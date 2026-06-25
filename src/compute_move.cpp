#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("move_capture" / ...)
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Move/grab brush — ported onto the gpu:: seam (Seam Step 2b). Three stateful
// kernels: move_capture (brute-force world-distance gate → weights + init pos +
// appends the affected list, once at pen-down), move_weight_smooth (ping-pong
// Laplacian over the affected set), move_apply (per-dab pos = init + total·w, mirror
// summed). Kernel logic lives in the canonical shaders/{glsl,wgsl}/move_*.* (embedded
// at build time). The move/affected/weights/init buffers stay GL-owned (wrapped in
// views at dispatch); clears/copies/readback stay raw GL. Limb reuses capture +
// weight_smooth + readback_move_affected verbatim.
// ---------------------------------------------------------------------------

namespace {
// 32-byte std140 block, byte-identical to move_capture.{comp,wgsl}'s Params.
struct MoveCaptureParamsGPU {
    float    anchor[3];    float world_radius;   // 16
    float    hardness;     uint32_t mirror_x;
    uint32_t vertex_count; uint32_t _pad0;       // 16
};
static_assert(sizeof(MoveCaptureParamsGPU) == 32, "move capture Params UBO must be 32 bytes");

// 16-byte std140 block, byte-identical to move_apply.{comp,wgsl}'s Params.
struct MoveApplyParamsGPU {
    float total[3];  float _pad0;
};
static_assert(sizeof(MoveApplyParamsGPU) == 16, "move apply Params UBO must be 16 bytes");
}

// ---------------------------------------------------------------------------
// Move brush methods
// ---------------------------------------------------------------------------

bool ComputeState::init_move() {
    if (!supported) return false;

    const gpu::BindEntry capture_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_WEIGHTS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_INIT,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(MoveCaptureParamsGPU) },
    };
    move_capture_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("move_capture"), capture_layout, 5);
    if (!move_capture_pipeline.handle) {
        std::printf("[compute] move_capture pipeline failed to compile\n");
        return false;
    }

    const gpu::BindEntry smooth_layout[] = {
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_AFFECTED,    gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS,     gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS_PONG,gpu::Bind::StorageReadWrite, 0 },
    };
    move_weight_smooth_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                      gpu::embedded_shader("move_weight_smooth"), smooth_layout, 6);
    if (!move_weight_smooth_pipeline.handle) {
        std::printf("[compute] move_weight_smooth pipeline failed to compile\n");
        gpu::release_compute_pipeline(move_capture_pipeline);
        return false;
    }

    const gpu::BindEntry apply_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS,  gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_INIT,     gpu::Bind::StorageRead,      0 },
        { BIND_MASK,          gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(MoveApplyParamsGPU) },
    };
    move_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("move_apply"), apply_layout, 6);
    if (!move_apply_pipeline.handle) {
        std::printf("[compute] move_apply pipeline failed to compile\n");
        gpu::release_compute_pipeline(move_capture_pipeline);
        gpu::release_compute_pipeline(move_weight_smooth_pipeline);
        return false;
    }

    move_capture_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(MoveCaptureParamsGPU), gpu::Usage::Uniform);
    move_apply_ubo   = gpu::create_buffer(gpu_dev, nullptr, sizeof(MoveApplyParamsGPU),   gpu::Usage::Uniform);
    std::printf("[compute] move pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::ensure_move_buffers(uint32_t vertex_count) {
    if (move_buffers_capacity >= vertex_count
        && move_affected_ssbo.handle && move_weights_ssbo.handle
        && move_weights_pong_ssbo.handle && move_init_ssbo.handle)
        return;

    // Grow-only seam-owned scratch (release + create). The capture pass clears the
    // counter/weights each pen-down, so no init data is needed here.
    gpu::release_buffer(move_affected_ssbo);
    move_affected_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                             (uint64_t)(1 + vertex_count) * sizeof(uint32_t), gpu::Usage::Storage);

    gpu::release_buffer(move_weights_ssbo);
    move_weights_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                             (uint64_t)vertex_count * 2 * sizeof(float), gpu::Usage::Storage);

    gpu::release_buffer(move_weights_pong_ssbo);
    move_weights_pong_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                             (uint64_t)vertex_count * 2 * sizeof(float), gpu::Usage::Storage);

    gpu::release_buffer(move_init_ssbo);
    move_init_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                             (uint64_t)vertex_count * 3 * sizeof(float), gpu::Usage::Storage);

    move_buffers_capacity = vertex_count;
}

void ComputeState::dispatch_move_capture(const MoveCaptureParams& p, GLuint pos_vbo) {
    if (!has_move()) return;
    const uint32_t vc = p.vertex_count;

    // Reset the affected counter + clear the weight buffers — raw-GL data movement on
    // the seam-owned handles (clear/reset is a web-stage concern; stays raw GL on GL).
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_affected_ssbo.handle);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_ssbo.handle);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RG32F, GL_RG, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_pong_ssbo.handle);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RG32F, GL_RG, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    MoveCaptureParamsGPU u = {};
    u.anchor[0] = p.anchor_x; u.anchor[1] = p.anchor_y; u.anchor[2] = p.anchor_z;
    u.world_radius = p.world_radius;
    u.hardness = p.hardness;
    u.mirror_x = p.mirror_x ? 1u : 0u;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, move_capture_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{      (uint64_t)vc * 3u * sizeof(float),       pos_vbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &posView,            posView.size },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,  &move_weights_ssbo,  move_weights_ssbo.size },
        { BIND_MOVE_INIT,     &move_init_ssbo,     move_init_ssbo.size },
        { BIND_PARAMS,        &move_capture_ubo, sizeof(MoveCaptureParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, move_capture_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, move_capture_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_move_weight_smooth(uint32_t vertex_count, int iterations, GLuint index_ebo) {
    if (!move_weight_smooth_pipeline.handle) return;
    const uint32_t cap = move_buffers_capacity;

    // Constant inputs across iterations. index_ebo is still a raw GL handle (view);
    // the move scratch + adjacency are seam-owned, bound directly. 0 = unguarded.
    gpu::Buffer idxView{    0,                                       index_ebo };

    // Two bind groups for the ping-pong: slot 9 = w_in, slot 10 = w_out. ping reads
    // weights→pong, pong reads pong→weights. create_bind_group is a POD fill on GL.
    const gpu::BindBufferEntry ping[] = {
        { BIND_INDICES,          &idxView,     idxView.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_MOVE_AFFECTED,    &move_affected_ssbo,     move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,     &move_weights_ssbo,      move_weights_ssbo.size },
        { BIND_MOVE_WEIGHTS_PONG,&move_weights_pong_ssbo, move_weights_pong_ssbo.size },
    };
    const gpu::BindBufferEntry pong[] = {
        { BIND_INDICES,          &idxView,     idxView.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_MOVE_AFFECTED,    &move_affected_ssbo,     move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,     &move_weights_pong_ssbo, move_weights_pong_ssbo.size },
        { BIND_MOVE_WEIGHTS_PONG,&move_weights_ssbo,      move_weights_ssbo.size },
    };
    gpu::BindGroup grp_ping = gpu::create_bind_group(gpu_dev, move_weight_smooth_pipeline, ping, 6);
    gpu::BindGroup grp_pong = gpu::create_bind_group(gpu_dev, move_weight_smooth_pipeline, pong, 6);

    const uint32_t groups = (vertex_count + 255u) / 256u;
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    for (int iter = 0; iter < iterations; iter++) {
        gpu::dispatch(b, move_weight_smooth_pipeline,
                      (iter & 1) == 0 ? grp_ping : grp_pong, groups);
    }
    gpu::submit(b);
    gpu::release_bind_group(grp_ping);
    gpu::release_bind_group(grp_pong);

    // Odd iteration count leaves the final result in pong → copy it back to weights so
    // apply always reads the canonical weights buffer. GL-owned copy, stays raw GL.
    if ((iterations & 1) == 1) {
        glBindBuffer(GL_COPY_READ_BUFFER, move_weights_pong_ssbo.handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, move_weights_ssbo.handle);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            (GLsizeiptr)cap * 2 * sizeof(float));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
}

void ComputeState::dispatch_move_apply(const MoveApplyParams& p, GLuint pos_vbo) {
    if (!move_apply_pipeline.handle || !mask_ssbo) return;
    const uint32_t vc = p.vertex_count;

    MoveApplyParamsGPU u = {};
    u.total[0] = p.total_dx; u.total[1] = p.total_dy; u.total[2] = p.total_dz;
    gpu::write_buffer(gpu_dev, move_apply_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{      (uint64_t)vc * 3u * sizeof(float),       pos_vbo };
    gpu::Buffer maskView{     (uint64_t)vc * sizeof(float),            mask_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &posView,            posView.size },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,  &move_weights_ssbo,  move_weights_ssbo.size },
        { BIND_MOVE_INIT,     &move_init_ssbo,     move_init_ssbo.size },
        { BIND_MASK,          &maskView,     maskView.size },
        { BIND_PARAMS,        &move_apply_ubo, sizeof(MoveApplyParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, move_apply_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, move_apply_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

uint32_t ComputeState::readback_move_affected(std::vector<uint32_t>& out) {
    if (!move_affected_ssbo.handle) return 0;
    uint32_t count = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_affected_ssbo.handle);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &count);
    if (count > move_buffers_capacity) count = move_buffers_capacity;
    out.resize(count);
    if (count > 0) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t),
                           count * sizeof(uint32_t), out.data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return count;
}
