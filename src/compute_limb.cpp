#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("limb_drag" / "limb_relax")
#include <cstdio>
#include <utility>

// ---------------------------------------------------------------------------
// Limb (snakehook) brush — ported onto the gpu:: seam (Seam Step 2b).
//
// Built on the MOVE sticky-capture spine (compute_move.cpp): the affected vert
// set, per-vertex falloff weights and mirror decomposition are all captured once
// at pen-down (via dispatch_move_capture / dispatch_move_weight_smooth) and reused
// here. Two passes run per dab, kernel logic in the canonical shaders/{glsl,wgsl}/
// limb_*.* (embedded at build time):
//
//   1. limb_drag   — INCREMENTAL grab: positions[v] += this-dab world delta * w.
//                    (Move's apply is absolute, init+total*w; that would reset
//                    positions every dab and wipe the relax. Incremental drag
//                    lets the redistribution accumulate.)
//   2. limb_relax  — tangential (normal-stripped) Laplacian over the captured
//                    set, run as a ping-pong between the live position VBO and a
//                    GL-owned scratch snapshot. Evens vertex spacing along the
//                    stretching shaft without deflating the form.
//
// The move/affected/weights buffers and the limb scratch SSBO stay GL-owned
// (wrapped in views at dispatch); the scratch snapshot/copy-back stay raw GL.
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to limb_drag.{comp,wgsl}'s Params.
struct LimbDragParamsGPU {
    float delta[3];  float _pad0;
};
static_assert(sizeof(LimbDragParamsGPU) == 16, "limb drag Params UBO must be 16 bytes");

// 32-byte std140 block, byte-identical to limb_relax.{comp,wgsl}'s Params.
struct LimbRelaxParamsGPU {
    uint32_t vertex_count;  float lambda;
    float    tip_bias;      float _pad0;     // 16
    float    tip_dir[3];    float _pad1;     // 16
};
static_assert(sizeof(LimbRelaxParamsGPU) == 32, "limb relax Params UBO must be 32 bytes");
}

// ---------------------------------------------------------------------------

bool ComputeState::init_limb() {
    if (!supported) return false;

    const gpu::BindEntry drag_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS,  gpu::Bind::StorageRead,      0 },
        { BIND_MASK,          gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(LimbDragParamsGPU) },
    };
    limb_drag_pipeline = gpu::create_compute_pipeline(gpu_dev,
                             gpu::embedded_shader("limb_drag"), drag_layout, 5);
    if (!limb_drag_pipeline.handle) {
        std::printf("[compute] limb_drag pipeline failed to compile\n");
        return false;
    }

    const gpu::BindEntry relax_layout[] = {
        { BIND_LIMB_POS_SRC,     gpu::Bind::StorageRead,      0 },
        { BIND_POSITIONS,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_NORMALS,          gpu::Bind::StorageRead,      0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS,     gpu::Bind::StorageRead,      0 },
        { BIND_MASK,             gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(LimbRelaxParamsGPU) },
    };
    limb_relax_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("limb_relax"), relax_layout, 9);
    if (!limb_relax_pipeline.handle) {
        std::printf("[compute] limb_relax pipeline failed to compile\n");
        gpu::release_compute_pipeline(limb_drag_pipeline);
        return false;
    }

    limb_drag_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(LimbDragParamsGPU),  gpu::Usage::Uniform);
    limb_relax_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(LimbRelaxParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] limb pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::ensure_limb_buffers(uint32_t vertex_count) {
    if (limb_scratch_capacity >= vertex_count && limb_pos_scratch_ssbo.handle) return;
    gpu::release_buffer(limb_pos_scratch_ssbo);
    limb_pos_scratch_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                (uint64_t)vertex_count * 3 * sizeof(float), gpu::Usage::Storage);
    limb_scratch_capacity = vertex_count;
}

void ComputeState::dispatch_limb_drag(const LimbDragParams& p, GLuint pos_vbo) {
    if (!has_limb() || !mask_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    LimbDragParamsGPU u = {};
    u.delta[0] = p.dx; u.delta[1] = p.dy; u.delta[2] = p.dz;
    gpu::write_buffer(gpu_dev, limb_drag_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{      (uint64_t)vc * 3u * sizeof(float),       pos_vbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &posView,            posView.size },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,  &move_weights_ssbo,  move_weights_ssbo.size },
        { BIND_MASK,          &mask_ssbo,    (uint64_t)vc * sizeof(float) },
        { BIND_PARAMS,        &limb_drag_ubo, sizeof(LimbDragParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, limb_drag_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, limb_drag_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_limb_relax(uint32_t vertex_count, int iterations, float lambda,
                                       float tip_dx, float tip_dy, float tip_dz, float tip_bias,
                                       GLuint pos_vbo, GLuint norm_vbo, GLuint index_ebo) {
    if (!has_limb() || !mask_ssbo.handle || iterations <= 0) return;
    ensure_limb_buffers(vertex_count);

    // src snapshot starts as a full copy of the live positions. pos_vbo is still a raw
    // GL handle; the scratch is seam-owned (raw-GL copy on its handle — web-stage concern).
    glBindBuffer(GL_COPY_READ_BUFFER,  pos_vbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, limb_pos_scratch_ssbo.handle);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                        (GLsizeiptr)vertex_count * 3 * sizeof(float));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    LimbRelaxParamsGPU u = {};
    u.vertex_count = vertex_count;
    u.lambda = lambda;
    u.tip_bias = tip_bias;
    u.tip_dir[0] = tip_dx; u.tip_dir[1] = tip_dy; u.tip_dir[2] = tip_dz;
    gpu::write_buffer(gpu_dev, limb_relax_ubo, 0, &u, sizeof(u));

    // Constant inputs across iterations (sizes unused on the GL backend — whole-buffer
    // bind; best-effort for the WebGPU min-size guard later). 0 = unguarded.
    gpu::Buffer normView{    (uint64_t)vertex_count * 3u * sizeof(float),    norm_vbo };
    gpu::Buffer idxView{     0,                                             index_ebo };
    // Ping-pong endpoints: scratch snapshot (seam-owned) and the live position VBO.
    gpu::Buffer scratchView{ (uint64_t)vertex_count * 3u * sizeof(float),    limb_pos_scratch_ssbo.handle };
    gpu::Buffer posView{     (uint64_t)vertex_count * 3u * sizeof(float),    pos_vbo };

    // Two bind groups differing only in slots 29 (src) / 0 (dst). ping reads
    // scratch→pos, pong reads pos→scratch. create_bind_group is a POD fill on GL.
    const gpu::BindBufferEntry ping[] = {
        { BIND_LIMB_POS_SRC,     &scratchView, scratchView.size },
        { BIND_POSITIONS,        &posView,     posView.size },
        { BIND_NORMALS,          &normView,    normView.size },
        { BIND_INDICES,          &idxView,     idxView.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_MOVE_WEIGHTS,     &move_weights_ssbo,     move_weights_ssbo.size },
        { BIND_MASK,             &mask_ssbo,   (uint64_t)vertex_count * sizeof(float) },
        { BIND_PARAMS,           &limb_relax_ubo, sizeof(LimbRelaxParamsGPU) },
    };
    const gpu::BindBufferEntry pong[] = {
        { BIND_LIMB_POS_SRC,     &posView,     posView.size },
        { BIND_POSITIONS,        &scratchView, scratchView.size },
        { BIND_NORMALS,          &normView,    normView.size },
        { BIND_INDICES,          &idxView,     idxView.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_MOVE_WEIGHTS,     &move_weights_ssbo,     move_weights_ssbo.size },
        { BIND_MASK,             &mask_ssbo,   (uint64_t)vertex_count * sizeof(float) },
        { BIND_PARAMS,           &limb_relax_ubo, sizeof(LimbRelaxParamsGPU) },
    };
    gpu::BindGroup grp_ping = gpu::create_bind_group(gpu_dev, limb_relax_pipeline, ping, 9);
    gpu::BindGroup grp_pong = gpu::create_bind_group(gpu_dev, limb_relax_pipeline, pong, 9);

    const uint32_t groups = (vertex_count + 255u) / 256u;
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    for (int iter = 0; iter < iterations; iter++) {
        // iter 0 reads scratch→pos (ping), iter 1 reads pos→scratch (pong), ...
        gpu::dispatch(b, limb_relax_pipeline,
                      (iter & 1) == 0 ? grp_ping : grp_pong, groups);
    }
    gpu::submit(b);
    gpu::release_bind_group(grp_ping);
    gpu::release_bind_group(grp_pong);

    // Even iteration count leaves the final result in the scratch buffer; copy it
    // back into the live position VBO. GL-owned copy, stays raw GL.
    if ((iterations & 1) == 0) {
        glBindBuffer(GL_COPY_READ_BUFFER,  limb_pos_scratch_ssbo.handle);
        glBindBuffer(GL_COPY_WRITE_BUFFER, pos_vbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            (GLsizeiptr)vertex_count * 3 * sizeof(float));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
}
