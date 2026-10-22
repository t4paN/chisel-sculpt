#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("pose_apply")
#include <cstdio>

// ---------------------------------------------------------------------------
// Pose brush. Which vertices a grab carries, and how much, is worked out once at
// pen-down on the CPU (pose_bind, src/pose.cpp): it needs ray-like skin sampling
// around each bone and a flood over the mesh's connectivity, neither of which fits
// a kernel. The result goes up compact, in the move brush's buffers where the rest
// of the stroke machinery (undo fold, normals, mirror sink) already reads them:
//
//   move_affected_ssbo  {count, ids[]}          the vertices that move
//   move_init_ssbo      3 floats per affected   their pen-down positions (compact)
//   pose_weights_ssbo   2 vec4 per affected     one weight per moving bone, then mirrored
//
// pose_apply then runs per dab over just those vertices.
// ---------------------------------------------------------------------------

bool ComputeState::init_pose() {
    if (!supported) return false;

    const gpu::BindEntry apply_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_WEIGHTS,  gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_INIT,     gpu::Bind::StorageRead,      0 },
        { BIND_MASK,          gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(PoseParamsGPU) },
    };
    pose_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("pose_apply"), apply_layout, 6);
    if (!pose_apply_pipeline.handle) {
        std::printf("[compute] pose_apply pipeline failed to compile\n");
        return false;
    }

    pose_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(PoseParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] pose pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::upload_pose_binding(const uint32_t* ids, const float* init, const float* w,
                                       uint32_t n) {
    if (!has_pose()) return;
    if (pose_weights_capacity < n || !pose_weights_ssbo.handle) {
        gpu::release_buffer(pose_weights_ssbo);
        const uint32_t cap = n < 1024u ? 1024u : n;
        pose_weights_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                (uint64_t)cap * 8u * sizeof(float), gpu::Usage::Storage);
        pose_weights_capacity = cap;
    }
    gpu::write_buffer(gpu_dev, move_affected_ssbo, 0, &n, sizeof(uint32_t));
    if (n == 0) return;
    gpu::write_buffer(gpu_dev, move_affected_ssbo, sizeof(uint32_t), ids, (uint64_t)n * sizeof(uint32_t));
    gpu::write_buffer(gpu_dev, move_init_ssbo, 0, init, (uint64_t)n * 3u * sizeof(float));
    gpu::write_buffer(gpu_dev, pose_weights_ssbo, 0, w, (uint64_t)n * 8u * sizeof(float));
}

void ComputeState::dispatch_pose_apply(const PoseParamsGPU& p, const gpu::Buffer& pos_vbo,
                                       uint32_t vc) {
    if (!has_pose() || !mask_ssbo.handle || p.affected_count == 0) return;

    gpu::write_buffer(gpu_dev, pose_ubo, 0, &p, sizeof(p));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &pos_vbo,            (uint64_t)vc * 3u * sizeof(float) },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_WEIGHTS,  &pose_weights_ssbo,  pose_weights_ssbo.size },
        { BIND_MOVE_INIT,     &move_init_ssbo,     move_init_ssbo.size },
        { BIND_MASK,          &mask_ssbo,          (uint64_t)vc * sizeof(float) },
        { BIND_PARAMS,        &pose_ubo,           sizeof(PoseParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, pose_apply_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, pose_apply_pipeline, grp, (p.affected_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
