#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("pose_capture" / "pose_apply")
#include <cstdio>

// ---------------------------------------------------------------------------
// Pose brush. Two kernels on the move brush's capture buffers:
//
//   1. pose_capture — once at pen-down: every vertex a moving bone reaches gets
//                     its position snapshotted into move_init and its id appended
//                     to move_affected (the list undo, normals and the mirror sink
//                     already read for move).
//   2. pose_apply   — per dab: positions = init + the bones' blended swing.
//
// Both read the same 3600-byte Params block (PoseParamsGPU): the grabbed set's
// bones, pen-down geometry plus this dab's rotation and stretch. The set is capped
// at 32 bones (kPoseMaxBones) so it fits one uniform block.
// ---------------------------------------------------------------------------

bool ComputeState::init_pose() {
    if (!supported) return false;

    const gpu::BindEntry capture_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_INIT,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(PoseParamsGPU) },
    };
    pose_capture_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("pose_capture"), capture_layout, 4);
    if (!pose_capture_pipeline.handle) {
        std::printf("[compute] pose_capture pipeline failed to compile\n");
        return false;
    }

    const gpu::BindEntry apply_layout[] = {
        { BIND_POSITIONS,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_MOVE_AFFECTED, gpu::Bind::StorageRead,      0 },
        { BIND_MOVE_INIT,     gpu::Bind::StorageRead,      0 },
        { BIND_MASK,          gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(PoseParamsGPU) },
    };
    pose_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("pose_apply"), apply_layout, 5);
    if (!pose_apply_pipeline.handle) {
        std::printf("[compute] pose_apply pipeline failed to compile\n");
        gpu::release_compute_pipeline(pose_capture_pipeline);
        return false;
    }

    pose_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(PoseParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] pose pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_pose_capture(const PoseParamsGPU& p, const gpu::Buffer& pos_vbo) {
    if (!has_pose()) return;
    const uint32_t vc = p.vertex_count;

    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, move_affected_ssbo, 0, &zero, sizeof(uint32_t));
    gpu::write_buffer(gpu_dev, pose_ubo, 0, &p, sizeof(p));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &pos_vbo,            (uint64_t)vc * 3u * sizeof(float) },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_INIT,     &move_init_ssbo,     move_init_ssbo.size },
        { BIND_PARAMS,        &pose_ubo,           sizeof(PoseParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, pose_capture_pipeline, bg, 4);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, pose_capture_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_pose_apply(const PoseParamsGPU& p, const gpu::Buffer& pos_vbo) {
    if (!has_pose() || !mask_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    gpu::write_buffer(gpu_dev, pose_ubo, 0, &p, sizeof(p));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,     &pos_vbo,            (uint64_t)vc * 3u * sizeof(float) },
        { BIND_MOVE_AFFECTED, &move_affected_ssbo, move_affected_ssbo.size },
        { BIND_MOVE_INIT,     &move_init_ssbo,     move_init_ssbo.size },
        { BIND_MASK,          &mask_ssbo,          (uint64_t)vc * sizeof(float) },
        { BIND_PARAMS,        &pose_ubo,           sizeof(PoseParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, pose_apply_pipeline, bg, 5);

    // The affected count lives on the GPU, so dispatch for every vertex; the kernel
    // returns past the count (same as move_apply).
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, pose_apply_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
