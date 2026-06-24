#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("crease_accum" / "pinch_accum")
#include <cstdio>

// ---------------------------------------------------------------------------
// Crease + pinch brushes — ported onto the gpu:: seam (Seam Step 2b). Both are
// accum-only kernels: they deposit into the shared accum buffer, and the brush
// reuses the already-seamed draw_apply / symmetrize / mirror_apply for the apply
// side. Kernel logic lives in the canonical shaders/{glsl,wgsl}/{crease,pinch}_accum.*
// (embedded at build time); this file drives the seam. accum / stroke-normal buffers
// stay GL-owned (wrapped in views at dispatch), as with draw.
// ---------------------------------------------------------------------------

namespace {
// 112-byte std140 block, byte-identical to crease_accum.{comp,wgsl}'s Params.
struct CreaseParamsGPU {
    float    anchor_a[3];        float world_radius;
    float    anchor_b[3];        float disp_amount;
    float    view_a[3];          float pinch_amount;
    float    view_b[3];          float hardness;
    float    anchor_normal_a[3]; float facing_threshold;
    float    anchor_normal_b[3]; uint32_t use_b;
    uint32_t vertex_count;       uint32_t _pad0; uint32_t _pad1; uint32_t _pad2;
};
static_assert(sizeof(CreaseParamsGPU) == 112, "crease Params UBO must be 112 bytes");

// 96-byte std140 block, byte-identical to pinch_accum.{comp,wgsl}'s Params.
struct PinchParamsGPU {
    float    anchor_a[3];        float world_radius;
    float    anchor_b[3];        float pinch_amount;
    float    view_a[3];          float hardness;
    float    view_b[3];          float facing_threshold;
    float    anchor_normal_a[3]; uint32_t use_b;
    float    anchor_normal_b[3]; uint32_t vertex_count;
};
static_assert(sizeof(PinchParamsGPU) == 96, "pinch Params UBO must be 96 bytes");

// Bind layout shared by both accum kernels: pos(read) / norm(read) / accum(rw) / UBO.
gpu::BindGroup make_accum_bind_group(ComputeState& cs, gpu::ComputePipeline& pipe,
                                     gpu::Buffer& posView, gpu::Buffer& normView,
                                     gpu::Buffer& accumView, gpu::Buffer& ubo, uint64_t ubo_size) {
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS, &posView,   posView.size },
        { BIND_NORMALS,   &normView,  normView.size },
        { BIND_ACCUM,     &accumView, accumView.size },
        { BIND_PARAMS,    &ubo,       ubo_size },
    };
    return gpu::create_bind_group(cs.gpu_dev, pipe, bg, 4);
}
}

// ---------------------------------------------------------------------------
// Crease
// ---------------------------------------------------------------------------

bool ComputeState::init_crease() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS, gpu::Bind::StorageRead,      0 },
        { BIND_NORMALS,   gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,    gpu::Bind::Uniform,          sizeof(CreaseParamsGPU) },
    };
    crease_accum_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("crease_accum"), layout, 4);
    if (!crease_accum_pipeline.handle) {
        std::printf("[compute] crease_accum pipeline failed to compile\n");
        return false;
    }
    crease_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(CreaseParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] crease_accum pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_crease_accum(const CreaseAccumParams& p, GLuint pos_vbo) {
    if (!has_crease() || !accum_ssbo || !stroke_norm_ssbo) return;
    const uint32_t vc = p.vertex_count;

    clear_accum_buffer();   // raw GL (GL-owned accum buffer)

    CreaseParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.disp_amount = p.disp_amount;
    u.view_a[0] = p.view_a_x; u.view_a[1] = p.view_a_y; u.view_a[2] = p.view_a_z;
    u.pinch_amount = p.pinch_amount;
    u.view_b[0] = p.view_b_x; u.view_b[1] = p.view_b_y; u.view_b[2] = p.view_b_z;
    u.hardness = p.hardness;
    u.anchor_normal_a[0] = p.anchor_normal_a_x; u.anchor_normal_a[1] = p.anchor_normal_a_y; u.anchor_normal_a[2] = p.anchor_normal_a_z;
    u.facing_threshold = p.facing_threshold;
    u.anchor_normal_b[0] = p.anchor_normal_b_x; u.anchor_normal_b[1] = p.anchor_normal_b_y; u.anchor_normal_b[2] = p.anchor_normal_b_z;
    u.use_b = (uint32_t)p.use_b;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, crease_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{  (uint64_t)vc * 3u * sizeof(float),    pos_vbo };
    gpu::Buffer normView{ (uint64_t)vc * 3u * sizeof(float),    stroke_norm_ssbo };
    gpu::Buffer accumView{(uint64_t)vc * 4u * sizeof(uint32_t), accum_ssbo };
    gpu::BindGroup grp = make_accum_bind_group(*this, crease_accum_pipeline,
                                               posView, normView, accumView,
                                               crease_ubo, sizeof(CreaseParamsGPU));

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, crease_accum_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

// ---------------------------------------------------------------------------
// Pinch
// ---------------------------------------------------------------------------

bool ComputeState::init_pinch() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS, gpu::Bind::StorageRead,      0 },
        { BIND_NORMALS,   gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,    gpu::Bind::Uniform,          sizeof(PinchParamsGPU) },
    };
    pinch_accum_pipeline = gpu::create_compute_pipeline(gpu_dev,
                               gpu::embedded_shader("pinch_accum"), layout, 4);
    if (!pinch_accum_pipeline.handle) {
        std::printf("[compute] pinch_accum pipeline failed to compile\n");
        return false;
    }
    pinch_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(PinchParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] pinch_accum pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_pinch_accum(const PinchAccumParams& p, GLuint pos_vbo) {
    if (!has_pinch() || !accum_ssbo || !stroke_norm_ssbo) return;
    const uint32_t vc = p.vertex_count;

    clear_accum_buffer();   // raw GL (GL-owned accum buffer)

    PinchParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.pinch_amount = p.pinch_amount;
    u.view_a[0] = p.view_a_x; u.view_a[1] = p.view_a_y; u.view_a[2] = p.view_a_z;
    u.hardness = p.hardness;
    u.view_b[0] = p.view_b_x; u.view_b[1] = p.view_b_y; u.view_b[2] = p.view_b_z;
    u.facing_threshold = p.facing_threshold;
    u.anchor_normal_a[0] = p.anchor_normal_a_x; u.anchor_normal_a[1] = p.anchor_normal_a_y; u.anchor_normal_a[2] = p.anchor_normal_a_z;
    u.use_b = (uint32_t)p.use_b;
    u.anchor_normal_b[0] = p.anchor_normal_b_x; u.anchor_normal_b[1] = p.anchor_normal_b_y; u.anchor_normal_b[2] = p.anchor_normal_b_z;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, pinch_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{  (uint64_t)vc * 3u * sizeof(float),    pos_vbo };
    gpu::Buffer normView{ (uint64_t)vc * 3u * sizeof(float),    stroke_norm_ssbo };
    gpu::Buffer accumView{(uint64_t)vc * 4u * sizeof(uint32_t), accum_ssbo };
    gpu::BindGroup grp = make_accum_bind_group(*this, pinch_accum_pipeline,
                                               posView, normView, accumView,
                                               pinch_ubo, sizeof(PinchParamsGPU));

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, pinch_accum_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
