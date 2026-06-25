#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("mask_paint")
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Mask paint — the first kernel ported onto the gpu:: seam (Seam Step 2b). The
// kernel logic lives in the canonical shaders/{glsl,wgsl}/mask_paint.* (embedded
// at build time); this file only drives the seam: pipeline, params UBO, bind
// group, dispatch. Per-vertex world-distance check, dual anchor (mirror), writes
// the mask buffer directly and appends touched vert ids to a compact dirty list.
// ---------------------------------------------------------------------------

namespace {
// std140 Params block, byte-identical to mask_paint.{comp,wgsl}'s `Params` (48
// bytes): a vec3 fills a 16-byte slot, so the trailing f32 packs into it.
struct MaskParamsGPU {
    float    anchor_a[3]; float world_radius;
    float    anchor_b[3]; float hardness;
    float    paint_strength; uint32_t use_b; uint32_t vertex_count; uint32_t _pad0;
};
static_assert(sizeof(MaskParamsGPU) == 48, "mask Params UBO must be 48 bytes (std140)");
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

bool ComputeState::init_mask() {
    if (!supported) return false;

    gpu::ShaderSources src = gpu::embedded_shader("mask_paint");
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageRead,      0 },
        { BIND_MASK,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(MaskParamsGPU) },
    };
    mask_pipeline = gpu::create_compute_pipeline(gpu_dev, src, layout, 4);
    if (!mask_pipeline.handle) {
        std::printf("[compute] mask_paint pipeline failed to compile\n");
        return false;
    }
    mask_params_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(MaskParamsGPU),
                                         gpu::Usage::Uniform);
    std::printf("[compute] mask_paint pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_mask_paint(const MaskPaintParams& p, GLuint pos_vbo) {
    if (!has_mask() || !mask_ssbo) return;

    ensure_smooth_dirty_buffer(p.vertex_count);

    const uint32_t vc = p.vertex_count;

    // Buffer views over the GL-owned working buffers (renderer VBOs + the dirty
    // list SSBO). On the GL backend a gpu::Buffer is just its handle; the seam binds
    // them at dispatch. Sizes are advisory on GL, the bind range on WebGPU.
    gpu::Buffer posView{   (uint64_t)vc * 3u * sizeof(float),    pos_vbo };
    gpu::Buffer maskView{  (uint64_t)vc * sizeof(float),         mask_ssbo };

    // Reset the dirty counter (slot 0), then upload this dab's params.
    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, smooth_dirty_ssbo, 0, &zero, sizeof(zero));

    MaskParamsGPU mp = {};
    mp.anchor_a[0] = p.anchor_a_x; mp.anchor_a[1] = p.anchor_a_y; mp.anchor_a[2] = p.anchor_a_z;
    mp.anchor_b[0] = p.anchor_b_x; mp.anchor_b[1] = p.anchor_b_y; mp.anchor_b[2] = p.anchor_b_z;
    mp.world_radius   = p.world_radius;
    mp.hardness       = p.hardness;
    mp.paint_strength = p.paint_strength;
    mp.use_b          = (uint32_t)p.use_b;
    mp.vertex_count   = vc;
    gpu::write_buffer(gpu_dev, mask_params_ubo, 0, &mp, sizeof(mp));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,   &posView,           posView.size },
        { BIND_MASK,        &maskView,          maskView.size },
        { BIND_DIRTY_VERTS, &smooth_dirty_ssbo, (uint64_t)(vc + 1u) * sizeof(uint32_t) },
        { BIND_PARAMS,      &mask_params_ubo,   sizeof(MaskParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, mask_pipeline, bg, 4);

    uint32_t groups = (vc + 255u) / 256u;
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, mask_pipeline, grp, groups);
    gpu::submit(b);   // GL backend issues the storage+vertex-attrib barriers here

    gpu::release_bind_group(grp);   // no-op on GL; frees the transient group on WebGPU
}
