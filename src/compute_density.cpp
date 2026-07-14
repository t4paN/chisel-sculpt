#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("density_paint")
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Density paint — the remesh-density field brush (Paint mode, density target).
// Clone of the mask brush pair (compute_mask.cpp): same Params shape, same dab
// mechanics, buffer swapped to BIND_DENSITY. density_colormap additionally
// writes colormap(density) into the display colour VBO — the field view.
// ---------------------------------------------------------------------------

namespace {
// std140 Params block, byte-identical to density_paint.{comp,wgsl}'s `Params`
// (48 bytes) — the same shape as the mask pair's.
struct DensityParamsGPU {
    float    anchor_a[3]; float world_radius;
    float    anchor_b[3]; float hardness;
    float    paint_strength; uint32_t use_b; uint32_t vertex_count; uint32_t _pad0;
};
static_assert(sizeof(DensityParamsGPU) == 48, "density Params UBO must be 48 bytes (std140)");

struct DensityColormapParamsGPU {
    uint32_t vertex_count; uint32_t _pad0; uint32_t _pad1; uint32_t _pad2;
};
static_assert(sizeof(DensityColormapParamsGPU) == 16, "colormap Params UBO must be 16 bytes");
}

bool ComputeState::init_density() {
    if (!supported) return false;

    gpu::ShaderSources src = gpu::embedded_shader("density_paint");
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,    gpu::Bind::StorageRead,      0 },
        { BIND_DENSITY,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_ALPHA_TEX,    gpu::Bind::StorageRead,      0 },
        { BIND_ALPHA_PARAMS, gpu::Bind::Uniform,          48 },
        { BIND_PARAMS,       gpu::Bind::Uniform,          sizeof(DensityParamsGPU) },
    };
    density_pipeline = gpu::create_compute_pipeline(gpu_dev, src, layout, 6);
    if (!density_pipeline.handle) {
        std::printf("[compute] density_paint pipeline failed to compile\n");
        return false;
    }
    density_params_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DensityParamsGPU),
                                            gpu::Usage::Uniform);

    const gpu::BindEntry smooth_layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageRead,      0 },
        { BIND_DENSITY,          gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(DensityParamsGPU) },
    };
    density_smooth_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                  gpu::embedded_shader("density_smooth"), smooth_layout, 7);
    if (density_smooth_pipeline.handle) {
        density_smooth_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DensityParamsGPU),
                                                gpu::Usage::Uniform);
    } else {
        std::printf("[compute] density_smooth pipeline failed to compile (density_paint OK)\n");
    }

    const gpu::BindEntry cmap_layout[] = {
        { BIND_DENSITY, gpu::Bind::StorageRead,      0 },
        { BIND_COLOR,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,  gpu::Bind::Uniform,          sizeof(DensityColormapParamsGPU) },
    };
    density_colormap_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                    gpu::embedded_shader("density_colormap"), cmap_layout, 3);
    if (density_colormap_pipeline.handle) {
        density_colormap_ubo = gpu::create_buffer(gpu_dev, nullptr,
                                   sizeof(DensityColormapParamsGPU), gpu::Usage::Uniform);
        std::printf("[compute] density_paint + density_smooth + density_colormap pipelines compiled (gpu:: seam)\n");
    } else {
        std::printf("[compute] density_colormap pipeline failed to compile\n");
    }
    return true;
}

void ComputeState::dispatch_density_paint(const MaskPaintParams& p, const gpu::Buffer& pos_vbo) {
    if (!has_density_kernels() || !density_ssbo.handle) return;

    ensure_smooth_dirty_buffer(p.vertex_count);

    const uint32_t vc = p.vertex_count;

    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, smooth_dirty_ssbo, 0, &zero, sizeof(zero));

    DensityParamsGPU dp = {};
    dp.anchor_a[0] = p.anchor_a_x; dp.anchor_a[1] = p.anchor_a_y; dp.anchor_a[2] = p.anchor_a_z;
    dp.anchor_b[0] = p.anchor_b_x; dp.anchor_b[1] = p.anchor_b_y; dp.anchor_b[2] = p.anchor_b_z;
    dp.world_radius   = p.world_radius;
    dp.hardness       = p.hardness;
    dp.paint_strength = p.paint_strength;
    dp.use_b          = (uint32_t)p.use_b;
    dp.vertex_count   = vc;
    gpu::write_buffer(gpu_dev, density_params_ubo, 0, &dp, sizeof(dp));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,    &pos_vbo,           (uint64_t)vc * 3u * sizeof(float) },
        { BIND_DENSITY,      &density_ssbo,      (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS,  &smooth_dirty_ssbo, (uint64_t)(vc + 1u) * sizeof(uint32_t) },
        { BIND_ALPHA_TEX,    &alpha_tex_ssbo,    (uint64_t)alpha_tex_w * alpha_tex_h * sizeof(float) },
        { BIND_ALPHA_PARAMS, &alpha_params_ubo,  48 },
        { BIND_PARAMS,       &density_params_ubo, sizeof(DensityParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, density_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, density_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_density_smooth(const MaskPaintParams& p, const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo) {
    if (!has_density_smooth() || !density_ssbo.handle) return;
    if (!adjacency_offset_ssbo.handle || !adjacency_list_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    ensure_smooth_dirty_buffer(vc);

    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, smooth_dirty_ssbo, 0, &zero, sizeof(zero));

    DensityParamsGPU dp = {};
    dp.anchor_a[0] = p.anchor_a_x; dp.anchor_a[1] = p.anchor_a_y; dp.anchor_a[2] = p.anchor_a_z;
    dp.anchor_b[0] = p.anchor_b_x; dp.anchor_b[1] = p.anchor_b_y; dp.anchor_b[2] = p.anchor_b_z;
    dp.world_radius   = p.world_radius;
    dp.hardness       = p.hardness;
    dp.paint_strength = p.paint_strength;   // smooth reuses paint_strength as the blend amount
    dp.use_b          = (uint32_t)p.use_b;
    dp.vertex_count   = vc;
    gpu::write_buffer(gpu_dev, density_smooth_ubo, 0, &dp, sizeof(dp));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,        &pos_vbo,           (uint64_t)vc * 3u * sizeof(float) },
        { BIND_DENSITY,          &density_ssbo,      (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS,      &smooth_dirty_ssbo, (uint64_t)(vc + 1u) * sizeof(uint32_t) },
        { BIND_INDICES,          &index_ebo,         index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_PARAMS,           &density_smooth_ubo, sizeof(DensityParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, density_smooth_pipeline, bg, 7);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, density_smooth_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_density_colormap(uint32_t vertex_count) {
    if (!density_colormap_pipeline.handle || !density_ssbo.handle || !color_ssbo.handle) return;

    DensityColormapParamsGPU cp = {};
    cp.vertex_count = vertex_count;
    gpu::write_buffer(gpu_dev, density_colormap_ubo, 0, &cp, sizeof(cp));

    const gpu::BindBufferEntry bg[] = {
        { BIND_DENSITY, &density_ssbo,        (uint64_t)vertex_count * sizeof(float) },
        { BIND_COLOR,   &color_ssbo,          (uint64_t)vertex_count * sizeof(uint32_t) },
        { BIND_PARAMS,  &density_colormap_ubo, sizeof(DensityColormapParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, density_colormap_pipeline, bg, 3);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, density_colormap_pipeline, grp, (vertex_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
