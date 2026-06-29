#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("color_paint" / "color_smooth")
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Paint (vpaint) brush — ported onto the gpu:: seam (Seam Step 2b). Two buffer-only
// kernels: color_paint (per-dab world-distance check, dual anchor mirror, lerps the
// packed RGBA8 albedo toward the brush colour, writes the colour buffer directly +
// appends to the dirty list) and color_smooth (blends each in-radius vertex's colour
// toward its 1-ring neighbour average). Kernel logic lives in the canonical
// shaders/{glsl,wgsl}/color_{paint,smooth}.* (embedded at build time). The colour /
// mask / dirty / adjacency buffers stay GL-owned (wrapped in views at dispatch); the
// dirty-counter reset stays raw GL.
// ---------------------------------------------------------------------------

namespace {
// 64-byte std140 block, byte-identical to color_paint.{comp,wgsl}'s Params.
struct ColorPaintParamsGPU {
    float    anchor_a[3];    float world_radius;     // 16
    float    anchor_b[3];    float hardness;         // 16
    float    paint_color[3]; float paint_strength;   // 16
    uint32_t use_b; uint32_t vertex_count; uint32_t _pad0; uint32_t _pad1; // 16
};
static_assert(sizeof(ColorPaintParamsGPU) == 64, "color paint Params UBO must be 64 bytes");

// 48-byte std140 block, byte-identical to color_smooth.{comp,wgsl}'s Params.
struct ColorSmoothParamsGPU {
    float    anchor_a[3]; float world_radius;        // 16
    float    anchor_b[3]; float hardness;            // 16
    float    strength; uint32_t use_b; uint32_t vertex_count; uint32_t _pad0; // 16
};
static_assert(sizeof(ColorSmoothParamsGPU) == 48, "color smooth Params UBO must be 48 bytes");

// Reset the dirty counter (the buffer's first word) to 0 via the seam.
void reset_dirty_counter(gpu::Device& dev, gpu::Buffer& dirty_ssbo) {
    uint32_t zero = 0;
    gpu::write_buffer(dev, dirty_ssbo, 0, &zero, sizeof(uint32_t));
}
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

bool ComputeState::init_color() {
    if (!supported) return false;

    const gpu::BindEntry paint_layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageRead,      0 },
        { BIND_COLOR,       gpu::Bind::StorageReadWrite, 0 },
        { BIND_MASK,        gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(ColorPaintParamsGPU) },
    };
    color_paint_pipeline = gpu::create_compute_pipeline(gpu_dev,
                               gpu::embedded_shader("color_paint"), paint_layout, 5);
    if (!color_paint_pipeline.handle) {
        std::printf("[compute] color_paint pipeline failed to compile\n");
        return false;
    }

    const gpu::BindEntry smooth_layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageRead,      0 },
        { BIND_COLOR,            gpu::Bind::StorageReadWrite, 0 },
        { BIND_MASK,             gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(ColorSmoothParamsGPU) },
    };
    color_smooth_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("color_smooth"), smooth_layout, 8);
    if (!color_smooth_pipeline.handle) {
        std::printf("[compute] color_smooth pipeline failed to compile\n");
        gpu::release_compute_pipeline(color_paint_pipeline);
        return false;
    }

    color_paint_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(ColorPaintParamsGPU),  gpu::Usage::Uniform);
    color_smooth_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ColorSmoothParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] color_paint + color_smooth pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_color_paint(const ColorPaintParams& p, const gpu::Buffer& pos_vbo) {
    if (!has_color() || !color_ssbo.handle || !mask_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    ensure_smooth_dirty_buffer(vc);
    reset_dirty_counter(gpu_dev, smooth_dirty_ssbo);

    ColorPaintParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.hardness = p.hardness;
    u.paint_color[0] = p.paint_r; u.paint_color[1] = p.paint_g; u.paint_color[2] = p.paint_b;
    u.paint_strength = p.paint_strength;
    u.use_b = (uint32_t)p.use_b;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, color_paint_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,   &pos_vbo,   (uint64_t)vc * 3u * sizeof(float) },
        { BIND_COLOR,       &color_ssbo, (uint64_t)vc * sizeof(uint32_t) },
        { BIND_MASK,        &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS, &smooth_dirty_ssbo, smooth_dirty_ssbo.size },
        { BIND_PARAMS,      &color_paint_ubo, sizeof(ColorPaintParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, color_paint_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, color_paint_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_color_smooth(const ColorPaintParams& p, const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo) {
    if (!has_color_smooth() || !color_ssbo.handle || !mask_ssbo.handle) return;
    if (!adjacency_offset_ssbo.handle || !adjacency_list_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    ensure_smooth_dirty_buffer(vc);
    reset_dirty_counter(gpu_dev, smooth_dirty_ssbo);

    ColorSmoothParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.hardness = p.hardness;
    u.strength = p.paint_strength;   // smooth reuses paint_strength as the blend amount
    u.use_b = (uint32_t)p.use_b;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, color_smooth_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,        &pos_vbo,   (uint64_t)vc * 3u * sizeof(float) },
        { BIND_COLOR,            &color_ssbo, (uint64_t)vc * sizeof(uint32_t) },
        { BIND_MASK,             &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS,      &smooth_dirty_ssbo, smooth_dirty_ssbo.size },
        { BIND_INDICES,          &index_ebo, index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_PARAMS,           &color_smooth_ubo, sizeof(ColorSmoothParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, color_smooth_pipeline, bg, 8);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, color_smooth_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
