#include "compute.h"
#include "paint_layers.h"
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

// 336-byte std140 block, byte-identical to color_composite.{comp,wgsl}'s Params.
struct ColorCompositeParamsGPU {
    uint32_t vertex_count; uint32_t layer_count; uint32_t list_mode; uint32_t _pad0; // 16
    float    opacity[16];                                                            // 64
    uint32_t layers[16][4];   // ids off, values off, count, mode | source << 8         256
};
static_assert(sizeof(ColorCompositeParamsGPU) == 336, "color composite Params UBO must be 336 bytes");
static_assert(kMaxPaintLayers == 16, "the composite's layer table is sized for 16 layers");

// Where a layer's values live, the table's source field (see color_composite.wgsl).
constexpr uint32_t kLayerSrcPacked = 0, kLayerSrcDense = 1, kLayerSrcActive = 2;

}  // namespace

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

bool ComputeState::init_color() {
    if (!supported) return false;

    const gpu::BindEntry paint_layout[] = {
        { BIND_POSITIONS,    gpu::Bind::StorageRead,      0 },
        { BIND_COLOR,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_MASK,         gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_ALPHA_TEX,    gpu::Bind::StorageRead,      0 },
        { BIND_ALPHA_PARAMS, gpu::Bind::Uniform,          48 },
        { BIND_BLOCK_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       gpu::Bind::Uniform,          sizeof(ColorPaintParamsGPU) },
    };
    color_paint_pipeline = gpu::create_compute_pipeline(gpu_dev,
                               gpu::embedded_shader("color_paint"), paint_layout, 9);
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
        { BIND_BLOCK_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(ColorSmoothParamsGPU) },
    };
    color_smooth_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("color_smooth"), smooth_layout, 10);
    if (!color_smooth_pipeline.handle) {
        std::printf("[compute] color_smooth pipeline failed to compile\n");
        gpu::release_compute_pipeline(color_paint_pipeline);
        return false;
    }

    color_paint_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(ColorPaintParamsGPU),  gpu::Usage::Uniform);
    color_smooth_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ColorSmoothParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] color_paint + color_smooth pipelines compiled (gpu:: seam)\n");

    // Paint layers composite. Optional: if it fails, painting still works on a single
    // layer exactly as before, so it never fails init_color.
    const gpu::BindEntry composite_layout[] = {
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageRead,      0 },
        { BIND_COLOR,            gpu::Bind::StorageRead,      0 },
        { BIND_COLOR_DISPLAY,    gpu::Bind::StorageReadWrite, 0 },
        { BIND_PAINT_LAYER_POOL, gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION,     gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(ColorCompositeParamsGPU) },
    };
    color_composite_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                   gpu::embedded_shader("color_composite"), composite_layout, 6);
    if (color_composite_pipeline.handle) {
        color_composite_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ColorCompositeParamsGPU),
                                                 gpu::Usage::Uniform);
        std::printf("[compute] color_composite pipeline compiled (paint layers)\n");
    } else {
        std::printf("[compute] color_composite pipeline failed to compile (paint layers off)\n");
    }
    return true;
}

void ComputeState::dispatch_color_paint(const ColorPaintParams& p, const gpu::Buffer& pos_vbo) {
    if (!has_color() || !color_ssbo.handle || !mask_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    ensure_smooth_dirty_buffer(vc);

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
    select_dab_blocks(u.anchor_a, u.anchor_b, u.world_radius, u.use_b != 0, vc);

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,    &pos_vbo,   (uint64_t)vc * 3u * sizeof(float) },
        { BIND_COLOR,        &color_ssbo, (uint64_t)vc * sizeof(uint32_t) },
        { BIND_MASK,         &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS,  &smooth_dirty_ssbo, smooth_dirty_ssbo.size },
        { BIND_ALPHA_TEX,    &alpha_tex_ssbo,   (uint64_t)alpha_tex_w * alpha_tex_h * sizeof(float) },
        { BIND_ALPHA_PARAMS, &alpha_params_ubo, 48 },
        { BIND_BLOCK_LIST,   &block_list_ssbo,   block_list_ssbo.size },
        { BIND_DIRTY_REGION, &dirty_region_ubo,  sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       &color_paint_ubo, sizeof(ColorPaintParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, color_paint_pipeline, bg, 9);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    dispatch_blocks_or_full(b, color_paint_pipeline, grp, vc);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_color_smooth(const ColorPaintParams& p, const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo) {
    if (!has_color_smooth() || !color_ssbo.handle || !mask_ssbo.handle) return;
    if (!adjacency_offset_ssbo.handle || !adjacency_list_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    ensure_smooth_dirty_buffer(vc);

    ColorSmoothParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.hardness = p.hardness;
    u.strength = p.paint_strength;   // smooth reuses paint_strength as the blend amount
    u.use_b = (uint32_t)p.use_b;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, color_smooth_ubo, 0, &u, sizeof(u));
    select_dab_blocks(u.anchor_a, u.anchor_b, u.world_radius, u.use_b != 0, vc);

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,        &pos_vbo,   (uint64_t)vc * 3u * sizeof(float) },
        { BIND_COLOR,            &color_ssbo, (uint64_t)vc * sizeof(uint32_t) },
        { BIND_MASK,             &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_DIRTY_VERTS,      &smooth_dirty_ssbo, smooth_dirty_ssbo.size },
        { BIND_INDICES,          &index_ebo, index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_BLOCK_LIST,   &block_list_ssbo,   block_list_ssbo.size },
        { BIND_DIRTY_REGION, &dirty_region_ubo,  sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           &color_smooth_ubo, sizeof(ColorSmoothParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, color_smooth_pipeline, bg, 10);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    dispatch_blocks_or_full(b, color_smooth_pipeline, grp, vc);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

// ---------------------------------------------------------------------------
// Paint layers composite
// ---------------------------------------------------------------------------

bool ComputeState::upload_paint_layers(const PaintLayerStack& stack, uint32_t vc) {
    paint_layers_uploaded = false;
    if (!has_color_composite()) return false;

    ColorCompositeParamsGPU u = {};
    u.vertex_count = vc;
    std::vector<uint32_t>& pool = paint_layer_pool_cpu;
    pool.clear();
    for (int i = 0; i < (int)stack.layers.size(); i++) {
        const PaintLayer& L = stack.layers[i];
        if (!L.visible || L.opacity <= 0.0f) continue;
        const uint32_t k = u.layer_count++;
        uint32_t* e = u.layers[k];
        uint32_t src;
        if (i == stack.active) {
            src = kLayerSrcActive;           // read straight from color_ssbo
        } else if (L.dense) {
            src  = kLayerSrcDense;
            e[1] = (uint32_t)pool.size();
            pool.insert(pool.end(), L.rgba.begin(), L.rgba.end());
            pool.resize(e[1] + (size_t)vc, 0u);   // never read past the mesh
        } else {
            src  = kLayerSrcPacked;
            e[0] = (uint32_t)pool.size();
            pool.insert(pool.end(), L.ids.begin(), L.ids.end());
            e[1] = (uint32_t)pool.size();
            pool.insert(pool.end(), L.rgba.begin(), L.rgba.end());
            e[2] = (uint32_t)L.ids.size();
        }
        e[3] = (uint32_t)L.mode | (src << 8);
        u.opacity[k] = L.opacity;
    }
    if (pool.empty()) pool.push_back(0u);    // a zero-size binding is invalid on WebGPU

    const uint64_t bytes = (uint64_t)pool.size() * sizeof(uint32_t);
    if (bytes > gpu::device_limits().max_storage_binding_size) {
        std::printf("[paint-layers] pool %.1f MB exceeds the storage binding limit (%.1f MB)\n",
                    bytes / 1048576.0,
                    gpu::device_limits().max_storage_binding_size / 1048576.0);
        return false;
    }
    if (paint_layer_pool_ssbo.size < bytes) {
        gpu::release_buffer(paint_layer_pool_ssbo);
        paint_layer_pool_ssbo = gpu::create_buffer(gpu_dev, nullptr, bytes, gpu::Usage::Storage);
    }
    gpu::write_buffer(gpu_dev, paint_layer_pool_ssbo, 0, pool.data(), bytes);
    std::memcpy(paint_layer_table, &u, sizeof(u));
    paint_layers_uploaded = true;
    return true;
}

static void dispatch_color_composite_impl(ComputeState& cs, uint32_t vc, bool dirty_only) {
    if (!cs.has_color_composite() || !cs.paint_layers_uploaded) return;
    if (!cs.color_ssbo.handle || !cs.color_display_ssbo.handle) return;
    // Until layers are wired, color_ssbo IS the display VBO; binding one buffer as both
    // read and read_write is a validation error on WebGPU.
    if (cs.color_ssbo.handle == cs.color_display_ssbo.handle) return;
    if (dirty_only && !cs.has_dirty_args()) return;

    ColorCompositeParamsGPU u;
    std::memcpy(&u, cs.paint_layer_table, sizeof(u));
    u.vertex_count = vc;
    u.list_mode    = dirty_only ? 1u : 0u;
    gpu::write_buffer(cs.gpu_dev, cs.color_composite_ubo, 0, &u, sizeof(u));

    cs.ensure_smooth_dirty_buffer(vc);
    const gpu::BindBufferEntry bg[] = {
        { BIND_DIRTY_VERTS,      &cs.smooth_dirty_ssbo,     cs.smooth_dirty_ssbo.size },
        { BIND_COLOR,            &cs.color_ssbo,            (uint64_t)vc * sizeof(uint32_t) },
        { BIND_COLOR_DISPLAY,    &cs.color_display_ssbo,    (uint64_t)vc * sizeof(uint32_t) },
        { BIND_PAINT_LAYER_POOL, &cs.paint_layer_pool_ssbo, cs.paint_layer_pool_ssbo.size },
        { BIND_DIRTY_REGION,     &cs.dirty_region_ubo,      sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           &cs.color_composite_ubo,   sizeof(ColorCompositeParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(cs.gpu_dev, cs.color_composite_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
    if (dirty_only) {
        // The dab's touched count only exists on the GPU: dirty_args turns it into the
        // workgroup count (256 per group, matching the kernel) — same as the mirror sink.
        const gpu::BindBufferEntry args_bg[] = {
            { BIND_DIRTY_VERTS,   &cs.smooth_dirty_ssbo,  cs.smooth_dirty_ssbo.size },
            { BIND_DISPATCH_ARGS, &cs.dispatch_args_ssbo, cs.dispatch_args_ssbo.size },
            { BIND_DIRTY_REGION,  &cs.dirty_region_ubo,   sizeof(DirtyRegionGPU) },
        };
        gpu::BindGroup args_grp = gpu::create_bind_group(cs.gpu_dev, cs.dirty_args_pipeline,
                                                         args_bg, 3);
        gpu::dispatch(b, cs.dirty_args_pipeline, args_grp, 1);
        gpu::dispatch_indirect(b, cs.color_composite_pipeline, grp, cs.dispatch_args_ssbo, 0);
        gpu::submit(b);
        gpu::release_bind_group(args_grp);
    } else {
        gpu::dispatch(b, cs.color_composite_pipeline, grp, (vc + 255u) / 256u);
        gpu::submit(b);
    }
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_color_composite_all(uint32_t vc)   { dispatch_color_composite_impl(*this, vc, false); }
void ComputeState::dispatch_color_composite_dirty(uint32_t vc) { dispatch_color_composite_impl(*this, vc, true); }
