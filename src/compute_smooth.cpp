#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("smooth_accum" / ...)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Smooth brush — ported onto the gpu:: seam (Seam Step 2b). Three buffer-only
// kernels: smooth_accum (world-distance gate → accum.w + dirty list), smooth_apply
// (uniform Laplacian, looped), smooth_mirror_apply (re-impose the X mirror after
// each iteration to keep the seam from creasing). It is buffer-only — the triid/bary pick read is CPU-side
// back-projection in brush.cpp, NOT a compute input — so it needs no texture-bind
// seam work. Kernel logic lives in shaders/{glsl,wgsl}/smooth_*.* (embedded at build
// time). accum / dirty / mirror-map / mask buffers stay GL-owned (wrapped in views at
// dispatch); the accum clear + dirty-counter reset stay raw GL.
// ---------------------------------------------------------------------------

namespace {
// 32-byte std140 block, byte-identical to smooth_accum.{comp,wgsl}'s Params.
struct SmoothAccumParamsGPU {
    float    anchor[3];    float world_radius;   // 16
    float    hardness;     uint32_t mirror_clip;
    uint32_t vertex_count; uint32_t _pad0;       // 16
    float    anchor_b[3];  uint32_t use_b;       // 16
};
static_assert(sizeof(SmoothAccumParamsGPU) == 48, "smooth accum Params UBO must be 48 bytes");

// 16-byte std140 block, byte-identical to smooth_apply.{comp,wgsl}'s Params.
struct SmoothApplyParamsGPU {
    uint32_t vertex_count; float strength; uint32_t _pad0; uint32_t _pad1;
};
static_assert(sizeof(SmoothApplyParamsGPU) == 16, "smooth apply Params UBO must be 16 bytes");

// 16-byte std140 block, byte-identical to smooth_mirror_apply.{comp,wgsl}'s Params.
struct SmoothMirrorParamsGPU {
    uint32_t vertex_count; float anchor_x; uint32_t _pad0; uint32_t _pad1;
};
static_assert(sizeof(SmoothMirrorParamsGPU) == 16, "smooth mirror Params UBO must be 16 bytes");
}

// ---------------------------------------------------------------------------
// Compute normals shader
// ---------------------------------------------------------------------------

static const char* compute_normals_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf { float positions[]; };
layout(std430, binding = 1) writeonly buffer NormBuf { float normals[]; };
layout(std430, binding = 2) readonly buffer IdxBuf { uint indices[]; };
layout(std430, binding = 4) readonly buffer AdjOffset { uint adj_offset[]; };
layout(std430, binding = 5) readonly buffer AdjList { uint adj_list[]; };

layout(std430, binding = 6) readonly buffer DirtyVerts { uint dirty_verts[]; };

uniform uint u_dirty_count;

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_dirty_count) return;

    uint v = dirty_verts[di];

    float nx = 0.0, ny = 0.0, nz = 0.0;

    uint start = adj_offset[v];
    uint end = adj_offset[v + 1u];

    for (uint j = start; j < end; j++) {
        uint t = adj_list[j];
        uint i0 = indices[t * 3u];
        uint i1 = indices[t * 3u + 1u];
        uint i2 = indices[t * 3u + 2u];

        vec3 p0 = vec3(positions[i0*3u], positions[i0*3u+1u], positions[i0*3u+2u]);
        vec3 p1 = vec3(positions[i1*3u], positions[i1*3u+1u], positions[i1*3u+2u]);
        vec3 p2 = vec3(positions[i2*3u], positions[i2*3u+1u], positions[i2*3u+2u]);

        vec3 e1 = p1 - p0;
        vec3 e2 = p2 - p0;
        vec3 fn = cross(e1, e2);
        float area2 = length(fn);
        if (area2 < 1e-7) continue;

        nx += fn.x;
        ny += fn.y;
        nz += fn.z;
    }

    float len = sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-8) {
        float inv = 1.0 / len;
        nx *= inv;
        ny *= inv;
        nz *= inv;
    } else {
        nx = 0.0; ny = 0.0; nz = 0.0;
    }

    uint base = v * 3u;
    normals[base]     = nx;
    normals[base + 1u] = ny;
    normals[base + 2u] = nz;
}
)";

// ---------------------------------------------------------------------------
// Stroke autosmooth shader
// ---------------------------------------------------------------------------

static const char* stroke_smooth_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf { float positions[]; };
layout(std430, binding = 2)  readonly buffer IdxBuf { uint indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffset { uint adj_offset[]; };
layout(std430, binding = 5)  readonly buffer AdjList { uint adj_list[]; };
layout(std430, binding = 6)  readonly buffer DirtyVerts { uint dirty_verts[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[]; };

uniform uint u_dirty_count;
uniform float u_strength;

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_dirty_count) return;
    uint v = dirty_verts[di];

    float mscale = 1.0 - mask[v];
    if (mscale <= 0.0) return;

    float cur_x = positions[v * 3u];
    float cur_y = positions[v * 3u + 1u];
    float cur_z = positions[v * 3u + 2u];

    float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    float count = 0.0;

    uint start = adj_offset[v];
    uint end = adj_offset[v + 1u];
    for (uint j = start; j < end; j++) {
        uint t = adj_list[j];
        uint i0 = indices[t * 3u];
        uint i1 = indices[t * 3u + 1u];
        uint i2 = indices[t * 3u + 2u];
        if (v == i0) {
            sum_x += positions[i1*3u] + positions[i2*3u];
            sum_y += positions[i1*3u+1u] + positions[i2*3u+1u];
            sum_z += positions[i1*3u+2u] + positions[i2*3u+2u];
        } else if (v == i1) {
            sum_x += positions[i0*3u] + positions[i2*3u];
            sum_y += positions[i0*3u+1u] + positions[i2*3u+1u];
            sum_z += positions[i0*3u+2u] + positions[i2*3u+2u];
        } else {
            sum_x += positions[i0*3u] + positions[i1*3u];
            sum_y += positions[i0*3u+1u] + positions[i1*3u+1u];
            sum_z += positions[i0*3u+2u] + positions[i1*3u+2u];
        }
        count += 2.0;
    }
    if (count <= 0.0) return;

    // Pen-up autosmooth keeps the plain uniform Laplacian: it's a single mild pass
    // (twins ride in snap_list so it's already symmetric) and was validated
    // pinch-free. The normal-projection directionality fix lives on the interactive
    // smooth brush, which has a brush radius to size the seam band from.
    float inv_c = 1.0 / count;
    float blend = u_strength * mscale;
    positions[v * 3u]     += (sum_x * inv_c - cur_x) * blend;
    positions[v * 3u + 1u] += (sum_y * inv_c - cur_y) * blend;
    positions[v * 3u + 2u] += (sum_z * inv_c - cur_z) * blend;
}
)";

// ---------------------------------------------------------------------------
// Smooth brush methods
// ---------------------------------------------------------------------------

bool ComputeState::init_smooth() {
    if (!supported) return false;

    const gpu::BindEntry accum_layout[] = {
        { BIND_POSITIONS,    gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_ALPHA_TEX,    gpu::Bind::StorageRead,      0 },
        { BIND_ALPHA_PARAMS, gpu::Bind::Uniform,          48 },
        { BIND_BLOCK_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       gpu::Bind::Uniform,          sizeof(SmoothAccumParamsGPU) },
    };
    smooth_accum_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("smooth_accum"), accum_layout, 8);
    if (!smooth_accum_pipeline.handle) {
        std::printf("[compute] smooth_accum pipeline failed to compile\n");
        return false;
    }

    const gpu::BindEntry apply_layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,            gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_MASK,             gpu::Bind::StorageRead,      0 },
        { BIND_BLOCK_LIST,       gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION,     gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(SmoothApplyParamsGPU) },
    };
    smooth_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("smooth_apply"), apply_layout, 9);
    if (!smooth_apply_pipeline.handle) {
        std::printf("[compute] smooth_apply pipeline failed to compile\n");
        gpu::release_compute_pipeline(smooth_accum_pipeline);
        return false;
    }

    const gpu::BindEntry mirror_layout[] = {
        { BIND_POSITIONS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_ACCUM,      gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP, gpu::Bind::StorageRead,      0 },
        { BIND_MASK,       gpu::Bind::StorageRead,      0 },
        { BIND_BLOCK_LIST, gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,        sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,     gpu::Bind::Uniform,          sizeof(SmoothMirrorParamsGPU) },
    };
    smooth_mirror_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                       gpu::embedded_shader("smooth_mirror_apply"), mirror_layout, 7);
    if (!smooth_mirror_apply_pipeline.handle) {
        std::printf("[compute] smooth_mirror_apply pipeline failed to compile\n");
        gpu::release_compute_pipeline(smooth_accum_pipeline);
        gpu::release_compute_pipeline(smooth_apply_pipeline);
        return false;
    }

    smooth_accum_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(SmoothAccumParamsGPU),  gpu::Usage::Uniform);
    smooth_apply_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(SmoothApplyParamsGPU),  gpu::Usage::Uniform);
    smooth_mirror_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(SmoothMirrorParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] smooth pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_smooth_mirror_apply(const gpu::Buffer& pos_vbo, uint32_t vertex_count, float anchor_x) {
    if (!smooth_mirror_apply_pipeline.handle || mirror_map_vertex_count == 0 || !mask_ssbo.handle) return;
    const uint32_t vc = vertex_count;

    SmoothMirrorParamsGPU u = {};
    u.vertex_count = vc;
    u.anchor_x = anchor_x;
    gpu::write_buffer(gpu_dev, smooth_mirror_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,  &pos_vbo,    (uint64_t)vc * 3u * sizeof(float) },
        { BIND_ACCUM,      &accum_ssbo, (uint64_t)vc * 4u * sizeof(uint32_t) },
        { BIND_MIRROR_MAP, &mirror_map_ssbo, (uint64_t)mirror_map_vertex_count * sizeof(uint32_t) },
        { BIND_MASK,       &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_BLOCK_LIST, &block_list_ssbo, block_list_ssbo.size },
        { BIND_DIRTY_REGION, &dirty_region_ubo, sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,     &smooth_mirror_ubo, sizeof(SmoothMirrorParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, smooth_mirror_apply_pipeline, bg, 7);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    dispatch_blocks_or_full(b, smooth_mirror_apply_pipeline, grp, vc);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_smooth(const SmoothAccumParams& p,
                                    const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo) {
    if (!has_smooth() || !mask_ssbo.handle) return;
    const uint32_t vc = p.vertex_count;

    // Select first, then clear only what this dab will write. The selection must
    // cover the far lobe under EITHER mirror mode: the geometric path smooths it with
    // a second lobe (use_b), while the topological path reaches it through
    // smooth_mirror_apply's twins. Selecting extra blocks is always safe; missing the
    // twins' blocks would leave them holding a previous dab's accum.
    const float sa[3] = { p.anchor_x, p.anchor_y, p.anchor_z };
    const float sb[3] = { p.anchor_b_x, p.anchor_b_y, p.anchor_b_z };
    if (has_block_cull()) {
        select_dab_blocks(sa, sb, p.world_radius, p.use_b != 0 || p.mirror_pairs, vc);
        clear_accum_blocks(vc);
    } else {
        clear_accum_buffer();   // raw GL (GL-owned buffer)
    }
    ensure_smooth_dirty_buffer(vc);

    // ---- Pass 1: accum (world-distance gate → accum.w + dirty list) ----
    SmoothAccumParamsGPU ua = {};
    ua.anchor[0] = p.anchor_x; ua.anchor[1] = p.anchor_y; ua.anchor[2] = p.anchor_z;
    ua.world_radius = p.world_radius;
    ua.hardness = p.hardness;
    ua.mirror_clip = p.mirror_pairs ? 1u : 0u;
    ua.vertex_count = vc;
    ua.anchor_b[0] = p.anchor_b_x; ua.anchor_b[1] = p.anchor_b_y; ua.anchor_b[2] = p.anchor_b_z;
    ua.use_b = p.use_b ? 1u : 0u;
    gpu::write_buffer(gpu_dev, smooth_accum_ubo, 0, &ua, sizeof(ua));

    {
        const gpu::BindBufferEntry bg[] = {
            { BIND_POSITIONS,    &pos_vbo,    (uint64_t)vc * 3u * sizeof(float) },
            { BIND_ACCUM,        &accum_ssbo, (uint64_t)vc * 4u * sizeof(uint32_t) },
            { BIND_DIRTY_VERTS,  &smooth_dirty_ssbo, smooth_dirty_ssbo.size },
            { BIND_ALPHA_TEX,    &alpha_tex_ssbo,   (uint64_t)alpha_tex_w * alpha_tex_h * sizeof(float) },
            { BIND_ALPHA_PARAMS, &alpha_params_ubo, 48 },
            { BIND_BLOCK_LIST,   &block_list_ssbo,   block_list_ssbo.size },
            { BIND_DIRTY_REGION, &dirty_region_ubo,  sizeof(DirtyRegionGPU) },
            { BIND_PARAMS,       &smooth_accum_ubo, sizeof(SmoothAccumParamsGPU) },
        };
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, smooth_accum_pipeline, bg, 8);
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        dispatch_blocks_or_full(b, smooth_accum_pipeline, grp, vc);
        gpu::submit(b);
        gpu::release_bind_group(grp);
    }

    // ---- Pass 2: apply (uniform Laplacian), looped ----
    SmoothApplyParamsGPU up = {};
    up.vertex_count = vc;
    up.strength = p.strength;
    gpu::write_buffer(gpu_dev, smooth_apply_ubo, 0, &up, sizeof(up));

    const gpu::BindBufferEntry apply_bg[] = {
        { BIND_POSITIONS,        &pos_vbo,    (uint64_t)vc * 3u * sizeof(float) },
        { BIND_INDICES,          &index_ebo,  index_ebo.size },
        { BIND_ACCUM,            &accum_ssbo, (uint64_t)vc * 4u * sizeof(uint32_t) },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_MASK,             &mask_ssbo,  (uint64_t)vc * sizeof(float) },
        { BIND_BLOCK_LIST,       &block_list_ssbo,  block_list_ssbo.size },
        { BIND_DIRTY_REGION,     &dirty_region_ubo, sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           &smooth_apply_ubo, sizeof(SmoothApplyParamsGPU) },
    };
    gpu::BindGroup apply_grp = gpu::create_bind_group(gpu_dev, smooth_apply_pipeline, apply_bg, 9);

    // TOPOLOGICAL path only. It clips the accum gate to the anchor's own side, so
    // the reflection has to be re-imposed after *each* Laplacian iteration, not just
    // at the end: a vert whose 1-ring crosses x=0 would otherwise average against a
    // stale wall every pass and the seam would stand proud as a symmetric crease.
    //
    // The geometric path needs none of this — its second lobe smooths the far side
    // in the same dispatch, from that side's own neighbours, so the cross-seam
    // 1-rings are live on both sides and nothing has to be teleported afterwards.
    bool do_mirror = p.mirror_pairs && smooth_mirror_apply_pipeline.handle
                     && mirror_map_vertex_count == vc;
    for (int iter = 0; iter < p.iterations; iter++) {
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        dispatch_blocks_or_full(b, smooth_apply_pipeline, apply_grp, vc);
        gpu::submit(b);

        if (do_mirror) {
            dispatch_smooth_mirror_apply(pos_vbo, vc, p.anchor_x);
        }
    }
    gpu::release_bind_group(apply_grp);
}

// Arena budget. Peak live id volume across a full sculpting session (measured at
// 5M tris, fast strokes and huge brushes alike) topped out around 23 MB, so 32 MB
// leaves room and still costs less than the mesh's own position buffer. Running
// out is not an error — dirty_arena_alloc fails and the caller stalls — so this
// trades VRAM against how often a very fast stroke has to wait.
static constexpr uint32_t kDirtyArenaBudgetWords = (32u * 1024u * 1024u) / 4u;

void ComputeState::ensure_smooth_dirty_buffer(uint32_t max_verts, uint64_t ring_words_hint) {
    uint32_t alloc = std::max(max_verts, 4096u);
    // Word 0 is a permanent bit bucket, never part of the ring: a dab that fails to
    // get a region still runs its kernels, and they will atomically bump whatever word
    // `base` names. Aiming those writes at word 0 (with cap 0, so no ids are stored)
    // keeps them away from a region another dab is still reading.
    //
    // One region must always be able to hold a worst-case dab (every vertex), so the
    // arena is at least that big even when the budget is smaller — at which point it
    // holds exactly one dab and every dab stalls, which is slow but still correct.
    uint64_t want = std::max((uint64_t)kDirtyArenaBudgetWords, (uint64_t)alloc + 2u);
    // Past ~8M verts that floor is ALSO the ceiling: a ring one mesh wide holds one
    // big dab, so the next dab in flight got a scrap region and overflowed — at L10
    // every big-brush dab did, each forcing a whole-mesh undo snapshot. The caller
    // asks for room for several of its measured dabs; grant it only while nothing is
    // live (a live region's ids would be lost with the old buffer), and never past
    // what the device can bind (128 MB on the WebGPU baseline).
    const bool must = !smooth_dirty_ssbo.handle || max_verts > smooth_dirty_capacity
                   || dirty_arena_words < want;
    if (ring_words_hint + 1u > want && (must || dirty_arena_live == 0)) {
        uint64_t limit = gpu::device_limits().max_storage_binding_size / sizeof(uint32_t);
        want = std::max(want, std::min(ring_words_hint + 1u, limit));
    }
    if (!must && dirty_arena_words >= want)
        return;

    gpu::release_buffer(smooth_dirty_ssbo);
    smooth_dirty_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                           want * sizeof(uint32_t), gpu::Usage::Storage);
    smooth_dirty_capacity = alloc;
    dirty_arena_words = (uint32_t)want;
    dirty_arena_reset();
}

uint32_t ComputeState::readback_accum_dirty(uint32_t vertex_count, std::vector<uint32_t>& out) {
    out.clear();
    if (!accum_ssbo.handle || vertex_count == 0) return 0;

    GLsizeiptr size = (GLsizeiptr)vertex_count * 4 * sizeof(uint32_t);
    if (readback_buf.size() < vertex_count * 4) {
        readback_buf.resize(vertex_count * 4);
    }

    gpu::read_buffer(gpu_dev, accum_ssbo, 0, size, readback_buf.data());

    uint32_t count = 0;
    for (uint32_t v = 0; v < vertex_count; v++) {
        float w = reinterpret_cast<float*>(readback_buf.data())[v * 4 + 3];
        if (w > 0.0f) {
            out.push_back(v);
            count++;
        }
    }

    return count;
}

// ---------------------------------------------------------------------------
// Compute normals methods
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to compute_normals.{comp,wgsl}'s Params.
struct ComputeNormalsParamsGPU {
    uint32_t dirty_count; uint32_t list_mode; uint32_t header_cap; uint32_t _pad2;
};
static_assert(sizeof(ComputeNormalsParamsGPU) == 16, "compute_normals Params UBO must be 16 bytes");
}

bool ComputeState::init_compute_normals() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageRead,      0 },
        { BIND_NORMALS,          gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(ComputeNormalsParamsGPU) },
    };
    compute_normals_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                   gpu::embedded_shader("compute_normals"), layout, 7);
    if (!compute_normals_pipeline.handle) {
        std::printf("[compute] compute_normals pipeline failed to compile\n");
        return false;
    }
    compute_normals_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ComputeNormalsParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] compute_normals pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::upload_adjacency(const uint32_t* offsets, uint32_t offset_count,
                                     const uint32_t* list, uint32_t list_count) {
    // Grow-only re-upload: the CSR sizes change after remesh, so release + recreate
    // each time (uploaded once per topology change, not in the hot path). Seam-owned.
    gpu::release_buffer(adjacency_offset_ssbo);
    adjacency_offset_ssbo = gpu::create_buffer(gpu_dev, offsets,
                                               (uint64_t)offset_count * sizeof(uint32_t), gpu::Usage::Storage);

    gpu::release_buffer(adjacency_list_ssbo);
    adjacency_list_ssbo = gpu::create_buffer(gpu_dev, list,
                                             (uint64_t)list_count * sizeof(uint32_t), gpu::Usage::Storage);

    adjacency_vertex_count = offset_count - 1;

    std::printf("[compute] adjacency uploaded: %u verts, %u entries\n",
                adjacency_vertex_count, list_count);
}

void ComputeState::dispatch_compute_normals(const uint32_t* dirty_verts, uint32_t dirty_count,
                                             const gpu::Buffer& pos_vbo, const gpu::Buffer& norm_vbo, const gpu::Buffer& index_ebo) {
    if (!has_normals() || dirty_count == 0) return;

    // Upload the dirty-vert id list — seam-owned buffer (shared with stroke_smooth /
    // multires). Grow-only (release + create); partial fill via write_buffer.
    if (!dirty_verts_ssbo.handle || dirty_count > dirty_verts_capacity) {
        uint32_t alloc_count = std::max(dirty_count, 4096u);
        gpu::release_buffer(dirty_verts_ssbo);
        dirty_verts_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                              (uint64_t)alloc_count * sizeof(uint32_t), gpu::Usage::Storage);
        dirty_verts_capacity = alloc_count;
    }
    gpu::write_buffer(gpu_dev, dirty_verts_ssbo, 0, dirty_verts, (uint64_t)dirty_count * sizeof(uint32_t));

    ComputeNormalsParamsGPU u = {};
    u.dirty_count = dirty_count;
    gpu::write_buffer(gpu_dev, compute_normals_ubo, 0, &u, sizeof(u));

    // pos/norm/index bound whole (the kernel scatters over the dirty list). Real buffer
    // sizes from the seam-owned gpu::Buffers. Adjacency + dirty list are seam-owned too.
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,        &pos_vbo,   pos_vbo.size },
        { BIND_NORMALS,          &norm_vbo,  norm_vbo.size },
        { BIND_INDICES,          &index_ebo, index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_DIRTY_VERTS,      &dirty_verts_ssbo,      dirty_verts_ssbo.size },
        { BIND_PARAMS,           &compute_normals_ubo, sizeof(ComputeNormalsParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, compute_normals_pipeline, bg, 7);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, compute_normals_pipeline, grp, (dirty_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

// ---------------------------------------------------------------------------
// GPU normals expansion (normals_expand + compute_normals list_mode 1)
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to normals_expand.{comp,wgsl}'s Params.
struct NormalsExpandParamsGPU {
    uint32_t stamp; uint32_t vertex_count; uint32_t use_mirror; uint32_t out_cap;
};
static_assert(sizeof(NormalsExpandParamsGPU) == 16, "normals_expand Params UBO must be 16 bytes");
}

bool ComputeState::init_normals_expand() {
    if (!supported) return false;
    // On by default; CHISEL_GPU_NORMALS=0 keeps the CPU expansion for an A/B.
    const char* e = getenv("CHISEL_GPU_NORMALS");
    if (e && *e == '0') {
        std::printf("[compute] GPU normals expansion DISABLED (CHISEL_GPU_NORMALS=0)\n");
        return false;
    }
    const gpu::BindEntry layout[] = {
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP,       gpu::Bind::StorageRead,      0 },
        { BIND_NORM_MARK,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_NORM_LIST,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_REGION,     gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(NormalsExpandParamsGPU) },
    };
    normals_expand_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                  gpu::embedded_shader("normals_expand"), layout, 9);
    if (!normals_expand_pipeline.handle) {
        std::printf("[compute] normals_expand pipeline failed to compile\n");
        return false;
    }
    normals_expand_ubo   = gpu::create_buffer(gpu_dev, nullptr, sizeof(NormalsExpandParamsGPU),
                                              gpu::Usage::Uniform);
    norm_src_region_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(DirtyRegionGPU),
                                              gpu::Usage::Uniform);
    norm_list_region_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DirtyRegionGPU),
                                              gpu::Usage::Uniform);
    norm_args_ssbo = gpu::create_buffer(gpu_dev, nullptr, 3 * sizeof(uint32_t),
                                        gpu::Usage::Storage | gpu::Usage::Indirect);
    gpu_normals_on = true;
    std::printf("[compute] normals_expand pipeline compiled (GPU normals expansion on)\n");
    return true;
}

void ComputeState::expand_normals_from(const gpu::Buffer& list, uint32_t base, uint32_t cap,
                                       uint32_t vertex_count, bool use_mirror,
                                       const gpu::Buffer& index_ebo) {
    if (!has_gpu_normals() || !list.handle || cap == 0 || vertex_count == 0) return;
    if (adjacency_vertex_count != vertex_count) return;

    // Sized to the whole mesh: the frame stamp dedupes, so the list can never hold
    // more than every vertex once. Grow-only, zeroed on (re)allocation — a zeroed mark
    // is "unclaimed", which is why the stamp starts at 1.
    if (norm_capacity < vertex_count || !norm_mark_ssbo.handle) {
        gpu::release_buffer(norm_mark_ssbo);
        gpu::release_buffer(norm_list_ssbo);
        norm_mark_ssbo = gpu::create_buffer(gpu_dev, nullptr, (uint64_t)vertex_count * sizeof(uint32_t),
                                            gpu::Usage::Storage);
        norm_list_ssbo = gpu::create_buffer(gpu_dev, nullptr, ((uint64_t)vertex_count + 1) * sizeof(uint32_t),
                                            gpu::Usage::Storage);
        gpu::clear_buffer(gpu_dev, norm_mark_ssbo, 0);
        gpu::clear_buffer(gpu_dev, norm_list_ssbo, 0);
        norm_capacity = vertex_count;
        norm_stamp = 1;
        norm_expands = 0;
    }

    // The pair-map twin is only meaningful when the uploaded map is this mesh's.
    const bool mirror = use_mirror && mirror_map_ssbo.handle
                        && mirror_map_vertex_count == vertex_count;

    DirtyRegionGPU src = { base, cap, 0, 0 };
    gpu::write_buffer(gpu_dev, norm_src_region_ubo, 0, &src, sizeof(src));
    NormalsExpandParamsGPU u = { norm_stamp, vertex_count, mirror ? 1u : 0u, norm_capacity };
    gpu::write_buffer(gpu_dev, normals_expand_ubo, 0, &u, sizeof(u));

    // The count lives in the list's header on the GPU, so dirty_args turns it into the
    // expand's workgroup count (both at 256 per group) — same pattern as the mirror sink.
    const gpu::BindBufferEntry args_bg[] = {
        { BIND_DIRTY_VERTS,   &list,                list.size },
        { BIND_DISPATCH_ARGS, &norm_args_ssbo,      norm_args_ssbo.size },
        { BIND_DIRTY_REGION,  &norm_src_region_ubo, sizeof(DirtyRegionGPU) },
    };
    gpu::BindGroup args_grp = gpu::create_bind_group(gpu_dev, dirty_args_pipeline, args_bg, 3);

    // With no mirror the map binding is never read, but it must still be bound; the
    // index buffer is read-only here already, so it stands in without a usage clash.
    const gpu::Buffer& map = mirror ? mirror_map_ssbo : index_ebo;
    const gpu::BindBufferEntry bg[] = {
        { BIND_INDICES,          &index_ebo,             index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_DIRTY_VERTS,      &list,                  list.size },
        { BIND_MIRROR_MAP,       &map,                   map.size },
        { BIND_NORM_MARK,        &norm_mark_ssbo,        norm_mark_ssbo.size },
        { BIND_NORM_LIST,        &norm_list_ssbo,        norm_list_ssbo.size },
        { BIND_DIRTY_REGION,     &norm_src_region_ubo,   sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,           &normals_expand_ubo,    sizeof(NormalsExpandParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, normals_expand_pipeline, bg, 9);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, dirty_args_pipeline, args_grp, 1);
    gpu::dispatch_indirect(b, normals_expand_pipeline, grp, norm_args_ssbo, 0);
    gpu::submit(b);
    gpu::release_bind_group(args_grp);
    gpu::release_bind_group(grp);
    norm_expands++;
    norm_vc = vertex_count;
}

bool ComputeState::flush_gpu_normals(uint32_t vertex_count, const gpu::Buffer& pos_vbo,
                                     const gpu::Buffer& norm_vbo, const gpu::Buffer& index_ebo) {
    if (!has_gpu_normals() || norm_expands == 0 || !norm_list_ssbo.handle) return false;
    norm_expands = 0;

    // Ids queued against a different mesh (an aborted stroke, then a level switch)
    // would index past the new adjacency — drop them; the rebuild recomputed normals.
    if (norm_vc == vertex_count && adjacency_vertex_count == vertex_count) {
        DirtyRegionGPU lr = { 0, norm_capacity, 0, 0 };
        gpu::write_buffer(gpu_dev, norm_list_region_ubo, 0, &lr, sizeof(lr));
        ComputeNormalsParamsGPU u = {};
        u.list_mode  = 1;
        u.header_cap = norm_capacity;
        gpu::write_buffer(gpu_dev, compute_normals_ubo, 0, &u, sizeof(u));

        const gpu::BindBufferEntry args_bg[] = {
            { BIND_DIRTY_VERTS,   &norm_list_ssbo,       norm_list_ssbo.size },
            { BIND_DISPATCH_ARGS, &norm_args_ssbo,       norm_args_ssbo.size },
            { BIND_DIRTY_REGION,  &norm_list_region_ubo, sizeof(DirtyRegionGPU) },
        };
        gpu::BindGroup args_grp = gpu::create_bind_group(gpu_dev, dirty_args_pipeline, args_bg, 3);
        const gpu::BindBufferEntry bg[] = {
            { BIND_POSITIONS,        &pos_vbo,   pos_vbo.size },
            { BIND_NORMALS,          &norm_vbo,  norm_vbo.size },
            { BIND_INDICES,          &index_ebo, index_ebo.size },
            { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
            { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
            { BIND_DIRTY_VERTS,      &norm_list_ssbo,        norm_list_ssbo.size },
            { BIND_PARAMS,           &compute_normals_ubo, sizeof(ComputeNormalsParamsGPU) },
        };
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, compute_normals_pipeline, bg, 7);

        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, dirty_args_pipeline, args_grp, 1);
        gpu::dispatch_indirect(b, compute_normals_pipeline, grp, norm_args_ssbo, 0);
        gpu::submit(b);
        gpu::release_bind_group(args_grp);
        gpu::release_bind_group(grp);
    }

    // New frame: empty the list and move the stamp on, so every vertex is claimable
    // again without touching the mark buffer. Queue-ordered after the submit above.
    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, norm_list_ssbo, 0, &zero, sizeof(zero));
    if (++norm_stamp == 0) {
        gpu::clear_buffer(gpu_dev, norm_mark_ssbo, 0);
        norm_stamp = 1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// GPU touched list (touched_fold)
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to touched_fold.{comp,wgsl}'s Params.
struct TouchedFoldParamsGPU {
    uint32_t stamp; uint32_t vertex_count; uint32_t use_mirror; uint32_t _pad;
};
static_assert(sizeof(TouchedFoldParamsGPU) == 16, "touched_fold Params UBO must be 16 bytes");
}

bool ComputeState::init_touched_fold() {
    if (!supported || !has_dirty_args()) return false;
    // On by default; CHISEL_GPU_TOUCHED=0 keeps the per-dab dirty readbacks for an A/B.
    const char* e = getenv("CHISEL_GPU_TOUCHED");
    if (e && *e == '0') {
        std::printf("[compute] GPU touched list DISABLED (CHISEL_GPU_TOUCHED=0)\n");
        return false;
    }
    const gpu::BindEntry layout[] = {
        { BIND_DIRTY_VERTS,  gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP,   gpu::Bind::StorageRead,      0 },
        { BIND_NORM_MARK,    gpu::Bind::StorageReadWrite, 0 },
        { BIND_NORM_LIST,    gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       gpu::Bind::Uniform,          sizeof(TouchedFoldParamsGPU) },
    };
    touched_fold_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("touched_fold"), layout, 6);
    if (!touched_fold_pipeline.handle) {
        std::printf("[compute] touched_fold pipeline failed to compile\n");
        return false;
    }
    touched_fold_ubo   = gpu::create_buffer(gpu_dev, nullptr, sizeof(TouchedFoldParamsGPU),
                                            gpu::Usage::Uniform);
    touched_region_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DirtyRegionGPU),
                                            gpu::Usage::Uniform);
    touched_args_ssbo  = gpu::create_buffer(gpu_dev, nullptr, 3 * sizeof(uint32_t),
                                            gpu::Usage::Storage | gpu::Usage::Indirect);
    gpu_touched_on = true;
    std::printf("[compute] touched_fold pipeline compiled (GPU touched list on)\n");
    return true;
}

void ComputeState::dispatch_touched_fold(const gpu::Buffer& src, uint32_t base, uint32_t cap,
                                         const gpu::Buffer& mark, const gpu::Buffer& dst,
                                         uint32_t stamp, uint32_t vertex_count, bool use_mirror) {
    if (!has_gpu_touched() || !src.handle || !mark.handle || !dst.handle) return;
    if (cap == 0 || vertex_count == 0) return;

    const bool mirror = use_mirror && mirror_map_ssbo.handle
                        && mirror_map_vertex_count == vertex_count;

    DirtyRegionGPU r = { base, cap, 0, 0 };
    gpu::write_buffer(gpu_dev, touched_region_ubo, 0, &r, sizeof(r));
    TouchedFoldParamsGPU u = { stamp, vertex_count, mirror ? 1u : 0u, 0 };
    gpu::write_buffer(gpu_dev, touched_fold_ubo, 0, &u, sizeof(u));

    // The source count lives on the GPU; dirty_args turns it into the workgroup count.
    const gpu::BindBufferEntry args_bg[] = {
        { BIND_DIRTY_VERTS,   &src,               src.size },
        { BIND_DISPATCH_ARGS, &touched_args_ssbo, touched_args_ssbo.size },
        { BIND_DIRTY_REGION,  &touched_region_ubo, sizeof(DirtyRegionGPU) },
    };
    gpu::BindGroup args_grp = gpu::create_bind_group(gpu_dev, dirty_args_pipeline, args_bg, 3);

    // The map binding must be live even when unused; the source list is read-only
    // here already, so it stands in without a usage clash.
    const gpu::Buffer& map = mirror ? mirror_map_ssbo : src;
    const gpu::BindBufferEntry bg[] = {
        { BIND_DIRTY_VERTS,  &src,                src.size },
        { BIND_MIRROR_MAP,   &map,                map.size },
        { BIND_NORM_MARK,    &mark,               mark.size },
        { BIND_NORM_LIST,    &dst,                dst.size },
        { BIND_DIRTY_REGION, &touched_region_ubo, sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       &touched_fold_ubo,   sizeof(TouchedFoldParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, touched_fold_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    // A zero-count region dispatches no groups and so skips the overflow check —
    // harmless, a region that received nothing cannot have overflowed.
    gpu::dispatch(b, dirty_args_pipeline, args_grp, 1);
    gpu::dispatch_indirect(b, touched_fold_pipeline, grp, touched_args_ssbo, 0);
    gpu::submit(b);
    gpu::release_bind_group(args_grp);
    gpu::release_bind_group(grp);
}

void ComputeState::begin_touched_stroke(uint32_t vertex_count) {
    if (!has_gpu_touched() || vertex_count == 0) return;
    // Sized to the whole mesh: the stamp dedupes, so the list never holds a vertex
    // twice. Grow-only; zeroed on (re)allocation, which is why the stamp starts at 1.
    if (touched_capacity < vertex_count || !touched_mark_ssbo.handle) {
        gpu::release_buffer(touched_mark_ssbo);
        gpu::release_buffer(touched_list_ssbo);
        touched_mark_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                               (uint64_t)vertex_count * sizeof(uint32_t),
                                               gpu::Usage::Storage);
        touched_list_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                               ((uint64_t)vertex_count + 2) * sizeof(uint32_t),
                                               gpu::Usage::Storage);
        gpu::clear_buffer(gpu_dev, touched_mark_ssbo, 0);
        touched_capacity = vertex_count;
        touched_stamp = 0;   // advanced to 1 below
    }
    if (++touched_stamp == 0) {
        gpu::clear_buffer(gpu_dev, touched_mark_ssbo, 0);
        touched_stamp = 1;
    }
    const uint32_t header[2] = { 0, 0 };
    gpu::write_buffer(gpu_dev, touched_list_ssbo, 0, header, sizeof(header));
    touched_vc = vertex_count;
}

void ComputeState::copy_touched_ids(const gpu::Buffer& dst, uint64_t dst_off, uint32_t count) {
    if (count == 0 || !touched_list_ssbo.handle || !dst.handle) return;
    gpu::copy_buffer(gpu_dev, touched_list_ssbo, 2 * sizeof(uint32_t), dst, dst_off,
                     (uint64_t)count * sizeof(uint32_t));
}

void ComputeState::load_touched_ids(const gpu::Buffer& src, uint64_t src_off, uint32_t count,
                                    uint32_t vertex_count) {
    if (count == 0 || count > vertex_count || !src.handle) return;
    begin_touched_stroke(vertex_count);
    if (!touched_list_ssbo.handle) return;
    const uint32_t header[2] = { 0, count };
    gpu::write_buffer(gpu_dev, touched_list_ssbo, 0, header, sizeof(header));
    gpu::copy_buffer(gpu_dev, src, src_off, touched_list_ssbo, 2 * sizeof(uint32_t),
                     (uint64_t)count * sizeof(uint32_t));
}

void ComputeState::ensure_dirty_verts(uint32_t count) {
    if (dirty_verts_ssbo.handle && count <= dirty_verts_capacity) return;
    uint32_t alloc_count = std::max(count, 4096u);
    gpu::release_buffer(dirty_verts_ssbo);
    dirty_verts_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                          (uint64_t)alloc_count * sizeof(uint32_t), gpu::Usage::Storage);
    dirty_verts_capacity = alloc_count;
}

// ---------------------------------------------------------------------------
// Stroke autosmooth methods
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to stroke_smooth_apply.{comp,wgsl}'s Params.
struct StrokeSmoothParamsGPU {
    uint32_t dirty_count; float strength; uint32_t _pad0; uint32_t _pad1;
};
static_assert(sizeof(StrokeSmoothParamsGPU) == 16, "stroke_smooth Params UBO must be 16 bytes");
}

bool ComputeState::init_stroke_smooth() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,      gpu::Bind::StorageRead,      0 },
        { BIND_MASK,             gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(StrokeSmoothParamsGPU) },
    };
    stroke_smooth_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                       gpu::embedded_shader("stroke_smooth_apply"), layout, 7);
    if (!stroke_smooth_apply_pipeline.handle) {
        std::printf("[compute] stroke_smooth_apply pipeline failed to compile\n");
        return false;
    }
    stroke_smooth_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(StrokeSmoothParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] stroke_smooth pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_stroke_smooth_apply(const uint32_t* vert_ids, uint32_t count,
                                                 float strength,
                                                 const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo) {
    if (!stroke_smooth_apply_pipeline.handle || count == 0 || !mask_ssbo.handle) return;

    // Upload the dirty-vert id list — seam-owned buffer (shared with compute_normals /
    // multires). A null list means the ids are already there (copied GPU-side from
    // the touched list), so there is nothing to upload.
    if (vert_ids) {
        ensure_dirty_verts(count);
        gpu::write_buffer(gpu_dev, dirty_verts_ssbo, 0, vert_ids, (uint64_t)count * sizeof(uint32_t));
    } else if (count > dirty_verts_capacity) {
        return;
    }

    StrokeSmoothParamsGPU u = {};
    u.dirty_count = count;
    u.strength = strength;
    gpu::write_buffer(gpu_dev, stroke_smooth_ubo, 0, &u, sizeof(u));

    // pos/index bound whole (kernel scatters over the dirty list). Real buffer sizes
    // from the seam-owned gpu::Buffers. Adjacency + dirty list + mask are seam-owned.
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,        &pos_vbo,   pos_vbo.size },
        { BIND_INDICES,          &index_ebo, index_ebo.size },
        { BIND_ADJACENCY_OFFSET, &adjacency_offset_ssbo, adjacency_offset_ssbo.size },
        { BIND_ADJACENCY_LIST,   &adjacency_list_ssbo,   adjacency_list_ssbo.size },
        { BIND_DIRTY_VERTS,      &dirty_verts_ssbo,      dirty_verts_ssbo.size },
        { BIND_MASK,             &mask_ssbo, mask_ssbo.size },
        { BIND_PARAMS,           &stroke_smooth_ubo, sizeof(StrokeSmoothParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, stroke_smooth_apply_pipeline, bg, 7);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, stroke_smooth_apply_pipeline, grp, (count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
