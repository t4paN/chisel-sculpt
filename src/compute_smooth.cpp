#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("smooth_accum" / ...)
#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Smooth brush — ported onto the gpu:: seam (Seam Step 2b). Three buffer-only
// kernels: smooth_accum (world-distance gate → accum.w + dirty list), smooth_apply
// (normal-projected Laplacian, looped), smooth_mirror_apply (re-impose the X mirror
// after each iteration). It is buffer-only — the triid/bary pick read is CPU-side
// back-projection in brush.cpp, NOT a compute input — so it needs no texture-bind
// seam work. Kernel logic lives in shaders/{glsl,wgsl}/smooth_*.* (embedded at build
// time). accum / dirty / mirror-map / mask buffers stay GL-owned (wrapped in views at
// dispatch); the accum clear + dirty-counter reset stay raw GL.
// ---------------------------------------------------------------------------

namespace {
// 32-byte std140 block, byte-identical to smooth_accum.{comp,wgsl}'s Params.
struct SmoothAccumParamsGPU {
    float    anchor[3];    float world_radius;   // 16
    float    hardness;     uint32_t mirror_x;
    uint32_t vertex_count; uint32_t _pad0;       // 16
};
static_assert(sizeof(SmoothAccumParamsGPU) == 32, "smooth accum Params UBO must be 32 bytes");

// 16-byte std140 block, byte-identical to smooth_apply.{comp,wgsl}'s Params.
struct SmoothApplyParamsGPU {
    uint32_t vertex_count; float strength; uint32_t mirror_x; float seam_band;
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
        { BIND_POSITIONS,   gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,       gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(SmoothAccumParamsGPU) },
    };
    smooth_accum_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("smooth_accum"), accum_layout, 4);
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
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(SmoothApplyParamsGPU) },
    };
    smooth_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("smooth_apply"), apply_layout, 7);
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
        { BIND_PARAMS,     gpu::Bind::Uniform,          sizeof(SmoothMirrorParamsGPU) },
    };
    smooth_mirror_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                       gpu::embedded_shader("smooth_mirror_apply"), mirror_layout, 5);
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

void ComputeState::dispatch_smooth_mirror_apply(GLuint pos_vbo, uint32_t vertex_count, float anchor_x) {
    if (!smooth_mirror_apply_pipeline.handle || mirror_map_vertex_count == 0 || !mask_ssbo) return;
    const uint32_t vc = vertex_count;

    SmoothMirrorParamsGPU u = {};
    u.vertex_count = vc;
    u.anchor_x = anchor_x;
    gpu::write_buffer(gpu_dev, smooth_mirror_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{   (uint64_t)vc * 3u * sizeof(float),       pos_vbo };
    gpu::Buffer accumView{ (uint64_t)vc * 4u * sizeof(uint32_t),    accum_ssbo };
    gpu::Buffer mirrorView{(uint64_t)mirror_map_vertex_count * sizeof(uint32_t), mirror_map_ssbo };
    gpu::Buffer maskView{  (uint64_t)vc * sizeof(float),            mask_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,  &posView,    posView.size },
        { BIND_ACCUM,      &accumView,  accumView.size },
        { BIND_MIRROR_MAP, &mirrorView, mirrorView.size },
        { BIND_MASK,       &maskView,   maskView.size },
        { BIND_PARAMS,     &smooth_mirror_ubo, sizeof(SmoothMirrorParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, smooth_mirror_apply_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, smooth_mirror_apply_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_smooth(const SmoothAccumParams& p,
                                    GLuint /*triid_tex*/, GLuint /*bary_tex*/,
                                    GLuint pos_vbo, GLuint index_ebo) {
    if (!has_smooth() || !mask_ssbo) return;
    const uint32_t vc = p.vertex_count;

    // accum clear + dirty-counter reset stay raw GL (GL-owned buffers).
    clear_accum_buffer();
    ensure_smooth_dirty_buffer(vc);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ---- Pass 1: accum (world-distance gate → accum.w + dirty list) ----
    SmoothAccumParamsGPU ua = {};
    ua.anchor[0] = p.anchor_x; ua.anchor[1] = p.anchor_y; ua.anchor[2] = p.anchor_z;
    ua.world_radius = p.world_radius;
    ua.hardness = p.hardness;
    ua.mirror_x = p.mirror_x ? 1u : 0u;
    ua.vertex_count = vc;
    gpu::write_buffer(gpu_dev, smooth_accum_ubo, 0, &ua, sizeof(ua));

    gpu::Buffer posView{   (uint64_t)vc * 3u * sizeof(float),                pos_vbo };
    gpu::Buffer accumView{ (uint64_t)vc * 4u * sizeof(uint32_t),            accum_ssbo };
    gpu::Buffer dirtyView{ (uint64_t)(1u + smooth_dirty_capacity) * sizeof(uint32_t), smooth_dirty_ssbo };
    {
        const gpu::BindBufferEntry bg[] = {
            { BIND_POSITIONS,   &posView,   posView.size },
            { BIND_ACCUM,       &accumView, accumView.size },
            { BIND_DIRTY_VERTS, &dirtyView, dirtyView.size },
            { BIND_PARAMS,      &smooth_accum_ubo, sizeof(SmoothAccumParamsGPU) },
        };
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, smooth_accum_pipeline, bg, 4);
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, smooth_accum_pipeline, grp, (vc + 255u) / 256u);
        gpu::submit(b);
        gpu::release_bind_group(grp);
    }

    // ---- Pass 2: apply (normal-projected Laplacian), looped ----
    // Region-blend seam band: within this distance of x=0 the kernel fades back to
    // the full uniform Laplacian so the seam relaxes instead of pinching. Half the
    // brush radius is a starting point; tune by feel.
    SmoothApplyParamsGPU up = {};
    up.vertex_count = vc;
    up.strength = p.strength;
    up.mirror_x = p.mirror_x ? 1u : 0u;
    up.seam_band = p.world_radius * 0.5f;
    gpu::write_buffer(gpu_dev, smooth_apply_ubo, 0, &up, sizeof(up));

    gpu::Buffer idxView{  0,                                    index_ebo };
    gpu::Buffer offView{  0,                                    adjacency_offset_ssbo };
    gpu::Buffer listView{ 0,                                    adjacency_list_ssbo };
    gpu::Buffer maskView{ (uint64_t)vc * sizeof(float),         mask_ssbo };
    const gpu::BindBufferEntry apply_bg[] = {
        { BIND_POSITIONS,        &posView,   posView.size },
        { BIND_INDICES,          &idxView,   idxView.size },
        { BIND_ACCUM,            &accumView, accumView.size },
        { BIND_ADJACENCY_OFFSET, &offView,   offView.size },
        { BIND_ADJACENCY_LIST,   &listView,  listView.size },
        { BIND_MASK,             &maskView,  maskView.size },
        { BIND_PARAMS,           &smooth_apply_ubo, sizeof(SmoothApplyParamsGPU) },
    };
    gpu::BindGroup apply_grp = gpu::create_bind_group(gpu_dev, smooth_apply_pipeline, apply_bg, 7);

    // Re-impose the mirror reflection after *each* Laplacian iteration (not just at
    // the end): the accum pass only weights the anchor-side lobe, so without this a
    // vert whose 1-ring crosses x=0 averages against a stale wall every pass and the
    // seam stands proud as a symmetric crease. Reflecting each iteration keeps the
    // cross-seam neighbours the fresh mirror of the just-smoothed lobe.
    bool do_mirror = p.mirror_x && smooth_mirror_apply_pipeline.handle
                     && mirror_map_vertex_count == vc;
    const uint32_t groups = (vc + 255u) / 256u;
    for (int iter = 0; iter < p.iterations; iter++) {
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, smooth_apply_pipeline, apply_grp, groups);
        gpu::submit(b);

        if (do_mirror) {
            dispatch_smooth_mirror_apply(pos_vbo, vc, p.anchor_x);
        }
    }
    gpu::release_bind_group(apply_grp);
}

void ComputeState::ensure_smooth_dirty_buffer(uint32_t max_verts) {
    if (smooth_dirty_ssbo && max_verts <= smooth_dirty_capacity) return;

    if (smooth_dirty_ssbo) glDeleteBuffers(1, &smooth_dirty_ssbo);
    glGenBuffers(1, &smooth_dirty_ssbo);
    uint32_t alloc = std::max(max_verts, 4096u);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (alloc + 1) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    smooth_dirty_capacity = alloc;
}

uint32_t ComputeState::readback_smooth_dirty(std::vector<uint32_t>& out) {
    out.clear();
    if (!smooth_dirty_ssbo) return 0;

    uint32_t count = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &count);

    if (count > 0) {
        if (count > smooth_dirty_capacity) count = smooth_dirty_capacity;
        out.resize(count);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t),
                           count * sizeof(uint32_t), out.data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return count;
}

uint32_t ComputeState::readback_accum_dirty(uint32_t vertex_count, std::vector<uint32_t>& out) {
    out.clear();
    if (!accum_ssbo || vertex_count == 0) return 0;

    GLsizeiptr size = (GLsizeiptr)vertex_count * 4 * sizeof(uint32_t);
    if (readback_buf.size() < vertex_count * 4) {
        readback_buf.resize(vertex_count * 4);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

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

bool ComputeState::init_compute_normals() {
    if (!supported) return false;
    compute_normals_program = compile_program(compute_normals_src);
    if (!compute_normals_program) {
        std::printf("[compute] compute_normals shader failed to compile\n");
        return false;
    }
    std::printf("[compute] compute_normals shader compiled\n");
    return true;
}

void ComputeState::upload_adjacency(const uint32_t* offsets, uint32_t offset_count,
                                     const uint32_t* list, uint32_t list_count) {
    if (!adjacency_offset_ssbo) glGenBuffers(1, &adjacency_offset_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, adjacency_offset_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, offset_count * sizeof(uint32_t), offsets, GL_STATIC_DRAW);

    if (!adjacency_list_ssbo) glGenBuffers(1, &adjacency_list_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, adjacency_list_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, list_count * sizeof(uint32_t), list, GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    adjacency_vertex_count = offset_count - 1;

    std::printf("[compute] adjacency uploaded: %u verts, %u entries\n",
                adjacency_vertex_count, list_count);
}

void ComputeState::dispatch_compute_normals(const uint32_t* dirty_verts, uint32_t dirty_count,
                                             GLuint pos_vbo, GLuint norm_vbo, GLuint index_ebo) {
    if (!compute_normals_program || dirty_count == 0) return;

    GLsizeiptr needed = (GLsizeiptr)dirty_count * sizeof(uint32_t);
    if (!dirty_verts_ssbo || dirty_count > dirty_verts_capacity) {
        if (dirty_verts_ssbo) glDeleteBuffers(1, &dirty_verts_ssbo);
        glGenBuffers(1, &dirty_verts_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
        uint32_t alloc_count = std::max(dirty_count, 4096u);
        glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_count * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        dirty_verts_capacity = alloc_count;
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, needed, dirty_verts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(compute_normals_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS, norm_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES, index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST, adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, dirty_verts_ssbo);

    glUniform1ui(glGetUniformLocation(compute_normals_program, "u_dirty_count"), dirty_count);

    int groups = (dirty_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---------------------------------------------------------------------------
// Stroke autosmooth methods
// ---------------------------------------------------------------------------

bool ComputeState::init_stroke_smooth() {
    if (!supported) return false;
    stroke_smooth_apply_program = compile_program(stroke_smooth_apply_src);
    if (!stroke_smooth_apply_program) {
        std::printf("[compute] stroke_smooth_apply shader failed to compile\n");
        return false;
    }
    std::printf("[compute] stroke_smooth shader compiled\n");
    return true;
}

void ComputeState::dispatch_stroke_smooth_apply(const uint32_t* vert_ids, uint32_t count,
                                                 float strength,
                                                 GLuint pos_vbo, GLuint index_ebo) {
    if (!stroke_smooth_apply_program || count == 0) return;

    GLsizeiptr needed = (GLsizeiptr)count * sizeof(uint32_t);
    if (!dirty_verts_ssbo || count > dirty_verts_capacity) {
        if (dirty_verts_ssbo) glDeleteBuffers(1, &dirty_verts_ssbo);
        glGenBuffers(1, &dirty_verts_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
        uint32_t alloc_count = std::max(count, 4096u);
        glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_count * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        dirty_verts_capacity = alloc_count;
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, needed, vert_ids);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(stroke_smooth_apply_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES, index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST, adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, dirty_verts_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform1ui(glGetUniformLocation(stroke_smooth_apply_program, "u_dirty_count"), count);
    glUniform1f(glGetUniformLocation(stroke_smooth_apply_program, "u_strength"), strength);

    int groups = (count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}
