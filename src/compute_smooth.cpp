#include "compute.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Smooth brush shaders
// ---------------------------------------------------------------------------

static const char* smooth_accum_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf { float positions[]; };
layout(std430, binding = 3) buffer AccumBuf { uint accum[]; };
layout(std430, binding = 6) buffer DirtyBuf { uint dirty_count; uint dirty_ids[]; };

uniform vec3  u_anchor_pos;
uniform float u_world_radius;
uniform float u_hardness;
uniform int   u_mirror_x;
uniform uint  u_vertex_count;

float brush_falloff(float dist, float radius) {
    float t = dist / radius;
    float inner = 0.15 + u_hardness * 0.55;
    if (t <= inner) return 1.0;
    float blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    vec3 p = vec3(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    if (u_mirror_x != 0 && u_anchor_pos.x * p.x < 0.0) return;

    float d = distance(p, u_anchor_pos);
    if (d >= u_world_radius) return;

    float w = brush_falloff(d, u_world_radius);
    if (w <= 0.0) return;

    accum[v * 4u + 3u] = floatBitsToUint(w);
    uint idx = atomicAdd(dirty_count, 1u);
    dirty_ids[idx] = v;
}
)";

static const char* smooth_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf { float positions[]; };
layout(std430, binding = 3)  buffer AccumBuf { uint accum[]; };
layout(std430, binding = 2)  readonly buffer IdxBuf { uint indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffset { uint adj_offset[]; };
layout(std430, binding = 5)  readonly buffer AdjList { uint adj_list[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[]; };

uniform uint u_vertex_count;
uniform float u_strength;
uniform int u_iteration;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    float w = uintBitsToFloat(accum[v * 4u + 3u]);
    if (w <= 0.0) return;

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

    float inv_c = 1.0 / count;
    float blend = w * u_strength * mscale;
    positions[v * 3u]     += (sum_x * inv_c - cur_x) * blend;
    positions[v * 3u + 1u] += (sum_y * inv_c - cur_y) * blend;
    positions[v * 3u + 2u] += (sum_z * inv_c - cur_z) * blend;
}
)";

static const char* smooth_mirror_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf { float positions[]; };
layout(std430, binding = 3)  readonly buffer AccumBuf { uint accum[]; };
layout(std430, binding = 7)  readonly buffer MirrorBuf { uint mirror_map[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[]; };

uniform uint u_vertex_count;
uniform float u_anchor_x;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    float w = uintBitsToFloat(accum[v * 4u + 3u]);
    if (w <= 0.0) return;

    float vx = positions[v * 3u];
    if (vx * u_anchor_x < 0.0) return;

    uint mv = mirror_map[v];
    if (mv == v) return;

    float mscale = 1.0 - mask[mv];
    if (mscale <= 0.0) return;

    uint src = v * 3u;
    uint dst = mv * 3u;
    positions[dst + 0u] += (-positions[src + 0u] - positions[dst + 0u]) * mscale;
    positions[dst + 1u] += ( positions[src + 1u] - positions[dst + 1u]) * mscale;
    positions[dst + 2u] += ( positions[src + 2u] - positions[dst + 2u]) * mscale;
}
)";

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
    smooth_accum_program = compile_program(smooth_accum_src);
    if (!smooth_accum_program) {
        std::printf("[compute] smooth_accum shader failed to compile\n");
        return false;
    }
    smooth_apply_program = compile_program(smooth_apply_src);
    if (!smooth_apply_program) {
        std::printf("[compute] smooth_apply shader failed to compile\n");
        glDeleteProgram(smooth_accum_program);
        smooth_accum_program = 0;
        return false;
    }
    smooth_mirror_apply_program = compile_program(smooth_mirror_apply_src);
    if (!smooth_mirror_apply_program) {
        std::printf("[compute] smooth_mirror_apply shader failed to compile\n");
        glDeleteProgram(smooth_accum_program); smooth_accum_program = 0;
        glDeleteProgram(smooth_apply_program); smooth_apply_program = 0;
        return false;
    }
    std::printf("[compute] smooth shaders compiled\n");
    return true;
}

void ComputeState::dispatch_smooth_mirror_apply(GLuint pos_vbo, uint32_t vertex_count, float anchor_x) {
    if (!smooth_mirror_apply_program || mirror_map_vertex_count == 0) return;

    glUseProgram(smooth_mirror_apply_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MIRROR_MAP, mirror_map_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform1ui(glGetUniformLocation(smooth_mirror_apply_program, "u_vertex_count"), vertex_count);
    glUniform1f(glGetUniformLocation(smooth_mirror_apply_program, "u_anchor_x"), anchor_x);

    int groups = (vertex_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::dispatch_smooth(const SmoothAccumParams& p,
                                    GLuint /*triid_tex*/, GLuint /*bary_tex*/,
                                    GLuint pos_vbo, GLuint index_ebo) {
    clear_accum_buffer();

    ensure_smooth_dirty_buffer(p.vertex_count);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(smooth_accum_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, smooth_dirty_ssbo);

    glUniform3f(glGetUniformLocation(smooth_accum_program, "u_anchor_pos"),
                p.anchor_x, p.anchor_y, p.anchor_z);
    glUniform1f(glGetUniformLocation(smooth_accum_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(smooth_accum_program, "u_hardness"), p.hardness);
    glUniform1i(glGetUniformLocation(smooth_accum_program, "u_mirror_x"), p.mirror_x ? 1 : 0);
    glUniform1ui(glGetUniformLocation(smooth_accum_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(smooth_apply_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES, index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST, adjacency_list_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform1ui(glGetUniformLocation(smooth_apply_program, "u_vertex_count"), p.vertex_count);
    glUniform1f(glGetUniformLocation(smooth_apply_program, "u_strength"), p.strength);

    groups = (int)((p.vertex_count + 255u) / 256u);

    // Re-impose the mirror reflection after *each* Laplacian iteration, not just
    // once at the end. The accum pass only weights the anchor-side lobe (mirror
    // gate), so without this the opposite lobe stays frozen and a vert whose
    // 1-ring crosses x=0 averages against a stale, un-smoothed wall every pass —
    // the seam band under-relaxes and stands proud as a symmetric crease. By
    // reflecting after every iteration, the cross-seam neighbours are the fresh
    // mirror of the just-smoothed lobe and the seam relaxes in lockstep. Each
    // side stays a byte-exact reflection of the other.
    bool do_mirror = p.mirror_x && smooth_mirror_apply_program
                     && mirror_map_vertex_count == p.vertex_count;
    GLint iter_loc = glGetUniformLocation(smooth_apply_program, "u_iteration");
    for (int iter = 0; iter < p.iterations; iter++) {
        glUseProgram(smooth_apply_program);
        glUniform1i(iter_loc, iter);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        if (do_mirror) {
            dispatch_smooth_mirror_apply(pos_vbo, p.vertex_count, p.anchor_x);
        }
    }

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
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
