#include "compute.h"
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Remesh per-tri selection shaders
// ---------------------------------------------------------------------------

static const char* select_stretched_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 13) readonly buffer InPosBuf { float pos[];     };
layout(std430, binding = 2)  readonly buffer IdxBuf   { uint  indices[]; };
layout(std430, binding = 16) buffer TriSelBuf { uint tri_sel[]; };

uniform float u_target_len;
uniform uint  u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    uint i0 = indices[t*3u+0u];
    uint i1 = indices[t*3u+1u];
    uint i2 = indices[t*3u+2u];

    vec3 p0 = vec3(pos[i0*3u], pos[i0*3u+1u], pos[i0*3u+2u]);
    vec3 p1 = vec3(pos[i1*3u], pos[i1*3u+1u], pos[i1*3u+2u]);
    vec3 p2 = vec3(pos[i2*3u], pos[i2*3u+1u], pos[i2*3u+2u]);

    float e01 = length(p1 - p0);
    float e12 = length(p2 - p1);
    float e20 = length(p0 - p2);

    float longest  = max(max(e01, e12), e20);
    float shortest = min(min(e01, e12), e20);

    // Sliver test: smallest angle below ~15° (cos > 0.966). Law of cosines on
    // all three corners; clamp guards against float drift past ±1. The
    // length-ratio test alone catches most slivers but misses tris where all
    // three edges fall under target_len yet the shape is still degenerate
    // (e.g. one corner almost-collinear in a dense region).
    bool sliver = false;
    if (shortest > 1e-8) {
        float e01_2 = e01*e01, e12_2 = e12*e12, e20_2 = e20*e20;
        float c0 = clamp((e12_2 + e20_2 - e01_2) / (2.0 * e12 * e20), -1.0, 1.0);
        float c1 = clamp((e01_2 + e20_2 - e12_2) / (2.0 * e01 * e20), -1.0, 1.0);
        float c2 = clamp((e01_2 + e12_2 - e20_2) / (2.0 * e01 * e12), -1.0, 1.0);
        float max_cos = max(max(c0, c1), c2);
        sliver = max_cos > 0.966;  // cos(15°)
    }

    bool sel = longest > 1.2 * u_target_len ||
               (shortest > 1e-8 && longest / shortest > 1.5) ||
               sliver;
    tri_sel[t] = sel ? 1u : 0u;
}
)";

static const char* select_unmasked_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 14) readonly buffer MaskBuf { float mask[];    };
layout(std430, binding = 2)  readonly buffer IdxBuf  { uint  indices[]; };
layout(std430, binding = 16) buffer TriSelBuf { uint tri_sel[]; };

uniform uint u_tri_count;
uniform uint u_mask_size;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    uint i0 = indices[t*3u+0u];
    uint i1 = indices[t*3u+1u];
    uint i2 = indices[t*3u+2u];

    bool u0 = (u_mask_size == 0u || i0 >= u_mask_size || mask[i0] < 0.5);
    bool u1 = (u_mask_size == 0u || i1 >= u_mask_size || mask[i1] < 0.5);
    bool u2 = (u_mask_size == 0u || i2 >= u_mask_size || mask[i2] < 0.5);

    tri_sel[t] = (u0 && u1 && u2) ? 1u : 0u;
}
)";

// ---------------------------------------------------------------------------
// Remesh per-tri ring-grow selection (BFS by shared vertex, one ring/dispatch)
// ---------------------------------------------------------------------------

static const char* grow_selection_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 18) readonly buffer InSelBuf  { uint in_sel[];  };
layout(std430, binding = 16)          buffer OutSelBuf { uint out_sel[]; };
layout(std430, binding = 2)  readonly buffer IdxBuf    { uint indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffBuf { uint adj_off[]; };
layout(std430, binding = 5)  readonly buffer AdjListBuf{ uint adj_list[];};

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (in_sel[t] != 0u) return;        // already in selection, leave the copy

    uint v0 = indices[t*3u + 0u];
    uint v1 = indices[t*3u + 1u];
    uint v2 = indices[t*3u + 2u];
    uint vs[3] = uint[3](v0, v1, v2);
    for (int i = 0; i < 3; i++) {
        uint v = vs[i];
        uint s = adj_off[v];
        uint e = adj_off[v + 1u];
        for (uint j = s; j < e; j++) {
            if (in_sel[adj_list[j]] != 0u) {
                out_sel[t] = 1u;
                return;
            }
        }
    }
}
)";

// ---------------------------------------------------------------------------
// Remesh per-tri mirror-selection spread (OR the symmetric tris in)
// ---------------------------------------------------------------------------

static const char* mirror_selection_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 18) readonly buffer InSelBuf  { uint in_sel[];   };
layout(std430, binding = 16)          buffer OutSelBuf { uint out_sel[];  };
layout(std430, binding = 2)  readonly buffer IdxBuf    { uint indices[];  };
layout(std430, binding = 4)  readonly buffer AdjOffBuf { uint adj_off[];  };
layout(std430, binding = 5)  readonly buffer AdjListBuf{ uint adj_list[]; };
layout(std430, binding = 7)  readonly buffer MirrorBuf { uint mirror[];   };

uniform uint u_tri_count;
uniform uint u_vertex_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (in_sel[t] == 0u) return;        // only selected tris seed mirror spread

    uint v0 = indices[t*3u + 0u];
    uint v1 = indices[t*3u + 1u];
    uint v2 = indices[t*3u + 2u];
    uint vs[3] = uint[3](v0, v1, v2);
    for (int i = 0; i < 3; i++) {
        uint v = vs[i];
        if (v >= u_vertex_count) continue;
        uint mv = mirror[v];
        if (mv == v || mv >= u_vertex_count) continue;
        uint s = adj_off[mv];
        uint e = adj_off[mv + 1u];
        for (uint j = s; j < e; j++) {
            atomicOr(out_sel[adj_list[j]], 1u);
        }
    }
}
)";

// ---------------------------------------------------------------------------
// Remesh per-vertex pinned-boundary detection
// ---------------------------------------------------------------------------

static const char* find_pinned_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 13) readonly buffer InPosBuf   { float pos[];     };
layout(std430, binding = 4)  readonly buffer AdjOffBuf  { uint  adj_off[]; };
layout(std430, binding = 5)  readonly buffer AdjListBuf { uint  adj_list[];};
layout(std430, binding = 16) readonly buffer TriSelBuf  { uint  tri_sel[]; };
layout(std430, binding = 15) buffer PinnedBuf { uint pinned[]; };

uniform uint  u_vertex_count;
uniform float u_seam_tol;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    if (abs(pos[v*3u]) < u_seam_tol) {
        pinned[v] = 1u;
        return;
    }

    uint t_start = adj_off[v];
    uint t_end   = adj_off[v + 1u];
    bool has_sel = false, has_unsel = false;
    for (uint j = t_start; j < t_end; j++) {
        if (tri_sel[adj_list[j]] != 0u) has_sel   = true;
        else                            has_unsel  = true;
        if (has_sel && has_unsel) { pinned[v] = 1u; return; }
    }
    pinned[v] = 0u;
}
)";

// ---------------------------------------------------------------------------
// Remesh per-vertex smooth weight computation
// ---------------------------------------------------------------------------

static const char* smooth_weights_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 4)  readonly buffer AdjOffBuf  { uint  adj_off[];  };
layout(std430, binding = 5)  readonly buffer AdjListBuf { uint  adj_list[]; };
layout(std430, binding = 14) buffer          WeightBuf  { float weights[];  };
layout(std430, binding = 15) readonly buffer PinnedBuf  { uint  pinned[];   };
layout(std430, binding = 16) readonly buffer FullSelBuf { uint  full_sel[]; };
layout(std430, binding = 17) readonly buffer CoreSelBuf { uint  core_sel[]; };

uniform uint u_vertex_count;
uniform int  u_support_rings;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    if (pinned[v] != 0u) { weights[v] = 0.0; return; }

    uint t_start = adj_off[v];
    uint t_end   = adj_off[v + 1u];

    bool any_full = false, all_core = true, any_core = false;
    for (uint j = t_start; j < t_end; j++) {
        uint t = adj_list[j];
        if (full_sel[t] != 0u) any_full  = true;
        if (core_sel[t] != 0u) any_core  = true;
        else                   all_core  = false;
    }

    if (!any_full) { weights[v] = 0.0; return; }

    if (all_core) {
        weights[v] = 1.0;
    } else {
        float ring = any_core ? 1.0 : 2.0;
        weights[v] = max(0.0, 1.0 - ring / float(u_support_rings + 1));
    }
}
)";

// ---------------------------------------------------------------------------
// Remesh tangential smooth (GPU ping-pong)
// ---------------------------------------------------------------------------

static const char* remesh_smooth_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer    OutPosBuf  { float out_pos[];  };
layout(std430, binding = 1)  readonly buffer NormBuf    { float normals[]; };
layout(std430, binding = 2)  readonly buffer IdxBuf     { uint  indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffBuf  { uint  adj_off[];  };
layout(std430, binding = 5)  readonly buffer AdjListBuf { uint  adj_list[]; };
layout(std430, binding = 13) readonly buffer InPosBuf   { float in_pos[];   };
layout(std430, binding = 14) readonly buffer WeightBuf  { float weights[];  };
layout(std430, binding = 15) readonly buffer PinnedBuf  { uint  pinned[];   };
layout(std430, binding = 16) readonly buffer TriSelBuf  { uint  tri_sel[];  };

uniform float u_lambda;
uniform float u_seam_tol;
uniform uint  u_vertex_count;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    out_pos[v*3u+0u] = in_pos[v*3u+0u];
    out_pos[v*3u+1u] = in_pos[v*3u+1u];
    out_pos[v*3u+2u] = in_pos[v*3u+2u];

    if (pinned[v] != 0u) return;

    float w = weights[v];
    if (w < 1e-6) return;

    uint t_start = adj_off[v];
    uint t_end   = adj_off[v + 1u];

    bool in_region = false;
    for (uint j = t_start; j < t_end; j++) {
        if (tri_sel[adj_list[j]] != 0u) { in_region = true; break; }
    }
    if (!in_region) return;

    uint nbrs[48];
    uint nbr_count = 0u;
    for (uint j = t_start; j < t_end; j++) {
        uint t = adj_list[j];
        for (int k = 0; k < 3; k++) {
            uint n = indices[t*3u + uint(k)];
            if (n == v) continue;
            bool found = false;
            for (uint l = 0u; l < nbr_count; l++) {
                if (nbrs[l] == n) { found = true; break; }
            }
            if (!found && nbr_count < 48u) nbrs[nbr_count++] = n;
        }
    }
    if (nbr_count == 0u) return;

    vec3 nbr_centroid = vec3(0.0);
    for (uint l = 0u; l < nbr_count; l++) {
        nbr_centroid.x += in_pos[nbrs[l]*3u+0u];
        nbr_centroid.y += in_pos[nbrs[l]*3u+1u];
        nbr_centroid.z += in_pos[nbrs[l]*3u+2u];
    }
    nbr_centroid /= float(nbr_count);

    vec3 pos = vec3(in_pos[v*3u+0u], in_pos[v*3u+1u], in_pos[v*3u+2u]);
    // A degenerate (zero-length) vertex normal has no tangent plane to project
    // onto. normalize() of a zero vector is 0/0 -> NaN in GLSL, which then leaks
    // into out_pos.y/z (the x slot is masked by the sign()/max() seam clamp) and
    // poisons mean-edge -> the post-remesh mirror rebuild goes tol=NaN, 0 paired,
    // shredding the model. Bail out leaving the vertex put when that happens.
    vec3 nrm_raw = vec3(normals[v*3u+0u], normals[v*3u+1u], normals[v*3u+2u]);
    float nrm_len = length(nrm_raw);
    if (nrm_len < 1e-12) {
        out_pos[v*3u+0u] = pos.x;
        out_pos[v*3u+1u] = pos.y;
        out_pos[v*3u+2u] = pos.z;
        return;
    }
    vec3 nrm = nrm_raw / nrm_len;
    vec3 d   = nbr_centroid - pos;
    vec3 td  = d - nrm * dot(d, nrm);

    float new_x = pos.x + td.x * (u_lambda * w);
    // Non-pinned verts are outside the seam band by construction (find_pinned
    // pins everything with |x| < seam_tol). Keep them there: don't allow
    // smoothing to drift |x| into the band or flip sides. Otherwise a vert
    // can wander across x=0 over multiple iters and leave a one-sided seam.
    new_x = sign(pos.x) * max(u_seam_tol, abs(new_x));

    out_pos[v*3u+0u] = new_x;
    out_pos[v*3u+1u] = pos.y + td.y * (u_lambda * w);
    out_pos[v*3u+2u] = pos.z + td.z * (u_lambda * w);
}
)";

// ---------------------------------------------------------------------------
// Post-remesh seam snap: snap near-x=0 verts using topological detection
// ---------------------------------------------------------------------------

static const char* seam_snap_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 13) buffer PosBuf     { float pos[];     };
layout(std430, binding = 4)  readonly buffer AdjOffBuf  { uint  adj_off[]; };
layout(std430, binding = 5)  readonly buffer AdjListBuf { uint  adj_list[];};
layout(std430, binding = 2)  readonly buffer IdxBuf     { uint  indices[]; };
layout(std430, binding = 12) readonly buffer MaskBuf    { float mask[];    };

uniform uint  u_vertex_count;
uniform uint  u_mask_size;
uniform float u_seam_tol;
uniform float u_snap_tol;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    // Skip fully masked
    if (u_mask_size > 0u && v < u_mask_size && mask[v] >= 1.0)
        return;

    float ax = abs(pos[v * 3u]);

    // Tight snap: already within seam_tol
    if (ax < u_seam_tol) {
        pos[v * 3u] = 0.0;
        return;
    }

    // Outside snap zone
    if (ax >= u_snap_tol) return;

    // Topological test: neighbors on both +x and -x sides
    bool has_pos = false, has_neg = false;
    uint t_start = adj_off[v];
    uint t_end   = adj_off[v + 1u];
    for (uint j = t_start; j < t_end && !(has_pos && has_neg); j++) {
        uint tri = adj_list[j];
        for (uint k = 0u; k < 3u; k++) {
            uint nv = indices[tri * 3u + k];
            if (nv == v) continue;
            float nx = pos[nv * 3u];
            if (nx > u_seam_tol) has_pos = true;
            else if (nx < -u_seam_tol) has_neg = true;
        }
    }

    if (has_pos && has_neg)
        pos[v * 3u] = 0.0;
}
)";

// ---------------------------------------------------------------------------
// Post-remesh seam weld: merge spatially-close verts at x=0
// ---------------------------------------------------------------------------

static const char* seam_weld_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 13) readonly buffer PosBuf  { float pos[];       };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[];      };
layout(std430, binding = 19) buffer MergeMapBuf      { uint  merge_map[]; };

uniform uint  u_vertex_count;
uniform uint  u_mask_size;
uniform float u_weld_tol;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    merge_map[v] = v;

    // Only process verts sitting on seam (x == 0)
    if (pos[v * 3u] != 0.0) return;

    // Skip fully masked
    if (u_mask_size > 0u && v < u_mask_size && mask[v] >= 1.0)
        return;

    float py = pos[v * 3u + 1u];
    float pz = pos[v * 3u + 2u];
    float weld_sq = u_weld_tol * u_weld_tol;

    uint best = v;
    for (uint u = 0u; u < v; u++) {
        if (pos[u * 3u] != 0.0) continue;
        if (u_mask_size > 0u && u < u_mask_size && mask[u] >= 1.0)
            continue;
        float dy = pos[u * 3u + 1u] - py;
        float dz = pos[u * 3u + 2u] - pz;
        float d2 = dy * dy + dz * dz;
        if (d2 < weld_sq && u < best)
            best = u;
    }
    merge_map[v] = best;
}
)";

// ---------------------------------------------------------------------------
// Remesh methods
// ---------------------------------------------------------------------------

bool ComputeState::init_remesh_select() {
    remesh_select_stretched_program = compile_program(select_stretched_src);
    remesh_select_unmasked_program  = compile_program(select_unmasked_src);
    if (!remesh_select_stretched_program || !remesh_select_unmasked_program) {
        std::fprintf(stderr, "[compute] remesh select shaders failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_grow_selection() {
    remesh_grow_selection_program = compile_program(grow_selection_src);
    if (!remesh_grow_selection_program) {
        std::fprintf(stderr, "[compute] remesh grow_selection shader failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_mirror_selection() {
    remesh_mirror_selection_program = compile_program(mirror_selection_src);
    if (!remesh_mirror_selection_program) {
        std::fprintf(stderr, "[compute] remesh mirror_selection shader failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_find_pinned() {
    remesh_find_pinned_program = compile_program(find_pinned_src);
    if (!remesh_find_pinned_program) {
        std::fprintf(stderr, "[compute] remesh find_pinned shader failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_smooth_weights() {
    remesh_smooth_weights_program = compile_program(smooth_weights_src);
    if (!remesh_smooth_weights_program) {
        std::fprintf(stderr, "[compute] remesh smooth_weights shader failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_smooth() {
    remesh_smooth_program = compile_program(remesh_smooth_src);
    if (!remesh_smooth_program) {
        std::fprintf(stderr, "[compute] remesh smooth shader failed\n");
        return false;
    }
    return true;
}

bool ComputeState::init_remesh_seam_snap_weld() {
    remesh_seam_snap_program = compile_program(seam_snap_src);
    remesh_seam_weld_program = compile_program(seam_weld_src);
    if (!remesh_seam_snap_program || !remesh_seam_weld_program) {
        std::fprintf(stderr, "[compute] remesh seam snap/weld shaders failed\n");
        return false;
    }
    return true;
}

void ComputeState::ensure_remesh_smooth_buffers(uint32_t vc, uint32_t tc) {
    auto alloc_ssbo = [](GLuint& ssbo, GLsizeiptr bytes) {
        if (!ssbo) glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
    };
    if (vc > remesh_vert_capacity) {
        remesh_vert_capacity = vc + vc / 4 + 256;
        alloc_ssbo(remesh_ping_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_pong_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_norm_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_weights_ssbo, (GLsizeiptr)remesh_vert_capacity     * sizeof(float));
        alloc_ssbo(remesh_pinned_ssbo,  (GLsizeiptr)remesh_vert_capacity     * sizeof(uint32_t));
    }
    if (tc > remesh_tri_capacity) {
        remesh_tri_capacity = tc + tc / 4 + 256;
        alloc_ssbo(remesh_trisel_ssbo,     (GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
        alloc_ssbo(remesh_indices_ssbo,    (GLsizeiptr)remesh_tri_capacity * 3 * sizeof(uint32_t));
        alloc_ssbo(remesh_core_sel_ssbo,   (GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
        alloc_ssbo(remesh_trisel_pong_ssbo,(GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_select_stretched(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float target_len)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);

    glUseProgram(remesh_select_stretched_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_IN_POS,  remesh_ping_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,        remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,  remesh_trisel_ssbo);

    glUniform1f(glGetUniformLocation(remesh_select_stretched_program, "u_target_len"), target_len);
    glUniform1ui(glGetUniformLocation(remesh_select_stretched_program, "u_tri_count"), tri_count);

    glDispatchCompute((tri_count + 255u) / 256u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_select_unmasked(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* mask, uint32_t mask_size)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);

    if (mask && mask_size > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mask_size*sizeof(float), mask);
    }

    glUseProgram(remesh_select_unmasked_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_WEIGHTS, remesh_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,        remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,  remesh_trisel_ssbo);

    glUniform1ui(glGetUniformLocation(remesh_select_unmasked_program, "u_tri_count"),  tri_count);
    glUniform1ui(glGetUniformLocation(remesh_select_unmasked_program, "u_mask_size"),  mask_size);

    glDispatchCompute((tri_count + 255u) / 256u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_grow_selection(
    uint32_t vertex_count, uint32_t tri_count, int rings)
{
    if (rings <= 0 || tri_count == 0 || !remesh_grow_selection_program) return;
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    glUseProgram(remesh_grow_selection_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,             remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET,    adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,      adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,       remesh_trisel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL_PONG,  remesh_trisel_pong_ssbo);
    glUniform1ui(glGetUniformLocation(remesh_grow_selection_program, "u_tri_count"), tri_count);

    GLuint groups = (tri_count + 255u) / 256u;
    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);

    for (int r = 0; r < rings; r++) {
        // Snapshot current selection into pong (the kernel's input).
        glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_trisel_pong_ssbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::dispatch_mirror_selection(uint32_t vertex_count, uint32_t tri_count)
{
    if (tri_count == 0 || !remesh_mirror_selection_program) return;
    if (mirror_map_vertex_count == 0 || !mirror_map_ssbo) return;
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);
    glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_trisel_pong_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(remesh_mirror_selection_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,             remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET,    adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,      adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MIRROR_MAP,          mirror_map_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,       remesh_trisel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL_PONG,  remesh_trisel_pong_ssbo);
    glUniform1ui(glGetUniformLocation(remesh_mirror_selection_program, "u_tri_count"),    tri_count);
    glUniform1ui(glGetUniformLocation(remesh_mirror_selection_program, "u_vertex_count"), vertex_count);

    glDispatchCompute((tri_count + 255u) / 256u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::snapshot_core_sel(uint32_t tri_count)
{
    if (tri_count == 0) return;
    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);
    glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_core_sel_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::readback_trisel(uint32_t tri_count, std::vector<uint32_t>& out)
{
    out.resize(tri_count);
    if (tri_count == 0) return;
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_trisel_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*sizeof(uint32_t), out.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_find_pinned(
    uint32_t vertex_count, uint32_t tri_count,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float seam_tol,
    std::vector<uint32_t>& out_pinned)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    glUseProgram(remesh_find_pinned_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_IN_POS,         remesh_ping_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET,      adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,        adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,         remesh_trisel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_PINNED,         remesh_pinned_ssbo);

    glUniform1ui(glGetUniformLocation(remesh_find_pinned_program, "u_vertex_count"), vertex_count);
    glUniform1f(glGetUniformLocation(remesh_find_pinned_program,  "u_seam_tol"),     seam_tol);

    glDispatchCompute((vertex_count + 255u) / 256u, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    out_pinned.resize(vertex_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_pinned_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*sizeof(uint32_t), out_pinned.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_compute_smooth_weights(
    uint32_t vertex_count, uint32_t tri_count,
    int support_rings)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    glUseProgram(remesh_smooth_weights_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET,  adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,    adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_WEIGHTS,    remesh_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_PINNED,     remesh_pinned_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL,     remesh_trisel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_CORE_SEL,   remesh_core_sel_ssbo);

    glUniform1ui(glGetUniformLocation(remesh_smooth_weights_program, "u_vertex_count"), vertex_count);
    glUniform1i (glGetUniformLocation(remesh_smooth_weights_program, "u_support_rings"), support_rings);

    glDispatchCompute((vertex_count + 255u) / 256u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::dispatch_remesh_smooth(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* norm_x, const float* norm_y, const float* norm_z,
    const std::vector<float>& weights,
    const std::vector<uint32_t>& pinned,
    float lambda,
    float seam_tol,
    float* out_pos_x, float* out_pos_y, float* out_pos_z,
    bool weights_on_gpu)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);

    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = norm_x[i];
        readback_buf[i*3+1] = norm_y[i];
        readback_buf[i*3+2] = norm_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_norm_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    if (!weights_on_gpu) {
        std::vector<float> w_buf(vertex_count, 1.0f);
        if (!weights.empty()) {
            uint32_t n = (uint32_t)weights.size();
            for (uint32_t i = 0; i < vertex_count; i++)
                w_buf[i] = (i < n) ? weights[i] : 1.0f;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*sizeof(float), w_buf.data());
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_pinned_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    std::min<uint32_t>(vertex_count, (uint32_t)pinned.size()) * sizeof(uint32_t),
                    pinned.data());

    // tri_sel is assumed already in remesh_trisel_ssbo (set by select+grow+mirror).

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);

    glUseProgram(remesh_smooth_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS,       remesh_norm_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,       remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_WEIGHTS,remesh_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_PINNED, remesh_pinned_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_TRISEL, remesh_trisel_ssbo);

    glUniform1f(glGetUniformLocation(remesh_smooth_program, "u_lambda"), lambda);
    glUniform1f(glGetUniformLocation(remesh_smooth_program, "u_seam_tol"), seam_tol);
    glUniform1ui(glGetUniformLocation(remesh_smooth_program, "u_vertex_count"), vertex_count);

    // Multi-step smoothing with ping-pong. The shader copies in_pos→out_pos for
    // pinned / out-of-region / weightless verts, so swapping buffers between
    // iterations is safe — every vertex is written each step. After the loop,
    // `out_buf` holds the final result (single CPU readback below).
    static constexpr int NUM_SMOOTH_STEPS = 3;
    uint32_t groups = (vertex_count + 255u) / 256u;
    GLuint in_buf  = remesh_ping_ssbo;
    GLuint out_buf = remesh_pong_ssbo;
    for (int step = 0; step < NUM_SMOOTH_STEPS; step++) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_IN_POS, in_buf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,     out_buf);
        glDispatchCompute(groups, 1, 1);
        if (step + 1 < NUM_SMOOTH_STEPS) {
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(in_buf, out_buf);
        }
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    readback_buf.resize(vertex_count * 3);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_buf);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for (uint32_t i = 0; i < vertex_count; i++) {
        out_pos_x[i] = readback_buf[i*3+0];
        out_pos_y[i] = readback_buf[i*3+1];
        out_pos_z[i] = readback_buf[i*3+2];
    }
}

void ComputeState::dispatch_seam_snap_weld(
    uint32_t vertex_count,
    float* pos_x, const float* pos_y, const float* pos_z,
    const float* mask_data, uint32_t mask_size,
    float seam_tol, float snap_tol, float weld_tol,
    std::vector<uint32_t>& out_merge_map)
{
    ensure_remesh_smooth_buffers(vertex_count, 1);

    // Interleave positions into ping SSBO
    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    vertex_count * 3 * sizeof(float), readback_buf.data());

    // Upload mask (reuse remesh_weights_ssbo as scratch)
    if (mask_data && mask_size > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        mask_size * sizeof(float), mask_data);
    }

    // Allocate weld merge map SSBO
    if (!seam_weld_map_ssbo) glGenBuffers(1, &seam_weld_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seam_weld_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 vertex_count * sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    uint32_t groups = (vertex_count + 255u) / 256u;

    // --- Pass 1: seam snap ---
    glUseProgram(remesh_seam_snap_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_IN_POS, remesh_ping_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,   adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES, remesh_indices_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, remesh_weights_ssbo);

    glUniform1ui(glGetUniformLocation(remesh_seam_snap_program, "u_vertex_count"), vertex_count);
    glUniform1ui(glGetUniformLocation(remesh_seam_snap_program, "u_mask_size"), mask_size);
    glUniform1f(glGetUniformLocation(remesh_seam_snap_program,  "u_seam_tol"), seam_tol);
    glUniform1f(glGetUniformLocation(remesh_seam_snap_program,  "u_snap_tol"), snap_tol);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // --- Pass 2: seam weld ---
    glUseProgram(remesh_seam_weld_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_REMESH_IN_POS, remesh_ping_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, remesh_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SEAM_WELD_MAP, seam_weld_map_ssbo);

    glUniform1ui(glGetUniformLocation(remesh_seam_weld_program, "u_vertex_count"), vertex_count);
    glUniform1ui(glGetUniformLocation(remesh_seam_weld_program, "u_mask_size"), mask_size);
    glUniform1f(glGetUniformLocation(remesh_seam_weld_program,  "u_weld_tol"), weld_tol);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // Readback snapped positions and count how many changed
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       vertex_count * 3 * sizeof(float), readback_buf.data());
    uint32_t n_snapped = 0;
    for (uint32_t i = 0; i < vertex_count; i++) {
        float new_x = readback_buf[i*3+0];
        if (new_x == 0.0f && pos_x[i] != 0.0f) n_snapped++;
        pos_x[i] = new_x;
    }

    // Readback merge map
    out_merge_map.resize(vertex_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seam_weld_map_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       vertex_count * sizeof(uint32_t), out_merge_map.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Chase merge chains to roots
    for (uint32_t v = 0; v < vertex_count; v++) {
        uint32_t root = v;
        while (out_merge_map[root] != root) root = out_merge_map[root];
        out_merge_map[v] = root;
    }

    uint32_t n_welded = 0;
    for (uint32_t v = 0; v < vertex_count; v++)
        if (out_merge_map[v] != v) n_welded++;

    if (n_snapped > 0 || n_welded > 0)
        std::printf("[mirror] GPU seam: snapped %u, welded %u verts\n", n_snapped, n_welded);
}
