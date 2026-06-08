#include "compute.h"
#include <cstdio>
#include <utility>

// ---------------------------------------------------------------------------
// Limb (snakehook) brush
//
// Built on the MOVE sticky-capture spine (compute_move.cpp): the affected vert
// set, per-vertex falloff weights and mirror decomposition are all captured once
// at pen-down and reused here. Two passes run per dab:
//
//   1. limb_drag   — INCREMENTAL grab: positions[v] += this-dab world delta * w.
//                    (Move's apply is absolute, init+total*w; that would reset
//                    positions every dab and wipe the relax. Incremental drag
//                    lets the redistribution accumulate.)
//   2. limb_relax  — tangential (normal-stripped) Laplacian over the captured
//                    set. Evens vertex spacing along the stretching shaft without
//                    deflating the form — verts flow up from the dense base to
//                    fill the pulled gaps. This is the "limbinator" feel: an
//                    extrusion that redistributes as it grows. No new triangles
//                    (no dynamic topology) — resolution is whatever was captured.
// ---------------------------------------------------------------------------

static const char* limb_drag_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf            { float positions[]; };
layout(std430, binding = 8)  readonly buffer AffectedBuf { uint affected_count; uint affected_ids[]; };
layout(std430, binding = 9)  readonly buffer WBuf        { vec2 move_weights[]; };
layout(std430, binding = 12) readonly buffer MaskBuf     { float mask[]; };

uniform vec3 u_delta;   // this dab's world-space drag increment

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= affected_count) return;
    uint v = affected_ids[idx];

    vec2 w = move_weights[v];
    // Mirror lobe (w.y) is X-negated; w.y is 0 when mirror is off, so this term
    // simply vanishes — same decomposition the move brush uses.
    vec3 mdelta = vec3(-u_delta.x, u_delta.y, u_delta.z);
    float mscale = 1.0 - mask[v];
    vec3 d = (u_delta * w.x + mdelta * w.y) * mscale;

    positions[v*3u + 0u] += d.x;
    positions[v*3u + 1u] += d.y;
    positions[v*3u + 2u] += d.z;
}
)";

// Ping-pong tangential Laplacian. Reads the src snapshot, writes dst. Dispatched
// over the whole vertex range: verts outside the captured set (w==0) pass through
// unchanged so both buffers stay fully valid for the next iteration's neighbour
// reads. The captured-set boundary reads unmoved neighbours, anchoring the base.
static const char* limb_relax_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 29) readonly  buffer PosSrc    { float src[]; };
layout(std430, binding = 0)  writeonly buffer PosDst    { float dst[]; };
layout(std430, binding = 1)  readonly  buffer NormBuf   { float normals[]; };
layout(std430, binding = 2)  readonly  buffer IdxBuf    { uint indices[]; };
layout(std430, binding = 4)  readonly  buffer AdjOffset { uint adj_offset[]; };
layout(std430, binding = 5)  readonly  buffer AdjList   { uint adj_list[]; };
layout(std430, binding = 9)  readonly  buffer WBuf      { vec2 move_weights[]; };
layout(std430, binding = 12) readonly  buffer MaskBuf   { float mask[]; };

uniform uint  u_vertex_count;
uniform float u_lambda;
uniform vec3  u_tip_dir;    // normalized world pull direction (points toward the tip)
uniform float u_tip_bias;   // 0 = symmetric relax; >0 drifts verts up-shaft → denser cap

// Accumulate one neighbour into the (tip-biased) centroid. Neighbours lying toward
// the tip are up-weighted, so each vert's relax target shifts up the shaft: the
// vertex distribution migrates toward the leading end, densifying the cap (and
// slightly thinning the base). The target stays a convex combination of the 1-ring,
// so this is a bounded redistribution, not an extra translation of the limb.
void accum_neighbour(uint vi, vec3 p, vec3 tdir, inout vec3 sum, inout float wsum) {
    vec3 q = vec3(src[vi*3u], src[vi*3u+1u], src[vi*3u+2u]);
    vec3 e = q - p;
    float el = length(e);
    float b = 1.0;
    if (el > 1e-8) b += u_tip_bias * max(0.0, dot(e / el, tdir));
    sum  += q * b;
    wsum += b;
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    vec3 p = vec3(src[v*3u], src[v*3u+1u], src[v*3u+2u]);

    vec2 w2 = move_weights[v];
    float w = max(w2.x, w2.y);
    if (w <= 0.0) {                       // outside the brush: pass through
        dst[v*3u] = p.x; dst[v*3u+1u] = p.y; dst[v*3u+2u] = p.z;
        return;
    }

    // Per-vert tip direction. Verts the mirror lobe owns (w.y dominant) drift toward
    // the X-flipped pull, matching the drag's mdelta = (-dx, dy, dz) decomposition;
    // a single world-space dir would skew the mirrored cap the wrong way.
    vec3 tdir = (w2.y > w2.x) ? vec3(-u_tip_dir.x, u_tip_dir.y, u_tip_dir.z) : u_tip_dir;

    // 1-ring centroid via adjacent triangles (shared neighbours counted by valence —
    // a uniform-enough weighting, matches move_smooth) with the tip bias folded in.
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    uint start = adj_offset[v];
    uint end   = adj_offset[v + 1u];
    for (uint j = start; j < end; j++) {
        uint t = adj_list[j];
        uint i0 = indices[t*3u]; uint i1 = indices[t*3u+1u]; uint i2 = indices[t*3u+2u];
        if (i0 != v) accum_neighbour(i0, p, tdir, sum, wsum);
        if (i1 != v) accum_neighbour(i1, p, tdir, sum, wsum);
        if (i2 != v) accum_neighbour(i2, p, tdir, sum, wsum);
    }
    if (wsum <= 0.0) {
        dst[v*3u] = p.x; dst[v*3u+1u] = p.y; dst[v*3u+2u] = p.z;
        return;
    }

    vec3 delta = sum / wsum - p;
    // Strip the normal component → tangential move only. This redistributes
    // spacing while holding the silhouette (a full Laplacian would shrink the
    // form, the same reason the smooth brush deflates).
    vec3 n = vec3(normals[v*3u], normals[v*3u+1u], normals[v*3u+2u]);
    float nl = length(n);
    if (nl > 1e-8) { n /= nl; delta -= n * dot(delta, n); }

    float mscale = 1.0 - mask[v];
    vec3 np = p + u_lambda * w * mscale * delta;
    dst[v*3u] = np.x; dst[v*3u+1u] = np.y; dst[v*3u+2u] = np.z;
}
)";

// ---------------------------------------------------------------------------

bool ComputeState::init_limb() {
    if (!supported) return false;
    limb_drag_program = compile_program(limb_drag_src);
    if (!limb_drag_program) {
        std::printf("[compute] limb_drag shader failed to compile\n");
        return false;
    }
    limb_relax_program = compile_program(limb_relax_src);
    if (!limb_relax_program) {
        std::printf("[compute] limb_relax shader failed to compile\n");
        glDeleteProgram(limb_drag_program); limb_drag_program = 0;
        return false;
    }
    std::printf("[compute] limb shaders compiled\n");
    return true;
}

void ComputeState::ensure_limb_buffers(uint32_t vertex_count) {
    if (limb_scratch_capacity >= vertex_count && limb_pos_scratch_ssbo) return;
    if (limb_pos_scratch_ssbo) { glDeleteBuffers(1, &limb_pos_scratch_ssbo); limb_pos_scratch_ssbo = 0; }
    glGenBuffers(1, &limb_pos_scratch_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, limb_pos_scratch_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)vertex_count * 3 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    limb_scratch_capacity = vertex_count;
}

void ComputeState::dispatch_limb_drag(const LimbDragParams& p, GLuint pos_vbo) {
    if (!limb_drag_program) return;
    glUseProgram(limb_drag_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,     pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_AFFECTED, move_affected_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,  move_weights_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform3f(glGetUniformLocation(limb_drag_program, "u_delta"), p.dx, p.dy, p.dz);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::dispatch_limb_relax(uint32_t vertex_count, int iterations, float lambda,
                                       float tip_dx, float tip_dy, float tip_dz, float tip_bias,
                                       GLuint pos_vbo, GLuint norm_vbo, GLuint index_ebo) {
    if (!limb_relax_program || iterations <= 0) return;
    ensure_limb_buffers(vertex_count);

    // src snapshot starts as a full copy of the live positions.
    glBindBuffer(GL_COPY_READ_BUFFER,  pos_vbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, limb_pos_scratch_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                        (GLsizeiptr)vertex_count * 3 * sizeof(float));
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    glUseProgram(limb_relax_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS,          norm_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,          index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,   adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,     move_weights_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);
    glUniform1ui(glGetUniformLocation(limb_relax_program, "u_vertex_count"), vertex_count);
    glUniform1f(glGetUniformLocation(limb_relax_program, "u_lambda"), lambda);
    glUniform3f(glGetUniformLocation(limb_relax_program, "u_tip_dir"), tip_dx, tip_dy, tip_dz);
    glUniform1f(glGetUniformLocation(limb_relax_program, "u_tip_bias"), tip_bias);

    GLuint src = limb_pos_scratch_ssbo;
    GLuint dst = pos_vbo;
    int groups = (int)((vertex_count + 255u) / 256u);
    for (int i = 0; i < iterations; i++) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_LIMB_POS_SRC, src);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,    dst);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        std::swap(src, dst);
    }
    // Even iteration count leaves the final result in the scratch buffer; copy
    // it back into the live position VBO.
    if ((iterations & 1) == 0) {
        glBindBuffer(GL_COPY_READ_BUFFER,  limb_pos_scratch_ssbo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, pos_vbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            (GLsizeiptr)vertex_count * 3 * sizeof(float));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}
