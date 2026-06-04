#include "compute.h"
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Draw brush shaders
// ---------------------------------------------------------------------------

static const char* draw_accum_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf  { float positions[]; };
layout(std430, binding = 1) readonly buffer NormBuf { float normals[];   };
layout(std430, binding = 3) buffer AccumBuf { uint accum[]; };

uniform vec3  u_anchor_a;
uniform vec3  u_anchor_b;
uniform int   u_use_b;
uniform float u_world_radius;
uniform float u_disp_amount;
uniform float u_hardness;
uniform float u_facing_threshold;
uniform vec3  u_view_a;
uniform vec3  u_view_b;
uniform vec3  u_anchor_normal_a;
uniform vec3  u_anchor_normal_b;
uniform uint  u_vertex_count;

void atomicAddFloat(uint idx, float val) {
    uint expected = accum[idx];
    for (int i = 0; i < 128; i++) {
        uint desired = floatBitsToUint(uintBitsToFloat(expected) + val);
        uint old = atomicCompSwap(accum[idx], expected, desired);
        if (old == expected) return;
        expected = old;
    }
}

float brush_falloff(float dist, float radius) {
    float t = dist / radius;
    float inner = 0.15 + u_hardness * 0.55;
    if (t <= inner) return 1.0;
    float blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

void deposit(uint v, vec3 anchor, vec3 view, vec3 anchor_n, vec3 vp, vec3 vn) {
    float dist = length(vp - anchor);
    if (dist >= u_world_radius) return;
    float facing = -dot(vn, view);
    if (facing < u_facing_threshold) return;
    float w = brush_falloff(dist, u_world_radius);
    if (w <= 0.0) return;
    vec3 dir = anchor_n;
    vec3 d = dir * (u_disp_amount * w);
    uint base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
    // Idempotent weight: any deposit sets weight to 1.0. Two anchors hitting the
    // same vert sum their X/Y/Z contributions (where X components from anchor_n_a
    // and anchor_n_b are opposite-signed), but weight stays 1 so apply doesn't
    // halve the amplitude. Without this, near-seam verts that pick up both
    // anchors would average to half-strength versus far-from-seam verts that pick
    // up only one — visible amplitude step at the secondary anchor's radius edge.
    atomicMax(accum[base + 3u], floatBitsToUint(1.0));
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;
    vec3 vp = vec3(positions[v*3u + 0u], positions[v*3u + 1u], positions[v*3u + 2u]);
    vec3 vn = vec3(normals[v*3u + 0u],   normals[v*3u + 1u],   normals[v*3u + 2u]);
    deposit(v, u_anchor_a, u_view_a, u_anchor_normal_a, vp, vn);
    if (u_use_b != 0) deposit(v, u_anchor_b, u_view_b, u_anchor_normal_b, vp, vn);
}
)";

static const char* draw_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf { float positions[]; };
layout(std430, binding = 3)  buffer AccumBuf { uint accum[]; };
layout(std430, binding = 6)  buffer DirtyBuf { uint dirty_count; uint dirty_ids[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[]; };

uniform uint u_vertex_count;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    uint base = v * 4u;
    float w = uintBitsToFloat(accum[base + 3u]);
    if (w <= 0.0) return;

    float scale = 1.0 - mask[v];
    if (scale <= 0.0) return;

    uint idx = atomicAdd(dirty_count, 1u);
    dirty_ids[idx] = v;

    float inv_w = 1.0 / w;
    float dx = uintBitsToFloat(accum[base + 0u]) * inv_w * scale;
    float dy = uintBitsToFloat(accum[base + 1u]) * inv_w * scale;
    float dz = uintBitsToFloat(accum[base + 2u]) * inv_w * scale;

    uint pidx = v * 3u;
    positions[pidx + 0u] += dx;
    positions[pidx + 1u] += dy;
    positions[pidx + 2u] += dz;
}
)";

// Symmetrize-accum pass. For each vertex v: if mv = mirror_map[v] != v, fold
// the X-mirrored version of accum[mv] into v's own accum and write to a
// separate output buffer. The parallel thread for mv does the same, producing
// out[mv] that is strictly the X-mirror of out[v]. Self-paired/seam verts
// (mv == v) and out-of-range partners copy through unchanged. Reading and
// writing different buffers avoids the read-after-write race that an in-place
// version would have.
static const char* draw_accum_symmetrize_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 3)  readonly  buffer AccumIn  { uint accum_in[]; };
layout(std430, binding = 20) writeonly buffer AccumOut { uint accum_out[]; };
layout(std430, binding = 7)  readonly  buffer MirrorBuf { uint mirror_map[]; };

uniform uint u_vertex_count;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    uint base_v = v * 4u;
    float vx = uintBitsToFloat(accum_in[base_v + 0u]);
    float vy = uintBitsToFloat(accum_in[base_v + 1u]);
    float vz = uintBitsToFloat(accum_in[base_v + 2u]);
    float vw = uintBitsToFloat(accum_in[base_v + 3u]);

    uint mv = mirror_map[v];
    if (mv == v || mv >= u_vertex_count) {
        accum_out[base_v + 0u] = floatBitsToUint(vx);
        accum_out[base_v + 1u] = floatBitsToUint(vy);
        accum_out[base_v + 2u] = floatBitsToUint(vz);
        accum_out[base_v + 3u] = floatBitsToUint(vw);
        return;
    }

    uint base_m = mv * 4u;
    float mx = uintBitsToFloat(accum_in[base_m + 0u]);
    float my = uintBitsToFloat(accum_in[base_m + 1u]);
    float mz = uintBitsToFloat(accum_in[base_m + 2u]);
    float mw = uintBitsToFloat(accum_in[base_m + 3u]);

    accum_out[base_v + 0u] = floatBitsToUint(vx + (-mx));
    accum_out[base_v + 1u] = floatBitsToUint(vy +  my);
    accum_out[base_v + 2u] = floatBitsToUint(vz +  mz);
    accum_out[base_v + 3u] = floatBitsToUint(vw +  mw);
}
)";

static const char* draw_mirror_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf { float positions[]; };
layout(std430, binding = 3)  buffer AccumBuf { uint accum[]; };
layout(std430, binding = 7)  readonly buffer MirrorBuf { uint mirror_map[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask[]; };

uniform uint u_vertex_count;

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    uint base = v * 4u;
    float w = uintBitsToFloat(accum[base + 3u]);
    if (w <= 0.0) return;

    uint mv = mirror_map[v];
    if (mv == v) return;

    float scale = 1.0 - mask[mv];
    if (scale <= 0.0) return;

    float inv_w = 1.0 / w;
    float dx = uintBitsToFloat(accum[base + 0u]) * inv_w * scale;
    float dy = uintBitsToFloat(accum[base + 1u]) * inv_w * scale;
    float dz = uintBitsToFloat(accum[base + 2u]) * inv_w * scale;

    uint pidx = mv * 3u;
    positions[pidx + 0u] += -dx;
    positions[pidx + 1u] +=  dy;
    positions[pidx + 2u] +=  dz;
}
)";

// ---------------------------------------------------------------------------
// Draw brush methods
// ---------------------------------------------------------------------------

void ComputeState::ensure_accum_buffer(uint32_t vertex_count) {
    if (accum_ssbo && accum_vertex_count >= vertex_count) return;
    if (accum_ssbo) glDeleteBuffers(1, &accum_ssbo);
    glGenBuffers(1, &accum_ssbo);
    GLsizeiptr size = (GLsizeiptr)vertex_count * 4 * sizeof(uint32_t);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

    if (accum_sym_ssbo) glDeleteBuffers(1, &accum_sym_ssbo);
    glGenBuffers(1, &accum_sym_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_sym_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    accum_vertex_count = vertex_count;
}

void ComputeState::clear_accum_buffer() {
    if (!accum_ssbo || accum_vertex_count == 0) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    uint32_t zero = 0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

bool ComputeState::init_draw_accum() {
    if (!supported) return false;
    draw_accum_program = compile_program(draw_accum_src);
    if (!draw_accum_program) {
        std::printf("[compute] draw_accum shader failed to compile\n");
        return false;
    }
    std::printf("[compute] draw_accum shader compiled\n");
    return true;
}

bool ComputeState::init_draw_accum_symmetrize() {
    if (!supported) return false;
    draw_accum_symmetrize_program = compile_program(draw_accum_symmetrize_src);
    if (!draw_accum_symmetrize_program) {
        std::printf("[compute] draw_accum_symmetrize shader failed to compile\n");
        return false;
    }
    std::printf("[compute] draw_accum_symmetrize shader compiled\n");
    return true;
}

void ComputeState::dispatch_draw_accum_symmetrize(uint32_t vertex_count) {
    if (!draw_accum_symmetrize_program || !accum_ssbo || !accum_sym_ssbo) return;
    if (!mirror_map_ssbo || mirror_map_vertex_count == 0) return;

    glUseProgram(draw_accum_symmetrize_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM,       accum_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM_SYM,   accum_sym_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MIRROR_MAP,  mirror_map_ssbo);
    glUniform1ui(glGetUniformLocation(draw_accum_symmetrize_program, "u_vertex_count"), vertex_count);

    int groups = (int)((vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::snapshot_stroke_normals(GLuint norm_vbo, uint32_t vertex_count) {
    // Working buffer holds only the active entity at offset 0 — copy [0, vertex_count).
    if (stroke_norm_capacity < vertex_count || !stroke_norm_ssbo) {
        if (stroke_norm_ssbo) { glDeleteBuffers(1, &stroke_norm_ssbo); stroke_norm_ssbo = 0; }
        glGenBuffers(1, &stroke_norm_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, stroke_norm_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)vertex_count * 3 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        stroke_norm_capacity = vertex_count;
    }
    GLsizeiptr byte_size = (GLsizeiptr)vertex_count * 3 * sizeof(float);
    glBindBuffer(GL_COPY_READ_BUFFER,  norm_vbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, stroke_norm_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, byte_size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::dispatch_draw_accum(const DrawAccumParams& p, GLuint pos_vbo) {
    glUseProgram(draw_accum_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS, stroke_norm_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_ssbo);

    glUniform3f(glGetUniformLocation(draw_accum_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(draw_accum_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(draw_accum_program, "u_use_b"), p.use_b);
    glUniform1f(glGetUniformLocation(draw_accum_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(draw_accum_program, "u_disp_amount"), p.disp_amount);
    glUniform1f(glGetUniformLocation(draw_accum_program, "u_hardness"), p.hardness);
    glUniform1f(glGetUniformLocation(draw_accum_program, "u_facing_threshold"), p.facing_threshold);
    glUniform3f(glGetUniformLocation(draw_accum_program, "u_view_a"),
                p.view_a_x, p.view_a_y, p.view_a_z);
    glUniform3f(glGetUniformLocation(draw_accum_program, "u_view_b"),
                p.view_b_x, p.view_b_y, p.view_b_z);
    glUniform3f(glGetUniformLocation(draw_accum_program, "u_anchor_normal_a"),
                p.anchor_normal_a_x, p.anchor_normal_a_y, p.anchor_normal_a_z);
    glUniform3f(glGetUniformLocation(draw_accum_program, "u_anchor_normal_b"),
                p.anchor_normal_b_x, p.anchor_normal_b_y, p.anchor_normal_b_z);
    glUniform1ui(glGetUniformLocation(draw_accum_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

void ComputeState::readback_accum(uint32_t vertex_count) {
    readback_buf.resize(vertex_count * 4);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       vertex_count * 4 * sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

bool ComputeState::init_draw_apply() {
    if (!supported) return false;
    draw_apply_program = compile_program(draw_apply_src);
    if (!draw_apply_program) {
        std::printf("[compute] draw_apply shader failed to compile\n");
        return false;
    }
    std::printf("[compute] draw_apply shader compiled\n");
    return true;
}

void ComputeState::upload_accum(const float* disp_x, const float* disp_y,
                                 const float* disp_z, const float* disp_weight,
                                 uint32_t vertex_count) {
    readback_buf.resize(vertex_count * 4);
    for (uint32_t v = 0; v < vertex_count; v++) {
        readback_buf[v * 4 + 0] = disp_x[v];
        readback_buf[v * 4 + 1] = disp_y[v];
        readback_buf[v * 4 + 2] = disp_z[v];
        readback_buf[v * 4 + 3] = disp_weight[v];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    vertex_count * 4 * sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_draw_apply(GLuint pos_vbo, uint32_t vertex_count,
                                        GLuint accum_override) {
    ensure_smooth_dirty_buffer(vertex_count);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(draw_apply_program);

    GLuint accum_src = accum_override ? accum_override : accum_ssbo;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_src);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, smooth_dirty_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform1ui(glGetUniformLocation(draw_apply_program, "u_vertex_count"), vertex_count);

    int groups = (vertex_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

bool ComputeState::init_draw_mirror_apply() {
    if (!supported) return false;
    draw_mirror_apply_program = compile_program(draw_mirror_apply_src);
    if (!draw_mirror_apply_program) {
        std::printf("[compute] draw_mirror_apply shader failed to compile\n");
        return false;
    }
    std::printf("[compute] draw_mirror_apply shader compiled\n");
    return true;
}

void ComputeState::upload_mirror_map(const std::vector<uint32_t>& map) {
    if (map.empty()) {
        mirror_map_vertex_count = 0;
        return;
    }

    if (!mirror_map_ssbo) glGenBuffers(1, &mirror_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mirror_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, map.size() * sizeof(uint32_t), map.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    mirror_map_vertex_count = map.size();

    std::printf("[compute] mirror_map uploaded: %u vertices\n", mirror_map_vertex_count);
}

void ComputeState::dispatch_draw_mirror_apply(GLuint pos_vbo, uint32_t vertex_count) {
    if (!draw_mirror_apply_program || mirror_map_vertex_count == 0) return;

    glUseProgram(draw_mirror_apply_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM, accum_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MIRROR_MAP, mirror_map_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform1ui(glGetUniformLocation(draw_mirror_apply_program, "u_vertex_count"), vertex_count);

    int groups = (vertex_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}
