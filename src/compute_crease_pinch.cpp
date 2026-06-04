#include "compute.h"
#include <cstdio>

// ---------------------------------------------------------------------------
// Crease brush shader
// ---------------------------------------------------------------------------

static const char* crease_accum_src = R"(
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
uniform float u_pinch_amount;
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

    vec3 d = anchor_n * (u_disp_amount * w);

    if (dist > 1e-6) {
        vec3 to_anchor = anchor - vp;
        vec3 tangent = to_anchor - dot(to_anchor, anchor_n) * anchor_n;
        d += tangent * (w * u_pinch_amount / u_world_radius);
    }

    uint base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
    // Idempotent weight: any deposit sets weight to 1.0. See compute_draw.cpp
    // for the rationale — without the half-space gate, near-seam verts pick up
    // both anchors, and weight must stay 1 so amplitude doesn't drop versus
    // far-seam verts that pick up only one.
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

// ---------------------------------------------------------------------------
// Pinch brush shader
// ---------------------------------------------------------------------------

static const char* pinch_accum_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosBuf  { float positions[]; };
layout(std430, binding = 1) readonly buffer NormBuf { float normals[];   };
layout(std430, binding = 3) buffer AccumBuf { uint accum[]; };

uniform vec3  u_anchor_a;
uniform vec3  u_anchor_b;
uniform int   u_use_b;
uniform float u_world_radius;
uniform float u_pinch_amount;
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

    vec3 d;
    if (u_pinch_amount >= 0.0) {
        if (dist < 1e-6) return;
        vec3 to_anchor = anchor - vp;
        vec3 tangent = to_anchor - dot(to_anchor, anchor_n) * anchor_n;
        d = tangent * (w * u_pinch_amount / u_world_radius);
    } else {
        float height = dot(vp - anchor, anchor_n);
        d = -anchor_n * (height * w * 1.5 * (-u_pinch_amount) / u_world_radius);
    }

    uint base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
    // Idempotent weight: see compute_draw.cpp for rationale.
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

// ---------------------------------------------------------------------------
// Crease & pinch methods
// ---------------------------------------------------------------------------

bool ComputeState::init_crease() {
    if (!supported) return false;
    crease_accum_program = compile_program(crease_accum_src);
    if (!crease_accum_program) {
        std::printf("[compute] crease_accum shader failed to compile\n");
        return false;
    }
    std::printf("[compute] crease_accum shader compiled\n");
    return true;
}

void ComputeState::dispatch_crease_accum(const CreaseAccumParams& p, GLuint pos_vbo) {
    clear_accum_buffer();

    glUseProgram(crease_accum_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS,   stroke_norm_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM,     accum_ssbo);

    glUniform3f(glGetUniformLocation(crease_accum_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(crease_accum_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(crease_accum_program, "u_use_b"),          p.use_b);
    glUniform1f(glGetUniformLocation(crease_accum_program, "u_world_radius"),   p.world_radius);
    glUniform1f(glGetUniformLocation(crease_accum_program, "u_disp_amount"),    p.disp_amount);
    glUniform1f(glGetUniformLocation(crease_accum_program, "u_pinch_amount"),   p.pinch_amount);
    glUniform1f(glGetUniformLocation(crease_accum_program, "u_hardness"),       p.hardness);
    glUniform1f(glGetUniformLocation(crease_accum_program, "u_facing_threshold"), p.facing_threshold);
    glUniform3f(glGetUniformLocation(crease_accum_program, "u_view_a"),
                p.view_a_x, p.view_a_y, p.view_a_z);
    glUniform3f(glGetUniformLocation(crease_accum_program, "u_view_b"),
                p.view_b_x, p.view_b_y, p.view_b_z);
    glUniform3f(glGetUniformLocation(crease_accum_program, "u_anchor_normal_a"),
                p.anchor_normal_a_x, p.anchor_normal_a_y, p.anchor_normal_a_z);
    glUniform3f(glGetUniformLocation(crease_accum_program, "u_anchor_normal_b"),
                p.anchor_normal_b_x, p.anchor_normal_b_y, p.anchor_normal_b_z);
    glUniform1ui(glGetUniformLocation(crease_accum_program, "u_vertex_count"),  p.vertex_count);

    int groups = ((int)p.vertex_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

bool ComputeState::init_pinch() {
    if (!supported) return false;
    pinch_accum_program = compile_program(pinch_accum_src);
    if (!pinch_accum_program) {
        std::printf("[compute] pinch_accum shader failed to compile\n");
        return false;
    }
    std::printf("[compute] pinch_accum shader compiled\n");
    return true;
}

void ComputeState::dispatch_pinch_accum(const PinchAccumParams& p, GLuint pos_vbo) {
    clear_accum_buffer();

    glUseProgram(pinch_accum_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_NORMALS,   stroke_norm_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ACCUM,     accum_ssbo);

    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(pinch_accum_program, "u_use_b"),          p.use_b);
    glUniform1f(glGetUniformLocation(pinch_accum_program, "u_world_radius"),   p.world_radius);
    glUniform1f(glGetUniformLocation(pinch_accum_program, "u_pinch_amount"),   p.pinch_amount);
    glUniform1f(glGetUniformLocation(pinch_accum_program, "u_hardness"),       p.hardness);
    glUniform1f(glGetUniformLocation(pinch_accum_program, "u_facing_threshold"), p.facing_threshold);
    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_view_a"),
                p.view_a_x, p.view_a_y, p.view_a_z);
    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_view_b"),
                p.view_b_x, p.view_b_y, p.view_b_z);
    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_anchor_normal_a"),
                p.anchor_normal_a_x, p.anchor_normal_a_y, p.anchor_normal_a_z);
    glUniform3f(glGetUniformLocation(pinch_accum_program, "u_anchor_normal_b"),
                p.anchor_normal_b_x, p.anchor_normal_b_y, p.anchor_normal_b_z);
    glUniform1ui(glGetUniformLocation(pinch_accum_program, "u_vertex_count"),  p.vertex_count);

    int groups = ((int)p.vertex_count + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}
