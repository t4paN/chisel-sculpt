#include "compute.h"
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Mask paint shader — per-vertex world-distance check, dual anchor (mirror),
// writes directly to the mask VBO/SSBO and appends to a compact dirty list.
// ---------------------------------------------------------------------------

static const char* mask_paint_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  readonly buffer PosBuf { float positions[]; };
layout(std430, binding = 12) buffer MaskBuf { float mask_buf[]; };
layout(std430, binding = 6)  buffer DirtyBuf { uint dirty_count; uint dirty_ids[]; };

uniform vec3  u_anchor_a;
uniform vec3  u_anchor_b;
uniform int   u_use_b;
uniform float u_world_radius;
uniform float u_hardness;
uniform float u_paint_strength;
uniform uint  u_vertex_count;

float brush_falloff(float dist, float radius) {
    float t = dist / radius;
    float inner = 0.15 + u_hardness * 0.55;
    if (t <= inner) return 1.0;
    float blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

void try_paint(uint v, vec3 anchor, vec3 vp) {
    if (u_use_b != 0 && anchor.x * vp.x < 0.0) return;
    float dist = length(vp - anchor);
    if (dist >= u_world_radius) return;
    float w = brush_falloff(dist, u_world_radius);
    if (w <= 0.0) return;

    float delta = u_paint_strength * w;
    float old_val = mask_buf[v];
    float new_val = clamp(old_val + delta, 0.0, 1.0);
    if (new_val == old_val) return;

    mask_buf[v] = new_val;
    uint idx = atomicAdd(dirty_count, 1u);
    dirty_ids[idx] = v;
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    vec3 vp = vec3(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    try_paint(v, u_anchor_a, vp);
    if (u_use_b != 0) try_paint(v, u_anchor_b, vp);
}
)";

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

bool ComputeState::init_mask() {
    if (!supported) return false;
    mask_paint_program = compile_program(mask_paint_src);
    if (!mask_paint_program) {
        std::printf("[compute] mask_paint shader failed to compile\n");
        return false;
    }
    std::printf("[compute] mask_paint shader compiled\n");
    return true;
}

void ComputeState::dispatch_mask_paint(const MaskPaintParams& p, GLuint pos_vbo) {
    if (!mask_paint_program || !mask_ssbo) return;

    ensure_smooth_dirty_buffer(p.vertex_count);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(mask_paint_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, smooth_dirty_ssbo);

    glUniform3f(glGetUniformLocation(mask_paint_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(mask_paint_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(mask_paint_program, "u_use_b"), p.use_b);
    glUniform1f(glGetUniformLocation(mask_paint_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(mask_paint_program, "u_hardness"), p.hardness);
    glUniform1f(glGetUniformLocation(mask_paint_program, "u_paint_strength"), p.paint_strength);
    glUniform1ui(glGetUniformLocation(mask_paint_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}
