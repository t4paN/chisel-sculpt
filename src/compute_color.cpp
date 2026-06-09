#include "compute.h"
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Paint (vpaint) shader — clone of the mask paint shader, but it lerps a packed
// RGBA8 albedo toward the brush color instead of accumulating a scalar. Per-dab
// world-distance check, dual anchor (mirror), writes directly to the color
// VBO/SSBO and appends touched verts to the compact dirty list. One owning
// invocation per vertex => no atomics on the color buffer (mirror of mask).
// ---------------------------------------------------------------------------

static const char* color_paint_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  readonly buffer PosBuf { float positions[]; };
layout(std430, binding = 28) buffer ColorBuf { uint color_buf[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask_buf[]; };
layout(std430, binding = 6)  buffer DirtyBuf { uint dirty_count; uint dirty_ids[]; };

uniform vec3  u_anchor_a;
uniform vec3  u_anchor_b;
uniform int   u_use_b;
uniform float u_world_radius;
uniform float u_hardness;
uniform float u_paint_strength;   // lerp amount toward u_paint_color (* falloff)
uniform vec3  u_paint_color;      // brush albedo, linear [0,1]
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

    // Sculpt mask protects the surface from paint too: masked verts (1) are
    // fully shielded, partial mask attenuates proportionally.
    w *= (1.0 - clamp(mask_buf[v], 0.0, 1.0));
    if (w <= 0.0) return;

    float a = clamp(u_paint_strength * w, 0.0, 1.0);
    vec4 old_rgba = unpackUnorm4x8(color_buf[v]);
    vec3 new_rgb  = mix(old_rgba.rgb, u_paint_color, a);   // lerp-to-color
    uint pk_rgba  = packUnorm4x8(vec4(new_rgb, 1.0));      // alpha forced opaque
    if (pk_rgba == color_buf[v]) return;

    color_buf[v] = pk_rgba;
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
// Paint-smooth shader — the smooth gesture (shift / double-shift) while a paint
// brush is active. Blends each in-radius vertex's colour toward the average of
// its 1-ring neighbours (CSR adjacency over incident tris) by strength*falloff.
// Same dual-anchor mirror + mask-shield as paint; in-place neighbour reads are
// Gauss-Seidel-racy across invocations exactly like the geometry smooth, which
// is harmless for a colour blur (a uint read is never torn, both values valid).
// ---------------------------------------------------------------------------

static const char* color_smooth_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  readonly buffer PosBuf { float positions[]; };
layout(std430, binding = 28) buffer ColorBuf { uint color_buf[]; };
layout(std430, binding = 12) readonly buffer MaskBuf { float mask_buf[]; };
layout(std430, binding = 6)  buffer DirtyBuf { uint dirty_count; uint dirty_ids[]; };
layout(std430, binding = 2)  readonly buffer IdxBuf { uint indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffset { uint adj_offset[]; };
layout(std430, binding = 5)  readonly buffer AdjList { uint adj_list[]; };

uniform vec3  u_anchor_a;
uniform vec3  u_anchor_b;
uniform int   u_use_b;
uniform float u_world_radius;
uniform float u_hardness;
uniform float u_strength;
uniform uint  u_vertex_count;

float brush_falloff(float dist, float radius) {
    float t = dist / radius;
    float inner = 0.15 + u_hardness * 0.55;
    if (t <= inner) return 1.0;
    float blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

float anchor_weight(vec3 anchor, vec3 vp) {
    if (u_use_b != 0 && anchor.x * vp.x < 0.0) return 0.0;
    float dist = length(vp - anchor);
    if (dist >= u_world_radius) return 0.0;
    return brush_falloff(dist, u_world_radius);
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    if (v >= u_vertex_count) return;

    vec3 vp = vec3(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);
    float w = anchor_weight(u_anchor_a, vp);
    if (u_use_b != 0) w = max(w, anchor_weight(u_anchor_b, vp));
    if (w <= 0.0) return;

    w *= (1.0 - clamp(mask_buf[v], 0.0, 1.0));   // mask shields paint-smooth too
    if (w <= 0.0) return;

    vec3 sum = vec3(0.0);
    float count = 0.0;
    uint start = adj_offset[v];
    uint end   = adj_offset[v + 1u];
    for (uint j = start; j < end; j++) {
        uint t = adj_list[j];
        uint i0 = indices[t*3u]; uint i1 = indices[t*3u+1u]; uint i2 = indices[t*3u+2u];
        uint n0, n1;
        if      (v == i0) { n0 = i1; n1 = i2; }
        else if (v == i1) { n0 = i0; n1 = i2; }
        else              { n0 = i0; n1 = i1; }
        sum += unpackUnorm4x8(color_buf[n0]).rgb;
        sum += unpackUnorm4x8(color_buf[n1]).rgb;
        count += 2.0;
    }
    if (count <= 0.0) return;

    vec3 avg     = sum / count;
    vec3 old_rgb = unpackUnorm4x8(color_buf[v]).rgb;
    float a      = clamp(u_strength * w, 0.0, 1.0);
    uint pk_rgba = packUnorm4x8(vec4(mix(old_rgb, avg, a), 1.0));
    if (pk_rgba == color_buf[v]) return;

    color_buf[v] = pk_rgba;
    uint idx = atomicAdd(dirty_count, 1u);
    dirty_ids[idx] = v;
}
)";

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

bool ComputeState::init_color() {
    if (!supported) return false;
    color_paint_program = compile_program(color_paint_src);
    if (!color_paint_program) {
        std::printf("[compute] color_paint shader failed to compile\n");
        return false;
    }
    color_smooth_program = compile_program(color_smooth_src);
    if (!color_smooth_program) {
        std::printf("[compute] color_smooth shader failed to compile\n");
        glDeleteProgram(color_paint_program);
        color_paint_program = 0;
        return false;
    }
    std::printf("[compute] color_paint + color_smooth shaders compiled\n");
    return true;
}

void ComputeState::dispatch_color_paint(const ColorPaintParams& p, GLuint pos_vbo) {
    if (!color_paint_program || !color_ssbo || !mask_ssbo) return;

    ensure_smooth_dirty_buffer(p.vertex_count);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(color_paint_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_COLOR, color_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, smooth_dirty_ssbo);

    glUniform3f(glGetUniformLocation(color_paint_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(color_paint_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(color_paint_program, "u_use_b"), p.use_b);
    glUniform1f(glGetUniformLocation(color_paint_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(color_paint_program, "u_hardness"), p.hardness);
    glUniform1f(glGetUniformLocation(color_paint_program, "u_paint_strength"), p.paint_strength);
    glUniform3f(glGetUniformLocation(color_paint_program, "u_paint_color"),
                p.paint_r, p.paint_g, p.paint_b);
    glUniform1ui(glGetUniformLocation(color_paint_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::dispatch_color_smooth(const ColorPaintParams& p, GLuint pos_vbo, GLuint index_ebo) {
    if (!color_smooth_program || !color_ssbo || !mask_ssbo) return;
    if (!adjacency_offset_ssbo || !adjacency_list_ssbo) return;

    ensure_smooth_dirty_buffer(p.vertex_count);
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_dirty_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(color_smooth_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS, pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_COLOR, color_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS, smooth_dirty_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES, index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST, adjacency_list_ssbo);

    glUniform3f(glGetUniformLocation(color_smooth_program, "u_anchor_a"),
                p.anchor_a_x, p.anchor_a_y, p.anchor_a_z);
    glUniform3f(glGetUniformLocation(color_smooth_program, "u_anchor_b"),
                p.anchor_b_x, p.anchor_b_y, p.anchor_b_z);
    glUniform1i(glGetUniformLocation(color_smooth_program, "u_use_b"), p.use_b);
    glUniform1f(glGetUniformLocation(color_smooth_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(color_smooth_program, "u_hardness"), p.hardness);
    glUniform1f(glGetUniformLocation(color_smooth_program, "u_strength"), p.paint_strength);
    glUniform1ui(glGetUniformLocation(color_smooth_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}
