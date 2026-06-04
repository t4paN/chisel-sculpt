#include "compute.h"
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Move brush shaders
// ---------------------------------------------------------------------------

static const char* move_capture_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  readonly buffer PosBuf      { float positions[]; };
layout(std430, binding = 8)  buffer AffectedBuf          { uint affected_count; uint affected_ids[]; };
layout(std430, binding = 9)  buffer WBuf                 { vec2 move_weights[]; };
layout(std430, binding = 11) buffer InitBuf              { float move_init[]; };

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

    float d1 = distance(p, u_anchor_pos);
    float w1 = (d1 < u_world_radius) ? brush_falloff(d1, u_world_radius) : 0.0;

    float w2 = 0.0;
    if (u_mirror_x != 0) {
        vec3 ma = vec3(-u_anchor_pos.x, u_anchor_pos.y, u_anchor_pos.z);
        float d2 = distance(p, ma);
        if (d2 < u_world_radius) w2 = brush_falloff(d2, u_world_radius);
    }

    if (w1 <= 0.0 && w2 <= 0.0) return;

    move_weights[v]      = vec2(w1, w2);
    move_init[v*3u]      = p.x;
    move_init[v*3u + 1u] = p.y;
    move_init[v*3u + 2u] = p.z;

    uint idx = atomicAdd(affected_count, 1u);
    affected_ids[idx] = v;
}
)";

static const char* move_weight_smooth_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 2)  readonly buffer IdxBuf      { uint indices[]; };
layout(std430, binding = 4)  readonly buffer AdjOffset   { uint adj_offset[]; };
layout(std430, binding = 5)  readonly buffer AdjList     { uint adj_list[]; };
layout(std430, binding = 8)  readonly buffer AffectedBuf { uint affected_count; uint affected_ids[]; };
layout(std430, binding = 9)  readonly buffer WInBuf      { vec2 w_in[]; };
layout(std430, binding = 10) buffer WOutBuf              { vec2 w_out[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= affected_count) return;
    uint v = affected_ids[idx];

    vec2 sum = w_in[v] * 4.0;
    uint count = 4u;

    uint start = adj_offset[v];
    uint end = adj_offset[v + 1u];
    for (uint j = start; j < end; j++) {
        uint t = adj_list[j];
        uint i0 = indices[t * 3u];
        uint i1 = indices[t * 3u + 1u];
        uint i2 = indices[t * 3u + 2u];
        if (i0 != v) { sum += w_in[i0]; count++; }
        if (i1 != v) { sum += w_in[i1]; count++; }
        if (i2 != v) { sum += w_in[i2]; count++; }
    }

    w_out[v] = sum / float(count);
}
)";

static const char* move_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf              { float positions[]; };
layout(std430, binding = 8)  readonly buffer AffectedBuf { uint affected_count; uint affected_ids[]; };
layout(std430, binding = 9)  readonly buffer WBuf        { vec2 move_weights[]; };
layout(std430, binding = 11) readonly buffer InitBuf    { float move_init[]; };
layout(std430, binding = 12) readonly buffer MaskBuf    { float mask[]; };

uniform vec3 u_total;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= affected_count) return;
    uint v = affected_ids[idx];

    vec2 w = move_weights[v];
    vec3 init = vec3(move_init[v*3u], move_init[v*3u+1u], move_init[v*3u+2u]);
    vec3 mirror_total = vec3(-u_total.x, u_total.y, u_total.z);
    float mscale = 1.0 - mask[v];
    vec3 target = init + (u_total * w.x + mirror_total * w.y) * mscale;

    positions[v*3u]      = target.x;
    positions[v*3u + 1u] = target.y;
    positions[v*3u + 2u] = target.z;
}
)";

// ---------------------------------------------------------------------------
// Move brush methods
// ---------------------------------------------------------------------------

bool ComputeState::init_move() {
    if (!supported) return false;
    move_capture_program = compile_program(move_capture_src);
    if (!move_capture_program) {
        std::printf("[compute] move_capture shader failed to compile\n");
        return false;
    }
    move_weight_smooth_program = compile_program(move_weight_smooth_src);
    if (!move_weight_smooth_program) {
        std::printf("[compute] move_weight_smooth shader failed to compile\n");
        glDeleteProgram(move_capture_program); move_capture_program = 0;
        return false;
    }
    move_apply_program = compile_program(move_apply_src);
    if (!move_apply_program) {
        std::printf("[compute] move_apply shader failed to compile\n");
        glDeleteProgram(move_capture_program); move_capture_program = 0;
        glDeleteProgram(move_weight_smooth_program); move_weight_smooth_program = 0;
        return false;
    }
    std::printf("[compute] move shaders compiled\n");
    return true;
}

void ComputeState::ensure_move_buffers(uint32_t vertex_count) {
    if (move_buffers_capacity >= vertex_count
        && move_affected_ssbo && move_weights_ssbo
        && move_weights_pong_ssbo && move_init_ssbo)
        return;

    if (move_affected_ssbo)     { glDeleteBuffers(1, &move_affected_ssbo);     move_affected_ssbo     = 0; }
    if (move_weights_ssbo)      { glDeleteBuffers(1, &move_weights_ssbo);      move_weights_ssbo      = 0; }
    if (move_weights_pong_ssbo) { glDeleteBuffers(1, &move_weights_pong_ssbo); move_weights_pong_ssbo = 0; }
    if (move_init_ssbo)         { glDeleteBuffers(1, &move_init_ssbo);         move_init_ssbo         = 0; }

    glGenBuffers(1, &move_affected_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_affected_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(1 + vertex_count) * sizeof(uint32_t),
                 nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &move_weights_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)vertex_count * 2 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &move_weights_pong_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_pong_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)vertex_count * 2 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &move_init_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_init_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)vertex_count * 3 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    move_buffers_capacity = vertex_count;
}

void ComputeState::dispatch_move_capture(const MoveCaptureParams& p, GLuint pos_vbo) {
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_affected_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RG32F, GL_RG, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_weights_pong_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RG32F, GL_RG, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(move_capture_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,      pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_AFFECTED,  move_affected_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,   move_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_INIT,      move_init_ssbo);

    glUniform3f(glGetUniformLocation(move_capture_program, "u_anchor_pos"),
                p.anchor_x, p.anchor_y, p.anchor_z);
    glUniform1f(glGetUniformLocation(move_capture_program, "u_world_radius"), p.world_radius);
    glUniform1f(glGetUniformLocation(move_capture_program, "u_hardness"), p.hardness);
    glUniform1i(glGetUniformLocation(move_capture_program, "u_mirror_x"), p.mirror_x ? 1 : 0);
    glUniform1ui(glGetUniformLocation(move_capture_program, "u_vertex_count"), p.vertex_count);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void ComputeState::dispatch_move_weight_smooth(uint32_t vertex_count, int iterations, GLuint index_ebo) {
    if (!move_weight_smooth_program) return;
    glUseProgram(move_weight_smooth_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_INDICES,          index_ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_OFFSET, adjacency_offset_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_ADJACENCY_LIST,   adjacency_list_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_AFFECTED,    move_affected_ssbo);

    int groups = (int)((vertex_count + 255u) / 256u);
    for (int iter = 0; iter < iterations; iter++) {
        if ((iter & 1) == 0) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,      move_weights_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS_PONG, move_weights_pong_ssbo);
        } else {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,      move_weights_pong_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS_PONG, move_weights_ssbo);
        }
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    if ((iterations & 1) == 1) {
        glBindBuffer(GL_COPY_READ_BUFFER, move_weights_pong_ssbo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, move_weights_ssbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                            (GLsizeiptr)move_buffers_capacity * 2 * sizeof(float));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
}

void ComputeState::dispatch_move_apply(const MoveApplyParams& p, GLuint pos_vbo) {
    if (!move_apply_program) return;
    glUseProgram(move_apply_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,     pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_AFFECTED, move_affected_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_WEIGHTS,  move_weights_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MOVE_INIT,     move_init_ssbo);
    if (mask_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MASK, mask_ssbo);

    glUniform3f(glGetUniformLocation(move_apply_program, "u_total"),
                p.total_dx, p.total_dy, p.total_dz);

    int groups = (int)((p.vertex_count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

uint32_t ComputeState::readback_move_affected(std::vector<uint32_t>& out) {
    if (!move_affected_ssbo) return 0;
    uint32_t count = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, move_affected_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &count);
    if (count > move_buffers_capacity) count = move_buffers_capacity;
    out.resize(count);
    if (count > 0) {
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t),
                           count * sizeof(uint32_t), out.data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return count;
}
