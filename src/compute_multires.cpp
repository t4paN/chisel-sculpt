#include "compute.h"
#include <algorithm>
#include <cstdio>

// GPU-resident undo, Phase 2b: pen-up multires diff.
//
// The CPU twin lives in brush.cpp finalize() (the gpu_positions_deferred
// readback loop). For each touched vert it forms the world-space stroke delta
//   d = live_vbo_pos - pen_down_pos
// and, for a displacement-level stroke, reprojects it into the vert's tangent
// frame, accumulating onto the pen-down disp:
//   disp += vec3(dot(t,d), dot(b,d), dot(n,d))
// For a base-level stroke it just stores the live position as the new base pos.
//
// This shader does the same on the GPU, reading the pen-down snapshots that
// MultiresGPU keeps resident (snap_pos from Phase 2a, disp/frames from Phase 1)
// plus the live working VBO, and writing the resident disp/base SSBO. Phase 2b
// keeps the CPU readback authoritative and only validates this against it under
// CHISEL_DEBUG_MULTIRES; Phase 2d drops the CPU readback and trusts this path.

static const char* multires_diff_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  readonly buffer PosBuf   { float live_pos[]; };
layout(std430, binding = 6)  readonly buffer DirtyBuf { uint  dirty[]; };
layout(std430, binding = 30)          buffer DispBuf  { float disp[]; };      // in: snap, out: new
layout(std430, binding = 31) readonly buffer FrameBuf { float frames[]; };
layout(std430, binding = 32) readonly buffer SnapBuf  { float snap_pos[]; };
layout(std430, binding = 33)          buffer BaseBuf  { float base_pos[]; };

uniform uint u_count;
uniform int  u_writes_to_base;

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_count) return;
    uint v = dirty[di];

    vec3 lp = vec3(live_pos[v*3u], live_pos[v*3u+1u], live_pos[v*3u+2u]);
    vec3 sp = vec3(snap_pos[v*3u], snap_pos[v*3u+1u], snap_pos[v*3u+2u]);

    if (u_writes_to_base != 0) {
        base_pos[v*3u]      = lp.x;
        base_pos[v*3u+1u]   = lp.y;
        base_pos[v*3u+2u]   = lp.z;
        return;
    }

    vec3 d = lp - sp;
    vec3 t = vec3(frames[v*9u],      frames[v*9u+1u], frames[v*9u+2u]);
    vec3 b = vec3(frames[v*9u+3u],   frames[v*9u+4u], frames[v*9u+5u]);
    vec3 n = vec3(frames[v*9u+6u],   frames[v*9u+7u], frames[v*9u+8u]);

    disp[v*3u]      += dot(t, d);
    disp[v*3u+1u]   += dot(b, d);
    disp[v*3u+2u]   += dot(n, d);
}
)";

bool ComputeState::init_multires_diff() {
    if (!supported) return false;
    multires_diff_program = compile_program(multires_diff_src);
    if (!multires_diff_program) {
        std::printf("[compute] multires_diff shader failed to compile\n");
        return false;
    }
    std::printf("[compute] multires_diff shader compiled\n");
    return true;
}

void ComputeState::dispatch_multires_diff(GLuint pos_vbo, GLuint disp_ssbo,
                                          GLuint frames_ssbo, GLuint snap_pos_ssbo,
                                          GLuint base_ssbo,
                                          const uint32_t* verts, uint32_t count,
                                          bool writes_to_base) {
    if (!multires_diff_program || count == 0) return;
    if (!pos_vbo || !snap_pos_ssbo) return;
    if (!writes_to_base && (!disp_ssbo || !frames_ssbo)) return;
    if (writes_to_base && !base_ssbo) return;

    // Upload the touched-vert list (same lazy-grow idiom as the other list-driven
    // kernels). dirty_verts_ssbo is free at pen-up — the only prior consumer this
    // frame is the autosmooth compute_normals pass, which has already run.
    GLsizeiptr needed = (GLsizeiptr)count * sizeof(uint32_t);
    if (!dirty_verts_ssbo || count > dirty_verts_capacity) {
        if (dirty_verts_ssbo) glDeleteBuffers(1, &dirty_verts_ssbo);
        glGenBuffers(1, &dirty_verts_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
        uint32_t alloc_count = std::max(count, 4096u);
        glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_count * sizeof(uint32_t),
                     nullptr, GL_DYNAMIC_DRAW);
        dirty_verts_capacity = alloc_count;
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, needed, verts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(multires_diff_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,        pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS,      dirty_verts_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_DISP,    disp_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_FRAMES,  frames_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_SNAP_POS, snap_pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_BASE,    base_ssbo);

    glUniform1ui(glGetUniformLocation(multires_diff_program, "u_count"), count);
    glUniform1i(glGetUniformLocation(multires_diff_program, "u_writes_to_base"),
                writes_to_base ? 1 : 0);

    int groups = (int)((count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

// GPU-resident undo, Phase 2c: undo/redo apply.
//
// The CPU twin is the same-level STROKE path of undo.cpp apply(). For a disp
// layer it scatters the target disp into disp_ssbo and reprojects the storage
// delta (target - source) through the tangent frame into the working VBO:
//   pos += dx*t + dy*b + dz*n   (matches undo.cpp:172-174, dx = target.x - source.x)
// For the base cage the surface IS the base, so it writes the absolute target
// into both base_ssbo and the VBO. The per-vert (target, source) pairs arrive in
// the staging buffer; the caller runs compute_normals over the same verts after.

static const char* multires_apply_src = R"(
#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0)  buffer PosBuf   { float pos[]; };
layout(std430, binding = 6)  readonly buffer DirtyBuf { uint dirty[]; };
layout(std430, binding = 30)          buffer DispBuf  { float disp[]; };
layout(std430, binding = 31) readonly buffer FrameBuf { float frames[]; };
layout(std430, binding = 33)          buffer BaseBuf  { float base_pos[]; };
layout(std430, binding = 34) readonly buffer StageBuf { float stage[]; };  // 6/vert: target,source

uniform uint u_count;
uniform int  u_targets_base;

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_count) return;
    uint v = dirty[di];

    vec3 tgt = vec3(stage[di*6u], stage[di*6u+1u], stage[di*6u+2u]);

    if (u_targets_base != 0) {
        base_pos[v*3u]    = tgt.x;
        base_pos[v*3u+1u] = tgt.y;
        base_pos[v*3u+2u] = tgt.z;
        pos[v*3u]    = tgt.x;
        pos[v*3u+1u] = tgt.y;
        pos[v*3u+2u] = tgt.z;
        return;
    }

    vec3 src = vec3(stage[di*6u+3u], stage[di*6u+4u], stage[di*6u+5u]);
    vec3 d   = tgt - src;

    disp[v*3u]    = tgt.x;
    disp[v*3u+1u] = tgt.y;
    disp[v*3u+2u] = tgt.z;

    vec3 t = vec3(frames[v*9u],      frames[v*9u+1u], frames[v*9u+2u]);
    vec3 b = vec3(frames[v*9u+3u],   frames[v*9u+4u], frames[v*9u+5u]);
    vec3 n = vec3(frames[v*9u+6u],   frames[v*9u+7u], frames[v*9u+8u]);

    pos[v*3u]    += d.x*t.x + d.y*b.x + d.z*n.x;
    pos[v*3u+1u] += d.x*t.y + d.y*b.y + d.z*n.y;
    pos[v*3u+2u] += d.x*t.z + d.y*b.z + d.z*n.z;
}
)";

bool ComputeState::init_multires_apply() {
    if (!supported) return false;
    multires_apply_program = compile_program(multires_apply_src);
    if (!multires_apply_program) {
        std::printf("[compute] multires_apply shader failed to compile\n");
        return false;
    }
    std::printf("[compute] multires_apply shader compiled\n");
    return true;
}

void ComputeState::dispatch_multires_apply(GLuint pos_vbo, GLuint disp_ssbo,
                                           GLuint frames_ssbo, GLuint base_ssbo,
                                           const uint32_t* verts, const float* stage,
                                           uint32_t count, bool targets_base) {
    if (!multires_apply_program || count == 0) return;
    if (!pos_vbo) return;
    if (targets_base && !base_ssbo) return;
    if (!targets_base && (!disp_ssbo || !frames_ssbo)) return;

    // Upload the touched-vert list (reuse dirty_verts_ssbo, as the diff does).
    GLsizeiptr verts_bytes = (GLsizeiptr)count * sizeof(uint32_t);
    if (!dirty_verts_ssbo || count > dirty_verts_capacity) {
        if (dirty_verts_ssbo) glDeleteBuffers(1, &dirty_verts_ssbo);
        glGenBuffers(1, &dirty_verts_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
        uint32_t alloc_count = std::max(count, 4096u);
        glBufferData(GL_SHADER_STORAGE_BUFFER, alloc_count * sizeof(uint32_t),
                     nullptr, GL_DYNAMIC_DRAW);
        dirty_verts_capacity = alloc_count;
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirty_verts_ssbo);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, verts_bytes, verts);

    // Upload the (target, source) staging pairs.
    GLsizeiptr stage_bytes = (GLsizeiptr)count * 6 * sizeof(float);
    if (!multires_stage_ssbo || count > multires_stage_capacity) {
        if (multires_stage_ssbo) glDeleteBuffers(1, &multires_stage_ssbo);
        glGenBuffers(1, &multires_stage_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, multires_stage_ssbo);
        uint32_t alloc_count = std::max(count, 4096u);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)alloc_count * 6 * sizeof(float),
                     nullptr, GL_DYNAMIC_DRAW);
        multires_stage_capacity = alloc_count;
    } else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, multires_stage_ssbo);
    }
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, stage_bytes, stage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(multires_apply_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,       pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS,     dirty_verts_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_DISP,   disp_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_FRAMES, frames_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_BASE,   base_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_STAGE,  multires_stage_ssbo);

    glUniform1ui(glGetUniformLocation(multires_apply_program, "u_count"), count);
    glUniform1i(glGetUniformLocation(multires_apply_program, "u_targets_base"),
                targets_base ? 1 : 0);

    int groups = (int)((count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}
