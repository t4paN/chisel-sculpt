#include "compute.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cmath>

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
layout(std430, binding = 35)          buffer RingBuf  { float ring[]; };      // float6/vert: old(3)+new(3)

uniform uint u_count;
uniform int  u_writes_to_base;
uniform uint u_ring_base;   // float offset of this stroke's span in the undo ring
uniform int  u_ring_on;     // write (old,new) into the ring when != 0

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_count) return;
    uint v = dirty[di];

    vec3 lp = vec3(live_pos[v*3u], live_pos[v*3u+1u], live_pos[v*3u+2u]);
    vec3 sp = vec3(snap_pos[v*3u], snap_pos[v*3u+1u], snap_pos[v*3u+2u]);

    vec3 old_v, new_v;

    if (u_writes_to_base != 0) {
        // Base-level stroke: the surface IS the base cage, so old = pen-down world
        // pos (snap), new = live pos.
        old_v = sp;
        new_v = lp;
        base_pos[v*3u]      = lp.x;
        base_pos[v*3u+1u]   = lp.y;
        base_pos[v*3u+2u]   = lp.z;
    } else {
        vec3 d = lp - sp;
        vec3 t = vec3(frames[v*9u],      frames[v*9u+1u], frames[v*9u+2u]);
        vec3 b = vec3(frames[v*9u+3u],   frames[v*9u+4u], frames[v*9u+5u]);
        vec3 n = vec3(frames[v*9u+6u],   frames[v*9u+7u], frames[v*9u+8u]);

        // disp[] still holds the pen-down disp on entry — that's old.
        old_v = vec3(disp[v*3u], disp[v*3u+1u], disp[v*3u+2u]);
        new_v = old_v + vec3(dot(t, d), dot(b, d), dot(n, d));
        disp[v*3u]      = new_v.x;
        disp[v*3u+1u]   = new_v.y;
        disp[v*3u+2u]   = new_v.z;
    }

    if (u_ring_on != 0) {
        // Pack (old,new) at this stroke's reserved span, indexed by local di so the
        // ring is densely packed per stroke (3b-iii undo capture).
        uint ro = u_ring_base + di * 6u;
        ring[ro]      = old_v.x;  ring[ro+1u] = old_v.y;  ring[ro+2u] = old_v.z;
        ring[ro+3u]   = new_v.x;  ring[ro+4u] = new_v.y;  ring[ro+5u] = new_v.z;
    }
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
                                          bool writes_to_base,
                                          GLuint ring_ssbo, uint32_t ring_base_floats) {
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
    // Undo-ring capture (3b-iii): bind the ring so the shader writes (old,new). A
    // valid binding is required even when off (some drivers demand it), so fall
    // back to the disp buffer as a harmless bind target when ring_ssbo == 0.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_UNDO_RING,
                     ring_ssbo ? ring_ssbo : disp_ssbo);

    glUniform1ui(glGetUniformLocation(multires_diff_program, "u_count"), count);
    glUniform1i(glGetUniformLocation(multires_diff_program, "u_writes_to_base"),
                writes_to_base ? 1 : 0);
    glUniform1ui(glGetUniformLocation(multires_diff_program, "u_ring_base"), ring_base_floats);
    glUniform1i(glGetUniformLocation(multires_diff_program, "u_ring_on"), ring_ssbo ? 1 : 0);

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
layout(std430, binding = 35) readonly buffer RingBuf  { float ring[]; };   // 6/vert: old,new

uniform uint u_count;
uniform int  u_targets_base;
// Source the (target,source) pair from the persistent undo ring instead of the
// transient CPU stage (3b-iv). The ring stores (old,new); undo wants target=old,
// redo target=new — u_forward picks the direction. ring_base is the FLOAT offset
// of this entry's span; slot index == di (verts aligned to the unfiltered capture).
uniform int  u_ring_mode;
uniform uint u_ring_base;
uniform int  u_forward;

void main() {
    uint di = gl_GlobalInvocationID.x;
    if (di >= u_count) return;
    uint v = dirty[di];

    vec3 tgt, src;
    if (u_ring_mode != 0) {
        uint ro = u_ring_base + di * 6u;
        vec3 old_v = vec3(ring[ro],      ring[ro+1u], ring[ro+2u]);
        vec3 new_v = vec3(ring[ro+3u],   ring[ro+4u], ring[ro+5u]);
        tgt = (u_forward != 0) ? new_v : old_v;
        src = (u_forward != 0) ? old_v : new_v;
    } else {
        tgt = vec3(stage[di*6u],      stage[di*6u+1u], stage[di*6u+2u]);
        src = vec3(stage[di*6u+3u],   stage[di*6u+4u], stage[di*6u+5u]);
    }

    if (u_targets_base != 0) {
        base_pos[v*3u]    = tgt.x;
        base_pos[v*3u+1u] = tgt.y;
        base_pos[v*3u+2u] = tgt.z;
        pos[v*3u]    = tgt.x;
        pos[v*3u+1u] = tgt.y;
        pos[v*3u+2u] = tgt.z;
        return;
    }

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
                                           uint32_t count, bool targets_base,
                                           GLuint ring_ssbo, uint32_t ring_base_floats,
                                           bool forward) {
    if (!multires_apply_program || count == 0) return;
    if (!pos_vbo) return;
    if (targets_base && !base_ssbo) return;
    if (!targets_base && (!disp_ssbo || !frames_ssbo)) return;

    // Ring mode (3b-iv): read (old,new) straight from the persistent undo ring, no
    // CPU stage upload. Stage mode: upload the (target,source) pairs as before.
    const bool ring_mode = (ring_ssbo != 0);

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

    // Upload the (target, source) staging pairs (stage mode only).
    if (!ring_mode && stage) {
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
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Every declared SSBO binding must point at a live buffer (some drivers fault on
    // an unbound slot even if the shader never reads it). In ring mode the stage slot
    // is unused; in stage mode the ring slot is. Fall back to disp_ssbo as a harmless
    // filler for whichever is idle.
    GLuint filler    = disp_ssbo ? disp_ssbo : base_ssbo;
    GLuint stage_bind = (!ring_mode && multires_stage_ssbo) ? multires_stage_ssbo : filler;
    GLuint ring_bind  = ring_mode ? ring_ssbo : filler;

    glUseProgram(multires_apply_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_POSITIONS,       pos_vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_DIRTY_VERTS,     dirty_verts_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_DISP,   disp_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_FRAMES, frames_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_BASE,   base_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_MULTIRES_STAGE,  stage_bind);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_UNDO_RING,       ring_bind);

    glUniform1ui(glGetUniformLocation(multires_apply_program, "u_count"), count);
    glUniform1i(glGetUniformLocation(multires_apply_program, "u_targets_base"),
                targets_base ? 1 : 0);
    glUniform1i(glGetUniformLocation(multires_apply_program, "u_ring_mode"),
                ring_mode ? 1 : 0);
    glUniform1ui(glGetUniformLocation(multires_apply_program, "u_ring_base"),
                 ring_mode ? ring_base_floats : 0u);
    glUniform1i(glGetUniformLocation(multires_apply_program, "u_forward"),
                forward ? 1 : 0);

    int groups = (int)((count + 255u) / 256u);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

// ===========================================================================
// GPU-resident undo ring (blood-moon 3b-ii → 3b-iv)
//
// Persistent history of per-vert (old,new) STROKE deltas in one SSBO. It grows
// (copy-preserving, doubling) toward the cap while there's headroom, then becomes
// a true circular buffer: once the buffer has reached the cap the bump cursor
// wraps to 0 instead of failing. Spans never straddle the end — a reserve that
// won't fit before the end wraps first, leaving a little dead padding. Eviction is
// driven from the consumer side: when a new span is written, UndoStack::
// ring_evict_overlap invalidates any older entry whose bytes it overwrites (the
// CPU arrays remain the spill target). The pen-up capture (3b-iii) writes (old,new)
// into a reserved span; the apply (3b-iv part 1) reads them back.
// ===========================================================================

void ComputeState::undo_ring_set_budget(size_t cap_bytes) {
    // Clamp to something sane; the ring never allocates this eagerly — it grows
    // toward the cap as history accumulates.
    if (cap_bytes < (1ull << 20)) cap_bytes = (1ull << 20);  // 1 MB floor (debug --ring-mb can go small)
    undo_ring_cap_bytes = cap_bytes;
}

void ComputeState::undo_ring_reset() {
    undo_ring_head = 0;   // drop history; keep the allocated buffer for reuse
}

size_t ComputeState::undo_ring_reserve(size_t float_count) {
    if (!supported || float_count == 0) return SIZE_MAX;

    const size_t bytes = float_count * sizeof(float);
    if (bytes > undo_ring_cap_bytes) return SIZE_MAX;   // span larger than the whole ring

    // Grow (copy-preserving, doubling) toward the cap while there's headroom. Once
    // the buffer has reached the cap, the wrap below recycles space instead.
    if ((!undo_ring_ssbo || undo_ring_head + bytes > undo_ring_bytes)
        && undo_ring_bytes < undo_ring_cap_bytes) {
        size_t newcap = undo_ring_bytes ? undo_ring_bytes : (16ull << 20);
        while (newcap < undo_ring_head + bytes && newcap < undo_ring_cap_bytes) newcap <<= 1;
        if (newcap > undo_ring_cap_bytes) newcap = undo_ring_cap_bytes;

        GLuint nb = 0;
        glGenBuffers(1, &nb);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, nb);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)newcap, nullptr, GL_DYNAMIC_COPY);
        if (undo_ring_ssbo && undo_ring_head) {
            glBindBuffer(GL_COPY_READ_BUFFER,  undo_ring_ssbo);
            glBindBuffer(GL_COPY_WRITE_BUFFER, nb);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                                (GLsizeiptr)undo_ring_head);
            glBindBuffer(GL_COPY_READ_BUFFER,  0);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        }
        if (undo_ring_ssbo) glDeleteBuffers(1, &undo_ring_ssbo);
        undo_ring_ssbo  = nb;
        undo_ring_bytes = newcap;
    }

    // At the cap (or the span won't fit before the end): wrap the cursor to 0 so
    // the span stays contiguous. The leftover tail bytes become dead padding; the
    // consumer (ring_evict_overlap) invalidates whatever live entries the new write
    // lands on.
    if (undo_ring_head + bytes > undo_ring_bytes) undo_ring_head = 0;

    const size_t offset = undo_ring_head;
    undo_ring_head = offset + bytes;
    return offset / sizeof(float);   // FLOAT offset for the shader uniform
}

size_t ComputeState::undo_ring_append(const float* data, size_t float_count) {
    // Reserve a span (handles grow + wrap) then upload into it. Returns the BYTE
    // offset of the span (the self-test checks byte offsets), or SIZE_MAX on fail.
    const size_t off_floats = undo_ring_reserve(float_count);
    if (off_floats == SIZE_MAX) return SIZE_MAX;
    const size_t byte_off = off_floats * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, undo_ring_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)byte_off,
                    (GLsizeiptr)(float_count * sizeof(float)), data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return byte_off;
}

void ComputeState::undo_ring_read(size_t byte_offset, size_t float_count, float* out) {
    if (!supported || !undo_ring_ssbo || float_count == 0) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, undo_ring_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)byte_offset,
                       (GLsizeiptr)(float_count * sizeof(float)), out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::undo_ring_selftest() {
#ifdef CHISEL_DEBUG_MULTIRES
    if (!supported) return;
    // Two appends of distinct patterns, read both back, confirm offsets + contents.
    float a[6] = { 1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f };
    float b[3] = { 7.5f, 8.5f, 9.5f };
    size_t head0 = undo_ring_head;
    size_t oa = undo_ring_append(a, 6);
    size_t ob = undo_ring_append(b, 3);
    bool offsets_ok = (oa == head0) && (ob == head0 + 6 * sizeof(float));

    float ra[6] = {0}, rb[3] = {0};
    undo_ring_read(oa, 6, ra);
    undo_ring_read(ob, 3, rb);
    double maxe = 0.0;
    for (int i = 0; i < 6; i++) maxe = std::max(maxe, (double)std::fabs(ra[i] - a[i]));
    for (int i = 0; i < 3; i++) maxe = std::max(maxe, (double)std::fabs(rb[i] - b[i]));

    std::printf("[undo-ring][debug] selftest offsets_ok=%d max|err|=%.3e "
                "(head=%zu cap=%zuMB alloc=%zuMB)\n",
                offsets_ok ? 1 : 0, maxe, undo_ring_head,
                undo_ring_cap_bytes >> 20, undo_ring_bytes >> 20);

    undo_ring_head = head0;   // rewind — selftest leaves no history behind
#endif
}
