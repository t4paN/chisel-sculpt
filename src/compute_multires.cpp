#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("multires_diff" / "multires_apply")
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// GPU-resident undo, multires diff + apply — ported onto the gpu:: seam
// (Seam Step 2b). The two compute kernels now dispatch through gpu::; their
// kernel logic lives in the canonical shaders/{glsl,wgsl}/multires_{diff,apply}.*
// (embedded at build time).
//
// All the resident multires SSBOs are now seam-owned gpu::Buffers and bound
// directly: the residency pool (disp/frames/snap_pos/base, in MultiresGPU) migrated
// in buffer-ownership Step 3c, the stage scratch + undo ring in Step 3b. The
// dispatch params take const gpu::Buffer& (no GLuint view fabrication). The undo
// ring's copy-preserving grow + readback and the dirty-vert id / stage uploads still
// move bytes via raw GL (.handle) — no seam copy/map primitive yet (web-stage
// concern). The CPU twins are brush.cpp finalize() (diff) and undo.cpp apply()
// (apply); under CHISEL_DEBUG_MULTIRES the CPU readback validates these.
// ---------------------------------------------------------------------------

namespace {
// 16-byte std140 block, byte-identical to multires_diff.{comp,wgsl}'s Params.
struct MultiresDiffParamsGPU {
    uint32_t count;          // 0
    int32_t  writes_to_base; // 4
    uint32_t ring_base;      // 8
    int32_t  ring_on;        // 12  (struct = 16)
};
static_assert(sizeof(MultiresDiffParamsGPU) == 16, "multires diff Params UBO must be 16 bytes");

// 32-byte std140 block, byte-identical to multires_apply.{comp,wgsl}'s Params.
struct MultiresApplyParamsGPU {
    uint32_t count;        // 0
    int32_t  targets_base; // 4
    int32_t  ring_mode;    // 8
    uint32_t ring_base;    // 12  (slot fills to 16)
    int32_t  forward;      // 16
    int32_t  _pad0;        // 20
    int32_t  _pad1;        // 24
    int32_t  _pad2;        // 28  (struct = 32)
};
static_assert(sizeof(MultiresApplyParamsGPU) == 32, "multires apply Params UBO must be 32 bytes");
}

// ---------------------------------------------------------------------------
// Phase 2b: pen-up multires diff.
// ---------------------------------------------------------------------------

bool ComputeState::init_multires_diff() {
    if (!supported) return false;

    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,         gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS,       gpu::Bind::StorageRead,      0 },
        { BIND_MULTIRES_DISP,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_MULTIRES_FRAMES,   gpu::Bind::StorageRead,      0 },
        { BIND_MULTIRES_SNAP_POS, gpu::Bind::StorageRead,      0 },
        { BIND_MULTIRES_BASE,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_UNDO_RING,         gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,            gpu::Bind::Uniform, sizeof(MultiresDiffParamsGPU) },
    };
    multires_diff_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                 gpu::embedded_shader("multires_diff"), layout, 8);
    if (!multires_diff_pipeline.handle) {
        std::printf("[compute] multires_diff pipeline failed to compile\n");
        return false;
    }
    multires_diff_ubo = gpu::create_buffer(gpu_dev, nullptr,
                                           sizeof(MultiresDiffParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] multires_diff pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_multires_diff(const gpu::Buffer& pos_vbo, const gpu::Buffer& disp_ssbo,
                                          const gpu::Buffer& frames_ssbo, const gpu::Buffer& snap_pos_ssbo,
                                          const gpu::Buffer& base_ssbo,
                                          const uint32_t* verts, uint32_t count,
                                          bool writes_to_base,
                                          bool ring_ssbo, uint32_t ring_base_floats) {
    if (!has_multires_diff() || count == 0) return;
    if (!pos_vbo.handle || !snap_pos_ssbo.handle) return;
    if (!writes_to_base && (!disp_ssbo.handle || !frames_ssbo.handle)) return;
    if (writes_to_base && !base_ssbo.handle) return;

    // Upload the touched-vert list (same lazy-grow idiom as the other list-driven
    // kernels). dirty_verts_ssbo is free at pen-up — the only prior consumer this
    // frame is the autosmooth compute_normals pass, which has already run. Seam-owned.
    if (!dirty_verts_ssbo.handle || count > dirty_verts_capacity) {
        uint32_t alloc_count = std::max(count, 4096u);
        gpu::release_buffer(dirty_verts_ssbo);
        dirty_verts_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                              (uint64_t)alloc_count * sizeof(uint32_t), gpu::Usage::Storage);
        dirty_verts_capacity = alloc_count;
    }
    gpu::write_buffer(gpu_dev, dirty_verts_ssbo, 0, verts, (uint64_t)count * sizeof(uint32_t));

    MultiresDiffParamsGPU u = {};
    u.count          = count;
    u.writes_to_base = writes_to_base ? 1 : 0;
    u.ring_base      = ring_base_floats;
    u.ring_on        = ring_ssbo ? 1 : 0;   // 3b-iii undo capture
    gpu::write_buffer(gpu_dev, multires_diff_ubo, 0, &u, sizeof(u));

    // Every declared binding must point at a live buffer (the seam binds all 8; a 0
    // handle would fault on some drivers). The pool buffers are seam-owned now (Step
    // 3c) and bound directly; pos_vbo is always present (checked above), so it's the
    // harmless filler for any slot the current mode doesn't touch. The ring falls back
    // to disp as the old GL path did when capture is off.
    const gpu::Buffer* disp_b  = disp_ssbo.handle   ? &disp_ssbo   : &pos_vbo;
    const gpu::Buffer* frame_b = frames_ssbo.handle ? &frames_ssbo : &pos_vbo;
    const gpu::Buffer* base_b  = base_ssbo.handle   ? &base_ssbo   : &pos_vbo;

    const bool ring_on = (ring_ssbo != 0);
    const gpu::Buffer* ring_b = ring_on ? &undo_ring_ssbo : disp_b;

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,         &pos_vbo,   pos_vbo.size },
        { BIND_DIRTY_VERTS,       &dirty_verts_ssbo, (uint64_t)count * sizeof(uint32_t) },
        { BIND_MULTIRES_DISP,     disp_b,     disp_b->size },
        { BIND_MULTIRES_FRAMES,   frame_b,    frame_b->size },
        { BIND_MULTIRES_SNAP_POS, &snap_pos_ssbo, snap_pos_ssbo.size },
        { BIND_MULTIRES_BASE,     base_b,     base_b->size },
        { BIND_UNDO_RING,         ring_b,     ring_b->size },
        { BIND_PARAMS,            &multires_diff_ubo, sizeof(MultiresDiffParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, multires_diff_pipeline, bg, 8);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, multires_diff_pipeline, grp, (count + 255u) / 256u);
    gpu::submit(b);   // seam issues SHADER_STORAGE|BUFFER_UPDATE so a debug readback sees it
    gpu::release_bind_group(grp);
}

// ---------------------------------------------------------------------------
// Phase 2c: undo/redo apply.
// ---------------------------------------------------------------------------

bool ComputeState::init_multires_apply() {
    if (!supported) return false;

    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,       gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS,     gpu::Bind::StorageRead,      0 },
        { BIND_MULTIRES_DISP,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_MULTIRES_FRAMES, gpu::Bind::StorageRead,      0 },
        { BIND_MULTIRES_BASE,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_MULTIRES_STAGE,  gpu::Bind::StorageRead,      0 },
        { BIND_UNDO_RING,       gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,          gpu::Bind::Uniform, sizeof(MultiresApplyParamsGPU) },
    };
    multires_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                  gpu::embedded_shader("multires_apply"), layout, 8);
    if (!multires_apply_pipeline.handle) {
        std::printf("[compute] multires_apply pipeline failed to compile\n");
        return false;
    }
    multires_apply_ubo = gpu::create_buffer(gpu_dev, nullptr,
                                            sizeof(MultiresApplyParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] multires_apply pipeline compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::dispatch_multires_apply(const gpu::Buffer& pos_vbo, const gpu::Buffer& disp_ssbo,
                                           const gpu::Buffer& frames_ssbo, const gpu::Buffer& base_ssbo,
                                           const uint32_t* verts, const float* stage,
                                           uint32_t count, bool targets_base,
                                           bool ring_ssbo, uint32_t ring_base_floats,
                                           bool forward) {
    if (!has_multires_apply() || count == 0) return;
    if (!pos_vbo.handle) return;
    if (targets_base && !base_ssbo.handle) return;
    if (!targets_base && (!disp_ssbo.handle || !frames_ssbo.handle)) return;

    // Ring mode (3b-iv): read (old,new) straight from the persistent undo ring, no
    // CPU stage upload. Stage mode: upload the (target,source) pairs as before.
    const bool ring_mode = (ring_ssbo != 0);

    // Upload the touched-vert list (reuse dirty_verts_ssbo, as the diff does). Seam-owned.
    if (!dirty_verts_ssbo.handle || count > dirty_verts_capacity) {
        uint32_t alloc_count = std::max(count, 4096u);
        gpu::release_buffer(dirty_verts_ssbo);
        dirty_verts_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                              (uint64_t)alloc_count * sizeof(uint32_t), gpu::Usage::Storage);
        dirty_verts_capacity = alloc_count;
    }
    gpu::write_buffer(gpu_dev, dirty_verts_ssbo, 0, verts, (uint64_t)count * sizeof(uint32_t));

    // Upload the (target, source) staging pairs (stage mode only). Seam-owned scratch (Step 3b).
    if (!ring_mode && stage) {
        if (!multires_stage_ssbo.handle || count > multires_stage_capacity) {
            uint32_t alloc_count = std::max(count, 4096u);
            gpu::release_buffer(multires_stage_ssbo);
            multires_stage_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                      (uint64_t)alloc_count * 6 * sizeof(float), gpu::Usage::Storage);
            multires_stage_capacity = alloc_count;
        }
        gpu::write_buffer(gpu_dev, multires_stage_ssbo, 0, stage, (uint64_t)count * 6 * sizeof(float));
    }

    MultiresApplyParamsGPU u = {};
    u.count        = count;
    u.targets_base = targets_base ? 1 : 0;
    u.ring_mode    = ring_mode ? 1 : 0;
    u.ring_base    = ring_mode ? ring_base_floats : 0u;
    u.forward      = forward ? 1 : 0;
    gpu::write_buffer(gpu_dev, multires_apply_ubo, 0, &u, sizeof(u));

    // Every declared binding must point at a live buffer (some drivers fault on an
    // unbound slot even if the shader never reads it). In ring mode the stage slot
    // is unused; in stage mode the ring slot is. The pool buffers are seam-owned now
    // (Step 3c) and bound directly; pos_vbo (always present) is the harmless filler
    // for whichever is idle and for any 0-handle slot the mode skips.
    const gpu::Buffer* disp_b  = disp_ssbo.handle   ? &disp_ssbo   : &pos_vbo;
    const gpu::Buffer* frame_b = frames_ssbo.handle ? &frames_ssbo : &pos_vbo;
    const gpu::Buffer* base_b  = base_ssbo.handle   ? &base_ssbo   : &pos_vbo;

    // stage + ring are seam-owned (Step 3b): bind the live member for the active mode,
    // pos as the filler for the idle one (every slot must point at a live buffer).
    const gpu::Buffer* stage_b = (!ring_mode && multires_stage_ssbo.handle)
                                     ? &multires_stage_ssbo : &pos_vbo;
    const gpu::Buffer* ring_b  = ring_mode ? &undo_ring_ssbo : &pos_vbo;

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,       &pos_vbo,   pos_vbo.size },
        { BIND_DIRTY_VERTS,     &dirty_verts_ssbo, (uint64_t)count * sizeof(uint32_t) },
        { BIND_MULTIRES_DISP,   disp_b,     disp_b->size },
        { BIND_MULTIRES_FRAMES, frame_b,    frame_b->size },
        { BIND_MULTIRES_BASE,   base_b,     base_b->size },
        { BIND_MULTIRES_STAGE,  stage_b,    stage_b->size },
        { BIND_UNDO_RING,       ring_b,     ring_b->size },
        { BIND_PARAMS,          &multires_apply_ubo, sizeof(MultiresApplyParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, multires_apply_pipeline, bg, 8);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, multires_apply_pipeline, grp, (count + 255u) / 256u);
    gpu::submit(b);   // seam issues SHADER_STORAGE|BUFFER_UPDATE for the follow-up normals/readback
    gpu::release_bind_group(grp);
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
//
// Raw GL buffer management (stays GL-owned through the port — the two seam kernels
// above wrap undo_ring_ssbo in a view at dispatch). Migrates to gpu::Buffer in the
// later buffer-ownership pass.
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
    if ((!undo_ring_ssbo.handle || undo_ring_head + bytes > undo_ring_bytes)
        && undo_ring_bytes < undo_ring_cap_bytes) {
        size_t newcap = undo_ring_bytes ? undo_ring_bytes : (16ull << 20);
        while (newcap < undo_ring_head + bytes && newcap < undo_ring_cap_bytes) newcap <<= 1;
        if (newcap > undo_ring_cap_bytes) newcap = undo_ring_cap_bytes;

        // Seam-owned now (Step 3b). The copy-preserving grow is a buffer→buffer copy
        // with no seam primitive yet, so it stays raw GL via .handle (web-stage concern).
        gpu::Buffer nb = gpu::create_buffer(gpu_dev, nullptr, (uint64_t)newcap, gpu::Usage::Storage);
        if (undo_ring_ssbo.handle && undo_ring_head) {
            glBindBuffer(GL_COPY_READ_BUFFER,  undo_ring_ssbo.handle);
            glBindBuffer(GL_COPY_WRITE_BUFFER, nb.handle);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
                                (GLsizeiptr)undo_ring_head);
            glBindBuffer(GL_COPY_READ_BUFFER,  0);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        }
        gpu::release_buffer(undo_ring_ssbo);
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
    gpu::write_buffer(gpu_dev, undo_ring_ssbo, (uint64_t)byte_off,
                      data, (uint64_t)(float_count * sizeof(float)));
    return byte_off;
}

void ComputeState::undo_ring_read(size_t byte_offset, size_t float_count, float* out) {
    if (!supported || !undo_ring_ssbo.handle || float_count == 0) return;
    // Readback: no seam primitive yet, stays raw GL via .handle (web-stage concern).
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, undo_ring_ssbo.handle);
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
