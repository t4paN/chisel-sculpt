#pragma once
#include <glad/glad.h>
#include <cstdint>
#include <vector>

struct MultiresStack;
struct Mesh;

// GPU residency mirror for the *active editing level* of a MultiresStack.
//
// Phase 1 of the GPU-resident-undo chain (see checked5jun-gpu-undo-groundwork.md):
// CPU disp/frames/base remain the source of truth; this struct keeps an SSBO copy
// of the layer currently being edited so the Phase-2 GPU diff/apply shaders have
// something to read/write directly. Buffers are grow-only and persist across
// strokes — no hot-path allocation. The whole struct is inert unless `supported`
// is set (mirrors ComputeState::supported); every consumer keeps its CPU path when
// compute is unavailable.
//
// Layout: disp_ssbo / base_ssbo are interleaved float3 per vertex (matches Vec3 and
// the working VBO position layout). frames_ssbo is 9 floats per vertex (t,b,n) —
// matches the CPU Frame struct, uploaded directly.
struct MultiresGPU {
    bool supported = false;

    // Absolute subdivision level currently mirrored (-1 = nothing uploaded yet).
    int   level = -1;

    GLuint disp_ssbo   = 0;   // float3 * V  (valid when level > base_level)
    GLuint frames_ssbo = 0;   // float9 * V  (valid when level > base_level)
    GLuint base_ssbo   = 0;   // float3 * V  (valid when level == base_level)
    uint32_t capacity  = 0;   // verts the disp/frames buffers can hold
    uint32_t base_capacity = 0;

    // Phase 2: pen-down snapshot of the working VBO positions (world space). The
    // GPU brush overwrites the VBO in place during the stroke, so the pre-stroke
    // world pos is gone by pen-up unless captured here. The pen-up diff shader
    // reads this + frames_ssbo + disp_ssbo (the latter already holds the pen-down
    // disp, since Phase 1 only re-syncs it at pen-up) to reproject into disp/base.
    GLuint snap_pos_ssbo   = 0;   // float3 * V, snapshot at pen-down
    uint32_t snap_pos_capacity = 0;

    // Allocate/resize buffers to hold `vertex_count` (active level) and
    // `base_vertex_count` verts. Grow-only; no-op if already large enough.
    void ensure(uint32_t vertex_count, uint32_t base_vertex_count);

    // Full re-upload of the active layer (`abs_level`) from CPU storage.
    // Call after lock, level switch, projection — any wholesale CPU mutation.
    void upload_level(const MultiresStack& stack, int abs_level);

    // Partial re-upload of just the listed verts of the active layer, after a
    // pen-up disp writeback or an undo/redo storage edit. `verts` need not be
    // sorted; uploaded as coalesced contiguous runs (mirrors the banded readback
    // in brush.cpp). No-op unless `abs_level == level` (the mirrored layer).
    void upload_disp_partial(const MultiresStack& stack, int abs_level,
                             const std::vector<uint32_t>& verts);

    // Snapshot the working VBO positions [0, vertex_count) into snap_pos_ssbo at
    // pen-down. GPU→GPU copy (glCopyBufferSubData), grow-only, one-shot per stroke.
    // No behavior dependence yet — consumed by the Phase-2 pen-up diff shader.
    void snapshot_positions(GLuint pos_vbo, uint32_t vertex_count);

    // ---- Phase 3 (deferred CPU writeback) -------------------------------------
    // From 3b on, the pen-up diff shader writes disp/base on the GPU and the CPU
    // copy is NOT refreshed inline. Instead the GPU is the source of truth and the
    // CPU `stack.disp[k]`/`stack.base` go stale; the affected verts accumulate here.
    // Every CPU consumer of disp/base (cascade, projection, save) must call
    // materialize_cpu() first to pull the GPU truth back down. Until 3b nothing
    // calls mark_cpu_dirty(), so materialize_cpu() is always a no-op early-return —
    // i.e. this machinery is inert and behavior-neutral when introduced in 3a.
    //
    // This syncs BOTH the *storage* layer (disp/base from the SSBOs) and the live
    // surface `mesh.pos` (from the working VBO) for the dirty verts — the surface
    // readers (remesh, sdf, mirror, bounding sphere) need the latter, the storage
    // readers (save, cascade, projection) the former. Same dirty set covers both
    // (a stroke makes the surface and its storage layer stale together). 2c-i added
    // the `mesh`/`vbo_pos` arms; still inert until 2c-iv starts marking dirty.
    bool cpu_dirty = false;
    std::vector<uint32_t> dirty_verts;   // active-level verts whose CPU disp/base/pos is stale

    // Accumulate verts whose GPU disp/base/pos now diverges from the CPU copy. Cheap;
    // no readback. Called at pen-up (and undo/redo) in place of the CPU writeback.
    void mark_cpu_dirty(const std::vector<uint32_t>& verts);

    // Pull the GPU disp/base (from the SSBOs) AND mesh.pos (from `vbo_pos`, the active
    // working VBO) for the dirty verts of the mirrored level back into CPU storage
    // (banded readback, inverse of upload_disp_partial), then clear the dirty set.
    // No-op unless cpu_dirty. Call before any CPU read of the active level's
    // disp/base/pos. `vbo_pos` is the active entity's working position VBO (offset 0).
    void materialize_cpu(MultiresStack& stack, Mesh& mesh, GLuint vbo_pos);

    void cleanup();
};
