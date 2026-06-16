#pragma once
#include <glad/glad.h>
#include <cstdint>
#include <vector>

struct MultiresStack;

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

    void cleanup();
};
