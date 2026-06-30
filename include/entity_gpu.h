#pragma once
#include "gpu/gpu.h"
#include <cstdint>

// Static display GPU resources for an INACTIVE entity. The active entity is
// drawn from the renderer's working buffers (vbo_*/ebo); every other alive
// entity owns one of these lightweight indexed buffer sets for the viewport
// draw and the entity-id pick pass. Refreshed (dirty) on geometry change or
// when the entity is flushed out of the working set on selection change.
// Seam-owned gpu::Buffers — the draw-time VAO/vertex-layout lives on the
// render pipeline, so there is no per-entity VAO.
struct EntityGpu {
    gpu::Buffer vbo_pos;
    gpu::Buffer vbo_norm;
    gpu::Buffer vbo_mask;
    gpu::Buffer vbo_color;
    gpu::Buffer ebo;
    uint32_t index_count = 0;
    bool dirty = true;       // needs re-upload from entity->mesh
};
