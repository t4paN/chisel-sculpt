#pragma once
#include "gpu/gpu.h"
#include <cstdint>
#include <memory>

struct MeshTexture;

// A textured mesh's extras on the GPU (an imported glTF): its corner UVs and its
// image. corners == 0 = untextured, drawn by the plain matcap pipeline. `src` is the
// image `tex` holds, kept alive so the identity check can never match a freed one.
struct TexturedGpu {
    gpu::Buffer  uv;            // 2 floats per corner, read-only storage
    gpu::Texture tex;
    std::shared_ptr<const MeshTexture> src;
    uint32_t     corners = 0;
};

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
    TexturedGpu textured;
    uint32_t index_count = 0;
    bool dirty = true;       // needs re-upload from entity->mesh
};
