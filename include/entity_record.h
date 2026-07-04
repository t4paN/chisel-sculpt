#pragma once
#include <cstdint>
#include "mesh.h"
#include "multires_stack.h"

// Plain, GPU-free description of one scene entity for serialization and scene
// reconstruction. Mirrors the *persistent* fields of MeshEntity (id,
// subdiv_level, mesh, multires) without the UndoStack or GL resources, so it
// can be shared by the project-file layer and the Scene rebuild path without
// either pulling in OpenGL.
struct EntityRecord {
    uint32_t      id           = 0;
    uint32_t      subdiv_level = 4;
    Mesh          mesh;
    MultiresStack multires;
    // Set by the project loader for v<=3 files: multires disp layers are indexed
    // under the saving platform's legacy (hash-order) midpoint numbering and must
    // be migrated to canonical numbering before the stack is usable. Transient —
    // never persisted.
    bool          legacy_numbering = false;
};
