#pragma once
#include "mesh.h"
#include "multires_stack.h"
#include "multires_gpu.h"
#include "undo.h"
#include "entity_gpu.h"

// A self-contained mesh entity: owns its Mesh data and MultiresStack
// independently of every other entity in the scene.
//
// Scene keeps a vector<unique_ptr<MeshEntity>>. Pointers are stable (no
// moves after construction), so callers may hold Mesh& / MultiresStack&
// references across frames as long as the entity stays alive.
struct MeshEntity {
    uint32_t id             = 0;
    uint32_t subdiv_level   = 4;
    bool     preview        = false;
    bool     alive          = true;

    Mesh          mesh;
    MultiresStack multires;

    // GPU residency mirror of the active editing level (Phase 1 of GPU-resident
    // undo). CPU `multires` stays authoritative; this is a read-side SSBO copy
    // kept in sync at each CPU mutation. Inert unless compute is supported.
    MultiresGPU   multires_gpu;

    // Per-model undo/redo history. Rides with the entity: set aside when the
    // entity stops being active, resumed when it is reselected. Undo always
    // targets the active entity, so no cross-entity routing is needed.
    UndoStack     undo;

    // Static display GL buffers, used when this entity is NOT the active one.
    // The active entity is drawn from the renderer's working buffers instead.
    EntityGpu     gpu;
};
