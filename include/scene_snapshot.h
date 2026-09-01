#pragma once
#include <vector>
#include <cstdint>
#include "entity_record.h"

class Scene;
struct ComputeState;

// Rescue copy of the scene, taken immediately before an operation that wipes the
// undo history. Only the two remeshers ('/' and 'J') get one, and not because
// they are the most destructive — because they are the only ones you cannot
// judge until the old model is already gone. Ctrl+O and drop-subdiv are pressed
// with intent and show their result instantly.
//
// v2, after v1 crashed the itch build. v1 deep-copied everything — surface,
// normals, CSR adjacency, mirror maps, tangent frames, topo cache, stack — at
// ~270 bytes per vertex, inside a 32-bit WASM heap, on top of the live scene and
// up to 1 GB of undo history. This version stores the ONLY thing that cannot be
// recomputed:
//
//   the multires stack — base cage, displacement layers, and the paint / mask /
//   density planes — and nothing else.
//
// Everything a snapshot used to carry is derived from that: Scene::load_entities
// cascades the surface back out of the stack and rebuilds normals, adjacency and
// mirror maps for every entity. So the copy is ~28 bytes per finest-level vertex
// instead of ~270, and the undo history is freed BEFORE it is taken rather than
// sitting alongside it.
struct SceneSnapshot {
    std::vector<EntityRecord> entities;
    std::vector<uint32_t>     selected_ids;
    uint32_t active_id = 0;
    uint32_t next_id   = 1;
    bool     mirror_use_topology = true;

    // Names the op in the countdown messages ("remesh" / "SDF remesh").
    const char* op = "";
    // Subdivision levels baked down by snapshot_prepare to get under the cap.
    int levels_baked = 0;

    // UndoStack::global_pushes at capture. The snapshot survives a few edits
    // rather than dying on the first, because stroking a fresh remesh is HOW you
    // evaluate it — expiring on stroke one would throw the rescue away at exactly
    // the moment it is wanted.
    uint64_t edits_at_capture = 0;
    static constexpr int GRACE_EDITS = 3;

    bool valid() const { return !entities.empty(); }
    int  edits_left() const;
    void clear();
};

// The cap is in TRIANGLES, not bytes: it is the number a sculptor can reason
// about, and past it the copy stops being cheap enough to be automatic.
constexpr uint64_t SNAPSHOT_TRI_CAP = 1000000;

// Triangles the scene carries at its entities' top levels. Read off the stacks
// (base tris * 4^layers — Loop quadruples exactly), never off the working mesh,
// so it stays correct while a bake is in flight.
uint64_t snapshot_scene_tris(const Scene& scene);

// Top subdivision levels that must be baked down before the scene fits the cap.
// 0 = fits already. -1 = cannot fit even with every layer gone. Pure query, safe
// to call every frame while a confirm prompt is up.
int snapshot_levels_to_bake(const Scene& scene);

// Free the undo history and bake the scene down until it fits the cap. Returns
// the number of levels baked, or -1 if it still does not fit (caller runs the op
// unprotected). MUTATES THE SCENE: baking is the same destructive project-down
// the burger menu's "delete highest subdiv" performs, so the prompt must have
// told the user how many levels it will cost before they said yes.
//
// The undo clear is not just tidiness — it is what makes the copy affordable.
// The op about to run wipes that history anyway, so spending it first costs
// nothing and takes up to 1 GB out of the peak.
int snapshot_prepare(Scene& scene, ComputeState& compute);

// Copy the prepared scene. Call snapshot_prepare first and only on a >= 0 return.
void snapshot_capture(SceneSnapshot& snap, Scene& scene, const char* op);

// Rebuild the scene from `snap`, consuming it. Restores the stacks, the paint and
// mask, the selection and the mirror mode, and leaves every undo history empty —
// the entries were spent before capture and their vertex ids died with the
// topology regardless. Does NOT restore the camera: the user will have orbited
// around to inspect the result, and yanking the view back reads as a bug rather
// than an undo. Caller re-fetches mesh/multires, refreshes the mirror map, syncs.
void snapshot_restore(SceneSnapshot& snap, Scene& scene);
