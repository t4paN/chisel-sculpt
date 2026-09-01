#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include "entity_record.h"

class Scene;

// Rescue copy of the whole scene, taken immediately before an operation that
// wipes the undo history.
//
// Only the two remeshers (isotropic '/' and SDF merge 'J') get one, and the
// reason is not how destructive they are — it is WHEN you find out. Ctrl+O and
// drop-subdiv are pressed with intent and their result is visible the instant
// they land. A remesh you cannot judge until the old model is already gone: it
// reports a healthy-looking vertex count and you only notice the silhouette
// went soft after you have been sculpting on it for a minute.
//
// Deliberately NOT part of UndoStack. Undo entries address vertices by index,
// and every index is meaningless the moment topology changes — which is exactly
// what these two ops do. A whole-scene copy sidesteps that instead of fighting it.
struct SceneSnapshot {
    std::vector<EntityRecord> entities;
    std::vector<uint32_t>     selected_ids;
    uint32_t active_id = 0;
    uint32_t next_id   = 1;
    bool     mirror_use_topology = true;

    // Names the op in the countdown notifications ("remesh" / "SDF merge").
    const char* op = "";

    // UndoStack::global_pushes at capture time. The snapshot survives the next
    // few edits rather than dying on the first, because stroking a fresh remesh
    // is HOW you evaluate it — expiring on stroke one would throw the rescue
    // away at precisely the moment it is wanted.
    uint64_t edits_at_capture = 0;
    static constexpr int GRACE_EDITS = 3;

    bool valid() const { return !entities.empty(); }
    // Edits remaining before this expires. 0 = expired (or never captured).
    int  edits_left() const;
    void clear();
};

// Exact byte cost of snapshotting `scene`, computed from container sizes without
// copying anything. The gate MUST be pre-flight: on web the 32-bit WASM heap has
// ALLOW_MEMORY_GROWTH with no ceiling, and a growth that fails aborts the whole
// tab rather than returning a null we could back out of.
size_t snapshot_bytes(const Scene& scene);
// 512 MB on every platform. Web used to get a smaller share, but the user's call
// (2026-09-01) is to run the same budget everywhere and let a machine that cannot
// take it find out — see snapshot_budget() for what that costs on web.
size_t snapshot_budget();
bool   snapshot_affordable(const Scene& scene);

// Deep-copy the scene. Call snapshot_affordable() first.
void snapshot_capture(SceneSnapshot& snap, Scene& scene, const char* op);

// Rebuild the scene from `snap`, consuming it. Restores per-entity subdiv levels,
// the selection and the mirror mode; does NOT restore the camera — the user will
// have orbited around to inspect the result, and yanking the view back reads as a
// bug rather than an undo. Caller re-fetches mesh/multires, refreshes the mirror
// map and syncs, exactly as after a project load.
void snapshot_restore(SceneSnapshot& snap, Scene& scene);
