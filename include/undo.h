#pragma once
#include "mesh.h"
#include "multires_stack.h"
#include <cstdint>
#include <deque>
#include <vector>

class Scene;
struct MeshEntity;
struct ComputeState;

struct UndoEntry {
    enum class Kind { STROKE, PROJECTION, MASK, LEVEL, PAINT };
    Kind kind = Kind::STROKE;

    // --- STROKE fields ---
    // verts holds LOCAL vertex indices relative to entity_id's mesh.
    std::vector<uint32_t> verts;
    std::vector<float> old_x, old_y, old_z;
    std::vector<float> new_x, new_y, new_z;

    // --- MASK fields ---
    std::vector<float> old_mask, new_mask;

    // --- PAINT fields --- (packed RGBA8, parallel to verts; level-agnostic like mask)
    std::vector<uint32_t> old_color, new_color;

    // Which storage layer this entry modifies
    int  level        = 0;
    bool targets_base = true;
    int  disp_index   = -1;   // -1 if targets_base, else level - base_level - 1

    // --- GPU undo ring (blood-moon 3b-iv) ---
    // When this STROKE's (old,new) deltas were captured into the GPU undo ring at
    // pen-up, ring_offset is the FLOAT offset of its span and ring_vcount the vert
    // count (== verts.size(), since ring capture records the unfiltered snap_list so
    // ring slot k aligns with verts[k]). SIZE_MAX => not in the ring; apply falls
    // back to the CPU old_*/new_* arrays. Part 1 keeps the CPU arrays authoritative;
    // they're the spill target once Part 2 drops the pen-up readback.
    size_t   ring_offset = SIZE_MAX;
    uint32_t ring_vcount = 0;

    // --- PROJECTION fields ---
    // Pre-projection snapshot of the affected storage. On undo, restore this.
    // On redo, re-run project_down_to_level(target_level).
    int              target_level = 0;
    MultiresSnapshot before;

    // --- LEVEL fields (multires view-level change; D / Shift-D) ---
    // The view moved from_level -> to_level. `before` (above) is populated only
    // for a descend that auto-projected; empty for a non-destructive ascend.
    int from_level = 0;
    int to_level   = 0;
};

class UndoStack {
public:
    // Total CPU undo-history budget in RAM (oldest entries evicted past this).
    // Runtime-configurable: defaults to 1 GB, dropped to 256 MB by --toaster. This
    // is the per-entity deep-history bound; the GPU ring below is a smaller hot
    // cache layered on top of it.
    static size_t max_bytes;

    // GPU-resident undo ring budget in VRAM (blood-moon 3b). Decoupled from
    // max_bytes (3b-iv part 2): the ring only caches the *active* entity's recent
    // strokes for zero-readback undo, so it's deliberately small. 256 MB default,
    // 64 MB with --toaster. Deep history beyond the ring lives in CPU RAM.
    static size_t ring_max_bytes;

    void push(UndoEntry&& e);
    bool can_undo() const { return !undo_stack.empty(); }
    bool can_redo() const { return !redo_stack.empty(); }

    // Undo/redo act on the entity that owns this stack (always the active one).
    // Returns true (needs_cascade) when the reverted layer sits below the
    // entity's current view level (entry.level < current_level) — the caller
    // must cascade_to_level(current_level) on `ent` then re-sync. Otherwise the
    // view is updated in place via scene.sync_partial_entity (entry.level ==
    // current_level) or left alone (entry.level > current_level).
    bool undo(MeshEntity& ent, Scene& scene);
    bool redo(MeshEntity& ent, Scene& scene);

    // Pass the ComputeState to also reset the GPU undo ring (blood-moon 3b-iv 2c) —
    // ONLY when clearing the ACTIVE entity's stack, since the ring caches the active
    // entity. A non-active clear must pass nullptr so it doesn't wipe the live ring.
    void clear(ComputeState* c = nullptr);

    // GPU undo ring (blood-moon 3b-iv part 2). Called from brush finalize right
    // after a new stroke's ring span [byte_off, byte_off+byte_len) is reserved and
    // before the diff shader fills it: invalidate (and, under debug, spill-validate
    // against the CPU arrays) any older STROKE entry whose ring bytes that write
    // overwrites, so it falls back to the CPU stage on a future apply. Keeps the
    // circular ring honest — an entry either points at its own live data or is
    // marked non-resident (ring_offset == SIZE_MAX).
    void ring_evict_overlap(size_t byte_off, size_t byte_len, ComputeState& c);

    // Park the GPU undo ring on active-entity switch (blood-moon 3b-iv 2c): mark
    // every resident STROKE entry non-resident (→ CPU stage on a future apply) and
    // reset the ring to head 0 so the incoming active entity starts fresh — the ring
    // is a hot cache for the active entity only. From 2c-iii on, spills each resident
    // entry's (old,new) from the ring into its CPU arrays first (the entries are this
    // entity's history and must survive the switch).
    void ring_park_all(ComputeState& c);

    size_t bytes_used() const { return total_bytes; }
    size_t undo_depth() const { return undo_stack.size(); }
    const UndoEntry* peek_undo() const { return undo_stack.empty() ? nullptr : &undo_stack.back(); }
    const UndoEntry* peek_redo() const { return redo_stack.empty() ? nullptr : &redo_stack.back(); }

private:
    std::deque<UndoEntry> undo_stack;
    std::deque<UndoEntry> redo_stack;
    size_t total_bytes = 0;
    std::vector<uint32_t> scratch_dirty;
    std::vector<uint32_t> scratch_gpu;

    static size_t entry_bytes(const UndoEntry& e) {
        size_t b;
        if (e.kind == UndoEntry::Kind::PROJECTION ||
            e.kind == UndoEntry::Kind::LEVEL) {
            b = e.before.bytes();
        } else if (e.kind == UndoEntry::Kind::MASK) {
            b = e.verts.size() * (sizeof(uint32_t) + 2 * sizeof(float));
        } else if (e.kind == UndoEntry::Kind::PAINT) {
            b = e.verts.size() * (sizeof(uint32_t) + 2 * sizeof(uint32_t));
        } else {
            b = e.verts.size() * (sizeof(uint32_t) + 6 * sizeof(float));
        }
        return b;
    }
    void evict_to_budget();
    // apply() reverts/replays the entry on the active entity `ent`. Non-const: a
    // cross-level apply of a still-ring-resident entry spills its (old,new) out of
    // the ring into the entry's CPU arrays (2c-iii) before reading them.
    bool apply(UndoEntry& e, MeshEntity& ent, Scene& scene, bool forward);
    bool apply_projection(const UndoEntry& e, MeshEntity& ent, bool forward);
    bool apply_level(const UndoEntry& e, MeshEntity& ent, bool forward);
};
