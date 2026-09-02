#pragma once
#include "mesh.h"
#include <cstdint>
#include <deque>
#include <vector>

class Scene;

// ---------------------------------------------------------------------------
// Object-transform undo (Select mode: move / scale / Q-E spin).
//
// A brush moves an arbitrary subset in an arbitrary way, so its undo entry has
// to store positions. An object transform is the SAME map applied to every
// vertex, so it stores as a handful of parameters — about 50 bytes per entity
// per gesture, against the 24 bytes PER VERTEX a position snapshot would need.
// On a million-vertex mesh that is the difference between 50 bytes and 24 MB,
// and a two-second hold of E would have emitted dozens of the latter. Undo is
// the exact parametric inverse.
//
// Why this is a SCENE-level stack and not part of UndoStack: UndoStack is
// per-entity, and one gesture can turn several selected meshes at once. And why
// it interleaves with UndoStack rather than living in its own siloed history:
// UndoEntry stores ABSOLUTE positions, so undoing a stroke that predates a
// transform would teleport those vertices back into the pre-transform frame.
// Strict last-in-first-out across both stacks is what makes that impossible —
// you can never undo *through* a transform. Both stacks stamp their entries
// from UndoStack::global_pushes, and Ctrl+Z pops whichever top is newer.
// ---------------------------------------------------------------------------

// One entity's share of one gesture.
struct ObjectXformTarget {
    uint32_t entity_id = 0;

    // --- MOVE ---
    // Total view-plane translation. The X rule is latched on the gesture's first
    // frame (see the move block in main.cpp): a piece centred on the mirror plane
    // locks X, a symmetrized pair moves as mirrored lobes. That lobe rule is
    // piecewise in x — no single matrix describes it — which is why these are
    // parameters and not an affine. Its inverse is exact anyway, because the
    // lobes keep their sign for the length of a sane drag.
    Vec3  delta        = {0, 0, 0};
    bool  lock_x       = false;
    // MOVE: the two lobes translate as mirrors (one right, its twin left).
    // SPIN: the two lobes turn as mirrors — see spin_apply_mesh().
    bool  mirror_lobes = false;
    float seam_eps     = 0.0f;   // latched too, so the replay splits the lobes identically

    // --- SCALE / SPIN ---
    // Pivot is latched with the gesture too. It is recomputed per frame by the
    // live code (a bounding centre is invariant under a scale or a turn about
    // itself), but only in exact arithmetic — replaying against the latched one
    // is what keeps undo from walking the object off its start position.
    // For a SPIN with mirror_lobes set, this is the +X LOBE's centre; the -x
    // lobe's is its reflection, which is why one Vec3 still describes both.
    Vec3  pivot  = {0, 0, 0};
    float factor = 1.0f;        // SCALE: accumulated exponential factor
    Vec3  axis   = {0, 0, 1};   // SPIN: view axis, latched with the pivot
    float angle  = 0.0f;        // SPIN: total radians
};

// The ONE definition of what a Select-mode spin does to a mesh. The live Q/E path
// and the undo replay both call it, so an inverse can never drift from the thing
// it inverts.
//
// Plain case: rotate every vertex about `pivot` by `angle` around `axis`.
//
// mirror_lobes: `pivot` is the +x lobe's centre, and the -x lobe gets the
// CONJUGATED map — reflect, turn, reflect back (M R M, about the reflected
// centre). That is exactly a turn of -angle about the reflected axis, and it is
// the only rotation of a mirrored pair that leaves it a mirrored pair: on screen
// the right lobe goes clockwise while the left goes anticlockwise. Without it a
// turn about anything but world X breaks the symmetry the mirror machinery
// assumes, and mirror_project then tears the mesh on the next stroke.
//
// The lobe a vertex belongs to is the sign of its x. That is only safe because
// spin_latch_target() refuses the lobe rule unless each lobe's own bounding
// sphere clears the plane — a lobe turning about its own centre stays inside
// that sphere, so no vertex can cross and get re-classified mid-gesture.
void spin_apply_mesh(Mesh& m, const Vec3& pivot, const Vec3& axis,
                     float angle, bool mirror_lobes);

// Latch one entity's pivot and lobe rule on the first frame of a spin gesture.
// Writes xt.pivot and xt.mirror_lobes; leaves angle/axis to the caller.
void spin_latch_target(const Mesh& m, bool mirror_on, ObjectXformTarget& xt);

struct ObjectXform {
    enum class Kind : uint8_t { MOVE, SCALE, SPIN };
    Kind kind = Kind::MOVE;
    std::vector<ObjectXformTarget> targets;
    uint64_t seq = 0;   // shared edit clock with UndoEntry::seq
};

class ObjectXformStack {
public:
    // --- gesture accumulation ---
    // A gesture is one press-to-release, not one frame: hold E for two seconds
    // and that is a single step back, not a hundred.
    void begin(ObjectXform::Kind k, Scene& scene);
    ObjectXformTarget& target(uint32_t entity_id);   // find-or-append, gesture must be open
    bool gesture_open() const { return open_; }
    ObjectXform::Kind gesture_kind() const { return pending_.kind; }
    // Push if the gesture actually moved something; no-op when none is open.
    // Takes the scene because landing a step has to kill the redo arm of every
    // entity it touched — the two stacks share one timeline, so a new step on
    // either branches away from whatever the other had queued.
    void commit(Scene& scene);
    void abandon();

    // --- history ---
    bool     can_undo() const { return !undo_.empty(); }
    bool     can_redo() const { return !redo_.empty(); }
    uint64_t undo_seq() const { return undo_.empty() ? 0 : undo_.back().seq; }
    uint64_t redo_seq() const { return redo_.empty() ? 0 : redo_.back().seq; }
    // Both return whether the geometry actually changed. Caller re-syncs GPU
    // residency, refreshes framing bounds and dirties the screen buffers.
    bool undo(Scene& scene);
    bool redo(Scene& scene);
    // Id of a targeted entity whose own sculpt history has an entry NEWER than
    // the transform on top, or 0 if none. UndoStack is per-entity and Ctrl+Z
    // only ever pops the ACTIVE one, so the seq comparison at the call site
    // orders this stack against that entity alone. If a transform moved three
    // meshes and one of the other two was sculpted afterwards, undoing the
    // transform now would leave that stroke's absolute positions stranded in a
    // frame the mesh is no longer in. Refuse and say which mesh to clear first;
    // closing the hole properly needs the scene-level undo that road2v2 backlog
    // item 4 is about.
    uint32_t newer_edit_entity(Scene& scene) const;
    // Was the step just undone/redone a spin? Tangent frames are directions on
    // the base surface, so only a turn invalidates them.
    bool last_was_spin() const { return last_spin_; }
    // "move" / "scale" / "rotation" — for the notification line.
    const char* last_label() const;

    void clear();
    void clear_redo();

    // global_pushes as of the last undo. Any push by anyone since then means the
    // user branched off this history, so the redo arm is dead — same rule
    // UndoStack::push applies to its own redo stack, just across two stacks.
    uint64_t redo_clock = 0;

private:
    void apply(const ObjectXform& x, bool inverse, Scene& scene);

    ObjectXform pending_;
    bool        open_ = false;
    bool        last_spin_ = false;
    ObjectXform::Kind last_kind_ = ObjectXform::Kind::MOVE;
    std::deque<ObjectXform> undo_, redo_;
    std::vector<uint32_t>   dirty_;   // reused full-range dirty list
};
