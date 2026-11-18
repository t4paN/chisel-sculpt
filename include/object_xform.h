#pragma once
#include "mesh.h"
#include <cstdint>
#include <deque>
#include <vector>

class Scene;
class Camera;
struct ImDrawList;

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
// per-entity, and one gesture can turn several selected meshes at once.
//
// The two histories are SEPARATE, one per mode: Ctrl+Z in Sculpt walks the
// strokes, Ctrl+Z in Object walks the transforms, and neither touches the other.
// (Until 2026-11-18 they shared one timeline, and a Ctrl+Z in Sculpt took back a
// turn the user had made in Object mode.) What makes that safe: UndoEntry stores
// ABSOLUTE positions, so every transform that lands on a mesh — done, undone or
// redone — is also appended to that mesh's pose log (UndoStack::note_pose). An
// entry remembers how long the log was when it was captured, and before it is
// applied its positions are carried through every pose step since (PoseStep,
// pose_map_points). A stroke from before a turn then comes off the TURNED mesh
// in the right place instead of teleporting back into the old frame.
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
    // MOVE with Mirror off: only the half on this side of the mirror plane moves (+1 / -1;
    // 0 = the whole mesh). Live and replay both pick that half out by the sign of x, which
    // is exact because the live move never lets the half cross the plane.
    int8_t lobe        = 0;

    // --- SCALE / SPIN ---
    // Pivot is latched with the gesture too. It is recomputed per frame by the
    // live code (a bounding centre is invariant under a scale or a turn about
    // itself), but only in exact arithmetic — replaying against the latched one
    // is what keeps undo from walking the object off its start position.
    // For a SPIN with mirror_lobes set, this is the +X LOBE's centre; the -x
    // lobe's is its reflection, which is why one Vec3 still describes both.
    Vec3  pivot  = {0, 0, 0};
    Vec3  scale  = {1, 1, 1};   // SCALE: accumulated factor per world axis (RMB: all three alike)
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

// The ONE definition of a scale: every vertex about `pivot`, by f per world axis. Live
// drags and the undo replay both call it. A scale that differs between axes also bends
// the normals (they take the inverse factors, renormalised) and, under multires, the
// tangent frames, which is why it counts as a turn for last_bent_frames().
void scale_apply_mesh(Mesh& m, const Vec3& pivot, const Vec3& f);
inline bool scale_is_uneven(const Vec3& f) { return f.x != f.y || f.y != f.z; }

// Latch one entity's pivot and lobe rule on the first frame of a spin gesture.
// Writes xt.pivot and xt.mirror_lobes; leaves angle/axis to the caller.
void spin_latch_target(const Mesh& m, bool mirror_on, ObjectXformTarget& xt);

// The point maps under the three mesh functions: positions only, any arrays. The
// mesh functions call these for their positions, and so does the pose log replay,
// so a stroke carried across a transform lands exactly where the mesh went.
void move_apply_points(const ObjectXformTarget& t, const Vec3& d,
                       float* x, float* y, float* z, size_t n);
void spin_apply_points(const Vec3& pivot, const Vec3& axis, float angle, bool mirror_lobes,
                       float* x, float* y, float* z, size_t n,
                       float* nx = nullptr, float* ny = nullptr, float* nz = nullptr);
void scale_apply_points(const Vec3& pivot, const Vec3& f,
                        float* x, float* y, float* z, size_t n);

struct ObjectXform {
    enum class Kind : uint8_t { MOVE, SCALE, SPIN };
    Kind kind = Kind::MOVE;
    std::vector<ObjectXformTarget> targets;
    uint64_t seq = 0;   // edit clock; orders this stack against mesh deletes/renames
};

// One transform as it landed on one mesh: the gesture, or its inverse when it was
// undone. Appended to that mesh's pose log (UndoStack::note_pose).
struct PoseStep {
    ObjectXform::Kind kind = ObjectXform::Kind::MOVE;
    ObjectXformTarget t;
    bool inverse = false;
    // A turn or an uneven scale: directions on the surface changed with it.
    bool bends() const {
        return kind == ObjectXform::Kind::SPIN
            || (kind == ObjectXform::Kind::SCALE && scale_is_uneven(t.scale));
    }
};
void pose_map_points(const PoseStep& s, float* x, float* y, float* z, size_t n);

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
    // Takes the scene because each mesh the step moved logs it as a pose step.
    void commit(Scene& scene);
    void abandon();

    // --- history ---
    bool     can_undo() const { return !undo_.empty(); }
    bool     can_redo() const { return !redo_.empty(); }
    uint64_t undo_seq() const { return undo_.empty() ? 0 : undo_.back().seq; }
    uint64_t redo_seq() const { return redo_.empty() ? 0 : redo_.back().seq; }
    size_t   undo_depth() const { return undo_.size(); }
    size_t   redo_depth() const { return redo_.size(); }
    // Both return whether the geometry actually changed. Caller re-syncs GPU
    // residency, refreshes framing bounds and dirties the screen buffers.
    bool undo(Scene& scene);
    bool redo(Scene& scene);
    // Did the step just undone/redone bend directions on the surface — a spin, or a
    // scale that differs between axes? Tangent frames are directions on the base
    // surface, so only those invalidate them.
    bool last_bent_frames() const { return last_spin_; }
    // "move" / "scale" / "rotation" — for the notification line.
    const char* last_label() const;

    void clear();

private:
    void apply(const ObjectXform& x, bool inverse, Scene& scene);

    ObjectXform pending_;
    bool        open_ = false;
    bool        last_spin_ = false;
    ObjectXform::Kind last_kind_ = ObjectXform::Kind::MOVE;
    std::deque<ObjectXform> undo_, redo_;
    std::vector<uint32_t>   dirty_;   // reused full-range dirty list
};

// ---------------------------------------------------------------------------
// Object mode's Move, Scale and Rotate gizmos: drawn on the selection, grabbed by the
// press. World axes (a transform is baked into the vertices, so there is no local
// frame), a fixed size on screen. Move and Scale: one arm per axis (a cone on Move's
// tips, a square on Scale's), a right-angle corner between each two (those two
// together) and a centre ring (Move: free in the view plane; Scale: all three).
// Rotate: one ring per axis.
// ---------------------------------------------------------------------------
enum GizmoKind { GK_MOVE, GK_SCALE, GK_ROTATE };

enum GizmoHandle {
    GZ_X, GZ_Y, GZ_Z,          // an axis arm (Move / Scale) or the ring about that axis (Rotate)
    GZ_YZ, GZ_ZX, GZ_XY,       // Move / Scale: the corner between two axes (GZ_YZ = 3 + the axis left out)
    GZ_ALL,                    // Move / Scale: the centre ring (Move: the view plane; Scale: all three)
};

struct GizmoView {
    static constexpr int kRingN = 64;
    static constexpr int kConeN = 16;
    bool  ok = false;
    float cx = 0, cy = 0;                 // the pivot on screen
    float arm_world = 0;                  // an arm's length in world units
    float px_world = 0;                   // the world size of a screen pixel at the pivot
    float ax[3][2] = {};                  // each axis arm, screen px (foreshortened)
    float ring[3][kRingN][2] = {};        // Rotate: the rings, screen px
    bool  front[3][kRingN] = {};          // that ring point is nearer the eye than the pivot
    float cone[3][kConeN + 1][2] = {};    // Move: each tip's cone, its base ring then the apex
    bool  cone_base_shows[3] = {};        // the cone points away from the eye: its base faces you
};

// The selection's centre: the mean of its meshes' centres, cached until the scene or the
// selection changes, or selection_centre_moved() says a live drag shifted it. Under
// `mirror_scale` its x is 0, the plane the RMB scale works about with Mirror on.
bool selection_centre(Scene& scene, bool mirror_scale, Vec3& out,
                      uint32_t lobe_entity = 0, int lobe_side = 0);
// Lobe moves (Mirror off, a symmetrized pair). A pair is two halves either side of the
// mirror plane with no triangle touching or crossing it (band = how near counts as on it).
bool  mesh_is_pair(const Mesh& m, float band);
// The half (+1 / -1) whose vertices come nearest the pointer on screen; 0 if none show.
int   lobe_under_pointer(const Mesh& m, const Camera& cam, int win_w, int win_h, float mx, float my);
// The half's nearest approach to the plane, as |x|; a huge value when that half is empty.
float lobe_inner_x(const Mesh& m, int side);
// Under a lobe move, the gizmo sits on the chosen half of the one selected pair
// (lobe_entity / lobe_side); 0 / 0 = the whole selection's centre.
void selection_centre_moved();

bool gizmo_view(const Camera& cam, const Vec3& pivot, int win_w, int win_h, int kind, GizmoView& g);
int  gizmo_hit(const GizmoView& g, int kind, float mx, float my);   // a GizmoHandle, or -1
void gizmo_draw(ImDrawList* dl, const GizmoView& g, int kind, int hot);
// Which gizmo the current Object tool shows, or -1 for none (Select).
inline int gizmo_kind_of(int object_tool) { return object_tool == 1 ? GK_MOVE : object_tool == 2 ? GK_SCALE
                                                 : object_tool == 3 ? GK_ROTATE : -1; }
