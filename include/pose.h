#pragma once
#include "mesh.h"
#include <cstdint>
#include <vector>

struct Camera;

// Pose brush: one hinge at a time. A line drawn along a limb leaves two handles in
// the middle of its volume: the ANCHOR where the line started (the joint) and the
// LEVER where it ended. Dragging the lever swings everything past the anchor about
// it; hardness sets how soft the bend is across the anchor. Drawing a new line
// replaces the hinge, so there is no rig to manage.
//
// Lives on the MeshEntity, in world space like the mesh. Not saved with the project.
// A swing is on the undo stack with the mesh, so the lever must follow it back
// (PoseMove below), or the next swing would start from where the mesh no longer is.

// One swing, keyed by the undo entry its stroke pushed: undoing that entry puts the
// lever back where the mesh went back to.
struct PoseMove { uint64_t seq; Vec3 from, to; };

struct PoseHinge {
    bool  set = false;
    Vec3  anchor, lever;
    float radius = 0.0f;   // the limb's half-thickness at the anchor: scales the bend
    std::vector<PoseMove> moves;

    void clear() { set = false; moves.clear(); }
    void on_undo(uint64_t seq);
    void on_redo(uint64_t seq);
};

// Handle ids for pose_pick and the hover state.
constexpr int kPoseAnchor = 0;
constexpr int kPoseLever  = 1;

// The cursor's view ray. Under ortho the origin moves with the pixel and the
// direction is the view axis; under perspective it is the eye and the pixel's
// direction. Either way the parameter along it is view depth (Camera::view_depth).
void pose_screen_ray(const Camera& cam, float sx, float sy, int w, int h,
                     Vec3& origin, Vec3& dir);

// Where a handle placed under the cursor sits: half-way between the first surface
// the ray enters and the next one it leaves through, so it lands inside the limb,
// not on its skin. radius = half that thickness. False when the ray misses the mesh;
// an open mesh with no far wall gives the entry point and radius 0.
bool pose_volume_center(const Mesh& mesh, const Camera& cam, float sx, float sy,
                        int w, int h, Vec3& center, float& radius);

// Screen pixels per world unit at the depth of `p` (the foreshortening term).
float pose_px_per_world(const Camera& cam, Vec3 p, int h);

// What one swing carries (pose_bind): the vertices, their pen-down positions and 8
// weights each. pose_apply's four lanes are the swing at 1/4, 2/4, 3/4 and all of
// its angle; a vertex part-way through the bend sits between two neighbouring lanes,
// so it turns by its share of the angle instead of cutting the corner (one lane
// blended linearly shrinks the joint on a big swing). The second four are the same
// for the mirrored hinge (zero with mirror off).
struct PoseBinding {
    std::vector<uint32_t> ids;
    std::vector<float>    init;   // 3 per id
    std::vector<float>    w;      // 8 per id
};

// Binds the hinge to the mesh at pen-down. What moves is what the lever's limb
// reaches past a cut through the anchor: a flood over the surface from the skin
// nearest the lever, stopped behind the cut. The cut is square to the line on a
// straight limb and leans to the bisector on a bent one (read off the skin round
// the anchor), so a bent elbow doesn't slice its upper arm lengthwise. Behind the cut
// the weight also fades past the limb's girth, so the chest beside a shoulder stays;
// past it the girth opens out, so a hand flaring off a wrist comes along whole. The
// weight ramps from 0 to 1 across the cut, over a band hardness sets. With mirror,
// the hinge reflected in x = 0 binds the other side (unless the hinge sits on the
// centre line, where it has no twin). mesh.pos must be current (materialize first).
void pose_bind(const Mesh& mesh, const PoseHinge& hinge, float hardness, bool mirror,
               PoseBinding& out);

// True when mirror has a twin to move: a hinge on the centre line is its own.
bool pose_mirror_twin(const PoseHinge& hinge);

// The handle under the cursor (kPoseAnchor / kPoseLever), or -1. Handles sit inside
// the mesh and are drawn through it, so this ignores the surface; the nearer one on
// screen wins.
int pose_pick(const PoseHinge& hinge, const Camera& cam, float sx, float sy, int w, int h);
