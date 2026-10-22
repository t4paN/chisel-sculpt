#pragma once
#include "mesh.h"
#include <cstdint>
#include <vector>

struct Camera;

// Pose brush rig: spheres ("joints") sitting in the middle of the volume, linked in
// pairs by bones. Spheres that link up form one SET; a drag only ever bends the set
// it grabbed, so an arm and a tail can be rigged separately on the same mesh.
//
// Lives on the MeshEntity, in world space like the mesh itself. Not saved with the
// project yet, and creating or deleting spheres is not on the undo stack; moving
// one is (PoseMove below), because a drag moves the mesh and the two must agree.

// Bones per set. The binding (pose_bind) samples every bone of the set against the
// grab's candidate triangles at pen-down, so this keeps that pause short. Only the
// bones touching the grabbed sphere reach the GPU (kPoseGpuBones, compute.h).
constexpr int kPoseMaxBones = 32;

struct PoseJoint { Vec3 pos; float radius; };
struct PoseBone  { uint16_t a, b; };

// One sphere drag, keyed by the undo entry its stroke pushed: undoing that entry
// puts the sphere back where the mesh went back to.
struct PoseMove { uint64_t seq; uint16_t joint; Vec3 from, to; };

struct PoseRig {
    std::vector<PoseJoint> joints;
    std::vector<PoseBone>  bones;
    std::vector<PoseMove>  moves;

    bool empty() const { return joints.empty(); }
    void clear() { joints.clear(); bones.clear(); moves.clear(); }
    int  add_joint(Vec3 p, float r);
    // false when the set it would join is already at kPoseMaxBones.
    bool add_bone(int a, int b);
    // Cancels a just-made sphere: it is always the last one, and so are its bones.
    void pop_joint();
    // Right-click removal: the sphere, its bones, its undo moves. Later spheres shift
    // down one index. A neighbour left with no bones stays, to grow from again.
    void remove_joint(int j);
    // A sphere placed by hand (Shift+drag) no longer sits where any earlier stroke
    // put it, so undoing that stroke must not snap it back there.
    void forget_moves(int j);

    // flags[j] = 1 for every joint linked to `j` (j included).
    void connected(int j, std::vector<uint8_t>& flags) const;
    int  bones_in_set_of(int j) const;

    void on_undo(uint64_t seq);
    void on_redo(uint64_t seq);
};

// The cursor's view ray. Under ortho the origin moves with the pixel and the
// direction is the view axis; under perspective it is the eye and the pixel's
// direction. Either way the parameter along it is view depth (Camera::view_depth).
void pose_screen_ray(const Camera& cam, float sx, float sy, int w, int h,
                     Vec3& origin, Vec3& dir);

// Where a sphere placed under the cursor sits: half-way between the first surface
// the ray enters and the next one it leaves through, so it lands inside the limb,
// not on its skin. radius = half that thickness. False when the ray misses the mesh;
// an open mesh with no far wall gives the entry point and radius 0.
bool pose_volume_center(const Mesh& mesh, const Camera& cam, float sx, float sy,
                        int w, int h, Vec3& center, float& radius);

// Screen pixels per world unit at the depth of `p` (the foreshortening term).
float pose_px_per_world(const Camera& cam, Vec3 p, int h);

// What one grab carries (pose_bind): the vertices, their pen-down positions and,
// per vertex, one weight per moving bone (lane k = moving[k] below), then the same
// four for the mirrored rig (zero with mirror off).
struct PoseBinding {
    std::vector<uint32_t> ids;
    std::vector<float>    init;   // 3 per id
    std::vector<float>    w;      // 8 per id
};

// Binds the set around `joint` to the mesh at pen-down. Each bone of the set gets a
// skin envelope: from its axis outward, the nearest surface facing away from it, in
// 12 slices along the bone x 16 directions around it — the limb's own cross-section,
// whatever its shape. A free end being dragged (a wrist) carries what hangs off it
// (the hand): the whole tip past it when that tip is clear of everything else,
// otherwise to ~3 sphere radii out. A vertex belongs to a bone when it lies
// inside that envelope (hardness sets the fade past it) AND is connected through the
// surface to the bone's own skin, so a neighbouring finger or the other leg that
// pokes into the envelope stays put. At a shared sphere the two bones hand over
// across the sphere's radius. With mirror, the rig reflected in x = 0 binds the
// other side too. mesh.pos must be current (materialize first).
// moving[k] (k < n_moving <= 4): rig bone index for weight lane k.
void pose_bind(const Mesh& mesh, const PoseRig& rig, int joint,
               const int* moving, int n_moving, float hardness, bool mirror,
               PoseBinding& out);

// The sphere under the cursor, or -1. Spheres sit inside the mesh and are drawn
// through it, so this ignores the surface: of the spheres whose screen disc holds
// the cursor, the nearest to the eye wins. Tiny spheres still catch within a few px.
int pose_pick(const PoseRig& rig, const Camera& cam, float sx, float sy, int w, int h);
