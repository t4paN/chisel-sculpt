#pragma once
#include "mesh.h"
#include <cstdint>
#include <vector>

struct Camera;

// Clip curve (L): a curve dragged across the screen, extruded straight into it, makes
// a wall. Every vertex on the shaded side is pushed onto that wall, through the whole
// model (back side too), so the shape comes out cut along it. Nothing is deleted and
// no edges are added: the cut is only as crisp as the mesh is dense.
//
// Not a stroke you paint: the curve is drawn first (main.cpp clip block) and the push
// happens once, on release, as a one-dab grab through the pose kernel, so undo,
// subdiv levels, normals and the mirror sink all take it like any other stroke.

// The curve the cut uses, in screen pixels: the drawn points smoothed and thinned,
// with both ends run on along their last direction far past the window, so the wall
// splits the whole screen in two. nrm = each segment's unit normal to the right of
// its direction (screen y down): the side that gets cut unless flipped.
struct ClipPath {
    std::vector<float> xy;    // x, y pairs, at least 2 points when ok
    std::vector<float> nrm;   // x, y per segment
    std::vector<float> cum;   // arc length at the start of each point
    // Even samples along the curve for clip_target's blend: x, y, arc length and the
    // stretch of curve each one stands for. 3 px apart on the drawn part, 12 px on the
    // run-on ends, and those only as far as a screen point could still be near.
    std::vector<float> smp;   // 4 floats per sample
    bool ok = false;
};

// Builds the path from the raw drawn points (x, y pairs). straight = only the first
// and last point count (Alt held). Not ok for a click or a scribble too short to
// give a direction.
void clip_prepare(const std::vector<float>& raw, bool straight, int win_w, int win_h,
                  ClipPath& out);

// Signed distance in pixels from (x, y) to the path, positive on the cut side (the
// right, or the left with flip). (qx, qy) = the nearest point on the path.
float clip_side(const ClipPath& path, bool flip, float x, float y, float& qx, float& qy);

// Where (x, y) goes when it is on the cut side (returns false when it isn't). Not
// simply the nearest point: inside a bend two stretches of the curve are about as
// near, and neighbours picking different ones would tear the mesh across the gap
// (the stretched "walls" of a C-shaped cut). The landing place is a soft blend along
// the curve over every stretch about as near as the nearest, and the deeper a point
// sits on the cut side the wider that blend, so the bay spreads evenly round the bend
// instead of piling onto the two arms. Near the curve it is the nearest point.
bool clip_target(const ClipPath& path, bool flip, float x, float y, float& tx, float& ty);

// What the release moves: the vertices that land somewhere else and where they land
// (3 floats per id), mask already applied (a fully masked vertex never makes the
// list). Each one on the cut side goes sideways on screen onto the curve (clip_target),
// keeping its view depth. With mirror, the wall reflected in x = 0 cuts
// the other side as well. mesh.pos must be current (materialize first).
void clip_bind(const Mesh& mesh, const Camera& cam, int w, int h, const ClipPath& path,
               bool flip, bool mirror, std::vector<uint32_t>& ids, std::vector<float>& targets);
