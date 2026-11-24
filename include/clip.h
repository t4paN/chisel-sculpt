#pragma once
#include "mesh.h"
#include <cstdint>
#include <vector>

struct Camera;

// Clip curve (L): a straight line dragged across the screen, extruded straight into
// it, makes a wall. Every vertex on the shaded side is pushed onto that wall, through
// the whole model (back side too), so the shape comes out cut flat. Nothing is
// deleted and no edges are added: the cut is only as crisp as the mesh is dense.
//
// Not a stroke you paint: the line is drawn first (main.cpp clip block) and the push
// happens once, on release, as a one-dab grab through the pose kernel, so undo,
// subdiv levels, normals and the mirror sink all take it like any other stroke.

// The line in screen pixels. The shaded side is to the right of a -> b (screen y
// down); flip takes the other side.
struct ClipLine {
    float ax = 0, ay = 0, bx = 0, by = 0;
    bool  flip = false;
};

// Too short to give a direction: a click, not a line.
bool clip_line_usable(const ClipLine& line);

// What the release moves: the vertices that land somewhere else and where they land
// (3 floats per id), mask already applied (a fully masked vertex never makes the
// list). With mirror, the line's wall reflected in x = 0 cuts the other side as well.
// mesh.pos must be current (materialize first).
void clip_bind(const Mesh& mesh, const Camera& cam, int w, int h, const ClipLine& line,
               bool mirror, std::vector<uint32_t>& ids, std::vector<float>& targets);
