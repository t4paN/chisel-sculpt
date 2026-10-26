#pragma once
#include "retopo.h"
#include <cstdint>
#include <vector>

// UV mode (key 9): the retopo result laid flat for texturing. The unwrap itself is
// xatlas (external/xatlas, MIT): it cuts the mesh into islands, flattens each one and
// packs them into the unit square. It runs on the low-poly retopo mesh only, never on
// the sculpt — a few thousand quads take milliseconds, a sculpt would take minutes.
//
// UVs belong to face corners, not vertices: a vertex on a seam sits in two islands at
// once and has a different UV in each. So the layout runs parallel to
// RetopoMesh::faces, the same way an OBJ's "f v/vt" does.

struct UvLayout {
    std::vector<float>    uv;          // 2 per face corner, parallel to RetopoMesh::faces; 0..1, v up (OBJ); faces run counter-clockwise, as in 3D
    std::vector<uint32_t> face_chart;  // per face, the island it landed in
    uint32_t charts = 0;
    // The topology it was made for (uv_topology_key). Moving vertices keeps a layout
    // usable (just stretched); adding, cutting or erasing faces doesn't.
    uint64_t key = 0;
    bool empty() const { return uv.empty(); }
    void clear() { uv.clear(); face_chart.clear(); charts = 0; key = 0; }
};

struct UvSettings {
    int   resolution = 1024;  // the texture it's packed for: padding is in its pixels
    int   padding    = 4;     // pixels between islands, so a baked texture doesn't bleed
    float max_cost   = 2.0f;  // xatlas chart cost: lower = more, flatter islands
};

// A hash of rm's faces (corners and counts), to tell when a layout has gone stale.
uint64_t uv_topology_key(const RetopoMesh& rm);

// Unwrap rm into out. Returns nullptr on success, else why not (out is left cleared).
// ms = how long it took.
const char* uv_unwrap(const RetopoMesh& rm, const UvSettings& s, UvLayout& out, double& ms);
