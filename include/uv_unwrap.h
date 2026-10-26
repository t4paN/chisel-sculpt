#pragma once
#include "retopo.h"
#include <cstdint>
#include <vector>

// UV mode (key 9): the retopo result laid flat for texturing. The unwrap itself is
// xatlas (external/xatlas, MIT): it cuts the mesh into islands, flattens each one and
// packs them into the unit square. It runs on the low-poly retopo mesh only, never on
// the sculpt — a few thousand quads take a fraction of a second, a sculpt would take
// minutes. The layout it makes (UvLayout) is in retopo.h, beside the mesh it maps.

struct UvSettings {
    int   resolution = 1024;  // the texture it's packed for: padding is in its pixels
    int   padding    = 4;     // pixels between islands, so a baked texture doesn't bleed
    float max_cost   = 2.0f;  // xatlas chart cost: lower = more, flatter islands
};

// A hash of rm's faces (corners and counts), to tell when a layout has gone stale.
uint64_t uv_topology_key(const RetopoMesh& rm);
// A hash of the seams that are still edges: has the user painted since the unwrap?
uint64_t uv_seam_key(const RetopoMesh& rm);

// Unwrap rm into out. Every painted seam (rm.seams) is cut; xatlas may cut more where a
// piece won't lie flat otherwise. Returns nullptr on success, else why not (out is left cleared).
// ms = how long it took.
const char* uv_unwrap(const RetopoMesh& rm, const UvSettings& s, UvLayout& out, double& ms);
