#pragma once
#include "retopo.h"
#include <cstdint>
#include <vector>

// UV mode (key 9): the retopo result laid flat for texturing. The islands are exactly
// what the painted seams and the mesh's borders fence off; each is flattened by
// least-squares conformal maps, then packed into the unit square by shape (biggest
// first, quarter turns allowed, never a reflection). It runs on the low-poly retopo mesh only, never on
// the sculpt — a few thousand quads take a fraction of a second, a sculpt would take
// minutes. The layout it makes (UvLayout) is in retopo.h, beside the mesh it maps.

struct UvSettings {
    int   resolution = 1024;  // the texture it's packed for: padding is in its pixels
    int   padding    = 4;     // pixels between islands, so a baked texture doesn't bleed
    bool  pack       = true;  // false: uv_unwrap leaves the islands flat, in mesh units, for uv_pack
};

// A hash of rm's faces (corners and counts), to tell when a layout has gone stale.
uint64_t uv_topology_key(const RetopoMesh& rm);
// A hash of the seams that are still edges: has the user painted since the unwrap?
uint64_t uv_seam_key(const RetopoMesh& rm);

// Unwrap rm into out, cut along the painted seams (rm.seams) and nowhere else. A piece
// that can't lie flat (closed, say) is still laid out, folded; out.needs_cut says which. Returns nullptr on success, else why not (out is left cleared).
// ms = how long it took.
const char* uv_unwrap(const RetopoMesh& rm, const UvSettings& s, UvLayout& out, double& ms);

// Several layouts packed into one square together (an atlas: one texture for every mesh
// in it), or one layout re-packed as it is. Each island keeps its shape and is scaled to
// its size on its mesh, so texel density matches across them all. Rewrites each layout's
// uv and used. nullptr on success, else why not (the layouts are left as they were).
struct UvPackPart { const RetopoMesh* rm; UvLayout* L; };
const char* uv_pack(const std::vector<UvPackPart>& parts, const UvSettings& s, double& ms);
