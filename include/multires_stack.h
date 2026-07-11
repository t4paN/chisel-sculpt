#pragma once

#include <vector>
#include "mesh.h"

constexpr int MULTIRES_MAX_LEVEL = 9;

// Per-vertex tangent frame derived from the subdivided-base surface (pre-displacement).
// Stored in local coordinates so that lower-level edits reorient upper-level detail.
struct Frame {
    Vec3 t;  // tangent
    Vec3 b;  // bitangent  (= n cross t)
    Vec3 n;  // normal
};

struct MultiresStack {
    // Base cage: icosphere at the level selected at lock time. Never modified after lock.
    Mesh base;

    // Subdivision level of `base` from a level-0 icosahedron.
    int base_level = 0;

    // Displacement layers above the base.
    // disp[k] holds per-vertex LOCAL-frame deltas for the mesh at (base_level + k + 1).
    std::vector<std::vector<Vec3>> disp;

    // Per-level tangent frame cache. frames[k] corresponds to disp[k].
    // frames[k][v] is the local frame at vertex v on the subdivided-base surface at
    // (base_level + k + 1), pre-displacement. Populated lazily by cascade_to_level.
    // frames[k] is empty when stale; cascade rebuilds on next visit.
    // Invariant: frames.size() == disp.size().
    std::vector<std::vector<Frame>> frames;

    // Topology-derived mirror maps. base_mirror[i] = mirror partner of vertex i in
    // stack.base (self-map on seam or fallback). mirror[k] is the partner map for
    // the mesh at absolute level (base_level + k + 1), parallel to disp[k]/frames[k].
    // Built once at lock and propagated through Loop subdivision in cascade_to_level.
    // Invalidated (cleared) by projection; rebuilt lazily on next cascade.
    std::vector<uint32_t> base_mirror;
    std::vector<std::vector<uint32_t>> mirror;

    // Finest-level paint planes. Canonical Loop numbering nests vertex ids
    // across levels (level-K verts are exactly [0, V_K) at every finer level),
    // so ONE array sized for L_max = base_level + disp.size() carries paint for
    // EVERY level: the working array at level K is the [0, V_K) prefix.
    // Empty = feature unused (no paint / no mask). Written by
    // multires_sync_paint (diff working prefix, re-interpolate descendants of
    // changed verts), read back by cascade_to_level. NOT touched by projection
    // snapshots — paint has its own undo category.
    std::vector<uint32_t> color;
    std::vector<float>    mask;

    // midpoint_parents[k]: for each midpoint vertex of the mesh at level
    // (base_level + k + 1), its two parent vertex ids at level (base_level + k)
    // — 2 ids per midpoint, midpoints indexed from V_k. Topology-only, built
    // lazily (captured during cascade, or replayed on demand); cleared on lock.
    std::vector<std::vector<uint32_t>> midpoint_parents;

    // Absolute subdivision level the user is currently editing at.
    int current_level = 0;

    // True once lock has happened and the stack is populated.
    bool locked = false;
};

void multires_stack_init_from_lock(MultiresStack& stack,
                                   const Mesh& locked_mesh,
                                   int icosphere_level);

void multires_stack_debug_print(const MultiresStack& stack);

// Compute per-vertex local frames from the mesh surface.
// Mesh must have adjacency (vert_tri_offset/list) and normals already computed.
// Tangent = direction to lowest-indexed neighbor, projected onto tangent plane.
void compute_frames(const Mesh& m, std::vector<Frame>& out);

// Fold the working mesh's paint (colour + mask) into the stack's finest-level
// planes. Call BEFORE any cascade that rebuilds the working surface. Diffs the
// working prefix against the stored plane and re-interpolates only the
// descendants of changed verts up to L_max — untouched fine detail survives
// exactly; repainted regions cover smoothly. The working level is derived from
// working.vertex_count() (robust when current_level was already moved). An
// empty working mask clears the mask plane (a cleared mask must not
// resurrect); an empty working colour leaves the colour plane alone.
void multires_sync_paint(MultiresStack& stack, const Mesh& working);

// Rebuild `out` to the stack's surface at absolute level K.
// Preconditions: stack.locked == true, base_level <= K <= MULTIRES_MAX_LEVEL.
// Runs (K - base_level) Loop passes from stack.base, lazily zero-fills disp[k] and
// frames[k], computes frames on the pre-displacement surface, then applies disp[k]
// in local-frame coordinates. out.color/out.mask are filled from the stack's
// finest-level paint planes ([0, V_K) prefix; extended first if disp grew) —
// cleared when the corresponding plane is empty.
// Rebuilds CSR adjacency on `out`. Mirror map NOT rebuilt — caller must do that.
void cascade_to_level(MultiresStack& stack, Mesh& out, int K);

// Re-encode a v<=3 stack (legacy platform-specific midpoint numbering, see
// mesh.h) into canonical numbering, validating the legacy replay against the
// current-level surface cached in the project file. Returns false when the
// file was saved on a different platform and cannot be decoded here — caller
// keeps the cached surface and flattens the stack.
bool migrate_legacy_numbering(MultiresStack& stack, const Mesh& cached_surface);

// Snapshot of the subset of multires state affected by a projection, used to
// make a single atomic undo entry.
struct MultiresSnapshot {
    bool has_base = false;
    std::vector<float> base_px, base_py, base_pz;

    int disp_start = 0;                             // first absolute disp index covered
    std::vector<std::vector<Vec3>>  disp;
    std::vector<std::vector<Frame>> frames;

    size_t bytes() const;
    bool   empty() const { return !has_base && disp.empty(); }
};

struct ProjectionStats {
    int    target_level    = 0;
    int    L_max           = 0;
    double elapsed_ms      = 0.0;
    bool   did_anything    = false;
    double max_reconstruction_error = 0.0; // filled iff CHISEL_DEBUG_MULTIRES
};

// Snapshot the multires storage that project_down_to_level(target_level) would
// overwrite: base (iff target_level == base_level) and disp/frames in
// [k_start .. disp.size()-1] where k_start = max(0, target_level - base_level - 1).
void capture_projection_snapshot(const MultiresStack& stack,
                                 int target_level,
                                 MultiresSnapshot& out);

// Restore a previously captured snapshot back into stack. Caller must cascade
// afterwards to refresh `mesh`.
void restore_projection_snapshot(MultiresStack& stack,
                                 const MultiresSnapshot& snap);

// Inverse-Loop projection from L_max down onto target_level. Destructively
// rewrites stack.base (iff target == base_level) or disp[target - base - 1]
// (iff target > base_level), plus disp[...] and frames[...] above, such that
// a cascade to L_max reproduces the pre-projection L_max surface exactly.
// L_max is derived from stack.disp.size(): L_max = base_level + disp.size().
// No-op if target_level >= L_max.
ProjectionStats project_down_to_level(MultiresStack& stack, int target_level);
