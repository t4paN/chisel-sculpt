#include "remesh.h"
#include "compute.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Edge table — temporary structure for local remeshing operations
// ---------------------------------------------------------------------------

static constexpr uint32_t INVALID = UINT32_MAX;

struct EdgeEntry {
    uint32_t v0, v1;
    uint32_t tri_a, tri_b;  // tri_b = INVALID if boundary
    bool dead;
};

// Open-addressed uint64 -> uint32 map, linear probing, power-of-two capacity.
//
// std::unordered_map allocates a node per insert. The edge table gets rebuilt
// at the top of every collapse sub-pass (up to 20 per outer iteration, up to
// 10 outer iterations) and does ~1.5 inserts per triangle, so those node
// allocations were the bulk of remesh's CPU time. Buffers here are reused
// across rebuilds — `reset` only reallocates when the mesh outgrows them.
//
// Erase leaves a tombstone and insert never claims one: every caller verifies
// absence first (a probe that walks to EMPTY), so refusing to reuse tombstones
// costs a little probing and removes any chance of a duplicate key.
struct EdgeMap {
    static constexpr uint64_t EMPTY = ~0ull;
    static constexpr uint64_t TOMB  = ~0ull - 1;

    std::vector<uint64_t> keys;
    std::vector<uint32_t> vals;
    uint32_t cap_mask = 0;
    uint32_t occupied = 0;   // live + tombstoned slots

    static uint64_t mix(uint64_t k) {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    void reset(size_t expected) {
        size_t want = 64;
        while (want < expected * 2) want <<= 1;
        if (keys.size() != want) { keys.resize(want); vals.resize(want); }
        std::fill(keys.begin(), keys.end(), EMPTY);
        cap_mask = (uint32_t)(want - 1);
        occupied = 0;
    }

    void grow() {
        std::vector<uint64_t> old_keys = keys;
        std::vector<uint32_t> old_vals = vals;
        size_t want = keys.size() * 2;
        keys.assign(want, EMPTY);
        vals.resize(want);
        cap_mask = (uint32_t)(want - 1);
        occupied = 0;
        for (size_t i = 0; i < old_keys.size(); i++)
            if (old_keys[i] != EMPTY && old_keys[i] != TOMB)
                insert(old_keys[i], old_vals[i]);
    }

    void insert(uint64_t k, uint32_t v) {
        // Keep load under 0.7 so probe chains stay short.
        if ((uint64_t)(occupied + 1) * 10 > (uint64_t)keys.size() * 7) grow();
        uint32_t i = (uint32_t)mix(k) & cap_mask;
        while (true) {
            uint64_t cur = keys[i];
            if (cur == k) { vals[i] = v; return; }
            if (cur == EMPTY) { keys[i] = k; vals[i] = v; occupied++; return; }
            i = (i + 1) & cap_mask;
        }
    }

    uint32_t find(uint64_t k) const {
        uint32_t i = (uint32_t)mix(k) & cap_mask;
        while (true) {
            uint64_t cur = keys[i];
            if (cur == k) return vals[i];
            if (cur == EMPTY) return INVALID;
            i = (i + 1) & cap_mask;
        }
    }

    void erase(uint64_t k) {
        uint32_t i = (uint32_t)mix(k) & cap_mask;
        while (true) {
            uint64_t cur = keys[i];
            if (cur == k) { keys[i] = TOMB; return; }
            if (cur == EMPTY) return;
            i = (i + 1) & cap_mask;
        }
    }
};

struct EdgeTable {
    std::vector<EdgeEntry> edges;
    EdgeMap lookup; // (v0,v1) -> edge index

    // CSR vert→edge_id: built by `build()`, immutable until the next build.
    // vert_edge_offset has size vc+1; vert_edge_list has size sum(valence).
    std::vector<uint32_t> vert_edge_offset;
    std::vector<uint32_t> vert_edge_list;
    // Edges appended via `add_edge()` after build (only flip_edges does this).
    // `for_edges_at()` walks the CSR then this side-buffer, so callers see
    // the full set without us shifting CSR slots. Cleared on each build().
    std::vector<std::vector<uint32_t>> vert_edge_extra;

    uint64_t key(uint32_t a, uint32_t b) const {
        if (a > b) { uint32_t t = a; a = b; b = t; }
        return ((uint64_t)a << 32) | (uint64_t)b;
    }

    void build(const Mesh& m) {
        edges.clear();
        vert_edge_extra.clear();
        uint32_t vc = m.vertex_count();
        uint32_t tc = m.tri_count();
        // A closed manifold has ~1.5 edges per triangle; 2x that is ample and
        // keeps the map from growing mid-build.
        lookup.reset((size_t)tc * 2);
        edges.reserve(tc * 3 / 2);

        for (uint32_t t = 0; t < tc; t++) {
            uint32_t idx[3] = { m.indices[t*3+0], m.indices[t*3+1], m.indices[t*3+2] };
            for (int e = 0; e < 3; e++) {
                uint32_t a = idx[e], b = idx[(e+1)%3];
                uint64_t k = key(a, b);
                uint32_t found = lookup.find(k);
                if (found == INVALID) {
                    uint32_t ei = (uint32_t)edges.size();
                    uint32_t lo = std::min(a, b), hi = std::max(a, b);
                    edges.push_back({lo, hi, t, INVALID, false});
                    lookup.insert(k, ei);
                } else {
                    edges[found].tri_b = t;
                }
            }
        }

        // Build CSR vert→edge_id from the edges array. Two-pass: count, then place.
        vert_edge_offset.assign(vc + 1, 0);
        for (const auto& e : edges) {
            vert_edge_offset[e.v0 + 1]++;
            vert_edge_offset[e.v1 + 1]++;
        }
        for (uint32_t v = 1; v <= vc; v++) vert_edge_offset[v] += vert_edge_offset[v-1];
        vert_edge_list.resize(vert_edge_offset[vc]);
        std::vector<uint32_t> cursor(vc, 0);
        for (uint32_t ei = 0; ei < (uint32_t)edges.size(); ei++) {
            const auto& e = edges[ei];
            vert_edge_list[vert_edge_offset[e.v0] + cursor[e.v0]++] = ei;
            vert_edge_list[vert_edge_offset[e.v1] + cursor[e.v1]++] = ei;
        }
    }

    // Iterate edge IDs incident on vert v. `f(ei) -> bool`: return false to stop early.
    // Walks the static CSR first, then any edges added since build via vert_edge_extra.
    template <typename F>
    void for_edges_at(uint32_t v, F&& f) const {
        if (v + 1 < vert_edge_offset.size()) {
            uint32_t s = vert_edge_offset[v], e = vert_edge_offset[v+1];
            for (uint32_t i = s; i < e; i++)
                if (!f(vert_edge_list[i])) return;
        }
        if (v < vert_edge_extra.size()) {
            for (uint32_t ei : vert_edge_extra[v])
                if (!f(ei)) return;
        }
    }

    uint32_t find_edge(uint32_t a, uint32_t b) const {
        return lookup.find(key(a, b));
    }

    void add_edge(uint32_t v0_, uint32_t v1_, uint32_t ta, uint32_t tb) {
        uint32_t lo = std::min(v0_, v1_), hi = std::max(v0_, v1_);
        uint32_t ei = (uint32_t)edges.size();
        edges.push_back({lo, hi, ta, tb, false});
        lookup.insert(key(lo, hi), ei);
        // Append to the per-vert extras (CSR is fixed once built).
        if (lo >= vert_edge_extra.size()) vert_edge_extra.resize(lo + 1);
        vert_edge_extra[lo].push_back(ei);
        if (hi >= vert_edge_extra.size()) vert_edge_extra.resize(hi + 1);
        vert_edge_extra[hi].push_back(ei);
    }

    void remove_edge(uint32_t ei) {
        edges[ei].dead = true;
        lookup.erase(key(edges[ei].v0, edges[ei].v1));
    }
};

// ---------------------------------------------------------------------------
// Topology defect tracer (CHISEL_DEBUG_REMESH)
// ---------------------------------------------------------------------------
//
// The audit at the end of perform_remesh reports that the output carries a
// handful of non-manifold and flipped-winding edges, but not WHERE they were
// born. This walks the same directed-use rule after every sub-pass and prints
// only when a count moves, so a defect gets pinned to one pass of one
// iteration. Two hypotheses it exists to separate:
//
//   1. Batched collapse. consumed[] marks only the two endpoints of a
//      collapsed edge, so two vertex-disjoint ops can still share a
//      neighbouring triangle or a link-condition neighbour — an earlier op in
//      the batch can invalidate a later one's link_condition, and nothing
//      re-validates after apply. The `ring` figure below counts, per sub-pass,
//      how many accepted ops had a 1-ring vertex already consumed by an
//      earlier op. Defects correlating with a nonzero `ring` confirms this.
//
//   2. flip_edges' edge-table fixup. After a flip it repairs only the (va,vd)
//      and (vb,vc) entries and assumes (va,vc)/(vb,vd) stay put. Its own
//      MAX_PASSES loop then reads that patched table. Defects appearing in
//      flip pass >= 1 rather than pass 0 confirms this.
//
// O(E) per checkpoint with a hash insert per directed edge — roughly a
// collapse sub-pass's worth of work each time, hence the compile gate.
#ifdef CHISEL_DEBUG_REMESH

static float edge_length(const Mesh& m, uint32_t a, uint32_t b);  // defined below

struct TopoCount { uint32_t nonmanifold = 0, bad_winding = 0, open_edges = 0; };

static TopoCount topo_defects(const Mesh& m, std::vector<uint64_t>* offenders = nullptr) {
    EdgeMap seen;
    const uint32_t tc = m.tri_count();
    const uint32_t vc = m.vertex_count();
    seen.reset((size_t)tc * 2);
    std::vector<uint32_t> fwd, bwd;
    std::vector<uint64_t> keys;
    fwd.reserve((size_t)tc * 2); bwd.reserve((size_t)tc * 2);
    keys.reserve((size_t)tc * 2);

    for (uint32_t t = 0; t < tc; t++) {
        for (int e = 0; e < 3; e++) {
            uint32_t a = m.indices[t*3+e], b = m.indices[t*3+(e+1)%3];
            if (a >= vc || b >= vc || a == b) continue;
            uint32_t lo = std::min(a, b), hi = std::max(a, b);
            uint64_t k = ((uint64_t)lo << 32) | (uint64_t)hi;
            uint32_t idx = seen.find(k);
            if (idx == INVALID) {
                idx = (uint32_t)fwd.size();
                fwd.push_back(0); bwd.push_back(0); keys.push_back(k);
                seen.insert(k, idx);
            }
            if (a == lo) fwd[idx]++; else bwd[idx]++;
        }
    }

    TopoCount c;
    for (size_t i = 0; i < fwd.size(); i++) {
        uint32_t uses = fwd[i] + bwd[i];
        bool bad = false;
        if (uses > 2)       { c.nonmanifold++; bad = true; }
        else if (uses == 2) { if (fwd[i] != 1) { c.bad_winding++; bad = true; } }
        else                  c.open_edges++;
        if (bad && offenders) offenders->push_back(keys[i]);
    }
    return c;
}

// Print the triangles sharing each offending edge. Two identical index triples
// means the defect is a DUPLICATE TRIANGLE (both traverse every shared edge the
// same way); three distinct triples on one edge is a genuine non-manifold fin.
// Only runs on a rise, so the O(F) scan per offender is paid rarely.
static void topo_dump_offenders(const Mesh& m, const std::vector<uint64_t>& offenders) {
    static constexpr int MAX_DUMP = 3;
    const uint32_t tc = m.tri_count();
    int shown = 0;
    for (uint64_t k : offenders) {
        if (shown >= MAX_DUMP) break;
        uint32_t v0 = (uint32_t)(k >> 32), v1 = (uint32_t)(k & 0xFFFFFFFFu);
        std::printf("[topo-trace]   edge (%u,%u) len=%.6f at (%.4f,%.4f,%.4f)\n",
                    v0, v1, edge_length(m, v0, v1),
                    m.pos_x[v0], m.pos_y[v0], m.pos_z[v0]);
        for (uint32_t t = 0; t < tc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            bool h0 = (i0 == v0 || i1 == v0 || i2 == v0);
            bool h1 = (i0 == v1 || i1 == v1 || i2 == v1);
            if (!h0 || !h1) continue;
            std::printf("[topo-trace]     tri %u = (%u, %u, %u)%s\n", t, i0, i1, i2,
                        (i0 == i1 || i1 == i2 || i0 == i2) ? "  DEGENERATE" : "");
        }
        shown++;
    }
}

static TopoCount g_topo_prev;
static bool      g_topo_armed = false;

static void topo_trace_reset(const Mesh& m) {
    g_topo_prev  = topo_defects(m);
    g_topo_armed = true;
    std::printf("[topo-trace] baseline: nonmanifold=%u winding=%u open=%u\n",
                g_topo_prev.nonmanifold, g_topo_prev.bad_winding,
                g_topo_prev.open_edges);
}

// `a` / `b` are stage-specific counters (sub-pass index, ops applied, ...);
// `extra` is the collapse ring-overlap count, -1 where it doesn't apply.
static void topo_trace(const Mesh& m, const char* stage, int a, int b, int extra) {
    if (!g_topo_armed) return;
    std::vector<uint64_t> offenders;
    TopoCount c = topo_defects(m, &offenders);
    const bool rose = (c.nonmanifold > g_topo_prev.nonmanifold ||
                       c.bad_winding > g_topo_prev.bad_winding);
    if (c.nonmanifold != g_topo_prev.nonmanifold ||
        c.bad_winding != g_topo_prev.bad_winding ||
        c.open_edges  != g_topo_prev.open_edges) {
        std::printf("[topo-trace] %s %d (n=%d", stage, a, b);
        if (extra >= 0) std::printf(", ring=%d", extra);
        std::printf("): nonmanifold %u->%u (%+d), winding %u->%u (%+d), "
                    "open %u->%u (%+d)\n",
                    g_topo_prev.nonmanifold, c.nonmanifold,
                    (int)c.nonmanifold - (int)g_topo_prev.nonmanifold,
                    g_topo_prev.bad_winding, c.bad_winding,
                    (int)c.bad_winding - (int)g_topo_prev.bad_winding,
                    g_topo_prev.open_edges, c.open_edges,
                    (int)c.open_edges - (int)g_topo_prev.open_edges);
        if (rose) topo_dump_offenders(m, offenders);
    }
    g_topo_prev = c;
}

#define REMESH_TOPO_RESET(m)                topo_trace_reset(m)
#define REMESH_TOPO_TRACE(m, s, a, b, x)    topo_trace((m), (s), (a), (b), (x))
#else
#define REMESH_TOPO_RESET(m)                ((void)0)
#define REMESH_TOPO_TRACE(m, s, a, b, x)    ((void)0)
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float edge_length(const Mesh& m, uint32_t a, uint32_t b) {
    float dx = m.pos_x[a] - m.pos_x[b];
    float dy = m.pos_y[a] - m.pos_y[b];
    float dz = m.pos_z[a] - m.pos_z[b];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static float compute_mean_edge_length(const Mesh& m) {
    double sum = 0.0;
    uint32_t tc = m.tri_count();
    uint32_t count = 0;
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0];
        uint32_t i1 = m.indices[t*3+1];
        uint32_t i2 = m.indices[t*3+2];
        sum += edge_length(m, i0, i1);
        sum += edge_length(m, i1, i2);
        sum += edge_length(m, i2, i0);
        count += 3;
    }
    return (count > 0) ? (float)(sum / count) : 1.0f;
}

static uint32_t tri_other_vert(const Mesh& m, uint32_t tri, uint32_t va, uint32_t vb) {
    uint32_t i0 = m.indices[tri*3+0];
    uint32_t i1 = m.indices[tri*3+1];
    uint32_t i2 = m.indices[tri*3+2];
    if (i0 != va && i0 != vb) return i0;
    if (i1 != va && i1 != vb) return i1;
    return i2;
}

static bool tri_contains_vert(const Mesh& m, uint32_t tri, uint32_t v) {
    return m.indices[tri*3+0] == v || m.indices[tri*3+1] == v || m.indices[tri*3+2] == v;
}

static void tri_replace_vert(Mesh& m, uint32_t tri, uint32_t old_v, uint32_t new_v) {
    for (int k = 0; k < 3; k++) {
        if (m.indices[tri*3+k] == old_v) {
            m.indices[tri*3+k] = new_v;
            return;
        }
    }
}

static bool tri_is_degenerate(const Mesh& m, uint32_t tri) {
    uint32_t i0 = m.indices[tri*3+0];
    uint32_t i1 = m.indices[tri*3+1];
    uint32_t i2 = m.indices[tri*3+2];
    return i0 == i1 || i1 == i2 || i0 == i2;
}

static Vec3 tri_normal(const Mesh& m, uint32_t tri) {
    Vec3 v0 = m.get_pos(m.indices[tri*3+0]);
    Vec3 v1 = m.get_pos(m.indices[tri*3+1]);
    Vec3 v2 = m.get_pos(m.indices[tri*3+2]);
    return (v1 - v0).cross(v2 - v0);
}



// ---------------------------------------------------------------------------
// Doubled-triangle removal
// ---------------------------------------------------------------------------
//
// The topology tracer showed every persistent non-manifold site in the output
// is the same structure: two triangles over the SAME three vertices wound in
// opposite directions — a zero-volume two-sided fin. Its three edges each carry
// two extra uses, which reads as non-manifold on the edge the fin shares with
// the real surface and as inconsistent winding elsewhere. Deleting both halves
// drops those edges back to exactly two triangles, consistently wound, and
// removes nothing a viewer could see: the pair encloses no volume.
//
// A same-winding duplicate is the degenerate cousin (two identical triangles);
// there only one copy goes.
//
// Nothing else in the pipeline can see these — they are geometrically
// coincident, so the angle, area and normal-inversion gates all pass them.
static bool tri_same_winding(uint32_t a0, uint32_t a1, uint32_t a2,
                             uint32_t b0, uint32_t b1, uint32_t b2) {
    return (a0 == b0 && a1 == b1 && a2 == b2) ||
           (a0 == b1 && a1 == b2 && a2 == b0) ||
           (a0 == b2 && a1 == b0 && a2 == b1);
}

static uint32_t remove_doubled_tris(Mesh& m) {
    const uint32_t tc = m.tri_count();
    if (tc == 0) return 0;

    EdgeMap seen;
    seen.reset((size_t)tc * 2);
    std::vector<uint8_t> drop(tc, 0);
    uint32_t removed = 0;

    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0], i1 = m.indices[t*3+1], i2 = m.indices[t*3+2];
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;   // compact_mesh's job

        uint32_t s0 = i0, s1 = i1, s2 = i2;
        if (s0 > s1) std::swap(s0, s1);
        if (s1 > s2) std::swap(s1, s2);
        if (s0 > s1) std::swap(s0, s1);
        uint64_t k = ((uint64_t)s0 * 0x9E3779B97F4A7C15ull) ^
                     ((uint64_t)s1 * 0xC2B2AE3D27D4EB4Full) ^
                     ((uint64_t)s2 * 0x165667B19E3779F9ull);
        if (k == EdgeMap::EMPTY || k == EdgeMap::TOMB) k ^= 1ull;

        uint32_t prev = seen.find(k);
        if (prev == INVALID) { seen.insert(k, t); continue; }
        if (drop[prev]) continue;

        // Verify the actual vertex sets — the key above is a hash, so a
        // collision must not be allowed to delete unrelated geometry.
        uint32_t p0 = m.indices[prev*3+0];
        uint32_t p1 = m.indices[prev*3+1];
        uint32_t p2 = m.indices[prev*3+2];
        uint32_t q0 = p0, q1 = p1, q2 = p2;
        if (q0 > q1) std::swap(q0, q1);
        if (q1 > q2) std::swap(q1, q2);
        if (q0 > q1) std::swap(q0, q1);
        if (q0 != s0 || q1 != s1 || q2 != s2) continue;

        if (tri_same_winding(i0, i1, i2, p0, p1, p2)) {
            drop[t] = 1;                    // exact duplicate — keep one
            removed += 1;
        } else {
            drop[t] = 1; drop[prev] = 1;    // opposing fin — both go
            removed += 2;
        }
    }

    if (removed == 0) return 0;

    std::vector<uint32_t> kept;
    kept.reserve(m.indices.size());
    for (uint32_t t = 0; t < tc; t++) {
        if (drop[t]) continue;
        kept.push_back(m.indices[t*3+0]);
        kept.push_back(m.indices[t*3+1]);
        kept.push_back(m.indices[t*3+2]);
    }
    m.indices = std::move(kept);
    return removed;
}

// ---------------------------------------------------------------------------
// Pass 1: Split long edges
// ---------------------------------------------------------------------------

static uint32_t split_long_edges(Mesh& m, EdgeTable& et,
                                 std::vector<uint32_t>& tri_selected,
                                 std::vector<uint32_t>& pinned,
                                 float high, float seam_tol,
                                 std::vector<float>* sel_mask,
                                 std::vector<float>* tlen) {
    uint32_t total_split = 0;
    // Sub-pass cap. Each sub-pass rebuilds adjacency at the top, so this directly
    // bounds wall-time. The touched_tris guard defers edges that share a tri with
    // an already-split edge in the same sub-pass; in practice these get caught in
    // 2-3 follow-up sub-passes, so 8 leaves comfortable headroom over the typical
    // ~3-4 sub-passes a stretched region needs. Raise if quality regresses on
    // dense long-edge clusters.
    static constexpr int MAX_ITERS = 8;
    // Minimum angle for triangles created by split. ~15°.
    // Compared against cosine (monotone-decreasing on [-1,1]): smallest angle ↔ largest cosine.
    static constexpr float MIN_TRI_ANGLE = 0.262f;     // rad — kept for clarity
    static const float    MAX_TRI_COS   = std::cos(MIN_TRI_ANGLE);

    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) { uint32_t t = a; a = b; b = t; }
        return ((uint64_t)a << 32) | (uint64_t)b;
    };

    // Returns the LARGEST cosine of the tri's three angles (= smallest angle).
    // For comparison: angle < threshold  iff  cos > cos(threshold).
    auto max_tri_cos = [](Vec3 p0, Vec3 p1, Vec3 p2) -> float {
        float l01 = (p1-p0).length(), l12 = (p2-p1).length(), l02 = (p2-p0).length();
        if (l01 < 1e-10f || l12 < 1e-10f || l02 < 1e-10f) return 1.0f; // degenerate → "0° angle"
        float c0 = std::max(-1.0f, std::min(1.0f, (p1-p0).dot(p2-p0) / (l01*l02)));
        float c1 = std::max(-1.0f, std::min(1.0f, (p0-p1).dot(p2-p1) / (l01*l12)));
        float c2 = std::max(-1.0f, std::min(1.0f, (p0-p2).dot(p1-p2) / (l02*l12)));
        return std::max({c0, c1, c2});
    };

    struct SplitEdge { uint32_t va, vb, tri_a, tri_b; };

    // Candidate buffers, reused across sub-passes so nothing reallocates per
    // pass. `cands` also replaces the old unordered_map's iteration order with
    // discovery order (ascending triangle index) — which edge wins the
    // touched_tris race changes, but the deferred ones are picked up by the
    // next sub-pass exactly as before, and this order is both deterministic
    // and cache-friendly.
    EdgeMap edge_seen;
    std::vector<SplitEdge> cands;
    std::vector<uint8_t> touched;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        // Rebuild adjacency so vert_tri_offset/vert_tri_list are current.
        m.build_adjacency();

        // Scan all selected tris for long edges.  Use adjacency (not the
        // edge table) to find both tris sharing each edge so the lookup is
        // never stale.
        uint32_t tc = m.tri_count();
        edge_seen.reset((size_t)tc * 2);
        cands.clear();

        for (uint32_t t = 0; t < tc; t++) {
            if (t >= (uint32_t)tri_selected.size() || !tri_selected[t]) continue;

            uint32_t idx[3] = { m.indices[t*3+0], m.indices[t*3+1], m.indices[t*3+2] };
            for (int e = 0; e < 3; e++) {
                uint32_t va = idx[e], vb = idx[(e+1)%3];

                uint64_t k = edge_key(va, vb);
                if (edge_seen.find(k) != INVALID) continue;
                // Adaptive: per-edge threshold from the graded target-length
                // field (mean of endpoint targets, spec §4.1). Uniform otherwise.
                float hi_local = high;
                if (tlen && va < (uint32_t)tlen->size() && vb < (uint32_t)tlen->size())
                    hi_local = 1.4f * 0.5f * ((*tlen)[va] + (*tlen)[vb]);
                if (edge_length(m, va, vb) <= hi_local) continue;  // was high*1.1 (1.54t) — left tris large

                // Find the (up to two) triangles sharing this edge.
                uint32_t tri_a = INVALID, tri_b = INVALID;
                uint32_t vstart = m.vert_tri_offset[va];
                uint32_t vend   = m.vert_tri_offset[va + 1];
                for (uint32_t j = vstart; j < vend; j++) {
                    uint32_t cand = m.vert_tri_list[j];
                    if (tri_contains_vert(m, cand, vb)) {
                        if (tri_a == INVALID) tri_a = cand;
                        else { tri_b = cand; break; }
                    }
                }

                bool sel_a = (tri_a != INVALID && tri_a < (uint32_t)tri_selected.size() && tri_selected[tri_a]);
                bool sel_b = (tri_b != INVALID && tri_b < (uint32_t)tri_selected.size() && tri_selected[tri_b]);
                if (!sel_a && !sel_b) continue;

                uint32_t lo = std::min(va, vb), hi_v = std::max(va, vb);
                edge_seen.insert(k, (uint32_t)cands.size());
                cands.push_back({lo, hi_v, tri_a, tri_b});
            }
        }

        if (cands.empty()) break;

        uint32_t num_split = 0;
        // Stamp array instead of a set. Splits append triangles, so it grows
        // on demand past the pre-split triangle count.
        touched.assign(tc, 0);
        auto mark_touched = [&](uint32_t t) {
            if (t >= touched.size()) touched.resize(t + 1, 0);
            touched[t] = 1;
        };
        auto is_touched = [&](uint32_t t) {
            return t < touched.size() && touched[t] != 0;
        };

        // Batch-split all found edges.  Two long edges that share a triangle
        // cannot both be split in the same sub-pass — the first split rewrites
        // the shared triangle in-place, making its index list inconsistent for
        // the second split and producing a one-sided split (boundary edge/hole).
        // Guard: skip any edge whose adjacent triangles were already touched by
        // an earlier split in this batch; the outer loop's next sub-pass will
        // catch the deferred edges with fresh adjacency.
        for (const SplitEdge& se : cands) {
            if (se.tri_a != INVALID && is_touched(se.tri_a)) continue;
            if (se.tri_b != INVALID && is_touched(se.tri_b)) continue;

            uint32_t va = se.va, vb = se.vb;

            float mx = (m.pos_x[va] + m.pos_x[vb]) * 0.5f;
            float my = (m.pos_y[va] + m.pos_y[vb]) * 0.5f;
            float mz = (m.pos_z[va] + m.pos_z[vb]) * 0.5f;
            float nx = (m.norm_x[va] + m.norm_x[vb]) * 0.5f;
            float ny = (m.norm_y[va] + m.norm_y[vb]) * 0.5f;
            float nz = (m.norm_z[va] + m.norm_z[vb]) * 0.5f;
            float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nl > 1e-8f) { nx /= nl; ny /= nl; nz /= nl; }

            Vec3 vm_p = {mx, my, mz};

            // Pre-check: would the split create thin triangles? Skip if so.
            bool angle_ok = true;
            uint32_t check_tris[2] = { se.tri_a, se.tri_b };
            for (uint32_t tri : check_tris) {
                if (tri == INVALID) continue;
                int va_pos = -1, vb_pos = -1;
                for (int kk = 0; kk < 3; kk++) {
                    if (m.indices[tri*3+kk] == va) va_pos = kk;
                    if (m.indices[tri*3+kk] == vb) vb_pos = kk;
                }
                if (va_pos < 0 || vb_pos < 0) continue;
                Vec3 pa = m.get_pos(va);
                Vec3 pb = m.get_pos(vb);
                Vec3 pc = m.get_pos(m.indices[tri*3 + (3 - va_pos - vb_pos)]);
                // Check modified triangle (va, vm, vc) and new triangle (vm, vb, vc).
                // cos > MAX_TRI_COS ⇔ angle < MIN_TRI_ANGLE.
                float cos_mod = max_tri_cos(pa, vm_p, pc);
                float cos_new = max_tri_cos(vm_p, pb, pc);
                if (cos_mod > MAX_TRI_COS || cos_new > MAX_TRI_COS) {
                    angle_ok = false;
                    break;
                }
            }
            if (!angle_ok) continue;

            bool new_pinned = pinned[va] && pinned[vb];
            bool seam_edge  = std::fabs(m.pos_x[va]) < seam_tol &&
                              std::fabs(m.pos_x[vb]) < seam_tol;
            // Never subdivide a fully-masked edge — even on the seam. Protected
            // tris must stay byte-identical on both sides; a one-sided split
            // makes a T-junction against the preserved -x twin.
            if (!m.mask.empty() &&
                va < (uint32_t)m.mask.size() && m.mask[va] >= 1.0f &&
                vb < (uint32_t)m.mask.size() && m.mask[vb] >= 1.0f) continue;
            // Off-seam pins are mask pins: refining fully-protected geometry is
            // pointless, and snapping their midpoint to x=0 would teleport it.
            if (new_pinned && !seam_edge) continue;
            float new_x = new_pinned ? 0.0f : mx;  // snap midpoint of two seam verts

            uint32_t vm = m.vertex_count();
            m.pos_x.push_back(new_x); m.pos_y.push_back(my); m.pos_z.push_back(mz);
            m.norm_x.push_back(nx); m.norm_y.push_back(ny); m.norm_z.push_back(nz);
            pinned.push_back(new_pinned ? 1u : 0u);
            ++num_split;

            if (!m.mirror_x_map.empty())
                m.mirror_x_map.push_back(vm);
            if (!m.mask.empty()) {
                float ma = (va < (uint32_t)m.mask.size()) ? m.mask[va] : 0.0f;
                float mb = (vb < (uint32_t)m.mask.size()) ? m.mask[vb] : 0.0f;
                m.mask.push_back((ma + mb) * 0.5f);
            }
            if (sel_mask && !sel_mask->empty()) {
                // AND rule: a midpoint counts as masked-for-selection only when
                // both parents do, so the frozen selection border never creeps
                // into the falloff as interpolated mask values drift toward 0.5.
                float sa = (va < (uint32_t)sel_mask->size()) ? (*sel_mask)[va] : 0.0f;
                float sb = (vb < (uint32_t)sel_mask->size()) ? (*sel_mask)[vb] : 0.0f;
                sel_mask->push_back((sa >= 0.5f && sb >= 0.5f) ? 1.0f : 0.0f);
            }
            if (!m.color.empty()) {
                uint32_t ca = (va < (uint32_t)m.color.size()) ? m.color[va] : 0xFFFFFFFFu;
                uint32_t cb = (vb < (uint32_t)m.color.size()) ? m.color[vb] : 0xFFFFFFFFu;
                m.color.push_back(color_avg(ca, cb));
            }
            if (!m.density.empty()) {
                float da = (va < (uint32_t)m.density.size()) ? m.density[va] : 0.5f;
                float db = (vb < (uint32_t)m.density.size()) ? m.density[vb] : 0.5f;
                m.density.push_back((da + db) * 0.5f);
            }
            if (tlen && !tlen->empty()) {
                float ta = (va < (uint32_t)tlen->size()) ? (*tlen)[va] : high / 1.4f;
                float tb = (vb < (uint32_t)tlen->size()) ? (*tlen)[vb] : high / 1.4f;
                tlen->push_back((ta + tb) * 0.5f);
            }

            uint32_t tris_to_split[2] = { se.tri_a, se.tri_b };
            for (uint32_t tri : tris_to_split) {
                if (tri == INVALID) continue;

                int va_pos = -1, vb_pos = -1;
                for (int kk = 0; kk < 3; kk++) {
                    if (m.indices[tri*3+kk] == va) va_pos = kk;
                    if (m.indices[tri*3+kk] == vb) vb_pos = kk;
                }
                if (va_pos < 0 || vb_pos < 0) continue;

                int vc_pos = 3 - va_pos - vb_pos;
                uint32_t vc_ = m.indices[tri*3+vc_pos];

                m.indices[tri*3+vb_pos] = vm;
                mark_touched(tri);

                uint32_t new_tri = m.tri_count();
                if ((va_pos + 1) % 3 == vb_pos) {
                    // va->vb in CCW order
                    m.indices.push_back(vm);
                    m.indices.push_back(vb);
                    m.indices.push_back(vc_);
                } else {
                    // vb->va in CCW order
                    m.indices.push_back(vb);
                    m.indices.push_back(vm);
                    m.indices.push_back(vc_);
                }
                tri_selected.push_back(1u);
                mark_touched(new_tri);
            }
        }

        // No adjacency rebuild here: the next sub-pass's first statement
        // is m.build_adjacency() (line ~185), and after the loop exits the
        // caller's rebuild_all() rebuilds it before anything reads it.
        total_split += num_split;
        std::printf("[split] sub-pass %d: split %u edges, mesh now %u verts %u tris\n",
                    iter, num_split, m.vertex_count(), m.tri_count());
        REMESH_TOPO_TRACE(m, "split sub-pass", iter, (int)num_split, -1);
    }

    // Rebuild edge table once now that topology is final.
    et.build(m);

    uint32_t boundary_count = 0;
    for (const auto& e : et.edges)
        if (!e.dead && e.tri_b == INVALID) ++boundary_count;
    std::printf("[split] boundary edges after split: %u\n", boundary_count);
    return total_split;
}

// ---------------------------------------------------------------------------
// Pass 2: Collapse short edges
// ---------------------------------------------------------------------------

static void compact_mesh(Mesh& m, std::vector<float>* aux = nullptr,
                         std::vector<float>* aux2 = nullptr); // forward declaration — defined after flip/smooth

static bool collapse_would_invert(const Mesh& m, const EdgeTable& et,
                                  uint32_t v_keep, uint32_t v_remove, Vec3 new_pos) {
    bool inverts = false;
    et.for_edges_at(v_remove, [&](uint32_t ei) {
        const auto& e = et.edges[ei];
        if (e.dead) return true;
        uint32_t tris[2] = { e.tri_a, e.tri_b };
        for (uint32_t tri : tris) {
            if (tri == INVALID) continue;
            // Skip tris that will become degenerate (contain both v_keep and v_remove)
            if (tri_contains_vert(m, tri, v_keep)) continue;

            Vec3 n_before = tri_normal(m, tri);

            // Simulate: replace v_remove with v_keep at new_pos
            uint32_t i0 = m.indices[tri*3+0];
            uint32_t i1 = m.indices[tri*3+1];
            uint32_t i2 = m.indices[tri*3+2];
            Vec3 p0 = (i0 == v_remove) ? new_pos : m.get_pos(i0);
            Vec3 p1 = (i1 == v_remove) ? new_pos : m.get_pos(i1);
            Vec3 p2 = (i2 == v_remove) ? new_pos : m.get_pos(i2);
            Vec3 n_after = (p1 - p0).cross(p2 - p0);

            if (n_before.dot(n_after) < 0.0f) { inverts = true; return false; }
        }
        return true;
    });
    if (inverts) return true;

    // Skip the v_keep loop entirely when v_keep doesn't move (pinned collapse,
    // or seam-snap that's already at x=0). With new_pos == get_pos(v_keep),
    // every simulated normal equals its original — no flip is possible.
    Vec3 keep_pos = m.get_pos(v_keep);
    Vec3 delta = new_pos - keep_pos;
    if (delta.dot(delta) > 0.0f) {
        et.for_edges_at(v_keep, [&](uint32_t ei) {
            const auto& e = et.edges[ei];
            if (e.dead) return true;
            uint32_t tris[2] = { e.tri_a, e.tri_b };
            for (uint32_t tri : tris) {
                if (tri == INVALID) continue;
                // Tris that also contain v_remove disappear in the collapse.
                if (tri_contains_vert(m, tri, v_remove)) continue;

                Vec3 n_before = tri_normal(m, tri);
                uint32_t i0 = m.indices[tri*3+0];
                uint32_t i1 = m.indices[tri*3+1];
                uint32_t i2 = m.indices[tri*3+2];
                Vec3 p0 = (i0 == v_keep) ? new_pos : m.get_pos(i0);
                Vec3 p1 = (i1 == v_keep) ? new_pos : m.get_pos(i1);
                Vec3 p2 = (i2 == v_keep) ? new_pos : m.get_pos(i2);
                Vec3 n_after = (p1 - p0).cross(p2 - p0);

                if (n_before.dot(n_after) < 0.0f) { inverts = true; return false; }
            }
            return true;
        });
    }
    return inverts;
}

// Link condition: v0 and v1 must share exactly 2 neighbor vertices (the diamond tips).
// If they share more, collapse creates non-manifold topology.
// Neighbours are gathered into a fixed stack array rather than an
// unordered_set: this runs for every short edge that clears the earlier gates,
// and a heap allocation per candidate was pure overhead. The caller already
// rejected val_a + val_b - 4 > 12, so a 1-ring can't reach MAX_LINK_NBRS here;
// overflowing it anyway means the mesh is degenerate, and refusing the collapse
// is the conservative answer.
static bool link_condition(const EdgeTable& et, uint32_t v0, uint32_t v1) {
    static constexpr int MAX_LINK_NBRS = 32;
    uint32_t nbrs0[MAX_LINK_NBRS];
    int n0 = 0;
    bool overflow = false;
    et.for_edges_at(v0, [&](uint32_t ei) {
        const auto& e = et.edges[ei];
        if (!e.dead) {
            uint32_t other = (e.v0 == v0) ? e.v1 : e.v0;
            if (other != v1) {
                if (n0 >= MAX_LINK_NBRS) { overflow = true; return false; }
                nbrs0[n0++] = other;
            }
        }
        return true;
    });
    if (overflow) return false;

    int shared = 0;
    et.for_edges_at(v1, [&](uint32_t ei) {
        const auto& e = et.edges[ei];
        if (!e.dead) {
            uint32_t other = (e.v0 == v1) ? e.v1 : e.v0;
            if (other != v0) {
                for (int i = 0; i < n0; i++)
                    if (nbrs0[i] == other) { shared++; break; }
            }
        }
        return true;
    });
    return shared == 2;
}

static uint32_t collapse_short_edges(Mesh& m, EdgeTable& et,
                                     std::vector<uint32_t>& tri_selected,
                                     std::vector<uint32_t>& pinned,
                                     float low, float seam_tol,
                                     std::vector<float>* sel_mask,
                                     std::vector<float>* tlen) {
    uint32_t total_collapse = 0;
    static constexpr int MAX_ITERS = 20;
    // Minimum angle for triangles after collapse. ~15°.
    static constexpr float MIN_TRI_ANGLE = 0.262f;
    static const float    MAX_TRI_COS   = std::cos(MIN_TRI_ANGLE);

    // Returns the LARGEST cosine of the tri's three angles (= smallest angle).
    auto max_tri_cos = [](Vec3 p0, Vec3 p1, Vec3 p2) -> float {
        float l01 = (p1-p0).length(), l12 = (p2-p1).length(), l02 = (p2-p0).length();
        if (l01 < 1e-10f || l12 < 1e-10f || l02 < 1e-10f) return 1.0f;
        float c0 = std::max(-1.0f, std::min(1.0f, (p1-p0).dot(p2-p0) / (l01*l02)));
        float c1 = std::max(-1.0f, std::min(1.0f, (p0-p1).dot(p2-p1) / (l01*l12)));
        float c2 = std::max(-1.0f, std::min(1.0f, (p0-p2).dot(p1-p2) / (l02*l12)));
        return std::max({c0, c1, c2});
    };

    struct CollapseOp { uint32_t v_keep, v_remove; Vec3 new_pos; };

    // Hoisted out of the sub-pass loop so the scratch buffers are allocated
    // once for the whole collapse, not once per sub-pass (up to 20 of them).
    std::vector<uint8_t> consumed;
    std::vector<int32_t> valence;
    std::vector<CollapseOp> ops;

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        m.build_adjacency();
        et.build(m);

        uint32_t vc_now = m.vertex_count();
        consumed.assign(vc_now, 0);
        ops.clear();
        // Accepted ops whose 1-ring already held a vertex claimed by an earlier
        // op in this batch — precisely the case consumed[] does NOT guard.
        [[maybe_unused]] uint32_t ring_overlaps = 0;

        // Valence for every vertex in one O(E) sweep. vertex_valence() walks
        // the CSR per query and got called twice per candidate edge. Every
        // collapse decision in this sub-pass is made before any op is applied,
        // so a snapshot taken here returns exactly what those calls did.
        valence.assign(vc_now, 0);
        for (const auto& e : et.edges) {
            if (e.dead) continue;
            if (e.v0 < vc_now) valence[e.v0]++;
            if (e.v1 < vc_now) valence[e.v1]++;
        }

        uint32_t num_edges = (uint32_t)et.edges.size();
        for (uint32_t ei = 0; ei < num_edges; ei++) {
            const auto& e = et.edges[ei];
            if (e.dead) continue;

            bool sel_a = (e.tri_a != INVALID && e.tri_a < (uint32_t)tri_selected.size() && tri_selected[e.tri_a]);
            bool sel_b = (e.tri_b != INVALID && e.tri_b < (uint32_t)tri_selected.size() && tri_selected[e.tri_b]);
            if (!sel_a && !sel_b) continue;

            // Adaptive: per-edge threshold from the graded target-length field
            // (mean of endpoint targets, spec §4.1). Uniform otherwise.
            float lo_local = low;
            if (tlen && e.v0 < (uint32_t)tlen->size() && e.v1 < (uint32_t)tlen->size())
                lo_local = 0.8f * 0.5f * ((*tlen)[e.v0] + (*tlen)[e.v1]);
            float len = edge_length(m, e.v0, e.v1);
            if (len >= lo_local) continue;

            uint32_t va = e.v0, vb = e.v1;

            if (va >= vc_now || vb >= vc_now) continue;
            if (consumed[va] || consumed[vb]) continue;

            // Pinned-pinned edges may only collapse when BOTH pins sit on the
            // mirror seam (that decimates the seam line to target spacing and
            // snaps to x=0 below). Any off-seam pin in the pair is a mask /
            // border pin: merging two of those would move protected geometry.
            //
            // A single off-seam pin with a free endpoint is ALLOWED: the free
            // vert collapses INTO the pin (v_keep = pin, new_pos = pin's exact
            // position below), which removes crunched border tris without the
            // protected region moving. Free-free edges on border-stitching tris
            // are also allowed — the collapse apply rewrites every adjacent tri
            // through full CSR adjacency, and the angle gate covers survivors,
            // so the stitch stays intact.
            {
                bool seam_a = std::fabs(m.pos_x[va]) < seam_tol;
                bool seam_b = std::fabs(m.pos_x[vb]) < seam_tol;
                if (pinned[va] && pinned[vb] && !(seam_a && seam_b)) continue;
                // Seam verts inside a masked patch are still protected: a
                // seam-decimation collapse there would rewrite original
                // protected tris on one side only.
                if (pinned[va] && pinned[vb] && !m.mask.empty() &&
                    ((va < (uint32_t)m.mask.size() && m.mask[va] >= 1.0f) ||
                     (vb < (uint32_t)m.mask.size() && m.mask[vb] >= 1.0f))) continue;
            }

            // Pinned-vs-near-seam-free leaves an orphaned mirror partner on the
            // far side; block. Pinned-vs-pinned is allowed below and snaps to x=0.
            if (pinned[va] && !pinned[vb] && std::fabs(m.pos_x[vb]) < seam_tol) continue;
            if (pinned[vb] && !pinned[va] && std::fabs(m.pos_x[va]) < seam_tol) continue;
            if (e.tri_a == INVALID || e.tri_b == INVALID) continue;

            int val_a = valence[va];
            int val_b = valence[vb];
            if (val_a + val_b - 4 > 12) continue;
            if (val_a <= 3 || val_b <= 3) continue;

            if (!link_condition(et, va, vb)) continue;

            uint32_t v_keep, v_remove;
            if (pinned[va]) { v_keep = va; v_remove = vb; }
            else if (pinned[vb]) { v_keep = vb; v_remove = va; }
            else { v_keep = va; v_remove = vb; }

            Vec3 new_pos;
            if (pinned[v_keep]) {
                new_pos = m.get_pos(v_keep);
                // Snap exact only for seam pins; mask pins keep their position.
                if (std::fabs(new_pos.x) < seam_tol) new_pos.x = 0.0f;
            } else {
                new_pos = (m.get_pos(va) + m.get_pos(vb)) * 0.5f;
            }

            // A non-pinned vert must never drift into the seam band — that
            // creates an unpaired near-seam vert that mirror_positive_half
            // would later cut into a duplicate seam tooth.
            if (!pinned[v_keep] && std::fabs(new_pos.x) < seam_tol) continue;

            if (collapse_would_invert(m, et, v_keep, v_remove, new_pos)) continue;

            // Block collapse that would create seam-straddling triangles: if
            // new_pos is on one side of x=0 and any neighbor of v_remove (that
            // isn't v_keep or being deleted) is on the other side, the surviving
            // triangle would cross the mirror plane.
            {
                bool straddle = false;
                float nx_sign = (new_pos.x > 0.0f) ? 1.0f : (new_pos.x < 0.0f ? -1.0f : 0.0f);
                if (nx_sign != 0.0f && m.vert_tri_offset.size() > v_remove + 1) {
                    uint32_t vs = m.vert_tri_offset[v_remove];
                    uint32_t ve = m.vert_tri_offset[v_remove + 1];
                    for (uint32_t j = vs; j < ve && !straddle; j++) {
                        uint32_t tri = m.vert_tri_list[j];
                        if (tri_contains_vert(m, tri, v_keep) &&
                            tri_contains_vert(m, tri, v_remove)) continue;
                        for (int k = 0; k < 3; k++) {
                            uint32_t nv = m.indices[tri*3+k];
                            if (nv == v_remove || nv == v_keep) continue;
                            float nvx = m.pos_x[nv];
                            if ((nx_sign > 0.0f && nvx < -seam_tol) ||
                                (nx_sign < 0.0f && nvx > seam_tol)) {
                                straddle = true;
                                break;
                            }
                        }
                    }
                }
                if (straddle) continue;
            }

            // Never mint a NEW fully-masked tri. The mirror step carries
            // fully-masked tris as "already present on both sides" — true only
            // for ORIGINAL protected tris (symmetric by construction), not for
            // tris a collapse rewires into [masked,masked,masked] on one side.
            // Those get skipped from reflection with no -x twin to stand in,
            // leaving open-edge cracks along the preserved patch border.
            if (!m.mask.empty() && v_keep < (uint32_t)m.mask.size() &&
                m.mask[v_keep] >= 1.0f) {
                bool mints_protected = false;
                uint32_t rs = m.vert_tri_offset[v_remove];
                uint32_t re = m.vert_tri_offset[v_remove + 1];
                for (uint32_t j = rs; j < re && !mints_protected; j++) {
                    uint32_t tri = m.vert_tri_list[j];
                    if (tri_contains_vert(m, tri, v_keep)) continue; // dies with the collapse
                    bool all_masked = true;
                    for (int k = 0; k < 3; k++) {
                        uint32_t tv = m.indices[tri*3+k];
                        if (tv == v_remove) continue; // rewired to v_keep (masked)
                        if (tv >= (uint32_t)m.mask.size() || m.mask[tv] < 1.0f) {
                            all_masked = false;
                            break;
                        }
                    }
                    if (all_masked) mints_protected = true;
                }
                if (mints_protected) continue;
            }

            // Check angle quality: all triangles around v_remove should not become too thin
            bool angle_ok = true;
            if (m.vert_tri_offset.size() > v_remove + 1) {
                uint32_t vstart = m.vert_tri_offset[v_remove];
                uint32_t vend   = m.vert_tri_offset[v_remove + 1];
                for (uint32_t j = vstart; j < vend; j++) {
                    uint32_t tri = m.vert_tri_list[j];
                    // Skip triangles that will be deleted
                    if (tri_contains_vert(m, tri, v_keep) && tri_contains_vert(m, tri, v_remove))
                        continue;
                    // Check remaining triangles after v_remove is replaced with v_keep
                    if (tri_contains_vert(m, tri, v_remove)) {
                        uint32_t i0 = m.indices[tri*3+0];
                        uint32_t i1 = m.indices[tri*3+1];
                        uint32_t i2 = m.indices[tri*3+2];
                        // Replace v_remove with v_keep in the angle check
                        i0 = (i0 == v_remove) ? v_keep : i0;
                        i1 = (i1 == v_remove) ? v_keep : i1;
                        i2 = (i2 == v_remove) ? v_keep : i2;
                        // Skip if would become degenerate
                        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                        Vec3 p0 = (i0 == v_keep) ? new_pos : m.get_pos(i0);
                        Vec3 p1 = (i1 == v_keep) ? new_pos : m.get_pos(i1);
                        Vec3 p2 = (i2 == v_keep) ? new_pos : m.get_pos(i2);
                        // cos > MAX_TRI_COS ⇔ angle < MIN_TRI_ANGLE.
                        if (max_tri_cos(p0, p1, p2) > MAX_TRI_COS) {
                            angle_ok = false;
                            break;
                        }
                    }
                }
            }
            if (!angle_ok) continue;

#ifdef CHISEL_DEBUG_REMESH
            {
                bool overlap = false;
                uint32_t ends[2] = { v_keep, v_remove };
                for (int q = 0; q < 2 && !overlap; q++) {
                    uint32_t rs = m.vert_tri_offset[ends[q]];
                    uint32_t re = m.vert_tri_offset[ends[q] + 1];
                    for (uint32_t j = rs; j < re && !overlap; j++) {
                        uint32_t tri = m.vert_tri_list[j];
                        for (int k = 0; k < 3; k++) {
                            uint32_t nv = m.indices[tri*3+k];
                            if (nv == va || nv == vb) continue;
                            if (nv < vc_now && consumed[nv]) { overlap = true; break; }
                        }
                    }
                }
                if (overlap) ring_overlaps++;
            }
#endif
            consumed[va] = 1;
            consumed[vb] = 1;
            ops.push_back({v_keep, v_remove, new_pos});
        }

        if (ops.empty()) break;
        total_collapse += (uint32_t)ops.size();

        // Apply all collected collapses.  Use vert_tri_offset/vert_tri_list
        // (built fresh at the top of this sub-pass) rather than the edge
        // table — consumed[] guarantees no two ops share a vertex, so
        // the pre-collapse adjacency for each v_remove is still valid when
        // we reach it here.
        for (const auto& op : ops) {
            m.set_pos(op.v_keep, op.new_pos);

            uint32_t vstart = m.vert_tri_offset[op.v_remove];
            uint32_t vend   = m.vert_tri_offset[op.v_remove + 1];
            for (uint32_t j = vstart; j < vend; j++) {
                uint32_t tri = m.vert_tri_list[j];
                if (tri_contains_vert(m, tri, op.v_keep) && tri_contains_vert(m, tri, op.v_remove)) {
                    m.indices[tri*3+0] = 0; m.indices[tri*3+1] = 0; m.indices[tri*3+2] = 0;
                } else if (tri_contains_vert(m, tri, op.v_remove)) {
                    tri_replace_vert(m, tri, op.v_remove, op.v_keep);
                }
            }
        }

        // Remap tri_selected and pinned through the same logic compact_mesh
        // uses, so the next sub-pass's index-based lookups stay correct.
        {
            uint32_t tc_old = m.tri_count();
            uint32_t vc_old = m.vertex_count();

            std::vector<bool> vert_used(vc_old, false);
            std::vector<uint32_t> new_ts;
            new_ts.reserve(tc_old);
            for (uint32_t t = 0; t < tc_old; t++) {
                if (tri_is_degenerate(m, t)) continue;
                new_ts.push_back(t < (uint32_t)tri_selected.size() ? tri_selected[t] : 0u);
                for (int k = 0; k < 3; k++) {
                    uint32_t v = m.indices[t*3+k];
                    if (v < vc_old) vert_used[v] = true;
                }
            }
            std::vector<uint32_t> new_pinned;
            new_pinned.reserve(vc_old);
            for (uint32_t v = 0; v < vc_old; v++) {
                if (!vert_used[v]) continue;
                new_pinned.push_back(v < (uint32_t)pinned.size() ? pinned[v] : 0u);
            }

            compact_mesh(m, sel_mask, tlen);
            tri_selected = std::move(new_ts);
            pinned       = std::move(new_pinned);
        }

        REMESH_TOPO_TRACE(m, "collapse sub-pass", iter, (int)ops.size(),
                          (int)ring_overlaps);
    }
    return total_collapse;
}

// ---------------------------------------------------------------------------
// Pass 3: Flip edges to equalize valence
// ---------------------------------------------------------------------------

static uint32_t flip_edges(Mesh& m, EdgeTable& et,
                           const std::vector<uint32_t>& tri_selected,
                           const std::vector<uint32_t>& pinned) {
    // Botsch-Kobbelt: loop until valence stops improving. Single-pass unordered
    // iteration leaves wins on the table — an edge that becomes flippable only
    // after its neighbor flips never gets caught in the same sweep.
    static constexpr int MAX_PASSES = 5;
    uint32_t vcount = m.vertex_count();
    std::vector<uint8_t> touched_tris(m.tri_count(), 0);

    // Valence snapshot, maintained incrementally. A flip drops edge (va,vb)
    // and adds (vc,vd), so it is exactly -1/-1/+1/+1 on those four verts —
    // the same deltas dev_after already models. Beats calling vertex_valence()
    // four times per candidate edge, each of which walks the CSR.
    std::vector<int32_t> valence(vcount, 0);
    for (const auto& e : et.edges) {
        if (e.dead) continue;
        if (e.v0 < vcount) valence[e.v0]++;
        if (e.v1 < vcount) valence[e.v1]++;
    }

    uint32_t total_flip = 0;
    int pass = 0;
    bool changed = true;
    while (changed && pass < MAX_PASSES) {
        changed = false;
        std::fill(touched_tris.begin(), touched_tris.end(), 0);
        uint32_t num_edges = (uint32_t)et.edges.size();
        for (uint32_t ei = 0; ei < num_edges; ei++) {
            const auto& e = et.edges[ei];
            if (e.dead) continue;
            if (e.tri_a == INVALID || e.tri_b == INVALID) continue;

            // Skip if either tri was already rewritten by an earlier flip this pass.
            if (e.tri_a >= touched_tris.size() || e.tri_b >= touched_tris.size()) continue;
            if (touched_tris[e.tri_a] || touched_tris[e.tri_b]) continue;

            bool sel_a = (e.tri_a < (uint32_t)tri_selected.size() && tri_selected[e.tri_a]);
            bool sel_b = (e.tri_b < (uint32_t)tri_selected.size() && tri_selected[e.tri_b]);
            if (!sel_a || !sel_b) continue;

            uint32_t va = e.v0, vb = e.v1;
            if (va >= (uint32_t)pinned.size() || vb >= (uint32_t)pinned.size()) continue;
            if (pinned[va] || pinned[vb]) continue;

            uint32_t vc = tri_other_vert(m, e.tri_a, va, vb);
            uint32_t vd = tri_other_vert(m, e.tri_b, va, vb);
            if (vc == INVALID || vd == INVALID) continue;
            if (vc == vd) continue; // would create degenerate

            // Block flips touching pinned opposite verts — rewiring connectivity
            // around a boundary vert can break the stitch to the protected region.
            if ((vc < (uint32_t)pinned.size() && pinned[vc]) ||
                (vd < (uint32_t)pinned.size() && pinned[vd])) continue;

            // Block flips that would create a seam-straddling edge: if vc and vd
            // sit on opposite sides of x=0, the new edge crosses the mirror plane.
            float xc = m.pos_x[vc], xd = m.pos_x[vd];
            if ((xc > 0.0f && xd < 0.0f) || (xc < 0.0f && xd > 0.0f)) continue;

            // Check: would the flipped edge already exist?
            if (et.find_edge(vc, vd) != INVALID) continue;

            // Valence improvement check
            if (vc >= vcount || vd >= vcount || va >= vcount || vb >= vcount) continue;
            int val_a = valence[va];
            int val_b = valence[vb];
            int val_c = valence[vc];
            int val_d = valence[vd];

            int dev_before = std::abs(val_a - 6) + std::abs(val_b - 6) +
                             std::abs(val_c - 6) + std::abs(val_d - 6);
            int dev_after  = std::abs(val_a - 1 - 6) + std::abs(val_b - 1 - 6) +
                             std::abs(val_c + 1 - 6) + std::abs(val_d + 1 - 6);

            if (dev_after >= dev_before) continue;

            // Check that flip doesn't invert normals
            Vec3 na_before = tri_normal(m, e.tri_a);
            Vec3 nb_before = tri_normal(m, e.tri_b);

            // After flip: tri_a = (vc, vd, va), tri_b = (vd, vc, vb)
            Vec3 pa = m.get_pos(va), pb = m.get_pos(vb);
            Vec3 pc = m.get_pos(vc), pd = m.get_pos(vd);
            Vec3 na_after = (pd - pc).cross(pa - pc);
            Vec3 nb_after = (pc - pd).cross(pb - pd);

            if (na_before.dot(na_after) <= 0.0f) continue;
            if (nb_before.dot(nb_after) <= 0.0f) continue;

            // Perform flip
            m.indices[e.tri_a*3+0] = vc;
            m.indices[e.tri_a*3+1] = vd;
            m.indices[e.tri_a*3+2] = va;

            m.indices[e.tri_b*3+0] = vd;
            m.indices[e.tri_b*3+1] = vc;
            m.indices[e.tri_b*3+2] = vb;

            // Update edge table: remove old, add new
            uint32_t ta = e.tri_a, tb = e.tri_b;
            et.remove_edge(ei);
            et.add_edge(vc, vd, ta, tb);

            // Fix adjacency of surrounding edges
            // Edge (va, vc) was in tri_a, still in tri_a — ok
            // Edge (vb, vc) was in tri_b... now in tri_b — ok wait, let's think:
            // Before: tri_a = (va, vb, vc), tri_b = (va, vb, vd) or similar
            // Actually the vertex arrangement depends on original winding.
            // The edges (va, vd) and (vb, vc) need their tri references updated.
            uint32_t e_va_vd = et.find_edge(va, vd);
            if (e_va_vd != INVALID) {
                auto& ee = et.edges[e_va_vd];
                // This edge was in tri_b, now should be in tri_a
                if (ee.tri_a == tb) ee.tri_a = ta;
                else if (ee.tri_b == tb) ee.tri_b = ta;
            }
            uint32_t e_vb_vc = et.find_edge(vb, vc);
            if (e_vb_vc != INVALID) {
                auto& ee = et.edges[e_vb_vc];
                // This edge was in tri_a, now should be in tri_b
                if (ee.tri_a == ta) ee.tri_a = tb;
                else if (ee.tri_b == ta) ee.tri_b = tb;
            }

            valence[va]--; valence[vb]--; valence[vc]++; valence[vd]++;

            touched_tris[ta] = 1;
            touched_tris[tb] = 1;
            changed = true;
            ++total_flip;
        }
        REMESH_TOPO_TRACE(m, "flip pass", pass, (int)total_flip, -1);
        pass++;
    }
    return total_flip;
}


// ---------------------------------------------------------------------------
// Compaction: remove degenerate tris and orphaned verts
// ---------------------------------------------------------------------------

static void compact_mesh(Mesh& m, std::vector<float>* aux, std::vector<float>* aux2) {
    uint32_t tc = m.tri_count();

    // Remove degenerate triangles
    std::vector<uint32_t> new_indices;
    new_indices.reserve(m.indices.size());
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0];
        uint32_t i1 = m.indices[t*3+1];
        uint32_t i2 = m.indices[t*3+2];
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        new_indices.push_back(i0);
        new_indices.push_back(i1);
        new_indices.push_back(i2);
    }
    m.indices = new_indices;

    // Find referenced vertices
    uint32_t vc = m.vertex_count();
    std::vector<bool> used(vc, false);
    for (uint32_t idx : m.indices) {
        if (idx < vc) used[idx] = true;
    }

    // Build remap
    std::vector<uint32_t> remap(vc, INVALID);
    uint32_t new_count = 0;
    for (uint32_t i = 0; i < vc; i++) {
        if (used[i]) remap[i] = new_count++;
    }

    if (new_count == vc) return; // nothing to compact

    // Compact position/normal arrays
    std::vector<float> npx(new_count), npy(new_count), npz(new_count);
    std::vector<float> nnx(new_count), nny(new_count), nnz(new_count);
    for (uint32_t i = 0; i < vc; i++) {
        if (remap[i] == INVALID) continue;
        uint32_t ni = remap[i];
        npx[ni] = m.pos_x[i]; npy[ni] = m.pos_y[i]; npz[ni] = m.pos_z[i];
        nnx[ni] = m.norm_x[i]; nny[ni] = m.norm_y[i]; nnz[ni] = m.norm_z[i];
    }
    m.pos_x = std::move(npx); m.pos_y = std::move(npy); m.pos_z = std::move(npz);
    m.norm_x = std::move(nnx); m.norm_y = std::move(nny); m.norm_z = std::move(nnz);

    // Remap indices
    for (uint32_t& idx : m.indices)
        idx = remap[idx];

    // Remap mirror_x_map through compaction. A vert whose partner was culled
    // falls back to SELF, not INVALID: consumers index this map raw, so a
    // UINT32_MAX left sitting in it is an out-of-bounds read later. Self is
    // also what the unpaired convention already means everywhere else.
    if (!m.mirror_x_map.empty()) {
        std::vector<uint32_t> new_mirror(new_count);
        for (uint32_t i = 0; i < new_count; i++) new_mirror[i] = i;
        uint32_t mm = (uint32_t)m.mirror_x_map.size();
        for (uint32_t i = 0; i < vc && i < mm; i++) {
            if (remap[i] == INVALID) continue;
            uint32_t mi = m.mirror_x_map[i];
            if (mi < vc && remap[mi] != INVALID) {
                new_mirror[remap[i]] = remap[mi];
            }
        }
        m.mirror_x_map = std::move(new_mirror);
    }

    if (!m.mask.empty()) {
        std::vector<float> new_mask(new_count, 0.0f);
        for (uint32_t i = 0; i < vc && i < (uint32_t)m.mask.size(); i++) {
            if (remap[i] == INVALID) continue;
            new_mask[remap[i]] = m.mask[i];
        }
        m.mask = std::move(new_mask);
    }

    if (!m.color.empty()) {
        std::vector<uint32_t> new_color(new_count, 0xFFFFFFFFu);
        for (uint32_t i = 0; i < vc && i < (uint32_t)m.color.size(); i++) {
            if (remap[i] == INVALID) continue;
            new_color[remap[i]] = m.color[i];
        }
        m.color = std::move(new_color);
    }

    if (!m.density.empty()) {
        std::vector<float> new_density(new_count, 0.5f);
        for (uint32_t i = 0; i < vc && i < (uint32_t)m.density.size(); i++) {
            if (remap[i] == INVALID) continue;
            new_density[remap[i]] = m.density[i];
        }
        m.density = std::move(new_density);
    }

    if (aux && !aux->empty()) {
        std::vector<float> new_aux(new_count, 0.0f);
        for (uint32_t i = 0; i < vc && i < (uint32_t)aux->size(); i++) {
            if (remap[i] == INVALID) continue;
            new_aux[remap[i]] = (*aux)[i];
        }
        *aux = std::move(new_aux);
    }

    if (aux2 && !aux2->empty()) {
        std::vector<float> new_aux(new_count, 0.0f);
        for (uint32_t i = 0; i < vc && i < (uint32_t)aux2->size(); i++) {
            if (remap[i] == INVALID) continue;
            new_aux[remap[i]] = (*aux2)[i];
        }
        *aux2 = std::move(new_aux);
    }
}

// ---------------------------------------------------------------------------
// Pre-mirror: consolidate the seam on the +x-only half
// ---------------------------------------------------------------------------
//
// Plane-clipping an arbitrary triangulation against x=0 mints sliver triangles
// wherever a pre-existing vertex sits just off the plane: the cut point on its
// neighbor edge lands almost on top of it, leaving a near-zero seam edge. Those
// are the "tiny triangles" hugging the mirror plane.
//
// Fix: after the clip drops the -x half (so the seam is exactly the boundary
// edges with both ends at x=0), snap any +x vertex that *hugs* the seam onto the
// plane, then weld coincident seam verts. Gated by both a band (|x| < snap_band)
// AND a short seam-incident edge (< max_edge) so a distant vert that merely
// touches one seam vert isn't flattened. Running on the +x half *before*
// reflection makes the seam a clean single edge-loop, so the mirror is symmetric
// by construction — no twin-chasing through mirror_x_map.
static uint32_t consolidate_seam(Mesh& m, float seam_tol, float target_edge) {
    float snap_band  = target_edge * 0.45f;   // how close to the plane counts as "should be on it"
    float max_edge   = target_edge * 0.6f;    // only snap across genuinely short seam edges (= `low`)
    float weld_tol   = target_edge * 0.0625f; // matches the GPU weld pass
    float weld_sq    = weld_tol * weld_tol;
    float short_seam = target_edge * 0.5f;    // seam edges under this get decimated (stage 3)
    bool has_mask = !m.mask.empty();
    auto masked = [&](uint32_t v) -> bool {
        return has_mask && v < (uint32_t)m.mask.size() && m.mask[v] >= 1.0f;
    };

    // Drop tris that went degenerate, or that collapsed flat onto the seam
    // (all three verts at x=0 — planar sails that survive as fins), then
    // renumber. Shared by the weld and decimate stages below.
    auto cull_and_compact = [&]() -> uint32_t {
        uint32_t tc = m.tri_count();
        std::vector<uint32_t> kept;
        kept.reserve(m.indices.size());
        uint32_t culled = 0;
        for (uint32_t t = 0; t < tc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            if (i0 == i1 || i1 == i2 || i0 == i2) { culled++; continue; }
            bool all_seam = std::fabs(m.pos_x[i0]) < seam_tol &&
                            std::fabs(m.pos_x[i1]) < seam_tol &&
                            std::fabs(m.pos_x[i2]) < seam_tol;
            // Preserve fully-masked tris — they're carried across as-is.
            if (all_seam && !(masked(i0) && masked(i1) && masked(i2))) { culled++; continue; }
            kept.push_back(i0); kept.push_back(i1); kept.push_back(i2);
        }
        m.indices = std::move(kept);
        compact_mesh(m);
        return culled;
    };

    // Recompute the seam set against current vertex numbering.
    std::vector<uint8_t> at_seam;
    auto mark_seam = [&]() -> uint32_t {
        uint32_t n = m.vertex_count();
        at_seam.assign(n, 0);
        uint32_t count = 0;
        for (uint32_t v = 0; v < n; v++)
            if (std::fabs(m.pos_x[v]) < seam_tol) { at_seam[v] = 1; count++; }
        return count;
    };

    m.build_adjacency();
    uint32_t vc = m.vertex_count();

    // Seam verts = exactly on x=0 (cut verts + pinned seam, post-clip).
    if (mark_seam() == 0) {
        std::printf("[seam-consolidate] no seam verts\n");
        return 0;
    }

    // --- 1. Snap near-seam verts onto the plane ---
    uint32_t snapped = 0;
    for (uint32_t v = 0; v < vc; v++) {
        float ax = std::fabs(m.pos_x[v]);
        if (ax < seam_tol || ax > snap_band) continue;
        if (masked(v)) continue;

        bool snap = false;
        uint32_t s = m.vert_tri_offset[v];
        uint32_t e = m.vert_tri_offset[v + 1];
        for (uint32_t j = s; j < e && !snap; j++) {
            uint32_t tri = m.vert_tri_list[j];
            for (int k = 0; k < 3; k++) {
                uint32_t nv = m.indices[tri*3+k];
                if (nv == v || nv >= vc || !at_seam[nv]) continue;
                if (edge_length(m, v, nv) < max_edge) { snap = true; break; }
            }
        }
        if (!snap) continue;

        m.pos_x[v] = 0.0f;
        at_seam[v] = 1;
        snapped++;
    }

    // --- 2. Weld coincident seam verts in (y,z) within weld_tol ---
    // Brute-force over the seam set only — small relative to vc.
    //
    // This runs unconditionally. It used to be gated behind `snapped > 0`,
    // which skipped it in exactly the case that needs it most: the plane clip
    // on its own can drop two cut verts on the same spot with nothing to snap
    // (two straddling edges meeting x=0 at the same point), and those
    // coincident pairs are the seed of the needle tris hugging the seam.
    std::vector<uint32_t> seam_verts;
    for (uint32_t v = 0; v < vc; v++)
        if (at_seam[v]) seam_verts.push_back(v);

    // Sorted by y so the inner scan stops as soon as the y gap alone exceeds
    // weld_tol. The seam is a 1-D loop, so this turns the all-pairs sweep into
    // roughly linear work — it was the one quadratic left in the pipeline, and
    // a dense seam is exactly where it bit hardest. Which vertex of a
    // coincident pair survives changes (y order, not index order); they are
    // within weld_tol of each other, so nothing downstream can tell.
    std::sort(seam_verts.begin(), seam_verts.end(),
              [&](uint32_t a, uint32_t b) { return m.pos_y[a] < m.pos_y[b]; });

    std::vector<uint32_t> remap(vc);
    for (uint32_t v = 0; v < vc; v++) remap[v] = v;

    uint32_t welded = 0;
    for (size_t i = 0; i < seam_verts.size(); i++) {
        uint32_t a = seam_verts[i];
        if (remap[a] != a || masked(a)) continue;
        for (size_t j = i + 1; j < seam_verts.size(); j++) {
            uint32_t b = seam_verts[j];
            float dy = m.pos_y[b] - m.pos_y[a];   // >= 0, sorted
            if (dy > weld_tol) break;
            if (remap[b] != b || masked(b)) continue;
            float dz = m.pos_z[a] - m.pos_z[b];
            if (dy*dy + dz*dz < weld_sq) { remap[b] = a; welded++; }
        }
    }
    if (welded > 0)
        for (uint32_t& idx : m.indices) idx = remap[idx];
    uint32_t culled = cull_and_compact();

    // --- 3. Decimate over-short seam edges ---
    // The seam-crossing clip mints cut verts wherever an edge meets x=0, at
    // whatever spacing the old triangulation happened to have — and nothing
    // downstream thins them: collapse_short_edges ran back when those verts
    // did not exist yet, and it treats seam verts as pinned anyway. Left alone
    // they are the second source of needle triangles at the mirror plane, the
    // one welding can't reach (these pairs are apart by more than weld_tol,
    // just far less than the target edge length).
    //
    // Merge b into a with a held fixed, one merge per vertex, guarded on
    // normal inversion. Doing it here — on the +x half, before reflection —
    // keeps the result symmetric by construction.
    m.build_adjacency();
    mark_seam();
    uint32_t nvc = m.vertex_count();

    auto merge_inverts = [&](uint32_t a, uint32_t b) -> bool {
        Vec3 pa = m.get_pos(a);
        uint32_t s = m.vert_tri_offset[b], e = m.vert_tri_offset[b + 1];
        for (uint32_t j = s; j < e; j++) {
            uint32_t tri = m.vert_tri_list[j];
            if (tri_contains_vert(m, tri, a)) continue;  // dies with the merge
            uint32_t i0 = m.indices[tri*3+0];
            uint32_t i1 = m.indices[tri*3+1];
            uint32_t i2 = m.indices[tri*3+2];
            Vec3 p0 = (i0 == b) ? pa : m.get_pos(i0);
            Vec3 p1 = (i1 == b) ? pa : m.get_pos(i1);
            Vec3 p2 = (i2 == b) ? pa : m.get_pos(i2);
            if (tri_normal(m, tri).dot((p1 - p0).cross(p2 - p0)) <= 0.0f) return true;
        }
        return false;
    };

    std::vector<uint32_t> dec(nvc);
    for (uint32_t v = 0; v < nvc; v++) dec[v] = v;
    // A vert that is either end of an accepted merge is off-limits for the
    // rest of the pass, so the pre-merge adjacency stays valid for every op
    // (same rule as consumed[] in collapse_short_edges).
    std::vector<uint8_t> touched(nvc, 0);
    uint32_t decimated = 0;

    for (uint32_t a = 0; a < nvc; a++) {
        if (!at_seam[a] || touched[a] || masked(a)) continue;
        uint32_t s = m.vert_tri_offset[a], e = m.vert_tri_offset[a + 1];
        bool merged = false;
        for (uint32_t j = s; j < e && !merged; j++) {
            uint32_t tri = m.vert_tri_list[j];
            for (int k = 0; k < 3; k++) {
                uint32_t b = m.indices[tri*3+k];
                if (b == a || b >= nvc) continue;
                if (!at_seam[b] || touched[b] || masked(b)) continue;
                if (edge_length(m, a, b) >= short_seam) continue;
                if (merge_inverts(a, b)) continue;
                dec[b] = a;
                touched[a] = touched[b] = 1;
                decimated++;
                merged = true;
                break;
            }
        }
    }

    uint32_t culled2 = 0;
    if (decimated > 0) {
        for (uint32_t& idx : m.indices) idx = dec[idx];
        culled2 = cull_and_compact();
    }

    std::printf("[seam-consolidate] snapped %u, welded %u, decimated %u short seam "
                "edges, culled %u sliver tris\n",
                snapped, welded, decimated, culled + culled2);
    return snapped + welded + decimated;
}

// ---------------------------------------------------------------------------
// Post-remesh: delete -x geometry, mirror +x to -x
// ---------------------------------------------------------------------------

static void mirror_positive_half(Mesh& m, float seam_tol, float target_edge, ComputeState* cs) {
    uint32_t vc = m.vertex_count();
    uint32_t tc = m.tri_count();
    if (vc == 0 || tc == 0) return;

    bool has_mask = !m.mask.empty();
    auto is_protected = [&](uint32_t v) -> bool {
        return has_mask && v < (uint32_t)m.mask.size() && m.mask[v] >= 1.0f;
    };
    auto tri_protected = [&](uint32_t t) -> bool {
        return has_mask &&
               is_protected(m.indices[t*3+0]) &&
               is_protected(m.indices[t*3+1]) &&
               is_protected(m.indices[t*3+2]);
    };

    // --- GPU seam snap + weld ---
    // Snap: verts near x=0 with tri-neighbors on both sides get pulled to x=0.
    // Weld: spatially-close verts at x=0 get merged (indices remapped).
    float snap_tol = seam_tol * 50.0f;
    float weld_tol = seam_tol * 6.25f;

    m.build_adjacency();
    cs->upload_adjacency(m.vert_tri_offset.data(),
                         (uint32_t)m.vert_tri_offset.size(),
                         m.vert_tri_list.data(),
                         (uint32_t)m.vert_tri_list.size());

    // Upload indices for the snap shader's neighbor lookups
    cs->ensure_remesh_smooth_buffers(vc, tc);
    gpu::write_buffer(cs->gpu_dev, cs->remesh_indices_ssbo, 0,
                      m.indices.data(), tc * 3 * sizeof(uint32_t));

    std::vector<uint32_t> merge_map;
    cs->dispatch_seam_snap_weld(
        vc, m.pos_x.data(), m.pos_y.data(), m.pos_z.data(),
        has_mask ? m.mask.data() : nullptr,
        has_mask ? (uint32_t)m.mask.size() : 0,
        seam_tol, snap_tol, weld_tol, merge_map);

    // Apply merge map to indices
    bool any_weld = false;
    for (uint32_t i = 0; i < (uint32_t)m.indices.size(); i++) {
        uint32_t v = m.indices[i];
        if (v < vc && merge_map[v] != v) {
            m.indices[i] = merge_map[v];
            any_weld = true;
        }
    }
    if (any_weld) compact_mesh(m);

    // Re-snapshot after potential compaction
    vc = m.vertex_count();
    tc = m.tri_count();

    // --- Split edges that cross x=0 so every triangle is cleanly on one side ---
    // Skip fully-masked tris: their geometry is preserved as-is.
    {
        auto side_of = [&](uint32_t v) -> int {
            if (std::fabs(m.pos_x[v]) < seam_tol) return 0;
            return (m.pos_x[v] > 0.0f) ? 1 : -1;
        };

        auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
            if (a > b) { uint32_t t = a; a = b; b = t; }
            return ((uint64_t)a << 32) | (uint64_t)b;
        };

        std::unordered_map<uint64_t, uint32_t> split_cache;

        auto get_or_create_split = [&](uint32_t a, uint32_t b) -> uint32_t {
            uint64_t k = edge_key(a, b);
            auto it = split_cache.find(k);
            if (it != split_cache.end()) return it->second;

            float xa = m.pos_x[a], xb = m.pos_x[b];
            float t = xa / (xa - xb);

            uint32_t nv = (uint32_t)m.pos_x.size();
            m.pos_x.push_back(0.0f);
            m.pos_y.push_back(m.pos_y[a] + t * (m.pos_y[b] - m.pos_y[a]));
            m.pos_z.push_back(m.pos_z[a] + t * (m.pos_z[b] - m.pos_z[a]));

            float nx = m.norm_x[a] + t * (m.norm_x[b] - m.norm_x[a]);
            float ny = m.norm_y[a] + t * (m.norm_y[b] - m.norm_y[a]);
            float nz = m.norm_z[a] + t * (m.norm_z[b] - m.norm_z[a]);
            float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nl > 1e-8f) { nx /= nl; ny /= nl; nz /= nl; }
            m.norm_x.push_back(nx);
            m.norm_y.push_back(ny);
            m.norm_z.push_back(nz);

            if (!m.mask.empty()) {
                float ma = (a < (uint32_t)m.mask.size()) ? m.mask[a] : 0.0f;
                float mb = (b < (uint32_t)m.mask.size()) ? m.mask[b] : 0.0f;
                m.mask.push_back(ma + t * (mb - ma));
            }
            if (!m.color.empty()) {
                uint32_t ca = (a < (uint32_t)m.color.size()) ? m.color[a] : 0xFFFFFFFFu;
                uint32_t cb = (b < (uint32_t)m.color.size()) ? m.color[b] : 0xFFFFFFFFu;
                m.color.push_back(color_lerp(ca, cb, t));
            }
            if (!m.density.empty()) {
                float da = (a < (uint32_t)m.density.size()) ? m.density[a] : 0.5f;
                float db = (b < (uint32_t)m.density.size()) ? m.density[b] : 0.5f;
                m.density.push_back(da + t * (db - da));
            }

            split_cache[k] = nv;
            return nv;
        };

        tc = m.tri_count();
        uint32_t num_split = 0;
        for (uint32_t t = 0; t < tc; t++) {
            if (tri_protected(t)) continue;

            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];

            int s0 = side_of(i0), s1 = side_of(i1), s2 = side_of(i2);
            bool has_pos = (s0 > 0 || s1 > 0 || s2 > 0);
            bool has_neg = (s0 < 0 || s1 < 0 || s2 < 0);
            if (!has_pos || !has_neg) continue;

            bool cross_01 = (s0 > 0 && s1 < 0) || (s0 < 0 && s1 > 0);
            bool cross_12 = (s1 > 0 && s2 < 0) || (s1 < 0 && s2 > 0);
            bool cross_20 = (s2 > 0 && s0 < 0) || (s2 < 0 && s0 > 0);
            int num_cross = (int)cross_01 + (int)cross_12 + (int)cross_20;

            if (num_cross == 2) {
                uint32_t vi, va, vb;
                if (cross_01 && cross_20)      { vi = i0; va = i1; vb = i2; }
                else if (cross_01 && cross_12) { vi = i1; va = i2; vb = i0; }
                else                           { vi = i2; va = i0; vb = i1; }

                uint32_t m1 = get_or_create_split(vi, va);
                uint32_t m2 = get_or_create_split(vi, vb);

                m.indices[t*3+0] = vi;
                m.indices[t*3+1] = m1;
                m.indices[t*3+2] = m2;

                m.indices.push_back(m1); m.indices.push_back(va); m.indices.push_back(vb);
                m.indices.push_back(m1); m.indices.push_back(vb); m.indices.push_back(m2);
                num_split++;
            } else if (num_cross == 1) {
                uint32_t ea, eb, ec;
                if (cross_01)      { ea = i0; eb = i1; ec = i2; }
                else if (cross_12) { ea = i1; eb = i2; ec = i0; }
                else               { ea = i2; eb = i0; ec = i1; }

                uint32_t mv = get_or_create_split(ea, eb);

                m.indices[t*3+0] = ea;
                m.indices[t*3+1] = mv;
                m.indices[t*3+2] = ec;

                m.indices.push_back(mv); m.indices.push_back(eb); m.indices.push_back(ec);
                num_split++;
            }
        }
        std::printf("[mirror] split %u seam-crossing triangles (%zu new seam verts)\n",
                    num_split, split_cache.size());

        // Stitch pass: protected tris are exempt from the clip above, but a
        // clipped NEIGHBOR may have minted a cut vert on a shared straddling
        // edge — a T-junction (zero-width slit along that edge). Subdivide the
        // protected tri at the same cached cut vert. On a masked-masked edge
        // the cut vert inherits mask 1.0, so the pieces stay protected; and a
        // straddling patch is its own mirror image, so symmetry is preserved.
        uint32_t stitched = 0;
        bool stitch_changed = true;
        while (stitch_changed) {
            stitch_changed = false;
            uint32_t cur_tc = m.tri_count();
            for (uint32_t t = 0; t < cur_tc; t++) {
                if (!tri_protected(t)) continue;
                for (int e = 0; e < 3; e++) {
                    uint32_t a = m.indices[t*3+e];
                    uint32_t b = m.indices[t*3+(e+1)%3];
                    auto it = split_cache.find(edge_key(a, b));
                    if (it == split_cache.end()) continue;
                    uint32_t mv = it->second;
                    uint32_t c = m.indices[t*3+(e+2)%3];
                    m.indices[t*3+0] = a;
                    m.indices[t*3+1] = mv;
                    m.indices[t*3+2] = c;
                    m.indices.push_back(mv);
                    m.indices.push_back(b);
                    m.indices.push_back(c);
                    stitched++;
                    stitch_changed = true;
                    break;
                }
            }
        }
        if (stitched > 0)
            std::printf("[mirror] stitched %u protected T-junction edges\n", stitched);
    }

    // Filter triangles that became degenerate during seam-split.
    {
        std::vector<uint32_t> filtered;
        filtered.reserve(m.indices.size());
        uint32_t dropped = 0;
        uint32_t cur_tc = m.tri_count();
        for (uint32_t t = 0; t < cur_tc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            if (i0 == i1 || i1 == i2 || i0 == i2) { dropped++; continue; }
            Vec3 e1 = m.get_pos(i1) - m.get_pos(i0);
            Vec3 e2 = m.get_pos(i2) - m.get_pos(i0);
            Vec3 cr = e1.cross(e2);
            float area2 = cr.dot(cr);
            if (area2 < 1e-20f) { dropped++; continue; }
            filtered.push_back(i0);
            filtered.push_back(i1);
            filtered.push_back(i2);
        }
        if (dropped > 0)
            std::printf("[mirror] dropped %u degenerate post-split tris\n", dropped);
        m.indices = std::move(filtered);
    }
    REMESH_TOPO_TRACE(m, "mirror seam-split", 0, 0, -1);

    // Re-snapshot counts after splitting
    vc = m.vertex_count();
    tc = m.tri_count();

    // Classify vertices: 0=seam, 1=+x, 2=-x
    std::vector<uint8_t> side(vc);
    int side_counts[3] = {0, 0, 0};
    for (uint32_t v = 0; v < vc; v++) {
        if (std::fabs(m.pos_x[v]) < seam_tol) {
            side[v] = 0;
            if (!is_protected(v)) m.pos_x[v] = 0.0f;
        } else if (m.pos_x[v] > 0.0f) side[v] = 1;
        else side[v] = 2;
        side_counts[side[v]]++;
    }
    std::printf("[mirror-debug] vertex distribution: seam=%d, +x=%d, -x=%d\n",
                side_counts[0], side_counts[1], side_counts[2]);

    // Keep +x and seam-only triangles; also keep fully-masked tris on -x.
    // Cull all-seam tris (all 3 verts at x=0) — these are degenerate planar
    // sails lying in the mirror plane, created when snap pulls straddle-verts
    // to x=0. They'd survive as fins/mohawks on the seam.
    std::vector<uint32_t> kept_indices;
    kept_indices.reserve(m.indices.size() / 2 + 100);
    uint32_t tri_dropped_before = 0, seam_culled = 0;
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0];
        uint32_t i1 = m.indices[t*3+1];
        uint32_t i2 = m.indices[t*3+2];
        bool all_seam = (side[i0]==0 && side[i1]==0 && side[i2]==0);
        if (all_seam && !tri_protected(t)) { seam_culled++; continue; }
        bool has_neg = (side[i0]==2 || side[i1]==2 || side[i2]==2);
        if (!has_neg || tri_protected(t)) {
            kept_indices.push_back(i0);
            kept_indices.push_back(i1);
            kept_indices.push_back(i2);
        } else {
            tri_dropped_before++;
        }
    }
    if (seam_culled > 0)
        std::printf("[mirror] culled %u all-seam sail tris\n", seam_culled);
    std::printf("[mirror-debug] kept %u tris, dropped %u tris with -x verts\n",
                (uint32_t)kept_indices.size() / 3, tri_dropped_before);
    m.indices = kept_indices;

    // Compact to remove orphaned vertices before mirroring
    compact_mesh(m);

    // Snap near-seam +x verts onto x=0 and weld, so the seam is a clean single
    // edge-loop before we reflect. Kills the clip slivers (see consolidate_seam).
    consolidate_seam(m, seam_tol, target_edge);
    REMESH_TOPO_TRACE(m, "mirror consolidate-seam", 0, 0, -1);

    // Re-snapshot after compaction
    vc = m.vertex_count();
    tc = m.tri_count();

    // Rebuild side classification on compacted vertices
    side.resize(vc);
    for (uint32_t v = 0; v < vc; v++) {
        if (std::fabs(m.pos_x[v]) < seam_tol) {
            side[v] = 0;
            if (!is_protected(v)) m.pos_x[v] = 0.0f;
        } else if (m.pos_x[v] > 0.0f) side[v] = 1;
        else side[v] = 2;
    }

    // Pair protected +x verts with their preserved -x counterparts.
    // Both are at original (pre-remesh) positions, so spatial match is tight.
    std::vector<uint32_t> masked_pair(vc, INVALID);
    if (has_mask) {
        std::vector<uint32_t> neg_protected;
        for (uint32_t v = 0; v < vc; v++)
            if (is_protected(v) && side[v] == 2) neg_protected.push_back(v);

        if (!neg_protected.empty()) {
            float mtol = seam_tol * 50.0f;
            float mtol_sq = mtol * mtol;
            // Closest-first, one-to-one. A border +x vert whose -x twin was
            // dropped (no fully-protected tri left to keep it alive) must NOT
            // grab a neighboring patch vert as partner — that folds reflected
            // border tris onto the wrong vert and leaves the preserved patch
            // edge uncovered (open-edge cracks). Sorting by distance lets true
            // twins (d ~ 0) claim each other first; anything left over falls
            // through to the mint-a-mirror-vert path, which is always safe.
            struct PairCand { float d2; uint32_t pos_v, neg_v; };
            std::vector<PairCand> cands;
            for (uint32_t v = 0; v < vc; v++) {
                if (side[v] != 1 || !is_protected(v)) continue;
                float best_d2 = mtol_sq;
                uint32_t best = INVALID;
                for (uint32_t nv : neg_protected) {
                    float dx = m.pos_x[v] + m.pos_x[nv];
                    float dy = m.pos_y[v] - m.pos_y[nv];
                    float dz = m.pos_z[v] - m.pos_z[nv];
                    float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < best_d2) { best_d2 = d2; best = nv; }
                }
                if (best != INVALID) cands.push_back({best_d2, v, best});
            }
            std::sort(cands.begin(), cands.end(),
                      [](const PairCand& a, const PairCand& b) { return a.d2 < b.d2; });
            uint32_t n_paired = 0, n_rejected = 0;
            for (const auto& c : cands) {
                if (masked_pair[c.pos_v] != INVALID ||
                    masked_pair[c.neg_v] != INVALID) { n_rejected++; continue; }
                masked_pair[c.pos_v] = c.neg_v;
                masked_pair[c.neg_v] = c.pos_v;
                n_paired++;
            }
            std::printf("[mirror] masked spatial pairs: %u (%u double-claims rejected)\n",
                        n_paired, n_rejected);
        }
    }

    // Create mirror vertex for each +x vertex.
    // Protected +x verts map to their preserved -x counterparts instead.
    std::vector<uint32_t> vert_mirror(vc, INVALID);
    for (uint32_t v = 0; v < vc; v++) {
        if (side[v] == 0) {
            vert_mirror[v] = v;
        } else if (side[v] == 1) {
            if (masked_pair[v] != INVALID) {
                vert_mirror[v] = masked_pair[v];
            } else {
                uint32_t mv = (uint32_t)m.pos_x.size();
                m.pos_x.push_back(-m.pos_x[v]);
                m.pos_y.push_back(m.pos_y[v]);
                m.pos_z.push_back(m.pos_z[v]);
                m.norm_x.push_back(-m.norm_x[v]);
                m.norm_y.push_back(m.norm_y[v]);
                m.norm_z.push_back(m.norm_z[v]);
                vert_mirror[v] = mv;
            }
        } else if (side[v] == 2 && masked_pair[v] != INVALID) {
            vert_mirror[v] = masked_pair[v];
        }
    }

    // Mirror mask values for newly created mirror verts only
    if (!m.mask.empty()) {
        m.mask.resize(m.vertex_count(), 0.0f);
        for (uint32_t v = 0; v < vc; v++) {
            if (side[v] == 1 && vert_mirror[v] != INVALID &&
                vert_mirror[v] >= vc && v < (uint32_t)m.mask.size())
                m.mask[vert_mirror[v]] = m.mask[v];
        }
    }

    // Mirror vertex paint onto the newly created mirror verts (white fill).
    if (!m.color.empty()) {
        m.color.resize(m.vertex_count(), 0xFFFFFFFFu);
        for (uint32_t v = 0; v < vc; v++) {
            if (side[v] == 1 && vert_mirror[v] != INVALID &&
                vert_mirror[v] >= vc && v < (uint32_t)m.color.size())
                m.color[vert_mirror[v]] = m.color[v];
        }
    }

    // Mirror the density field onto the newly created mirror verts (neutral
    // fill). Without this the -x half resets to 0.5 on every remesh.
    if (!m.density.empty()) {
        m.density.resize(m.vertex_count(), 0.5f);
        for (uint32_t v = 0; v < vc; v++) {
            if (side[v] == 1 && vert_mirror[v] != INVALID &&
                vert_mirror[v] >= vc && v < (uint32_t)m.density.size())
                m.density[vert_mirror[v]] = m.density[v];
        }
    }

    // Create mirrored triangles with flipped winding.
    // Skip fully-masked tris — they're already present on both sides.
    tc = m.tri_count();
    for (uint32_t t = 0; t < tc; t++) {
        if (tri_protected(t)) continue;

        uint32_t i0 = m.indices[t*3+0];
        uint32_t i1 = m.indices[t*3+1];
        uint32_t i2 = m.indices[t*3+2];

        if (side[i0]==0 && side[i1]==0 && side[i2]==0) continue;

        uint32_t mi0 = vert_mirror[i0];
        uint32_t mi1 = vert_mirror[i1];
        uint32_t mi2 = vert_mirror[i2];
        if (mi0 == INVALID || mi1 == INVALID || mi2 == INVALID) continue;
        // Two source verts mapping to one target would make a degenerate
        // reflected tri (and an open edge next to it) — never emit those.
        if (mi0 == mi1 || mi1 == mi2 || mi0 == mi2) continue;

        m.indices.push_back(mi0);
        m.indices.push_back(mi2);
        m.indices.push_back(mi1);
    }

    // Build mirror_x_map from duplication table + masked spatial pairs
    uint32_t total_vc = m.vertex_count();
    m.mirror_x_map.resize(total_vc);
    for (uint32_t v = 0; v < total_vc; v++)
        m.mirror_x_map[v] = v;
    for (uint32_t v = 0; v < vc; v++) {
        if (side[v] == 1 && vert_mirror[v] != INVALID) {
            m.mirror_x_map[v] = vert_mirror[v];
            m.mirror_x_map[vert_mirror[v]] = v;
        }
    }

    uint32_t paired = 0, seam = 0, unpaired = 0;
    uint32_t final_vc = m.vertex_count();
    for (uint32_t i = 0; i < final_vc; i++) {
        if (i < (uint32_t)m.mirror_x_map.size() && m.mirror_x_map[i] != i)
            paired++;
        else if (m.pos_x[i] == 0.0f)
            seam++;
        else
            unpaired++;
    }
    std::printf("[mirror] geometry mirror: %u paired, %u seam, %u unpaired, "
                "%u verts %u tris\n",
                paired, seam, unpaired, m.vertex_count(), m.tri_count());
    REMESH_TOPO_TRACE(m, "mirror reflect", 0, 0, -1);

    // --- Connected-component cleanup ---
    // Union-find over triangle connectivity. Discard all components except the
    // largest — catches disconnected sail fragments and other orphan geometry.
    // TODO(sdf): reuse this keep-largest-component pass for voxel-merge cleanup (spec chunk 4).
    {
        uint32_t fvc = m.vertex_count();
        uint32_t ftc = m.tri_count();
        std::vector<uint32_t> parent(fvc);
        for (uint32_t i = 0; i < fvc; i++) parent[i] = i;

        std::function<uint32_t(uint32_t)> find = [&](uint32_t x) -> uint32_t {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = find(a); b = find(b);
            if (a != b) parent[a] = b;
        };

        for (uint32_t t = 0; t < ftc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            unite(i0, i1);
            unite(i1, i2);
        }

        // Find largest component
        std::unordered_map<uint32_t, uint32_t> comp_size;
        for (uint32_t t = 0; t < ftc; t++) {
            uint32_t root = find(m.indices[t*3]);
            comp_size[root]++;
        }
        uint32_t best_root = 0, best_count = 0;
        for (auto& [root, cnt] : comp_size) {
            if (cnt > best_count) { best_count = cnt; best_root = root; }
        }

        if (comp_size.size() > 1) {
            std::vector<uint32_t> clean_indices;
            clean_indices.reserve(m.indices.size());
            uint32_t removed = 0;
            for (uint32_t t = 0; t < ftc; t++) {
                if (find(m.indices[t*3]) == best_root) {
                    clean_indices.push_back(m.indices[t*3+0]);
                    clean_indices.push_back(m.indices[t*3+1]);
                    clean_indices.push_back(m.indices[t*3+2]);
                } else {
                    removed++;
                }
            }
            m.indices = clean_indices;
            compact_mesh(m);
            std::printf("[mirror] component cleanup: removed %u tris from %u orphan components\n",
                        removed, (uint32_t)comp_size.size() - 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-pass: relax flipped triangles
// ---------------------------------------------------------------------------
//
// Detector: a tri whose face normal points opposite the average of its three
// vertex normals is inverted relative to its 1-ring. Vertex normals are
// area-weighted averages of incident face normals, so a single bad tri among
// good neighbors flips this dot product negative. A genuine surface fold
// (where the whole region agrees) does not trigger.
//
// Repair: move each "bad" vert (member of any flipped tri) toward its
// 1-ring vertex centroid by alpha. Seam verts (|x| < seam_tol) stay on x=0.
// Masked verts (mask >= 1.0) never move. Iterate; the outer boundary of a
// flipped patch converges first, then propagates inward.
static uint32_t repair_flipped_tris(Mesh& m, float seam_tol,
                                    int max_iters = 5, float alpha = 0.5f) {
    bool has_mask = !m.mask.empty();
    uint32_t total_relaxed = 0;
    uint32_t prev_flipped = UINT32_MAX;

    m.build_adjacency();

    for (int iter = 0; iter < max_iters; iter++) {
        m.recompute_normals();

        uint32_t tc = m.tri_count();
        uint32_t vc = m.vertex_count();

        std::vector<uint8_t> bad_vert(vc, 0);
        uint32_t flipped_count = 0;

        for (uint32_t t = 0; t < tc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            Vec3 tn = tri_normal(m, t);
            Vec3 vn = m.get_normal(i0) + m.get_normal(i1) + m.get_normal(i2);
            if (tn.dot(vn) < 0.0f) {
                bad_vert[i0] = bad_vert[i1] = bad_vert[i2] = 1;
                flipped_count++;
            }
        }

        if (flipped_count == 0) {
            if (iter == 0) std::printf("[repair] no flipped tris\n");
            break;
        }
        // Relaxation only fixes tris that are geometrically tangled. A count
        // that stops falling means what's left is TOPOLOGICAL — an edge with
        // three triangles on it, or two that disagree on winding — and no
        // amount of centroid-averaging will touch it. Keep iterating and we
        // just drag verts around for nothing.
        if (flipped_count >= prev_flipped) {
            std::printf("[repair] stalled at %u flipped tris after %d iters - "
                        "topological, not geometric (see the remesh-audit "
                        "non-manifold / flipped-winding counts)\n",
                        flipped_count, iter);
            break;
        }
        prev_flipped = flipped_count;

        std::vector<Vec3> new_pos(vc);
        // Which verts the gather loop actually produced a position for. The
        // apply loop used to re-test bad_vert+mask, which is a WEAKER predicate
        // than the gather loop's (that one also bails on n == 0) — so a vert
        // with no 1-ring got written back a default-constructed Vec3, i.e.
        // teleported to the origin. Track it explicitly instead.
        std::vector<uint8_t> relaxed(vc, 0);
        uint32_t n_moved = 0;
        for (uint32_t v = 0; v < vc; v++) {
            if (!bad_vert[v]) continue;
            if (has_mask && v < (uint32_t)m.mask.size() && m.mask[v] >= 1.0f) continue;

            Vec3 sum(0,0,0);
            int n = 0;
            uint32_t s = m.vert_tri_offset[v];
            uint32_t e = m.vert_tri_offset[v+1];
            for (uint32_t j = s; j < e; j++) {
                uint32_t tri = m.vert_tri_list[j];
                for (int k = 0; k < 3; k++) {
                    uint32_t nv = m.indices[tri*3+k];
                    if (nv != v) { sum += m.get_pos(nv); n++; }
                }
            }
            if (n == 0) continue;
            Vec3 cur = m.get_pos(v);
            Vec3 centroid = sum * (1.0f / (float)n);
            Vec3 moved = cur + (centroid - cur) * alpha;
            // Seam-respecting: a vert sitting on x=0 must stay there.
            if (std::fabs(cur.x) < seam_tol) moved.x = 0.0f;
            new_pos[v] = moved;
            relaxed[v] = 1;
            n_moved++;
        }

        for (uint32_t v = 0; v < vc; v++)
            if (relaxed[v]) m.set_pos(v, new_pos[v]);
        total_relaxed += n_moved;
        std::printf("[repair] iter %d: %u flipped tris, relaxed %u verts\n",
                    iter, flipped_count, n_moved);
    }

    if (total_relaxed > 0) m.recompute_normals();
    return total_relaxed;
}

// ---------------------------------------------------------------------------
// Watertightness audit (diagnostic)
// ---------------------------------------------------------------------------
// A closed, consistently-wound mesh uses every edge exactly twice, once in each
// direction. Each way that can fail is a defect we can name:
//   1 use          -> crack (open edge)
//   >2 uses        -> non-manifold fin
//   2 the same way -> flipped winding; shading inverts across that edge
// The old version built an EdgeTable, which stores only tri_a/tri_b and
// silently overwrites tri_b on a third triangle — so non-manifold edges read
// back as watertight, and winding was never checked at all. Count directed
// uses directly instead. Open edges are still classified by mask value + seam
// proximity to localize the culprit. Also reports needle triangles, which is
// the seam-sliver symptom stated as a number.
static void audit_open_edges(const Mesh& m, float seam_tol, const char* stage,
                             float target_edge) {
    struct EdgeUse { uint32_t v0, v1; uint16_t fwd, bwd; };
    std::unordered_map<uint64_t, EdgeUse> use;
    uint32_t tc = m.tri_count();
    uint32_t vc = m.vertex_count();
    use.reserve(tc * 3);
    for (uint32_t t = 0; t < tc; t++) {
        for (int e = 0; e < 3; e++) {
            uint32_t a = m.indices[t*3+e], b = m.indices[t*3+(e+1)%3];
            if (a >= vc || b >= vc || a == b) continue;
            uint32_t lo = std::min(a, b), hi = std::max(a, b);
            uint64_t k = ((uint64_t)lo << 32) | (uint64_t)hi;
            auto& u = use.try_emplace(k, EdgeUse{lo, hi, 0, 0}).first->second;
            if (a == lo) u.fwd++; else u.bwd++;
        }
    }

    uint32_t open_total = 0, open_seam = 0;
    uint32_t open_masked = 0, open_falloff = 0, open_unmasked = 0;
    uint32_t nonmanifold = 0, bad_winding = 0;
    uint32_t n_sample = 0;
    static constexpr uint32_t MAX_SAMPLES = 6;
    struct { float x, y, z, m0, m1; } samples[MAX_SAMPLES];

    for (const auto& [k, e] : use) {
        uint32_t uses = (uint32_t)e.fwd + (uint32_t)e.bwd;
        if (uses > 2) { nonmanifold++; continue; }
        if (uses == 2) {
            if (e.fwd != 1) bad_winding++;   // both tris traverse it the same way
            continue;
        }
        open_total++;
        bool seam_edge = std::fabs(m.pos_x[e.v0]) < seam_tol &&
                         std::fabs(m.pos_x[e.v1]) < seam_tol;
        if (seam_edge) { open_seam++; continue; }
        float m0 = (e.v0 < (uint32_t)m.mask.size()) ? m.mask[e.v0] : 0.0f;
        float m1 = (e.v1 < (uint32_t)m.mask.size()) ? m.mask[e.v1] : 0.0f;
        float mmax = std::max(m0, m1);
        if (mmax >= 1.0f)      open_masked++;
        else if (mmax > 0.0f)  open_falloff++;
        else                   open_unmasked++;
        if (n_sample < MAX_SAMPLES) {
            samples[n_sample] = {
                (m.pos_x[e.v0] + m.pos_x[e.v1]) * 0.5f,
                (m.pos_y[e.v0] + m.pos_y[e.v1]) * 0.5f,
                (m.pos_z[e.v0] + m.pos_z[e.v1]) * 0.5f,
                m0, m1 };
            n_sample++;
        }
    }

    // Needle triangles: shortest edge far under target. Split out by seam
    // proximity — a seam-heavy count is the mirror-clip sliver signature,
    // a spread-out count means the edge ops themselves are leaving junk.
    uint32_t needles = 0, needles_seam = 0;
    if (target_edge > 0.0f) {
        const float needle_len = target_edge * 0.15f;
        for (uint32_t t = 0; t < tc; t++) {
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            if (i0 >= vc || i1 >= vc || i2 >= vc) continue;
            float shortest = std::min({ edge_length(m, i0, i1),
                                        edge_length(m, i1, i2),
                                        edge_length(m, i2, i0) });
            if (shortest >= needle_len) continue;
            needles++;
            if (std::fabs(m.pos_x[i0]) < seam_tol ||
                std::fabs(m.pos_x[i1]) < seam_tol ||
                std::fabs(m.pos_x[i2]) < seam_tol) needles_seam++;
        }
    }

    if (open_total == 0 && nonmanifold == 0 && bad_winding == 0 && needles == 0) {
        std::printf("[remesh-audit] %s: watertight, manifold, consistent winding\n", stage);
        return;
    }
    if (open_total > 0)
        std::printf("[remesh-audit] %s: %u OPEN EDGES (on-seam=%u, masked=%u, "
                    "falloff=%u, unmasked=%u)\n",
                    stage, open_total, open_seam, open_masked, open_falloff,
                    open_unmasked);
    if (nonmanifold > 0)
        std::printf("[remesh-audit] %s: %u NON-MANIFOLD edges (>2 tris)\n",
                    stage, nonmanifold);
    if (bad_winding > 0)
        std::printf("[remesh-audit] %s: %u FLIPPED-WINDING edges (normals invert "
                    "across these)\n", stage, bad_winding);
    if (needles > 0)
        std::printf("[remesh-audit] %s: %u needle tris (%u touching the seam)\n",
                    stage, needles, needles_seam);
    for (uint32_t i = 0; i < n_sample; i++)
        std::printf("[remesh-audit]   open edge at (%.4f, %.4f, %.4f) "
                    "mask=(%.3f, %.3f)\n",
                    samples[i].x, samples[i].y, samples[i].z,
                    samples[i].m0, samples[i].m1);
}

// ---------------------------------------------------------------------------
// Geometric quality audit (diagnostic)
// ---------------------------------------------------------------------------
// audit_open_edges answers one question: is the connectivity a valid closed
// manifold. It can come back perfectly clean on a mesh that still looks broken,
// because every defect it knows how to name is a connectivity defect. The seam
// artifact that outlives it — small triangles sitting inside larger ones — is
// therefore geometric: the connectivity is a legal 2-manifold, the embedding of
// that manifold into 3-space is not. Three shapes produce it, listed in the
// order they are worth suspecting:
//
//   fold   two triangles sharing an edge whose normals point back at each
//          other. A vertex snapped onto the mirror plane travels up to
//          0.45 * target_edge in x under no quality guard at all
//          (consolidate_seam stage 1, and the GPU snap in
//          mirror_positive_half) — far enough to drag its triangles back
//          across their own neighbours. Renders as one triangle lying on top
//          of another.
//   cap    a triangle carrying an angle near 180 degrees. Flat, near-zero
//          area, yet all three edges can be long enough that the needle test
//          (which looks at the SHORTEST edge) never sees it.
//   tetra  a valence-3 vertex whose three neighbours already form a triangle.
//          That is a tetrahedron embedded in the surface: four triangles,
//          every edge used exactly twice, wound consistently, enclosing
//          almost no volume. Textbook "triangle within a triangle", and
//          textbook invisible to a connectivity audit. Note that
//          collapse_short_edges cannot clear one even in principle — it
//          refuses any edge with a valence-3 endpoint.
//
// One hash pass, O(E + F), so this runs unconditionally next to the
// watertightness audit rather than behind CHISEL_DEBUG_REMESH.
static void audit_geometry(const Mesh& m, float seam_tol, const char* stage,
                           float target_edge) {
    const uint32_t tc = m.tri_count();
    const uint32_t vc = m.vertex_count();
    if (tc == 0 || vc == 0) return;

    // Folds born of a seam snap sit a ring or two off the plane, not on it, so
    // "near the seam" has to be measured in target edges — seam_tol is far too
    // tight to catch the neighbourhood that the snap actually disturbs.
    const float seam_band = (target_edge > 0.0f) ? target_edge * 1.5f
                                                 : seam_tol * 50.0f;
    auto near_seam = [&](uint32_t v) -> bool {
        return std::fabs(m.pos_x[v]) < seam_band;
    };

    // --- edge -> its (up to two) triangles --------------------------------
    struct EUse { uint32_t t0, t1, uses; };
    EdgeMap emap;
    emap.reset((size_t)tc * 2);
    std::vector<EUse>     eu;
    std::vector<uint64_t> ekey;
    eu.reserve((size_t)tc * 2);
    ekey.reserve((size_t)tc * 2);

    for (uint32_t t = 0; t < tc; t++) {
        for (int e = 0; e < 3; e++) {
            uint32_t a = m.indices[t*3+e], b = m.indices[t*3+(e+1)%3];
            if (a >= vc || b >= vc || a == b) continue;
            uint32_t lo = std::min(a, b), hi = std::max(a, b);
            uint64_t k = ((uint64_t)lo << 32) | (uint64_t)hi;
            uint32_t idx = emap.find(k);
            if (idx == INVALID) {
                idx = (uint32_t)eu.size();
                eu.push_back({t, INVALID, 1});
                ekey.push_back(k);
                emap.insert(k, idx);
            } else {
                if (eu[idx].uses == 1) eu[idx].t1 = t;
                eu[idx].uses++;
            }
        }
    }

    auto unit_normal = [&](uint32_t t, Vec3& out) -> bool {
        Vec3 n = tri_normal(m, t);
        float l = n.length();
        if (l < 1e-20f) return false;
        out = n * (1.0f / l);
        return true;
    };

    // --- 1. Folds ---------------------------------------------------------
    // dot < -0.5 is a dihedral under 60 degrees: sharper than any sculpted
    // crease the brushes can leave, so anything here is a defect, not detail.
    // dot < -0.9 is folded flat back on itself, which is the visible artifact.
    static constexpr float FOLD_COS = -0.5f;
    static constexpr float HARD_COS = -0.9f;
    static constexpr uint32_t MAX_SAMPLES = 4;
    uint32_t folds = 0, folds_hard = 0, folds_seam = 0, n_fs = 0;
    struct { float x, y, z, dot; } fsample[MAX_SAMPLES];

    for (size_t i = 0; i < eu.size(); i++) {
        if (eu[i].uses != 2 || eu[i].t1 == INVALID) continue;
        Vec3 na, nb;
        if (!unit_normal(eu[i].t0, na) || !unit_normal(eu[i].t1, nb)) continue;
        float d = na.dot(nb);
        if (d >= FOLD_COS) continue;
        uint32_t v0 = (uint32_t)(ekey[i] >> 32);
        uint32_t v1 = (uint32_t)(ekey[i] & 0xFFFFFFFFu);
        folds++;
        if (d < HARD_COS) folds_hard++;
        if (near_seam(v0) || near_seam(v1)) folds_seam++;
        if (n_fs < MAX_SAMPLES) {
            fsample[n_fs] = { (m.pos_x[v0] + m.pos_x[v1]) * 0.5f,
                              (m.pos_y[v0] + m.pos_y[v1]) * 0.5f,
                              (m.pos_z[v0] + m.pos_z[v1]) * 0.5f, d };
            n_fs++;
        }
    }

    // --- 2. Cap triangles -------------------------------------------------
    // cos < -0.985 is an angle over 170 degrees.
    static constexpr float CAP_COS = -0.985f;
    uint32_t caps = 0, caps_seam = 0;
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0], i1 = m.indices[t*3+1], i2 = m.indices[t*3+2];
        if (i0 >= vc || i1 >= vc || i2 >= vc) continue;
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        Vec3 p0 = m.get_pos(i0), p1 = m.get_pos(i1), p2 = m.get_pos(i2);
        float l01 = (p1-p0).length(), l12 = (p2-p1).length(), l02 = (p2-p0).length();
        if (l01 < 1e-12f || l12 < 1e-12f || l02 < 1e-12f) continue;
        float c0 = (p1-p0).dot(p2-p0) / (l01*l02);
        float c1 = (p0-p1).dot(p2-p1) / (l01*l12);
        float c2 = (p0-p2).dot(p1-p2) / (l02*l12);
        if (std::min({c0, c1, c2}) > CAP_COS) continue;
        caps++;
        if (near_seam(i0) || near_seam(i1) || near_seam(i2)) caps_seam++;
    }

    // --- 3. Embedded tetrahedra ------------------------------------------
    // Valence and the first three neighbours per vertex, straight off the edge
    // list. Valence saturates so a high-valence vertex costs nothing extra.
    std::vector<uint8_t>  val(vc, 0);
    std::vector<uint32_t> nbr((size_t)vc * 3, INVALID);
    auto push_nbr = [&](uint32_t v, uint32_t o) {
        if (val[v] < 3) nbr[(size_t)v*3 + val[v]] = o;
        if (val[v] < 255) val[v]++;
    };
    for (size_t i = 0; i < eu.size(); i++) {
        uint32_t v0 = (uint32_t)(ekey[i] >> 32);
        uint32_t v1 = (uint32_t)(ekey[i] & 0xFFFFFFFFu);
        push_nbr(v0, v1);
        push_nbr(v1, v0);
    }

    // Sorted-triple hash of every triangle, so "do these three verts already
    // span a face" is a lookup. Same key shape remove_doubled_tris uses.
    auto tri_key = [](uint32_t a, uint32_t b, uint32_t c) -> uint64_t {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        uint64_t k = ((uint64_t)a * 0x9E3779B97F4A7C15ull) ^
                     ((uint64_t)b * 0xC2B2AE3D27D4EB4Full) ^
                     ((uint64_t)c * 0x165667B19E3779F9ull);
        if (k == EdgeMap::EMPTY || k == EdgeMap::TOMB) k ^= 1ull;
        return k;
    };
    EdgeMap tmap;
    tmap.reset(tc);
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0], i1 = m.indices[t*3+1], i2 = m.indices[t*3+2];
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        tmap.insert(tri_key(i0, i1, i2), t);
    }

    uint32_t tetras = 0, tetras_seam = 0, n_ts = 0;
    struct { float x, y, z; uint32_t v; } tsample[MAX_SAMPLES];
    for (uint32_t v = 0; v < vc; v++) {
        if (val[v] != 3) continue;
        uint32_t a = nbr[(size_t)v*3+0], b = nbr[(size_t)v*3+1], c = nbr[(size_t)v*3+2];
        if (a == INVALID || b == INVALID || c == INVALID) continue;
        uint32_t t = tmap.find(tri_key(a, b, c));
        if (t == INVALID) continue;
        // Hash hit must be confirmed against the real vertex set — a collision
        // here would invent a defect that isn't there.
        uint32_t q0 = m.indices[t*3+0], q1 = m.indices[t*3+1], q2 = m.indices[t*3+2];
        bool ha = (q0 == a || q1 == a || q2 == a);
        bool hb = (q0 == b || q1 == b || q2 == b);
        bool hc = (q0 == c || q1 == c || q2 == c);
        if (!ha || !hb || !hc) continue;
        tetras++;
        if (near_seam(v)) tetras_seam++;
        if (n_ts < MAX_SAMPLES) {
            tsample[n_ts] = { m.pos_x[v], m.pos_y[v], m.pos_z[v], v };
            n_ts++;
        }
    }

    // --- 4. Connected components -----------------------------------------
    // A stray shard reads as clean topology on its own. mirror_positive_half
    // culls all but the largest component, so anything above 1 here at
    // post-mirror means the cull let something through.
    std::vector<uint32_t> parent(tc);
    for (uint32_t t = 0; t < tc; t++) parent[t] = t;
    auto find_root = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (size_t i = 0; i < eu.size(); i++) {
        if (eu[i].t1 == INVALID) continue;
        uint32_t ra = find_root(eu[i].t0), rb = find_root(eu[i].t1);
        if (ra != rb) parent[ra] = rb;
    }
    std::vector<uint32_t> csize(tc, 0);
    for (uint32_t t = 0; t < tc; t++) csize[find_root(t)]++;
    uint32_t components = 0, largest = 0;
    for (uint32_t t = 0; t < tc; t++) {
        if (csize[t] == 0) continue;
        components++;
        largest = std::max(largest, csize[t]);
    }

    if (folds == 0 && caps == 0 && tetras == 0 && components == 1) {
        std::printf("[remesh-geom] %s: no folds, caps or embedded tetrahedra\n", stage);
        return;
    }
    if (folds > 0)
        std::printf("[remesh-geom] %s: %u FOLD edges (%u folded flat, %u within "
                    "%.4f of the seam)\n", stage, folds, folds_hard, folds_seam,
                    seam_band);
    if (caps > 0)
        std::printf("[remesh-geom] %s: %u CAP tris, angle >170 deg (%u near seam)\n",
                    stage, caps, caps_seam);
    if (tetras > 0)
        std::printf("[remesh-geom] %s: %u EMBEDDED TETRAHEDRA, valence-3 vert over an "
                    "existing tri (%u near seam)\n", stage, tetras, tetras_seam);
    if (components != 1)
        std::printf("[remesh-geom] %s: %u components, largest holds %u/%u tris\n",
                    stage, components, largest, tc);
    for (uint32_t i = 0; i < n_fs; i++)
        std::printf("[remesh-geom]   fold at (%.4f, %.4f, %.4f) n.n=%.3f\n",
                    fsample[i].x, fsample[i].y, fsample[i].z, fsample[i].dot);
    for (uint32_t i = 0; i < n_ts; i++)
        std::printf("[remesh-geom]   tetra tip v%u at (%.4f, %.4f, %.4f)\n",
                    tsample[i].v, tsample[i].x, tsample[i].y, tsample[i].z);
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

RemeshResult perform_remesh(Mesh& mesh, MultiresStack& stack,
                            float target_edge_length, int iterations,
                            ComputeState* cs,
                            float density_coarse_mult, float density_fine_mult) {
    RemeshResult r;
    r.old_verts = mesh.vertex_count();
    r.old_tris  = mesh.tri_count();

    // --- Entry validation ---
    // Every bail below leaves `mesh` untouched and reports through r.error,
    // which main.cpp already prints + toasts. Without these the first cs->
    // call was a null deref on a compute-less device (main.cpp passes nullptr
    // there), and r.error — the only thing the failure toast has to say —
    // was never written by anyone.
    if (cs == nullptr || !cs->supported) {
        r.error = "GPU compute unavailable - remesh needs a compute-capable device";
        std::printf("[remesh] abort: %s\n", r.error.c_str());
        return r;
    }
    if (r.old_verts == 0 || r.old_tris == 0) {
        r.error = "mesh is empty";
        std::printf("[remesh] abort: %s\n", r.error.c_str());
        return r;
    }
    if (mesh.indices.size() % 3 != 0) {
        r.error = "index buffer is not a whole number of triangles";
        std::printf("[remesh] abort: %s (%zu indices)\n",
                    r.error.c_str(), mesh.indices.size());
        return r;
    }
    {
        // One linear scan for out-of-range indices. Everything downstream
        // indexes pos_*/mask/pinned raw, so a single bad index is a silent
        // OOB read that only surfaces later as garbage geometry.
        uint32_t bad = INVALID;
        for (uint32_t idx : mesh.indices)
            if (idx >= r.old_verts) { bad = idx; break; }
        if (bad != INVALID) {
            r.error = "index buffer references out-of-range vertices";
            std::printf("[remesh] abort: %s (index %u >= %u verts)\n",
                        r.error.c_str(), bad, r.old_verts);
            return r;
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    // Ensure adjacency is built
    if (mesh.vert_tri_offset.empty()) mesh.build_adjacency();

    // Compute target edge length from mean if auto
    if (target_edge_length <= 0.0f)
        target_edge_length = compute_mean_edge_length(mesh);

    float high = 1.4f * target_edge_length;
    float low  = 0.8f * target_edge_length;   // canonical 4/3:4/5 band (was 0.6 — too wide, left fine tris)
    float seam_tol = std::max(1e-5f, target_edge_length * 0.01f);

    // Repair brush-induced flipped tris before the edge ops can amplify them.
    repair_flipped_tris(mesh, seam_tol);

    // Baseline on the incoming mesh. Sculpting can fold a surface on its own,
    // so without this line every fold at pre-mirror looks like the remesher's
    // doing and there is no way to tell inherited from minted.
    audit_geometry(mesh, seam_tol, "entry", target_edge_length);

    // Build edge table
    EdgeTable et;
    et.build(mesh);

    // ---- Adaptive sizing (spec §4): painted density field → per-vertex
    // target edge length. The field maps log-space (edge lengths are
    // multiplicative): density 0 → L_base×coarse_mult, 1 → L_base×fine_mult,
    // 0.5 → exactly L_base when coarse×fine == 1. Empty field = uniform path,
    // byte-identical to before the feature existed.
    std::vector<float> target_len;   // per-vert; empty = uniform
    const bool adaptive = !mesh.density.empty() &&
                          density_coarse_mult != density_fine_mult;
    if (adaptive) {
        const float lc = std::log(std::max(0.01f, density_coarse_mult));
        const float lf = std::log(std::max(0.01f, density_fine_mult));
        const uint32_t vc = mesh.vertex_count();
        target_len.resize(vc);
        for (uint32_t v = 0; v < vc; v++) {
            float d = (v < (uint32_t)mesh.density.size()) ? mesh.density[v] : 0.5f;
            d = std::min(1.0f, std::max(0.0f, d));
            target_len[v] = target_edge_length * std::exp(lc + d * (lf - lc));
        }
        // Gradient limiting (spec §4.2, required): a hard green→red boundary
        // is a 4× target jump on one edge — walls of degenerate transition
        // tris. Sweep the WORKING COPY (the paint itself is never mutated)
        // until no adjacent pair exceeds factor g; this manufactures the
        // yellow buffer bands automatically. Denser (shorter) side wins.
        static constexpr float GRADE_G = 1.3f;
        int sweeps = 0;
        for (; sweeps < 64; sweeps++) {
            bool changed = false;
            for (const auto& e : et.edges) {
                if (e.dead) continue;
                float& la = target_len[e.v0];
                float& lb = target_len[e.v1];
                if (la > GRADE_G * lb)      { la = GRADE_G * lb; changed = true; }
                else if (lb > GRADE_G * la) { lb = GRADE_G * la; changed = true; }
            }
            if (!changed) break;
        }
        std::printf("[remesh] adaptive: density field drives sizing "
                    "(coarse x%.2f, fine x%.2f), graded in %d sweeps\n",
                    density_coarse_mult, density_fine_mult, sweeps);
    }
    std::vector<float>* tlen = adaptive ? &target_len : nullptr;

    // With a painted field every tri is a sizing candidate (its local band
    // decides what happens) — the stretched-only heuristic would skip painted
    // regions whose edges look fine against the GLOBAL target. target 0 ⇒ the
    // select kernel flags everything.
    const float select_target = adaptive ? 0.0f : target_edge_length;

    // Decide selection strategy: mask-driven or auto-detect.
    // tri_selected mirrors remesh_trisel_ssbo on GPU; we only readback when CPU
    // consumers (split/collapse/flip, counts) need it.
    bool using_mask = !mesh.mask.empty();
    std::vector<uint32_t> tri_selected;
    static constexpr int SUPPORT_RINGS = 2;

    // Selection mask, binarized once at entry and maintained across topology ops
    // (split appends with an AND rule, compact_mesh remaps). Selection is driven
    // by THIS instead of live mask values: split midpoints get averaged mask
    // values that drift across the 0.5 threshold, which made the core border —
    // and with it the pinned set — churn on every rebuild.
    std::vector<float> sel_mask;
    if (using_mask) {
        sel_mask.resize(mesh.mask.size());
        for (size_t i = 0; i < mesh.mask.size(); i++)
            sel_mask[i] = (mesh.mask[i] >= 0.5f) ? 1.0f : 0.0f;
    }

    // GPU grow_selection / mirror_selection need adjacency on GPU; upload before select.
    cs->upload_adjacency(mesh.vert_tri_offset.data(),
                         (uint32_t)mesh.vert_tri_offset.size(),
                         mesh.vert_tri_list.data(),
                         (uint32_t)mesh.vert_tri_list.size());

    if (using_mask) {
        cs->dispatch_select_unmasked(mesh.vertex_count(), mesh.tri_count(),
            mesh.indices.data(),
            sel_mask.data(), (uint32_t)sel_mask.size());
        cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
    } else {
        cs->dispatch_select_stretched(mesh.vertex_count(), mesh.tri_count(),
            mesh.indices.data(),
            mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
            select_target);
    }

    // For both paths, readback once so CPU can count + drive split/collapse/flip.
    cs->readback_trisel(mesh.tri_count(), tri_selected);

    // Count before growing (non-mask path grows later)
    uint32_t raw_selected = 0;
    for (uint32_t b : tri_selected) if (b) raw_selected++;

    if (raw_selected == 0) {
        auto t1 = std::chrono::steady_clock::now();
        r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.new_verts = mesh.vertex_count();
        r.new_tris  = mesh.tri_count();
        r.selected_tris = 0;
        r.success = true;
        std::printf("[remesh] no stretched triangles found, nothing to do (%.1f ms)\n",
                    r.elapsed_ms);
        return r;
    }

    // Non-mask path: grow for smooth transition (on GPU).
    if (!using_mask) {
        cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), 8);
    }

    // Mirror the selection for symmetry (skip if no mirror map).
    if (!mesh.mirror_x_map.empty())
        cs->dispatch_mirror_selection(mesh.vertex_count(), mesh.tri_count());

    // Final readback after grow + mirror — CPU now has the authoritative selection.
    cs->readback_trisel(mesh.tri_count(), tri_selected);

    uint32_t total_selected = 0;
    for (uint32_t b : tri_selected) if (b) total_selected++;
    r.selected_tris = total_selected;

    // Find pinned boundary vertices (includes mirror seam). tri_sel already on GPU.
    std::vector<uint32_t> pinned;

    // Fully-masked verts are hard pins regardless of selection topology: the
    // grown selection reaches SUPPORT_RINGS into masked territory, and without
    // this the smooth pass and collapse midpoints could move protected geometry.
    auto apply_mask_pins = [&]() {
        if (!using_mask) return;
        uint32_t n = std::min<uint32_t>(mesh.vertex_count(), (uint32_t)mesh.mask.size());
        n = std::min<uint32_t>(n, (uint32_t)pinned.size());
        for (uint32_t v = 0; v < n; v++)
            if (mesh.mask[v] >= 1.0f) pinned[v] = 1u;
    };

    cs->dispatch_find_pinned(mesh.vertex_count(), mesh.tri_count(),
                             mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                             seam_tol, pinned);
    {
        uint32_t vc = mesh.vertex_count();
        // split/collapse index pinned[] raw (unlike tri_selected, which they
        // bounds-check). Keep it at least vertex_count long so an under-filled
        // readback can't turn into an OOB read deep in the edge ops.
        if (pinned.size() < vc) pinned.resize(vc, 0u);
        for (uint32_t v = 0; v < vc; v++)
            if (pinned[v] && std::fabs(mesh.pos_x[v]) < seam_tol)
                mesh.pos_x[v] = 0.0f;
    }
    apply_mask_pins();

    // Per-vertex smoothing weights (mask path only): mask-proportional falloff,
    // computed on CPU and uploaded by dispatch_remesh_smooth. weight = 1 - mask
    // for any unpinned vert touching the selection — a continuous transition
    // instead of the old 2-ring quantized ramp, which hit zero exactly where the
    // border band needed relaxing most.
    std::vector<float> smooth_weights;
    auto compute_weights_cpu = [&]() {
        if (!using_mask) return;
        uint32_t vc = mesh.vertex_count();
        smooth_weights.assign(vc, 0.0f);
        for (uint32_t v = 0; v < vc; v++) {
            if (v < (uint32_t)pinned.size() && pinned[v]) continue;
            uint32_t ts = mesh.vert_tri_offset[v];
            uint32_t te = mesh.vert_tri_offset[v + 1];
            bool any_sel = false;
            for (uint32_t j = ts; j < te; j++) {
                uint32_t t = mesh.vert_tri_list[j];
                if (t < (uint32_t)tri_selected.size() && tri_selected[t]) { any_sel = true; break; }
            }
            if (!any_sel) continue;
            float mv = (v < (uint32_t)mesh.mask.size()) ? mesh.mask[v] : 0.0f;
            smooth_weights[v] = std::max(0.0f, 1.0f - mv);
        }
    };
    compute_weights_cpu();

    std::printf("[remesh] %u raw selected, %u total (after grow+support), "
                "target_edge=%.4f, %d iters\n",
                raw_selected, total_selected, target_edge_length, iterations);

    // Helper: full rebuild of all transient state after topology changes
    auto rebuild_all = [&]() {
        compact_mesh(mesh, &sel_mask, tlen);
        mesh.mirror_x_map.clear();
        mesh.build_adjacency();
        cs->upload_adjacency(mesh.vert_tri_offset.data(),
                             (uint32_t)mesh.vert_tri_offset.size(),
                             mesh.vert_tri_list.data(),
                             (uint32_t)mesh.vert_tri_list.size());
        mesh.recompute_normals();

        if (using_mask) {
            cs->dispatch_select_unmasked(mesh.vertex_count(), mesh.tri_count(),
                mesh.indices.data(),
                sel_mask.data(), (uint32_t)sel_mask.size());
            cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
        } else {
            cs->dispatch_select_stretched(mesh.vertex_count(), mesh.tri_count(),
                mesh.indices.data(),
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                select_target);
            cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), 8);
        }

        // mirror_x_map was just cleared, so mirror spread is naturally a no-op
        // until the final mirror_positive_half pass — no dispatch needed here.

        cs->readback_trisel(mesh.tri_count(), tri_selected);

        cs->dispatch_find_pinned(mesh.vertex_count(), mesh.tri_count(),
                                 mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                                 seam_tol, pinned);
        uint32_t vc = mesh.vertex_count();
        if (pinned.size() < vc) pinned.resize(vc, 0u);
        for (uint32_t v = 0; v < vc; v++)
            if (pinned[v] && std::fabs(mesh.pos_x[v]) < seam_tol)
                mesh.pos_x[v] = 0.0f;
        apply_mask_pins();
        compute_weights_cpu();
    };

    // Flip changes indices but not vertex/tri count and not which verts belong
    // to the selection (flip is gated on both adjacent tris being selected,
    // and the four involved verts keep the same set of adjacent selected tris).
    // So pinned, weights, selection, and mirror map are all
    // invariant — just refresh CPU+GPU adjacency and recompute normals.
    auto rebuild_after_flip = [&]() {
        mesh.build_adjacency();
        cs->upload_adjacency(mesh.vert_tri_offset.data(),
                             (uint32_t)mesh.vert_tri_offset.size(),
                             mesh.vert_tri_list.data(),
                             (uint32_t)mesh.vert_tri_list.size());
        mesh.recompute_normals();
    };

    // Smooth touches positions only — no topology change, pinned verts don't
    // move (shader gates on pinned[v]), so adjacency/selection/pinned/weights
    // are all still valid. Only normals need refreshing for the next iter's
    // split-midpoint normal interpolation.
    auto rebuild_after_smooth = [&]() {
        mesh.recompute_normals();
    };

    // Remeshing iterations — rebuild topology where it changes.
    // Convergence break: once split+collapse+flip drops below max(8, 0.1% of tri count)
    // the next iter buys ~nothing topologically (Botsch-Kobbelt observation), so we
    // run a final smooth and bail. Floor of 8 prevents tiny meshes from doing all
    // 10 iters for 1-2 trivial ops.
    int iters_done = 0;
    double t_split = 0.0, t_collapse = 0.0, t_flip = 0.0, t_rebuild = 0.0;
    auto tick = [] { return std::chrono::steady_clock::now(); };
    auto since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - a).count();
    };

    // Baseline BEFORE any edge op, so a defect present in the incoming mesh is
    // attributed to the input rather than to whichever pass runs first.
    REMESH_TOPO_RESET(mesh);

    for (int iter = 0; iter < iterations; iter++) {
        auto ts = tick();
        uint32_t n_split = split_long_edges(mesh, et, tri_selected, pinned, high, seam_tol, &sel_mask, tlen);
        t_split += since(ts);

        // Skip the rebuild when a pass changed nothing. split_long_edges and
        // collapse_short_edges both rebuild adjacency and the edge table
        // internally, so with zero ops the topology, normals and GPU-side CSR
        // are already exactly what rebuild_all would reproduce. Every skipped
        // call saves two GPU readback stalls plus a full-mesh reselect — and
        // the late iterations, which is most of them, do little or nothing.
        if (n_split > 0) { ts = tick(); rebuild_all(); t_rebuild += since(ts); }

        ts = tick();
        uint32_t n_collapse = collapse_short_edges(mesh, et, tri_selected, pinned, low, seam_tol, &sel_mask, tlen);
        t_collapse += since(ts);

        // Sweep fins before they propagate: split refines them into more fins,
        // and every later pass is blind to them.
        uint32_t n_doubled = remove_doubled_tris(mesh);
        if (n_doubled > 0)
            std::printf("[remesh] iter %d: removed %u doubled tris\n", iter, n_doubled);
        REMESH_TOPO_TRACE(mesh, "dedup iter", iter, (int)n_doubled, -1);

        if (n_collapse > 0 || n_doubled > 0) {
            ts = tick();
            rebuild_all();
            // Rebuilt AFTER rebuild_all, not before: rebuild_all compacts, and
            // a compaction renumbers verts and tris out from under the edge
            // table that flip_edges is about to walk.
            et.build(mesh);
            t_rebuild += since(ts);
        }

        ts = tick();
        uint32_t n_flip = flip_edges(mesh, et, tri_selected, pinned);
        t_flip += since(ts);
        if (n_flip > 0) rebuild_after_flip();

        if (cs->has_remesh_smooth()) {
            cs->dispatch_remesh_smooth(
                mesh.vertex_count(), mesh.tri_count(),
                mesh.indices.data(),
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                mesh.norm_x.data(), mesh.norm_y.data(), mesh.norm_z.data(),
                smooth_weights, pinned,
                0.8f, seam_tol,
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data());
        }
        // Control point: smoothing moves positions only, so this must never
        // fire. If it does, the smooth kernel is corrupting indices.
        REMESH_TOPO_TRACE(mesh, "smooth iter", iter, 0, -1);

        uint32_t total_ops = n_split + n_collapse + n_flip;
        uint32_t threshold = std::max<uint32_t>(8, mesh.tri_count() / 1000);
        iters_done = iter + 1;
        std::printf("[remesh] iter %d: %u split, %u collapse, %u flip (threshold=%u)\n",
                    iter, n_split, n_collapse, n_flip, threshold);
        if (total_ops < threshold) {
            std::printf("[remesh] converged at iter %d (%u ops < %u threshold)\n",
                        iter, total_ops, threshold);
            break;
        }
        if (iter + 1 < iterations) rebuild_after_smooth();
    }
    std::printf("[remesh] outer loop: %d/%d iters (split %.0f ms, collapse %.0f ms, "
                "flip %.0f ms, rebuild %.0f ms)\n",
                iters_done, iterations, t_split, t_collapse, t_flip, t_rebuild);

    // Final sweep before the mirror — the reflection duplicates whatever it is
    // handed, so a fin that survives to here comes out as two fins.
    {
        uint32_t n_doubled = remove_doubled_tris(mesh);
        if (n_doubled > 0)
            std::printf("[remesh] pre-mirror: removed %u doubled tris\n", n_doubled);
    }

    // Final compaction and rebuild
    compact_mesh(mesh);

    audit_open_edges(mesh, seam_tol, "pre-mirror", target_edge_length);
    audit_geometry(mesh, seam_tol, "pre-mirror", target_edge_length);

    mirror_positive_half(mesh, seam_tol, target_edge_length, cs);

    mesh.build_adjacency();
    mesh.recompute_normals();

    audit_open_edges(mesh, seam_tol, "post-mirror", target_edge_length);
    // Run on both sides of the mirror: the seam snap, the plane clip and the
    // reflection all live inside mirror_positive_half, so a defect that appears
    // only in the second line was minted there.
    audit_geometry(mesh, seam_tol, "post-mirror", target_edge_length);

    // The mirror step is the last thing that can leave inverted geometry: the
    // reflected half is emitted with reversed winding, and the component cull
    // can strip a tri that was propping up its neighbours. Everything before
    // here got a repair pass at entry; this one had none.
    repair_flipped_tris(mesh, seam_tol);

    // Replace multires stack (fresh level-0 lock on the new topology). Paint
    // planes restart from the remeshed surface: colour survived the remesh
    // (edge ops interpolate it), the mask dies with the old topology
    // (main.cpp clears mesh.mask right after).
    stack.base          = mesh;
    stack.base_level    = 0;
    stack.current_level = 0;
    stack.disp.clear();
    stack.frames.clear();
    stack.mirror.clear();
    stack.base_mirror.clear();
    stack.midpoint_parents.clear();
    // Remesh is a relock onto brand-new topology, so it owes everything
    // multires_stack_init_from_lock() clears. These two were being missed:
    // topo_cache still described the OLD subdivision chain (and is flagged
    // ready), so the next subdivide-up fed a stale tc.vcount into
    // build_parent_map as the fine-level count — underflowing V_fine -
    // V_coarse into a ~34 GB allocation. lock_stamp keys the GPU cascade's
    // VRAM tables to a chain identity, so a carried-over stamp would let the
    // GPU replay reuse tables built for the pre-remesh topology.
    stack.topo_cache.clear();
    stack.lock_stamp = 0;
    stack.color = mesh.color;
    if (!stack.color.empty()) stack.color.resize(mesh.vertex_count(), 0xFFFFFFFFu);
    stack.mask.clear();
    // Density survives like colour (edge ops interpolate it) — it exists to
    // steer exactly this kind of restructuring, so it must outlive it.
    stack.density = mesh.density;
    if (!stack.density.empty()) stack.density.resize(mesh.vertex_count(), 0.5f);
    stack.base.color.clear();
    stack.base.mask.clear();
    stack.base.density.clear();
    stack.locked = true;

    auto t1 = std::chrono::steady_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.new_verts = mesh.vertex_count();
    r.new_tris  = mesh.tri_count();
    r.success   = true;

    std::printf("[remesh] %u verts %u tris -> %u verts %u tris "
                "(%.1f ms, edge=%.4f, iters=%d)\n",
                r.old_verts, r.old_tris, r.new_verts, r.new_tris,
                r.elapsed_ms, target_edge_length, iterations);
    return r;
}

uint64_t predict_adaptive_tris(const Mesh& mesh, float target_edge_length,
                               float coarse_mult, float fine_mult) {
    if (target_edge_length <= 0.0f)
        target_edge_length = compute_mean_edge_length(mesh);
    const float lc = std::log(std::max(0.01f, coarse_mult));
    const float lf = std::log(std::max(0.01f, fine_mult));
    const uint32_t tc = mesh.tri_count();
    const uint32_t dc = (uint32_t)mesh.density.size();
    double pred = 0.0;
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = mesh.indices[t*3+0];
        uint32_t i1 = mesh.indices[t*3+1];
        uint32_t i2 = mesh.indices[t*3+2];
        float le = (edge_length(mesh, i0, i1) +
                    edge_length(mesh, i1, i2) +
                    edge_length(mesh, i2, i0)) * (1.0f / 3.0f);
        float d0 = (i0 < dc) ? mesh.density[i0] : 0.5f;
        float d1 = (i1 < dc) ? mesh.density[i1] : 0.5f;
        float d2 = (i2 < dc) ? mesh.density[i2] : 0.5f;
        float d = std::min(1.0f, std::max(0.0f, (d0 + d1 + d2) * (1.0f / 3.0f)));
        float lt = target_edge_length * std::exp(lc + d * (lf - lc));
        float ratio = (lt > 1e-12f) ? le / lt : 1.0f;
        // A tri whose edges exceed its local target splits into ~ratio² pieces;
        // never credit collapses (conservative — this is a device-loss guard).
        pred += std::max(1.0f, ratio * ratio);
    }
    return (uint64_t)pred;
}
