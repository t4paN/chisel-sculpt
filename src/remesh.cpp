#include "remesh.h"
#include "compute.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <functional>
#include <unordered_set>
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

struct EdgeTable {
    std::vector<EdgeEntry> edges;
    std::unordered_map<uint64_t, uint32_t> lookup; // (v0,v1) -> edge index

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
        lookup.clear();
        vert_edge_extra.clear();
        uint32_t vc = m.vertex_count();
        uint32_t tc = m.tri_count();
        lookup.reserve(tc * 3);
        edges.reserve(tc * 3 / 2);

        for (uint32_t t = 0; t < tc; t++) {
            uint32_t idx[3] = { m.indices[t*3+0], m.indices[t*3+1], m.indices[t*3+2] };
            for (int e = 0; e < 3; e++) {
                uint32_t a = idx[e], b = idx[(e+1)%3];
                uint64_t k = key(a, b);
                auto it = lookup.find(k);
                if (it == lookup.end()) {
                    uint32_t ei = (uint32_t)edges.size();
                    uint32_t lo = std::min(a, b), hi = std::max(a, b);
                    edges.push_back({lo, hi, t, INVALID, false});
                    lookup[k] = ei;
                } else {
                    edges[it->second].tri_b = t;
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
        auto it = lookup.find(key(a, b));
        return (it != lookup.end()) ? it->second : INVALID;
    }

    void add_edge(uint32_t v0_, uint32_t v1_, uint32_t ta, uint32_t tb) {
        uint32_t lo = std::min(v0_, v1_), hi = std::max(v0_, v1_);
        uint32_t ei = (uint32_t)edges.size();
        edges.push_back({lo, hi, ta, tb, false});
        lookup[key(lo, hi)] = ei;
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

static int vertex_valence(const EdgeTable& et, uint32_t v) {
    int val = 0;
    et.for_edges_at(v, [&](uint32_t ei) {
        if (!et.edges[ei].dead) val++;
        return true;
    });
    return val;
}


// ---------------------------------------------------------------------------
// Pass 1: Split long edges
// ---------------------------------------------------------------------------

static uint32_t split_long_edges(Mesh& m, EdgeTable& et,
                                 std::vector<uint32_t>& tri_selected,
                                 std::vector<uint32_t>& pinned,
                                 float high, float seam_tol) {
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

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        // Rebuild adjacency so vert_tri_offset/vert_tri_list are current.
        m.build_adjacency();

        // Scan all selected tris for long edges.  Use adjacency (not the
        // edge table) to find both tris sharing each edge so the lookup is
        // never stale.
        uint32_t tc = m.tri_count();
        std::unordered_map<uint64_t, SplitEdge> edge_map;

        for (uint32_t t = 0; t < tc; t++) {
            if (t >= (uint32_t)tri_selected.size() || !tri_selected[t]) continue;

            uint32_t idx[3] = { m.indices[t*3+0], m.indices[t*3+1], m.indices[t*3+2] };
            for (int e = 0; e < 3; e++) {
                uint32_t va = idx[e], vb = idx[(e+1)%3];

                uint64_t k = edge_key(va, vb);
                if (edge_map.count(k)) continue;
                if (edge_length(m, va, vb) <= high) continue;  // was high*1.1 (1.54t) — left tris large

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
                edge_map[k] = {lo, hi_v, tri_a, tri_b};
            }
        }

        if (edge_map.empty()) break;

        uint32_t num_split = 0;
        std::unordered_set<uint32_t> touched_tris;

        // Batch-split all found edges.  Two long edges that share a triangle
        // cannot both be split in the same sub-pass — the first split rewrites
        // the shared triangle in-place, making its index list inconsistent for
        // the second split and producing a one-sided split (boundary edge/hole).
        // Guard: skip any edge whose adjacent triangles were already touched by
        // an earlier split in this batch; the outer loop's next sub-pass will
        // catch the deferred edges with fresh adjacency.
        for (auto& [k, se] : edge_map) {
            if (se.tri_a != INVALID && touched_tris.count(se.tri_a)) continue;
            if (se.tri_b != INVALID && touched_tris.count(se.tri_b)) continue;

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
            if (!m.color.empty()) {
                uint32_t ca = (va < (uint32_t)m.color.size()) ? m.color[va] : 0xFFFFFFFFu;
                uint32_t cb = (vb < (uint32_t)m.color.size()) ? m.color[vb] : 0xFFFFFFFFu;
                m.color.push_back(color_avg(ca, cb));
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
                touched_tris.insert(tri);

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
                touched_tris.insert(new_tri);
            }
        }

        // No adjacency rebuild here: the next sub-pass's first statement
        // is m.build_adjacency() (line ~185), and after the loop exits the
        // caller's rebuild_all() rebuilds it before anything reads it.
        total_split += num_split;
        std::printf("[split] sub-pass %d: split %u edges, mesh now %u verts %u tris\n",
                    iter, num_split, m.vertex_count(), m.tri_count());
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

static void compact_mesh(Mesh& m); // forward declaration — defined after flip/smooth

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
static bool link_condition(const EdgeTable& et, uint32_t v0, uint32_t v1) {
    std::unordered_set<uint32_t> nbrs0;
    et.for_edges_at(v0, [&](uint32_t ei) {
        const auto& e = et.edges[ei];
        if (!e.dead) {
            uint32_t other = (e.v0 == v0) ? e.v1 : e.v0;
            if (other != v1) nbrs0.insert(other);
        }
        return true;
    });
    int shared = 0;
    et.for_edges_at(v1, [&](uint32_t ei) {
        const auto& e = et.edges[ei];
        if (!e.dead) {
            uint32_t other = (e.v0 == v1) ? e.v1 : e.v0;
            if (other != v0 && nbrs0.count(other)) shared++;
        }
        return true;
    });
    return shared == 2;
}

static uint32_t collapse_short_edges(Mesh& m, EdgeTable& et,
                                     std::vector<uint32_t>& tri_selected,
                                     std::vector<uint32_t>& pinned,
                                     float low, float seam_tol) {
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

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        m.build_adjacency();
        et.build(m);

        std::unordered_set<uint32_t> consumed_verts;
        std::vector<CollapseOp> ops;

        uint32_t num_edges = (uint32_t)et.edges.size();
        for (uint32_t ei = 0; ei < num_edges; ei++) {
            const auto& e = et.edges[ei];
            if (e.dead) continue;

            bool sel_a = (e.tri_a != INVALID && e.tri_a < (uint32_t)tri_selected.size() && tri_selected[e.tri_a]);
            bool sel_b = (e.tri_b != INVALID && e.tri_b < (uint32_t)tri_selected.size() && tri_selected[e.tri_b]);
            if (!sel_a && !sel_b) continue;

            float len = edge_length(m, e.v0, e.v1);
            if (len >= low) continue;

            uint32_t va = e.v0, vb = e.v1;

            if (consumed_verts.count(va) || consumed_verts.count(vb)) continue;

            // Block collapse of edges whose adjacent tris stitch a PROTECTED
            // boundary — an off-seam pin (mask / grown-selection border) together
            // with an interior vert. Collapsing those breaks connectivity to the
            // protected region and leaves holes at the border.
            //
            // Mirror-seam pins (|x| < seam_tol) are deliberately NOT counted as a
            // boundary pin here: seam verts are pinned for the whole loop, so the
            // center line otherwise stays frozen at input density while the interior
            // re-spaces to target — a visibly finer seam band. Letting a pure seam
            // edge [seam, seam, interior] collapse (pinned-pinned, snapped to x=0
            // below) decimates the seam to target spacing. A true mask border
            // [off-seam-pin, interior, interior] still trips the guard and is blocked.
            {
                bool boundary_edge = false;
                uint32_t adj_tris[2] = { e.tri_a, e.tri_b };
                for (uint32_t tri : adj_tris) {
                    if (tri == INVALID) continue;
                    bool has_border_pin = false, has_interior = false;
                    for (int k = 0; k < 3; k++) {
                        uint32_t tv = m.indices[tri*3+k];
                        bool is_pinned = (tv < (uint32_t)pinned.size() && pinned[tv]);
                        bool on_seam   = (std::fabs(m.pos_x[tv]) < seam_tol);
                        if (is_pinned && !on_seam) has_border_pin = true;
                        else if (!is_pinned)       has_interior   = true;
                    }
                    if (has_border_pin && has_interior) { boundary_edge = true; break; }
                }
                if (boundary_edge) continue;
            }

            // Pinned-vs-near-seam-free leaves an orphaned mirror partner on the
            // far side; block. Pinned-vs-pinned is allowed below and snaps to x=0.
            if (pinned[va] && !pinned[vb] && std::fabs(m.pos_x[vb]) < seam_tol) continue;
            if (pinned[vb] && !pinned[va] && std::fabs(m.pos_x[va]) < seam_tol) continue;
            if (e.tri_a == INVALID || e.tri_b == INVALID) continue;

            int val_a = vertex_valence(et, va);
            int val_b = vertex_valence(et, vb);
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
                new_pos.x = 0.0f;  // snap exact: pinned must sit on x=0
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

            consumed_verts.insert(va);
            consumed_verts.insert(vb);
            ops.push_back({v_keep, v_remove, new_pos});
        }

        if (ops.empty()) break;
        total_collapse += (uint32_t)ops.size();

        // Apply all collected collapses.  Use vert_tri_offset/vert_tri_list
        // (built fresh at the top of this sub-pass) rather than the edge
        // table — consumed_verts guarantees no two ops share a vertex, so
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

            compact_mesh(m);
            tri_selected = std::move(new_ts);
            pinned       = std::move(new_pinned);
        }
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
    std::unordered_set<uint32_t> touched_tris;

    uint32_t total_flip = 0;
    int pass = 0;
    bool changed = true;
    while (changed && pass < MAX_PASSES) {
        changed = false;
        touched_tris.clear();
        uint32_t num_edges = (uint32_t)et.edges.size();
        for (uint32_t ei = 0; ei < num_edges; ei++) {
            const auto& e = et.edges[ei];
            if (e.dead) continue;
            if (e.tri_a == INVALID || e.tri_b == INVALID) continue;

            // Skip if either tri was already rewritten by an earlier flip this pass.
            if (touched_tris.count(e.tri_a) || touched_tris.count(e.tri_b)) continue;

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
            int val_a = vertex_valence(et, va);
            int val_b = vertex_valence(et, vb);
            int val_c = vertex_valence(et, vc);
            int val_d = vertex_valence(et, vd);

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

            touched_tris.insert(ta);
            touched_tris.insert(tb);
            changed = true;
            ++total_flip;
        }
        pass++;
    }
    return total_flip;
}


// ---------------------------------------------------------------------------
// Compaction: remove degenerate tris and orphaned verts
// ---------------------------------------------------------------------------

static void compact_mesh(Mesh& m) {
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

    // Remap mirror_x_map through compaction
    if (!m.mirror_x_map.empty()) {
        std::vector<uint32_t> new_mirror(new_count, INVALID);
        for (uint32_t i = 0; i < vc; i++) {
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
    float snap_band = target_edge * 0.45f;   // how close to the plane counts as "should be on it"
    float max_edge  = target_edge * 0.6f;    // only snap across genuinely short seam edges (= `low`)
    float weld_tol  = target_edge * 0.0625f; // matches the GPU weld pass
    float weld_sq   = weld_tol * weld_tol;
    bool has_mask = !m.mask.empty();
    auto masked = [&](uint32_t v) -> bool {
        return has_mask && v < (uint32_t)m.mask.size() && m.mask[v] >= 1.0f;
    };

    m.build_adjacency();
    uint32_t vc = m.vertex_count();

    // Seam verts = exactly on x=0 (cut verts + pinned seam, post-clip).
    std::vector<uint8_t> at_seam(vc, 0);
    for (uint32_t v = 0; v < vc; v++)
        if (std::fabs(m.pos_x[v]) < seam_tol) at_seam[v] = 1;

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

    if (snapped == 0) {
        std::printf("[seam-consolidate] no near-seam verts to snap\n");
        return 0;
    }

    // Weld coincident seam verts in (y,z) within weld_tol. Brute-force over the
    // seam set only — small relative to vc.
    std::vector<uint32_t> seam_verts;
    for (uint32_t v = 0; v < vc; v++)
        if (at_seam[v]) seam_verts.push_back(v);

    std::vector<uint32_t> remap(vc);
    for (uint32_t v = 0; v < vc; v++) remap[v] = v;

    uint32_t welded = 0;
    for (size_t i = 0; i < seam_verts.size(); i++) {
        uint32_t a = seam_verts[i];
        if (remap[a] != a || masked(a)) continue;
        for (size_t j = i + 1; j < seam_verts.size(); j++) {
            uint32_t b = seam_verts[j];
            if (remap[b] != b || masked(b)) continue;
            float dy = m.pos_y[a] - m.pos_y[b];
            float dz = m.pos_z[a] - m.pos_z[b];
            if (dy*dy + dz*dz < weld_sq) { remap[b] = a; welded++; }
        }
    }
    for (uint32_t& idx : m.indices) idx = remap[idx];

    // Drop tris that welded to degenerate or collapsed onto the seam (all three
    // verts at x=0). Preserve fully-masked tris — they're carried across as-is.
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
        if (all_seam && !(masked(i0) && masked(i1) && masked(i2))) { culled++; continue; }
        kept.push_back(i0); kept.push_back(i1); kept.push_back(i2);
    }
    m.indices = std::move(kept);
    compact_mesh(m);

    std::printf("[seam-consolidate] snapped %u, welded %u, culled %u sliver tris\n",
                snapped, welded, culled);
    return snapped;
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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cs->remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    tc * 3 * sizeof(uint32_t), m.indices.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

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
                if (best != INVALID) {
                    masked_pair[v] = best;
                    masked_pair[best] = v;
                }
            }
            uint32_t n_paired = 0;
            for (uint32_t v = 0; v < vc; v++) if (masked_pair[v] != INVALID) n_paired++;
            std::printf("[mirror] masked spatial pairs: %u\n", n_paired / 2);
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

        std::vector<Vec3> new_pos(vc);
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
            n_moved++;
        }

        for (uint32_t v = 0; v < vc; v++) {
            if (!bad_vert[v]) continue;
            if (has_mask && v < (uint32_t)m.mask.size() && m.mask[v] >= 1.0f) continue;
            m.set_pos(v, new_pos[v]);
        }
        total_relaxed += n_moved;
        std::printf("[repair] iter %d: %u flipped tris, relaxed %u verts\n",
                    iter, flipped_count, n_moved);
    }

    if (total_relaxed > 0) m.recompute_normals();
    return total_relaxed;
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

RemeshResult perform_remesh(Mesh& mesh, MultiresStack& stack,
                            float target_edge_length, int iterations,
                            ComputeState* cs) {
    RemeshResult r;
    r.old_verts = mesh.vertex_count();
    r.old_tris  = mesh.tri_count();

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

    // Build edge table
    EdgeTable et;
    et.build(mesh);

    // Decide selection strategy: mask-driven or auto-detect.
    // tri_selected mirrors remesh_trisel_ssbo on GPU; we only readback when CPU
    // consumers (split/collapse/flip, counts) need it.
    bool using_mask = !mesh.mask.empty();
    std::vector<uint32_t> tri_selected;
    static constexpr int SUPPORT_RINGS = 2;

    // GPU grow_selection / mirror_selection need adjacency on GPU; upload before select.
    cs->upload_adjacency(mesh.vert_tri_offset.data(),
                         (uint32_t)mesh.vert_tri_offset.size(),
                         mesh.vert_tri_list.data(),
                         (uint32_t)mesh.vert_tri_list.size());

    if (using_mask) {
        cs->dispatch_select_unmasked(mesh.vertex_count(), mesh.tri_count(),
            mesh.indices.data(),
            mesh.mask.empty() ? nullptr : mesh.mask.data(),
            (uint32_t)mesh.mask.size());
        // Snapshot pre-grow selection for smooth_weights (GPU copy).
        cs->snapshot_core_sel(mesh.tri_count());
        cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
    } else {
        cs->dispatch_select_stretched(mesh.vertex_count(), mesh.tri_count(),
            mesh.indices.data(),
            mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
            target_edge_length);
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
    cs->dispatch_find_pinned(mesh.vertex_count(), mesh.tri_count(),
                             mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                             seam_tol, pinned);
    {
        uint32_t vc = mesh.vertex_count();
        for (uint32_t v = 0; v < vc; v++)
            if (pinned[v] && std::fabs(mesh.pos_x[v]) < seam_tol)
                mesh.pos_x[v] = 0.0f;
    }

    // Per-vertex smoothing weights (mask path only). core_sel already on GPU.
    std::vector<float> smooth_weights;
    bool weights_on_gpu = false;
    if (using_mask) {
        cs->dispatch_compute_smooth_weights(
            mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
        weights_on_gpu = true;
    }

    std::printf("[remesh] %u raw selected, %u total (after grow+support), "
                "target_edge=%.4f, %d iters\n",
                raw_selected, total_selected, target_edge_length, iterations);

    // Helper: full rebuild of all transient state after topology changes
    auto rebuild_all = [&]() {
        compact_mesh(mesh);
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
                mesh.mask.empty() ? nullptr : mesh.mask.data(),
                (uint32_t)mesh.mask.size());
            cs->snapshot_core_sel(mesh.tri_count());
            cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
        } else {
            cs->dispatch_select_stretched(mesh.vertex_count(), mesh.tri_count(),
                mesh.indices.data(),
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                target_edge_length);
            cs->dispatch_grow_selection(mesh.vertex_count(), mesh.tri_count(), 8);
        }

        // mirror_x_map was just cleared, so mirror spread is naturally a no-op
        // until the final mirror_positive_half pass — no dispatch needed here.

        cs->readback_trisel(mesh.tri_count(), tri_selected);

        cs->dispatch_find_pinned(mesh.vertex_count(), mesh.tri_count(),
                                 mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                                 seam_tol, pinned);
        uint32_t vc = mesh.vertex_count();
        for (uint32_t v = 0; v < vc; v++)
            if (pinned[v] && std::fabs(mesh.pos_x[v]) < seam_tol)
                mesh.pos_x[v] = 0.0f;
        if (using_mask) {
            cs->dispatch_compute_smooth_weights(
                mesh.vertex_count(), mesh.tri_count(), SUPPORT_RINGS);
            weights_on_gpu = true;
        }
    };

    // Flip changes indices but not vertex/tri count and not which verts belong
    // to the selection (flip is gated on both adjacent tris being selected,
    // and the four involved verts keep the same set of adjacent selected tris).
    // So pinned, weights, selection, mirror map, and core_selected are all
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
    for (int iter = 0; iter < iterations; iter++) {
        uint32_t n_split = split_long_edges(mesh, et, tri_selected, pinned, high, seam_tol);
        rebuild_all();

        uint32_t n_collapse = collapse_short_edges(mesh, et, tri_selected, pinned, low, seam_tol);
        et.build(mesh);
        rebuild_all();

        uint32_t n_flip = flip_edges(mesh, et, tri_selected, pinned);
        rebuild_after_flip();

        if (cs->has_remesh_smooth()) {
            cs->dispatch_remesh_smooth(
                mesh.vertex_count(), mesh.tri_count(),
                mesh.indices.data(),
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                mesh.norm_x.data(), mesh.norm_y.data(), mesh.norm_z.data(),
                smooth_weights, pinned,
                0.8f, seam_tol,
                mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data(),
                weights_on_gpu);
        }

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
    std::printf("[remesh] outer loop: %d/%d iters\n", iters_done, iterations);

    // Final compaction and rebuild
    compact_mesh(mesh);

    mirror_positive_half(mesh, seam_tol, target_edge_length, cs);

    mesh.build_adjacency();
    mesh.recompute_normals();

    // Replace multires stack
    stack.base          = mesh;
    stack.base_level    = 0;
    stack.current_level = 0;
    stack.disp.clear();
    stack.frames.clear();
    stack.mirror.clear();
    stack.base_mirror.clear();
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
