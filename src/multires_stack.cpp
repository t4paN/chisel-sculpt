#include "multires_stack.h"
#include "compute.h"   // optional GPU cascade replay (compute_cascade.cpp)
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Topology-derived mirror propagation
// ---------------------------------------------------------------------------

// loop_subdivide's Pass 4 emits 4 fine triangles per coarse face t with a
// fixed index layout at fine.indices[t*12 + 0..11]:
//   [1] = midpoint of coarse edge (a, b)
//   [4] = midpoint of coarse edge (b, c)
//   [7] = midpoint of coarse edge (c, a)
// This lets us recover the exact edge->midpoint mapping without re-running
// loop_subdivide's unordered hash table (whose iteration order is unspecified).
static void build_fine_mirror(
    uint32_t                        V_coarse,
    const std::vector<uint32_t>&    coarse_idx,
    const std::vector<uint32_t>&    coarse_mirror,
    uint32_t                        V_fine,
    const std::vector<uint32_t>&    fine_idx,
    std::vector<uint32_t>&          out)
{
    out.assign(V_fine, UINT32_MAX);

    // Old vertices: inherit mirror from coarse level.
    for (uint32_t i = 0; i < V_coarse; i++)
        out[i] = coarse_mirror[i];

    auto ekey = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) { uint32_t t = a; a = b; b = t; }
        return ((uint64_t)a << 32) | b;
    };

    // Pass 1: build edge -> fine midpoint map from the known Pass-4 layout.
    const uint32_t F_coarse = (uint32_t)(coarse_idx.size() / 3);
    std::unordered_map<uint64_t, uint32_t> edge_to_mid;
    edge_to_mid.reserve(V_fine - V_coarse);

    for (uint32_t t = 0; t < F_coarse; t++) {
        uint32_t a  = coarse_idx[t*3+0], b = coarse_idx[t*3+1], c = coarse_idx[t*3+2];
        edge_to_mid[ekey(a, b)] = fine_idx[t*12+1];
        edge_to_mid[ekey(b, c)] = fine_idx[t*12+4];
        edge_to_mid[ekey(c, a)] = fine_idx[t*12+7];
    }

    // Pass 2: for each midpoint, look up the mirror of its edge.
    for (uint32_t t = 0; t < F_coarse; t++) {
        uint32_t a  = coarse_idx[t*3+0], b = coarse_idx[t*3+1], c = coarse_idx[t*3+2];
        uint32_t ma = coarse_mirror[a],   mb = coarse_mirror[b],  mc = coarse_mirror[c];
        uint32_t ab = fine_idx[t*12+1],   bc = fine_idx[t*12+4],  ca = fine_idx[t*12+7];

        auto set = [&](uint32_t mid, uint32_t ea, uint32_t eb) {
            auto it = edge_to_mid.find(ekey(ea, eb));
            out[mid] = (it != edge_to_mid.end()) ? it->second : mid;
        };
        set(ab, ma, mb);
        set(bc, mb, mc);
        set(ca, mc, ma);
    }

    // Fallback: self-map any vertex that wasn't reached (degenerate topology).
    for (uint32_t i = 0; i < V_fine; i++)
        if (out[i] == UINT32_MAX) out[i] = i;

#ifndef NDEBUG
    for (uint32_t i = 0; i < V_fine; i++)
        assert(out[out[i]] == i);
#endif
}

// ---------------------------------------------------------------------------
// Finest-level paint planes (colour + mask)
// ---------------------------------------------------------------------------

// Vertex count of the mesh at level (base_level + k). disp[k] is sized per
// level, so the counts come free of any topology replay.
static uint32_t level_vcount(const MultiresStack& s, int k) {
    return k == 0 ? s.base.vertex_count() : (uint32_t)s.disp[k - 1].size();
}

// Vertex count at L_max — the size a non-empty paint plane must have.
static uint32_t plane_vcount(const MultiresStack& s) {
    return level_vcount(s, (int)s.disp.size());
}

// k such that level_vcount(s, k) == vc, or -1. V strictly grows per level, so
// the match is unique.
static int level_index_for_vcount(const MultiresStack& s, uint32_t vc) {
    for (int k = 0; k <= (int)s.disp.size(); k++)
        if (level_vcount(s, k) == vc) return k;
    return -1;
}

// Recover the edge->midpoint parent pairs from loop_subdivide's fixed Pass-4
// layout (same trick as build_fine_mirror above): fine_idx[t*12+1] is the
// midpoint of coarse edge (a,b), [+4] of (b,c), [+7] of (c,a).
static void build_parent_map(uint32_t                     V_coarse,
                             const std::vector<uint32_t>& coarse_idx,
                             const std::vector<uint32_t>& fine_idx,
                             uint32_t                     V_fine,
                             std::vector<uint32_t>&       out) {
    out.assign((size_t)(V_fine - V_coarse) * 2, UINT32_MAX);
    const uint32_t F = (uint32_t)(coarse_idx.size() / 3);
    for (uint32_t t = 0; t < F; t++) {
        uint32_t a = coarse_idx[t*3+0], b = coarse_idx[t*3+1], c = coarse_idx[t*3+2];
        uint32_t ab = fine_idx[t*12+1], bc = fine_idx[t*12+4], ca = fine_idx[t*12+7];
        out[(size_t)(ab - V_coarse)*2] = a; out[(size_t)(ab - V_coarse)*2+1] = b;
        out[(size_t)(bc - V_coarse)*2] = b; out[(size_t)(bc - V_coarse)*2+1] = c;
        out[(size_t)(ca - V_coarse)*2] = c; out[(size_t)(ca - V_coarse)*2+1] = a;
    }
}

// Make sure midpoint_parents[0 .. up_to_k) exist. Normally captured for free
// during cascade_to_level's replay; this fallback replays the subdivision from
// base for the gap case (e.g. paint sync right after load, before any cascade
// has walked the upper levels). One-shot — maps are cached until relock.
static void ensure_parent_maps(MultiresStack& s, int up_to_k) {
    bool complete = (int)s.midpoint_parents.size() >= up_to_k;
    for (int k = 0; complete && k < up_to_k; k++)
        if (s.midpoint_parents[k].empty()) complete = false;
    if (complete) return;

    Mesh m = s.base;
    for (int k = 0; k < up_to_k; k++) {
        uint32_t Vc = m.vertex_count();
        std::vector<uint32_t> cidx = m.indices;
        m.build_adjacency();
        m = loop_subdivide(m);
        if ((int)s.midpoint_parents.size() <= k) s.midpoint_parents.emplace_back();
        if (s.midpoint_parents[k].empty())
            build_parent_map(Vc, cidx, m.indices, m.vertex_count(), s.midpoint_parents[k]);
    }
}

// Midpoint-interpolate a plane's levels (k_from, k_to]: every midpoint at each
// level gets the average of its two parents. Parent maps must exist.
template <typename T, typename AVG>
static void interpolate_plane_up(std::vector<T>& plane, const MultiresStack& s,
                                 int k_from, int k_to, AVG avg) {
    for (int k = k_from; k < k_to; k++) {
        const std::vector<uint32_t>& par = s.midpoint_parents[k];
        const uint32_t Vc = level_vcount(s, k);
        const size_t   n  = par.size() / 2;
        for (size_t mi = 0; mi < n; mi++)
            plane[Vc + mi] = avg(plane[par[mi*2]], plane[par[mi*2 + 1]]);
    }
}

// Grow non-empty planes to V(L_max) when disp gained layers since the plane
// was last touched — new fine verts start as interpolation of the level below,
// exactly like disp zero-fill.
static void extend_planes(MultiresStack& s) {
    const uint32_t V_top = plane_vcount(s);
    const int      k_top = (int)s.disp.size();
    auto extend = [&](auto& plane, auto def, auto avg) {
        if (plane.empty() || plane.size() >= V_top) return;
        int k_from = level_index_for_vcount(s, (uint32_t)plane.size());
        if (k_from < 0) { plane.clear(); return; }   // stale size — drop, don't guess
        ensure_parent_maps(s, k_top);
        plane.resize(V_top, def);
        interpolate_plane_up(plane, s, k_from, k_top, avg);
    };
    extend(s.color, (uint32_t)0xFFFFFFFFu,
           [](uint32_t a, uint32_t b) { return color_avg(a, b); });
    extend(s.mask, 0.0f,
           [](float a, float b) { return 0.5f * (a + b); });
    extend(s.density, 0.5f,
           [](float a, float b) { return 0.5f * (a + b); });
}

// Diff the working array's prefix against the plane and re-interpolate only
// the descendants of changed verts — the linear pass + touched-region walk
// that makes coarse repaint cover smoothly while untouched fine detail
// survives exactly. Initialises the plane from the working array when empty.
// [paint-audit] fold telemetry: how many working verts differed from the plane
// and how many descendants got re-interpolated. A huge `changed` right after a
// small stroke = a stale working array being folded in (the jumble suspects).
struct PaintSyncCounts { uint32_t changed = 0; uint32_t propagated = 0; };

template <typename T, typename AVG>
static PaintSyncCounts sync_plane(std::vector<T>& plane, const std::vector<T>& work,
                                  uint32_t Vw, MultiresStack& s, int kw, T def, AVG avg) {
    const int      k_top = (int)s.disp.size();
    const uint32_t V_top = plane_vcount(s);

    if (plane.empty()) {
        ensure_parent_maps(s, k_top);
        plane.assign(V_top, def);
        for (uint32_t v = 0; v < Vw; v++)
            plane[v] = (v < (uint32_t)work.size()) ? work[v] : def;
        interpolate_plane_up(plane, s, kw, k_top, avg);
        return { Vw, V_top - Vw };
    }

    PaintSyncCounts counts;
    std::vector<uint8_t> changed(V_top, 0);
    for (uint32_t v = 0; v < Vw; v++) {
        T wv = (v < (uint32_t)work.size()) ? work[v] : def;
        if (!(plane[v] == wv)) { plane[v] = wv; changed[v] = 1; counts.changed++; }
    }
    if (!counts.changed) return counts;

    ensure_parent_maps(s, k_top);
    for (int k = kw; k < k_top; k++) {
        const std::vector<uint32_t>& par = s.midpoint_parents[k];
        const uint32_t Vc = level_vcount(s, k);
        const size_t   n  = par.size() / 2;
        for (size_t mi = 0; mi < n; mi++) {
            uint32_t p0 = par[mi*2], p1 = par[mi*2 + 1];
            if (changed[p0] | changed[p1]) {
                plane[Vc + mi]   = avg(plane[p0], plane[p1]);
                changed[Vc + mi] = 1;
                counts.propagated++;
            }
        }
    }
    return counts;
}

void multires_sync_paint(MultiresStack& stack, const Mesh& working) {
    if (!stack.locked) return;
    const uint32_t Vw = working.vertex_count();
    const int      kw = level_index_for_vcount(stack, Vw);
    if (kw < 0) {
        // working mesh is not a stack level — nothing to fold. Should never
        // fire mid-session; if it does, a fold got silently skipped.
        std::printf("[paint-audit] sync SKIP: Vw %u matches no stack level\n", Vw);
        return;
    }

    extend_planes(stack);  // disp may have grown since the planes were touched

    const bool color_init = stack.color.empty();
    PaintSyncCounts cc, mc;
    if (!working.color.empty())
        cc = sync_plane(stack.color, working.color, Vw, stack, kw, (uint32_t)0xFFFFFFFFu,
                        [](uint32_t a, uint32_t b) { return color_avg(a, b); });

    const bool mask_init = stack.mask.empty();
    bool mask_cleared = false;
    if (working.mask.empty()) {
        mask_cleared = !stack.mask.empty();
        stack.mask.clear();    // a cleared mask must not resurrect on cascade
    } else {
        mc = sync_plane(stack.mask, working.mask, Vw, stack, kw, 0.0f,
                        [](float a, float b) { return 0.5f * (a + b); });
    }

    // Density folds like colour: an empty working array means "feature unused",
    // never "cleared" (there is no density-clear op), so the plane is left alone.
    if (!working.density.empty())
        sync_plane(stack.density, working.density, Vw, stack, kw, 0.5f,
                   [](float a, float b) { return 0.5f * (a + b); });

    std::printf("[paint-audit] sync L%d (Vw %u): color %u changed / %u prop%s,"
                " mask %u / %u%s%s\n",
                stack.base_level + kw, Vw, cc.changed, cc.propagated,
                (color_init && !working.color.empty()) ? " (init)" : "",
                mc.changed, mc.propagated,
                (mask_init && !working.mask.empty()) ? " (init)" : "",
                mask_cleared ? " (plane cleared)" : "");
}

void multires_stack_init_from_lock(MultiresStack& stack,
                                   const Mesh& locked_mesh,
                                   int icosphere_level) {
    stack.base = locked_mesh;
    stack.base_level = icosphere_level;
    stack.current_level = icosphere_level;
    stack.disp.clear();
    stack.frames.clear();
    stack.mirror.clear();
    stack.midpoint_parents.clear();
    stack.topo_cache.clear();   // pure function of base topology — new base, new chain
    stack.lock_stamp = 0;       // new chain identity; GPU cascade tables re-key lazily
    stack.locked = true;

    // Paint lives in the finest-level planes, not in base — at lock the
    // whole working array IS the prefix. Strip base's copies so cascades
    // don't carry a second, stale source of paint.
    stack.color = locked_mesh.color;
    if (!stack.color.empty()) stack.color.resize(locked_mesh.vertex_count(), 0xFFFFFFFFu);
    stack.mask = locked_mesh.mask;
    if (!stack.mask.empty()) stack.mask.resize(locked_mesh.vertex_count(), 0.0f);
    stack.density = locked_mesh.density;
    if (!stack.density.empty()) stack.density.resize(locked_mesh.vertex_count(), 0.5f);
    stack.base.color.clear();
    stack.base.mask.clear();
    stack.base.density.clear();

    build_mirror_spatial(stack.base, stack.base_mirror);

    uint32_t V = (uint32_t)stack.base_mirror.size();
    uint32_t S = 0;
    for (uint32_t i = 0; i < V; i++)
        if (stack.base_mirror[i] == i) S++;
    std::printf("[mirror] base_mirror: %u verts, %u self-mapped, %u pairs\n",
                V, S, (V - S) / 2);
}

void compute_frames(const Mesh& m, std::vector<Frame>& out) {
    const uint32_t vc = m.vertex_count();
    out.resize(vc);

    for (uint32_t v = 0; v < vc; v++) {
        Vec3 n = m.get_normal(v).normalized();

        // Find lowest-indexed neighbor via CSR adjacency for deterministic tangent
        uint32_t best_nbr = UINT32_MAX;
        uint32_t start = m.vert_tri_offset[v];
        uint32_t end   = m.vert_tri_offset[v + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t t  = m.vert_tri_list[j];
            uint32_t i0 = m.indices[t*3+0];
            uint32_t i1 = m.indices[t*3+1];
            uint32_t i2 = m.indices[t*3+2];
            if (i0 != v && i0 < best_nbr) best_nbr = i0;
            if (i1 != v && i1 < best_nbr) best_nbr = i1;
            if (i2 != v && i2 < best_nbr) best_nbr = i2;
        }

        Vec3 t_raw = (best_nbr != UINT32_MAX)
            ? m.get_pos(best_nbr) - m.get_pos(v)
            : Vec3{1, 0, 0};

        // Project onto tangent plane
        float dn = t_raw.dot(n);
        Vec3 t_proj = { t_raw.x - n.x * dn,
                        t_raw.y - n.y * dn,
                        t_raw.z - n.z * dn };

        float t_len = t_proj.length();
        if (t_len < 1e-6f) {
            // Axis-aligned fallback: pick world axis most perpendicular to n
            float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
            Vec3 axis = (ax <= ay && ax <= az) ? Vec3{1,0,0}
                      : (ay <= az)             ? Vec3{0,1,0}
                                               : Vec3{0,0,1};
            float d2 = axis.dot(n);
            t_proj = { axis.x - n.x*d2, axis.y - n.y*d2, axis.z - n.z*d2 };
            t_len  = t_proj.length();
        }

        Vec3 t_final = { t_proj.x / t_len, t_proj.y / t_len, t_proj.z / t_len };
        Vec3 b_final = n.cross(t_final).normalized();

        out[v] = { t_final, b_final, n };
    }
}

// Slow replay: full subdivision chain from base. Warms the topology cache as
// it goes; still the only path for a cold cache (and the truth reference for
// the fast path under CHISEL_DEBUG_MULTIRES).
static void cascade_replay_slow(MultiresStack& stack, Mesh& out, int passes) {
    Mesh m = stack.base;

    for (int i = 0; i < passes; i++) {
        // Save coarse topology before subdivision for mirror propagation.
        uint32_t V_coarse = m.vertex_count();
        std::vector<uint32_t> coarse_idx = m.indices;

        // Topology-cache warm: capture the stencil + fine topology while the
        // slow replay computes them anyway. ready entries are consumed by the
        // fast path; a miss here costs one extra build_adjacency at most.
        if ((int)stack.topo_cache.size() <= i) stack.topo_cache.emplace_back();
        MultiresStack::LevelTopo& tc = stack.topo_cache[i];
        const bool tc_miss = !tc.ready;

        m.build_adjacency();        // loop_subdivide reads CSR for vertex relocation
        m = loop_subdivide(m, false, tc_miss ? &tc.stencil : nullptr);
        if (tc_miss) {
            m.build_adjacency();
            tc.indices   = m.indices;
            tc.vt_offset = m.vert_tri_offset;
            tc.vt_list   = m.vert_tri_list;
            tc.vcount    = m.vertex_count();
            tc.ready     = true;
        }

        if ((int)stack.disp.size() <= i)
            stack.disp.emplace_back(m.vertex_count(), Vec3{0.0f, 0.0f, 0.0f});
        if ((int)stack.frames.size() <= i)
            stack.frames.emplace_back();

        // Compute local frames from the pre-displacement surface at this level.
        // m right now is the subdivided base before any displacements — exactly right.
        if (stack.frames[i].empty()) {
            if (!tc_miss) m.build_adjacency();  // fresh already on a cache miss
            m.recompute_normals();  // frame normals from subdivided base
            compute_frames(m, stack.frames[i]);
        }

        // Lazy midpoint->parents capture for the paint planes (topology-only,
        // recovered from the Pass-4 layout — free while we're replaying anyway).
        if ((int)stack.midpoint_parents.size() <= i) stack.midpoint_parents.emplace_back();
        if (stack.midpoint_parents[i].empty())
            build_parent_map(V_coarse, coarse_idx, m.indices,
                             m.vertex_count(), stack.midpoint_parents[i]);

        // Lazy topology mirror propagation.
        if ((int)stack.mirror.size() <= i) stack.mirror.emplace_back();
        if (stack.mirror[i].empty() && !stack.base_mirror.empty()) {
            const std::vector<uint32_t>& cm =
                (i == 0) ? stack.base_mirror : stack.mirror[i - 1];
            if (!cm.empty())
                build_fine_mirror(V_coarse, coarse_idx, cm,
                                  m.vertex_count(), m.indices, stack.mirror[i]);
        }

        // Apply displacements: local frame -> world
        const uint32_t vc = m.vertex_count();
        for (uint32_t v = 0; v < vc; v++) {
            const Frame& f = stack.frames[i][v];
            const Vec3&  d = stack.disp[i][v];
            m.set_pos(v, m.get_pos(v) + f.t * d.x + f.b * d.y + f.n * d.z);
        }
    }

    m.recompute_normals();
    m.build_adjacency();    // CSR for post-switch brush ops
    out = std::move(m);
}

// Positions-only replay of one cached subdivision pass. Mirrors loop_subdivide
// Pass 3 (relocation) and Pass 2 (midpoints) — same expressions, same iteration
// order, so the output is bit-identical to a real subdivide — reading topology
// from the cache instead of re-extracting edges through the hash map.
static void apply_stencil_positions(
        const std::vector<float>& cpx, const std::vector<float>& cpy,
        const std::vector<float>& cpz,
        const std::vector<uint32_t>& c_indices,
        const std::vector<uint32_t>& c_off, const std::vector<uint32_t>& c_list,
        const MultiresStack::LevelTopo& tc,
        std::vector<float>& fpx, std::vector<float>& fpy, std::vector<float>& fpz)
{
    const uint32_t Vc = (uint32_t)cpx.size();
    fpx.resize(tc.vcount); fpy.resize(tc.vcount); fpz.resize(tc.vcount);

    auto cpos = [&](uint32_t i) -> Vec3 { return {cpx[i], cpy[i], cpz[i]}; };
    auto fset = [&](uint32_t i, Vec3 p) { fpx[i]=p.x; fpy[i]=p.y; fpz[i]=p.z; };

    // Relocate original verts with Loop weights (loop_subdivide Pass 3).
    const bool has_bnd = !tc.stencil.is_bnd.empty();
    for (uint32_t i = 0; i < Vc; i++) {
        Vec3 p = cpos(i);
        Vec3 new_pos;
        if (has_bnd && tc.stencil.is_bnd[i]) {
            Vec3 pb0 = cpos(tc.stencil.bnd_a[i]);
            Vec3 pb1 = cpos(tc.stencil.bnd_b[i]);
            new_pos = p * 0.75f + (pb0 + pb1) * (1.0f/8.0f);
        } else {
            uint32_t n = c_off[i+1] - c_off[i];
            float beta = (n == 3) ? (3.0f/16.0f) : (3.0f / (8.0f * (float)n));
            Vec3 nbr_sum = {0, 0, 0};
            for (uint32_t tj = c_off[i]; tj < c_off[i+1]; tj++) {
                uint32_t tri = c_list[tj];
                uint32_t a = c_indices[tri*3+0];
                uint32_t b = c_indices[tri*3+1];
                uint32_t c = c_indices[tri*3+2];
                if      (a == i) { nbr_sum += cpos(b); nbr_sum += cpos(c); }
                else if (b == i) { nbr_sum += cpos(a); nbr_sum += cpos(c); }
                else             { nbr_sum += cpos(a); nbr_sum += cpos(b); }
            }
            new_pos = p * (1.0f - (float)n * beta) + nbr_sum * (beta * 0.5f);
        }
        fset(i, new_pos);
    }

    // Midpoints, in canonical numbering order (loop_subdivide Pass 2).
    const uint32_t E = (uint32_t)(tc.stencil.mid.size() / 4);
    for (uint32_t e = 0; e < E; e++) {
        const uint32_t* s = &tc.stencil.mid[(size_t)e * 4];
        Vec3 p0 = cpos(s[0]);
        Vec3 p1 = cpos(s[1]);
        Vec3 pos;
        if (s[3] != UINT32_MAX) {
            Vec3 p2 = cpos(s[2]);
            Vec3 p3 = cpos(s[3]);
            pos = (p0 + p1) * (3.0f/8.0f) + (p2 + p3) * (1.0f/8.0f);
        } else {
            pos = (p0 + p1) * 0.5f;
        }
        fset(Vc + e, pos);
    }
}

// Fast replay: positions-only cascade over the warmed topology cache — no
// loop_subdivide, no build_adjacency, no hash maps. Requires topo_cache[0..
// passes-1].ready (caller checks). Lazy structures (disp, frames, parent maps,
// mirrors) fill exactly like the slow path; the frames rebuild after a
// projection is the one case that still materializes a temp mesh (for normals).
static void cascade_replay_fast(MultiresStack& stack, Mesh& out, int passes) {
    if (stack.base.vert_tri_offset.size() != stack.base.vertex_count() + 1)
        stack.base.build_adjacency();   // base immutable after lock — build once

    std::vector<float> ax = stack.base.pos_x, ay = stack.base.pos_y,
                       az = stack.base.pos_z;
    std::vector<float> bx, by, bz;
    const std::vector<uint32_t>* c_idx  = &stack.base.indices;
    const std::vector<uint32_t>* c_off  = &stack.base.vert_tri_offset;
    const std::vector<uint32_t>* c_list = &stack.base.vert_tri_list;

    for (int i = 0; i < passes; i++) {
        MultiresStack::LevelTopo& tc = stack.topo_cache[i];
        const uint32_t V_coarse = (uint32_t)ax.size();

        apply_stencil_positions(ax, ay, az, *c_idx, *c_off, *c_list,
                                tc, bx, by, bz);

        if ((int)stack.disp.size() <= i)
            stack.disp.emplace_back(tc.vcount, Vec3{0.0f, 0.0f, 0.0f});
        if ((int)stack.frames.size() <= i)
            stack.frames.emplace_back();

        // Frames stale (post-projection): rebuild from the pre-displacement
        // surface. compute_frames needs a Mesh with normals + CSR — borrow the
        // cached topology into a temp. Rare; the steady state skips this.
        if (stack.frames[i].empty()) {
            Mesh tmp;
            tmp.pos_x = bx; tmp.pos_y = by; tmp.pos_z = bz;
            tmp.norm_x.resize(tc.vcount); tmp.norm_y.resize(tc.vcount);
            tmp.norm_z.resize(tc.vcount);
            tmp.indices = tc.indices;
            tmp.vert_tri_offset = tc.vt_offset;
            tmp.vert_tri_list   = tc.vt_list;
            tmp.recompute_normals();
            compute_frames(tmp, stack.frames[i]);
        }

        if ((int)stack.midpoint_parents.size() <= i) stack.midpoint_parents.emplace_back();
        if (stack.midpoint_parents[i].empty())
            build_parent_map(V_coarse, *c_idx, tc.indices,
                             tc.vcount, stack.midpoint_parents[i]);

        if ((int)stack.mirror.size() <= i) stack.mirror.emplace_back();
        if (stack.mirror[i].empty() && !stack.base_mirror.empty()) {
            const std::vector<uint32_t>& cm =
                (i == 0) ? stack.base_mirror : stack.mirror[i - 1];
            if (!cm.empty())
                build_fine_mirror(V_coarse, *c_idx, cm,
                                  tc.vcount, tc.indices, stack.mirror[i]);
        }

        // Apply displacements: local frame -> world (same expression as slow).
        for (uint32_t v = 0; v < tc.vcount; v++) {
            const Frame& f = stack.frames[i][v];
            const Vec3&  d = stack.disp[i][v];
            Vec3 p = Vec3{bx[v], by[v], bz[v]} + f.t * d.x + f.b * d.y + f.n * d.z;
            bx[v] = p.x; by[v] = p.y; bz[v] = p.z;
        }

        std::swap(ax, bx); std::swap(ay, by); std::swap(az, bz);
        c_idx  = &tc.indices;
        c_off  = &tc.vt_offset;
        c_list = &tc.vt_list;
    }

    const MultiresStack::LevelTopo& top = stack.topo_cache[passes - 1];
    out = Mesh();
    out.pos_x = std::move(ax); out.pos_y = std::move(ay); out.pos_z = std::move(az);
    out.norm_x.resize(top.vcount); out.norm_y.resize(top.vcount);
    out.norm_z.resize(top.vcount);
    out.indices         = top.indices;
    out.vert_tri_offset = top.vt_offset;
    out.vert_tri_list   = top.vt_list;
    out.topo_version    = 1;   // fresh mesh; mirror stamp is synced by the caller
    out.recompute_normals();
}

// GPU replay: same contract as cascade_replay_fast, positions + normals via the
// compute kernels (compute_cascade.cpp) with one readback. Handles the same
// lazy structure fills — but only the topology-only ones; a stale frames layer
// (post-projection) needs pre-displacement positions the GPU flow doesn't keep
// on the CPU, so that one switch falls back to the CPU fast path (which
// rebuilds + caches the frames; the GPU path resumes on the next switch).
// Returns false with `out` untouched when ineligible — caller falls back.
static bool cascade_replay_gpu(MultiresStack& stack, ComputeState& cs,
                               Mesh& out, int passes) {
    if (!cs.has_cascade()) return false;

    for (int i = 0; i < passes; i++) {
        MultiresStack::LevelTopo& tc = stack.topo_cache[i];
        if (!tc.stencil.is_bnd.empty()) {
            std::printf("[cascade-gpu] fallback: open mesh (boundary stencil at pass %d)\n", i);
            return false;   // open mesh — CPU keeps it
        }
        const uint32_t V_coarse = (i == 0) ? stack.base.vertex_count()
                                           : stack.topo_cache[i - 1].vcount;
        const std::vector<uint32_t>& c_idx =
            (i == 0) ? stack.base.indices : stack.topo_cache[i - 1].indices;

        if ((int)stack.disp.size() <= i)
            stack.disp.emplace_back(tc.vcount, Vec3{0.0f, 0.0f, 0.0f});
        if ((int)stack.frames.size() <= i)
            stack.frames.emplace_back();
        // Stale frames (a sculpt below cleared them) are NOT a bail: the GPU
        // replay rebuilds them on the GPU from the pre-disp surface and reads
        // the target level's back for the brush residency (compute_cascade.cpp).
        if (stack.disp[i].size() != (size_t)tc.vcount) {
            std::printf("[cascade-gpu] fallback: disp[%d] size %zu != %u\n",
                        i, stack.disp[i].size(), tc.vcount);
            return false;
        }

        if ((int)stack.midpoint_parents.size() <= i) stack.midpoint_parents.emplace_back();
        if (stack.midpoint_parents[i].empty())
            build_parent_map(V_coarse, c_idx, tc.indices,
                             tc.vcount, stack.midpoint_parents[i]);

        if ((int)stack.mirror.size() <= i) stack.mirror.emplace_back();
        if (stack.mirror[i].empty() && !stack.base_mirror.empty()) {
            const std::vector<uint32_t>& cm =
                (i == 0) ? stack.base_mirror : stack.mirror[i - 1];
            if (!cm.empty())
                build_fine_mirror(V_coarse, c_idx, cm,
                                  tc.vcount, tc.indices, stack.mirror[i]);
        }
    }

    Mesh fresh;
    if (!cs.gpu_cascade_replay(stack, fresh, passes)) return false;

    const MultiresStack::LevelTopo& top = stack.topo_cache[passes - 1];
    fresh.indices         = top.indices;
    fresh.vert_tri_offset = top.vt_offset;
    fresh.vert_tri_list   = top.vt_list;
    fresh.topo_version    = 1;   // fresh mesh; mirror stamp is synced by the caller
    out = std::move(fresh);
    return true;
}

void cascade_to_level(MultiresStack& stack, Mesh& out, int K, ComputeState* compute) {
    const int passes = K - stack.base_level;

    // A CPU replay leaves the GPU cascade scratch holding a previous switch's
    // output — make sure no one copies it (gpu_cascade_replay re-arms on success).
    if (compute) compute->cascade_out_valid = false;

    bool fast = passes > 0 && (int)stack.topo_cache.size() >= passes;
    for (int i = 0; fast && i < passes; i++)
        if (!stack.topo_cache[i].ready) fast = false;

    const auto t0 = std::chrono::steady_clock::now();
    const char* mode = "slow";
    if (fast) {
        bool gpu_done = false;
        if (compute && compute->supported)
            gpu_done = cascade_replay_gpu(stack, *compute, out, passes);
        if (gpu_done) {
            mode = "gpu";
#ifdef CHISEL_DEBUG_MULTIRES
            {
                // Truth check: the CPU fast replay must agree to float rounding.
                // Not bit-exact — the kernels repeat the same per-thread
                // accumulation order but GPU FMA contraction rounds differently.
                Mesh chk;
                cascade_replay_fast(stack, chk, passes);
                const bool topo_ok = chk.indices == out.indices
                                  && chk.vert_tri_offset == out.vert_tri_offset
                                  && chk.vert_tri_list   == out.vert_tri_list;
                uint32_t bad = 0; float maxd = 0.0f;
                for (uint32_t v = 0; v < chk.vertex_count(); v++) {
                    float dx = std::fabs(chk.pos_x[v] - out.pos_x[v]);
                    float dy = std::fabs(chk.pos_y[v] - out.pos_y[v]);
                    float dz = std::fabs(chk.pos_z[v] - out.pos_z[v]);
                    float dm = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
                    if (dm > 1e-5f) bad++;
                    if (dm > maxd) maxd = dm;
                }
                std::printf("[cascade-check] L%d gpu vs fast: topo %s, %u/%u pos beyond 1e-5, max |d| %.3g\n",
                            K, topo_ok ? "OK" : "MISMATCH",
                            bad, chk.vertex_count(), (double)maxd);
            }
#endif
        } else {
            mode = "fast";
            cascade_replay_fast(stack, out, passes);
#ifdef CHISEL_DEBUG_MULTIRES
            {
                // Truth check: the slow replay must agree bit-for-bit — the fast
                // path repeats its arithmetic in the same order.
                Mesh chk;
                cascade_replay_slow(stack, chk, passes);
                const bool topo_ok = chk.indices == out.indices
                                  && chk.vert_tri_offset == out.vert_tri_offset
                                  && chk.vert_tri_list   == out.vert_tri_list;
                uint32_t bad = 0; float maxd = 0.0f;
                for (uint32_t v = 0; v < chk.vertex_count(); v++) {
                    float dx = std::fabs(chk.pos_x[v] - out.pos_x[v]);
                    float dy = std::fabs(chk.pos_y[v] - out.pos_y[v]);
                    float dz = std::fabs(chk.pos_z[v] - out.pos_z[v]);
                    float dm = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
                    if (dm > 0.0f) bad++;
                    if (dm > maxd) maxd = dm;
                }
                std::printf("[cascade-check] L%d fast vs slow: topo %s, %u/%u pos differ, max |d| %.3g\n",
                            K, topo_ok ? "OK" : "MISMATCH",
                            bad, chk.vertex_count(), (double)maxd);
            }
#endif
        }
    } else {
        cascade_replay_slow(stack, out, passes);
    }
    const double cascade_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[cascade] L%d %s %.1f ms\n", K, mode, cascade_ms);

    // Paint rides the finest-level planes: the working array at level K is the
    // [0, V_K) prefix. Extend first — the pass loop above may have grown disp.
    extend_planes(stack);
    const uint32_t V_K = out.vertex_count();
    if (!stack.color.empty())
        out.color.assign(stack.color.begin(), stack.color.begin() + V_K);
    else
        out.color.clear();
    if (!stack.mask.empty())
        out.mask.assign(stack.mask.begin(), stack.mask.begin() + V_K);
    else
        out.mask.clear();
    if (!stack.density.empty())
        out.density.assign(stack.density.begin(), stack.density.begin() + V_K);
    else
        out.density.clear();

    // [paint-audit] a non-empty plane must be exactly V(L_max) — anything else
    // means the prefix handed to the working mesh is misaligned.
    std::printf("[paint-audit] cascade L%d: V_K %u, color plane %zu/%u, mask %zu/%u\n",
                K, V_K, stack.color.size(), plane_vcount(stack),
                stack.mask.size(), plane_vcount(stack));

    // Populate topology mirror map in the output mesh. Sync the topo stamp so
    // the persistent-map gate in refresh_mirror_map treats the cached map as
    // current (a rebuild from sculpted positions would reclassify drifted verts).
    if (K == stack.base_level) {
        if (!stack.base_mirror.empty()) {
            out.mirror_x_map = stack.base_mirror;
            out.mirror_topo_version = out.topo_version;
        }
    } else {
        int k = K - stack.base_level - 1;
        if (k < (int)stack.mirror.size() && !stack.mirror[k].empty()) {
            out.mirror_x_map = stack.mirror[k];
            out.mirror_topo_version = out.topo_version;
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy-numbering migration (v<=3 project files)
// ---------------------------------------------------------------------------
// v<=3 files store disp[k] indexed under the midpoint numbering that this
// platform's std::unordered_map iteration produced at save time (see the
// legacy_numbering note in mesh.h). Replay the cascade under that numbering,
// validate the replay against the current-level surface cached in the file,
// and re-encode every layer under the canonical numbering. Returns false when
// the replay does not match the cached surface — the file was saved under a
// different stdlib (another platform); its stack cannot be decoded here.
bool migrate_legacy_numbering(MultiresStack& stack, const Mesh& cached_surface) {
    if (stack.disp.empty()) return true;

    Mesh m_leg = stack.base;   // legacy chain (displaced, legacy numbering)
    Mesh m_can = stack.base;   // canonical chain (displaced, canonical numbering)

    const int layers = (int)stack.disp.size();
    const int cur_k  = stack.current_level - stack.base_level - 1;
    // Cached surface at base level: the base is stored verbatim, nothing to
    // validate against. Above the stored layers: nothing to validate WITH.
    if (cur_k >= layers) return false;
    bool validated = (cur_k < 0);

    std::vector<std::vector<Vec3>>  new_disp(layers);
    std::vector<std::vector<Frame>> new_frames(layers);

    for (int k = 0; k < layers; k++) {
        // Legacy replay — exactly the sequence the v<=3 cascade ran.
        m_leg.build_adjacency();
        Mesh sub_leg = loop_subdivide(m_leg, /*legacy_numbering=*/true);
        sub_leg.build_adjacency();
        sub_leg.recompute_normals();
        std::vector<Frame> fr_leg;
        compute_frames(sub_leg, fr_leg);

        // Canonical twin of the same level.
        m_can.build_adjacency();
        Mesh sub_can = loop_subdivide(m_can, /*legacy_numbering=*/false);
        sub_can.build_adjacency();
        sub_can.recompute_normals();
        compute_frames(sub_can, new_frames[k]);

        // Fine-level permutation: Pass 4 emits sub-faces in coarse-face order on
        // both chains, so the parallel index arrays pair every legacy id with
        // its canonical id.
        const uint32_t vc = sub_leg.vertex_count();
        if (sub_can.vertex_count() != vc ||
            sub_can.indices.size() != sub_leg.indices.size()) return false;
        std::vector<uint32_t> perm(vc, UINT32_MAX);
        for (size_t j = 0; j < sub_leg.indices.size(); j++) {
            uint32_t l = sub_leg.indices[j], c = sub_can.indices[j];
            if (perm[l] == UINT32_MAX) perm[l] = c;
            else if (perm[l] != c) return false;    // chains not parallel — bail
        }

        const std::vector<Vec3>& disp_leg = stack.disp[k];
        if (disp_leg.size() != vc) return false;

        // Displace the legacy chain with the stored layer.
        for (uint32_t v = 0; v < vc; v++) {
            const Frame& f = fr_leg[v];
            const Vec3&  d = disp_leg[v];
            sub_leg.set_pos(v, sub_leg.get_pos(v) + f.t * d.x + f.b * d.y + f.n * d.z);
        }

        // Validate the replay at the level the file cached its surface for.
        if (k == cur_k) {
            if (cached_surface.vertex_count() != vc) return false;
            Vec3 lo = cached_surface.get_pos(0), hi = lo;
            for (uint32_t v = 1; v < vc; v++) {
                Vec3 p = cached_surface.get_pos(v);
                lo = { std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z) };
                hi = { std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z) };
            }
            float diag = (hi - lo).length();
            float max_d = 0.0f;
            for (uint32_t v = 0; v < vc; v++)
                max_d = std::max(max_d, (sub_leg.get_pos(v) - cached_surface.get_pos(v)).length());
            // GPU-materialized save vs CPU replay leaves ~1e-6 relative slack;
            // a numbering mismatch scrambles at edge-length scale (>1e-1).
            if (!(max_d <= 1e-3f * diag)) {
                std::printf("[migrate] legacy replay mismatch: max_d=%g diag=%g\n",
                            max_d, diag);
                return false;
            }
            validated = true;
        }

        // Re-encode: world delta against the canonical pre-displacement surface,
        // decomposed in the canonical frame; then advance the canonical chain by
        // exactly what a future cascade will reproduce.
        new_disp[k].resize(vc);
        for (uint32_t v = 0; v < vc; v++) {
            uint32_t c = perm[v];
            Vec3 delta = sub_leg.get_pos(v) - sub_can.get_pos(c);
            const Frame& f = new_frames[k][c];
            new_disp[k][c] = { delta.dot(f.t), delta.dot(f.b), delta.dot(f.n) };
        }
        for (uint32_t v = 0; v < vc; v++) {
            const Frame& f = new_frames[k][v];
            const Vec3&  d = new_disp[k][v];
            sub_can.set_pos(v, sub_can.get_pos(v) + f.t * d.x + f.b * d.y + f.n * d.z);
        }

        m_leg = std::move(sub_leg);
        m_can = std::move(sub_can);
    }

    if (!validated) return false;

    stack.disp   = std::move(new_disp);
    stack.frames = std::move(new_frames);
    for (auto& mv : stack.mirror) mv.clear();   // lazily rebuilt from base_mirror
    return true;
}

// ---------------------------------------------------------------------------
// Projection (Chunk 4a)
// ---------------------------------------------------------------------------

size_t MultiresSnapshot::bytes() const {
    size_t n = 0;
    n += (base_px.size() + base_py.size() + base_pz.size()) * sizeof(float);
    for (const auto& d : disp)   n += d.size() * sizeof(Vec3);
    for (const auto& f : frames) n += f.size() * sizeof(Frame);
    return n;
}

static int projection_k_start(const MultiresStack& s, int target_level) {
    int k = target_level - s.base_level - 1;
    return k < 0 ? 0 : k;
}

void capture_projection_snapshot(const MultiresStack& stack,
                                 int target_level,
                                 MultiresSnapshot& out) {
    out = MultiresSnapshot{};
    if (target_level == stack.base_level) {
        out.has_base = true;
        out.base_px  = stack.base.pos_x;
        out.base_py  = stack.base.pos_y;
        out.base_pz  = stack.base.pos_z;
    }
    int k_start = projection_k_start(stack, target_level);
    out.disp_start = k_start;
    for (int k = k_start; k < (int)stack.disp.size(); k++) {
        out.disp.push_back(stack.disp[k]);
        if (k < (int)stack.frames.size()) out.frames.push_back(stack.frames[k]);
        else                               out.frames.emplace_back();
    }
}

void restore_projection_snapshot(MultiresStack& stack,
                                 const MultiresSnapshot& snap) {
    if (snap.has_base) {
        stack.base.pos_x = snap.base_px;
        stack.base.pos_y = snap.base_py;
        stack.base.pos_z = snap.base_pz;
    }
    for (size_t i = 0; i < snap.disp.size(); i++) {
        int k = snap.disp_start + (int)i;
        if (k >= (int)stack.disp.size()) stack.disp.emplace_back();
        stack.disp[k] = snap.disp[i];
        if (k >= (int)stack.frames.size()) stack.frames.emplace_back();
        stack.frames[k] = snap.frames[i];
    }
    // Clear mirror caches for restored layers so they rebuild lazily on cascade.
    if (snap.has_base) {
        for (auto& mv : stack.mirror) mv.clear();
    } else {
        for (int k = snap.disp_start; k < (int)stack.mirror.size(); k++)
            stack.mirror[k].clear();
    }
}

// One-step inverse Loop: given the fine-level positions (vertex-point indices
// coincide with coarse indices — verified: loop_subdivide writes updated coarse
// verts to [0, V_coarse) and new edge midpoints to [V_coarse, V_fine)) and the
// coarse-level topology in coarse_mesh, solve for coarse positions.
//
// Warren's weights:   beta = 3/(8n) for n > 3, 3/16 for n == 3.
// Forward relation:   fine_v = (1 - n*beta) * coarse_v + beta * sum_unique_neighbors(coarse)
// Replacing coarse_v_i with fine_v_i (mathematically sufficient per Warren/Weiss):
//   coarse_v = ( fine_v - beta * sum_unique(fine_neighbors) ) / (1 - n*beta)
static void inverse_loop_one_step(const Mesh& coarse_mesh,
                                  const Mesh& fine_mesh,
                                  std::vector<Vec3>& new_coarse_pos) {
    const uint32_t V = coarse_mesh.vertex_count();
    new_coarse_pos.assign(V, Vec3{0, 0, 0});

    for (uint32_t v = 0; v < V; v++) {
        uint32_t start = coarse_mesh.vert_tri_offset[v];
        uint32_t end   = coarse_mesh.vert_tri_offset[v + 1];
        uint32_t n     = end - start;  // valence for closed manifold interior

        // Each distinct neighbor appears in exactly 2 incident tris, so the
        // raw sum over tris double-counts. Same trick loop_subdivide uses
        // forward — mirrored here.
        Vec3 nbr_sum_doubled = {0, 0, 0};
        for (uint32_t tj = start; tj < end; tj++) {
            uint32_t tri = coarse_mesh.vert_tri_list[tj];
            uint32_t a   = coarse_mesh.indices[tri*3+0];
            uint32_t b   = coarse_mesh.indices[tri*3+1];
            uint32_t c   = coarse_mesh.indices[tri*3+2];
            if      (a == v) { nbr_sum_doubled += fine_mesh.get_pos(b);
                               nbr_sum_doubled += fine_mesh.get_pos(c); }
            else if (b == v) { nbr_sum_doubled += fine_mesh.get_pos(a);
                               nbr_sum_doubled += fine_mesh.get_pos(c); }
            else             { nbr_sum_doubled += fine_mesh.get_pos(a);
                               nbr_sum_doubled += fine_mesh.get_pos(b); }
        }
        float beta  = (n == 3) ? (3.0f / 16.0f) : (3.0f / (8.0f * (float)n));
        float denom = 1.0f - (float)n * beta;    // 0.625 for n=5 or n=6
        Vec3  fine_v        = fine_mesh.get_pos(v);
        Vec3  nbr_sum_unique = nbr_sum_doubled * 0.5f;
        Vec3  num = fine_v - nbr_sum_unique * beta;
        new_coarse_pos[v] = num * (1.0f / denom);
    }
}

// One subdivision step for the projection paths: cached stencil replay when
// warm (bit-identical to a real subdivide, no edge extraction), real
// loop_subdivide otherwise — warming the cache exactly like cascade does.
// Returns the fine mesh with positions + indices + CSR; normals are zeroed —
// callers recompute them only where actually consumed (compute_frames).
static Mesh subdiv_step_cached(MultiresStack& stack, Mesh& coarse, int i) {
    if ((int)stack.topo_cache.size() <= i) stack.topo_cache.emplace_back();
    MultiresStack::LevelTopo& tc = stack.topo_cache[i];
    Mesh fine;
    if (tc.ready) {
        if (coarse.vert_tri_offset.size() != coarse.vertex_count() + 1)
            coarse.build_adjacency();
        apply_stencil_positions(coarse.pos_x, coarse.pos_y, coarse.pos_z,
                                coarse.indices, coarse.vert_tri_offset,
                                coarse.vert_tri_list, tc,
                                fine.pos_x, fine.pos_y, fine.pos_z);
        fine.indices         = tc.indices;
        fine.vert_tri_offset = tc.vt_offset;
        fine.vert_tri_list   = tc.vt_list;
        fine.norm_x.resize(tc.vcount);
        fine.norm_y.resize(tc.vcount);
        fine.norm_z.resize(tc.vcount);
        fine.topo_version    = coarse.topo_version + 1;
    } else {
        coarse.build_adjacency();
        fine = loop_subdivide(coarse, false, &tc.stencil);
        fine.build_adjacency();
        tc.indices   = fine.indices;
        tc.vt_offset = fine.vert_tri_offset;
        tc.vt_list   = fine.vert_tri_list;
        tc.vcount    = fine.vertex_count();
        tc.ready     = true;
    }
    return fine;
}

ProjectionStats project_down_to_level(MultiresStack& stack, int target_level) {
    ProjectionStats stats;
    stats.target_level = target_level;

    const int L_max = stack.base_level + (int)stack.disp.size();
    stats.L_max = L_max;
    assert(target_level >= stack.base_level);
    assert(target_level <= L_max);
    if (target_level >= L_max) {
        return stats; // nothing to project
    }

    auto t0 = std::chrono::steady_clock::now();

    // ---- Phase 1: cascade-with-capture. Save per-level meshes (positions +
    //      topology) for every level in [base_level, L_max]. These are the
    //      "truth" we're preserving.
    const int passes = L_max - stack.base_level;
    std::vector<Mesh> mesh_at(passes + 1);   // mesh_at[k] corresponds to level base_level + k

    Mesh m = stack.base;
    m.build_adjacency();
    m.recompute_normals();
    mesh_at[0] = m;

    for (int i = 0; i < passes; i++) {
        m = subdiv_step_cached(stack, mesh_at[i], i);
        if ((int)stack.frames.size() <= i) stack.frames.emplace_back();
        if (stack.frames[i].empty()) {
            m.recompute_normals();   // frame normals from the pre-disp surface
            compute_frames(m, stack.frames[i]);
        }
        const uint32_t vc = m.vertex_count();
        for (uint32_t v = 0; v < vc; v++) {
            const Frame& f = stack.frames[i][v];
            const Vec3&  d = stack.disp[i][v];
            m.set_pos(v, m.get_pos(v) + f.t*d.x + f.b*d.y + f.n*d.z);
        }
        mesh_at[i + 1] = std::move(m);   // topology + post-disp positions
    }

    // ---- Phase 2: inverse Loop from L_max down to target_level. Walk the
    //      chain of fine meshes, replacing positions at each coarser level.
    std::vector<Vec3> scratch_coarse_pos;
    for (int k = L_max; k > target_level; k--) {
        int coarse_idx = k - 1 - stack.base_level;
        int fine_idx   = k     - stack.base_level;
        inverse_loop_one_step(mesh_at[coarse_idx], mesh_at[fine_idx],
                              scratch_coarse_pos);
        Mesh& cm = mesh_at[coarse_idx];
        uint32_t vc = cm.vertex_count();
        for (uint32_t v = 0; v < vc; v++) cm.set_pos(v, scratch_coarse_pos[v]);
    }

    // ---- Phase 3: forward rebuild. Start at target_level with the new
    //      positions and walk up to L_max, overwriting disp[] and frames[]
    //      so that cascade reproduces mesh_at[k] for every k > target_level.
    const int k_target = target_level - stack.base_level;       // 0..passes
    const int k_start_disp = projection_k_start(stack, target_level); // first disp idx to write

    Mesh rebuild;
    if (target_level == stack.base_level) {
        // Write new base positions, drop stale frames (all depend on base).
        uint32_t vc = stack.base.vertex_count();
        for (uint32_t v = 0; v < vc; v++)
            stack.base.set_pos(v, mesh_at[0].get_pos(v));
        for (auto& fv : stack.frames) fv.clear();
        rebuild = stack.base;
        rebuild.build_adjacency();
        rebuild.recompute_normals();
    } else {
        // Rewrite disp[k_start_disp] in local frame against the pre-disp
        // surface at target (obtained by subdividing mesh_at[k_target-1]
        // which still holds the unchanged lower-layer cascade). frames at
        // this layer depend only on disp[0..k_start_disp-1] (unchanged), so
        // they stay valid — we use them but do not rewrite them.
        Mesh pre_disp = subdiv_step_cached(stack, mesh_at[k_target - 1], k_target - 1);
        if ((int)stack.frames.size() <= k_start_disp) stack.frames.resize(k_start_disp + 1);
        if (stack.frames[k_start_disp].empty()) {
            pre_disp.recompute_normals();   // only consumer of these normals
            compute_frames(pre_disp, stack.frames[k_start_disp]);
        }
        const auto& frames_at = stack.frames[k_start_disp];
        const auto& target_pos = mesh_at[k_target];
        uint32_t vc = pre_disp.vertex_count();
        if ((int)stack.disp[k_start_disp].size() != (int)vc)
            stack.disp[k_start_disp].assign(vc, Vec3{0, 0, 0});
        for (uint32_t v = 0; v < vc; v++) {
            Vec3 world_delta = target_pos.get_pos(v) - pre_disp.get_pos(v);
            const Frame& f = frames_at[v];
            stack.disp[k_start_disp][v] = {
                world_delta.dot(f.t),
                world_delta.dot(f.b),
                world_delta.dot(f.n)
            };
            pre_disp.set_pos(v, target_pos.get_pos(v));  // now matches target
        }
        // Frames above this layer depend on the surface we just changed.
        for (int k = k_start_disp + 1; k < (int)stack.frames.size(); k++)
            stack.frames[k].clear();
        rebuild = pre_disp;   // at target_level with new positions, adjacency valid
    }

    // Walk up from target_level to L_max, recomputing disp and frames.
    int first_up_disp_idx = (target_level == stack.base_level)
                              ? 0
                              : (k_start_disp + 1);
    for (int kk = first_up_disp_idx; kk < passes; kk++) {
        rebuild = subdiv_step_cached(stack, rebuild, kk);
        rebuild.recompute_normals();   // compute_frames below reads them
        stack.frames[kk].clear();
        compute_frames(rebuild, stack.frames[kk]);

        const auto& frames_k   = stack.frames[kk];
        const auto& target_pos = mesh_at[kk + 1];
        uint32_t vc = rebuild.vertex_count();
        if ((int)stack.disp[kk].size() != (int)vc)
            stack.disp[kk].assign(vc, Vec3{0, 0, 0});
        for (uint32_t v = 0; v < vc; v++) {
            Vec3 world_delta = target_pos.get_pos(v) - rebuild.get_pos(v);
            const Frame& f = frames_k[v];
            stack.disp[kk][v] = {
                world_delta.dot(f.t),
                world_delta.dot(f.b),
                world_delta.dot(f.n)
            };
            rebuild.set_pos(v, target_pos.get_pos(v));
        }
    }

    // Rebuild base mirror and clear per-level mirror caches. Mirror maps are
    // topology-only, but the base positions changed (even for target > base_level
    // we re-run spatial hash defensively). Per-level maps rebuild lazily on cascade.
    build_mirror_spatial(stack.base, stack.base_mirror);
    stack.mirror.clear();

#ifdef CHISEL_DEBUG_MULTIRES
    // Reconstruction check: cascade to L_max and compare to saved truth.
    double max_err = 0.0;
    {
        Mesh check;
        cascade_to_level(stack, check, L_max);
        const auto& truth = mesh_at[passes];
        uint32_t vc = check.vertex_count();
        for (uint32_t v = 0; v < vc; v++) {
            Vec3 a = check.get_pos(v);
            Vec3 b = truth.get_pos(v);
            double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
            double e = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (e > max_err) max_err = e;
        }
    }
    stats.max_reconstruction_error = max_err;
#endif

    auto t1 = std::chrono::steady_clock::now();
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.did_anything = true;
    return stats;
}

void multires_stack_debug_print(const MultiresStack& stack) {
    std::printf("[multires] locked=%s base_level=%d current_level=%d\n",
                stack.locked ? "true" : "false",
                stack.base_level,
                stack.current_level);
    std::printf("[multires] base: %u verts, %u tris\n",
                stack.base.vertex_count(),
                stack.base.tri_count());
    std::printf("[multires] disp layers: %zu\n", stack.disp.size());
    for (size_t k = 0; k < stack.disp.size(); k++) {
        std::printf("[multires]   disp[%zu]: %zu entries\n", k, stack.disp[k].size());
    }
}
