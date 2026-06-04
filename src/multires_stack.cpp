#include "multires_stack.h"
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

void multires_stack_init_from_lock(MultiresStack& stack,
                                   const Mesh& locked_mesh,
                                   int icosphere_level) {
    stack.base = locked_mesh;
    stack.base_level = icosphere_level;
    stack.current_level = icosphere_level;
    stack.disp.clear();
    stack.frames.clear();
    stack.mirror.clear();
    stack.locked = true;

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

void cascade_to_level(MultiresStack& stack, Mesh& out, int K) {
    Mesh m = stack.base;
    const int passes = K - stack.base_level;

    for (int i = 0; i < passes; i++) {
        // Save coarse topology before subdivision for mirror propagation.
        uint32_t V_coarse = m.vertex_count();
        std::vector<uint32_t> coarse_idx = m.indices;

        m.build_adjacency();        // loop_subdivide reads CSR for vertex relocation
        m = loop_subdivide(m);

        if ((int)stack.disp.size() <= i)
            stack.disp.emplace_back(m.vertex_count(), Vec3{0.0f, 0.0f, 0.0f});
        if ((int)stack.frames.size() <= i)
            stack.frames.emplace_back();

        // Compute local frames from the pre-displacement surface at this level.
        // m right now is the subdivided base before any displacements — exactly right.
        if (stack.frames[i].empty()) {
            m.build_adjacency();    // needed for tangent (lowest-neighbor) lookup
            m.recompute_normals();  // frame normals from subdivided base
            compute_frames(m, stack.frames[i]);
        }

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

    // Populate topology mirror map in the output mesh.
    if (K == stack.base_level) {
        if (!stack.base_mirror.empty())
            out.mirror_x_map = stack.base_mirror;
    } else {
        int k = K - stack.base_level - 1;
        if (k < (int)stack.mirror.size() && !stack.mirror[k].empty())
            out.mirror_x_map = stack.mirror[k];
    }
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
        m.build_adjacency();
        m = loop_subdivide(m);
        m.build_adjacency();
        m.recompute_normals();
        if ((int)stack.frames.size() <= i) stack.frames.emplace_back();
        if (stack.frames[i].empty()) compute_frames(m, stack.frames[i]);
        const uint32_t vc = m.vertex_count();
        for (uint32_t v = 0; v < vc; v++) {
            const Frame& f = stack.frames[i][v];
            const Vec3&  d = stack.disp[i][v];
            m.set_pos(v, m.get_pos(v) + f.t*d.x + f.b*d.y + f.n*d.z);
        }
        mesh_at[i + 1] = m;   // includes topology + post-disp positions
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
        Mesh pre_disp = mesh_at[k_target - 1];
        pre_disp.build_adjacency();
        pre_disp = loop_subdivide(pre_disp);
        pre_disp.build_adjacency();
        pre_disp.recompute_normals();
        if ((int)stack.frames.size() <= k_start_disp) stack.frames.resize(k_start_disp + 1);
        if (stack.frames[k_start_disp].empty())
            compute_frames(pre_disp, stack.frames[k_start_disp]);
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
        rebuild.build_adjacency();
        rebuild = loop_subdivide(rebuild);
        rebuild.build_adjacency();
        rebuild.recompute_normals();
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
