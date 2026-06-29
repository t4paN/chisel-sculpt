#include "sdf.h"
#include "scene.h"
#include "mesh_entity.h"
#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("sdf_count" / ...)
#include "mc_tables.h"
#include "chisel_debug.h"
#include <glad/glad.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <functional>
#include <algorithm>
#include <limits>
#include <cstdlib>

// ============================================================================
//  SDF voxel-merge — GPU-native pooled-soup + generalized winding number.
//
//  Three compute dispatches over flat SSBOs:
//    1. distance splat   (one thread per triangle → unsigned band distance)
//    2. winding sign     (one thread per corner → signed marching-cubes field)
//    3. marching cubes   (one thread per cell → triangle soup, atomic-appended)
//  The only CPU touch is a hash-weld on readback into an indexed Mesh, which is
//  unavoidable because the result must land as a Mesh for scene/undo/export.
//
//  Everything here is one-shot and user-paced (like perform_remesh): allocation
//  and a single readback are fine; the hot-path zero-alloc rules do not apply.
//  All GL resources are created at the top of voxel_merge_selected and freed at
//  the bottom — nothing persists in ComputeState.
// ============================================================================

namespace {

constexpr int   BAND_DILATE = 2;        // splat AABB dilation in cells (band thickness)
constexpr float BAND_FAR     = 1e18f;   // sentinel for off-band corners (atomicMin floor)
constexpr int   GRID_PAD     = 4;       // cells of empty margin so the surface never touches the wall

// ---- Soup gather ----------------------------------------------------------
// Concatenate world-space triangles into one soup. Entities are baked in world
// space (twins are real tris), so no transform.
//   - ADDITIVE part: the current selection, normal winding (unioned).
//   - CUTTER part (subtract only): every unselected (red) committed entity, with
//     each triangle wound backwards. The winding-sign pass sums signed solid
//     angles over the whole soup, so a flipped tri negates its contribution and
//     the cutter's interior reads as "outside" -> it carves the union (A − B)
//     instead of joining it. Distance (unsigned) is winding-agnostic, so the
//     cavity walls still get correct band distances and MC tessellates them.
//     See sdf_sign.wgsl. Cutters are NOT consumed by the merge — the splice only
//     replaces the selection.
// Additive verts are emitted first; `additive_floats` marks the end of the
// additive block in `pos` so the caller can bound the grid to the kept region
// (cutters poking outside it still sum into the winding correctly).
uint32_t gather_soup(const Scene& scene,
                     std::vector<float>&    pos,
                     std::vector<uint32_t>& idx,
                     std::vector<uint32_t>& vcol,  // per-source-vertex colour (white if unpainted)
                     bool&                  any_paint,
                     bool                   subtract,
                     uint32_t&              n_additive,
                     uint32_t&              n_cutter,
                     size_t&                additive_floats)
{
    n_additive = 0;
    n_cutter   = 0;
    any_paint  = false;

    auto add_entity = [&](const MeshEntity* e, bool cutter) -> bool {
        if (!e) return false;
        const Mesh& m = e->mesh;
        if (m.vertex_count() == 0 || m.indices.empty()) return false;

        const bool painted = !m.color.empty();
        if (painted) any_paint = true;

        uint32_t base = (uint32_t)(pos.size() / 3);
        pos.reserve(pos.size() + m.vertex_count() * 3);
        vcol.reserve(vcol.size() + m.vertex_count());
        for (uint32_t v = 0; v < m.vertex_count(); v++) {
            pos.push_back(m.pos_x[v]);
            pos.push_back(m.pos_y[v]);
            pos.push_back(m.pos_z[v]);
            vcol.push_back((painted && v < m.color.size()) ? m.color[v] : 0xFFFFFFFFu);
        }
        idx.reserve(idx.size() + m.indices.size());
        if (cutter) {
            for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
                idx.push_back(base + m.indices[t + 0]);
                idx.push_back(base + m.indices[t + 2]);
                idx.push_back(base + m.indices[t + 1]);
            }
        } else {
            for (uint32_t i : m.indices) idx.push_back(base + i);
        }
        return true;
    };

    // Additive: the current selection.
    for (uint32_t id : scene.selected_ids())
        if (add_entity(scene.find_entity(id), /*cutter=*/false)) n_additive++;
    additive_floats = pos.size();

    // Cutters: every unselected (red) committed entity — subtract merge only.
    if (subtract) {
        for (const auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview) continue;
            bool selected = false;
            for (uint32_t sid : scene.selected_ids())
                if (sid == up->id) { selected = true; break; }
            if (selected) continue;
            if (add_entity(up.get(), /*cutter=*/true)) n_cutter++;
        }
    }
    return n_additive + n_cutter;
}

// ---- Fast Winding Number tree ---------------------------------------------
// The winding-sign pass used to be a brute O(band_corners * tris) solid-angle sum:
// every band corner looped over EVERY soup triangle. Replace the per-corner inner
// loop with a Barnes-Hut traversal (Barill et al. 2018, "Fast Winding Numbers for
// Soups and Clouds"): build a median-split BVH over the triangles once on the CPU,
// precompute each node's dipole (order-0) + order-1 aggregates, and per corner
// traverse — a node far from the query (|q-p| > beta*radius) contributes via its
// Taylor expansion; only near leaves are summed exactly. O(corners * log tris).
//
// Per node we store, about an area-weighted expansion point p:
//   g0 = Σ aₜnₜ                         (order-0 area-weighted normal sum)
//   T  = Σ aₜnₜ ⊗ (cₜ - p)              (order-1 tensor; cₜ = tri centroid)
// where aₜnₜ = ½(b-a)×(c-a) is already the area-weighted normal. The winding
// contribution of a far node at query q (d = p-q, r = |d|) is, before /4π:
//   w0 = d·g0 / r³
//   w1 = tr(T)/r³ - 3 (dᵀTd)/r⁵
// (the 0th/1st terms of expanding the Poisson-kernel gradient (x-q)/|x-q|³ about p).
// radius = max |vertex - p| over the subtree, the conservative acceptance bound.
constexpr int   FWN_LEAF = 8;       // tris per leaf below which we stop splitting
constexpr float FWN_BETA = 2.0f;    // far-field acceptance: |q-p| > BETA*radius ⇒ expand

// std430/host-mirrored node (96 bytes; vec4-aligned, no padding holes).
struct FwnNodeGPU {
    float px, py, pz, r;            // expansion point (xyz) + subtree radius (w)
    float g0x, g0y, g0z, _g;       // order-0: Σ area-weighted normals
    float t0x, t0y, t0z, _0;       // T row 0
    float t1x, t1y, t1z, _1;       // T row 1
    float t2x, t2y, t2z, _2;       // T row 2
    int32_t left, right, tri_start, tri_count;  // children (-1 if leaf) / leaf tri range
};
static_assert(sizeof(FwnNodeGPU) == 96, "FwnNode must match the std430 shader struct");

// Build the FWN tree over the soup. `order` is reordered in place into leaf-contiguous
// runs; `nodes` receives the flat tree (root at index 0). One-shot CPU work (allowed
// here — the zero-alloc / no-readback rules are stroke-only).
static void build_fwn_tree(const std::vector<float>& pos, const std::vector<uint32_t>& idx,
                           std::vector<FwnNodeGPU>& nodes, std::vector<uint32_t>& order)
{
    const uint32_t nt = (uint32_t)(idx.size() / 3);
    nodes.clear();
    order.resize(nt);
    if (nt == 0) return;

    // Per-tri centroid + area-weighted normal (an = ½(b-a)×(c-a); |an| = area).
    std::vector<Vec3> cen(nt), an(nt);
    for (uint32_t t = 0; t < nt; t++) {
        uint32_t i0 = idx[3*t], i1 = idx[3*t+1], i2 = idx[3*t+2];
        Vec3 a(pos[3*i0], pos[3*i0+1], pos[3*i0+2]);
        Vec3 b(pos[3*i1], pos[3*i1+1], pos[3*i1+2]);
        Vec3 c(pos[3*i2], pos[3*i2+1], pos[3*i2+2]);
        cen[t] = (a + b + c) * (1.0f/3.0f);
        an[t]  = (b - a).cross(c - a) * 0.5f;
        order[t] = t;
    }

    nodes.reserve(nt / FWN_LEAF * 2 + 8);

    // Recursive median split over order[start,end). Returns the new node's index.
    std::function<int32_t(uint32_t,uint32_t)> build = [&](uint32_t start, uint32_t end) -> int32_t {
        int32_t self = (int32_t)nodes.size();
        nodes.push_back({});                       // reserve slot (children fill it after recursion)

        // Aggregate dipole + order-1 + radius over this range.
        Vec3 g0(0,0,0), wcen(0,0,0);
        double area = 0.0;
        for (uint32_t s = start; s < end; s++) {
            uint32_t t = order[s];
            float at = an[t].length();
            g0   += an[t];
            wcen += cen[t] * at;
            area += at;
        }
        Vec3 p = (area > 0.0) ? wcen * (float)(1.0/area)
                              : [&]{ Vec3 m(0,0,0); for (uint32_t s=start;s<end;s++) m+=cen[order[s]];
                                     return m * (1.0f/(float)(end-start)); }();
        // T = Σ an ⊗ (cen - p); radius = max |vertex - p|.
        float t00=0,t01=0,t02=0, t10=0,t11=0,t12=0, t20=0,t21=0,t22=0;
        float r2 = 0.0f;
        for (uint32_t s = start; s < end; s++) {
            uint32_t t = order[s];
            Vec3 dlt = cen[t] - p, n = an[t];
            t00 += n.x*dlt.x; t01 += n.x*dlt.y; t02 += n.x*dlt.z;
            t10 += n.y*dlt.x; t11 += n.y*dlt.y; t12 += n.y*dlt.z;
            t20 += n.z*dlt.x; t21 += n.z*dlt.y; t22 += n.z*dlt.z;
            for (int k = 0; k < 3; k++) {
                Vec3 v(pos[3*idx[3*t+k]], pos[3*idx[3*t+k]+1], pos[3*idx[3*t+k]+2]);
                r2 = std::max(r2, (v - p).dot(v - p));
            }
        }

        FwnNodeGPU& nd = nodes[self];
        nd.px=p.x; nd.py=p.y; nd.pz=p.z; nd.r=std::sqrt(r2);
        nd.g0x=g0.x; nd.g0y=g0.y; nd.g0z=g0.z; nd._g=0;
        nd.t0x=t00; nd.t0y=t01; nd.t0z=t02; nd._0=0;
        nd.t1x=t10; nd.t1y=t11; nd.t1z=t12; nd._1=0;
        nd.t2x=t20; nd.t2y=t21; nd.t2z=t22; nd._2=0;

        uint32_t count = end - start;
        if (count <= (uint32_t)FWN_LEAF) {         // leaf
            nd.left = nd.right = -1;
            nd.tri_start = (int32_t)start; nd.tri_count = (int32_t)count;
            return self;
        }

        // Split on the longest centroid-extent axis at the median centroid.
        Vec3 clo = cen[order[start]], chi = clo;
        for (uint32_t s = start+1; s < end; s++) {
            Vec3 c = cen[order[s]];
            clo.x=std::min(clo.x,c.x); clo.y=std::min(clo.y,c.y); clo.z=std::min(clo.z,c.z);
            chi.x=std::max(chi.x,c.x); chi.y=std::max(chi.y,c.y); chi.z=std::max(chi.z,c.z);
        }
        Vec3 ext = chi - clo;
        int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
        uint32_t mid = start + count/2;
        std::nth_element(order.begin()+start, order.begin()+mid, order.begin()+end,
            [&](uint32_t a, uint32_t b){ const Vec3& ca=cen[a]; const Vec3& cb=cen[b];
                return (axis==0?ca.x:axis==1?ca.y:ca.z) < (axis==0?cb.x:axis==1?cb.y:cb.z); });

        int32_t L = build(start, mid);
        int32_t Rr = build(mid, end);
        nodes[self].left = L; nodes[self].right = Rr;  // re-index: vector may have realloc'd
        nodes[self].tri_start = nodes[self].tri_count = 0;
        return self;
    };
    build(0, nt);
}

// ---- Shader sources -------------------------------------------------------
// The 5 SDF compute kernels now live as self-contained canonical shaders, embedded at
// build time: shaders/{glsl,wgsl}/sdf_{count,expand,sign,mc,nets}.* — referenced by
// gpu::embedded_shader("<stem>"). The shared GLSL snippets (pt_tri_dist / degenerate /
// band-AABB / linear_gid) that used to be string-concatenated here are inlined into
// each stem (the embedder does no include injection). Loose grid uniforms became an
// anonymous std140 Params UBO at binding 63 (one struct per kernel, below).

// ---- Hash-weld ------------------------------------------------------------
// Snap each soup vertex to a voxel*1e-3 lattice, dedup by quantized key →
// shared vertices ⇒ watertight indexed mesh. Single CPU touch, O(verts).
struct QKey {
    int64_t x, y, z;
    bool operator==(const QKey& o) const { return x==o.x && y==o.y && z==o.z; }
};
struct QKeyHash {
    size_t operator()(const QKey& k) const {
        // 64-bit splitmix-ish mix of the three lattice coords.
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](int64_t v){ h ^= (uint64_t)v; h *= 1099511628211ull; };
        mix(k.x); mix(k.y); mix(k.z);
        return (size_t)h;
    }
};

void hash_weld(const std::vector<float>& soup,  // 18 floats per tri (pos+nrm x3)
               uint32_t out_tris, float snap, Mesh& out)
{
    out.pos_x.clear(); out.pos_y.clear(); out.pos_z.clear();
    out.norm_x.clear(); out.norm_y.clear(); out.norm_z.clear();
    out.indices.clear();
    out.mask.clear();
    out.indices.reserve(out_tris * 3);

    float inv = 1.0f / snap;
    std::unordered_map<QKey, uint32_t, QKeyHash> table;
    table.reserve(out_tris * 2);

    for (uint32_t t = 0; t < out_tris; t++) {
        for (int v = 0; v < 3; v++) {
            const float* p = &soup[(t * 3 + v) * 6];
            QKey key{ (int64_t)std::llround(p[0]*inv),
                      (int64_t)std::llround(p[1]*inv),
                      (int64_t)std::llround(p[2]*inv) };
            auto it = table.find(key);
            uint32_t vi;
            if (it == table.end()) {
                vi = (uint32_t)out.pos_x.size();
                table.emplace(key, vi);
                out.pos_x.push_back(p[0]);
                out.pos_y.push_back(p[1]);
                out.pos_z.push_back(p[2]);
                out.norm_x.push_back(p[3]);
                out.norm_y.push_back(p[4]);
                out.norm_z.push_back(p[5]);
            } else {
                vi = it->second;
            }
            out.indices.push_back(vi);
        }
    }
}

// ---- Manifold / watertight report ------------------------------------------
// Surfaced after merge so a bad result is visible before printing. A clean
// watertight result has 0 boundary edges, 0 non-manifold edges, 1 component.
void manifold_report(const Mesh& m, uint32_t& boundary,
                     uint32_t& nonmanifold, uint32_t& components)
{
    boundary = nonmanifold = components = 0;
    uint32_t nt = m.tri_count(), nv = m.vertex_count();
    if (nt == 0 || nv == 0) return;

    auto ekey = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) std::swap(a, b);
        return ((uint64_t)a << 32) | (uint64_t)b;
    };
    std::unordered_map<uint64_t, uint32_t> edge_count;
    edge_count.reserve(nt * 3);

    // Union-find over vertices (path-halving + union by size omitted for brevity).
    std::vector<uint32_t> parent(nv);
    for (uint32_t v = 0; v < nv; v++) parent[v] = v;
    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint32_t a, uint32_t b) { parent[find(a)] = find(b); };

    for (uint32_t t = 0; t < nt; t++) {
        uint32_t i0 = m.indices[t*3+0], i1 = m.indices[t*3+1], i2 = m.indices[t*3+2];
        edge_count[ekey(i0, i1)]++;
        edge_count[ekey(i1, i2)]++;
        edge_count[ekey(i2, i0)]++;
        unite(i0, i1); unite(i1, i2);
    }
    for (auto& kv : edge_count) {
        if (kv.second == 1)      boundary++;
        else if (kv.second > 2)  nonmanifold++;
    }

    std::unordered_set<uint32_t> roots;
    std::vector<char> used(nv, 0);
    for (uint32_t i : m.indices) used[i] = 1;
    for (uint32_t v = 0; v < nv; v++) if (used[v]) roots.insert(find(v));
    components = (uint32_t)roots.size();
}

// ---- Off-band sign flood fill --------------------------------------------
// The winding pass signs only band corners (|f| < band_far) with ±dist; off-band
// corners come back marked |f| == band_far. Each connected off-band region
// touches the band on exactly one side, so its sign is well-defined: multi-source
// BFS from the signed band corners propagates the inside/outside bit into the
// bulk. Off-band corners become ±band_far with the correct sign, so MC never
// produces a false crossing at the band boundary. O(corners).
void flood_fill_sign(std::vector<float>& field, uint32_t R, float band_far) {
    int R1 = (int)R + 1;
    int64_t N = (int64_t)R1 * R1 * R1;
    const float known_thresh = band_far * 0.5f;  // band corners are |f| << band_far

    std::vector<uint8_t> known((size_t)N, 0);
    std::vector<int32_t> work;
    work.reserve((size_t)N / 4 + 16);
    for (int64_t c = 0; c < N; c++) {
        if (std::fabs(field[c]) < known_thresh) { known[c] = 1; work.push_back((int32_t)c); }
    }

    auto decode = [&](int64_t c, int& i, int& j, int& k) {
        i = (int)(c % R1); j = (int)((c / R1) % R1); k = (int)(c / ((int64_t)R1 * R1));
    };
    // Index-based worklist (BFS order irrelevant: each region is sign-uniform).
    for (size_t head = 0; head < work.size(); head++) {
        int64_t c = work[head];
        float s = (field[c] < 0.0f) ? -band_far : band_far;
        int i, j, k; decode(c, i, j, k);
        const int di[6] = {1,-1,0,0,0,0};
        const int dj[6] = {0,0,1,-1,0,0};
        const int dk[6] = {0,0,0,0,1,-1};
        for (int n = 0; n < 6; n++) {
            int ni = i+di[n], nj = j+dj[n], nk = k+dk[n];
            if (ni<0||ni>=R1||nj<0||nj>=R1||nk<0||nk>=R1) continue;
            int64_t nc = ni + (int64_t)R1*(nj + (int64_t)R1*nk);
            if (known[nc]) continue;
            known[nc] = 1;
            field[nc] = s;                 // inherit the touching band side's sign
            work.push_back((int32_t)nc);
        }
    }
    // Any region never reached (no band neighbour at all) defaults to outside.
    for (int64_t c = 0; c < N; c++)
        if (!known[c]) field[c] = band_far;
}

// ---- Tangential relaxation toward the iso-surface --------------------------
// Trilinearly sample the signed MC field (read back to CPU) at a world point.
// Corner (i,j,k) index = i + S*(j + S*k), S = R+1 — matches the MC shader.
float sample_field(const std::vector<float>& f, const SdfGrid& g, Vec3 p) {
    float gx = (p.x - g.origin.x) / g.voxel;
    float gy = (p.y - g.origin.y) / g.voxel;
    float gz = (p.z - g.origin.z) / g.voxel;
    float maxc = (float)g.R - 1e-4f;          // keep i+1 in [0, R]
    gx = std::min(std::max(gx, 0.0f), maxc);
    gy = std::min(std::max(gy, 0.0f), maxc);
    gz = std::min(std::max(gz, 0.0f), maxc);
    uint32_t i = (uint32_t)gx, j = (uint32_t)gy, k = (uint32_t)gz;
    float fx = gx - i, fy = gy - j, fz = gz - k;
    uint32_t S = g.R + 1;

    // Gather the 8 cell corners. Off-band corners carry the ±BAND_FAR sentinel
    // (set by flood_fill_sign). Left raw, one such corner would swamp the
    // trilinear blend with a ~1e18 value and fling the relax reprojection off to
    // infinity. Replace each sentinel with a finite, sign-preserving stand-in —
    // the largest genuine |distance| in this stencil (a real band value), or a
    // 1-voxel floor if the whole cell is off-band — so the sample stays bounded
    // and the inside/outside sign is still correct. In-band corners (all MC and
    // the normal relax case ever touch) pass through unchanged, so the blend is
    // bit-identical to before whenever no sentinel is present.
    float c[8] = {
        f[ i   + S*( j    + S* k   )], f[(i+1) + S*( j    + S* k   )],
        f[ i   + S*((j+1) + S* k   )], f[(i+1) + S*((j+1) + S* k   )],
        f[ i   + S*( j    + S*(k+1))], f[(i+1) + S*( j    + S*(k+1))],
        f[ i   + S*((j+1) + S*(k+1))], f[(i+1) + S*((j+1) + S*(k+1))],
    };
    const float SENTINEL = BAND_FAR * 0.5f;
    float maxmag = 0.0f;
    for (float v : c) if (std::fabs(v) < SENTINEL) maxmag = std::max(maxmag, std::fabs(v));
    float fill = (maxmag > 0.0f) ? maxmag : g.voxel;   // floor: cell is wholly off-band
    for (float& v : c) if (std::fabs(v) >= SENTINEL) v = (v < 0.0f ? -fill : fill);

    // c[idx], idx = ii + 2*jj + 4*kk  (ii/jj/kk ∈ {0,1} select corner+0/+1 on x/y/z)
    auto CC = [&](int ii, int jj, int kk) { return c[ii + 2*jj + 4*kk]; };
    float c00 = CC(0,0,0)*(1-fx) + CC(1,0,0)*fx;
    float c10 = CC(0,1,0)*(1-fx) + CC(1,1,0)*fx;
    float c01 = CC(0,0,1)*(1-fx) + CC(1,0,1)*fx;
    float c11 = CC(0,1,1)*(1-fx) + CC(1,1,1)*fx;
    float c0 = c00*(1-fy) + c10*fy;
    float c1 = c01*(1-fy) + c11*fy;
    return c0*(1-fz) + c1*fz;
}

// Even out the marching-cubes triangulation WITHOUT losing shape. Each pass
// moves a vertex toward its 1-ring centroid but only in the tangent plane (the
// normal component, taken from the SDF gradient, is stripped), then snaps the
// residual back onto the field's zero level set. Plain (Laplacian) smoothing
// deflates the form because it moves vertices off-surface; constraining to the
// tangent plane + reprojecting keeps every vertex exactly on the merged surface
// while the triangles spread to uniform spacing. Self-contained — reuses the
// field we already built; no perform_remesh (which would mirror-symmetrise).
// `pinned` (optional): verts flagged 1 are held fixed across every pass. Used by
// the mirror path to lock the x=0 seam loop so the two halves stay weldable.
void relax_to_field(Mesh& m, const std::vector<float>& field,
                    const SdfGrid& grid, int iters, float lambda,
                    const std::vector<uint8_t>* pinned = nullptr) {
    uint32_t nv = m.vertex_count();
    if (nv == 0 || m.indices.empty() || iters <= 0) return;

    // Unique 1-ring neighbours from the triangle list.
    std::vector<std::vector<uint32_t>> nbr(nv);
    {
        std::vector<std::unordered_set<uint32_t>> sets(nv);
        for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
            uint32_t a = m.indices[t], b = m.indices[t+1], c = m.indices[t+2];
            sets[a].insert(b); sets[a].insert(c);
            sets[b].insert(a); sets[b].insert(c);
            sets[c].insert(a); sets[c].insert(b);
        }
        for (uint32_t v = 0; v < nv; v++) nbr[v].assign(sets[v].begin(), sets[v].end());
    }

    float h = 0.5f * grid.voxel;              // central-difference step for the gradient
    std::vector<float> nx(nv), ny(nv), nz(nv);

    for (int it = 0; it < iters; it++) {
        for (uint32_t v = 0; v < nv; v++) {
            const std::vector<uint32_t>& N = nbr[v];
            Vec3 p(m.pos_x[v], m.pos_y[v], m.pos_z[v]);
            if (N.empty()) { nx[v]=p.x; ny[v]=p.y; nz[v]=p.z; continue; }

            // Seam vert (mirror path): relax it as a 1D curve in the x=0 plane —
            // average only its seam neighbours, force x=0, reproject onto the
            // surface using the in-plane (x-stripped) gradient. Evens the seam
            // spacing so the first interior row isn't pulled into slivers, while
            // staying exactly on the mirror plane (self-mapped partner preserved).
            if (pinned && (*pinned)[v]) {
                Vec3 c(0,0,0); int cnt = 0;
                for (uint32_t u : N) if ((*pinned)[u]) { c += Vec3(m.pos_x[u],m.pos_y[u],m.pos_z[u]); cnt++; }
                if (cnt == 0) { nx[v]=p.x; ny[v]=p.y; nz[v]=p.z; continue; }
                Vec3 pn = p + (c * (1.0f/(float)cnt) - p) * lambda;
                pn.x = 0.0f;
                Vec3 g(0.0f,
                    sample_field(field,grid,Vec3(pn.x,pn.y+h,pn.z)) - sample_field(field,grid,Vec3(pn.x,pn.y-h,pn.z)),
                    sample_field(field,grid,Vec3(pn.x,pn.y,pn.z+h)) - sample_field(field,grid,Vec3(pn.x,pn.y,pn.z-h)));
                float gl = g.length();
                if (gl > 1e-12f) { Vec3 n = g*(1.0f/gl); pn = pn - n*sample_field(field,grid,pn); pn.x = 0.0f; }
                nx[v]=pn.x; ny[v]=pn.y; nz[v]=pn.z;
                continue;
            }

            Vec3 c(0,0,0);
            for (uint32_t u : N) c += Vec3(m.pos_x[u], m.pos_y[u], m.pos_z[u]);
            c = c * (1.0f / (float)N.size());
            Vec3 d = c - p;                    // umbrella (uniform-Laplacian) vector

            // Surface normal from the SDF gradient (central differences).
            Vec3 grad(
                sample_field(field,grid,Vec3(p.x+h,p.y,p.z)) - sample_field(field,grid,Vec3(p.x-h,p.y,p.z)),
                sample_field(field,grid,Vec3(p.x,p.y+h,p.z)) - sample_field(field,grid,Vec3(p.x,p.y-h,p.z)),
                sample_field(field,grid,Vec3(p.x,p.y,p.z+h)) - sample_field(field,grid,Vec3(p.x,p.y,p.z-h)));
            float gl = grad.length();
            Vec3 pn;
            if (gl > 1e-12f) {
                Vec3 n = grad * (1.0f / gl);
                Vec3 dt = d - n * d.dot(n);     // tangential component only
                pn = p + dt * lambda;
                pn = pn - n * sample_field(field, grid, pn);  // reproject onto zero level set
            } else {
                pn = p + d * lambda;            // no usable gradient — fall back to plain move
            }
            nx[v]=pn.x; ny[v]=pn.y; nz[v]=pn.z;
        }
        m.pos_x = nx; m.pos_y = ny; m.pos_z = nz;   // Jacobi update (all verts move together)
    }
}

// ---- Mirror-symmetric extraction -------------------------------------------
// The MC case table bakes a fixed grid-space diagonal per ambiguous face, so a
// +x feature and its -x twin tessellate the SAME world direction — geometry can
// be symmetric while topology is not, which defeats the vertex partner map. Fix:
// keep ONLY the +x half (the grid is aligned so x=0 is a corner layer, hence no
// straddling cells — every +x tri has all verts at x>=0, seam verts exactly 0),
// relax it, then reflect it to -x with swapped winding. Symmetric by construction.

// Drop -x tris, compact, and report the x=0 seam vert set (post-compaction).
void clip_to_plus_x(Mesh& m, float voxel, std::vector<uint8_t>& seam_out) {
    float seam_tol = voxel * 1e-4f;
    uint32_t vc = m.vertex_count();

    for (uint32_t v = 0; v < vc; v++)
        if (std::fabs(m.pos_x[v]) < seam_tol) m.pos_x[v] = 0.0f;   // snap to the plane

    std::vector<uint32_t> keep;
    keep.reserve(m.indices.size());
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        uint32_t a = m.indices[t], b = m.indices[t+1], c = m.indices[t+2];
        if (m.pos_x[a] < -seam_tol || m.pos_x[b] < -seam_tol || m.pos_x[c] < -seam_tol)
            continue;   // belongs to a -x cell
        if (m.pos_x[a] == 0.0f && m.pos_x[b] == 0.0f && m.pos_x[c] == 0.0f)
            continue;   // degenerate planar sail lying in the seam
        keep.push_back(a); keep.push_back(b); keep.push_back(c);
    }
    m.indices.swap(keep);

    // Compact away verts no kept tri references.
    std::vector<uint32_t> remap(vc, 0xFFFFFFFFu);
    uint32_t nv = 0;
    for (uint32_t idx : m.indices)
        if (remap[idx] == 0xFFFFFFFFu) remap[idx] = nv++;

    std::vector<float> px(nv), py(nv), pz(nv), nxx(nv), nyy(nv), nzz(nv);
    for (uint32_t v = 0; v < vc; v++) {
        uint32_t r = remap[v];
        if (r == 0xFFFFFFFFu) continue;
        px[r]=m.pos_x[v]; py[r]=m.pos_y[v]; pz[r]=m.pos_z[v];
        nxx[r]=m.norm_x[v]; nyy[r]=m.norm_y[v]; nzz[r]=m.norm_z[v];
    }
    m.pos_x.swap(px); m.pos_y.swap(py); m.pos_z.swap(pz);
    m.norm_x.swap(nxx); m.norm_y.swap(nyy); m.norm_z.swap(nzz);
    for (uint32_t& idx : m.indices) idx = remap[idx];

    seam_out.assign(nv, 0);
    for (uint32_t v = 0; v < nv; v++)
        if (m.pos_x[v] == 0.0f) seam_out[v] = 1;
}

// Append the -x reflection: X-negated copies of non-seam verts (seam verts are
// shared), mirrored tris with 2nd/3rd index swapped so they face outward.
void reflect_across_x(Mesh& m, const std::vector<uint8_t>& seam) {
    uint32_t vc = m.vertex_count();
    uint32_t tn = (uint32_t)m.indices.size();
    std::vector<uint32_t> refl(vc);
    for (uint32_t v = 0; v < vc; v++) {
        if (seam[v]) { refl[v] = v; continue; }
        refl[v] = m.vertex_count();
        m.pos_x.push_back(-m.pos_x[v]);
        m.pos_y.push_back( m.pos_y[v]);
        m.pos_z.push_back( m.pos_z[v]);
        m.norm_x.push_back(-m.norm_x[v]);
        m.norm_y.push_back( m.norm_y[v]);
        m.norm_z.push_back( m.norm_z[v]);
    }
    for (uint32_t t = 0; t < tn; t += 3) {
        m.indices.push_back(refl[m.indices[t  ]]);
        m.indices.push_back(refl[m.indices[t+2]]);
        m.indices.push_back(refl[m.indices[t+1]]);
    }
}

// Drop verts no tri references; carry the parallel seam mask through the remap.
void compact_with_seam(Mesh& m, std::vector<uint8_t>& seam) {
    uint32_t vc = m.vertex_count();
    std::vector<uint32_t> remap(vc, 0xFFFFFFFFu);
    uint32_t nv = 0;
    for (uint32_t idx : m.indices)
        if (remap[idx] == 0xFFFFFFFFu) remap[idx] = nv++;

    std::vector<float> px(nv), py(nv), pz(nv), nxx(nv), nyy(nv), nzz(nv);
    std::vector<uint8_t> sm(nv, 0);
    for (uint32_t v = 0; v < vc; v++) {
        uint32_t r = remap[v];
        if (r == 0xFFFFFFFFu) continue;
        px[r]=m.pos_x[v]; py[r]=m.pos_y[v]; pz[r]=m.pos_z[v];
        nxx[r]=m.norm_x[v]; nyy[r]=m.norm_y[v]; nzz[r]=m.norm_z[v];
        sm[r]=seam[v];
    }
    m.pos_x.swap(px); m.pos_y.swap(py); m.pos_z.swap(pz);
    m.norm_x.swap(nxx); m.norm_y.swap(nyy); m.norm_z.swap(nzz);
    seam.swap(sm);
    for (uint32_t& idx : m.indices) idx = remap[idx];
}

// H-A: post-reflect seam-band weld. reflect_across_x shares only the FLAGGED
// seam verts; any boundary vert that belonged on the plane but slipped the flag
// got a fresh ±ε twin instead → a hairline seam down x=0. Mop those up
// independently of the flag: snap every vert in the |x|<band slab onto x=0, then
// weld verts that coincide in (y,z) — a reflection pair shares (y,z) exactly, so
// the ±ε twins collapse to one self-symmetric seam vert. Catches whatever clip /
// classify / H-C leaked. Runs after reflection; normals are rebuilt downstream.
// One-shot op — allocation is fine here (zero-alloc rule is stroke-only).
static void weld_seam_band(Mesh& m, float voxel) {
    const float band = voxel * 1e-3f;
    if (band <= 0.0f) return;
    const uint32_t vc = m.vertex_count();

    // Key packs the (y,z) cell with no aliasing: cell_y in the high 32 bits,
    // cell_z in the low 32, so distinct cells never collide. Identical (y,z) (the
    // twins) map to the same cell; real verts are ~voxel apart ≫ band, so they
    // never share one. Representative = first vert seen in the cell.
    std::unordered_map<int64_t, uint32_t> rep;
    std::vector<uint32_t> remap(vc);
    for (uint32_t v = 0; v < vc; v++) remap[v] = v;
    auto cell = [&](float a) -> int64_t { return (int64_t)std::llround(a / band); };

    uint32_t merged = 0;
    for (uint32_t v = 0; v < vc; v++) {
        if (std::fabs(m.pos_x[v]) >= band) continue;
        m.pos_x[v] = 0.0f;                                       // snap onto x=0
        int64_t k = (cell(m.pos_y[v]) << 32) ^ (cell(m.pos_z[v]) & 0xffffffffLL);
        auto it = rep.find(k);
        if (it == rep.end()) rep.emplace(k, v);
        else { remap[v] = it->second; merged++; }
    }
    if (merged == 0) return;

    for (uint32_t& idx : m.indices) idx = remap[idx];

    // Drop tris that collapsed onto the seam.
    std::vector<uint32_t> keep; keep.reserve(m.indices.size());
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        uint32_t a=m.indices[t], b=m.indices[t+1], c=m.indices[t+2];
        if (a==b || b==c || a==c) continue;
        keep.push_back(a); keep.push_back(b); keep.push_back(c);
    }
    m.indices.swap(keep);

    // Compact orphaned verts (pos+norm; mask/colour are built downstream).
    std::vector<uint32_t> vm(vc, 0xFFFFFFFFu);
    uint32_t nv = 0;
    for (uint32_t idx : m.indices) if (vm[idx] == 0xFFFFFFFFu) vm[idx] = nv++;
    std::vector<float> px(nv),py(nv),pz(nv),nx(nv),ny(nv),nz(nv);
    for (uint32_t v = 0; v < vc; v++) {
        uint32_t r = vm[v]; if (r == 0xFFFFFFFFu) continue;
        px[r]=m.pos_x[v]; py[r]=m.pos_y[v]; pz[r]=m.pos_z[v];
        nx[r]=m.norm_x[v]; ny[r]=m.norm_y[v]; nz[r]=m.norm_z[v];
    }
    m.pos_x.swap(px); m.pos_y.swap(py); m.pos_z.swap(pz);
    m.norm_x.swap(nx); m.norm_y.swap(ny); m.norm_z.swap(nz);
    for (uint32_t& idx : m.indices) idx = vm[idx];

    std::printf("[sdf-mirror] seam-band weld: %u escapee vert(s) merged\n", merged);
}

// 1-ring vertex neighbours of v via the vert→tri adjacency (must be built).
static void ring_neighbors(const Mesh& m, uint32_t v, std::unordered_set<uint32_t>& out) {
    out.clear();
    uint32_t s = m.vert_tri_offset[v], e = m.vert_tri_offset[v + 1];
    for (uint32_t j = s; j < e; j++) {
        uint32_t tri = m.vert_tri_list[j];
        for (int k = 0; k < 3; k++) {
            uint32_t w = m.indices[tri*3 + k];
            if (w != v) out.insert(w);
        }
    }
}

// H-C: is a P→S near-seam collapse topology-safe? `weld_near_seam` used to weld
// purely on distance, which could fold a vertex umbrella and leave an edge in 3+
// tris after reflection — the `nonmf` failures the H-D gate now catches. Two
// guards (same idea as the remesher's collapse link-condition in remesh.cpp):
//   (1) Link condition — P and S must share EXACTLY 2 ring-neighbours (the two
//       verts opposite the P–S edge). More shared ⇒ the collapse is non-manifold.
//   (2) Seam-degree — P may only touch S and S's existing seam-loop neighbours;
//       a sliver bridging to a further seam vert would hand S a third seam edge,
//       branching the reflected seam loop (also non-manifold).
// Evaluated against the pre-remap adjacency, so it's order-independent.
static bool collapse_is_seam_safe(const Mesh& m, const std::vector<uint8_t>& seam,
                                  uint32_t P, uint32_t S) {
    std::unordered_set<uint32_t> nP, nS;
    ring_neighbors(m, P, nP);
    ring_neighbors(m, S, nS);

    int shared = 0;
    for (uint32_t w : nP)
        if (w != S && nS.count(w)) shared++;
    if (shared != 2) return false;                 // (1) link condition

    for (uint32_t w : nP) {                         // (2) seam-degree
        if (w == S || !seam[w]) continue;
        if (!nS.count(w)) return false;             // seam vert adjacent to P but not S
    }
    return true;
}

// Kill the seam sliver column: a corner-aligned grid puts the first +x vertex on
// a seam→interior edge at x=voxel·t, and when the iso-crossing sits near the seam
// corner (t→0) that vert is ~coincident with its seam vert → a near-zero-width
// tri. Weld each such near-seam +x vert onto the closest seam vert in its 1-ring
// (short edge only), then drop the tris that collapse to a line and compact.
// Runs on the +x half before reflection, so symmetry is preserved for free.
void weld_near_seam(Mesh& m, float voxel, std::vector<uint8_t>& seam) {
    float band     = voxel * 0.30f;          // x-distance that counts as "on the seam"
    float merge_sq = (voxel * 0.50f) * (voxel * 0.50f);  // only weld genuinely short P–S edges
    m.build_adjacency();
    uint32_t vc = m.vertex_count();

    std::vector<uint32_t> remap(vc);
    for (uint32_t v = 0; v < vc; v++) remap[v] = v;

    uint32_t welded = 0;
    for (uint32_t v = 0; v < vc; v++) {
        if (seam[v] || m.pos_x[v] >= band) continue;
        uint32_t best = 0xFFFFFFFFu; float best_sq = merge_sq;
        uint32_t s = m.vert_tri_offset[v], e = m.vert_tri_offset[v + 1];
        for (uint32_t j = s; j < e; j++) {
            uint32_t tri = m.vert_tri_list[j];
            for (int k = 0; k < 3; k++) {
                uint32_t nv = m.indices[tri*3+k];
                if (nv == v || !seam[nv]) continue;
                float dx=m.pos_x[v]-m.pos_x[nv], dy=m.pos_y[v]-m.pos_y[nv], dz=m.pos_z[v]-m.pos_z[nv];
                float d = dx*dx+dy*dy+dz*dz;
                if (d < best_sq) { best_sq = d; best = nv; }
            }
        }
        // H-C: only collapse when it stays manifold + keeps the seam loop intact;
        // an unsafe sliver is left for the post-reflect band weld / H-D gate.
        if (best != 0xFFFFFFFFu && collapse_is_seam_safe(m, seam, v, best)) {
            remap[v] = best; welded++;
        }
    }
    if (welded == 0) return;

    for (uint32_t& idx : m.indices) idx = remap[idx];

    // Drop tris that collapsed to a line/point (a welded edge).
    std::vector<uint32_t> keep;
    keep.reserve(m.indices.size());
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        uint32_t a=m.indices[t], b=m.indices[t+1], c=m.indices[t+2];
        if (a==b || b==c || a==c) continue;
        keep.push_back(a); keep.push_back(b); keep.push_back(c);
    }
    m.indices.swap(keep);
    compact_with_seam(m, seam);
    std::printf("[sdf-mirror] welded %u near-seam slivers\n", welded);
}

// H-B: validate (and lightly repair) the seam loop on the clipped +x half BEFORE
// reflection. The +x half is an open surface whose ONLY boundary may be the seam
// loop at x=0 — i.e. every boundary edge (an edge used by exactly 1 tri) must have
// both endpoints flagged `seam`. An escapee endpoint still physically on the plane
// (|x|<band) is snapped+flagged here, so reflect_across_x shares it and the seam
// closes by construction — caught earlier/cleaner than the post-reflect H-A band
// weld. A boundary endpoint genuinely off the plane is a real hole in the +x half:
// snapping can't fix it, so it's logged and left for the H-D gate to refuse.
// Acceptance: every boundary edge ends up with both endpoints `seam`.
// One-shot op — allocation is fine here (zero-alloc rule is stroke-only).
static void validate_seam_loop(Mesh& m, float voxel, std::vector<uint8_t>& seam) {
    const float band = voxel * 1e-3f;

    // Edge → incident-tri count (undirected, endpoints packed low/high into key).
    std::unordered_map<int64_t, uint32_t> ecount;
    ecount.reserve(m.indices.size());
    auto key = [](uint32_t a, uint32_t b) -> int64_t {
        if (a > b) std::swap(a, b);
        return ((int64_t)a << 32) | (int64_t)b;
    };
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        uint32_t a=m.indices[t], b=m.indices[t+1], c=m.indices[t+2];
        ecount[key(a,b)]++; ecount[key(b,c)]++; ecount[key(c,a)]++;
    }

    uint32_t snapped = 0;
    std::unordered_set<uint32_t> off_plane;
    float off_min = 1e30f, off_max = 0.0f;                   // |x|/voxel range of escapees
    for (auto& kv : ecount) {
        if (kv.second != 1) continue;                       // not a boundary edge
        uint32_t ends[2] = { (uint32_t)(kv.first >> 32),
                             (uint32_t)(kv.first & 0xffffffffu) };
        for (uint32_t v : ends) {
            if (seam[v]) continue;                          // already on the loop
            if (std::fabs(m.pos_x[v]) < band) {             // escapee still on plane
                m.pos_x[v] = 0.0f; seam[v] = 1; snapped++;
            } else {
                off_plane.insert(v);                        // boundary vert off x=0
                float d = std::fabs(m.pos_x[v]) / voxel;
                off_min = std::min(off_min, d); off_max = std::max(off_max, d);
            }
        }
    }
    if (snapped)
        std::printf("[sdf-mirror] seam validate: %u escapee boundary vert(s) snapped+flagged\n", snapped);
    if (!off_plane.empty())
        std::printf("[sdf-mirror] seam validate: WARNING %zu boundary vert(s) off the plane "
                    "(|x| = %.4f..%.4f voxel; band=%.0e voxel) — genuine hole in +x half (H-D will refuse)\n",
                    off_plane.size(), off_min, off_max, band / voxel);
}

} // namespace

// ============================================================================

// ---- Vertex-paint transfer (world-space nearest source vertex) ------------
// The voxel merge mints brand-new MC topology with no correspondence to the
// source meshes, so paint can't ride along an index map the way it does through
// remesh. Instead we reproject it: each output vertex copies the colour of the
// nearest SOURCE vertex in world space. A uniform grid (cell = voxel) over the
// source verts keeps it near-linear; the expanding Chebyshev-ring search stops
// once no unscanned cell can hold a closer vertex. One-shot op — heavier CPU is
// allowed (the no-readback / zero-alloc rules are stroke-only).
static void transfer_colors_world(Mesh& out,
                                  const std::vector<float>&    src_pos,  // 3 per vert
                                  const std::vector<uint32_t>& src_col,
                                  Vec3 origin, float cell, uint32_t R) {
    const uint32_t ns = (uint32_t)src_col.size();
    const uint32_t vc = out.vertex_count();
    out.color.assign(vc, 0xFFFFFFFFu);
    if (ns == 0 || cell <= 0.0f) return;

    const float inv = 1.0f / cell;
    auto cell_of = [&](float x, float y, float z, int& i, int& j, int& k) {
        i = (int)std::floor((x - origin.x) * inv);
        j = (int)std::floor((y - origin.y) * inv);
        k = (int)std::floor((z - origin.z) * inv);
    };
    // Grid spans the padded AABB plus slop for source verts outside it; bias keeps
    // the packed key positive across the whole range (R <= 256, pad small).
    auto key = [](int i, int j, int k) -> int64_t {
        return (int64_t)(uint32_t)(i + 4096)
             | ((int64_t)(uint32_t)(j + 4096) << 21)
             | ((int64_t)(uint32_t)(k + 4096) << 42);
    };

    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    grid.reserve(ns);
    for (uint32_t s = 0; s < ns; s++) {
        int i, j, k; cell_of(src_pos[s*3], src_pos[s*3+1], src_pos[s*3+2], i, j, k);
        grid[key(i, j, k)].push_back(s);
    }

    const int RMAX = (int)R + 4;
    for (uint32_t v = 0; v < vc; v++) {
        const float px = out.pos_x[v], py = out.pos_y[v], pz = out.pos_z[v];
        int ci, cj, ck; cell_of(px, py, pz, ci, cj, ck);
        float    best     = std::numeric_limits<float>::max();
        uint32_t best_col = 0xFFFFFFFFu;
        for (int r = 0; r <= RMAX; r++) {
            for (int dk = -r; dk <= r; dk++)
            for (int dj = -r; dj <= r; dj++)
            for (int di = -r; di <= r; di++) {
                // Chebyshev shell only — interior cells were scanned at smaller r.
                if (std::abs(di) != r && std::abs(dj) != r && std::abs(dk) != r) continue;
                auto it = grid.find(key(ci + di, cj + dj, ck + dk));
                if (it == grid.end()) continue;
                for (uint32_t s : it->second) {
                    const float dx = px - src_pos[s*3],
                                dy = py - src_pos[s*3+1],
                                dz = pz - src_pos[s*3+2];
                    const float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < best) { best = d2; best_col = src_col[s]; }
                }
            }
            // The nearest point of any cell at Chebyshev radius r+1 is >= r*cell
            // away, so once best <= (r*cell)^2 no farther shell can improve it.
            if (best < std::numeric_limits<float>::max()) {
                const float reach = (float)r * cell;
                if (reach * reach >= best) break;
            }
        }
        out.color[v] = best_col;
    }
}

// ---- Shared dispatch helper (file scope: tick uses it) ----------------------
// Dispatch `threads` logical invocations through the gpu:: seam, splitting the 1D
// workgroup count into a 2D grid past the 65535 per-dim limit (the SDF passes hit it
// at R>=128). The kernels recover their linear index from num_workgroups.x.
static void sdf_dispatch_seam(gpu::ComputeBatch& b, gpu::ComputePipeline& pipe,
                              gpu::BindGroup& grp, uint32_t threads) {
    uint32_t groups = (threads + 63u) / 64u;
    uint32_t gx = groups, gy = 1u;
    const uint32_t MAXG = 65535u;
    if (groups > MAXG) { gx = MAXG; gy = (groups + MAXG - 1u) / MAXG; }
    gpu::dispatch(b, pipe, grp, gx, gy);
}

// ---- Per-kernel std140 Params UBO payloads (binding 63) ---------------------
// Byte-identical to the anonymous Params block declared in each sdf_*.comp/.wgsl:
// vec3 origin fills a 16-byte slot, the following float (voxel) packs into bytes
// 12-15, and every block is padded to a 16-byte multiple.
struct SdfCountParamsGPU {
    float ox, oy, oz;     float voxel;       // origin.xyz (0,4,8), voxel (12)
    int32_t R;            int32_t band;      // R (16), band (20)
    uint32_t triCount;    uint32_t _pad0;    // triCount (24)
};
struct SdfExpandParamsGPU {
    float ox, oy, oz;     float voxel;
    int32_t R;            uint32_t triCount;
    uint32_t workCount;   uint32_t _pad0;
};
struct SdfSignParamsGPU {
    float ox, oy, oz;     float voxel;
    int32_t R;            uint32_t triCount;
    uint32_t cornerCount; float bandFar;
    uint32_t cornerOffset; float beta;     // FWN far-field acceptance (|q-p| > beta*radius ⇒ expand)
    uint32_t _pad1, _pad2;
};
struct SdfMeshParamsGPU {                    // shared by mc + nets (same uniforms)
    float ox, oy, oz;     float voxel;
    int32_t R;            uint32_t cellCount;
    int32_t count_only;   uint32_t cap;
};
static_assert(sizeof(SdfCountParamsGPU)  == 32, "");
static_assert(sizeof(SdfExpandParamsGPU) == 32, "");
static_assert(sizeof(SdfSignParamsGPU)   == 48, "");
static_assert(sizeof(SdfMeshParamsGPU)   == 32, "");

// Drain + report GL errors after a stage. In CHISEL_DEBUG, glFinish() first so a
// GPU hang/TDR is attributed to THIS dispatch (the last "stage OK" before a crash
// names the dispatch that hung) and report its wall time.
static void sdf_gl_check(const char* stage) {
#ifdef CHISEL_DEBUG
    auto t0 = std::chrono::steady_clock::now();
    glFinish();
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
#endif
    GLenum e;
    while ((e = glGetError()) != GL_NO_ERROR)
        std::printf("[sdf] GL error 0x%04x after %s\n", (unsigned)e, stage);
#ifdef CHISEL_DEBUG
    std::printf("[sdf][dbg] stage OK: %s (%.1f ms)\n", stage, ms);
    std::fflush(stdout);
#endif
}

// Cross-frame state for a tick-driven voxel merge. Heap-owned by the caller. Holds
// the GL resources so they survive between frames; the render loop clobbers GL
// *binding* state in between, so each tick re-establishes program + SSBO bindings
// + uniforms before dispatching. Phases advance one step per tick (Sign — the
// dominant pass — is budgeted across several ticks).
struct VoxelMergeJob {
    enum class Phase { Splat, Sign, Mesh, Finish, Done, Failed };
    Phase phase = Phase::Splat;

    bool     mirror = false;
    bool     use_nets = false;     // Surface Nets extractor chosen (vs Marching Cubes)
    SdfGrid  grid;
    int      band = 0;
    uint32_t tri_count    = 0;
    uint32_t corner_count = 0;
    uint32_t cell_count   = 0;
    uint32_t work_count   = 0;
    uint32_t out_tris     = 0;
    uint32_t sign_off     = 0;     // resumable winding-sign cursor (corner index)

    std::vector<float>    pos;     // source soup, kept for relax + paint transfer
    std::vector<uint32_t> vcol;    // per-source-vertex paint
    bool                  any_paint = false;

    std::vector<float> field_cpu;  // post-flood signed field (read once, reused by relax)
    std::vector<float> soup;       // MC output triangle soup (read once in Mesh)

    // SDF scratch SSBOs — seam-owned gpu::Buffers (buffer-ownership migration Step 5).
    // Created via gpu::create_buffer, bound to the bind groups directly (no view
    // fabrication). Data movement with no seam primitive (the far_bits clear on dist,
    // the mc_out in-place resize, the CPU readbacks) stays raw GL via .handle — a
    // web-stage concern (WebGPU: copy_buffer→MapRead staging + a bind-group rebuild
    // around the mc_out resize). Released explicitly in the destructor.
    gpu::Buffer soup_pos, soup_idx, dist, field, mc_out, mc_cnt, tritab, splat_box, splat_off;

    // Fast-winding-number tree (sign pass): node array + leaf-contiguous tri order.
    gpu::Buffer fwn_nodes, fwn_order;

    // Compute pipelines + their Params UBOs, on the gpu:: seam (Seam Step 2b).
    // mc_pipe is compiled from either sdf_mc or sdf_nets per use_nets. Pipelines/UBOs
    // are POD handles → released in the destructor.
    gpu::ComputePipeline count_pipe, expand_pipe, sign_pipe, mc_pipe;
    gpu::Buffer          count_ubo, expand_ubo, sign_ubo, mc_ubo;

    std::chrono::high_resolution_clock::time_point t0;
    VoxelMergeResult res;

    ~VoxelMergeJob() {
        gpu::release_compute_pipeline(count_pipe);
        gpu::release_compute_pipeline(expand_pipe);
        gpu::release_compute_pipeline(sign_pipe);
        gpu::release_compute_pipeline(mc_pipe);
        gpu::release_buffer(count_ubo);
        gpu::release_buffer(expand_ubo);
        gpu::release_buffer(sign_ubo);
        gpu::release_buffer(mc_ubo);
        gpu::release_buffer(soup_pos);  gpu::release_buffer(soup_idx);
        gpu::release_buffer(dist);      gpu::release_buffer(field);
        gpu::release_buffer(mc_out);    gpu::release_buffer(mc_cnt);
        gpu::release_buffer(tritab);    gpu::release_buffer(splat_box);
        gpu::release_buffer(splat_off);
        gpu::release_buffer(fwn_nodes); gpu::release_buffer(fwn_order);
    }
};

// Mirror the signed field about the x=0 corner layer by copying the kept +x half
// onto the -x half, so an extractor reads an EXACTLY symmetric field. Used by the
// Surface-Nets mirror path: SN has no vertices on the plane to seam (unlike MC), so
// instead of clip/reflect surgery we symmetrise the field and extract the whole
// surface — its vertices come out in exact mirror pairs, continuous through x=0.
// (The grid origin was already snapped so x=0 is a corner layer, hence seam is an
// integer corner index and the mirror is index-exact.)
static void symmetrise_field_x(std::vector<float>& field, const SdfGrid& grid) {
    const int R1   = (int)grid.R + 1;
    const int seam = (int)std::lround(-grid.origin.x / grid.voxel);  // x=0 corner index
    if (seam <= 0 || seam >= (int)grid.R) return;                    // plane off-grid: nothing to do
    for (int k = 0; k < R1; k++)
    for (int j = 0; j < R1; j++)
        for (int i = 0; i < seam; i++) {            // -x corner <- mirrored +x corner
            const int src = 2 * seam - i;
            if (src > (int)grid.R) continue;
            field[i + R1 * (j + R1 * k)] = field[src + R1 * (j + R1 * k)];
        }
}

VoxelMergeJob* voxel_merge_begin(Scene& scene, ComputeState& cs,
                                 int resolution, bool mirror, bool surface_nets,
                                 bool subtract) {
    VoxelMergeJob* job = new VoxelMergeJob();
    job->mirror = mirror;
    job->t0 = std::chrono::high_resolution_clock::now();

    auto fail = [&](const std::string& msg) -> VoxelMergeJob* {
        job->res.error = msg;
        job->phase = VoxelMergeJob::Phase::Failed;
        return job;
    };

    if (!cs.supported) return fail("GPU compute unavailable");

    // ---- 1. Gather soup ----
    std::vector<uint32_t> idx;
    uint32_t n_additive = 0, n_cutter = 0;
    size_t   additive_floats = 0;
    job->res.in_entities = gather_soup(scene, job->pos, idx, job->vcol, job->any_paint,
                                       subtract, n_additive, n_cutter, additive_floats);
    job->res.in_tris     = (uint32_t)(idx.size() / 3);
    if (idx.empty()) return fail("selection has no triangles");
    if (subtract && n_additive == 0)
        return fail("subtract needs at least one selected (kept) mesh");
    const uint32_t tri_count = job->res.in_tris;
    job->tri_count = tri_count;

    // ---- 2. Grid (cubic, padded so the surface never touches the wall) ----
    // Bound to the ADDITIVE (kept) verts: the carve region lives inside them, and
    // cutters poking outside still sum into the winding correctly without stealing
    // resolution. (additive_floats == pos.size() for a plain union.)
    const std::vector<float>& pos = job->pos;
    Vec3 lo(pos[0], pos[1], pos[2]);
    Vec3 hi = lo;
    for (size_t i = 0; i < additive_floats; i += 3) {
        lo.x = std::min(lo.x, pos[i  ]); hi.x = std::max(hi.x, pos[i  ]);
        lo.y = std::min(lo.y, pos[i+1]); hi.y = std::max(hi.y, pos[i+1]);
        lo.z = std::min(lo.z, pos[i+2]); hi.z = std::max(hi.z, pos[i+2]);
    }
    // Mirror mode symmetrises about the app's mirror plane (world x=0): widen the
    // x window to [-X, +X] so the kept +x half spans the whole result.
    if (mirror) {
        float X = std::max(std::fabs(lo.x), std::fabs(hi.x));
        lo.x = -X; hi.x = X;
    }

    int R = std::max(16, std::min(resolution, 256));
    float ext = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
    if (!(ext > 0.0f)) return fail("degenerate selection bounds");

    SdfGrid& grid = job->grid;
    grid.R     = (uint32_t)R;
    grid.voxel = ext / (float)(R - 2 * GRID_PAD);     // R-2*PAD cells span the AABB
    grid.origin = lo - Vec3(grid.voxel, grid.voxel, grid.voxel) * (float)GRID_PAD;
    job->res.R = grid.R;

    // Sign band thickness is a constant ABSOLUTE width, not a constant voxel count:
    // the flood fill assumes the band is a closed separating shell, so at a fixed
    // voxel count it thins in world units as R grows and leaks through thin features.
    // Anchor BAND_DILATE voxels at R=128 and scale; floor at BAND_DILATE.
    job->band = std::max(BAND_DILATE,
        (int)std::lround((double)BAND_DILATE * (R - 2 * GRID_PAD) / (128 - 2 * GRID_PAD)));

    // Snap the x origin so a corner layer lands exactly on x=0 (shift < half voxel),
    // so every cell sits wholly on one side and clip_to_plus_x gives a clean seam.
    if (mirror) {
        int seam_i = (int)std::lround(-grid.origin.x / grid.voxel);
        seam_i = std::max(0, std::min(seam_i, (int)grid.R));
        grid.origin.x = -(float)seam_i * grid.voxel;
    }

    job->corner_count = grid.corners();
    job->cell_count   = grid.cells();

    // ---- 3. Allocate SSBOs + upload soup (seam-owned gpu::Buffers) ----
    using gpu::Usage;
    const uint32_t zero = 0u;
    // Source soup (uploaded at create).
    job->soup_pos = gpu::create_buffer(cs.gpu_dev, pos.data(), pos.size()*sizeof(float),  Usage::Storage);
    job->soup_idx = gpu::create_buffer(cs.gpu_dev, idx.data(), idx.size()*sizeof(uint32_t), Usage::Storage);

    // Fast-winding-number tree for the sign pass (replaces the brute all-tris sum).
    std::vector<FwnNodeGPU> fwn_nodes;
    std::vector<uint32_t>   fwn_order;
    build_fwn_tree(pos, idx, fwn_nodes, fwn_order);
    job->fwn_nodes = gpu::create_buffer(cs.gpu_dev, fwn_nodes.data(),
                                        fwn_nodes.size()*sizeof(FwnNodeGPU), Usage::Storage);
    job->fwn_order = gpu::create_buffer(cs.gpu_dev, fwn_order.data(),
                                        fwn_order.size()*sizeof(uint32_t),  Usage::Storage);

    // Load-balanced splat scratch: per-tri voxel box (6 uints: g0.xyz+span.xyz) +
    // the exclusive scan of footprints (offset). box is GPU-written then read back
    // for a CPU scan (CopySrc); offset is uploaded back for the expand pass's owner search.
    job->splat_box = gpu::create_buffer(cs.gpu_dev, nullptr, (uint64_t)tri_count*6*sizeof(uint32_t), Usage::Storage | Usage::CopySrc);
    job->splat_off = gpu::create_buffer(cs.gpu_dev, nullptr, (uint64_t)tri_count*sizeof(uint32_t),   Usage::Storage);

    // Packed-distance accumulator: atomicMin target in expand, read by sign (GPU only,
    // never read back). Cleared to BAND_FAR's bit pattern via the seam clear primitive
    // (non-zero fill word: CPU-upload on WebGPU, ClearBuffer-with-pattern on GL).
    job->dist = gpu::create_buffer(cs.gpu_dev, nullptr, (uint64_t)job->corner_count*sizeof(uint32_t), Usage::Storage);
    uint32_t far_bits;
    { float bf = BAND_FAR; std::memcpy(&far_bits, &bf, sizeof(float)); }
    gpu::clear_buffer(cs.gpu_dev, job->dist, far_bits);

    // Signed field: written by sign, read back for the flood fill, uploaded back, read by MC (CopySrc).
    job->field = gpu::create_buffer(cs.gpu_dev, nullptr, (uint64_t)job->corner_count*sizeof(float), Usage::Storage | Usage::CopySrc);

    // Extractor output-triangle counter (atomic): read back to size mc_out, then reset (CopySrc).
    job->mc_cnt = gpu::create_buffer(cs.gpu_dev, &zero, sizeof(uint32_t), Usage::Storage | Usage::CopySrc);

    // MC/Nets triangle soup output: tiny placeholder, resized in-place after the count pass (CopySrc).
    job->mc_out = gpu::create_buffer(cs.gpu_dev, nullptr, sizeof(float)*18, Usage::Storage | Usage::CopySrc);

    // Static MC case table.
    job->tritab = gpu::create_buffer(cs.gpu_dev, MC_TRI_TABLE, sizeof(MC_TRI_TABLE), Usage::Storage);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ---- Compile pipelines (gpu:: seam) ----
    using gpu::BindEntry; using gpu::Bind;
    const BindEntry count_layout[] = {
        { BIND_SDF_SOUP_POS,  Bind::StorageRead,      0 },
        { BIND_SDF_SOUP_IDX,  Bind::StorageRead,      0 },
        { BIND_SDF_SPLAT_BOX, Bind::StorageReadWrite, 0 },  // writeonly box[]
        { BIND_PARAMS,        Bind::Uniform,          sizeof(SdfCountParamsGPU) },
    };
    job->count_pipe = gpu::create_compute_pipeline(cs.gpu_dev,
        gpu::embedded_shader("sdf_count"), count_layout, 4);

    const BindEntry expand_layout[] = {
        { BIND_SDF_SOUP_POS,     Bind::StorageRead,      0 },
        { BIND_SDF_SOUP_IDX,     Bind::StorageRead,      0 },
        { BIND_SDF_DIST,         Bind::StorageReadWrite, 0 },  // atomicMin
        { BIND_SDF_SPLAT_BOX,    Bind::StorageRead,      0 },
        { BIND_SDF_SPLAT_OFFSET, Bind::StorageRead,      0 },
        { BIND_PARAMS,           Bind::Uniform,          sizeof(SdfExpandParamsGPU) },
    };
    job->expand_pipe = gpu::create_compute_pipeline(cs.gpu_dev,
        gpu::embedded_shader("sdf_expand"), expand_layout, 6);

    const BindEntry sign_layout[] = {
        { BIND_SDF_SOUP_POS,    Bind::StorageRead,      0 },
        { BIND_SDF_SOUP_IDX,    Bind::StorageRead,      0 },
        { BIND_SDF_DIST,        Bind::StorageRead,      0 },  // read (written atomically in expand)
        { BIND_SDF_FIELD,       Bind::StorageReadWrite, 0 },
        { BIND_SDF_FWN_NODES,   Bind::StorageRead,      0 },  // FWN tree
        { BIND_SDF_FWN_TRIORDER,Bind::StorageRead,      0 },  // leaf tri order
        { BIND_PARAMS,          Bind::Uniform,          sizeof(SdfSignParamsGPU) },
    };
    job->sign_pipe = gpu::create_compute_pipeline(cs.gpu_dev,
        gpu::embedded_shader("sdf_sign"), sign_layout, 7);

    // Iso-surface extractor: Surface Nets (smoother, quad-dominant) or MC (default,
    // manifold-robust). Either way it drives the same mc_pipe two-pass dispatch in
    // the Mesh phase, so nothing downstream of here cares which one was compiled.
    // Mirror handling differs by extractor. MC lands vertices exactly on the x=0
    // corner layer, so the mirror path keeps the +x half and reflects it (the
    // clip_to_plus_x / reflect / H-D seam machinery). Surface Nets puts vertices at
    // cell centres — none sit on the plane — so that surgery can't seam it. Instead
    // the SN mirror path symmetrises the FIELD about x=0 in the Mesh phase and
    // extracts the whole thing: vertices come out in exact mirror pairs, continuous
    // through the plane, no seam to close. (Handled in Mesh/Finish via use_nets.)
    const bool use_nets = surface_nets;
    job->use_nets       = use_nets;
    if (use_nets) {
        const BindEntry nets_layout[] = {
            { BIND_SDF_FIELD,    Bind::StorageRead,      0 },
            { BIND_SDF_MC_OUT,   Bind::StorageReadWrite, 0 },
            { BIND_SDF_MC_COUNT, Bind::StorageReadWrite, 0 },  // atomic counter
            { BIND_PARAMS,       Bind::Uniform,          sizeof(SdfMeshParamsGPU) },
        };
        job->mc_pipe = gpu::create_compute_pipeline(cs.gpu_dev,
            gpu::embedded_shader("sdf_nets"), nets_layout, 4);
    } else {
        const BindEntry mc_layout[] = {
            { BIND_SDF_FIELD,    Bind::StorageRead,      0 },
            { BIND_SDF_MC_OUT,   Bind::StorageReadWrite, 0 },
            { BIND_SDF_MC_COUNT, Bind::StorageReadWrite, 0 },  // atomic counter
            { BIND_SDF_TRITABLE, Bind::StorageRead,      0 },
            { BIND_PARAMS,       Bind::Uniform,          sizeof(SdfMeshParamsGPU) },
        };
        job->mc_pipe = gpu::create_compute_pipeline(cs.gpu_dev,
            gpu::embedded_shader("sdf_mc"), mc_layout, 5);
    }
    if (!job->count_pipe.handle || !job->expand_pipe.handle ||
        !job->sign_pipe.handle  || !job->mc_pipe.handle)
        return fail("SDF shader compile failed (see stderr)");

    // Persistent Params UBOs (one per kernel; mc + nets share the SdfMeshParamsGPU layout).
    job->count_ubo  = gpu::create_buffer(cs.gpu_dev, nullptr, sizeof(SdfCountParamsGPU),  gpu::Usage::Uniform);
    job->expand_ubo = gpu::create_buffer(cs.gpu_dev, nullptr, sizeof(SdfExpandParamsGPU), gpu::Usage::Uniform);
    job->sign_ubo   = gpu::create_buffer(cs.gpu_dev, nullptr, sizeof(SdfSignParamsGPU),   gpu::Usage::Uniform);
    job->mc_ubo     = gpu::create_buffer(cs.gpu_dev, nullptr, sizeof(SdfMeshParamsGPU),   gpu::Usage::Uniform);

    sdf_gl_check("pre-merge (stale)");   // drain inherited errors so attribution is clean
    job->phase = VoxelMergeJob::Phase::Splat;
    return job;
}

VoxelMergeStatus voxel_merge_tick(Scene& scene, ComputeState& cs,
                                  VoxelMergeJob& j, VoxelMergeResult& out) {
    // Pipelines were compiled in begin; tick dispatches them through cs.gpu_dev.
    using Phase = VoxelMergeJob::Phase;
    const SdfGrid& grid = j.grid;
    const uint32_t zero = 0u;

    auto done = [&](VoxelMergeStatus s) { out = j.res; return s; };
    auto fail = [&](const std::string& msg) {
        j.res.error = msg;
        j.phase = Phase::Failed;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glUseProgram(0);
        out = j.res;
        return VoxelMergeStatus::Failed;
    };

    switch (j.phase) {
    case Phase::Splat: {

        // ---- Pass A (count): per-tri voxel box, degenerates -> 0 ----
        // GL-owned SSBOs wrapped in transient gpu::Buffer views; the seam's per-dispatch
        // + submit barriers are a superset of the old explicit glMemoryBarrier.
        SdfCountParamsGPU cu = {};
        cu.ox = grid.origin.x; cu.oy = grid.origin.y; cu.oz = grid.origin.z;
        cu.voxel = grid.voxel; cu.R = (int)grid.R; cu.band = j.band; cu.triCount = j.tri_count;
        gpu::write_buffer(cs.gpu_dev, j.count_ubo, 0, &cu, sizeof(cu));

        const gpu::BindBufferEntry cbg[] = {
            { BIND_SDF_SOUP_POS,  &j.soup_pos,  j.soup_pos.size },
            { BIND_SDF_SOUP_IDX,  &j.soup_idx,  j.soup_idx.size },
            { BIND_SDF_SPLAT_BOX, &j.splat_box, j.splat_box.size },
            { BIND_PARAMS,        &j.count_ubo, sizeof(SdfCountParamsGPU) },
        };
        gpu::BindGroup cgrp = gpu::create_bind_group(cs.gpu_dev, j.count_pipe, cbg, 4);
        {
            gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
            sdf_dispatch_seam(b, j.count_pipe, cgrp, j.tri_count);
            gpu::submit(b);
        }
        gpu::release_bind_group(cgrp);
        sdf_gl_check("count dispatch");

        // ---- Pass B (exclusive scan, CPU): read boxes, scan footprints to offsets,
        //      get total work items. On WebGPU this becomes a GPU scan + mapAsync. ----
        std::vector<uint32_t> box_cpu((size_t)j.tri_count*6), off_cpu(j.tri_count);
        gpu::read_buffer(cs.gpu_dev, j.splat_box, 0,
                         (uint64_t)j.tri_count*6*sizeof(uint32_t), box_cpu.data());
        uint64_t acc = 0;
        for (uint32_t t = 0; t < j.tri_count; t++) {
            uint64_t footprint = (uint64_t)box_cpu[6*t+3] * box_cpu[6*t+4] * box_cpu[6*t+5];
            off_cpu[t] = (uint32_t)acc;
            acc += footprint;
        }
        j.work_count = (uint32_t)acc;          // total voxel-touches across all tris

        // Hard guard: the expand pass's owner search needs off[] strictly monotonic.
        // A uint32 work-count overflow wraps off[] mid-scan -> garbage owner -> OOB
        // atomicMin into dist[] (the old non-deterministic crash). Bail cleanly.
        if (acc > 0xFFFFFFFFull) {
            char ebuf[176];
            std::snprintf(ebuf, sizeof(ebuf),
                "SDF splat overflow: %llu voxel-touches > 2^32 (tris=%u R=%d) - lower merge resolution",
                (unsigned long long)acc, j.tri_count, grid.R);
            std::printf("[sdf] %s\n", ebuf);
            return fail(ebuf);
        }
#ifdef CHISEL_DEBUG
        {
            // Validate every counted box stays inside [0,R]^3 (free: box_cpu is on CPU).
            uint32_t bad_box = 0, max_idx = 0;
            for (uint32_t t = 0; t < j.tri_count; t++) {
                uint32_t sx = box_cpu[6*t+3];
                if (sx == 0) continue;
                uint32_t hi_i = box_cpu[6*t  ] + sx            - 1;
                uint32_t hi_j = box_cpu[6*t+1] + box_cpu[6*t+4] - 1;
                uint32_t hi_k = box_cpu[6*t+2] + box_cpu[6*t+5] - 1;
                if (hi_i > (uint32_t)grid.R || hi_j > (uint32_t)grid.R || hi_k > (uint32_t)grid.R) bad_box++;
                uint32_t m = std::max(hi_i, std::max(hi_j, hi_k));
                if (m > max_idx) max_idx = m;
            }
            std::printf("[sdf][dbg] tris=%u corners=%u cells=%u R=%d band=%d | work=%u acc=%llu | box max-axis-idx=%u out-of-range=%u\n",
                        j.tri_count, j.corner_count, j.cell_count, grid.R, j.band,
                        j.work_count, (unsigned long long)acc, max_idx, bad_box);
            if (bad_box)
                std::printf("[sdf][dbg] WARNING %u boxes exceed [0,R] - expand pass writes OOB into dist[]\n", bad_box);
        }
#endif
        gpu::write_buffer(cs.gpu_dev, j.splat_off, 0, off_cpu.data(),
                          (uint64_t)j.tri_count*sizeof(uint32_t));

        // ---- Pass C (expand + splat): one thread per work item ----
        if (j.work_count > 0u) {
            SdfExpandParamsGPU eu = {};
            eu.ox = grid.origin.x; eu.oy = grid.origin.y; eu.oz = grid.origin.z;
            eu.voxel = grid.voxel; eu.R = (int)grid.R;
            eu.triCount = j.tri_count; eu.workCount = j.work_count;
            gpu::write_buffer(cs.gpu_dev, j.expand_ubo, 0, &eu, sizeof(eu));

            const gpu::BindBufferEntry ebg[] = {
                { BIND_SDF_SOUP_POS,     &j.soup_pos,  j.soup_pos.size },
                { BIND_SDF_SOUP_IDX,     &j.soup_idx,  j.soup_idx.size },
                { BIND_SDF_DIST,         &j.dist,      j.dist.size },
                { BIND_SDF_SPLAT_BOX,    &j.splat_box, j.splat_box.size },
                { BIND_SDF_SPLAT_OFFSET, &j.splat_off, j.splat_off.size },
                { BIND_PARAMS,           &j.expand_ubo, sizeof(SdfExpandParamsGPU) },
            };
            gpu::BindGroup egrp = gpu::create_bind_group(cs.gpu_dev, j.expand_pipe, ebg, 6);
            {
                gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
                sdf_dispatch_seam(b, j.expand_pipe, egrp, j.work_count);
                gpu::submit(b);
            }
            gpu::release_bind_group(egrp);
            sdf_gl_check("expand dispatch");
        }

        j.sign_off = 0;
        j.phase = Phase::Sign;
        return done(VoxelMergeStatus::Working);
    }

    case Phase::Sign: {
        // Winding sign + field (one thread per corner). The single heaviest pass
        // (O(band_corners * tris), seconds at R>=128) -> budgeted across frames, a
        // few 32k-corner slices per tick. Each slice is its own ring submission via
        // glFlush so no GPU job approaches the watchdog; the window keeps pumping.
        SdfSignParamsGPU su = {};
        su.ox = grid.origin.x; su.oy = grid.origin.y; su.oz = grid.origin.z;
        su.voxel = grid.voxel; su.R = (int)grid.R;
        su.triCount = j.tri_count; su.cornerCount = j.corner_count; su.bandFar = BAND_FAR;
        su.beta = FWN_BETA;

        const gpu::BindBufferEntry sbg[] = {
            { BIND_SDF_SOUP_POS,     &j.soup_pos,  j.soup_pos.size },
            { BIND_SDF_SOUP_IDX,     &j.soup_idx,  j.soup_idx.size },
            { BIND_SDF_DIST,         &j.dist,      j.dist.size },
            { BIND_SDF_FIELD,        &j.field,     j.field.size },
            { BIND_SDF_FWN_NODES,    &j.fwn_nodes, j.fwn_nodes.size },
            { BIND_SDF_FWN_TRIORDER, &j.fwn_order, j.fwn_order.size },
            { BIND_PARAMS,           &j.sign_ubo,  sizeof(SdfSignParamsGPU) },
        };
        gpu::BindGroup sgrp = gpu::create_bind_group(cs.gpu_dev, j.sign_pipe, sbg, 7);
        const uint32_t SIGN_SLICE      = 32768u;  // worst-case all-band slice ~ <1s on Vega 8
        const uint32_t SLICES_PER_TICK = 2u;      // tune: smaller = smoother window, more ticks.
                                                  // Leaning small on purpose — user prefers a
                                                  // responsive low-FPS window over a multi-second
                                                  // freeze; never let one tick block for seconds.
        for (uint32_t s = 0; s < SLICES_PER_TICK && j.sign_off < j.corner_count; s++) {
            uint32_t n = std::min(SIGN_SLICE, j.corner_count - j.sign_off);
            su.cornerOffset = j.sign_off;       // only field that changes per slice
            gpu::write_buffer(cs.gpu_dev, j.sign_ubo, 0, &su, sizeof(su));
            gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
            sdf_dispatch_seam(b, j.sign_pipe, sgrp, n);
            gpu::submit(b);
            glFlush();   // separate ring submission per slice (own watchdog window)
            j.sign_off += n;
        }
        gpu::release_bind_group(sgrp);
        if (j.sign_off >= j.corner_count) {
            sdf_gl_check("sign dispatch (chunked)");
            j.phase = Phase::Mesh;
        }
        return done(VoxelMergeStatus::Working);
    }

    case Phase::Mesh: {
        // ---- Off-band sign flood (CPU): read field, propagate inside/outside into
        //      the off-band bulk so MC sees no false crossing, upload for MC. Keep the
        //      CPU copy for the relax in Finish (field SSBO isn't written after this,
        //      so one readback serves both). ----
        j.field_cpu.assign(j.corner_count, 0.0f);
        gpu::read_buffer(cs.gpu_dev, j.field, 0,
                         (uint64_t)j.corner_count*sizeof(float), j.field_cpu.data());
        flood_fill_sign(j.field_cpu, grid.R, BAND_FAR);
        // Surface-Nets mirror: make the field exactly symmetric here, so the SN
        // dispatch below extracts a continuous mirror-paired surface (no seam to
        // close). MC's mirror path leaves the field as-is and seams at extraction.
        if (j.mirror && j.use_nets)
            symmetrise_field_x(j.field_cpu, grid);
        gpu::write_buffer(cs.gpu_dev, j.field, 0, j.field_cpu.data(),
                          (uint64_t)j.corner_count*sizeof(float));

        // ---- Extractor bind group (one, reused across both passes; mc_out is realloc'd
        //      in place between them so its GL id — what the GL backend records — stays
        //      valid). Surface Nets omits the tri-table binding. ----
        SdfMeshParamsGPU mu = {};
        mu.ox = grid.origin.x; mu.oy = grid.origin.y; mu.oz = grid.origin.z;
        mu.voxel = grid.voxel; mu.R = (int)grid.R; mu.cellCount = j.cell_count;

        // One bind group per extractor pass. mc_out is resized between the count and
        // write passes via the seam resize_buffer: on GL the handle is preserved, but
        // on WebGPU a resize RECREATES the buffer (new handle), so the bind group that
        // references mc_out must be rebuilt after the resize. Factor creation into a
        // lambda used for both builds. Surface Nets omits the tri-table binding.
        auto build_mgrp = [&]() -> gpu::BindGroup {
            if (j.use_nets) {
                const gpu::BindBufferEntry mbg[] = {
                    { BIND_SDF_FIELD,    &j.field,  j.field.size },
                    { BIND_SDF_MC_OUT,   &j.mc_out, j.mc_out.size },
                    { BIND_SDF_MC_COUNT, &j.mc_cnt, j.mc_cnt.size },
                    { BIND_PARAMS,       &j.mc_ubo, sizeof(SdfMeshParamsGPU) },
                };
                return gpu::create_bind_group(cs.gpu_dev, j.mc_pipe, mbg, 4);
            }
            const gpu::BindBufferEntry mbg[] = {
                { BIND_SDF_FIELD,    &j.field,  j.field.size },
                { BIND_SDF_MC_OUT,   &j.mc_out, j.mc_out.size },
                { BIND_SDF_MC_COUNT, &j.mc_cnt, j.mc_cnt.size },
                { BIND_SDF_TRITABLE, &j.tritab, j.tritab.size },
                { BIND_PARAMS,       &j.mc_ubo, sizeof(SdfMeshParamsGPU) },
            };
            return gpu::create_bind_group(cs.gpu_dev, j.mc_pipe, mbg, 5);
        };
        gpu::BindGroup mgrp = build_mgrp();

        // ---- Dispatch 3a: extractor count pass ----
        mu.count_only = 1; mu.cap = 0u;
        gpu::write_buffer(cs.gpu_dev, j.mc_ubo, 0, &mu, sizeof(mu));
        {
            gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
            sdf_dispatch_seam(b, j.mc_pipe, mgrp, j.cell_count);
            gpu::submit(b);
        }
        sdf_gl_check("mc-count dispatch");

        uint32_t out_tris = 0;
        gpu::read_buffer(cs.gpu_dev, j.mc_cnt, 0, sizeof(uint32_t), &out_tris);
        if (out_tris == 0) { gpu::release_bind_group(mgrp); return fail("merge produced no geometry (raise resolution?)"); }
        j.out_tris = out_tris;

        // ---- Size MC_OUT exactly via the seam resize, then rebuild mgrp (the write
        //      pass binds the possibly-recreated mc_out — see the bind-group note above)
        //      and reset the counter via the seam. ----
        gpu::resize_buffer(cs.gpu_dev, j.mc_out, (uint64_t)out_tris * 18 * sizeof(float),
                           gpu::Usage::Storage | gpu::Usage::CopySrc);
        gpu::release_bind_group(mgrp);
        mgrp = build_mgrp();
        gpu::write_buffer(cs.gpu_dev, j.mc_cnt, 0, &zero, sizeof(uint32_t));

        // ---- Dispatch 3b: extractor write pass ----
        mu.count_only = 0; mu.cap = out_tris;
        gpu::write_buffer(cs.gpu_dev, j.mc_ubo, 0, &mu, sizeof(mu));
        {
            gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
            sdf_dispatch_seam(b, j.mc_pipe, mgrp, j.cell_count);
            gpu::submit(b);
        }
        gpu::release_bind_group(mgrp);
        sdf_gl_check("mc-write dispatch");

        // ---- Readback the MC soup (the only large readback). On WebGPU -> mapAsync. ----
        j.soup.assign((size_t)out_tris * 18, 0.0f);
        gpu::read_buffer(cs.gpu_dev, j.mc_out, 0,
                         (uint64_t)j.soup.size()*sizeof(float), j.soup.data());

        j.phase = Phase::Finish;
        return done(VoxelMergeStatus::Working);
    }

    case Phase::Finish: {
        // ---- Hash-weld -> watertight indexed Mesh ----
        // Tight snap (voxel*1e-5): the MC shader canonicalizes edge orientation so a
        // shared edge yields a bit-identical vertex from both cells; a looser tol only
        // risks false fusion of a distinct nearby sheet into a non-manifold edge.
        Mesh welded;
        hash_weld(j.soup, j.out_tris, grid.voxel * 1e-5f, welded);

#ifdef CHISEL_DEBUG
        {
            uint32_t b=0, nm=0, comp=0;
            manifold_report(welded, b, nm, comp);
            std::printf("[sdf][dbg] raw MC weld: %u tris, %u verts | boundary_edges=%u nonmanifold=%u components=%u\n",
                        welded.tri_count(), welded.vertex_count(), b, nm, comp);
        }
#endif

        // ---- Relax the MC triangulation onto the (already read-back) signed field ----
        // Spreads the blocky/uneven MC tris to uniform spacing while holding the
        // silhouette. Reuses j.field_cpu from the Mesh phase (no second readback).
        if (j.mirror && !j.use_nets) {
            // MC keep-+x-and-reflect: relax that half alone (seam pinned), then
            // mirror it -> both sides identical, exact partner map, no seam fight.
            std::vector<uint8_t> seam;
            clip_to_plus_x(welded, grid.voxel, seam);
            weld_near_seam(welded, grid.voxel, seam);   // B: collapse the seam sliver column
            relax_to_field(welded, j.field_cpu, grid, /*iters=*/8, /*lambda=*/0.5f, &seam);  // A: 1D seam relax
            validate_seam_loop(welded, grid.voxel, seam);  // H-B: close seam escapees before reflecting
            reflect_across_x(welded, seam);
            weld_seam_band(welded, grid.voxel);   // H-A: weld any unflagged escapees onto x=0
        } else {
            // Faithful merge, OR the SN mirror path: the field was already made
            // symmetric (Mesh phase) so the full extracted surface is mirror-paired
            // and continuous — a plain relax keeps that symmetry (Jacobi + symmetric
            // field => mirror-paired verts move identically). No seam surgery.
            relax_to_field(welded, j.field_cpu, grid, /*iters=*/8, /*lambda=*/0.5f);
        }

        welded.build_adjacency();
        welded.recompute_normals();   // clean per-vertex normals (GPU normals were orientation-only)

        // Reproject vertex paint from the source meshes (skipped if nothing painted).
        if (j.any_paint)
            transfer_colors_world(welded, j.pos, j.vcol, grid.origin, grid.voxel, grid.R);

        j.res.out_tris  = welded.tri_count();
        j.res.out_verts = welded.vertex_count();

        // ---- Manifold report (surfaced so a bad merge is visible before print) ----
        manifold_report(welded, j.res.boundary_edges, j.res.nonmanifold_edges, j.res.components);

        // ---- H-D: gate the splice on the report (MC mirror path only). A reflected
        //      MC merge MUST be watertight by construction; any boundary/non-manifold
        //      edge means the seam failed to close -> refuse rather than detonate
        //      downstream. SN is exempt: like the faithful path it tolerates the
        //      naive-SN two-sheet non-manifold edges (MC remains the print default). ----
        if (j.mirror && !j.use_nets && (j.res.boundary_edges != 0 || j.res.nonmanifold_edges != 0)) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "mirror seam did not close (%u bnd, %u nonmf) - merge refused "
                          "to avoid corrupt topology",
                          j.res.boundary_edges, j.res.nonmanifold_edges);
            return fail(buf);
        }

        // ---- Splice into the scene as the merged entity ----
        uint32_t merged_id = scene.merge_selected_into(welded, 0);
        if (merged_id == 0) return fail("scene splice failed");

        auto t1 = std::chrono::high_resolution_clock::now();
        j.res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - j.t0).count();
        j.res.success = true;
        j.phase = Phase::Done;
        return done(VoxelMergeStatus::Done);
    }

    case Phase::Done:   return done(VoxelMergeStatus::Done);
    case Phase::Failed: return done(VoxelMergeStatus::Failed);
    }
    return done(VoxelMergeStatus::Working);  // unreachable
}

float voxel_merge_progress(const VoxelMergeJob& j) {
    switch (j.phase) {
        case VoxelMergeJob::Phase::Splat:  return 0.05f;
        case VoxelMergeJob::Phase::Sign:   return j.corner_count
                                               ? 0.10f + 0.70f * ((float)j.sign_off / (float)j.corner_count)
                                               : 0.10f;
        case VoxelMergeJob::Phase::Mesh:   return 0.85f;
        case VoxelMergeJob::Phase::Finish: return 0.95f;
        case VoxelMergeJob::Phase::Done:   return 1.0f;
        case VoxelMergeJob::Phase::Failed: return 1.0f;
    }
    return 0.0f;
}

void voxel_merge_destroy(VoxelMergeJob* job) { delete job; }

// Synchronous convenience: drive the tick job to completion in one (blocking) call.
VoxelMergeResult voxel_merge_selected(Scene& scene, ComputeState& cs,
                                      int resolution, bool mirror, bool surface_nets,
                                      bool subtract) {
    VoxelMergeJob* job = voxel_merge_begin(scene, cs, resolution, mirror, surface_nets, subtract);
    VoxelMergeResult out;
    while (voxel_merge_tick(scene, cs, *job, out) == VoxelMergeStatus::Working) {}
    voxel_merge_destroy(job);
    return out;
}
