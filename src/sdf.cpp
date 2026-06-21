#include "sdf.h"
#include "scene.h"
#include "mesh_entity.h"
#include "compute.h"
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
// Concatenate world-space triangles of every selected entity into one soup.
// Entities are baked in world space (twins are real tris), so no transform.
// Returns the number of entities gathered.
uint32_t gather_soup(const Scene& scene,
                     std::vector<float>&    pos,
                     std::vector<uint32_t>& idx,
                     std::vector<uint32_t>& vcol,  // per-source-vertex colour (white if unpainted)
                     bool&                  any_paint)
{
    uint32_t n_ent = 0;
    any_paint = false;
    for (uint32_t id : scene.selected_ids()) {
        const MeshEntity* e = scene.find_entity(id);
        if (!e) continue;
        const Mesh& m = e->mesh;
        if (m.vertex_count() == 0 || m.indices.empty()) continue;

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
        for (uint32_t i : m.indices) idx.push_back(base + i);
        n_ent++;
    }
    return n_ent;
}

// ---- Shader sources -------------------------------------------------------

// Ericson closest-point-on-triangle, returns Euclidean distance. Degenerate
// (zero-area) tris are filtered before the splat, so the face denom is safe.
const char* PT_TRI_DIST = R"(
float pt_tri_dist(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return length(ap);
    vec3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return length(bp);
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        float v = d1 / (d1 - d3);
        return length(p - (a + v*ab));
    }
    vec3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return length(cp);
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        float w = d2 / (d2 - d6);
        return length(p - (a + w*ac));
    }
    float va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(p - (b + w*(c - b)));
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    return length(p - (a + ab*v + ac*w));
}
)";

// Degenerate-triangle predicate + soup vertex fetch. Injected into the count,
// expand AND winding shaders so a degenerate tri (zero area / non-finite vertex)
// reaches neither distance nor winding evaluation. Needs only p[]/ix[] (Pos/Idx).
const char* SDF_DEGEN_SRC = R"(
bool nonfinite(vec3 v) { return any(isnan(v)) || any(isinf(v)); }
// Zero-area (covers fully-collapsed edges) or any non-finite vertex.
bool tri_degenerate(vec3 a, vec3 b, vec3 c) {
    if (nonfinite(a) || nonfinite(b) || nonfinite(c)) return true;
    return length(cross(b - a, c - a)) < 1e-20;
}
vec3 tri_vert(uint t, uint k) {           // vertex k (0..2) of triangle t, world space
    uint v = ix[3u*t + k];
    return vec3(p[3u*v], p[3u*v+1u], p[3u*v+2u]);
}
)";

// Clamped, band-dilated corner-index AABB [g0,g1] (inclusive) for a triangle.
// Injected into BOTH the count (Pass A) and expand (Pass C) shaders so they agree
// bit-for-bit on the per-triangle voxel footprint. If they disagreed, the expand
// pass would unravel a work-item to the wrong voxel and write out of bounds
// (load-balancing spec, invariant 1). Needs u_origin/u_voxel/u_R/u_band.
const char* SDF_AABB_SRC = R"(
void tri_band_aabb(vec3 a, vec3 b, vec3 c, out ivec3 g0, out ivec3 g1) {
    vec3 lo = min(min(a,b),c), hi = max(max(a,b),c);
    g0 = ivec3(floor((lo - u_origin)/u_voxel)) - u_band;
    g1 = ivec3(ceil ((hi - u_origin)/u_voxel)) + u_band;
    g0 = clamp(g0, ivec3(0), ivec3(u_R));
    g1 = clamp(g1, ivec3(0), ivec3(u_R));
}
)";

// Linear work index across a 2D workgroup grid. All SDF passes dispatch through
// dispatch_linear() on the host, which splits the group count into X*Y when it
// exceeds the 65535 per-dimension GL_MAX_COMPUTE_WORK_GROUP_COUNT limit (hit at
// R=128+ for the expanded splat, and R=256 for the per-corner/per-cell passes).
const char* SDF_GID_SRC = R"(
uint linear_gid() {
    return gl_GlobalInvocationID.x
         + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
}
)";

// Pass A — count: one thread per triangle. Compute the clamped band-AABB ONCE and
// store it (g0.xyz + span.xyz) into box[] so Pass C can reuse the exact integers
// instead of recomputing the float math — if the two passes recomputed and a 1-ULP
// rounding difference flipped a floor() at a voxel boundary, Pass C would unravel a
// work item out of the counted box and write dist[] out of bounds (load-balancing
// spec, invariant 1 → the non-deterministic crash). Degenerate tris store span 0 →
// zero work items, skipped by the expand + winding passes.
const char* COUNT_SRC = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 21) readonly  buffer Pos { float p[]; };
layout(std430, binding = 22) readonly  buffer Idx { uint  ix[]; };
layout(std430, binding = 36) writeonly buffer Box { uint  box[]; };  // 6 per tri
uniform vec3  u_origin;  uniform float u_voxel;  uniform int u_R;
uniform int   u_band;    uniform uint  u_triCount;
)" /* SDF_DEGEN_SRC + SDF_AABB_SRC injected here */ R"(
void main() {
    uint t = linear_gid();
    if (t >= u_triCount) return;
    uint o = 6u*t;
    vec3 a = tri_vert(t,0u), b = tri_vert(t,1u), c = tri_vert(t,2u);
    if (tri_degenerate(a,b,c)) {
        box[o]=0u; box[o+1u]=0u; box[o+2u]=0u; box[o+3u]=0u; box[o+4u]=0u; box[o+5u]=0u;
        return;
    }
    ivec3 g0, g1; tri_band_aabb(a,b,c,g0,g1);
    uvec3 span = uvec3(g1 - g0 + ivec3(1));      // g0 clamped to [0,R] ⇒ non-negative
    box[o]=uint(g0.x); box[o+1u]=uint(g0.y); box[o+2u]=uint(g0.z);
    box[o+3u]=span.x;  box[o+4u]=span.y;     box[o+5u]=span.z;
}
)";

// Pass C — expand + splat: one thread per work item. Binary-search the owner
// triangle in the exclusive scan off[], read its stored box, unravel the local
// index into a voxel, evaluate one distance, atomic-min into the field. Writes
// every AABB cell (no per-voxel band predicate) to stay bit-identical to the old
// thread-per-triangle splat — the band is decided downstream in the winding pass
// via the BAND_FAR sentinel, exactly as before.
const char* EXPAND_SRC = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 21) readonly buffer Pos { float p[]; };
layout(std430, binding = 22) readonly buffer Idx { uint  ix[]; };
layout(std430, binding = 23)          buffer Dist{ uint  d[]; };
layout(std430, binding = 36) readonly buffer Box { uint  box[]; };
layout(std430, binding = 37) readonly buffer Off { uint  off[]; };
uniform vec3  u_origin;  uniform float u_voxel;  uniform int u_R;
uniform uint  u_triCount;  uniform uint u_workCount;
)" /* SDF_DEGEN_SRC + pt_tri_dist injected here */ R"(
void main() {
    uint gid = linear_gid();
    if (gid >= u_workCount) return;
    // Owner = upper_bound(off, gid) - 1 over [0, u_triCount): first i with off[i] > gid.
    uint lo = 0u, hi = u_triCount;
    while (lo < hi) { uint mid = (lo + hi) >> 1u; if (off[mid] <= gid) lo = mid + 1u; else hi = mid; }
    uint t = lo - 1u;
    uint j = gid - off[t];                      // local work index within triangle t
    uint o = 6u*t;
    ivec3 g0   = ivec3(int(box[o]), int(box[o+1u]), int(box[o+2u]));
    uvec3 span = uvec3(box[o+3u], box[o+4u], box[o+5u]);
    uint dx =  j % span.x;
    uint dy = (j / span.x) % span.y;
    uint dz =  j / (span.x * span.y);
    int gi = g0.x + int(dx), gj = g0.y + int(dy), gk = g0.z + int(dz);
    vec3 a = tri_vert(t,0u), b = tri_vert(t,1u), c = tri_vert(t,2u);
    vec3 q = u_origin + u_voxel*vec3(gi,gj,gk);
    float dd = pt_tri_dist(q, a, b, c);
    int R1 = u_R + 1;
    uint ci = uint(gi + R1*(gj + R1*gk));
    atomicMin(d[ci], floatBitsToUint(dd));       // monotonic for dd >= 0
}
)";

// Winding sign — evaluated ONLY at band corners (the surface shell, ~O(R^2)).
// Off-band corners are deferred: marked with |f| == u_bandFar so the CPU flood
// fill can assign their inside/outside sign. This is the spec's spartan
// optimisation: the O(corners*tris) brute force is infeasible on weak GPUs, but
// MC only interpolates band-to-band edges, so off-band corners just need a
// consistent sign — never a winding number. (sdf-remesh-spec.md gotcha 3.)
const char* SIGN_SRC = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 21) readonly buffer Pos  { float p[]; };
layout(std430, binding = 22) readonly buffer Idx  { uint  ix[]; };
layout(std430, binding = 23) readonly buffer Dist { uint  d[]; };
layout(std430, binding = 24)          buffer Field{ float f[]; };
uniform vec3 u_origin; uniform float u_voxel; uniform int u_R;
uniform uint u_triCount; uniform uint u_cornerCount; uniform float u_bandFar;
uniform uint u_cornerOffset;   // chunked dispatch: base corner index of this slice
const float FOUR_PI = 12.566370614359172;
)" /* SDF_DEGEN_SRC injected here */ R"(
void main() {
    uint ci = linear_gid() + u_cornerOffset;
    if (ci >= u_cornerCount) return;
    float dist = uintBitsToFloat(d[ci]);
    if (dist >= u_bandFar) { f[ci] = u_bandFar; return; }   // off-band → CPU flood

    int R1 = u_R + 1;
    int i =  int(ci) % R1;
    int j = (int(ci) / R1) % R1;
    int k =  int(ci) / (R1*R1);
    vec3 q = u_origin + u_voxel*vec3(i,j,k);

    float omega = 0.0;
    for (uint t = 0u; t < u_triCount; t++) {
        vec3 va = tri_vert(t,0u), vb = tri_vert(t,1u), vc = tri_vert(t,2u);
        if (tri_degenerate(va,vb,vc)) continue;   // keep degenerates out of the sign sum
        vec3 A = va - q, B = vb - q, C = vc - q;
        float la=length(A), lb=length(B), lc=length(C);
        float num = dot(A, cross(B,C));
        float den = la*lb*lc + dot(A,B)*lc + dot(B,C)*la + dot(C,A)*lb;
        omega += 2.0 * atan(num, den);
    }
    float w = omega / FOUR_PI;
    f[ci] = (w > 0.5 ? -1.0 : 1.0) * dist;          // signed real distance (band)
}
)";

const char* MC_SRC = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 24) readonly buffer Field { float f[]; };
layout(std430, binding = 25)          buffer McOut { float o[]; };   // 18 floats / tri
layout(std430, binding = 26)          buffer McCnt { uint  cnt[]; };
layout(std430, binding = 27) readonly buffer TriTb { int   tri[]; }; // 256*16
uniform vec3 u_origin; uniform float u_voxel; uniform int u_R;
uniform uint u_cellCount; uniform int u_count_only; uniform uint u_cap;

const ivec3 CORNER[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,1,0), ivec3(0,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(1,1,1), ivec3(0,1,1));
const ivec2 EDGE[12] = ivec2[12](
    ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),
    ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),
    ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7));

float fld(ivec3 c, int R1) {
    c = clamp(c, ivec3(0), ivec3(u_R));
    return f[c.x + R1*(c.y + R1*c.z)];
}
vec3 grad(ivec3 c, int R1) {
    return vec3(fld(c+ivec3(1,0,0),R1) - fld(c-ivec3(1,0,0),R1),
                fld(c+ivec3(0,1,0),R1) - fld(c-ivec3(0,1,0),R1),
                fld(c+ivec3(0,0,1),R1) - fld(c-ivec3(0,0,1),R1));
}

void main() {
    uint cell = linear_gid();
    if (cell >= u_cellCount) return;
    int R1 = u_R + 1;
    int cx =  int(cell) % u_R;
    int cy = (int(cell) / u_R) % u_R;
    int cz =  int(cell) / (u_R*u_R);
    ivec3 base = ivec3(cx, cy, cz);

    float val[8];
    int cubeindex = 0;
    for (int s = 0; s < 8; s++) {
        ivec3 c = base + CORNER[s];
        val[s] = f[c.x + R1*(c.y + R1*c.z)];
        if (val[s] < 0.0) cubeindex |= (1 << s);
    }
    if (cubeindex == 0 || cubeindex == 255) return;

    int row = cubeindex * 16;

    // Count pass: the triangle count is fully determined by the case table — no need
    // to interpolate edge positions/normals (the expensive grad() field reads) just
    // to throw them away, and no need to hit the global counter once per triangle.
    // Tally locally, reserve once. Halves the count pass's field traffic and collapses
    // millions of single-address atomics into one per active cell — the R=256 TDR fix.
    if (u_count_only == 1) {
        uint ntri = 0u;
        for (int i = 0; i < 15 && tri[row+i] != -1; i += 3) ntri++;
        if (ntri > 0u) atomicAdd(cnt[0], ntri);
        return;
    }

    vec3 vpos[12];
    vec3 vnrm[12];
    for (int e = 0; e < 12; e++) {
        int a = EDGE[e].x, b = EDGE[e].y;
        float fa = val[a], fb = val[b];
        if ((fa < 0.0) == (fb < 0.0)) continue;   // edge not crossed
        ivec3 ca = base + CORNER[a];
        ivec3 cb = base + CORNER[b];
        // Canonicalize the edge by its GLOBAL corner coords so both cells sharing this
        // edge interpolate from the identical endpoint order. fa/(fa-fb) and fb/(fb-fa)
        // are equal in real arithmetic but NOT bit-identical in float, and the large
        // integer corner coords (up to R) amplify the difference to ~1e-5 voxel — enough
        // to split the shared vertex into two and leave a crack the weld can't close at a
        // tight snap. Sorting the endpoints makes the vertex bit-identical from both cells.
        if (ca.x != cb.x ? ca.x > cb.x
          : ca.y != cb.y ? ca.y > cb.y
          :                ca.z > cb.z) {
            ivec3 tc = ca; ca = cb; cb = tc;
            float tf = fa; fa = fb; fb = tf;
        }
        float tt = fa / (fa - fb);
        vpos[e] = u_origin + u_voxel * (vec3(ca) + tt * vec3(cb - ca));
        vec3 nn = mix(grad(ca, R1), grad(cb, R1), tt);
        vnrm[e] = (dot(nn,nn) > 0.0) ? normalize(nn) : vec3(0.0, 0.0, 1.0);
    }

    for (int i = 0; i < 15 && tri[row+i] != -1; i += 3) {
        int e0 = tri[row+i], e1 = tri[row+i+1], e2 = tri[row+i+2];
        vec3 p0 = vpos[e0], p1 = vpos[e1], p2 = vpos[e2];
        vec3 n0 = vnrm[e0], n1 = vnrm[e1], n2 = vnrm[e2];
        // Orient so face winding matches the field gradient (outward).
        if (dot(cross(p1 - p0, p2 - p0), n0 + n1 + n2) < 0.0) {
            vec3 tp = p1; p1 = p2; p2 = tp;
            vec3 tn = n1; n1 = n2; n2 = tn;
        }
        uint slot = atomicAdd(cnt[0], 1u);
        if (slot >= u_cap) continue;   // overflow guard (exact in practice)
        uint b0 = slot * 18u;
        o[b0+0u]=p0.x;  o[b0+1u]=p0.y;  o[b0+2u]=p0.z;  o[b0+3u]=n0.x;  o[b0+4u]=n0.y;  o[b0+5u]=n0.z;
        o[b0+6u]=p1.x;  o[b0+7u]=p1.y;  o[b0+8u]=p1.z;  o[b0+9u]=n1.x;  o[b0+10u]=n1.y; o[b0+11u]=n1.z;
        o[b0+12u]=p2.x; o[b0+13u]=p2.y; o[b0+14u]=p2.z; o[b0+15u]=n2.x; o[b0+16u]=n2.y; o[b0+17u]=n2.z;
    }
}
)";

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

// Small RAII for the throwaway GL objects.
struct GlBuf {
    GLuint id = 0;
    void gen() { if (!id) glGenBuffers(1, &id); }
    ~GlBuf() { if (id) glDeleteBuffers(1, &id); }
};
struct GlProg {
    GLuint id = 0;
    ~GlProg() { if (id) glDeleteProgram(id); }
};

void set_grid_uniforms(GLuint prog, const SdfGrid& g) {
    glUniform3f(glGetUniformLocation(prog, "u_origin"), g.origin.x, g.origin.y, g.origin.z);
    glUniform1f(glGetUniformLocation(prog, "u_voxel"), g.voxel);
    glUniform1i(glGetUniformLocation(prog, "u_R"), (int)g.R);
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

VoxelMergeResult voxel_merge_selected(Scene& scene, ComputeState& cs,
                                      int resolution, bool mirror) {
    VoxelMergeResult res;
    if (!cs.supported) { res.error = "GPU compute unavailable"; return res; }

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- 1. Gather soup ----
    std::vector<float>    pos;
    std::vector<uint32_t> idx;
    std::vector<uint32_t> vcol;          // per-source-vertex paint, for world-space transfer
    bool                  any_paint = false;
    res.in_entities = gather_soup(scene, pos, idx, vcol, any_paint);
    res.in_tris     = (uint32_t)(idx.size() / 3);
    if (idx.empty()) { res.error = "selection has no triangles"; return res; }
    uint32_t tri_count = res.in_tris;

    // ---- 2. Grid (cubic, padded so the surface never touches the wall) ----
    Vec3 lo( pos[0], pos[1], pos[2]);
    Vec3 hi = lo;
    for (size_t i = 0; i < pos.size(); i += 3) {
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
    if (!(ext > 0.0f)) { res.error = "degenerate selection bounds"; return res; }

    SdfGrid grid;
    grid.R     = (uint32_t)R;
    grid.voxel = ext / (float)(R - 2 * GRID_PAD);     // R-2*PAD cells span the AABB
    grid.origin = lo - Vec3(grid.voxel, grid.voxel, grid.voxel) * (float)GRID_PAD;
    res.R = grid.R;

    // Sign band thickness must be a constant ABSOLUTE width, not a constant voxel
    // count. The winding-sign winding number is only evaluated at band corners; off-band
    // corners get their sign by flood fill, which assumes the band is a closed separating
    // shell between inside and outside. At a fixed BAND_DILATE voxel count the band shrinks
    // in world units as R grows, so at high R it no longer separates the two surfaces of a
    // thin feature → the flood fill leaks → torn iso-surface (boundary_edges>0). Anchor the
    // absolute thickness to the value that works at R=128 (BAND_DILATE voxels there) and
    // scale the voxel count with resolution. Floor at BAND_DILATE: thicker is only slower
    // (the sign pass is chunked, so any duration is watchdog-safe), never wrong; thinner
    // would risk leaks at lower R too.
    const int band = std::max(BAND_DILATE,
        (int)std::lround((double)BAND_DILATE * (R - 2 * GRID_PAD) / (128 - 2 * GRID_PAD)));

    // Snap the x origin so a corner layer lands exactly on x=0 (shift < ½ voxel).
    // Then every cell sits wholly on one side — clip_to_plus_x gives a clean seam.
    if (mirror) {
        int seam_i = (int)std::lround(-grid.origin.x / grid.voxel);
        seam_i = std::max(0, std::min(seam_i, (int)grid.R));
        grid.origin.x = -(float)seam_i * grid.voxel;
    }

    uint32_t corner_count = grid.corners();
    uint32_t cell_count   = grid.cells();

    // ---- 3. Allocate SSBOs + upload soup ----
    GlBuf soup_pos, soup_idx, dist, field, mc_out, mc_cnt, tritab, splat_box, splat_off;
    soup_pos.gen(); soup_idx.gen(); dist.gen(); field.gen();
    mc_out.gen();   mc_cnt.gen();   tritab.gen();
    splat_box.gen(); splat_off.gen();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, soup_pos.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, pos.size()*sizeof(float), pos.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, soup_idx.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, idx.size()*sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

    // Load-balanced splat scratch: per-tri voxel box (6 uints: g0.xyz+span.xyz) +
    // the exclusive scan of footprints (offset). box is GPU-written then read back
    // for a CPU scan; offset is uploaded back for the expand pass's owner search.
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, splat_box.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)tri_count*6*sizeof(uint32_t), nullptr, GL_DYNAMIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, splat_off.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)tri_count*sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, dist.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)corner_count*sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    uint32_t far_bits;
    { float bf = BAND_FAR; std::memcpy(&far_bits, &bf, sizeof(float)); }
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &far_bits);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, field.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)corner_count*sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_cnt.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    uint32_t zero = 0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_out.id);   // tiny placeholder; resized after count pass
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*18, nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tritab.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(MC_TRI_TABLE), MC_TRI_TABLE, GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ---- Compile programs ----
    // Helper: build a program from a base source, injecting the given includes
    // just before main(). Mirrors the original Ericson-helper injection.
    auto build_prog = [&](const char* base, std::initializer_list<const char*> includes) -> GLuint {
        std::string src(base);
        size_t mpos = src.find("void main()");
        for (const char* inc : includes) { src.insert(mpos, inc); mpos += std::strlen(inc); }
        return cs.compile_program(src.c_str());
    };
    GlProg count_prog, expand_prog, sign_prog, mc_prog;
    count_prog.id  = build_prog(COUNT_SRC,  { SDF_GID_SRC, SDF_DEGEN_SRC, SDF_AABB_SRC });
    expand_prog.id = build_prog(EXPAND_SRC, { SDF_GID_SRC, SDF_DEGEN_SRC, PT_TRI_DIST });
    sign_prog.id   = build_prog(SIGN_SRC,   { SDF_GID_SRC, SDF_DEGEN_SRC });
    mc_prog.id     = build_prog(MC_SRC,     { SDF_GID_SRC });
    if (!count_prog.id || !expand_prog.id || !sign_prog.id || !mc_prog.id) {
        res.error = "SDF shader compile failed (see stderr)";
        return res;
    }

    // All SDF passes dispatch through this: 1D logical thread count, split into a
    // 2D workgroup grid when it would exceed the 65535 per-dim workgroup limit.
    // Shaders recover the linear index via linear_gid() (SDF_GID_SRC).
    auto dispatch_linear = [](uint32_t threads) {
        uint32_t groups = (threads + 63u) / 64u;
        uint32_t gx = groups, gy = 1u;
        const uint32_t MAXG = 65535u;
        if (groups > MAXG) { gx = MAXG; gy = (groups + MAXG - 1u) / MAXG; }
        glDispatchCompute(gx, gy, 1u);
    };

    // One-shot diagnostic: drain + report GL errors after a stage. The merge is a
    // modal op (not a stroke), so the implicit sync from glGetError costs nothing.
    // In debug, glFinish() first so a GPU hang/TDR is attributed to THIS dispatch
    // (otherwise the async queue surfaces the context loss several calls later) —
    // the last "stage OK" printed before a crash names the dispatch that hung.
    auto gl_check = [](const char* stage) {
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
    };
    gl_check("pre-merge (stale)");   // drain anything inherited so attribution is clean

    // ---- Dispatch 1: load-balanced distance splat ----
    // Pass A (count): compute + store each triangle's voxel box, degenerates → 0.
    glUseProgram(count_prog.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_POS,  soup_pos.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_IDX,  soup_idx.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SPLAT_BOX, splat_box.id);
    set_grid_uniforms(count_prog.id, grid);
    glUniform1i (glGetUniformLocation(count_prog.id, "u_band"),     band);
    glUniform1ui(glGetUniformLocation(count_prog.id, "u_triCount"), tri_count);
    dispatch_linear(tri_count);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    gl_check("count dispatch");

    // Pass B (exclusive scan, CPU): read the boxes, scan footprints (span product)
    // to offsets, get total work items. The merge already reads back (it's one-shot,
    // not a stroke), so a CPU scan is in budget; the WebGPU port swaps this for a
    // GPU scan + mapAsync.
    std::vector<uint32_t> box_cpu((size_t)tri_count*6), off_cpu(tri_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, splat_box.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       (GLsizeiptr)tri_count*6*sizeof(uint32_t), box_cpu.data());
    uint64_t acc = 0;
    for (uint32_t t = 0; t < tri_count; t++) {
        uint64_t footprint = (uint64_t)box_cpu[6*t+3] * box_cpu[6*t+4] * box_cpu[6*t+5];
        off_cpu[t] = (uint32_t)acc;
        acc += footprint;
    }
    uint32_t work_count = (uint32_t)acc;          // total voxel-touches across all tris

    // Hard guard (permanent, not debug-only): the expand pass's owner search needs
    // off[] strictly monotonic. A uint32 work-count overflow wraps off[] mid-scan →
    // a work item resolves to a garbage owner triangle → out-of-bounds atomicMin into
    // dist[]. That's the non-deterministic crash. Bail cleanly before touching the GPU.
    if (acc > 0xFFFFFFFFull) {
        char ebuf[176];
        std::snprintf(ebuf, sizeof(ebuf),
            "SDF splat overflow: %llu voxel-touches > 2^32 (tris=%u R=%d) — lower merge resolution",
            (unsigned long long)acc, tri_count, grid.R);
        res.error = ebuf;
        std::printf("[sdf] %s\n", ebuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return res;
    }
#ifdef CHISEL_DEBUG
    {
        // Validate every counted box stays inside the corner grid [0,R]^3 — a box that
        // exceeds R makes the expand pass index dist[] out of bounds. Free: box_cpu is
        // already on the CPU. (sx==0 ⇒ degenerate / zero-footprint, contributes no work.)
        uint32_t bad_box = 0, max_idx = 0;
        for (uint32_t t = 0; t < tri_count; t++) {
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
                    tri_count, corner_count, cell_count, grid.R, band,
                    work_count, (unsigned long long)acc, max_idx, bad_box);
        if (bad_box)
            std::printf("[sdf][dbg] WARNING %u boxes exceed [0,R] — expand pass writes OOB into dist[]\n", bad_box);
    }
#endif
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, splat_off.id);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)tri_count*sizeof(uint32_t), off_cpu.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Pass C (expand + splat): one thread per work item → uniform per-thread load.
    if (work_count > 0u) {
        glUseProgram(expand_prog.id);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_POS,     soup_pos.id);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_IDX,     soup_idx.id);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_DIST,         dist.id);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SPLAT_BOX,    splat_box.id);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SPLAT_OFFSET, splat_off.id);
        set_grid_uniforms(expand_prog.id, grid);
        glUniform1ui(glGetUniformLocation(expand_prog.id, "u_triCount"),  tri_count);
        glUniform1ui(glGetUniformLocation(expand_prog.id, "u_workCount"), work_count);
        dispatch_linear(work_count);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        gl_check("expand dispatch");
    }

    // ---- Dispatch 2: winding sign + field (one thread per corner) ----
    glUseProgram(sign_prog.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_POS, soup_pos.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_IDX, soup_idx.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_DIST,     dist.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_FIELD,    field.id);
    set_grid_uniforms(sign_prog.id, grid);
    glUniform1ui(glGetUniformLocation(sign_prog.id, "u_triCount"),    tri_count);
    glUniform1ui(glGetUniformLocation(sign_prog.id, "u_cornerCount"), corner_count);
    glUniform1f (glGetUniformLocation(sign_prog.id, "u_bandFar"),     BAND_FAR);
    // The winding sign is O(band_corners * tris) — the single heaviest pass. As one
    // dispatch it ran ~10s at R=128 and tripped the GPU watchdog (ring gfx timeout →
    // soft recovery → context lost; the soft-recovered glFinish then falsely reported
    // the *next* dispatch as the culprit). Split it into corner slices, each its own
    // ring submission via glFlush, so no single GPU job exceeds the watchdog. Total
    // work / wall time is unchanged; only the submission granularity changes.
    GLint sign_off_loc = glGetUniformLocation(sign_prog.id, "u_cornerOffset");
    const uint32_t SIGN_SLICE = 32768u;   // worst-case all-band slice ≈ <1s on Vega 8
    for (uint32_t off = 0; off < corner_count; off += SIGN_SLICE) {
        uint32_t n = std::min(SIGN_SLICE, corner_count - off);
        glUniform1ui(sign_off_loc, off);
        dispatch_linear(n);
        glFlush();   // force a separate ring submission per slice (own watchdog window)
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    gl_check("sign dispatch (chunked)");

    // ---- Off-band sign flood (CPU): the winding pass only signed band corners;
    //      propagate the inside/outside bit into the off-band bulk so MC never
    //      sees a false crossing at the band boundary. One readback + one upload. ----
    std::vector<float> field_cpu(corner_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, field.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       (GLsizeiptr)corner_count*sizeof(float), field_cpu.data());
    flood_fill_sign(field_cpu, grid.R, BAND_FAR);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)corner_count*sizeof(float), field_cpu.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ---- Dispatch 3a: marching cubes — count pass ----
    glUseProgram(mc_prog.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_FIELD,    field.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_MC_OUT,   mc_out.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_MC_COUNT, mc_cnt.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_TRITABLE, tritab.id);
    set_grid_uniforms(mc_prog.id, grid);
    glUniform1ui(glGetUniformLocation(mc_prog.id, "u_cellCount"),  cell_count);
    glUniform1i (glGetUniformLocation(mc_prog.id, "u_count_only"), 1);
    glUniform1ui(glGetUniformLocation(mc_prog.id, "u_cap"),        0u);
    dispatch_linear(cell_count);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    gl_check("mc-count dispatch");

    uint32_t out_tris = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_cnt.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &out_tris);
    if (out_tris == 0) {
        res.error = "merge produced no geometry (raise resolution?)";
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return res;
    }

    // ---- Size MC_OUT exactly, reset counter ----
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_out.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)out_tris * 18 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_cnt.id);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ---- Dispatch 3b: marching cubes — write pass ----
    glUseProgram(mc_prog.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_FIELD,    field.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_MC_OUT,   mc_out.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_MC_COUNT, mc_cnt.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_TRITABLE, tritab.id);
    glUniform1i (glGetUniformLocation(mc_prog.id, "u_count_only"), 0);
    glUniform1ui(glGetUniformLocation(mc_prog.id, "u_cap"),        out_tris);
    dispatch_linear(cell_count);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    gl_check("mc-write dispatch");

    // ---- Readback (the only readback) ----
    std::vector<float> soup(out_tris * 18);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_out.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       (GLsizeiptr)soup.size()*sizeof(float), soup.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    // ---- Hash-weld → watertight indexed Mesh ----
    Mesh welded;
    // Tight snap (voxel*1e-5, was 1e-3): now that the MC shader canonicalizes edge
    // orientation, a shared edge produces a bit-identical vertex from both adjacent cells,
    // so legit shared verts weld at an essentially exact key. The old 1e-3 tolerance only
    // added risk of FALSE fusion — pulling a distinct nearby sheet onto a welded pair into
    // a non-manifold edge (shared by >2 tris). 1e-5 is far below any real feature spacing
    // yet far above float noise, so it welds the real shares and fuses nothing spurious.
    hash_weld(soup, out_tris, grid.voxel * 1e-5f, welded);

#ifdef CHISEL_DEBUG
    // Is the raw MC weld watertight, BEFORE any mirror clip/relax? Isolates whether a
    // hole originates in MC/field/flood (here) or in the seam pipeline (downstream).
    {
        uint32_t b=0, nm=0, comp=0;
        manifold_report(welded, b, nm, comp);
        std::printf("[sdf][dbg] raw MC weld: %u tris, %u verts | boundary_edges=%u nonmanifold=%u components=%u\n",
                    welded.tri_count(), welded.vertex_count(), b, nm, comp);
    }
#endif

    // ---- Read back the signed field, relax the MC triangulation onto it ----
    // Spreads the blocky/uneven MC tris (and dissolves the cell-corner slivers)
    // to uniform spacing while holding the silhouette exactly. Second readback,
    // one-shot like the soup above — the no-readback rule is stroke-only.
    {
        std::vector<float> field_cpu(corner_count);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, field.id);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           (GLsizeiptr)field_cpu.size()*sizeof(float), field_cpu.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        if (mirror) {
            // Keep +x, relax that half alone (seam pinned), then mirror it. Both
            // sides end up identical → exact partner map, no smooth seam fight.
            std::vector<uint8_t> seam;
            clip_to_plus_x(welded, grid.voxel, seam);
            weld_near_seam(welded, grid.voxel, seam);   // B: collapse the seam sliver column
            relax_to_field(welded, field_cpu, grid, /*iters=*/8, /*lambda=*/0.5f, &seam);  // A: 1D seam relax
            validate_seam_loop(welded, grid.voxel, seam);  // H-B: close seam escapees before reflecting
            reflect_across_x(welded, seam);
            weld_seam_band(welded, grid.voxel);   // H-A: weld any unflagged ±ε escapees onto x=0
        } else {
            relax_to_field(welded, field_cpu, grid, /*iters=*/8, /*lambda=*/0.5f);
        }
    }

    welded.build_adjacency();
    welded.recompute_normals();   // clean per-vertex normals (GPU normals were for orientation only)

    // Reproject vertex paint from the source meshes onto the merged surface.
    // Skipped entirely when nothing in the selection was painted (result stays
    // unpainted → renders white, no cost). World-space nearest-vertex transfer
    // is symmetric for free when the source paint is symmetric.
    if (any_paint)
        transfer_colors_world(welded, pos, vcol, grid.origin, grid.voxel, grid.R);

    res.out_tris  = welded.tri_count();
    res.out_verts = welded.vertex_count();

    // ---- Manifold report (surfaced so a bad merge is visible before print) ----
    manifold_report(welded, res.boundary_edges, res.nonmanifold_edges, res.components);

    // ---- H-D: gate the splice on the report (mirror path) ----
    // A closed mirror-merged mesh MUST be watertight by construction (the +x half
    // reflected across x=0). So in mirror mode any boundary or non-manifold edge
    // means the seam failed to close — corrupt topology that crashes/garbles the
    // next remesh or voxel-merge (the remesher's EdgeTable mangles such input).
    // Until the seam-repair passes (H-A/H-B) land, refuse-and-surface rather than
    // let it into the scene to detonate downstream. The scene is untouched until
    // the splice below, so returning here leaves the selection intact.
    if (mirror && (res.boundary_edges != 0 || res.nonmanifold_edges != 0)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "mirror seam did not close (%u bnd, %u nonmf) — merge refused "
                      "to avoid corrupt topology",
                      res.boundary_edges, res.nonmanifold_edges);
        res.error = buf;
        return res;
    }

    // ---- Splice into the scene as the merged entity ----
    uint32_t merged_id = scene.merge_selected_into(welded, 0);
    if (merged_id == 0) { res.error = "scene splice failed"; return res; }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.success = true;
    return res;
}
