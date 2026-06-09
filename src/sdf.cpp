#include "sdf.h"
#include "scene.h"
#include "mesh_entity.h"
#include "compute.h"
#include "mc_tables.h"
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

const char* SPLAT_SRC = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 21) readonly buffer Pos { float p[]; };
layout(std430, binding = 22) readonly buffer Idx { uint  ix[]; };
layout(std430, binding = 23)          buffer Dist{ uint  d[]; };
uniform vec3  u_origin;  uniform float u_voxel;  uniform int u_R;
uniform int   u_band;    uniform uint  u_triCount;
)" /* pt_tri_dist injected here */ R"(
void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_triCount) return;
    vec3 a = vec3(p[3u*ix[3u*t  ]], p[3u*ix[3u*t  ]+1u], p[3u*ix[3u*t  ]+2u]);
    vec3 b = vec3(p[3u*ix[3u*t+1u]], p[3u*ix[3u*t+1u]+1u], p[3u*ix[3u*t+1u]+2u]);
    vec3 c = vec3(p[3u*ix[3u*t+2u]], p[3u*ix[3u*t+2u]+1u], p[3u*ix[3u*t+2u]+2u]);
    if (length(cross(b - a, c - a)) < 1e-20) return;   // skip degenerate tris
    vec3 lo = min(min(a,b),c), hi = max(max(a,b),c);
    ivec3 g0 = ivec3(floor((lo - u_origin)/u_voxel)) - u_band;
    ivec3 g1 = ivec3(ceil ((hi - u_origin)/u_voxel)) + u_band;
    g0 = clamp(g0, ivec3(0), ivec3(u_R));
    g1 = clamp(g1, ivec3(0), ivec3(u_R));
    int R1 = u_R + 1;
    for (int k=g0.z;k<=g1.z;k++)
    for (int j=g0.y;j<=g1.y;j++)
    for (int i=g0.x;i<=g1.x;i++) {
        vec3 q = u_origin + u_voxel*vec3(i,j,k);
        float dd = pt_tri_dist(q, a, b, c);
        uint  ci = uint(i + R1*(j + R1*k));
        atomicMin(d[ci], floatBitsToUint(dd));   // monotonic for dd >= 0
    }
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
const float FOUR_PI = 12.566370614359172;

void main() {
    uint ci = gl_GlobalInvocationID.x;
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
        vec3 A = vec3(p[3u*ix[3u*t  ]], p[3u*ix[3u*t  ]+1u], p[3u*ix[3u*t  ]+2u]) - q;
        vec3 B = vec3(p[3u*ix[3u*t+1u]], p[3u*ix[3u*t+1u]+1u], p[3u*ix[3u*t+1u]+2u]) - q;
        vec3 C = vec3(p[3u*ix[3u*t+2u]], p[3u*ix[3u*t+2u]+1u], p[3u*ix[3u*t+2u]+2u]) - q;
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

int g_R1;
float fld(ivec3 c) {
    c = clamp(c, ivec3(0), ivec3(u_R));
    return f[c.x + g_R1*(c.y + g_R1*c.z)];
}
vec3 grad(ivec3 c) {
    return vec3(fld(c+ivec3(1,0,0)) - fld(c-ivec3(1,0,0)),
                fld(c+ivec3(0,1,0)) - fld(c-ivec3(0,1,0)),
                fld(c+ivec3(0,0,1)) - fld(c-ivec3(0,0,1)));
}

void main() {
    uint cell = gl_GlobalInvocationID.x;
    if (cell >= u_cellCount) return;
    g_R1 = u_R + 1;
    int cx =  int(cell) % u_R;
    int cy = (int(cell) / u_R) % u_R;
    int cz =  int(cell) / (u_R*u_R);
    ivec3 base = ivec3(cx, cy, cz);

    float val[8];
    int cubeindex = 0;
    for (int s = 0; s < 8; s++) {
        ivec3 c = base + CORNER[s];
        val[s] = f[c.x + g_R1*(c.y + g_R1*c.z)];
        if (val[s] < 0.0) cubeindex |= (1 << s);
    }
    if (cubeindex == 0 || cubeindex == 255) return;

    vec3 vpos[12];
    vec3 vnrm[12];
    for (int e = 0; e < 12; e++) {
        int a = EDGE[e].x, b = EDGE[e].y;
        float fa = val[a], fb = val[b];
        if ((fa < 0.0) == (fb < 0.0)) continue;   // edge not crossed
        ivec3 ca = base + CORNER[a];
        ivec3 cb = base + CORNER[b];
        float tt = fa / (fa - fb);
        vpos[e] = u_origin + u_voxel * (vec3(ca) + tt * vec3(cb - ca));
        vec3 nn = mix(grad(ca), grad(cb), tt);
        vnrm[e] = (dot(nn,nn) > 0.0) ? normalize(nn) : vec3(0.0, 0.0, 1.0);
    }

    int row = cubeindex * 16;
    for (int i = 0; i < 15 && tri[row+i] != -1; i += 3) {
        int e0 = tri[row+i], e1 = tri[row+i+1], e2 = tri[row+i+2];

        if (u_count_only == 1) { atomicAdd(cnt[0], 1u); continue; }

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
    auto C = [&](uint32_t ii, uint32_t jj, uint32_t kk) { return f[ii + S*(jj + S*kk)]; };
    float c00 = C(i,j,  k  )*(1-fx) + C(i+1,j,  k  )*fx;
    float c10 = C(i,j+1,k  )*(1-fx) + C(i+1,j+1,k  )*fx;
    float c01 = C(i,j,  k+1)*(1-fx) + C(i+1,j,  k+1)*fx;
    float c11 = C(i,j+1,k+1)*(1-fx) + C(i+1,j+1,k+1)*fx;
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
        if (best != 0xFFFFFFFFu) { remap[v] = best; welded++; }
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
    GlBuf soup_pos, soup_idx, dist, field, mc_out, mc_cnt, tritab;
    soup_pos.gen(); soup_idx.gen(); dist.gen(); field.gen();
    mc_out.gen();   mc_cnt.gen();   tritab.gen();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, soup_pos.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, pos.size()*sizeof(float), pos.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, soup_idx.id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, idx.size()*sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);

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
    std::string splat_full = std::string(SPLAT_SRC);
    {   // inject the Ericson helper just before main()
        size_t mpos = splat_full.find("void main()");
        splat_full.insert(mpos, PT_TRI_DIST);
    }
    GlProg splat_prog, sign_prog, mc_prog;
    splat_prog.id = cs.compile_program(splat_full.c_str());
    sign_prog.id  = cs.compile_program(SIGN_SRC);
    mc_prog.id    = cs.compile_program(MC_SRC);
    if (!splat_prog.id || !sign_prog.id || !mc_prog.id) {
        res.error = "SDF shader compile failed (see stderr)";
        return res;
    }

    // ---- Dispatch 1: distance splat (one thread per triangle) ----
    glUseProgram(splat_prog.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_POS, soup_pos.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_SOUP_IDX, soup_idx.id);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BIND_SDF_DIST,     dist.id);
    set_grid_uniforms(splat_prog.id, grid);
    glUniform1i (glGetUniformLocation(splat_prog.id, "u_band"),     BAND_DILATE);
    glUniform1ui(glGetUniformLocation(splat_prog.id, "u_triCount"), tri_count);
    glDispatchCompute((tri_count + 63u) / 64u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

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
    glDispatchCompute((corner_count + 63u) / 64u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

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
    glDispatchCompute((cell_count + 63u) / 64u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

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
    glDispatchCompute((cell_count + 63u) / 64u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    // ---- Readback (the only readback) ----
    std::vector<float> soup(out_tris * 18);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mc_out.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       (GLsizeiptr)soup.size()*sizeof(float), soup.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    // ---- Hash-weld → watertight indexed Mesh ----
    Mesh welded;
    hash_weld(soup, out_tris, grid.voxel * 1e-3f, welded);

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
            reflect_across_x(welded, seam);
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

    // ---- Splice into the scene as the merged entity ----
    uint32_t merged_id = scene.merge_selected_into(welded, 0);
    if (merged_id == 0) { res.error = "scene splice failed"; return res; }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.success = true;
    return res;
}
