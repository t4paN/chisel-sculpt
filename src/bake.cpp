#include "bake.h"
#include "retopo.h"
#include "uv_unwrap.h"
#include "mesh.h"
#include "paint_layers.h"
#include "mikktspace/mikktspace.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// The sculpt's triangles, binned into boxes. A node is a leaf when n > 0 (triangles
// tri[a .. a + n)), else its children are nodes a and a + 1. Up to 8 triangles a leaf:
// a 4M-triangle sculpt then needs ~1.3M nodes (40 MB), which the 32-bit web heap takes.
struct Node { float lo[3], hi[3]; uint32_t a, n; };
constexpr uint32_t kLeaf = 8;

struct BuildTask { uint32_t node, begin, end; };

enum class Phase { Prep, Tree, Raster, Shade, Dilate, Done };

}  // namespace

struct BakeWork {
    BakeSettings s;
    Phase phase = Phase::Prep;
    Clock::time_point t0;
    // The sculpt it started on, to notice it changing under the job.
    const Mesh* high = nullptr;
    uint32_t high_v = 0, high_t = 0;
    std::vector<float>    hn;       // the sculpt's smooth normals, 3 per vertex (its own CPU ones may be stale)
    std::vector<uint32_t> hcol;     // the sculpt's paint, composited (empty = no colour map)
    const MeshTexture*    htex = nullptr;   // or its texture, through its corner UVs (wins over paint)
    // The tree.
    std::vector<Node>     nodes;
    std::vector<uint32_t> tri;
    std::vector<float>    cen;      // triangle centres, 3 per triangle: build only
    std::vector<BuildTask> todo;
    uint32_t tree_done = 0;         // triangles placed in leaves (progress)
    // The low, as the soup the result carries (filled in Prep).
    BakeResult out;
    float reach = 0.0f;             // world units
    // Raster: per texel, the soup triangle covering its centre + 1 (0 = none).
    std::vector<uint32_t> owner;
    uint32_t raster_tri = 0;
    // Shade, row by row; filled = has a value (an island texel whose ray hit).
    std::vector<uint8_t> filled;
    uint32_t shade_row = 0;
    uint64_t covered = 0, misses = 0;
    // Dilate: the empty texels next to filled ones, grown out a ring per pass.
    std::vector<uint32_t> front, next;
    int dilate_pass = 0, dilate_passes = 0;
};

// ---- Low mesh: tangents (MikkTSpace) ------------------------------------------------

namespace {

// MikkTSpace sees the faces as they are where it can (triangles and quads: that's how
// the apps on the other end hand them over too), fanned beyond that.
struct MikkFace { uint32_t c[4]; int k; };
struct MikkData {
    const RetopoMesh* rm;
    const UvLayout* L;
    const std::vector<Vec3>* vn;
    std::vector<MikkFace> faces;
    std::vector<float>* tan;   // 4 per corner of rm.faces
};

int mk_faces(const SMikkTSpaceContext* c) { return (int)((MikkData*)c->m_pUserData)->faces.size(); }
int mk_verts(const SMikkTSpaceContext* c, int f) { return ((MikkData*)c->m_pUserData)->faces[f].k; }
void mk_pos(const SMikkTSpaceContext* c, float out[], int f, int v) {
    const MikkData& d = *(MikkData*)c->m_pUserData;
    const Vec3 p = d.rm->v[d.rm->faces[d.faces[f].c[v]]];
    out[0] = p.x; out[1] = p.y; out[2] = p.z;
}
void mk_nrm(const SMikkTSpaceContext* c, float out[], int f, int v) {
    const MikkData& d = *(MikkData*)c->m_pUserData;
    const Vec3 n = (*d.vn)[d.rm->faces[d.faces[f].c[v]]];
    out[0] = n.x; out[1] = n.y; out[2] = n.z;
}
void mk_uv(const SMikkTSpaceContext* c, float out[], int f, int v) {
    const MikkData& d = *(MikkData*)c->m_pUserData;
    const uint32_t k = d.faces[f].c[v];
    out[0] = d.L->uv[2 * k]; out[1] = d.L->uv[2 * k + 1];
}
void mk_set(const SMikkTSpaceContext* c, const float t[], float sign, int f, int v) {
    MikkData& d = *(MikkData*)c->m_pUserData;
    float* o = d.tan->data() + 4 * (size_t)d.faces[f].c[v];
    o[0] = t[0]; o[1] = t[1]; o[2] = t[2]; o[3] = sign;
}

// The soup: every face with an island, fanned from its first corner (as OBJ readers
// triangulate), each corner carrying position, smooth normal, tangent and UV.
void build_low(BakeWork& w, const RetopoMesh& rm, const UvLayout& L) {
    std::vector<Vec3> vn;
    retopo_smooth_normals(rm, vn);
    std::vector<float> tan(rm.faces.size() * 4, 0.0f);
    MikkData md{&rm, &L, &vn, {}, &tan};
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        if (L.face_chart[f] >= L.charts) continue;
        const uint32_t o = rm.face_off[f], k = rm.face_size(f);
        if (k == 3 || k == 4) { md.faces.push_back({{o, o + 1, o + 2, k == 4 ? o + 3 : 0}, (int)k}); continue; }
        for (uint32_t j = 1; j + 1 < k; j++) md.faces.push_back({{o, o + j, o + j + 1, 0}, 3});
    }
    SMikkTSpaceInterface mi = {};
    mi.m_getNumFaces = mk_faces;
    mi.m_getNumVerticesOfFace = mk_verts;
    mi.m_getPosition = mk_pos;
    mi.m_getNormal = mk_nrm;
    mi.m_getTexCoord = mk_uv;
    mi.m_setTSpaceBasic = mk_set;
    SMikkTSpaceContext mc = {&mi, &md};
    genTangSpaceDefault(&mc);

    BakeResult& r = w.out;
    double edge = 0.0;
    uint32_t edges = 0;
    auto corner = [&](uint32_t k) {
        const Vec3 p = rm.v[rm.faces[k]], n = vn[rm.faces[k]];
        r.pos.insert(r.pos.end(), {p.x, p.y, p.z});
        r.nrm.insert(r.nrm.end(), {n.x, n.y, n.z});
        r.tan.insert(r.tan.end(), tan.begin() + 4 * k, tan.begin() + 4 * k + 4);
        r.uv.insert(r.uv.end(), {L.uv[2 * k], L.uv[2 * k + 1]});
    };
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        if (L.face_chart[f] >= L.charts) continue;
        const uint32_t o = rm.face_off[f], k = rm.face_size(f);
        for (uint32_t j = 0; j < k; j++) {
            edge += (rm.v[rm.faces[o + j]] - rm.v[rm.faces[o + (j + 1) % k]]).length();
            edges++;
        }
        for (uint32_t j = 1; j + 1 < k; j++) { corner(o); corner(o + j); corner(o + j + 1); }
    }
    w.reach = (float)(edge / std::max(1u, edges)) * w.s.reach;
}

// ---- The sculpt: normals, paint, tree --------------------------------------------------

void tri_box(const Mesh& m, uint32_t t, float lo[3], float hi[3]) {
    for (int c = 0; c < 3; c++) {
        const uint32_t v = m.indices[3 * t + c];
        const float p[3] = {m.pos_x[v], m.pos_y[v], m.pos_z[v]};
        for (int a = 0; a < 3; a++) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
    }
}

void prep_high(BakeWork& w, const Mesh& m) {
    const uint32_t nv = m.vertex_count(), nt = m.tri_count();
    w.hn.assign((size_t)nv * 3, 0.0f);
    w.cen.resize((size_t)nt * 3);
    for (uint32_t t = 0; t < nt; t++) {
        const uint32_t a = m.indices[3 * t], b = m.indices[3 * t + 1], c = m.indices[3 * t + 2];
        const Vec3 pa = m.get_pos(a), pb = m.get_pos(b), pc = m.get_pos(c);
        const Vec3 n = (pb - pa).cross(pc - pa);   // area-weighted, as the sculpt shades
        for (uint32_t v : {a, b, c}) { w.hn[3 * v] += n.x; w.hn[3 * v + 1] += n.y; w.hn[3 * v + 2] += n.z; }
        const Vec3 ce = (pa + pb + pc) * (1.0f / 3.0f);
        w.cen[3 * t] = ce.x; w.cen[3 * t + 1] = ce.y; w.cen[3 * t + 2] = ce.z;
    }
    for (uint32_t v = 0; v < nv; v++) {
        float* n = &w.hn[3 * v];
        const float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l > 1e-20f) { n[0] /= l; n[1] /= l; n[2] /= l; }
    }
    // A textured sculpt (an imported glTF) bakes its texture: that's the detail a
    // game asset needs, and vertex paint on a mesh that coarse holds almost none of it.
    // The shared_ptr in `m` keeps the pixels alive; the job stops if `m` changes.
    if (w.s.color && m.texture && m.has_uv()) w.htex = m.texture.get();
    else if (w.s.color) {
        w.hcol.resize(nv);
        paint_composite(m.paint_layers, m.color, nv, w.hcol.data());
    }
    w.tri.resize(nt);
    for (uint32_t t = 0; t < nt; t++) w.tri[t] = t;
    w.nodes.reserve(2 * (nt / 4 + 1));
    w.nodes.push_back(Node{});
    w.todo.push_back({0, 0, nt});
}

// One node: its box, then a leaf or a split at the median centre of its longest side.
void build_one(BakeWork& w, const Mesh& m) {
    const BuildTask t = w.todo.back();
    w.todo.pop_back();
    Node nd;
    float clo[3] = {1e30f, 1e30f, 1e30f}, chi[3] = {-1e30f, -1e30f, -1e30f};
    for (int a = 0; a < 3; a++) { nd.lo[a] = 1e30f; nd.hi[a] = -1e30f; }
    for (uint32_t i = t.begin; i < t.end; i++) {
        tri_box(m, w.tri[i], nd.lo, nd.hi);
        const float* c = &w.cen[3 * w.tri[i]];
        for (int a = 0; a < 3; a++) { clo[a] = std::min(clo[a], c[a]); chi[a] = std::max(chi[a], c[a]); }
    }
    const uint32_t n = t.end - t.begin;
    int ax = 0;
    for (int a = 1; a < 3; a++) if (chi[a] - clo[a] > chi[ax] - clo[ax]) ax = a;
    if (n <= kLeaf || chi[ax] - clo[ax] <= 0.0f) {
        nd.a = t.begin; nd.n = n;
        w.nodes[t.node] = nd;
        w.tree_done += n;
        return;
    }
    const uint32_t mid = t.begin + n / 2;
    const float* cen = w.cen.data();
    std::nth_element(w.tri.begin() + t.begin, w.tri.begin() + mid, w.tri.begin() + t.end,
                     [cen, ax](uint32_t x, uint32_t y) { return cen[3 * x + ax] < cen[3 * y + ax]; });
    nd.a = (uint32_t)w.nodes.size(); nd.n = 0;
    w.nodes[t.node] = nd;
    w.nodes.push_back(Node{});
    w.nodes.push_back(Node{});
    w.todo.push_back({nd.a, t.begin, mid});
    w.todo.push_back({nd.a + 1, mid, t.end});
}

// The ray: the line through o along d (unit), t in [-reach, reach]. The hit nearest the
// low (smallest |t|) on a triangle facing along d, so a ray never takes the inside of a
// thin part (the far side of an ear) or a face turned away. False = nothing there.
struct Hit { uint32_t t; float u, v, at; };

bool box_reach(const Node& nd, const float o[3], const float inv[3], float r, float& near) {
    float t0 = -r, t1 = r;
    for (int a = 0; a < 3; a++) {
        float ta = (nd.lo[a] - o[a]) * inv[a], tb = (nd.hi[a] - o[a]) * inv[a];
        if (ta > tb) std::swap(ta, tb);
        if (!(ta <= tb)) { ta = -1e30f; tb = 1e30f; }   // o on a slab plane of a flat axis: 0 * inf
        t0 = std::max(t0, ta); t1 = std::min(t1, tb);
        if (t0 > t1) return false;
    }
    near = t0 > 0.0f ? t0 : t1 < 0.0f ? -t1 : 0.0f;
    return true;
}

bool cast(const BakeWork& w, const Mesh& m, const Vec3& o3, const Vec3& d3, Hit& best) {
    const float o[3] = {o3.x, o3.y, o3.z}, d[3] = {d3.x, d3.y, d3.z};
    const float inv[3] = {1.0f / d[0], 1.0f / d[1], 1.0f / d[2]};
    best.at = w.reach;
    bool any = false;
    uint32_t stack[96];
    int sp = 0;
    float near;
    if (!box_reach(w.nodes[0], o, inv, w.reach, near)) return false;
    stack[sp++] = 0;
    while (sp) {
        const Node& nd = w.nodes[stack[--sp]];
        if (!box_reach(nd, o, inv, w.reach, near) || near >= best.at) continue;
        if (nd.n) {
            for (uint32_t i = nd.a; i < nd.a + nd.n; i++) {
                const uint32_t t = w.tri[i];
                const uint32_t ia = m.indices[3 * t], ib = m.indices[3 * t + 1], ic = m.indices[3 * t + 2];
                const Vec3 a = m.get_pos(ia), e1 = m.get_pos(ib) - a, e2 = m.get_pos(ic) - a;
                if (e1.cross(e2).dot(d3) <= 0.0f) continue;   // turned away from the low
                // Moller-Trumbore, both ways along the line.
                const Vec3 pv = d3.cross(e2);
                const float det = e1.dot(pv);
                if (std::fabs(det) < 1e-30f) continue;
                const float id = 1.0f / det;
                const Vec3 tv = o3 - a;
                const float u = tv.dot(pv) * id;
                if (u < 0.0f || u > 1.0f) continue;
                const Vec3 qv = tv.cross(e1);
                const float v = d3.dot(qv) * id;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float at = e2.dot(qv) * id;
                if (std::fabs(at) >= best.at) continue;
                best = {t, u, v, std::fabs(at)};
                any = true;
            }
            continue;
        }
        // Children: the nearer last, so it pops first and tightens best.at for the other.
        float n0, n1;
        const bool h0 = box_reach(w.nodes[nd.a], o, inv, w.reach, n0);
        const bool h1 = box_reach(w.nodes[nd.a + 1], o, inv, w.reach, n1);
        if (sp + 2 > 96) continue;   // can't happen with median splits; never overrun
        if (h0 && h1) {
            if (n0 <= n1) { stack[sp++] = nd.a + 1; stack[sp++] = nd.a; }
            else          { stack[sp++] = nd.a; stack[sp++] = nd.a + 1; }
        } else if (h0) stack[sp++] = nd.a;
        else if (h1) stack[sp++] = nd.a + 1;
    }
    return any;
}

// ---- Texels ------------------------------------------------------------------------------

// Soup triangle t into the owner map: every texel whose centre lies inside (first wins).
void raster_tri(BakeWork& w, uint32_t t) {
    const int R = w.out.res;
    const float* uv = &w.out.uv[6 * (size_t)t];
    // Texel space, y down from the top row (v = 1).
    float x[3], y[3];
    for (int i = 0; i < 3; i++) { x[i] = uv[2 * i] * R; y[i] = (1.0f - uv[2 * i + 1]) * R; }
    const float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if (std::fabs(area) < 1e-12f) return;
    const int x0 = std::max(0, (int)std::floor(std::min({x[0], x[1], x[2]})));
    const int x1 = std::min(R - 1, (int)std::ceil(std::max({x[0], x[1], x[2]})));
    const int y0 = std::max(0, (int)std::floor(std::min({y[0], y[1], y[2]})));
    const int y1 = std::min(R - 1, (int)std::ceil(std::max({y[0], y[1], y[2]})));
    const float eps = -1e-4f * std::fabs(area);
    for (int py = y0; py <= y1; py++)
        for (int px = x0; px <= x1; px++) {
            const float cx = px + 0.5f, cy = py + 0.5f;
            float b[3];
            for (int i = 0; i < 3; i++) {
                const int j = (i + 1) % 3, k = (i + 2) % 3;
                b[i] = ((x[k] - x[j]) * (cy - y[j]) - (cx - x[j]) * (y[k] - y[j])) * (area > 0 ? 1.0f : -1.0f);
            }
            if (b[0] < eps || b[1] < eps || b[2] < eps) continue;
            uint32_t& o = w.owner[(size_t)py * R + px];
            if (!o) o = t + 1;
        }
}

uint32_t lerp_rgba(uint32_t a, uint32_t b, uint32_t c, float wa, float wb, float wc) {
    uint32_t out = 0;
    for (int s = 0; s < 32; s += 8) {
        const float v = ((a >> s) & 255) * wa + ((b >> s) & 255) * wb + ((c >> s) & 255) * wc;
        out |= (uint32_t)std::max(0.0f, std::min(255.0f, v + 0.5f)) << s;
    }
    return out;
}

// Bilinear, wrapping (glTF's default sampler), at OBJ-convention (u, v): v = 1 is the
// image's top row.
uint32_t sample_tex(const MeshTexture& t, float u, float v) {
    const float fx = u * t.w - 0.5f, fy = (1.0f - v) * t.h - 0.5f;
    const float x0f = std::floor(fx), y0f = std::floor(fy);
    const float ax = fx - x0f, ay = fy - y0f;
    auto wrap = [](long i, int n) { i %= n; return (int)(i < 0 ? i + n : i); };
    const int x0 = wrap((long)x0f, t.w), x1 = wrap((long)x0f + 1, t.w);
    const int y0 = wrap((long)y0f, t.h), y1 = wrap((long)y0f + 1, t.h);
    const uint8_t* p00 = &t.rgba[((size_t)y0 * t.w + x0) * 4], *p10 = &t.rgba[((size_t)y0 * t.w + x1) * 4];
    const uint8_t* p01 = &t.rgba[((size_t)y1 * t.w + x0) * 4], *p11 = &t.rgba[((size_t)y1 * t.w + x1) * 4];
    uint32_t out = 0;
    for (int c = 0; c < 4; c++) {
        const float top = p00[c] + (p10[c] - p00[c]) * ax, bot = p01[c] + (p11[c] - p01[c]) * ax;
        out |= (uint32_t)std::lround(top + (bot - top) * ay) << (8 * c);
    }
    return out;
}

void put(uint8_t* px, uint32_t rgba) {
    px[0] = rgba & 255; px[1] = (rgba >> 8) & 255; px[2] = (rgba >> 16) & 255; px[3] = 255;
}

// One row of texels: ray, then the sculpt's normal in the low's tangent frame and its
// paint where the ray landed.
void shade_row(BakeWork& w, const Mesh& m, uint32_t py) {
    const int R = w.out.res;
    BakeResult& r = w.out;
    for (int px = 0; px < R; px++) {
        const size_t texel = (size_t)py * R + px;
        const uint32_t own = w.owner[texel];
        if (!own) continue;
        w.covered++;
        const uint32_t t = own - 1;
        const float* uv = &r.uv[6 * (size_t)t];
        const float cu = (px + 0.5f) / R, cv = 1.0f - (py + 0.5f) / R;
        const float x0 = uv[0], y0 = uv[1];
        const float ax = uv[2] - x0, ay = uv[3] - y0, bx = uv[4] - x0, by = uv[5] - y0;
        const float den = ax * by - bx * ay;
        if (std::fabs(den) < 1e-20f) { w.misses++; continue; }
        const float b1 = ((cu - x0) * by - bx * (cv - y0)) / den;
        const float b2 = (ax * (cv - y0) - (cu - x0) * ay) / den;
        const float b0 = 1.0f - b1 - b2;
        auto at3 = [&](const std::vector<float>& a, int stride) {
            const float* c = &a[(size_t)stride * 3 * t];
            return Vec3(c[0] * b0 + c[stride] * b1 + c[2 * stride] * b2,
                        c[1] * b0 + c[stride + 1] * b1 + c[2 * stride + 1] * b2,
                        c[2] * b0 + c[stride + 2] * b1 + c[2 * stride + 2] * b2);
        };
        const Vec3 P = at3(r.pos, 3), N = at3(r.nrm, 3), T = at3(r.tan, 4);
        const float sign = r.tan[(size_t)12 * t + 3] < 0.0f ? -1.0f : 1.0f;
        const Vec3 dir = N.normalized();
        Hit h;
        if (dir.length() < 0.5f || !cast(w, m, P, dir, h)) { w.misses++; continue; }
        const uint32_t ia = m.indices[3 * h.t], ib = m.indices[3 * h.t + 1], ic = m.indices[3 * h.t + 2];
        const float ha = 1.0f - h.u - h.v;
        if (r.has_normal) {
            const float* na = &w.hn[3 * ia], *nb = &w.hn[3 * ib], *nc = &w.hn[3 * ic];
            const Vec3 Nh = Vec3(na[0] * ha + nb[0] * h.u + nc[0] * h.v,
                                 na[1] * ha + nb[1] * h.u + nc[1] * h.v,
                                 na[2] * ha + nb[2] * h.u + nc[2] * h.v).normalized();
            // The exact inverse of the shader's n = normalize(x T + y B + z N), with
            // B = sign * cross(N, T) and T, N as interpolated (mikktspace.h's sampler).
            const Vec3 B = N.cross(T) * sign;
            const Vec3 r0 = B.cross(N), r1 = N.cross(T), r2 = T.cross(B);
            const float fs = T.dot(r0) < 0.0f ? -1.0f : 1.0f;
            Vec3 tn = Vec3(Nh.dot(r0), Nh.dot(r1), Nh.dot(r2)) * fs;
            tn = tn.normalized();
            if (r.flip_green) tn.y = -tn.y;
            uint8_t* o = &r.normal_rgba[4 * texel];
            o[0] = (uint8_t)std::lround((tn.x * 0.5f + 0.5f) * 255.0f);
            o[1] = (uint8_t)std::lround((tn.y * 0.5f + 0.5f) * 255.0f);
            o[2] = (uint8_t)std::lround((tn.z * 0.5f + 0.5f) * 255.0f);
            o[3] = 255;
        }
        if (r.has_color) {
            uint32_t c = 0xFFFFFFFFu;
            if (w.htex) {
                const float* hu = &m.uv[6 * (size_t)h.t];
                c = sample_tex(*w.htex, hu[0] * ha + hu[2] * h.u + hu[4] * h.v,
                                        hu[1] * ha + hu[3] * h.u + hu[5] * h.v);
            } else if (!w.hcol.empty()) {
                c = lerp_rgba(w.hcol[ia], w.hcol[ib], w.hcol[ic], ha, h.u, h.v);
            }
            put(&r.color_rgba[4 * texel], c);
        }
        w.filled[texel] = 1;
    }
}

// Every empty texel next to a filled one: the first ring to grow.
void dilate_start(BakeWork& w) {
    const int R = w.out.res;
    w.front.clear();
    for (int y = 0; y < R; y++)
        for (int x = 0; x < R; x++) {
            if (w.filled[(size_t)y * R + x]) continue;
            bool near = false;
            for (int dy = -1; dy <= 1 && !near; dy++)
                for (int dx = -1; dx <= 1 && !near; dx++) {
                    const int nx = x + dx, ny = y + dy;
                    near = nx >= 0 && ny >= 0 && nx < R && ny < R && w.filled[(size_t)ny * R + nx] == 1;
                }
            if (near) w.front.push_back((uint32_t)(y * R + x));
        }
}

// One ring: each front texel takes the mean of its filled neighbours; then they count
// as filled and their empty neighbours are the next ring. Keeps texture filtering and
// mip maps from pulling background into the islands' edges, and fills small misses.
void dilate_pass(BakeWork& w) {
    const int R = w.out.res;
    BakeResult& r = w.out;
    const uint8_t gen = (uint8_t)(2 + w.dilate_pass % 250);   // marks this ring (1 = baked)
    for (uint32_t i : w.front) {
        const int x = (int)(i % R), y = (int)(i / R);
        int nsum[4] = {0, 0, 0, 0}, csum[4] = {0, 0, 0, 0}, cnt = 0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                const int nx = x + dx, ny = y + dy;
                if ((!dx && !dy) || nx < 0 || ny < 0 || nx >= R || ny >= R) continue;
                const size_t j = (size_t)ny * R + nx;
                const uint8_t f = w.filled[j];
                if (!f || f == gen) continue;   // empty, or filled this same ring
                for (int c = 0; c < 4; c++) {
                    if (r.has_normal) nsum[c] += r.normal_rgba[4 * j + c];
                    if (r.has_color)  csum[c] += r.color_rgba[4 * j + c];
                }
                cnt++;
            }
        if (!cnt) continue;
        for (int c = 0; c < 4; c++) {
            if (r.has_normal) r.normal_rgba[4 * (size_t)i + c] = (uint8_t)((nsum[c] + cnt / 2) / cnt);
            if (r.has_color)  r.color_rgba[4 * (size_t)i + c]  = (uint8_t)((csum[c] + cnt / 2) / cnt);
        }
        w.filled[i] = gen;
    }
    w.next.clear();
    for (uint32_t i : w.front) {
        if (w.filled[i] != gen) continue;
        const int x = (int)(i % R), y = (int)(i / R);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                const int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= R || ny >= R) continue;
                const uint32_t j = (uint32_t)(ny * R + nx);
                if (!w.filled[j]) { w.filled[j] = 255; w.next.push_back(j); }   // 255: queued once
            }
    }
    for (uint32_t j : w.next) w.filled[j] = 0;
    w.front.swap(w.next);
    w.dilate_pass++;
}

// Past the last ring: a flat normal, and the mean paint (no hard step for mips to find).
void finish(BakeWork& w) {
    BakeResult& r = w.out;
    const size_t n = (size_t)r.res * r.res;
    uint64_t sum[3] = {0, 0, 0}, cnt = 0;
    if (r.has_color)
        for (size_t i = 0; i < n; i++)
            if (w.filled[i]) { for (int c = 0; c < 3; c++) sum[c] += r.color_rgba[4 * i + c]; cnt++; }
    const uint32_t mean = cnt ? (uint32_t)(sum[0] / cnt) | (uint32_t)(sum[1] / cnt) << 8 | (uint32_t)(sum[2] / cnt) << 16
                              : 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        if (w.filled[i]) continue;
        if (r.has_normal) put(&r.normal_rgba[4 * i], 0xFFFF8080u);   // (128, 128, 255)
        if (r.has_color)  put(&r.color_rgba[4 * i], mean);
    }
    r.missed = w.covered ? (float)((double)w.misses / (double)w.covered) : 0.0f;
}

}  // namespace

// ---- Public -----------------------------------------------------------------------------

uint64_t bake_key(const RetopoMesh& rm, const UvLayout& L) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(rm.faces.data(), rm.faces.size() * sizeof(uint32_t));
    mix(rm.face_off.data(), rm.face_off.size() * sizeof(uint32_t));
    mix(rm.v.data(), rm.v.size() * sizeof(Vec3));
    mix(L.uv.data(), L.uv.size() * sizeof(float));
    return h;
}

void bake_refresh(BakeState& b, const RetopoMesh& rm, const UvLayout& L, uint64_t rev) {
    if (rev == b.checked_rev) return;
    b.checked_rev = rev;
    b.fresh = !b.result.empty() && bake_key(rm, L) == b.result.key;
}

const char* bake_start(BakeState& b, const RetopoMesh& rm, const UvLayout& L, const Mesh& high) {
    if (b.job) return "Already baking";
    if (rm.faces.empty()) return "Build a retopo mesh in mode 0 first";
    if (L.empty() || !L.fits(rm) || L.key != uv_topology_key(rm)) return "Unwrap first: the bake needs UVs";
    if (high.tri_count() == 0) return "No sculpt to bake from";
    const BakeSettings& s = b.settings;
    if (!s.normal && !s.color) return "Pick a map to bake";
    auto w = std::make_shared<BakeWork>();
    w->s = s;
    w->t0 = Clock::now();
    w->high = &high;
    w->high_v = high.vertex_count();
    w->high_t = high.tri_count();
    BakeResult& r = w->out;
    r.res = std::max(64, std::min(4096, s.resolution));
    r.has_normal = s.normal;
    r.has_color = s.color;
    r.flip_green = s.flip_green;
    r.key = bake_key(rm, L);
    r.high_tris = high.tri_count();
    build_low(*w, rm, L);
    if (r.pos.empty()) return "No faces in an island: Unwrap again";
    b.job = std::move(w);
    return nullptr;
}

void bake_cancel(BakeState& b) { b.job.reset(); }

int bake_step(BakeState& b, const Mesh& m, double budget_ms, const char** why) {
    if (why) *why = nullptr;
    if (!b.job) return -1;
    BakeWork& w = *b.job;
    if (&m != w.high || m.vertex_count() != w.high_v || m.tri_count() != w.high_t) {
        if (why) *why = "Bake stopped: the sculpt changed";
        b.job.reset();
        return -1;
    }
    const Clock::time_point t0 = Clock::now();
    const int R = w.out.res;
    while (ms_since(t0) < budget_ms) {
        switch (w.phase) {
        case Phase::Prep:
            prep_high(w, m);
            w.phase = Phase::Tree;
            break;
        case Phase::Tree:
            for (int i = 0; i < 256 && !w.todo.empty(); i++) build_one(w, m);
            if (w.todo.empty()) {
                std::vector<float>().swap(w.cen);
                w.owner.assign((size_t)R * R, 0);
                w.phase = Phase::Raster;
            }
            break;
        case Phase::Raster: {
            const uint32_t n = (uint32_t)(w.out.pos.size() / 9);
            for (int i = 0; i < 64 && w.raster_tri < n; i++) raster_tri(w, w.raster_tri++);
            if (w.raster_tri >= n) {
                if (w.out.has_normal) w.out.normal_rgba.assign((size_t)R * R * 4, 0);
                if (w.out.has_color)  w.out.color_rgba.assign((size_t)R * R * 4, 0);
                w.filled.assign((size_t)R * R, 0);
                w.phase = Phase::Shade;
            }
            break;
        }
        case Phase::Shade:
            shade_row(w, m, w.shade_row++);
            if (w.shade_row >= (uint32_t)R) {
                std::vector<uint32_t>().swap(w.owner);
                // The tree's done its job: hand its memory back before the next one.
                std::vector<Node>().swap(w.nodes);
                std::vector<uint32_t>().swap(w.tri);
                // Rings to grow: enough to cover a mip chain's reach a few levels down.
                w.dilate_passes = std::max(8, std::min(64, R / 64));
                dilate_start(w);
                w.phase = Phase::Dilate;
            }
            break;
        case Phase::Dilate:
            if (w.dilate_pass < w.dilate_passes && !w.front.empty()) dilate_pass(w);
            else { finish(w); w.phase = Phase::Done; }
            break;
        case Phase::Done: {
            // Never repeats in a run: a new project starts a fresh BakeState, and the
            // renderer must still see its first bake as new.
            static uint32_t serial = 0;
            w.out.ms = ms_since(w.t0);
            w.out.serial = ++serial;
            b.result = std::move(w.out);
            b.checked_rev = ~0ull;
            std::printf("[bake] %dx%d from %u sculpt tris: %.1f%% of texels missed, %.0f ms\n",
                        b.result.res, b.result.res, b.result.high_tris, b.result.missed * 100.0f, b.result.ms);
            b.job.reset();
            return 1;
        }
        }
    }
    return 0;
}

float bake_progress(const BakeState& b) {
    if (!b.job) return 0.0f;
    const BakeWork& w = *b.job;
    const float R = (float)w.out.res;
    switch (w.phase) {
    case Phase::Prep:   return 0.0f;
    case Phase::Tree:   return 0.02f + 0.18f * (float)w.tree_done / (float)std::max(1u, w.high_t);
    case Phase::Raster: return 0.20f + 0.05f * (float)w.raster_tri / std::max(1.0f, (float)w.out.pos.size() / 9.0f);
    case Phase::Shade:  return 0.25f + 0.70f * (float)w.shade_row / R;
    case Phase::Dilate: return 0.95f + 0.05f * (float)w.dilate_pass / (float)std::max(1, w.dilate_passes);
    case Phase::Done:   return 1.0f;
    }
    return 0.0f;
}

const char* bake_phase(const BakeState& b) {
    if (!b.job) return "";
    switch (b.job->phase) {
    case Phase::Prep:   return "Reading the sculpt";
    case Phase::Tree:   return "Sorting the sculpt";
    case Phase::Raster: return "Laying out texels";
    case Phase::Shade:  return "Casting rays";
    case Phase::Dilate: return "Padding the edges";
    case Phase::Done:   return "Done";
    }
    return "";
}

bool bake_write_png(const BakeResult& r, int which, const char* path) {
    const std::vector<uint8_t>& src = which == 0 ? r.normal_rgba : r.color_rgba;
    if (r.empty() || src.size() != (size_t)r.res * r.res * 4) return false;
    std::vector<uint8_t> rgb((size_t)r.res * r.res * 3);
    for (size_t i = 0, n = (size_t)r.res * r.res; i < n; i++)
        for (int c = 0; c < 3; c++) rgb[3 * i + c] = src[4 * i + c];
    return stbi_write_png(path, r.res, r.res, 3, rgb.data(), r.res * 3) != 0;
}
