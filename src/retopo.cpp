#include "retopo.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

// ---- Stroke list -------------------------------------------------------------------

bool RetopoState::finish_live() {
    RetopoStroke s = std::move(live);
    live = RetopoStroke{};
    drawing = false;
    if (s.pts.size() < 3) return false;

    float len = 0.0f, hw = 0.0f;
    for (size_t i = 1; i < s.pts.size(); i++) len += (s.pts[i] - s.pts[i - 1]).length();
    for (float w : s.half_w) hw += w;
    hw /= (float)s.half_w.size();
    if (len < hw) return false;   // a tap, or a wobble on the spot

    // Closed when the pen came back to where it started. "Back" is judged against the
    // band's own width, so a fat donut doesn't have to be closed to the pixel, and the
    // length test keeps a short fat dab from reading as a ring round nothing.
    const float gap = (s.pts.back() - s.pts.front()).length();
    s.closed = gap < 1.5f * hw && len > 6.0f * hw;
    if (s.closed && gap > 1e-6f) {
        // Drop the overlap: a ring drawn past its start would double back on itself.
        // Trim the tail to the point nearest the start.
        size_t best = s.pts.size() - 1;
        float best_d = gap;
        for (size_t i = s.pts.size() / 2; i < s.pts.size(); i++) {
            float d = (s.pts[i] - s.pts.front()).length();
            if (d < best_d) { best_d = d; best = i; }
        }
        s.pts.resize(best + 1); s.nrm.resize(best + 1); s.half_w.resize(best + 1);
    }

    s.id = next_id++;
    char nm[32];
    std::snprintf(nm, sizeof(nm), "%s %u", s.closed ? "Donut" : "Strip", s.id);
    s.name = nm;
    strokes.push_back(std::move(s));
    redo.clear();
    return true;
}

void RetopoState::undo() {
    if (strokes.empty()) return;
    redo.push_back(std::move(strokes.back()));
    strokes.pop_back();
}

void RetopoState::redo_one() {
    if (redo.empty()) return;
    strokes.push_back(std::move(redo.back()));
    redo.pop_back();
}

void RetopoState::clear_all() {
    // Clear goes onto the redo stack in reverse, so Ctrl+Shift+Z can walk them back.
    while (!strokes.empty()) undo();
    result.clear();
    built = false;
}

// ---- Surface snapping --------------------------------------------------------------
//
// Build walks across the sculpt and keeps snapping onto it, so it needs closest-point
// queries. Only triangles near the strokes can matter, so a sorted cell list over just
// those (key, tri) pairs does the job: no hash map, one sort, binary-search lookups.

namespace {

struct SnapGrid {
    const Mesh* m = nullptr;
    Vec3  lo;
    float cell = 1.0f;
    int   nx = 1, ny = 1, nz = 1;
    std::vector<uint64_t> keys;   // (cell << 32) | tri, sorted

    Vec3 P(uint32_t v) const { return {m->pos_x[v], m->pos_y[v], m->pos_z[v]}; }

    int64_t cell_of(int x, int y, int z) const { return ((int64_t)z * ny + y) * nx + x; }

    void build(const Mesh& mesh, Vec3 box_lo, Vec3 box_hi, float cell_size) {
        m = &mesh;
        lo = box_lo;
        cell = cell_size;
        nx = std::max(1, (int)std::ceil((box_hi.x - box_lo.x) / cell));
        ny = std::max(1, (int)std::ceil((box_hi.y - box_lo.y) / cell));
        nz = std::max(1, (int)std::ceil((box_hi.z - box_lo.z) / cell));
        keys.clear();
        const uint32_t tc = mesh.tri_count();
        for (uint32_t t = 0; t < tc; t++) {
            Vec3 a = P(mesh.indices[t * 3]), b = P(mesh.indices[t * 3 + 1]),
                 c = P(mesh.indices[t * 3 + 2]);
            Vec3 tlo = {std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}), std::min({a.z, b.z, c.z})};
            Vec3 thi = {std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}), std::max({a.z, b.z, c.z})};
            if (thi.x < box_lo.x || thi.y < box_lo.y || thi.z < box_lo.z ||
                tlo.x > box_hi.x || tlo.y > box_hi.y || tlo.z > box_hi.z) continue;
            int x0 = clampi((int)((tlo.x - lo.x) / cell), nx), x1 = clampi((int)((thi.x - lo.x) / cell), nx);
            int y0 = clampi((int)((tlo.y - lo.y) / cell), ny), y1 = clampi((int)((thi.y - lo.y) / cell), ny);
            int z0 = clampi((int)((tlo.z - lo.z) / cell), nz), z1 = clampi((int)((thi.z - lo.z) / cell), nz);
            for (int z = z0; z <= z1; z++)
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++)
                        keys.push_back(((uint64_t)cell_of(x, y, z) << 32) | t);
        }
        std::sort(keys.begin(), keys.end());
    }

    static int clampi(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }

    // Ericson, Real-Time Collision Detection 5.1.5.
    static Vec3 closest_on_tri(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
        Vec3 ab = b - a, ac = c - a, ap = p - a;
        float d1 = ab.dot(ap), d2 = ac.dot(ap);
        if (d1 <= 0 && d2 <= 0) return a;
        Vec3 bp = p - b;
        float d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0 && d4 <= d3) return b;
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
        Vec3 cp = p - c;
        float d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0 && d5 <= d6) return c;
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        float den = 1.0f / (va + vb + vc);
        return a + ab * (vb * den) + ac * (vc * den);
    }

    // Closest surface point to p, searching outward ring by ring. False when nothing
    // is within `max_rings` cells (the point wandered off the sculpt's near region).
    bool closest(Vec3 p, Vec3& out, Vec3& out_n, int max_rings = 4) const {
        const int cx = (int)std::floor((p.x - lo.x) / cell);
        const int cy = (int)std::floor((p.y - lo.y) / cell);
        const int cz = (int)std::floor((p.z - lo.z) / cell);
        float best = 1e30f;
        uint32_t best_t = 0xFFFFFFFFu;
        for (int r = 0; r <= max_rings; r++) {
            for (int z = cz - r; z <= cz + r; z++)
                for (int y = cy - r; y <= cy + r; y++)
                    for (int x = cx - r; x <= cx + r; x++) {
                        // Shell only: the inside was searched by smaller rings.
                        if (std::max({std::abs(x - cx), std::abs(y - cy), std::abs(z - cz)}) != r) continue;
                        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue;
                        const uint64_t k = (uint64_t)cell_of(x, y, z) << 32;
                        auto it = std::lower_bound(keys.begin(), keys.end(), k);
                        for (; it != keys.end() && (*it >> 32) == (k >> 32); ++it) {
                            const uint32_t t = (uint32_t)(*it & 0xFFFFFFFFu);
                            Vec3 a = P(m->indices[t * 3]), b = P(m->indices[t * 3 + 1]),
                                 c = P(m->indices[t * 3 + 2]);
                            Vec3 q = closest_on_tri(p, a, b, c);
                            float d = (q - p).dot(q - p);
                            if (d < best) { best = d; out = q; best_t = t; }
                        }
                    }
            // Anything beyond ring r is at least r cells away: stop once the best beats it.
            if (best_t != 0xFFFFFFFFu && std::sqrt(best) <= r * cell) break;
        }
        if (best_t == 0xFFFFFFFFu) return false;
        Vec3 a = P(m->indices[best_t * 3]), b = P(m->indices[best_t * 3 + 1]),
             c = P(m->indices[best_t * 3 + 2]);
        out_n = (b - a).cross(c - a).normalized();
        return true;
    }
};

Vec3 project_tangent(Vec3 v, Vec3 n) { return (v - n * v.dot(n)).normalized(); }

// Even arc-length resample of a polyline (closed: the wrap segment counts too).
void resample(const RetopoStroke& s, float step, std::vector<Vec3>& out_p,
              std::vector<float>& out_w) {
    const size_t n = s.pts.size();
    const size_t segs = s.closed ? n : n - 1;
    std::vector<float> acc(segs + 1, 0.0f);
    for (size_t i = 0; i < segs; i++)
        acc[i + 1] = acc[i] + (s.pts[(i + 1) % n] - s.pts[i]).length();
    const float L = acc[segs];
    int count = std::max(s.closed ? 6 : 2, (int)std::lround(L / step));
    // Closed: `count` samples share the loop evenly. Open: both ends are kept.
    const float d = s.closed ? L / count : L / std::max(1, count);
    const int samples = s.closed ? count : count + 1;
    out_p.clear(); out_w.clear();
    size_t seg = 0;
    for (int k = 0; k < samples; k++) {
        float t = std::min(k * d, L);
        while (seg + 1 < segs && acc[seg + 1] < t) seg++;
        float sl = acc[seg + 1] - acc[seg];
        float f = sl > 1e-9f ? (t - acc[seg]) / sl : 0.0f;
        const Vec3 a = s.pts[seg], b = s.pts[(seg + 1) % n];
        out_p.push_back(a + (b - a) * f);
        out_w.push_back(s.half_w[seg] + (s.half_w[(seg + 1) % n] - s.half_w[seg]) * f);
    }
}

// Pen jitter shows up as wiggly loops; two passes of neighbour averaging on the
// captured centreline (ends pinned on a strip) take it out without moving the shape.
void smooth_centreline(RetopoStroke& s, int passes) {
    const size_t n = s.pts.size();
    if (n < 3) return;
    std::vector<Vec3> tmp(n);
    for (int p = 0; p < passes; p++) {
        for (size_t i = 0; i < n; i++) {
            if (!s.closed && (i == 0 || i == n - 1)) { tmp[i] = s.pts[i]; continue; }
            const Vec3 a = s.pts[(i + n - 1) % n], b = s.pts[(i + 1) % n];
            tmp[i] = s.pts[i] * 0.5f + (a + b) * 0.25f;
        }
        s.pts.swap(tmp);
    }
}

void add_edge(RetopoMesh& rm, uint32_t a, uint32_t b, uint32_t owner) {
    rm.edges.push_back(a); rm.edges.push_back(b);
    rm.edge_owner.push_back(owner);
}

// One stroke → one quad ribbon appended to rm.
void build_stroke(const RetopoStroke& in, const SnapGrid& g, int rings, RetopoMesh& rm) {
    RetopoStroke s = in;
    smooth_centreline(s, 2);

    float avg_w = 0.0f;
    for (float w : s.half_w) avg_w += w;
    avg_w /= (float)s.half_w.size();
    // `rings` loops across the full band: K steps each side of the centre loop. The
    // same spacing runs along the stroke, so the quads come out roughly square.
    const int K = std::max(1, rings / 2);
    const float step = avg_w / (float)K;

    std::vector<Vec3> cp;
    std::vector<float> cw;
    resample(s, step, cp, cw);
    const int n = (int)cp.size();
    if (n < 2) return;

    // Snap the centre samples and take the surface normal there.
    std::vector<Vec3> cn(n);
    for (int i = 0; i < n; i++) {
        Vec3 q, qn;
        if (g.closest(cp[i], q, qn)) { cp[i] = q; cn[i] = qn; }
        else                         cn[i] = Vec3(0, 0, 1);
    }

    // Tangent and side (in the surface, across the stroke) at each sample.
    std::vector<Vec3> side(n);
    std::vector<float> curv(n, 0.0f);   // signed turn rate toward +side
    std::vector<Vec3> tan(n);
    for (int i = 0; i < n; i++) {
        int ia = s.closed ? (i + n - 1) % n : std::max(0, i - 1);
        int ib = s.closed ? (i + 1) % n : std::min(n - 1, i + 1);
        tan[i] = project_tangent(cp[ib] - cp[ia], cn[i]);
        side[i] = cn[i].cross(tan[i]).normalized();
    }
    // Smooth the spoke directions along the stroke: on a saddle, raw sides can fan
    // or cross between neighbours.
    for (int pass = 0; pass < 2; pass++) {
        std::vector<Vec3> t2 = side;
        for (int i = 0; i < n; i++) {
            if (!s.closed && (i == 0 || i == n - 1)) continue;
            Vec3 m = side[(i + n - 1) % n] + side[i] * 2.0f + side[(i + 1) % n];
            t2[i] = project_tangent(m, cn[i]);
        }
        side.swap(t2);
    }
    for (int i = 0; i < n; i++) {
        int ia = s.closed ? (i + n - 1) % n : std::max(0, i - 1);
        int ib = s.closed ? (i + 1) % n : std::min(n - 1, i + 1);
        float ds = (cp[ib] - cp[ia]).length();
        if (ds > 1e-9f) curv[i] = (tan[ib] - tan[ia]).dot(side[i]) / ds;
    }

    // Walk each spoke out across the surface. Small sub-steps, re-snapping and
    // re-flattening the direction every time, so the spoke follows the surface over
    // a curve (an eye socket) instead of shooting off along a flat tangent.
    const int SUB = 3;
    const int W = 2 * K + 1;                       // vertices per spoke
    const uint32_t base = (uint32_t)rm.v.size();
    rm.v.resize(base + (size_t)n * W);
    rm.n.resize(base + (size_t)n * W);
    for (int i = 0; i < n; i++) {
        rm.v[base + i * W + K] = cp[i];
        rm.n[base + i * W + K] = cn[i];
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float ext = cw[i];
            // The inside of a bend: spokes converge and cross if the band is wider
            // than the bend's radius (a small eye drawn with a fat brush). Clamp there.
            const float k = curv[i] * (float)sgn;
            if (k > 1e-6f) ext = std::min(ext, 0.8f / k);
            const float h = ext / (float)(K * SUB);
            Vec3 p = cp[i], nrm = cn[i], dir = side[i] * (float)sgn;
            for (int j = 1; j <= K; j++) {
                for (int sub = 0; sub < SUB; sub++) {
                    Vec3 q, qn;
                    p = p + dir * h;
                    if (g.closest(p, q, qn)) { p = q; nrm = qn; }
                    dir = project_tangent(dir, nrm);
                }
                const int col = K + sgn * j;
                rm.v[base + i * W + col] = p;
                rm.n[base + i * W + col] = nrm;
            }
        }
    }

    // Quads between neighbouring spokes. Winding is picked from the first quad
    // against the surface normal so every quad faces out of the sculpt.
    auto V = [&](int i, int j) { return base + (uint32_t)((i % n) * W + j); };
    const int spans = s.closed ? n : n - 1;
    bool flip = false;
    {
        Vec3 a = rm.v[V(0, 0)], b = rm.v[V(1, 0)], c = rm.v[V(1, 1)];
        flip = (b - a).cross(c - a).dot(cn[0]) < 0.0f;
    }
    for (int i = 0; i < spans; i++)
        for (int j = 0; j < W - 1; j++) {
            uint32_t q[4] = {V(i, j), V(i + 1, j), V(i + 1, j + 1), V(i, j + 1)};
            if (flip) std::swap(q[1], q[3]);
            rm.quads.insert(rm.quads.end(), q, q + 4);
        }
    // Edges: the loops (along) and the spokes (across), each once.
    for (int i = 0; i < spans; i++)
        for (int j = 0; j < W; j++) add_edge(rm, V(i, j), V(i + 1, j), in.id);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < W - 1; j++) add_edge(rm, V(i, j), V(i, j + 1), in.id);
}

RetopoStroke mirrored(const RetopoStroke& s) {
    // Reversed as well as reflected: a reflection flips handedness, and walking it
    // backwards flips it again, so the twin's quads wind the same way as the original's.
    RetopoStroke m = s;
    std::reverse(m.pts.begin(), m.pts.end());
    std::reverse(m.nrm.begin(), m.nrm.end());
    std::reverse(m.half_w.begin(), m.half_w.end());
    for (Vec3& p : m.pts) p.x = -p.x;
    for (Vec3& v : m.nrm) v.x = -v.x;
    return m;
}

}  // namespace

double retopo_build(RetopoState& st, const Mesh& m, bool mirror) {
    const auto t0 = std::chrono::steady_clock::now();
    st.result.clear();
    st.built = false;

    std::vector<RetopoStroke> todo;
    for (const RetopoStroke& s : st.strokes) {
        if (!s.visible || s.pts.size() < 3) continue;
        todo.push_back(s);
        if (!mirror) continue;
        // A stroke wholly on one side gets a twin; one that crosses x=0 (a mouth) is
        // built as drawn — symmetrising it is a later job.
        bool pos = false, neg = false;
        for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
        if (pos != neg) todo.push_back(mirrored(s));
    }
    if (todo.empty() || m.tri_count() == 0) return 0.0;

    // Region the strokes can reach: their bounds grown by the widest band (spokes walk
    // that far) plus a margin for the snap search.
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    float max_w = 0.0f, min_w = 1e30f;
    for (const RetopoStroke& s : todo)
        for (size_t i = 0; i < s.pts.size(); i++) {
            const Vec3& p = s.pts[i];
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
            max_w = std::max(max_w, s.half_w[i]);
            min_w = std::min(min_w, s.half_w[i]);
        }
    const float pad = max_w * 1.6f;
    lo = lo - Vec3(pad, pad, pad);
    hi = hi + Vec3(pad, pad, pad);
    // A cell about one ring step wide: small enough that a query touches few
    // triangles, big enough that the region stays a few hundred thousand cells.
    float cell = std::max(min_w / (float)std::max(1, st.rings / 2), 1e-5f);
    const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    cell = std::max(cell, ext / 96.0f);

    SnapGrid g;
    g.build(m, lo, hi, cell);
    for (const RetopoStroke& s : todo) build_stroke(s, g, st.rings, st.result);
    st.built = !st.result.quads.empty();

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[retopo] build: %zu strokes -> %u quads, %zu verts, %zu grid refs, %.1f ms\n",
                todo.size(), st.result.quad_count(), st.result.v.size(), g.keys.size(), ms);
    return ms;
}

bool retopo_export_obj(const RetopoMesh& rm, const char* path) {
    if (rm.quads.empty()) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "# Chisel retopo: %u quads\n", rm.quad_count());
    for (const Vec3& p : rm.v) std::fprintf(f, "v %.6f %.6f %.6f\n", p.x, p.y, p.z);
    for (size_t q = 0; q < rm.quads.size(); q += 4)
        std::fprintf(f, "f %u %u %u %u\n", rm.quads[q] + 1, rm.quads[q + 1] + 1,
                     rm.quads[q + 2] + 1, rm.quads[q + 3] + 1);
    std::fclose(f);
    return true;
}
