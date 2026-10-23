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

    s.rings = rings; s.density = density; s.width = width;
    // The fresh stroke is the selection: the sliders tune it straight away.
    for (RetopoStroke& o : strokes) o.selected = false;
    s.selected = true;
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
    // Its quads go with it; brought back, it is a sketch again until the next Build.
    redo.back().frozen = false;
    redo.back().selected = false;
    prune();
}

void RetopoState::redo_one() {
    if (redo.empty()) return;
    strokes.push_back(std::move(redo.back()));
    redo.pop_back();
}

const char* RetopoState::click_select(uint32_t id, bool add) {
    RetopoStroke* hit = id ? find(id) : nullptr;
    if (!add) {
        for (RetopoStroke& s : strokes) s.selected = &s == hit;
        return nullptr;
    }
    if (!hit) return nullptr;              // Ctrl+click on nothing keeps what's there
    if (hit->selected) { hit->selected = false; return nullptr; }
    const RetopoStroke* sel = first_selected();
    if (sel && sel->frozen != hit->frozen)
        return hit->frozen ? "Edited islands can't join a selection of editable ones"
                           : "Editable islands can't join a selection of edited ones";
    hit->selected = true;
    return nullptr;
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

// A stroke's spokes before they become quads: n spokes of W vertices, v[i*W + j].
struct Grid {
    int n = 0, W = 0;
    bool closed = false;
    bool flip = false;              // quad winding that faces out of the sculpt
    std::vector<Vec3> v, nrm;
    Vec3 centre(int i) const {
        const int Q = W - 1;
        return Q % 2 == 0 ? v[i * W + Q / 2] : (v[i * W + Q / 2] + v[i * W + Q / 2 + 1]) * 0.5f;
    }
};

// One stroke → its spokes, walked across the sculpt.
bool stroke_grid(const RetopoStroke& in, const SnapGrid& g, Grid& G) {
    const int rings = in.rings;
    const float density = in.density;
    RetopoStroke s = in;
    smooth_centreline(s, 2);

    float avg_w = 0.0f;
    for (float w : s.half_w) avg_w += w;
    avg_w /= (float)s.half_w.size();
    // `rings` = quads across the band (1..8), odd counts included: then no vertex sits
    // on the centreline. The same spacing runs along the stroke at density 1, so the
    // quads come out about square; lower density stretches them along it.
    const int Q = std::max(1, rings);
    const float step = 2.0f * avg_w / (float)Q;

    std::vector<Vec3> cp;
    std::vector<float> cw;
    resample(s, step / std::max(0.1f, density), cp, cw);
    const int n = (int)cp.size();
    if (n < 2) return false;

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
    // Spoke vertex j (0..Q) sits at signed distance ext * (2j/Q - 1) from the centre:
    // each half is walked in Q*SUB sub-steps of ext/(Q*SUB), so every vertex lands on
    // a whole sub-step, |2j - Q| * SUB of them out. Even Q puts one on the centre.
    const int SUB = 3;
    const int W = Q + 1;                           // vertices per spoke
    G.n = n; G.W = W; G.closed = s.closed;
    G.v.assign((size_t)n * W, Vec3());
    G.nrm.assign((size_t)n * W, Vec3());
    for (int i = 0; i < n; i++) {
        if (Q % 2 == 0) {
            G.v[i * W + Q / 2] = cp[i];
            G.nrm[i * W + Q / 2] = cn[i];
        }
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            float ext = cw[i];
            // The inside of a bend: spokes converge and cross if the band is wider
            // than the bend's radius (a small eye drawn with a fat brush). Clamp there.
            const float k = curv[i] * (float)sgn;
            if (k > 1e-6f) ext = std::min(ext, 0.8f / k);
            const float h = ext / (float)(Q * SUB);
            Vec3 p = cp[i], nrm = cn[i], dir = side[i] * (float)sgn;
            for (int s_i = 1; s_i <= Q * SUB; s_i++) {
                Vec3 q, qn;
                p = p + dir * h;
                if (g.closest(p, q, qn)) { p = q; nrm = qn; }
                dir = project_tangent(dir, nrm);
                if (s_i % SUB) continue;
                const int out = s_i / SUB;              // = |2j - Q|
                if ((out + Q) % 2) continue;            // not a vertex position
                const int col = (Q + sgn * out) / 2;
                G.v[i * W + col] = p;
                G.nrm[i * W + col] = nrm;
            }
        }
    }

    // Winding is picked from the first quad against the surface normal so every quad
    // faces out of the sculpt.
    const Vec3 a = G.v[0], b = G.v[W], c = G.v[W + 1];
    G.flip = (b - a).cross(c - a).dot(cn[0]) < 0.0f;
    return true;
}

// Quads and edges for spokes i0..i1 of G (a closed one wraps), vertex indices from idx.
void emit_faces(RetopoMesh& rm, const Grid& G, int i0, int i1, const std::vector<uint32_t>& idx,
                uint32_t owner, bool flip) {
    const int n = G.n, W = G.W;
    auto V = [&](int i, int j) { return idx[(size_t)(i % n) * W + j]; };
    const int span_end = G.closed ? n : i1;
    for (int i = i0; i < span_end; i++)
        for (int j = 0; j < W - 1; j++) {
            uint32_t q[4] = {V(i, j), V(i + 1, j), V(i + 1, j + 1), V(i, j + 1)};
            if (flip) std::swap(q[1], q[3]);
            rm.quads.insert(rm.quads.end(), q, q + 4);
            rm.quad_owner.push_back(owner);
        }
    // Edges: the loops (along) and the spokes (across). A joined end spoke repeats
    // the island's own edges; dedupe_edges drops those.
    for (int i = i0; i < span_end; i++)
        for (int j = 0; j < W; j++) add_edge(rm, V(i, j), V(i + 1, j), owner);
    for (int i = i0; i <= i1; i++)
        for (int j = 0; j < W - 1; j++) add_edge(rm, V(i, j), V(i, j + 1), owner);
}

// Spokes i0..i1 of G as quads owned by `owner`. A ~0u slot in idx gets a new vertex;
// any other (a joined end) reuses the island vertex given. Fills idx in.
void emit_grid(RetopoMesh& rm, const Grid& G, int i0, int i1, std::vector<uint32_t>& idx,
               uint32_t owner) {
    for (int i = i0; i <= i1; i++)
        for (int j = 0; j < G.W; j++) {
            uint32_t& k = idx[(size_t)i * G.W + j];
            if (k != ~0u) continue;
            k = (uint32_t)rm.v.size();
            rm.v.push_back(G.v[(size_t)i * G.W + j]);
            rm.n.push_back(G.nrm[(size_t)i * G.W + j]);
            rm.v_owner.push_back(owner);
            rm.mirror.push_back(~0u);
        }
    emit_faces(rm, G, i0, i1, idx, owner, G.flip);
}

// The mirror twin of what emit_grid just made: each vertex reflected in x = 0 and
// paired with its original by index (Grab and Smooth move pairs, never guess by
// position), quads wound the other way, since a reflection flips handedness. A joined
// vertex maps to its island's twin of it; an island without twins (one across x = 0,
// a mouth) lends the vertex sitting at the reflection, if there is one.
void emit_twin(RetopoMesh& rm, const Grid& G, int i0, int i1, const std::vector<uint32_t>& idx,
               uint32_t first_new, uint32_t owner, bool seam0, bool seam1) {
    std::vector<uint32_t> tw(idx.size(), ~0u);
    for (int i = i0; i <= i1; i++)
        for (int j = 0; j < G.W; j++) {
            const size_t k = (size_t)i * G.W + j;
            const uint32_t a = idx[k];
            // A half-mouth's end spoke lies on x = 0: it is its own twin, which is
            // what welds the two halves into one ring.
            if (a >= first_new && ((seam0 && i == i0) || (seam1 && i == i1))) {
                tw[k] = a;
                rm.mirror[a] = a;
                continue;
            }
            if (rm.mirror[a] != ~0u) { tw[k] = rm.mirror[a]; continue; }
            Vec3 p = rm.v[a], nn = rm.n[a];
            p.x = -p.x; nn.x = -nn.x;
            if (a < first_new) {
                const int jn = j + 1 < G.W ? j + 1 : j - 1;
                const float tol = 0.5f * (G.v[k] - G.v[(size_t)i * G.W + jn]).length();
                float bd = tol * tol;
                for (uint32_t b = 0; b < first_new; b++) {
                    const Vec3 d = rm.v[b] - p;
                    if (d.dot(d) < bd) { bd = d.dot(d); tw[k] = b; }
                }
                if (tw[k] != ~0u) continue;
            }
            tw[k] = (uint32_t)rm.v.size();
            rm.v.push_back(p);
            rm.n.push_back(nn);
            rm.v_owner.push_back(owner);
            rm.mirror.push_back(a);
            if (a >= first_new) rm.mirror[a] = tw[k];
        }
    emit_faces(rm, G, i0, i1, tw, owner, !G.flip);
}

// Merge edges listed twice (a joined end spoke, a weld); the first owner listed keeps it.
void dedupe_edges(RetopoMesh& rm) {
    std::vector<std::pair<uint64_t, uint32_t>> es;
    es.reserve(rm.edge_owner.size());
    for (size_t e = 0; e < rm.edge_owner.size(); e++) {
        const uint32_t x = rm.edges[e * 2], y = rm.edges[e * 2 + 1];
        if (x == y) continue;
        es.push_back({((uint64_t)std::min(x, y) << 32) | std::max(x, y), rm.edge_owner[e]});
    }
    std::stable_sort(es.begin(), es.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    rm.edges.clear(); rm.edge_owner.clear();
    for (size_t i = 0; i < es.size(); i++) {
        if (i > 0 && es[i].first == es[i - 1].first) continue;
        rm.edges.push_back((uint32_t)(es[i].first >> 32));
        rm.edges.push_back((uint32_t)(es[i].first & 0xFFFFFFFFu));
        rm.edge_owner.push_back(es[i].second);
    }
}

// Neighbours and borders for Smooth. A border edge belongs to one quad only.
void build_topology(RetopoMesh& rm) {
    const uint32_t nv = (uint32_t)rm.v.size();
    rm.nb_off.assign(nv + 1, 0);
    for (size_t e = 0; e < rm.edges.size(); e++) rm.nb_off[rm.edges[e] + 1]++;
    for (uint32_t i = 0; i < nv; i++) rm.nb_off[i + 1] += rm.nb_off[i];
    rm.nb.resize(rm.edges.size());
    std::vector<uint32_t> fill(rm.nb_off.begin(), rm.nb_off.end() - 1);
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) {
        rm.nb[fill[rm.edges[e]]++] = rm.edges[e + 1];
        rm.nb[fill[rm.edges[e + 1]]++] = rm.edges[e];
    }
    std::vector<uint64_t> qe;
    qe.reserve(rm.quads.size());
    for (size_t q = 0; q < rm.quads.size(); q += 4)
        for (int k = 0; k < 4; k++) {
            uint32_t a = rm.quads[q + k], b = rm.quads[q + (k + 1) % 4];
            qe.push_back(((uint64_t)std::min(a, b) << 32) | std::max(a, b));
        }
    std::sort(qe.begin(), qe.end());
    rm.border_nb.assign((size_t)nv * 2, ~0u);
    rm.pinned.assign(nv, 0);
    std::vector<uint8_t> nborder(nv, 0);
    for (size_t i = 0; i < qe.size(); ) {
        size_t j = i;
        while (j < qe.size() && qe[j] == qe[i]) j++;
        if (j - i == 1) {
            const uint32_t a = (uint32_t)(qe[i] >> 32), b = (uint32_t)(qe[i] & 0xFFFFFFFFu);
            if (nborder[a] < 2) rm.border_nb[a * 2 + nborder[a]] = b;
            if (nborder[b] < 2) rm.border_nb[b * 2 + nborder[b]] = a;
            nborder[a]++; nborder[b]++;
        }
        i = j;
    }
    for (uint32_t i = 0; i < nv; i++) {
        if (nborder[i] == 0) continue;
        if (nborder[i] != 2) { rm.pinned[i] = 1; continue; }
        // A corner (a strip's end): the two border edges meet at an angle rather than
        // running on through it. Sliding it would round the strip off.
        const Vec3 a = (rm.v[rm.border_nb[i * 2]] - rm.v[i]).normalized();
        const Vec3 b = (rm.v[rm.border_nb[i * 2 + 1]] - rm.v[i]).normalized();
        if (a.dot(b) > -0.7f) rm.pinned[i] = 1;
    }
}

// A stroke across x = 0 (a mouth) with mirror on: keep its longest run on one side, cut
// exactly at x = 0, so Build makes that half plus its twin and they meet on the seam.
// seam0 / seam1: which ends of `out` sit on the seam (a strip's own end doesn't).
bool half_stroke(const RetopoStroke& s, RetopoStroke& out, bool& seam0, bool& seam1) {
    const size_t n = s.pts.size();
    const size_t segs = s.closed ? n : n - 1;
    // The side holding more of the stroke is the one drawn with care.
    float lp = 0.0f, ln = 0.0f;
    for (size_t i = 0; i < segs; i++) {
        const Vec3 a = s.pts[i], b = s.pts[(i + 1) % n];
        ((a.x + b.x) > 0.0f ? lp : ln) += (b - a).length();
    }
    const float sg = lp >= ln ? 1.0f : -1.0f;
    auto in = [&](size_t i) { return s.pts[i % n].x * sg > 0.0f; };
    // Runs of points on that side, in stroke order (closed: starting after a point that
    // isn't, so no run is split by the wrap). The longest by length wins.
    size_t start = 0;
    if (s.closed) { while (start < n && in(start)) start++; if (start == n) return false; start++; }
    const size_t count = s.closed ? n : n;
    size_t best_a = 0, best_len = 0;
    float best_l = -1.0f;
    for (size_t k = 0; k < count; ) {
        if (!in(start + k)) { k++; continue; }
        size_t e = k;
        float l = 0.0f;
        while (e + 1 < count && in(start + e + 1)) {
            l += (s.pts[(start + e + 1) % n] - s.pts[(start + e) % n]).length();
            e++;
        }
        if (l > best_l) { best_l = l; best_a = k; best_len = e - k + 1; }
        k = e + 1;
    }
    if (best_l < 0.0f) return false;
    out = s;
    out.pts.clear(); out.nrm.clear(); out.half_w.clear();
    out.closed = false;
    // The point where segment i→j crosses x = 0, exactly on the plane.
    auto cross = [&](size_t i, size_t j) {
        const Vec3 A = s.pts[i % n], B = s.pts[j % n];
        const float t = A.x / (A.x - B.x);
        Vec3 p = A + (B - A) * t;
        p.x = 0.0f;
        out.pts.push_back(p);
        out.nrm.push_back((s.nrm[i % n] + (s.nrm[j % n] - s.nrm[i % n]) * t).normalized());
        out.half_w.push_back(s.half_w[i % n] + (s.half_w[j % n] - s.half_w[i % n]) * t);
    };
    const size_t a = start + best_a, b = a + best_len - 1;
    seam0 = s.closed || a > 0;
    seam1 = s.closed || b < n - 1;
    if (seam0) cross(a - 1, a);
    for (size_t i = a; i <= b; i++) {
        out.pts.push_back(s.pts[i % n]);
        out.nrm.push_back(s.nrm[i % n]);
        out.half_w.push_back(s.half_w[i % n]);
    }
    if (seam1) cross(b, b + 1);
    return out.pts.size() >= 3;
}

bool alive(const RetopoMesh& rm, uint32_t i) { return rm.nb_off[i + 1] > rm.nb_off[i]; }
bool on_border(const RetopoMesh& rm, uint32_t i) { return rm.border_nb[i * 2] != ~0u; }

// Distance from p to the nearest quad not owned by `owner`; its owner in `who`.
float dist_to_others(const RetopoMesh& rm, uint32_t owner, Vec3 p, uint32_t& who) {
    float best = 1e30f;
    for (size_t q = 0; q < rm.quad_owner.size(); q++) {
        if (rm.quad_owner[q] == owner) continue;
        const Vec3 a = rm.v[rm.quads[q * 4]], b = rm.v[rm.quads[q * 4 + 1]],
                   c = rm.v[rm.quads[q * 4 + 2]], d = rm.v[rm.quads[q * 4 + 3]];
        for (int t = 0; t < 2; t++) {
            const Vec3 x = SnapGrid::closest_on_tri(p, a, t ? c : b, t ? d : c) - p;
            const float d2 = x.dot(x);
            if (d2 < best) { best = d2; who = rm.quad_owner[q]; }
        }
    }
    return std::sqrt(best);
}

// Bridges. A strip end that lands on an island built before it (or stops just short
// of one) is cut back to that island's outer edge, and its end spoke takes over the
// island's run of edge vertices that lines up best with it: shared vertices, shared
// edges, no T-junction. The island keeps its shape; the strip's last row stretches to
// meet it. `end` 0 = the first spoke, 1 = the last; i0/i1 step in past what's cut.
// rm's topology must be current.
bool join_end(RetopoMesh& rm, const Grid& G, int end, uint32_t owner, int& i0, int& i1,
              std::vector<uint32_t>& idx) {
    const int W = G.W, dir = end == 0 ? 1 : -1;
    int ie = end == 0 ? i0 : i1;
    const float along = (G.centre(ie + dir) - G.centre(ie)).length();
    // Overlapping = the spoke's centre lies on another island's quads. 0.3 of a step
    // allows for flat quads on a curved surface.
    const float eps = std::max(1e-6f, 0.3f * along);
    uint32_t target = 0;
    while (i1 - i0 >= 2) {                        // keep a row of quads, at least
        uint32_t who = 0;
        if (dist_to_others(rm, owner, G.centre(ie), who) >= eps) break;
        target = who;
        if (end == 0) i0++; else i1--;
        ie += dir;
    }
    const Vec3 c = G.centre(ie);
    const float half = 0.5f * (G.v[(size_t)ie * W + W - 1] - G.v[(size_t)ie * W]).length();
    // Overlapped: that island's edge. Not: any island's edge within reach of the end.
    const float reach = std::max(half, along) * 1.5f;
    uint32_t b0 = ~0u;
    float bd = target ? 1e30f : reach * reach;
    for (uint32_t v = 0; v < (uint32_t)rm.v.size(); v++) {
        if (!alive(rm, v) || !on_border(rm, v) || rm.v_owner[v] == owner) continue;
        if (target && rm.v_owner[v] != target) continue;
        const Vec3 d = rm.v[v] - c;
        if (d.dot(d) < bd) { bd = d.dot(d); b0 = v; }
    }
    if (b0 == ~0u) return false;
    // The edge both ways from b0, then the window of W in a row that sits closest to
    // the end spoke, in either direction.
    auto walk = [&](uint32_t from, std::vector<uint32_t>& out) {
        uint32_t prev = b0, cur = from;
        while (cur != ~0u && cur != b0 && (int)out.size() < W - 1) {
            out.push_back(cur);
            const uint32_t nx = rm.border_nb[cur * 2] == prev ? rm.border_nb[cur * 2 + 1] : rm.border_nb[cur * 2];
            prev = cur; cur = nx;
        }
    };
    std::vector<uint32_t> L, R, chain;
    walk(rm.border_nb[b0 * 2], L);
    walk(rm.border_nb[b0 * 2 + 1], R);
    chain.assign(L.rbegin(), L.rend());
    chain.push_back(b0);
    chain.insert(chain.end(), R.begin(), R.end());
    float best = 1e30f;
    int best_s = -1;
    bool best_rev = false;
    for (int s0 = 0; s0 + W <= (int)chain.size(); s0++) {
        bool distinct = true;   // a short loop walked both ways meets itself
        for (int a = 0; a < W && distinct; a++)
            for (int b = a + 1; b < W; b++)
                if (chain[s0 + a] == chain[s0 + b]) { distinct = false; break; }
        if (!distinct) continue;
        for (int rev = 0; rev < 2; rev++) {
            float cost = 0.0f;
            for (int j = 0; j < W; j++) {
                const Vec3 d = rm.v[chain[s0 + (rev ? W - 1 - j : j)]] - G.v[(size_t)ie * W + j];
                cost += d.dot(d);
            }
            if (cost < best) { best = cost; best_s = s0; best_rev = rev; }
        }
    }
    if (best_s < 0) return false;
    // A run far off the strip's end means the edge there doesn't face it: leave it be.
    const float lim = 2.0f * std::max(half, along);
    if (best > (float)W * lim * lim) return false;
    for (int j = 0; j < W; j++)
        idx[(size_t)ie * W + j] = chain[best_s + (best_rev ? W - 1 - j : j)];
    return true;
}

// Copy into dst the quads whose owner passes keep(), the vertices they use, and the
// kept owners' edges between those vertices.
template <typename Keep>
void extract(const RetopoMesh& src, RetopoMesh& dst, Keep keep) {
    dst.clear();
    std::vector<uint32_t> remap(src.v.size(), ~0u);
    for (size_t q = 0; q < src.quad_owner.size(); q++) {
        if (!keep(src.quad_owner[q])) continue;
        for (int k = 0; k < 4; k++) {
            const uint32_t i = src.quads[q * 4 + k];
            if (remap[i] == ~0u) {
                remap[i] = (uint32_t)dst.v.size();
                dst.v.push_back(src.v[i]);
                dst.n.push_back(src.n[i]);
                dst.v_owner.push_back(src.v_owner[i]);
                dst.mirror.push_back(src.mirror[i]);   // old index, remapped below
            }
            dst.quads.push_back(remap[i]);
        }
        dst.quad_owner.push_back(src.quad_owner[q]);
    }
    for (uint32_t& m : dst.mirror) m = m == ~0u ? ~0u : remap[m];
    for (size_t e = 0; e < src.edge_owner.size(); e++) {
        const uint32_t a = remap[src.edges[e * 2]], b = remap[src.edges[e * 2 + 1]];
        if (a == ~0u || b == ~0u || !keep(src.edge_owner[e])) continue;
        dst.edges.push_back(a); dst.edges.push_back(b);
        dst.edge_owner.push_back(src.edge_owner[e]);
    }
}


}  // namespace

double retopo_build(RetopoState& st, const Mesh& m, bool mirror) {
    const auto t0 = std::chrono::steady_clock::now();

    // Frozen strokes' quads are carried over untouched (still editable by Grab and
    // Smooth, just out of reach of rings and density); everything else is rebuilt.
    // The old result must survive until this copy: clearing it first lost them all.
    {
        RetopoMesh keep;
        extract(st.result, keep, [&](uint32_t owner) {
            const RetopoStroke* s = st.find(owner);
            return s && s->frozen;
        });
        st.result = std::move(keep);
    }
    // Donuts first, then strips in the order drawn: a strip joins anything built
    // before it. A stroke wholly on one side of x = 0 gets a twin with mirror on; one
    // that crosses it (a mouth) is built as drawn — symmetrising it is a later job.
    // With mirror, one that crosses it (a mouth) is cut at x = 0 and built as a half plus
    // its twin, welded along the seam: symmetric, one piece.
    struct Todo { RetopoStroke s; bool twin; bool seam0 = false, seam1 = false; };
    std::vector<Todo> todo;
    for (int pass = 0; pass < 2; pass++)
        for (const RetopoStroke& s : st.strokes) {
            if (s.frozen || !s.visible || s.pts.size() < 3 || s.closed != (pass == 0)) continue;
            bool pos = false, neg = false;
            for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
            Todo t{s, mirror && pos != neg};
            if (mirror && pos && neg && half_stroke(s, t.s, t.seam0, t.seam1)) t.twin = true;
            for (float& w : t.s.half_w) w *= s.width;
            todo.push_back(std::move(t));
        }
    if (todo.empty() || m.tri_count() == 0) {
        st.built = !st.result.quads.empty();
        if (st.built) build_topology(st.result);
        std::printf("[retopo] build: nothing to rebuild, %u frozen quads kept\n", st.result.quad_count());
        return 0.0;
    }

    // Region the strokes can reach: their bounds grown by the widest band (spokes walk
    // that far) plus a margin for the snap search.
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    // A cell about one ring step wide (the finest stroke's): small enough that a query
    // touches few triangles, big enough that the region stays a few hundred thousand cells.
    float max_w = 0.0f, cell = 1e30f;
    for (const Todo& t : todo)
        for (size_t i = 0; i < t.s.pts.size(); i++) {
            const Vec3& p = t.s.pts[i];
            lo = {std::min(lo.x, t.twin ? -std::abs(p.x) : p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, t.twin ? std::abs(p.x) : p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
            max_w = std::max(max_w, t.s.half_w[i]);
            cell = std::min(cell, 2.0f * t.s.half_w[i] / (float)std::max(1, t.s.rings));
        }
    const float pad = max_w * 1.6f;
    lo = lo - Vec3(pad, pad, pad);
    hi = hi + Vec3(pad, pad, pad);
    cell = std::max(cell, 1e-5f);
    const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    cell = std::max(cell, ext / 96.0f);

    SnapGrid g;
    g.build(m, lo, hi, cell);
    RetopoMesh& rm = st.result;
    int joins = 0;
    for (const Todo& t : todo) {
        Grid G;
        if (!stroke_grid(t.s, g, G)) continue;
        std::vector<uint32_t> idx((size_t)G.n * G.W, ~0u);
        int i0 = 0, i1 = G.n - 1;
        // A half-mouth's seam spokes go flat onto x = 0 (the snap can drift a hair).
        for (int e = 0; e < 2; e++) {
            if (!(e ? t.seam1 : t.seam0)) continue;
            const int i = e ? G.n - 1 : 0;
            for (int j = 0; j < G.W; j++) {
                G.v[(size_t)i * G.W + j].x = 0.0f;
                Vec3& nn = G.nrm[(size_t)i * G.W + j];
                nn.x = 0.0f;
                if (nn.length() > 1e-6f) nn = nn.normalized();
            }
        }
        if (!G.closed && !rm.quads.empty()) {
            dedupe_edges(rm);
            build_topology(rm);
            // Seam ends meet their twin, not an island.
            if (!t.seam0) joins += join_end(rm, G, 0, t.s.id, i0, i1, idx);
            if (!t.seam1) joins += join_end(rm, G, 1, t.s.id, i0, i1, idx);
        }
        const uint32_t first_new = (uint32_t)rm.v.size();
        emit_grid(rm, G, i0, i1, idx, t.s.id);
        if (t.twin) emit_twin(rm, G, i0, i1, idx, first_new, t.s.id, t.seam0, t.seam1);
    }
    dedupe_edges(rm);
    st.built = !rm.quads.empty();
    if (st.built) build_topology(rm);

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[retopo] build: %zu strokes -> %u quads, %zu verts, %d strip ends joined, %.1f ms\n",
                todo.size(), st.result.quad_count(), st.result.v.size(), joins, ms);
    return ms;
}

// ---- Picking -----------------------------------------------------------------------

uint32_t retopo_pick(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                     float x, float y, bool mirror) {
    auto facing = [&](Vec3 p, Vec3 n) {
        const Vec3 to_eye = cam.perspective ? cam.get_position() - p : cam.get_view_direction() * -1.0f;
        return n.dot(to_eye) > 0.0f;
    };
    // Built quads first: the cursor inside a front-facing quad, the nearest one wins.
    const RetopoMesh& rm = st.result;
    uint32_t best = 0;
    float best_d = 1e30f;
    for (size_t q = 0; q < rm.quad_owner.size(); q++) {
        float sx[4], sy[4];
        bool ok = true;
        for (int k = 0; k < 4 && ok; k++)
            ok = cam.world_to_screen(rm.v[rm.quads[q * 4 + k]], win_w, win_h, sx[k], sy[k]);
        if (!ok || !facing(rm.v[rm.quads[q * 4]], rm.n[rm.quads[q * 4]])) continue;
        // Inside = on the same side of all four edges (either winding).
        int pos = 0, neg = 0;
        for (int k = 0; k < 4; k++) {
            const int j = (k + 1) & 3;
            const float c = (sx[j] - sx[k]) * (y - sy[k]) - (sy[j] - sy[k]) * (x - sx[k]);
            pos += c > 0.0f; neg += c < 0.0f;
        }
        if (pos && neg) continue;
        const float d = cam.view_depth(rm.v[rm.quads[q * 4]]);
        if (d < best_d) { best_d = d; best = rm.quad_owner[q]; }
    }
    if (best) return best;

    // Then unbuilt strokes: within the band's drawn width of the centreline (at least a
    // few pixels, for a thin one). Closest relative to its width wins.
    float best_r = 1.0f;
    for (const RetopoStroke& s : st.strokes) {
        if (s.frozen || !s.visible || s.pts.size() < 2) continue;
        bool pos = false, neg = false;
        for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
        const int sides = mirror && pos != neg ? 2 : 1;
        for (int side = 0; side < sides; side++) {
            float px = 0, py = 0;
            bool prev = false;
            const size_t segs = s.closed ? s.pts.size() + 1 : s.pts.size();
            for (size_t i = 0; i < segs; i++) {
                const size_t k = i % s.pts.size();
                Vec3 p = s.pts[k], n = s.nrm[k];
                if (side) { p.x = -p.x; n.x = -n.x; }
                float qx, qy;
                const bool ok = cam.world_to_screen(p, win_w, win_h, qx, qy) && facing(p, n);
                if (ok && prev) {
                    const float ex = qx - px, ey = qy - py, l2 = ex * ex + ey * ey;
                    const float t = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, ((x - px) * ex + (y - py) * ey) / l2)) : 0.0f;
                    const float d = std::hypot(x - (px + ex * t), y - (py + ey * t));
                    const float wpp = 2.0f * cam.half_height_at(cam.view_depth(p)) / (float)win_h;
                    const float r = std::max(6.0f, wpp > 0.0f ? s.half_w[k] * s.width / wpp : 0.0f);
                    if (d / r < best_r) { best_r = d / r; best = s.id; }
                }
                px = qx; py = qy; prev = ok;
            }
        }
    }
    return best;
}

// ---- Prune, freeze, weld ---------------------------------------------------------

void RetopoState::prune() {
    RetopoMesh keep;
    extract(result, keep, [&](uint32_t owner) { return find(owner) != nullptr; });
    result = std::move(keep);
    built = !result.quads.empty();
    if (built) build_topology(result);
}

namespace {

// b is a or within two edges of it: welding there would fold a quad onto itself.
bool near_in_topology(const RetopoMesh& rm, uint32_t a, uint32_t b) {
    for (uint32_t k = rm.nb_off[a]; k < rm.nb_off[a + 1]; k++) {
        const uint32_t n = rm.nb[k];
        if (n == b) return true;
        for (uint32_t k2 = rm.nb_off[n]; k2 < rm.nb_off[n + 1]; k2++)
            if (rm.nb[k2] == b) return true;
    }
    return false;
}

// Fold vertex a into b: every quad and edge that used a now uses b, edges that became
// duplicates go. a is left behind unreferenced (export skips it).
void weld(RetopoMesh& rm, uint32_t a, uint32_t b) {
    for (uint32_t& q : rm.quads) if (q == a) q = b;
    for (uint32_t& e : rm.edges) if (e == a) e = b;
    dedupe_edges(rm);
    build_topology(rm);
}

// After an edit moved `moved`: freeze their strokes, then weld any moved outer-edge
// vertex that has reached another outer-edge vertex. The reach is 0.4 of the vertex's
// shortest edge, so it takes a deliberate push, not a near miss, to zip two edges.
void after_edit(RetopoState& st, const std::vector<uint32_t>& moved, bool mirror) {
    RetopoMesh& rm = st.result;
    std::vector<uint32_t> touched;
    for (uint32_t i : moved) if (i < rm.v_owner.size()) touched.push_back(rm.v_owner[i]);
    std::vector<uint32_t> border;
    for (uint32_t i = 0; i < (uint32_t)rm.v.size(); i++)
        if (alive(rm, i) && on_border(rm, i)) border.push_back(i);
    int welds = 0;
    for (uint32_t a : moved) {
        if (a >= rm.v.size() || !alive(rm, a) || !on_border(rm, a)) continue;
        float emin = 1e30f;
        for (uint32_t k = rm.nb_off[a]; k < rm.nb_off[a + 1]; k++)
            emin = std::min(emin, (rm.v[rm.nb[k]] - rm.v[a]).length());
        const float tol = 0.4f * emin;
        float best_d2 = tol * tol;
        uint32_t best = ~0u;
        for (uint32_t b : border) {
            if (b == a || !alive(rm, b) || !on_border(rm, b)) continue;
            const Vec3 d = rm.v[b] - rm.v[a];
            const float d2 = d.dot(d);
            if (d2 >= best_d2 || near_in_topology(rm, a, b)) continue;
            best_d2 = d2;
            best = b;
        }
        if (best == ~0u) continue;
        touched.push_back(rm.v_owner[best]);
        const uint32_t pa = mirror ? rm.mirror[a] : ~0u, pb = mirror ? rm.mirror[best] : ~0u;
        weld(rm, a, best);
        welds++;
        if (pa == best) {
            // A vertex met its own twin: one vertex now, on the seam.
            rm.v[best].x = 0.0f;
            rm.mirror[best] = best;
        } else if (pa != ~0u && pb != ~0u && pa != pb && alive(rm, pa) && alive(rm, pb)
                   && !near_in_topology(rm, pa, pb)) {
            weld(rm, pa, pb);   // the twins weld too, so the sides stay alike
            welds++;
        }
    }
    // A joined group is one piece: freezing one island freezes every island that shares
    // a vertex with it, or the next Build would rebuild a neighbour out from under the
    // frozen one's shared vertices.
    uint32_t max_id = 0;
    for (uint32_t o : rm.quad_owner) max_id = std::max(max_id, o);
    std::vector<uint32_t> up(max_id + 1);
    for (uint32_t i = 0; i <= max_id; i++) up[i] = i;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    std::vector<uint32_t> first(rm.v.size(), ~0u);
    for (size_t q = 0; q < rm.quad_owner.size(); q++)
        for (int k = 0; k < 4; k++) {
            uint32_t& f = first[rm.quads[q * 4 + k]];
            if (f == ~0u) f = rm.quad_owner[q];
            else          up[root(f)] = root(rm.quad_owner[q]);
        }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (RetopoStroke& s : st.strokes) {
        if (s.frozen || s.id > max_id) continue;
        for (uint32_t id : touched)
            if (id <= max_id && root(id) == root(s.id)) { s.frozen = true; break; }
    }
    if (welds) std::printf("[retopo] welded %d vertex pair(s)\n", welds);
}

}  // namespace

// ---- Smooth ---------------------------------------------------------------------

namespace {
SnapGrid g_smooth_grid;          // one retopo session at a time; rebuilt every pen-down
std::vector<Vec3>  g_smooth_old; // Jacobi: every vertex reads last frame's neighbours
std::vector<float> g_smooth_w;
// Grab: what the press took, and where it was; with mirror, their twins follow.
std::vector<uint32_t> g_grab_idx;
std::vector<float>    g_grab_w;
std::vector<Vec3>     g_grab_start;
std::vector<uint8_t>  g_grab_in;      // per vertex: taken by the press
bool g_grab_mirror = false;

// With mirror: set each moved vertex's twin to its reflection, and list both in `out`
// for after_edit. When both of a pair moved (the ring straddles x = 0) they meet
// halfway; a seam vertex stays on the seam.
void mirror_pairs(RetopoMesh& rm, const std::vector<uint32_t>& moved, const std::vector<uint8_t>& in,
                  std::vector<uint32_t>& out) {
    out = moved;
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };
    for (uint32_t i : moved) {
        // Welded away mid-drag: it no longer stands for anything, don't let it steer.
        if (!alive(rm, i)) continue;
        const uint32_t p = rm.mirror[i];
        if (p == ~0u || !alive(rm, p)) continue;
        if (p == i) { rm.v[i].x = 0.0f; continue; }
        if (in[p]) {
            if (i > p) continue;
            const Vec3 m = (rm.v[i] + refl(rm.v[p])) * 0.5f;
            rm.v[i] = m; rm.v[p] = refl(m);
            rm.n[p] = refl(rm.n[i]);
            continue;
        }
        rm.v[p] = refl(rm.v[i]);
        rm.n[p] = refl(rm.n[i]);
        out.push_back(p);
    }
}

// Ring weight of p (normal n) around the cursor: smooth to zero at the edge, zero when
// it faces away or projects outside.
float ring_weight(const Camera& cam, int win_w, int win_h, float cx, float cy, float r2,
                  Vec3 p, Vec3 n) {
    const Vec3 to_eye = cam.perspective ? cam.get_position() - p : cam.get_view_direction() * -1.0f;
    if (n.dot(to_eye) <= 0.0f) return 0.0f;
    float x, y;
    if (!cam.world_to_screen(p, win_w, win_h, x, y)) return 0.0f;
    const float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    if (d2 >= r2) return 0.0f;
    const float t = 1.0f - d2 / r2;
    return t * t;
}
}

void retopo_edit_begin(RetopoState& st, const Mesh& m) {
    const RetopoMesh& rm = st.result;
    if (rm.v.empty() || m.tri_count() == 0) return;
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    float edge = 0.0f;
    for (const Vec3& p : rm.v) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2)
        edge += (rm.v[rm.edges[e]] - rm.v[rm.edges[e + 1]]).length();
    edge /= (float)std::max<size_t>(1, rm.edges.size() / 2);
    const float pad = edge * 2.0f;
    lo = lo - Vec3(pad, pad, pad);
    hi = hi + Vec3(pad, pad, pad);
    const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    g_smooth_grid.build(m, lo, hi, std::max(edge * 0.5f, ext / 96.0f));
    g_smooth_old.resize(rm.v.size());
    g_smooth_w.resize(rm.v.size());
}

int retopo_smooth_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float cx, float cy, float radius_px, float strength, bool mirror) {
    RetopoMesh& rm = st.result;
    if (!st.built || g_smooth_grid.m == nullptr || g_smooth_w.size() != rm.v.size()) return 0;
    const float r2 = radius_px * radius_px;
    int moved = 0;
    for (size_t i = 0; i < rm.v.size(); i++) {
        const float w = ring_weight(cam, win_w, win_h, cx, cy, r2, rm.v[i], rm.n[i]);
        g_smooth_w[i] = rm.pinned[i] ? 0.0f : w * strength;
        moved += g_smooth_w[i] > 0.0f;
    }
    if (!moved) return 0;
    // Slide p along the line through a and b toward their midpoint. Only the part along
    // the line counts: on a curved loop the midpoint sits inside the curve, and taking
    // that pull too is what shrinks loops.
    auto slide = [](Vec3 p, Vec3 a, Vec3 b) {
        const Vec3 t = (b - a).normalized();
        return t * ((a + b) * 0.5f - p).dot(t);
    };
    g_smooth_old = rm.v;
    for (size_t i = 0; i < rm.v.size(); i++) {
        const float w = g_smooth_w[i];
        if (w <= 0.0f || !alive(rm, (uint32_t)i)) continue;
        const Vec3 p = g_smooth_old[i];
        const uint32_t k0 = rm.nb_off[i], k1 = rm.nb_off[i + 1];
        Vec3 d;
        if (rm.border_nb[i * 2] != ~0u) {
            // Border: slide along the border line only, so widths and holes hold.
            d = slide(p, g_smooth_old[rm.border_nb[i * 2]], g_smooth_old[rm.border_nb[i * 2 + 1]]);
        } else {
            // A grid vertex (four neighbours) has two straight-through pairs, one along
            // its loop and one across the band: slide along each. Averaging all four
            // instead pulled every loop toward the inside of its bend, so a donut's
            // middle loops crept into the hole; even spacing is where this one rests.
            // Pairs are found by direction (most opposite), so welds can't confuse them.
            int pa = -1;
            if (k1 - k0 == 4) {
                Vec3 u[4];
                for (int k = 0; k < 4; k++) u[k] = (g_smooth_old[rm.nb[k0 + k]] - p).normalized();
                float best = -0.3f;   // not opposite enough: not a grid vertex, average instead
                for (int k = 1; k < 4; k++)
                    if (u[0].dot(u[k]) < best) { best = u[0].dot(u[k]); pa = k; }
            }
            if (pa > 0) {
                const int r0 = pa == 1 ? 2 : 1, r1 = pa == 3 ? 2 : 3;
                auto N = [&](int k) { return g_smooth_old[rm.nb[k0 + k]]; };
                d = slide(p, N(0), N(pa)) + slide(p, N(r0), N(r1));
            } else {
                // Anything else (a weld's odd valence): toward the neighbours' average.
                Vec3 avg;
                for (uint32_t k = k0; k < k1; k++) avg += g_smooth_old[rm.nb[k]];
                d = avg * (1.0f / (float)std::max(1u, k1 - k0)) - p;
            }
            // Tangential only: along the normal would sink the patch into the sculpt.
            d = d - rm.n[i] * d.dot(rm.n[i]);
        }
        Vec3 q, qn;
        if (g_smooth_grid.closest(p + d * std::min(1.0f, w), q, qn)) { rm.v[i] = q; rm.n[i] = qn; }
    }
    static std::vector<uint32_t> list, both;
    static std::vector<uint8_t> in;
    list.clear();
    in.assign(rm.v.size(), 0);
    for (size_t i = 0; i < rm.v.size(); i++)
        if (g_smooth_w[i] > 0.0f) { list.push_back((uint32_t)i); in[i] = 1; }
    if (mirror) mirror_pairs(rm, list, in, both);
    after_edit(st, mirror ? both : list, mirror);
    return moved;
}

int retopo_grab_begin(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float cx, float cy, float radius_px, bool mirror) {
    const RetopoMesh& rm = st.result;
    g_grab_idx.clear(); g_grab_w.clear(); g_grab_start.clear();
    g_grab_in.assign(rm.v.size(), 0);
    g_grab_mirror = mirror;
    if (!st.built || g_smooth_grid.m == nullptr) return 0;
    const float r2 = radius_px * radius_px;
    for (size_t i = 0; i < rm.v.size(); i++) {
        const float w = ring_weight(cam, win_w, win_h, cx, cy, r2, rm.v[i], rm.n[i]);
        if (w <= 0.0f) continue;
        g_grab_idx.push_back((uint32_t)i);
        g_grab_w.push_back(w);
        g_grab_start.push_back(rm.v[i]);
        g_grab_in[i] = 1;
    }
    return (int)g_grab_idx.size();
}

void retopo_grab_drag(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float dx_px, float dy_px) {
    RetopoMesh& rm = st.result;
    if (!st.built || g_grab_idx.empty()) return;
    const Vec3 cam_pos = cam.get_position();
    const Vec3 fwd = (cam.target - cam_pos).normalized();
    const Vec3 right = fwd.cross(Vec3(0, 1, 0)).normalized();
    const Vec3 up = right.cross(fwd).normalized();
    for (size_t k = 0; k < g_grab_idx.size(); k++) {
        const uint32_t i = g_grab_idx[k];
        if (i >= rm.v.size()) continue;
        const Vec3 p0 = g_grab_start[k];
        // The pointer's pixels, as world distance at this vertex's own depth (the
        // foreshortening term: under perspective a far vertex needs more world per pixel).
        const float wpp = 2.0f * cam.half_height_at(cam.view_depth(p0)) / (float)win_h;
        const Vec3 d = (right * dx_px - up * dy_px) * (wpp * g_grab_w[k]);
        Vec3 q, qn;
        if (g_smooth_grid.closest(p0 + d, q, qn, 8)) { rm.v[i] = q; rm.n[i] = qn; }
    }
    if (dx_px == 0.0f && dy_px == 0.0f) return;
    if (g_grab_in.size() != rm.v.size()) g_grab_in.resize(rm.v.size(), 0);   // a weld can't add, but be safe
    static std::vector<uint32_t> both;
    if (g_grab_mirror) mirror_pairs(rm, g_grab_idx, g_grab_in, both);
    after_edit(st, g_grab_mirror ? both : g_grab_idx, g_grab_mirror);
}

bool retopo_export_obj(const RetopoMesh& rm, const char* path) {
    if (rm.quads.empty()) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "# Chisel retopo: %u quads\n", rm.quad_count());
    // Welds leave folded-away vertices behind: write only the ones a quad uses.
    std::vector<uint32_t> id(rm.v.size(), 0);
    for (uint32_t i : rm.quads) id[i] = 1;
    uint32_t next = 0;
    for (size_t i = 0; i < rm.v.size(); i++) {
        if (!id[i]) continue;
        id[i] = ++next;
        std::fprintf(f, "v %.6f %.6f %.6f\n", rm.v[i].x, rm.v[i].y, rm.v[i].z);
    }
    for (size_t q = 0; q < rm.quads.size(); q += 4)
        std::fprintf(f, "f %u %u %u %u\n", id[rm.quads[q]], id[rm.quads[q + 1]],
                     id[rm.quads[q + 2]], id[rm.quads[q + 3]]);
    std::fclose(f);
    return true;
}
