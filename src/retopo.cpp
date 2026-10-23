#include "retopo.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

// ---- Stroke list -------------------------------------------------------------------

bool RetopoState::finish_live(int rings, float density) {
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

    s.rings = rings; s.density = density; s.width = 1.0f;   // the brush Size already set its band
    // The fresh stroke is the selection: the sliders tune it straight away.
    for (RetopoStroke& o : strokes) o.selected = false;
    s.selected = true;
    checkpoint();
    s.id = next_id++;
    char nm[32];
    std::snprintf(nm, sizeof(nm), "%s %u", s.closed ? "Donut" : "Strip", s.id);
    s.name = nm;
    strokes.push_back(std::move(s));
    return true;
}

// ---- History -----------------------------------------------------------------------

size_t RetopoSnapshot::bytes() const {
    size_t b = result.v.size() * (2 * sizeof(Vec3) + 3 * sizeof(uint32_t) + 3)
             + (result.faces.size() + result.face_off.size()) * sizeof(uint32_t) * 2
             + result.edges.size() * sizeof(uint32_t) * 3;
    for (const RetopoStroke& k : strokes) b += k.pts.size() * (2 * sizeof(Vec3) + sizeof(float)) + 128;
    return b;
}

namespace {
// Plenty for a session of retopo; the byte cap keeps a huge one inside the web build's heap.
constexpr size_t kHistorySteps = 64;
constexpr size_t kHistoryBytes = 64u << 20;

RetopoSnapshot snapshot_of(const RetopoState& st) {
    RetopoSnapshot s;
    s.strokes = st.strokes;
    s.result = st.result;
    s.built = st.built;
    s.next_id = st.next_id;
    return s;
}

void push_capped(std::vector<RetopoSnapshot>& stack, RetopoSnapshot&& s) {
    stack.push_back(std::move(s));
    size_t total = 0;
    for (const RetopoSnapshot& x : stack) total += x.bytes();
    size_t drop = 0;
    while (stack.size() - drop > 1 && (stack.size() - drop > kHistorySteps || total > kHistoryBytes))
        total -= stack[drop++].bytes();
    stack.erase(stack.begin(), stack.begin() + drop);
}

// Put a step back, keeping what isn't history: the eyes and the selection stay as the
// user has them now (a stroke that comes back gets its own from back then).
void restore(RetopoState& st, RetopoSnapshot&& s) {
    for (RetopoStroke& k : s.strokes)
        if (const RetopoStroke* now = st.find(k.id)) { k.visible = now->visible; k.selected = now->selected; }
    st.strokes = std::move(s.strokes);
    st.result = std::move(s.result);
    st.built = s.built;
    st.next_id = std::max(st.next_id, s.next_id);   // ids are never reused
}
}  // namespace

void RetopoState::checkpoint(int tag) {
    if (tag != 0 && tag == last_tag) return;
    last_tag = tag;
    loop_edges.clear();
    push_capped(undo_stack, snapshot_of(*this));
    redo_stack.clear();
}

void RetopoState::begin_edit() {
    pending = snapshot_of(*this);
    has_pending = true;
}

void RetopoState::end_edit() {
    if (!has_pending) return;
    has_pending = false;
    const RetopoMesh& a = pending.result;
    const bool same = a.faces == result.faces && a.face_off == result.face_off && a.v.size() == result.v.size()
                   && std::equal(a.v.begin(), a.v.end(), result.v.begin(),
                                 [](const Vec3& l, const Vec3& r) { return l.x == r.x && l.y == r.y && l.z == r.z; });
    if (!same) {
        last_tag = 0;
        push_capped(undo_stack, std::move(pending));
        redo_stack.clear();
    }
    pending = RetopoSnapshot{};
}

bool RetopoState::undo() {
    if (undo_stack.empty()) return false;
    last_tag = 0;
    loop_edges.clear();
    push_capped(redo_stack, snapshot_of(*this));
    restore(*this, std::move(undo_stack.back()));
    undo_stack.pop_back();
    return true;
}

bool RetopoState::redo() {
    if (redo_stack.empty()) return false;
    last_tag = 0;
    loop_edges.clear();
    push_capped(undo_stack, snapshot_of(*this));
    restore(*this, std::move(redo_stack.back()));
    redo_stack.pop_back();
    return true;
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

int RetopoState::merge_selected() {
    if (merge_candidates() < 2) return 0;
    checkpoint();
    uint32_t keep = ~0u;
    for (const RetopoStroke& s : strokes) if (s.selected && s.frozen) keep = std::min(keep, s.id);
    std::vector<uint8_t> into(next_id, 0);
    int n = 0;
    for (const RetopoStroke& s : strokes)
        if (s.selected && s.frozen && s.id != keep && s.id < next_id) { into[s.id] = 1; n++; }
    auto remap = [&](std::vector<uint32_t>& owners) {
        for (uint32_t& o : owners) if (o < into.size() && into[o]) o = keep;
    };
    remap(result.face_owner);
    remap(result.v_owner);
    remap(result.edge_owner);
    strokes.erase(std::remove_if(strokes.begin(), strokes.end(), [&](const RetopoStroke& s) {
        return s.id < into.size() && into[s.id];
    }), strokes.end());
    if (RetopoStroke* k = find(keep)) {
        k->selected = true;
        // A hidden island merged into a shown one shows; the merged piece is one thing.
        k->visible = true;
    }
    std::printf("[retopo] merge: %d island(s) into id %u\n", n, keep);
    return n;
}

void RetopoState::clear_all() {
    if (strokes.empty() && result.faces.empty()) return;
    checkpoint();   // one Ctrl+Z brings it all back
    strokes.clear();
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

    // The first triangle facing back at the ray (o + d t, d unit, 0 <= t <= tmax),
    // walking the cells it passes through in order (Amanatides & Woo). Triangles seen
    // from behind are passed through: from inside a lip, the ray goes on to the surface
    // it was aimed at. False when it leaves the grid or reaches tmax without one.
    bool ray(Vec3 o, Vec3 d, float tmax, Vec3& hit, Vec3& hit_n) const {
        const float O[3] = {o.x, o.y, o.z}, D[3] = {d.x, d.y, d.z}, L0[3] = {lo.x, lo.y, lo.z};
        const int nc[3] = {nx, ny, nz};
        float t0 = 0.0f, t1 = tmax;
        for (int a = 0; a < 3; a++) {
            const float hi = L0[a] + nc[a] * cell;
            if (std::fabs(D[a]) < 1e-12f) {
                if (O[a] < L0[a] || O[a] > hi) return false;
                continue;
            }
            float ta = (L0[a] - O[a]) / D[a], tb = (hi - O[a]) / D[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
        if (t0 > t1) return false;
        int c[3], step[3];
        float tnext[3], tdelta[3];
        for (int a = 0; a < 3; a++) {
            c[a] = clampi((int)std::floor((O[a] + D[a] * t0 - L0[a]) / cell), nc[a]);
            step[a] = D[a] > 0.0f ? 1 : -1;
            if (std::fabs(D[a]) < 1e-12f) { tnext[a] = tdelta[a] = 1e30f; continue; }
            tnext[a] = (L0[a] + (c[a] + (D[a] > 0.0f ? 1 : 0)) * cell - O[a]) / D[a];
            tdelta[a] = cell / std::fabs(D[a]);
        }
        for (;;) {
            const float exit = std::min({tnext[0], tnext[1], tnext[2], t1});
            float best = 1e30f;
            uint32_t best_t = 0xFFFFFFFFu;
            const uint64_t k = (uint64_t)cell_of(c[0], c[1], c[2]) << 32;
            for (auto it = std::lower_bound(keys.begin(), keys.end(), k);
                 it != keys.end() && (*it >> 32) == (k >> 32); ++it) {
                const uint32_t t = (uint32_t)(*it & 0xFFFFFFFFu);
                const Vec3 a = P(m->indices[t * 3]), e1 = P(m->indices[t * 3 + 1]) - a,
                           e2 = P(m->indices[t * 3 + 2]) - a;
                if (e1.cross(e2).dot(d) >= 0.0f) continue;   // seen from behind
                const Vec3 pv = d.cross(e2);
                const float det = e1.dot(pv);
                if (std::fabs(det) < 1e-20f) continue;
                const float inv = 1.0f / det;
                const Vec3 tv = o - a;
                const float u = tv.dot(pv) * inv;
                if (u < 0.0f || u > 1.0f) continue;
                const Vec3 qv = tv.cross(e1);
                const float v = d.dot(qv) * inv;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float th = e2.dot(qv) * inv;
                if (th >= t0 && th <= t1 && th < best) { best = th; best_t = t; }
            }
            // A triangle spans cells: a hit past this cell may hide a nearer one in the next.
            if (best_t != 0xFFFFFFFFu && best <= exit) {
                hit = o + d * best;
                const Vec3 a = P(m->indices[best_t * 3]);
                hit_n = (P(m->indices[best_t * 3 + 1]) - a).cross(P(m->indices[best_t * 3 + 2]) - a).normalized();
                return true;
            }
            const int a = tnext[0] < tnext[1] ? (tnext[0] < tnext[2] ? 0 : 2) : (tnext[1] < tnext[2] ? 1 : 2);
            if (tnext[a] > t1) return false;
            c[a] += step[a];
            if (c[a] < 0 || c[a] >= nc[a]) return false;
            tnext[a] += tdelta[a];
        }
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

// A face's corners into c, any corner repeated by a weld dropped. Returns how many.
int face_corners(const RetopoMesh& rm, size_t f, std::vector<uint32_t>& c) {
    const uint32_t* q = rm.face(f);
    const uint32_t k = rm.face_size(f);
    c.clear();
    for (uint32_t i = 0; i < k; i++)
        if (c.empty() || q[i] != c.back()) c.push_back(q[i]);
    while (c.size() > 1 && c.back() == c.front()) c.pop_back();
    return (int)c.size();
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
            rm.add_face(q, 4, owner);
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
    qe.reserve(rm.faces.size());
    std::vector<uint32_t> c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n && n >= 3; k++) {
            const uint32_t a = c[k], b = c[(k + 1) % n];
            qe.push_back(((uint64_t)std::min(a, b) << 32) | std::max(a, b));
        }
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

    // Outlines, walked round edge by edge. A vertex where two outlines touch (a weld at
    // one point) has four outer edges: the walk takes any unused one, so the two may come
    // out as one figure-eight rim, still closed. A walk that dead-ends isn't a rim.
    rm.rim_off.assign(1, 0); rm.rim.clear(); rm.rim_hole.clear();
    std::vector<uint32_t> bo(nv + 1, 0), bn, be;   // outer edges per vertex (CSR): other end, edge
    std::vector<std::pair<uint32_t, uint32_t>> outer;
    for (size_t i = 0; i < qe.size(); ) {
        size_t j = i;
        while (j < qe.size() && qe[j] == qe[i]) j++;
        if (j - i == 1) outer.push_back({(uint32_t)(qe[i] >> 32), (uint32_t)(qe[i] & 0xFFFFFFFFu)});
        i = j;
    }
    if (outer.empty()) return;
    for (const auto& e : outer) { bo[e.first + 1]++; bo[e.second + 1]++; }
    for (uint32_t i = 0; i < nv; i++) bo[i + 1] += bo[i];
    bn.resize(bo[nv]); be.resize(bo[nv]);
    {
        std::vector<uint32_t> fill(bo.begin(), bo.end() - 1);
        for (uint32_t e = 0; e < (uint32_t)outer.size(); e++) {
            const uint32_t a = outer[e].first, b = outer[e].second;
            bn[fill[a]] = b; be[fill[a]++] = e;
            bn[fill[b]] = a; be[fill[b]++] = e;
        }
    }
    std::vector<uint8_t> used(outer.size(), 0);
    std::vector<uint32_t> walk;
    std::vector<float> rim_len;
    for (uint32_t e0 = 0; e0 < (uint32_t)outer.size(); e0++) {
        if (used[e0]) continue;
        used[e0] = 1;
        const uint32_t start = outer[e0].first;
        uint32_t cur = outer[e0].second;
        walk.assign(1, start);
        float len = (rm.v[cur] - rm.v[start]).length();
        bool closed = false;
        while (walk.size() <= outer.size()) {
            if (cur == start) { closed = true; break; }
            walk.push_back(cur);
            uint32_t nx = ~0u;
            for (uint32_t k = bo[cur]; k < bo[cur + 1]; k++)
                if (!used[be[k]]) { used[be[k]] = 1; nx = bn[k]; break; }
            if (nx == ~0u) break;
            len += (rm.v[nx] - rm.v[cur]).length();
            cur = nx;
        }
        if (!closed || walk.size() < 3) continue;
        rm.rim.insert(rm.rim.end(), walk.begin(), walk.end());
        rm.rim_off.push_back((uint32_t)rm.rim.size());
        rim_len.push_back(len);
    }
    // Pieces: vertices joined by faces (union-find). Each piece's longest rim is its
    // outer edge; every other rim of it is a hole.
    std::vector<uint32_t> up(nv);
    for (uint32_t i = 0; i < nv; i++) up[i] = i;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    for (size_t f = 0; f < rm.face_count(); f++) {
        const uint32_t* q = rm.face(f);
        for (uint32_t k = 1; k < rm.face_size(f); k++) up[root(q[k])] = root(q[0]);
    }
    const uint32_t nr = (uint32_t)rim_len.size();
    rm.rim_hole.assign(nr, 1);
    std::vector<uint32_t> longest(nv, ~0u);   // per piece root: its longest rim
    for (uint32_t r = 0; r < nr; r++) {
        uint32_t& l = longest[root(rm.rim[rm.rim_off[r]])];
        if (l == ~0u || rim_len[r] > rim_len[l]) l = r;
    }
    for (uint32_t r = 0; r < nr; r++)
        if (longest[root(rm.rim[rm.rim_off[r]])] == r) rm.rim_hole[r] = 0;
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

// Distance from p to the nearest face not owned by `owner`; its owner in `who`.
float dist_to_others(const RetopoMesh& rm, uint32_t owner, Vec3 p, uint32_t& who) {
    float best = 1e30f;
    for (size_t f = 0; f < rm.face_count(); f++) {
        if (rm.face_owner[f] == owner) continue;
        const uint32_t* q = rm.face(f);
        const uint32_t k = rm.face_size(f);
        for (uint32_t t = 1; t + 1 < k; t++) {   // a fan from the first corner
            const Vec3 x = SnapGrid::closest_on_tri(p, rm.v[q[0]], rm.v[q[t]], rm.v[q[t + 1]]) - p;
            const float d2 = x.dot(x);
            if (d2 < best) { best = d2; who = rm.face_owner[f]; }
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

// Copy into dst the faces whose owner passes keep(), the vertices they use, and the
// kept owners' edges between those vertices.
template <typename Keep>
void extract(const RetopoMesh& src, RetopoMesh& dst, Keep keep) {
    dst.clear();
    std::vector<uint32_t> remap(src.v.size(), ~0u);
    for (size_t f = 0; f < src.face_count(); f++) {
        if (!keep(src.face_owner[f])) continue;
        for (uint32_t k = src.face_off[f]; k < src.face_off[f + 1]; k++) {
            const uint32_t i = src.faces[k];
            if (remap[i] == ~0u) {
                remap[i] = (uint32_t)dst.v.size();
                dst.v.push_back(src.v[i]);
                dst.n.push_back(src.n[i]);
                dst.v_owner.push_back(src.v_owner[i]);
                dst.mirror.push_back(src.mirror[i]);   // old index, remapped below
            }
            dst.faces.push_back(remap[i]);
        }
        dst.face_off.push_back((uint32_t)dst.faces.size());
        dst.face_owner.push_back(src.face_owner[f]);
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
    st.loop_edges.clear();   // the vertices get new numbers

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
            if (s.frozen || s.pts.size() < 3 || s.closed != (pass == 0)) continue;
            bool pos = false, neg = false;
            for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
            Todo t{s, mirror && pos != neg};
            if (mirror && pos && neg && half_stroke(s, t.s, t.seam0, t.seam1)) t.twin = true;
            for (float& w : t.s.half_w) w *= s.width;
            todo.push_back(std::move(t));
        }
    if (todo.empty() || m.tri_count() == 0) {
        st.built = !st.result.faces.empty();
        if (st.built) build_topology(st.result);
        std::printf("[retopo] build: nothing to rebuild, %u frozen faces kept\n", st.result.face_count());
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
        if (!G.closed && !rm.faces.empty()) {
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
    st.built = !rm.faces.empty();
    if (st.built) build_topology(rm);

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[retopo] build: %zu strokes -> %u faces, %zu verts, %d strip ends joined, %.1f ms\n",
                todo.size(), st.result.face_count(), st.result.v.size(), joins, ms);
    return ms;
}

// ---- Picking -----------------------------------------------------------------------

uint32_t retopo_pick(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                     float x, float y, bool mirror) {
    auto facing = [&](Vec3 p, Vec3 n) {
        const Vec3 to_eye = cam.perspective ? cam.get_position() - p : cam.get_view_direction() * -1.0f;
        return n.dot(to_eye) > 0.0f;
    };
    // Built faces first: the cursor inside a front-facing face, the nearest one wins.
    const RetopoMesh& rm = st.result;
    uint32_t best = 0;
    float best_d = 1e30f;
    std::vector<float> sx, sy;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const uint32_t* q = rm.face(f);
        const uint32_t k = rm.face_size(f);
        if (k < 3 || st.hidden(rm.face_owner[f]) || !facing(rm.v[q[0]], rm.n[q[0]])) continue;
        sx.resize(k); sy.resize(k);
        bool ok = true;
        for (uint32_t i = 0; i < k && ok; i++) ok = cam.world_to_screen(rm.v[q[i]], win_w, win_h, sx[i], sy[i]);
        if (!ok) continue;
        // Inside = an odd number of sides crossed by a ray to the right (any shape, any
        // winding: a merged face needn't be convex).
        bool in = false;
        for (uint32_t i = 0, j = k - 1; i < k; j = i++)
            if ((sy[i] > y) != (sy[j] > y)
                && x < sx[j] + (sx[i] - sx[j]) * (y - sy[j]) / (sy[i] - sy[j])) in = !in;
        if (!in) continue;
        const float d = cam.view_depth(rm.v[q[0]]);
        if (d < best_d) { best_d = d; best = rm.face_owner[f]; }
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
    loop_edges.clear();
    built = !result.faces.empty();
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
    for (uint32_t& q : rm.faces) if (q == a) q = b;
    for (uint32_t& e : rm.edges) if (e == a) e = b;
    dedupe_edges(rm);
    build_topology(rm);
}

void freeze_touched(RetopoState& st, std::vector<uint32_t>& touched);

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
            if (b == a || !alive(rm, b) || !on_border(rm, b) || st.hidden(rm.v_owner[b])) continue;
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
    freeze_touched(st, touched);
    if (welds) std::printf("[retopo] welded %d vertex pair(s)\n", welds);
}

// A joined group is one piece: freezing one island freezes every island that shares
// a vertex with it, or the next Build would rebuild a neighbour out from under the
// frozen one's shared vertices.
void freeze_touched(RetopoState& st, std::vector<uint32_t>& touched) {
    RetopoMesh& rm = st.result;
    uint32_t max_id = 0;
    for (uint32_t o : rm.face_owner) max_id = std::max(max_id, o);
    std::vector<uint32_t> up(max_id + 1);
    for (uint32_t i = 0; i <= max_id; i++) up[i] = i;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    std::vector<uint32_t> first(rm.v.size(), ~0u);
    for (size_t q = 0; q < rm.face_count(); q++)
        for (uint32_t k = rm.face_off[q]; k < rm.face_off[q + 1]; k++) {
            uint32_t& f = first[rm.faces[k]];
            if (f == ~0u) f = rm.face_owner[q];
            else          up[root(f)] = root(rm.face_owner[q]);
        }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (RetopoStroke& s : st.strokes) {
        if (s.frozen || s.id > max_id) continue;
        for (uint32_t id : touched)
            if (id <= max_id && root(id) == root(s.id)) { s.frozen = true; break; }
    }
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
        g_smooth_w[i] = rm.pinned[i] || st.hidden(rm.v_owner[i]) ? 0.0f : w * strength;
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
        if (w <= 0.0f || st.hidden(rm.v_owner[i])) continue;
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

// ---- Cut ------------------------------------------------------------------------

namespace {

// How much of a crossing's offset from its edge's middle survives the aim. A hand-drawn
// line wanders; the loop it cuts is wanted straight, so 1/5 along an edge lands at 0.43.
constexpr float kCutAimKeep = 0.25f;
// A vertex cut must start or end this close to the vertex (screen pixels): deliberate.
constexpr float kSnapPx = 8.0f;

// 1 - |cos| of the corner at b: 1 square, 0 folded flat or doubled back.
float corner_q(Vec3 a, Vec3 b, Vec3 c) {
    return 1.0f - std::fabs((a - b).normalized().dot((c - b).normalized()));
}
float poly_q(const RetopoMesh& rm, const uint32_t* p, int n) {
    float q = 1.0f;
    for (int k = 0; k < n; k++)
        q = std::min(q, corner_q(rm.v[p[(k + n - 1) % n]], rm.v[p[k]], rm.v[p[(k + 1) % n]]));
    return q;
}

// A face list under construction (a RetopoMesh's faces, face_off and face_owner).
struct FaceList {
    std::vector<uint32_t> faces, off{0}, owner;
    void add(const uint32_t* p, int k, uint32_t who) {
        faces.insert(faces.end(), p, p + k);
        off.push_back((uint32_t)faces.size());
        owner.push_back(who);
    }
    void swap_into(RetopoMesh& rm) {
        rm.faces.swap(faces); rm.face_off.swap(off); rm.face_owner.swap(owner);
    }
};

// One piece of a cut face (in the face's own winding) into the face list. Cutting a
// quad or a triangle, a pentagon piece becomes a quad and a triangle, split where both
// come out squarest (that also keeps a triangle off three points of one straight
// edge). A piece of a bigger face (merged by the eraser) goes in as it is.
void emit_piece(const RetopoMesh& rm, FaceList& out, const uint32_t* p, int n, uint32_t who,
                bool quadify) {
    if (n != 5 || !quadify) { out.add(p, n, who); return; }
    int best = 0;
    float best_q = -1.0f;
    for (int i = 0; i < 5; i++) {
        const uint32_t t[3] = {p[i], p[(i + 1) % 5], p[(i + 2) % 5]};
        const uint32_t q[4] = {p[(i + 2) % 5], p[(i + 3) % 5], p[(i + 4) % 5], p[i]};
        const float s = std::min(poly_q(rm, t, 3), poly_q(rm, q, 4));
        if (s > best_q) { best_q = s; best = i; }
    }
    const int i = best;
    const uint32_t t[3] = {p[i], p[(i + 1) % 5], p[(i + 2) % 5]};
    const uint32_t q[4] = {p[(i + 2) % 5], p[(i + 3) % 5], p[(i + 4) % 5], p[i]};
    out.add(t, 3, who);
    out.add(q, 4, who);
}

uint64_t edge_key(uint32_t a, uint32_t b) { return ((uint64_t)std::min(a, b) << 32) | std::max(a, b); }

// Faces again from scratch after a face edit: edges from their sides, then topology.
void rebuild_edges(RetopoMesh& rm) {
    rm.edges.clear(); rm.edge_owner.clear();
    std::vector<uint32_t> c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n && n >= 2; k++) add_edge(rm, c[k], c[(k + 1) % n], rm.face_owner[f]);
    }
    dedupe_edges(rm);
    build_topology(rm);
}

}  // namespace

uint32_t retopo_snap_vertex(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                            float x, float y) {
    const RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return ~0u;
    uint32_t best = ~0u;
    float best_d2 = kSnapPx * kSnapPx;
    for (uint32_t i = 0; i < (uint32_t)rm.v.size(); i++) {
        if (!alive(rm, i) || st.hidden(rm.v_owner[i])) continue;
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
        float sx, sy;
        if (rm.n[i].dot(to_eye) <= 0.0f || !cam.world_to_screen(rm.v[i], win_w, win_h, sx, sy)) continue;
        const float d2 = (sx - x) * (sx - x) + (sy - y) * (sy - y);
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}

int retopo_select_loop(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool add, bool mirror) {
    if (!add) st.loop_edges.clear();
    const RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    if (!st.built || rm.nb_off.size() != nv + 1) return 0;

    // The edge under the pointer: both ends facing the eye and on screen, nearest in pixels.
    uint32_t ea = ~0u, eb = ~0u;
    float best = kSnapPx;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (a == b || st.hidden(rm.edge_owner[e / 2])) continue;
        float ax, ay, bx, by;
        bool ok = true;
        for (uint32_t i : {a, b}) {
            const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
            ok = ok && rm.n[i].dot(to_eye) > 0.0f;
        }
        if (!ok || !cam.world_to_screen(rm.v[a], win_w, win_h, ax, ay)
                || !cam.world_to_screen(rm.v[b], win_w, win_h, bx, by)) continue;
        const float ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
        const float t = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, ((x - ax) * ex + (y - ay) * ey) / l2)) : 0.0f;
        const float d = std::hypot(x - (ax + ex * t), y - (ay + ey * t));
        if (d < best) { best = d; ea = a; eb = b; }
    }
    if (ea == ~0u) return 0;

    // Faces round each vertex (CSR), once per pick: retopo is a few thousand faces.
    std::vector<uint32_t> vf_off(nv + 1, 0), vf, c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) vf_off[c[k] + 1]++;
    }
    for (uint32_t i = 0; i < nv; i++) vf_off[i + 1] += vf_off[i];
    vf.resize(vf_off[nv]);
    {
        std::vector<uint32_t> fill(vf_off.begin(), vf_off.end() - 1);
        for (size_t f = 0; f < rm.face_count(); f++) {
            const int n = face_corners(rm, f, c);
            for (int k = 0; k < n; k++) vf[fill[c[k]]++] = (uint32_t)f;
        }
    }
    // v's faces' sides out of v: for each face, the corners either side of v. `quads`
    // turns false if any of them isn't a quad.
    auto around = [&](uint32_t v, std::vector<std::pair<uint32_t, uint32_t>>& sides, bool& quads) {
        sides.clear();
        quads = true;
        for (uint32_t k = vf_off[v]; k < vf_off[v + 1]; k++) {
            const int n = face_corners(rm, vf[k], c);
            quads = quads && n == 4;
            for (int i = 0; i < n; i++)
                if (c[i] == v) sides.push_back({c[(i + n - 1) % n], c[(i + 1) % n]});
        }
    };
    auto faces_on = [&](uint32_t w, const std::vector<std::pair<uint32_t, uint32_t>>& sides) {
        int k = 0;
        for (const auto& s : sides) k += (s.first == w) + (s.second == w);
        return k;
    };
    // The loop's next vertex past b, coming from a; ~0u = it stops at b.
    std::vector<std::pair<uint32_t, uint32_t>> sides;
    std::vector<uint32_t> nbs;
    auto next = [&](uint32_t a, uint32_t b) -> uint32_t {
        bool quads;
        around(b, sides, quads);
        nbs.clear();
        for (const auto& s : sides)
            for (uint32_t w : {s.first, s.second})
                if (std::find(nbs.begin(), nbs.end(), w) == nbs.end()) nbs.push_back(w);
        if (faces_on(a, sides) == 1) {
            // On the outline: on round it, through corners too, while it's a simple one.
            uint32_t on = ~0u;
            int outline = 0;
            for (uint32_t w : nbs)
                if (faces_on(w, sides) == 1) { outline++; if (w != a) on = w; }
            return outline == 2 ? on : ~0u;
        }
        // Inside: straight across a vertex with four quads round it, to the one neighbour
        // that shares no face with a-b.
        if (!quads || sides.size() != 4 || nbs.size() != 4) return ~0u;
        for (uint32_t w : nbs) if (faces_on(w, sides) != 2) return ~0u;
        uint32_t across = ~0u;
        for (uint32_t w : nbs) {
            if (w == a) continue;
            bool beside = false;
            for (const auto& s : sides)
                if ((s.first == a && s.second == w) || (s.first == w && s.second == a)) beside = true;
            if (!beside) { if (across != ~0u) return ~0u; across = w; }
        }
        return across;
    };

    std::vector<uint64_t> seen{edge_key(ea, eb)};
    std::vector<uint32_t> loop{ea, eb};
    bool closed = false;
    for (int dir = 0; dir < 2 && !closed; dir++) {
        uint32_t p = dir ? eb : ea, q = dir ? ea : eb;
        for (size_t step = 0; step < rm.edges.size(); step++) {
            const uint32_t r = next(p, q);
            if (r == ~0u || r == p) break;
            const uint64_t k = edge_key(q, r);
            if (std::find(seen.begin(), seen.end(), k) != seen.end()) { closed = true; break; }
            seen.push_back(k);
            loop.push_back(q); loop.push_back(r);
            p = q; q = r;
        }
    }
    if (mirror && rm.mirror.size() == nv) {
        const size_t n = loop.size();
        for (size_t i = 0; i + 1 < n; i += 2) {
            const uint32_t a = rm.mirror[loop[i]], b = rm.mirror[loop[i + 1]];
            if (a >= nv || b >= nv || a == b) continue;
            const uint64_t k = edge_key(a, b);
            if (std::find(seen.begin(), seen.end(), k) != seen.end()) continue;   // its own twin
            seen.push_back(k);
            loop.push_back(a); loop.push_back(b);
        }
    }

    // Ctrl on a loop already picked takes it out; otherwise it joins what's there.
    bool had = false;
    for (size_t i = 0; i + 1 < st.loop_edges.size(); i += 2)
        had |= edge_key(st.loop_edges[i], st.loop_edges[i + 1]) == edge_key(ea, eb);
    std::vector<uint32_t> keep;
    for (size_t i = 0; i + 1 < st.loop_edges.size(); i += 2) {
        const uint64_t k = edge_key(st.loop_edges[i], st.loop_edges[i + 1]);
        if (std::find(seen.begin(), seen.end(), k) != seen.end()) continue;
        keep.push_back(st.loop_edges[i]); keep.push_back(st.loop_edges[i + 1]);
    }
    if (!had) keep.insert(keep.end(), loop.begin(), loop.end());
    st.loop_edges.swap(keep);
    const bool outline = vf_off[ea + 1] - vf_off[ea] > 0 && [&] {
        bool q; around(ea, sides, q); return faces_on(eb, sides) == 1;
    }();
    std::printf("[retopo] loop: %s %zu edge(s), %s, %s; %zu picked in all\n", had ? "dropped" : "picked",
                loop.size() / 2, outline ? "outline" : "inner", closed ? "closed" : "open",
                st.loop_edges.size() / 2);
    return (int)(loop.size() / 2);
}

int retopo_cut(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
               bool mirror) {
    RetopoMesh& rm = st.result;
    const size_t np = st.cut_pts.size();
    if (!st.built || np < 2) return 0;
    retopo_edit_begin(st, m);   // the snap grid, for the new vertices
    const uint32_t from_v = st.cut_from_v < rm.v.size() ? st.cut_from_v : ~0u;
    const uint32_t to_v   = st.cut_to_v   < rm.v.size() && st.cut_to_v != from_v ? st.cut_to_v : ~0u;
    // A snapped end starts (or stops) exactly on its vertex, not a few pixels off it.
    std::vector<Vec3> pts = st.cut_pts;
    if (from_v != ~0u) pts.front() = rm.v[from_v];
    if (to_v != ~0u)   pts.back()  = rm.v[to_v];

    // The line on screen. Its points lie on the sculpt where the pen was, so they're
    // visible by construction.
    std::vector<float> lx(np), ly(np);
    std::vector<uint8_t> lok(np);
    float bx0 = 1e30f, by0 = 1e30f, bx1 = -1e30f, by1 = -1e30f;
    for (size_t i = 0; i < np; i++) {
        lok[i] = cam.world_to_screen(pts[i], win_w, win_h, lx[i], ly[i]);
        if (!lok[i]) continue;
        bx0 = std::min(bx0, lx[i]); bx1 = std::max(bx1, lx[i]);
        by0 = std::min(by0, ly[i]); by1 = std::max(by1, ly[i]);
    }
    const size_t nv = rm.v.size();
    std::vector<float> sx(nv), sy(nv);
    std::vector<uint8_t> vok(nv);
    for (size_t i = 0; i < nv; i++) {
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
        vok[i] = cam.world_to_screen(rm.v[i], win_w, win_h, sx[i], sy[i]) && rm.n[i].dot(to_eye) > 0.0f
              && !st.hidden(rm.v_owner[i]);
    }

    // Every edge the line crosses, once (a line that doubles back keeps its first pass).
    // s = how far along the line, for ordering. Seen from the eye, an edge crossing must
    // also sit on the line's own bit of surface: a front-facing island behind the sculpt
    // (the far side of a limb) passes the screen test but not this one.
    struct Cross { uint32_t a, b; float t, s; uint32_t twin = ~0u, vid = ~0u; bool slave = false; };
    std::vector<Cross> cx;
    std::unordered_map<uint64_t, uint32_t> at;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (a == b || !vok[a] || !vok[b]) continue;
        if (std::max(sx[a], sx[b]) < bx0 || std::min(sx[a], sx[b]) > bx1
            || std::max(sy[a], sy[b]) < by0 || std::min(sy[a], sy[b]) > by1) continue;
        const float rx = sx[b] - sx[a], ry = sy[b] - sy[a];
        const float len = (rm.v[b] - rm.v[a]).length();
        for (size_t i = 0; i + 1 < np; i++) {
            if (!lok[i] || !lok[i + 1]) continue;
            const float qx = lx[i + 1] - lx[i], qy = ly[i + 1] - ly[i];
            const float den = rx * qy - ry * qx;
            if (std::fabs(den) < 1e-9f) continue;
            const float wx = lx[i] - sx[a], wy = ly[i] - sy[a];
            const float t = (wx * qy - wy * qx) / den, u = (wx * ry - wy * rx) / den;
            if (t < 0.0f || t > 1.0f || u < 0.0f || u > 1.0f) continue;
            // Leaving (or reaching) a snapped vertex grazes its own edges right at it:
            // those aren't crossings, the vertex is the cut point.
            bool at_snap = false;
            for (uint32_t sv : {from_v, to_v})
                if (sv != ~0u && (a == sv || b == sv))
                    at_snap |= std::hypot(sx[a] + rx * t - sx[sv], sy[a] + ry * t - sy[sv]) < 2.0f * kSnapPx;
            if (at_snap) continue;
            const Vec3 on_edge = rm.v[a] + (rm.v[b] - rm.v[a]) * t;
            const Vec3 on_line = pts[i] + (pts[i + 1] - pts[i]) * u;
            if ((on_edge - on_line).length() > 0.5f * len) continue;
            at[edge_key(a, b)] = (uint32_t)cx.size();
            cx.push_back({a, b, 0.5f + (t - 0.5f) * kCutAimKeep, (float)i + u});
            break;
        }
    }
    auto corners = [&](size_t f, std::vector<uint32_t>& c) { return face_corners(rm, f, c); };

    // Vertex cuts: (face, corner) pairs where the cut starts at a corner instead of on
    // a side. The start vertex pairs with the face it leaves through: the one that has
    // both it and the first crossed edge. Likewise the end with the last.
    std::vector<std::pair<uint32_t, uint32_t>> chords;
    auto is_chord = [&](uint32_t f, uint32_t v) {
        return std::find(chords.begin(), chords.end(), std::make_pair(f, v)) != chords.end();
    };
    auto chord_face = [&](uint32_t f) {
        for (auto& ch : chords) if (ch.first == f) return true;
        return false;
    };
    {
        std::vector<uint32_t> c;
        // The face with corner v and (a != ~0u) side a-b, or (a == ~0u) corner b not next
        // to v. ~0u for none.
        auto face_of = [&](uint32_t v, uint32_t a, uint32_t b) {
            for (size_t f = 0; f < rm.face_count(); f++) {
                const int n = corners(f, c);
                int iv = -1;
                for (int k = 0; k < n; k++) if (c[k] == v) iv = k;
                if (iv < 0) continue;
                for (int k = 0; k < n; k++) {
                    const uint32_t x = c[k], y = c[(k + 1) % n];
                    if (a != ~0u && ((x == a && y == b) || (x == b && y == a))) return (uint32_t)f;
                }
                if (a == ~0u)
                    for (int k = 0; k < n; k++)
                        if (c[k] == b && k != (iv + 1) % n && (k + 1) % n != iv) return (uint32_t)f;
            }
            return ~0u;
        };
        if (!cx.empty()) {
            size_t first = 0, last = 0;
            for (size_t i = 1; i < cx.size(); i++) {
                if (cx[i].s < cx[first].s) first = i;
                if (cx[i].s > cx[last].s) last = i;
            }
            if (from_v != ~0u) {
                const uint32_t f = face_of(from_v, cx[first].a, cx[first].b);
                if (f != ~0u) chords.push_back({f, from_v});
            }
            if (to_v != ~0u) {
                const uint32_t f = face_of(to_v, cx[last].a, cx[last].b);
                if (f != ~0u && !is_chord(f, to_v)) chords.push_back({f, to_v});
            }
        } else if (from_v != ~0u && to_v != ~0u) {
            // No sides crossed: corner to corner across one face.
            const uint32_t f = face_of(from_v, ~0u, to_v);
            if (f != ~0u) { chords.push_back({f, from_v}); chords.push_back({f, to_v}); }
        }
    }
    if (cx.empty() && chords.empty()) {
        std::printf("[retopo] cut: the line crossed no edges\n");
        return 0;
    }

    // Mirror: each crossing's twin edge is cut at the same place. Where the line itself
    // crossed both sides (drawn over the seam), the earlier crossing leads and the other
    // follows it, so the halves stay exact reflections. `insert` = cut a twin edge the
    // line missed; without it, only pair up twins that are both cut already.
    auto pair_twins = [&](bool insert) {
        std::vector<uint32_t> order(cx.size());
        for (uint32_t i = 0; i < (uint32_t)order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t l, uint32_t r) { return cx[l].s < cx[r].s; });
        for (uint32_t i : order) {
            if (cx[i].slave || cx[i].twin != ~0u) continue;
            const uint32_t ma = rm.mirror[cx[i].a], mb = rm.mirror[cx[i].b];
            if (ma == ~0u || mb == ~0u) continue;
            const uint64_t k = edge_key(ma, mb);
            if (k == edge_key(cx[i].a, cx[i].b)) {
                // Its own twin: an edge across the seam (cut at its middle, which is on
                // the seam) or one lying along it.
                if (ma == cx[i].b) cx[i].t = 0.5f;
                cx[i].twin = i;
                continue;
            }
            auto it = at.find(k);
            uint32_t j;
            if (it == at.end()) {
                if (!insert) continue;
                j = (uint32_t)cx.size();
                at[k] = j;
                cx.push_back({ma, mb, cx[i].t, cx[i].s});
            } else {
                j = it->second;
                cx[j].a = ma; cx[j].b = mb; cx[j].t = cx[i].t;
            }
            cx[j].slave = true;
            cx[i].twin = j; cx[j].twin = i;
        }
    };
    if (mirror) {
        pair_twins(true);
        // A vertex cut's twin: the twin vertex, in the face made of the twin corners.
        std::vector<uint32_t> c, mk, fk;
        const size_t nch = chords.size();
        for (size_t i = 0; i < nch; i++) {
            const uint32_t mv = rm.mirror[chords[i].second];
            if (mv == ~0u) continue;
            corners(chords[i].first, c);
            mk.clear();
            for (uint32_t x : c) mk.push_back(rm.mirror[x]);
            if (std::find(mk.begin(), mk.end(), ~0u) != mk.end()) continue;
            std::sort(mk.begin(), mk.end());
            for (size_t f = 0; f < rm.face_count(); f++) {
                corners(f, fk);
                std::sort(fk.begin(), fk.end());
                if (fk != mk) continue;
                if (!is_chord((uint32_t)f, mv)) chords.push_back({(uint32_t)f, mv});
                break;
            }
        }
    }

    // Quads, not triangles. A face splits into quads only when its outline (corners +
    // cut points) has an even count; a line that stops inside a quad leaves five. So
    // such a face gets one more cut, in the middle of the side facing its lone cut (or
    // of its one uncut side): the support runs on through it, and on through the face
    // beyond, until it reaches the island's edge or meets the cut again. A wrong count
    // only ever moves to the next face, so it ends at a border or meets another.
    {
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> faces_of;
        std::vector<uint32_t> c;
        for (size_t f = 0; f < rm.face_count(); f++) {
            const int n = corners(f, c);
            for (int k = 0; k < n && n >= 3; k++) {
                auto ins = faces_of.emplace(edge_key(c[k], c[(k + 1) % n]),
                                            std::make_pair((uint32_t)f, ~0u));
                if (!ins.second) ins.first->second.second = (uint32_t)f;
            }
        }
        std::vector<uint32_t> work;
        for (const Cross& c : cx) {
            auto it = faces_of.find(edge_key(c.a, c.b));
            if (it == faces_of.end()) continue;
            work.push_back(it->second.first);
            if (it->second.second != ~0u) work.push_back(it->second.second);
        }
        float s_more = 1e9f;   // ordered after every drawn crossing: those lead in the mirror pairing
        while (!work.empty()) {
            const uint32_t f = work.back();
            work.pop_back();
            const int n = corners(f, c);
            // Only quads and triangles are kept to quads; a bigger face (the eraser's)
            // takes a lone cut point as one more corner. A vertex cut's face splits as
            // asked, triangle or not.
            if (n < 3 || n > 4 || chord_face(f)) continue;
            int cuts = 0, lone = -1, uncut = -1;
            for (int k = 0; k < n; k++) {
                if (at.count(edge_key(c[k], c[(k + 1) % n]))) { cuts++; lone = k; }
                else uncut = k;
            }
            if (cuts == 0 || (n + cuts) % 2 == 0) continue;
            const int k = n == 4 && cuts == 1 ? (lone + 2) % 4 : uncut;
            const uint64_t key = edge_key(c[k], c[(k + 1) % n]);
            at[key] = (uint32_t)cx.size();
            cx.push_back({c[k], c[(k + 1) % n], 0.5f, s_more++});
            const auto& fs = faces_of[key];
            work.push_back(fs.first == f ? fs.second : fs.first);
            if (work.back() == ~0u) work.pop_back();   // a border: done
        }
    }
    if (mirror) pair_twins(false);

    st.checkpoint();   // before anything changes

    auto add_vertex = [&](Vec3 p, Vec3 n, uint32_t owner) {
        rm.v.push_back(p); rm.n.push_back(n);
        rm.v_owner.push_back(owner); rm.mirror.push_back(~0u);
        return (uint32_t)rm.v.size() - 1;
    };
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };
    auto snapped = [&](Vec3 p, Vec3 n_fallback, Vec3& n) {
        Vec3 q;
        if (g_smooth_grid.m && g_smooth_grid.closest(p, q, n)) return q;
        n = n_fallback.normalized();
        return p;
    };
    for (Cross& c : cx) {
        if (c.vid != ~0u || c.slave) continue;
        Vec3 n;
        Vec3 p = snapped(rm.v[c.a] + (rm.v[c.b] - rm.v[c.a]) * c.t,
                         rm.n[c.a] + (rm.n[c.b] - rm.n[c.a]) * c.t, n);
        if (c.twin == (uint32_t)(&c - cx.data())) p.x = 0.0f;
        c.vid = add_vertex(p, n, rm.v_owner[c.a]);
        if (c.twin == ~0u) continue;
        Cross& w = cx[c.twin];
        if (&w == &c) { rm.mirror[c.vid] = c.vid; continue; }
        w.vid = add_vertex(refl(p), refl(n), rm.v_owner[w.a]);
        rm.mirror[c.vid] = w.vid; rm.mirror[w.vid] = c.vid;
    }

    // Split every face with a cut side. Its outline (corners and cut points, in its own
    // winding) is cut between consecutive cut points: two of them split it straight
    // through (a corner clipped: round a turning vertex), three or more meet at a new
    // middle vertex.
    FaceList out;
    out.faces.reserve(rm.faces.size() + cx.size() * 8);
    std::vector<uint32_t> touched;
    struct Centre { std::vector<uint32_t> key, corners; uint32_t vid; };
    std::vector<Centre> centres;
    auto face_key = [](std::vector<uint32_t> c) { std::sort(c.begin(), c.end()); return c; };
    std::vector<uint32_t> c, cut, ring, ci, piece;
    int split = 0;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const uint32_t who = rm.face_owner[f];
        const int n = corners(f, c);
        cut.assign(n, ~0u);
        int cuts = 0;
        for (int k = 0; k < n && n >= 3; k++) {
            auto it = at.find(edge_key(c[k], c[(k + 1) % n]));
            if (it != at.end()) { cut[k] = cx[it->second].vid; cuts++; }
        }
        // The outline, and where its cut points sit in it: new points on its sides, and
        // (vertex cuts) corners. A walk between two neighbouring corners would be a
        // two-cornered piece: then the corners don't count and it cuts as a plain face.
        int L = 0, corner_cuts = 0;
        for (int use_corners = 1; use_corners >= 0; use_corners--) {
            ring.clear(); ci.clear();
            corner_cuts = 0;
            for (int k = 0; k < n; k++) {
                if (use_corners && is_chord((uint32_t)f, c[k])) { ci.push_back((uint32_t)ring.size()); corner_cuts++; }
                ring.push_back(c[k]);
                if (cut[k] != ~0u) { ci.push_back((uint32_t)ring.size()); ring.push_back(cut[k]); }
            }
            L = (int)ring.size();
            bool ok = true;
            if (ci.size() == 2)
                for (int k = 0; k < 2; k++) ok &= ((int)ci[(k + 1) % 2] - (int)ci[k] + L) % L >= 2;
            if (ok || !corner_cuts) break;
        }
        cuts = (int)ci.size();
        if (n < 3 || cuts == 0 || (cuts == 1 && corner_cuts == 1)) {
            out.add(rm.face(f), (int)rm.face_size(f), who);
            continue;
        }
        split++;
        touched.push_back(who);
        const bool quadify = n <= 4;
        if (cuts == 1) {
            // A triangle with one cut side is a quad already (a quad's lone cut ran on
            // through its far side above); a bigger face takes the point as a corner.
            emit_piece(rm, out, ring.data(), L, who, quadify);
            continue;
        }
        auto add_centre = [&](Vec3 at_p, Vec3 at_n) {
            Vec3 nn;
            const uint32_t v = add_vertex(snapped(at_p, at_n, nn), nn, who);
            centres.push_back({face_key(c), c, v});
            return v;
        };
        if (cuts == 2 && n == 4 && !corner_cuts && (ci[1] - ci[0] == 2 || ci[0] + L - ci[1] == 2)) {
            // A corner clipped: the cut turns inside the quad. Three quads round a new
            // vertex M where the turn is (p and q's corner reflected across them), the
            // cut running p -> M -> q and one support from M to the far corner.
            const int a = ci[1] - ci[0] == 2 ? ci[0] : ci[1];
            const uint32_t P = ring[a], V1 = ring[(a + 1) % 6], Q = ring[(a + 2) % 6],
                           V2 = ring[(a + 3) % 6], V3 = ring[(a + 4) % 6], V0 = ring[(a + 5) % 6];
            const uint32_t M = add_centre(rm.v[P] + rm.v[Q] - rm.v[V1], rm.n[P] + rm.n[Q]);
            const uint32_t f0[4] = {P, V1, Q, M}, f1[4] = {Q, V2, V3, M}, f2[4] = {V3, V0, P, M};
            out.add(f0, 4, who);
            out.add(f1, 4, who);
            out.add(f2, 4, who);
            continue;
        }
        uint32_t mid = ~0u;
        if (cuts >= 3) {
            Vec3 avg, an;
            for (int k = 0; k < cuts; k++) { avg += rm.v[ring[ci[k]]]; an += rm.n[ring[ci[k]]]; }
            mid = add_centre(avg * (1.0f / (float)cuts), an);
        }
        for (int k = 0; k < cuts; k++) {
            // From this cut point round to the next, then (three or more) the middle.
            const int from = ci[k], to = ci[(k + 1) % cuts];
            piece.clear();
            for (int i = from;; i = (i + 1) % L) {
                piece.push_back(ring[i]);
                if (i == to && piece.size() > 1) break;
            }
            if (mid != ~0u) piece.push_back(mid);
            emit_piece(rm, out, piece.data(), (int)piece.size(), who, quadify);
        }
    }
    // Middle vertices of a face and of its twin face pair up the same way.
    if (mirror)
        for (Centre& ce : centres) {
            std::vector<uint32_t> mc;
            bool ok = rm.mirror[ce.vid] == ~0u;
            for (size_t k = 0; k < ce.corners.size() && ok; k++) {
                mc.push_back(rm.mirror[ce.corners[k]]);
                ok = mc.back() != ~0u;
            }
            if (!ok) continue;
            mc = face_key(mc);
            for (Centre& o : centres) {
                if (o.key != mc) continue;
                if (&o == &ce) { rm.v[ce.vid].x = 0.0f; rm.mirror[ce.vid] = ce.vid; }
                else {
                    rm.v[o.vid] = refl(rm.v[ce.vid]); rm.n[o.vid] = refl(rm.n[ce.vid]);
                    rm.mirror[ce.vid] = o.vid; rm.mirror[o.vid] = ce.vid;
                }
                break;
            }
        }
    out.swap_into(rm);

    // Edges again, from the faces: the cut sides are gone, their halves and the new
    // edges across are in.
    rebuild_edges(rm);
    freeze_touched(st, touched);
    std::printf("[retopo] cut: %zu edges, %zu vertex end(s), %d faces split, now %u faces\n",
                cx.size(), chords.size(), split, rm.face_count());
    return split;
}

// ---- Erase (E) ------------------------------------------------------------------

namespace {

// A vertex left on a straight run of two edges is just a point on a line: it goes.
// Straight = the two edges within ~30 degrees of opposite.
constexpr float kDissolveStraight = -0.85f;

// Every face that uses the side a-b, either way round (one for an outer edge, two for
// an inner one).
void faces_with_side(const RetopoMesh& rm, uint32_t a, uint32_t b, std::vector<uint32_t>& out) {
    out.clear();
    std::vector<uint32_t> c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) {
            const uint32_t x = c[k], y = c[(k + 1) % n];
            if ((x == a && y == b) || (x == b && y == a)) { out.push_back((uint32_t)f); break; }
        }
    }
}

// Faces f1 and f2 (sharing side a-b) as one: f1's outline from b round to a, then f2's
// from a round to b. A line left poking into the new face (x, y, x in its outline: a
// vertex that only hung off the removed edge's end) is folded away. Returns the outline.
std::vector<uint32_t> merged_outline(const RetopoMesh& rm, uint32_t f1, uint32_t f2, uint32_t a, uint32_t b) {
    std::vector<uint32_t> c1, c2;
    face_corners(rm, f1, c1);
    face_corners(rm, f2, c2);
    // Run c from u round to w, where u follows w in it (the side w -> u is the one
    // being removed). Reversed when the side runs the other way (inconsistent winding).
    auto run = [](std::vector<uint32_t> c, uint32_t w, uint32_t u) {
        const int n = (int)c.size();
        int iu = -1;
        for (int k = 0; k < n; k++) if (c[k] == u && c[(k + n - 1) % n] == w) iu = k;
        if (iu < 0) {
            std::reverse(c.begin(), c.end());
            for (int k = 0; k < n; k++) if (c[k] == u && c[(k + n - 1) % n] == w) iu = k;
        }
        std::rotate(c.begin(), c.begin() + std::max(0, iu), c.end());
        return c;   // u ... w
    };
    // Go f1's way round, so the new face keeps its winding (and its normal out).
    bool ab = false;
    for (size_t k = 0; k < c1.size(); k++) ab |= c1[k] == a && c1[(k + 1) % c1.size()] == b;
    if (!ab) std::swap(a, b);
    std::vector<uint32_t> r1 = run(c1, a, b);   // b ... a
    std::vector<uint32_t> r2 = run(c2, b, a);   // a ... b
    std::vector<uint32_t> out(r1.begin(), r1.end());
    out.insert(out.end(), r2.begin() + 1, r2.end() - 1);
    for (bool more = true; more && out.size() >= 3;) {
        more = false;
        const size_t n = out.size();
        for (size_t i = 0; i < n; i++) {
            if (out[(i + n - 1) % n] != out[(i + 1) % n]) continue;
            // x, y, x: drop y and the second x.
            const size_t j = (i + 1) % n;
            if (i < j) { out.erase(out.begin() + j); out.erase(out.begin() + i); }
            else       { out.erase(out.begin() + i); out.erase(out.begin() + j); }
            more = true;
            break;
        }
    }
    return out;
}

// The vertices joined to v by a face side, each once.
void side_neighbours(const RetopoMesh& rm, uint32_t v, std::vector<uint32_t>& out) {
    out.clear();
    std::vector<uint32_t> c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) {
            if (c[k] != v) continue;
            for (uint32_t w : {c[(k + n - 1) % n], c[(k + 1) % n]})
                if (std::find(out.begin(), out.end(), w) == out.end()) out.push_back(w);
        }
    }
}

// Take out the inner edge a-b: its two faces become one. Then an end left as a point on
// a straight line dissolves out of every face it's in, as long as each keeps three
// corners. False when a-b isn't an inner edge. Owners of what changed go in `touched`.
bool erase_edge(RetopoMesh& rm, uint32_t a, uint32_t b, std::vector<uint32_t>& touched) {
    std::vector<uint32_t> fs;
    faces_with_side(rm, a, b, fs);
    if (fs.size() != 2) return false;   // an outer edge, or not a side at all
    const std::vector<uint32_t> merged = merged_outline(rm, fs[0], fs[1], a, b);
    touched.push_back(rm.face_owner[fs[0]]);
    touched.push_back(rm.face_owner[fs[1]]);
    FaceList out;
    for (size_t f = 0; f < rm.face_count(); f++) {
        if (f == fs[1]) continue;
        if (f == fs[0]) {
            if (merged.size() >= 3) out.add(merged.data(), (int)merged.size(), rm.face_owner[f]);
            continue;
        }
        out.add(rm.face(f), (int)rm.face_size(f), rm.face_owner[f]);
    }
    out.swap_into(rm);

    std::vector<uint32_t> nb, c;
    for (uint32_t v : {a, b}) {
        side_neighbours(rm, v, nb);
        if (nb.size() != 2) continue;
        const Vec3 d0 = (rm.v[nb[0]] - rm.v[v]).normalized(), d1 = (rm.v[nb[1]] - rm.v[v]).normalized();
        if (d0.dot(d1) > kDissolveStraight) continue;
        bool ok = true;
        for (size_t f = 0; f < rm.face_count() && ok; f++) {
            const int n = face_corners(rm, f, c);
            if (std::find(c.begin(), c.end(), v) != c.end()) ok = n >= 4;
        }
        if (!ok) continue;
        FaceList keep;
        for (size_t f = 0; f < rm.face_count(); f++) {
            const int n = face_corners(rm, f, c);
            c.erase(std::remove(c.begin(), c.end(), v), c.end());
            if (c.size() < (size_t)n) touched.push_back(rm.face_owner[f]);
            keep.add(c.data(), (int)c.size(), rm.face_owner[f]);
        }
        keep.swap_into(rm);
    }
    return true;
}

}  // namespace

int retopo_erase_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                     float cx, float cy, float radius_px, bool mirror) {
    RetopoMesh& rm = st.result;
    if (!st.built || rm.edges.empty()) return 0;
    const size_t nv = rm.v.size();
    static std::vector<float> sx, sy;
    static std::vector<uint8_t> vok;
    sx.resize(nv); sy.resize(nv); vok.resize(nv);
    for (size_t i = 0; i < nv; i++) {
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
        vok[i] = cam.world_to_screen(rm.v[i], win_w, win_h, sx[i], sy[i]) && rm.n[i].dot(to_eye) > 0.0f
              && !st.hidden(rm.v_owner[i]);
    }
    // The inner edge whose middle is nearest the centre, inside the ring.
    uint32_t ba = ~0u, bb = ~0u;
    float best = radius_px * radius_px;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (!vok[a] || !vok[b]) continue;
        if (rm.border_nb[a * 2] == b || rm.border_nb[a * 2 + 1] == b) continue;   // the outline stays
        const float mx = 0.5f * (sx[a] + sx[b]) - cx, my = 0.5f * (sy[a] + sy[b]) - cy;
        const float d2 = mx * mx + my * my;
        if (d2 < best) { best = d2; ba = a; bb = b; }
    }
    if (ba == ~0u) return 0;
    std::vector<uint32_t> touched;
    int gone = erase_edge(rm, ba, bb, touched) ? 1 : 0;
    if (gone && mirror) {
        const uint32_t ma = rm.mirror[ba], mb = rm.mirror[bb];
        if (ma != ~0u && mb != ~0u && edge_key(ma, mb) != edge_key(ba, bb))
            gone += erase_edge(rm, ma, mb, touched) ? 1 : 0;
    }
    if (!gone) return 0;
    rebuild_edges(rm);
    freeze_touched(st, touched);
    std::printf("[retopo] erase: %d edge(s), now %u faces (largest %u sides)\n", gone, rm.face_count(),
                [&] { uint32_t m = 0; for (size_t f = 0; f < rm.face_count(); f++) m = std::max(m, rm.face_size(f)); return m; }());
    return gone;
}

uint32_t retopo_face_at(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                        float cx, float cy, float radius_px) {
    const RetopoMesh& rm = st.result;
    if (!st.built) return ~0u;
    // The face under the centre (nearest the eye), else the one whose middle is nearest
    // it inside the ring.
    uint32_t under = ~0u, near = ~0u;
    float under_d = 1e30f, near_d2 = radius_px * radius_px;
    std::vector<float> sx, sy;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const uint32_t* q = rm.face(f);
        const uint32_t k = rm.face_size(f);
        if (k < 3 || st.hidden(rm.face_owner[f])) continue;
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[q[0]] : cam.get_view_direction() * -1.0f;
        if (rm.n[q[0]].dot(to_eye) <= 0.0f) continue;
        sx.resize(k); sy.resize(k);
        bool ok = true;
        float mx = 0.0f, my = 0.0f;
        for (uint32_t i = 0; i < k && ok; i++) {
            ok = cam.world_to_screen(rm.v[q[i]], win_w, win_h, sx[i], sy[i]);
            mx += sx[i]; my += sy[i];
        }
        if (!ok) continue;
        bool in = false;
        for (uint32_t i = 0, j = k - 1; i < k; j = i++)
            if ((sy[i] > cy) != (sy[j] > cy)
                && cx < sx[j] + (sx[i] - sx[j]) * (cy - sy[j]) / (sy[i] - sy[j])) in = !in;
        if (in) {
            const float d = cam.view_depth(rm.v[q[0]]);
            if (d < under_d) { under_d = d; under = (uint32_t)f; }
        }
        mx = mx / (float)k - cx; my = my / (float)k - cy;
        if (mx * mx + my * my < near_d2) { near_d2 = mx * mx + my * my; near = (uint32_t)f; }
    }
    return under != ~0u ? under : near;
}

int retopo_erase_face_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                          float cx, float cy, float radius_px, bool mirror) {
    RetopoMesh& rm = st.result;
    const uint32_t f = retopo_face_at(st, cam, win_w, win_h, cx, cy, radius_px);
    if (f == ~0u) return 0;
    std::vector<uint8_t> drop(rm.face_count(), 0);
    drop[f] = 1;
    if (mirror) {
        // Its twin: the face made of its corners' twins.
        std::vector<uint32_t> c, mk, fk;
        face_corners(rm, f, c);
        bool ok = true;
        for (uint32_t x : c) { mk.push_back(rm.mirror[x]); ok &= mk.back() != ~0u; }
        if (ok) {
            std::sort(mk.begin(), mk.end());
            for (size_t g = 0; g < rm.face_count(); g++) {
                face_corners(rm, g, fk);
                std::sort(fk.begin(), fk.end());
                if (fk == mk) { drop[g] = 1; break; }
            }
        }
    }
    std::vector<uint32_t> touched;
    FaceList out;
    int gone = 0;
    for (size_t g = 0; g < rm.face_count(); g++) {
        if (drop[g]) { touched.push_back(rm.face_owner[g]); gone++; continue; }
        out.add(rm.face(g), (int)rm.face_size(g), rm.face_owner[g]);
    }
    out.swap_into(rm);
    // Vertices only the deleted faces used are left behind unreferenced, like a weld's
    // (export skips them, nothing else sees them once the edges are rebuilt).
    rebuild_edges(rm);
    freeze_touched(st, touched);
    st.built = rm.face_count() > 0;
    std::printf("[retopo] erase: %d face(s) deleted, now %u faces\n", gone, rm.face_count());
    return gone;
}

// ---- Bridge (Draw from an edited island) ------------------------------------------

const char* retopo_bridge(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                          uint32_t to_id, bool mirror) {
    RetopoMesh& rm = st.result;
    const uint32_t from_id = st.bridge_from;
    const std::vector<Vec3>& pts = st.live.pts;
    const size_t np = pts.size();
    if (!st.built || !from_id || np < 2) return "Draw out of the island's edge to bridge it";
    if (to_id && (!st.find(to_id) || !st.find(to_id)->frozen))
        return "Bridges join edited islands only: Build or edit the other one first";

    // The line and the islands' outlines on screen. The bridge leaves `from` where the
    // line first crosses its edge, and lands where the line last crosses into `to`'s.
    std::vector<float> lx(np), ly(np);
    for (size_t i = 0; i < np; i++) cam.world_to_screen(pts[i], win_w, win_h, lx[i], ly[i]);
    const size_t nv = rm.v.size();
    std::vector<float> sx(nv), sy(nv);
    std::vector<uint8_t> vok(nv);
    for (size_t i = 0; i < nv; i++) vok[i] = cam.world_to_screen(rm.v[i], win_w, win_h, sx[i], sy[i]);
    struct Hit { uint32_t a, b, owner; float s; };
    std::vector<Hit> hits;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1], ow = rm.edge_owner[e / 2];
        if (ow != from_id && ow != to_id) continue;
        if (!vok[a] || !vok[b] || !(rm.border_nb[a * 2] == b || rm.border_nb[a * 2 + 1] == b)) continue;
        const float rx = sx[b] - sx[a], ry = sy[b] - sy[a];
        for (size_t i = 0; i + 1 < np; i++) {
            const float qx = lx[i + 1] - lx[i], qy = ly[i + 1] - ly[i];
            const float den = rx * qy - ry * qx;
            if (std::fabs(den) < 1e-9f) continue;
            const float wx = lx[i] - sx[a], wy = ly[i] - sy[a];
            const float t = (wx * qy - wy * qx) / den, u = (wx * ry - wy * rx) / den;
            if (t < 0.0f || t > 1.0f || u < 0.0f || u > 1.0f) continue;
            hits.push_back({a, b, ow, (float)i + u});
        }
    }
    const Hit* out = nullptr;
    for (const Hit& h : hits) if (h.owner == from_id && (!out || h.s < out->s)) out = &h;
    if (!out) return "Draw out across the island's edge to bridge it";
    const Hit* in = nullptr;
    if (to_id)
        for (const Hit& h : hits)
            if (h.owner == to_id && h.s > out->s && edge_key(h.a, h.b) != edge_key(out->a, out->b)
                && (!in || h.s > in->s)) in = &h;
    if (to_id && !in) return "Land the bridge across the other island's edge";

    // The centreline: from the middle of the edge it leaves, along the line as drawn,
    // to the middle of the edge it lands on (or the line's end: a loose strip).
    auto at = [&](float s) {
        const size_t i = std::min(np - 2, (size_t)s);
        return pts[i] + (pts[i + 1] - pts[i]) * (s - (float)i);
    };
    std::vector<Vec3> line;
    line.push_back((rm.v[out->a] + rm.v[out->b]) * 0.5f);
    const float s_end = in ? in->s : (float)(np - 1);
    for (size_t i = (size_t)std::ceil(out->s); (float)i < s_end && i < np; i++) line.push_back(pts[i]);
    line.push_back(in ? (rm.v[in->a] + rm.v[in->b]) * 0.5f : at(s_end));
    std::vector<float> arc(line.size(), 0.0f);
    for (size_t i = 1; i < line.size(); i++) arc[i] = arc[i - 1] + (line[i] - line[i - 1]).length();
    const float len = arc.back();
    auto along = [&](float d) {
        size_t i = 1;
        while (i + 1 < arc.size() && arc[i] < d) i++;
        const float seg = std::max(1e-9f, arc[i] - arc[i - 1]);
        return line[i - 1] + (line[i] - line[i - 1]) * std::min(1.0f, std::max(0.0f, (d - arc[i - 1]) / seg));
    };

    // Steps the size of the quads it joins: the mean side of the faces on those edges.
    std::vector<uint32_t> c, fs;
    auto face_size_at = [&](uint32_t a, uint32_t b) {
        float sum = 0.0f;
        int n = 0;
        for (size_t f = 0; f < rm.face_count(); f++) {
            const int k = face_corners(rm, f, c);
            bool has = false;
            for (int i = 0; i < k; i++)
                has |= (c[i] == a && c[(i + 1) % k] == b) || (c[i] == b && c[(i + 1) % k] == a);
            if (!has) continue;
            for (int i = 0; i < k; i++) { sum += (rm.v[c[(i + 1) % k]] - rm.v[c[i]]).length(); n++; }
        }
        return n ? sum / (float)n : (rm.v[a] - rm.v[b]).length();
    };
    float q = face_size_at(out->a, out->b);
    if (in) q = 0.5f * (q + face_size_at(in->a, in->b));
    const int n = std::max(1, (int)std::lround(len / std::max(q, 1e-6f)));

    // Which end vertex is on the line's left: the side = normal x heading.
    auto left_first = [&](uint32_t a, uint32_t b, Vec3 heading) {
        const Vec3 nn = (rm.n[a] + rm.n[b]).normalized();
        const Vec3 side = nn.cross(heading);
        return (rm.v[a] - rm.v[b]).dot(side) > 0.0f;
    };
    const Vec3 h0 = (line[std::min<size_t>(1, line.size() - 1)] - line[0]).normalized();
    const uint32_t LA = left_first(out->a, out->b, h0) ? out->a : out->b;
    const uint32_t RA = LA == out->a ? out->b : out->a;
    uint32_t LB = ~0u, RB = ~0u;
    if (in) {
        const Vec3 h1 = (line.back() - line[line.size() - 2]).normalized();
        LB = left_first(in->a, in->b, h1) ? in->a : in->b;
        RB = LB == in->a ? in->b : in->a;
    }
    const float wA = (rm.v[LA] - rm.v[RA]).length();
    const float wB = in ? (rm.v[LB] - rm.v[RB]).length() : wA;
    const Vec3 dA = (rm.v[LA] - rm.v[RA]).normalized();
    const Vec3 dB = in ? (rm.v[LB] - rm.v[RB]).normalized() : dA;

    // The snap grid, over the islands and the line (a loose strip can run past them).
    SnapGrid g;
    {
        Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
        auto grow = [&](Vec3 p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        };
        for (const Vec3& p : line) grow(p);
        const float pad = std::max(wA, wB) * 2.0f + q;
        lo = lo - Vec3(pad, pad, pad);
        hi = hi + Vec3(pad, pad, pad);
        const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        g.build(m, lo, hi, std::max(q * 0.5f, ext / 96.0f));
    }
    auto snap = [&](Vec3 p, Vec3& nn) {
        Vec3 r;
        if (g.closest(p, r, nn, 8)) return r;
        nn = Vec3(0, 0, 0);
        return p;
    };

    st.checkpoint();
    // Spokes 1..n-1 (and n for a loose end): centre on the line, across = between the
    // line's own sideways and the two end edges' directions, width eased from one end
    // edge's length to the other's.
    const uint32_t keep = to_id ? std::min(from_id, to_id) : from_id;
    std::vector<uint32_t> L(n + 1), R(n + 1);
    L[0] = LA; R[0] = RA;
    if (in) { L[n] = LB; R[n] = RB; }
    const uint32_t first_new = (uint32_t)rm.v.size();
    for (int i = 1; i <= n; i++) {
        if (i == n && in) break;
        const float f = (float)i / (float)n;
        Vec3 nn;
        const Vec3 cen = snap(along(f * len), nn);
        const Vec3 tan = (along(std::min(len, (f + 0.5f / n) * len)) - along(std::max(0.0f, (f - 0.5f / n) * len))).normalized();
        Vec3 side = nn.cross(tan).normalized();
        const Vec3 ends = (dA * (1.0f - f) + dB * f).normalized();
        if (side.dot(ends) < 0.0f) side = side * -1.0f;
        Vec3 across = (side + ends).normalized();
        across = (across - nn * across.dot(nn)).normalized();
        const float w = wA + (wB - wA) * f;
        for (int k = 0; k < 2; k++) {
            Vec3 vn;
            const Vec3 p = snap(cen + across * (k ? -0.5f * w : 0.5f * w), vn);
            rm.v.push_back(p);
            rm.n.push_back(vn.length() > 0.5f ? vn : nn);
            rm.v_owner.push_back(keep);
            rm.mirror.push_back(~0u);
            (k ? R : L)[i] = (uint32_t)rm.v.size() - 1;
        }
    }
    // Wound against the island's face on the edge it leaves, so the bridge's normals
    // agree with it (a shared side runs opposite ways in its two faces).
    bool flip = false;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int k = face_corners(rm, f, c);
        for (int i = 0; i < k; i++)
            if (c[i] == RA && c[(i + 1) % k] == LA) flip = true;
    }
    auto quad = [&](int i, uint32_t* qd) {
        qd[0] = L[i]; qd[1] = L[i + 1]; qd[2] = R[i + 1]; qd[3] = R[i];
        if (flip) std::reverse(qd, qd + 4);
    };
    const uint32_t first_face = rm.face_count();
    for (int i = 0; i < n; i++) { uint32_t qd[4]; quad(i, qd); rm.add_face(qd, 4, keep); }

    // Mirror. A bridge from an island to its own twin is its own twin: spoke i pairs with
    // spoke n - i, evened out across the seam. Otherwise, with both ends twinned and the
    // line on one side, it gets a twin bridge, reflected vertex by vertex.
    int twin = 0;
    if (mirror && rm.mirror[LA] != ~0u && rm.mirror[RA] != ~0u) {
        auto refl = [](Vec3 v) { v.x = -v.x; return v; };
        const bool self = in && ((rm.mirror[LA] == LB && rm.mirror[RA] == RB) || (rm.mirror[LA] == RB && rm.mirror[RA] == LB));
        bool one_side = true;
        for (const Vec3& p : line) one_side &= (p.x > 0.0f) == (line[0].x > 0.0f);
        if (self) {
            const bool same_rail = rm.mirror[LA] == LB;
            for (int i = 1; i <= n / 2; i++) {
                const int j = n - i;
                for (int k = 0; k < 2; k++) {
                    const uint32_t x = (k ? R : L)[i];
                    const uint32_t y = same_rail ? (k ? R : L)[j] : (k ? L : R)[j];
                    if (x == y) { rm.v[x].x = 0.0f; rm.mirror[x] = x; continue; }
                    const Vec3 mid = (rm.v[x] + refl(rm.v[y])) * 0.5f;
                    rm.v[x] = mid; rm.v[y] = refl(mid);
                    rm.n[y] = refl(rm.n[x]);
                    rm.mirror[x] = y; rm.mirror[y] = x;
                }
            }
            twin = 1;
        } else if (one_side && (!in || (rm.mirror[LB] != ~0u && rm.mirror[RB] != ~0u))) {
            std::vector<uint32_t> tw(rm.v.size(), ~0u);
            auto tw_of = [&](uint32_t v) {
                if (v < first_new) return rm.mirror[v];
                if (tw[v] == ~0u) {
                    tw[v] = (uint32_t)rm.v.size();
                    rm.v.push_back(refl(rm.v[v]));
                    rm.n.push_back(refl(rm.n[v]));
                    rm.v_owner.push_back(keep);
                    rm.mirror.push_back(v);
                    rm.mirror[v] = tw[v];
                    tw.push_back(~0u);
                }
                return tw[v];
            };
            for (int i = 0; i < n; i++) {
                const uint32_t* qd = rm.face(first_face + i);
                const uint32_t t4[4] = {tw_of(qd[3]), tw_of(qd[2]), tw_of(qd[1]), tw_of(qd[0])};
                rm.add_face(t4, 4, keep);
            }
            twin = 2;
        }
    }

    // One piece now: the islands it joins are one island (the oldest keeps its name).
    if (to_id && to_id != from_id) {
        const uint32_t gone = keep == from_id ? to_id : from_id;
        for (std::vector<uint32_t>* ow : {&rm.face_owner, &rm.v_owner, &rm.edge_owner})
            for (uint32_t& o : *ow) if (o == gone) o = keep;
        st.strokes.erase(std::remove_if(st.strokes.begin(), st.strokes.end(),
                                        [&](const RetopoStroke& s) { return s.id == gone; }), st.strokes.end());
    }
    rebuild_edges(rm);
    std::printf("[retopo] bridge: %s, %d quads (step %.3f, width %.3f -> %.3f)%s\n",
                to_id ? (to_id == from_id ? "within one island" : "two islands joined") : "loose strip",
                n, q, wA, wB, twin == 1 ? ", its own twin" : twin == 2 ? ", twin built" : "");
    return nullptr;
}

// ---- Fill --------------------------------------------------------------------------

namespace {
// The aim starts this far above the rim (x its size: an eyeball can bulge out past it)
// and reaches this far below (a socket deeper than its opening is wide is rare).
constexpr float kFillAbove = 0.25f, kFillBelow = 1.5f;
// Rims with this many sides or more get a ring of quads inside before the grid.
constexpr int kFillRingMin = 12;
}  // namespace

int retopo_fill(RetopoState& st, const Mesh& m, bool mirror, int& rims, int& odd) {
    rims = odd = 0;
    RetopoMesh& rm = st.result;
    const uint32_t nr = rm.rim_count();
    if (!st.built || nr == 0) return 0;
    auto rim_n = [&](uint32_t r) { return rm.rim_off[r + 1] - rm.rim_off[r]; };
    auto rim_v = [&](uint32_t r, uint32_t k) { return rm.rim[rm.rim_off[r] + k % rim_n(r)]; };

    // Which rims: the picked loops that are whole outlines, else every hole. Hidden
    // islands' rims stay as they are.
    std::vector<uint8_t> want(nr, 0);
    bool picked = false;
    if (!st.loop_edges.empty()) {
        std::vector<uint64_t> sel;
        for (size_t i = 0; i + 1 < st.loop_edges.size(); i += 2)
            sel.push_back(edge_key(st.loop_edges[i], st.loop_edges[i + 1]));
        std::sort(sel.begin(), sel.end());
        for (uint32_t r = 0; r < nr; r++) {
            bool all = true;
            for (uint32_t k = 0; k < rim_n(r) && all; k++)
                all = std::binary_search(sel.begin(), sel.end(), edge_key(rim_v(r, k), rim_v(r, k + 1)));
            want[r] = all;
            picked |= all;
        }
    }
    if (!picked)
        for (uint32_t r = 0; r < nr; r++) want[r] = rm.rim_hole[r];
    for (uint32_t r = 0; r < nr; r++)
        for (uint32_t k = 0; k < rim_n(r) && want[r]; k++)
            if (st.hidden(rm.v_owner[rim_v(r, k)])) want[r] = 0;

    // Mirror: each rim's twin (its vertices' twins, all in one rim of the same size).
    // A rim that is its own twin straddles x = 0.
    std::vector<uint32_t> rim_of(rm.v.size(), ~0u), twin(nr, ~0u);
    for (uint32_t r = 0; r < nr; r++)
        for (uint32_t k = 0; k < rim_n(r); k++) rim_of[rim_v(r, k)] = r;
    if (mirror && rm.mirror.size() == rm.v.size())
        for (uint32_t r = 0; r < nr; r++) {
            const uint32_t t0 = rm.mirror[rim_v(r, 0)];
            const uint32_t t = t0 < rim_of.size() ? rim_of[t0] : ~0u;
            bool ok = t != ~0u && rim_n(t) == rim_n(r);
            for (uint32_t k = 0; k < rim_n(r) && ok; k++) {
                const uint32_t w = rm.mirror[rim_v(r, k)];
                ok = w < rim_of.size() && rim_of[w] == t;
            }
            if (ok) twin[r] = t;
        }
    // A twin pair is filled once, from the side found first; the other is its
    // reflection (picking either side fills both).
    for (uint32_t r = 0; r < nr; r++)
        if (twin[r] != ~0u && twin[r] > r && want[r]) want[twin[r]] = 0;
    for (uint32_t r = 0; r < nr; r++)
        if (want[r] && rim_n(r) % 2) { odd++; want[r] = 0; }
    uint32_t todo = 0;
    for (uint32_t r = 0; r < nr; r++) todo += want[r];
    if (!todo) return 0;

    // Each rim's facing: its area normal (Newell), turned to agree with the vertices'
    // own normals, and its size. The inside is aimed down this normal onto the sculpt,
    // not to the nearest surface: across a deep socket the nearest surface is the wall
    // just under the rim, and everything would pile up there.
    auto rim_facing = [&](uint32_t rr, Vec3& cen, float& diam) {
        Vec3 nw(0, 0, 0), ns(0, 0, 0);
        cen = Vec3(0, 0, 0);
        const uint32_t n = rim_n(rr);
        for (uint32_t k = 0; k < n; k++) {
            const Vec3 a = rm.v[rim_v(rr, k)], b = rm.v[rim_v(rr, k + 1)];
            nw = nw + Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
            ns = ns + rm.n[rim_v(rr, k)];
            cen = cen + a;
        }
        cen = cen * (1.0f / (float)n);
        diam = 0.0f;
        for (uint32_t k = 0; k < n; k++) diam = std::max(diam, 2.0f * (rm.v[rim_v(rr, k)] - cen).length());
        if (nw.length() < 1e-12f) nw = ns;
        if (nw.dot(ns) < 0.0f) nw = nw * -1.0f;
        return nw.normalized();
    };

    // The snap grid over every rim being filled (and its twin), deep enough for the aim.
    SnapGrid g;
    float edge = 0.0f;
    {
        Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
        auto grow = [&](Vec3 p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        };
        int ne = 0;
        for (uint32_t r = 0; r < nr; r++) {
            if (!want[r]) continue;
            for (uint32_t rr : {r, twin[r]}) {
                if (rr == ~0u) continue;
                Vec3 cen;
                float diam;
                const Vec3 nf = rim_facing(rr, cen, diam);
                for (uint32_t k = 0; k < rim_n(rr); k++) {
                    const Vec3 p = rm.v[rim_v(rr, k)];
                    grow(p);
                    grow(p + nf * (kFillAbove * diam));
                    grow(p - nf * (kFillBelow * diam));
                    edge += (rm.v[rim_v(rr, k + 1)] - p).length();
                    ne++;
                }
            }
        }
        edge /= (float)std::max(1, ne);
        const float pad = edge * 3.0f;
        lo = lo - Vec3(pad, pad, pad);
        hi = hi + Vec3(pad, pad, pad);
        const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        g.build(m, lo, hi, std::max(edge * 0.5f, ext / 96.0f));
    }
    auto snap = [&](Vec3 p, Vec3& nn) {
        Vec3 r;
        if (g.closest(p, r, nn, 8)) return r;
        nn = Vec3(0, 0, 0);
        return p;
    };
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };

    st.checkpoint();
    std::vector<uint32_t> touched;
    std::vector<Vec3> L;
    std::vector<uint32_t> Li, Ri, c;
    int made = 0;
    for (uint32_t r = 0; r < nr; r++) {
        if (!want[r]) continue;
        const int N = (int)rim_n(r), h = N / 2;
        Ri.resize(N);
        for (int k = 0; k < N; k++) Ri[k] = rim_v(r, k);
        for (uint32_t k : Ri) touched.push_back(rm.v_owner[k]);
        const bool self = twin[r] == r;
        // Its own twin: the reflection maps rim vertex k to vertex (t - k) mod N.
        int t = 0;
        if (self) for (int k = 0; k < N; k++) if (Ri[k] == rm.mirror[Ri[0]]) t = k;
        auto md = [&](int k) { return ((k % N) + N) % N; };
        Vec3 cen;
        float diam;
        const Vec3 nf = rim_facing(r, cen, diam);
        const uint32_t owner = rm.v_owner[Ri[0]];
        const uint32_t first_new = (uint32_t)rm.v.size();
        auto add_vertex = [&](Vec3 p, Vec3 nn) {
            rm.v.push_back(p);
            rm.n.push_back(nn.length() > 0.5f ? nn : nf);
            rm.v_owner.push_back(owner);
            rm.mirror.push_back(~0u);
            return (uint32_t)rm.v.size() - 1;
        };
        // Down the rim's facing from a little above the rim, onto the first surface
        // that faces back (the socket's floor, or an eyeball that bulges out past the
        // rim); nearest surface only if the aim finds none.
        auto aim = [&](Vec3 p, Vec3& nn) {
            Vec3 hit, hn;
            const float up = kFillAbove * diam;
            if (g.ray(p + nf * up, nf * -1.0f, up + kFillBelow * diam, hit, hn)) { nn = hn; return hit; }
            return snap(p, nn);
        };

        // The ring: one loop of quads just inside the rim, each vertex walked across
        // the surface (snapped and re-stretched to a rim edge's length, like Build's
        // spokes) so it hugs a socket's wall instead of jumping to its floor. The grid
        // then fills what's left: its four corners sit inside, off the rim.
        const bool ring = N >= kFillRingMin;
        float rim_edge = 0.0f;
        for (int k = 0; k < N; k++) rim_edge += (rm.v[Ri[md(k + 1)]] - rm.v[Ri[k]]).length();
        rim_edge /= (float)N;
        Li = Ri;
        if (ring) {
            for (int k = 0; k < N; k++) {
                const Vec3 p = rm.v[Ri[k]];
                const Vec3 tg = rm.v[Ri[md(k + 1)]] - rm.v[Ri[md(k - 1)]];
                Vec3 in = nf.cross(tg).normalized();
                if (in.dot(cen - p) < 0.0f) in = in * -1.0f;
                // A rim edge in, but never near the middle: a narrow hole's two sides
                // would cross.
                Vec3 off = cen - p;
                off = off - nf * off.dot(nf);
                const float step = std::min(rim_edge, 0.45f * off.length());
                Vec3 q = p + in * step, nn;
                for (int it = 0; it < 3; it++) {
                    q = snap(q, nn);
                    const Vec3 d = q - p;
                    if (d.length() < 1e-9f) break;
                    q = p + d.normalized() * step;
                }
                q = snap(q, nn);
                Li[k] = add_vertex(q, nn);
            }
        }
        L.resize(N);
        for (int k = 0; k < N; k++) L[k] = rm.v[Li[k]];

        // Corners: four where the loop turns most, opposite sides the same count (m, n,
        // m, n), quads as near square as the side lengths allow.
        std::vector<float> turn(N), arc(N + 1, 0.0f);
        for (int k = 0; k < N; k++) {
            const Vec3 a = (L[k] - L[md(k - 1)]).normalized(), b = (L[md(k + 1)] - L[k]).normalized();
            turn[k] = 1.0f - a.dot(b);
            arc[k + 1] = arc[k] + (L[md(k + 1)] - L[k]).length();
        }
        auto span = [&](int a, int n) {   // length from vertex a, n sides on
            const int b = a + n;
            return b <= N ? arc[b] - arc[a] : arc[N] - arc[a] + arc[b - N];
        };
        int bs = 0, bm = std::max(1, h / 2);
        float best = -1e30f;
        for (int s = 0; s < h; s++)
            for (int mm = 1; mm < h; mm++) {
                const int nn = h - mm;
                const int cs[4] = {s, md(s + mm), md(s + h), md(s + h + mm)};
                if (self) {
                    bool sym = true;
                    for (int a = 0; a < 4 && sym; a++) {
                        const int w = md(t - cs[a]);
                        sym = w == cs[0] || w == cs[1] || w == cs[2] || w == cs[3];
                    }
                    if (!sym) continue;
                }
                const float s1 = span(cs[0], mm), s2 = span(cs[1], nn), s3 = span(cs[2], mm), s4 = span(cs[3], nn);
                const float aspect = std::fabs(std::log(((s1 + s3) / mm) / std::max(1e-9f, (s2 + s4) / nn)));
                const float score = turn[cs[0]] + turn[cs[1]] + turn[cs[2]] + turn[cs[3]] - 0.5f * aspect
                                  - 0.25f * (std::fabs(std::log(s1 / std::max(1e-9f, s3))) + std::fabs(std::log(s2 / std::max(1e-9f, s4))));
                if (score > best) { best = score; bs = s; bm = mm; }
            }
        const int M = bm, Nn = h - bm;   // grid: M quads one way, Nn the other

        // Grid (M + 1) x (Nn + 1): its outside is the loop, from corner bs round.
        // bottom (i, 0) = L[bs + i], right (M, j) = L[bs + M + j],
        // top (i, Nn) = L[bs + h + (M - i)], left (0, j) = L[bs + h + M + (Nn - j)].
        const int GW = M + 1, GH = Nn + 1;
        std::vector<uint32_t> gi(GW * GH, ~0u);
        auto G = [&](int i, int j) -> uint32_t& { return gi[j * GW + i]; };
        for (int i = 0; i <= M; i++) { G(i, 0) = Li[md(bs + i)]; G(i, Nn) = Li[md(bs + h + (M - i))]; }
        for (int j = 0; j <= Nn; j++) { G(M, j) = Li[md(bs + M + j)]; G(0, j) = Li[md(bs + h + M + (Nn - j))]; }
        // Inside: Coons (each point blended from the four sides, minus the corners'
        // double count), aimed down the rim's facing onto the sculpt.
        const Vec3 P00 = rm.v[G(0, 0)], P10 = rm.v[G(M, 0)], P01 = rm.v[G(0, Nn)], P11 = rm.v[G(M, Nn)];
        for (int j = 1; j < Nn; j++)
            for (int i = 1; i < M; i++) {
                const float u = (float)i / M, v = (float)j / Nn;
                const Vec3 p = rm.v[G(i, 0)] * (1 - v) + rm.v[G(i, Nn)] * v
                             + rm.v[G(0, j)] * (1 - u) + rm.v[G(M, j)] * u
                             - (P00 * ((1 - u) * (1 - v)) + P10 * (u * (1 - v)) + P01 * ((1 - u) * v) + P11 * (u * v));
                Vec3 nn;
                const Vec3 q = aim(p, nn);
                G(i, j) = add_vertex(q, nn);
            }
        // Relax: each inside vertex toward its four neighbours' middle, but only along
        // the surface (the part along its normal dropped), then back onto it. A plain
        // average would lift the floor of a socket toward the rim, and the nearest
        // surface from there is the wall again.
        std::vector<Vec3> tmp(GW * GH);
        for (int pass = 0; pass < 4 && M > 1 && Nn > 1; pass++) {
            for (int j = 1; j < Nn; j++)
                for (int i = 1; i < M; i++) {
                    const Vec3 p = rm.v[G(i, j)], n = rm.n[G(i, j)];
                    Vec3 d = (rm.v[G(i - 1, j)] + rm.v[G(i + 1, j)] + rm.v[G(i, j - 1)] + rm.v[G(i, j + 1)]) * 0.25f - p;
                    d = d - n * d.dot(n);
                    tmp[j * GW + i] = p + d;
                }
            for (int j = 1; j < Nn; j++)
                for (int i = 1; i < M; i++) {
                    Vec3 nn;
                    rm.v[G(i, j)] = snap(tmp[j * GW + i], nn);
                    if (nn.length() > 0.5f) rm.n[G(i, j)] = nn;
                }
        }
        // Its own twin: pair the new vertices with their reflections and even them out
        // (ring and grid are symmetric, the corners were chosen so); on the seam, x = 0.
        if (self) {
            const uint32_t end = (uint32_t)rm.v.size();
            for (uint32_t x = first_new; x < end; x++) {
                if (rm.mirror[x] != ~0u) continue;
                uint32_t y = x;
                float bd = 1e30f;
                for (uint32_t z = first_new; z < end; z++) {
                    if (rm.mirror[z] != ~0u && z != x) continue;
                    const Vec3 d = rm.v[z] - refl(rm.v[x]);
                    if (d.dot(d) < bd) { bd = d.dot(d); y = z; }
                }
                if (y == x) { rm.v[x].x = 0.0f; rm.mirror[x] = x; continue; }
                const Vec3 mid = (rm.v[x] + refl(rm.v[y])) * 0.5f;
                rm.v[x] = mid; rm.v[y] = refl(mid); rm.n[y] = refl(rm.n[x]);
                rm.mirror[x] = y; rm.mirror[y] = x;
            }
        }
        // Wound against the face already on the rim: a shared side runs opposite ways
        // in its two faces. The ring's quad k has the side R[k] -> R[k + 1]; the grid's
        // (0, 0) has L[bs] -> L[bs + 1], the same way round as the ring's inner side
        // runs the other way, so one flip suits both.
        bool flip = false;
        for (size_t f = 0; f < rm.face_count() && !flip; f++) {
            const int k = face_corners(rm, f, c);
            for (int i = 0; i < k; i++) if (c[i] == Ri[0] && c[(i + 1) % k] == Ri[1]) flip = true;
        }
        const uint32_t first_face = rm.face_count();
        if (ring)
            for (int k = 0; k < N; k++) {
                uint32_t q[4] = {Ri[k], Ri[md(k + 1)], Li[md(k + 1)], Li[k]};
                if (flip) std::reverse(q, q + 4);
                rm.add_face(q, 4, owner);
            }
        for (int j = 0; j < Nn; j++)
            for (int i = 0; i < M; i++) {
                uint32_t q[4] = {G(i, j), G(i + 1, j), G(i + 1, j + 1), G(i, j + 1)};
                if (flip) std::reverse(q, q + 4);
                rm.add_face(q, 4, owner);
            }
        const int nq = M * Nn + (ring ? N : 0);
        made += nq;
        rims++;
        std::printf("[retopo] fill: rim of %d sides -> %s%d x %d grid, %d quads (depth %.3f of %.3f across)%s\n",
                    N, ring ? "ring + " : "", M, Nn, nq,
                    [&] { float dmax = 0; for (uint32_t x = first_new; x < rm.v.size(); x++)
                              dmax = std::max(dmax, (cen - rm.v[x]).dot(nf)); return dmax; }(), diam,
                    self ? ", its own twin" : twin[r] != ~0u ? ", twin too" : "");

        // A separate twin: its fill is this one reflected, vertex by vertex.
        if (twin[r] != ~0u && !self) {
            for (uint32_t k : Ri) touched.push_back(rm.v_owner[rm.mirror[k]]);
            const uint32_t end = (uint32_t)rm.v.size();
            std::vector<uint32_t> tw(end - first_new, ~0u);
            auto tw_of = [&](uint32_t v) {
                if (v < first_new) return rm.mirror[v];
                uint32_t& w = tw[v - first_new];
                if (w == ~0u) {
                    w = (uint32_t)rm.v.size();
                    rm.v.push_back(refl(rm.v[v]));
                    rm.n.push_back(refl(rm.n[v]));
                    rm.v_owner.push_back(rm.v_owner[rm.mirror[Ri[0]]]);
                    rm.mirror.push_back(v);
                    rm.mirror[v] = w;
                }
                return w;
            };
            const uint32_t nface = rm.face_count();
            for (uint32_t f = first_face; f < nface; f++) {
                const uint32_t* q = rm.face(f);
                const uint32_t t4[4] = {tw_of(q[3]), tw_of(q[2]), tw_of(q[1]), tw_of(q[0])};
                rm.add_face(t4, 4, rm.v_owner[rm.mirror[Ri[0]]]);
            }
            made += nq;
            rims++;
        }
    }

    // One piece now: every island round a filled rim joins the oldest of them, and it's
    // geometry (frozen) from here on.
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    uint32_t keep = ~0u;
    for (uint32_t id : touched) if (st.find(id)) keep = std::min(keep, id);
    if (keep != ~0u) {
        for (uint32_t id : touched) if (RetopoStroke* s = st.find(id)) s->frozen = true;
        for (std::vector<uint32_t>* ow : {&rm.face_owner, &rm.v_owner, &rm.edge_owner})
            for (uint32_t& o : *ow) if (std::binary_search(touched.begin(), touched.end(), o)) o = keep;
        bool vis = false, sel = false;
        for (const RetopoStroke& s : st.strokes)
            if (s.id != keep && std::binary_search(touched.begin(), touched.end(), s.id)) { vis |= s.visible; sel |= s.selected; }
        st.strokes.erase(std::remove_if(st.strokes.begin(), st.strokes.end(), [&](const RetopoStroke& s) {
            return s.id != keep && std::binary_search(touched.begin(), touched.end(), s.id);
        }), st.strokes.end());
        if (RetopoStroke* k = st.find(keep)) { k->visible |= vis; k->selected |= sel; }
    }
    st.loop_edges.clear();
    rebuild_edges(rm);
    std::printf("[retopo] fill: %d rim(s), %d quads, %d odd rim(s) left; %u holes remain\n",
                rims, made, odd, (uint32_t)std::count(rm.rim_hole.begin(), rm.rim_hole.end(), 1));
    return made;
}

bool retopo_export_obj(const RetopoMesh& rm, const char* path) {
    if (rm.faces.empty()) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "# Chisel retopo: %u faces\n", rm.face_count());
    // Welds leave folded-away vertices behind: write only the ones a quad uses.
    std::vector<uint32_t> id(rm.v.size(), 0);
    for (uint32_t i : rm.faces) id[i] = 1;
    uint32_t next = 0;
    for (size_t i = 0; i < rm.v.size(); i++) {
        if (!id[i]) continue;
        id[i] = ++next;
        std::fprintf(f, "v %.6f %.6f %.6f\n", rm.v[i].x, rm.v[i].y, rm.v[i].z);
    }
    for (size_t q = 0; q < rm.face_count(); q++) {
        std::fprintf(f, "f");
        for (uint32_t k = rm.face_off[q]; k < rm.face_off[q + 1]; k++) std::fprintf(f, " %u", id[rm.faces[k]]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return true;
}
