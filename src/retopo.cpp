#include "retopo.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <set>
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
             + result.edges.size() * sizeof(uint32_t) * 3 + result.seams.size() * sizeof(uint32_t);
    for (const RetopoStroke& k : strokes) b += k.pts.size() * (2 * sizeof(Vec3) + sizeof(float)) + 128;
    return b;
}

namespace {
// Plenty for a session of retopo; the byte cap keeps a huge one inside the web build's heap.
// ---- Reach: "is this meant to touch that?" -------------------------------------
// Every tolerance that decides whether two pieces join, weld or snap, in one place.
// Screen reaches are pixels; world reaches are fractions of the local edge or quad, so
// they scale with the topology, not the sculpt. (Draw's snap, kRetopoDrawSnapPx, is in
// retopo.h: the overlay needs it too.)
//   Screen snaps (Cut's vertex ends, edge picks, loop/path clicks): the same reach as
//   Draw's lines. Cut used a tighter 8 px "to be deliberate" until 2026-10-25; the user
//   found that too fiddly and asked for one reach everywhere.
constexpr float kSnapPx = kRetopoDrawSnapPx;
//   Edits (Grab, Smooth): a moved outline vertex welds to another outline vertex this
//   close, as a share of its shortest edge: a deliberate push, not a near miss.
constexpr float kEditWeldFrac = 0.4f;
//   Wand lasso: vertices closer than this share of their edges' mean length are one
//   that missed its weld (hand merges, zips a hair off): the user's slivers sat at ~0.15.
constexpr float kLassoMergeFrac = 0.35f;
//   Build, side zip: a strip's side vertex takes an island outline vertex within this
//   share of the strip's own quad there (across or along, whichever is bigger).
constexpr float kZipReach = 0.75f;
//   Build, end join: an island outline run within this many quad sizes of a strip's end.
constexpr float kEndJoinReach = 2.0f;
//   Two directions count as running alongside within ~45 degrees (|cos| above this):
//   an end join refuses a run that parallel to the strip, alongside wants one.
constexpr float kParallelCos = 0.7f;
//   A bridge line is a strip beside an island when this share of it runs alongside an
//   outline edge, within kAlongsideReach half-widths of the band.
constexpr float kAlongsideShare = 0.6f;
constexpr float kAlongsideReach = 2.0f;
//   Notch legs keep going while each next outline edge heads out within ~60 degrees.
constexpr float kNotchLegCos = 0.5f;
//   Gap-joining lines (notch, edge to vertex) may lie over faces for this share at most.
constexpr float kOpenSpaceOver = 0.3f;
//   Mirror on: a line runs down the seam (and is forced onto it, symmetric) when its
//   longest stretch within one band width of x = 0 holds this share of its length, and
//   laid flat on x = 0 that stretch keeps kSeamFlatShare of its length (a short stroke
//   across the seam flattens to nothing, so it stays a mouth).
constexpr float kSeamLineShare = 0.6f;
constexpr float kSeamFlatShare = 0.5f;

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
    rev++;
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
                   && a.mask == result.mask && a.seams == result.seams
                   && std::equal(a.v.begin(), a.v.end(), result.v.begin(),
                                 [](const Vec3& l, const Vec3& r) { return l.x == r.x && l.y == r.y && l.z == r.z; });
    if (!same) {
        last_tag = 0;
        rev++;
        push_capped(undo_stack, std::move(pending));
        redo_stack.clear();
    }
    pending = RetopoSnapshot{};
}

bool RetopoState::undo() {
    if (undo_stack.empty()) return false;
    last_tag = 0;
    loop_edges.clear();
    rev++;
    push_capped(redo_stack, snapshot_of(*this));
    restore(*this, std::move(undo_stack.back()));
    undo_stack.pop_back();
    return true;
}

bool RetopoState::redo() {
    if (redo_stack.empty()) return false;
    last_tag = 0;
    loop_edges.clear();
    rev++;
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

// Whether some face already runs a -> b. New faces on an outline go the other way round
// (a shared side runs opposite ways in its two faces), so their normals agree.
bool has_side(const RetopoMesh& rm, uint32_t a, uint32_t b) {
    std::vector<uint32_t> c;
    for (size_t f = 0; f < rm.face_count(); f++) {
        const int k = face_corners(rm, f, c);
        for (int i = 0; i < k; i++) if (c[i] == a && c[(i + 1) % k] == b) return true;
    }
    return false;
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
// a mouth) lends the vertex sitting at the reflection, if there is one. seam_j: a strip
// run down the seam (seam_strip) has that rail on x = 0, its own twin like a seam spoke.
void emit_twin(RetopoMesh& rm, const Grid& G, int i0, int i1, const std::vector<uint32_t>& idx,
               uint32_t first_new, uint32_t owner, bool seam0, bool seam1, int seam_j) {
    std::vector<uint32_t> tw(idx.size(), ~0u);
    for (int i = i0; i <= i1; i++)
        for (int j = 0; j < G.W; j++) {
            const size_t k = (size_t)i * G.W + j;
            const uint32_t a = idx[k];
            // A half-mouth's end spoke lies on x = 0: it is its own twin, which is
            // what welds the two halves into one ring.
            if (a >= first_new && ((seam0 && i == i0) || (seam1 && i == i1) || j == seam_j)) {
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

// Mirror on, a line drawn down the seam rather than across it: it gets forced onto x = 0
// (2026-10-25: a strip down the middle wandered over the seam, hooked off into -x as a
// J, and was built as drawn). reach[i]: how far from x = 0 point i may stray. The
// longest stretch within reach (the one from point 0, if from_start) must hold
// kSeamLineShare of the line and, flat on x = 0, keep kSeamFlatShare of its length.
// keep: that stretch's points, a quarter reach apart at least, cut where it first turns
// back on itself once flat (a hook's curl would fold the strip over).
bool seam_line(const std::vector<Vec3>& pts, const std::vector<float>& reach, bool from_start,
               std::vector<size_t>& keep) {
    const size_t n = pts.size();
    if (n < 2) return false;
    float total = 0.0f;
    for (size_t i = 0; i + 1 < n; i++) total += (pts[i + 1] - pts[i]).length();
    auto near = [&](size_t i) { return std::fabs(pts[i].x) <= reach[i]; };
    size_t a = 0, b = 0;
    float best = -1.0f;
    for (size_t i = 0; i < n; ) {
        if (!near(i)) { if (from_start) break; i++; continue; }
        size_t e = i;
        float l = 0.0f;
        while (e + 1 < n && near(e + 1)) { l += (pts[e + 1] - pts[e]).length(); e++; }
        if (l > best) { best = l; a = i; b = e; }
        if (from_start) break;
        i = e + 1;
    }
    if (best <= 0.0f || best < kSeamLineShare * total) return false;
    auto flat = [&](size_t i) { Vec3 p = pts[i]; p.x = 0.0f; return p; };
    keep.assign(1, a);
    Vec3 heading(0, 0, 0);
    float flat_len = 0.0f;
    for (size_t i = a + 1; i <= b; i++) {
        const Vec3 d = flat(i) - flat(keep.back());
        if (d.length() < 0.25f * reach[i]) continue;
        if (heading.dot(d) < 0.0f) break;
        heading = d;
        flat_len += d.length();
        keep.push_back(i);
    }
    float raw = 0.0f;
    for (size_t i = a; i < keep.back(); i++) raw += (pts[i + 1] - pts[i]).length();
    return keep.size() >= 2 && flat_len >= kSeamFlatShare * raw;
}

// A strip drawn down the seam, mirror on: built as a half from x = 0 out to the side
// most of it was drawn on, plus its twin, so the seam is a rail of vertices (as across a
// mouth) and the quads keep the size drawn: two across where one was drawn.
bool seam_strip(const RetopoStroke& s, RetopoStroke& out) {
    if (s.closed || s.pts.size() < 3) return false;
    std::vector<float> reach(s.pts.size());
    for (size_t i = 0; i < reach.size(); i++) reach[i] = 2.0f * s.half_w[i] * s.width;
    std::vector<size_t> keep;
    if (!seam_line(s.pts, reach, false, keep)) return false;
    float side = 0.0f;
    for (size_t i : keep) side += s.pts[i].x;
    const float sg = side >= 0.0f ? 1.0f : -1.0f;
    out = s;
    out.pts.clear(); out.nrm.clear(); out.half_w.clear();
    for (size_t i : keep) {
        Vec3 p = s.pts[i];
        p.x = sg * s.half_w[i] * s.width;   // the half's centre: one rail on x = 0
        out.pts.push_back(p);
        out.nrm.push_back(s.nrm[i]);
        out.half_w.push_back(s.half_w[i]);
    }
    return out.pts.size() >= 3;
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
    const float lim = kEndJoinReach * std::max(half, along);
    if (best > (float)W * lim * lim) return false;
    // So does one that runs along the strip rather than across its end (a strip drawn
    // beside an island starts next to it): that's join_side's, and taking it here laid
    // the end spoke down the island's edge.
    if (W >= 2) {
        const Vec3 run = rm.v[chain[best_s + W - 1]] - rm.v[chain[best_s]];
        const Vec3 spoke = G.v[(size_t)ie * W + W - 1] - G.v[(size_t)ie * W];
        const float lr = run.length(), ls = spoke.length();
        if (lr > 1e-9f && ls > 1e-9f && std::fabs(run.dot(spoke)) < kParallelCos * lr * ls) return false;
    }
    for (int j = 0; j < W; j++)
        idx[(size_t)ie * W + j] = chain[best_s + (best_rev ? W - 1 - j : j)];
    return true;
}

// After a strip's ends joined (or a side zipped): its rows were laid evenly along the
// line as drawn, then the joined rows jumped to the islands' edges, so the quads next
// to them stretched while the middle stayed tight (user's screenshot, 2026-10-26). Per
// column, the free vertices are spaced evenly by length along that column's own path,
// between the pinned ones (joined/zipped vertices, the end rows), then snapped back
// onto the sculpt. One pass, so slider drags stay live. seam_j = a rail on x = 0.
void even_out_rows(const RetopoMesh& rm, const SnapGrid& g, Grid& G, int i0, int i1,
                   const std::vector<uint32_t>& idx, int seam_j) {
    const int W = G.W;
    if (i1 - i0 < 2) return;
    std::vector<Vec3> P;
    std::vector<float> L;
    for (int j = 0; j < W; j++) {
        auto at = [&](int i) {
            const uint32_t k = idx[(size_t)i * W + j];
            return k != ~0u ? rm.v[k] : G.v[(size_t)i * W + j];
        };
        P.clear(); L.clear();
        for (int i = i0; i <= i1; i++) {
            P.push_back(at(i));
            L.push_back(L.empty() ? 0.0f : L.back() + (P.back() - P[P.size() - 2]).length());
        }
        // Point at arc length t along P (the path as it stood, not as it's being moved).
        auto along = [&](float t) {
            size_t k = 1;
            while (k + 1 < L.size() && L[k] < t) k++;
            const float seg = L[k] - L[k - 1];
            const float u = seg > 1e-12f ? std::clamp((t - L[k - 1]) / seg, 0.0f, 1.0f) : 0.0f;
            return P[k - 1] + (P[k] - P[k - 1]) * u;
        };
        int a = i0;
        for (int b = i0 + 1; b <= i1; b++) {
            if (b < i1 && idx[(size_t)b * W + j] == ~0u) continue;
            // a and b pinned: the rows between them evenly spaced.
            for (int i = a + 1; i < b; i++) {
                const float t = L[a - i0] + (L[b - i0] - L[a - i0]) * (float)(i - a) / (float)(b - a);
                Vec3 q, qn;
                if (!g.closest(along(t), q, qn)) continue;
                if (j == seam_j) { q.x = 0.0f; qn.x = 0.0f; if (qn.length() > 1e-6f) qn = qn.normalized(); }
                G.v[(size_t)i * W + j] = q;
                G.nrm[(size_t)i * W + j] = qn;
            }
            a = b;
        }
    }
}

// Sides. A strip drawn alongside an island built before it (its band's edge on or near
// that island's outer edge) zips onto it: over the stretch where its side runs along
// the edge, the spokes are re-laid one per island edge vertex, the side vertex IS that
// vertex, and the rest of each spoke shifts across with it, fading out to the far side.
// Shared vertices and edges, the island's loops run on into the strip, no T-junction
// (the user's "parallel strips magnetise", 2026-10-24). One side per strip: zipping
// both would need the two islands' vertex counts to agree. Returns the island vertices
// taken; rm's topology must be current.
int join_side(RetopoMesh& rm, const SnapGrid& g, Grid& G, uint32_t owner, int i0, int& i1,
              std::vector<uint32_t>& idx, bool pin0, bool pin1) {
    const int W = G.W, Q = W - 1;
    if (G.closed || Q < 1 || i1 - i0 < 1) return 0;
    std::vector<uint32_t> border;
    for (uint32_t v = 0; v < (uint32_t)rm.v.size(); v++)
        if (alive(rm, v) && on_border(rm, v) && rm.v_owner[v] != owner && rm.border_nb[v * 2] != ~0u)
            border.push_back(v);
    if (border.empty()) return 0;

    // Per side, each spoke's side vertex: the island edge vertex within reach (3/4 of
    // the strip's own quad there, across or along, whichever is bigger), if any. The
    // longest unbroken run on one island wins, either side.
    int best_s = -1, best_a = 0, best_b = -1;
    std::vector<uint32_t> near((size_t)G.n), best_near;
    for (int s : {0, Q}) {
        const int sn = s == 0 ? 1 : Q - 1;
        for (int i = i0; i <= i1; i++) {
            near[i] = ~0u;
            if (idx[(size_t)i * W + s] != ~0u) continue;   // a joined end spoke's corner
            // A half-mouth's seam spoke lies on x = 0 and must stay there (it welds the
            // halves): it may only take the island's own vertex on the seam.
            const bool pinned = (pin0 && i == 0) || (pin1 && i == G.n - 1);
            const Vec3 p = G.v[(size_t)i * W + s];
            const int in = i < i1 ? i + 1 : i - 1;
            const float r = kZipReach * std::max((G.v[(size_t)i * W + sn] - p).length(),
                                             (G.v[(size_t)in * W + s] - p).length());
            float bd = r * r;
            for (uint32_t b : border) {
                if (pinned && !(rm.mirror.size() == rm.v.size() ? rm.mirror[b] == b : std::fabs(rm.v[b].x) < 1e-5f))
                    continue;
                const Vec3 d = rm.v[b] - p;
                if (d.dot(d) < bd) { bd = d.dot(d); near[i] = b; }
            }
        }
        for (int a = i0; a <= i1;) {
            if (near[a] == ~0u) { a++; continue; }
            int b = a;
            while (b + 1 <= i1 && near[b + 1] != ~0u && rm.v_owner[near[b + 1]] == rm.v_owner[near[a]]) b++;
            if (b > a && near[b] != near[a] && b - a > best_b - best_a) {
                best_s = s; best_a = a; best_b = b; best_near = near;
            }
            a = b + 1;
        }
    }
    if (best_s < 0) {
        std::printf("[retopo] zip: no stretch of either side within reach of an island's edge\n");
        return 0;
    }
    const int s = best_s, ra = best_a, rb = best_b;

    // The island's edge from the run's first vertex to its last, the short way.
    const uint32_t bA = best_near[ra], bB = best_near[rb];
    const size_t lim = (size_t)(4 * (rb - ra) + 8);
    std::vector<uint32_t> chain;
    for (int dir = 0; dir < 2; dir++) {
        std::vector<uint32_t> c{bA};
        uint32_t prev = bA, cur = rm.border_nb[bA * 2 + dir];
        while (cur != ~0u && cur != bA && c.size() <= lim) {
            c.push_back(cur);
            if (cur == bB) break;
            const uint32_t nx = rm.border_nb[cur * 2] == prev ? rm.border_nb[cur * 2 + 1] : rm.border_nb[cur * 2];
            prev = cur; cur = nx;
        }
        if (c.back() == bB && (chain.empty() || c.size() < chain.size())) chain.swap(c);
    }
    if (chain.size() < 2) {
        std::printf("[retopo] zip: side %d, spokes %d..%d near the edge, but the edge between "
                    "can't be walked (a corner where outlines touch?)\n", s, ra, rb);
        return 0;
    }
    const int m = (int)chain.size();

    // Where along the run each chain vertex sits: nearest point on the side's polyline,
    // as a spoke index with a fraction, kept in order.
    std::vector<float> t(m);
    for (int k = 0; k < m; k++) {
        const Vec3 p = rm.v[chain[k]];
        float bd = 1e30f;
        for (int i = ra; i < rb; i++) {
            const Vec3 a = G.v[(size_t)i * W + s], e = G.v[(size_t)(i + 1) * W + s] - a;
            const float l2 = e.dot(e);
            const float u = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, (p - a).dot(e) / l2)) : 0.0f;
            const Vec3 d = a + e * u - p;
            if (d.dot(d) < bd) { bd = d.dot(d); t[k] = (float)i + u; }
        }
        if (k > 0) t[k] = std::max(t[k], t[k - 1]);
    }
    t[0] = (float)ra;
    t[m - 1] = (float)rb;

    // The new spokes, spliced in for ra..rb.
    std::vector<Vec3> nv, nn;
    std::vector<uint32_t> ni;
    for (int i = 0; i < ra; i++)
        for (int j = 0; j < W; j++) {
            nv.push_back(G.v[(size_t)i * W + j]); nn.push_back(G.nrm[(size_t)i * W + j]);
            ni.push_back(idx[(size_t)i * W + j]);
        }
    for (int k = 0; k < m; k++) {
        const int a = std::min((int)t[k], rb - 1);
        const float u = t[k] - (float)a;
        auto row = [&](int j) { return G.v[(size_t)a * W + j] * (1.0f - u) + G.v[(size_t)(a + 1) * W + j] * u; };
        const Vec3 shift = rm.v[chain[k]] - row(s);
        for (int j = 0; j < W; j++) {
            if (j == s) {
                nv.push_back(rm.v[chain[k]]); nn.push_back(rm.n[chain[k]]); ni.push_back(chain[k]);
                continue;
            }
            const float w = 1.0f - (float)std::abs(j - s) / (float)Q;
            Vec3 p = row(j) + shift * w, q, qn;
            Vec3 pn = G.nrm[(size_t)a * W + j];
            if (g.closest(p, q, qn)) { p = q; pn = qn; }
            nv.push_back(p); nn.push_back(pn); ni.push_back(~0u);
        }
    }
    for (int i = rb + 1; i < G.n; i++)
        for (int j = 0; j < W; j++) {
            nv.push_back(G.v[(size_t)i * W + j]); nn.push_back(G.nrm[(size_t)i * W + j]);
            ni.push_back(idx[(size_t)i * W + j]);
        }
    G.v.swap(nv);
    G.nrm.swap(nn);
    idx.swap(ni);
    std::printf("[retopo] zip: side %d, spokes %d..%d of %d onto %d island vertices\n", s, ra, rb, G.n, m);
    G.n += m - (rb - ra + 1);
    i1 += m - (rb - ra + 1);
    return m;
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
                dst.mask.push_back(src.mask_at(i));
            }
            dst.faces.push_back(remap[i]);
        }
        dst.face_off.push_back((uint32_t)dst.faces.size());
        dst.face_owner.push_back(src.face_owner[f]);
    }
    for (uint32_t& m : dst.mirror) m = m == ~0u ? ~0u : remap[m];
    // Seams follow their vertices' new numbers (re-sorted: the order was by the old ones).
    std::vector<uint64_t> sk;
    for (size_t i = 0; i + 1 < src.seams.size(); i += 2) {
        const uint32_t a = remap[src.seams[i]], b = remap[src.seams[i + 1]];
        if (a != ~0u && b != ~0u && a != b) sk.push_back(((uint64_t)std::min(a, b) << 32) | std::max(a, b));
    }
    std::sort(sk.begin(), sk.end());
    sk.erase(std::unique(sk.begin(), sk.end()), sk.end());
    for (uint64_t k : sk) { dst.seams.push_back((uint32_t)(k >> 32)); dst.seams.push_back((uint32_t)k); }
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
    // before it. A stroke wholly on one side of x = 0 gets a twin with mirror on.
    // With mirror, one that crosses it (a mouth) is cut at x = 0 and built as a half plus
    // its twin, welded along the seam: symmetric, one piece.
    // One drawn down the seam (seam_strip) is forced onto it: a half plus its twin
    // sharing a rail on x = 0, whichever side of the seam it wandered to.
    struct Todo { RetopoStroke s; bool twin; bool seam0 = false, seam1 = false, rail = false; };
    std::vector<Todo> todo;
    for (int pass = 0; pass < 2; pass++)
        for (const RetopoStroke& s : st.strokes) {
            if (s.frozen || s.pts.size() < 3 || s.closed != (pass == 0)) continue;
            bool pos = false, neg = false;
            for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
            Todo t{s, mirror && pos != neg};
            if (mirror && seam_strip(s, t.s)) t.twin = t.rail = true;
            else if (mirror && pos && neg && half_stroke(s, t.s, t.seam0, t.seam1)) t.twin = true;
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
    int joins = 0, zips = 0;
    for (const Todo& t : todo) {
        Grid G;
        if (!stroke_grid(t.s, g, G)) continue;
        std::vector<uint32_t> idx((size_t)G.n * G.W, ~0u);
        int i0 = 0, i1 = G.n - 1;
        // A seam strip's rail: the outer column nearer x = 0, laid flat on it.
        int seam_j = -1;
        auto flat_rail = [&]() {
            if (seam_j < 0) return;
            for (int i = 0; i < G.n; i++) {
                G.v[(size_t)i * G.W + seam_j].x = 0.0f;
                Vec3& nn = G.nrm[(size_t)i * G.W + seam_j];
                nn.x = 0.0f;
                if (nn.length() > 1e-6f) nn = nn.normalized();
            }
        };
        if (t.rail) {
            float x0 = 0.0f, x1 = 0.0f;
            for (int i = 0; i < G.n; i++) {
                x0 += std::fabs(G.v[(size_t)i * G.W].x);
                x1 += std::fabs(G.v[(size_t)i * G.W + G.W - 1].x);
            }
            seam_j = x0 <= x1 ? 0 : G.W - 1;
            flat_rail();
        }
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
            const int joins0 = joins, zips0 = zips;
            if (!t.seam0) joins += join_end(rm, G, 0, t.s.id, i0, i1, idx);
            if (!t.seam1) joins += join_end(rm, G, 1, t.s.id, i0, i1, idx);
            // A half-mouth zips too (a strip across the seam drawn beside an island);
            // its seam spoke only onto the island's seam vertex, then flat on x = 0 again.
            zips += join_side(rm, g, G, t.s.id, i0, i1, idx, t.seam0, t.seam1) > 0;
            if (joins != joins0 || zips != zips0) even_out_rows(rm, g, G, i0, i1, idx, seam_j);
            for (int e = 0; e < 2; e++) {
                if (!(e ? t.seam1 : t.seam0)) continue;
                const int i = e ? G.n - 1 : 0;
                for (int j = 0; j < G.W; j++) G.v[(size_t)i * G.W + j].x = 0.0f;
            }
            flat_rail();   // a zip may have pulled it off
        }
        const uint32_t first_new = (uint32_t)rm.v.size();
        emit_grid(rm, G, i0, i1, idx, t.s.id);
        if (t.twin) emit_twin(rm, G, i0, i1, idx, first_new, t.s.id, t.seam0, t.seam1, seam_j);
    }
    dedupe_edges(rm);
    st.built = !rm.faces.empty();
    if (st.built) build_topology(rm);

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[retopo] build: %zu strokes -> %u faces, %zu verts, %d strip ends joined, %d sides zipped, %.1f ms\n",
                todo.size(), st.result.face_count(), st.result.v.size(), joins, zips, ms);
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
        const float tol = kEditWeldFrac * emin;
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
// Per vertex, its side of x = 0 at pen-down (+1 / -1, 0 = a seam vertex). With mirror,
// Grab and Smooth never carry a vertex across: it stops on x = 0 (user's rule,
// 2026-10-26), so a half can't fold over its twin. Coincident pairs left on the seam
// weld in after_edit when they're outline vertices; anything else is a hand fix.
std::vector<int8_t>   g_side;

void hold_side(Vec3& q, uint32_t i) {
    if (i >= g_side.size()) return;
    if ((g_side[i] > 0 && q.x < 0.0f) || (g_side[i] < 0 && q.x > 0.0f)) q.x = 0.0f;
}

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

// Smooth's step for vertex i, read off `old` (last pass's positions, Jacobi): the way
// it slides toward even spacing with its neighbours, along the surface. Paste's
// clean-up pass uses it too.
Vec3 relax_dir(const RetopoMesh& rm, const std::vector<Vec3>& old, uint32_t i) {
    // Slide p along the line through a and b toward their midpoint. Only the part along
    // the line counts: on a curved loop the midpoint sits inside the curve, and taking
    // that pull too is what shrinks loops.
    auto slide = [](Vec3 p, Vec3 a, Vec3 b) {
        const Vec3 t = (b - a).normalized();
        return t * ((a + b) * 0.5f - p).dot(t);
    };
    const Vec3 p = old[i];
    const uint32_t k0 = rm.nb_off[i], k1 = rm.nb_off[i + 1];
    Vec3 d;
    if (rm.border_nb[i * 2] != ~0u) {
        // Border: slide along the border line only, so widths and holes hold.
        d = slide(p, old[rm.border_nb[i * 2]], old[rm.border_nb[i * 2 + 1]]);
    } else {
        // A grid vertex (four neighbours) has two straight-through pairs, one along
        // its loop and one across the band: slide along each. Averaging all four
        // instead pulled every loop toward the inside of its bend, so a donut's
        // middle loops crept into the hole; even spacing is where this one rests.
        // Pairs are found by direction (most opposite), so welds can't confuse them.
        int pa = -1;
        if (k1 - k0 == 4) {
            Vec3 u[4];
            for (int k = 0; k < 4; k++) u[k] = (old[rm.nb[k0 + k]] - p).normalized();
            float best = -0.3f;   // not opposite enough: not a grid vertex, average instead
            for (int k = 1; k < 4; k++)
                if (u[0].dot(u[k]) < best) { best = u[0].dot(u[k]); pa = k; }
        }
        if (pa > 0) {
            const int r0 = pa == 1 ? 2 : 1, r1 = pa == 3 ? 2 : 3;
            auto N = [&](int k) { return old[rm.nb[k0 + k]]; };
            d = slide(p, N(0), N(pa)) + slide(p, N(r0), N(r1));
        } else {
            // Anything else (a weld's odd valence): toward the neighbours' average.
            Vec3 avg;
            for (uint32_t k = k0; k < k1; k++) avg += old[rm.nb[k]];
            d = avg * (1.0f / (float)std::max(1u, k1 - k0)) - p;
        }
        // Tangential only: along the normal would sink the patch into the sculpt.
        d = d - rm.n[i] * d.dot(rm.n[i]);
    }
    return d;
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
    // A vertex already on x = 0 that isn't its own twin (one stopped there by an
    // earlier stroke) belongs to the side its neighbours are on.
    g_side.assign(rm.v.size(), 0);
    const bool topo = rm.nb_off.size() == rm.v.size() + 1;
    for (uint32_t i = 0; i < (uint32_t)rm.v.size(); i++) {
        float x = rm.v[i].x;
        if (std::fabs(x) <= 1e-6f && i < rm.mirror.size() && rm.mirror[i] != i && topo) {
            x = 0.0f;
            for (uint32_t k = rm.nb_off[i]; k < rm.nb_off[i + 1]; k++) x += rm.v[rm.nb[k]].x;
        }
        g_side[i] = x > 1e-6f ? 1 : x < -1e-6f ? -1 : 0;
    }
}

int retopo_smooth_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float cx, float cy, float radius_px, float strength, bool mirror) {
    RetopoMesh& rm = st.result;
    if (!st.built || g_smooth_grid.m == nullptr || g_smooth_w.size() != rm.v.size()) return 0;
    const float r2 = radius_px * radius_px;
    int moved = 0;
    for (size_t i = 0; i < rm.v.size(); i++) {
        const float w = ring_weight(cam, win_w, win_h, cx, cy, r2, rm.v[i], rm.n[i]);
        g_smooth_w[i] = rm.pinned[i] || st.hidden(rm.v_owner[i]) ? 0.0f : w * strength * (1.0f - rm.mask_at((uint32_t)i));
        moved += g_smooth_w[i] > 0.0f;
    }
    if (!moved) return 0;
    g_smooth_old = rm.v;
    for (size_t i = 0; i < rm.v.size(); i++) {
        const float w = g_smooth_w[i];
        if (w <= 0.0f || !alive(rm, (uint32_t)i)) continue;
        const Vec3 p = g_smooth_old[i];
        const Vec3 d = relax_dir(rm, g_smooth_old, (uint32_t)i);
        Vec3 q, qn;
        if (g_smooth_grid.closest(p + d * std::min(1.0f, w), q, qn)) {
            if (mirror) hold_side(q, (uint32_t)i);
            rm.v[i] = q; rm.n[i] = qn;
        }
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
        const float w = ring_weight(cam, win_w, win_h, cx, cy, r2, rm.v[i], rm.n[i])
                      * (1.0f - rm.mask_at((uint32_t)i));
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
        if (g_smooth_grid.closest(p0 + d, q, qn, 8)) {
            if (g_grab_mirror) hold_side(q, i);
            rm.v[i] = q; rm.n[i] = qn;
        }
    }
    if (dx_px == 0.0f && dy_px == 0.0f) return;
    if (g_grab_in.size() != rm.v.size()) g_grab_in.resize(rm.v.size(), 0);   // a weld can't add, but be safe
    static std::vector<uint32_t> both;
    if (g_grab_mirror) mirror_pairs(rm, g_grab_idx, g_grab_in, both);
    after_edit(st, g_grab_mirror ? both : g_grab_idx, g_grab_mirror);
}

// ---- Mask -----------------------------------------------------------------------

namespace {
// The sculpt Mask brush's falloff (BrushStroke::falloff): flat out to an inner radius
// set by hardness, then a smoothstep to zero at the rim. Screen pixels here.
float mask_falloff(float dist, float radius, float hardness) {
    if (dist >= radius) return 0.0f;
    const float t = dist / radius, inner = 0.15f + hardness * 0.55f;
    if (t <= inner) return 1.0f;
    float b = (t - inner) / (1.0f - inner + 1e-6f);
    b = b * b * (3.0f - 2.0f * b);
    return 1.0f - b;
}

// Per vertex: its falloff under the ring (0 when it faces away, is hidden or off screen).
void mask_weights(const RetopoState& st, const Camera& cam, int win_w, int win_h, float cx, float cy,
                  float radius_px, float hardness, std::vector<float>& w) {
    const RetopoMesh& rm = st.result;
    w.assign(rm.v.size(), 0.0f);
    for (size_t i = 0; i < rm.v.size(); i++) {
        if (!alive(rm, (uint32_t)i) || st.hidden(rm.v_owner[i])) continue;
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
        float x, y;
        if (rm.n[i].dot(to_eye) <= 0.0f || !cam.world_to_screen(rm.v[i], win_w, win_h, x, y)) continue;
        w[i] = mask_falloff(std::hypot(x - cx, y - cy), radius_px, hardness);
    }
}

// After a mask edit: twins take the value (mirror), and the islands touched are frozen
// so Build keeps their vertices (and with them the mask).
void mask_done(RetopoState& st, const std::vector<uint32_t>& changed, bool mirror) {
    RetopoMesh& rm = st.result;
    std::vector<uint32_t> touched;
    for (uint32_t i : changed) {
        touched.push_back(rm.v_owner[i]);
        const uint32_t t = mirror && i < rm.mirror.size() ? rm.mirror[i] : ~0u;
        if (t < rm.v.size() && t != i) { rm.mask[t] = rm.mask[i]; touched.push_back(rm.v_owner[t]); }
    }
    freeze_touched(st, touched);
}
}  // namespace

int retopo_mask_dab(RetopoState& st, const Camera& cam, int win_w, int win_h, float cx, float cy,
                    float radius_px, float strength, float hardness, bool unmask, bool mirror) {
    RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return 0;
    rm.mask.resize(rm.v.size(), 0.0f);
    static std::vector<float> w;
    mask_weights(st, cam, win_w, win_h, cx, cy, radius_px, hardness, w);
    static std::vector<uint32_t> changed;
    changed.clear();
    for (size_t i = 0; i < rm.v.size(); i++) {
        if (w[i] <= 0.0f) continue;
        const float nv = std::max(0.0f, std::min(1.0f, rm.mask[i] + (unmask ? -strength : strength) * w[i]));
        if (nv == rm.mask[i]) continue;
        rm.mask[i] = nv;
        changed.push_back((uint32_t)i);
    }
    mask_done(st, changed, mirror);
    return (int)changed.size();
}

int retopo_mask_smooth_dab(RetopoState& st, const Camera& cam, int win_w, int win_h, float cx, float cy,
                           float radius_px, float strength, float hardness, bool mirror) {
    RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return 0;
    rm.mask.resize(rm.v.size(), 0.0f);
    static std::vector<float> w, old;
    mask_weights(st, cam, win_w, win_h, cx, cy, radius_px, hardness, w);
    old = rm.mask;   // every vertex reads its neighbours as they were
    static std::vector<uint32_t> changed;
    changed.clear();
    for (size_t i = 0; i < rm.v.size(); i++) {
        const uint32_t k0 = rm.nb_off[i], k1 = rm.nb_off[i + 1];
        if (w[i] <= 0.0f || k1 == k0) continue;
        float avg = 0.0f;
        for (uint32_t k = k0; k < k1; k++) avg += old[rm.nb[k]];
        avg /= (float)(k1 - k0);
        const float a = std::min(1.0f, strength * w[i]);
        const float nv = std::max(0.0f, std::min(1.0f, old[i] + (avg - old[i]) * a));
        if (nv == rm.mask[i]) continue;
        rm.mask[i] = nv;
        changed.push_back((uint32_t)i);
    }
    mask_done(st, changed, mirror);
    return (int)changed.size();
}

void retopo_mask_invert(RetopoState& st) {
    RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return;
    st.checkpoint();
    rm.mask.resize(rm.v.size(), 0.0f);
    std::vector<uint32_t> all;
    for (uint32_t i = 0; i < (uint32_t)rm.v.size(); i++) {
        rm.mask[i] = 1.0f - rm.mask[i];
        if (alive(rm, i)) all.push_back(i);
    }
    mask_done(st, all, false);
}

void retopo_mask_clear(RetopoState& st) {
    RetopoMesh& rm = st.result;
    if (rm.mask.empty()) return;
    st.checkpoint();
    rm.mask.clear();
}

// ---- Cut ------------------------------------------------------------------------

namespace {

// How much of a crossing's offset from its edge's middle survives the aim. A hand-drawn
// line wanders; the loop it cuts is wanted straight, so 1/5 along an edge lands at 0.43.
constexpr float kCutAimKeep = 0.25f;

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
                            float x, float y, float px) {
    const RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return ~0u;
    uint32_t best = ~0u;
    if (px <= 0.0f) px = kSnapPx;
    float best_d2 = px * px;
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

namespace {
// The edge under the pointer: both ends facing the eye and on screen, nearest in pixels.
bool edge_under(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y,
                uint32_t& ea, uint32_t& eb, float px = kSnapPx) {
    const RetopoMesh& rm = st.result;
    ea = eb = ~0u;
    float best = px;
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
    return ea != ~0u;
}
}  // namespace

int retopo_select_loop(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool add, bool mirror) {
    if (!add) st.loop_edges.clear();
    st.path_a = st.path_b = ~0u;
    const RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    if (!st.built || rm.nb_off.size() != nv + 1) return 0;

    uint32_t ea, eb;
    if (!edge_under(st, cam, win_w, win_h, x, y, ea, eb)) return 0;

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
    if (!had) {
        keep.insert(keep.end(), loop.begin(), loop.end());
        st.path_a = ea; st.path_b = eb;   // a Shift+Ctrl+click runs on from here
    }
    st.loop_edges.swap(keep);
    const bool outline = vf_off[ea + 1] - vf_off[ea] > 0 && [&] {
        bool q; around(ea, sides, q); return faces_on(eb, sides) == 1;
    }();
    std::printf("[retopo] loop: %s %zu edge(s), %s, %s; %zu picked in all\n", had ? "dropped" : "picked",
                loop.size() / 2, outline ? "outline" : "inner", closed ? "closed" : "open",
                st.loop_edges.size() / 2);
    return (int)(loop.size() / 2);
}

int retopo_select_path(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool mirror) {
    const RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    if (!st.built || rm.nb_off.size() != nv + 1) return 0;
    uint32_t ta, tb;
    if (!edge_under(st, cam, win_w, win_h, x, y, ta, tb)) return 0;

    std::vector<uint64_t> have;
    for (size_t i = 0; i + 1 < st.loop_edges.size(); i += 2)
        have.push_back(edge_key(st.loop_edges[i], st.loop_edges[i + 1]));
    std::sort(have.begin(), have.end());
    const uint32_t pa = st.path_a, pb = st.path_b;
    const bool from = pa < nv && pb < nv && std::binary_search(have.begin(), have.end(), edge_key(pa, pb));
    if (from && edge_key(pa, pb) == edge_key(ta, tb)) return 0;
    // No run to carry on (nothing picked, or what's picked came from Extrude or a loop
    // dropped): a fresh pick. Joining the old one gave Extrude a branched or wrong chain.
    if (!from) { st.loop_edges.clear(); have.clear(); }

    std::vector<uint32_t> run;   // 2 per edge
    const char* how = "first edge";
    // Both ends on one outline: along it, not across the faces. A run that cuts through
    // the inside isn't an outline chain, and Extrude needs one.
    int r_from = -1, r_to = -1;
    uint32_t k_from = 0, k_to = 0;
    for (uint32_t r = 0; from && r < rm.rim_count(); r++) {
        const uint32_t o = rm.rim_off[r], n = rm.rim_off[r + 1] - o;
        for (uint32_t k = 0; k < n; k++) {
            const uint64_t key = edge_key(rm.rim[o + k], rm.rim[o + (k + 1) % n]);
            if (key == edge_key(pa, pb)) { r_from = (int)r; k_from = k; }
            if (key == edge_key(ta, tb)) { r_to = (int)r; k_to = k; }
        }
    }
    if (!from) {
        run = {ta, tb};
    } else if (r_from >= 0 && r_from == r_to) {
        // Rim side k runs rim[k] -> rim[k + 1]. Forward or back from the start side to
        // the clicked one, whichever is shorter on the surface.
        const uint32_t o = rm.rim_off[r_from], n = rm.rim_off[r_from + 1] - o;
        auto side_len = [&](uint32_t k) { return (rm.v[rm.rim[o + (k + 1) % n]] - rm.v[rm.rim[o + k % n]]).length(); };
        const uint32_t n_fwd = (k_to + n - k_from) % n, n_bwd = (k_from + n - k_to) % n;
        float fwd = 0.0f, bwd = 0.0f;
        for (uint32_t i = 1; i <= n_fwd; i++) fwd += side_len(k_from + i);
        for (uint32_t i = 1; i <= n_bwd; i++) bwd += side_len(k_from + n - i);
        for (uint32_t i = 1; i <= (fwd <= bwd ? n_fwd : n_bwd); i++) {
            const uint32_t k = fwd <= bwd ? k_from + i : k_from + n - i;
            run.push_back(rm.rim[o + k % n]);
            run.push_back(rm.rim[o + (k + 1) % n]);
        }
        how = "along the outline";
    } else {
        // Anywhere else: the shortest way over the edges (Dijkstra by length), from
        // either end of the start edge to either end of the clicked one.
        std::vector<float> dist(nv, 1e30f);
        std::vector<uint32_t> prev(nv, ~0u);
        using QE = std::pair<float, uint32_t>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
        dist[pa] = dist[pb] = 0.0f;
        q.push({0.0f, pa});
        q.push({0.0f, pb});
        uint32_t end = ~0u;
        while (!q.empty()) {
            const auto [d, u] = q.top();
            q.pop();
            if (d > dist[u]) continue;
            if (u == ta || u == tb) { end = u; break; }
            for (uint32_t k = rm.nb_off[u]; k < rm.nb_off[u + 1]; k++) {
                const uint32_t w = rm.nb[k];
                if (st.hidden(rm.v_owner[w])) continue;
                const float nd = d + (rm.v[w] - rm.v[u]).length();
                if (nd < dist[w]) { dist[w] = nd; prev[w] = u; q.push({nd, w}); }
            }
        }
        if (end == ~0u) {
            std::printf("[retopo] path: no edges join those two\n");
            return -1;
        }
        for (uint32_t u = end; prev[u] != ~0u; u = prev[u]) { run.push_back(prev[u]); run.push_back(u); }
        run.push_back(ta);
        run.push_back(tb);
        how = "across";
    }
    if (mirror && rm.mirror.size() == nv) {
        const size_t n = run.size();
        for (size_t i = 0; i + 1 < n; i += 2) {
            const uint32_t a = rm.mirror[run[i]], b = rm.mirror[run[i + 1]];
            if (a < nv && b < nv && a != b) { run.push_back(a); run.push_back(b); }
        }
    }
    int added = 0;
    for (size_t i = 0; i + 1 < run.size(); i += 2) {
        const uint64_t k = edge_key(run[i], run[i + 1]);
        const auto at = std::lower_bound(have.begin(), have.end(), k);
        if (at != have.end() && *at == k) continue;
        have.insert(at, k);
        st.loop_edges.push_back(run[i]);
        st.loop_edges.push_back(run[i + 1]);
        added++;
    }
    st.path_a = ta; st.path_b = tb;
    std::printf("[retopo] path: %d edge(s) added, %s; %zu picked in all\n", added, how,
                st.loop_edges.size() / 2);
    return added;
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
              && !st.hidden(rm.v_owner[i]) && rm.mask_at((uint32_t)i) < 0.5f;   // masked: leave it
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
    for (uint32_t k = rm.face_off[f]; k < rm.face_off[f + 1]; k++)
        if (rm.mask_at(rm.faces[k]) >= 0.5f) return 0;   // a masked corner: the face stays
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

// ---- Dissolve (Del on picked edges) ------------------------------------------------

int retopo_audit(const RetopoState& st) {
    const RetopoMesh& rm = st.result;
    // Every side, both ways round: (lo, hi, forward?, face).
    struct Side { uint32_t lo, hi, fwd, f; };
    std::vector<Side> sides;
    std::vector<uint32_t> c;
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n && n >= 2; k++) {
            const uint32_t a = c[k], b = c[(k + 1) % n];
            if (a != b) sides.push_back({std::min(a, b), std::max(a, b), a < b ? 1u : 0u, f});
        }
    }
    std::sort(sides.begin(), sides.end(), [](const Side& x, const Side& y) {
        return x.lo != y.lo ? x.lo < y.lo : x.hi != y.hi ? x.hi < y.hi : x.fwd < y.fwd;
    });
    int same_way = 0, crowded = 0;
    std::vector<uint32_t> bad_faces;
    for (size_t i = 0; i < sides.size(); ) {
        size_t j = i;
        while (j < sides.size() && sides[j].lo == sides[i].lo && sides[j].hi == sides[i].hi) j++;
        bool dup = false;
        for (size_t k = i + 1; k < j; k++) dup |= sides[k].fwd == sides[k - 1].fwd;
        if (j - i > 2) crowded++;
        else if (dup) same_way++;
        if (j - i > 2 || dup)
            for (size_t k = i; k < j; k++)
                if (std::find(bad_faces.begin(), bad_faces.end(), sides[k].f) == bad_faces.end()) bad_faces.push_back(sides[k].f);
        i = j;
    }
    if (!same_way && !crowded) return 0;
    std::printf("[retopo-audit] %d side(s) run the same way by two faces (one over the other), "
                "%d under 3+ faces; %zu face(s) involved:", same_way, crowded, bad_faces.size());
    for (size_t i = 0; i < bad_faces.size() && i < 6; i++) {
        const int n = face_corners(rm, bad_faces[i], c);
        std::printf(" [f%u:", bad_faces[i]);
        for (int k = 0; k < n; k++) std::printf(" v%u(%.2f,%.2f,%.2f)", c[k], rm.v[c[k]].x, rm.v[c[k]].y, rm.v[c[k]].z);
        std::printf("]");
    }
    std::printf("\n");
    return same_way + crowded;
}

const char* retopo_dissolve(RetopoState& st, bool mirror, int& dissolved) {
    dissolved = 0;
    RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    if (!st.built || st.loop_edges.empty()) return "Pick edges first: double-click a loop, Shift+Ctrl+click a run";

    std::vector<uint64_t> sel;
    for (size_t i = 0; i + 1 < st.loop_edges.size(); i += 2) {
        const uint32_t a = st.loop_edges[i], b = st.loop_edges[i + 1];
        if (a >= nv || b >= nv || a == b) continue;
        sel.push_back(edge_key(a, b));
        if (mirror && rm.mirror.size() == nv) {
            const uint32_t ta = rm.mirror[a], tb = rm.mirror[b];
            if (ta < nv && tb < nv && ta != tb) sel.push_back(edge_key(ta, tb));
        }
    }
    std::sort(sel.begin(), sel.end());
    sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
    auto picked = [&](uint32_t a, uint32_t b) { return std::binary_search(sel.begin(), sel.end(), edge_key(a, b)); };

    // Each picked edge's faces. Two = they merge; one = the outline, nothing to merge
    // with (refused for now, user); none = an edge that stopped existing.
    const uint32_t nf = rm.face_count();
    std::vector<uint32_t> fa(sel.size(), ~0u), fb(sel.size(), ~0u), cnt(sel.size(), 0);
    std::vector<uint32_t> c;
    for (uint32_t f = 0; f < nf; f++) {
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) {
            const uint64_t key = edge_key(c[k], c[(k + 1) % n]);
            const auto it = std::lower_bound(sel.begin(), sel.end(), key);
            if (it == sel.end() || *it != key) continue;
            const size_t i = (size_t)(it - sel.begin());
            (cnt[i]++ == 0 ? fa[i] : fb[i]) = f;
        }
    }
    std::vector<uint32_t> up(nf);
    for (uint32_t f = 0; f < nf; f++) up[f] = f;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    for (size_t i = 0; i < sel.size(); i++) {
        if (cnt[i] == 0) continue;
        if (cnt[i] == 1) return "Outline edges can't be dissolved: pick edges with quads on both sides";
        if (cnt[i] > 2 || fa[i] == fb[i]) return "That edge isn't between two faces: can't dissolve it";
        up[root(fa[i])] = root(fb[i]);
        dissolved++;
    }
    if (!dissolved) return "Those edges are gone: pick again";

    // Each group of faces joined across picked edges becomes one face: its sides that
    // aren't picked, walked round in the faces' own winding.
    std::vector<std::vector<uint32_t>> members(nf);
    for (uint32_t f = 0; f < nf; f++) members[root(f)].push_back(f);
    FaceList out;
    std::vector<uint32_t> touched, from, to, poly;
    int merged = 0, made = 0;
    for (uint32_t r = 0; r < nf; r++) {
        if (members[r].empty()) continue;
        if (members[r].size() == 1) {
            out.add(rm.face(r), (int)rm.face_size(r), rm.face_owner[r]);
            continue;
        }
        from.clear(); to.clear();
        for (uint32_t f : members[r]) {
            touched.push_back(rm.face_owner[f]);
            const int n = face_corners(rm, f, c);
            for (int k = 0; k < n; k++)
                if (!picked(c[k], c[(k + 1) % n])) { from.push_back(c[k]); to.push_back(c[(k + 1) % n]); }
        }
        // A side starting where another does: the merged face would pinch or ring round
        // a hole. A walk that can't go on or ends early: the faces either side are wound
        // opposite ways. Neither is one face.
        for (size_t i = 0; i < from.size(); i++)
            for (size_t j = i + 1; j < from.size(); j++)
                if (from[i] == from[j]) return "Dissolving there would make a face with a hole in it";
        poly.clear();
        uint32_t at = from.empty() ? ~0u : from[0];
        for (size_t step = 0; step < from.size(); step++) {
            const size_t i = (size_t)(std::find(from.begin(), from.end(), at) - from.begin());
            if (i == from.size()) return "The faces there are flipped against each other: can't merge them";
            poly.push_back(at);
            at = to[i];
            if (at == from[0]) break;
        }
        if (poly.size() != from.size() || poly.size() < 3)
            return "Dissolving there would make a face with a hole in it";
        out.add(poly.data(), (int)poly.size(), rm.face_owner[members[r][0]]);
        merged += (int)members[r].size();
        made++;
    }

    // The picked edges' vertices left with two edges (up and down a dissolved row, or an
    // outline vertex the row ended on) go: the two edges become one.
    std::vector<uint8_t> cand(nv, 0);
    for (uint64_t k : sel) { cand[k >> 32] = 1; cand[k & 0xFFFFFFFFu] = 1; }
    std::vector<uint64_t> near;
    for (size_t f = 0; f + 1 < out.off.size(); f++)
        for (uint32_t k = out.off[f]; k < out.off[f + 1]; k++) {
            const uint32_t a = out.faces[k], b = out.faces[k + 1 < out.off[f + 1] ? k + 1 : out.off[f]];
            if (cand[a] || cand[b]) near.push_back(edge_key(a, b));
        }
    std::sort(near.begin(), near.end());
    near.erase(std::unique(near.begin(), near.end()), near.end());
    std::vector<uint32_t> deg(nv, 0);
    for (uint64_t k : near) { deg[k >> 32]++; deg[k & 0xFFFFFFFFu]++; }
    int gone = 0;
    for (uint32_t i = 0; i < nv; i++) gone += cand[i] && deg[i] == 2;
    FaceList fin;
    for (size_t f = 0; f + 1 < out.off.size(); f++) {
        poly.clear();
        for (uint32_t k = out.off[f]; k < out.off[f + 1]; k++)
            if (!(cand[out.faces[k]] && deg[out.faces[k]] == 2)) poly.push_back(out.faces[k]);
        if (poly.size() < 3) return "Dissolving there would leave a face with under three corners";
        fin.add(poly.data(), (int)poly.size(), out.owner[f]);
    }

    st.checkpoint();   // one Ctrl+Z brings the row back
    fin.swap_into(rm);
    // The dropped vertices stay behind unreferenced, like an erase's.
    rebuild_edges(rm);
    freeze_touched(st, touched);
    std::printf("[retopo] dissolve: %d edge(s), %d faces merged into %d, %d vertices gone, now %u faces\n",
                dissolved, merged, made, gone, rm.face_count());
    return nullptr;
}

// ---- Alongside (Draw from an island, along its edge) -------------------------------

bool retopo_alongside(RetopoState& st) {
    const RetopoMesh& rm = st.result;
    RetopoStroke& L = st.live;
    const size_t np = L.pts.size();
    if (!st.built || np < 3 || L.half_w.size() != np || rm.nb_off.size() != rm.v.size() + 1) return false;

    // The outer edges, each with the way out of its one face, in the surface.
    std::unordered_map<uint64_t, uint32_t> face_of;
    std::vector<uint32_t> c;
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        if (st.hidden(rm.face_owner[f])) continue;
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) {
            const uint64_t key = edge_key(c[k], c[(k + 1) % n]);
            auto it = face_of.find(key);
            if (it == face_of.end()) face_of.emplace(key, f);
            else                     it->second = ~0u;   // two faces: not an outer edge
        }
    }
    struct Side { Vec3 a, e, out; };
    std::vector<Side> sides;
    for (const auto& [key, f] : face_of) {
        if (f == ~0u) continue;
        const uint32_t a = (uint32_t)(key >> 32), b = (uint32_t)(key & 0xFFFFFFFFu);
        Vec3 mid(0, 0, 0);
        const int n = face_corners(rm, f, c);
        for (int k = 0; k < n; k++) mid = mid + rm.v[c[k]];
        mid = mid * (1.0f / (float)n);
        const Vec3 e = rm.v[b] - rm.v[a];
        Vec3 out = (rm.n[a] + rm.n[b]).cross(e).normalized();
        if (out.dot(rm.v[a] + e * 0.5f - mid) < 0.0f) out = out * -1.0f;
        sides.push_back({rm.v[a], e, out});
    }
    if (sides.empty()) return false;

    // Each line point: the nearest outer edge, how far out of it the point sits (signed,
    // + = outside), and whether the line runs along it there: within a band's width and
    // within ~45 degrees of parallel.
    std::vector<float> off(np, 0.0f);
    std::vector<Vec3> out(np);
    std::vector<uint8_t> near(np, 0);
    float total = 0.0f, along = 0.0f;
    for (size_t i = 0; i < np; i++) {
        const Vec3 p = L.pts[i];
        float bd = 1e30f;
        const Side* best = nullptr;
        Vec3 at;
        for (const Side& s : sides) {
            const float l2 = s.e.dot(s.e);
            const float u = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, (p - s.a).dot(s.e) / l2)) : 0.0f;
            const Vec3 q = s.a + s.e * u, d = p - q;
            if (d.dot(d) < bd) { bd = d.dot(d); best = &s; at = q; }
        }
        const Vec3 t = L.pts[std::min(np - 1, i + 1)] - L.pts[i > 0 ? i - 1 : 0];
        const float seg = 0.5f * t.length();
        total += seg;
        if (!best || std::sqrt(bd) > kAlongsideReach * L.half_w[i]) continue;
        near[i] = 1;
        off[i] = (p - at).dot(best->out);
        out[i] = best->out;
        const float le = best->e.length();
        if (le > 1e-9f && seg > 1e-9f && std::fabs(t.dot(best->e)) > kParallelCos * 2.0f * seg * le) along += seg;
    }
    std::printf("[retopo] alongside: %.0f%% of the line runs along an edge%s\n",
                total > 0.0f ? 100.0f * along / total : 0.0f, along >= kAlongsideShare * total ? ": a strip beside it" : "");
    if (along < kAlongsideShare * total) return false;

    // Slide it off: the band's near edge onto the island's (centre half a band out).
    // Points that strayed from the edge carry their neighbour's shift; a light smooth
    // keeps it from kinking where the nearest edge changes.
    std::vector<Vec3> shift(np, Vec3(0, 0, 0));
    for (size_t i = 0; i < np; i++)
        if (near[i]) shift[i] = out[i] * (L.half_w[i] - off[i]);
    for (size_t i = 1; i < np; i++) if (!near[i] && near[i - 1]) { shift[i] = shift[i - 1]; near[i] = 1; }
    for (size_t i = np - 1; i-- > 0;) if (!near[i] && near[i + 1]) { shift[i] = shift[i + 1]; near[i] = 1; }
    for (int pass = 0; pass < 3; pass++) {
        std::vector<Vec3> s2 = shift;
        for (size_t i = 1; i + 1 < np; i++) s2[i] = (shift[i - 1] + shift[i] * 2.0f + shift[i + 1]) * 0.25f;
        shift.swap(s2);
    }
    for (size_t i = 0; i < np; i++) L.pts[i] = L.pts[i] + shift[i];   // Build snaps them back on
    return true;
}

// ---- Corner (a bridge line round an inside corner) --------------------------------

namespace {
// A fan piece per this much turned: a square corner takes two, round its turn.
constexpr float kCornerFanDeg = 70.0f;

// The line leaves one outline edge, turns, and comes back to another outline edge that
// meets the first at a corner, or all but meets it: two corner vertices a hair apart with
// a sliver hole between, welded here. Wanted (user, 2026-10-25): not a full-width quad
// with a square corner of its own, but a fan round the corner vertex whose outer rail
// runs through where the line turned. Off a bridge's start edge (ea, eb), the line from
// pts[first]; landing on `land` when the bridge found one, else wherever the line ends.
bool bridge_corner(RetopoState& st, const std::vector<Vec3>& pts, const std::vector<Vec3>& nrm, size_t first,
                   uint32_t ea, uint32_t eb, uint64_t land, uint32_t from_id, bool mirror) {
    RetopoMesh& rm = st.result;
    const size_t np = pts.size();
    if (np < first + 2 || rm.edges.empty()) return false;
    // Outline sides (one face only), on islands that can take a bridge.
    std::unordered_map<uint64_t, int> uses;
    std::vector<uint32_t> c;
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        if (st.hidden(rm.face_owner[f])) continue;
        const int k = face_corners(rm, f, c);
        for (int i = 0; i < k; i++) uses[edge_key(c[i], c[(i + 1) % k])]++;
    }
    float me = 0.0f;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2) me += (rm.v[rm.edges[e]] - rm.v[rm.edges[e + 1]]).length();
    me /= (float)(rm.edges.size() / 2);
    auto seg_dist = [](Vec3 p, Vec3 a, Vec3 b) {
        const Vec3 e = b - a;
        const float l2 = e.dot(e), u = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, (p - a).dot(e) / l2)) : 0.0f;
        return (p - (a + e * u)).length();
    };
    auto adjacent = [&](uint32_t a, uint32_t b) {
        for (uint32_t k = rm.nb_off[a]; k < rm.nb_off[a + 1]; k++) if (rm.nb[k] == b) return true;
        return false;
    };
    // The landing edge: an outline edge with an end at (or a weld's reach from) one of
    // the start edge's ends, and the line's end beside it.
    const Vec3 E = pts.back();
    uint32_t c0 = ~0u, o0 = ~0u, c1 = ~0u, o1 = ~0u;
    float best = 1e30f;
    const uint64_t out_key = edge_key(ea, eb);
    for (const auto& [key, n] : uses) {
        if (n != 1 || key == out_key || (land && key != land)) continue;
        const uint32_t x = (uint32_t)(key >> 32), y = (uint32_t)(key & 0xFFFFFFFFu);
        const RetopoStroke* ow = st.find(rm.v_owner[x]);
        if (!ow || !ow->frozen) continue;
        for (int side = 0; side < 2; side++) {
            const uint32_t a = side ? eb : ea, oa = side ? ea : eb;
            for (int k = 0; k < 2; k++) {
                const uint32_t b = k ? y : x, ob = k ? x : y;
                if (ob == oa || ob == a || b == oa) continue;
                if (b != a && ((rm.v[b] - rm.v[a]).length() > kLassoMergeFrac * me || adjacent(a, b))) continue;
                const float d = seg_dist(E, rm.v[b], rm.v[ob]);
                if (!land && d > 0.5f * (rm.v[ob] - rm.v[b]).length()) continue;
                if (d < best) { best = d; c0 = a; o0 = oa; c1 = b; o1 = ob; }
            }
        }
    }
    if (c0 == ~0u) return false;

    // Turned round the corner the open way: angles about it, from the start edge's far end,
    // on the side the line went. Faces already at the corner must all lie outside that.
    const Vec3 cp = c1 == c0 ? rm.v[c0] : (rm.v[c0] + rm.v[c1]) * 0.5f;
    const Vec3 cn = (rm.n[c0] + rm.n[c1]).normalized();
    auto flat = [&](Vec3 v) { v = v - cp; return v - cn * v.dot(cn); };
    const Vec3 u0 = flat(rm.v[o0]).normalized(), w0 = cn.cross(u0);
    auto ang = [&](Vec3 p) { const Vec3 d = flat(p); return std::atan2(d.dot(w0), d.dot(u0)); };
    float mean = 0.0f;
    for (size_t i = first; i < np; i++) mean += ang(pts[i]);
    const float sgn = mean >= 0.0f ? 1.0f : -1.0f;
    constexpr float kTau = 6.2831853f;
    auto turned = [&](Vec3 p) { float a = ang(p) * sgn; return a < 0.0f ? a + kTau : a; };
    const float sweep = turned(rm.v[o1]);
    const float deg = sweep * 360.0f / kTau;
    if (deg < 15.0f || deg > 200.0f) return false;
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        if (st.hidden(rm.face_owner[f])) continue;
        const int k = face_corners(rm, f, c);
        Vec3 mid(0, 0, 0);
        bool at = false;
        for (int i = 0; i < k; i++) { mid = mid + rm.v[c[i]]; at |= c[i] == c0 || c[i] == c1; }
        if (!at) continue;
        const float a = turned(mid * (1.0f / (float)k));
        if (a > 0.05f && a < sweep - 0.05f) return false;   // a face inside the turn: not open
    }

    // The outer rail: the start edge's far end, the line where it turned, the landing
    // edge's far end. One turn vertex: the line's point farthest out past the chord
    // between the far ends, away from the corner (a J's hook; the line's first stretch
    // runs between chord and corner, so plain distance picked that). None past it: the
    // point farthest from the corner. More: where it crosses each even share of the turn.
    const int pieces = deg < 50.0f ? 1 : std::max(2, std::min(4, (int)std::lround(deg / kCornerFanDeg)));
    std::vector<size_t> turn;
    if (pieces == 2) {
        const Vec3 f0 = flat(rm.v[o0]), ch = flat(rm.v[o1]) - f0;
        Vec3 away = cn.cross(ch).normalized();
        if (away.dot(flat(cp) - f0) > 0.0f) away = away * -1.0f;
        float far = 0.0f;
        size_t at = ~(size_t)0;
        for (size_t i = first; i < np; i++) {
            const float d = (flat(pts[i]) - f0).dot(away);
            if (d > far) { far = d; at = i; }
        }
        if (at == ~(size_t)0) {
            float dc = -1.0f;
            for (size_t i = first; i < np; i++)
                if ((pts[i] - cp).length() > dc) { dc = (pts[i] - cp).length(); at = i; }
        }
        turn.push_back(at);
    } else {
        size_t from = first;
        for (int j = 1; j < pieces; j++) {
            const float t = sweep * (float)j / (float)pieces;
            size_t at = ~(size_t)0;
            float bd = 1e30f;
            for (size_t i = from; i < np; i++)
                if (std::fabs(turned(pts[i]) - t) < bd) { bd = std::fabs(turned(pts[i]) - t); at = i; }
            if (at == ~(size_t)0) return false;
            turn.push_back(at);
            from = at + 1;
        }
    }

    st.checkpoint();
    const uint32_t other = rm.v_owner[o1];
    const uint32_t keep = std::min(from_id, other);
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };
    const bool twins = mirror && rm.mirror.size() == rm.v.size();
    if (c1 != c0) {
        // The near miss closes: one vertex between them, the sliver gone.
        rm.v[c0] = cp; rm.n[c0] = cn;
        const uint32_t t0 = twins ? rm.mirror[c0] : ~0u, t1 = twins ? rm.mirror[c1] : ~0u;
        weld(rm, c1, c0);
        if (t0 < rm.v.size() && t1 < rm.v.size() && t0 != c0 && t1 != c1 && t0 != t1) {
            rm.v[t0] = refl(cp); rm.n[t0] = refl(cn);
            weld(rm, t1, t0);
        }
    }
    std::vector<uint32_t> ring{o0};
    const uint32_t first_new = (uint32_t)rm.v.size();
    for (size_t i : turn) {
        rm.v.push_back(pts[i]);
        rm.n.push_back(i < nrm.size() ? nrm[i] : cn);
        rm.v_owner.push_back(keep);
        rm.mirror.push_back(~0u);
        ring.push_back((uint32_t)rm.v.size() - 1);
    }
    ring.push_back(o1);
    // Wound against the faces on the corner's two edges (a shared side runs opposite ways).
    const bool flip = has_side(rm, c0, o0);
    const uint32_t first_face = rm.face_count();
    for (size_t i = 0; i + 1 < ring.size(); i++) {
        const uint32_t t[3] = {flip ? ring[i] : ring[i + 1], c0, flip ? ring[i + 1] : ring[i]};
        rm.add_face(t, 3, keep);
    }
    // Mirror: the twin corner gets the twin fan, reflected vertex by vertex.
    bool twin = false;
    if (twins && rm.mirror[c0] < first_new && rm.mirror[o0] < first_new && rm.mirror[o1] < first_new
        && rm.mirror[o0] != o1 && rm.mirror[o0] != o0 && rm.mirror[o1] != o1) {
        bool one_side = true;
        for (size_t i : turn) one_side &= (pts[i].x > 0.0f) == (rm.v[o0].x > 0.0f);
        if (one_side) {
            auto tw = [&](uint32_t v) {
                if (v < first_new) return rm.mirror[v];
                if (rm.mirror[v] == ~0u) {
                    rm.v.push_back(refl(rm.v[v]));
                    rm.n.push_back(refl(rm.n[v]));
                    rm.v_owner.push_back(keep);
                    rm.mirror.push_back(v);
                    rm.mirror[v] = (uint32_t)rm.v.size() - 1;
                }
                return rm.mirror[v];
            };
            const uint32_t made = rm.face_count();
            for (uint32_t f = first_face; f < made; f++) {
                const uint32_t* q = rm.face(f);
                const uint32_t t[3] = {tw(q[2]), tw(q[1]), tw(q[0])};
                rm.add_face(t, 3, keep);
            }
            twin = true;
        }
    }
    // One piece now: the islands it joins are one island (the oldest keeps its name).
    if (other != from_id) {
        const uint32_t gone = keep == from_id ? other : from_id;
        for (std::vector<uint32_t>* ow : {&rm.face_owner, &rm.v_owner, &rm.edge_owner})
            for (uint32_t& o : *ow) if (o == gone) o = keep;
        st.strokes.erase(std::remove_if(st.strokes.begin(), st.strokes.end(),
                                        [&](const RetopoStroke& s) { return s.id == gone; }), st.strokes.end());
    }
    rebuild_edges(rm);
    std::printf("[retopo] bridge: round a corner, %.0f degrees, %d triangle(s)%s%s\n", deg, pieces,
                c1 != c0 ? ", its near miss welded" : "", twin ? ", twin built" : "");
    return true;
}
}  // namespace

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
            // A hair of slack: a line snapped onto an outline vertex starts or ends
            // exactly on the edge's end, and rounding mustn't lose that crossing.
            constexpr float e = 1e-3f;
            if (t < -e || t > 1.0f + e || u < -e || u > 1.0f + e) continue;
            hits.push_back({a, b, ow, (float)i + std::max(0.0f, std::min(1.0f, u))});
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
    // Round an inside corner, back onto the outline beside it: a fan, not a strip. A line
    // leaving from an outline vertex leaves both its edges at once: either may be the one.
    for (const Hit& h : hits)
        if (h.owner == from_id && h.s <= out->s + 0.01f
            && bridge_corner(st, pts, st.live.nrm, (size_t)std::floor(h.s) + 1, h.a, h.b,
                             in ? edge_key(in->a, in->b) : 0, from_id, mirror))
            return nullptr;
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
    // Samples strictly past the crossing: a line snapped onto the edge starts ON it, and
    // that first sample would make the heading run along the edge.
    for (size_t i = (size_t)std::floor(out->s) + 1; (float)i < s_end && i < np; i++) line.push_back(pts[i]);
    line.push_back(in ? (rm.v[in->a] + rm.v[in->b]) * 0.5f : at(s_end));

    // Mirror on, a line run down the seam (seam_line, reach = the strip's width) is forced
    // onto it. Off an edge across the seam: centred on x = 0, its own twin (seam 1). Off an
    // edge with one end on the seam: that end's rail runs down x = 0 and the far side gets
    // a twin (seam 2), like a band down the seam. A landing edge has to be the same kind,
    // and reached: a line cut short of it stays as drawn.
    int seam = 0;
    uint32_t seam_v = ~0u;   // seam 2: the start edge's end on x = 0
    if (mirror && rm.mirror.size() == nv) {
        auto kind = [&](uint32_t a, uint32_t b) {
            if (rm.mirror[a] == b && a != b) return 1;
            return (rm.mirror[a] == a) != (rm.mirror[b] == b) ? 2 : 0;
        };
        const int k = kind(out->a, out->b);
        std::vector<size_t> keep;
        const std::vector<float> reach(line.size(), (rm.v[out->a] - rm.v[out->b]).length());
        if (k && (!in || kind(in->a, in->b) == k) && seam_line(line, reach, true, keep)
            && (!in || keep.back() == line.size() - 1)) {
            seam = k;
            seam_v = rm.mirror[out->a] == out->a ? out->a : out->b;
            const uint32_t far = seam_v == out->a ? out->b : out->a;
            const float x = k == 1 ? 0.0f : (rm.v[far].x > 0.0f ? 0.5f : -0.5f) * reach[0];
            std::vector<Vec3> flat;
            flat.push_back(line[0]);   // the edges' own middles stay where they are
            for (size_t i = 1; i < keep.size(); i++) {
                if (in && keep[i] == line.size() - 1) break;
                Vec3 p = line[keep[i]];
                p.x = x;
                flat.push_back(p);
            }
            if (in) flat.push_back(line.back());
            line.swap(flat);
        }
    }
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
    // Headings from half a quad in, not the first sample: a sample a hair off the edge
    // points anywhere, and picked the rails at random (a bow-tie quad, 2026-10-25).
    const Vec3 h0 = (along(std::min(len, 0.5f * q)) - line[0]).normalized();
    const uint32_t LA = left_first(out->a, out->b, h0) ? out->a : out->b;
    const uint32_t RA = LA == out->a ? out->b : out->a;
    uint32_t LB = ~0u, RB = ~0u;
    if (in) {
        const Vec3 h1 = (line.back() - along(std::max(0.0f, len - 0.5f * q))).normalized();
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
    // Spokes 1..n-1 (and n for a loose end): centre on the line, across = the line's own
    // left (normal x heading, the side LA and LB were picked on), leaning toward an end
    // edge's direction only within a couple of quads of it, width eased from one end
    // edge's length to the other's. The lean used to run the whole length, flipped to
    // agree with the start edge: once the line turned back on itself (a J), that put
    // the spoke's ends on the wrong rails and the quad there twisted.
    const uint32_t keep = to_id ? std::min(from_id, to_id) : from_id;
    std::vector<uint32_t> L(n + 1), R(n + 1);
    L[0] = LA; R[0] = RA;
    if (in) { L[n] = LB; R[n] = RB; }
    const uint32_t first_new = (uint32_t)rm.v.size();
    const float step = len / (float)n;
    for (int i = 1; i <= n; i++) {
        if (i == n && in) break;
        const float f = (float)i / (float)n;
        Vec3 nn;
        const Vec3 cen = snap(along(f * len), nn);
        const Vec3 tan = (along(std::min(len, (f + 0.5f / n) * len)) - along(std::max(0.0f, (f - 0.5f / n) * len))).normalized();
        const Vec3 side = nn.cross(tan).normalized();
        const float wa = std::max(0.0f, 1.0f - f * len / (2.0f * q));
        const float wb = in ? std::max(0.0f, 1.0f - (1.0f - f) * len / (2.0f * q)) : 0.0f;
        Vec3 across = side + dA * wa + dB * wb;
        across = (across - nn * across.dot(nn)).normalized();
        const float w = wA + (wB - wA) * f;
        Vec3 sp[2], sn[2];
        for (int k = 0; k < 2; k++) {
            // A bend tighter than the strip is wide: the inside rail would step back past
            // the last spoke and fold the quad over. Pull that end in until it moves on.
            const Vec3 prev = rm.v[(k ? R : L)[i - 1]];
            float o = 0.5f * w;
            Vec3 vn, p = snap(cen + across * (k ? -o : o), vn);
            for (int it = 0; it < 6 && (p - prev).dot(tan) < 0.25f * step; it++) {
                o *= 0.6f;
                p = snap(cen + across * (k ? -o : o), vn);
            }
            sp[k] = p;
            sn[k] = vn.length() > 0.5f ? vn : nn;
        }
        // Whatever picked the sides, the rails must not cross: a spoke whose ends sit
        // nearer the other rail's last vertex is a bow-tie waiting to happen. Swap it.
        const Vec3 pl = rm.v[L[i - 1]], pr = rm.v[R[i - 1]];
        if ((sp[1] - pl).length() + (sp[0] - pr).length() < (sp[0] - pl).length() + (sp[1] - pr).length()) {
            std::swap(sp[0], sp[1]);
            std::swap(sn[0], sn[1]);
        }
        for (int k = 0; k < 2; k++) {
            rm.v.push_back(sp[k]);
            rm.n.push_back(sn[k]);
            rm.v_owner.push_back(keep);
            rm.mirror.push_back(~0u);
            (k ? R : L)[i] = (uint32_t)rm.v.size() - 1;
        }
    }
    // The landing edge's ends too: picked by heading, kept only if the rails don't cross.
    if (in) {
        const Vec3 pl = rm.v[L[n - 1]], pr = rm.v[R[n - 1]];
        if ((rm.v[R[n]] - pl).length() + (rm.v[L[n]] - pr).length()
            < (rm.v[L[n]] - pl).length() + (rm.v[R[n]] - pr).length())
            std::swap(L[n], R[n]);
    }
    // Wound against the island's face on the edge it leaves, so the bridge's normals
    // agree with it (a shared side runs opposite ways in its two faces).
    const bool flip = has_side(rm, RA, LA);
    auto quad = [&](int i, uint32_t* qd) {
        qd[0] = L[i]; qd[1] = L[i + 1]; qd[2] = R[i + 1]; qd[3] = R[i];
        if (flip) std::reverse(qd, qd + 4);
    };
    const uint32_t first_face = rm.face_count();
    for (int i = 0; i < n; i++) { uint32_t qd[4]; quad(i, qd); rm.add_face(qd, 4, keep); }

    // Mirror. A bridge from an island to its own twin is its own twin: spoke i pairs with
    // spoke n - i, evened out across the seam. Otherwise, with both ends twinned and the
    // line on one side, it gets a twin bridge, reflected vertex by vertex. One forced onto
    // the seam (above) is its own twin spoke by spoke, or gets a twin sharing its seam rail.
    int twin = 0;
    // Mirror on but no twin: why, for the log (it used to skip silently).
    const char* no_twin = mirror ? "no twin: the edge it leaves has no mirror pair" : nullptr;
    if (mirror && rm.mirror[LA] != ~0u && rm.mirror[RA] != ~0u) {
        auto refl = [](Vec3 v) { v.x = -v.x; return v; };
        const bool self = in && ((rm.mirror[LA] == LB && rm.mirror[RA] == RB) || (rm.mirror[LA] == RB && rm.mirror[RA] == LB));
        bool one_side = true;
        for (const Vec3& p : line) one_side &= (p.x > 0.0f) == (line[0].x > 0.0f);
        const int last = in ? n - 1 : n;   // spokes made here; a landing edge's are the island's
        if (seam == 2 && !self) {
            // The rail from the seam vertex lies on x = 0, its own twin; the twin bridge
            // below reflects the rest and shares it.
            const std::vector<uint32_t>& rail = seam_v == LA ? L : R;
            for (int i = 1; i <= last; i++) {
                const uint32_t x = rail[i];
                rm.v[x].x = 0.0f;
                rm.n[x].x = 0.0f;
                if (rm.n[x].length() > 1e-6f) rm.n[x] = rm.n[x].normalized();
                rm.mirror[x] = x;
            }
        }
        if (seam == 1) {
            // Centred on the seam: each spoke's ends are each other's twins.
            for (int i = 1; i <= last; i++) {
                const uint32_t x = L[i], y = R[i];
                const Vec3 mid = (rm.v[x] + refl(rm.v[y])) * 0.5f;
                rm.v[x] = mid; rm.v[y] = refl(mid);
                rm.n[y] = refl(rm.n[x]);
                rm.mirror[x] = y; rm.mirror[y] = x;
            }
            twin = 1;
        } else if (self) {
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
        } else if (seam == 2 || (one_side && (!in || (rm.mirror[LB] != ~0u && rm.mirror[RB] != ~0u)))) {
            std::vector<uint32_t> tw(rm.v.size(), ~0u);
            auto tw_of = [&](uint32_t v) {
                if (v < first_new || rm.mirror[v] == v) return rm.mirror[v];
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
        } else {
            no_twin = !one_side ? "no twin: the line crosses x = 0" : "no twin: the edge it lands on has no mirror pair";
        }
    }
    if (twin) no_twin = nullptr;

    // One piece now: the islands it joins are one island (the oldest keeps its name).
    if (to_id && to_id != from_id) {
        const uint32_t gone = keep == from_id ? to_id : from_id;
        for (std::vector<uint32_t>* ow : {&rm.face_owner, &rm.v_owner, &rm.edge_owner})
            for (uint32_t& o : *ow) if (o == gone) o = keep;
        st.strokes.erase(std::remove_if(st.strokes.begin(), st.strokes.end(),
                                        [&](const RetopoStroke& s) { return s.id == gone; }), st.strokes.end());
    }
    rebuild_edges(rm);
    std::printf("[retopo] bridge: %s, %d quads (step %.3f, width %.3f -> %.3f)%s%s%s%s\n",
                to_id ? (to_id == from_id ? "within one island" : "two islands joined") : "loose strip",
                n, q, wA, wB, twin == 1 ? ", its own twin" : twin == 2 ? ", twin built" : "",
                seam ? ", forced onto the seam" : "", no_twin ? ", " : "", no_twin ? no_twin : "");
    return nullptr;
}

// ---- Extrude -----------------------------------------------------------------------

bool retopo_on_loop(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y) {
    const RetopoMesh& rm = st.result;
    // The margin grows with the edge: a pen-down just inside the quad beside a long edge
    // means that edge. 12 px alone missed most presses on a short run of big quads, which
    // then drew a one-quad bridge instead (user, 2026-10-24).
    float best = 1e30f, best_lim = 0.0f;
    for (size_t e = 0; e + 1 < st.loop_edges.size(); e += 2) {
        const uint32_t a = st.loop_edges[e], b = st.loop_edges[e + 1];
        if (a >= rm.v.size() || b >= rm.v.size()) continue;
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[a] : cam.get_view_direction() * -1.0f;
        float ax, ay, bx, by;
        if (rm.n[a].dot(to_eye) <= 0.0f || !cam.world_to_screen(rm.v[a], win_w, win_h, ax, ay)
            || !cam.world_to_screen(rm.v[b], win_w, win_h, bx, by)) continue;
        const float ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
        const float t = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, ((x - ax) * ex + (y - ay) * ey) / l2)) : 0.0f;
        const float d = std::hypot(x - (ax + ex * t), y - (ay + ey * t));
        const float lim = std::max(1.5f * kSnapPx, 0.5f * std::sqrt(l2));
        if (d - lim < best - best_lim) { best = d; best_lim = lim; }
    }
    if (!st.loop_edges.empty())
        std::printf("[retopo] press %.0f px from the pick (counts within %.0f): %s\n", best, best_lim,
                    best < best_lim ? "extrude" : "not on it");
    return best < best_lim;
}

const char* retopo_extrude(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                           bool mirror) {
    RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    const std::vector<Vec3>& line = st.live.pts;
    if (!st.built || line.size() < 2 || st.loop_edges.empty()) return "Pick a loop first (double-click an outline)";

    // The picked edge nearest the line's start, on screen.
    float lx, ly;
    cam.world_to_screen(line.front(), win_w, win_h, lx, ly);
    size_t e0 = ~(size_t)0;
    float best = 1e30f;
    for (size_t e = 0; e + 1 < st.loop_edges.size(); e += 2) {
        const uint32_t a = st.loop_edges[e], b = st.loop_edges[e + 1];
        float ax, ay, bx, by;
        if (a >= nv || b >= nv || !cam.world_to_screen(rm.v[a], win_w, win_h, ax, ay)
            || !cam.world_to_screen(rm.v[b], win_w, win_h, bx, by)) continue;
        const float ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
        const float t = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, ((lx - ax) * ex + (ly - ay) * ey) / l2)) : 0.0f;
        const float d = std::hypot(lx - (ax + ex * t), ly - (ay + ey * t));
        if (d < best) { best = d; e0 = e; }
    }
    if (e0 == ~(size_t)0) return "Pick a loop first (double-click an outline)";

    // Its chain: the picked edges joined end to end, in order. Open or closed.
    std::vector<uint32_t> deg(nv, 0);
    std::vector<uint64_t> sel;
    for (size_t e = 0; e + 1 < st.loop_edges.size(); e += 2) {
        const uint32_t a = st.loop_edges[e], b = st.loop_edges[e + 1];
        if (a >= nv || b >= nv) continue;
        sel.push_back(edge_key(a, b));
    }
    std::sort(sel.begin(), sel.end());
    sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
    for (uint64_t k : sel) { deg[k >> 32]++; deg[k & 0xFFFFFFFFu]++; }
    auto picked = [&](uint32_t a, uint32_t b) { return std::binary_search(sel.begin(), sel.end(), edge_key(a, b)); };
    auto step_on = [&](uint32_t prev, uint32_t cur) {   // the picked neighbour of cur that isn't prev
        for (uint32_t k = rm.nb_off[cur]; k < rm.nb_off[cur + 1]; k++) {
            const uint32_t w = rm.nb[k];
            if (w != prev && picked(cur, w)) return w;
        }
        return ~0u;
    };
    const uint32_t ea = st.loop_edges[e0], eb = st.loop_edges[e0 + 1];
    if (deg[ea] > 2 || deg[eb] > 2) return "That loop branches: pick a simple one";
    std::vector<uint32_t> C{ea, eb};
    bool closed = false;
    for (uint32_t p = ea, q = eb; C.size() <= nv;) {   // forward from eb
        const uint32_t r = step_on(p, q);
        if (r == ~0u) break;
        if (deg[r] > 2) return "That loop branches: pick a simple one";
        if (r == C.front()) { closed = true; break; }
        C.push_back(r);
        p = q; q = r;
    }
    if (!closed) {
        std::vector<uint32_t> back;
        for (uint32_t p = eb, q = ea; back.size() <= nv;) {
            const uint32_t r = step_on(p, q);
            if (r == ~0u) break;
            if (deg[r] > 2) return "That loop branches: pick a simple one";
            back.push_back(r);
            p = q; q = r;
        }
        C.insert(C.begin(), back.rbegin(), back.rend());
    }
    const int n = (int)C.size();
    const int ne = closed ? n : n - 1;
    auto at = [&](int k) { return closed ? C[((k % n) + n) % n] : C[std::max(0, std::min(n - 1, k))]; };

    // Outward, per edge: across it, in the surface, away from the one face it borders.
    std::vector<uint32_t> fs, c;
    std::vector<Vec3> out_e(ne);
    for (int k = 0; k < ne; k++) {
        const uint32_t a = at(k), b = at(k + 1);
        faces_with_side(rm, a, b, fs);
        if (fs.size() != 1)
            return "Extrude grows from an outline: this loop runs between faces (double-click the island's edge)";
        const int fc = face_corners(rm, fs[0], c);
        Vec3 fcen(0, 0, 0);
        for (int i = 0; i < fc; i++) fcen = fcen + rm.v[c[i]];
        fcen = fcen * (1.0f / (float)fc);
        const Vec3 nn = (rm.n[a] + rm.n[b]).normalized();
        Vec3 o = nn.cross(rm.v[b] - rm.v[a]).normalized();
        if (o.dot((rm.v[a] + rm.v[b]) * 0.5f - fcen) < 0.0f) o = o * -1.0f;
        out_e[k] = o;
    }
    std::vector<Vec3> out(n);
    float edge = 0.0f;
    for (int k = 0; k < ne; k++) edge += (rm.v[at(k + 1)] - rm.v[at(k)]).length();
    edge /= (float)std::max(1, ne);
    for (int k = 0; k < n; k++) {
        Vec3 o(0, 0, 0);
        if (closed || k > 0)      o = o + out_e[closed ? (k + ne - 1) % ne : k - 1];
        if (closed || k < n - 1)  o = o + out_e[closed ? k % ne : k];
        const Vec3 nn = rm.n[C[k]];
        out[k] = (o - nn * o.dot(nn)).normalized();
    }

    // How far: the line's end, measured out from the loop vertex nearest its start.
    int kn = 0;
    float bd = 1e30f;
    for (int k = 0; k < n; k++) {
        const float d = (rm.v[C[k]] - line.front()).length();
        if (d < bd) { bd = d; kn = k; }
    }
    const float dist = (line.back() - rm.v[C[kn]]).dot(out[kn]);
    if (dist < 0.25f * edge) return "Draw outward, away from the faces the loop borders";
    const int rows = std::max(1, (int)std::lround(dist / std::max(1e-9f, edge)));
    const float step = dist / (float)rows;

    // Snap grid over the loop and as far out as it grows.
    SnapGrid g;
    {
        Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
        for (int k = 0; k < n; k++)
            for (const Vec3 p : {rm.v[C[k]], rm.v[C[k]] + out[k] * dist}) {
                lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
                hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
            }
        const float pad = edge * 3.0f + dist * 0.5f;
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

    // Mirror: the loop across x = 0 is its own twin (vertex k's twin at index tk[k] in
    // C); else every vertex's twin elsewhere is a twin loop that grows by reflection.
    bool self = false, twinned = false;
    std::vector<int> tk(n, -1);
    if (mirror && rm.mirror.size() == nv) {
        bool all_in = true, all_out = true;
        for (int k = 0; k < n; k++) {
            const uint32_t w = rm.mirror[C[k]];
            const auto it = w < nv ? std::find(C.begin(), C.end(), w) : C.end();
            if (it != C.end()) { tk[k] = (int)(it - C.begin()); all_out = false; }
            else { all_in = false; if (w >= nv) all_out = false; }
        }
        self = all_in;
        twinned = !self && all_out;
    }

    st.checkpoint();
    const uint32_t owner = rm.v_owner[C[0]];
    // Wound against the face on the loop: quad (prev k, prev k+1, new k+1, new k) has the
    // side C[k] -> C[k+1], which that face must run the other way.
    bool flip = false;
    faces_with_side(rm, at(0), at(1), fs);
    {
        const int fc = face_corners(rm, fs[0], c);
        for (int i = 0; i < fc; i++) if (c[i] == at(0) && c[(i + 1) % fc] == at(1)) flip = true;
    }
    const uint32_t first_new = (uint32_t)rm.v.size(), first_face = rm.face_count();
    std::vector<uint32_t> prev(C), cur(n);
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < n; k++) {
            // Walked across the surface: out a step, onto it, stretched back to the step.
            const Vec3 p = rm.v[prev[k]];
            Vec3 dir = out[k] - rm.n[prev[k]] * out[k].dot(rm.n[prev[k]]);
            if (dir.length() < 1e-9f) dir = out[k];
            Vec3 q = p + dir.normalized() * step, nn;
            for (int it = 0; it < 2; it++) {
                q = snap(q, nn);
                const Vec3 d = q - p;
                if (d.length() < 1e-9f) break;
                q = p + d.normalized() * step;
            }
            q = snap(q, nn);
            rm.v.push_back(q);
            rm.n.push_back(nn.length() > 0.5f ? nn : rm.n[prev[k]]);
            rm.v_owner.push_back(owner);
            rm.mirror.push_back(~0u);
            cur[k] = (uint32_t)rm.v.size() - 1;
        }
        if (self)
            for (int k = 0; k < n; k++) {
                const int j = tk[k];
                if (j < 0) continue;
                const uint32_t x = cur[k], y = cur[j];
                if (x == y) { rm.v[x].x = 0.0f; rm.mirror[x] = x; continue; }
                if (j < k) continue;
                const Vec3 mid = (rm.v[x] + refl(rm.v[y])) * 0.5f;
                rm.v[x] = mid; rm.v[y] = refl(mid); rm.n[y] = refl(rm.n[x]);
                rm.mirror[x] = y; rm.mirror[y] = x;
            }
        for (int k = 0; k < ne; k++) {
            const int k1 = closed ? (k + 1) % n : k + 1;
            uint32_t q[4] = {prev[k], prev[k1], cur[k1], cur[k]};
            if (flip) std::reverse(q, q + 4);
            rm.add_face(q, 4, owner);
        }
        prev = cur;
    }
    // A twin loop elsewhere: the same growth, reflected vertex by vertex.
    std::vector<uint32_t> tw_last;
    if (twinned) {
        const uint32_t end = (uint32_t)rm.v.size();
        std::vector<uint32_t> tw(end - first_new, ~0u);
        auto tw_of = [&](uint32_t v) {
            if (v < first_new) return rm.mirror[v];
            uint32_t& w = tw[v - first_new];
            if (w == ~0u) {
                w = (uint32_t)rm.v.size();
                rm.v.push_back(refl(rm.v[v]));
                rm.n.push_back(refl(rm.n[v]));
                rm.v_owner.push_back(rm.v_owner[rm.mirror[C[0]]]);
                rm.mirror.push_back(v);
                rm.mirror[v] = w;
            }
            return w;
        };
        const uint32_t nface = rm.face_count();
        for (uint32_t f = first_face; f < nface; f++) {
            const uint32_t* q = rm.face(f);
            const uint32_t t4[4] = {tw_of(q[3]), tw_of(q[2]), tw_of(q[1]), tw_of(q[0])};
            rm.add_face(t4, 4, rm.v_owner[rm.mirror[C[0]]]);
        }
        for (uint32_t v : prev) tw_last.push_back(tw_of(v));
    }

    std::vector<uint32_t> touched{owner};
    if (twinned) touched.push_back(rm.v_owner[rm.mirror[C[0]]]);
    rebuild_edges(rm);
    freeze_touched(st, touched);
    // The new outer loop (and its twin) is the pick now: the next stroke carries on.
    st.loop_edges.clear();
    for (const std::vector<uint32_t>* L : {&prev, &tw_last})
        for (int k = 0; k + 1 < (int)L->size() + (closed ? 1 : 0); k++) {
            if (L->empty()) break;
            st.loop_edges.push_back((*L)[k]);
            st.loop_edges.push_back((*L)[(k + 1) % L->size()]);
        }
    std::printf("[retopo] extrude: %s loop of %d verts, %d row(s) of %.3f (edge %.3f)%s\n",
                closed ? "closed" : "open", n, rows, step, edge,
                self ? ", its own twin" : twinned ? ", twin too" : "");
    return nullptr;
}

// ---- Fill --------------------------------------------------------------------------

namespace {
// The aim starts this far above the rim (x its size: an eyeball can bulge out past it)
// and reaches this far below (a socket deeper than its opening is wide is rare).
constexpr float kFillAbove = 0.25f, kFillBelow = 1.5f;
// A lasso grid's aim starts higher: its inside can bulge up past its outline.
constexpr float kGridAbove = 0.6f;
// Passes that slide a lasso grid's ring along itself, spacing out what its corners bunched.
constexpr int kGridRingSlides = 12;
// Rims with this many sides or more get a ring of quads inside before the grid.
constexpr int kFillRingMin = 12;

// One piece now: every island round a filled rim joins the oldest of them, and it's
// geometry (frozen) from here on.
void join_round(RetopoState& st, std::vector<uint32_t>& touched) {
    RetopoMesh& rm = st.result;
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    uint32_t keep = ~0u;
    for (uint32_t id : touched) if (st.find(id)) keep = std::min(keep, id);
    if (keep == ~0u) return;
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

// Fill's work. only_rim >= 0: that rim alone (the wand), picks ignored; record = false:
// no undo step (the wand works on a copy and takes its own step when one is kept).
int fill_rims(RetopoState& st, const Mesh& m, bool mirror, int& rims, int& odd, int only_rim, bool record,
              bool no_ring = false) {
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
    if (only_rim >= 0) {
        if ((uint32_t)only_rim >= nr) return 0;
        want[only_rim] = 1;
        picked = true;
    } else if (!st.loop_edges.empty()) {
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
    // No twin rim, but every rim vertex has its pair and the pairs' sides are all open:
    // the twin gap is there, only not a hole of its own (an outline running into it makes
    // it part of that outline; 2026-10-25, Fill filled one of a pair). Its fill is this
    // one reflected all the same. Otherwise say why, so a lopsided fill isn't a mystery.
    std::vector<uint8_t> loose(nr, 0);
    if (mirror && rm.mirror.size() == rm.v.size())
        for (uint32_t r = 0; r < nr; r++) {
            if (!want[r] || twin[r] != ~0u) continue;
            const char* why = nullptr;
            for (uint32_t k = 0; k < rim_n(r) && !why; k++) {
                const uint32_t v = rim_v(r, k), w = rm.mirror[v];
                if (w >= rm.v.size()) why = "a rim vertex has no mirror pair";
                else if (w == v || rim_of[w] == r) why = "it reaches over x = 0 but isn't its own reflection";
            }
            for (uint32_t k = 0; k < rim_n(r) && !why; k++) {
                const uint32_t a = rm.mirror[rim_v(r, k)], b = rm.mirror[rim_v(r, k + 1)];
                // Open = one face along it (border_nb keeps two neighbours, so it can't
                // tell at a vertex where an outline touches the gap).
                if (has_side(rm, a, b) == has_side(rm, b, a))
                    why = "the mirror side isn't open there (faces already, or edges not joined)";
            }
            loose[r] = !why;
            if (why && record) std::printf("[retopo] fill: no twin for the rim of %u sides: %s\n", rim_n(r), why);
        }
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

    if (record) st.checkpoint();
    std::vector<uint32_t> touched;
    std::vector<Vec3> L;
    std::vector<uint32_t> Li, Ri, c;
    int made = 0;
    for (uint32_t r = 0; r < nr; r++) {
        if (!want[r]) continue;
        const int N = (int)rim_n(r), h = N / 2;
        Ri.resize(N);
        for (int k = 0; k < N; k++) Ri[k] = rim_v(r, k);
        // Still open all round? A twin filled earlier in this pass may have covered it
        // (rims found as each other's twin only one way round were filled twice).
        bool open = true;
        for (int k = 0; k < N && open; k++) open = has_side(rm, Ri[k], Ri[(k + 1) % N]) != has_side(rm, Ri[(k + 1) % N], Ri[k]);
        if (!open) continue;
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
        const bool ring = !no_ring && N >= kFillRingMin;
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
        const bool flip = has_side(rm, Ri[0], Ri[1]);
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
        if (record) std::printf("[retopo] fill: rim of %d sides -> %s%d x %d grid, %d quads (depth %.3f of %.3f across)%s\n",
                    N, ring ? "ring + " : "", M, Nn, nq,
                    [&] { float dmax = 0; for (uint32_t x = first_new; x < rm.v.size(); x++)
                              dmax = std::max(dmax, (cen - rm.v[x]).dot(nf)); return dmax; }(), diam,
                    self ? ", its own twin" : twin[r] != ~0u ? ", twin too" : loose[r] ? ", twin too (by vertex pairs)" : "");

        // A separate twin: its fill is this one reflected, vertex by vertex (unless
        // something already closed it).
        bool twin_open = (twin[r] != ~0u || loose[r]) && !self;
        for (int k = 0; k < N && twin_open; k++) {
            const uint32_t a = rm.mirror[Ri[k]], b = rm.mirror[Ri[(k + 1) % N]];
            twin_open = has_side(rm, a, b) != has_side(rm, b, a);
        }
        if (twin_open) {
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

    join_round(st, touched);
    st.loop_edges.clear();
    rebuild_edges(rm);
    if (record)
        std::printf("[retopo] fill: %d rim(s), %d quads, %d odd rim(s) left; %u holes remain\n",
                    rims, made, odd, (uint32_t)std::count(rm.rim_hole.begin(), rm.rim_hole.end(), 1));
    return made;
}
}  // namespace

int retopo_fill(RetopoState& st, const Mesh& m, bool mirror, int& rims, int& odd) {
    return fill_rims(st, m, mirror, rims, odd, -1, true);
}

// ---- Wand ----------------------------------------------------------------------

namespace {
// The hole whose rim, on screen, holds (x, y), facing the eye; nearest by its middle.
int hole_under(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y) {
    const RetopoMesh& rm = st.result;
    int best = -1;
    float best_d = 1e30f;
    std::vector<float> sx, sy;
    for (uint32_t r = 0; r < rm.rim_count(); r++) {
        if (!rm.rim_hole[r]) continue;
        const uint32_t k0 = rm.rim_off[r], n = rm.rim_off[r + 1] - k0;
        if (n < 3) continue;
        sx.resize(n); sy.resize(n);
        bool ok = true;
        Vec3 cen(0, 0, 0), ns(0, 0, 0);
        for (uint32_t i = 0; i < n && ok; i++) {
            const uint32_t v = rm.rim[k0 + i];
            ok = !st.hidden(rm.v_owner[v]) && cam.world_to_screen(rm.v[v], win_w, win_h, sx[i], sy[i]);
            cen = cen + rm.v[v];
            ns = ns + rm.n[v];
        }
        if (!ok) continue;
        cen = cen * (1.0f / (float)n);
        const Vec3 to_eye = cam.perspective ? cam.get_position() - cen : cam.get_view_direction() * -1.0f;
        if (ns.dot(to_eye) <= 0.0f) continue;   // its twin round the back
        bool in = false;
        for (uint32_t i = 0, j = n - 1; i < n; j = i++)
            if ((sy[i] > y) != (sy[j] > y)
                && x < sx[j] + (sx[i] - sx[j]) * (y - sy[j]) / (sy[i] - sy[j])) in = !in;
        if (!in) continue;
        const float d = cam.view_depth(cen);
        if (d < best_d) { best_d = d; best = (int)r; }
    }
    return best;
}

// The rim's reflection across x = 0 as a shift: rim vertex k's twin is vertex t - k. Its
// vertices' mirror pairs when they have them; else by position (a donut drawn across the
// seam in one stroke has no pairs, only a shape that is nearly symmetric): the t whose
// reflected rim lands nearest, if every vertex lands within 3/4 of a rim edge of its twin
// (hand-drawn: the user's first test hole was off by 0.3 of an edge, a mouth twice that).
// -1 = the rim isn't its own reflection.
int rim_reflection(const RetopoMesh& rm, const std::vector<uint32_t>& R) {
    const int N = (int)R.size();
    auto md = [&](int k) { return ((k % N) + N) % N; };
    if (rm.mirror.size() == rm.v.size()) {
        int t = -1;
        for (int k = 0; k < N; k++) if (R[k] == rm.mirror[R[0]]) t = k;
        bool ok = t >= 0;
        for (int k = 0; k < N && ok; k++) ok = rm.mirror[R[k]] == R[md(t - k)];
        if (ok) return t;
    }
    float edge = 0.0f;
    for (int k = 0; k < N; k++) edge += (rm.v[R[md(k + 1)]] - rm.v[R[k]]).length();
    edge /= (float)N;
    int best = -1;
    float best_d = 0.75f * edge;
    for (int t = 0; t < N; t++) {
        float worst = 0.0f;
        for (int k = 0; k < N && worst < best_d; k++) {
            Vec3 q = rm.v[R[k]];
            q.x = -q.x;
            worst = std::max(worst, (q - rm.v[R[md(t - k)]]).length());
        }
        if (worst < best_d) { best_d = worst; best = t; }
    }
    return best;
}

// New faces over rim R, given as rim indices with every rim side in them running up the
// rim, wound against the faces already on it; the islands round it become one.
void emit_rim_faces(RetopoState& st, const std::vector<uint32_t>& R, const std::vector<std::vector<int>>& F) {
    RetopoMesh& rm = st.result;
    const bool flip = has_side(rm, R[0], R[1]);
    const uint32_t owner = rm.v_owner[R[0]];
    std::vector<uint32_t> q;
    for (const std::vector<int>& f : F) {
        q.clear();
        for (int k : f) q.push_back(R[k]);
        if (flip) std::reverse(q.begin(), q.end());
        rm.add_face(q.data(), (int)q.size(), owner);
    }
    std::vector<uint32_t> touched;
    for (uint32_t v : R) touched.push_back(rm.v_owner[v]);
    join_round(st, touched);
    rebuild_edges(rm);
}

// The two halves of a rim that is its own reflection (rim_reflection's t). In half-steps
// (vertex k at 2k, the side after it at 2k + 1) the seam crosses the rim at t and t + N:
// a vertex where that's even, a side's middle where it's odd. Half A runs up from the
// first crossing to the second, half B on round from there; each has E + 1 vertices and
// A(i)'s twin is B(E - i). False when a half is a single vertex.
struct RimHalves {
    int N = 0, a0 = 0, b0 = 0, E = 0;
    int md(int k) const { return ((k % N) + N) % N; }
    int A(int i) const { return md(a0 + i); }
    int B(int i) const { return md(b0 + i); }
    bool set(int n, int t) {
        N = n;
        a0 = (t + 1) / 2; E = (t + N) / 2 - a0; b0 = (t + N + 1) / 2;
        return E >= 1;
    }
};

// Ladders over a rim that is its own reflection (see retopo_wand_open). kind 0 = Columns,
// 1 and 2 = Columns with one half started a step on (bottom, top), 3 = Rows. Only new
// faces between the rim's own vertices. False = this kind doesn't fit (nothing done).
bool ladder_fill(RetopoState& st, const std::vector<uint32_t>& R, int kind) {
    RetopoMesh& rm = st.result;
    const int N = (int)R.size();
    if (N < 3) return false;
    const int t = rim_reflection(rm, R);
    RimHalves h;
    if (t < 0 || !h.set(N, t)) return false;
    const int E = h.E;
    auto md = [&](int k) { return h.md(k); };
    auto A = [&](int i) { return h.A(i); };
    auto B = [&](int i) { return h.B(i); };

    // Faces as rim indices, every rim side in them running up the rim.
    std::vector<std::vector<int>> F;
    if (kind == 3) {
        for (int i = 0; i < E; i++) {
            std::vector<int> f{A(i), A(i + 1), B(E - i - 1), B(E - i)};
            f.erase(std::unique(f.begin(), f.end()), f.end());
            if (f.size() > 1 && f.front() == f.back()) f.pop_back();
            if (f.size() < 3) return false;
            F.push_back(f);
        }
    } else {
        const int i0 = kind == 2 ? 1 : 0, j0 = kind == 1 ? E - 1 : E;
        if (j0 - i0 < 1) return false;
        // Rungs from (i, j) inward: the pair after each is (i + 1, j - 1); they meet at the
        // sides in a rim side (all quads) or a vertex (a triangle).
        auto ladder = [&](auto H, int i, int j) {
            for (; j - i >= 3; i++, j--) F.push_back({H(i), H(i + 1), H(j - 1), H(j)});
            if (j - i == 2) F.push_back({H(i), H(i + 1), H(j)});
        };
        ladder(A, i0, j0);
        ladder(B, E - j0, E - i0);
        // The middle: between the first rung and its twin, round both seam crossings.
        std::vector<int> P;
        auto walk = [&](int x, int y) { for (int k = x;; k = md(k + 1)) { P.push_back(k); if (k == y) break; } };
        walk(A(j0), B(E - j0));
        walk(B(E - i0), A(i0));
        P.erase(std::unique(P.begin(), P.end()), P.end());
        if (P.size() > 1 && P.front() == P.back()) P.pop_back();
        const int p = (int)P.size();
        if (p == 3 || p == 4) {
            F.push_back(P);
        } else if (p == 5 || p == 6) {
            // Split along a chord that is its own reflection: twins, or two seam vertices.
            int bx = -1, by = -1, bq = -1;
            for (int x = 0; x < p; x++)
                for (int y = x + 2; y < p; y++) {
                    const bool sym = md(t - P[x]) == P[y] || (md(t - P[x]) == P[x] && md(t - P[y]) == P[y]);
                    const int s1 = y - x + 1, s2 = p - (y - x) + 1;
                    if (!sym || s1 < 3 || s1 > 4 || s2 < 3 || s2 > 4) continue;
                    if (s1 + s2 > bq) { bq = s1 + s2; bx = x; by = y; }
                }
            if (bx < 0) return false;
            F.emplace_back(P.begin() + bx, P.begin() + by + 1);
            std::vector<int> g(P.begin() + by, P.end());
            g.insert(g.end(), P.begin(), P.begin() + bx + 1);
            F.push_back(g);
        } else if (p > 6) {
            return false;
        }
    }
    if (F.empty()) return false;
    emit_rim_faces(st, R, F);
    return true;
}

// Fewest faces: triangles and quads of the rim's own vertices, no new ones. Every way of
// cutting the hole up is scored (a dynamic program over the rim's chords, quick for the
// rims a retopo has) and the cheapest wins:
constexpr float kFaceCost = 1.0f;    // each face: fewer polygons first
constexpr float kTriCost = 0.3f;     // a triangle a little more than a quad...
constexpr float kTriMidCost = 0.3f;  // ...and more again toward the hole's middle (ends first)
constexpr float kSliverCost = 4.0f;  // a needle or flat corner, a quad bent nearly straight
// Off square: (1 - q)^2 times this, q = area against the sum of the squared sides (1 for a
// square or an equilateral triangle). Angles alone let long kites through: down the seam
// of a tall hole, two of them per side scored cheaper than rows across it.
constexpr float kShapeCost = 3.0f;
constexpr int   kFewestMaxRim = 96;  // past this the search is skipped (O(n^4) quads)

// The rim flattened onto its own plane, and what a face over it costs.
struct RimShape {
    std::vector<float> x, y;
    float ori = 1.0f, cx = 0.0f, cy = 0.0f, rmax = 1.0f;
    std::vector<uint8_t> ok;   // chord (a, b) lies inside the hole, n x n
    int n = 0;

    void set(const RetopoMesh& rm, const std::vector<uint32_t>& R) {
        n = (int)R.size();
        Vec3 nw(0, 0, 0), cen(0, 0, 0);
        for (int k = 0; k < n; k++) {
            const Vec3 a = rm.v[R[k]], b = rm.v[R[(k + 1) % n]];
            nw = nw + Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
            cen = cen + a;
        }
        cen = cen * (1.0f / (float)n);
        const Vec3 nn = nw.length() > 1e-12f ? nw.normalized() : Vec3(0, 0, 1);
        const Vec3 u = (std::fabs(nn.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0)).cross(nn).normalized();
        const Vec3 v = nn.cross(u);
        x.resize(n); y.resize(n);
        float area = 0.0f;
        for (int k = 0; k < n; k++) { const Vec3 d = rm.v[R[k]] - cen; x[k] = d.dot(u); y[k] = d.dot(v); }
        for (int k = 0; k < n; k++) area += x[k] * y[(k + 1) % n] - x[(k + 1) % n] * y[k];
        ori = area >= 0.0f ? 1.0f : -1.0f;
        rmax = 1e-9f;
        for (int k = 0; k < n; k++) rmax = std::max(rmax, std::hypot(x[k], y[k]));
        // Chords: crossing no rim side (bar the ones at its ends), middle inside.
        ok.assign(n * n, 0);
        auto cross2 = [&](float ax, float ay, float bx, float by) { return ax * by - ay * bx; };
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                bool good = true;
                if (b - a != 1 && !(a == 0 && b == n - 1)) {
                    for (int e = 0; e < n && good; e++) {
                        const int e1 = (e + 1) % n;
                        if (e == a || e == b || e1 == a || e1 == b) continue;
                        const float d1 = cross2(x[b] - x[a], y[b] - y[a], x[e] - x[a], y[e] - y[a]);
                        const float d2 = cross2(x[b] - x[a], y[b] - y[a], x[e1] - x[a], y[e1] - y[a]);
                        const float d3 = cross2(x[e1] - x[e], y[e1] - y[e], x[a] - x[e], y[a] - y[e]);
                        const float d4 = cross2(x[e1] - x[e], y[e1] - y[e], x[b] - x[e], y[b] - y[e]);
                        if ((d1 > 0) != (d2 > 0) && (d3 > 0) != (d4 > 0)) good = false;
                    }
                    if (good) {
                        const float mx = 0.5f * (x[a] + x[b]), my = 0.5f * (y[a] + y[b]);
                        bool in = false;
                        for (int i = 0, j = n - 1; i < n; j = i++)
                            if ((y[i] > my) != (y[j] > my) && mx < x[j] + (x[i] - x[j]) * (my - y[j]) / (y[i] - y[j])) in = !in;
                        good = in;
                    }
                }
                ok[a * n + b] = ok[b * n + a] = good;
            }
    }
    bool chord(int a, int b) const { return a == b || ok[a * n + b]; }

    // A face over rim indices q (in rim order): its cost, -1 if it isn't one (a corner
    // turned inward, a chord outside the hole).
    float cost(const int* q, int k) const {
        float amin = 180.0f, amax = 0.0f, area = 0.0f, ss = 0.0f;
        for (int i = 0; i < k; i++) {
            const int a = q[(i + k - 1) % k], b = q[i], c = q[(i + 1) % k];
            if (!chord(b, c)) return -1.0f;
            const float ux = x[a] - x[b], uy = y[a] - y[b], vx = x[c] - x[b], vy = y[c] - y[b];
            const float lu = std::hypot(ux, uy), lv = std::hypot(vx, vy);
            if (lu < 1e-12f || lv < 1e-12f) return -1.0f;
            // Running up the rim, the inside is to the polygon's own side of each corner.
            if ((vx * uy - vy * ux) * ori <= 1e-6f * lu * lv) return -1.0f;
            const float ang = std::acos(std::max(-1.0f, std::min(1.0f, (ux * vx + uy * vy) / (lu * lv)))) * 57.29578f;
            amin = std::min(amin, ang);
            amax = std::max(amax, ang);
            area += x[b] * y[c] - x[c] * y[b];
            ss += lv * lv;
        }
        const float qual = std::fabs(area) * 0.5f * (k == 3 ? 6.9282f : 4.0f) / std::max(1e-12f, ss);
        auto ramp = [](float v) { v = std::max(0.0f, std::min(1.0f, v)); return v * v; };
        float c = kFaceCost + kSliverCost * ramp((25.0f - amin) / 25.0f) + kShapeCost * (1.0f - qual) * (1.0f - qual);
        if (k == 3) {
            float mx = 0.0f, my = 0.0f;
            for (int i = 0; i < 3; i++) { mx += x[q[i]] / 3.0f; my += y[q[i]] / 3.0f; }
            c += kTriCost + kTriMidCost * (1.0f - std::min(1.0f, std::hypot(mx - cx, my - cy) / rmax))
               + kSliverCost * ramp((amax - 120.0f) / 50.0f);
        } else {
            c += kSliverCost * ramp((amax - 160.0f) / 20.0f);
        }
        return c;
    }
};

// The cheapest cut of the chain Q (rim indices in rim order, its two ends joined by a
// chord) into triangles and quads: best(i, j) for every piece Q[i..j].
struct ChainCut {
    int m = 0;
    std::vector<float> best;
    std::vector<int> pk, pl;   // the face on side (i, j): (i, k, j) or (i, k, l, j); l = -1 for a triangle
    float& B(int i, int j) { return best[i * m + j]; }

    void solve(const RimShape& sh, const std::vector<int>& Q) {
        m = (int)Q.size();
        best.assign(m * m, 1e30f);
        pk.assign(m * m, -1);
        pl.assign(m * m, -1);
        for (int i = 0; i + 1 < m; i++) B(i, i + 1) = 0.0f;
        for (int len = 2; len < m; len++)
            for (int i = 0; i + len < m; i++) {
                const int j = i + len;
                if (!sh.chord(Q[i], Q[j])) continue;
                float& b = B(i, j);
                for (int k = i + 1; k < j; k++) {
                    if (B(i, k) >= 1e29f || B(k, j) >= 1e29f) continue;
                    const int t3[3] = {Q[i], Q[k], Q[j]};
                    const float c3 = sh.cost(t3, 3);
                    if (c3 >= 0.0f && c3 + B(i, k) + B(k, j) < b) { b = c3 + B(i, k) + B(k, j); pk[i * m + j] = k; pl[i * m + j] = -1; }
                    for (int l = k + 1; l < j; l++) {
                        if (B(k, l) >= 1e29f || B(l, j) >= 1e29f) continue;
                        const int q4[4] = {Q[i], Q[k], Q[l], Q[j]};
                        const float c4 = sh.cost(q4, 4);
                        const float tot = c4 + B(i, k) + B(k, l) + B(l, j);
                        if (c4 >= 0.0f && tot < b) { b = tot; pk[i * m + j] = k; pl[i * m + j] = l; }
                    }
                }
            }
    }
    void faces(const std::vector<int>& Q, int i, int j, std::vector<std::vector<int>>& F) const {
        if (j - i < 2) return;
        const int k = pk[i * m + j], l = pl[i * m + j];
        if (l < 0) {
            F.push_back({Q[i], Q[k], Q[j]});
            faces(Q, i, k, F); faces(Q, k, j, F);
        } else {
            F.push_back({Q[i], Q[k], Q[l], Q[j]});
            faces(Q, i, k, F); faces(Q, k, l, F); faces(Q, l, j, F);
        }
    }
};

// Fewest faces over rim r (see above). With mirror and a rim that is its own reflection
// the cut is symmetric: down the seam a stack of faces that are their own twins (a
// triangle on a seam vertex, a quad between two rungs, each rung a vertex and its twin),
// the pockets beside them cut on one side and mirrored; or, both seam crossings being
// vertices, the seam itself as a chord and the halves mirrored. False = no cut found.
bool fewest_fill(RetopoState& st, const std::vector<uint32_t>& R, bool mirror) {
    RetopoMesh& rm = st.result;
    const int N = (int)R.size();
    if (N < 3 || N > kFewestMaxRim) return false;
    RimShape sh;
    sh.set(rm, R);
    std::vector<std::vector<int>> F;
    const int t = mirror ? rim_reflection(rm, R) : -1;
    RimHalves h;
    if (t >= 0 && h.set(N, t)) {
        const int E = h.E;
        auto tw = [&](int k) { return h.md(t - k); };
        std::vector<int> L(E + 1);
        for (int i = 0; i <= E; i++) L[i] = h.A(i);
        ChainCut cut;
        cut.solve(sh, L);
        // The face down the seam between rungs i and j (deduped: a seam vertex is its own
        // twin), its cost; -1 = none fits.
        auto mid_face = [&](int i, int j, std::vector<int>& f) {
            f = {L[i], L[j], tw(L[j]), tw(L[i])};
            f.erase(std::unique(f.begin(), f.end()), f.end());
            if (f.size() > 1 && f.front() == f.back()) f.pop_back();
            if (f.size() < 3) return -1.0f;
            return sh.cost(f.data(), (int)f.size());
        };
        std::vector<float> C(E + 1, 1e30f);
        std::vector<int> from(E + 1, -1);
        std::vector<int> f;
        C[0] = 0.0f;
        for (int j = 1; j <= E; j++)
            for (int i = 0; i < j; i++) {
                if (C[i] >= 1e29f || cut.B(i, j) >= 1e29f) continue;
                const float c = mid_face(i, j, f);
                if (c < 0.0f) continue;
                const float tot = C[i] + c + 2.0f * cut.B(i, j);
                if (tot < C[j]) { C[j] = tot; from[j] = i; }
            }
        // The seam as a chord: only with a vertex at both crossings.
        const bool split_ok = tw(L[0]) == L[0] && tw(L[E]) == L[E] && cut.B(0, E) < 1e29f;
        const float split = split_ok ? 2.0f * cut.B(0, E) : 1e30f;
        if (std::min(C[E], split) >= 1e29f) return false;
        std::vector<std::pair<int, int>> pockets;
        if (split < C[E]) {
            pockets.push_back({0, E});
        } else {
            for (int j = E; j > 0; j = from[j]) {
                mid_face(from[j], j, f);
                F.push_back(f);
                pockets.push_back({from[j], j});
            }
        }
        for (const auto& pq : pockets) {
            const size_t f0 = F.size();
            cut.faces(L, pq.first, pq.second, F);
            const size_t f1 = F.size();
            for (size_t q = f0; q < f1; q++) {   // the twin pocket: reflected, wound back up the rim
                std::vector<int> g;
                for (int k : F[q]) g.push_back(tw(k));
                std::reverse(g.begin(), g.end());
                F.push_back(g);
            }
        }
    } else {
        std::vector<int> Q(N);
        for (int k = 0; k < N; k++) Q[k] = k;
        ChainCut cut;
        cut.solve(sh, Q);
        if (cut.B(0, N - 1) >= 1e29f) return false;
        cut.faces(Q, 0, N - 1, F);
    }
    if (F.empty()) return false;
    // A hole whose twin is elsewhere and still open: the same faces there too (2026-10-25:
    // Fewest faces filled one of a mirrored pair and left the other open). Rim vertices
    // only, so the twin rim is the whole job; emit_rim_faces winds it against its island.
    std::vector<uint32_t> Rt;
    if (mirror && t < 0 && rm.mirror.size() == rm.v.size()) {
        for (uint32_t v : R) {
            const uint32_t w = rm.mirror[v];
            if (w >= rm.v.size() || w == v || std::find(R.begin(), R.end(), w) != R.end()) { Rt.clear(); break; }
            Rt.push_back(w);
        }
        for (size_t k = 0; k < Rt.size(); k++) {
            const uint32_t a = Rt[k], b = Rt[(k + 1) % Rt.size()];
            if (has_side(rm, a, b) == has_side(rm, b, a)) { Rt.clear(); break; }   // open: one face along it
        }
    }
    emit_rim_faces(st, R, F);
    if (!Rt.empty()) emit_rim_faces(st, Rt, F);
    return true;
}

// A step put back as it was: preview swaps, selection and eyes included.
void put(RetopoState& st, const RetopoSnapshot& s) {
    st.strokes = s.strokes;
    st.result = s.result;
    st.built = s.built;
    st.next_id = std::max(st.next_id, s.next_id);
}
}  // namespace

namespace {
// A drawn spine (see retopo_wand_open_spine): its two rim vertices and the line.
struct SpineLine {
    uint32_t a = ~0u, b = ~0u;
    std::vector<Vec3> pts, nrm;
};
const char* spine_fill(RetopoState& st, const std::vector<uint32_t>& R, int r, const SpineLine& sl,
                       bool mirror, bool ends_tri);

// A notch (see retopo_wand_open_notch), closed as a hole: the outline from one foot of
// its legs to the other, and a flat mouth back between the feet.
struct Notch {
    std::vector<uint32_t> run;          // outline: foot A', up to A, round to B, down to foot B'
    std::vector<Vec3> mouth, mouth_n;   // the mouth's new vertices, from B' back toward A'
    uint32_t a = ~0u, b = ~0u;          // the line's ends
    int legs = 0;                       // outline edges from A down to A' (and B to B')
    bool seam = false;                  // the feet are mirror twins: the mouth pairs across x = 0
    bool twin = false;                  // mirror, off the seam: the twin notch gets the same faces
};

// The notch's rim, its mouth vertices added to the mesh (no faces yet: a fill makes them).
std::vector<uint32_t> notch_rim(RetopoState& st, const Notch& nt) {
    RetopoMesh& rm = st.result;
    std::vector<uint32_t> R = nt.run;
    const uint32_t owner = rm.v_owner[nt.a], first = (uint32_t)rm.v.size();
    const uint32_t k = (uint32_t)nt.mouth.size();
    const bool pairs = rm.mirror.size() == rm.v.size();
    for (uint32_t j = 0; j < k; j++) {
        rm.v.push_back(nt.mouth[j]);
        rm.n.push_back(nt.mouth_n[j]);
        rm.v_owner.push_back(owner);
        if (pairs) rm.mirror.push_back(nt.seam ? first + (k - 1 - j) : ~0u);
        R.push_back(first + j);
    }
    return R;
}

// The faces from first_face on, reflected across x = 0 onto their twins: a vertex older
// than first_v by its mirror pair, a newer one by a new reflected vertex.
void reflect_new_faces(RetopoState& st, uint32_t first_v, uint32_t first_face, uint32_t tw_owner,
                       std::vector<uint32_t>& touched) {
    RetopoMesh& rm = st.result;
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };
    const uint32_t end = (uint32_t)rm.v.size();
    std::vector<uint32_t> tw(end - first_v, ~0u);
    auto tw_of = [&](uint32_t v) {
        if (v < first_v) { touched.push_back(rm.v_owner[rm.mirror[v]]); return rm.mirror[v]; }
        uint32_t& w = tw[v - first_v];
        if (w == ~0u) {
            w = (uint32_t)rm.v.size();
            rm.v.push_back(refl(rm.v[v]));
            rm.n.push_back(refl(rm.n[v]));
            rm.v_owner.push_back(tw_owner);
            rm.mirror.push_back(v);
            rm.mirror[v] = w;
        }
        return w;
    };
    const uint32_t nface = rm.face_count();
    std::vector<uint32_t> q;
    for (uint32_t f = first_face; f < nface; f++) {
        q.assign(rm.face(f), rm.face(f) + rm.face_size(f));
        for (uint32_t& v : q) v = tw_of(v);
        std::reverse(q.begin(), q.end());
        rm.add_face(q.data(), (int)q.size(), tw_owner);
    }
}

// The wand's menu over rim r, or over a notch: every fill, each worked out on a copy
// (with a spine, the fills from it come first). nullptr = open, else why not (nothing
// changed).
const char* wand_menu(RetopoState& st, const Mesh& m, int r, bool mirror, const SpineLine* sl,
                      const Notch* nt = nullptr) {
    st.loop_edges.clear();
    st.wand_base = snapshot_of(st);
    const RetopoMesh& base = st.wand_base.result;
    const uint32_t N = nt ? (uint32_t)(nt->run.size() + nt->mouth.size()) : base.rim_off[r + 1] - base.rim_off[r];
    const uint32_t fc0 = base.face_count();
    // The rim each fill works on (a notch's mouth vertices go in with it), and the twin
    // notch's copy of whatever it made.
    auto rim = [&] {
        if (nt) return notch_rim(st, *nt);
        const RetopoMesh& rm = st.result;
        return std::vector<uint32_t>(rm.rim.begin() + rm.rim_off[r], rm.rim.begin() + rm.rim_off[r + 1]);
    };
    auto fill = [&](auto&& how) {
        const uint32_t v0 = (uint32_t)st.result.v.size(), f0 = st.result.face_count();
        const std::vector<uint32_t> R = rim();
        if (!how(R)) return false;
        if (nt && nt->twin) {
            RetopoMesh& rm = st.result;
            std::vector<uint32_t> touched;
            reflect_new_faces(st, v0, f0, rm.v_owner[rm.mirror[nt->a]], touched);
            join_round(st, touched);
            rebuild_edges(rm);
        }
        return true;
    };
    std::vector<std::vector<uint32_t>> seen;   // each option's new faces, to drop repeats
    auto attempt = [&](const char* label, auto&& apply) {
        put(st, st.wand_base);
        if (!apply()) return;
        const RetopoMesh& rm = st.result;
        int quads = 0, tris = 0, other = 0;
        std::vector<uint32_t> sig;
        for (uint32_t f = fc0; f < rm.face_count(); f++) {
            const uint32_t k = rm.face_size(f);
            (k == 4 ? quads : k == 3 ? tris : other)++;
            std::vector<uint32_t> c(rm.face(f), rm.face(f) + k);
            std::sort(c.begin(), c.end());
            sig.insert(sig.end(), c.begin(), c.end());
            sig.push_back(~0u);
        }
        if (quads + tris + other == 0) return;
        for (const std::vector<uint32_t>& z : seen) if (z == sig) return;
        seen.push_back(sig);
        RetopoState::WandOption o;
        std::snprintf(o.label, sizeof o.label, "%s", label);
        if (tris) std::snprintf(o.counts, sizeof o.counts, "%d quad%s, %d tri%s", quads, quads == 1 ? "" : "s",
                                tris, tris == 1 ? "" : "s");
        else      std::snprintf(o.counts, sizeof o.counts, "%d quad%s", quads, quads == 1 ? "" : "s");
        o.after = snapshot_of(st);
        st.wand_opts.push_back(std::move(o));
    };
    const char* spine_why = nullptr;
    if (sl) {
        attempt("Your line", [&] {
            return fill([&](const std::vector<uint32_t>& R) { return !(spine_why = spine_fill(st, R, r, *sl, mirror, false)); });
        });
        attempt("Your line, tris at its ends", [&] {
            return fill([&](const std::vector<uint32_t>& R) { return !spine_fill(st, R, r, *sl, mirror, true); });
        });
        if (st.wand_opts.empty()) {
            put(st, st.wand_base);
            st.wand_base = RetopoSnapshot{};
            return spine_why ? spine_why : "That line doesn't make a spine here";
        }
    }
    attempt("Fewest faces", [&] { return fill([&](const std::vector<uint32_t>& R) { return fewest_fill(st, R, mirror); }); });
    if (mirror) {
        static const char* const ladders[] = {"Columns", "Columns, staggered", "Columns, staggered flip", "Rows"};
        for (int kind = 0; kind < 4; kind++)
            attempt(ladders[kind], [&] { return fill([&](const std::vector<uint32_t>& R) { return ladder_fill(st, R, kind); }); });
    }
    if (N % 2 == 0 && !nt) {
        attempt("Grid", [&] { int rims = 0, odd = 0; return fill_rims(st, m, mirror, rims, odd, r, false) > 0; });
        attempt("Grid, no ring", [&] { int rims = 0, odd = 0; return fill_rims(st, m, mirror, rims, odd, r, false, true) > 0; });
    }
    put(st, st.wand_base);
    if (st.wand_opts.empty()) {
        st.wand_base = RetopoSnapshot{};
        return mirror ? "Nothing fits this hole yet: odd count of sides, and not across the mirror line"
                      : "Odd count of sides: turn mirror on for a hole across x = 0, or cut a loop into the rim";
    }
    st.wand_open = true;
    st.wand_shown = -1;
    st.wand_keep_shown = false;
    std::printf("[retopo] wand%s: rim of %u sides, %zu option(s):", nt ? " (notch)" : sl ? " (spine)" : "", N,
                st.wand_opts.size());
    for (const RetopoState::WandOption& o : st.wand_opts) std::printf(" [%s: %s]", o.label, o.counts);
    std::printf("\n");
    return nullptr;
}
}  // namespace

const char* retopo_wand_open(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                             float x, float y, bool mirror) {
    retopo_wand_close(st, -1);
    if (!st.built) return "Nothing built yet: Build first (Space)";
    const int r = hole_under(st, cam, win_w, win_h, x, y);
    if (r < 0) return "Click inside a hole (the green or amber wash)";
    return wand_menu(st, m, r, mirror, nullptr);
}

void retopo_wand_show(RetopoState& st, int k) {
    if (!st.wand_open || k == st.wand_shown || k >= (int)st.wand_opts.size()) return;
    put(st, k < 0 ? st.wand_base : st.wand_opts[k].after);
    st.wand_shown = k;
}

void retopo_wand_close(RetopoState& st, int keep) {
    if (!st.wand_open) return;
    put(st, st.wand_base);
    if (keep >= 0 && keep < (int)st.wand_opts.size()) {
        st.checkpoint();
        put(st, st.wand_opts[keep].after);
        std::printf("[retopo] wand: kept %s (%s)\n", st.wand_opts[keep].label, st.wand_opts[keep].counts);
    }
    st.wand_opts.clear();
    st.wand_base = RetopoSnapshot{};
    st.wand_open = false;
    st.wand_shown = -1;
    st.wand_keep_shown = false;
}

// ---- Spine ---------------------------------------------------------------------

namespace {
// The hole rim through vertex v (the first, if it's on more than one), -1 for none.
int hole_rim_of(const RetopoMesh& rm, uint32_t v) {
    for (uint32_t r = 0; r < rm.rim_count(); r++)
        if (rm.rim_hole[r])
            for (uint32_t k = rm.rim_off[r]; k < rm.rim_off[r + 1]; k++)
                if (rm.rim[k] == v) return (int)r;
    return -1;
}

// The spine's ends: rim r through st.spine_from, the vertex b of that rim the line ended
// on (the one under the pen, else the nearest to the line's end within half a rim edge
// on screen), and whether the line ran through the hole (its middle mostly inside the
// rim as seen). False = not a spine.
bool spine_ends(const RetopoState& st, const Camera& cam, int win_w, int win_h, float ex, float ey,
                int& r, uint32_t& b) {
    const RetopoMesh& rm = st.result;
    if (!st.built || st.spine_from == ~0u || st.live.pts.size() < 3) return false;
    r = hole_rim_of(rm, st.spine_from);
    if (r < 0) return false;
    const uint32_t k0 = rm.rim_off[r], n = rm.rim_off[r + 1] - k0;
    std::vector<float> sx(n), sy(n);
    float edge_px = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        if (!cam.world_to_screen(rm.v[rm.rim[k0 + i]], win_w, win_h, sx[i], sy[i])) return false;
    for (uint32_t i = 0; i < n; i++) edge_px += std::hypot(sx[(i + 1) % n] - sx[i], sy[(i + 1) % n] - sy[i]);
    edge_px /= (float)n;
    // Through the hole: the line's middle (its ends sit on the rim) mostly inside.
    const size_t m = st.live.pts.size();
    int in_n = 0, all = 0;
    for (size_t i = m / 5; i < m - m / 5; i++) {
        float px, py;
        if (!cam.world_to_screen(st.live.pts[i], win_w, win_h, px, py)) continue;
        bool in = false;
        for (uint32_t a = 0, c = n - 1; a < n; c = a++)
            if ((sy[a] > py) != (sy[c] > py) && px < sx[c] + (sx[a] - sx[c]) * (py - sy[c]) / (sy[a] - sy[c])) in = !in;
        in_n += in;
        all++;
    }
    if (all == 0 || in_n * 10 < all * 7) return false;
    // Its end: on this rim, not where it started.
    b = retopo_snap_vertex(st, cam, win_w, win_h, ex, ey, kRetopoDrawSnapPx);
    bool on = false;
    for (uint32_t i = 0; i < n && b != ~0u; i++) on |= rm.rim[k0 + i] == b;
    if (!on) {
        float lx, ly;
        if (!cam.world_to_screen(st.live.pts.back(), win_w, win_h, lx, ly)) return false;
        b = ~0u;
        float best = std::max(12.0f, 0.5f * edge_px);
        for (uint32_t i = 0; i < n; i++) {
            const float d = std::hypot(sx[i] - lx, sy[i] - ly);
            if (d < best) { best = d; b = rm.rim[k0 + i]; }
        }
    }
    return b != ~0u && b != st.spine_from;
}
}  // namespace

uint32_t retopo_hole_vertex(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y) {
    if (!st.built) return ~0u;
    const uint32_t v = retopo_snap_vertex(st, cam, win_w, win_h, x, y, kRetopoDrawSnapPx);
    return v != ~0u && hole_rim_of(st.result, v) >= 0 ? v : ~0u;
}

void retopo_draw_snap(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float x, float y, bool any) {
    st.draw_v = st.draw_ea = st.draw_eb = ~0u;
    const RetopoMesh& rm = st.result;
    if (!st.built || rm.nb_off.size() != rm.v.size() + 1) return;
    auto frozen = [&](uint32_t owner) {
        for (const RetopoStroke& s : st.strokes) if (s.id == owner) return s.frozen;
        return false;
    };
    // A press lights only what it would draw a line from (a bridge or a spine): a dot on
    // a plain strip's vertex would promise a line and deliver a band.
    const uint32_t v = retopo_snap_vertex(st, cam, win_w, win_h, x, y, kRetopoDrawSnapPx);
    if (v != ~0u && (any || frozen(rm.v_owner[v]) || hole_rim_of(rm, v) >= 0)) {
        st.draw_v = v;
        return;
    }
    uint32_t a, b;
    if (!edge_under(st, cam, win_w, win_h, x, y, a, b, kRetopoDrawSnapPx)) return;
    if (!any && !(frozen(rm.v_owner[a]) && frozen(rm.v_owner[b]))) return;
    float ax, ay, bx, by;
    cam.world_to_screen(rm.v[a], win_w, win_h, ax, ay);
    cam.world_to_screen(rm.v[b], win_w, win_h, bx, by);
    const float ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
    st.draw_ea = a;
    st.draw_eb = b;
    st.draw_t = l2 > 0.0f ? std::max(0.0f, std::min(1.0f, ((x - ax) * ex + (y - ay) * ey) / l2)) : 0.5f;
}

bool retopo_draw_snap_point(const RetopoState& st, Vec3& p, Vec3& n) {
    const RetopoMesh& rm = st.result;
    if (st.draw_v < rm.v.size()) {
        p = rm.v[st.draw_v];
        n = rm.n[st.draw_v];
        return true;
    }
    if (st.draw_ea < rm.v.size() && st.draw_eb < rm.v.size()) {
        const float t = st.draw_t;
        p = rm.v[st.draw_ea] * (1.0f - t) + rm.v[st.draw_eb] * t;
        n = (rm.n[st.draw_ea] * (1.0f - t) + rm.n[st.draw_eb] * t).normalized();
        return true;
    }
    return false;
}

bool retopo_spine_gesture(const RetopoState& st, const Camera& cam, int win_w, int win_h, float ex, float ey) {
    int r;
    uint32_t b;
    return spine_ends(st, cam, win_w, win_h, ex, ey, r, b);
}

const char* retopo_wand_open_spine(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                                   float ex, float ey, bool mirror) {
    retopo_wand_close(st, -1);
    int r;
    SpineLine sl;
    if (!spine_ends(st, cam, win_w, win_h, ex, ey, r, sl.b)) return "Draw from a vertex of a hole's rim across it to another";
    sl.a = st.spine_from;
    sl.pts = st.live.pts;
    sl.nrm = st.live.nrm;
    if (const char* why = wand_menu(st, m, r, mirror, &sl)) return why;
    // Drawn on purpose: the line's own fill is up at once and stays unless another is
    // picked (click away keeps it, Esc takes it back).
    retopo_wand_show(st, 0);
    st.wand_keep_shown = true;
    return nullptr;
}

// ---- Notch ---------------------------------------------------------------------

namespace {
// Lines that join across a gap (notch, edge to vertex) must cross open space: false when
// the middle three fifths of the line mostly lie over faces (more than kOpenSpaceOver).
bool over_faces(const RetopoState& st, const Camera& cam, int win_w, int win_h, const std::vector<Vec3>& L) {
    int over = 0, all = 0;
    for (size_t i = L.size() / 5; i < L.size() - L.size() / 5; i++) {
        float x, y;
        if (!cam.world_to_screen(L[i], win_w, win_h, x, y)) continue;
        over += retopo_face_at(st, cam, win_w, win_h, x, y, 0.0f) != ~0u;
        all++;
    }
    return all && (float)over > kOpenSpaceOver * (float)all;
}


// The notch a line from outline vertex a to outline vertex b closes (see
// retopo_wand_open_notch). With `m`, its mouth too, snapped onto the sculpt; without,
// only whether it is one. nullptr = a notch, else why not.
const char* notch_find(const RetopoState& st, const Mesh* m, const Camera& cam, int win_w, int win_h,
                       uint32_t a, uint32_t b, const std::vector<Vec3>& L, bool mirror, Notch& nt) {
    const RetopoMesh& rm = st.result;
    if (!st.built || a == b || a >= rm.v.size() || b >= rm.v.size() || L.size() < 2)
        return "Draw from one outline vertex to another";
    // Both ends on one outer outline.
    int r = -1, ia = -1, ib = -1;
    for (uint32_t q = 0; q < rm.rim_count() && r < 0; q++) {
        if (rm.rim_hole[q]) continue;
        int fa = -1, fb = -1;
        for (uint32_t k = rm.rim_off[q]; k < rm.rim_off[q + 1]; k++) {
            if (rm.rim[k] == a) fa = (int)(k - rm.rim_off[q]);
            if (rm.rim[k] == b) fb = (int)(k - rm.rim_off[q]);
        }
        if (fa >= 0 && fb >= 0) { r = (int)q; ia = fa; ib = fb; }
    }
    if (r < 0) return "A notch closes between two vertices of one outline";
    const int M = (int)(rm.rim_off[r + 1] - rm.rim_off[r]);
    auto O = [&](int k) { return rm.rim[rm.rim_off[r] + ((k % M) + M) % M]; };
    // The notch is the short way round from a to b.
    const int fwd = ((ib - ia) % M + M) % M;
    const int dir = fwd <= M - fwd ? 1 : -1, nP = std::min(fwd, M - fwd);
    if (nP < 2) return "Those two are neighbours: nothing between them to close";
    if (over_faces(st, cam, win_w, win_h, L)) return "That line runs over faces: a notch closes across open space";
    // Out of the notch: from the middle of the outline round it toward the line's middle.
    Vec3 pc(0, 0, 0);
    for (int s = 0; s <= nP; s++) pc = pc + rm.v[O(ia + dir * s)];
    pc = pc * (1.0f / (float)(nP + 1));
    Vec3 out = (rm.v[a] + rm.v[b]) * 0.5f - pc;
    if (out.length() < 1e-9f) return "Can't tell which way that notch opens";
    out = out.normalized();
    // Legs, in step: past a and past b while both keep heading out (a Pi's walls below
    // the line). Where either turns off, both stop: those are the feet.
    int legs = 0;
    auto heads_out = [&](int k, int step) {
        const Vec3 e = rm.v[O(k + step)] - rm.v[O(k)];
        return e.length() > 1e-9f && e.normalized().dot(out) > kNotchLegCos;
    };
    while (nP + 2 * (legs + 1) < M && heads_out(ia - dir * legs, -dir) && heads_out(ib + dir * legs, dir)) legs++;
    nt = Notch{};
    nt.a = a;
    nt.b = b;
    nt.legs = legs;
    for (int s = -legs; s <= nP + legs; s++) nt.run.push_back(O(ia + dir * s));
    const uint32_t fa = nt.run.front(), fb = nt.run.back();
    const bool pairs = mirror && rm.mirror.size() == rm.v.size();
    nt.seam = pairs && rm.mirror[fa] == fb;
    if (pairs && !nt.seam) {
        nt.twin = true;
        for (uint32_t v : nt.run) {
            const uint32_t w = rm.mirror[v];
            nt.twin &= w < rm.v.size() && w != v && std::find(nt.run.begin(), nt.run.end(), w) == nt.run.end();
        }
    }
    if (!m) return nullptr;

    // The mouth: as many edges as make the two sides of the line even (each rung a quad),
    // at least one. Feet to feet in a straight line snapped onto the sculpt; no legs ->
    // the mouth is the line itself.
    const int me = std::max(1, legs ? nP - 2 * legs : nP - 2);
    std::vector<float> arc(L.size(), 0.0f);
    for (size_t i = 1; i < L.size(); i++) arc[i] = arc[i - 1] + (L[i] - L[i - 1]).length();
    auto on_line = [&](float d) {   // d along the line from a
        size_t i = 1;
        while (i + 1 < arc.size() && arc[i] < d) i++;
        const float seg = std::max(1e-9f, arc[i] - arc[i - 1]);
        return L[i - 1] + (L[i] - L[i - 1]) * std::min(1.0f, std::max(0.0f, (d - arc[i - 1]) / seg));
    };
    SnapGrid g;
    {
        Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
        float edge = 0.0f;
        for (size_t i = 0; i < nt.run.size(); i++) {
            const Vec3 p = rm.v[nt.run[i]];
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
            if (i) edge += (p - rm.v[nt.run[i - 1]]).length();
        }
        edge /= (float)std::max<size_t>(1, nt.run.size() - 1);
        const Vec3 pad(2.0f * edge, 2.0f * edge, 2.0f * edge);
        lo = lo - pad;
        hi = hi + pad;
        const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        g.build(*m, lo, hi, std::max(edge * 0.5f, ext / 96.0f));
    }
    for (int j = 1; j < me; j++) {   // from B' (j = 1) toward A'
        const float u = (float)j / (float)me;
        Vec3 p = legs ? rm.v[fb] * (1.0f - u) + rm.v[fa] * u : on_line(arc.back() * (1.0f - u));
        Vec3 n = (rm.n[fb] * (1.0f - u) + rm.n[fa] * u).normalized();
        if (nt.seam && 2 * j > me) {   // the twin of one already placed
            p = nt.mouth[me - j - 1];
            n = nt.mouth_n[me - j - 1];
            p.x = -p.x;
            n.x = -n.x;
        } else {
            Vec3 q, qn;
            if (g.closest(p, q, qn, 8)) { p = q; n = qn; }
            if (nt.seam && 2 * j == me) { p.x = 0.0f; n.x = 0.0f; n = n.normalized(); }
        }
        nt.mouth.push_back(p);
        nt.mouth_n.push_back(n);
    }
    return nullptr;
}
}  // namespace

bool retopo_notch_gesture(const RetopoState& st, const Camera& cam, int win_w, int win_h, bool mirror) {
    if (st.line_from_v == ~0u || st.draw_v == ~0u) return false;
    Notch nt;
    const char* why = notch_find(st, nullptr, cam, win_w, win_h, st.line_from_v, st.draw_v, st.live.pts, mirror, nt);
    if (why) std::printf("[retopo] notch: no (%s)\n", why);
    return !why;
}

const char* retopo_wand_open_notch(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                                   bool mirror) {
    retopo_wand_close(st, -1);
    Notch nt;
    if (const char* why = notch_find(st, &m, cam, win_w, win_h, st.line_from_v, st.draw_v, st.live.pts, mirror, nt))
        return why;
    std::printf("[retopo] notch: %zu outline edges, legs %d, mouth %zu new vertices%s\n", nt.run.size() - 1,
                nt.legs, nt.mouth.size(), nt.seam ? ", across the seam" : nt.twin ? ", twin notch too" : "");
    // With legs the line runs through the notch: its spine fills first. Without, the line
    // is the mouth and the notch is an ordinary hole.
    SpineLine sl;
    sl.a = nt.a;
    sl.b = nt.b;
    sl.pts = st.live.pts;
    sl.nrm = st.live.nrm;
    if (const char* why = wand_menu(st, m, -1, mirror, nt.legs ? &sl : nullptr, &nt)) return why;
    // Drawn on purpose, as a spine is: a fill is up at once and stays unless another is
    // picked (click away keeps it, Esc takes it back). The line's own when it has one,
    // else the first all-quad fill (Fewest faces leads the list and likes triangles).
    int show = 0;
    if (!nt.legs)
        for (int k = (int)st.wand_opts.size() - 1; k >= 0; k--)
            if (!std::strstr(st.wand_opts[k].counts, "tri")) show = k;
    retopo_wand_show(st, show);
    st.wand_keep_shown = true;
    return nullptr;
}

// ---- Edge to vertex ------------------------------------------------------------

namespace {
// The quad x, y, vn, v: outline edge (x, y) rung over to outline vertex v and its outline
// neighbour vn; vo is v's other neighbour (the wedge side, past v from vn).
struct EdgeVertex { uint32_t x = ~0u, y = ~0u, v = ~0u, vn = ~0u, vo = ~0u; };

// Which edge and vertex the live line joins: started on a lit edge and ended on a lit
// vertex, or the other way round. False = neither.
bool edge_vertex_ends(const RetopoState& st, uint32_t& ea, uint32_t& eb, uint32_t& v) {
    if (st.line_from_ea != ~0u && st.draw_v != ~0u) { ea = st.line_from_ea; eb = st.line_from_eb; v = st.draw_v; return true; }
    if (st.line_from_v != ~0u && st.draw_ea != ~0u) { ea = st.draw_ea; eb = st.draw_eb; v = st.line_from_v; return true; }
    return false;
}

// The quads that fit, one per side of the vertex (best rungs first); empty = why says.
const char* edge_vertex_find(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                             std::vector<EdgeVertex>& out) {
    out.clear();
    const RetopoMesh& rm = st.result;
    uint32_t ea, eb, v;
    if (!st.built || !edge_vertex_ends(st, ea, eb, v)) return "Draw from an outline edge to an outline vertex";
    const uint32_t nv = (uint32_t)rm.v.size();
    if (ea >= nv || eb >= nv || v >= nv || rm.border_nb.size() != 2 * (size_t)nv) return "Nothing to join there";
    if (!(rm.border_nb[ea * 2] == eb || rm.border_nb[ea * 2 + 1] == eb)) return "That edge isn't on an outline";
    if (!on_border(rm, v)) return "That vertex isn't on an outline";
    if (v == ea || v == eb) return "The vertex is on that edge";
    if (over_faces(st, cam, win_w, win_h, st.live.pts))
        return "That line runs over faces: edge to vertex joins across open space";
    // Per side of v (which neighbour the quad takes): the shortest rungs that don't cross,
    // either way round. Crossing is judged flat, in the plane of the corners' mean normal.
    float side_len[2] = {1e30f, 1e30f};
    EdgeVertex side_ev[2];
    for (int side = 0; side < 2; side++) {
        const uint32_t vn = rm.border_nb[v * 2 + side], vo = rm.border_nb[v * 2 + 1 - side];
        if (vn >= nv || vn == ea || vn == eb) continue;
        float& best = side_len[side];
        EdgeVertex& ev = side_ev[side];
        for (int way = 0; way < 2; way++) {
            const uint32_t x = way ? eb : ea, y = way ? ea : eb;
            const Vec3 nn = (rm.n[x] + rm.n[y] + rm.n[v] + rm.n[vn]).normalized();
            Vec3 u = (rm.v[y] - rm.v[x]);
            u = (u - nn * u.dot(nn)).normalized();
            const Vec3 w = nn.cross(u);
            auto flat = [&](uint32_t i, float& a, float& b) { a = rm.v[i].dot(u); b = rm.v[i].dot(w); };
            float x0, y0, x1, y1, x2, y2, x3, y3;
            flat(x, x0, y0); flat(v, x1, y1); flat(y, x2, y2); flat(vn, x3, y3);
            const float rx = x1 - x0, ry = y1 - y0, qx = x3 - x2, qy = y3 - y2, den = rx * qy - ry * qx;
            bool cross = false;
            if (std::fabs(den) > 1e-12f) {
                const float t = ((x2 - x0) * qy - (y2 - y0) * qx) / den, k = ((x2 - x0) * ry - (y2 - y0) * rx) / den;
                cross = t > 0.0f && t < 1.0f && k > 0.0f && k < 1.0f;
            }
            const float len = (rm.v[v] - rm.v[x]).length() + (rm.v[vn] - rm.v[y]).length();
            if (cross || len >= best) continue;
            // Wound against the faces on the edge (as edge_vertex_fill will), the quad's
            // other sides must not run the way a face already does: that is a face laid
            // over another (an edge joined to a corner of its own mirror twin did it).
            uint32_t q[4] = {x, y, vn, v};
            if (has_side(rm, x, y)) std::reverse(q, q + 4);
            bool over = false;
            for (int i = 0; i < 4 && !over; i++) over = has_side(rm, q[i], q[(i + 1) % 4]);
            if (over) continue;
            best = len;
            ev = {x, y, v, vn, vo < nv && vo != x && vo != y ? vo : ~0u};
        }
    }
    const int first = side_len[1] < side_len[0] ? 1 : 0;
    for (int side : {first, 1 - first})
        if (side_len[side] < 1e29f) out.push_back(side_ev[side]);
    return out.empty() ? "No quad fits between that edge and vertex" : nullptr;
}

// Adds the quad (and with `wedge` the triangle x, v, vo), wound against the faces already
// on the edge; with mirror, the twin too unless it's its own; the pieces become one.
// True when a face from f0 on runs a side the same way as another face (one over the
// other). Faces before f0 are only compared against the new ones, so an old overlap
// elsewhere doesn't block anything.
bool new_faces_double(const RetopoMesh& rm, uint32_t f0) {
    std::vector<std::pair<uint32_t, uint32_t>> old, made;
    std::vector<uint32_t> c;
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        const int k = face_corners(rm, f, c);
        for (int i = 0; i < k; i++) (f < f0 ? old : made).push_back({c[i], c[(i + 1) % k]});
    }
    std::sort(old.begin(), old.end());
    std::sort(made.begin(), made.end());
    if (std::adjacent_find(made.begin(), made.end()) != made.end()) return true;
    for (const auto& e : made)
        if (std::binary_search(old.begin(), old.end(), e)) return true;
    return false;
}

bool edge_vertex_fill(RetopoState& st, const EdgeVertex& ev, bool wedge, bool mirror) {
    RetopoMesh& rm = st.result;
    if (wedge && ev.vo == ~0u) return false;
    const bool flip = has_side(rm, ev.x, ev.y);
    const uint32_t owner = rm.v_owner[ev.v], f0 = rm.face_count();
    std::vector<std::vector<uint32_t>> F{{ev.x, ev.y, ev.vn, ev.v}};
    if (wedge) F.push_back({ev.x, ev.v, ev.vo});
    for (std::vector<uint32_t>& f : F) {
        if (flip) std::reverse(f.begin(), f.end());
        rm.add_face(f.data(), (int)f.size(), owner);
    }
    std::vector<uint32_t> touched{rm.v_owner[ev.x], rm.v_owner[ev.y], rm.v_owner[ev.v], rm.v_owner[ev.vn]};
    if (mirror && rm.mirror.size() == rm.v.size()) {
        std::vector<uint32_t> mine{ev.x, ev.y, ev.v, ev.vn}, twins;
        if (wedge) mine.push_back(ev.vo);
        bool ok = true;
        for (uint32_t i : mine) { ok &= rm.mirror[i] < rm.v.size(); if (ok) twins.push_back(rm.mirror[i]); }
        std::vector<uint32_t> a = mine, b = twins;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (ok && a != b) reflect_new_faces(st, (uint32_t)rm.v.size(), f0, rm.v_owner[rm.mirror[ev.v]], touched);
    }
    // The wedge, or the twin, landing over a face: not an option (the wand puts the base back).
    if (new_faces_double(rm, f0)) return false;
    join_round(st, touched);
    rebuild_edges(rm);
    return true;
}
}  // namespace

bool retopo_edge_vertex_gesture(const RetopoState& st, const Camera& cam, int win_w, int win_h) {
    uint32_t ea, eb, v;
    if (!edge_vertex_ends(st, ea, eb, v)) return false;
    std::vector<EdgeVertex> ev;
    const char* why = edge_vertex_find(st, cam, win_w, win_h, ev);
    if (why) std::printf("[retopo] edge to vertex: no (%s)\n", why);
    return !why;
}

const char* retopo_wand_open_edge_vertex(RetopoState& st, const Camera& cam, int win_w, int win_h, bool mirror) {
    retopo_wand_close(st, -1);
    std::vector<EdgeVertex> ev;
    if (const char* why = edge_vertex_find(st, cam, win_w, win_h, ev)) return why;
    st.loop_edges.clear();
    st.wand_base = snapshot_of(st);
    // The vertex brings both its outline neighbours (user's rule): three vertices to the
    // edge's two, a quad on one side of it and a triangle on the other. The quad with the
    // better rungs first, then the other way round, then the quad alone.
    auto option = [&](const EdgeVertex& e, bool wedge, const char* label) {
        put(st, st.wand_base);
        if (!edge_vertex_fill(st, e, wedge, mirror)) return;
        RetopoState::WandOption o;
        std::snprintf(o.label, sizeof o.label, "%s", label);
        std::snprintf(o.counts, sizeof o.counts, "%s", wedge ? "1 quad, 1 tri" : "1 quad");
        o.after = snapshot_of(st);
        st.wand_opts.push_back(std::move(o));
    };
    option(ev[0], true, "Both neighbours");
    if (ev.size() > 1) option(ev[1], true, "Both neighbours, flipped");
    option(ev[0], false, "One quad");
    put(st, st.wand_base);
    if (st.wand_opts.empty()) { st.wand_base = RetopoSnapshot{}; return "No quad fits there"; }
    st.wand_open = true;
    st.wand_keep_shown = false;
    st.wand_shown = -1;
    std::printf("[retopo] edge to vertex:");
    for (const RetopoState::WandOption& o : st.wand_opts) std::printf(" [%s: %s]", o.label, o.counts);
    std::printf("\n");
    // Drawn on purpose: the first is up at once and stays unless another is picked.
    retopo_wand_show(st, 0);
    st.wand_keep_shown = true;
    return nullptr;
}

// ---- Lasso cleanup (wand drag) --------------------------------------------------


const char* retopo_lasso_cleanup(RetopoState& st, const Camera& cam, int win_w, int win_h,
                                 const std::vector<float>& xy, bool mirror) {
    static char msg[160];
    RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    const size_t np = xy.size() / 2;
    if (!st.built || np < 3 || rm.nb_off.size() != nv + 1) return "Nothing built to clean up";

    // What the lasso holds: vertices in use, facing the eye, inside it on screen; with
    // mirror, their twins too.
    std::vector<uint8_t> in(nv, 0);
    std::vector<uint32_t> S;
    for (uint32_t v = 0; v < nv; v++) {
        if (!alive(rm, v) || st.hidden(rm.v_owner[v])) continue;
        const Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[v] : cam.get_view_direction() * -1.0f;
        float x, y;
        if (rm.n[v].dot(to_eye) <= 0.0f || !cam.world_to_screen(rm.v[v], win_w, win_h, x, y)) continue;
        bool inside = false;
        for (size_t a = 0, c = np - 1; a < np; c = a++) {
            const float ax = xy[a * 2], ay = xy[a * 2 + 1], cx = xy[c * 2], cy = xy[c * 2 + 1];
            if ((ay > y) != (cy > y) && x < cx + (ax - cx) * (y - cy) / (ay - cy)) inside = !inside;
        }
        if (inside) { in[v] = 1; S.push_back(v); }
    }
    const bool pairs = mirror && rm.mirror.size() == nv;
    if (pairs)
        for (size_t k = 0, n = S.size(); k < n; k++) {
            const uint32_t w = rm.mirror[S[k]];
            if (w < nv && !in[w] && alive(rm, w)) { in[w] = 1; S.push_back(w); }
        }
    if (S.empty()) return "Lasso some vertices to clean up";

    // Merge by distance, the reach set per vertex by its own edges.
    std::vector<float> reach(nv, 0.0f);
    for (uint32_t v : S) {
        float sum = 0.0f;
        const uint32_t k0 = rm.nb_off[v], k1 = rm.nb_off[v + 1];
        for (uint32_t k = k0; k < k1; k++) sum += (rm.v[rm.nb[k]] - rm.v[v]).length();
        reach[v] = k1 > k0 ? kLassoMergeFrac * sum / (float)(k1 - k0) : 0.0f;
    }
    std::vector<uint32_t> up(nv);
    for (uint32_t v = 0; v < nv; v++) up[v] = v;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    for (size_t i = 0; i < S.size(); i++)
        for (size_t j = i + 1; j < S.size(); j++) {
            const uint32_t a = S[i], b = S[j];
            const float r = std::min(reach[a], reach[b]);
            const Vec3 d = rm.v[a] - rm.v[b];
            if (d.dot(d) < r * r) {
                const uint32_t ra = root(a), rb = root(b);
                if (ra != rb) up[std::max(ra, rb)] = std::min(ra, rb);
            }
        }
    // Each group to its lowest index, at the group's middle; a group on the seam (a seam
    // vertex in it, or a vertex and its own twin) flat onto x = 0.
    std::vector<Vec3> sum_p(nv), sum_n(nv);
    std::vector<int> cnt(nv, 0);
    std::vector<uint8_t> seam(nv, 0);
    int merged = 0;
    for (uint32_t v : S) {
        const uint32_t r = root(v);
        sum_p[r] = sum_p[r] + rm.v[v];
        sum_n[r] = sum_n[r] + rm.n[v];
        cnt[r]++;
        if (pairs && (rm.mirror[v] == v || (rm.mirror[v] < nv && root(rm.mirror[v]) == r))) seam[r] = 1;
    }
    std::vector<uint32_t> touched;
    for (uint32_t v : S) {
        const uint32_t r = root(v);
        if (v != r) { merged++; touched.push_back(rm.v_owner[v]); touched.push_back(rm.v_owner[r]); }
    }

    // The faces again: merged corners repeat, so runs collapse; a face left with under
    // three corners (it spanned the gap that closed) or pinched on a corner goes, and so
    // does a second copy of one already there.
    FaceList out;
    std::vector<uint32_t> c;
    int dropped = 0;
    std::vector<std::pair<std::vector<uint32_t>, uint32_t>> kept;
    for (size_t f = 0; f < rm.face_count(); f++) {
        c.clear();
        const uint32_t* q = rm.face(f);
        for (uint32_t i = 0; i < rm.face_size(f); i++) {
            const uint32_t v = in[q[i]] ? root(q[i]) : q[i];
            if (c.empty() || c.back() != v) c.push_back(v);
        }
        while (c.size() > 1 && c.back() == c.front()) c.pop_back();
        std::vector<uint32_t> sorted = c;
        std::sort(sorted.begin(), sorted.end());
        const bool pinched = std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
        if (c.size() < 3 || pinched) { dropped++; touched.push_back(rm.face_owner[f]); continue; }
        bool dup = false;
        for (const auto& k : kept) dup |= k.first == sorted;
        if (dup) { dropped++; continue; }
        kept.push_back({sorted, rm.face_owner[f]});
        out.add(c.data(), (int)c.size(), rm.face_owner[f]);
    }
    // The history step is taken at the end, from this copy, only if something changed.
    RetopoMesh before = rm;
    for (uint32_t v : S) {
        const uint32_t r = root(v);
        if (v != r || cnt[r] < 2) continue;
        rm.v[r] = sum_p[r] * (1.0f / (float)cnt[r]);
        if (sum_n[r].length() > 1e-9f) rm.n[r] = sum_n[r].normalized();
        if (seam[r]) rm.v[r].x = 0.0f;
    }
    if (pairs) {
        for (uint32_t v : S) {
            const uint32_t r = root(v);
            if (seam[r]) { rm.mirror[r] = r; continue; }
            const uint32_t w = rm.mirror[v];
            if (w < nv) rm.mirror[r] = in[w] ? root(w) : w;
        }
        // Twins' groups average to exact reflections of each other.
        for (uint32_t v : S) {
            const uint32_t r = root(v), w = rm.mirror[r];
            if (v != r || w >= nv || w == r || w < r) continue;
            Vec3 m = (rm.v[r] + Vec3(-rm.v[w].x, rm.v[w].y, rm.v[w].z)) * 0.5f;
            rm.v[r] = m;
            rm.v[w] = Vec3(-m.x, m.y, m.z);
        }
    }
    out.swap_into(rm);
    rebuild_edges(rm);

    // Small holes the welds left (3 or 4 sides, every corner in the lasso): one face.
    int filled = 0;
    for (uint32_t r = 0; r < rm.rim_count(); r++) {
        if (!rm.rim_hole[r]) continue;
        const uint32_t k0 = rm.rim_off[r], n = rm.rim_off[r + 1] - k0;
        if (n < 3 || n > 4) continue;
        bool all = true;
        for (uint32_t k = 0; k < n; k++) all &= in[rm.rim[k0 + k]] != 0;
        if (!all) continue;
        std::vector<uint32_t> R(rm.rim.begin() + k0, rm.rim.begin() + k0 + n);
        if (has_side(rm, R[0], R[1])) std::reverse(R.begin(), R.end());
        rm.add_face(R.data(), (int)n, rm.v_owner[R[0]]);
        touched.push_back(rm.v_owner[R[0]]);
        filled++;
    }
    if (filled) rebuild_edges(rm);
    if (merged == 0 && dropped == 0 && filled == 0) {
        rm = std::move(before);   // nothing changed: no empty step in the history
        std::snprintf(msg, sizeof msg, "Nothing to clean in there (%zu vertices, none close enough to weld)", S.size());
        return msg;
    }
    std::swap(rm, before);
    st.checkpoint();
    std::swap(rm, before);
    // Pieces the welds joined are one piece now (the oldest keeps its name); the rest of
    // what was touched is frozen, as any edit does.
    {
        uint32_t max_id = 0;
        for (uint32_t o : rm.face_owner) max_id = std::max(max_id, o);
        std::vector<uint32_t> grp(max_id + 1), first(rm.v.size(), ~0u);
        for (uint32_t i = 0; i <= max_id; i++) grp[i] = i;
        auto top = [&](uint32_t x) { while (grp[x] != x) x = grp[x] = grp[grp[x]]; return x; };
        for (size_t q = 0; q < rm.face_count(); q++)
            for (uint32_t k = rm.face_off[q]; k < rm.face_off[q + 1]; k++) {
                uint32_t& f = first[rm.faces[k]];
                if (f == ~0u) f = rm.face_owner[q];
                else          grp[top(f)] = top(rm.face_owner[q]);
            }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        std::vector<uint8_t> done(max_id + 1, 0);
        for (uint32_t id : touched) {
            if (id > max_id || done[top(id)]) continue;
            done[top(id)] = 1;
            std::vector<uint32_t> group;
            for (uint32_t o = 0; o <= max_id; o++) if (top(o) == top(id) && st.find(o)) group.push_back(o);
            if (group.size() > 1) join_round(st, group);
        }
    }
    freeze_touched(st, touched);
    std::printf("[retopo] lasso: %zu vertices, %d welded, %d faces dropped, %d small holes filled, now %u faces\n",
                S.size(), merged, dropped, filled, rm.face_count());
    std::snprintf(msg, sizeof msg, "Cleaned: %d welded, %d faces dropped, %d holes closed", merged, dropped, filled);
    return msg;
}

namespace {
const char* spine_fill(RetopoState& st, const std::vector<uint32_t>& R, int r, const SpineLine& sl,
                       bool mirror, bool ends_tri) {
    const uint32_t bv = sl.b;
    RetopoMesh& rm = st.result;
    const int N = (int)R.size();
    auto md = [&](int k) { return ((k % N) + N) % N; };
    const int ia = (int)(std::find(R.begin(), R.end(), sl.a) - R.begin());
    const int ib = (int)(std::find(R.begin(), R.end(), bv) - R.begin());
    // Both sides from a to b: S up the rim, T down it.
    std::vector<uint32_t> S, T;
    for (int k = ia;; k = md(k + 1)) { S.push_back(R[k]); if (k == ib) break; }
    for (int k = ia;; k = md(k - 1)) { T.push_back(R[k]); if (k == ib) break; }
    const int n1 = (int)S.size() - 1, n2 = (int)T.size() - 1;
    if (std::min(n1, n2) < (ends_tri ? 2 : 3))
        return "Too close: a spine needs three rim edges or more on each side (end it further round)";
    // As many spine edges as the shorter side has, less the two its end quads take; or,
    // with triangles at its ends, one per edge of the shorter side.
    const int k = std::min(n1, n2) - (ends_tri ? 0 : 2);
    // Spine vertex i's rung lands on rim vertex at(n, i) of a side with n edges: the end
    // faces take two rim edges each (one, with triangles there), the middle ones share
    // the rest out.
    auto at = [&](int n, int i) {
        if (i <= 0) return 0;
        if (i >= k) return n;
        if (ends_tri) return (int)std::lround((double)i * n / k);
        if (k == 2) return n / 2;
        return 2 + (int)std::lround((double)(i - 1) * (n - 4) / (k - 2));
    };

    // Mirror: a hole that is its own reflection, spine from seam vertex to seam vertex ->
    // the spine lies on x = 0. A hole whose twin is elsewhere -> the twin is reflected.
    const int t = mirror ? rim_reflection(rm, R) : -1;
    const bool on_seam = t >= 0 && md(t - ia) == ia && md(t - ib) == ib;
    // ...from a vertex to its own twin -> the spine crosses x = 0: its halves are twins
    // (a notch across the seam, closed wall to wall).
    const bool across = t >= 0 && !on_seam && md(t - ia) == ib;
    bool twinned = false;
    if (mirror && t < 0 && rm.mirror.size() == rm.v.size()) {
        twinned = true;
        for (uint32_t v : R) twinned &= rm.mirror[v] < rm.v.size() && rm.mirror[v] != v;
        const int tr = twinned ? hole_rim_of(rm, rm.mirror[R[0]]) : -1;
        twinned = tr >= 0 && tr != r;
    }

    // Spine vertices: on the drawn line, each at the line sample nearest the middle of its
    // two rungs' rim ends, walking on along the line (never back).
    const uint32_t owner = rm.v_owner[R[0]];
    const uint32_t first_new = (uint32_t)rm.v.size();
    std::vector<uint32_t> sp(k + 1);
    sp[0] = S[0];
    sp[k] = S[n1];
    size_t from = 0;
    const std::vector<Vec3>& L = sl.pts;
    for (int i = 1; i < k; i++) {
        const Vec3 mid = (rm.v[S[at(n1, i)]] + rm.v[T[at(n2, i)]]) * 0.5f;
        size_t best = from;
        float bd = 1e30f;
        for (size_t j = from; j < L.size(); j++) {
            const float d = (L[j] - mid).length();
            if (d < bd) { bd = d; best = j; }
        }
        from = best;
        Vec3 p = L[best], nn = sl.nrm[best];
        const bool self_twin = on_seam || (across && 2 * i == k);
        if (across && 2 * i > k) {   // the twin of one already placed
            p = rm.v[sp[k - i]];
            nn = rm.n[sp[k - i]];
            p.x = -p.x;
            nn.x = -nn.x;
        }
        if (self_twin) { p.x = 0.0f; nn.x = 0.0f; nn = nn.length() > 1e-6f ? nn.normalized() : rm.n[R[ia]]; }
        rm.v.push_back(p);
        rm.n.push_back(nn);
        rm.v_owner.push_back(owner);
        rm.mirror.push_back(self_twin ? (uint32_t)rm.v.size() - 1 : ~0u);
        sp[i] = (uint32_t)rm.v.size() - 1;
        if (across && 2 * i > k) {
            rm.mirror[sp[i]] = sp[k - i];
            rm.mirror[sp[k - i]] = sp[i];
        }
    }

    // Faces, every rim side in them running up the rim: on S, its run up then the spine
    // back; on T, its run (read backwards, which is up the rim) then the spine forward.
    std::vector<std::vector<uint32_t>> F;
    auto add = [&](std::vector<uint32_t> f) {
        f.erase(std::unique(f.begin(), f.end()), f.end());
        if (f.size() > 1 && f.front() == f.back()) f.pop_back();
        if (f.size() < 3) return;
        if (f.size() <= 4) { F.push_back(f); return; }
        // More rim than a quad holds (one side longer than the other): cut it the way
        // Fewest faces would, triangles toward the rim; a fan if nothing fits.
        RimShape sh;
        sh.set(rm, f);
        std::vector<int> Q(f.size());
        for (size_t j = 0; j < f.size(); j++) Q[j] = (int)j;
        ChainCut cut;
        cut.solve(sh, Q);
        std::vector<std::vector<int>> G;
        if (cut.B(0, (int)f.size() - 1) < 1e29f) cut.faces(Q, 0, (int)f.size() - 1, G);
        else for (size_t j = 1; j + 1 < f.size(); j++) G.push_back({0, (int)j, (int)j + 1});
        for (const std::vector<int>& g : G) {
            std::vector<uint32_t> q;
            for (int j : g) q.push_back(f[j]);
            F.push_back(q);
        }
    };
    for (int i = 0; i < k; i++) {
        std::vector<uint32_t> f;
        for (int j = at(n1, i); j <= at(n1, i + 1); j++) f.push_back(S[j]);
        f.push_back(sp[i + 1]);
        f.push_back(sp[i]);
        add(f);
        std::vector<uint32_t> g;
        for (int j = at(n2, i + 1); j >= at(n2, i); j--) g.push_back(T[j]);
        g.push_back(sp[i]);
        g.push_back(sp[i + 1]);
        add(g);
    }
    // Wound against the faces already on the rim, as Fill does.
    const bool flip = has_side(rm, R[0], R[1]);
    const uint32_t first_face = rm.face_count();
    for (std::vector<uint32_t>& f : F) {
        if (flip) std::reverse(f.begin(), f.end());
        rm.add_face(f.data(), (int)f.size(), owner);
    }
    std::vector<uint32_t> touched;
    for (uint32_t v : R) touched.push_back(rm.v_owner[v]);
    // A twin hole elsewhere: the same spine and faces, reflected vertex by vertex.
    if (twinned) reflect_new_faces(st, first_new, first_face, rm.v_owner[rm.mirror[R[0]]], touched);
    join_round(st, touched);
    rebuild_edges(rm);
    return nullptr;
}
}  // namespace

// ---- Symmetrize ------------------------------------------------------------------

namespace {
// A vertex this close to x = 0 (x the mean edge) with no twin of its own is on the seam.
constexpr float kSymSeamFrac = 0.1f;
}  // namespace

const char* retopo_symmetrize(RetopoState& st, const Mesh& m, int keep, int& faces_out) {
    const RetopoMesh& rm = st.result;
    faces_out = 0;
    if (!st.built || rm.face_count() == 0) return "Nothing built to symmetrize";
    const uint32_t nv = (uint32_t)rm.v.size();
    const float s = keep < 0 ? -1.0f : 1.0f;
    const bool pairs = rm.mirror.size() == nv;
    auto refl = [](Vec3 v) { v.x = -v.x; return v; };
    float edge = 0.0f;
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2)
        edge += (rm.v[rm.edges[e]] - rm.v[rm.edges[e + 1]]).length();
    edge /= (float)std::max<size_t>(1, rm.edges.size() / 2);

    // Each vertex's side: 1 kept, 0 on the seam, -1 the side that gets rebuilt.
    std::vector<uint8_t> used(nv, 0);
    for (uint32_t i : rm.faces) used[i] = 1;
    std::vector<int8_t> side(nv, 0);
    for (uint32_t i = 0; i < nv; i++) {
        const uint32_t tw = pairs ? rm.mirror[i] : ~0u;
        const bool own_twin = tw < nv && tw != i;
        const float ax = std::fabs(rm.v[i].x);
        const bool seam = tw == i || ax <= 1e-7f || (!own_twin && ax < kSymSeamFrac * edge);
        side[i] = seam ? 0 : (rm.v[i].x * s > 0.0f ? 1 : -1);
    }

    // Faces across the seam keep their topology when every far-side corner has a partner
    // on the kept side: its twin, else the kept vertex nearest its reflection. A piece
    // where that fails anywhere is cut at x = 0 instead (all of it: a face paired beside
    // a face cut would leave a crack between them).
    std::vector<uint32_t> up(nv);
    for (uint32_t i = 0; i < nv; i++) up[i] = i;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    for (size_t f = 0; f < rm.face_count(); f++)
        for (uint32_t k = rm.face_off[f] + 1; k < rm.face_off[f + 1]; k++)
            up[root(rm.faces[k])] = root(rm.faces[rm.face_off[f]]);
    std::vector<uint32_t> partner(nv, ~0u), claimed(nv, ~0u);
    std::vector<uint8_t> cut_piece(nv, 0);   // by root
    auto straddles = [&](size_t f, int& nk, int& nd) {
        nk = nd = 0;
        for (uint32_t k = rm.face_off[f]; k < rm.face_off[f + 1]; k++) {
            nk += side[rm.faces[k]] > 0;
            nd += side[rm.faces[k]] < 0;
        }
        return nk && nd;
    };
    const float reach2 = (kLassoMergeFrac * edge) * (kLassoMergeFrac * edge);
    for (size_t f = 0; f < rm.face_count(); f++) {
        int nk, nd;
        if (!straddles(f, nk, nd)) continue;
        for (uint32_t k = rm.face_off[f]; k < rm.face_off[f + 1]; k++) {
            const uint32_t d = rm.faces[k];
            if (side[d] >= 0 || partner[d] != ~0u) continue;
            uint32_t p = pairs && rm.mirror[d] < nv && rm.mirror[d] != d && used[rm.mirror[d]]
                         && side[rm.mirror[d]] >= 0 ? rm.mirror[d] : ~0u;
            if (p == ~0u) {
                float best = reach2;
                const Vec3 want = refl(rm.v[d]);
                for (uint32_t i = 0; i < nv; i++) {
                    if (!used[i] || side[i] < 0) continue;
                    const float d2 = (rm.v[i] - want).dot(rm.v[i] - want);
                    if (d2 < best) { best = d2; p = i; }
                }
            }
            if (p == ~0u || (claimed[p] != ~0u && claimed[p] != d)) cut_piece[root(d)] = 1;
            if (p != ~0u) { partner[d] = p; claimed[p] = d; }
        }
    }
    // ...and a partner mustn't land on another corner of the same face (it would fold).
    std::vector<uint64_t> u;
    for (size_t f = 0; f < rm.face_count(); f++) {
        int nk, nd;
        if (!straddles(f, nk, nd) || cut_piece[root(rm.faces[rm.face_off[f]])]) continue;
        u.clear();
        for (uint32_t k = rm.face_off[f]; k < rm.face_off[f + 1]; k++) {
            const uint32_t i = rm.faces[k], p = partner[i];
            u.push_back(side[i] >= 0 ? (uint64_t)i * 2 : (uint64_t)p * 2 + (side[p] != 0));
        }
        std::sort(u.begin(), u.end());
        if (std::adjacent_find(u.begin(), u.end()) != u.end()) cut_piece[root(rm.faces[rm.face_off[f]])] = 1;
    }

    // The kept half, into `out`: kept and seam vertices as they are (seam ones onto x = 0),
    // a far-side corner as its partner's reflection, seam crossings of cut faces new.
    RetopoMesh out;
    std::vector<uint32_t> poly;
    int paired = 0, cut = 0, layered = 0;
    for (;;) {
        out = RetopoMesh{};
        layered = 0;
        std::vector<uint32_t> ik(nv, ~0u), ir(nv, ~0u), cross_v;
        std::unordered_map<uint64_t, uint32_t> crossing;
        auto push = [&](Vec3 p, Vec3 n, uint32_t owner, float mk) {
            out.v.push_back(p); out.n.push_back(n); out.v_owner.push_back(owner);
            out.mirror.push_back(~0u); out.mask.push_back(mk);
            return (uint32_t)out.v.size() - 1;
        };
        auto K = [&](uint32_t i) {
            if (ik[i] == ~0u) {
                Vec3 p = rm.v[i];
                if (side[i] == 0) p.x = 0.0f;
                ik[i] = push(p, rm.n[i], rm.v_owner[i], rm.mask_at(i));
                if (side[i] == 0) out.mirror[ik[i]] = ik[i];
            }
            return ik[i];
        };
        auto R = [&](uint32_t i) {
            if (side[i] == 0) return K(i);
            if (ir[i] == ~0u) {
                const uint32_t a = K(i);
                ir[i] = push(refl(rm.v[i]), refl(rm.n[i]), rm.v_owner[i], rm.mask_at(i));
                out.mirror[a] = ir[i];
                out.mirror[ir[i]] = a;
            }
            return ir[i];
        };
        auto X = [&](uint32_t a, uint32_t b, uint32_t owner) {
            uint32_t& c = crossing.emplace(edge_key(a, b), ~0u).first->second;
            if (c == ~0u) {
                const float t = rm.v[a].x / (rm.v[a].x - rm.v[b].x);
                Vec3 p = rm.v[a] + (rm.v[b] - rm.v[a]) * t;
                p.x = 0.0f;
                c = push(p, (rm.n[a] + (rm.n[b] - rm.n[a]) * t).normalized(), owner,
                         rm.mask_at(a) + (rm.mask_at(b) - rm.mask_at(a)) * t);
                out.mirror[c] = c;
                cross_v.push_back(c);
            }
            return c;
        };
        FaceList half;
        std::vector<uint32_t> half_piece;   // per half face, its piece (root)
        paired = cut = 0;
        for (size_t f = 0; f < rm.face_count(); f++) {
            int nk, nd;
            straddles(f, nk, nd);
            if (!nk) continue;   // wholly on the far side or the seam: the reflection replaces it
            const uint32_t* c = rm.face(f);
            const int n = (int)rm.face_size(f);
            const uint32_t who = rm.face_owner[f];
            poly.clear();
            if (!nd || !cut_piece[root(c[0])]) {
                for (int k = 0; k < n; k++) poly.push_back(side[c[k]] >= 0 ? K(c[k]) : R(partner[c[k]]));
                paired += nd > 0;
                half.add(poly.data(), n, who);
                half_piece.push_back(root(c[0]));
            } else {
                for (int k = 0; k < n; k++) {
                    const uint32_t a = c[k], b = c[(k + 1) % n];
                    if (side[a] >= 0) poly.push_back(K(a));
                    if ((side[a] > 0 && side[b] < 0) || (side[a] < 0 && side[b] > 0)) poly.push_back(X(a, b, who));
                }
                cut++;
                if (poly.size() >= 3) emit_piece(out, half, poly.data(), (int)poly.size(), who, true);
                half_piece.resize(half.owner.size(), root(c[0]));
            }
        }
        if (half.owner.empty()) return "Nothing on that side to keep";

        // Crossings onto the sculpt, staying on x = 0.
        if (!cross_v.empty() && m.tri_count()) {
            Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
            for (uint32_t c : cross_v) {
                const Vec3 p = out.v[c];
                lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
                hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
            }
            const float pad = edge * 2.0f;
            lo = lo - Vec3(pad, pad, pad);
            hi = hi + Vec3(pad, pad, pad);
            const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
            SnapGrid g;
            g.build(m, lo, hi, std::max(edge * 0.5f, ext / 96.0f));
            for (uint32_t c : cross_v) {
                Vec3 q, qn;
                if (g.closest(out.v[c], q, qn, 8)) { out.v[c] = q; out.v[c].x = 0.0f; out.n[c] = qn; }
            }
        }

        // Every face and its reflection (wound the other way), each once: a face across the
        // seam that is its own reflection, or two that were each other's, come out once.
        for (size_t i = 0; i < out.v.size(); i++)
            if (out.mirror[i] == ~0u) {
                // A kept vertex no far-side corner was mapped onto: its reflection is new.
                const uint32_t a = (uint32_t)i;
                const uint32_t b = push(refl(out.v[a]), refl(out.n[a]), out.v_owner[a], out.mask[a]);
                out.mirror[a] = b;
                out.mirror[b] = a;
            }
        std::set<std::vector<uint32_t>> seen;
        std::vector<uint32_t> out_piece;
        auto add = [&](const uint32_t* p, int n, uint32_t who, uint32_t piece) {
            std::vector<uint32_t> u(p, p + n);
            std::sort(u.begin(), u.end());
            if (seen.insert(u).second) { out.add_face(p, n, who); out_piece.push_back(piece); }
        };
        for (size_t f = 0; f < half.owner.size(); f++) {
            const uint32_t* p = half.faces.data() + half.off[f];
            const int n = (int)(half.off[f + 1] - half.off[f]);
            add(p, n, half.owner[f], half_piece[f]);
            poly.assign(p, p + n);
            for (uint32_t& v : poly) v = out.mirror[v];
            std::reverse(poly.begin(), poly.end());
            add(poly.data(), n, half.owner[f], half_piece[f]);
        }

        // A face whose every corner is a corner of a bigger face is a layer lying on it (a
        // triangle over half a quad, a seam face built again as its own twin: the 2026-10-25
        // save had both): it goes. A mesh in that state can't be symmetrized cleanly.
        {
            std::vector<std::vector<uint32_t>> at(out.v.size());
            for (uint32_t f = 0; f < out.face_count(); f++)
                for (uint32_t k = 0; k < out.face_size(f); k++) at[out.face(f)[k]].push_back(f);
            auto corner_of = [&](uint32_t g, uint32_t v) {
                for (uint32_t k = 0; k < out.face_size(g); k++) if (out.face(g)[k] == v) return true;
                return false;
            };
            FaceList kept;
            std::vector<uint32_t> kept_piece;
            for (uint32_t f = 0; f < out.face_count(); f++) {
                const uint32_t* p = out.face(f);
                const uint32_t n = out.face_size(f);
                bool layer = false;
                for (uint32_t g : at[p[0]]) {
                    if (g == f || out.face_size(g) <= n) continue;
                    bool all = true;
                    for (uint32_t k = 1; k < n && all; k++) all = corner_of(g, p[k]);
                    if (all) { layer = true; break; }
                }
                if (layer) { layered++; continue; }
                kept.add(p, (int)n, out.face_owner[f]);
                kept_piece.push_back(out_piece[f]);
            }
            kept.swap_into(out);
            out_piece.swap(kept_piece);
        }

        // Pairing by nearest reflection can match corners that aren't really mirrors, and
        // then a face lands on another: any piece with a side run twice the same way or under
        // three faces is done again, cut at x = 0. A piece already cut stays as it came out
        // (what's left is the mesh's own, the audit's to report).
        std::unordered_map<uint64_t, int> dir, und;
        for (uint32_t f = 0; f < out.face_count(); f++)
            for (uint32_t k = 0, n = out.face_size(f); k < n; k++) {
                const uint32_t x = out.face(f)[k], y = out.face(f)[(k + 1) % n];
                dir[((uint64_t)x << 32) | y]++;
                und[edge_key(x, y)]++;
            }
        bool again = false;
        for (uint32_t f = 0; f < out.face_count(); f++)
            for (uint32_t k = 0, n = out.face_size(f); k < n; k++) {
                const uint32_t x = out.face(f)[k], y = out.face(f)[(k + 1) % n];
                if ((dir[((uint64_t)x << 32) | y] > 1 || und[edge_key(x, y)] > 2) && !cut_piece[out_piece[f]])
                    cut_piece[out_piece[f]] = 1, again = true;
            }
        if (!again) break;
    }
    rebuild_edges(out);

    // Islands: one that lost every face goes (it was all on the far side, its reflection
    // came from the kept one), the rest are geometry now.
    std::vector<uint32_t> before(rm.face_owner), after(out.face_owner);
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    const uint32_t n_before = rm.face_count();
    st.checkpoint();
    st.strokes.erase(std::remove_if(st.strokes.begin(), st.strokes.end(), [&](const RetopoStroke& k) {
        return std::binary_search(before.begin(), before.end(), k.id)
               && !std::binary_search(after.begin(), after.end(), k.id);
    }), st.strokes.end());
    for (RetopoStroke& k : st.strokes)
        if (std::binary_search(after.begin(), after.end(), k.id)) k.frozen = true;
    st.result = std::move(out);
    st.loop_edges.clear();
    st.path_a = st.path_b = ~0u;
    faces_out = (int)st.result.face_count();
    std::printf("[retopo] symmetrize: kept %s, %u -> %d faces, %d across the seam paired, %d cut at x = 0, "
                "%d layered face(s) dropped\n", keep < 0 ? "-X" : "+X", n_before, faces_out, paired, cut, layered);
    return nullptr;
}

void retopo_load(RetopoState& st, RetopoSnapshot&& s) {
    const bool show_result = st.show_result, show_strokes = st.show_strokes;
    const uint64_t rev = st.rev + 1;
    st = RetopoState{};
    st.show_result = show_result;
    st.show_strokes = show_strokes;
    st.rev = rev;
    st.strokes = std::move(s.strokes);
    st.result = std::move(s.result);
    // Ids are never reused: past every stroke and every owner, whatever the file said.
    uint32_t next = std::max<uint32_t>(1, s.next_id);
    for (const RetopoStroke& k : st.strokes) next = std::max(next, k.id + 1);
    for (uint32_t o : st.result.face_owner) next = std::max(next, o + 1);
    st.next_id = next;
    st.built = s.built && st.result.face_count() > 0;
    if (st.built) rebuild_edges(st.result);
    else          st.result.clear();
    if (st.built && s.uv.fits(st.result)) st.uv = std::move(s.uv);
}

void retopo_save(RetopoState& st, RetopoSnapshot& out) {
    // A plain wand preview isn't work to save; a spine's fill is.
    retopo_wand_close(st, st.wand_keep_shown ? st.wand_shown : -1);
    out.strokes = st.strokes;
    out.result  = st.result;
    out.built   = st.built;
    out.next_id = st.next_id;
    out.uv      = st.uv;
}

void retopo_smooth_normals(const RetopoMesh& rm, std::vector<Vec3>& out) {
    out.assign(rm.v.size(), Vec3());
    for (uint32_t f = 0; f < rm.face_count(); f++) {
        const uint32_t* q = rm.face(f);
        const uint32_t k = rm.face_size(f);
        Vec3 fn;   // Newell: right for a quad that isn't quite flat
        for (uint32_t j = 0; j < k; j++) fn += rm.v[q[j]].cross(rm.v[q[(j + 1) % k]]);
        fn = fn.normalized();
        for (uint32_t j = 0; j < k; j++) {
            const Vec3 a = (rm.v[q[(j + k - 1) % k]] - rm.v[q[j]]).normalized();
            const Vec3 b = (rm.v[q[(j + 1) % k]] - rm.v[q[j]]).normalized();
            const float c = std::max(-1.0f, std::min(1.0f, a.dot(b)));
            out[q[j]] += fn * std::acos(c);
        }
    }
    for (Vec3& n : out) n = n.normalized();
}

bool retopo_export_obj(const RetopoMesh& rm, const char* path, const UvLayout* uv) {
    if (rm.faces.empty()) return false;
    if (uv && !uv->fits(rm)) uv = nullptr;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "# Chisel retopo: %u faces%s\n", rm.face_count(), uv ? ", UVs" : "");
    // Welds leave folded-away vertices behind: write only the ones a quad uses.
    std::vector<uint32_t> id(rm.v.size(), 0);
    for (uint32_t i : rm.faces) id[i] = 1;
    uint32_t next = 0;
    for (size_t i = 0; i < rm.v.size(); i++) {
        if (!id[i]) continue;
        id[i] = ++next;
        std::fprintf(f, "v %.6f %.6f %.6f\n", rm.v[i].x, rm.v[i].y, rm.v[i].z);
    }
    // Normals the bake used, one per written vertex (vn k = v k): left to the importer,
    // each app smooths its own way and a baked normal map shades wrong on top.
    std::vector<Vec3> vn;
    retopo_smooth_normals(rm, vn);
    for (size_t i = 0; i < rm.v.size(); i++)
        if (id[i]) std::fprintf(f, "vn %.6f %.6f %.6f\n", vn[i].x, vn[i].y, vn[i].z);
    // One vt per face corner, in corner order: corner k is vt k + 1. Corners that share
    // a vertex inside one island repeat the same value; OBJ readers weld those.
    if (uv)
        for (size_t k = 0; k < rm.faces.size(); k++) std::fprintf(f, "vt %.6f %.6f\n", uv->uv[2 * k], uv->uv[2 * k + 1]);
    std::fprintf(f, "s 1\n");
    for (size_t q = 0; q < rm.face_count(); q++) {
        std::fprintf(f, "f");
        for (uint32_t k = rm.face_off[q]; k < rm.face_off[q + 1]; k++) {
            const uint32_t v = id[rm.faces[k]];
            if (uv) std::fprintf(f, " %u/%u/%u", v, k + 1, v);
            else    std::fprintf(f, " %u//%u", v, v);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return true;
}

// ---- UV seams ----------------------------------------------------------------------

void retopo_seam_edges(const RetopoMesh& rm, std::vector<uint32_t>& out) {
    out.clear();
    const uint32_t nv = (uint32_t)rm.v.size();
    if (rm.nb_off.size() != nv + 1) return;
    for (size_t i = 0; i + 1 < rm.seams.size(); i += 2) {
        const uint32_t a = rm.seams[i], b = rm.seams[i + 1];
        if (a >= nv || b >= nv) continue;
        bool edge = false;
        for (uint32_t k = rm.nb_off[a]; k < rm.nb_off[a + 1] && !edge; k++) edge = rm.nb[k] == b;
        if (edge) { out.push_back(a); out.push_back(b); }
    }
}

int retopo_seam_path(RetopoState& st, uint32_t from, uint32_t to, bool erase, bool mirror) {
    RetopoMesh& rm = st.result;
    const uint32_t nv = (uint32_t)rm.v.size();
    if (!st.built || from >= nv || to >= nv || from == to || rm.nb_off.size() != nv + 1) return 0;
    // The shortest way over the edges (Dijkstra by length), as the path pick does. Brush
    // samples are a few pixels apart, so this is usually one or two edges.
    std::vector<float> dist(nv, 1e30f);
    std::vector<uint32_t> prev(nv, ~0u);
    using QE = std::pair<float, uint32_t>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> q;
    dist[from] = 0.0f;
    q.push({0.0f, from});
    while (!q.empty()) {
        const auto [d, u] = q.top();
        q.pop();
        if (d > dist[u]) continue;
        if (u == to) break;
        for (uint32_t k = rm.nb_off[u]; k < rm.nb_off[u + 1]; k++) {
            const uint32_t w = rm.nb[k];
            if (st.hidden(rm.v_owner[w])) continue;
            const float nd = d + (rm.v[w] - rm.v[u]).length();
            if (nd < dist[w]) { dist[w] = nd; prev[w] = u; q.push({nd, w}); }
        }
    }
    if (prev[to] == ~0u) return 0;
    std::vector<uint64_t> have;
    for (size_t i = 0; i + 1 < rm.seams.size(); i += 2) have.push_back(edge_key(rm.seams[i], rm.seams[i + 1]));
    std::vector<uint64_t> path;
    std::vector<uint32_t> touched;
    for (uint32_t u = to; prev[u] != ~0u; u = prev[u]) {
        path.push_back(edge_key(prev[u], u));
        touched.push_back(rm.v_owner[u]);
        touched.push_back(rm.v_owner[prev[u]]);
        // Mirror: the twin edge too, when both ends have twins (on the x = 0 seam a
        // vertex is its own twin, so an edge along it is its own).
        const uint32_t ta = mirror && u < rm.mirror.size() ? rm.mirror[u] : ~0u;
        const uint32_t tb = mirror && prev[u] < rm.mirror.size() ? rm.mirror[prev[u]] : ~0u;
        if (ta < nv && tb < nv && ta != tb) { path.push_back(edge_key(ta, tb)); touched.push_back(rm.v_owner[ta]); }
    }
    int changed = 0;
    for (uint64_t k : path) {
        const auto it = std::lower_bound(have.begin(), have.end(), k);
        const bool there = it != have.end() && *it == k;
        if (erase && there)        { have.erase(it); changed++; }
        else if (!erase && !there) { have.insert(it, k); changed++; }
    }
    if (!changed) return 0;
    rm.seams.clear();
    for (uint64_t k : have) { rm.seams.push_back((uint32_t)(k >> 32)); rm.seams.push_back((uint32_t)k); }
    freeze_touched(st, touched);
    return changed;
}

bool retopo_seams_clear(RetopoState& st) {
    if (st.result.seams.empty()) return false;
    st.checkpoint();
    st.result.seams.clear();
    return true;
}

// ---- Copy / paste ------------------------------------------------------------------
//
// The user's "horn ring" (2026-10-26): copy a donut off one horn, lasso the next horn,
// paste, and the ring drops onto it. The fit is a pose match, not a shape match: the
// sculpt under the copy and the sculpt in the lasso are each measured the same way
// (retopo_measure: centre, axis, spread), the piece rides from one pose to the other,
// then snaps onto the surface and is smoothed. What the pose can't know (twist, how high
// up the horn, a lasso drawn loose) is the panel's Scale / Twist / Offset.

bool retopo_measure(const std::vector<Vec3>& p, const std::vector<Vec3>& n, Vec3& c, Vec3& a, float& size) {
    double W = 0.0;
    Vec3 cs, ns;
    for (size_t i = 0; i < p.size(); i++) {
        const float w = n[i].length();
        cs += p[i] * w;
        ns += n[i];
        W += w;
    }
    // Facing every way at once (a whole closed shell) has no axis.
    if (W <= 0.0 || ns.length() < 0.05f * (float)W) return false;
    c = cs * (float)(1.0 / W);
    a = ns.normalized();
    double s2 = 0.0;
    for (size_t i = 0; i < p.size(); i++) {
        Vec3 d = p[i] - c;
        d = d - a * d.dot(a);
        s2 += (double)n[i].length() * d.dot(d);
    }
    size = (float)std::sqrt(s2 / W);
    return size > 0.0f;
}

namespace {
// v turned by angle about unit axis k (Rodrigues).
Vec3 turn_about(Vec3 v, Vec3 k, float angle) {
    const float c = std::cos(angle), s = std::sin(angle);
    return v * c + k.cross(v) * s + k * (k.dot(v) * (1.0f - c));
}

// v turned by the smallest rotation that takes unit a onto unit b.
Vec3 turn_onto(Vec3 v, Vec3 a, Vec3 b) {
    Vec3 k = a.cross(b);
    const float s = k.length(), c = a.dot(b);
    if (s < 1e-6f) {
        if (c > 0.0f) return v;
        // Opposite: half a turn about any line square to a.
        k = a.cross(std::fabs(a.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0)).normalized();
        return turn_about(v, k, 3.14159265f);
    }
    return turn_about(v, k * (1.0f / s), std::atan2(s, c));
}

// The clip placed on the target with the panel's settings, as a new island (paste_id),
// onto `m`, smoothed; its twin with mirror. st.result is whatever it's pasted into.
void paste_apply(RetopoState& st, const Mesh& m) {
    const RetopoState::Clip& C = st.clip;
    RetopoMesh& rm = st.result;
    // Across x = 0 from where it was copied: its mirror image, not a turn (user's call).
    const bool flip = C.c.x * st.paste_c.x < 0.0f && std::fabs(C.c.x) > 0.25f * C.size
                   && std::fabs(st.paste_c.x) > 0.25f * st.paste_size;
    auto refl = [&](Vec3 v) { if (flip) v.x = -v.x; return v; };
    const Vec3 cs = refl(C.c), as = refl(C.a), at = st.paste_a;
    const float k = st.paste_scale * st.paste_size / std::max(C.size, 1e-9f);
    const float tw = st.paste_twist * 3.14159265f / 180.0f;
    const Vec3 ct = st.paste_c + at * (st.paste_offset * st.paste_size);

    const uint32_t v0 = (uint32_t)rm.v.size(), f0 = rm.face_count(), id = st.paste_id;
    for (size_t i = 0; i < C.v.size(); i++) {
        const Vec3 d = turn_about(turn_onto(refl(C.v[i]) - cs, as, at), at, tw) * k;
        rm.v.push_back(ct + d);
        rm.n.push_back(turn_about(turn_onto(refl(C.n[i]), as, at), at, tw));
        rm.v_owner.push_back(id);
        rm.mirror.push_back(~0u);
    }
    std::vector<uint32_t> q;
    for (size_t f = 0; f + 1 < C.face_off.size(); f++) {
        q.assign(C.faces.begin() + C.face_off[f], C.faces.begin() + C.face_off[f + 1]);
        for (uint32_t& x : q) x += v0;
        if (flip) std::reverse(q.begin(), q.end());   // a mirror image winds the other way
        rm.add_face(q.data(), (int)q.size(), id);
    }

    // Onto the sculpt: a grid over where it landed, then the snap, then a few passes of
    // Smooth's own step (the target is never quite the same shape), snapping each.
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (uint32_t i = v0; i < rm.v.size(); i++) {
        const Vec3& p = rm.v[i];
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    float edge = 0.0f;
    int ne = 0;
    for (uint32_t f = f0; f < rm.face_count(); f++)
        for (uint32_t j = 0; j < rm.face_size(f); j++, ne++)
            edge += (rm.v[rm.face(f)[j]] - rm.v[rm.face(f)[(j + 1) % rm.face_size(f)]]).length();
    edge /= (float)std::max(1, ne);
    const float pad = std::max(edge * 2.0f, 0.5f * st.paste_size);
    lo = lo - Vec3(pad, pad, pad);
    hi = hi + Vec3(pad, pad, pad);
    const float ext = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    SnapGrid g;
    g.build(m, lo, hi, std::max(edge * 0.5f, ext / 96.0f));
    int lost = 0;
    for (uint32_t i = v0; i < rm.v.size(); i++) {
        Vec3 p, pn;
        if (g.closest(rm.v[i], p, pn, 16)) { rm.v[i] = p; rm.n[i] = pn; }
        else lost++;
    }
    rebuild_edges(rm);
    std::vector<Vec3> old;
    for (int pass = 0; pass < 4; pass++) {
        old = rm.v;
        for (uint32_t i = v0; i < rm.v.size(); i++) {
            if (rm.pinned[i] || !alive(rm, i)) continue;
            Vec3 p, pn;
            if (g.closest(old[i] + relax_dir(rm, old, i) * 0.5f, p, pn)) { rm.v[i] = p; rm.n[i] = pn; }
        }
    }

    // Mirror: the twin, when the piece sits wholly on one side of x = 0.
    bool pos = false, neg = false;
    for (uint32_t i = v0; i < rm.v.size(); i++) { pos |= rm.v[i].x > 1e-6f; neg |= rm.v[i].x < -1e-6f; }
    const bool twin = st.paste_mirror && pos != neg;
    if (twin) {
        std::vector<uint32_t> touched;
        reflect_new_faces(st, v0, f0, id, touched);
        rebuild_edges(rm);
    }

    RetopoStroke s;
    s.id = id;
    s.frozen = true;
    s.closed = C.closed;
    s.name = C.name + " copy";
    for (RetopoStroke& o : st.strokes) o.selected = false;
    s.selected = true;
    st.strokes.push_back(s);
    st.built = true;
    st.rev++;
    std::printf("[retopo] paste: %zu faces, scale %.2f (x%.2f), twist %.0f, offset %.2f%s%s%s\n",
                C.face_off.size() - 1, k, st.paste_scale, st.paste_twist, st.paste_offset,
                flip ? ", mirrored across x = 0" : "", twin ? ", twin too" : "",
                lost ? ", some vertices found no surface" : "");
}
}  // namespace

const char* retopo_copy(RetopoState& st, const Mesh& m, uint32_t f) {
    const RetopoMesh& rm = st.result;
    if (!st.built || f >= rm.face_count()) return "Right-click on an island to copy it";
    // The piece: faces joined through shared vertices (union-find over corners).
    std::vector<uint32_t> up(rm.v.size());
    for (uint32_t i = 0; i < up.size(); i++) up[i] = i;
    auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
    for (uint32_t k = 0; k < rm.face_count(); k++)
        for (uint32_t j = 1; j < rm.face_size(k); j++) up[root(rm.face(k)[j])] = root(rm.face(k)[0]);
    const uint32_t r0 = root(rm.face(f)[0]);

    RetopoState::Clip C;
    std::vector<uint32_t> local(rm.v.size(), ~0u);
    Vec3 axis;
    for (uint32_t k = 0; k < rm.face_count(); k++) {
        if (root(rm.face(k)[0]) != r0) continue;
        for (uint32_t j = 0; j < rm.face_size(k); j++) {
            const uint32_t v = rm.face(k)[j];
            if (local[v] == ~0u) {
                local[v] = (uint32_t)C.v.size();
                C.v.push_back(rm.v[v]);
                C.n.push_back(rm.n[v]);
                axis += rm.n[v];
            }
            C.faces.push_back(local[v]);
        }
        C.face_off.push_back((uint32_t)C.faces.size());
    }
    if (axis.length() < 1e-6f) return "That piece faces every way at once: nothing to fit it by";
    axis = axis.normalized();

    // Its outer outline (the piece's one rim that isn't a hole), flat in the plane
    // square to the piece's axis: what the sculpt is measured inside.
    const Vec3 u = axis.cross(std::fabs(axis.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0)).normalized();
    const Vec3 w = axis.cross(u);
    std::vector<float> ox, oy;
    for (uint32_t r = 0; r < rm.rim_count() && ox.empty(); r++) {
        if (rm.rim_hole[r] || root(rm.rim[rm.rim_off[r]]) != r0) continue;
        for (uint32_t k = rm.rim_off[r]; k < rm.rim_off[r + 1]; k++) {
            ox.push_back(rm.v[rm.rim[k]].dot(u));
            oy.push_back(rm.v[rm.rim[k]].dot(w));
        }
    }
    if (ox.size() < 3) return "That piece has no outline to copy by";
    float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f, h0 = 0.0f, reach = 0.0f;
    for (size_t i = 0; i < ox.size(); i++) {
        x0 = std::min(x0, ox[i]); x1 = std::max(x1, ox[i]);
        y0 = std::min(y0, oy[i]); y1 = std::max(y1, oy[i]);
    }
    for (const Vec3& p : C.v) h0 += p.dot(axis);
    h0 /= (float)C.v.size();
    reach = 4.0f * std::max(x1 - x0, y1 - y0);   // how far along the axis the sculpt counts
    std::vector<Vec3> P, N;
    for (uint32_t t = 0; t < m.tri_count(); t++) {
        const Vec3 a = m.get_pos(m.indices[t * 3]), b = m.get_pos(m.indices[t * 3 + 1]),
                   c = m.get_pos(m.indices[t * 3 + 2]);
        const Vec3 n = (b - a).cross(c - a), mid = (a + b + c) * (1.0f / 3.0f);
        if (n.dot(axis) <= 0.0f || std::fabs(mid.dot(axis) - h0) > reach) continue;
        const float x = mid.dot(u), y = mid.dot(w);
        if (x < x0 || x > x1 || y < y0 || y > y1) continue;
        bool in = false;
        for (size_t i = 0, j = ox.size() - 1; i < ox.size(); j = i++)
            if ((oy[i] > y) != (oy[j] > y) && x < ox[j] + (ox[i] - ox[j]) * (y - oy[j]) / (oy[i] - oy[j])) in = !in;
        if (in) { P.push_back(mid); N.push_back(n); }
    }
    if (!retopo_measure(P, N, C.c, C.a, C.size)) return "No sculpt under that piece to measure it by";
    const RetopoStroke* s = st.find(rm.face_owner[f]);
    C.name = s ? s->name : "Island";
    C.closed = s && s->closed;
    std::printf("[retopo] copy: %s, %zu faces, %zu verts; sculpt under it %zu tris, axis (%.2f,%.2f,%.2f), size %.3f\n",
                C.name.c_str(), C.face_off.size() - 1, C.v.size(), P.size(), C.a.x, C.a.y, C.a.z, C.size);
    st.clip = std::move(C);
    return nullptr;
}

const char* retopo_paste(RetopoState& st, const Mesh& m, bool mirror) {
    if (!st.has_clip()) return "Nothing copied: right-click an island with the wand, Copy island";
    if (!st.paste_ready) return "Lasso where it goes first: drag the wand round bare sculpt";
    st.checkpoint();   // one Ctrl+Z takes the paste back, sliders and all
    st.paste_base = snapshot_of(st);
    st.paste_id = st.next_id++;
    st.paste_scale = 1.0f;
    st.paste_twist = st.paste_offset = 0.0f;
    st.paste_mirror = mirror;
    st.paste_open = true;
    paste_apply(st, m);
    st.paste_ready = false;
    st.paste_lasso.clear();
    return nullptr;
}

void retopo_paste_redo(RetopoState& st, const Mesh& m) {
    if (!st.paste_open) return;
    put(st, st.paste_base);
    paste_apply(st, m);
}

void retopo_paste_close(RetopoState& st) {
    st.paste_open = false;
    st.paste_base = RetopoSnapshot{};
}

namespace {
// The lasso grid onto `result` (see retopo_grid): st.grid_pts with st.grid_sides round
// it. Fill's recipe on an outline that isn't there yet: ring, corners, Coons, aim, relax.
const char* grid_apply(RetopoState& st, const Mesh& m) {
    RetopoMesh& rm = st.result;
    const std::vector<Vec3>& B = st.grid_pts;
    const size_t nb = B.size();
    const int N = st.grid_sides, h = N / 2;
    if (nb < 3 || N < 8 || N % 2) return "That lasso is too small for a grid";
    auto md = [&](int k) { return ((k % N) + N) % N; };

    // The outline's facing (its area normal, turned to agree with the sculpt's normals
    // under it), centre and size: the inside is aimed down this, as Fill aims down a rim's.
    Vec3 nw(0, 0, 0), ns(0, 0, 0), cen(0, 0, 0);
    for (size_t k = 0; k < nb; k++) {
        const Vec3 a = B[k], b = B[(k + 1) % nb];
        nw = nw + Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
        ns = ns + st.grid_nrm[k];
        cen = cen + a;
    }
    cen = cen * (1.0f / (float)nb);
    float diam = 0.0f;
    for (const Vec3& p : B) diam = std::max(diam, 2.0f * (p - cen).length());
    if (nw.length() < 1e-12f) nw = ns;
    if (nw.dot(ns) < 0.0f) nw = nw * -1.0f;
    if (nw.length() < 1e-12f || diam < 1e-9f) return "No surface under that lasso";
    const Vec3 nf = nw.normalized();

    // N vertices evenly round the outline, by length.
    std::vector<float> acc(nb + 1, 0.0f);
    for (size_t k = 0; k < nb; k++) acc[k + 1] = acc[k] + (B[(k + 1) % nb] - B[k]).length();
    const float per = acc[nb], edge = per / (float)N;
    std::vector<Vec3> O(N), On(N);
    for (int k = 0, seg = 0; k < N; k++) {
        const float t = per * (float)k / (float)N;
        while (seg + 1 < (int)nb && acc[seg + 1] < t) seg++;
        const float sl = acc[seg + 1] - acc[seg];
        const float f = sl > 1e-12f ? (t - acc[seg]) / sl : 0.0f;
        O[k] = B[seg] + (B[(seg + 1) % nb] - B[seg]) * f;
    }

    // The snap grid round it, deep enough for the aim.
    SnapGrid g;
    {
        Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
        auto grow = [&](Vec3 p) {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        };
        for (const Vec3& p : B) {
            grow(p);
            grow(p + nf * (kGridAbove * diam));
            grow(p - nf * (kFillBelow * diam));
        }
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
    // Down the facing from above, onto the first surface that faces back. Higher than
    // Fill starts: a hole's inside sits below its rim, but a lasso's inside on a bulge
    // (a chest, a skull) sits above its outline's chords.
    auto aim = [&](Vec3 p, Vec3& nn) {
        Vec3 hit, hn;
        const float up = kGridAbove * diam;
        if (g.ray(p + nf * up, nf * -1.0f, up + kFillBelow * diam, hit, hn)) { nn = hn; return hit; }
        return snap(p, nn);
    };
    for (int k = 0; k < N; k++) O[k] = snap(O[k], On[k]);

    // The ring: one loop of quads just inside the outline, so the grid's four corners
    // sit inside and the outline itself stays one smooth loop. Each vertex steps in a
    // quad's length along the mitred inward direction (the mean of its two sides'
    // normals, longer at a corner: a drawn rectangle's corner quads folded without it).
    // A shape too narrow for a ring and a grid side by side (a thin oval at a low count)
    // gets the grid alone: the ring squeezed it flat.
    std::vector<Vec3> sn(N);   // each side's inward normal, along the surface
    float wind = 0.0f;
    for (int k = 0; k < N; k++) {
        sn[k] = nf.cross(O[md(k + 1)] - O[k]).normalized();
        wind += sn[k].dot(cen - (O[k] + O[md(k + 1)]) * 0.5f);
    }
    if (wind < 0.0f) for (Vec3& v : sn) v = v * -1.0f;
    float room = 1e30f;
    for (int k = 0; k < N; k++) {
        Vec3 off = cen - O[k];
        off = off - nf * off.dot(nf);
        room = std::min(room, 0.45f * off.length());
    }
    const bool ring = N >= kFillRingMin && room >= 0.75f * edge;
    std::vector<Vec3> L = O, Ln = On;
    if (ring)
        for (int k = 0; k < N; k++) {
            const Vec3 p = O[k];
            Vec3 in = (sn[md(k - 1)] + sn[k]).normalized();
            const float miter = 1.0f / std::max(0.5f, in.dot(sn[k]));
            Vec3 off = cen - p;
            off = off - nf * off.dot(nf);
            const float step = std::min(edge * miter, 0.45f * off.length());
            Vec3 q = p + in * step, nn;
            for (int it = 0; it < 3; it++) {
                q = snap(q, nn);
                const Vec3 d = q - p;
                if (d.length() < 1e-9f) break;
                q = p + d.normalized() * step;
            }
            L[k] = snap(q, Ln[k]);
        }
    // Then evened out along its own loop: at a sharp corner the corner and both its
    // neighbours step in onto one point, and round a tip tighter than a quad the steps
    // cross. Three plain passes untangle; then passes that only slide along the loop
    // space them out (plain ones would keep shrinking the ring into the grid).
    if (ring) {
        std::vector<Vec3> t(N);
        for (int pass = 0; pass < 3 + kGridRingSlides; pass++) {
            for (int k = 0; k < N; k++) {
                const Vec3 a = L[md(k - 1)], b = L[md(k + 1)];
                Vec3 d = (a + b) * 0.5f - L[k];
                if (pass >= 3) {
                    const Vec3 tg = (b - a).normalized();
                    d = tg * d.dot(tg);
                }
                t[k] = L[k] + d * 0.5f;
            }
            for (int k = 0; k < N; k++) L[k] = snap(t[k], Ln[k]);
        }
    }

    // Corners: four where the ring turns most, opposite sides the same count (m, n, m,
    // n), quads as near square as the shape allows. Not Fill's square test: that one
    // reads the spacing along a rim, and an outline resampled evenly is spaced alike
    // whatever the split, so a round lasso came out 11 x 1. Here it's the distance
    // across, side middle to opposite side middle, over the quads that span it.
    std::vector<float> turn(N), arc(N + 1, 0.0f);
    for (int k = 0; k < N; k++) {
        const Vec3 a = (L[k] - L[md(k - 1)]).normalized(), b = (L[md(k + 1)] - L[k]).normalized();
        turn[k] = 1.0f - a.dot(b);
        arc[k + 1] = arc[k] + (L[md(k + 1)] - L[k]).length();
    }
    auto span = [&](int a, int n) {
        const int b = a + n;
        return b <= N ? arc[b] - arc[a] : arc[N] - arc[a] + arc[b - N];
    };
    auto mid = [&](int a, int n) {   // the middle of the side from vertex a, n sides on
        return n % 2 ? (L[md(a + n / 2)] + L[md(a + n / 2 + 1)]) * 0.5f : L[md(a + n / 2)];
    };
    int bs = 0, bm = std::max(1, h / 2);
    float best = -1e30f;
    for (int s = 0; s < h; s++)
        for (int mm = 1; mm < h; mm++) {
            const int nn = h - mm;
            const int cs[4] = {s, md(s + mm), md(s + h), md(s + h + mm)};
            const float s1 = span(cs[0], mm), s2 = span(cs[1], nn), s3 = span(cs[2], mm), s4 = span(cs[3], nn);
            const float wide = (mid(cs[1], nn) - mid(cs[3], nn)).length();   // spanned by mm quads
            const float tall = (mid(cs[0], mm) - mid(cs[2], mm)).length();   // spanned by nn quads
            const float aspect = std::fabs(std::log((wide / mm) / std::max(1e-9f, tall / nn)));
            const float score = turn[cs[0]] + turn[cs[1]] + turn[cs[2]] + turn[cs[3]] - aspect
                              - 0.25f * (std::fabs(std::log(s1 / std::max(1e-9f, s3))) + std::fabs(std::log(s2 / std::max(1e-9f, s4))));
            if (score > best) { best = score; bs = s; bm = mm; }
        }
    const int M = bm, Nn = h - bm;

    const uint32_t id = st.grid_id, v0 = (uint32_t)rm.v.size(), f0 = rm.face_count();
    auto add_vertex = [&](Vec3 p, Vec3 nn) {
        rm.v.push_back(p);
        rm.n.push_back(nn.length() > 0.5f ? nn : nf);
        rm.v_owner.push_back(id);
        rm.mirror.push_back(~0u);
        return (uint32_t)rm.v.size() - 1;
    };
    std::vector<uint32_t> Oi(N), Li(N);
    for (int k = 0; k < N; k++) Oi[k] = add_vertex(O[k], On[k]);
    for (int k = 0; k < N; k++) Li[k] = ring ? add_vertex(L[k], Ln[k]) : Oi[k];

    // Grid (M + 1) x (Nn + 1), its outside the ring from corner bs round, as in Fill.
    const int GW = M + 1, GH = Nn + 1;
    std::vector<uint32_t> gi(GW * GH, ~0u);
    auto G = [&](int i, int j) -> uint32_t& { return gi[j * GW + i]; };
    for (int i = 0; i <= M; i++) { G(i, 0) = Li[md(bs + i)]; G(i, Nn) = Li[md(bs + h + (M - i))]; }
    for (int j = 0; j <= Nn; j++) { G(M, j) = Li[md(bs + M + j)]; G(0, j) = Li[md(bs + h + M + (Nn - j))]; }
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
    // Relax along the surface, then back onto it (Fill's). A bigger grid gets more
    // passes: Coons bunches quads at a round outline's corners, and a pass only reaches
    // one vertex further in.
    const int passes = std::clamp(std::max(M, Nn) / 2, 4, 16);
    std::vector<Vec3> tmp(GW * GH);
    for (int pass = 0; pass < passes && M > 1 && Nn > 1; pass++) {
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

    // Faces: the ring's quads, then the grid's. Ring quad k runs Li[k + 1] -> Li[k] on
    // its inner side, the grid's first row Li[bs] -> Li[bs + 1]: wound alike. Then all
    // turned over together if they face into the sculpt.
    std::vector<uint32_t> Q;
    if (ring)
        for (int k = 0; k < N; k++) Q.insert(Q.end(), {Oi[k], Oi[md(k + 1)], Li[md(k + 1)], Li[k]});
    for (int j = 0; j < Nn; j++)
        for (int i = 0; i < M; i++) Q.insert(Q.end(), {G(i, j), G(i + 1, j), G(i + 1, j + 1), G(i, j + 1)});
    float facing = 0.0f;
    for (size_t q = 0; q < Q.size(); q += 4) {
        const Vec3 a = rm.v[Q[q]], b = rm.v[Q[q + 1]], c = rm.v[Q[q + 2]];
        facing += (b - a).cross(c - a).dot(rm.n[Q[q]]);
    }
    for (size_t q = 0; q < Q.size(); q += 4) {
        if (facing < 0.0f) std::swap(Q[q + 1], Q[q + 3]);
        rm.add_face(&Q[q], 4, id);
    }
    rebuild_edges(rm);

    // Mirror: the twin, when the grid sits wholly on one side of x = 0 (as a paste).
    bool pos = false, neg = false;
    for (uint32_t i = v0; i < rm.v.size(); i++) { pos |= rm.v[i].x > 1e-6f; neg |= rm.v[i].x < -1e-6f; }
    const bool twin = st.grid_mirror && pos != neg;
    if (twin) {
        std::vector<uint32_t> touched;
        reflect_new_faces(st, v0, f0, id, touched);
        rebuild_edges(rm);
    }

    RetopoStroke s;
    s.id = id;
    s.frozen = true;
    s.closed = true;
    s.name = "Grid";
    for (RetopoStroke& o : st.strokes) o.selected = false;
    s.selected = true;
    st.strokes.push_back(s);
    st.built = true;
    st.rev++;
    std::printf("[retopo] grid: %d round, %d x %d inside%s, %u faces%s\n", N, M, Nn,
                ring ? " a ring" : "", rm.face_count() - f0, twin ? ", twin too" : "");
    return nullptr;
}
}  // namespace

const char* retopo_grid(RetopoState& st, const Mesh& m, const std::vector<Vec3>& pts,
                        const std::vector<Vec3>& nrm, int sides, bool mirror) {
    if (pts.size() < 3 || pts.size() != nrm.size()) return "No surface under that lasso";
    st.checkpoint();   // one Ctrl+Z takes the grid back, whatever the wheel did
    st.grid_base = snapshot_of(st);
    st.grid_id = st.next_id++;
    st.grid_pts = pts;
    st.grid_nrm = nrm;
    st.grid_sides = sides;
    st.grid_mirror = mirror;
    st.grid_wheel = 0.0f;
    st.grid_open = true;
    if (const char* why = grid_apply(st, m)) {
        retopo_grid_cancel(st);
        return why;
    }
    return nullptr;
}

void retopo_grid_redo(RetopoState& st, const Mesh& m, int sides) {
    if (!st.grid_open) return;
    st.grid_sides = sides;
    put(st, st.grid_base);
    if (grid_apply(st, m)) st.rev++;   // refused: the lasso's spot stays bare
}

void retopo_grid_close(RetopoState& st) {
    st.grid_open = false;
    st.grid_base = RetopoSnapshot{};
    st.grid_pts.clear();
    st.grid_nrm.clear();
    st.grid_wheel = 0.0f;
}

void retopo_grid_cancel(RetopoState& st) {
    if (!st.grid_open) return;
    put(st, st.grid_base);
    if (!st.undo_stack.empty()) st.undo_stack.pop_back();   // the grid's own checkpoint
    st.rev++;
    retopo_grid_close(st);
}
