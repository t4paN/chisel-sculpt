#include "uv_unwrap.h"
#include "xatlas/xatlas.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

uint64_t uv_topology_key(const RetopoMesh& rm) {
    // FNV-1a over the corner lists and the face boundaries: two meshes with the same
    // vertex ids in the same faces get the same key, whatever their positions.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint32_t x) { for (int i = 0; i < 4; i++) { h ^= (x >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
    for (uint32_t i : rm.faces) mix(i);
    for (uint32_t i : rm.face_off) mix(i);
    return h;
}

uint64_t uv_seam_key(const RetopoMesh& rm) {
    std::vector<uint32_t> e;
    retopo_seam_edges(rm, e);
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i : e) for (int b = 0; b < 4; b++) { h ^= (i >> (8 * b)) & 0xff; h *= 1099511628211ull; }
    return h;
}

// Ear-clip face f into triangles, as corner positions within the face (0..k-1), in the
// face's own plane (Newell normal). Quads are most of a retopo mesh, but the eraser
// merges faces into bigger, sometimes concave ones, and a fan would fold those.
static void triangulate(const RetopoMesh& rm, uint32_t f, std::vector<uint32_t>& out) {
    const uint32_t* c = rm.face(f);
    const uint32_t k = rm.face_size(f);
    Vec3 n;
    for (uint32_t j = 0; j < k; j++) {
        const Vec3& a = rm.v[c[j]];
        const Vec3& b = rm.v[c[(j + 1) % k]];
        n += Vec3((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    std::vector<uint32_t> ring(k);
    for (uint32_t j = 0; j < k; j++) ring[j] = j;
    auto convex = [&](uint32_t a, uint32_t b, uint32_t d) {
        return (rm.v[c[b]] - rm.v[c[a]]).cross(rm.v[c[d]] - rm.v[c[b]]).dot(n) > 0.0f;
    };
    auto inside = [&](uint32_t p, uint32_t a, uint32_t b, uint32_t d) {
        const Vec3& P = rm.v[c[p]];
        return (rm.v[c[b]] - rm.v[c[a]]).cross(P - rm.v[c[a]]).dot(n) >= 0.0f &&
               (rm.v[c[d]] - rm.v[c[b]]).cross(P - rm.v[c[b]]).dot(n) >= 0.0f &&
               (rm.v[c[a]] - rm.v[c[d]]).cross(P - rm.v[c[d]]).dot(n) >= 0.0f;
    };
    while (ring.size() > 3) {
        const size_t m = ring.size();
        size_t ear = m;
        for (size_t i = 0; i < m && ear == m; i++) {
            const uint32_t a = ring[(i + m - 1) % m], b = ring[i], d = ring[(i + 1) % m];
            if (!convex(a, b, d)) continue;
            bool clear = true;
            for (size_t j = 0; j < m && clear; j++) {
                const uint32_t p = ring[j];
                if (p != a && p != b && p != d && inside(p, a, b, d)) clear = false;
            }
            if (clear) ear = i;
        }
        if (ear == m) ear = 0;   // nothing clean (a twisted face): clip anyway, it still covers it
        out.push_back(ring[(ear + m - 1) % m]);
        out.push_back(ring[ear]);
        out.push_back(ring[(ear + 1) % m]);
        ring.erase(ring.begin() + ear);
    }
    out.insert(out.end(), ring.begin(), ring.end());
}

// One island laid flat by least-squares conformal maps: every triangle keeps its angles
// as well as it can, and the island's own border goes wherever that puts it. Two
// vertices are pinned (the pair farthest apart, on the border if there is one), the
// rest solved by conjugate gradients from a flat projection as the first guess.
// `tris` are triangle numbers into wv (3 vertex copies each); positions are pos[orig[copy]].
// Writes uv (2 per copy) for the island's copies, in mesh units.
struct LscmResult { bool closed = false; bool mirrored = false; uint32_t folds = 0; };

static LscmResult lscm(const std::vector<Vec3>& pos, const std::vector<uint32_t>& orig,
                       const std::vector<uint32_t>& wv, const std::vector<uint32_t>& tris, std::vector<float>& uv) {
    LscmResult res;
    if (tris.empty()) return res;
    // The island's copies, numbered 0..n-1.
    std::map<uint32_t, uint32_t> local;
    std::vector<uint32_t> copy;
    for (uint32_t t : tris)
        for (int j = 0; j < 3; j++)
            if (local.emplace(wv[3 * t + j], (uint32_t)copy.size()).second) copy.push_back(wv[3 * t + j]);
    const uint32_t n = (uint32_t)copy.size();
    auto P = [&](uint32_t i) { return pos[orig[copy[i]]]; };

    // Its border: edges only one triangle uses. A piece with none (closed) can't lie flat;
    // a ring or a tube can (as a flat ring, squeezed on the inside).
    std::map<uint64_t, int> edge_use;
    for (uint32_t t : tris)
        for (int j = 0; j < 3; j++) {
            const uint32_t a = local[wv[3 * t + j]], b = local[wv[3 * t + (j + 1) % 3]];
            edge_use[((uint64_t)std::min(a, b) << 32) | std::max(a, b)]++;
        }
    std::vector<uint8_t> border(n, 0);
    for (const auto& e : edge_use)
        if (e.second == 1) border[(uint32_t)(e.first >> 32)] = border[(uint32_t)e.first] = 1;

    // First guess: projected onto the island's mean plane, seen from the side its faces
    // face (so it isn't mirrored).
    Vec3 nsum, mid;
    for (uint32_t i = 0; i < n; i++) mid += P(i);
    mid = mid * (1.0f / (float)n);
    for (uint32_t t : tris) {
        const Vec3 a = pos[orig[wv[3 * t]]], b = pos[orig[wv[3 * t + 1]]], d = pos[orig[wv[3 * t + 2]]];
        nsum += (b - a).cross(d - a);
    }
    Vec3 nz = nsum.length() > 1e-20f ? nsum.normalized() : Vec3(0, 0, 1);
    Vec3 ax = std::fabs(nz.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    ax = (ax - nz * ax.dot(nz)).normalized();
    const Vec3 ay = nz.cross(ax);
    std::vector<double> x(2 * (size_t)n);   // u's, then v's
    for (uint32_t i = 0; i < n; i++) { x[i] = (P(i) - mid).dot(ax); x[n + i] = (P(i) - mid).dot(ay); }

    // Pins: the farthest pair (two sweeps), among border vertices if there are any.
    const bool any_border = std::find(border.begin(), border.end(), 1) != border.end();
    res.closed = !any_border;
    auto far_from = [&](uint32_t from) {
        uint32_t best = from; float bd = -1.0f;
        for (uint32_t i = 0; i < n; i++) {
            if (any_border && !border[i]) continue;
            const float d = (P(i) - P(from)).length();
            if (d > bd) { bd = d; best = i; }
        }
        return best;
    };
    uint32_t start = 0;
    for (uint32_t i = 0; i < n; i++) if (border[i]) { start = i; break; }
    const uint32_t p0 = far_from(start), p1 = far_from(p0);

    // Each triangle in its own plane: corners q0 = (0,0), q1 = (|e|,0), q2 above the
    // x axis. Conformal means sum_j W_j U_j = 0 (complex), W_j the edge opposite corner j
    // (q_{j+2} - q_{j+1}), U_j = u_j + i v_j; weighted by 1/sqrt(area) so each triangle
    // counts by its size. Two real rows per triangle.
    struct Row { uint32_t c[3]; double a[3], b[3]; };
    std::vector<Row> rows;
    rows.reserve(tris.size());
    for (uint32_t t : tris) {
        Row r;
        Vec3 Q[3];
        for (int j = 0; j < 3; j++) { r.c[j] = local[wv[3 * t + j]]; Q[j] = P(r.c[j]); }
        const Vec3 e1 = Q[1] - Q[0], e2 = Q[2] - Q[0];
        const double l1 = e1.length();
        const Vec3 nn = e1.cross(e2);
        if (l1 < 1e-12 || nn.length() < 1e-14f) continue;   // degenerate: no say
        const Vec3 xa = e1 * (float)(1.0 / l1), ya = nn.normalized().cross(xa);
        const double qx[3] = {0.0, l1, e2.dot(xa)}, qy[3] = {0.0, 0.0, e2.dot(ya)};
        const double s = 1.0 / std::sqrt(l1 * qy[2]);
        for (int j = 0; j < 3; j++) {
            const int a = (j + 1) % 3, b = (j + 2) % 3;
            r.a[j] = (qx[b] - qx[a]) * s;
            r.b[j] = (qy[b] - qy[a]) * s;
        }
        rows.push_back(r);
    }
    // M = AᵀA applied to y, as rows of A: re = sum a u - b v, im = sum b u + a v.
    auto apply = [&](const std::vector<double>& y, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (const Row& r : rows) {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < 3; j++) {
                re += r.a[j] * y[r.c[j]] - r.b[j] * y[n + r.c[j]];
                im += r.b[j] * y[r.c[j]] + r.a[j] * y[n + r.c[j]];
            }
            for (int j = 0; j < 3; j++) {
                out[r.c[j]]     += r.a[j] * re + r.b[j] * im;
                out[n + r.c[j]] += -r.b[j] * re + r.a[j] * im;
            }
        }
    };
    std::vector<double> diag(2 * (size_t)n, 0.0);
    for (const Row& r : rows)
        for (int j = 0; j < 3; j++) {
            const double d = r.a[j] * r.a[j] + r.b[j] * r.b[j];
            diag[r.c[j]] += d;
            diag[n + r.c[j]] += d;
        }
    // Fixed: the pins, and any vertex only degenerate triangles touch (no equation).
    std::vector<uint8_t> fixed(2 * (size_t)n, 0);
    fixed[p0] = fixed[n + p0] = fixed[p1] = fixed[n + p1] = 1;
    for (size_t i = 0; i < diag.size(); i++) if (diag[i] <= 0.0) fixed[i] = 1;

    // Preconditioned conjugate gradients on the free entries.
    std::vector<double> r(2 * (size_t)n), z(r.size()), p(r.size()), Ap(r.size());
    apply(x, r);
    for (size_t i = 0; i < r.size(); i++) r[i] = fixed[i] ? 0.0 : -r[i];
    for (size_t i = 0; i < r.size(); i++) z[i] = fixed[i] ? 0.0 : r[i] / diag[i];
    p = z;
    double rz = 0.0, r0 = 0.0;
    for (size_t i = 0; i < r.size(); i++) { rz += r[i] * z[i]; r0 += r[i] * r[i]; }
    const uint32_t max_it = std::max(200u, 4u * n);
    for (uint32_t it = 0; it < max_it && r0 > 0.0; it++) {
        apply(p, Ap);
        double pAp = 0.0;
        for (size_t i = 0; i < p.size(); i++) if (!fixed[i]) pAp += p[i] * Ap[i];
        if (pAp <= 0.0) break;
        const double al = rz / pAp;
        double rr = 0.0;
        for (size_t i = 0; i < p.size(); i++) {
            if (fixed[i]) continue;
            x[i] += al * p[i];
            r[i] -= al * Ap[i];
            rr += r[i] * r[i];
        }
        if (rr < 1e-16 * r0) break;
        double rz2 = 0.0;
        for (size_t i = 0; i < r.size(); i++) {
            z[i] = fixed[i] ? 0.0 : r[i] / diag[i];
            rz2 += r[i] * z[i];
        }
        const double be = rz2 / rz;
        rz = rz2;
        for (size_t i = 0; i < p.size(); i++) p[i] = fixed[i] ? 0.0 : z[i] + be * p[i];
    }

    // Area in UV against area on the mesh: the scale, and (signed) whether it came out
    // mirrored, which a conformal map shouldn't; folds = triangles turned over.
    double a_uv = 0.0, a_3d = 0.0;
    for (uint32_t t : tris) {
        const uint32_t a = local[wv[3 * t]], b = local[wv[3 * t + 1]], d = local[wv[3 * t + 2]];
        a_uv += 0.5 * ((x[b] - x[a]) * (x[n + d] - x[n + a]) - (x[d] - x[a]) * (x[n + b] - x[n + a]));
        a_3d += 0.5 * (P(b) - P(a)).cross(P(d) - P(a)).length();
    }
    res.mirrored = a_uv < 0.0;
    const double sc = std::fabs(a_uv) > 1e-30 ? std::sqrt(a_3d / std::fabs(a_uv)) : 1.0;
    for (uint32_t i = 0; i < n; i++) {
        uv[2 * copy[i]]     = (float)((res.mirrored ? -x[i] : x[i]) * sc);
        uv[2 * copy[i] + 1] = (float)(x[n + i] * sc);
    }
    for (uint32_t t : tris) {
        const uint32_t a = wv[3 * t], b = wv[3 * t + 1], d = wv[3 * t + 2];
        const float ar = (uv[2 * b] - uv[2 * a]) * (uv[2 * d + 1] - uv[2 * a + 1])
                       - (uv[2 * d] - uv[2 * a]) * (uv[2 * b + 1] - uv[2 * a + 1]);
        res.folds += ar < 0.0f;
    }
    return res;
}

const char* uv_unwrap(const RetopoMesh& rm, const UvSettings& s, UvLayout& out, double& ms) {
    out.clear();
    ms = 0.0;
    const uint32_t nf = rm.face_count();
    if (nf == 0) return "no retopo mesh to unwrap (Build one in mode 0 first)";
    const auto t0 = std::chrono::steady_clock::now();

    // Faces as triangles (the flattening works on triangles), quads put back below.
    // tri_corner holds, per triangle corner, which corner of rm.faces it came from.
    std::vector<uint32_t> tri_idx, tri_corner, tri_face, local;
    for (uint32_t f = 0; f < nf; f++) {
        if (rm.face_size(f) < 3) continue;
        local.clear();
        triangulate(rm, f, local);
        for (uint32_t j : local) {
            tri_corner.push_back(rm.face_off[f] + j);
            tri_idx.push_back(rm.faces[rm.face_off[f] + j]);
        }
        for (size_t t = 0; t < local.size() / 3; t++) tri_face.push_back(f);
    }
    const uint32_t nt = (uint32_t)tri_face.size();

    // Islands are exactly what the painted seams (and the mesh's own borders) fence off:
    // nothing else is cut. Each vertex on a seam is split into one copy per wedge (the
    // faces round it between seams); everything joined through a shared vertex copy is
    // one island. xatlas segments by its own cost rules and can't be told to stop, so it
    // only packs (pass 2); the flattening is ours (lscm, above).
    std::vector<uint32_t> seam_e;
    retopo_seam_edges(rm, seam_e);
    std::vector<uint32_t> wv = tri_idx;    // per triangle corner: its vertex copy
    std::vector<uint32_t> orig(rm.v.size());   // per copy: the mesh vertex
    for (uint32_t i = 0; i < orig.size(); i++) orig[i] = i;
    if (!seam_e.empty()) {
        std::vector<uint64_t> seam_k;
        std::vector<uint8_t> on_seam(rm.v.size(), 0);
        for (size_t i = 0; i < seam_e.size(); i += 2) {
            seam_k.push_back(((uint64_t)std::min(seam_e[i], seam_e[i + 1]) << 32) | std::max(seam_e[i], seam_e[i + 1]));
            on_seam[seam_e[i]] = on_seam[seam_e[i + 1]] = 1;
        }
        std::sort(seam_k.begin(), seam_k.end());
        auto is_seam = [&](uint32_t a, uint32_t b) {
            return std::binary_search(seam_k.begin(), seam_k.end(), ((uint64_t)std::min(a, b) << 32) | std::max(a, b));
        };
        // Wedges: a seam vertex's corners, joined across the non-seam edges they share.
        std::vector<uint32_t> up(rm.faces.size());
        for (uint32_t c = 0; c < up.size(); c++) up[c] = c;
        auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
        std::map<std::pair<uint64_t, uint32_t>, uint32_t> at_edge;   // (edge, its end v) -> a corner at v beside it
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t o = rm.face_off[f], k = rm.face_size(f);
            for (uint32_t j = 0; j < k; j++) {
                const uint32_t v = rm.faces[o + j];
                if (!on_seam[v]) continue;
                for (uint32_t w : {rm.faces[o + (j + 1) % k], rm.faces[o + (j + k - 1) % k]}) {
                    if (w == v || is_seam(v, w)) continue;
                    const auto key = std::make_pair(((uint64_t)std::min(v, w) << 32) | std::max(v, w), v);
                    const auto it = at_edge.find(key);
                    if (it == at_edge.end()) at_edge.emplace(key, o + j);
                    else up[root(o + j)] = root(it->second);
                }
            }
        }
        // Copies: the first wedge round a vertex keeps its number, the rest get new ones.
        std::vector<uint32_t> wedge_vert(rm.faces.size(), ~0u), wedges_at(rm.v.size(), 0);
        for (uint32_t i = 0; i < tri_idx.size(); i++) {
            const uint32_t v = tri_idx[i];
            if (!on_seam[v]) continue;
            const uint32_t r = root(tri_corner[i]);
            if (wedge_vert[r] == ~0u) {
                if (wedges_at[v]++ == 0) wedge_vert[r] = v;
                else { wedge_vert[r] = (uint32_t)orig.size(); orig.push_back(v); }
            }
            wv[i] = wedge_vert[r];
        }
    }
    const uint32_t nw = (uint32_t)orig.size();
    // Islands: copies joined by the triangles they share.
    std::vector<uint32_t> isl(nw);
    {
        std::vector<uint32_t> up(nw);
        for (uint32_t i = 0; i < nw; i++) up[i] = i;
        auto root = [&](uint32_t x) { while (up[x] != x) x = up[x] = up[up[x]]; return x; };
        for (uint32_t t = 0; t < nt; t++) {
            up[root(wv[3 * t + 1])] = root(wv[3 * t]);
            up[root(wv[3 * t + 2])] = root(wv[3 * t]);
        }
        std::vector<uint32_t> num(nw, ~0u);
        for (uint32_t t = 0; t < nt; t++) {
            const uint32_t r = root(wv[3 * t]);
            if (num[r] == ~0u) num[r] = out.charts++;
        }
        for (uint32_t i = 0; i < nw; i++) isl[i] = num[root(i)];
    }
    out.face_chart.assign(nf, ~0u);
    uint32_t orphans = 0;
    for (uint32_t t = 0; t < nt; t++) out.face_chart[tri_face[t]] = isl[wv[3 * t]];
    for (uint32_t c : out.face_chart) orphans += c == ~0u;

    // Each island laid flat, then scaled so its area in UV matches its area on the mesh
    // (pass 2 packs them all at one scale, so a big island stays big).
    std::vector<std::vector<uint32_t>> isl_tris(out.charts);
    for (uint32_t t = 0; t < nt; t++) isl_tris[isl[wv[3 * t]]].push_back(t);
    std::vector<float> wuv(2 * (size_t)nw, 0.0f);
    out.needs_cut.assign(out.charts, 0);
    uint32_t mirrored = 0, folds = 0, not_flat = 0;
    for (uint32_t c = 0; c < out.charts; c++) {
        const LscmResult r = lscm(rm.v, orig, wv, isl_tris[c], wuv);
        mirrored += r.mirrored;
        folds += r.folds;
        if (r.closed || r.folds) { out.needs_cut[c] = 1; not_flat++; }
    }

    // Corner UVs from their copies. `flat` is in mesh units: one scale for every island,
    // which is all pass 2 needs.
    std::vector<float> flat(rm.faces.size() * 2, 0.0f);
    std::vector<uint8_t> have(rm.faces.size(), 0);
    for (uint32_t i = 0; i < wv.size(); i++) {
        flat[2 * tri_corner[i]]     = wuv[2 * wv[i]];
        flat[2 * tri_corner[i] + 1] = wuv[2 * wv[i] + 1];
    }

    // Each island turned (a true rotation, never a reflection) so its long axis lies along
    // u: the principal axis of its corners, about their centre. That's the packing help
    // xatlas's own turns gave, without the mirroring.
    {
        std::vector<double> sx(out.charts, 0.0), sy(out.charts, 0.0), n(out.charts, 0.0),
                            sxx(out.charts, 0.0), syy(out.charts, 0.0), sxy(out.charts, 0.0);
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t c = out.face_chart[f];
            if (c == ~0u) continue;
            for (uint32_t j = rm.face_off[f]; j < rm.face_off[f + 1]; j++) {
                const double x = flat[2 * j], y = flat[2 * j + 1];
                sx[c] += x; sy[c] += y; n[c] += 1.0;
                sxx[c] += x * x; syy[c] += y * y; sxy[c] += x * y;
            }
        }
        std::vector<float> cs(out.charts, 1.0f), sn(out.charts, 0.0f), mx(out.charts), my(out.charts);
        for (uint32_t c = 0; c < out.charts; c++) {
            if (n[c] <= 0.0) continue;
            mx[c] = (float)(sx[c] / n[c]); my[c] = (float)(sy[c] / n[c]);
            const double cxx = sxx[c] / n[c] - mx[c] * (double)mx[c], cyy = syy[c] / n[c] - my[c] * (double)my[c];
            const double cxy = sxy[c] / n[c] - mx[c] * (double)my[c];
            const double a = -0.5 * std::atan2(2.0 * cxy, cxx - cyy);   // turn the major axis onto u
            cs[c] = (float)std::cos(a); sn[c] = (float)std::sin(a);
        }
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t c = out.face_chart[f];
            if (c == ~0u) continue;
            for (uint32_t j = rm.face_off[f]; j < rm.face_off[f + 1]; j++) {
                const float x = flat[2 * j] - mx[c], y = flat[2 * j + 1] - my[c];
                flat[2 * j]     = mx[c] + x * cs[c] - y * sn[c];
                flat[2 * j + 1] = my[c] + x * sn[c] + y * cs[c];
            }
        }
    }

    // ---- Pass 2: pack the islands into the square, with padding. xatlas finds
    // the islands by shared UV corners; one vertex per (island, mesh vertex, UV) keeps a
    // seam vertex's two copies apart, and the island as material keeps islands from
    // merging where they happen to touch.
    std::vector<float> pv;
    std::vector<uint32_t> pidx(tri_idx.size()), pmat(nt);
    {
        std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> seen;
        for (uint32_t i = 0; i < tri_idx.size(); i++) {
            const uint32_t c = tri_corner[i], isl = out.face_chart[tri_face[i / 3]];
            uint32_t ub, vb;
            std::memcpy(&ub, &flat[2 * c], 4);
            std::memcpy(&vb, &flat[2 * c + 1], 4);
            const auto key = std::make_tuple(isl, tri_idx[i], ub, vb);
            auto it = seen.find(key);
            if (it == seen.end()) {
                it = seen.emplace(key, (uint32_t)(pv.size() / 2)).first;
                pv.push_back(flat[2 * c]);
                pv.push_back(flat[2 * c + 1]);
            }
            pidx[i] = it->second;
        }
        for (uint32_t t = 0; t < nt; t++) pmat[t] = out.face_chart[tri_face[t]];
    }
    xatlas::UvMeshDecl ud;
    ud.vertexUvData     = pv.data();
    ud.vertexStride     = 2 * sizeof(float);
    ud.vertexCount      = (uint32_t)(pv.size() / 2);
    ud.indexData        = pidx.data();
    ud.indexFormat      = xatlas::IndexFormat::UInt32;
    ud.indexCount       = (uint32_t)pidx.size();
    ud.faceMaterialData = pmat.data();
    xatlas::Atlas* atlas = xatlas::Create();
    if (xatlas::AddUvMesh(atlas, ud) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        out.clear();
        return "the packer refused the islands";
    }
    xatlas::PackOptions po;
    po.resolution = (uint32_t)s.resolution;
    po.padding    = (uint32_t)s.padding;
    // Never let xatlas's packer turn an island: both its turns are reflections (the
    // quarter turn swaps u and v; the turn onto the long axis projects onto a frame that
    // can be left-handed), so they hand islands back mirrored. Ours above is a rotation.
    po.rotateCharts = po.rotateChartsToAxis = false;
    xatlas::Generate(atlas, xatlas::ChartOptions(), po);   // on a UV mesh: find its islands, then pack

    const char* why = nullptr;
    float used = 0.0f;
    uint32_t turned = 0, dotted = 0;
    if (atlas->meshCount != 1 || atlas->width == 0 || atlas->height == 0 || atlas->meshes[0].indexCount != pidx.size()) {
        why = "the packer produced nothing";
    } else if (atlas->atlasCount > 1) {
        // Fixed resolution with an estimated scale should always fit one square; more
        // than one means the islands didn't, and a second sheet has nowhere to go.
        why = "the islands didn't fit one texture";
    } else {
        const xatlas::Mesh& pm = atlas->meshes[0];
        used = atlas->utilization ? atlas->utilization[0] : 0.0f;
        const float iw = 1.0f / (float)atlas->width, ih = 1.0f / (float)atlas->height;
        out.uv.assign(rm.faces.size() * 2, 0.0f);
        std::fill(have.begin(), have.end(), 0);
        for (uint32_t i = 0; i < pidx.size(); i++) {
            const xatlas::Vertex& v = pm.vertexArray[pm.indexArray[i]];
            if (v.chartIndex < 0) continue;   // a triangle too thin to pack: its corners come from the others
            out.uv[2 * tri_corner[i]]     = v.uv[0] * iw;
            out.uv[2 * tri_corner[i] + 1] = v.uv[1] * ih;
            have[tri_corner[i]] = 1;
        }
        // Tripwire: with the packer's turns off, no island should come out backwards.
        // (Mirroring one here would be wrong anyway: islands nest by shape, so flipping
        // one across its box lands it on its neighbours. That was the 2026-10-26 overlap.)
        {
            std::vector<double> area(out.charts, 0.0);
            for (uint32_t f = 0; f < nf; f++) {
                const uint32_t c = out.face_chart[f];
                if (c == ~0u) continue;
                const uint32_t o = rm.face_off[f], k = rm.face_size(f);
                for (uint32_t j = 0; j < k; j++) {
                    const uint32_t a = o + j, b = o + (j + 1) % k;
                    area[c] += (double)out.uv[2 * a] * out.uv[2 * b + 1] - (double)out.uv[2 * b] * out.uv[2 * a + 1];
                }
            }
            for (uint32_t c = 0; c < out.charts; c++) turned += area[c] < 0.0;
        }
        // Whatever no packed triangle reached (a face in no island, or a corner only a
        // sliver touched) takes its face's first placed corner: a dot, not a stray line.
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t o = rm.face_off[f], k = rm.face_size(f);
            uint32_t src = ~0u;
            for (uint32_t j = 0; j < k && src == ~0u; j++) if (have[o + j]) src = o + j;
            for (uint32_t j = 0; j < k; j++) dotted += !have[o + j];
            for (uint32_t j = 0; j < k; j++)
                if (!have[o + j] && src != ~0u) { out.uv[2 * (o + j)] = out.uv[2 * src]; out.uv[2 * (o + j) + 1] = out.uv[2 * src + 1]; }
        }
        out.key = uv_topology_key(rm);
        out.seam_key = uv_seam_key(rm);
        out.used = used;
    }
    xatlas::Destroy(atlas);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (why) { out.clear(); return why; }
    std::printf("[uv] unwrap: %u faces, %zu painted seam edge(s) -> %u islands, %.0f%% of the square used, %.1f ms"
                " (not flat-able %u, folded triangles %u, mirrored back %u, BACKWARDS AFTER PACKING %u, faces in no island %u, corners dotted %u)\n",
                nf, seam_e.size() / 2, out.charts, used * 100.0f, ms, not_flat, folds, mirrored, turned, orphans, dotted);
    return nullptr;
}
