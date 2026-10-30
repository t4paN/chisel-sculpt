#include "uv_unwrap.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>

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
    struct Row { uint32_t c[3]; double a[3], b[3], qx[3], qy[3], w[3]; };
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
            r.qx[j] = qx[j];
            r.qy[j] = qy[j];
            // The cotangent of the angle at corner j: the weight of the edge facing it.
            const double ux = qx[a] - qx[j], uy = qy[a] - qy[j], vx = qx[b] - qx[j], vy = qy[b] - qy[j];
            r.w[j] = 0.5 * (ux * vx + uy * vy) / std::max(1e-300, ux * vy - uy * vx);
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

    // Conformal alone isn't enough on a dome (a face, the back of a head): a curved cap has
    // a whole family of conformal flattenings, like shining a light on it from different
    // spots, and they score alike, so the solve settles on a lopsided one, big on one
    // side and shrinking toward the other like a road into the horizon (2026-10-26,
    // Goggins: one half 1.75x the other). As-rigid-as-possible fixes the sizes: each
    // triangle is held to the nearest turned copy of its true shape (local step), then
    // the vertices are re-solved to fit those (global step, the cotangent Laplacian).
    // Started from the conformal map, a few rounds settle it.
    if (!rows.empty()) {
        // L (cotangent weights) as a sparse matrix over the vertices, built once.
        std::vector<std::map<uint32_t, double>> Lm(n);
        for (const Row& r : rows)
            for (int j = 0; j < 3; j++) {
                const uint32_t a = r.c[(j + 1) % 3], b = r.c[(j + 2) % 3];
                Lm[a][b] -= r.w[j]; Lm[b][a] -= r.w[j];
                Lm[a][a] += r.w[j]; Lm[b][b] += r.w[j];
            }
        std::vector<uint32_t> off(n + 1, 0), col;
        std::vector<double> val, ldiag(n, 0.0);
        for (uint32_t i = 0; i < n; i++) {
            for (const auto& kv : Lm[i]) { col.push_back(kv.first); val.push_back(kv.second); if (kv.first == i) ldiag[i] = kv.second; }
            off[i + 1] = (uint32_t)col.size();
        }
        // One vertex holds still (the island may otherwise drift); a vertex with no
        // weight of its own (only degenerate triangles) too.
        std::vector<uint8_t> hold(n, 0);
        hold[p0] = 1;
        for (uint32_t i = 0; i < n; i++) if (ldiag[i] <= 1e-300) hold[i] = 1;
        auto solve = [&](double* y, const std::vector<double>& rhs) {   // L y = rhs, PCG from y
            std::vector<double> rr(n), zz(n), pp(n), Ap(n);
            auto mul = [&](const double* in, std::vector<double>& o) {
                for (uint32_t i = 0; i < n; i++) {
                    double acc = 0.0;
                    for (uint32_t k = off[i]; k < off[i + 1]; k++) acc += val[k] * in[col[k]];
                    o[i] = acc;
                }
            };
            mul(y, Ap);
            double rz = 0.0, r0 = 0.0;
            for (uint32_t i = 0; i < n; i++) {
                rr[i] = hold[i] ? 0.0 : rhs[i] - Ap[i];
                zz[i] = hold[i] ? 0.0 : rr[i] / ldiag[i];
                pp[i] = zz[i];
                rz += rr[i] * zz[i]; r0 += rr[i] * rr[i];
            }
            for (uint32_t it = 0; it < 4 * n + 100 && r0 > 0.0; it++) {
                mul(pp.data(), Ap);
                double pAp = 0.0;
                for (uint32_t i = 0; i < n; i++) if (!hold[i]) pAp += pp[i] * Ap[i];
                if (pAp <= 0.0) break;
                const double al = rz / pAp;
                double rn = 0.0, rz2 = 0.0;
                for (uint32_t i = 0; i < n; i++) {
                    if (hold[i]) continue;
                    y[i] += al * pp[i];
                    rr[i] -= al * Ap[i];
                    rn += rr[i] * rr[i];
                }
                if (rn < 1e-20 * r0) break;
                for (uint32_t i = 0; i < n; i++) { zz[i] = hold[i] ? 0.0 : rr[i] / ldiag[i]; rz2 += rr[i] * zz[i]; }
                const double be = rz2 / rz;
                rz = rz2;
                for (uint32_t i = 0; i < n; i++) pp[i] = hold[i] ? 0.0 : zz[i] + be * pp[i];
            }
        };
        // The conformal map's size is arbitrary: bring it to true size first, so the
        // turned copies (true size) and the map agree on scale.
        double au = 0.0, a3 = 0.0;
        for (const Row& r : rows) {
            const uint32_t a = r.c[0], b = r.c[1], d = r.c[2];
            au += 0.5 * ((x[b] - x[a]) * (x[n + d] - x[n + a]) - (x[d] - x[a]) * (x[n + b] - x[n + a]));
            a3 += 0.5 * r.qx[1] * r.qy[2];
        }
        if (au > 1e-30) {   // a mirrored conformal map (shouldn't happen) is left as it is
            const double k = std::sqrt(a3 / au);
            for (double& t : x) t *= k;
            std::vector<double> bu(n), bv(n);
            for (int round = 0; round < 12; round++) {
                std::fill(bu.begin(), bu.end(), 0.0);
                std::fill(bv.begin(), bv.end(), 0.0);
                for (const Row& r : rows) {
                    // Local: the turn that best fits the triangle's current UV shape.
                    double s00 = 0.0, s01 = 0.0, s10 = 0.0, s11 = 0.0;
                    for (int j = 0; j < 3; j++) {
                        const int a = (j + 1) % 3, b = (j + 2) % 3;
                        const double du = x[r.c[a]] - x[r.c[b]], dv = x[n + r.c[a]] - x[n + r.c[b]];
                        const double dx = r.qx[a] - r.qx[b], dy = r.qy[a] - r.qy[b];
                        s00 += r.w[j] * du * dx; s01 += r.w[j] * du * dy;
                        s10 += r.w[j] * dv * dx; s11 += r.w[j] * dv * dy;
                    }
                    const double th = std::atan2(s10 - s01, s00 + s11), cs = std::cos(th), sn = std::sin(th);
                    // Global's right-hand side: each edge's turned true vector, both ends.
                    for (int j = 0; j < 3; j++) {
                        const int a = (j + 1) % 3, b = (j + 2) % 3;
                        const double dx = r.qx[a] - r.qx[b], dy = r.qy[a] - r.qy[b];
                        const double ru = cs * dx - sn * dy, rv = sn * dx + cs * dy;
                        bu[r.c[a]] += r.w[j] * ru; bv[r.c[a]] += r.w[j] * rv;
                        bu[r.c[b]] -= r.w[j] * ru; bv[r.c[b]] -= r.w[j] * rv;
                    }
                }
                solve(x.data(), bu);
                solve(x.data() + n, bv);
            }
        }
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

// ---- Packing. Each island is drawn onto a grid of cells (every cell one of its
// triangles so much as touches), grown by the padding, and placed biggest first wherever
// its top ends lowest, trying all four quarter turns: true rotations, since a reflection
// would hand the island back mirrored. Islands fit by shape, not by box, so a small one
// can sit in a big one's bay. A search over the scale finds the largest that fits.

struct PackShape {
    int w = 0, h = 0;              // the grown mask, cells
    float mnx = 0.0f, mny = 0.0f;  // the turned island's lower corner, mesh units
    std::vector<uint8_t> raw;      // the cells the island itself covers, w*h
    std::vector<int32_t> runs;     // the grown mask as runs (y, x0, x1), bottom row first
    int cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;   // a rectangle wholly inside the grown mask
};

struct PackPlace { int q = 0, x = 0, y = 0; float mnx = 0.0f, mny = 0.0f; };

static void pack_turn(int q, float x, float y, float& ox, float& oy) {
    switch (q) {
        case 0:  ox = x;  oy = y;  break;
        case 1:  ox = -y; oy = x;  break;
        case 2:  ox = -x; oy = -y; break;
        default: ox = y;  oy = -x; break;
    }
}

// Island `tris` (6 floats per triangle, mesh units) turned q quarter turns, at s cells
// per mesh unit, grown by pad cells. False when it can't fit an n-cell square at all.
static bool pack_raster(const std::vector<float>& tris, int q, float s, int pad, int n, PackShape& o) {
    float mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
    for (size_t i = 0; i < tris.size(); i += 2) {
        float x, y;
        pack_turn(q, tris[i], tris[i + 1], x, y);
        mnx = std::min(mnx, x); mxx = std::max(mxx, x);
        mny = std::min(mny, y); mxy = std::max(mxy, y);
    }
    const double fw = (double)(mxx - mnx) * s, fh = (double)(mxy - mny) * s;
    if (fw + 1 + 2 * pad > n || fh + 1 + 2 * pad > n) return false;
    o.mnx = mnx; o.mny = mny;
    o.w = (int)fw + 1 + 2 * pad;
    o.h = (int)fh + 1 + 2 * pad;
    const int w = o.w, h = o.h;
    o.raw.assign((size_t)w * h, 0);
    for (size_t i = 0; i + 5 < tris.size(); i += 6) {
        float px[3], py[3];
        for (int k = 0; k < 3; k++) {
            pack_turn(q, tris[i + 2 * k], tris[i + 2 * k + 1], px[k], py[k]);
            px[k] = (px[k] - mnx) * s + pad;
            py[k] = (py[k] - mny) * s + pad;
        }
        if ((px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]) < 0.0f) {
            std::swap(px[1], px[2]); std::swap(py[1], py[2]);
        }
        auto cell = [](float v, int lo, int hi) { return std::min(hi, std::max(lo, (int)std::floor(v))); };
        const int x0 = cell(std::min({px[0], px[1], px[2]}), pad, w - pad - 1), x1 = cell(std::max({px[0], px[1], px[2]}), pad, w - pad - 1);
        const int y0 = cell(std::min({py[0], py[1], py[2]}), pad, h - pad - 1), y1 = cell(std::max({py[0], py[1], py[2]}), pad, h - pad - 1);
        // A cell is in if, for every edge, its corner furthest inside that edge is inside:
        // with the box test that's exact overlap, and a flat triangle gets the cells its
        // line crosses rather than none.
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                bool in = true;
                for (int k = 0; k < 3 && in; k++) {
                    const float ex = px[(k + 1) % 3] - px[k], ey = py[(k + 1) % 3] - py[k];
                    const float cx = ey < 0.0f ? x + 1.0f : (float)x, cy = ex > 0.0f ? y + 1.0f : (float)y;
                    in = ex * (cy - py[k]) - ey * (cx - px[k]) >= 0.0f;
                }
                if (in) o.raw[(size_t)y * w + x] = 1;
            }
    }
    // Grown by pad cells every way (a square, so two islands' cells end up at least pad+1
    // apart along some axis: a gap of at least pad cells between the real edges).
    std::vector<uint8_t> a((size_t)w * h, 0), g((size_t)w * h, 0);
    for (int y = 0; y < h; y++)
        for (int x = pad; x < w - pad; x++)
            if (o.raw[(size_t)y * w + x]) for (int d = -pad; d <= pad; d++) a[(size_t)y * w + x + d] = 1;
    for (int y = pad; y < h - pad; y++)
        for (int x = 0; x < w; x++)
            if (a[(size_t)y * w + x]) for (int d = -pad; d <= pad; d++) g[(size_t)(y + d) * w + x] = 1;
    o.runs.clear();
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w;) {
            if (!g[(size_t)y * w + x]) { x++; continue; }
            int e = x;
            while (e < w && g[(size_t)y * w + e]) e++;
            o.runs.push_back(y); o.runs.push_back(x); o.runs.push_back(e);
            x = e;
        }
    // The biggest rectangle inside the grown mask (row by row, as a histogram): one sum
    // over it rules out most places before the runs are looked at.
    std::vector<int> ht(w + 1, 0), stk;
    long best = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) ht[x] = g[(size_t)y * w + x] ? ht[x] + 1 : 0;
        stk.clear();
        for (int x = 0; x <= w; x++) {
            const int hx = x < w ? ht[x] : 0;
            while (!stk.empty() && ht[stk.back()] >= hx) {
                const int top = ht[stk.back()];
                stk.pop_back();
                const int left = stk.empty() ? 0 : stk.back() + 1;
                if ((long)top * (x - left) > best) {
                    best = (long)top * (x - left);
                    o.cx0 = left; o.cx1 = x; o.cy0 = y - top + 1; o.cy1 = y + 1;
                }
            }
            stk.push_back(x);
        }
    }
    return best > 0;
}

// The square being filled: which cells are taken, with sums to ask "anything taken
// in here?" in one step, and run lengths to hop over a row's taken or too-short stretches.
struct PackGrid {
    int n = 0;
    std::vector<uint8_t>  occ;    // n*n
    std::vector<int32_t>  row;    // per row, running count: n*(n+1)
    std::vector<int32_t>  sat;    // summed area: (n+1)*(n+1)
    std::vector<uint16_t> run;    // per cell: cells to the end of its run (free or taken)
    std::vector<uint16_t> widest; // per row: its longest free run
    void reset(int size) {
        n = size;
        occ.assign((size_t)n * n, 0);
        row.assign((size_t)n * (n + 1), 0);
        sat.assign((size_t)(n + 1) * (n + 1), 0);
        run.assign((size_t)n * n, 0);
        widest.assign(n, 0);
        fix(0, 0, n);
    }
    // Rows y0..y1 changed, from column x0 on.
    void fix(int x0, int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            const uint8_t* o = &occ[(size_t)y * n];
            int32_t* r = &row[(size_t)y * (n + 1)];
            uint16_t* u = &run[(size_t)y * n];
            for (int x = x0; x < n; x++) r[x + 1] = r[x] + o[x];
            int longest = 0;
            for (int x = n - 1; x >= 0; x--) {
                u[x] = (uint16_t)(x + 1 < n && o[x + 1] == o[x] ? u[x + 1] + 1 : 1);
                if (!o[x]) longest = std::max(longest, (int)u[x]);
            }
            widest[y] = (uint16_t)longest;
        }
        for (int y = y0; y < n; y++)
            for (int x = x0 + 1; x <= n; x++)
                sat[(size_t)(y + 1) * (n + 1) + x] = sat[(size_t)y * (n + 1) + x] + row[(size_t)y * (n + 1) + x];
    }
    int32_t taken(int x0, int y0, int x1, int y1) const {
        const size_t m = n + 1;
        return sat[y1 * m + x1] - sat[y0 * m + x1] - sat[y1 * m + x0] + sat[y0 * m + x0];
    }
    // The lowest, then leftmost, spot for sh whose top stays under `below`.
    bool find(const PackShape& sh, int below, int& bx, int& by) const {
        const int cw = sh.cx1 - sh.cx0;
        for (int y = 0; y + sh.h <= n && y + sh.h < below; y++) {
            const int yb = y + sh.cy0;
            if (widest[yb] < cw) continue;
            for (int x = 0; x + sh.w <= n;) {
                const size_t c = (size_t)yb * n + x + sh.cx0;
                if (occ[c] || run[c] < cw) { x += run[c]; continue; }
                // On a clash, hop to the first x that clears the rightmost taken cell
                // it hit: every x short of that still covers it.
                const int c0 = x + sh.cx0, c1 = x + sh.cx1, r0 = y + sh.cy0, r1 = y + sh.cy1;
                if (taken(c0, r0, c1, r1)) {
                    int lo = c0, hi = c1 - 1;   // the last column with something taken from it on
                    while (lo < hi) { const int m = (lo + hi + 1) / 2; if (taken(m, r0, c1, r1)) lo = m; else hi = m - 1; }
                    x = lo - sh.cx0 + 1;
                    continue;
                }
                int next = -1;
                if (taken(x, y, x + sh.w, y + sh.h)) {
                    for (size_t i = 0; i < sh.runs.size(); i += 3) {
                        const int32_t* r = &row[(size_t)(y + sh.runs[i]) * (n + 1)];
                        const int a = x + sh.runs[i + 1], b = x + sh.runs[i + 2];
                        if (r[b] == r[a]) continue;
                        int lo = a, hi = b - 1;
                        while (lo < hi) { const int m = (lo + hi + 1) / 2; if (r[b] != r[m]) lo = m; else hi = m - 1; }
                        next = lo - sh.runs[i + 1] + 1;
                        break;
                    }
                }
                if (next < 0) { bx = x; by = y; return true; }
                x = next;
            }
        }
        return false;
    }
    void stamp(const PackShape& sh, int x, int y) {
        for (int j = 0; j < sh.h; j++)
            for (int i = 0; i < sh.w; i++)
                if (sh.raw[(size_t)j * sh.w + i]) occ[(size_t)(y + j) * n + x + i] = 1;
        fix(x, y, y + sh.h);
    }
};

// Every island placed at s cells per mesh unit, biggest first; false when one won't fit.
static bool pack_try(const std::vector<std::vector<float>>& isl, const std::vector<uint32_t>& order,
                     float s, int pad, PackGrid& g, std::vector<PackPlace>& at) {
    g.reset(g.n);
    at.assign(isl.size(), PackPlace());
    PackShape sh[4];
    for (uint32_t i : order) {
        int best = g.n + 1, bq = -1, bx = 0, by = 0;
        for (int q = 0; q < 2; q++) {
            if (!pack_raster(isl[i], q, s, pad, g.n, sh[q])) continue;
            int x, y;
            if (g.find(sh[q], best, x, y)) { best = y + sh[q].h; bq = q; bx = x; by = y; }
        }
        if (bq < 0) return false;
        g.stamp(sh[bq], bx, by);
        at[i] = PackPlace{bq, bx, by, sh[bq].mnx, sh[bq].mny};
    }
    return true;
}

// Packs the islands (each a list of triangles, 6 floats apiece, mesh units) into an
// n-cell square with pad cells between them. Writes where each went; returns the scale
// (cells per mesh unit), 0 when they can't be made to fit.
static float pack_islands(const std::vector<std::vector<float>>& isl, int n, int pad, std::vector<PackPlace>& out) {
    double area = 0.0;
    std::vector<double> isl_area(isl.size(), 0.0);
    for (size_t i = 0; i < isl.size(); i++) {
        const std::vector<float>& t = isl[i];
        for (size_t k = 0; k + 5 < t.size(); k += 6)
            isl_area[i] += 0.5 * std::fabs((double)(t[k + 2] - t[k]) * (t[k + 5] - t[k + 1]) - (double)(t[k + 4] - t[k]) * (t[k + 3] - t[k + 1]));
        area += isl_area[i];
    }
    if (isl.empty()) return 0.0f;
    std::vector<uint32_t> order(isl.size());
    for (uint32_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return isl_area[a] > isl_area[b]; });
    PackGrid g;
    g.n = n;
    std::vector<PackPlace> trial;
    // A first guess from the area, stepped until one scale fits and a bigger one doesn't,
    // then halved between them.
    float s = (float)std::sqrt(0.6 * n * n / std::max(area, 1e-12)), lo = 0.0f, hi = 0.0f;
    for (int i = 0; i < 150 && (lo == 0.0f || hi == 0.0f); i++) {
        if (pack_try(isl, order, s, pad, g, trial)) { lo = s; out = trial; s *= 1.1f; }
        else { hi = s; s /= 1.1f; }
    }
    if (lo == 0.0f) return 0.0f;
    for (int i = 0; i < 4 && hi > 0.0f; i++) {
        const float mid = std::sqrt(lo * hi);
        if (pack_try(isl, order, mid, pad, g, trial)) { lo = mid; out = trial; }
        else hi = mid;
    }
    return lo;
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
    // one island.
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
    // u: the principal axis of its corners, about their centre, so the packer's quarter
    // turns have a box-shaped island to work with.
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

    // ---- Pass 2: pack the islands into the square, with padding. The grid is at most
    // 512 cells a side (2 px a cell at 1024), the padding rounded up to whole cells.
    const int res = std::max(1, s.resolution);
    const int n = std::min(512, res);
    const int pad = std::max(1, (int)std::ceil(s.padding * (float)n / res - 1e-4f));
    std::vector<std::vector<float>> isl_flat(out.charts);
    for (uint32_t c = 0; c < out.charts; c++)
        for (uint32_t t : isl_tris[c])
            for (int k = 0; k < 3; k++) {
                isl_flat[c].push_back(flat[2 * tri_corner[3 * t + k]]);
                isl_flat[c].push_back(flat[2 * tri_corner[3 * t + k] + 1]);
            }
    std::vector<PackPlace> at;
    const float sc = pack_islands(isl_flat, n, pad, at);

    const char* why = nullptr;
    double used = 0.0;
    uint32_t turned = 0, dotted = 0;
    if (sc <= 0.0f) {
        why = "the islands didn't fit one texture";
    } else {
        out.uv.assign(rm.faces.size() * 2, 0.0f);
        std::fill(have.begin(), have.end(), 0);
        for (uint32_t f = 0; f < nf; f++) {
            const uint32_t c = out.face_chart[f];
            if (c == ~0u) continue;
            const PackPlace& P = at[c];
            for (uint32_t j = rm.face_off[f]; j < rm.face_off[f + 1]; j++) {
                float x, y;
                pack_turn(P.q, flat[2 * j], flat[2 * j + 1], x, y);
                out.uv[2 * j]     = ((x - P.mnx) * sc + pad + P.x) / n;
                out.uv[2 * j + 1] = ((y - P.mny) * sc + pad + P.y) / n;
                have[j] = 1;
            }
        }
        // Tripwire: the packer only rotates, so no island should come out backwards.
        // How much of the square is used: the islands' own area, face by face.
        {
            std::vector<double> area(out.charts, 0.0);
            for (uint32_t f = 0; f < nf; f++) {
                const uint32_t c = out.face_chart[f];
                if (c == ~0u) continue;
                const uint32_t o = rm.face_off[f], k = rm.face_size(f);
                double a2 = 0.0;
                for (uint32_t j = 0; j < k; j++) {
                    const uint32_t a = o + j, b = o + (j + 1) % k;
                    a2 += (double)out.uv[2 * a] * out.uv[2 * b + 1] - (double)out.uv[2 * b] * out.uv[2 * a + 1];
                }
                area[c] += a2;
                used += std::fabs(a2) * 0.5;
            }
            for (uint32_t c = 0; c < out.charts; c++) turned += area[c] < 0.0;
        }
        // A face in no island takes its first placed corner: a dot, not a stray line.
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
        out.used = (float)std::min(1.0, used);
    }
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (why) { out.clear(); return why; }
    std::printf("[uv] unwrap: %u faces, %zu painted seam edge(s) -> %u islands, %.0f%% of the square used, %.1f ms"
                " (not flat-able %u, folded triangles %u, mirrored back %u, BACKWARDS AFTER PACKING %u, faces in no island %u, corners dotted %u)\n",
                nf, seam_e.size() / 2, out.charts, used * 100.0, ms, not_flat, folds, mirrored, turned, orphans, dotted);
    return nullptr;
}
