#include "clip.h"
#include "camera.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#ifndef __EMSCRIPTEN__
#include <thread>
#endif

namespace {

// How wide clip_target's blend opens with depth: tau = 6 px + this x the distance past
// the curve. 0 = nearest point (plus the 6 px hand-over); 1 spreads a point about its
// own depth along the curve either way.
constexpr float kClipSpread = 1.0f;

// Douglas-Peucker on x, y pairs: keeps the points the curve's shape needs, so the
// per-vertex search runs over tens of segments, not the hundreds a slow drag lays.
void simplify(const std::vector<float>& in, float tol, std::vector<float>& out) {
    const size_t n = in.size() / 2;
    std::vector<char> keep(n, 0);
    keep[0] = keep[n - 1] = 1;
    std::vector<std::pair<size_t, size_t>> todo{{0, n - 1}};
    while (!todo.empty()) {
        auto [a, b] = todo.back();
        todo.pop_back();
        if (b <= a + 1) continue;
        const float ax = in[a * 2], ay = in[a * 2 + 1];
        const float dx = in[b * 2] - ax, dy = in[b * 2 + 1] - ay;
        const float len = std::max(std::sqrt(dx * dx + dy * dy), 1e-6f);
        float worst = -1.0f;
        size_t wi = a;
        for (size_t i = a + 1; i < b; i++) {
            const float px = in[i * 2] - ax, py = in[i * 2 + 1] - ay;
            float d;
            if (len < 1e-3f) d = std::sqrt(px * px + py * py);
            else             d = std::fabs(px * dy - py * dx) / len;
            if (d > worst) { worst = d; wi = i; }
        }
        if (worst > tol) {
            keep[wi] = 1;
            todo.push_back({a, wi});
            todo.push_back({wi, b});
        }
    }
    out.clear();
    for (size_t i = 0; i < n; i++)
        if (keep[i]) { out.push_back(in[i * 2]); out.push_back(in[i * 2 + 1]); }
}

// The direction the curve leaves an end along: from the point about `reach` px back
// along it, so the last jitter of the hand doesn't aim the run-on.
void end_dir(const std::vector<float>& p, bool at_start, float reach, float& ux, float& uy) {
    const size_t n = p.size() / 2;
    const size_t e = at_start ? 0 : n - 1;
    const float ex = p[e * 2], ey = p[e * 2 + 1];
    float fx = ex, fy = ey;
    for (size_t k = 1; k < n; k++) {
        const size_t i = at_start ? k : n - 1 - k;
        fx = p[i * 2]; fy = p[i * 2 + 1];
        if (std::hypot(fx - ex, fy - ey) >= reach) break;
    }
    const float dx = ex - fx, dy = ey - fy;
    const float l = std::max(std::sqrt(dx * dx + dy * dy), 1e-6f);
    ux = dx / l; uy = dy / l;
}

}  // namespace

void clip_prepare(const std::vector<float>& raw, bool straight, int win_w, int win_h,
                  ClipPath& out) {
    out.xy.clear();
    out.nrm.clear();
    out.ok = false;
    const size_t n = raw.size() / 2;
    if (n < 2) return;

    std::vector<float> pts;
    if (straight) {
        pts = {raw[0], raw[1], raw[(n - 1) * 2], raw[(n - 1) * 2 + 1]};
    } else {
        // Two passes of a 1-2-1 average take the hand's tremble out (the ends stay
        // where they were drawn), then thin to what the shape needs.
        std::vector<float> s = raw, t = raw;
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 1; i + 1 < n; i++)
                for (int c = 0; c < 2; c++)
                    t[i * 2 + c] = 0.25f * s[(i - 1) * 2 + c] + 0.5f * s[i * 2 + c]
                                 + 0.25f * s[(i + 1) * 2 + c];
            s.swap(t);
        }
        simplify(s, 0.75f, pts);
    }
    const size_t m = pts.size() / 2;
    const float span = std::hypot(pts[(m - 1) * 2] - pts[0], pts[(m - 1) * 2 + 1] - pts[1]);
    float length = 0.0f;
    for (size_t i = 0; i + 1 < m; i++)
        length += std::hypot(pts[i * 2 + 2] - pts[i * 2], pts[i * 2 + 3] - pts[i * 2 + 1]);
    if (length < 4.0f || (m == 2 && span < 4.0f)) return;   // a click

    // Run both ends on far past the window, so the wall splits the whole screen.
    const float far = 4.0f * (float)(win_w + win_h);
    float sx, sy, ex, ey;
    end_dir(pts, true, 16.0f, sx, sy);
    end_dir(pts, false, 16.0f, ex, ey);
    out.xy.reserve(pts.size() + 4);
    out.xy.push_back(pts[0] + sx * far);
    out.xy.push_back(pts[1] + sy * far);
    out.xy.insert(out.xy.end(), pts.begin(), pts.end());
    out.xy.push_back(pts[(m - 1) * 2] + ex * far);
    out.xy.push_back(pts[(m - 1) * 2 + 1] + ey * far);

    const size_t k = out.xy.size() / 2;
    out.nrm.resize((k - 1) * 2);
    out.cum.assign(k, 0.0f);
    for (size_t i = 0; i + 1 < k; i++) {
        const float dx = out.xy[i * 2 + 2] - out.xy[i * 2];
        const float dy = out.xy[i * 2 + 3] - out.xy[i * 2 + 1];
        const float l = std::max(std::sqrt(dx * dx + dy * dy), 1e-6f);
        out.nrm[i * 2]     = -dy / l;   // right of the direction, screen y down
        out.nrm[i * 2 + 1] =  dx / l;
        out.cum[i + 1] = out.cum[i] + l;
    }
    const float reach = 1.5f * std::hypot((float)win_w, (float)win_h);
    for (int L = 0; L < ClipPath::kClipLevels; L++) {
        std::vector<float>& smp = out.smp[L];
        smp.clear();
        const float step_draw = 3.0f * (float)(1 << L);
        const float step_run = std::max(12.0f, step_draw);
        for (size_t i = 0; i + 1 < k; i++) {
            const float ax = out.xy[i * 2], ay = out.xy[i * 2 + 1];
            const float bx = out.xy[i * 2 + 2], by = out.xy[i * 2 + 3];
            const float len = out.cum[i + 1] - out.cum[i];
            // The run-on ends: sampled from the drawn end outward, as far as `reach`.
            const bool run_start = i == 0, run_end = i + 2 == k;
            const float step = run_start || run_end ? step_run : step_draw;
            float t0 = 0.0f, t1 = len;
            if (run_start) t0 = std::max(0.0f, len - reach);
            if (run_end)   t1 = std::min(len, reach);
            const int n = std::max(1, (int)std::ceil((t1 - t0) / step));
            const float ds = (t1 - t0) / (float)n;
            for (int j = 0; j < n; j++) {
                const float u = t0 + ds * ((float)j + 0.5f);
                const float f = u / len;
                smp.push_back(ax + (bx - ax) * f);
                smp.push_back(ay + (by - ay) * f);
                smp.push_back(out.cum[i] + u);
                smp.push_back(ds);
            }
        }
        std::vector<float>& box = out.box[L];
        box.clear();
        const size_t ns = smp.size() / 4;
        for (size_t r = 0; r < ns; r += ClipPath::kClipRun) {
            float x0 = 3.4e38f, y0 = 3.4e38f, x1 = -3.4e38f, y1 = -3.4e38f;
            for (size_t i = r; i < std::min(ns, r + ClipPath::kClipRun); i++) {
                x0 = std::min(x0, smp[i * 4]);     x1 = std::max(x1, smp[i * 4]);
                y0 = std::min(y0, smp[i * 4 + 1]); y1 = std::max(y1, smp[i * 4 + 1]);
            }
            box.insert(box.end(), {x0, y0, x1, y1});
        }
    }
    out.ok = true;
}

float clip_side(const ClipPath& path, bool flip, float x, float y, float& qx, float& qy) {
    const size_t segs = path.nrm.size() / 2;
    float best = 3.4e38f, bt = 0.0f;
    size_t bs = 0;
    for (size_t i = 0; i < segs; i++) {
        const float ax = path.xy[i * 2], ay = path.xy[i * 2 + 1];
        const float dx = path.xy[i * 2 + 2] - ax, dy = path.xy[i * 2 + 3] - ay;
        const float l2 = dx * dx + dy * dy;
        float t = l2 > 0.0f ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0.0f;
        t = std::min(1.0f, std::max(0.0f, t));
        const float cx = ax + dx * t - x, cy = ay + dy * t - y;
        const float d2 = cx * cx + cy * cy;
        if (d2 < best) { best = d2; bs = i; bt = t; }
    }
    const float ax = path.xy[bs * 2], ay = path.xy[bs * 2 + 1];
    qx = ax + (path.xy[bs * 2 + 2] - ax) * bt;
    qy = ay + (path.xy[bs * 2 + 3] - ay) * bt;
    // Nearest to a corner: the side is read off both segments' normals together
    // (the corner's pseudo-normal), or the inside of a bend would read as outside.
    float nx = path.nrm[bs * 2], ny = path.nrm[bs * 2 + 1];
    if (bt <= 0.0f && bs > 0)            { nx += path.nrm[bs * 2 - 2]; ny += path.nrm[bs * 2 - 1]; }
    else if (bt >= 1.0f && bs + 1 < segs) { nx += path.nrm[bs * 2 + 2]; ny += path.nrm[bs * 2 + 3]; }
    const float side = (x - qx) * nx + (y - qy) * ny;
    const float d = std::sqrt(best);
    const float s = side >= 0.0f ? d : -d;
    return flip ? -s : s;
}

bool clip_target(const ClipPath& path, bool flip, float x, float y, float& tx, float& ty) {
    float qx, qy;
    const float side = clip_side(path, flip, x, y, qx, qy);
    if (side <= 0.0f) return false;

    // Soft-min over even samples of the curve: weight = the stretch a sample stands
    // for x exp(-(d - d_min) / tau), and the landing place is the weighted mean arc
    // length. tau grows with depth (d_min, how far the point sits past the curve), so a
    // point just past the curve lands on its nearest point, while one deep in a bend
    // spreads along the curve round it, and the two arms of a C hand over smoothly
    // through the bend instead of tearing across it.
    const float dmin = std::fabs(side);
    const float tau = 6.0f + kClipSpread * dmin;
    // The weight is exp(-e / tau) for e = how much farther than the nearest a sample
    // is, out to 6 tau, then fades smoothly to exactly 0 at 8 tau (so a sample dropping
    // out never moves anything by a jump). Runs of samples whose box is out of reach
    // are skipped whole, the rest on the cheap squared test.
    const float reach = 8.0f * tau, fade0 = 6.0f * tau;
    const float lim = dmin + reach, lim2 = lim * lim;
    auto blend = [&](int L, float& s_out) {
        const std::vector<float>& smp = path.smp[L];
        const std::vector<float>& box = path.box[L];
        float sw = 0.0f, ss = 0.0f;
        const size_t ns = smp.size() / 4;
        for (size_t r = 0; r * 4 < box.size(); r++) {
            const float* b = &box[r * 4];
            const float bx = std::max(std::max(b[0] - x, x - b[2]), 0.0f);
            const float by = std::max(std::max(b[1] - y, y - b[3]), 0.0f);
            if (bx * bx + by * by >= lim2) continue;
            const size_t i1 = std::min(ns, (r + 1) * (size_t)ClipPath::kClipRun);
            for (size_t i = r * ClipPath::kClipRun; i < i1; i++) {
                const float* q = &smp[i * 4];
                const float dx = q[0] - x, dy = q[1] - y;
                const float d2 = dx * dx + dy * dy;
                if (d2 >= lim2) continue;
                const float e = std::max(0.0f, std::sqrt(d2) - dmin);
                float w = q[3] * std::exp(-e / tau);
                if (e > fade0) {
                    const float f = 1.0f - (e - fade0) / (reach - fade0);
                    w *= f * f * (3.0f - 2.0f * f);
                }
                sw += w; ss += w * q[2];
            }
        }
        if (sw <= 0.0f) return false;
        s_out = ss / sw;
        return true;
    };
    // The level whose spacing is about tau / 2 (fine enough to sum the blend well:
    // tau / 8 lands the same to a pixel, at four times the cost),
    // and the next coarser one, mixed by the fraction between them so the landing
    // place slides smoothly with depth instead of stepping where the level changes.
    const float lv = std::min(std::max(std::log2(tau / 6.0f), 0.0f),
                              (float)(ClipPath::kClipLevels - 1));
    const int L0 = (int)lv;
    const int L1 = std::min(L0 + 1, ClipPath::kClipLevels - 1);
    const float fr = lv - (float)L0;
    float s0 = 0.0f, s1 = 0.0f;
    const bool ok0 = blend(L0, s0);
    const bool ok1 = fr > 0.0f && L1 != L0 ? blend(L1, s1) : ok0;
    if (fr <= 0.0f || L1 == L0) s1 = s0;
    if (!ok0 && !ok1) { tx = qx; ty = qy; return true; }
    const float s = !ok0 ? s1 : !ok1 ? s0 : s0 + (s1 - s0) * fr;

    // Back from arc length to a point on the curve.
    const size_t k = path.cum.size();
    size_t i = std::upper_bound(path.cum.begin(), path.cum.end(), s) - path.cum.begin();
    i = std::min(std::max<size_t>(i, 1), k - 1) - 1;
    const float l = path.cum[i + 1] - path.cum[i];
    const float t = l > 0.0f ? (s - path.cum[i]) / l : 0.0f;
    tx = path.xy[i * 2] + (path.xy[i * 2 + 2] - path.xy[i * 2]) * t;
    ty = path.xy[i * 2 + 1] + (path.xy[i * 2 + 3] - path.xy[i * 2 + 1]) * t;
    return true;
}

void clip_bind(const Mesh& mesh, const Camera& cam, int w, int h, const ClipPath& path,
               bool flip, bool mirror, std::vector<uint32_t>& ids, std::vector<float>& targets) {
    ids.clear();
    targets.clear();
    if (!path.ok) return;
    const auto t0 = std::chrono::steady_clock::now();

    // The projection pose_screen_ray unprojects with, written out once so the loop
    // over a few million vertices doesn't rebuild the camera's frame per vertex.
    const Vec3 eye = cam.get_position();
    const Vec3 fwd = cam.get_view_direction();
    const Vec3 right = fwd.cross({0.0f, 1.0f, 0.0f}).normalized();
    const Vec3 up = right.cross(fwd).normalized();
    const float aspect = (float)w / (float)std::max(1, h);
    const bool persp = cam.perspective;
    const float hh1 = persp ? cam.half_height_at(1.0f) : cam.half_height();
    auto hh_at = [&](float depth) { return persp ? hh1 * depth : hh1; };

    // Onto the wall: on screen to the nearest point of the curve, at the same depth.
    auto cut = [&](Vec3 p) -> Vec3 {
        const Vec3 d = p - eye;
        const float vz = d.dot(fwd);
        if (persp && vz <= 1e-5f) return p;   // behind the eye: not on screen
        const float hh = hh_at(vz);
        const float sx = (d.dot(right) / (hh * aspect) + 1.0f) * 0.5f * (float)w;
        const float sy = (1.0f - d.dot(up) / hh) * 0.5f * (float)h;
        float qx, qy;
        if (!clip_target(path, flip, sx, sy, qx, qy)) return p;
        const float nx = 2.0f * qx / (float)w - 1.0f;
        const float ny = 1.0f - 2.0f * qy / (float)h;
        return eye + fwd * vz + right * (nx * hh * aspect) + up * (ny * hh);
    };

    const bool has_mask = !mesh.mask.empty();
    const uint32_t vc = mesh.vertex_count();
    auto run = [&](uint32_t v0, uint32_t v1, std::vector<uint32_t>& out_ids,
                   std::vector<float>& out_t) {
        for (uint32_t v = v0; v < v1; v++) {
            const float m = has_mask && v < mesh.mask.size() ? mesh.mask[v] : 0.0f;
            if (m >= 1.0f) continue;
            const Vec3 p = mesh.get_pos(v);
            Vec3 q = cut(p);
            if (mirror) {
                // The wall reflected in x = 0: reflect, cut, reflect back.
                const Vec3 r = cut(Vec3(-q.x, q.y, q.z));
                q = Vec3(-r.x, r.y, r.z);
            }
            if (q.x == p.x && q.y == p.y && q.z == p.z) continue;
            const Vec3 t = p + (q - p) * (1.0f - m);
            out_ids.push_back(v);
            out_t.push_back(t.x);
            out_t.push_back(t.y);
            out_t.push_back(t.z);
        }
    };

    // Every vertex is independent, so on the desktop the mesh is cut in slices, one per
    // core, and the slices' lists joined in order (the list stays sorted by id, as one
    // pass would make it). The web build has no threads: one pass there.
    unsigned nt = 1;
#ifndef __EMSCRIPTEN__
    if (vc >= 50000u) nt = std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
#endif
    if (nt <= 1) {
        run(0, vc, ids, targets);
    } else {
#ifndef __EMSCRIPTEN__
        std::vector<std::vector<uint32_t>> part_ids(nt);
        std::vector<std::vector<float>> part_t(nt);
        std::vector<std::thread> pool;
        const uint32_t per = (vc + nt - 1) / nt;
        for (unsigned k = 0; k < nt; k++) {
            const uint32_t v0 = std::min(vc, k * per), v1 = std::min(vc, v0 + per);
            pool.emplace_back([&, k, v0, v1]() { run(v0, v1, part_ids[k], part_t[k]); });
        }
        for (std::thread& th : pool) th.join();
        for (unsigned k = 0; k < nt; k++) {
            ids.insert(ids.end(), part_ids[k].begin(), part_ids[k].end());
            targets.insert(targets.end(), part_t[k].begin(), part_t[k].end());
        }
#endif
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    std::printf("[clip] bind: %zu segments, %zu samples, %u verts, %u thread%s, %.1f ms\n",
                path.nrm.size() / 2, path.smp[0].size() / 4, vc, nt, nt == 1 ? "" : "s", ms);
}
