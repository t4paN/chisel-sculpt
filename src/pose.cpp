#include "pose.h"
#include "camera.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

void PoseHinge::on_undo(uint64_t seq) {
    for (const PoseMove& m : moves)
        if (m.seq == seq) lever = m.from;
}

void PoseHinge::on_redo(uint64_t seq) {
    for (const PoseMove& m : moves)
        if (m.seq == seq) lever = m.to;
}

void pose_screen_ray(const Camera& cam, float sx, float sy, int w, int h,
                     Vec3& origin, Vec3& dir) {
    Vec3 cam_pos = cam.get_position();
    Vec3 fwd   = cam.get_view_direction();
    Vec3 right = fwd.cross({0.0f, 1.0f, 0.0f}).normalized();
    Vec3 up    = right.cross(fwd).normalized();
    float aspect = (float)w / (float)std::max(1, h);
    float ndc_x = 2.0f * sx / (float)w - 1.0f;
    float ndc_y = 1.0f - 2.0f * sy / (float)h;
    // Same unprojection as set_anchor: cam + fwd*d + right*ndc_x*hw(d) + up*ndc_y*hh(d).
    // half_height_at is linear in d under perspective and constant under ortho, so
    // the point is linear in d and this splits into origin + d * dir.
    if (cam.perspective) {
        float t = cam.half_height_at(1.0f);
        origin = cam_pos;
        dir = fwd + right * (ndc_x * t * aspect) + up * (ndc_y * t);
    } else {
        float hh = cam.half_height();
        origin = cam_pos + right * (ndc_x * hh * aspect) + up * (ndc_y * hh);
        dir = fwd;
    }
}

bool pose_volume_center(const Mesh& mesh, const Camera& cam, float sx, float sy,
                        int w, int h, Vec3& center, float& radius) {
    Vec3 o, d;
    pose_screen_ray(cam, sx, sy, w, h, o, d);
    // Ortho shows what sits behind the camera point too (near plane is negative),
    // so only perspective cuts the ray at the eye.
    const float t_min = cam.perspective ? 1e-4f : -1e30f;

    // Brute force over every triangle: this runs on a click and while a new sphere
    // is dragged out, never inside a stroke. Two passes: nearest hit, then the
    // nearest hit past it — the wall the ray leaves the volume through.
    const uint32_t nt = (uint32_t)mesh.indices.size() / 3;
    const float* px = mesh.pos_x.data();
    const float* py = mesh.pos_y.data();
    const float* pz = mesh.pos_z.data();
    const uint32_t* ix = mesh.indices.data();
    auto hit = [&](uint32_t t, float& out) {
        const uint32_t a = ix[t * 3], b = ix[t * 3 + 1], c = ix[t * 3 + 2];
        const Vec3 p0(px[a], py[a], pz[a]);
        const Vec3 e1 = Vec3(px[b], py[b], pz[b]) - p0;
        const Vec3 e2 = Vec3(px[c], py[c], pz[c]) - p0;
        const Vec3 pv = d.cross(e2);
        const float det = e1.dot(pv);
        if (std::fabs(det) < 1e-12f) return false;
        const float inv = 1.0f / det;
        const Vec3 tv = o - p0;
        const float u = tv.dot(pv) * inv;
        if (u < 0.0f || u > 1.0f) return false;
        const Vec3 qv = tv.cross(e1);
        const float v = d.dot(qv) * inv;
        if (v < 0.0f || u + v > 1.0f) return false;
        out = e2.dot(qv) * inv;
        return true;
    };

    float t_near = 1e30f;
    for (uint32_t t = 0; t < nt; t++) {
        float s;
        if (hit(t, s) && s > t_min && s < t_near) t_near = s;
    }
    if (t_near >= 1e30f) return false;

    // Past the entry by a hair, so the entry triangle and its neighbours sharing the
    // hit edge can't answer as the far wall.
    const float eps = std::max(1e-5f, std::fabs(t_near) * 1e-5f);
    float t_far = 1e30f;
    for (uint32_t t = 0; t < nt; t++) {
        float s;
        if (hit(t, s) && s > t_near + eps && s < t_far) t_far = s;
    }
    if (t_far >= 1e30f) {
        center = o + d * t_near;
        radius = 0.0f;
        return true;
    }
    const float tm = 0.5f * (t_near + t_far);
    center = o + d * tm;
    radius = 0.5f * (t_far - t_near) * d.length();
    return true;
}

float pose_px_per_world(const Camera& cam, Vec3 p, int h) {
    float hh = cam.half_height_at(std::max(cam.view_depth(p), 1e-4f));
    return (float)h / (2.0f * std::max(hh, 1e-8f));
}

int pose_pick(const PoseHinge& hinge, const Camera& cam, float sx, float sy, int w, int h) {
    if (!hinge.set) return -1;
    int best = -1;
    float best_d2 = 12.0f * 12.0f;   // catch radius, px
    const Vec3 at[2] = { hinge.anchor, hinge.lever };
    for (int k = 0; k < 2; k++) {
        if (cam.perspective && cam.view_depth(at[k]) <= 1e-4f) continue;
        float px, py;
        cam.world_to_screen(at[k], w, h, px, py);
        const float d2 = (sx - px) * (sx - px) + (sy - py) * (sy - py);
        if (d2 < best_d2) { best_d2 = d2; best = k; }
    }
    return best;
}

bool pose_mirror_twin(const PoseHinge& hinge) {
    return std::fabs(hinge.anchor.x) > std::max(hinge.radius, 1e-6f);
}

// ---- pose_bind -------------------------------------------------------------------

namespace {
float smooth01(float x) {
    x = std::min(1.0f, std::max(0.0f, x));
    return x * x * (3.0f - 2.0f * x);
}
}  // namespace

// How the surface joins up, for the floods below. The mesh's own vertex -> triangle
// table when it has one (built here otherwise: with no table to walk, the old
// fallback moved everything past the cut, connected or not). And `weld`: vertices
// sitting on the same spot are one point of the surface. An imported model splits
// its vertices along every UV seam, so without this a flood stops at each seam and
// a hand on its own UV island would be left behind.
namespace {
struct PoseConn {
    const uint32_t* off = nullptr;
    const uint32_t* list = nullptr;
    std::vector<uint32_t> own_off, own_list;
    std::vector<uint32_t> weld;      // next vertex on the same spot, round a ring
    uint32_t welded = 0;             // vertices that share their spot with another
};

void build_conn(const Mesh& mesh, PoseConn& c) {
    const uint32_t vc = mesh.vertex_count();
    const uint32_t nt = (uint32_t)mesh.indices.size() / 3;
    const uint32_t* ix = mesh.indices.data();
    if (mesh.vert_tri_offset.size() == (size_t)vc + 1) {
        c.off = mesh.vert_tri_offset.data();
        c.list = mesh.vert_tri_list.data();
    } else {
        c.own_off.assign(vc + 1, 0);
        for (uint32_t i = 0; i < nt * 3; i++) c.own_off[ix[i] + 1]++;
        for (uint32_t v = 0; v < vc; v++) c.own_off[v + 1] += c.own_off[v];
        c.own_list.resize(nt * 3);
        static std::vector<uint32_t> fill;
        fill.assign(c.own_off.begin(), c.own_off.end() - 1);
        for (uint32_t t = 0; t < nt; t++)
            for (int k = 0; k < 3; k++) c.own_list[fill[ix[t * 3 + k]]++] = t;
        c.off = c.own_off.data();
        c.list = c.own_list.data();
    }

    // Same spot = same position on a fine grid (a millionth of the model's size), so
    // a seam that drifted by float noise under the brushes still joins.
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    const float* p[3] = {mesh.pos_x.data(), mesh.pos_y.data(), mesh.pos_z.data()};
    for (uint32_t v = 0; v < vc; v++)
        for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], p[k][v]); hi[k] = std::max(hi[k], p[k][v]); }
    const float size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-6f});
    const float inv = 1.0f / (size * 1e-6f);
    static std::vector<uint64_t> key;
    static std::vector<uint32_t> order;
    key.resize(vc);
    order.resize(vc);
    for (uint32_t v = 0; v < vc; v++) {
        uint64_t q[3];
        for (int k = 0; k < 3; k++) q[k] = (uint64_t)std::llround((p[k][v] - lo[k]) * inv) & 0x1FFFFF;
        key[v] = (q[0] << 42) | (q[1] << 21) | q[2];
        order[v] = v;
    }
    std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
        return key[x] != key[y] ? key[x] < key[y] : x < y;
    });
    c.weld.resize(vc);
    c.welded = 0;
    for (uint32_t i = 0; i < vc;) {
        uint32_t j = i + 1;
        while (j < vc && key[order[j]] == key[order[i]]) j++;
        for (uint32_t k = i; k < j; k++) c.weld[order[k]] = order[k + 1 < j ? k + 1 : i];
        if (j - i > 1) c.welded += j - i;
        i = j;
    }
}
}  // namespace

// One side of the binding: t[v] = how far through the bend v is (0 stays, 1 rides
// the lever rigidly), for every v the swing reaches (has[v] = 1). Returns how many,
// and whether the limb had to be kept to its own tube (`tube`).
static uint32_t bind_side(const Mesh& mesh, const PoseConn& conn, Vec3 a, Vec3 l,
                          float r, float hardness, std::vector<float>& t,
                          std::vector<uint8_t>& has, bool& tube) {
    tube = false;
    const uint32_t vc = mesh.vertex_count();
    Vec3 axis = l - a;
    const float len = axis.length();
    if (len < 1e-6f || vc == 0) return 0;
    const Vec3 u = axis * (1.0f / len);
    r = std::max(r, 1e-6f);

    const float* px = mesh.pos_x.data();
    const float* py = mesh.pos_y.data();
    const float* pz = mesh.pos_z.data();
    const uint32_t* ix = mesh.indices.data();
    auto pos = [&](uint32_t v) { return Vec3(px[v], py[v], pz[v]); };

    static std::vector<uint8_t> seen;
    static std::vector<uint32_t> stack, ball;
    auto nearest = [&](Vec3 q) {
        uint32_t best_v = UINT32_MAX;
        float best = 1e30f;
        for (uint32_t v = 0; v < vc; v++) {
            const Vec3 d = pos(v) - q;
            const float d2 = d.dot(d);
            if (d2 < best) { best = d2; best_v = v; }
        }
        return best_v;
    };
    // Surface flood from `seed` through every vertex `ok` lets in; seen[v] = 1 for
    // those, 2 for the ones it looked at and turned away. A vertex on the same spot
    // as one let in is let in with it.
    auto flood = [&](uint32_t seed, auto&& ok) {
        seen.assign(vc, 0);
        stack.clear();
        if (seed == UINT32_MAX || !ok(seed)) return;
        auto take = [&](uint32_t w) {
            for (uint32_t x = w;;) {
                seen[x] = 1;
                stack.push_back(x);
                x = conn.weld[x];
                if (x == w || seen[x]) break;
            }
        };
        take(seed);
        while (!stack.empty()) {
            const uint32_t v = stack.back();
            stack.pop_back();
            for (uint32_t k = conn.off[v]; k < conn.off[v + 1]; k++) {
                const uint32_t tri = conn.list[k];
                for (int c = 0; c < 3; c++) {
                    const uint32_t w = ix[tri * 3 + c];
                    if (seen[w]) continue;
                    seen[w] = 2;   // looked at; 1 only once it is let in
                    if (ok(w)) take(w);
                }
            }
        }
    };

    // The cut. Square to the line is right for a straight limb, but on a bent joint
    // that plane runs along the limb behind it (square to the forearm, it slices the
    // upper arm lengthwise). So the cut leans to the bisector of the two segments:
    // the one behind is read off the skin round the anchor, as the centre of what
    // lies behind the current cut, a few rounds so the two settle together.
    const float reach = 3.0f * r;
    flood(nearest(a), [&](uint32_t v) { return (pos(v) - a).length() < reach; });
    ball.clear();
    for (uint32_t v = 0; v < vc; v++) if (seen[v] == 1) ball.push_back(v);
    Vec3 n = u;
    for (int round = 0; round < 4; round++) {
        Vec3 sum(0, 0, 0);
        int cnt = 0;
        for (uint32_t v : ball) {
            const Vec3 q = pos(v) - a;
            if (q.dot(n) < 0.0f) { sum = sum + q; cnt++; }
        }
        if (cnt == 0) break;
        const Vec3 back = sum * (1.0f / (float)cnt);
        if (back.length() < 1e-6f) break;
        Vec3 m = (u - back.normalized()).normalized();
        // A joint folded tighter than ~40 degrees has no sensible bisector left;
        // keep the cut within 70 degrees of square to the line.
        const float c = m.dot(u);
        if (c < 0.342f) m = (u * 0.342f + (m - u * c).normalized() * 0.94f).normalized();
        n = m;
    }

    // The bend straddles the cut: from -band (still) to +band (all the way).
    // Hardness 1 is a crisp hinge, 0 spreads the bend two girths each way.
    const float band = r * (0.05f + 1.95f * (1.0f - hardness));
    // Behind the cut, a vertex also has to lie within the limb's girth round the line
    // to bend: the chest beside a shoulder stays put while the arm turns. Past the cut
    // the girth opens out as a cone, so a hand flaring past a wrist rides along whole.
    // In a tube (below) the cone only opens past the lever, at the limb's far end.
    const float girth_in = 1.1f * r, girth_out = 1.75f * r;
    float cone_start = 0.0f;
    auto held = [&](Vec3 q, float d) {
        const float rad = (q - u * q.dot(u)).length();
        const float open = 2.0f * std::max(d - cone_start, 0.0f);
        return 1.0f - smooth01((rad - girth_in - open) / (girth_out - girth_in));
    };
    auto allowed = [&](uint32_t v) {
        const Vec3 q = pos(v) - a;
        const float d = q.dot(n);
        if (d <= -band) return false;
        // Past the girth the weight is zero: stop there, so the flood can't run on
        // round the body. Behind the cut always; past it only in a tube.
        if ((d < 0.0f || tube) && held(q, d) <= 0.0f) return false;
        return true;
    };

    // Seed: the skin nearest the lever, which sits inside the limb being swung.
    const uint32_t seed = nearest(l);
    flood(seed, allowed);

    // Connected only through the limb's own ring. If what the flood reached also
    // meets the cut well away from the anchor, it isn't just the limb: an arm hanging
    // down the body, cut square at the shoulder, has the whole torso past the cut,
    // joined to it at the armpit. Then the limb keeps to its own tube up to the
    // lever, and only opens out past it (the user draws to the limb's far end).
    {
        uint32_t wide = 0;
        for (uint32_t v = 0; v < vc && wide < 8; v++) {
            if (seen[v] != 1) continue;
            const Vec3 pv = pos(v);
            const float dv = (pv - a).dot(n);
            if (dv <= 0.0f) continue;
            for (uint32_t k = conn.off[v]; k < conn.off[v + 1] && wide < 8; k++) {
                const uint32_t tri = conn.list[k];
                for (int c = 0; c < 3; c++) {
                    const Vec3 pw = pos(ix[tri * 3 + c]);
                    const float dw = (pw - a).dot(n);
                    if (dw > 0.0f) continue;
                    const Vec3 x = pv + (pw - pv) * (dv / (dv - dw));
                    if ((x - a).length() > 3.0f * r) wide++;
                }
            }
        }
        if (wide >= 8) {
            tube = true;
            cone_start = std::max(0.0f, (l - a).dot(n) - r);
            flood(seed, allowed);
        }
    }

    uint32_t count = 0;
    for (uint32_t v = 0; v < vc; v++) {
        if (seen[v] != 1) continue;
        const Vec3 q = pos(v) - a;
        const float d = q.dot(n);
        const float w = smooth01((d + band) / (2.0f * band)) * held(q, d);
        if (w < 1e-4f) continue;
        t[v] = w;
        has[v] = 1;
        count++;
    }
    return count;
}

// t through the bend -> weights on pose_apply's four lanes (the swing at 1/4 .. 4/4
// of its angle). Between two lanes the vertex blends the neighbouring pair, below
// the first it blends lane 0 with standing still.
static void lanes_of(float t, float* w) {
    w[0] = w[1] = w[2] = w[3] = 0.0f;
    if (t <= 0.0f) return;
    const float s = std::min(t, 1.0f) * 4.0f;
    if (s >= 4.0f) { w[3] = 1.0f; return; }
    const int k = (int)s;
    const float f = s - (float)k;
    w[k] = f;
    if (k > 0) w[k - 1] = 1.0f - f;
}

void pose_bind(const Mesh& mesh, const PoseHinge& hinge, float hardness, bool mirror,
               PoseBinding& out) {
    out.ids.clear(); out.init.clear(); out.w.clear();
    const uint32_t vc = mesh.vertex_count();
    if (!hinge.set || vc == 0) return;

    static std::vector<float> own, mir;
    static std::vector<uint8_t> has_own, has_mir;
    static PoseConn conn;
    build_conn(mesh, conn);
    own.assign(vc, 0.0f);
    has_own.assign(vc, 0);
    bool tube = false, tube_m = false;
    const uint32_t n_own = bind_side(mesh, conn, hinge.anchor, hinge.lever, hinge.radius,
                                     hardness, own, has_own, tube);
    uint32_t n_mir = 0;

    // Mirror: the hinge reflected in x = 0 binds the other side the same way, and
    // pose_apply carries those vertices through the reflected swing. Every pair then
    // moves as each other's reflection, which the pair-map projection (built for
    // brushes that always grab both sides) needs: fed one side only, it averages half
    // the pairs and skips the rest, and tears the mesh.
    mirror = mirror && pose_mirror_twin(hinge);
    if (mirror) {
        const Vec3 fa(-hinge.anchor.x, hinge.anchor.y, hinge.anchor.z);
        const Vec3 fl(-hinge.lever.x, hinge.lever.y, hinge.lever.z);
        mir.assign(vc, 0.0f);
        has_mir.assign(vc, 0);
        n_mir = bind_side(mesh, conn, fa, fl, hinge.radius, hardness, mir, has_mir, tube_m);
    }

    for (uint32_t v = 0; v < vc; v++) {
        const bool a = has_own[v] != 0, b = mirror && has_mir[v];
        if (!a && !b) continue;
        float w[8];
        lanes_of(a ? own[v] : 0.0f, w);
        lanes_of(b ? mir[v] : 0.0f, w + 4);
        // Where both reach (the two hinges' limbs touch), they share the vertex.
        if (a && b) for (float& x : w) x *= 0.5f;
        out.ids.push_back(v);
        out.init.push_back(mesh.pos_x[v]); out.init.push_back(mesh.pos_y[v]);
        out.init.push_back(mesh.pos_z[v]);
        out.w.insert(out.w.end(), w, w + 8);
    }
    std::printf("[pose] bind: %u of %u verts move (%s%s), mirror %u%s | %s table, %u seam verts joined\n",
                n_own, vc, tube ? "kept to the limb's tube" : "whole limb past the cut",
                n_own == 0 ? ", NOTHING: lever off the limb?" : "",
                n_mir, mirror ? (tube_m ? " (tube)" : "") : " (off)",
                conn.own_off.empty() ? "mesh's own" : "rebuilt", conn.welded);
}
