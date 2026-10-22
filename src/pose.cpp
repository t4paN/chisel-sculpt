#include "pose.h"
#include "camera.h"
#include <algorithm>
#include <cmath>
#include <cstring>

int PoseRig::add_joint(Vec3 p, float r) {
    joints.push_back({p, r});
    return (int)joints.size() - 1;
}

bool PoseRig::add_bone(int a, int b) {
    if (bones_in_set_of(a) >= kPoseMaxBones) return false;
    bones.push_back({(uint16_t)a, (uint16_t)b});
    return true;
}

void PoseRig::pop_joint() {
    if (joints.empty()) return;
    const uint16_t j = (uint16_t)(joints.size() - 1);
    while (!bones.empty() && (bones.back().a == j || bones.back().b == j)) bones.pop_back();
    joints.pop_back();
}

void PoseRig::remove_joint(int j) {
    if (j < 0 || j >= (int)joints.size()) return;
    joints.erase(joints.begin() + j);
    size_t k = 0;
    for (const PoseBone& b : bones) {
        if (b.a == j || b.b == j) continue;
        bones[k++] = {(uint16_t)(b.a > j ? b.a - 1 : b.a), (uint16_t)(b.b > j ? b.b - 1 : b.b)};
    }
    bones.resize(k);
    forget_moves(j);
    for (PoseMove& m : moves)
        if (m.joint > j) m.joint--;
}

void PoseRig::forget_moves(int j) {
    size_t k = 0;
    for (const PoseMove& m : moves)
        if (m.joint != j) moves[k++] = m;
    moves.resize(k);
}

void PoseRig::connected(int j, std::vector<uint8_t>& flags) const {
    flags.assign(joints.size(), 0);
    if (j < 0 || j >= (int)joints.size()) return;
    flags[j] = 1;
    // A set is a few dozen bones at most: sweep until nothing new joins.
    for (bool grew = true; grew;) {
        grew = false;
        for (const PoseBone& b : bones) {
            if (flags[b.a] != flags[b.b]) { flags[b.a] = flags[b.b] = 1; grew = true; }
        }
    }
}

int PoseRig::bones_in_set_of(int j) const {
    std::vector<uint8_t> f;
    connected(j, f);
    int n = 0;
    for (const PoseBone& b : bones) n += f[b.a] ? 1 : 0;
    return n;
}

void PoseRig::on_undo(uint64_t seq) {
    for (const PoseMove& m : moves)
        if (m.seq == seq && m.joint < joints.size()) joints[m.joint].pos = m.from;
}

void PoseRig::on_redo(uint64_t seq) {
    for (const PoseMove& m : moves)
        if (m.seq == seq && m.joint < joints.size()) joints[m.joint].pos = m.to;
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

int pose_pick(const PoseRig& rig, const Camera& cam, float sx, float sy, int w, int h) {
    int best = -1;
    float best_depth = 1e30f;
    for (int j = 0; j < (int)rig.joints.size(); j++) {
        const PoseJoint& jt = rig.joints[j];
        float px, py;
        cam.world_to_screen(jt.pos, w, h, px, py);
        if (cam.perspective && cam.view_depth(jt.pos) <= 1e-4f) continue;
        const float r = std::max(6.0f, jt.radius * pose_px_per_world(cam, jt.pos, h));
        const float dx = sx - px, dy = sy - py;
        if (dx * dx + dy * dy > r * r) continue;
        const float d = cam.view_depth(jt.pos);
        if (d < best_depth) { best_depth = d; best = j; }
    }
    return best;
}

// ---- pose_bind -------------------------------------------------------------------

namespace {
constexpr int   kSlices = 12;    // envelope samples along a bone
constexpr int   kAround = 16;    // ... and around it
constexpr float kInf    = 1e30f;
constexpr float kTwoPi  = 6.2831853f;

struct BoneEnv {
    Vec3  a, b, u, e1, e2;
    float len, ra, rb;
    float reach;       // farthest the side envelope may extend
    float cap_reach;   // ... and past a free moving end
    int   cap_end;     // -1, or 0 / 1: the end (a / b) that is a free moving end
    int   lane;        // weight lane if this bone moves, else -1
    float side[kSlices * kAround];
};

float smooth01(float x) {
    x = std::min(1.0f, std::max(0.0f, x));
    return x * x * (3.0f - 2.0f * x);
}

// Angle of `r` (perpendicular to the bone) around it, as a continuous bin index.
float around_of(const BoneEnv& e, Vec3 r) {
    float ang = std::atan2(r.dot(e.e2), r.dot(e.e1));
    if (ang < 0.0f) ang += kTwoPi;
    return ang / kTwoPi * (float)kAround;
}

// Bilinear over (slice, around); around wraps.
float side_at(const BoneEnv& e, float t, Vec3 rad) {
    float f = std::min((float)kSlices - 1.0f, std::max(0.0f, t * kSlices - 0.5f));
    int i0 = (int)f, i1 = std::min(i0 + 1, kSlices - 1);
    float ft = f - (float)i0;
    float g = around_of(e, rad) - 0.5f;
    if (g < 0.0f) g += (float)kAround;
    int j0 = (int)g % kAround, j1 = (j0 + 1) % kAround;
    float gt = g - std::floor(g);
    float r0 = e.side[i0 * kAround + j0] * (1 - gt) + e.side[i0 * kAround + j1] * gt;
    float r1 = e.side[i1 * kAround + j0] * (1 - gt) + e.side[i1 * kAround + j1] * gt;
    return r0 * (1 - ft) + r1 * ft;
}

// 1 inside the skin (with a hair of margin for the skin itself and its bumps),
// fading out over a band that hardness sets.
float skin_weight(float rho, float h, float hardness) {
    const float edge = h * 1.12f;
    if (rho <= edge) return 1.0f;
    const float band = h * (0.15f + 0.6f * (1.0f - hardness)) + 1e-9f;
    return 1.0f - smooth01((rho - edge) / band);
}

float bone_weight(const BoneEnv& e, Vec3 p, float hardness) {
    const Vec3 q = p - e.a;
    const float t = q.dot(e.u) / e.len;
    const Vec3 rad = q - e.u * (t * e.len);
    const float rho = rad.length();
    if (t >= 0.0f && t <= 1.0f) return skin_weight(rho, side_at(e, t, rad), hardness);
    // Past an end the envelope fades out across that sphere's radius: at a shared
    // sphere the next bone takes over, at a fixed one the mesh stays put.
    const int end = t < 0.0f ? 0 : 1;
    const float past = (end == 0 ? -t : t - 1.0f) * e.len;
    const float r_end = std::max(end == 0 ? e.ra : e.rb, 1e-6f);
    const float side = skin_weight(rho, side_at(e, end == 0 ? 0.0f : 1.0f, rad), hardness)
                     * (1.0f - smooth01(past / r_end));
    if (end != e.cap_end) return side;
    // A free end being dragged takes what hangs off it (the hand past the wrist), when
    // the tip rule found no clean tip: all of it to 1.5 radii out, fading by 3.5 (a
    // two-radii fade; one radius stepped visibly on coarse meshes). It ramps in from the end plane,
    // where the side envelope still rules, so the two meet without a step.
    const float dist = (p - (end == 0 ? e.a : e.b)).length();
    const float hang = (1.0f - smooth01((dist - 1.5f * r_end) / (2.0f * r_end)))
                     * smooth01(past / r_end);
    return std::max(side, hang);
}
}  // namespace

// One side of the binding: lanes[v*4 + k] = how much moving bone k carries v, for
// every v the rig reaches (has[v] = 1). Both sized to the vertex count by the caller.
static void bind_side(const Mesh& mesh, const PoseRig& rig, int joint,
                      const int* moving, int n_moving, float hardness,
                      std::vector<float>& lanes, std::vector<uint8_t>& has) {
    const uint32_t vc = mesh.vertex_count();

    static std::vector<uint8_t> set;
    rig.connected(joint, set);
    // How many of the set's bones meet at each sphere: a moving end met by just one
    // is free, and carries what hangs off it.
    static std::vector<int> degree;
    degree.assign(rig.joints.size(), 0);
    for (const PoseBone& b : rig.bones)
        if (set[b.a]) { degree[b.a]++; degree[b.b]++; }

    static std::vector<BoneEnv> env;
    env.clear();
    Vec3 lo(kInf, kInf, kInf), hi(-kInf, -kInf, -kInf);
    for (int bi = 0; bi < (int)rig.bones.size(); bi++) {
        const PoseBone& b = rig.bones[bi];
        if (!set[b.a]) continue;
        BoneEnv e;
        e.a = rig.joints[b.a].pos;
        e.b = rig.joints[b.b].pos;
        e.ra = rig.joints[b.a].radius;
        e.rb = rig.joints[b.b].radius;
        Vec3 ab = e.b - e.a;
        e.len = ab.length();
        if (e.len < 1e-6f) continue;
        e.u = ab * (1.0f / e.len);
        Vec3 ref = std::fabs(e.u.x) < 0.9f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        e.e1 = e.u.cross(ref).normalized();
        e.e2 = e.u.cross(e.e1);
        e.reach = 2.0f * std::max(e.ra, e.rb);
        e.cap_end = -1;
        if (b.a == joint && degree[b.a] == 1) e.cap_end = 0;
        if (b.b == joint && degree[b.b] == 1) e.cap_end = 1;
        e.cap_reach = 3.5f * (e.cap_end == 0 ? e.ra : e.rb);
        e.lane = -1;
        for (int k = 0; k < n_moving; k++) if (moving[k] == bi) e.lane = k;
        for (float& h : e.side) h = kInf;
        const float r = std::max(e.reach, e.cap_end >= 0 ? e.cap_reach : 0.0f);
        lo = Vec3(std::min({lo.x, e.a.x - r, e.b.x - r}), std::min({lo.y, e.a.y - r, e.b.y - r}),
                  std::min({lo.z, e.a.z - r, e.b.z - r}));
        hi = Vec3(std::max({hi.x, e.a.x + r, e.b.x + r}), std::max({hi.y, e.a.y + r, e.b.y + r}),
                  std::max({hi.z, e.a.z + r, e.b.z + r}));
        env.push_back(e);
    }
    if (env.empty()) return;

    const float* px = mesh.pos_x.data();
    const float* py = mesh.pos_y.data();
    const float* pz = mesh.pos_z.data();
    auto pos = [&](uint32_t v) { return Vec3(px[v], py[v], pz[v]); };

    // Candidates: the box around the whole set. Everything below only looks at these.
    static std::vector<uint8_t> flag;   // bit 1 candidate, 2 member, 4 reached
    flag.assign(vc, 0);
    for (uint32_t v = 0; v < vc; v++) {
        if (px[v] >= lo.x && px[v] <= hi.x && py[v] >= lo.y && py[v] <= hi.y
            && pz[v] >= lo.z && pz[v] <= hi.z) flag[v] = 1;
    }

    // Skin sampling. Every candidate triangle offers its corners and centroid; a
    // sample counts for a bin when the triangle faces away from the bone there (the
    // first skin a ray from the axis would leave through), and the bin keeps the
    // nearest. The sample's vertex is remembered: it is the bone's own skin, and
    // seeds the connectivity flood.
    static std::vector<uint32_t> seed_side;
    seed_side.assign(env.size() * kSlices * kAround, UINT32_MAX);
    const uint32_t* ix = mesh.indices.data();
    const uint32_t nt = (uint32_t)mesh.indices.size() / 3;
    for (uint32_t t = 0; t < nt; t++) {
        const uint32_t i0 = ix[t * 3], i1 = ix[t * 3 + 1], i2 = ix[t * 3 + 2];
        if (!(flag[i0] | flag[i1] | flag[i2])) continue;
        const Vec3 p0 = pos(i0), p1 = pos(i1), p2 = pos(i2);
        const Vec3 n = (p1 - p0).cross(p2 - p0);
        const Vec3 smp[4] = { p0, p1, p2, (p0 + p1 + p2) * (1.0f / 3.0f) };
        const uint32_t sv[4] = { i0, i1, i2, i0 };
        for (size_t k = 0; k < env.size(); k++) {
            BoneEnv& e = env[k];
            for (int s = 0; s < 4; s++) {
                const Vec3 q = smp[s] - e.a;
                const float tt = q.dot(e.u) / e.len;
                if (tt >= 0.0f && tt <= 1.0f) {
                    const Vec3 rad = q - e.u * (tt * e.len);
                    const float rho = rad.length();
                    if (rho > 1e-9f && rho <= e.reach && n.dot(rad) > 0.0f) {
                        const int bin = std::min(kSlices - 1, (int)(tt * kSlices)) * kAround
                                      + (int)around_of(e, rad) % kAround;
                        if (rho < e.side[bin]) {
                            e.side[bin] = rho;
                            seed_side[k * kSlices * kAround + bin] = sv[s];
                        }
                    }
                }
            }
        }
    }

    // Bins no skin landed in (a coarse mesh, an opening) borrow their slice's average,
    // or the reach if the whole slice is empty.
    for (BoneEnv& e : env) {
        for (int i = 0; i < kSlices; i++) {
            float sum = 0.0f; int n = 0;
            for (int j = 0; j < kAround; j++)
                if (e.side[i * kAround + j] < kInf) { sum += e.side[i * kAround + j]; n++; }
            const float fill = n ? sum / (float)n : e.reach;
            for (int j = 0; j < kAround; j++)
                if (e.side[i * kAround + j] >= kInf) e.side[i * kAround + j] = fill;
        }
    }

    const bool have_adj = mesh.vert_tri_offset.size() == (size_t)vc + 1;
    static std::vector<uint32_t> stack;

    // Tip rule. Cut the model with a plane through the free end being dragged, square
    // to its bone: the piece past the cut that hangs off this limb's own skin is its
    // tip (the rest of the finger, the hand past the wrist), and all of it rides the
    // sphere. A fixed reach cut long tips short and stretched them. It only counts as
    // a tip if it is away from everything else: if the piece runs further than twice
    // the bone's length, spreads wider than 2.5x the limb at the cut (a palm, the
    // body), or comes within any other sphere, it is not a tip, and the reach rule
    // (bone_weight's hang) stays in charge.
    static std::vector<uint8_t> tip;    // 1 visited, 2 tip
    static std::vector<uint32_t> visited;
    int tip_bone = -1;
    float tip_ramp = 1.0f;
    Vec3 tip_e, tip_o;
    for (size_t k = 0; k < env.size(); k++) if (env[k].cap_end >= 0) tip_bone = (int)k;
    if (tip_bone >= 0 && have_adj) {
        const BoneEnv& e = env[tip_bone];
        tip_e = e.cap_end == 1 ? e.b : e.a;
        tip_o = e.cap_end == 1 ? e.u : e.u * -1.0f;
        const float r_end = std::max(e.cap_end == 1 ? e.rb : e.ra, 1e-6f);
        tip_ramp = 0.25f * r_end;
        const float limit = std::max(2.0f * e.len, 4.0f * r_end);
        const float slice = e.len / (float)kSlices;
        const int last = e.cap_end == 1 ? kSlices - 1 : 0;
        float h_end = 0.0f;
        for (int j = 0; j < kAround; j++) h_end = std::max(h_end, e.side[last * kAround + j]);
        const float wide = 2.5f * h_end;
        tip.assign(vc, 0);
        visited.clear();
        stack.clear();
        // Walk from the last slice's skin, through that slice (kept inside its
        // envelope, so a limb lying against the body doesn't leak into it) and on past
        // the cut, where anything goes until the limits say otherwise.
        auto allowed = [&](uint32_t v, bool& escaped) {
            const Vec3 d = pos(v) - tip_e;
            const float along = d.dot(tip_o);
            if (along > 0.0f) {
                if (d.length() > limit || (d - tip_o * along).length() > wide) {
                    escaped = true;
                    return false;
                }
                for (int j = 0; j < (int)rig.joints.size(); j++) {
                    if (j == joint) continue;
                    if ((pos(v) - rig.joints[j].pos).length() < rig.joints[j].radius) {
                        escaped = true;
                        return false;
                    }
                }
                return true;
            }
            if (along < -slice) return false;
            const Vec3 q = pos(v) - e.a;
            const float t = q.dot(e.u) / e.len;
            const Vec3 rad = q - e.u * (t * e.len);
            return rad.length() <= 1.5f * side_at(e, std::min(1.0f, std::max(0.0f, t)), rad);
        };
        bool escaped = false;
        for (int j = 0; j < kAround; j++) {
            const uint32_t v = seed_side[((size_t)tip_bone * kSlices + last) * kAround + j];
            if (v == UINT32_MAX || tip[v]) continue;
            tip[v] = 1; visited.push_back(v); stack.push_back(v);
        }
        while (!stack.empty() && !escaped) {
            const uint32_t v = stack.back();
            stack.pop_back();
            for (uint32_t k = mesh.vert_tri_offset[v]; k < mesh.vert_tri_offset[v + 1] && !escaped; k++) {
                const uint32_t t = mesh.vert_tri_list[k];
                for (int c = 0; c < 3; c++) {
                    const uint32_t w = ix[t * 3 + c];
                    if (tip[w]) continue;
                    tip[w] = 1;
                    visited.push_back(w);
                    if (allowed(w, escaped)) stack.push_back(w);
                    if (escaped) break;
                }
            }
        }
        stack.clear();
        // Without an escape, everything visited past the cut was let through.
        int n_tip = 0;
        for (uint32_t v : visited) {
            tip[v] = 0;
            if (!escaped && (pos(v) - tip_e).dot(tip_o) > 0.0f) { tip[v] = 2; flag[v] |= 1; n_tip++; }
        }
        if (escaped || n_tip == 0) tip_bone = -1;
    } else {
        tip_bone = -1;
    }
    // A tip vertex is fully the end bone's, ramped in over a quarter radius past the
    // cut so it meets the envelope's own fade there without a step.
    auto weight_of = [&](int k, uint32_t v, Vec3 p) {
        const float w = bone_weight(env[k], p, hardness);
        if (k != tip_bone || tip[v] != 2) return w;
        const float ramp = smooth01((p - tip_e).dot(tip_o) / tip_ramp);
        return 1.0f - (1.0f - w) * (1.0f - ramp);
    };

    // Members: inside some bone's envelope.
    for (uint32_t v = 0; v < vc; v++) {
        if (!(flag[v] & 1)) continue;
        const Vec3 p = pos(v);
        for (int k = 0; k < (int)env.size(); k++)
            if (weight_of(k, v, p) > 0.0f) { flag[v] |= 2; break; }
    }

    // Connected to the bones' own skin, through members only. Without the vertex ->
    // triangle table there is no surface to walk, so every member counts.
    stack.clear();
    if (have_adj) {
        auto push = [&](uint32_t v) {
            if (v < vc && (flag[v] & 2) && !(flag[v] & 4)) { flag[v] |= 4; stack.push_back(v); }
        };
        for (uint32_t v : seed_side) if (v != UINT32_MAX) push(v);
        while (!stack.empty()) {
            const uint32_t v = stack.back();
            stack.pop_back();
            for (uint32_t k = mesh.vert_tri_offset[v]; k < mesh.vert_tri_offset[v + 1]; k++) {
                const uint32_t t = mesh.vert_tri_list[k];
                push(ix[t * 3]); push(ix[t * 3 + 1]); push(ix[t * 3 + 2]);
            }
        }
    } else {
        for (uint32_t v = 0; v < vc; v++) if (flag[v] & 2) flag[v] |= 4;
    }

    // Weights. The strongest bone sets how firmly the vertex is held at all; the
    // moving bones then take their share of it. A vertex near a pinned sphere is held
    // by that sphere's bone too, which is what keeps it still.
    for (uint32_t v = 0; v < vc; v++) {
        if (!(flag[v] & 4)) continue;
        const Vec3 p = pos(v);
        float wsum = 0.0f, wmax = 0.0f, lane[4] = {0, 0, 0, 0};
        for (int k = 0; k < (int)env.size(); k++) {
            const float w = weight_of(k, v, p);
            if (w <= 0.0f) continue;
            wsum += w;
            wmax = std::max(wmax, w);
            if (env[k].lane >= 0) lane[env[k].lane] += w;
        }
        if (wsum <= 0.0f) continue;
        const float k = wmax / wsum;
        const float l0 = lane[0] * k, l1 = lane[1] * k, l2 = lane[2] * k, l3 = lane[3] * k;
        if (std::max(std::max(l0, l1), std::max(l2, l3)) < 1e-4f) continue;
        has[v] = 1;
        lanes[v * 4] = l0; lanes[v * 4 + 1] = l1; lanes[v * 4 + 2] = l2; lanes[v * 4 + 3] = l3;
    }
}

void pose_bind(const Mesh& mesh, const PoseRig& rig, int joint,
               const int* moving, int n_moving, float hardness, bool mirror,
               PoseBinding& out) {
    out.ids.clear(); out.init.clear(); out.w.clear();
    const uint32_t vc = mesh.vertex_count();
    if (joint < 0 || joint >= (int)rig.joints.size() || vc == 0) return;

    static std::vector<float> own, mir;
    static std::vector<uint8_t> has_own, has_mir;
    own.assign((size_t)vc * 4, 0.0f);
    has_own.assign(vc, 0);
    bind_side(mesh, rig, joint, moving, n_moving, hardness, own, has_own);

    // Mirror: the rig reflected in x = 0 binds the other side the same way, and
    // pose_apply carries those vertices through the reflected bones. Every pair then
    // moves as each other's reflection, which the pair-map projection (built for
    // brushes that always grab both sides) needs: fed one side only, it averaged
    // half the pairs and skipped the rest, and tore the mesh.
    if (mirror) {
        static PoseRig flipped;
        flipped.joints = rig.joints;
        flipped.bones = rig.bones;
        for (PoseJoint& j : flipped.joints) j.pos.x = -j.pos.x;
        mir.assign((size_t)vc * 4, 0.0f);
        has_mir.assign(vc, 0);
        bind_side(mesh, flipped, joint, moving, n_moving, hardness, mir, has_mir);
    }

    for (uint32_t v = 0; v < vc; v++) {
        const bool a = has_own[v] != 0, b = mirror && has_mir[v];
        if (!a && !b) continue;
        float w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            if (a) w[k] = own[v * 4 + k];
            if (b) w[4 + k] = mir[v * 4 + k];
            sum += w[k] + w[4 + k];
        }
        // Where both reach (a rig on the centre line), they share the vertex.
        if (sum > 1.0f) for (float& x : w) x /= sum;
        out.ids.push_back(v);
        out.init.push_back(mesh.pos_x[v]); out.init.push_back(mesh.pos_y[v]);
        out.init.push_back(mesh.pos_z[v]);
        out.w.insert(out.w.end(), w, w + 8);
    }
}
