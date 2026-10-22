#include "pose.h"
#include "camera.h"
#include <algorithm>
#include <cmath>

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
