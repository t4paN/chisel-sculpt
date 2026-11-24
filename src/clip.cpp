#include "clip.h"
#include "camera.h"
#include "pose.h"     // pose_screen_ray
#include <cmath>

bool clip_line_usable(const ClipLine& l) {
    const float dx = l.bx - l.ax, dy = l.by - l.ay;
    return dx * dx + dy * dy >= 16.0f;   // 4 px
}

void clip_bind(const Mesh& mesh, const Camera& cam, int w, int h, const ClipLine& line,
               bool mirror, std::vector<uint32_t>& ids, std::vector<float>& targets) {
    ids.clear();
    targets.clear();
    if (!clip_line_usable(line)) return;

    // The wall holds both ends' view rays. Under ortho the rays are parallel (one
    // direction, two origins); under perspective they share the eye. A point on each
    // ray, one step out, covers both: n = da x ((ob + db) - oa).
    Vec3 oa, da, ob, db;
    pose_screen_ray(cam, line.ax, line.ay, w, h, oa, da);
    pose_screen_ray(cam, line.bx, line.by, w, h, ob, db);
    Vec3 n = da.cross((ob + db) - oa);
    const float nl = n.length();
    if (nl < 1e-12f) return;
    n = n * (1.0f / nl);

    // Which way is the shaded side: a pixel just right of the line's middle, on its
    // ray one step out, must land on the positive side.
    {
        const float dx = line.bx - line.ax, dy = line.by - line.ay;
        const float len = std::sqrt(dx * dx + dy * dy);
        float rx = -dy / len, ry = dx / len;   // right of a -> b with y down
        if (line.flip) { rx = -rx; ry = -ry; }
        const float mx = 0.5f * (line.ax + line.bx) + rx * 8.0f;
        const float my = 0.5f * (line.ay + line.by) + ry * 8.0f;
        Vec3 oc, dc;
        pose_screen_ray(cam, mx, my, w, h, oc, dc);
        if (n.dot((oc + dc) - oa) < 0.0f) n = n * -1.0f;
    }

    // Straight onto the wall, square to it. Under ortho the wall holds the view axis,
    // so that keeps each vertex's depth; under perspective it is close to it, and it
    // keeps the cut a true plane either way.
    auto cut = [&](Vec3 p) {
        const float s = n.dot(p - oa);
        return s > 0.0f ? p - n * s : p;
    };

    const bool has_mask = !mesh.mask.empty();
    const uint32_t vc = mesh.vertex_count();
    for (uint32_t v = 0; v < vc; v++) {
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
        ids.push_back(v);
        targets.push_back(t.x);
        targets.push_back(t.y);
        targets.push_back(t.z);
    }
}
