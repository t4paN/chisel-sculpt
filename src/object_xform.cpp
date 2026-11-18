#include "object_xform.h"
#include "scene.h"
#include "mesh_entity.h"
#include "undo.h"
#include "camera.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
// Deep history here is worthless — nobody walks back forty object moves — and
// each gesture is tiny, so the cap is about not leaking across a long session
// rather than about bytes.
constexpr size_t MAX_GESTURES = 64;

// A transform landed on `e`: log it for the entries captured before it, and carry the
// pose brush's hinge along. The hinge is world space like the mesh, and it used to
// stay behind — after a turn the next swing started from where the limb no longer was.
// Its recorded swings move too, since undoing one puts the lever back to its `from`.
void land_pose_step(MeshEntity& e, const PoseStep& s) {
    e.undo.note_pose(s);
    PoseHinge& h = e.pose;
    if (!h.set && h.moves.empty()) return;
    auto carry = [&](Vec3& p) { pose_map_points(s, &p.x, &p.y, &p.z, 1); };
    carry(h.anchor);
    carry(h.lever);
    for (PoseMove& m : h.moves) { carry(m.from); carry(m.to); }
}
}

void move_apply_points(const ObjectXformTarget& t, const Vec3& d,
                       float* x, float* y, float* z, size_t n) {
    for (size_t v = 0; v < n; v++) {
        if (t.lobe && (t.lobe > 0 ? x[v] <= 0.0f : x[v] >= 0.0f)) continue;
        float sx = d.x;
        if (t.lock_x) sx = 0.0f;
        else if (t.mirror_lobes) {
            const float px = x[v];
            sx = (px >  t.seam_eps) ?  d.x
               : (px < -t.seam_eps) ? -d.x : 0.0f;
        }
        x[v] += sx;
        y[v] += d.y;
        z[v] += d.z;
    }
}

void pose_map_points(const PoseStep& s, float* x, float* y, float* z, size_t n) {
    const ObjectXformTarget& t = s.t;
    switch (s.kind) {
        case ObjectXform::Kind::MOVE: {
            const Vec3 d = s.inverse ? Vec3(-t.delta.x, -t.delta.y, -t.delta.z) : t.delta;
            move_apply_points(t, d, x, y, z, n);
            break;
        }
        case ObjectXform::Kind::SCALE: {
            const Vec3 f = s.inverse ? Vec3(1.0f / t.scale.x, 1.0f / t.scale.y, 1.0f / t.scale.z)
                                     : t.scale;
            scale_apply_points(t.pivot, f, x, y, z, n);
            break;
        }
        case ObjectXform::Kind::SPIN:
            spin_apply_points(t.pivot, t.axis, s.inverse ? -t.angle : t.angle, t.mirror_lobes,
                              x, y, z, n);
            break;
    }
}

void spin_apply_points(const Vec3& pivot, const Vec3& axis, float angle, bool mirror_lobes,
                       float* px, float* py, float* pz, size_t n,
                       float* nxs, float* nys, float* nzs) {
    const Vec3& k = axis;
    const float cs = std::cos(angle), sn = std::sin(angle);
    const float omc = 1.0f - cs;

    // Rodrigues about the ORIGIN: v' = v cos + (k x v) sin + k (k.v)(1 - cos).
    // Positions bracket it with the pivot; directions use it bare.
    auto rot = [&](float& x, float& y, float& z) {
        const float kd = k.x * x + k.y * y + k.z * z;
        const float cx = k.y * z - k.z * y;
        const float cy = k.z * x - k.x * z;
        const float cz = k.x * y - k.y * x;
        const float rx = x * cs + cx * sn + k.x * kd * omc;
        const float ry = y * cs + cy * sn + k.y * kd * omc;
        const float rz = z * cs + cz * sn + k.z * kd * omc;
        x = rx; y = ry; z = rz;
    };

    for (size_t v = 0; v < n; v++) {
        // -x lobe: reflect in, turn, reflect out. The two flips ARE the whole
        // conjugation — no second rotation matrix, no negated angle to keep in
        // step with the first one. Read the side BEFORE the position is written.
        const bool neg = mirror_lobes && (px[v] < 0.0f);

        float x = px[v], y = py[v], z = pz[v];
        if (neg) x = -x;
        x -= pivot.x; y -= pivot.y; z -= pivot.z;
        rot(x, y, z);
        x += pivot.x; y += pivot.y; z += pivot.z;
        if (neg) x = -x;
        px[v] = x; py[v] = y; pz[v] = z;

        // Normals are DIRECTIONS: same map, no pivot — they turn with the surface
        // but do not translate. Leaving them is not a subtle error: the renderer
        // uploads mesh.norm_* verbatim (Renderer::update_mesh_verts), so a turned
        // mesh comes out lit as though it were still in its old orientation, and
        // the shading looks welded to the geometry. Only a rotation does this,
        // which is why move and scale ignore normals — translation and uniform
        // scale leave a unit normal valid. Under mirror_lobes the map is M R M,
        // whose determinant is +1, so it is a plain rotation here too: no winding
        // flip, no renormalisation needed.
        if (!nxs) continue;
        float nx = nxs[v], ny = nys[v], nz = nzs[v];
        if (neg) nx = -nx;
        rot(nx, ny, nz);
        if (neg) nx = -nx;
        nxs[v] = nx; nys[v] = ny; nzs[v] = nz;
    }
}

void spin_apply_mesh(Mesh& m, const Vec3& pivot, const Vec3& axis,
                     float angle, bool mirror_lobes) {
    // A mesh mid-load can have positions without normals; never index past them.
    const bool has_norm = (m.norm_x.size() == m.pos_x.size());
    spin_apply_points(pivot, axis, angle, mirror_lobes,
                      m.pos_x.data(), m.pos_y.data(), m.pos_z.data(), m.vertex_count(),
                      has_norm ? m.norm_x.data() : nullptr,
                      has_norm ? m.norm_y.data() : nullptr,
                      has_norm ? m.norm_z.data() : nullptr);

    // A turn about anything but world X leaves the mesh off the mirror plane, so
    // the cached symmetry verdict has to be re-measured. Topology did not change,
    // so nothing else would ever trigger that. The lobe path lands back on exact
    // symmetry and will simply measure ok again.
    m.invalidate_mirror_symmetry();
}

void scale_apply_points(const Vec3& pivot, const Vec3& f,
                        float* x, float* y, float* z, size_t n) {
    for (size_t v = 0; v < n; v++) {
        x[v] = pivot.x + (x[v] - pivot.x) * f.x;
        y[v] = pivot.y + (y[v] - pivot.y) * f.y;
        z[v] = pivot.z + (z[v] - pivot.z) * f.z;
    }
}

void scale_apply_mesh(Mesh& m, const Vec3& pivot, const Vec3& f) {
    const uint32_t n = m.vertex_count();
    scale_apply_points(pivot, f, m.pos_x.data(), m.pos_y.data(), m.pos_z.data(), n);
    // An even scale leaves a unit normal valid. An uneven one does not: a normal maps
    // by the inverse transpose, here just the inverse factors, then back to unit length.
    if (!scale_is_uneven(f) || m.norm_x.size() != m.pos_x.size()) return;
    const float ix = 1.0f / f.x, iy = 1.0f / f.y, iz = 1.0f / f.z;
    for (uint32_t v = 0; v < n; v++) {
        const float nx = m.norm_x[v] * ix, ny = m.norm_y[v] * iy, nz = m.norm_z[v] * iz;
        const float l = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (l <= 1e-20f) continue;
        m.norm_x[v] = nx / l; m.norm_y[v] = ny / l; m.norm_z[v] = nz / l;
    }
}

void spin_latch_target(const Mesh& m, bool mirror_on, ObjectXformTarget& xt) {
    Vec3 c; float r;
    m.compute_bounding_sphere(c, r);
    const float rr = (r > 0.0f) ? r : 1.0f;

    // Default: this entity turns about its OWN centre. That alone is the fix for
    // a multi-selection orbiting one shared point instead of each piece turning
    // where it sits.
    xt.pivot        = c;
    xt.mirror_lobes = false;
    if (!mirror_on) return;

    // Same classification the move block uses, and for the same reason: a bounding
    // centre at x=0 cannot tell a centred single piece from a symmetrized PAIR,
    // whose centre is also 0. A piece whose geometry is continuous across the
    // plane cannot be lobe-turned at all — the two halves would shear apart at the
    // seam — so only a genuinely disjoint pair qualifies.
    if (std::fabs(c.x) >= 1e-3f * rr) return;
    const float band = 1e-3f * rr;
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        bool pos = false, neg = false;
        for (int i = 0; i < 3; i++) {
            float x = m.pos_x[m.indices[t + i]];
            if (x > band) pos = true;
            else if (x < -band) neg = true;
            else { pos = true; neg = true; }   // in the seam band
        }
        if (pos && neg) return;                // spans the seam: no lobe rule
    }

    // The +x lobe's own centre and radius. The -x lobe's are the reflection.
    float mnx = 0, mny = 0, mnz = 0, mxx = 0, mxy = 0, mxz = 0;
    bool  any = false;
    const uint32_t n = m.vertex_count();
    for (uint32_t v = 0; v < n; v++) {
        if (m.pos_x[v] <= 0.0f) continue;
        float x = m.pos_x[v], y = m.pos_y[v], z = m.pos_z[v];
        if (!any) { mnx = mxx = x; mny = mxy = y; mnz = mxz = z; any = true; continue; }
        if (x < mnx) mnx = x;  if (x > mxx) mxx = x;
        if (y < mny) mny = y;  if (y > mxy) mxy = y;
        if (z < mnz) mnz = z;  if (z > mxz) mxz = z;
    }
    if (!any) return;
    Vec3 cp = { 0.5f * (mnx + mxx), 0.5f * (mny + mxy), 0.5f * (mnz + mxz) };

    float rad_sq = 0.0f;
    for (uint32_t v = 0; v < n; v++) {
        if (m.pos_x[v] <= 0.0f) continue;
        float dx = m.pos_x[v] - cp.x, dy = m.pos_y[v] - cp.y, dz = m.pos_z[v] - cp.z;
        float d = dx * dx + dy * dy + dz * dz;
        if (d > rad_sq) rad_sq = d;
    }

    // The guard that makes classifying by the sign of x sound: a lobe turning
    // about its own centre never leaves the sphere it is already inside, so if
    // that sphere clears the plane no vertex can cross it and be re-classified
    // mid-gesture. Fail it and the entity turns as one piece and loses symmetry —
    // the stroke gate catches that and says so, rather than tearing.
    if (cp.x <= std::sqrt(rad_sq)) return;

    xt.pivot        = cp;
    xt.mirror_lobes = true;
}

void ObjectXformStack::begin(ObjectXform::Kind k, Scene& scene) {
    if (open_ && pending_.kind == k) return;
    commit(scene);               // a kind change ends the previous gesture
    pending_ = ObjectXform();
    pending_.kind = k;
    open_ = true;
}

ObjectXformTarget& ObjectXformStack::target(uint32_t entity_id) {
    for (auto& t : pending_.targets)
        if (t.entity_id == entity_id) return t;
    pending_.targets.push_back(ObjectXformTarget());
    pending_.targets.back().entity_id = entity_id;
    return pending_.targets.back();
}

void ObjectXformStack::abandon() {
    pending_ = ObjectXform();
    open_ = false;
}

void ObjectXformStack::commit(Scene& scene) {
    if (!open_) return;
    open_ = false;

    // Drop gestures that did nothing: a click that never dragged, a tap of E
    // short enough that the accumulated angle rounds to nothing. They would
    // otherwise sit on the stack eating Ctrl+Z presses invisibly.
    bool moved = false;
    for (const auto& t : pending_.targets) {
        switch (pending_.kind) {
            case ObjectXform::Kind::MOVE:
                moved |= (t.delta.x != 0.0f || t.delta.y != 0.0f || t.delta.z != 0.0f);
                break;
            case ObjectXform::Kind::SCALE:
                moved |= (t.scale.x != 1.0f || t.scale.y != 1.0f || t.scale.z != 1.0f); break;
            case ObjectXform::Kind::SPIN:  moved |= (t.angle  != 0.0f); break;
        }
    }
    if (!moved || pending_.targets.empty()) { pending_ = ObjectXform(); return; }

    // One line per gesture, so a model that changed shape can be traced to the press
    // that did it (the click log alone can't tell a gizmo drag from a pick).
    for (const auto& t : pending_.targets) {
        switch (pending_.kind) {
            case ObjectXform::Kind::MOVE:
                std::printf("[xform] move mesh %u by (%.4f %.4f %.4f)%s%s%s\n", t.entity_id,
                            t.delta.x, t.delta.y, t.delta.z, t.lock_x ? " lock-x" : "", t.mirror_lobes ? " lobes" : "",
                            t.lobe > 0 ? " +x half only" : t.lobe < 0 ? " -x half only" : "");
                break;
            case ObjectXform::Kind::SCALE:
                std::printf("[xform] scale mesh %u by (%.4f %.4f %.4f) about (%.4f %.4f %.4f)\n", t.entity_id,
                            t.scale.x, t.scale.y, t.scale.z, t.pivot.x, t.pivot.y, t.pivot.z);
                break;
            case ObjectXform::Kind::SPIN:
                std::printf("[xform] turn mesh %u by %.2f deg about (%.3f %.3f %.3f) at (%.4f %.4f %.4f)%s\n",
                            t.entity_id, t.angle * 57.29578f, t.axis.x, t.axis.y, t.axis.z,
                            t.pivot.x, t.pivot.y, t.pivot.z, t.mirror_lobes ? " lobes" : "");
                break;
        }
    }
    std::fflush(stdout);

    pending_.seq = UndoStack::next_seq();
    redo_.clear();
    // The live drag already moved the vertices; the pose log only has to hear of it.
    for (const auto& t : pending_.targets)
        if (MeshEntity* e = scene.find_entity(t.entity_id))
            land_pose_step(*e, PoseStep{pending_.kind, t, false});
    undo_.push_back(std::move(pending_));
    pending_ = ObjectXform();
    while (undo_.size() > MAX_GESTURES) undo_.pop_front();
}

void ObjectXformStack::clear() {
    undo_.clear();
    redo_.clear();
    abandon();
}


void ObjectXformStack::apply(const ObjectXform& x, bool inverse, Scene& scene) {
    // The active entity's positions may still live only in VRAM after a stroke.
    scene.materialize_active_cpu();
    bool bent = false;   // any target turned, or scaled unevenly

    for (const auto& t : x.targets) {
        MeshEntity* e = scene.find_entity(t.entity_id);
        if (!e) continue;          // deleted since; the rest of the gesture still applies

        // The three maps, each written once and handed both meshes below. Every
        // one is its own exact inverse under the flipped parameter, which is the
        // whole reason object transforms are affordable in undo.
        auto move_mesh = [&](Mesh& m) {
            Vec3 d = inverse ? Vec3(-t.delta.x, -t.delta.y, -t.delta.z) : t.delta;
            move_apply_points(t, d, m.pos_x.data(), m.pos_y.data(), m.pos_z.data(),
                              m.vertex_count());
        };

        const Vec3 sf = inverse ? Vec3(1.0f / t.scale.x, 1.0f / t.scale.y, 1.0f / t.scale.z) : t.scale;
        auto scale_mesh = [&](Mesh& m) { scale_apply_mesh(m, t.pivot, sf); };
        const bool bends = x.kind == ObjectXform::Kind::SPIN
                        || (x.kind == ObjectXform::Kind::SCALE && scale_is_uneven(t.scale));
        if (bends) bent = true;

        // Same function the live Q/E path calls, including the lobe rule — the
        // conjugated map is its own inverse under a negated angle, so undoing a
        // mirrored turn lands the pair back on exact symmetry.
        auto spin_mesh = [&](Mesh& m) {
            spin_apply_mesh(m, t.pivot, t.axis,
                            inverse ? -t.angle : t.angle, t.mirror_lobes);
        };

        auto run = [&](Mesh& m) {
            switch (x.kind) {
                case ObjectXform::Kind::MOVE:  move_mesh(m);  break;
                case ObjectXform::Kind::SCALE: scale_mesh(m); break;
                case ObjectXform::Kind::SPIN:  spin_mesh(m);  break;
            }
        };

        uint32_t vc = e->mesh.vertex_count();
        run(e->mesh);
        land_pose_step(*e, PoseStep{x.kind, t, inverse});
        // The low rides along, as it did live (retopo_transform). Its stroke widths take
        // the mean factor (exact for an even scale).
        const float size = x.kind != ObjectXform::Kind::SCALE ? 1.0f
                         : std::cbrt(sf.x * sf.y * sf.z);
        retopo_transform(e->retopo, run, size);
        // Any of the three can leave the mesh off the world mirror plane, and none
        // of them touches topology, so the cached symmetry verdict would otherwise
        // never be re-measured. (spin_apply_mesh already does this; move and scale
        // need it here.)
        e->mesh.invalidate_mirror_symmetry();
        if (e->multires.locked) {
            // The base cage has to follow the surface, or the next cascade
            // regenerates the mesh back where it started.
            run(e->multires.base);
            if (bends) {
                // Tangent frames are DIRECTIONS on the base surface and the disp
                // layers are expressed in them. Translation and uniform scale
                // leave them valid; a turn does not. Same invalidation the live
                // spin does — sizes stay (frames.size() tracks disp.size()),
                // contents go.
                for (auto& lvl : e->multires.frames) lvl.clear();
                e->multires_gpu.cleanup();   // VRAM mirror holds stale frames
            }
        }

        if (dirty_.size() != vc) {
            dirty_.resize(vc);
            for (uint32_t i = 0; i < vc; i++) dirty_[i] = i;
        }
        scene.sync_partial_entity(t.entity_id, dirty_);
    }

    last_spin_ = bent;
    last_kind_ = x.kind;
}

const char* ObjectXformStack::last_label() const {
    switch (last_kind_) {
        case ObjectXform::Kind::MOVE:  return "move";
        case ObjectXform::Kind::SCALE: return "scale";
        case ObjectXform::Kind::SPIN:  return "rotation";
    }
    return "transform";
}

bool ObjectXformStack::undo(Scene& scene) {
    if (undo_.empty()) return false;
    scene.mark_edited();
    ObjectXform x = std::move(undo_.back());
    undo_.pop_back();
    apply(x, true, scene);
    redo_.push_back(std::move(x));
    return true;
}

bool ObjectXformStack::redo(Scene& scene) {
    if (redo_.empty()) return false;
    scene.mark_edited();
    ObjectXform x = std::move(redo_.back());
    redo_.pop_back();
    apply(x, false, scene);
    undo_.push_back(std::move(x));
    return true;
}

// ---- Object mode's gizmos ---------------------------------------------------

namespace {
struct CentreCache {
    bool     valid = false;
    uint64_t mark = 0;
    std::vector<uint32_t> sel;
    uint32_t lobe_entity = 0;
    int      lobe_side = 0;
    Vec3     c = {0, 0, 0};
};
CentreCache g_centre;

constexpr float kArmPx    = 110.0f;   // Scale: arm length on screen
constexpr float kRingPx   = 95.0f;    // Rotate: ring radius on screen
constexpr float kCornerT  = 0.35f;    // Scale: the corner sits this far up both arms
constexpr float kCentrePx = 12.0f;    // Scale: the all-axes ring
constexpr float kGrabPx   = 8.0f;     // how near a line has to be to grab it

const ImU32 kAxisCol[3] = {IM_COL32(232, 64, 64, 255), IM_COL32(108, 200, 64, 255),
                           IM_COL32(64, 128, 240, 255)};
const ImU32 kHotCol = IM_COL32(255, 214, 64, 255);

float seg_dist(float px, float py, float ax, float ay, float bx, float by) {
    const float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    float t = l2 > 0.0f ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0f;
    t = std::max(0.0f, std::min(1.0f, t));
    return std::hypot(px - (ax + dx * t), py - (ay + dy * t));
}
}  // namespace

void selection_centre_moved() { g_centre.valid = false; }

bool mesh_is_pair(const Mesh& m, float band) {
    bool any_pos = false, any_neg = false;
    for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        bool pos = false, neg = false;
        for (int k = 0; k < 3; k++) {
            const float x = m.pos_x[m.indices[t + k]];
            if (x > band) pos = true;
            else if (x < -band) neg = true;
            else return false;               // on the plane: one piece through it
        }
        if (pos && neg) return false;         // a triangle across the plane
        any_pos |= pos; any_neg |= neg;
    }
    return any_pos && any_neg;
}

int lobe_under_pointer(const Mesh& m, const Camera& cam, int win_w, int win_h, float mx, float my) {
    // A click-paced pass, nearest vertex on screen; within a few pixels the nearer one
    // to the eye wins, so a half seen through its twin's silhouette edge isn't taken.
    float best = 1e30f, best_depth = 1e30f;
    int side = 0;
    const uint32_t n = m.vertex_count();
    for (uint32_t v = 0; v < n; v++) {
        const Vec3 p(m.pos_x[v], m.pos_y[v], m.pos_z[v]);
        const float dep = cam.view_depth(p);
        if (cam.perspective && dep <= 1e-4f) continue;
        float sx, sy;
        cam.world_to_screen(p, win_w, win_h, sx, sy);
        const float d2 = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
        const bool closer = d2 < 16.0f && best < 16.0f ? dep < best_depth : d2 < best;
        if (closer) { best = d2; best_depth = dep; side = p.x > 0.0f ? 1 : -1; }
    }
    return side;
}

float lobe_inner_x(const Mesh& m, int side) {
    float inner = 1e30f;
    const uint32_t n = m.vertex_count();
    for (uint32_t v = 0; v < n; v++) {
        const float sx = side * m.pos_x[v];
        if (sx > 0.0f && sx < inner) inner = sx;
    }
    return inner;
}

bool selection_centre(Scene& scene, bool mirror_scale, Vec3& out, uint32_t lobe_entity, int lobe_side) {
    const std::vector<uint32_t>& sel = scene.selected_ids();
    if (sel.empty()) return false;
    if (lobe_side == 0 || sel.size() != 1 || sel[0] != lobe_entity) { lobe_entity = 0; lobe_side = 0; }
    // A full pass over every selected mesh, so only when something changed: the scene's
    // edit clock, the selection, or a live drag (selection_centre_moved).
    if (!g_centre.valid || g_centre.mark != scene.edit_mark() || g_centre.sel != sel
        || g_centre.lobe_entity != lobe_entity || g_centre.lobe_side != lobe_side) {
        scene.materialize_active_cpu();
        Vec3 sum = {0, 0, 0};
        uint32_t n = 0;
        if (lobe_side != 0) {
            // The chosen half of a pair: its bounding-box centre.
            MeshEntity* e = scene.find_entity(lobe_entity);
            Vec3 c; float r;
            if (e && e->mesh.vertex_count()) {
                e->mesh.compute_bounding_sphere(c, r);
                if (mesh_is_pair(e->mesh, 1e-3f * (r > 0.0f ? r : 1.0f))) {
                    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
                    const Mesh& m = e->mesh;
                    for (uint32_t v = 0; v < m.vertex_count(); v++) {
                        if (lobe_side * m.pos_x[v] <= 0.0f) continue;
                        lo.x = std::min(lo.x, m.pos_x[v]); hi.x = std::max(hi.x, m.pos_x[v]);
                        lo.y = std::min(lo.y, m.pos_y[v]); hi.y = std::max(hi.y, m.pos_y[v]);
                        lo.z = std::min(lo.z, m.pos_z[v]); hi.z = std::max(hi.z, m.pos_z[v]);
                    }
                    if (lo.x <= hi.x) { sum = (lo + hi) * 0.5f; n = 1; }
                }
            }
        }
        if (n == 0) for (uint32_t id : sel) {
            MeshEntity* e = scene.find_entity(id);
            if (!e || e->mesh.vertex_count() == 0) continue;
            Vec3 c; float r;
            e->mesh.compute_bounding_sphere(c, r);
            sum += c;
            n++;
        }
        if (n == 0) return false;
        g_centre.c = Vec3(sum.x / n, sum.y / n, sum.z / n);
        g_centre.mark = scene.edit_mark();
        g_centre.sel = sel;
        g_centre.lobe_entity = lobe_entity;
        g_centre.lobe_side = lobe_side;
        g_centre.valid = true;
    }
    out = g_centre.c;
    if (mirror_scale) out.x = 0.0f;
    return true;
}

namespace {
// Convex hull of n screen points (monotone chain), counter-clockwise; returns its size.
int hull2d(ImVec2* p, int n, ImVec2* out) {
    std::sort(p, p + n, [](const ImVec2& a, const ImVec2& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    auto cross = [](const ImVec2& o, const ImVec2& a, const ImVec2& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    int k = 0;
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(out[k - 2], out[k - 1], p[i]) <= 0) k--;
        out[k++] = p[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; i--) {
        while (k >= t && cross(out[k - 2], out[k - 1], p[i]) <= 0) k--;
        out[k++] = p[i];
    }
    return k - 1;
}
constexpr float kConeBase = 0.84f, kConeTip = 1.10f, kConeR = 0.075f;   // along / across an arm
}  // namespace

bool gizmo_view(const Camera& cam, const Vec3& pivot, int win_w, int win_h, int kind, GizmoView& g) {
    const bool rotate = kind == GK_ROTATE;
    g.ok = false;
    if (win_h <= 0) return false;
    const float depth = cam.view_depth(pivot);
    if (cam.perspective && depth <= 1e-4f) return false;   // behind the eye
    float sx, sy;
    cam.world_to_screen(pivot, win_w, win_h, sx, sy);
    g.cx = sx; g.cy = sy;
    const float px_world = 2.0f * cam.half_height_at(depth) / (float)win_h;   // world size of a pixel
    g.px_world = px_world;
    const Vec3 e[3] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
    if (!rotate) {
        const float len = kArmPx * px_world;
        g.arm_world = len;
        for (int a = 0; a < 3; a++) {
            float ex, ey;
            if (cam.perspective && cam.view_depth(pivot + e[a] * len * kConeTip) <= 1e-4f) { g.ax[a][0] = g.ax[a][1] = 0; continue; }
            cam.world_to_screen(pivot + e[a] * len, win_w, win_h, ex, ey);
            g.ax[a][0] = ex - sx; g.ax[a][1] = ey - sy;
            if (kind != GK_MOVE) continue;
            // The tip's cone, as 3D points projected: its base ring, then its apex.
            const Vec3 u = e[(a + 1) % 3], v = e[(a + 2) % 3];
            const Vec3 base = pivot + e[a] * (len * kConeBase), apex = pivot + e[a] * (len * kConeTip);
            for (int i = 0; i < GizmoView::kConeN; i++) {
                const float t = 6.2831853f * (float)i / (float)GizmoView::kConeN;
                cam.world_to_screen(base + (u * std::cos(t) + v * std::sin(t)) * (len * kConeR), win_w, win_h,
                                    g.cone[a][i][0], g.cone[a][i][1]);
            }
            cam.world_to_screen(apex, win_w, win_h, g.cone[a][GizmoView::kConeN][0], g.cone[a][GizmoView::kConeN][1]);
            g.cone_base_shows[a] = cam.view_depth(apex) > cam.view_depth(base);
        }
    } else {
        const float r = kRingPx * px_world;
        for (int a = 0; a < 3; a++) {
            const Vec3 u = e[(a + 1) % 3], v = e[(a + 2) % 3];
            for (int i = 0; i < GizmoView::kRingN; i++) {
                const float t = 6.2831853f * (float)i / (float)GizmoView::kRingN;
                const Vec3 p = pivot + (u * std::cos(t) + v * std::sin(t)) * r;
                float qx, qy;
                cam.world_to_screen(p, win_w, win_h, qx, qy);
                g.ring[a][i][0] = qx; g.ring[a][i][1] = qy;
                g.front[a][i] = cam.view_depth(p) <= depth;
            }
        }
    }
    g.ok = true;
    return true;
}

int gizmo_hit(const GizmoView& g, int kind, float mx, float my) {
    if (!g.ok) return -1;
    const bool rotate = kind == GK_ROTATE;
    const float reach = kind == GK_MOVE ? kConeTip : 1.0f;   // Move's arm runs out to the cone's apex
    if (rotate) {
        // Nearest ring, the half facing you first: where two cross, the front one wins.
        int best = -1;
        float best_d = kGrabPx;
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < GizmoView::kRingN; i++) {
                const int j = (i + 1) % GizmoView::kRingN;
                float d = seg_dist(mx, my, g.ring[a][i][0], g.ring[a][i][1], g.ring[a][j][0], g.ring[a][j][1]);
                if (!g.front[a][i]) d += 3.0f;
                if (d < best_d) { best_d = d; best = a; }
            }
        return best;
    }
    if (std::hypot(mx - g.cx, my - g.cy) <= kCentrePx + 3.0f) return GZ_ALL;
    // Arms: from past the corner out to the tip square. One pointing at you is too short
    // to tell from the centre, so it can't be grabbed.
    int best = -1;
    float best_d = kGrabPx;
    for (int a = 0; a < 3; a++) {
        const float lx = g.ax[a][0], ly = g.ax[a][1];
        if (lx * lx + ly * ly < 24.0f * 24.0f) continue;
        const float d = seg_dist(mx, my, g.cx + lx * kCornerT, g.cy + ly * kCornerT,
                                 g.cx + lx * reach, g.cy + ly * reach);
        if (d < best_d) { best_d = d; best = a; }
    }
    if (best >= 0) return best;
    // Corners: inside the little parallelogram the two arms span up to the corner.
    for (int k = 0; k < 3; k++) {
        const int a = (k + 1) % 3, b = (k + 2) % 3;
        const float ax = g.ax[a][0] * kCornerT, ay = g.ax[a][1] * kCornerT;
        const float bx = g.ax[b][0] * kCornerT, by = g.ax[b][1] * kCornerT;
        const float det = ax * by - ay * bx;
        if (std::fabs(det) < 60.0f) continue;   // seen edge-on
        const float qx = mx - g.cx, qy = my - g.cy;
        const float s = (qx * by - qy * bx) / det, t = (ax * qy - ay * qx) / det;
        if (s >= 0.0f && s <= 1.0f && t >= 0.0f && t <= 1.0f) return GZ_YZ + k;
    }
    return -1;
}

void gizmo_draw(ImDrawList* dl, const GizmoView& g, int kind, int hot) {
    if (!g.ok || !dl) return;
    const bool rotate = kind == GK_ROTATE;
    const ImVec2 c(g.cx, g.cy);
    if (rotate) {
        // Back halves first, thin and faint; then the front halves over them.
        for (int pass = 0; pass < 2; pass++)
            for (int a = 0; a < 3; a++) {
                const bool h = hot == a;
                for (int i = 0; i < GizmoView::kRingN; i++) {
                    const int j = (i + 1) % GizmoView::kRingN;
                    const bool fr = g.front[a][i] && g.front[a][j];
                    if (fr != (pass == 1)) continue;
                    ImU32 col = h ? kHotCol : kAxisCol[a];
                    if (!fr) col = (col & 0x00FFFFFF) | (h ? 0xC0000000 : 0x60000000);
                    dl->AddLine(ImVec2(g.ring[a][i][0], g.ring[a][i][1]), ImVec2(g.ring[a][j][0], g.ring[a][j][1]),
                                col, fr ? (h ? 3.5f : 2.5f) : (h ? 2.0f : 1.25f));
                }
            }
        dl->AddCircleFilled(c, 2.5f, IM_COL32(235, 235, 235, 255));
        return;
    }
    // Corners: a right-angle mark between each two arms, in the third axis' colour.
    for (int k = 0; k < 3; k++) {
        const int a = (k + 1) % 3, b = (k + 2) % 3;
        const ImVec2 pa(c.x + g.ax[a][0] * kCornerT, c.y + g.ax[a][1] * kCornerT);
        const ImVec2 pb(c.x + g.ax[b][0] * kCornerT, c.y + g.ax[b][1] * kCornerT);
        const ImVec2 pc(pa.x + g.ax[b][0] * kCornerT, pa.y + g.ax[b][1] * kCornerT);
        const bool h = hot == GZ_YZ + k;
        if (h) {
            const ImVec2 q[4] = {c, pa, pc, pb};
            dl->AddConvexPolyFilled(q, 4, IM_COL32(255, 214, 64, 70));
        }
        dl->AddLine(pa, pc, h ? kHotCol : kAxisCol[k], h ? 2.5f : 1.75f);
        dl->AddLine(pc, pb, h ? kHotCol : kAxisCol[k], h ? 2.5f : 1.75f);
    }
    // Arms, the farthest drawn first doesn't matter on screen: all three meet at the pivot.
    for (int a = 0; a < 3; a++) {
        const bool h = hot == a || (hot == GZ_ALL && kind == GK_SCALE);
        const ImU32 col = h ? kHotCol : kAxisCol[a];
        if (kind == GK_MOVE) {
            // The arm stops at the cone's base; the cone is its silhouette (the hull of
            // its base ring and apex), its base drawn darker when it faces you.
            const ImVec2 foot(c.x + g.ax[a][0] * kConeBase, c.y + g.ax[a][1] * kConeBase);
            dl->AddLine(c, foot, col, h ? 3.5f : 2.5f);
            ImVec2 pts[GizmoView::kConeN + 1], hull[2 * (GizmoView::kConeN + 1)];
            for (int i = 0; i <= GizmoView::kConeN; i++) pts[i] = ImVec2(g.cone[a][i][0], g.cone[a][i][1]);
            const int nh = hull2d(pts, GizmoView::kConeN + 1, hull);
            if (nh >= 3) dl->AddConvexPolyFilled(hull, nh, col);
            if (g.cone_base_shows[a]) {
                ImVec2 ring[GizmoView::kConeN];
                float area = 0.0f;
                for (int i = 0; i < GizmoView::kConeN; i++) {
                    ring[i] = ImVec2(g.cone[a][i][0], g.cone[a][i][1]);
                    const int j = (i + 1) % GizmoView::kConeN;
                    area += g.cone[a][i][0] * g.cone[a][j][1] - g.cone[a][j][0] * g.cone[a][i][1];
                }
                // Same winding as the hull, or the antialiased fringe lands inside.
                if (area < 0.0f) std::reverse(ring, ring + GizmoView::kConeN);
                const ImU32 dark = IM_COL32((col & 0xFF) * 3 / 5, ((col >> 8) & 0xFF) * 3 / 5,
                                            ((col >> 16) & 0xFF) * 3 / 5, 255);
                dl->AddConvexPolyFilled(ring, GizmoView::kConeN, dark);
            }
            continue;
        }
        const ImVec2 tip(c.x + g.ax[a][0], c.y + g.ax[a][1]);
        dl->AddLine(c, tip, col, h ? 3.5f : 2.5f);
        const float s = h ? 5.5f : 4.5f;
        dl->AddRectFilled(ImVec2(tip.x - s, tip.y - s), ImVec2(tip.x + s, tip.y + s), col);
    }
    const bool h = hot == GZ_ALL;
    dl->AddCircle(c, kCentrePx, h ? kHotCol : IM_COL32(235, 235, 235, 255), 0, h ? 3.0f : 2.0f);
}
