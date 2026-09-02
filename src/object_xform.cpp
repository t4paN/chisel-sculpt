#include "object_xform.h"
#include "scene.h"
#include "mesh_entity.h"
#include "undo.h"
#include <cmath>

namespace {
// Deep history here is worthless — nobody walks back forty object moves — and
// each gesture is tiny, so the cap is about not leaking across a long session
// rather than about bytes.
constexpr size_t MAX_GESTURES = 64;
}

void spin_apply_mesh(Mesh& m, const Vec3& pivot, const Vec3& axis,
                     float angle, bool mirror_lobes) {
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

    // A mesh mid-load can have positions without normals; never index past them.
    const bool has_norm = (m.norm_x.size() == m.pos_x.size());
    const uint32_t n = m.vertex_count();
    for (uint32_t v = 0; v < n; v++) {
        // -x lobe: reflect in, turn, reflect out. The two flips ARE the whole
        // conjugation — no second rotation matrix, no negated angle to keep in
        // step with the first one. Read the side BEFORE the position is written.
        const bool neg = mirror_lobes && (m.pos_x[v] < 0.0f);

        float x = m.pos_x[v], y = m.pos_y[v], z = m.pos_z[v];
        if (neg) x = -x;
        x -= pivot.x; y -= pivot.y; z -= pivot.z;
        rot(x, y, z);
        x += pivot.x; y += pivot.y; z += pivot.z;
        if (neg) x = -x;
        m.pos_x[v] = x; m.pos_y[v] = y; m.pos_z[v] = z;

        // Normals are DIRECTIONS: same map, no pivot — they turn with the surface
        // but do not translate. Leaving them is not a subtle error: the renderer
        // uploads mesh.norm_* verbatim (Renderer::update_mesh_verts), so a turned
        // mesh comes out lit as though it were still in its old orientation, and
        // the shading looks welded to the geometry. Only a rotation does this,
        // which is why move and scale ignore normals — translation and uniform
        // scale leave a unit normal valid. Under mirror_lobes the map is M R M,
        // whose determinant is +1, so it is a plain rotation here too: no winding
        // flip, no renormalisation needed.
        if (!has_norm) continue;
        float nx = m.norm_x[v], ny = m.norm_y[v], nz = m.norm_z[v];
        if (neg) nx = -nx;
        rot(nx, ny, nz);
        if (neg) nx = -nx;
        m.norm_x[v] = nx; m.norm_y[v] = ny; m.norm_z[v] = nz;
    }

    // A turn about anything but world X leaves the mesh off the mirror plane, so
    // the cached symmetry verdict has to be re-measured. Topology did not change,
    // so nothing else would ever trigger that. The lobe path lands back on exact
    // symmetry and will simply measure ok again.
    m.invalidate_mirror_symmetry();
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
            case ObjectXform::Kind::SCALE: moved |= (t.factor != 1.0f); break;
            case ObjectXform::Kind::SPIN:  moved |= (t.angle  != 0.0f); break;
        }
    }
    if (!moved || pending_.targets.empty()) { pending_ = ObjectXform(); return; }

    pending_.seq = UndoStack::next_seq();
    redo_.clear();
    for (const auto& t : pending_.targets)
        if (MeshEntity* e = scene.find_entity(t.entity_id)) e->undo.clear_redo();
    undo_.push_back(std::move(pending_));
    pending_ = ObjectXform();
    while (undo_.size() > MAX_GESTURES) undo_.pop_front();
}

void ObjectXformStack::clear() {
    undo_.clear();
    redo_.clear();
    abandon();
}

void ObjectXformStack::clear_redo() { redo_.clear(); }

void ObjectXformStack::apply(const ObjectXform& x, bool inverse, Scene& scene) {
    // The active entity's positions may still live only in VRAM after a stroke.
    scene.materialize_active_cpu();

    for (const auto& t : x.targets) {
        MeshEntity* e = scene.find_entity(t.entity_id);
        if (!e) continue;          // deleted since; the rest of the gesture still applies

        // The three maps, each written once and handed both meshes below. Every
        // one is its own exact inverse under the flipped parameter, which is the
        // whole reason object transforms are affordable in undo.
        auto move_mesh = [&](Mesh& m) {
            Vec3 d = inverse ? Vec3(-t.delta.x, -t.delta.y, -t.delta.z) : t.delta;
            uint32_t n = m.vertex_count();
            for (uint32_t v = 0; v < n; v++) {
                float sx = d.x;
                if (t.lock_x) sx = 0.0f;
                else if (t.mirror_lobes) {
                    float px = m.pos_x[v];
                    sx = (px >  t.seam_eps) ?  d.x
                       : (px < -t.seam_eps) ? -d.x : 0.0f;
                }
                m.pos_x[v] += sx;
                m.pos_y[v] += d.y;
                m.pos_z[v] += d.z;
            }
        };

        auto scale_mesh = [&](Mesh& m) {
            float f = inverse ? (1.0f / t.factor) : t.factor;
            uint32_t n = m.vertex_count();
            for (uint32_t v = 0; v < n; v++) {
                m.pos_x[v] = t.pivot.x + (m.pos_x[v] - t.pivot.x) * f;
                m.pos_y[v] = t.pivot.y + (m.pos_y[v] - t.pivot.y) * f;
                m.pos_z[v] = t.pivot.z + (m.pos_z[v] - t.pivot.z) * f;
            }
        };

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
        // Any of the three can leave the mesh off the world mirror plane, and none
        // of them touches topology, so the cached symmetry verdict would otherwise
        // never be re-measured. (spin_apply_mesh already does this; move and scale
        // need it here.)
        e->mesh.invalidate_mirror_symmetry();
        if (e->multires.locked) {
            // The base cage has to follow the surface, or the next cascade
            // regenerates the mesh back where it started.
            run(e->multires.base);
            if (x.kind == ObjectXform::Kind::SPIN) {
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

    last_spin_ = (x.kind == ObjectXform::Kind::SPIN);
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

uint32_t ObjectXformStack::newer_edit_entity(Scene& scene) const {
    if (undo_.empty()) return 0;
    const uint64_t seq = undo_.back().seq;
    for (const auto& t : undo_.back().targets) {
        if (t.entity_id == scene.active_mesh_id()) continue;   // ordered at the call site
        const MeshEntity* e = scene.find_entity(t.entity_id);
        if (!e) continue;
        const UndoEntry* top = e->undo.peek_undo();
        if (top && top->seq > seq) return t.entity_id;
    }
    return 0;
}

bool ObjectXformStack::undo(Scene& scene) {
    if (undo_.empty()) return false;
    ObjectXform x = std::move(undo_.back());
    undo_.pop_back();
    apply(x, true, scene);
    redo_.push_back(std::move(x));
    redo_clock = UndoStack::global_pushes;
    return true;
}

bool ObjectXformStack::redo(Scene& scene) {
    if (redo_.empty()) return false;
    ObjectXform x = std::move(redo_.back());
    redo_.pop_back();
    apply(x, false, scene);
    undo_.push_back(std::move(x));
    return true;
}
