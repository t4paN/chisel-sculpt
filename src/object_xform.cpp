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

        auto spin_mesh = [&](Mesh& m) {
            float ang = inverse ? -t.angle : t.angle;
            const Vec3 k = t.axis;
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float omc = 1.0f - cs;
            uint32_t n = m.vertex_count();
            for (uint32_t v = 0; v < n; v++) {
                float px = m.pos_x[v] - t.pivot.x;
                float py = m.pos_y[v] - t.pivot.y;
                float pz = m.pos_z[v] - t.pivot.z;
                float kd = k.x * px + k.y * py + k.z * pz;
                float cx = k.y * pz - k.z * py;
                float cy = k.z * px - k.x * pz;
                float cz = k.x * py - k.y * px;
                m.pos_x[v] = t.pivot.x + px * cs + cx * sn + k.x * kd * omc;
                m.pos_y[v] = t.pivot.y + py * cs + cy * sn + k.y * kd * omc;
                m.pos_z[v] = t.pivot.z + pz * cs + cz * sn + k.z * kd * omc;
            }
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
