// The IDLE half of the frame's state-specific update, moved out of main.cpp as is:
// camera orbit/pan/zoom drags, Select-mode object move / scale / Q-E spin, mask ops,
// undo/redo, and the IDLE-only one-shot actions. SCULPTING (the hot path) stays in main.
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#include "app.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "mesh.h"
#include "camera.h"
#include "renderer.h"
#include "input.h"
#include "settings.h"
#include "brush.h"
#include "undo.h"
#include "multires_stack.h"
#include "compute.h"
#include "sdf.h"
#include "insert_controller.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include "retopo.h"
#include "scene_snapshot.h"
#include "object_xform.h"

void app_idle_update(App& a, int win_w, int win_h, float dx, float dy) {
    auto& window = a.window;
    auto& input = a.input;
    auto& scene = a.scene;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& camera = a.camera;
    auto& compute = a.compute;
    auto& renderer = a.renderer;
    auto& brush_stroke = a.brush_stroke;
    auto& xforms = a.xforms;
    auto& app_state = a.app_state;
    auto& mesh_center = a.mesh_center;
    auto& mesh_radius = a.mesh_radius;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto& rescue = a.rescue;
    auto& rescue_shown = a.rescue_shown;
    auto& rot_axis = a.rot_axis;
    auto& rot_active = a.rot_active;
    auto& rot_last_time = a.rot_last_time;
    auto& rot_dirty = a.rot_dirty;
    auto& scale_pivot = a.scale_pivot;
    auto print_undo_top = [&](const char* tag) { app_print_undo_top(a, tag); };
    auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(a); };
    auto arm_geometry_handoff = [&](const Mesh& m) { app_arm_geometry_handoff(a, m); };
    // Camera controls
    if (input.drag_mode == InputState::DragMode::ORBIT) {
        camera.orbit(dx, dy);
        screen_buffers_dirty = true;
        wrap_cursor(window, input, win_w, win_h);
    }
    if (input.drag_mode == InputState::DragMode::PAN || input.mouse3_down) {
        camera.pan(dx, dy, win_w, win_h);
        if (dx != 0.0f || dy != 0.0f) screen_buffers_dirty = true;
        wrap_cursor(window, input, win_w, win_h);
    }
    if (input.zoom_modifier_held() && input.mouse2_down) {
        camera.zoom(-dy * 0.05f);
        if (dy != 0.0f) screen_buffers_dirty = true;
        wrap_cursor(window, input, win_w, win_h);
    }

    // Object move: drag selected meshes in view-plane
    if (input.drag_mode == InputState::DragMode::MOVE_OBJECT) {
        // Framing-relative, matching Camera::pan — see the note there for why
        // this is not keyed off camera.distance.
        float scale = camera.half_height() * 0.0048284f;
        Vec3 cam_pos = camera.get_position();
        Vec3 fwd = (camera.target - cam_pos).normalized();
        Vec3 world_up = {0, 1, 0};
        Vec3 right = fwd.cross(world_up).normalized();
        Vec3 up = right.cross(fwd).normalized();
        // Signs chosen so the mesh tracks the cursor 1:1: +dx (cursor right)
        // moves right; -dy (GLFW y grows downward, so cursor up is dy<0) moves
        // up. Both selectmove specs agree on this.
        Vec3 delta = right * (dx * scale) + up * (-dy * scale);

        if (delta.x != 0.0f || delta.y != 0.0f || delta.z != 0.0f) {
            // 2c-iii: if the active entity was left stale by the flipped
            // pen-up path, its mesh.pos must be fresh before move_mesh /
            // compute_bounding_sphere read it (no-op when not dirty).
            scene.materialize_active_cpu();
            for (uint32_t sel_id : scene.selected_ids()) {
                MeshEntity* e = scene.find_entity(sel_id);
                if (!e) continue;
                uint32_t vc = e->mesh.vertex_count();

                // X-axis behaviour depends on WHAT sits on the mirror plane.
                // Bounding centre at x=0 alone can't tell a centered single
                // piece from a symmetrized pair — the pair's centre is also
                // at 0 — so also test whether the geometry is continuous
                // across the plane (some triangle touches/crosses x=0):
                // - Centered single (spans the seam): lock X so it stays
                //   centered, regardless of the symmetry toggle.
                // - Symmetrized pair (two disjoint lobes, nothing crossing):
                //   with symmetry on, move the lobes as exact mirrors
                //   (spread/converge); with symmetry off, translate freely.
                // - Off-centre pieces always translate freely on all axes.
                Vec3 c; float r;
                e->mesh.compute_bounding_sphere(c, r);
                float rr = (r > 0.0f) ? r : 1.0f;
                bool centered = std::fabs(c.x) < 1e-3f * rr;
                float seam_eps = 1e-4f * rr;

                bool spans_seam = false;
                if (centered) {
                    const Mesh& sm = e->mesh;
                    float band = 1e-3f * rr;
                    for (size_t t = 0; t + 2 < sm.indices.size() && !spans_seam; t += 3) {
                        bool pos = false, neg = false;
                        for (int k = 0; k < 3; k++) {
                            float x = sm.pos_x[sm.indices[t + k]];
                            if (x > band) pos = true;
                            else if (x < -band) neg = true;
                            else { pos = true; neg = true; }  // in the seam band
                        }
                        spans_seam = pos && neg;
                    }
                }
                bool lock_x = centered && spans_seam;
                bool mirror_lobes = centered && !spans_seam && input.mirror_x;

                // Record it. The X rule is latched on this entity's first
                // frame of the drag and reused for the rest of it, so the
                // undo replays the same split it was moved with even if a
                // lobe drifts close enough to the seam to re-classify.
                xforms.begin(ObjectXform::Kind::MOVE, scene);
                ObjectXformTarget& xt = xforms.target(sel_id);
                if (xt.delta.x == 0.0f && xt.delta.y == 0.0f && xt.delta.z == 0.0f) {
                    xt.lock_x       = lock_x;
                    xt.mirror_lobes = mirror_lobes;
                    xt.seam_eps     = seam_eps;
                } else {
                    lock_x       = xt.lock_x;
                    mirror_lobes = xt.mirror_lobes;
                    seam_eps     = xt.seam_eps;
                }
                xt.delta += delta;

                auto move_mesh = [&](Mesh& m) {
                    uint32_t n = m.vertex_count();
                    for (uint32_t v = 0; v < n; v++) {
                        float sx = delta.x;
                        if (lock_x) sx = 0.0f;
                        else if (mirror_lobes) {
                            float x = m.pos_x[v];
                            sx = (x >  seam_eps) ?  delta.x
                               : (x < -seam_eps) ? -delta.x : 0.0f;
                        }
                        m.pos_x[v] += sx;
                        m.pos_y[v] += delta.y;
                        m.pos_z[v] += delta.z;
                    }
                };
                move_mesh(e->mesh);
                e->mesh.invalidate_mirror_symmetry();   // see the scale block
                // The base cage follows for EVERY moved entity, not just
                // the active one. It used to be active-only, which left a
                // non-active locked mesh with a moved surface over a base
                // still at the old spot — invisible until you selected it
                // and switched level, at which point the cascade snapped it
                // back. Spin already did it this way; move and scale now
                // agree, which is also what makes their undo an honest
                // inverse rather than an over-correction.
                if (e->multires.locked) move_mesh(e->multires.base);
                retopo_transform(e->retopo, move_mesh, 1.0f);   // its low comes along

                std::vector<uint32_t> local_dirty(vc);
                for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                scene.sync_partial_entity(sel_id, local_dirty);
            }
            screen_buffers_dirty = true;
        }
        wrap_cursor(window, input, win_w, win_h);
    }

    // Object scale: RMB drag in SELECT mode scales selected meshes about
    // their shared centroid. Drag right grows, left shrinks (exponential
    // so the feel is uniform regardless of current size).
    if (input.drag_mode == InputState::DragMode::SCALE_OBJECT && dx != 0.0f) {
        const auto& sel = scene.selected_ids();
        if (!sel.empty()) {
            scene.materialize_active_cpu();

            // Pivot = centroid of selected bounding centres. Invariant under
            // uniform scale about itself, so recomputing per frame is stable.
            Vec3 pivot = {0, 0, 0};
            uint32_t np = 0;
            for (uint32_t sel_id : sel) {
                MeshEntity* e = scene.find_entity(sel_id);
                if (!e) continue;
                Vec3 c; float r;
                e->mesh.compute_bounding_sphere(c, r);
                pivot.x += c.x; pivot.y += c.y; pivot.z += c.z;
                np++;
            }
            if (np > 0) {
                pivot.x /= np; pivot.y /= np; pivot.z /= np;
                // With symmetry on, scale X about the mirror plane so the
                // piece (and its -x twin) stay exact mirrors.
                if (input.mirror_x) pivot.x = 0.0f;

                float f = std::exp(dx * 0.005f);

                // Record it, and latch the pivot on the drag's first
                // frame for the LIVE scale too: a bounding centre is
                // invariant under a scale about itself in exact
                // arithmetic only, so letting it drift per frame would
                // both creep the object and put the undo somewhere the
                // forward pass never was.
                const bool scale_first = !(xforms.gesture_open()
                    && xforms.gesture_kind() == ObjectXform::Kind::SCALE);
                xforms.begin(ObjectXform::Kind::SCALE, scene);
                if (scale_first) scale_pivot = pivot;
                else             pivot       = scale_pivot;

                auto scale_mesh = [&](Mesh& m) {
                    uint32_t n = m.vertex_count();
                    for (uint32_t v = 0; v < n; v++) {
                        m.pos_x[v] = pivot.x + (m.pos_x[v] - pivot.x) * f;
                        m.pos_y[v] = pivot.y + (m.pos_y[v] - pivot.y) * f;
                        m.pos_z[v] = pivot.z + (m.pos_z[v] - pivot.z) * f;
                    }
                };

                for (uint32_t sel_id : sel) {
                    MeshEntity* e = scene.find_entity(sel_id);
                    if (!e) continue;
                    uint32_t vc = e->mesh.vertex_count();
                    ObjectXformTarget& xt = xforms.target(sel_id);
                    xt.pivot   = pivot;
                    xt.factor *= f;
                    scale_mesh(e->mesh);
                    if (e->multires.locked) scale_mesh(e->multires.base);   // see move
                    retopo_transform(e->retopo, scale_mesh, f);
                    // Scaling a symmetric piece about an off-plane pivot
                    // takes it off the mirror plane, and no topology changed,
                    // so nothing else would re-measure. See the stroke gate.
                    e->mesh.invalidate_mirror_symmetry();

                    std::vector<uint32_t> local_dirty(vc);
                    for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                    scene.sync_partial_entity(sel_id, local_dirty);
                }
                screen_buffers_dirty = true;
            }
        }
        wrap_cursor(window, input, win_w, win_h);
    }

    // Object spin: Q/E in SELECT mode turn the selection about the VIEW
    // axis, so the turn follows what you are looking at rather than a world
    // axis — face the model from any angle and Q/E always read as
    // anticlockwise/clockwise on screen.
    //
    // The key is POLLED rather than driven off key events: events give you
    // the OS auto-repeat, which is a delay followed by discrete jumps, and
    // no rate you pick makes that feel continuous. Polling with a real dt is
    // smooth at any frame rate and cannot get stuck on a release the modal
    // gate swallowed. Constant rate, no ramp. Like move and scale the whole
    // press-to-release is recorded as ONE step on the object-transform
    // stack (object_xform.h), which Ctrl+Z reaches from Select mode.
    {
        const bool modal = input.quit_requested || input.remesh_confirm_pending
                        || input.voxel_merge_confirm_pending
                        || input.drop_level_confirm_pending
                        || input.reset_confirm_pending
                        || input.pose_clear_confirm_pending
                        || input.drop_confirm_pending
                        || input.export_dialog_active || input.import_dialog_active
                        || input.save_dialog_active
                        || input.keybind_menu_open || input.brush_props_open;
        // The spin keys are whatever Previous/Next alpha are bound to (Q/E by
        // default) — the same keys step the alpha outside Select mode.
        const KeyBind& spin_cw  = input.keybinds[(int)KeyAction::BRUSH_NEXT];
        const KeyBind& spin_ccw = input.keybinds[(int)KeyAction::BRUSH_PREV];
        int dir = 0;
        if (input.interaction_mode == InputState::InteractionMode::SELECT
            && !modal && !input.ctrl_held && !scene.selected_ids().empty()) {
            if (spin_cw.bound()  && glfwGetKey(window, spin_cw.key)  == GLFW_PRESS) dir += 1;
            if (spin_ccw.bound() && glfwGetKey(window, spin_ccw.key) == GLFW_PRESS) dir -= 1;
        }

        const double now = glfwGetTime();
        if (dir == 0) {
            rot_active = false;
        } else {
            const auto& sel = scene.selected_ids();
            scene.materialize_active_cpu();

            if (!rot_active) {
                // The pivot is now PER ENTITY (latched below, in the record),
                // not one centroid for the whole selection. A shared pivot
                // made everything ORBIT it — select a head and a pair of ears
                // and the ears swung around the head instead of turning where
                // they sat, which is not what "spin" means.
                //
                // get_view_direction() runs camera -> target, i.e. INTO
                // the screen. A right-handed positive turn about an axis
                // pointing away from the viewer reads clockwise, so E is +.
                rot_axis      = camera.get_view_direction().normalized();
                rot_last_time = now;
                rot_active    = true;
                xforms.begin(ObjectXform::Kind::SPIN, scene);
            }

            // Clamped so a hitch (a cascade, a window drag) cannot bank up
            // into one violent jump on the frame after it.
            float dt = (float)(now - rot_last_time);
            rot_last_time = now;
            if (dt > 0.1f) dt = 0.1f;

            const float ang = (float)dir * 1.5707963f * dt;   // 90 deg/sec
            if (dt > 0.0f && ang != 0.0f) {
                for (uint32_t sel_id : sel) {
                    MeshEntity* e = scene.find_entity(sel_id);
                    if (!e) continue;
                    uint32_t vc = e->mesh.vertex_count();
                    ObjectXformTarget& xt = xforms.target(sel_id);
                    // Latch this entity's pivot and lobe rule on its first
                    // frame and reuse them for the rest of the gesture. Both
                    // for the reason the axis is latched — recomputing a
                    // bounding centre per frame lets the piece wander as it
                    // turns, because the centre is only rotation-invariant
                    // in exact arithmetic — and so the undo replays the same
                    // split the turn was made with.
                    if (xt.angle == 0.0f)
                        spin_latch_target(e->mesh, input.mirror_x, xt);
                    xt.axis   = rot_axis;
                    xt.angle += ang;
                    auto rotate_mesh = [&](Mesh& m) {
                        spin_apply_mesh(m, xt.pivot, rot_axis, ang,
                                        xt.mirror_lobes);
                    };
                    rotate_mesh(e->mesh);
                    retopo_transform(e->retopo, rotate_mesh, 1.0f);
                    if (e->multires.locked) {
                        // The base cage has to turn with the surface or the
                        // next cascade regenerates the mesh unrotated.
                        rotate_mesh(e->multires.base);
                        // Tangent frames are DIRECTIONS on the base surface and
                        // the disp layers are expressed in them. Translation and
                        // uniform scale leave them valid — which is why move and
                        // scale ignore them — but a rotation does not. Mark every
                        // level stale so the next cascade rebuilds them on the
                        // turned surface; leaving them re-applies your detail in
                        // the old orientation, one subdiv step later. Sizes stay
                        // (frames.size() must track disp.size()), contents go.
                        for (auto& lvl : e->multires.frames) lvl.clear();
                        e->multires_gpu.cleanup();   // VRAM mirror holds stale frames
                    }
                    if (rot_dirty.size() != vc) {
                        rot_dirty.resize(vc);
                        for (uint32_t i = 0; i < vc; i++) rot_dirty[i] = i;
                    }
                    scene.sync_partial_entity(sel_id, rot_dirty);
                }
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
            }
        }
    }

    // Close the object-transform gesture the frame the drag or the key
    // ends. One press-to-release is one step back: hold E for two
    // seconds and that is a single Ctrl+Z, not a hundred. Sited here so
    // all three paths share it and none can leak a half-open gesture.
    if (input.drag_mode != InputState::DragMode::MOVE_OBJECT
        && input.drag_mode != InputState::DragMode::SCALE_OBJECT
        && !rot_active)
        xforms.commit(scene);

    // One-shot actions (IDLE only)

    // Mesh layers dock: a row clicked. Ctrl toggles it in the selection (as Ctrl+click on
    // the model does in Select mode); a plain click makes it the active mesh, the same
    // switch as a Select-mode click, and F then frames it.
    if (input.mesh_layer_pick) {
        const uint32_t id = input.mesh_layer_pick;
        input.mesh_layer_pick = 0;
        if (input.mesh_layer_pick_add) {
            if (scene.toggle_selected(id)) {
                screen_buffers_dirty = true;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Selection: %u meshes", (uint32_t)scene.selected_ids().size());
                input.notification_timer = 1.5f;
            }
        } else if (scene.select(id)) {
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            scene.refresh_mirror_map();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            brush_stroke.vertex_count = 0;
            brush_stroke.phase = StrokePhase::NONE;
            screen_buffers_dirty = true;
        }
    }
    // ...or renamed (through the scene, so Ctrl+Z can take it back).
    if (input.mesh_layer_rename) {
        scene.rename_entity(input.mesh_layer_rename, input.mesh_layer_rename_to);
        input.mesh_layer_rename = 0;
    }
    // ...or its trash confirmed: make that one the selected mesh, then the Delete path.
    if (input.mesh_layer_delete) {
        const uint32_t id = input.mesh_layer_delete;
        input.mesh_layer_delete = 0;
        if (scene.find_entity(id) && (id == scene.active_mesh_id() || scene.select(id)))
            input.delete_mesh_requested = true;
    }

    // Delete selected mesh (+ mirror pair if linked)
    if (input.delete_mesh_requested) {
        input.delete_mesh_requested = false;
        if (scene.selected()) {
            char gone[64], buf[24];
            std::snprintf(gone, sizeof gone, "%s", Scene::display_name(*scene.selected(), buf, sizeof buf));
            auto r = scene.delete_selected();
            if (r.blocked_only_mesh) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Cannot delete the only mesh");
                input.notification_timer = 2.0f;
            } else if (r.deleted) {
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                // The deleted entity is held, history and all, for
                // Scene::DELETE_GRACE edits (Ctrl+Z brings it back); the
                // now-active entity keeps its own. No clear here.
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Deleted %s, now on %s - %s brings it back", gone,
                              Scene::display_name(scene.active_entity(), buf, sizeof buf),
                              input.key_label(KeyAction::UNDO));
                input.notification_timer = 2.0f;
            }
        }
    }

    // Mode 0: Ctrl+I / Ctrl+A act on the retopo mask; the sculpt's stays as it is.
    if (input.interaction_mode == InputState::InteractionMode::RETOPO) {
        if (input.mask_invert_requested && !input.uv_mode) retopo_mask_invert(a.scene.active_entity().retopo);
        if (input.mask_clear_requested && !input.uv_mode) retopo_mask_clear(a.scene.active_entity().retopo);
        input.mask_invert_requested = input.mask_clear_requested = false;
    }
    // Mask invert (Ctrl+I)
    if (input.mask_invert_requested) {
        input.mask_invert_requested = false;
        if (mesh->mask.empty()) {
            mesh->mask.assign(mesh->vertex_count(), 0.0f);
        } else if (mesh->mask.size() < mesh->vertex_count()) {
            mesh->mask.resize(mesh->vertex_count(), 0.0f);
        }
        // Record the flip as an undoable MASK entry (mirrors mask-clear below),
        // capturing every vertex whose value actually changes (mask != 0.5) as
        // old->new. Without this the invert leaves nothing on the stack and
        // Ctrl+Z silently skips over it.
        UndoEntry mask_e;
        mask_e.kind = UndoEntry::Kind::MASK;
        for (uint32_t v = 0; v < mesh->vertex_count(); v++) {
            float old_val = mesh->mask[v];
            float new_val = 1.0f - old_val;
            if (old_val != new_val) {
                mask_e.verts.push_back(v);
                mask_e.old_mask.push_back(old_val);
                mask_e.new_mask.push_back(new_val);
            }
            mesh->mask[v] = new_val;
        }
        // Invert the finest-level plane too: midpoint averaging is
        // linear (avg(1-a,1-b) == 1-avg(a,b)), so an elementwise flip
        // keeps every level consistent — fine detail inverts in place
        // instead of being re-interpolated from the coarse prefix.
        for (float& mv : multires->mask) mv = 1.0f - mv;
        renderer.update_mask(*mesh);
        if (!mask_e.verts.empty())
            scene.active_undo().push(std::move(mask_e));
    }

    // Mask clear (Ctrl+A when mask brush is active)
    if (input.mask_clear_requested) {
        input.mask_clear_requested = false;
        // Drop the finest-level plane outright: clearing only the
        // working prefix would leave fine-level detail to resurrect
        // on the next level switch.
        multires->mask.clear();
        if (!mesh->mask.empty()) {
            UndoEntry mask_e;
            mask_e.kind      = UndoEntry::Kind::MASK;
            for (uint32_t v = 0; v < mesh->vertex_count() && v < (uint32_t)mesh->mask.size(); v++) {
                if (mesh->mask[v] != 0.0f) {
                    mask_e.verts.push_back(v);
                    mask_e.old_mask.push_back(mesh->mask[v]);
                    mask_e.new_mask.push_back(0.0f);
                    mesh->mask[v] = 0.0f;
                }
            }
            // Sync before move: mask_e.verts invalid after std::move.
            scene.sync_mask_partial_entity(scene.active_mesh_id(), mask_e.verts);
            scene.active_undo().push(std::move(mask_e));
        }
    }

    // Undo/redo act on the active entity's own per-model stack. When the
    // reverted layer sits below the current view level, rebuild the active
    // surface from its multires stack (the pre-multimesh single-mesh flow).
    auto cascade_active = [&](bool needs_cascade) {
        if (!needs_cascade) return;
        // An in-place GPU-resident undo (undo.cpp Case 1) writes the working
        // VBO + disp/base SSBO and only mark_cpu_dirty()s the touched verts —
        // CPU disp/base/pos stay stale. Flush them back NOW, while the surface
        // is still at the current (pre-cascade) level so multires_gpu's dirty
        // vert indices are in range: cascade_to_level below reads CPU disp/base
        // (else the descended surface is wrong), and splice_active shrinks the
        // mesh — a later materialize against the smaller mesh would index its
        // adjacency out of bounds (wasm traps; native was lenient). Same choke
        // every other rebuild path already runs first (level-switch/remesh/merge).
        scene.materialize_active_cpu();
        MeshEntity& ent = scene.active_entity();
        multires_sync_paint(ent.multires, ent.mesh);
        // Same-level cascade: restore the working mask verbatim after the
        // rebuild (exact — no fold/interpolate round-trip blur).
        auto saved_mask = std::move(ent.mesh.mask);
        Mesh solo;
        cascade_to_level(ent.multires, solo, ent.multires.current_level, &compute);
        if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
            solo.mask = std::move(saved_mask);
        arm_geometry_handoff(solo);
        scene.splice_active(solo);  // reuses cascade CSR/normals + full sync
        mesh     = &scene.active_mesh();
        multires = &scene.active_multires();
        scene.refresh_mirror_map();
        refresh_active_gpu_residency();
    };
    // UndoStack is per-entity sculpt history, so outside Edit mode there is
    // still nothing IT can correctly revert — insert placement and selection
    // push no entries. Reverting an unrelated older stroke instead is worse
    // than doing nothing, so that side stays clamped to a no-op. Object
    // transforms are the exception and are handled below: they have their
    // own scene-level stack now, so Ctrl+Z reaches them from Select mode.
    // Revert the last remesh. Deliberately ahead of the Edit-mode clamp
    // below: that clamp exists because per-entity sculpt history cannot
    // correctly revert scene-level work, and this snapshot is exactly the
    // scene-level undo it is missing. A merge is usually driven from Select
    // mode, so gating it on Edit would put the rescue out of reach in the
    // one case it was built for. Only fires once the entity's own history
    // is exhausted, so it is always the LAST step back, never a shortcut.
    // A dab's dirty list lands ASYNCHRONOUSLY, and an undo can move — or, across
    // a level switch, RENUMBER — every vertex. Ids still in flight would then be
    // applied to a mesh they no longer describe, which puts arbitrary vertices
    // in arbitrary directions: the spikes reported on 2026-09-17. finalize
    // already refuses to commit while reads are outstanding; undo never did, and
    // rapid undo presses are exactly how the window gets hit.
    //
    // Deferred, not drained in place: on the web the callbacks that land these
    // reads come from the event loop, so blocking here would deadlock the frame.
    // The request stays set and fires on a later frame, once the stroke's own
    // finalize/post_frame drain has emptied the queue.
    // A remesh job swaps the mesh out on its last step and clears the history;
    // an undo landing mid-job would be overwritten anyway.
    if (input.remesh_in_progress) input.undo_requested = input.redo_requested = false;
    const bool undo_reads_pending =
        (input.undo_requested || input.redo_requested)
        && brush_stroke.dab_readbacks_pending();

    if (input.undo_requested && !undo_reads_pending
        && rescue.valid() && rescue.edits_left() > 0
        && !scene.active_undo().can_undo() && !xforms.can_undo()
        && scene.mesh_op_seq() <= rescue.edits_at_capture) {
        input.undo_requested = false;
        const char* what = rescue.op;
        snapshot_restore(rescue, scene);
        scene.mark_edited();
        rescue_shown = -1;
        // Same fixups the .chisel load path runs: the scene was rebuilt
        // wholesale, so every cached pointer and GPU mirror is stale.
        // No level hint on the mirror refresh — that argument is only an
        // icosphere cache lookup and a restored scene is arbitrary geometry.
        scene.refresh_mirror_map();
        scene.sync();
        mesh     = &scene.active_mesh();
        multires = &scene.active_multires();
        refresh_active_gpu_residency();
        // Framing bounds only — the CAMERA is deliberately left where the
        // user put it while they were inspecting the remesh.
        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
        scene.active_undo().clear(&compute);   // ring cached the dead topology
        xforms.clear();                        // and this scene is a rebuilt one
        brush_stroke.vertex_count = 0;
        brush_stroke.phase = StrokePhase::NONE;
        app_state = AppState::IDLE;
        screen_buffers_dirty = true;
        std::snprintf(input.notification, sizeof(input.notification),
                      "Reverted the %s (%u v, %u t) - no undo history",
                      what, mesh->vertex_count(), mesh->tri_count());
        input.notification_timer = 3.0f;
    }

    // Which of the two histories does this Ctrl+Z belong to? Both stamp
    // their entries from UndoStack's scene-wide clock, so undo takes
    // whichever top is NEWER and redo whichever is OLDER — strict
    // last-in-first-out across the pair. That ordering is not a nicety:
    // UndoEntry stores absolute positions, so undoing a stroke that
    // predates an object transform would drop those vertices back into
    // the pre-transform frame and tear the mesh. Interleaving makes it
    // impossible to undo *through* a transform.
    //
    // A push on either stack kills the other's redo arm, since the user
    // has branched off that timeline. The xform side clears the entity
    // stacks it touches at commit; this clock check covers the reverse.
    if (input.redo_requested && xforms.can_redo()
        && UndoStack::global_pushes != xforms.redo_clock)
        xforms.clear_redo();

    const UndoEntry* utop = scene.active_undo().peek_undo();
    const UndoEntry* rtop = scene.active_undo().peek_redo();
    // A mesh deleted or renamed in the dock: undoable from any mode, when it's the
    // newest thing on the clock. No redo (a re-delete is one click).
    const uint64_t mop = scene.mesh_op_seq();
    const bool meshop_undo_next = mop
        && (!utop || mop > utop->seq) && (!xforms.can_undo() || mop > xforms.undo_seq());
    const bool xform_undo_next = !meshop_undo_next && xforms.can_undo()
        && (!utop || xforms.undo_seq() > utop->seq);
    const bool xform_redo_next = xforms.can_redo()
        && (!rtop || xforms.redo_seq() < rtop->seq);

    // Per-entity sculpt history still can't correctly revert scene-level
    // work outside Edit mode, so the old clamp stands — but an object
    // transform IS scene-level, and it is driven from Select mode, so it
    // has to stay reachable there.
    bool undo_allowed = input.interaction_mode == InputState::InteractionMode::EDIT;
    if (input.undo_requested && !undo_allowed && (xform_undo_next || meshop_undo_next)) undo_allowed = true;
    if (input.redo_requested && !undo_allowed && xform_redo_next) undo_allowed = true;
    if ((input.undo_requested || input.redo_requested) && !undo_allowed
        && !undo_reads_pending) {
        input.undo_requested = false;
        input.redo_requested = false;
        std::snprintf(input.notification, sizeof(input.notification),
                      "Undo only works in Edit mode (1)");
        input.notification_timer = 1.5f;
    }
    if (input.undo_requested && !undo_reads_pending) {
        input.undo_requested = false;
        uint32_t xform_blocker = xform_undo_next ? xforms.newer_edit_entity(scene) : 0;
        if (xform_blocker) {
            // See newer_edit_entity(): a sibling of this transform has been
            // sculpted since, and its stroke positions are absolute.
            std::snprintf(input.notification, sizeof(input.notification),
                          "Undo the newer edits on mesh %u first", xform_blocker);
            input.notification_timer = 2.5f;
        } else if (meshop_undo_next) {
            std::string what;
            if (scene.undo_mesh_op(what)) {
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                scene.refresh_mirror_map();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                std::snprintf(input.notification, sizeof(input.notification), "%s", what.c_str());
                input.notification_timer = 2.0f;
            }
        } else if (xform_undo_next) {
            xforms.undo(scene);
            if (xforms.last_was_spin()) refresh_active_gpu_residency();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            std::snprintf(input.notification, sizeof(input.notification),
                          "Undid object %s", xforms.last_label());
            input.notification_timer = 1.5f;
        } else {
            const UndoEntry* top = scene.active_undo().peek_undo();
            const uint64_t seq = top ? top->seq : 0;
            cascade_active(scene.active_undo().undo(scene.active_entity(), scene));
            scene.active_entity().pose.on_undo(seq);
            print_undo_top("ctrl-z");
        }
        screen_buffers_dirty = true;
    }
    if (input.redo_requested && !undo_reads_pending) {
        input.redo_requested = false;
        if (xform_redo_next) {
            xforms.redo(scene);
            if (xforms.last_was_spin()) refresh_active_gpu_residency();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            std::snprintf(input.notification, sizeof(input.notification),
                          "Redid object %s", xforms.last_label());
            input.notification_timer = 1.5f;
        } else {
            const UndoEntry* top = scene.active_undo().peek_redo();
            const uint64_t seq = top ? top->seq : 0;
            cascade_active(scene.active_undo().redo(scene.active_entity(), scene));
            scene.active_entity().pose.on_redo(seq);
            print_undo_top("ctrl-shift-z");
        }
        screen_buffers_dirty = true;
    }

    // Manual projection (P post-lock)
    if (input.project_requested) {
        input.project_requested = false;
        const int L_max = multires->base_level + (int)multires->disp.size();
        const int target = multires->current_level;
        if (target >= L_max) {
            std::printf("[project] already at top level, nothing to project\n");
        } else {
            std::printf("[project] projecting from level %d (truth) down to level %d (current)\n",
                        L_max, target);
            std::printf("[project]   snapshotting levels %d..%d\n",
                        target, L_max);

            UndoEntry e;
            e.kind         = UndoEntry::Kind::PROJECTION;
            e.target_level = target;
            capture_projection_snapshot(*multires, target, e.before);

            ProjectionStats ps = project_down_to_level(*multires, target);
            report_projection_check(ps);

            for (int k = L_max; k > target; k--)
                std::printf("[project]   inverse-Loop %d -> %d ... done\n", k, k - 1);

            int k_first = (target == multires->base_level) ? 0
                            : (target - multires->base_level - 1);
            std::printf("[project]   rewriting disp[%d..%d] ... done\n",
                        k_first, (int)multires->disp.size() - 1);

            std::printf("[project] done in %.2f ms\n", ps.elapsed_ms);
#ifdef CHISEL_DEBUG_MULTIRES
            std::printf("[project][debug] max reconstruction error at L%d: %.3e\n",
                        L_max, ps.max_reconstruction_error);
#endif

            scene.active_undo().push(std::move(e));
            print_undo_top("project");

            multires_sync_paint(*multires, *mesh);
            if (scene.alive_count() <= 1) {
                auto saved_mask = std::move(mesh->mask);
                cascade_to_level(*multires, *mesh, multires->current_level, &compute);
                if (!saved_mask.empty() && saved_mask.size() == mesh->vertex_count())
                    mesh->mask = std::move(saved_mask);
                arm_geometry_handoff(*mesh);
                scene.refresh_mirror_map();
                scene.sync();
            } else {
                // Same-level cascade — keep the exact working mask (see
                // the undo cascade above).
                auto saved_mask = std::move(mesh->mask);
                Mesh solo;
                cascade_to_level(*multires, solo, multires->current_level, &compute);
                if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
                    solo.mask = std::move(saved_mask);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);
                scene.refresh_mirror_map();
            }
            refresh_active_gpu_residency();
            screen_buffers_dirty = true;
        }
    }
}
