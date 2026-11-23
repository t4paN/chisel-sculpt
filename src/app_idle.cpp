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
#include <algorithm>

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

    // Object move: drag selected meshes in view-plane. The Move tool's gizmo comes
    // through here too: an arm moves along its axis only, a corner across the two it
    // sits between, the centre ring in the view plane — each tracking the pointer.
    const bool gizmo_move = input.drag_mode == InputState::DragMode::GIZMO
                         && input.interaction_mode == InputState::InteractionMode::SELECT
                         && input.object_tool == InputState::ObjectTool::MOVE
                         && input.gizmo_handle >= 0;
    if (input.drag_mode == InputState::DragMode::MOVE_OBJECT || (gizmo_move && (dx != 0.0f || dy != 0.0f))) {
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
        Vec3 gpivot;
        GizmoView g;
        if (gizmo_move && selection_centre(scene, false, gpivot, input.mirror_x ? 0 : input.lobe_entity,
                                           input.lobe_side)
            && gizmo_view(camera, gpivot, win_w, win_h, GK_MOVE, g)) {
            const int h = input.gizmo_handle;
            const Vec3 e[3] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
            if (h == GZ_ALL) {
                delta = right * (dx * g.px_world) + up * (-dy * g.px_world);
            } else if (h <= GZ_Z) {
                // The pointer's travel along the arm on screen, as world units along it.
                const float ux = g.ax[h][0], uy = g.ax[h][1], l2 = ux * ux + uy * uy;
                delta = l2 > 1.0f ? e[h] * ((dx * ux + dy * uy) / l2 * g.arm_world) : Vec3(0, 0, 0);
            } else {
                // Split the pointer's travel between the two arms' screen directions.
                const int a1 = (h - GZ_YZ + 1) % 3, a2 = (h - GZ_YZ + 2) % 3;
                const float ax = g.ax[a1][0], ay = g.ax[a1][1], bx = g.ax[a2][0], by = g.ax[a2][1];
                const float det = ax * by - ay * bx;
                if (std::fabs(det) > 60.0f) {
                    const float s1 = (dx * by - dy * bx) / det, s2 = (ax * dy - ay * dx) / det;
                    delta = (e[a1] * s1 + e[a2] * s2) * g.arm_world;
                } else {
                    delta = Vec3(0, 0, 0);
                }
            }
        } else if (gizmo_move) {
            delta = Vec3(0, 0, 0);
        }

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
                // Record it. The X rule is latched on this entity's first
                // frame of the drag and reused for the rest of it, so the
                // undo replays the same split it was moved with even if a
                // lobe drifts close enough to the seam to re-classify.
                xforms.begin(ObjectXform::Kind::MOVE, scene);
                ObjectXformTarget& xt = xforms.target(sel_id);
                const bool first = xt.delta.x == 0.0f && xt.delta.y == 0.0f && xt.delta.z == 0.0f;
                Vec3 c; float r;
                if (first) e->mesh.compute_bounding_sphere(c, r);
                else { c = Vec3(0, 0, 0); r = 1.0f; }
                float rr = (r > 0.0f) ? r : 1.0f;
                bool centered = first && std::fabs(c.x) < 1e-3f * rr;
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
                int  lobe = 0;

                if (first) {
                    // Mirror off and the clicked half of a pair: that half alone, which
                    // breaks the pair's symmetry (said once, while it still held).
                    if (!input.mirror_x && sel_id == input.lobe_entity && input.lobe_side != 0
                        && mesh_is_pair(e->mesh, 1e-3f * rr)) {
                        lobe = input.lobe_side;
                        lock_x = mirror_lobes = false;
                        if (centered) {
                            std::snprintf(input.notification, sizeof(input.notification),
                                          "Pair symmetry disabled, revert with Ctrl+Z to keep");
                            input.notification_timer = 3.0f;
                        }
                    }
                    xt.lock_x       = lock_x;
                    xt.mirror_lobes = mirror_lobes;
                    xt.seam_eps     = seam_eps;
                    xt.lobe         = (int8_t)lobe;
                } else {
                    lock_x       = xt.lock_x;
                    mirror_lobes = xt.mirror_lobes;
                    seam_eps     = xt.seam_eps;
                    lobe         = xt.lobe;
                }
                Vec3 d = delta;
                if (lobe) {
                    // The half may not reach the plane: the undo picks it out by the
                    // sign of x, so crossing over would strand it. Its base cage too.
                    float inner = lobe_inner_x(e->mesh, lobe);
                    if (e->multires.locked) inner = std::min(inner, lobe_inner_x(e->multires.base, lobe));
                    const float floor_dx = 2.0f * seam_eps - inner;   // least lobe * dx allowed
                    if (lobe * d.x < floor_dx) d.x = lobe * floor_dx;
                }
                xt.delta += d;

                // xt carries the rule latched above (lock_x / lobes / seam / half),
                // and the undo replay and the pose log run this same function.
                auto move_mesh = [&](Mesh& m) {
                    move_apply_points(xt, d, m.pos_x.data(), m.pos_y.data(), m.pos_z.data(),
                                      m.vertex_count());
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
            selection_centre_moved();   // the gizmo follows
            screen_buffers_dirty = true;
        }
        if (!gizmo_move) wrap_cursor(window, input, win_w, win_h);
    }

    // Object scale: RMB drag in SELECT mode scales selected meshes about
    // their shared centroid, all three axes alike. Drag right grows, left
    // shrinks (exponential so the feel is uniform regardless of current size).
    // The Scale tool's gizmo comes through here too, per axis: an arm scales
    // along its axis only, a corner the two it sits between, the centre ring
    // all three. Each measures the drag along its own direction on screen.
    using IM_ = InputState::InteractionMode;
    const bool gizmo_scale = input.drag_mode == InputState::DragMode::GIZMO
                          && input.interaction_mode == IM_::SELECT
                          && input.object_tool == InputState::ObjectTool::SCALE
                          && input.gizmo_handle >= 0;
    if ((input.drag_mode == InputState::DragMode::SCALE_OBJECT && dx != 0.0f)
        || (gizmo_scale && (dx != 0.0f || dy != 0.0f))) {
        const auto& sel = scene.selected_ids();
        if (!sel.empty()) {
            scene.materialize_active_cpu();

            // Pivot = centroid of selected bounding centres, x = 0 under Mirror so
            // the piece (and its -x twin) stay exact mirrors. Invariant under a
            // scale about itself, so recomputing per frame is stable.
            Vec3 pivot;
            if (selection_centre(scene, input.mirror_x, pivot)) {
                Vec3 f3(1, 1, 1);
                if (!gizmo_scale) {
                    const float f = std::exp(dx * 0.005f);
                    f3 = Vec3(f, f, f);
                } else {
                    GizmoView g;
                    gizmo_view(camera, pivot, win_w, win_h, GK_SCALE, g);
                    const int h = input.gizmo_handle;
                    // The drag direction this handle reads: its arm, the diagonal
                    // between two arms, or (centre ring) right and up.
                    float ux = 1.0f, uy = -1.0f;
                    if (h <= GZ_Z) { ux = g.ax[h][0]; uy = g.ax[h][1]; }
                    else if (h <= GZ_XY) {
                        const int a1 = (h - GZ_YZ + 1) % 3, a2 = (h - GZ_YZ + 2) % 3;
                        ux = g.ax[a1][0] + g.ax[a2][0]; uy = g.ax[a1][1] + g.ax[a2][1];
                    }
                    const float ul = std::hypot(ux, uy);
                    const float along = ul > 1e-3f ? (dx * ux + dy * uy) / ul : 0.0f;
                    const float f = std::exp(along * 0.006f);
                    if (h == GZ_ALL) f3 = Vec3(f, f, f);
                    else if (h <= GZ_Z) { float* c = &f3.x; c[h] = f; }
                    else { float* c = &f3.x; c[(h - GZ_YZ + 1) % 3] = f; c[(h - GZ_YZ + 2) % 3] = f; }
                }

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
                const bool uneven = scale_is_uneven(f3);
                auto scale_mesh = [&](Mesh& m) { scale_apply_mesh(m, pivot, f3); };

                for (uint32_t sel_id : sel) {
                    MeshEntity* e = scene.find_entity(sel_id);
                    if (!e) continue;
                    uint32_t vc = e->mesh.vertex_count();
                    ObjectXformTarget& xt = xforms.target(sel_id);
                    xt.pivot    = pivot;
                    xt.scale.x *= f3.x; xt.scale.y *= f3.y; xt.scale.z *= f3.z;
                    scale_mesh(e->mesh);
                    if (e->multires.locked) {
                        scale_mesh(e->multires.base);   // see move
                        // An uneven scale bends the tangent frames the detail
                        // layers sit in, as a turn does: same invalidation.
                        if (uneven) {
                            for (auto& lvl : e->multires.frames) lvl.clear();
                            e->multires_gpu.cleanup();
                        }
                    }
                    retopo_transform(e->retopo, scale_mesh, std::cbrt(f3.x * f3.y * f3.z));
                    // Scaling a symmetric piece about an off-plane pivot
                    // takes it off the mirror plane, and no topology changed,
                    // so nothing else would re-measure. See the stroke gate.
                    e->mesh.invalidate_mirror_symmetry();

                    std::vector<uint32_t> local_dirty(vc);
                    for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                    scene.sync_partial_entity(sel_id, local_dirty);
                }
                if (uneven) refresh_active_gpu_residency();
                selection_centre_moved();
                screen_buffers_dirty = true;
            }
        }
        if (!gizmo_scale) wrap_cursor(window, input, win_w, win_h);
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
                        || input.subdiv_confirm_pending
                        || input.drop_confirm_pending
                        || input.export_dialog_active || input.import_dialog_active
                        || input.save_dialog_active
                        || input.keybind_menu_open || input.brush_props_open;
        // The spin keys are whatever Previous/Next alpha are bound to (Q/E by
        // default) — the same keys step the alpha outside Select mode.
        const KeyBind& spin_cw  = input.keybinds[(int)KeyAction::BRUSH_NEXT];
        const KeyBind& spin_ccw = input.keybinds[(int)KeyAction::BRUSH_PREV];
        // Turn every selected mesh by `ang` about `axis`, each about its own pivot.
        auto spin_selection = [&](const Vec3& axis, float ang) {
            for (uint32_t sel_id : scene.selected_ids()) {
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
                xt.axis   = axis;
                xt.angle += ang;
                auto rotate_mesh = [&](Mesh& m) {
                    spin_apply_mesh(m, xt.pivot, axis, ang,
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
            selection_centre_moved();
            screen_buffers_dirty = true;
        };
        // The Rotate tool's gizmo: a ring dragged turns the selection about that world
        // axis, each piece about its own centre as Q/E does. Any direction drags it:
        // right or up turns one way, left or down the other (X the other way round, so
        // it reads like the other two from the default view). Shift: 5-degree steps,
        // like Shift+Q/E, counted from where the drag started, with the running total.
        {
            static bool  was_rot = false;
            static float pend = 0.0f;     // drag not yet turned into steps
            static int   steps = 0;       // this drag's stepped total, in 5s
            const bool rot_drag = input.drag_mode == InputState::DragMode::GIZMO && !modal
                && input.interaction_mode == InputState::InteractionMode::SELECT
                && input.object_tool == InputState::ObjectTool::ROTATE
                && input.gizmo_handle >= GZ_X && input.gizmo_handle <= GZ_Z
                && !scene.selected_ids().empty();
            if (rot_drag && !was_rot) { pend = 0.0f; steps = 0; }
            was_rot = rot_drag;
            if (rot_drag && (dx != 0.0f || dy != 0.0f)) {
                scene.materialize_active_cpu();
                Vec3 axis(0, 0, 0);
                (&axis.x)[input.gizmo_handle] = 1.0f;
                float ang = (dx - dy) * 0.008f;
                if (input.gizmo_handle == GZ_X) ang = -ang;
                xforms.begin(ObjectXform::Kind::SPIN, scene);
                if (!input.shift_held) {
                    pend = 0.0f;
                    spin_selection(axis, ang);
                } else {
                    constexpr float kStep = 0.08726646f;   // 5 degrees
                    pend += ang;
                    int n = 0;
                    while (pend >=  kStep) { pend -= kStep; n++; }
                    while (pend <= -kStep) { pend += kStep; n--; }
                    if (n != 0) {
                        spin_selection(axis, n * kStep);
                        steps += n;
                        static const char* names[3] = {"X", "Y", "Z"};
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Turn %+d\xC2\xB0 about %s", steps * 5, names[input.gizmo_handle]);
                        input.notification_timer = 1.5f;
                    }
                }
            }
        }
        int dir = 0;
        const bool can_spin = input.interaction_mode == InputState::InteractionMode::SELECT
                           && !modal && !input.ctrl_held && !scene.selected_ids().empty();
        // Shift turns Q/E into 5-degree steps (key events, below), so the hold stands down.
        if (can_spin && !input.shift_held) {
            if (spin_cw.bound()  && glfwGetKey(window, spin_cw.key)  == GLFW_PRESS) dir += 1;
            if (spin_ccw.bound() && glfwGetKey(window, spin_ccw.key) == GLFW_PRESS) dir -= 1;
        }

        const double now = glfwGetTime();
        if (dir == 0) {
            rot_active = false;
        } else {
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
            if (dt > 0.0f && ang != 0.0f) spin_selection(rot_axis, ang);
        }

        // Shift+Q/E: one 5-degree step per press (auto-repeat while held), each its own
        // undo step. The axis is the view axis snapped to the world axis it is within
        // ~20 degrees of, so the F-key views (and near enough) turn about an exact axis
        // and 18 steps land a clean 90, with no drift from a slightly tilted camera.
        static Vec3 step_axis(0, 0, 0);            // the last step's, for the running total
        static int  step_total = 0;
        static std::vector<uint32_t> step_sel;
        if (input.spin_step != 0 && !rot_active && can_spin) {
            scene.materialize_active_cpu();
            Vec3 axis = camera.get_view_direction().normalized();
            const float ax = std::fabs(axis.x), ay = std::fabs(axis.y), az = std::fabs(axis.z);
            const char* axis_name = nullptr;
            if (ax >= ay && ax >= az && ax > 0.94f)      { axis = Vec3(axis.x > 0 ? 1.f : -1.f, 0, 0); axis_name = "X"; }
            else if (ay >= az && ay > 0.94f)             { axis = Vec3(0, axis.y > 0 ? 1.f : -1.f, 0); axis_name = "Y"; }
            else if (az > 0.94f)                         { axis = Vec3(0, 0, axis.z > 0 ? 1.f : -1.f); axis_name = "Z"; }
            const float ang = (float)input.spin_step * 0.08726646f;   // 5 degrees
            xforms.begin(ObjectXform::Kind::SPIN, scene);
            spin_selection(axis, ang);
            xforms.commit(scene);
            // Running total, so the user can see when they are at 90. Restarts when
            // the axis or the selection changes.
            const bool same = axis.x == step_axis.x && axis.y == step_axis.y && axis.z == step_axis.z
                           && scene.selected_ids() == step_sel;
            step_total = (same ? step_total : 0) + input.spin_step * 5;
            step_axis = axis;
            step_sel  = scene.selected_ids();
            std::snprintf(input.notification, sizeof(input.notification),
                          "Turn %+d\xC2\xB0 about %s", step_total,
                          axis_name ? axis_name : "the view");
            input.notification_timer = 1.5f;
        }
        input.spin_step = 0;
    }

    // Close the object-transform gesture the frame the drag or the key
    // ends. One press-to-release is one step back: hold E for two
    // seconds and that is a single Ctrl+Z, not a hundred. Sited here so
    // all three paths share it and none can leak a half-open gesture.
    if (input.drag_mode != InputState::DragMode::MOVE_OBJECT
        && input.drag_mode != InputState::DragMode::SCALE_OBJECT
        && input.drag_mode != InputState::DragMode::GIZMO
        && !rot_active)
        xforms.commit(scene);

    // One-shot actions (IDLE only)

    // Ctrl+A in Object mode: every visible mesh joins the selection (the active one kept).
    if (input.select_all_requested) {
        input.select_all_requested = false;
        const std::vector<uint32_t> had = scene.selected_ids();
        uint32_t n = 0;
        for (auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview || !up->visible) continue;
            n++;
            if (std::find(had.begin(), had.end(), up->id) == had.end()) scene.toggle_selected(up->id);
        }
        screen_buffers_dirty = true;
        std::snprintf(input.notification, sizeof(input.notification), "Selected all: %u meshes", n);
        input.notification_timer = 1.5f;
    }

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
    // UV mode's panel picked an island on another mesh it shows: that mesh becomes the
    // active one (its undo, its bake), the others stay in the view.
    if (input.uv_focus) {
        const uint32_t id = input.uv_focus;
        input.uv_focus = 0;
        if (id != scene.active_mesh_id() && scene.focus(id)) {
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
        multires_reclaim_live_topo(ent.multires, ent.mesh);   // ent.mesh is replaced by solo
        cascade_to_level(ent.multires, solo, ent.multires.current_level, &compute, true);
        if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
            solo.mask = std::move(saved_mask);
        arm_geometry_handoff(solo);
        scene.splice_active(solo);  // reuses cascade CSR/normals + full sync
        mesh     = &scene.active_mesh();
        multires = &scene.active_multires();
        scene.refresh_mirror_map();
        refresh_active_gpu_residency();
    };
    // A turn or an uneven scale clears a locked mesh's tangent frames (they are
    // directions on the old surface), and only a cascade rebuilds them. Without one,
    // the next stroke at the same level reprojects its world deltas into disp
    // through EMPTY frames (zero on the GPU, out of range on the CPU path): the
    // stroke shows but never reaches disp, so the next rebuild drops it. So once the
    // gesture is over, cascade the level in place; the result is the same surface.
    {
        static uint64_t tried_at = ~0ull;   // one attempt per edit, never a per-frame loop
        if (app_state == AppState::IDLE && input.drag_mode == InputState::DragMode::NONE
            && !rot_active && tried_at != scene.edit_mark()) {
            const uint32_t active_id = scene.active_entity().id;
            for (auto& up : scene.entities()) {
                if (!up || !up->alive || !up->multires.locked) continue;
                MultiresStack& st = up->multires;
                const int k = st.current_level - st.base_level - 1;
                if (k < 0 || k >= (int)st.disp.size()) continue;
                if (k < (int)st.frames.size() && st.frames[k].size() == st.disp[k].size()) continue;
                std::printf("[frames] mesh %u L%d: frames cleared by a transform, rebuilding\n",
                            up->id, st.current_level);
                if (up->id == active_id) {
                    cascade_active(true);
                } else {
                    Mesh scratch;   // only the frames it fills are wanted
                    cascade_to_level(st, scratch, st.current_level, nullptr);
                }
                screen_buffers_dirty = true;
            }
            tried_at = scene.edit_mark();
        }
    }

    // Revert the last remesh. Deliberately ahead of the per-mode routing
    // below: the Edit-mode side of it exists because per-entity sculpt history cannot
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

    // Which history this Ctrl+Z belongs to is decided by the MODE, not by age.
    // Sculpt (and Paint, which is Sculpt with paint_mode) walks the active mesh's
    // own strokes; Object mode walks the transforms and the dock's deletes/renames.
    // Neither reaches into the other — a turn made in Object mode survives undoing
    // every stroke in Sculpt, and comes off only in Object mode. That is safe only
    // because strokes captured before a transform are carried into the current pose
    // when they apply (UndoStack::bring_to_pose); see object_xform.h.
    //
    // Inside Object mode the transforms and the mesh deletes/renames still share a
    // clock with each other: both are object-level, and the newer one goes first.
    const bool sculpt_scope = input.interaction_mode == InputState::InteractionMode::EDIT;
    const bool object_scope = input.object_mode();

    const UndoEntry* utop = scene.active_undo().peek_undo();
    const UndoEntry* rtop = scene.active_undo().peek_redo();
    // A mesh deleted or renamed in the dock. No redo (a re-delete is one click).
    const uint64_t mop = scene.mesh_op_seq();
    const bool meshop_undo_next = object_scope && mop
        && (!xforms.can_undo() || mop > xforms.undo_seq());
    const bool xform_undo_next = object_scope && !meshop_undo_next && xforms.can_undo();
    const bool xform_redo_next = object_scope && xforms.can_redo();

    const bool have_undo = sculpt_scope ? utop != nullptr
                         : (meshop_undo_next || xform_undo_next);
    const bool have_redo = sculpt_scope ? rtop != nullptr : xform_redo_next;
    if ((input.undo_requested || input.redo_requested) && !undo_reads_pending
        && !(input.undo_requested ? have_undo : have_redo)) {
        // Say why, and point at the other mode when IT has something to take back.
        const bool undo = input.undo_requested;
        const bool strokes_left  = undo ? utop != nullptr : rtop != nullptr;
        const bool objects_left  = undo ? (xforms.can_undo() || mop != 0) : xforms.can_redo();
        const char* msg =
            (!sculpt_scope && strokes_left)
                ? (undo ? "Nothing to undo here - strokes undo in Sculpt mode (1)"
                        : "Nothing to redo here - strokes redo in Sculpt mode (1)")
          : (!object_scope && objects_left)
                ? (undo ? "Nothing to undo here - moves undo in Object mode (2)"
                        : "Nothing to redo here - moves redo in Object mode (2)")
                : (undo ? "Nothing to undo" : "Nothing to redo");
        std::printf("[undo-pick] %s in mode %d: nothing in this mode's history "
                    "(strokes %s, objects %s)\n",
                    undo ? "ctrl-z" : "ctrl-shift-z", (int)input.interaction_mode,
                    strokes_left ? "left" : "none", objects_left ? "left" : "none");
        std::fflush(stdout);
        input.undo_requested = false;
        input.redo_requested = false;
        std::snprintf(input.notification, sizeof(input.notification), "%s", msg);
        input.notification_timer = 1.5f;
    }
    if ((input.undo_requested || input.redo_requested) && !undo_reads_pending) {
        // Which history this press went to, so a surprising step can be traced.
        const bool undo = input.undo_requested;
        std::printf("[undo-pick] %s in mode %d: mesh %u top %s seq %llu, object top seq %llu (%zu), "
                    "mesh-op seq %llu -> %s\n",
                    undo ? "ctrl-z" : "ctrl-shift-z", (int)input.interaction_mode,
                    scene.active_entity().id,
                    undo ? (utop ? "undo" : "none") : (rtop ? "redo" : "none"),
                    (unsigned long long)(undo ? (utop ? utop->seq : 0) : (rtop ? rtop->seq : 0)),
                    (unsigned long long)(undo ? xforms.undo_seq() : xforms.redo_seq()),
                    undo ? xforms.undo_depth() : xforms.redo_depth(),
                    (unsigned long long)mop,
                    sculpt_scope ? "mesh"
                    : undo ? (meshop_undo_next ? "mesh op" : "object") : "object");
        std::fflush(stdout);
    }
    if (input.undo_requested && !undo_reads_pending) {
        input.undo_requested = false;
        if (meshop_undo_next) {
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
            if (xforms.last_bent_frames()) refresh_active_gpu_residency();
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
            if (xforms.last_bent_frames()) refresh_active_gpu_residency();
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
            multires_reclaim_live_topo(*multires, *mesh);   // live mesh is rebuilt below
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
                cascade_to_level(*multires, *mesh, multires->current_level, &compute, true);
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
                cascade_to_level(*multires, solo, multires->current_level, &compute, true);
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
