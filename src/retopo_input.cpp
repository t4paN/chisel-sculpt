// Retopo mode's (key 0) share of the main loop: what the pen, clicks and panel
// requests do to the retopo state each frame, the F framing, the ring, and the
// overlay + dock. retopo.cpp is the geometry; this is the wiring, kept out of main.cpp.
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include "retopo.h"
#include "input.h"
#include "renderer.h"
#include "tablet.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// F in retopo: frame the piece being worked on, whole, from the current angle:
// the connected faces nearest the last press (a mirror twin is a piece of its
// own, so only this side), or an unbuilt stroke if one of those is nearer.
// A piece bigger than a few quads round the press (a cage over the whole
// model) frames just that neighbourhood, centred on the press: framing it
// whole is framing the sculpt. No press yet: the newest stroke as drawn.
bool retopo_focus(const RetopoState& retopo, Camera& camera, float mesh_radius) {
    if (retopo.strokes.empty()) return false;
    const RetopoMesh& rm = retopo.result;
    const RetopoStroke* stroke = nullptr;
    uint32_t vert = ~0u;
    if (retopo.work_valid) {
        float best = 1e30f;
        if (retopo.built)
            for (size_t i = 0; i < rm.v.size(); i++) {
                const float d = (rm.v[i] - retopo.work_pos).length();
                if (d < best) { best = d; vert = (uint32_t)i; }
            }
        for (const RetopoStroke& s : retopo.strokes) {
            if (s.frozen) continue;
            bool has_faces = false;
            for (uint32_t o : rm.face_owner) if (o == s.id) { has_faces = true; break; }
            if (has_faces && retopo.built) continue;   // its faces stand for it
            for (const Vec3& p : s.pts) {
                const float d = (p - retopo.work_pos).length();
                if (d < best) { best = d; stroke = &s; vert = ~0u; }
            }
        }
    } else {
        for (const RetopoStroke& s : retopo.strokes)
            if (!stroke || s.id > stroke->id) stroke = &s;
    }
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    auto grow = [&](Vec3 p) {
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    };
    if (vert != ~0u) {
        // The piece: vertices joined through faces (union-find over corners).
        std::vector<uint32_t> up(rm.v.size());
        for (uint32_t i = 0; i < up.size(); i++) up[i] = i;
        auto root = [&](uint32_t x) {
            while (up[x] != x) x = up[x] = up[up[x]];
            return x;
        };
        for (uint32_t f = 0; f < rm.face_count(); f++) {
            const uint32_t* q = rm.face(f);
            for (uint32_t i = 1; i < rm.face_size(f); i++) up[root(q[i])] = root(q[0]);
        }
        const uint32_t r0 = root(vert);
        for (uint32_t i = 0; i < up.size(); i++) if (root(i) == r0) grow(rm.v[i]);
    } else if (stroke) {
        for (const Vec3& p : stroke->pts) grow(p);
    }
    if (lo.x <= hi.x) {
        float r = std::max(0.5f * (hi - lo).length(), mesh_radius * 0.02f);
        Vec3 c = (lo + hi) * 0.5f;
        if (vert != ~0u) {
            float edge = 0.0f;
            for (size_t e = 0; e + 1 < rm.edges.size(); e += 2)
                edge += (rm.v[rm.edges[e]] - rm.v[rm.edges[e + 1]]).length();
            edge /= (float)std::max<size_t>(1, rm.edges.size() / 2);
            const float near_r = std::max(6.0f * edge, mesh_radius * 0.02f);
            if (r > near_r) { r = near_r; c = retopo.work_pos; }
        }
        camera.set_target(c);
        camera.distance = r * 2.5f;
        std::printf("[retopo] focus: %s, radius %.3f\n", vert != ~0u ? "built piece" : "stroke", r);
        return true;
    }
    return false;
}

// Stroke capture, its own undo, Build.
// Strokes are sampled off the pen-down plane cache like a brush anchor, so
// capture never reads the GPU back. Retopo never edits the sculpt, so the
// cache stays good for the whole stroke.
void retopo_frame(RetopoState& retopo, InputState& input, Scene& scene, Mesh& mesh,
                  Renderer& renderer, const Camera& camera, const Tablet& tablet,
                  int win_w, int win_h, bool screen_buffers_dirty, bool imgui_wants_mouse) {
    const bool retopo_on = input.interaction_mode == InputState::InteractionMode::RETOPO;
    // The sculpt under screen point (mx, my), off the pen-down plane cache: the point,
    // its normal, and world units per pixel at that depth. False off the model.
    auto retopo_surface_at = [&](float mx, float my, Vec3& p, Vec3& n, float& world_per_px) {
        float d = 0.0f;
        if (!renderer.sample_depth_bilinear(mx, my, &d, 0.0f) || d >= 500.0f) return false;
        Vec3 cam_pos = camera.get_position();
        Vec3 fwd = (camera.target - cam_pos).normalized();
        Vec3 right = fwd.cross(Vec3(0, 1, 0)).normalized();
        Vec3 up = right.cross(fwd).normalized();
        const float half_h = camera.half_height_at(d);
        const float half_w = half_h * ((float)win_w / (float)win_h);
        const float ndc_x = 2.0f * mx / (float)win_w - 1.0f;
        const float ndc_y = 1.0f - 2.0f * my / (float)win_h;
        p = cam_pos + fwd * d + right * (ndc_x * half_w) + up * (ndc_y * half_h);
        float nt[3];
        n = renderer.sample_normal((int)mx, (int)my, nt)
          ? Vec3(nt[0], nt[1], nt[2]).normalized() : fwd * -1.0f;
        world_per_px = 2.0f * half_h / (float)win_h;
        return true;
    };
    // Smooth (Shift / double Shift) works on the built quads, not the strokes.
    // Decided at the press: letting go of Shift mid-drag doesn't start a stroke.
    static bool retopo_smoothing = false;
    static bool retopo_grabbing = false;
    static bool retopo_erasing = false;
    static bool retopo_erasing_faces = false;   // Ctrl was held at the press
    static bool retopo_masking = false;
    static bool retopo_mask_smooth = false;     // Shift was held at the press: soften the edge
    const bool retopo_press = retopo_on && input.drag_mode == InputState::DragMode::RETOPO;
    // Shift+Ctrl in Draw: a path-pick click (below), never a stroke. Decided at the press.
    static bool retopo_path_press = false;
    // Where the pen went down. A stroke can start a few frames later (plane cache
    // still dirty) with the pen already moving; what it starts on is judged here.
    static float retopo_press_x = 0.0f, retopo_press_y = 0.0f;
    // The press landed on a lit vertex or edge: the line's first point goes there.
    static bool retopo_snap_first = false;
    static Vec3 retopo_snap_p, retopo_snap_n;
    {
        static bool was_press = false;
        if (retopo_press && !was_press) {
            retopo_path_press = input.shift_held && input.ctrl_held
                             && input.retopo_tool == InputState::RetopoTool::DRAW;
            retopo_press_x = (float)input.mouse_x;
            retopo_press_y = (float)input.mouse_y;
        }
        Vec3 p, n;
        float wpp;
        if (retopo_press && !was_press && !screen_buffers_dirty
            && retopo_surface_at((float)input.mouse_x, (float)input.mouse_y, p, n, wpp)) {
            retopo.work_pos = p;
            retopo.work_valid = true;
        }
        was_press = retopo_press;
    }
    // Wand lasso let go: clean up what it holds. A few pixels is a click (the menu).
    if (!retopo_press && !retopo.lasso.empty()) {
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        for (size_t i = 0; i + 1 < retopo.lasso.size(); i += 2) {
            x0 = std::min(x0, retopo.lasso[i]); x1 = std::max(x1, retopo.lasso[i]);
            y0 = std::min(y0, retopo.lasso[i + 1]); y1 = std::max(y1, retopo.lasso[i + 1]);
        }
        if (retopo_on && retopo.lasso.size() >= 6 && std::max(x1 - x0, y1 - y0) > 8.0f) {
            if (retopo.wand_open) retopo_wand_close(retopo, retopo.wand_keep_shown ? retopo.wand_shown : -1);
            std::snprintf(input.notification, sizeof(input.notification), "%s",
                          retopo_lasso_cleanup(retopo, camera, win_w, win_h, retopo.lasso, input.mirror_x));
            input.notification_timer = 2.5f;
        }
        retopo.lasso.clear();
    }
    // A line's pen-up that opened a wand menu (spine, notch, edge to vertex): the
    // menu goes up, or the toast says why not; the line is done either way.
    auto line_menu = [&](const char* why) {
        if (why) {
            std::snprintf(input.notification, sizeof(input.notification), "%s", why);
            input.notification_timer = 2.5f;
        } else {
            input.retopo_wand_request = true;
        }
        retopo.end_line();
    };
    // Pen-up of a line (bridge, spine) on a lit vertex or edge: it ends exactly
    // there. The last sample, if it's that close anyway, moves onto it.
    if (!retopo_press && retopo.drawing && !retopo.extruding
        && (retopo.bridge_from || retopo.spine_from != ~0u) && !retopo.live.pts.empty()) {
        retopo_draw_snap(retopo, camera, win_w, win_h, (float)input.mouse_x, (float)input.mouse_y, true);
        Vec3 sp, sn;
        if (retopo_draw_snap_point(retopo, sp, sn)) {
            float lx, ly, qx, qy;
            const bool close = retopo.live.pts.size() >= 2
                && camera.world_to_screen(retopo.live.pts.back(), win_w, win_h, lx, ly)
                && camera.world_to_screen(sp, win_w, win_h, qx, qy)
                && std::hypot(lx - qx, ly - qy) < 4.0f;
            if (close) {
                retopo.live.pts.back() = sp;
                retopo.live.nrm.back() = sn;
            } else {
                retopo.live.pts.push_back(sp);
                retopo.live.nrm.push_back(sn);
                retopo.live.half_w.push_back(retopo.live.half_w.back());
            }
        }
    }
    if (retopo_press && !retopo.drawing
        && (retopo_masking || (input.retopo_tool == InputState::RetopoTool::MASK && !retopo_smoothing))) {
        // Mask: the sculpt Mask brush's own strength and hardness, Ctrl unmasks,
        // Shift (at the press) blends the mask toward its neighbours instead.
        if (!retopo_masking) {
            if (!retopo.built) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Nothing to mask: Build first (Space)");
                input.notification_timer = 2.0f;
                input.drag_mode = InputState::DragMode::NONE;
            } else {
                retopo.begin_edit();
                retopo_masking = true;
                retopo_mask_smooth = input.is_smooth_active();
            }
        }
        if (retopo_masking) {
            const BrushSettings& mb = input.per_brush[(int)BrushType::MASK];
            if (retopo_mask_smooth)
                retopo_mask_smooth_dab(retopo, camera, win_w, win_h, (float)input.mouse_x, (float)input.mouse_y,
                                       input.brush_size, 0.5f, mb.hardness, input.mirror_x);
            else
                retopo_mask_dab(retopo, camera, win_w, win_h, (float)input.mouse_x, (float)input.mouse_y,
                                input.brush_size, mb.strength, mb.hardness, input.is_subtract_active(),
                                input.mirror_x);
        }
    } else if (retopo_press && !retopo.drawing
        && (retopo_smoothing || input.is_smooth_active())) {
        if (!retopo_smoothing) {
            if (!retopo.built) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Nothing to smooth: Build first (Space)");
                input.notification_timer = 2.0f;
                input.drag_mode = InputState::DragMode::NONE;
            } else {
                scene.materialize_active_cpu();   // the snap re-reads mesh.pos
                retopo.begin_edit();
                retopo_edit_begin(retopo, mesh);
                retopo_smoothing = true;
            }
        }
        if (retopo_smoothing)
            retopo_smooth_dab(retopo, camera, win_w, win_h, (float)input.mouse_x,
                              // Fixed for now: Size is the only knob (user).
                              (float)input.mouse_y, input.brush_size,
                              0.5f, input.mirror_x);
    } else if (retopo_press && !retopo.drawing
               && (retopo_grabbing || input.retopo_tool == InputState::RetopoTool::GRAB)) {
        // Grab: the press takes what's under the ring, the drag carries it.
        static float grab_x = 0.0f, grab_y = 0.0f;
        if (!retopo_grabbing) {
            if (!retopo.built) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Nothing to grab: Build first (Space)");
                input.notification_timer = 2.0f;
                input.drag_mode = InputState::DragMode::NONE;
            } else {
                scene.materialize_active_cpu();
                retopo.begin_edit();
                retopo_edit_begin(retopo, mesh);
                grab_x = (float)input.mouse_x;
                grab_y = (float)input.mouse_y;
                retopo_grab_begin(retopo, camera, win_w, win_h, grab_x, grab_y,
                                  input.brush_size, input.mirror_x);
                retopo_grabbing = true;
            }
        }
        if (retopo_grabbing)
            retopo_grab_drag(retopo, camera, win_w, win_h,
                             (float)input.mouse_x - grab_x, (float)input.mouse_y - grab_y);
    } else if (retopo_press && !retopo.drawing
               && (retopo_erasing || input.retopo_tool == InputState::RetopoTool::ERASE)) {
        // Erase: a dab at the press and each time the pointer moves on. Holding
        // still takes nothing more, so it only eats where it's led.
        static float er_x = 0.0f, er_y = 0.0f;
        const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
        if (!retopo_erasing) {
            if (!retopo.built) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Nothing to erase: Build first (Space)");
                input.notification_timer = 2.0f;
                input.drag_mode = InputState::DragMode::NONE;
            } else {
                retopo.begin_edit();
                retopo_erasing = true;
                retopo_erasing_faces = input.ctrl_held;
                er_x = mx - 1e3f;   // the press always dabs
            }
        }
        if (retopo_erasing && std::hypot(mx - er_x, my - er_y) >= 2.0f) {
            if (retopo_erasing_faces)
                retopo_erase_face_dab(retopo, camera, win_w, win_h, mx, my, input.brush_size, input.mirror_x);
            else
                retopo_erase_dab(retopo, camera, win_w, win_h, mx, my, input.brush_size, input.mirror_x);
            er_x = mx;
            er_y = my;
        }
    } else if (retopo_press && !retopo.drawing && !screen_buffers_dirty
               && (retopo.cutting || input.retopo_tool == InputState::RetopoTool::CUT)) {
        // Cut: a bare line on the surface, sampled like a stroke, no band.
        const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
        if (!retopo.cutting && !retopo.built) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Nothing to cut: Build first (Space)");
            input.notification_timer = 2.0f;
            input.drag_mode = InputState::DragMode::NONE;
        } else if (!retopo.cutting) {
            retopo.cutting = true;
            retopo.cut_from_v = retopo.snap_v;
            retopo.cut_pts.clear();
            retopo.cut_nrm.clear();
            retopo.last_sx = mx - 1e3f;   // the press point always lands
        }
        Vec3 p, n;
        float world_per_px;
        if (retopo.cutting && std::hypot(mx - retopo.last_sx, my - retopo.last_sy) >= 3.0f
            && retopo_surface_at(mx, my, p, n, world_per_px)) {
            retopo.cut_pts.push_back(p);
            retopo.cut_nrm.push_back(n);
            retopo.last_sx = mx;
            retopo.last_sy = my;
        }
    } else if (retopo_press && !retopo.drawing
               && input.retopo_tool == InputState::RetopoTool::WAND) {
        // A wand click opens a hole's menu (on release, below); a drag draws a
        // lasso whose inside gets cleaned up when it's let go.
        const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
        const size_t k = retopo.lasso.size();
        if (k == 0) {
            retopo.lasso = {retopo_press_x, retopo_press_y};
        } else if (std::hypot(mx - retopo.lasso[k - 2], my - retopo.lasso[k - 1]) >= 3.0f) {
            retopo.lasso.push_back(mx);
            retopo.lasso.push_back(my);
        }
    } else if (retopo_press && !screen_buffers_dirty && !retopo_path_press) {
        const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
        if (!retopo.drawing) {
            retopo.drawing = true;
            retopo.live = RetopoStroke{};
            retopo.last_sx = mx - 1e3f;   // the press point always lands
            // Starting on a picked loop: this stroke extrudes it. Else, starting on
            // an edited island: a bridge out of it.
            const float px = retopo_press_x, py = retopo_press_y;
            retopo.extruding = retopo.built && retopo_on_loop(retopo, camera, win_w, win_h, px, py);
            const uint32_t id = retopo.built && !retopo.extruding
                              ? retopo_pick(retopo, camera, win_w, win_h, px, py, input.mirror_x) : 0;
            const RetopoStroke* on = id ? retopo.find(id) : nullptr;
            retopo.bridge_from = on && on->frozen ? id : 0;
            // On a hole's rim vertex: maybe a spine across it (judged at pen-up).
            retopo.spine_from = retopo.extruding ? ~0u
                              : retopo_hole_vertex(retopo, camera, win_w, win_h, px, py);
            // On a lit vertex or edge (orange dot / bold blue): the line starts
            // exactly there. An edited island's lit outline corner is a bridge
            // even where the press itself falls just off its faces.
            retopo_snap_first = false;
            retopo.line_from_v = retopo.line_from_ea = retopo.line_from_eb = ~0u;
            if (retopo.built && !retopo.extruding) {
                retopo_draw_snap(retopo, camera, win_w, win_h, px, py, false);
                Vec3 sp, sn;
                if (retopo_draw_snap_point(retopo, sp, sn)) {
                    const uint32_t sv = retopo.draw_v != ~0u ? retopo.draw_v : retopo.draw_ea;
                    const RetopoStroke* so = retopo.find(retopo.result.v_owner[sv]);
                    if (!retopo.bridge_from && retopo.spine_from == ~0u && so && so->frozen)
                        retopo.bridge_from = so->id;
                    retopo_snap_first = retopo.bridge_from || retopo.spine_from != ~0u;
                    if (retopo_snap_first) {
                        retopo.line_from_v = retopo.draw_v;
                        if (retopo.draw_v == ~0u) {
                            retopo.line_from_ea = retopo.draw_ea;
                            retopo.line_from_eb = retopo.draw_eb;
                        }
                    }
                    retopo_snap_p = sp;
                    retopo_snap_n = sn;
                }
            }
        }
        Vec3 p, n;
        float world_per_px;
        if (std::hypot(mx - retopo.last_sx, my - retopo.last_sy) >= 3.0f
            && retopo_surface_at(mx, my, p, n, world_per_px)) {
            // Band width = the brush ring, in world units where the pen is.
            // Pen pressure thins it, so a stroke can taper.
            // Same pen test as the brush: plugged in isn't enough, the pen has
            // to be what's driving (a resting pen reads ~0 forever).
            const bool pen = tablet.available()
                          && input.active_profile == InputProfile::TABLET;
            const float pr = pen ? std::max(0.2f, tablet.pressure()) : 1.0f;
            retopo.live.pts.push_back(p);
            retopo.live.nrm.push_back(n);
            retopo.live.half_w.push_back(input.brush_size * world_per_px * pr);
            retopo.last_sx = mx;
            retopo.last_sy = my;
            if (retopo_snap_first && retopo.live.pts.size() == 1) {
                retopo.live.pts[0] = retopo_snap_p;
                retopo.live.nrm[0] = retopo_snap_n;
            }
        }
    } else if (retopo.drawing && retopo.spine_from != ~0u
               && retopo_spine_gesture(retopo, camera, win_w, win_h,
                                       (float)input.mouse_x, (float)input.mouse_y)) {
        // Pen up on a spine: from a rim vertex through the hole to another. Its fill
        // goes up at once with the wand's menu beside it for the other ways.
        scene.materialize_active_cpu();   // the menu's grid snaps onto mesh.pos
        const char* why = retopo_wand_open_spine(retopo, mesh, camera, win_w, win_h,
                                                 (float)input.mouse_x, (float)input.mouse_y,
                                                 input.mirror_x);
        line_menu(why);
    } else if (retopo.drawing && retopo.line_from_v != ~0u && retopo.draw_v != ~0u
               && retopo_notch_gesture(retopo, camera, win_w, win_h, input.mirror_x)) {
        // Pen up on a line from an outline vertex to another of the same outline,
        // across open space: the notch between them closes, its menu beside it.
        scene.materialize_active_cpu();   // the mouth snaps onto mesh.pos
        const char* why = retopo_wand_open_notch(retopo, mesh, camera, win_w, win_h, input.mirror_x);
        line_menu(why);
    } else if (retopo.drawing && retopo_edge_vertex_gesture(retopo, camera, win_w, win_h)) {
        // Pen up on a line from an outline edge to an outline vertex (or back):
        // one quad joins them, its menu beside it (the wedge closed or not).
        const char* why = retopo_wand_open_edge_vertex(retopo, camera, win_w, win_h, input.mirror_x);
        line_menu(why);
    } else if (retopo.drawing && retopo.extruding) {
        // Pen up on an extrude. A few pixels is a click (or half a double-click):
        // the pick stays, no toast.
        if (retopo.live.pts.size() >= 3) {
            scene.materialize_active_cpu();   // new loops snap onto mesh.pos
            const char* why = retopo_extrude(retopo, mesh, camera, win_w, win_h, input.mirror_x);
            std::snprintf(input.notification, sizeof(input.notification), "%s",
                          why ? why : "Extruded - draw from the new loop to carry on");
            input.notification_timer = why ? 2.5f : 1.5f;
        }
        retopo.end_line();
    } else if (retopo.drawing && retopo.bridge_from) {
        // Pen up on a bridge: built now, it joins geometry, not sketches. A few
        // pixels is a click (it selects the island), not a bridge: no toast.
        // A line that ran along the island's edge instead is a strip beside it:
        // slid off the island and kept like any stroke, Build zips it on.
        if (retopo.live.pts.size() >= 3 && retopo_alongside(retopo)) {
            if (retopo.finish_live(input.retopo_rings, input.retopo_density)) {
                input.retopo_rings = 1;   // more rings are a one-off (user)
                input.retopo_build_requested = true;
                input.retopo_build_quiet = true;
                input.retopo_menu_auto = true;
            }
        } else if (retopo.live.pts.size() >= 3) {
            const uint32_t to = retopo_pick(retopo, camera, win_w, win_h, (float)input.mouse_x,
                                            (float)input.mouse_y, input.mirror_x);
            scene.materialize_active_cpu();   // its spokes snap onto mesh.pos
            const char* why = retopo_bridge(retopo, mesh, camera, win_w, win_h, to, input.mirror_x);
            std::snprintf(input.notification, sizeof(input.notification), "%s",
                          why ? why : to ? "Bridged" : "Strip grown out of the edge");
            input.notification_timer = why ? 2.5f : 1.5f;
        }
        retopo.end_line();
    } else if (retopo.drawing) {
        // Pen up (or the mode changed under it).
        const bool kept = retopo.finish_live(input.retopo_rings, input.retopo_density);
        if (kept) {
            input.retopo_rings = 1;   // more rings are a one-off (user)
            // Built on release, no Space needed. Quiet: finish_live took the undo
            // step (one Ctrl+Z takes the stroke and its quads), and no toast.
            input.retopo_build_requested = true;
            input.retopo_build_quiet = true;
            input.retopo_menu_auto = true;   // its sliders, right where the pen let go
        }
    } else if (retopo.cutting) {
        // Pen up: the knife goes through the built quads along the line.
        retopo.cut_to_v = retopo.snap_v;
        if (retopo.cut_pts.size() >= 2) {
            scene.materialize_active_cpu();   // new vertices snap onto mesh.pos
            const int split = retopo_cut(retopo, mesh, camera, win_w, win_h, input.mirror_x);
            if (split)
                std::snprintf(input.notification, sizeof(input.notification),
                              "Cut through %d faces - Ctrl+Z takes it back", split);
            else
                std::snprintf(input.notification, sizeof(input.notification),
                              "The cut crossed no edges");
            input.notification_timer = 2.0f;
        }
        retopo.cutting = false;
        retopo.cut_from_v = retopo.cut_to_v = ~0u;
        retopo.cut_pts.clear();
        retopo.cut_nrm.clear();
    }
    if (!retopo_press) {
        if (retopo_smoothing || retopo_grabbing || retopo_erasing || retopo_masking) retopo.end_edit();
        retopo_smoothing = retopo_grabbing = retopo_erasing = retopo_erasing_faces = false;
        retopo_masking = retopo_mask_smooth = false;
    }
    // A slider drag is one undo step; letting go ends it.
    if (!input.mouse1_down) retopo.last_tag = 0;
    // The wand's menu is a preview: anything else that edits puts the hole back
    // as it was first, so it never acts on (or saves) a fill nobody kept.
    // A spine's menu keeps what it shows (the line was drawn on purpose), so Ctrl+Z
    // then takes the spine itself back.
    if (retopo.wand_open
        && (!retopo_on || input.is_smooth_active()
            || (!retopo.wand_keep_shown && input.retopo_tool != InputState::RetopoTool::WAND)
            || input.undo_requested || input.redo_requested || input.retopo_build_requested
            || input.retopo_fill_requested || input.retopo_dissolve_requested
            || input.retopo_symmetrize_request || input.retopo_export_requested))
        retopo_wand_close(retopo, retopo.wand_keep_shown ? retopo.wand_shown : -1);

    // A click (pressed and let go within a few pixels) picks an island: plain
    // selects only it, Ctrl adds or takes out, on nothing clears. Shift is Smooth's,
    // except Shift+Ctrl in Draw: the shortest edge path from the last edge picked.
    // A press on the bar never gets here: ImGui ate mouse1_just_pressed.
    static bool click_armed = false, click_path = false;
    static float click_x = 0.0f, click_y = 0.0f;
    // Ctrl+click with the eraser deletes a face; it doesn't pick.
    const bool path_mods = input.shift_held && input.ctrl_held
                        && input.retopo_tool == InputState::RetopoTool::DRAW;
    if (retopo_on && input.mouse1_just_pressed && (!input.shift_held || path_mods) && !input.alt_held
        && !(input.ctrl_held && input.retopo_tool == InputState::RetopoTool::ERASE)) {
        click_armed = true;
        click_path = path_mods;
        click_x = (float)input.mouse_x;
        click_y = (float)input.mouse_y;
    }
    if (click_armed && std::hypot((float)input.mouse_x - click_x, (float)input.mouse_y - click_y) > 4.0f)
        click_armed = false;
    // A second click soon after, on the same spot, with the pencil out: the edge
    // loop there instead (the first click already picked its island).
    static double last_click_t = -1.0;
    static float last_click_x = 0.0f, last_click_y = 0.0f;
    if (click_armed && input.mouse1_just_released) {
        click_armed = false;
        if (retopo_on && click_path) {
            last_click_t = -1.0;   // never half of a double-click
            if (retopo_select_path(retopo, camera, win_w, win_h, click_x, click_y, input.mirror_x) < 0) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "No edges join those two: separate islands");
                input.notification_timer = 2.0f;
            }
        } else if (retopo_on) {
            const double now = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const bool dbl = input.retopo_tool == InputState::RetopoTool::DRAW
                          && now - last_click_t < 0.4
                          && std::hypot(click_x - last_click_x, click_y - last_click_y) <= 6.0f;
            last_click_t = dbl ? -1.0 : now;   // a third click starts over
            last_click_x = click_x;
            last_click_y = click_y;
            if (input.retopo_tool == InputState::RetopoTool::WAND) {
                scene.materialize_active_cpu();   // the grid option snaps onto mesh.pos
                if (const char* why = retopo_wand_open(retopo, mesh, camera, win_w, win_h,
                                                       click_x, click_y, input.mirror_x)) {
                    std::snprintf(input.notification, sizeof(input.notification), "%s", why);
                    input.notification_timer = 2.5f;
                } else {
                    input.retopo_wand_request = true;
                }
            } else if (dbl) {
                retopo_select_loop(retopo, camera, win_w, win_h, click_x, click_y,
                                   input.ctrl_held, input.mirror_x);
            } else {
                if (!input.ctrl_held) retopo.loop_edges.clear();
                const uint32_t id = retopo_pick(retopo, camera, win_w, win_h, click_x, click_y,
                                                input.mirror_x);
                if (const char* msg = retopo.click_select(id, input.ctrl_held)) {
                    std::snprintf(input.notification, sizeof(input.notification), "%s", msg);
                    input.notification_timer = 2.0f;
                }
            }
        }
    }
    // Right-click on an editable island: the rings / density / width menu, for the
    // selection if the island is in it, else for the island alone.
    if (retopo_on && input.mouse2_just_released && !imgui_wants_mouse
        && !input.zoom_modifier_held()
        && std::hypot((float)input.mouse_x - input.mouse2_press_x,
                      (float)input.mouse_y - input.mouse2_press_y) <= 4.0f) {
        const uint32_t id = retopo_pick(retopo, camera, win_w, win_h, input.mouse2_press_x,
                                        input.mouse2_press_y, input.mirror_x);
        RetopoStroke* s = id ? retopo.find(id) : nullptr;
        if (s && s->frozen && s->selected && retopo.merge_candidates() >= 2) {
            input.retopo_menu_request = true;   // the menu offers Merge
        } else if (s && s->frozen) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "%s is edited geometry now: rings, density and width don't reach it",
                          s->name.c_str());
            input.notification_timer = 2.5f;
        } else if (s) {
            if (!s->selected) retopo.click_select(id, false);
            input.retopo_menu_request = true;
        }
    }
    // Ctrl+Z / Ctrl+Shift+Z walk the stroke list here, never the sculpt's
    // history: retopo doesn't touch the sculpt.
    if (retopo_on && (input.undo_requested || input.redo_requested)) {
        const bool ok = input.undo_requested ? retopo.undo() : retopo.redo();
        if (!ok) {
            std::snprintf(input.notification, sizeof(input.notification),
                          input.undo_requested ? "Nothing to undo" : "Nothing to redo");
            input.notification_timer = 1.2f;
        }
        input.undo_requested = input.redo_requested = false;
    }
    if (input.retopo_build_requested) {
        input.retopo_build_requested = false;
        scene.materialize_active_cpu();   // Build snaps onto mesh.pos
        const bool quiet = input.retopo_build_quiet;
        if (!quiet) retopo.checkpoint();   // a slider took its step at the drag's start
        const double ms = retopo_build(retopo, mesh, input.mirror_x);
        input.retopo_build_quiet = false;
        if (quiet) {
            // Slider drags rebuild every step: the bar's readout says enough.
        } else if (retopo.built)
            std::snprintf(input.notification, sizeof(input.notification),
                          "Built %u faces (%.0f ms)", retopo.result.face_count(), ms);
        else
            std::snprintf(input.notification, sizeof(input.notification),
                          "Nothing to build: sketch a stroke first");
        if (!quiet) input.notification_timer = 2.5f;
    }
    if (input.retopo_fill_requested) {
        input.retopo_fill_requested = false;
        if (retopo.built) scene.materialize_active_cpu();   // the fill snaps onto mesh.pos
        int rims = 0, odd = 0;
        const int made = retopo_fill(retopo, mesh, input.mirror_x, rims, odd);
        if (made && odd)
            std::snprintf(input.notification, sizeof(input.notification),
                          "Filled %d hole%s with %d quads; %d with an odd count of sides left open",
                          rims, rims == 1 ? "" : "s", made, odd);
        else if (made)
            std::snprintf(input.notification, sizeof(input.notification),
                          "Filled %d hole%s with %d quads - Ctrl+Z takes it back",
                          rims, rims == 1 ? "" : "s", made);
        else if (odd)
            std::snprintf(input.notification, sizeof(input.notification),
                          "Odd count of sides: all-quad fill needs an even one (cut a loop into the rim)");
        else
            std::snprintf(input.notification, sizeof(input.notification),
                          "No holes to fill (double-click an outline to fill it anyway)");
        input.notification_timer = 2.5f;
    }
    if (input.retopo_symmetrize_request) {
        const int keep = input.retopo_symmetrize_request;
        input.retopo_symmetrize_request = 0;
        if (retopo.built) scene.materialize_active_cpu();   // seam cuts snap onto mesh.pos
        int faces = 0;
        const char* why = retopo_symmetrize(retopo, mesh, keep, faces);
        if (why) std::snprintf(input.notification, sizeof(input.notification), "%s", why);
        else     std::snprintf(input.notification, sizeof(input.notification),
                               "Symmetrized from %s: %d faces - Ctrl+Z takes it back",
                               keep > 0 ? "+X" : "-X", faces);
        input.notification_timer = 2.5f;
    }
    if (input.retopo_dissolve_requested) {
        input.retopo_dissolve_requested = false;
        int n = 0;
        const char* why = retopo_dissolve(retopo, input.mirror_x, n);
        if (why) std::snprintf(input.notification, sizeof(input.notification), "%s", why);
        else     std::snprintf(input.notification, sizeof(input.notification),
                               "Dissolved %d edge%s - Ctrl+Z takes it back", n, n == 1 ? "" : "s");
        input.notification_timer = 2.5f;
    }

    // What the pointer is on, for the overlay and the next press: worked out last,
    // after this frame's edits. A delete or an undo renumbers faces, and an index
    // taken before it pointed at whatever face slid into that slot (the red outline
    // jumped across the model for a frame after every delete).
    // Cut's vertex snap, every frame the knife is out: the overlay lights it, the
    // press and the pen-up read it.
    const bool knife = retopo_on && retopo.built && !input.is_smooth_active()
                    && input.retopo_tool == InputState::RetopoTool::CUT;
    retopo.snap_v = knife ? retopo_snap_vertex(retopo, camera, win_w, win_h,
                                               (float)input.mouse_x, (float)input.mouse_y)
                          : ~0u;
    // Draw's snap: before a press, what a line would start on; during a bridge or a
    // spine, what it would end on. Nothing while a band is drawing (it doesn't snap).
    const bool line_live = retopo.drawing && !retopo.extruding
                        && (retopo.bridge_from || retopo.spine_from != ~0u);
    if (retopo_on && retopo.built && !input.is_smooth_active() && !retopo.wand_open
        && input.retopo_tool == InputState::RetopoTool::DRAW && (!retopo.drawing || line_live))
        retopo_draw_snap(retopo, camera, win_w, win_h, (float)input.mouse_x, (float)input.mouse_y,
                         line_live);
    else
        retopo.draw_v = retopo.draw_ea = retopo.draw_eb = ~0u;
    // Ctrl with the eraser takes whole faces: show which one before it goes.
    const bool face_eraser = retopo_on && retopo.built && !input.is_smooth_active()
                          && input.retopo_tool == InputState::RetopoTool::ERASE
                          && (input.ctrl_held || retopo_erasing_faces);
    retopo.face_hover = face_eraser ? retopo_face_at(retopo, camera, win_w, win_h, (float)input.mouse_x,
                                                     (float)input.mouse_y, input.brush_size)
                                    : ~0u;
}

// The low-poly is a by-product to take elsewhere, not a document: no dialog, next to
// the project (or home), named after it. Web writes it to MEMFS for main.cpp to hand
// to the browser.
std::string retopo_export(const RetopoState& retopo, InputState& input, const std::string& project_path) {
#if defined(__EMSCRIPTEN__)
    const std::string name = "retopo.obj", path = "/" + name;
#else
    std::string stem = project_path;
    if (stem.empty()) {
        const char* home = std::getenv("HOME");
        stem = std::string(home ? home : ".") + "/chisel";
    } else {
        const auto dot = stem.find_last_of('.');
        if (dot != std::string::npos && stem.find_last_of('/') < dot) stem.resize(dot);
    }
    const std::string path = stem + "-retopo.obj", name = path;
#endif
    const bool ok = retopo_export_obj(retopo.result, path.c_str());
    if (ok)
        std::snprintf(input.notification, sizeof(input.notification), "Exported: %.400s", name.c_str());
    else
        std::snprintf(input.notification, sizeof(input.notification),
                      retopo.built ? "Export failed: %.400s" : "Build first%.0s", name.c_str());
    input.notification_timer = 3.0f;
    return ok ? path : std::string();
}

void draw_retopo_cursor(Renderer& renderer, const Camera& camera, const InputState& input,
                        float cursor_x, float cursor_y, int win_w, int win_h) {
    // Matrix green (#00FF41): a ring no sculpt brush wears, so the hand
    // knows it's in retopo before the eye finds the dock.
    static const float kRetopoGreen[3] = {0.0f, 1.0f, 0.255f};
    // Cut draws a bare line and the wand only clicks: a small ring marks the tip.
    const bool cut = (input.retopo_tool == InputState::RetopoTool::CUT
                      || input.retopo_tool == InputState::RetopoTool::WAND)
                  && !input.is_smooth_active();
    // The eraser's ring is red: it takes things away.
    static const float kEraseRed[3] = {1.0f, 0.25f, 0.2f};
    const bool erase = input.retopo_tool == InputState::RetopoTool::ERASE
                    && !input.is_smooth_active();
    renderer.draw_cursor(camera, cursor_x, cursor_y, cut ? 6.0f : input.brush_size,
                         input.cursor_nx, input.cursor_ny, input.cursor_nz,
                         1.0f, 0.5f, win_w, win_h, input.on_model,
                         erase ? kEraseRed : kRetopoGreen);
}

void draw_retopo_ui(InputState& input, RetopoState& retopo, Renderer& renderer, const Camera& camera,
                    int win_w, int win_h, bool screen_buffers_dirty) {
    static uint64_t audited_rev = 0;   // retopo.rev last checked for faces laid over faces
    const bool cam_moving = input.drag_mode == InputState::DragMode::ORBIT
                         || input.drag_mode == InputState::DragMode::PAN
                         || input.drag_mode == InputState::DragMode::ZOOM
                         || input.mouse2_down || input.mouse3_down;
    // Dock first: it sets which stroke the cursor is over, and the overlay lights it
    // the same frame (background list, so it still draws under the dock).
    draw_retopo_dock(input, retopo, win_w, win_h);
    if (retopo.rev != audited_rev) { audited_rev = retopo.rev; retopo_audit(retopo); }
    draw_retopo_overlay(retopo, renderer, camera, win_w, win_h,
                        !cam_moving && !screen_buffers_dirty, input.mirror_x);
}
