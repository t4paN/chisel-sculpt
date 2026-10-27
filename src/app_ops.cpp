// The frame's one-shot operations and shared helpers, moved out of main.cpp as is.
#include "debug_console.h"
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#include "app.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>

#include "mesh.h"
#include "camera.h"
#include "renderer.h"
#include "input.h"
#include "settings.h"
#include "brush.h"
#include "undo.h"
#include "multires_stack.h"
#include "remesh.h"
#include "compute.h"
#include "sdf.h"
#include "insert_controller.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include "retopo.h"
#include "brush_alpha.h"
#include "scene_snapshot.h"
#include "object_xform.h"

// Reports the projection truth check when it ran (CHISEL_PROJECT_CHECK=1, or a
// CHISEL_DEBUG_MULTIRES build). project_down_to_level promises that cascading back to
// L_max reproduces the pre-projection surface EXACTLY, so any real distance here is a
// broken contract, not a tolerance to tune. Silent when the check did not run, so the
// normal log is unchanged.
//
// The 1e-4 line is a heuristic, not a spec: a few cascade steps of float arithmetic on
// O(1) coordinates land around 1e-6, while a mean edge at L9 is ~5e-3 — so anything at
// 1e-4 is already far outside rounding and heading for visible.
void report_projection_check(const ProjectionStats& ps) {
    if (!ps.checked) return;
    const bool bad = ps.max_reconstruction_error > 1e-4;
    std::printf("[project] CHECK reconstruction max error %.6g (worst vert %u)%s\n",
                ps.max_reconstruction_error, ps.worst_vertex,
                bad ? "  <-- LOSSY, cascade does not reproduce the surface" : "  ok");
    std::fflush(stdout);
}

// Phase 1 GPU residency: full re-upload of the active entity's mirrored
// level after any wholesale CPU mutation (lock, level switch, projection,
// cascade). Cheap no-op until the stack is locked / compute supported.
void app_refresh_active_gpu_residency(App& a) {
    auto& scene = a.scene;
    auto& compute = a.compute;
    MeshEntity& ent = scene.active_entity();
    ent.multires_gpu.supported = compute.supported;
    ent.multires_gpu.dev       = &compute.gpu_dev;
    // Residency dedupe: right after a GPU cascade replay, the target
    // level's disp/frames still sit in the replay's layer scratch —
    // upload_level copies them device-local instead of re-sending from
    // CPU. The handoff is consumed here (one switch only) so no later
    // refresh can copy stale scratch.
    const gpu::Buffer* dsrc = nullptr;
    const gpu::Buffer* fsrc = nullptr;
    if (compute.cascade_out_valid &&
        compute.cascade_out_passes ==
            ent.multires.current_level - ent.multires.base_level) {
        dsrc = &compute.cascade_disp_ssbo;
        fsrc = &compute.cascade_frames_ssbo;
        std::printf("[residency] disp/frames via gpu copy\n");
    }
    ent.multires_gpu.upload_level(ent.multires, ent.multires.current_level, dsrc, fsrc);
    compute.cascade_out_valid = false;
}

// Debug helper: print undo stack top after key operations
void app_print_undo_top(App& a, const char* tag) {
    auto& scene = a.scene;
    UndoStack& undo_stack = scene.active_undo();
    const UndoEntry* e = undo_stack.peek_undo();
    if (e) {
        if (e->kind == UndoEntry::Kind::PROJECTION)
            std::printf("[undo-trace][%s] depth=%zu top=PROJECTION target_level=%d\n",
                        tag, undo_stack.undo_depth(), e->target_level);
        else if (e->kind == UndoEntry::Kind::LEVEL)
            std::printf("[undo-trace][%s] depth=%zu top=LEVEL from=%d to=%d proj=%d\n",
                        tag, undo_stack.undo_depth(), e->from_level, e->to_level,
                        (int)!e->before.empty());
        else if (e->kind == UndoEntry::Kind::MASK)
            std::printf("[undo-trace][%s] depth=%zu top=MASK verts=%zu\n",
                        tag, undo_stack.undo_depth(), e->verts.size());
        else if (e->kind == UndoEntry::Kind::PAINT)
            std::printf("[undo-trace][%s] depth=%zu top=PAINT verts=%zu\n",
                        tag, undo_stack.undo_depth(), e->verts.size());
        else
            std::printf("[undo-trace][%s] depth=%zu top=STROKE level=%d targets_base=%d disp_index=%d\n",
                        tag, undo_stack.undo_depth(), e->level, (int)e->targets_base, e->disp_index);
    } else {
        std::printf("[undo-trace][%s] depth=%zu (empty)\n", tag, undo_stack.undo_depth());
    }
}

// Prepare + capture the rescue snapshot, shared by '/' and 'J'. Frees the
// undo history, bakes the scene down to the triangle cap, then copies the
// stacks.
void app_take_rescue(App& a, const char* op) {
    auto& scene = a.scene;
    auto& compute = a.compute;
    auto& rescue = a.rescue;
    auto& rescue_shown = a.rescue_shown;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& mesh_center = a.mesh_center;
    auto& mesh_radius = a.mesh_radius;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(a); };
    int baked = snapshot_prepare(scene, compute);
    if (baked < 0) { rescue.clear(); return; }   // over cap, unprotected
    if (baked > 0) {
        // snapshot_prepare rebuilt the scene through load_entities to
        // re-cascade the baked stacks, so every cached pointer is stale.
        mesh     = &scene.active_mesh();
        multires = &scene.active_multires();
        scene.refresh_mirror_map();
        scene.sync();
        refresh_active_gpu_residency();
        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
        screen_buffers_dirty = true;
    }
    rescue.levels_baked = baked;
    snapshot_capture(rescue, scene, op);
    rescue_shown = -1;
}

// Arm the renderer's one-shot geometry handoff for the sync following a
// GPU cascade replay: the working VBOs/EBO then fill by GPU→GPU copy from
// the replay scratch (positions/normals) and the VRAM level tables
// (indices). No-op when the replay didn't run (CPU fallback).
void app_arm_geometry_handoff(App& a, const Mesh& m) {
    auto& compute = a.compute;
    auto& renderer = a.renderer;
    if (!compute.cascade_out_valid || !compute.cascade_out_pos) return;
    if (compute.cascade_out_passes <= 0 ||
        compute.cascade_out_passes >= (int)compute.cascade_levels.size()) return;
    renderer.set_geometry_source(compute.cascade_out_pos, &compute.cascade_norm_ssbo,
                                 &compute.cascade_levels[compute.cascade_out_passes].indices,
                                 m.vertex_count(), (uint32_t)m.indices.size());
}

// Colour picker and fill bucket: both read the screen FBO under a click on the active
// entity. (`app`, not `a`: the bucket keeps its own `a`, the gradient's start point.)
void app_ops_paint_clicks(App& app, int win_w, int win_h) {
    auto& input = app.input;
    auto& scene = app.scene;
    auto& camera = app.camera;
    auto& renderer = app.renderer;
    // Colour picker (C in paint mode): sample the stored albedo under the
    // click — tri + barycentrics from the screen FBO, colours from the CPU
    // mesh (synced at pen-up), so shading/matcap never leaks into the pick.
    if (input.color_pick_click) {
        input.color_pick_click = false;
        int cx = (int)input.mouse_x;
        int cy = (int)input.mouse_y;
        if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
            uint32_t tid;
            renderer.read_triid_region(cx, cy, 1, 1, &tid);
            const Mesh& pm = scene.active_mesh();   // screen FBO holds the active entity
            if (tid != 0xFFFFFFFF && tid < pm.tri_count()) {
                float bary[2];
                renderer.read_bary_region(cx, cy, 1, 1, bary);
                float bu = bary[0], bv = bary[1], bw = 1.0f - bu - bv;
                uint32_t i0 = pm.indices[tid*3+0];
                uint32_t i1 = pm.indices[tid*3+1];
                uint32_t i2 = pm.indices[tid*3+2];
                auto vcol = [&](uint32_t v) {   // what you see: the layers' composite
                    return paint_composite_vertex(pm.paint_layers, pm.color, v);
                };
                uint32_t c0 = vcol(i0), c1 = vcol(i1), c2 = vcol(i2);
                for (int ch = 0; ch < 3; ch++) {
                    float f0 = (float)((c0 >> (ch*8)) & 0xFF);
                    float f1 = (float)((c1 >> (ch*8)) & 0xFF);
                    float f2 = (float)((c2 >> (ch*8)) & 0xFF);
                    input.paint_color[ch] = (bu*f0 + bv*f1 + bw*f2) / 255.0f;
                }
                input.color_pick_active = false;
                snprintf(input.notification, sizeof(input.notification),
                         "Picked #%02X%02X%02X",
                         (int)(input.paint_color[0]*255.0f + 0.5f),
                         (int)(input.paint_color[1]*255.0f + 0.5f),
                         (int)(input.paint_color[2]*255.0f + 0.5f));
                input.notification_timer = 1.5f;
            }
        }
    }

    // Fill bucket (paint mode): flood the unmasked island under the click on the
    // active layer, flat or through a pattern (main colour to secondary); a gradient
    // runs from the press (A, main) to the release (B, secondary). Points come from the screen FBO exactly as the picker's
    // do (tri + barycentrics); mask and colour are the CPU copies, both synced at
    // pen-up. One PAINT undo entry, so Ctrl+Z takes the fill back.
    if (input.bucket_click) {
        input.bucket_click = false;
        Mesh& pm = scene.active_mesh();
        const int pattern = input.fill_pattern;
        const bool gradient = pattern == kFillGradient;
        const bool solid = pattern == kFillSolid || pattern < 0 || pattern >= fill_patterns().count();
        // Hardness below 1 feathers the fill off the mask, measured along the surface.
        const bool feathered = input.brush_hardness < 0.99f;
        // Patterns, gradients and the feather read positions/normals: pull the GPU's
        // back first.
        if (!solid || feathered) scene.materialize_active_cpu();
        // The surface under a screen pixel: nearest corner (the seed) and the point.
        float click_n[3] = {0.0f, 0.0f, 1.0f};   // the surface normal at the last surface_at hit
        auto surface_at = [&](float fx, float fy, uint32_t* seed, float out[3]) -> bool {
            const int cx = (int)fx, cy = (int)fy;
            if (cx < 0 || cx >= win_w || cy < 0 || cy >= win_h) return false;
            uint32_t tid = 0xFFFFFFFF;
            renderer.read_triid_region(cx, cy, 1, 1, &tid);
            if (tid == 0xFFFFFFFF || tid >= pm.tri_count()) return false;
            float bary[2];
            renderer.read_bary_region(cx, cy, 1, 1, bary);
            const float bw[3] = {bary[0], bary[1], 1.0f - bary[0] - bary[1]};
            const uint32_t ix[3] = {pm.indices[tid*3+0], pm.indices[tid*3+1], pm.indices[tid*3+2]};
            if (seed) *seed = (bw[0] >= bw[1] && bw[0] >= bw[2]) ? ix[0] : (bw[1] >= bw[2] ? ix[1] : ix[2]);
            if (out) {
                out[0] = out[1] = out[2] = 0.0f;
                for (int c = 0; c < 3; c++) {
                    out[0] += bw[c] * pm.pos_x[ix[c]];
                    out[1] += bw[c] * pm.pos_y[ix[c]];
                    out[2] += bw[c] * pm.pos_z[ix[c]];
                }
            }
            if (pm.norm_x.size() >= pm.vertex_count()) {
                Vec3 n{0, 0, 0};
                for (int c = 0; c < 3; c++)
                    n = n + Vec3{pm.norm_x[ix[c]], pm.norm_y[ix[c]], pm.norm_z[ix[c]]} * bw[c];
                if (n.length() > 1e-6f) {
                    n = n.normalized();
                    click_n[0] = n.x; click_n[1] = n.y; click_n[2] = n.z;
                }
            }
            return true;
        };
        uint32_t seed = 0;
        float a[3], b[3];
        const float drag_px = std::hypot(input.bucket_bx - input.bucket_ax, input.bucket_by - input.bucket_ay);
        if (!surface_at(input.bucket_ax, input.bucket_ay, &seed, a)) {
            // Missed the model (on_model and the FBO can disagree by a pixel).
        } else if (gradient && drag_px < 4.0f) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Drag across the model to set the gradient");
            input.notification_timer = 1.5f;
        } else {
            if (gradient && !surface_at(input.bucket_bx, input.bucket_by, nullptr, b)) {
                // Released off the model: B on the view plane through A.
                Vec3 cam_pos = camera.get_position();
                Vec3 fwd = (camera.target - cam_pos).normalized();
                Vec3 right = fwd.cross(Vec3{0, 1, 0}).normalized();
                Vec3 up = right.cross(fwd).normalized();
                const float d = camera.view_depth(Vec3{a[0], a[1], a[2]});
                const float half_h = camera.half_height_at(d);
                const float half_w = half_h * ((float)win_w / (float)win_h);
                const float ndc_x = 2.0f * input.bucket_bx / (float)win_w - 1.0f;
                const float ndc_y = 1.0f - 2.0f * input.bucket_by / (float)win_h;
                Vec3 p = cam_pos + fwd * d + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                b[0] = p.x; b[1] = p.y; b[2] = p.z;
            }
            // Patterns run from the main colour (black) to the secondary (white).
            auto pack = [](const float* rgb) {
                uint32_t c = 0xFFu << 24;
                for (int ch = 0; ch < 3; ch++)
                    c |= (uint32_t)(std::max(0.0f, std::min(1.0f, rgb[ch])) * 255.0f + 0.5f) << (ch * 8);
                return c;
            };
            const uint32_t main_c = pack(input.paint_color), second_c = pack(input.paint_color_alt);
            const double t0 = glfwGetTime();
            static std::vector<uint32_t> island;
            static std::vector<float> value, edge;
            // World size of a screen pixel at the clicked spot: the brush ring there.
            const float wpp = 2.0f * camera.half_height_at(camera.view_depth(Vec3{a[0], a[1], a[2]}))
                            / (float)win_h;
            // Hardness = the edge against the walls; at 0 the fade is one brush radius
            // wide (the bucket's own slot: live_brush_slot() is BUCKET while armed).
            const float feather = feathered ? (1.0f - input.brush_hardness) * input.brush_size * wpp : 0.0f;
            // Repeat fills of one island with the same paint creep the edge further
            // across the walls, half a band per click. A repeat = the undo stack's top
            // is still the fill recorded at that depth, so Ctrl+Z steps the creep back
            // with it, and any other edit in between starts it over.
            // A run of repeats reuses its first fill's walls and island (the chain):
            // with paint as walls, that fill's own soft edge would otherwise read as
            // new walls, or new floor, on the next click and shift them.
            struct BucketRepeat {
                uint32_t entity, chain, main_c, second_c;
                int pattern; bool erase;
                size_t depth; size_t top_n; uint32_t top_v, top_c;
                int n;
            };
            static std::vector<BucketRepeat> repeats;
            static std::vector<float> chain_block;        // the chain's walls, per vertex
            static std::vector<uint32_t> chain_island;    // its flood, sorted
            static uint32_t chain_id = 0;
            BucketRepeat rep{scene.active_entity().id, 0, main_c, second_c, pattern,
                             input.bucket_erase, 0, 0, 0, 0, 0};
            bool repeat = false;
            {
                UndoStack& us = scene.active_undo();
                while (!repeats.empty() && repeats.back().depth > us.undo_depth()) repeats.pop_back();
                const UndoEntry* top = us.peek_undo();
                if (!repeats.empty() && top && top->kind == UndoEntry::Kind::PAINT && !top->verts.empty()) {
                    const BucketRepeat& r = repeats.back();
                    repeat = r.depth == us.undo_depth() && r.top_n == top->verts.size()
                          && r.top_v == top->verts[0] && r.top_c == top->new_color[0]
                          && r.entity == rep.entity && r.chain == chain_id
                          && r.main_c == rep.main_c && r.second_c == rep.second_c
                          && r.pattern == rep.pattern && r.erase == rep.erase
                          && chain_block.size() == pm.vertex_count()
                          && std::binary_search(chain_island.begin(), chain_island.end(), seed);
                    if (repeat) rep.n = r.n + 1;
                }
            }
            if (!repeat) {
                // Walls: the mask, plus (Stop at paint) paint unlike the clicked colour,
                // read off what the viewport shows, every visible layer.
                static std::vector<uint32_t> shown;
                if (input.bucket_paint_walls) {
                    shown.resize(pm.vertex_count());
                    paint_composite(pm.paint_layers, pm.color, pm.vertex_count(), shown.data());
                }
                constexpr float kPaintTolerance = 0.15f;   // largest channel difference that walls
                paint_bucket_walls(pm, input.bucket_paint_walls ? shown.data() : nullptr, seed,
                                   kPaintTolerance, chain_block);
                paint_bucket_island(pm, chain_block.data(), seed, chain_island);
                std::sort(chain_island.begin(), chain_island.end());
                chain_id++;
            }
            rep.chain = chain_id;
            island = chain_island;
            paint_bucket_edge(pm, chain_block.data(), island, input.brush_hardness, feather,
                              0.5f * feather * (float)rep.n, edge);
            const float* cv = nullptr;
            if (gradient) {
                paint_cover_gradient(pm, island, a, b, value);
                cv = value.data();
            } else if (!solid) {
                // The click's frame: its normal, and the camera's up laid onto the
                // surface there, so Rotate 0 reads upright from where you look.
                // (surface_at(A) ran last on this path, so click_n is A's.)
                Vec3 n{click_n[0], click_n[1], click_n[2]};
                Vec3 cam_fwd = (camera.target - camera.get_position()).normalized();
                Vec3 cam_right = cam_fwd.cross(Vec3{0, 1, 0}).normalized();
                Vec3 cam_up = cam_right.cross(cam_fwd).normalized();
                Vec3 up = cam_up - n * n.dot(cam_up);
                if (up.length() < 1e-3f) up = cam_right - n * n.dot(cam_right);
                up = up.normalized();
                Vec3 right = up.cross(n).normalized();
                const float frame[9] = {right.x, right.y, right.z, up.x, up.y, up.z, n.x, n.y, n.z};
                // Size = one tile across the brush circle at the clicked spot, so
                // the ring shows the repeat before you click.
                const AlphaEntry& pe = fill_patterns().get(pattern);
                paint_cover_triplanar(pm, island, pe.data.data(), pe.w, pe.h,
                                      2.0f * input.brush_size * wpp,
                                      input.fill_angle * 3.14159265f / 180.0f, a, frame, value);
                cv = value.data();
            }
            UndoEntry e;
            e.kind = UndoEntry::Kind::PAINT;
            e.paint_layer_id = paint_layers_active_id(pm.paint_layers);
            paint_bucket_apply(pm, island, main_c, second_c, cv, edge.data(), input.bucket_erase,
                               input.brush_strength, e.verts, e.old_color, e.new_color);
            const double ms = (glfwGetTime() - t0) * 1000.0;
            std::printf("[bucket] seed %u, %s, hardness %.2f, repeat %d%s: reached %zu / %u verts, changed %zu, %.1f ms%s\n",
                        seed, fill_patterns().get(solid ? kFillSolid : pattern).name.c_str(),
                        input.brush_hardness, rep.n, input.bucket_paint_walls ? ", stops at paint" : "",
                        island.size(), pm.vertex_count(), e.verts.size(), ms,
                        input.bucket_erase ? " (erase)" : "");
            if (island.empty()) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "That spot is masked");
                input.notification_timer = 1.5f;
            } else if (!e.verts.empty()) {
                rep.top_n = e.verts.size(); rep.top_v = e.verts[0]; rep.top_c = e.new_color[0];
                scene.sync_color_partial_entity(scene.active_entity().id, e.verts);
                scene.active_undo().push(std::move(e));
                rep.depth = scene.active_undo().undo_depth();
                repeats.push_back(rep);
                if (repeats.size() > 64) repeats.erase(repeats.begin());
            }
        }
    }
}

// Level switch, delete-top-level, focus and snap views: camera and multires ops that run
// between the pen-down sampling and the stroke.
void app_ops_levels_and_views(App& a) {
    auto& input = a.input;
    auto& scene = a.scene;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& camera = a.camera;
    auto& compute = a.compute;
    auto& brush_stroke = a.brush_stroke;
    auto& xforms = a.xforms;
    auto& app_state = a.app_state;
    auto& mesh_center = a.mesh_center;
    auto& mesh_radius = a.mesh_radius;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto& max_level = a.max_level;
    auto& swallow_level_switch = a.swallow_level_switch;
    auto print_undo_top = [&](const char* tag) { app_print_undo_top(a, tag); };
    auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(a); };
    auto arm_geometry_handoff = [&](const Mesh& m) { app_arm_geometry_handoff(a, m); };
    // Multires level switch (D / Shift-D post-lock). Recorded on the undo
    // timeline as a LEVEL entry so Ctrl-Z retraces the literal action sequence
    // (…draw, subd-up, draw, subd-down…) with the view level following along.
    if (input.level_switch_delta != 0 && swallow_level_switch) {
        // Drop input queued while the previous switch froze the frame —
        // keymashing D must not stack further multi-second builds.
        input.level_switch_delta = 0;
    }
    swallow_level_switch = false;
    if (input.level_switch_delta != 0 && app_state != AppState::IDLE) {
        // Mid-stroke D: the cascade would rebuild the mesh under the live
        // brush (pen-down FBO cache, GPU accumulators, dirty lists all
        // reference the old buffers) — crash. Level switch is idle-only.
        input.level_switch_delta = 0;
    }
    if (input.level_switch_delta != 0) {
        int delta = input.level_switch_delta;
        input.level_switch_delta = 0;
        const int from   = multires->current_level;
        const int target = from + delta;
        // Subdivision guard: before going up a level, predict the largest
        // resulting GPU buffer against the limits the device actually granted.
        // Subdividing quadruples tris; the index SSBO and CSR adjacency both
        // weigh tris x 12 bytes and are the biggest allocations. Refusing here
        // beats attempting the build and losing the device (WebGPU treats an
        // over-limit buffer as a validation error -> device loss).
        if (delta > 0) {
            const uint64_t tris_next  = (uint64_t)mesh->tri_count() * 4ull;
            const uint64_t bytes_next = tris_next * 12ull;
            const gpu::DeviceLimits dl = gpu::device_limits();
            const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                                  ? dl.max_buffer_size : dl.max_storage_binding_size;
            if (bytes_next > budget) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Subdivision would exceed GPU limits (%.1fM tris)",
                              (double)tris_next / 1e6);
                input.notification_timer = 4.0f;
                std::printf("[multires] refused L%d -> L%d: %.1fM tris needs %llu MB, device grants %llu MB\n",
                            from, target, (double)tris_next / 1e6,
                            (unsigned long long)(bytes_next >> 20),
                            (unsigned long long)(budget >> 20));
                delta = 0;
            }
        }
        const double switch_t0 = glfwGetTime();
        if (delta != 0 && target >= multires->base_level && target <= max_level) {
            scene.materialize_active_cpu();  // 2b: projection/cascade read disp/base
            UndoEntry lvl_e;
            lvl_e.kind       = UndoEntry::Kind::LEVEL;
            lvl_e.from_level = from;
            lvl_e.to_level   = target;
            if (delta < 0) {
                // Auto-project before descending: bake detail from L_max down to
                // target. The snapshot rides on the LEVEL entry — undo restores it,
                // redo replays the projection.
                const int L_max = multires->base_level + (int)multires->disp.size();
                capture_projection_snapshot(*multires, target, lvl_e.before);
                ProjectionStats ps = project_down_to_level(*multires, target);
                std::printf("[project] auto L%d -> L%d in %.2f ms\n", L_max, target, ps.elapsed_ms);
                report_projection_check(ps);
            }
            scene.active_undo().push(std::move(lvl_e));
            multires->current_level = target;
            multires_sync_paint(*multires, *mesh);
            if (scene.alive_count() <= 1) {
                // Paint/mask ride the cascade (finest-level planes) — no
                // save/restore needed across the rebuild.
                cascade_to_level(*multires, *mesh, target, &compute);
                arm_geometry_handoff(*mesh);
                scene.refresh_mirror_map();
                scene.sync();  // rebinds active: rebuilds adjacency from new topology
            } else {
                Mesh solo;
                cascade_to_level(*multires, solo, target, &compute);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);  // splice_active marks topo dirty
                scene.refresh_mirror_map();
            }
            refresh_active_gpu_residency();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            screen_buffers_dirty = true;
            std::printf("[multires] switched to level %d (%u verts, %u tris) in %.1f ms\n",
                        target, mesh->vertex_count(), mesh->tri_count(),
                        (glfwGetTime() - switch_t0) * 1000.0);
            print_undo_top("level-switch");
            // A switch slow enough to freeze the frame has stale input queued
            // behind it (GLFW delivers it all in the next poll) — swallow it.
            if (glfwGetTime() - switch_t0 > 0.15)
                swallow_level_switch = true;
        }
    }

    // Delete the highest subdivision level (burger menu -> Y/N confirm).
    // Destructive and deliberately NOT undoable: the whole point is to hand
    // the top layer's memory back, and an undo entry holding it would give
    // exactly none of that back. The history is cleared instead — its STROKE
    // entries carry vert ids from a level that no longer exists, and a LEVEL
    // entry replaying upward would resurrect the level as a flat zero layer.
    if (input.drop_level_requested) {
        input.drop_level_requested = false;
        const int L_max = multires->base_level + (int)multires->disp.size();
        if (!multires->locked || multires->disp.empty()) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "No subdivision level to delete");
            input.notification_timer = 2.5f;
        } else if (app_state != AppState::IDLE) {
            // Same reason the level switch is idle-only: the cascade would
            // rebuild the mesh out from under a live stroke.
            std::snprintf(input.notification, sizeof(input.notification),
                          "Finish the stroke first");
            input.notification_timer = 2.0f;
        } else {
            const double drop_t0 = glfwGetTime();
            scene.mark_edited();
            const int target = L_max - 1;
            scene.materialize_active_cpu();  // projection/cascade read disp/base

            // Bake first, then drop: the same inverse-Loop projection a level-down
            // does, so the form survives into the layer below and only the
            // residual detail dies with the layer.
            ProjectionStats ps = project_down_to_level(*multires, target);
            report_projection_check(ps);
            // Only forced down if the view was ON the layer being removed —
            // deleting L5 while editing at L3 must leave you at L3.
            if (multires->current_level > target) multires->current_level = target;
            const int view = multires->current_level;
            multires_sync_paint(*multires, *mesh);   // fold paint in BEFORE the planes shrink
            multires_drop_top_level(*multires);

            scene.active_undo().clear(&compute);
            xforms.clear();          // history wiped means history wiped
            // Grow-only GPU mirror still sized for the removed level; drop it
            // so the refresh below re-allocates at the new (smaller) level.
            scene.active_entity().multires_gpu.cleanup();

            if (scene.alive_count() <= 1) {
                cascade_to_level(*multires, *mesh, view, &compute);
                arm_geometry_handoff(*mesh);
                scene.refresh_mirror_map();
                scene.sync();
            } else {
                Mesh solo;
                cascade_to_level(*multires, solo, view, &compute);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);
                scene.refresh_mirror_map();
            }
            refresh_active_gpu_residency();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            screen_buffers_dirty = true;
            std::snprintf(input.notification, sizeof(input.notification),
                          "Deleted subdiv level %d - max is now L%d (%.2fM tris here)",
                          L_max, target, (double)mesh->tri_count() / 1e6);
            input.notification_timer = 3.0f;
            std::printf("[multires] delete L%d: projected in %.2f ms, max now L%d, "
                        "viewing L%d (%u verts, %u tris), total %.1f ms\n",
                        L_max, ps.elapsed_ms, target, view, mesh->vertex_count(),
                        mesh->tri_count(), (glfwGetTime() - drop_t0) * 1000.0);
        }
    }

    // Handle focus request (F key)
    if (input.focus_requested) {
        input.focus_requested = false;
        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
        // Retopo (0) frames the piece being worked on (retopo_focus).
        const bool framed = input.interaction_mode == InputState::InteractionMode::RETOPO
                         && retopo_focus(a.retopo, camera, mesh_radius);
        if (!framed) {
            camera.set_target(brush_stroke.last_stroke_valid ? brush_stroke.last_stroke_pos : mesh_center);
            camera.distance = std::max(camera.distance * 0.675f, mesh_radius * 0.05f);
        }
        screen_buffers_dirty = true;
    }

    // Snap views (F1/F2/F3)
    if (input.snap_view_requested != InputState::SnapView::NONE) {
        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
        camera.set_target(mesh_center);
        camera.distance = mesh_radius * 2.5f;
        switch (input.snap_view_requested) {
            case InputState::SnapView::FRONT:
                camera.yaw = 0.0f;
                camera.pitch = 0.0f;
                break;
            case InputState::SnapView::SIDE:
                camera.yaw = (float)(M_PI / 2.0);
                camera.pitch = 0.0f;
                break;
            case InputState::SnapView::TOP:
                camera.yaw = 0.0f;
                camera.pitch = (float)(M_PI / 2.0 - 0.001);
                break;
            default: break;
        }
        input.snap_view_requested = InputState::SnapView::NONE;
        screen_buffers_dirty = true;
    }
}

// Remesh and SDF merge: the remesh runs inside one frame (after a frame that shows its
// panel), the merge spans frames, one budgeted step each.
void app_ops_remesh_and_merge(App& a) {
    auto& input = a.input;
    auto& scene = a.scene;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& compute = a.compute;
    auto& brush_stroke = a.brush_stroke;
    auto& xforms = a.xforms;
    auto& vmerge_job = a.vmerge_job;
    auto& app_state = a.app_state;
    auto& mesh_center = a.mesh_center;
    auto& mesh_radius = a.mesh_radius;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto& rescue = a.rescue;
    auto& remesh_chained_from_merge = a.remesh_chained_from_merge;
    auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(a); };
    auto take_rescue = [&](const char* op) { app_take_rescue(a, op); };
    // ---- Remesh execution ----
    // A remesh runs inside this one frame, so the "Remeshing..." panel would never
    // reach the screen: the first frame only raises remesh_in_progress (the panel
    // draws and presents), the next one does the work.
    static bool remesh_shown = false;
    if (input.remesh_requested && !remesh_shown) {
        remesh_shown = true;
        input.remesh_in_progress = true;
    } else if (input.remesh_requested) {
        remesh_shown = false;
        input.remesh_requested = false;
        input.remesh_in_progress = true;

        scene.materialize_active_cpu();  // 2b: remesh reads the live surface (mesh.pos)

        // Adaptive-remesh guard (density spec Q2): a red-heavy field at a
        // small fine-mult can explode tri counts the same way keymashed
        // subdivision does. Same predicted-size check, same refusal toast.
        bool remesh_refused = false;
        // The detail slider multiplies the count on either path, so it rides
        // the same guard — a x4 on a big mesh is the uniform version of the
        // red-heavy field this was written for.
        const float detail = input.remesh_detail;
        if (!mesh->density.empty() || detail > 1.0f) {
            const uint64_t base_pred = mesh->density.empty()
                ? (uint64_t)mesh->tri_count()
                : predict_adaptive_tris(*mesh, 0.0f, input.density_coarse_mult,
                                        input.density_fine_mult);
            const uint64_t tris_pred = (uint64_t)((double)base_pred * std::max(1.0f, detail));
            const uint64_t bytes_pred = tris_pred * 12ull;  // index/CSR SSBOs
            const gpu::DeviceLimits dl = gpu::device_limits();
            const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                                  ? dl.max_buffer_size : dl.max_storage_binding_size;
            if (bytes_pred > budget) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Remesh would exceed GPU limits (%.1fM tris)",
                              (double)tris_pred / 1e6);
                input.notification_timer = 4.0f;
                std::printf("[remesh] refused: %.1fM predicted tris needs %llu MB, device grants %llu MB\n",
                            (double)tris_pred / 1e6,
                            (unsigned long long)(bytes_pred >> 20),
                            (unsigned long long)(budget >> 20));
                remesh_refused = true;
            }
        }

        // Rescue copy before the topology goes. A second remesh replaces the
        // first's snapshot — one step back means the LAST remesh — but a
        // remesh chained off a merge leaves the merge's snapshot alone.
        if (!remesh_refused && !remesh_chained_from_merge) take_rescue("remesh");

        // Destructive remesh breaks topology mirror — switch to spatial
        // mode afterward. Mirror setting (input.mirror_x) is preserved.
        // Dev hook: CHISEL_REMESH_SRC_DUMP=<path.obj> writes the mesh going IN.
        if (!remesh_refused)
            if (const char* dump = std::getenv("CHISEL_REMESH_SRC_DUMP"))
                mesh->export_obj(dump);
        RemeshResult result;
        if (!remesh_refused)
            result = perform_remesh(*mesh, *multires, 0.0f, 10,
                                    compute.supported ? &compute : nullptr,
                                    input.density_coarse_mult,
                                    input.density_fine_mult,
                                    input.remesh_keep_detail,
                                    input.remesh_detail);

        // Dev hook: CHISEL_REMESH_DUMP=<path.obj> writes the remesh result;
        // CHISEL_AUTO_QUIT=1 then exits, so a scripted run ends on its own.
        if (result.success)
            if (const char* dump = std::getenv("CHISEL_REMESH_DUMP"))
                mesh->export_obj(dump);
        if (std::getenv("CHISEL_AUTO_QUIT")) {
            std::fflush(stdout);
            debug_console::pump();   // stdout is piped through the ~ console
            std::_Exit(result.success ? 0 : 3);
        }
        if (result.success) {
            scene.mark_edited();
            mesh->mask.clear();
            // perform_remesh rebuilt the ACTIVE entity's mesh + multires base
            // in place. Leave every other entity untouched — only the active
            // one was remeshed. Its topology is fresh (base level 0), so reset
            // its subdiv level, refresh its mirror map, and re-sync the working
            // set. Spatial mirror mode: destructive remesh breaks topology mirror.
            scene.set_mirror_topology(false);
            scene.active_entity().subdiv_level = 0;
            // Remesh is single-entity: it only rebuilt the active mesh, so a
            // leftover multi-selection is stale. Collapse it to the active
            // entity now — clears the deselected tint and keeps downstream
            // ops (edit-mode entry, merge) from acting on a mixed set.
            scene.collapse_selection_to_active();
            scene.refresh_mirror_map();
            scene.sync();
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            // Same resync as the voxel merge: the GPU copy of the base still
            // holds the PRE-remesh mesh, and strokes at the base level are
            // diffed against it and later materialized from it. Without this,
            // every stroke after a remesh landed as old-mesh positions at the
            // new mesh's ids — invisible until the next level change rebuilt
            // the model from them (2026-09-25).
            refresh_active_gpu_residency();
            screen_buffers_dirty = true;
            brush_stroke.vertex_count = 0;
            brush_stroke.phase = StrokePhase::NONE;
            app_state = AppState::IDLE;
            if (result.selected_tris > 0) { scene.active_undo().clear(&compute); xforms.clear(); }
            std::snprintf(input.notification, sizeof(input.notification),
                          "Remesh: %u sel, %u/%u -> %u/%u v/t (spatial mirror)",
                          result.selected_tris,
                          result.old_verts, result.old_tris,
                          result.new_verts, result.new_tris);
            input.notification_timer = 4.0f;
        } else if (!remesh_refused) {
            // Nothing changed, so there is nothing this remesh can rescue —
            // but a chained one must leave the merge's snapshot standing.
            if (!remesh_chained_from_merge) rescue.clear();
            std::printf("[remesh] FAILED: %s\n", result.error.c_str());
            std::snprintf(input.notification, sizeof(input.notification),
                          "Remesh FAILED — check console");
            input.notification_timer = 4.0f;
        }

        remesh_chained_from_merge = false;
        input.remesh_in_progress = false;
    }

    // ---- Voxel merge execution (SDF join-for-print) ----
    // Kick off a merge: build the job (gather/grid/alloc/compile), then let the
    // per-frame advance below tick it to completion. The merge spans frames so the
    // window stays responsive and the progress HUD animates (the winding-sign pass
    // alone is seconds at R>=128).
    if (input.voxel_merge_requested) {
        input.voxel_merge_requested = false;
        if (!compute.supported) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "SDF remesh needs GPU compute (unavailable)");
            input.notification_timer = 4.0f;
        } else if (!vmerge_job) {
            take_rescue("SDF remesh");
            scene.materialize_active_cpu();  // 2b: merge reads the live surface (mesh.pos)
            if (const char* dump = std::getenv("CHISEL_MERGE_SRC_DUMP"))   // dev: pre-merge mesh
                scene.active_mesh().export_obj(dump);
            vmerge_job = voxel_merge_begin(scene, compute,
                                           input.voxel_merge_resolution,
                                           input.voxel_merge_mirror,
                                           input.voxel_merge_surface_nets,
                                           input.voxel_merge_subtract,
                                           input.remesh_keep_detail);
            input.voxel_merge_in_progress = true;
        }
    }

    // Advance an in-flight merge by one budgeted step.
    if (vmerge_job) {
        VoxelMergeResult vm;
        VoxelMergeStatus st = voxel_merge_tick(scene, compute, *vmerge_job, vm);
        if (st != VoxelMergeStatus::Working) {
            voxel_merge_destroy(vmerge_job);
            vmerge_job = nullptr;
            input.voxel_merge_in_progress = false;

            if (st == VoxelMergeStatus::Done && vm.success) {
                scene.mark_edited();
                // Mirror merge yields a tessellation-symmetric mesh → topology
                // mirror gives an exact partner map. Faithful merge is generic
                // geometry → spatial mirror. Either way refresh from the fresh mesh.
                scene.set_mirror_topology(input.voxel_merge_mirror);
                scene.refresh_mirror_map();
                scene.sync();
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                // The merge replaces the active entity wholesale; resync the
                // GPU-residency mirror to the fresh mesh's current level so the
                // Phase-2 diff/apply shaders read a valid disp/base layer.
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                bool watertight = (vm.boundary_edges == 0 && vm.nonmanifold_edges == 0);
                std::snprintf(input.notification, sizeof(input.notification),
                              "SDF remesh %u -> %u v / %u t (R=%u, %.0f ms) | %s: %u comp, %u bnd, %u nonmf",
                              vm.in_entities, vm.out_verts, vm.out_tris,
                              vm.R, vm.elapsed_ms,
                              watertight ? "watertight" : "NOT watertight",
                              vm.components, vm.boundary_edges, vm.nonmanifold_edges);
                input.notification_timer = 6.0f;
                std::printf("[voxel-merge] %s\n", input.notification);
                // Dev hook: CHISEL_MERGE_DUMP=<path.obj> writes the result for offline checks.
                if (const char* dump = std::getenv("CHISEL_MERGE_DUMP"))
                    mesh->export_obj(dump);
                // Chain the adaptive remesh: the merge output is uniform-dense
                // MC soup; when it carries a painted density field the user
                // wants the topology to follow it right away (D in the merge
                // dialog opts out). Runs through the normal remesh block next
                // frame — same GPU-limit guard, same notification.
                if (input.voxel_merge_adaptive && !mesh->density.empty()) {
                    std::printf("[voxel-merge] density field present -> chaining adaptive remesh\n");
                    input.remesh_requested = true;
                    remesh_chained_from_merge = true;
                }
            } else {
                rescue.clear();   // scene untouched, nothing to revert to
                std::printf("[voxel-merge] FAILED: %s\n", vm.error.c_str());
                std::snprintf(input.notification, sizeof(input.notification),
                              "SDF remesh failed: %.200s", vm.error.c_str());
                input.notification_timer = 4.0f;
            }
        }
    }
}
