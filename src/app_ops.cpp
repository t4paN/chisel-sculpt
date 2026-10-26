// The frame's one-shot operations and shared helpers, moved out of main.cpp as is.
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include "app.h"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cctype>
#include <algorithm>
#include <string>

#include "mesh.h"
#include "camera.h"
#include "renderer.h"
#include "input.h"
#include "settings.h"
#include "text_overlay.h"
#include "brush.h"
#include "tablet.h"
#include "undo.h"
#include "multires_stack.h"
#include "remesh.h"
#include "compute.h"
#include "chisel_debug.h"
#include "scene.h"
#include "sdf.h"
#include "insert_controller.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include "web_bridge.h"
#include "gpu/wgpu_surface.h"
#include "ui_backdrop.h"
#include "debug_console.h"
#include "bench.h"
#include "imgui.h"
#include "imgui_internal.h"   // InputEventsQueue: the idle frame skip reads it
#include "imgui_impl_glfw.h"
#ifdef CHISEL_BACKEND_WEBGPU
#include "imgui_impl_wgpu.h"
#else
#include "imgui_impl_opengl3.h"
#endif
#include "ImGuiFileDialog.h"
#include "brush_alpha.h"
#include "project_file.h"
#include "scene_snapshot.h"
#include "object_xform.h"
#include <string>
#include <filesystem>

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
// stacks. Lives here rather than outside the loop because it needs
// refresh_active_gpu_residency, which is itself a per-frame lambda.
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
