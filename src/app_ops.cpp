// The frame's one-shot operations and shared helpers, moved out of main.cpp as is.
#include "bench.h"
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
#if defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <malloc.h>
#endif
#include <cstring>

// Memory probe for the level-down (2026-11-17: the web heap aborted descending a
// 30M-tri model). Native Linux: resident set and its high-water mark, which
// /proc/self/clear_refs "5" resets so the peak is this operation's. Web: the WASM
// heap size, which only grows, so its final size IS the peak.
static size_t proc_status_kb(const char* key) {
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
    size_t kb = 0;
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        const size_t kl = std::strlen(key);
        while (std::fgets(line, sizeof(line), f))
            if (std::strncmp(line, key, kl) == 0) { kb = (size_t)std::strtoull(line + kl, nullptr, 10); break; }
        std::fclose(f);
    }
    return kb;
#else
    (void)key; return 0;
#endif
}
static size_t mem_now_mb() {
#if defined(__EMSCRIPTEN__)
    return emscripten_get_heap_size() >> 20;
#else
    return proc_status_kb("VmRSS:") >> 10;
#endif
}
static void mem_peak_reset() {
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
    if (FILE* f = std::fopen("/proc/self/clear_refs", "w")) { std::fputs("5", f); std::fclose(f); }
#endif
}
static size_t mem_peak_mb() {
#if defined(__EMSCRIPTEN__)
    return emscripten_get_heap_size() >> 20;
#else
    return proc_status_kb("VmHWM:") >> 10;
#endif
}

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
    // Grow-only scratch follows the mesh back down (subd-memory-second-trip-handoff).
    // The shared buffers keep room for the biggest mesh in the scene, so switching
    // between a big sculpt and a small one never frees and regrows them.
    uint32_t biggest = 0;
    for (const auto& e : scene.entities())
        if (e && e->alive) biggest = std::max(biggest, e->mesh.vertex_count());
    compute.trim_scratch(biggest);
    ent.multires_gpu.trim_scratch(ent.mesh.vertex_count());
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
// Same 4 GB wall as the level switches: SDF on a 30.8M-tri model aborted the page
// before its first log line (2026-11-25). The merge's CPU side holds the triangle soup
// (~12 B/tri indices + 12 B/vert positions), the splat boxes and offsets read back for
// the scan (28 B/tri), and up to three grids ((R+1)^3 floats each). Native: always fits.
static bool app_sdf_fits_web_heap(App& a) {
#if defined(__EMSCRIPTEN__)
    auto& scene = a.scene;
    auto& input = a.input;
    uint64_t tris = 0, verts = 0;
    for (uint32_t id : scene.selected_ids()) {
        const MeshEntity* e = scene.find_entity(id);
        if (!e || !e->alive) continue;
        tris  += e->mesh.tri_count();
        verts += e->mesh.vertex_count();
    }
    const uint64_t R     = (uint64_t)std::min(input.voxel_merge_resolution,
                                              voxel_merge_max_resolution()) + 1;
    const uint64_t need  = tris * 40 + verts * 24 + 3 * R * R * R * 4;   // 64-bit: size_t is 32 here
    const uint64_t used  = mallinfo().uordblks;
    const uint64_t cap   = emscripten_get_heap_max();
    const uint64_t have  = cap > used ? cap - used : 0;
    const uint64_t slack = (uint64_t)256 << 20;
    std::printf("[sdf] web heap check: %.1fM tris needs ~%u MB, %u MB left\n",
                (double)tris / 1e6, (unsigned)(need >> 20), (unsigned)(have >> 20));
    if (need + slack > have) {
        std::snprintf(input.notification, sizeof(input.notification),
                      "Too big to SDF in the browser (needs %u MB more, %u MB left). "
                      "Go down a level first, or use the desktop app.",
                      (unsigned)(need >> 20), (unsigned)(have >> 20));
        input.notification_timer = 6.0f;
        return false;
    }
#else
    (void)a;
#endif
    return true;
}

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
    // Colour picker (C in paint mode): sample the stored albedo under the cursor —
    // tri + barycentrics from the screen FBO, colours from the CPU mesh (synced at
    // pen-up), so shading/matcap never leaks into the pick. While the press is held
    // every frame samples into the preview swatch (a 1-px read, like the hover depth
    // read); the release takes the last sample. Off the model the swatch holds.
    if (input.color_pick_dragging || input.color_pick_click) {
        int cx = (int)input.mouse_x;
        int cy = (int)input.mouse_y;
        if (!input.color_pick_active) {
            input.color_pick_dragging = false;
            input.color_pick_click = false;
        } else if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
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
                // A textured mesh shows paint x texture (the textured draw), so a plain
                // pick takes that: the texture at the cursor's corner UVs. Ctrl picks
                // the paint alone, since repainting the product over the texture
                // darkens each round (red over grey concrete comes back dark red).
                uint32_t tc = 0xFFFFFFFFu;
                const float tmix = input.ctrl_held ? 0.0f
                                 : pm.paint_layers.texture_mix();   // the Texture row
                if (pm.texture && pm.has_uv() && tmix > 0.0f) {
                    const float* uv = &pm.uv[6 * (size_t)tid];
                    tc = pm.texture->sample(bu * uv[0] + bv * uv[2] + bw * uv[4],
                                            bu * uv[1] + bv * uv[3] + bw * uv[5]);
                }
                for (int ch = 0; ch < 3; ch++) {
                    float f0 = (float)((c0 >> (ch*8)) & 0xFF);
                    float f1 = (float)((c1 >> (ch*8)) & 0xFF);
                    float f2 = (float)((c2 >> (ch*8)) & 0xFF);
                    const float t = 1.0f + ((float)((tc >> (ch*8)) & 0xFF) / 255.0f - 1.0f) * tmix;
                    input.color_pick_preview[ch] = (bu*f0 + bv*f1 + bw*f2) / 255.0f * t;
                }
                input.color_pick_have = true;
            }
        }
        // Released: take the last sample. A drag that never touched the model
        // leaves the picker armed.
        if (input.color_pick_click) {
            input.color_pick_click = false;
            if (input.color_pick_have) {
                for (int ch = 0; ch < 3; ch++) input.paint_color[ch] = input.color_pick_preview[ch];
                input.color_pick_active = false;
                input.color_pick_have = false;
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

// The size guards (subdivide, remesh) predict the biggest single GPU buffer a triangle
// count needs and hold it against the device's limit: the index list and the CSR
// adjacency list, 12 B/tri each. (Until 2026-11-11 the pick pass kept a flat copy of
// the mesh at 36 B/tri, which this figure missed; that copy is gone.)
static const uint64_t kBiggestBufferBytesPerTri = 12ull;
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
    gpu::user_memory_gb = input.gpu_memory_gb;   // Settings -> the seam, where the size rules live
    // A remesh job owns the mesh until its last step: no level switch under it.
    if (input.remesh_in_progress) input.level_switch_delta = 0;
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
    // A Yolo whose level-up was dropped above must not wave a later one through.
    if (input.level_switch_delta == 0) input.subdiv_confirmed = false;
    if (input.level_switch_delta != 0) {
        int delta = input.level_switch_delta;
        input.level_switch_delta = 0;
        const int from   = multires->current_level;
        const int target = from + delta;
        bool deferred = false;   // web: a big switch waits one frame behind its notice
        // Subdivision guard: before going up a level, predict the largest
        // resulting GPU buffer against the limits the device actually granted.
        // Subdividing quadruples tris. Refusing here beats attempting the build
        // and losing the device (WebGPU treats an over-limit buffer as a
        // validation error -> device loss).
        if (delta > 0 && target > max_level) {
            // Said here, before the web's "will freeze" notice below, which would
            // otherwise promise a switch that the level cap then silently drops.
            std::snprintf(input.notification, sizeof(input.notification),
                          "Highest subdivision level reached (L%d)", max_level);
            input.notification_timer = 3.0f;
            delta = 0;
        }
        if (delta > 0) {
            const uint64_t tris_next  = (uint64_t)mesh->tri_count() * 4ull;
            const uint64_t bytes_next = tris_next * kBiggestBufferBytesPerTri;
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
            } else if (tris_next > gpu::fits_max_tris() && !input.subdiv_confirmed) {
                // It fits the device's limits, but by the Graphics memory setting it is
                // more than the card holds: the machine may crawl (62.9M tris on 10 GB
                // spilled into system RAM, 650 ms a frame) or the driver may give up. The user's call, not ours: ask
                // (flashing skull). Y comes back here as the same level-up with
                // subdiv_confirmed set.
                input.subdiv_confirm_tris = tris_next;
                input.subdiv_confirm_pending = true;
                std::printf("[multires] L%d -> L%d is %.1fM tris: asking first\n",
                            from, target, (double)tris_next / 1e6);
                delta = 0;
            }
        }
#if defined(__EMSCRIPTEN__)
        // The browser's heap is capped (4 GB, CMakeLists). A level-down on a
        // model with tens of millions of tris can need more than is left, and an
        // allocation failure there aborts the whole page, work and all
        // (2026-11-17). Refuse up front instead. "In use" is malloc's live total, not
        // the heap size: the heap never shrinks, so after one big descent its size
        // would refuse every later one. The slack covers fragmentation (big vectors
        // want contiguous space) and the estimate's rounding.
        if (delta < 0 && target >= multires->base_level) {
            const uint64_t need  = projection_memory_estimate(*multires, target);
            const uint64_t used  = mallinfo().uordblks;
            const uint64_t cap   = emscripten_get_heap_max();
            const uint64_t have  = cap > used ? cap - used : 0;
            const uint64_t slack = (uint64_t)256 << 20;
            if (need + slack > have) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Too big to go down a level in the browser (needs %u MB "
                              "more, %u MB left). The desktop app can.",
                              (unsigned)(need >> 20), (unsigned)(have >> 20));
                input.notification_timer = 6.0f;
                std::printf("[project] refused L%d -> L%d on web: needs ~%u MB, %u MB "
                            "left under the heap cap\n", from, target, (unsigned)(need >> 20), (unsigned)(have >> 20));
                delta = 0;
            }
        }
        // Same wall going up: 7.7M -> 30.8M tris aborted the page (bench 2026-11-23). The
        // GPU-limit check above passes, the heap is what runs out.
        if (delta > 0) {
            const uint64_t need  = level_up_memory_estimate(*multires, *mesh, target);
            const uint64_t used  = mallinfo().uordblks;
            const uint64_t cap   = emscripten_get_heap_max();
            const uint64_t have  = cap > used ? cap - used : 0;
            const uint64_t slack = (uint64_t)256 << 20;
            if (need + slack > have) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Too big to subdivide in the browser (needs %u MB more, %u MB "
                              "left). The desktop app can.",
                              (unsigned)(need >> 20), (unsigned)(have >> 20));
                input.notification_timer = 6.0f;
                std::printf("[multires] refused L%d -> L%d on web: needs ~%u MB, %u MB "
                            "left under the heap cap\n", from, target, (unsigned)(need >> 20), (unsigned)(have >> 20));
                delta = 0;
            }
        }
        // A big switch freezes the tab for many seconds (~23 s for 7.7M -> 30.8M) and the
        // page can't draw while it runs, so a user sees a hung tab and clicks into it. Say
        // so first: show the notice this frame, run the switch on the next one.
        {
            // When the notice went up; a switch within a second of it is the deferred one.
            // A time, not a flag: a switch dropped in between (a stroke began) can't leave
            // the next big one without its notice.
            static double notice_at = -1e9;
            const bool notice_shown = glfwGetTime() - notice_at < 1.0;
            const double tris_now  = (double)mesh->tri_count();
            const double tris_then = delta > 0 ? tris_now * 4.0 : tris_now / 4.0;
            const bool big = delta != 0 && std::max(tris_now, tris_then) >= 8e6;
            if (big && !notice_shown) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "%s to %.1fM tris - the page will freeze for a while, "
                              "please wait", delta > 0 ? "Subdividing" : "Going down",
                              tris_then / 1e6);
                input.notification_timer = 4.0f;
                input.level_switch_delta = delta;   // run it next frame; subdiv_confirmed rides along
                notice_at = glfwGetTime();
                delta = 0;
            } else if (big) {
                notice_at = -1e9;
            }
            deferred = input.level_switch_delta != 0;
        }
#endif
        // Good for the one switch it was given for; a deferred switch keeps it for its real run.
        if (!deferred) input.subdiv_confirmed = false;
        const double switch_t0 = glfwGetTime();
        // Level-up memory, logged so level_up_memory_estimate can be calibrated (web: heap
        // size, which never shrinks, so "peak" is the high-water mark; native: VmHWM).
        const size_t up_mem0 = mem_now_mb();
        const size_t up_est  = delta > 0 ? level_up_memory_estimate(*multires, *mesh, target) >> 20 : 0;
        if (delta > 0) mem_peak_reset();
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
                const size_t mem0 = mem_now_mb();
                const size_t est  = projection_memory_estimate(*multires, target) >> 20;
                mem_peak_reset();
                // The live mesh is rebuilt below; its level's topology goes back to the
                // cache first so the projection finds it warm (see LevelTopo::on_live).
                multires_reclaim_live_topo(*multires, *mesh);
                capture_projection_snapshot(*multires, target, lvl_e.before);
                ProjectionStats ps = project_down_to_level(*multires, target);
                std::printf("[project] auto L%d -> L%d in %.2f ms | memory %zu MB before, "
                            "peak %zu MB (estimate +%zu), undo copy %zu MB\n", L_max, target,
                            ps.elapsed_ms, mem0, mem_peak_mb(), est,
                            lvl_e.before.bytes() >> 20);
                report_projection_check(ps);
            }
            scene.active_undo().push(std::move(lvl_e));
            multires->current_level = target;
            bench::pd_arm_level(compute.gpu_dev);
            multires_sync_paint(*multires, *mesh);
            bench::pd_mark("level: sync paint"); memtrace_level("app: sync paint");
            if (scene.alive_count() <= 1) {
                // Paint/mask ride the cascade (finest-level planes) — no
                // save/restore needed across the rebuild.
                cascade_to_level(*multires, *mesh, target, &compute, true);
                bench::pd_mark("level: cascade"); memtrace_level("app: cascade done");
                arm_geometry_handoff(*mesh);
                bench::pd_mark("level: geometry handoff"); memtrace_level("app: geometry handoff");
                scene.refresh_mirror_map();
                bench::pd_mark("level: mirror map"); memtrace_level("app: mirror map");
                scene.sync();  // rebinds active: rebuilds adjacency from new topology
                bench::pd_mark("level: scene sync (adjacency)"); memtrace_level("app: scene sync");
            } else {
                Mesh solo;
                multires_reclaim_live_topo(*multires, *mesh);   // *mesh is replaced by solo
                cascade_to_level(*multires, solo, target, &compute, true);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);  // splice_active marks topo dirty
                scene.refresh_mirror_map();
            }
            refresh_active_gpu_residency();
            bench::pd_mark("level: gpu residency"); memtrace_level("app: gpu residency");
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            screen_buffers_dirty = true;
            std::printf("[multires] switched to level %d (%u verts, %u tris) in %.1f ms\n",
                        target, mesh->vertex_count(), mesh->tri_count(),
                        (glfwGetTime() - switch_t0) * 1000.0);
            if (delta > 0)
                std::printf("[multires] level-up L%d -> L%d memory %zu MB before, peak %zu MB "
                            "(estimate +%zu)\n", from, target, up_mem0, mem_peak_mb(), up_est);
            print_undo_top("level-switch");
            {
                char tag[48];
                std::snprintf(tag, sizeof tag, "after switch L%d -> L%d", from, target);
                gpu::dump_live_buffers(tag);
            }
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
            multires_reclaim_live_topo(*multires, *mesh);   // live mesh is rebuilt below
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
                cascade_to_level(*multires, *mesh, view, &compute, true);
                arm_geometry_handoff(*mesh);
                scene.refresh_mirror_map();
                scene.sync();
            } else {
                Mesh solo;
                cascade_to_level(*multires, solo, view, &compute, true);
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
                         && retopo_focus(a.scene.active_entity().retopo, camera, mesh_radius);
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

// Remesh and SDF merge: both span frames, one budgeted step each, so their panels
// show real progress and the window keeps drawing.
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
    auto arm_geometry_handoff = [&](const Mesh& m) { app_arm_geometry_handoff(a, m); };
    // ---- Remesh execution ----
    // The remesh is a job: setup, each outer iteration and the two finishing
    // passes are one step a frame. The first frame only raises remesh_in_progress
    // so the panel is up before setup (the slowest step) runs.
    static bool remesh_shown = false;
    static RemeshJob* remesh_job = nullptr;
    if (input.remesh_requested && !remesh_shown) {
        remesh_shown = true;
        input.remesh_in_progress = true;
        input.remesh_progress = 0.0f;
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
            const uint64_t bytes_pred = tris_pred * kBiggestBufferBytesPerTri;
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
        if (!remesh_refused)
            remesh_job = remesh_begin(*mesh, *multires, 0.0f, 10,
                                      compute.supported ? &compute : nullptr,
                                      input.density_coarse_mult,
                                      input.density_fine_mult,
                                      input.remesh_keep_detail,
                                      input.remesh_detail,
                                      // A chained remesh finishes the merge, so it
                                      // keeps the side the merge dialog picked.
                                      remesh_chained_from_merge ? input.voxel_merge_mirror
                                                                : input.mirror_x);
        else {
            remesh_chained_from_merge = false;
            input.remesh_in_progress = false;
            if (std::getenv("CHISEL_AUTO_QUIT")) {   // scripted run: a refusal ends it too
                std::fflush(stdout);
                debug_console::pump();
                std::_Exit(3);
            }
        }
    }

    // ---- Rebuild levels ----
    // A dense mesh with no levels gets a stack under it, in three phases a frame or
    // more apart: (1) the normal remesh job takes it down to a coarse base, (2) one
    // level a frame is subdivided back up and shrinkwrapped onto the original, kept
    // here as a soup, (3) the lower levels are fitted to the top and the paint is
    // carried. After that it is an ordinary stack: D / Shift+D, cascade, projection.
    struct RebuildLevels {
        int stage = 0;                 // 0 idle, 1 remeshing down, 2 wrapping levels
        int levels = 0, done = 0;
        std::vector<float> pos;        // the original, xyz per vertex
        std::vector<uint32_t> idx, color;
        float base_edge = 0.0f, src_edge = 0.0f;
        double t0 = 0.0;
        bool flow_base = false;        // L0 is a Flow merge's grid, not a remesh
    };
    static RebuildLevels rb;
    if (input.rebuild_levels_requested) {
        input.rebuild_levels_requested = false;
        const bool chained = input.rebuild_levels_chained;
        input.rebuild_levels_chained = false;
        int N = std::min(MULTIRES_MAX_LEVEL, std::max(1, input.rebuild_levels_count));
        const uint64_t T = mesh->tri_count();
        // A Flow merge laid its grid for exactly these levels: that grid is L0, no remesh.
        std::vector<float> fb_pos; std::vector<uint32_t> fb_idx; int fb_levels = 0; float fb_edge = 0.0f;
        const bool flow_base = chained && sdf_take_flow_base(fb_pos, fb_idx, fb_levels, fb_edge)
                            && fb_levels == N;
        if (flow_base) {
            scene.materialize_active_cpu();
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            const uint32_t vc = mesh->vertex_count();
            double area = 0.0;
            for (uint32_t t = 0; t < mesh->tri_count(); t++) {
                const Vec3 p0 = mesh->get_pos(mesh->indices[t*3]);
                area += 0.5 * (double)(mesh->get_pos(mesh->indices[t*3+1]) - p0)
                                  .cross(mesh->get_pos(mesh->indices[t*3+2]) - p0).length();
            }
            rb = RebuildLevels{};
            rb.stage = 2; rb.levels = N; rb.t0 = glfwGetTime(); rb.flow_base = true;
            rb.base_edge = fb_edge;
            rb.src_edge  = (float)std::sqrt(4.0 * area / (std::sqrt(3.0) * std::max<double>(1.0, (double)T)));
            rb.pos.resize((size_t)vc * 3);
            for (uint32_t v = 0; v < vc; v++) {
                rb.pos[v*3] = mesh->pos_x[v]; rb.pos[v*3+1] = mesh->pos_y[v]; rb.pos[v*3+2] = mesh->pos_z[v];
            }
            rb.idx = mesh->indices;
            if (mesh->color.size() == vc) rb.color = mesh->color;
            // Install the grid as the live mesh and a fresh level-0 lock, as a remesh
            // would leave it (see the end of remesh_tick).
            Mesh b;
            const uint32_t bv = (uint32_t)(fb_pos.size() / 3);
            b.pos_x.resize(bv); b.pos_y.resize(bv); b.pos_z.resize(bv);
            b.norm_x.resize(bv); b.norm_y.resize(bv); b.norm_z.resize(bv);   // recompute_normals fills, doesn't size
            for (uint32_t v = 0; v < bv; v++) { b.pos_x[v] = fb_pos[v*3]; b.pos_y[v] = fb_pos[v*3+1]; b.pos_z[v] = fb_pos[v*3+2]; }
            b.indices = std::move(fb_idx);
            b.build_adjacency();
            b.recompute_normals();
            if (const char* dump = std::getenv("CHISEL_FLOW_BASE_DUMP")) b.export_obj(dump);   // dev: the grid as handed over
            scene.splice_active(b);
            scene.set_mirror_topology(false);
            scene.active_entity().subdiv_level = 0;
            scene.collapse_selection_to_active();
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            paint_layers_drop_stale(mesh->paint_layers, mesh->vertex_count());
            MultiresStack& stack = *multires;
            stack.base = *mesh;
            stack.base.paint_layers = PaintLayerStack{};
            stack.base_level = 0; stack.current_level = 0;
            stack.disp.clear(); stack.frames.clear(); stack.mirror.clear(); stack.base_mirror.clear();
            stack.midpoint_parents.clear(); stack.topo_cache.clear(); stack.lock_stamp = 0;
            stack.color.clear(); stack.mask.clear(); stack.density.clear();
            stack.base.color.clear(); stack.base.mask.clear(); stack.base.density.clear();
            stack.locked = true;
            mesh->mask.clear();
            scene.refresh_mirror_map();
            scene.sync();
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            refresh_active_gpu_residency();
            screen_buffers_dirty = true;
            input.rebuild_levels_running = true;
            input.remesh_in_progress = true;
            input.remesh_progress = 0.0f;
            std::printf("[rebuild] %d levels over %llu tris: Flow grid as the base (%u verts, %zu tris, edge %.4f), source edge %.4f\n",
                        N, (unsigned long long)T, bv, b.indices.size() / 3, rb.base_edge, rb.src_edge);
        } else {
        // Chained off an SDF merge, the dialog couldn't know the result's size: take as
        // many of the asked-for levels as the result has room for, rather than refuse.
        if (chained)
            while (N > 1 && (double)T / std::pow(4.0, N) < 400.0) N--;
        const double base_tris = (double)T / std::pow(4.0, N);
        const char* refuse = nullptr;
        if (rb.stage || remesh_job || input.remesh_requested || input.remesh_in_progress || vmerge_job)
            refuse = "Busy - try again when it finishes";
        else if (app_state != AppState::IDLE) refuse = "Finish the stroke first";
        else if (!multires->locked)           refuse = "Nothing to rebuild";
        else if (!multires->disp.empty())     refuse = "It already has subdiv levels - delete them first";
        // Below a few hundred tris the remesher can't hold a shape at all.
        else if (base_tris < 400.0)           refuse = "Too few triangles for that many levels";
        if (refuse) {
            std::snprintf(input.notification, sizeof(input.notification), "Rebuild levels: %s", refuse);
            input.notification_timer = 3.0f;
            std::printf("[rebuild] refused: %s\n", refuse);
        } else {
            scene.materialize_active_cpu();   // reads the live surface
            // After a merge, Undo should reach back past both, so the merge's copy stays.
            if (!chained) take_rescue("rebuild levels");
            mesh = &scene.active_mesh();       // the rescue may have rebuilt the scene
            multires = &scene.active_multires();
            const uint32_t vc = mesh->vertex_count();
            double area = 0.0;
            for (uint32_t t = 0; t < mesh->tri_count(); t++) {
                const Vec3 p0 = mesh->get_pos(mesh->indices[t*3]);
                area += 0.5 * (double)(mesh->get_pos(mesh->indices[t*3+1]) - p0)
                                  .cross(mesh->get_pos(mesh->indices[t*3+2]) - p0).length();
            }
            // Edge of an even mesh of n tris over this area. 0.87: the remesher settles
            // ~15% longer than it is aimed (same correction as its detail slider).
            auto edge_for = [&](double n) {
                return (float)std::sqrt(4.0 * area / (std::sqrt(3.0) * std::max(1.0, n)));
            };
            rb = RebuildLevels{};
            rb.stage = 1; rb.levels = N; rb.t0 = glfwGetTime();
            rb.base_edge = 0.87f * edge_for(base_tris);
            rb.src_edge  = edge_for((double)T);
            rb.pos.resize((size_t)vc * 3);
            for (uint32_t v = 0; v < vc; v++) {
                rb.pos[v*3] = mesh->pos_x[v]; rb.pos[v*3+1] = mesh->pos_y[v]; rb.pos[v*3+2] = mesh->pos_z[v];
            }
            rb.idx = mesh->indices;
            if (mesh->color.size() == vc) rb.color = mesh->color;
            mesh->mask.clear();                // the whole mesh is rebuilt; a mask would pin parts
            std::printf("[rebuild] %d levels over %llu tris: base edge %.4f (~%.0f tris), source edge %.4f\n",
                        N, (unsigned long long)T, rb.base_edge, base_tris, rb.src_edge);
            remesh_job = remesh_begin(*mesh, *multires, rb.base_edge, 10 + 4 * N,
                                      compute.supported ? &compute : nullptr,
                                      input.density_coarse_mult, input.density_fine_mult,
                                      true, 1.0f, input.mirror_x);
            // The top comes out at base x 4^N: a base that stops short of its target
            // puts the top that far over the original's count.
            remesh_set_target_tris(*remesh_job, (uint32_t)base_tris);
            input.rebuild_levels_running = true;
            input.remesh_in_progress = true;
            input.remesh_progress = 0.0f;
        }
        }
    }

    // Advance an in-flight remesh by one step; the rest runs once it is done.
    RemeshResult result;
    if (remesh_job && remesh_tick(*remesh_job, result)) {
        remesh_destroy(remesh_job);
        remesh_job = nullptr;

        // Dev hook: CHISEL_REMESH_DUMP=<path.obj> writes the remesh result;
        // CHISEL_AUTO_QUIT=1 then exits, so a scripted run ends on its own.
        if (result.success)
            if (const char* dump = std::getenv("CHISEL_REMESH_DUMP"))
                mesh->export_obj(dump);
        if (std::getenv("CHISEL_AUTO_QUIT") && rb.stage != 1) {
            std::fflush(stdout);
            debug_console::pump();   // stdout is piped through the ~ console
            std::_Exit(result.success ? 0 : 3);
        }
        if (result.success) {
            scene.mark_edited();
            mesh->mask.clear();
            // The remesh rebuilt the ACTIVE entity's mesh + multires base. Leave every other entity untouched — only the active
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
                          "Remesh: %u sel, %u/%u -> %u/%u v/t",
                          result.selected_tris,
                          result.old_verts, result.old_tris,
                          result.new_verts, result.new_tris);
            input.notification_timer = 4.0f;
        } else {
            // Nothing changed, so there is nothing this remesh can rescue —
            // but a chained one must leave the merge's snapshot standing.
            if (!remesh_chained_from_merge) rescue.clear();
            std::printf("[remesh] FAILED: %s\n", result.error.c_str());
            std::snprintf(input.notification, sizeof(input.notification),
                          "Remesh failed: %.200s",
                          result.error.empty() ? "check console" : result.error.c_str());
            input.notification_timer = 4.0f;
        }

        remesh_chained_from_merge = false;
        input.remesh_in_progress = false;
        if (rb.stage == 1) {
            if (result.success) {
                rb.stage = 2;                       // levels from next frame
                input.remesh_in_progress = true;    // keeps the guards and the panel up
            } else {
                rb = RebuildLevels{};
                input.rebuild_levels_running = false;
            }
        }
    }
    if (remesh_job) input.remesh_progress = remesh_progress(*remesh_job) / (float)(rb.stage == 1 ? rb.levels + 1 : 1);

    // Rebuild levels, phase 2-3: one level a frame, then the fit.
    if (rb.stage == 2 && !remesh_job) {
        // The gap between a fresh subdivided level and the original is a fraction of
        // the level below's edge; reach one whole edge. The grid cell stays near the
        // source's own edge, or every lookup at the coarse levels scans thousands of tris.
        const float parent_edge = rb.base_edge / (float)(1u << rb.done);
        const float reach = std::max(parent_edge, 3.0f * rb.src_edge);
        const float cell  = std::max(2.0f * rb.src_edge, 0.25f * reach);
        uint32_t filled = 0;
        rebuild_wrap_level(*multires, rb.pos, rb.idx, reach, cell, filled);
        rb.done++;
        input.remesh_progress = (float)(rb.done + 1) / (float)(rb.levels + 1);
        if (rb.done >= rb.levels || multires->base_level + (int)multires->disp.size() >= MULTIRES_MAX_LEVEL) {
            rebuild_finish(*multires, rb.pos, rb.idx, rb.color, 2.0f * std::max(rb.src_edge, parent_edge * 0.5f),
                           !rb.flow_base);   // a Flow base stays as laid: fitting it only grew spikes, the top came out identical
            const int top = multires->base_level + (int)multires->disp.size();
            multires->current_level = top;
            if (scene.alive_count() <= 1) {
                cascade_to_level(*multires, *mesh, top, &compute, true);
                arm_geometry_handoff(*mesh);
                scene.refresh_mirror_map();
                scene.sync();
            } else {
                Mesh solo;
                multires_reclaim_live_topo(*multires, *mesh);
                cascade_to_level(*multires, solo, top, &compute, true);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);
                scene.refresh_mirror_map();
            }
            mesh = &scene.active_mesh();
            multires = &scene.active_multires();
            // New topology at every level: no stroke or transform in the history fits it.
            // The rescue copy (taken at the start) is what Undo reaches instead.
            scene.active_undo().clear(&compute);
            xforms.clear();
            scene.mark_edited();
            refresh_active_gpu_residency();
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            screen_buffers_dirty = true;
            std::snprintf(input.notification, sizeof(input.notification),
                          "Rebuilt %d subdiv levels: base %u tris, top %u tris",
                          top - multires->base_level, multires->base.tri_count(), mesh->tri_count());
            input.notification_timer = 4.0f;
            std::printf("[rebuild] done in %.0f ms: L%d..L%d, base %u tris, top %u tris\n",
                        (glfwGetTime() - rb.t0) * 1000.0, multires->base_level, top,
                        multires->base.tri_count(), mesh->tri_count());
            if (const char* dump = std::getenv("CHISEL_REBUILD_DUMP")) {
                mesh->export_obj(dump);
                Mesh base_view;
                cascade_to_level(*multires, base_view, multires->base_level);
                std::string bp = std::string(dump) + ".base.obj";
                base_view.export_obj(bp.c_str());
            }
            rb = RebuildLevels{};
            input.rebuild_levels_running = false;
            input.remesh_in_progress = false;
            if (std::getenv("CHISEL_AUTO_QUIT")) {
                std::fflush(stdout);
                debug_console::pump();
                std::_Exit(0);
            }
        }
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
        } else if (!vmerge_job && !app_sdf_fits_web_heap(a)) {
            // refused: the toast and log line are set inside
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
                                           input.remesh_keep_detail,
                                           input.voxel_merge_close_holes,
                                           SdfPatternOpts{ input.voxel_merge_pattern, input.voxel_merge_poles,
                                                           input.voxel_merge_poles_on,
                                                           input.voxel_merge_rebuild_levels });
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
                if (vm.wrapped) {
                    const size_t at = std::strlen(input.notification);
                    std::snprintf(input.notification + at, sizeof(input.notification) - at,
                                  " | thin skin: wrapped");
                }
                if (scene.merge_uv_dropped()) {
                    const size_t at = std::strlen(input.notification);
                    std::snprintf(input.notification + at, sizeof(input.notification) - at,
                                  " | retopos joined, UVs need a re-unwrap");
                }
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
                if (input.voxel_merge_rebuild_levels > 0) {
                    // Fresh subdiv levels under the merge result, the top kept near the
                    // merge's own count. Its remesh follows a density field too, so it
                    // stands in for the adaptive remesh below.
                    std::printf("[voxel-merge] chaining Rebuild levels (%d)\n",
                                input.voxel_merge_rebuild_levels);
                    input.rebuild_levels_count = input.voxel_merge_rebuild_levels;
                    input.rebuild_levels_requested = true;
                    input.rebuild_levels_chained = true;
                } else if (input.voxel_merge_adaptive && !mesh->density.empty()) {
                    std::printf("[voxel-merge] density field present -> chaining adaptive remesh\n");
                    input.remesh_requested = true;
                    remesh_chained_from_merge = true;
                }
            } else {
                // Untouched on the desktop: the rescue copy no longer bakes levels there.
                // The browser does not either: over its 1M cap there is no copy at all.
                rescue.clear();
                std::printf("[voxel-merge] FAILED: %s\n", vm.error.c_str());
                std::snprintf(input.notification, sizeof(input.notification),
                              "SDF remesh failed: %.200s", vm.error.c_str());
                input.notification_timer = 12.0f;   // 4 s went by unread (2026-11-25)
            }
            // CHISEL_AUTO_QUIT: a scripted merge ends on its own, like a scripted remesh
            // (a chained Rebuild levels ends it instead).
            if (std::getenv("CHISEL_AUTO_QUIT") && !input.rebuild_levels_requested) {
                std::fflush(stdout);
                debug_console::pump();   // stdout is piped through the ~ console
                std::_Exit(vm.success ? 0 : 3);
            }
        }
    }
}
