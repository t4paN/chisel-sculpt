// The frame's drawing, moved out of main.cpp as is: cursor visibility, the scene pass
// (background, entities, debug mesh, pose rig, brush cursor), overlays and dialogs, then
// the UI tail (button islands, panels, error popup) and the ImGui pass + present.
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include "app.h"
#include <cstdio>
#include <string>

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
#include "gpu/wgpu_surface.h"
#include "ui_backdrop.h"
#include "debug_console.h"
#include "imgui.h"
#ifdef CHISEL_BACKEND_WEBGPU
#include "imgui_impl_wgpu.h"
#else
#include "imgui_impl_opengl3.h"
#endif
#include "brush_alpha.h"
#include "scene_snapshot.h"
#include "object_xform.h"

void app_render_frame(App& a, int win_w, int win_h, bool imgui_wants_mouse) {
    auto& window = a.window;
    auto& input = a.input;
    auto& scene = a.scene;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& camera = a.camera;
    auto& renderer = a.renderer;
    auto& text = a.text;
    auto& alpha_lib = a.alpha_lib;
    auto& vmerge_job = a.vmerge_job;
    auto& app_state = a.app_state;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto& current_project_path = a.current_project_path;
    auto& error_popup_msg = a.error_popup_msg;
    auto& error_popup_trigger = a.error_popup_trigger;
    auto& fps_display = a.fps_display;
#ifdef CHISEL_BACKEND_WEBGPU
    auto& fbw = a.fbw;
    auto& fbh = a.fbh;
#endif
    // ---- Cursor visibility ----
    // Retopo counts as an edit mode here: it shows the brush ring (the band width).
    // UV has no tools, so it keeps the OS pointer.
    const bool retopo_mode = input.retopo_tools_on();
    bool non_edit_mode = input.interaction_mode != InputState::InteractionMode::EDIT
                      && !retopo_mode;
    // Hide the OS cursor while sculpting (we draw our own brush cursor).
    bool show_os_cursor = input.quit_requested || input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.remesh_confirm_pending || input.voxel_merge_confirm_pending || input.drop_confirm_pending || input.reset_confirm_pending || input.pose_clear_confirm_pending || input.help_popup_open || input.keybind_menu_open || input.brush_props_open || imgui_wants_mouse || non_edit_mode;
#ifndef __EMSCRIPTEN__
    // A slider drag owns the cursor mode while it runs: it holds GLFW_CURSOR_DISABLED
    // to capture the pointer. This line ran unconditionally every frame and put the
    // mode straight back to NORMAL/HIDDEN, which destroys the compositor's pointer
    // lock ~50us after the drag created it. Two owners of one piece of state, and
    // this one won sixty times a second — so the capture never actually held, the
    // pointer really did travel, and glfwSetCursorPos was refused for want of a lock.
    if (input.slider_mode == InputState::SliderMode::NONE)
        glfwSetInputMode(window, GLFW_CURSOR, show_os_cursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
#else
    // Emscripten's GLFW has no GLFW_CURSOR_HIDDEN (warns every frame) —
    // drive the canvas CSS cursor instead, only on state change.
    static bool css_cursor_hidden = false;
    if (css_cursor_hidden == show_os_cursor) {
        css_cursor_hidden = !show_os_cursor;
        EM_ASM({ Module['canvas'].style.cursor = $0 ? 'none' : 'auto'; }, css_cursor_hidden);
    }
#endif

    // ---- Render ----
#ifdef CHISEL_BACKEND_WEBGPU
    // Acquire this frame's swapchain colour view and inject it as the default
    // render target; default-screen passes render into it (see webgpu_backend).
    // The colour/depth CLEAR happens via the first pass's loadOp=Clear, owned by
    // renderer.draw_background (the GL build instead clears explicitly below).
    WGPUSurfaceTexture surfTex = {};
    wgpuSurfaceGetCurrentTexture(g_surface, &surfTex);
#ifndef __EMSCRIPTEN__
    // An outdated/lost surface (the window changed under the swapchain) hands back
    // no texture and must be reconfigured before it can present again; those are
    // logged.
    //
    // "Suboptimal" gets ONE reconfigure per window size, on the next frame. Intel's
    // Windows Vulkan driver reports it on nearly every frame after any resize, so it
    // isn't logged and isn't chased beyond that one retry — but the retry matters:
    // without it, a window restored (or created) hanging under the taskbar shows
    // black triangles while orbiting/sculpting; with it, it doesn't (user-tested
    // both ways). Presumably the reconfigure made at the resize lands while the
    // window is still mid-transition, and the second one sees it settled.
    {
        static int subopt_w = -1, subopt_h = -1;
        if (surfTex.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal &&
            (subopt_w != win_w || subopt_h != win_h)) {
            subopt_w = win_w; subopt_h = win_h;
            g_surface_reconfigure = true;
        }
        const bool lost = surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
                          surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Lost;
        if (lost || surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Timeout ||
            surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Error)
            std::printf("[win] surface status %d (window %dx%d, configured %dx%d)\n",
                        (int)surfTex.status, win_w, win_h, fbw, fbh);
        if (!surfTex.texture && lost) {
            fbw = win_w; fbh = win_h;
            configureSurface(fbw, fbh);
            makeDepth(fbw, fbh);
            screen_buffers_dirty = true;
            wgpuSurfaceGetCurrentTexture(g_surface, &surfTex);
        }
    }
#endif
    WGPUTextureView frameView = surfTex.texture
        ? wgpuTextureCreateView(surfTex.texture, nullptr) : nullptr;
    bool spare_frame = false;
#ifndef __EMSCRIPTEN__
    // No texture this frame: draw into the stand-in and don't present (see
    // spareColorView). Said once per run of missed frames, not every frame.
    {
        static bool missing = false;
        if (!frameView) {
            if (!missing)
                std::printf("[win] no swapchain frame (status %d): drawing offscreen "
                            "until it comes back\n", (int)surfTex.status);
            frameView   = spareColorView(fbw, fbh);
            spare_frame = true;
        }
        missing = spare_frame;
    }
#endif
    gpu::webgpu_set_default_color(frameView);
#else
    // glClear(DEPTH) is gated by the depth write-mask; force it TRUE so the clear
    // can never be silently no-op'd by whatever pipeline drew last (e.g. a HUD
    // pipeline leaving depthMask FALSE). Don't rely on every pipeline upholding it.
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#endif

    // Background gradient
    renderer.draw_background(win_w, win_h);

    // Paint stays visible while the paint brush is active regardless of the
    // toggle, so you can always see what you're laying down.
    renderer.paint_visible =
        (input.paint_visible || input.current_brush == BrushType::PAINT) ? 1.0f : 0.0f;
    renderer.matcap_contrast = input.matcap_contrast;
    renderer.flat_shading = input.flat_shading ? 1.0f : 0.0f;
    for (int i = 0; i < 3; i++) renderer.unselected_tint[i] = input.unselected_tint[i];

    // Mesh: N draws. The active entity is drawn from the working VAO; every
    // other alive entity from its static display VAO. Depth test composes them.
    // UV mode with a bake to preview: the retopo mesh wearing it stands in for the
    // sculpt it was baked from (the active one, the one retopo works on).
    const BakeState& bake = a.retopo.bake;
    const bool baked = input.uv_on() && bake.showing();
    if (baked) {
        const BakeResult& r = bake.result;
        renderer.upload_baked(r.serial, r.pos.data(), r.nrm.data(), r.tan.data(), r.uv.data(), r.corners(),
                              r.res, r.has_color ? r.color_rgba.data() : nullptr,
                              r.has_normal ? r.normal_rgba.data() : nullptr);
        renderer.draw_baked(camera, win_w, win_h, r.has_normal, r.has_color && renderer.paint_visible > 0.5f,
                            r.flip_green);
    }
    {
        const std::vector<uint32_t>& sel = scene.selected_ids();
        uint32_t active_id = scene.active_mesh_id();
        for (auto& up : scene.entities()) {
            if (!up || !up->alive) continue;
            // An entity is tinted (deselected) unless it's in the selection set.
            // An empty set is a real, reachable state (the user deselected
            // everything) → every entity tinted: a fully-deselected scene.
            bool selected = false;
            for (uint32_t id : sel) if (id == up->id) { selected = true; break; }
            if (up->id == active_id && baked) continue;
            if (up->id == active_id)
                renderer.draw_mesh(camera, win_w, win_h,
                                   (uint32_t)scene.active_mesh().indices.size(),
                                   selected);
            else
                renderer.draw_display(camera, up->gpu, win_w, win_h, selected);
        }
    }

    // Debug mesh overlay (Y key toggle). Invalidate cached edge buffer on
    // every transition so the next "on" rebuilds against current topology.
    static bool prev_show_debug_mesh = false;
    if (input.show_debug_mesh != prev_show_debug_mesh) {
        renderer.invalidate_debug_mesh();
        prev_show_debug_mesh = input.show_debug_mesh;
    }
    if (input.show_debug_mesh) {
        renderer.draw_debug_mesh(camera, *mesh, win_w, win_h);
    }

    // Pose rig: shown only while the pose brush is in hand. The reach ring is the
    // farthest a bone's skin envelope may extend from it: two sphere radii (pose_bind).
    if (input.pose_brush_on() && !non_edit_mode) {
        draw_pose_rig(scene.active_entity().pose, camera, win_w, win_h,
                      input.pose_hover_joint, 1.0f,
                      input.ctrl_held && app_state == AppState::IDLE);
    }

    // Brush cursor — draw at locked position during slider, normal position otherwise
    if (!input.quit_requested && !input.export_dialog_active && !input.import_dialog_active && !input.save_dialog_active && !input.remesh_confirm_pending && !input.voxel_merge_confirm_pending && !input.drop_confirm_pending && !input.reset_confirm_pending && !input.pose_clear_confirm_pending && !input.help_popup_open && !input.keybind_menu_open && !input.brush_props_open && !imgui_wants_mouse && !non_edit_mode) {
        float cursor_x, cursor_y;
        if (input.slider_mode != InputState::SliderMode::NONE) {
            cursor_x = (float)input.slider_start_x;
            cursor_y = (float)input.slider_start_y;
        } else {
            cursor_x = (float)input.mouse_x;
            cursor_y = (float)input.mouse_y;
        }
        if (retopo_mode) {
            draw_retopo_cursor(renderer, camera, input, cursor_x, cursor_y, win_w, win_h);
        } else if (input.color_pick_active) {
            // Armed colour picker: eyedropper glyph instead of the brush
            // ring — the ring's size/hardness reading is meaningless here.
            draw_pick_cursor(cursor_x, cursor_y, input.on_model);
        } else if (input.pose_brush_on() && !input.is_smooth_active()) {
            draw_pose_cursor(cursor_x, cursor_y,
                             input.on_model || input.pose_hover_joint >= 0);
        } else if (input.bucket_active && input.current_brush == BrushType::PAINT
                   && !input.is_smooth_active()) {
            if (input.bucket_dragging) {
                draw_gradient_drag(input.bucket_ax, input.bucket_ay, cursor_x, cursor_y);
            } else {
                // A tiling pattern repeats once across the brush circle, and the
                // mask feather (Hardness below 1) is up to a radius wide: show it.
                if ((input.fill_pattern != kFillSolid && input.fill_pattern != kFillGradient)
                    || input.brush_hardness < 0.99f)
                    renderer.draw_cursor(camera, cursor_x, cursor_y, input.brush_size,
                                         input.cursor_nx, input.cursor_ny, input.cursor_nz,
                                         input.brush_hardness, input.brush_strength,
                                         win_w, win_h, input.on_model);
                draw_bucket_cursor(cursor_x, cursor_y, input.on_model);
            }
        } else {
            renderer.draw_cursor(
                camera,
                cursor_x, cursor_y,
                input.brush_size,
                input.cursor_nx, input.cursor_ny, input.cursor_nz,
                input.brush_hardness, input.brush_strength,
                win_w, win_h, input.on_model);

            // Symmetry lobe marker. Where the second dab lands is a WORLD
            // fact (reflect the anchor through x=0, same as every brush's
            // anchor_b), not a screen one — mirroring the cursor pixel about
            // the window centre would only be right for a centred, axis-on
            // camera. So unproject the pen through the cached depth plane,
            // flip X, project back. Off-model there is no anchor to reflect.
            if (input.mirror_x && input.on_model) {
                float hit_depth = 0.0f;
                if (renderer.sample_depth_bilinear(cursor_x, cursor_y, &hit_depth, 0.0f)) {
                    Vec3 cam_pos = camera.get_position();
                    Vec3 fwd = (camera.target - cam_pos).normalized();
                    Vec3 wup = {0, 1, 0};
                    Vec3 right = fwd.cross(wup).normalized();
                    Vec3 up = right.cross(fwd).normalized();
                    float half_h = camera.half_height_at(hit_depth);
                    float half_w = half_h * ((float)win_w / (float)win_h);
                    float ndc_x = 2.0f * cursor_x / (float)win_w - 1.0f;
                    float ndc_y = 1.0f - 2.0f * cursor_y / (float)win_h;
                    Vec3 anchor = cam_pos + fwd * hit_depth
                                + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                    anchor.x = -anchor.x;
                    float mx, my;
                    if (camera.world_to_screen(anchor, win_w, win_h, mx, my)) {
                        // Occlusion: the reflected anchor sits on the mesh's far
                        // side whenever it is deeper than whatever the depth plane
                        // already holds at that pixel. Fade rather than hide — you
                        // still want to know the lobe is there, just not to mistake
                        // it for something in front. The tolerance is a few pixels'
                        // worth of world depth, so a point on its OWN near-side
                        // surface (where the two depths agree to within the plane's
                        // own quantisation) never counts as hidden.
                        float alpha = 1.0f;
                        float front_depth = 0.0f;
                        if (renderer.sample_depth((int)mx, (int)my, &front_depth)) {
                            float world_per_pixel = (2.0f * half_h) / (float)win_h;
                            float tol = world_per_pixel * 6.0f;
                            if (camera.view_depth(anchor) > front_depth + tol)
                                alpha = 0.30f;
                        }
                        renderer.draw_mirror_cursor(mx, my, input.brush_hardness,
                                                    alpha, win_w, win_h);
                    }
                }
            }
        }
    }

    // ---- Overlays ----
    // Remesh and SDF-merge prompts: what they need from the scene is gathered once
    // here. The classic skin draws them as bitmap text right below; every other
    // skin draws real dialogs (draw_skin_dialogs, after the skin's UI).
    DialogInfo dlg;
    if (input.remesh_confirm_pending || input.voxel_merge_confirm_pending)
        dlg.levels_to_bake = snapshot_levels_to_bake(scene);
    if (input.voxel_merge_confirm_pending) {
        dlg.n_selected = (int)scene.selected_ids().size();
        // Count unselected (red) committed entities — candidate cutters for subtract.
        for (const auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview) continue;
            bool sel = false;
            for (uint32_t sid : scene.selected_ids())
                if (sid == up->id) { sel = true; break; }
            if (!sel) dlg.n_unselected++;
        }
        // Density field on any selected mesh → the merge output will carry
        // the heatmap; offer the chained adaptive remesh line.
        for (uint32_t sid : scene.selected_ids()) {
            const MeshEntity* e = scene.find_entity(sid);
            if (e && e->alive && !e->mesh.density.empty()) { dlg.has_density = true; break; }
        }
    }
    if (vmerge_job) dlg.merge_progress = voxel_merge_progress(*vmerge_job);
    const bool bitmap_dialogs = input.ui_classic();

    if (bitmap_dialogs && input.quit_requested)
        draw_quit_dialog(text, win_w, win_h);
    if (input.drop_confirm_pending)
        draw_drop_confirm(text, input.drop_path, win_w, win_h);
    if (bitmap_dialogs && input.remesh_confirm_pending)
        draw_remesh_confirm(text, dlg.levels_to_bake, win_w, win_h);
    if (input.drop_level_confirm_pending)
        draw_drop_level_confirm(text, multires->base_level + (int)multires->disp.size(),
                                win_w, win_h);
    if (bitmap_dialogs && input.reset_confirm_pending)
        draw_reset_confirm(text, win_w, win_h);
    if (bitmap_dialogs && input.pose_clear_confirm_pending)
        draw_pose_clear_confirm(text, win_w, win_h);
    if (bitmap_dialogs && input.voxel_merge_confirm_pending)
        draw_voxel_merge_confirm(text, input.voxel_merge_resolution,
                                 dlg.n_selected, dlg.n_unselected,
                                 input.voxel_merge_surface_nets,
                                 dlg.has_density, input.voxel_merge_adaptive,
                                 dlg.levels_to_bake, win_w, win_h);
    if (bitmap_dialogs && input.remesh_in_progress)
        draw_remesh_progress(text, win_w, win_h);
    if (bitmap_dialogs && input.voxel_merge_in_progress)
        draw_voxel_merge_progress(text, win_w, win_h, dlg.merge_progress);
    // The bitmap HUD belongs to the classic skin; the modern skin draws its own
    // readouts, toast and slider in draw_modern_ui below.
    if (input.ui_classic()) {
        if (input.toolbar_visible)
            draw_toolbar(text, input, mesh->tri_count(), mesh->vertex_count(), CHISEL_VERSION,
                         multires->current_level,
                         current_project_path.c_str(), win_w, win_h);
        if (input.slider_mode != InputState::SliderMode::NONE)
            draw_slider(text, input, win_w, win_h);
        if (input.interaction_mode == InputState::InteractionMode::SELECT)
            draw_mode_indicator(text, "SELECT", win_w, win_h);
        else if (input.interaction_mode == InputState::InteractionMode::INSERT)
            draw_mode_indicator(text, "INSERT", win_w, win_h);
        draw_notification(text, input, win_w, win_h);
        draw_mirror_unavailable(text, input, win_w, win_h);
        if (input.show_fps)
            draw_fps(text, fps_display, win_w, win_h);
    }
    debug_console::draw(text, win_w, win_h);

    app_files_frame(a);   // app_files.cpp: open/import, dev hooks, save, export, dialogs

    // ---- Button islands (brush selection + ops) ----
    MultiresInfo mres_info;
    mres_info.locked     = multires->locked;
    mres_info.base_level = multires->base_level;
    mres_info.lmax       = multires->base_level + (int)multires->disp.size();
    if (input.ui_classic()) {
        draw_button_islands(input, win_w, win_h, &alpha_lib, mres_info);
    } else {
        // Frosted glass: blur what has been drawn so far (the scene, cursor and any
        // bitmap dialog) for the panels to sit on. Only the ImGui pass is still to
        // come, so the copy is exactly what is behind them.
        float blur_px = 0.0f, saturate = 1.0f;
        ui_skin_backdrop_params(input, &blur_px, &saturate);
#ifdef CHISEL_BACKEND_WEBGPU
        ui_backdrop::capture(win_w, win_h, blur_px, saturate, surfTex.texture);
#else
        ui_backdrop::capture(win_w, win_h, blur_px, saturate, nullptr);
#endif
        SkinStats st;
        st.version = CHISEL_VERSION;
        st.tris = mesh->tri_count();
        st.verts = mesh->vertex_count();
        st.level = multires->current_level;
        st.fps = fps_display;
        st.project_path = current_project_path.c_str();
        draw_modern_ui(input, win_w, win_h, &alpha_lib, mres_info, st);
        draw_skin_dialogs(input, win_w, win_h, dlg);
    }

    // Key bindings window (burger menu). Top level, so every skin shares it.
    draw_keybind_window(input, win_w, win_h);
    draw_brush_props_window(input, win_w, win_h);
    draw_paint_layers_panel(input, scene, win_w, win_h);
    if (input.interaction_mode == InputState::InteractionMode::RETOPO)
        draw_retopo_ui(input, a.retopo, renderer, camera, win_w, win_h, screen_buffers_dirty);

    // ---- Error popup ----
    if (error_popup_trigger) {
        ImGui::OpenPopup("Error");
        error_popup_trigger = false;
    }
    if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(error_popup_msg.c_str());
        if (ImGui::Button("OK", ImVec2(120, 0)))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::Render();
#ifdef CHISEL_BACKEND_WEBGPU
    // UI pass: load the scene already drawn into the swapchain view, draw ImGui on
    // top (no depth), then present.
    if (frameView) {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
        WGPURenderPassColorAttachment uiColor = {};
        uiColor.view       = frameView;
        uiColor.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        uiColor.loadOp     = WGPULoadOp_Load;
        uiColor.storeOp    = WGPUStoreOp_Store;
        WGPURenderPassDescriptor uiRp = {};
        uiRp.colorAttachmentCount = 1;
        uiRp.colorAttachments = &uiColor;
        WGPURenderPassEncoder uiPass = wgpuCommandEncoderBeginRenderPass(enc, &uiRp);
        ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), uiPass);
        wgpuRenderPassEncoderEnd(uiPass);
        wgpuRenderPassEncoderRelease(uiPass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(wgpuDeviceGetQueue(g_device), 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }
#ifndef __EMSCRIPTEN__
    // Native (wgpu-native) presents explicitly. On the web the browser composites
    // the canvas automatically when the requestAnimationFrame callback returns, and
    // emdawnwebgpu aborts if wgpuSurfacePresent is called at all — so skip it.
    if (surfTex.texture) wgpuSurfacePresent(g_surface);   // nothing acquired, nothing to show
#endif
    gpu::webgpu_set_default_color(nullptr);
    if (frameView && !spare_frame) wgpuTextureViewRelease(frameView);
    if (surfTex.texture) wgpuTextureRelease(surfTex.texture);
#else
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
#endif
}
