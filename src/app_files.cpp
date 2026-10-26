// The frame's file work, moved out of main.cpp as is: open/import a path, the dev
// auto-import hooks, save / save-as / incremental save, export, and their dialogs
// (native: the ImGui file browser; web: name prompts, browser download, file picker).
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#include "app.h"
#include "web_bridge.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <string>

#include "mesh.h"
#include "camera.h"
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
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "brush_alpha.h"
#include "project_file.h"
#include "scene_snapshot.h"
#include "object_xform.h"
#include <filesystem>

#ifndef __EMSCRIPTEN__
// Native file dialogs (Open, Save, Export, Load Brush Alpha) open centred on the
// screen rather than wherever ImGui last left them (often under the button that
// opened them). Appearing only, so the user can still drag one aside. The position
// is set only when this key's dialog is up: otherwise it would land on whatever
// window begins next.
static bool display_file_dialog(IGFD::FileDialog* fd, const char* key) {
    if (fd->IsOpened(key))
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
    return fd->Display(key, ImGuiWindowFlags_NoCollapse, ImVec2(600, 400));
}
#endif

void app_files_frame(App& a) {
    auto& window = a.window;
    auto& input = a.input;
    auto& scene = a.scene;
    auto& mesh = a.mesh;
    auto& multires = a.multires;
    auto& camera = a.camera;
    auto& compute = a.compute;
    auto& brush_stroke = a.brush_stroke;
    auto& alpha_lib = a.alpha_lib;
    auto& xforms = a.xforms;
    auto& vmerge_job = a.vmerge_job;
    auto& app_state = a.app_state;
    auto& mesh_center = a.mesh_center;
    auto& mesh_radius = a.mesh_radius;
    auto& screen_buffers_dirty = a.screen_buffers_dirty;
    auto& clean_edit_mark = a.clean_edit_mark;
    auto& cli_open_path = a.cli_open_path;
    auto& default_browse_path = a.default_browse_path;
    auto& current_project_path = a.current_project_path;
    auto& error_popup_msg = a.error_popup_msg;
    auto& error_popup_trigger = a.error_popup_trigger;
#ifdef __EMSCRIPTEN__
    auto& web_save_name = a.web_save_name;
    auto& web_export_name = a.web_export_name;
#endif
    auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(a); };

    // ---- Open/import a path (shared by the native dialog and the web picker) ----
    auto do_import_path = [&](const std::string& path) {
        auto dot = path.find_last_of('.');
        std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
        for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);

        if (ext == "chisel") {
            ProjectData proj;
            LoadResult lr = load_project(path.c_str(), proj);
            if (lr == LoadResult::OK && !proj.entities.empty()) {
                camera = proj.camera;
                input.mirror_x = proj.mirror_x;
                input.subdiv_level = proj.subdiv_level;
                input.mesh_locked = true;
                current_project_path = path;

                // Rebuild the whole multimesh scene. load_entities does
                // per-entity cascade/adjacency/normals/mirror and restores
                // the saved selection; the active entity's mirror map is
                // (re)cached by refresh_mirror_map before sync.
                scene.set_mirror_topology(proj.mirror_use_topology);
                scene.load_entities(proj.entities, proj.active_id,
                                    proj.selected_ids, proj.next_id);
                xforms.clear();      // records point at the outgoing scene
                scene.refresh_mirror_map(input.subdiv_level);
                scene.sync();
                clean_edit_mark = scene.edit_mark();

                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                screen_buffers_dirty = true;

                if (scene.load_flattened() > 0) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Loaded with %d mesh(es) flattened: file is from "
                                  "another platform — re-save there to migrate",
                                  scene.load_flattened());
                    input.notification_timer = 6.0f;
                } else {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Loaded: %.400s (%zu mesh%s, active %u v, %u t)",
                                  path.c_str(), proj.entities.size(),
                                  proj.entities.size() == 1 ? "" : "es",
                                  mesh->vertex_count(), mesh->tri_count());
                    input.notification_timer = 3.0f;
                }
            } else {
                error_popup_msg = std::string("Load failed: ") + result_string(lr) + "\n" + path;
                error_popup_trigger = true;
            }
        } else {
            Mesh loaded;
            bool imported = (ext == "ply") ? Mesh::import_ply(path.c_str(), loaded)
                                           : Mesh::import_obj(path.c_str(), loaded);
            if (imported && input.import_append) {
                // Append: add the mesh as a NEW scene entity at its authored
                // scale, leaving existing entities, the camera, and their undo
                // untouched. Same commit path insert uses (add_preview →
                // commit_preview → multires/mirror init).
                loaded.mask.clear();
                uint32_t new_id = scene.add_preview(loaded, 0);
                scene.commit_preview(new_id);
                multires_stack_init_from_lock(scene.active_multires(),
                                              scene.active_mesh(), 0);
                scene.set_mirror_topology(false);
                scene.refresh_mirror_map();
                scene.sync();

                input.mesh_locked = true;
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                screen_buffers_dirty = true;

                std::snprintf(input.notification, sizeof(input.notification),
                              "Appended: %.400s (%u v, %u t)",
                              path.c_str(), mesh->vertex_count(), mesh->tri_count());
                input.notification_timer = 3.0f;
            } else if (imported) {
                *mesh = std::move(loaded);
                mesh->mask.clear();

                scene.set_mirror_topology(false);
                // build_adjacency FIRST: it bumps topo_version, and
                // refresh_mirror_map stamps mirror_topo_version with whatever
                // topo_version reads at the time. Refreshing first left the
                // stamp permanently one behind, so the guard in
                // refresh_mirror_map never short-circuited again and every later
                // call rebuilt the map from *sculpted* positions — exactly the
                // reclassification that guard exists to prevent.
                mesh->build_adjacency();
                scene.refresh_mirror_map();

                input.mesh_locked = true;
                multires_stack_init_from_lock(*multires, *mesh, 0);
                scene.reset_to_single_mesh(0);
                scene.sync();
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                camera.set_target(mesh_center);
                camera.distance = mesh_radius * 2.5f;
                current_project_path.clear();

                scene.active_undo().clear(&compute);
                xforms.clear();
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                screen_buffers_dirty = true;
                clean_edit_mark = scene.edit_mark();   // the imported file still holds it

                std::snprintf(input.notification, sizeof(input.notification),
                              "Imported: %.400s (%u v, %u t)",
                              path.c_str(), mesh->vertex_count(), mesh->tri_count());
                input.notification_timer = 3.0f;
            } else {
                error_popup_msg = "Import failed: " + path;
                error_popup_trigger = true;
            }
        }
    };

    // Dev hook: CHISEL_AUTO_IMPORT=<path> imports a file on the first frame
    // (headless load-path testing without driving the UI). No-op when unset.
    // A bare path on the command line ("Open With" / chisel foo.chisel)
    // rides the same first-frame import; CLI path wins over the env hook.
    {
        static bool auto_import_done = false;
        if (!auto_import_done) {
            auto_import_done = true;
            const char* auto_path = cli_open_path.empty()
                                  ? std::getenv("CHISEL_AUTO_IMPORT")
                                  : cli_open_path.c_str();
            if (auto_path) {
                if (FILE* tf = std::fopen(auto_path, "rb")) {
                    std::fclose(tf);
                    std::printf("[auto-import] loading %s\n", auto_path);
                    do_import_path(auto_path);
                } else {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Cannot open: %.400s", auto_path);
                    input.notification_timer = 4.0f;
                }
            }
        }
    }

    // Dev hook: CHISEL_AUTO_SUBD=<n> requests one level-up per frame for the
    // first n frames — drives the subdivision guard without the UI (pair with
    // CHISEL_LIMITS_MB to watch it refuse). No-op when unset.
    {
        static int auto_subd_left = -1;
        if (auto_subd_left < 0) {
            const char* env = std::getenv("CHISEL_AUTO_SUBD");
            auto_subd_left = env ? std::atoi(env) : 0;
        }
        if (auto_subd_left > 0) {
            --auto_subd_left;
            input.level_switch_delta = +1;
        }
    }

    // Dev hook: CHISEL_REMESH_KEEP_DETAIL=0 starts with the Settings box
    // "Remesh keeps detail" unticked, so a scripted remesh can be run both
    // ways on the same file and the two DRIFT lines compared.
    {
        static bool keep_detail_env_done = false;
        if (!keep_detail_env_done) {
            keep_detail_env_done = true;
            if (const char* env = std::getenv("CHISEL_REMESH_KEEP_DETAIL"))
                input.remesh_keep_detail = std::atoi(env) != 0;
            if (const char* env = std::getenv("CHISEL_REMESH_DETAIL"))
                input.remesh_detail = std::min(4.0f, std::max(0.25f, (float)std::atof(env)));
            // SDF remesh knobs for CHISEL_AUTO_MERGE runs.
            if (const char* env = std::getenv("CHISEL_MERGE_RES"))
                input.voxel_merge_resolution = std::min(256, std::max(64, std::atoi(env)));
            if (const char* env = std::getenv("CHISEL_MERGE_MIRROR"))
                input.voxel_merge_mirror = std::atoi(env) != 0;
        }
    }

    // Dev hook: CHISEL_AUTO_REMESH=1 fires one plain remesh (no density
    // paint) once the scene has settled — pairs with CHISEL_AUTO_IMPORT and
    // the two env knobs above to A/B a file headless. No-op when unset.
    {
        static int auto_remesh_frames = -1;
        if (auto_remesh_frames < 0) {
            const char* env = std::getenv("CHISEL_AUTO_REMESH");
            auto_remesh_frames = (env && std::atoi(env) != 0) ? 1 : 0;
        }
        if (auto_remesh_frames > 0 && ++auto_remesh_frames > 12 &&
            app_state == AppState::IDLE && !input.remesh_in_progress) {
            input.remesh_requested = true;
            auto_remesh_frames = 0;
        }
    }

    // Dev hook: CHISEL_AUTO_MERGE=1 fires one SDF remesh of the selection the
    // same way, so CHISEL_REMESH_KEEP_DETAIL=0/1 A/Bs the merge's snap too.
    {
        static int auto_merge_frames = -1;
        if (auto_merge_frames < 0) {
            const char* env = std::getenv("CHISEL_AUTO_MERGE");
            auto_merge_frames = (env && std::atoi(env) != 0) ? 1 : 0;
        }
        if (auto_merge_frames > 0 && ++auto_merge_frames > 12 &&
            app_state == AppState::IDLE && !input.voxel_merge_in_progress) {
            input.voxel_merge_requested = true;
            auto_merge_frames = 0;
        }
    }

    // Dev hook: CHISEL_AUTO_DENSITY_REMESH=1 paints a hard half-and-half
    // density field on the active mesh (y above centroid ⇒ red 1.0, below
    // ⇒ green 0.0) and fires one remesh — drives the adaptive remesher
    // headless, then prints per-bucket edge-length means so the split can
    // be verified from the log (red mean ≈ fine_mult/coarse_mult × green
    // mean). CHISEL_AUTO_DENSITY_MERGE=1 paints the same field but fires a
    // voxel merge instead (mirror MC, default R), then prints the merged
    // mesh's mean density above/below the centroid — verifies the
    // nearest-source carry spatially (above ≈ 1, below ≈ 0). No-op when
    // both are unset.
    {
        static int  auto_density_state = -1;
        static int  auto_density_warmup = 0;
        static bool auto_density_merge = false;
        if (auto_density_state < 0) {
            const char* env = std::getenv("CHISEL_AUTO_DENSITY_REMESH");
            auto_density_state = (env && std::atoi(env) != 0) ? 1 : 0;
            const char* menv = std::getenv("CHISEL_AUTO_DENSITY_MERGE");
            if (menv && std::atoi(menv) != 0) {
                auto_density_state = 1;
                auto_density_merge = true;
            }
        }
        // Warmup lets a CHISEL_AUTO_SUBD ladder finish first (composable:
        // subdivide N levels, then paint + adaptive-remesh the result).
        if (auto_density_state == 1 && ++auto_density_warmup > 10 &&
            app_state == AppState::IDLE &&
            input.level_switch_delta == 0 && !input.remesh_in_progress &&
            !input.voxel_merge_in_progress && !vmerge_job) {
            const uint32_t vc = mesh->vertex_count();
            double cy = 0.0;
            for (uint32_t v = 0; v < vc; v++) cy += mesh->pos_y[v];
            cy = (vc > 0) ? cy / vc : 0.0;
            mesh->density.assign(vc, 0.0f);
            for (uint32_t v = 0; v < vc; v++)
                if (mesh->pos_y[v] > cy) mesh->density[v] = 1.0f;
            std::printf("[auto-density] painted half/half field (centroid y=%.4f), requesting %s\n",
                        cy, auto_density_merge ? "voxel merge" : "remesh");
            if (auto_density_merge) {
                input.voxel_merge_mirror       = true;
                input.voxel_merge_subtract     = false;
                input.voxel_merge_surface_nets = false;
                input.voxel_merge_requested    = true;
            } else {
                input.remesh_requested = true;
            }
            auto_density_state = 2;
        } else if (auto_density_state == 2 && auto_density_merge &&
                   !input.voxel_merge_requested && !input.voxel_merge_in_progress) {
            const uint32_t vc = mesh->vertex_count();
            const uint32_t dc = (uint32_t)mesh->density.size();
            double cy = 0.0;
            for (uint32_t v = 0; v < vc; v++) cy += mesh->pos_y[v];
            cy = (vc > 0) ? cy / vc : 0.0;
            double sum[2] = {0, 0}; uint64_t cnt[2] = {0, 0}; uint64_t mid = 0;
            for (uint32_t v = 0; v < vc && v < dc; v++) {
                int b = (mesh->pos_y[v] > cy) ? 1 : 0;
                sum[b] += mesh->density[v]; cnt[b]++;
                if (mesh->density[v] > 0.25f && mesh->density[v] < 0.75f) mid++;
            }
            std::printf("[auto-density] post-merge density means: "
                        "above=%.3f (n=%llu) below=%.3f (n=%llu) mid-band=%llu, field %u/%u verts\n",
                        cnt[1] ? sum[1]/cnt[1] : -1.0, (unsigned long long)cnt[1],
                        cnt[0] ? sum[0]/cnt[0] : -1.0, (unsigned long long)cnt[0],
                        (unsigned long long)mid, dc, vc);
            auto_density_state = 3;
        } else if (auto_density_state == 2 && !auto_density_merge &&
                   !input.remesh_requested && !input.remesh_in_progress) {
            double sum[3] = {0, 0, 0};
            uint64_t cnt[3] = {0, 0, 0};
            const uint32_t tc = mesh->tri_count();
            const uint32_t dc = (uint32_t)mesh->density.size();
            for (uint32_t t = 0; t < tc; t++) {
                for (int e = 0; e < 3; e++) {
                    uint32_t a = mesh->indices[t*3+e];
                    uint32_t b = mesh->indices[t*3+(e+1)%3];
                    float da = (a < dc) ? mesh->density[a] : 0.5f;
                    float db = (b < dc) ? mesh->density[b] : 0.5f;
                    float d = 0.5f * (da + db);
                    int bucket = (d > 0.75f) ? 2 : (d < 0.25f) ? 0 : 1;
                    float dx = mesh->pos_x[a] - mesh->pos_x[b];
                    float dy = mesh->pos_y[a] - mesh->pos_y[b];
                    float dz = mesh->pos_z[a] - mesh->pos_z[b];
                    sum[bucket] += std::sqrt(dx*dx + dy*dy + dz*dz);
                    cnt[bucket]++;
                }
            }
            std::printf("[auto-density] post-remesh edge means: "
                        "green=%.5f (n=%llu) mid=%.5f (n=%llu) red=%.5f (n=%llu), green/red=%.2f\n",
                        cnt[0] ? sum[0]/cnt[0] : 0.0, (unsigned long long)cnt[0],
                        cnt[1] ? sum[1]/cnt[1] : 0.0, (unsigned long long)cnt[1],
                        cnt[2] ? sum[2]/cnt[2] : 0.0, (unsigned long long)cnt[2],
                        (cnt[0] && cnt[2] && sum[2] > 0.0)
                            ? (sum[0]/cnt[0]) / (sum[2]/cnt[2]) : 0.0);
            auto_density_state = 3;
        }
    }

#ifndef __EMSCRIPTEN__
    // ---- ImGui file dialogs (native: browse the real filesystem) ----
    auto* fd = IGFD::FileDialog::Instance();

    if (input.export_dialog_active && !fd->IsOpened("ExportKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = default_browse_path;
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
        fd->OpenDialog("ExportKey", "Export Mesh", ".obj,.stl,.ply", cfg);
    }
    if (input.import_dialog_active && !fd->IsOpened("ImportKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = default_browse_path;
        cfg.flags = ImGuiFileDialogFlags_None;
        cfg.sidePaneWidth = 190.0f;
        cfg.sidePane = [&](const char*, IGFD::UserDatas, bool*) {
            ImGui::Checkbox("Append to scene", &input.import_append);
            ImGui::Spacing();
            ImGui::TextWrapped("On: add the mesh as a new object at its "
                               "own scale (OBJ/PLY only). Off: replace the "
                               "scene. .chisel projects always replace.");
        };
        fd->OpenDialog("ImportKey", "Open File", ".chisel,.obj,.ply", cfg);
    }

    if (display_file_dialog(fd, "ExportKey")) {
        if (fd->IsOk()) {
            std::string path = fd->GetFilePathName();
            auto dot = path.find_last_of('.');
            std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
            for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
            // GPU-resident sculpting/undo leave mesh.pos stale on the CPU —
            // pull the live surface back first (save does the same).
            scene.materialize_active_cpu();
            bool ok = (ext == "stl") ? mesh->export_stl(path.c_str())
                    : (ext == "ply") ? mesh->export_ply(path.c_str())
                                     : mesh->export_obj(path.c_str());
            if (ok) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Exported: %.400s", path.c_str());
                input.notification_timer = 3.0f;
            } else {
                error_popup_msg = "Export failed: " + path;
                error_popup_trigger = true;
            }
        }
        fd->Close();
        input.export_dialog_active = false;
    }
    if (!input.export_dialog_active && fd->IsOpened("ExportKey"))
        fd->Close();

    if (display_file_dialog(fd, "ImportKey")) {
        if (fd->IsOk())
            do_import_path(fd->GetFilePathName());
        fd->Close();
        input.import_dialog_active = false;
    }
    if (!input.import_dialog_active && fd->IsOpened("ImportKey"))
        fd->Close();

    // Brush-alpha custom image loader (mirrors ImportKey). Loads a grayscale
    // bitmap as a new pool entry and selects it.
    if (input.load_alpha_dialog_active && !fd->IsOpened("AlphaKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = default_browse_path;
        cfg.flags = ImGuiFileDialogFlags_None;
        fd->OpenDialog("AlphaKey", "Load Brush Alpha", ".png,.jpg,.jpeg,.tga,.bmp", cfg);
    }
    if (display_file_dialog(fd, "AlphaKey")) {
        if (fd->IsOk()) {
            int idx = alpha_lib.load_custom(fd->GetFilePathName().c_str());
            if (idx > 0) {
                input.active_alpha = idx;   // upload-on-change picks it up next frame
            } else {
                error_popup_msg = "Failed to load alpha image: " + fd->GetFilePathName();
                error_popup_trigger = true;
            }
        }
        fd->Close();
        input.load_alpha_dialog_active = false;
    }
    if (!input.load_alpha_dialog_active && fd->IsOpened("AlphaKey"))
        fd->Close();
#endif // !__EMSCRIPTEN__

    // ---- Save project (.chisel) ----
    auto do_save_project = [&](const std::string& path) {
        scene.materialize_active_cpu();  // 2b: copies the active entity's mesh + multires to disk
        ProjectData proj;
        // Persist the whole multimesh scene: every alive, committed entity
        // (skip transient INSERT previews) with its own mesh + multires.
        for (auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview) continue;
            EntityRecord rec;
            rec.id           = up->id;
            rec.subdiv_level = up->subdiv_level;
            rec.mesh         = up->mesh;
            rec.multires     = up->multires;
            proj.entities.push_back(std::move(rec));
        }
        proj.active_id           = scene.active_mesh_id();
        proj.selected_ids        = scene.selected_ids();
        proj.next_id             = scene.next_id();
        proj.mirror_use_topology = scene.mirror_topology();
        proj.camera       = camera;
        proj.mirror_x     = input.mirror_x;
        proj.subdiv_level = input.subdiv_level;
        SaveResult sr = save_project(path.c_str(), proj);
        if (sr == SaveResult::OK && input.quit_after_save) {
            input.quit_after_save = false;
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (sr != SaveResult::OK) input.quit_after_save = false;   // stay; the error popup says why
        if (sr == SaveResult::OK) {
            current_project_path = path;
            clean_edit_mark = scene.edit_mark();
#ifdef __EMSCRIPTEN__
            // A MEMFS "save" only counts once the bytes leave as a download
            // (which also unlinks the MEMFS copy; Ctrl+S recreates it).
            web_download_file(path.c_str(),
                              path.substr(path.find_last_of('/') + 1).c_str());
#endif
            std::snprintf(input.notification, sizeof(input.notification),
                          "Saved: %.400s", path.c_str());
            input.notification_timer = 2.0f;
        } else {
            error_popup_msg = std::string("Save failed: ") + result_string(sr) + "\n" + path;
            error_popup_trigger = true;
        }
    };

    // ---- Drag-and-drop open (prompt was answered in the key callback) ----
    if (input.drop_open_requested) {
        input.drop_open_requested = false;
        do_import_path(input.drop_path);
    }

    if (input.save_requested) {
        input.save_requested = false;
        if (current_project_path.empty()) {
            input.save_dialog_active = true;
        } else {
            do_save_project(current_project_path);
        }
    }
    if (input.save_as_requested) {
        input.save_as_requested = false;
        input.save_dialog_active = true;
    }

    // ---- Incremental save ("+"): sculpt.chisel -> sculpt_001.chisel -> _002 ----
    // A trailing _NNN is treated as the counter rather than part of the stem, so
    // repeated presses walk the series instead of nesting suffixes. Existing files
    // are skipped, so an incremental save can never overwrite a previous version
    // (on web nothing persists in MEMFS, so it just counts up from the last name).
    auto next_incremental_path = [&](const std::string& path) -> std::string {
        const std::string ext = ".chisel";
        std::string stem = path;
        if (stem.size() >= ext.size()
            && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
            stem.resize(stem.size() - ext.size());
        // A browser renames a colliding download to "bust_001 (1).chisel", and
        // opening that file makes it the current project. Strip the marker before
        // reading the counter below, or "bust_001 (1)" parses as an un-numbered
        // stem and the next save compounds into "bust_001 (1)_001".
        if (!stem.empty() && stem.back() == ')') {
            size_t op = stem.find_last_of('(');
            if (op != std::string::npos && op + 1 < stem.size() - 1) {
                bool all_digits = true;
                for (size_t i = op + 1; i + 1 < stem.size(); i++)
                    if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
                if (all_digits) {
                    while (op > 0 && stem[op - 1] == ' ') op--;   // Chrome " (1)", Firefox "(1)"
                    stem.resize(op);
                }
            }
        }
        int n = 0;
        size_t us = stem.find_last_of('_');
        if (us != std::string::npos && us + 1 < stem.size()) {
            bool all_digits = true;
            for (size_t i = us + 1; i < stem.size(); i++)
                if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
            if (all_digits) {
                n = std::atoi(stem.c_str() + us + 1);
                stem.resize(us);
            }
        }
        char suffix[16];
        for (int i = n + 1; i <= 9999; i++) {
            std::snprintf(suffix, sizeof(suffix), "_%03d", i);
            std::string cand = stem + suffix + ext;
            std::error_code ec;
            if (!std::filesystem::exists(cand, ec)) return cand;
        }
        return stem + "_9999" + ext;   // series exhausted: overwrite the last one
    };

    if (input.save_incremental_requested) {
        input.save_incremental_requested = false;
        if (current_project_path.empty()) {
            // Nothing to number off yet — ask for a name first, like plain save.
            input.save_dialog_active = true;
        } else {
            do_save_project(next_incremental_path(current_project_path));
        }
    }

#ifndef __EMSCRIPTEN__
    if (input.save_dialog_active && !fd->IsOpened("SaveKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = default_browse_path;
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
        fd->OpenDialog("SaveKey", "Save Project", ".chisel", cfg);
    }

    if (display_file_dialog(fd, "SaveKey")) {
        if (fd->IsOk()) {
            do_save_project(fd->GetFilePathName());
        }
        fd->Close();
        input.save_dialog_active = false;
    }
    if (!input.save_dialog_active && fd->IsOpened("SaveKey"))
        fd->Close();
#else
    // ---- Web dialogs: name prompt + browser download; picker for open ----
    // (same *_dialog_active flags as native, so the sculpt-input and hotkey
    // gating keeps working unchanged)

    // Export: pick a name, a format button writes via the normal exporter
    // into MEMFS, web_download_file hands the bytes to the browser.
    if (input.export_dialog_active && !ImGui::IsPopupOpen("Export Mesh"))
        ImGui::OpenPopup("Export Mesh");
    if (ImGui::BeginPopupModal("Export Mesh", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!input.export_dialog_active)   // ESC cleared the flag
            ImGui::CloseCurrentPopup();
        ImGui::TextUnformatted("Download as:");
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(280);
        ImGui::InputText("##export_name", web_export_name, sizeof(web_export_name));
        auto web_export = [&](const std::string& ext) {
            std::string name = web_export_name[0] ? web_export_name : "sculpt";
            // Don't double the extension if the user typed it.
            if (name.size() < ext.size()
                || name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
                name += ext;
            std::string fs_path = "/" + name;
            // GPU-resident sculpting/undo leave mesh.pos stale on the CPU —
            // pull the live surface back first (save does the same).
            scene.materialize_active_cpu();
            bool ok = (ext == ".stl") ? mesh->export_stl(fs_path.c_str())
                    : (ext == ".ply") ? mesh->export_ply(fs_path.c_str())
                                      : mesh->export_obj(fs_path.c_str());
            if (ok) {
                web_download_file(fs_path.c_str(), name.c_str());
                std::snprintf(input.notification, sizeof(input.notification),
                              "Exported: %.400s", name.c_str());
                input.notification_timer = 3.0f;
            } else {
                error_popup_msg = "Export failed: " + name;
                error_popup_trigger = true;
            }
            input.export_dialog_active = false;
            ImGui::CloseCurrentPopup();
        };
        if (ImGui::Button(".obj")) web_export(".obj");
        ImGui::SameLine();
        if (ImGui::Button(".stl")) web_export(".stl");
        ImGui::SameLine();
        if (ImGui::Button(".ply")) web_export(".ply");
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            input.export_dialog_active = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Save project: one name field, ".chisel" appended, straight to download.
    if (input.save_dialog_active && !ImGui::IsPopupOpen("Save Project"))
        ImGui::OpenPopup("Save Project");
    if (ImGui::BeginPopupModal("Save Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!input.save_dialog_active)
            ImGui::CloseCurrentPopup();
        ImGui::TextUnformatted("Download project as:");
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(220);
        bool commit = ImGui::InputText("##save_name", web_save_name, sizeof(web_save_name),
                                       ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        ImGui::TextUnformatted(".chisel");
        if (ImGui::Button("Save") || commit) {
            std::string name = web_save_name[0] ? web_save_name : "sculpt";
            const std::string ext = ".chisel";
            if (name.size() >= ext.size()
                && name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
                name.resize(name.size() - ext.size());
            do_save_project("/" + name + ext);
            input.save_dialog_active = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            input.save_dialog_active = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Open/import: no ImGui — the browser's picker IS the dialog. The picked
    // file lands in MEMFS async (chisel_web_import_done); consume it here,
    // then free the RAM copy (current_project_path stays valid — a save
    // recreates the file before downloading it).
    if (input.import_dialog_active) {
        input.import_dialog_active = false;
        web_open_file_picker();
    }
    if (!g_web_import_path.empty()) {
        std::string path = g_web_import_path;
        g_web_import_path.clear();
        do_import_path(path);
        ::remove(path.c_str());
    }

    // Brush-alpha custom image: browser picker IS the dialog. Bytes land in MEMFS
    // (chisel_web_alpha_done); decode into the pool, select it, free the RAM copy.
    if (input.load_alpha_dialog_active) {
        input.load_alpha_dialog_active = false;
        web_open_alpha_picker();
    }
    if (!g_web_alpha_path.empty()) {
        std::string path = g_web_alpha_path;
        g_web_alpha_path.clear();
        int idx = alpha_lib.load_custom(path.c_str());
        if (idx > 0) input.active_alpha = idx;
        ::remove(path.c_str());
    }
#endif // __EMSCRIPTEN__

    // Menu > Reset settings, after its Y/N confirm.
    if (input.reset_settings_requested) {
        input.reset_settings_requested = false;
        settings_reset(input);
        std::snprintf(input.notification, sizeof(input.notification), "Settings reset to defaults");
        input.notification_timer = 2.0f;
    }

    // Save & Quit whose Save window was cancelled: forget the quit. A successful
    // save already closed the window inside do_save_project.
    if (input.quit_after_save && !input.save_dialog_active && !input.save_requested)
        input.quit_after_save = false;
}
