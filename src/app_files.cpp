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
#include "retopo.h"
#include "uv_unwrap.h"
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

// A name as part of a file name: anything odd becomes _.
static std::string file_part(const char* s) {
    std::string n;
    for (const char* c = s; c && *c; c++) {
        const unsigned char u = (unsigned char)*c;
        n += (std::isalnum(u) || u == '-' || u == '_' || u >= 0x80) ? (char)u : '_';
    }
    return n;
}

// UVs go along with a low only when its layout was made for it as it is now.
static bool lo_uv_ok(const RetopoState& rt) {
    return !rt.uv.empty() && rt.uv.fits(rt.result) && rt.uv.key == uv_topology_key(rt.result);
}

// The export dialog's list: every mesh's sculpt (-HI) and, once built, its retopo (-LO),
// each with a tick. Refilled each time the dialog opens, the selected meshes ticked (Ctrl+
// click in Select mode or the Meshes dock picks them; none = the active). File scope because
// the native dialog's side pane draws it frames after the call that opened it.
struct ExportPick {
    uint32_t    id = 0;
    bool        lo = false;
    bool        on = true;
    std::string label;   // the object / file name: <mesh>-HI or <mesh>-LO
    std::string note;
};
static std::vector<ExportPick> g_export_picks;

static void draw_export_picks() {
    ImGui::TextUnformatted("Export these:");
    for (size_t i = 0; i < g_export_picks.size(); i++) {
        ExportPick& p = g_export_picks[i];
        ImGui::PushID((int)i);
        ImGui::Checkbox(p.label.c_str(), &p.on);
        ImGui::SameLine();
        ImGui::TextDisabled("%s", p.note.c_str());
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::TextWrapped("OBJ: one file, each mesh its own object. "
                       "PLY / STL hold one mesh: several make one file each, "
                       "<name>-<mesh>.");
}

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
    auto& clean_retopo_rev = a.clean_retopo_rev;
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

    // ---- Export: which meshes, what name, and the writing ----
    auto fill_export_picks = [&]() {
        g_export_picks.clear();
        const std::vector<uint32_t>& sel = scene.selected_ids();
        auto picked = [&](uint32_t id) {
            return sel.empty() ? id == scene.active_mesh_id()
                               : std::find(sel.begin(), sel.end(), id) != sel.end();
        };
        for (auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview) continue;
            char buf[24];
            std::string base = file_part(Scene::display_name(*up, buf, sizeof buf));
            // Two meshes with one name would write over each other's files.
            for (const ExportPick& p : g_export_picks)
                if (!p.lo && p.label == base + "-HI") { base += "_" + std::to_string(up->id); break; }
            char note[48];
            std::snprintf(note, sizeof note, "%u tris", up->mesh.tri_count());
            g_export_picks.push_back({up->id, false, picked(up->id), base + "-HI", note});
            const RetopoState& rt = up->retopo;
            if (rt.built && !rt.result.faces.empty()) {
                std::snprintf(note, sizeof note, "%u faces%s", rt.result.face_count(),
                              lo_uv_ok(rt) ? ", UVs" : "");
                g_export_picks.push_back({up->id, true, picked(up->id), base + "-LO", note});
            }
        }
    };
    // One thing ticked: its own name. One mesh's HI + LO: the mesh's. Several meshes: the
    // project's (files inside are per mesh).
    auto export_default_name = [&]() -> std::string {
        const ExportPick* first = nullptr;
        int n_on = 0;
        bool one_mesh = true;
        for (const ExportPick& p : g_export_picks) {
            if (!p.on) continue;
            if (!first) first = &p;
            else if (p.id != first->id) one_mesh = false;
            n_on++;
        }
        if (n_on == 1) return first->label;
        if (n_on > 1 && one_mesh) return first->label.substr(0, first->label.size() - 3);   // drop "-HI"
        std::string stem = current_project_path.substr(current_project_path.find_last_of("/\\") + 1);
        const auto dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem.resize(dot);
        return stem.empty() ? std::string("chisel") : stem;
    };
    // Writes the ticked meshes. OBJ holds them all, one object each; PLY and STL hold
    // one mesh, so with several ticked each gets <path minus extension>-<label>.<ext>.
    // Returns the files written; tells the user itself how it went.
    auto export_picks = [&](const std::string& path) -> std::vector<std::string> {
        const size_t slash = path.find_last_of("/\\");
        const size_t dot = path.find_last_of('.');
        const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
        std::string ext = has_ext ? path.substr(dot + 1) : "";
        for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        if (ext != "stl" && ext != "ply") ext = "obj";
        const std::string stem = has_ext ? path.substr(0, dot) : path;

        std::vector<const ExportPick*> on;
        for (const ExportPick& p : g_export_picks)
            if (p.on && scene.find_entity(p.id)) on.push_back(&p);
        std::vector<std::string> wrote;
        if (on.empty()) {
            std::snprintf(input.notification, sizeof(input.notification), "Nothing ticked to export");
            input.notification_timer = 3.0f;
            return wrote;
        }
        // GPU-resident sculpting/undo leave mesh.pos stale on the CPU —
        // pull the live surface back first (save does the same). The others
        // are already current: switching away from a mesh reads it back.
        scene.materialize_active_cpu();
        if (ext == "obj") {
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                std::fprintf(f, "# OBJ export from Chisel: %zu object%s\n",
                             on.size(), on.size() == 1 ? "" : "s");
                uint32_t vb = 0, vtb = 0;
                for (const ExportPick* p : on) {
                    const MeshEntity& e = *scene.find_entity(p->id);
                    if (p->lo) retopo_write_obj(f, e.retopo.result, lo_uv_ok(e.retopo) ? &e.retopo.uv : nullptr,
                                                p->label.c_str(), vb, vtb);
                    else       vb += e.mesh.write_obj(f, p->label.c_str(), vb);
                }
                if (std::fclose(f) == 0) wrote.push_back(path);
            }
        } else {
            for (const ExportPick* p : on) {
                const MeshEntity& e = *scene.find_entity(p->id);
                const std::string out = on.size() == 1 ? path : stem + "-" + p->label + "." + ext;
                const bool ok = p->lo ? (ext == "stl" ? retopo_export_stl(e.retopo.result, out.c_str())
                                                      : retopo_export_ply(e.retopo.result, out.c_str()))
                                      : (ext == "stl" ? e.mesh.export_stl(out.c_str())
                                                      : e.mesh.export_ply(out.c_str()));
                if (ok) wrote.push_back(out);
            }
        }
        const size_t want = ext == "obj" ? 1 : on.size();
        if (wrote.size() < want) {
            error_popup_msg = wrote.empty() ? "Export failed: " + path
                            : "Exported only " + std::to_string(wrote.size()) + " of "
                              + std::to_string(want) + " files next to " + stem;
            error_popup_trigger = true;
        } else if (wrote.size() == 1) {
            std::snprintf(input.notification, sizeof(input.notification), "Exported%s: %.400s",
                          on.size() > 1 ? (" " + std::to_string(on.size()) + " meshes").c_str() : "",
                          wrote[0].c_str());
            input.notification_timer = 3.0f;
        } else {
            std::snprintf(input.notification, sizeof(input.notification), "Exported %zu files: %.400s-*.%s",
                          wrote.size(), stem.c_str(), ext.c_str());
            input.notification_timer = 3.0f;
        }
        return wrote;
    };

    // ---- Open/import a path (shared by the native dialog and the web picker) ----
    auto do_import_path = [&](const std::string& path) {
        auto dot = path.find_last_of('.');
        std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
        for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);

        // An imported mesh's layer name: the file's, without folder or extension.
        std::string stem = path.substr(path.find_last_of("/\\") + 1);
        if (stem.find_last_of('.') != std::string::npos) stem.resize(stem.find_last_of('.'));
        if (stem.size() > 60) stem.resize(60);

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
                // Each mesh brings its own retopo (none in the file = none now).
                scene.load_entities(proj.entities, proj.active_id,
                                    proj.selected_ids, proj.next_id);
                clean_retopo_rev = scene.retopo_rev();
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
            std::vector<ObjPart> parts;   // an OBJ of 2+ objects: one mesh each
            std::string why;              // glTF: why it failed, or what it left out
            const bool gltf = ext == "glb" || ext == "gltf";
            bool imported = (ext == "ply") ? Mesh::import_ply(path.c_str(), loaded)
                          : gltf           ? Mesh::import_glb(path.c_str(), loaded, &parts, &why)
                                           : Mesh::import_obj(path.c_str(), loaded, &parts);
            // The whole file frames the camera; the first object then stands in for
            // "the mesh" below and the others are appended after it.
            Vec3 file_center; float file_radius = 0.0f;
            if (imported) loaded.compute_bounding_sphere(file_center, file_radius);
            // Our own export names objects <mesh>-HI / <mesh>-LO. The HI is a sculpt named
            // <mesh> again; an LO whose HI is in the file is that sculpt's retopo, not
            // another sculpt. Any other object is a sculpt under its own name.
            auto cut_suffix = [](std::string& n, const char* tag) {
                if (n.size() < 4) return false;
                const char sep = n[n.size() - 3];
                if (sep != '-' && sep != '_') return false;
                for (int k = 0; k < 2; k++)
                    if (std::toupper((unsigned char)n[n.size() - 2 + k]) != tag[k]) return false;
                n.resize(n.size() - 3);
                return true;
            };
            std::vector<std::string> names(parts.size());
            std::vector<size_t> sculpts;                     // parts that become meshes
            std::vector<int> lo_for(parts.size(), -1);       // per sculpt part: its LO part
            {
                std::vector<int> is_hi(parts.size(), 0), is_lo(parts.size(), 0);
                for (size_t i = 0; i < parts.size(); i++) {
                    names[i] = parts[i].name;
                    is_hi[i] = cut_suffix(names[i], "HI");
                    if (!is_hi[i]) is_lo[i] = cut_suffix(names[i], "LO");
                }
                for (size_t i = 0; i < parts.size(); i++) {
                    if (is_lo[i])
                        for (size_t j = 0; j < parts.size(); j++)
                            if (is_hi[j] && lo_for[j] < 0 && names[j] == names[i]) { lo_for[j] = (int)i; break; }
                }
                for (size_t i = 0; i < parts.size(); i++) {
                    bool paired = false;
                    for (int l : lo_for) paired = paired || l == (int)i;
                    if (paired) continue;
                    sculpts.push_back(i);
                    if (is_lo[i]) names[i] = parts[i].name;   // a lone LO keeps its whole name
                }
                for (std::string& n : names) {
                    if (n.empty()) n = stem;
                    if (n.size() > 60) n.resize(60);
                }
            }
            // The LO paired with sculpt part i, if any, becomes the active mesh's retopo.
            int n_retopos = 0;
            auto attach_lo = [&](size_t i) {
                RetopoSnapshot snap;
                if (lo_for[i] >= 0 && retopo_from_obj(parts[lo_for[i]], snap)) n_retopos++;
                retopo_load(scene.active_entity().retopo, std::move(snap));
            };
            // ", 3 meshes, 1 retopo" for the toast; nothing for a plain one-mesh file.
            auto count_note = [&]() {
                char b[48] = "";
                int at = 0;
                if (sculpts.size() > 1) at = std::snprintf(b, sizeof b, ", %zu meshes", sculpts.size());
                if (n_retopos > 0)
                    std::snprintf(b + at, sizeof b - at, ", %d retopo%s", n_retopos, n_retopos == 1 ? "" : "s");
                std::string s = b;
                if (gltf) {
                    const Mesh& m = scene.active_mesh();
                    if (m.texture) s += ", textured " + std::to_string(m.texture->w) + "x" + std::to_string(m.texture->h);
                    if (!why.empty()) s += ", " + why;
                }
                return s;
            };
            std::string first_name = stem;
            if (!sculpts.empty()) {
                loaded = std::move(parts[sculpts[0]].mesh);
                first_name = names[sculpts[0]];
            }
            // A NEW scene entity at its authored scale, leaving existing entities, the
            // camera, and their undo untouched. Same commit path insert uses
            // (add_preview → commit_preview → multires/mirror init). Ends active.
            auto append_mesh = [&](Mesh& m, const std::string& name) {
                m.mask.clear();
                const uint32_t new_id = scene.add_preview(m, 0);
                scene.commit_preview(new_id);
                scene.active_entity().name = name;
                multires_stack_init_from_lock(scene.active_multires(), scene.active_mesh(), 0);
                return new_id;
            };
            // The rest of the sculpts, then back to the first, which stays the active one.
            auto append_other_parts = [&](uint32_t first_id) {
                for (size_t k = 1; k < sculpts.size(); k++) {
                    append_mesh(parts[sculpts[k]].mesh, names[sculpts[k]]);
                    attach_lo(sculpts[k]);
                }
                if (sculpts.size() > 1) scene.select(first_id);
            };
            if (imported && input.import_append) {
                const uint32_t first_id = append_mesh(loaded, first_name);
                if (!sculpts.empty()) attach_lo(sculpts[0]);
                append_other_parts(first_id);
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
                              "Appended: %.400s (%u v, %u t%s)",
                              path.c_str(), mesh->vertex_count(), mesh->tri_count(), count_note().c_str());
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
                // A new sculpt: the old one's name and retopo don't belong to it.
                scene.active_entity().name = first_name;
                scene.active_entity().visible = true;
                if (!sculpts.empty()) attach_lo(sculpts[0]);
                else retopo_load(scene.active_entity().retopo, RetopoSnapshot{});
                append_other_parts(scene.active_mesh_id());
                scene.refresh_mirror_map();
                scene.sync();
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                camera.set_target(file_center);
                camera.distance = file_radius * 2.5f;
                current_project_path.clear();

                scene.active_undo().clear(&compute);
                xforms.clear();
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                screen_buffers_dirty = true;
                clean_edit_mark = scene.edit_mark();   // the imported file still holds it
                clean_retopo_rev = scene.retopo_rev();

                std::snprintf(input.notification, sizeof(input.notification),
                              "Imported: %.400s (%u v, %u t%s)",
                              path.c_str(), mesh->vertex_count(), mesh->tri_count(), count_note().c_str());
                input.notification_timer = 3.0f;
            } else {
                error_popup_msg = "Import failed: " + path;
                if (!why.empty()) error_popup_msg += "\n" + why;
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
            // CHISEL_AUTO_MASK_Y=<y>: fully mask everything above y first, so
            // a scripted run exercises the masked-remesh invariant.
            if (const char* my = std::getenv("CHISEL_AUTO_MASK_Y")) {
                const float y0 = (float)std::atof(my);
                mesh->mask.assign(mesh->vertex_count(), 0.0f);
                for (uint32_t v = 0; v < mesh->vertex_count(); v++)
                    if (mesh->pos_y[v] > y0) mesh->mask[v] = 1.0f;
            }
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
        fill_export_picks();
        cfg.fileName = export_default_name() + ".obj";
        // Nothing to choose with one mesh and no low: no side pane.
        if (g_export_picks.size() > 1) {
            cfg.sidePaneWidth = 230.0f;
            cfg.sidePane = [](const char*, IGFD::UserDatas, bool*) { draw_export_picks(); };
        }
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
        fd->OpenDialog("ImportKey", "Open File", ".chisel,.obj,.ply,.glb,.gltf", cfg);
    }

    if (display_file_dialog(fd, "ExportKey")) {
        if (fd->IsOk())
            export_picks(fd->GetFilePathName());
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
            rec.name         = up->name;
            rec.visible      = up->visible;
            retopo_save(up->retopo, rec.retopo);
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
            clean_retopo_rev = scene.retopo_rev();
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
        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
            fill_export_picks();
            std::snprintf(web_export_name, sizeof(web_export_name), "%s", export_default_name().c_str());
        }
        ImGui::SetNextItemWidth(280);
        ImGui::InputText("##export_name", web_export_name, sizeof(web_export_name));
        auto web_export = [&](const std::string& ext) {
            std::string name = web_export_name[0] ? web_export_name : "sculpt";
            // Don't double the extension if the user typed it.
            if (name.size() < ext.size()
                || name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
                name += ext;
            // Several files are several downloads (the browser may ask once to allow that).
            for (const std::string& out : export_picks("/" + name))
                web_download_file(out.c_str(), out.c_str() + 1);
            input.export_dialog_active = false;
            ImGui::CloseCurrentPopup();
        };
        if (g_export_picks.size() > 1) {
            ImGui::Spacing();
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 300.0f);
            draw_export_picks();
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
        }
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
