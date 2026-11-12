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
#include "uv_export.h"
#include "bake.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "file_browser.h"
#include "brush_alpha.h"
#include "project_file.h"
#include "session_files.h"
#include "scene_snapshot.h"
#include "object_xform.h"
#include <filesystem>

#ifndef __EMSCRIPTEN__
// Native file dialogs (Open, Save, Export, Load Brush Alpha) open centred on the
// screen rather than wherever ImGui last left them (often under the button that
// opened them). Appearing only, so the user can still drag one aside. The position
// is set only when this key's dialog is up: otherwise it would land on whatever
// window begins next. The minimum size leaves room for the Places pane, the file
// list and an options pane side by side.
static bool display_file_dialog(IGFD::FileDialog* fd, const char* key) {
    if (fd->IsOpened(key)) {
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(900, 560), ImGuiCond_FirstUseEver);
    }
    return fd->Display(key, ImGuiWindowFlags_NoCollapse, ImVec2(760, 440));
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


// ---- UV mode's exports (UV dock): the layout picture, and model + textures ----------
// What the two windows can offer, worked out as one opens (file scope: the native
// dialog's side pane draws it frames later).
struct UvExportAvail {
    int  parts = 0, with_uv = 0;      // meshes shown, and how many have a current layout
    bool normal = false, color = false, id = false;   // maps in the bake
    bool fresh = true;                // the bake still fits the meshes
    int  res = 0;                     // the bake's size
};
static UvExportAvail g_uvx;

// Size chips shared by the windows: 512 / 1K / 2K / 4K.
static void size_chips(const char* id, int& v) {
    static const int sizes[4] = {512, 1024, 2048, 4096};
    static const char* lab[4] = {"512", "1K", "2K", "4K"};
    ImGui::PushID(id);
    for (int i = 0; i < 4; i++) {
        if (i) ImGui::SameLine();
        if (ImGui::RadioButton(lab[i], v == sizes[i])) v = sizes[i];
    }
    ImGui::PopID();
}

static void draw_uvmap_options(InputState& in) {
    ImGui::TextUnformatted("Size");
    size_chips("uvmSize", in.uvmap_size);
    ImGui::SetNextItemWidth(200.0f);
    ImGui::SliderFloat("##uvmWire", &in.uvmap_wire, 0.5f, 8.0f, "wire %.1f px");
    ImGui::Checkbox("Fill the islands", &in.uvmap_fill);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("On: each island in its colour under the wire. Off: just the wire");
    ImGui::BeginDisabled(!g_uvx.id);
    bool cid = in.uvmap_color_id && g_uvx.id;
    if (ImGui::Checkbox("Color ID map too", &cid)) in.uvmap_color_id = cid;
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", g_uvx.id ? "<mesh>-ColorID_<size>.png beside it: the baked ColorID paint layer"
                                         : "Bake with Color ID on first (needs a paint layer named ColorID)");
    if (g_uvx.parts > 1) ImGui::TextDisabled("%d meshes in one square", g_uvx.with_uv);
}

static void draw_uvx_options(InputState& in) {
    ImGui::Checkbox("Model", &in.uvx_model);
    ImGui::BeginDisabled(!in.uvx_model);
    ImGui::Indent();
    ImGui::RadioButton("GLB", &in.uvx_format, 2);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One file: the meshes with their colour and normal maps inside");
    ImGui::SameLine();
    ImGui::RadioButton("OBJ", &in.uvx_format, 0);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("With an .mtl that points at the maps");
    ImGui::SameLine();
    ImGui::RadioButton("STL", &in.uvx_format, 1);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Shape only: no UVs, no textures");
    ImGui::Unindent();
    ImGui::EndDisabled();
    if (g_uvx.parts > 1) ImGui::TextDisabled("%d meshes", g_uvx.parts);

    ImGui::Spacing();
    ImGui::Checkbox("Textures", &in.uvx_textures);
    ImGui::BeginDisabled(!in.uvx_textures);
    ImGui::Indent();
    auto tex = [&](const char* label, bool& v, bool have, const char* why) {
        ImGui::BeginDisabled(!have);
        bool on = v && have;
        if (ImGui::Checkbox(label, &on)) v = on;
        ImGui::EndDisabled();
        if (!have && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", why);
    };
    tex("Colour", in.uvx_tex_color, g_uvx.color, "Bake with Colour on first");
    tex("Normal", in.uvx_tex_normal, g_uvx.normal, "Bake with Normal on first");
    tex("Color ID", in.uvx_tex_id, g_uvx.id, "Bake with Color ID on first (needs a paint layer named ColorID)");
    tex("UV map", in.uvx_tex_uvmap, g_uvx.with_uv > 0, "Unwrap first");
    ImGui::Unindent();
    ImGui::EndDisabled();
    if (g_uvx.res && !g_uvx.fresh) ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f), "The bake is out of date");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 220.0f);
    ImGui::TextDisabled("%s", in.uvx_model && in.uvx_format == 2
        ? "GLB carries Colour and Normal inside; Color ID and the UV map go beside it as PNGs."
        : "Maps go beside the model as PNGs: <name>-Color_1024.png and so on.");
    ImGui::PopTextWrapPos();
}

// Every alive, committed entity (not INSERT previews) with its own mesh + multires,
// plus the scene-wide state: what a .chisel holds. Copies the meshes, so the result
// stands alone (the autosave writer hands it to another thread). Main thread only:
// materialize_active_cpu may read back from the GPU, which no stroke may be doing.
void app_build_project(App& a, ProjectData& proj) {
    Scene& scene = a.scene;
    scene.materialize_active_cpu();  // 2b: copies the active entity's mesh + multires to disk
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
    proj.camera       = a.camera;
    proj.mirror_x     = a.input.mirror_x;
    proj.subdiv_level = a.input.subdiv_level;
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


    // ---- The UV map and model + textures windows: the meshes the UV panel shows (the active one, then the rest
    // of the selection), as Bake takes them. Each low goes with its layout when the layout
    // is current for it.
    auto uv_parts = [&]() {
        std::vector<UvExportPart> out;
        auto take = [&](uint32_t id) {
            MeshEntity* e = scene.find_entity(id);
            if (!e || !e->alive || e->preview || !e->retopo.built || e->retopo.result.faces.empty()) return;
            for (const UvExportPart& p : out) if (p.id == id) return;
            char buf[24];
            out.push_back({id, file_part(Scene::display_name(*e, buf, sizeof buf)), &e->retopo.result,
                           lo_uv_ok(e->retopo) ? &e->retopo.uv : nullptr});
        };
        take(scene.active_mesh_id());
        for (uint32_t id : scene.selected_ids()) take(id);
        return out;
    };
    auto fill_uvx_avail = [&]() {
        const std::vector<UvExportPart> parts = uv_parts();
        g_uvx = UvExportAvail{};
        g_uvx.parts = (int)parts.size();
        for (const UvExportPart& p : parts) g_uvx.with_uv += p.uv ? 1 : 0;
        const BakeState& bk = scene.active_entity().retopo.bake;
        const BakeResult& r = bk.res();
        g_uvx.res = r.res;
        g_uvx.normal = r.has_normal;
        g_uvx.color = r.has_color;
        g_uvx.id = r.has_id;
        g_uvx.fresh = bk.fresh;
    };
    auto active_part_name = [&]() {
        char buf[24];
        return file_part(Scene::display_name(scene.active_entity(), buf, sizeof buf));
    };
    // "<dir>/<name>" without the extension; and the "-UVmap_<n>" a name may carry set to
    // the size actually written (the native dialog's name was made before the size moved).
    auto strip_ext = [](const std::string& path) {
        const size_t slash = path.find_last_of("/\\"), dot = path.find_last_of('.');
        return dot != std::string::npos && (slash == std::string::npos || dot > slash) ? path.substr(0, dot) : path;
    };
    auto retag = [](std::string stem, const char* tag, int size, const char* to_tag) {
        const size_t at = stem.rfind(tag);
        const std::string want = std::string(to_tag) + std::to_string(size);
        if (at != std::string::npos) {
            size_t e = at + std::strlen(tag);
            while (e < stem.size() && std::isdigit((unsigned char)stem[e])) e++;
            if (e == stem.size()) return stem.substr(0, at) + want;
        }
        return stem + want;
    };
    auto toast_written = [&](const std::vector<std::string>& wrote, size_t want, const std::string& where) {
        if (wrote.size() < want) {
            error_popup_msg = wrote.empty() ? "Export failed: " + where
                            : "Exported only " + std::to_string(wrote.size()) + " of " + std::to_string(want)
                              + " files next to " + where;
            error_popup_trigger = true;
        } else {
            if (wrote.size() == 1)
                std::snprintf(input.notification, sizeof(input.notification), "Exported: %.400s", wrote[0].c_str());
            else
                std::snprintf(input.notification, sizeof(input.notification), "Exported %zu files: %.400s*",
                              wrote.size(), where.c_str());
            input.notification_timer = 3.0f;
        }
    };
    // The layout picture (and the Color ID map beside it). Returns the files written.
    auto export_uv_map = [&](const std::string& path) {
        std::vector<std::string> wrote;
        std::vector<UvExportPart> parts = uv_parts();
        parts.erase(std::remove_if(parts.begin(), parts.end(), [](const UvExportPart& p) { return !p.uv; }), parts.end());
        if (parts.empty()) {
            std::snprintf(input.notification, sizeof(input.notification), "Unwrap first: the UV map is the layout");
            input.notification_timer = 3.0f;
            return wrote;
        }
        const int size = input.uvmap_size;
        const std::string stem = retag(strip_ext(path), "-UVmap_", size, "-UVmap_");
        size_t want = 1;
        if (uv_map_write_png(parts, size, input.uvmap_wire, input.uvmap_fill, (stem + ".png").c_str()))
            wrote.push_back(stem + ".png");
        const BakeResult& r = scene.active_entity().retopo.bake.res();
        if (input.uvmap_color_id && r.has_id) {
            want++;
            const std::string id = retag(stem, "-UVmap_", size, "-ColorID_") + ".png";
            if (bake_write_png(r, 2, id.c_str(), size)) wrote.push_back(id);
        }
        toast_written(wrote, want, stem);
        return wrote;
    };
    // Model and textures: the lows as OBJ (+ MTL) / STL / GLB, and the ticked maps beside
    // them (a GLB carries Colour and Normal inside instead).
    auto export_uv_assets = [&](const std::string& path) {
        std::vector<std::string> wrote;
        const std::vector<UvExportPart> parts = uv_parts();
        if (parts.empty()) {
            std::snprintf(input.notification, sizeof(input.notification), "Build a retopo mesh in mode 0 first");
            input.notification_timer = 3.0f;
            return wrote;
        }
        const std::string stem = strip_ext(path);
        const BakeResult& r = scene.active_entity().retopo.bake.res();
        const bool tex = input.uvx_textures;
        const bool t_col = tex && input.uvx_tex_color && r.has_color;
        const bool t_nrm = tex && input.uvx_tex_normal && r.has_normal;
        const bool t_id  = tex && input.uvx_tex_id && r.has_id;
        const bool t_uv = tex && input.uvx_tex_uvmap;
        const bool model = input.uvx_model;
        const int fmt = input.uvx_format;
        const bool glb_carries = model && fmt == 2;
        const std::string res = "_" + std::to_string(r.res);
        const std::string col_png = stem + "-Color" + res + ".png", nrm_png = stem + "-Normal" + res + ".png";
        auto base = [](const std::string& p) { return p.substr(p.find_last_of("/\\") + 1); };
        size_t want = 0;
        if (model) {
            if (fmt == 2) {
                want++;
                std::vector<uint8_t> cp, np;
                if (t_col) cp = bake_png_bytes(r, 1);
                if (t_nrm) np = bake_png_bytes(r, 0);
                if (glb_write_lows((stem + ".glb").c_str(), parts, t_col ? &cp : nullptr, t_nrm ? &np : nullptr))
                    wrote.push_back(stem + ".glb");
            } else if (fmt == 0) {
                want++;
                const bool mtl = t_col || t_nrm;
                if (mtl) {
                    want++;
                    if (FILE* m = std::fopen((stem + ".mtl").c_str(), "w")) {
                        std::fprintf(m, "# Chisel bake\nnewmtl baked\nKd 1 1 1\nKs 0 0 0\n");
                        if (t_col) std::fprintf(m, "map_Kd %s\n", base(col_png).c_str());
                        if (t_nrm) std::fprintf(m, "map_Bump %s\n", base(nrm_png).c_str());
                        if (std::fclose(m) == 0) wrote.push_back(stem + ".mtl");
                    }
                }
                if (FILE* f = std::fopen((stem + ".obj").c_str(), "w")) {
                    std::fprintf(f, "# OBJ export from Chisel: %zu retopo mesh%s\n", parts.size(), parts.size() == 1 ? "" : "es");
                    if (mtl) std::fprintf(f, "mtllib %s\n", base(stem + ".mtl").c_str());
                    uint32_t vb = 0, vtb = 0;
                    for (const UvExportPart& p : parts) {
                        std::fprintf(f, "o %s-LO\n", p.name.c_str());
                        if (mtl) std::fprintf(f, "usemtl baked\n");
                        retopo_write_obj(f, *p.rm, p.uv, nullptr, vb, vtb);
                    }
                    if (std::fclose(f) == 0) wrote.push_back(stem + ".obj");
                }
            } else {
                for (const UvExportPart& p : parts) {
                    want++;
                    const std::string out = parts.size() == 1 ? stem + ".stl" : stem + "-" + p.name + "-LO.stl";
                    if (retopo_export_stl(*p.rm, out.c_str())) wrote.push_back(out);
                }
            }
        }
        if (t_col && !glb_carries) { want++; if (bake_write_png(r, 1, col_png.c_str())) wrote.push_back(col_png); }
        if (t_nrm && !glb_carries) { want++; if (bake_write_png(r, 0, nrm_png.c_str())) wrote.push_back(nrm_png); }
        if (t_id) {
            want++;
            const std::string id = stem + "-ColorID" + res + ".png";
            if (bake_write_png(r, 2, id.c_str())) wrote.push_back(id);
        }
        if (t_uv) {
            std::vector<UvExportPart> lp = parts;
            lp.erase(std::remove_if(lp.begin(), lp.end(), [](const UvExportPart& p) { return !p.uv; }), lp.end());
            if (!lp.empty()) {
                want++;
                const std::string uvp = stem + "-UVmap_" + std::to_string(input.uvmap_size) + ".png";
                if (uv_map_write_png(lp, input.uvmap_size, input.uvmap_wire, input.uvmap_fill, uvp.c_str()))
                    wrote.push_back(uvp);
            }
        }
        if (!want) {
            std::snprintf(input.notification, sizeof(input.notification), "Nothing ticked to export");
            input.notification_timer = 3.0f;
            return wrote;
        }
        toast_written(wrote, want, stem);
        // Maps ticked with no bake to put in them: say so, or the file looks broken
        // (Windows 2026-11-12: a bare GLB exported before the bake, read as a bug).
        const bool asked_maps = tex && (input.uvx_tex_color || input.uvx_tex_normal);
        const bool no_maps = asked_maps && !r.has_color && !r.has_normal;
        if (model && no_maps && wrote.size() == want) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Exported geometry only: no bake yet (UV mode, Bake) for the maps");
            input.notification_timer = 4.0f;
        }
        std::printf("[export] %s: %zu of %zu written, %zu mesh%s; colour %s, normal %s%s\n",
                    stem.c_str(), wrote.size(), want, parts.size(), parts.size() == 1 ? "" : "es",
                    t_col ? (glb_carries ? "inside" : "png") : "no",
                    t_nrm ? (glb_carries ? "inside" : "png") : "no",
                    no_maps ? " (maps ticked, nothing baked)" : "");
        std::fflush(stdout);
        return wrote;
    };
    auto uvx_ext = [&]() { return input.uvx_model ? (input.uvx_format == 2 ? ".glb" : input.uvx_format == 1 ? ".stl" : ".obj") : ""; };
    // A closed export window always leaves the kind at MESH for the toolbar's next one.
    if (!input.export_dialog_active) input.export_kind = InputState::ExportKind::MESH;
    const bool exp_mesh = input.export_dialog_active && input.export_kind == InputState::ExportKind::MESH;
    const bool exp_uvmap = input.export_dialog_active && input.export_kind == InputState::ExportKind::UV_MAP;
    const bool exp_uvx = input.export_dialog_active && input.export_kind == InputState::ExportKind::UV_ASSETS;

    // ---- Put a whole project into the scene: Open (.chisel), File > New, Recover ----
    // `path` becomes the save target (empty = Unsaved Project). Clean as loaded.
    auto apply_project = [&](ProjectData& proj, const std::string& path) {
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
    };

    // ---- Recover an autosave: its work, but not its file ----
    // Opens as Unsaved Project, so Ctrl+S asks where and never writes into the autosave
    // folder, and dirty, so closing still asks. The autosave itself is deleted once this
    // session holds its own copy (session_files.h).
    auto do_recover = [&](const std::string& path) {
        ProjectData proj;
        LoadResult lr = load_project(path.c_str(), proj);
        if (lr != LoadResult::OK || proj.entities.empty()) {
            error_popup_msg = std::string("Recover failed: ") + result_string(lr) + "\n" + path;
            error_popup_trigger = true;
            return;
        }
        apply_project(proj, std::string());
        clean_edit_mark = ~0ull;   // matches no real edit mark: unsaved until a save
        autosave_mark_recovered(path);
        std::snprintf(input.notification, sizeof(input.notification),
                      "Recovered unsaved work (%zu mesh%s). Ctrl+S to keep it",
                      proj.entities.size(), proj.entities.size() == 1 ? "" : "es");
        input.notification_timer = 4.0f;
    };

    // ---- File > New: the opening ball again, as a fresh launch builds it ----
    // Goes through the load path, so nothing of the old scene (meshes, retopo, layers,
    // undo) carries over. Mirror and every setting stay as they are.
    auto do_new_project = [&]() {
        static const int kStartLevel = InputState().subdiv_level;
        const bool uv = input.sphere_kind == InputState::SphereKind::UV;
        const int level = uv ? 0 : kStartLevel;
        ProjectData proj;
        EntityRecord rec;
        rec.id = 1;
        rec.subdiv_level = (uint32_t)level;
        rec.mesh = uv ? uv_sphere(32, 16) : icosphere(level);
        rec.mesh.recompute_normals();
        rec.mesh.build_adjacency();
        multires_stack_init_from_lock(rec.multires, rec.mesh, level);
        proj.entities.push_back(std::move(rec));
        proj.active_id = 1;
        proj.selected_ids = {1};
        proj.next_id = 2;
        proj.mirror_use_topology = scene.mirror_topology();
        proj.mirror_x = input.mirror_x;
        proj.subdiv_level = level;
        apply_project(proj, std::string());
        camera = Camera();
        camera.set_target(mesh_center);
        camera.distance = mesh_radius * 2.5f;
        std::snprintf(input.notification, sizeof(input.notification), "New project");
        input.notification_timer = 2.0f;
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
                apply_project(proj, path);
                recents_add(path);
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
            if (const char* env = std::getenv("CHISEL_REMESH_MIRROR"))
                input.mirror_x = std::atoi(env) != 0;
            if (const char* env = std::getenv("CHISEL_REMESH_DETAIL"))
                input.remesh_detail = std::min(4.0f, std::max(0.25f, (float)std::atof(env)));
            // SDF remesh knobs for CHISEL_AUTO_MERGE runs.
            if (const char* env = std::getenv("CHISEL_MERGE_RES"))
                input.voxel_merge_resolution = std::min(512, std::max(64, std::atoi(env)));
            if (const char* env = std::getenv("CHISEL_MERGE_MIRROR"))
                input.voxel_merge_mirror = std::atoi(env) != 0;
            if (const char* env = std::getenv("CHISEL_MERGE_CLOSE_HOLES"))
                input.voxel_merge_close_holes = std::min(2, std::max(0, std::atoi(env)));
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
    auto* fd = file_browser();
    // Dev hook for screenshots (no input injection on Wayland): CHISEL_SHOT_DIALOG=open,
    // save or folder opens that dialog once at startup. Inert unless set.
    {
        static bool shot_done = false;
        if (!shot_done) {
            shot_done = true;
            if (const char* e = std::getenv("CHISEL_SHOT_DIALOG")) {
                const std::string w = e;
                if (w == "open") input.import_dialog_active = true;
                else if (w == "save") input.save_dialog_active = true;
                else if (w == "folder") input.browse_default_request = true;
            }
        }
    }
    // Every dialog starts where the last file went to or came from (file_browser.cpp).
    auto start_dir = [&]() { return file_browser_start_dir(input.browse_default_dir, default_browse_path); };

    if (exp_mesh && !fd->IsOpened("ExportKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = start_dir();
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite | ImGuiFileDialogFlags_DontShowHiddenFiles;
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
        cfg.path = start_dir();
        cfg.flags = ImGuiFileDialogFlags_DontShowHiddenFiles;
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
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
            export_picks(fd->GetFilePathName());
        }
        fd->Close();
        input.export_dialog_active = false;
    }
    if (!exp_mesh && fd->IsOpened("ExportKey"))
        fd->Close();

    // UV mode's two: the layout picture, and model + textures. Their options sit in the
    // side pane; the name is a starting point (the size and format chosen there win).
    InputState* inp = &input;
    if (exp_uvmap && !fd->IsOpened("UvMapKey")) {
        fill_uvx_avail();
        IGFD::FileDialogConfig cfg;
        cfg.path = start_dir();
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite | ImGuiFileDialogFlags_DontShowHiddenFiles;
        cfg.fileName = active_part_name() + "-UVmap_" + std::to_string(input.uvmap_size) + ".png";
        cfg.sidePaneWidth = 240.0f;
        cfg.sidePane = [inp](const char*, IGFD::UserDatas, bool*) { draw_uvmap_options(*inp); };
        fd->OpenDialog("UvMapKey", "Export UV Map", ".png", cfg);
    }
    if (display_file_dialog(fd, "UvMapKey")) {
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
            export_uv_map(fd->GetFilePathName());
        }
        fd->Close();
        input.export_dialog_active = false;
    }
    if (!exp_uvmap && fd->IsOpened("UvMapKey"))
        fd->Close();

    if (exp_uvx && !fd->IsOpened("UvAssetsKey")) {
        fill_uvx_avail();
        IGFD::FileDialogConfig cfg;
        cfg.path = start_dir();
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite | ImGuiFileDialogFlags_DontShowHiddenFiles;
        cfg.fileName = active_part_name() + uvx_ext();
        cfg.sidePaneWidth = 250.0f;
        cfg.sidePane = [inp](const char*, IGFD::UserDatas, bool*) { draw_uvx_options(*inp); };
        fd->OpenDialog("UvAssetsKey", "Export Model and Textures", ".glb,.obj,.stl,.png", cfg);
    }
    if (display_file_dialog(fd, "UvAssetsKey")) {
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
            export_uv_assets(fd->GetFilePathName());
        }
        fd->Close();
        input.export_dialog_active = false;
    }
    if (!exp_uvx && fd->IsOpened("UvAssetsKey"))
        fd->Close();

    if (display_file_dialog(fd, "ImportKey")) {
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
            do_import_path(fd->GetFilePathName());
        }
        fd->Close();
        input.import_dialog_active = false;
    }
    if (!input.import_dialog_active && fd->IsOpened("ImportKey"))
        fd->Close();

    // Brush-alpha custom image loader (mirrors ImportKey). Loads a grayscale
    // bitmap as a new pool entry and selects it.
    if (input.load_alpha_dialog_active && !fd->IsOpened("AlphaKey")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = start_dir();
        cfg.flags = ImGuiFileDialogFlags_DontShowHiddenFiles;
        fd->OpenDialog("AlphaKey", "Load Brush Alpha", ".png,.jpg,.jpeg,.tga,.bmp", cfg);
    }
    if (display_file_dialog(fd, "AlphaKey")) {
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
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

    // Burger menu > Default folder: a folder pick (no filters = folders only). Picking
    // one also makes it the folder the next dialog opens in.
    if (input.browse_default_request && !fd->IsOpened("FolderKey")) {
        IGFD::FileDialogConfig cfg;
        std::error_code ec;
        start_dir();   // tells the Places pane which folder is the default
        cfg.path = std::filesystem::is_directory(input.browse_default_dir, ec) ? input.browse_default_dir
                                                                               : default_browse_path;
        cfg.flags = ImGuiFileDialogFlags_DontShowHiddenFiles;
        cfg.sidePaneWidth = 220.0f;
        const std::string builtin = default_browse_path;
        cfg.sidePane = [inp, builtin](const char*, IGFD::UserDatas, bool*) {
            ImGui::TextWrapped("Open, Save and Export start in the folder you last used. "
                               "This one is where they start before that, and it is "
                               "always in Places.");
            ImGui::Spacing();
            ImGui::TextWrapped("Go into a folder (or click one) and press OK.");
            if (!inp->browse_default_dir.empty()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                if (ImGui::Button("Use the built-in folder")) {
                    inp->browse_default_dir.clear();
                    file_browser_note(builtin, false);
                    inp->browse_default_request = false;
                }
                ImGui::TextDisabled("%s", builtin.c_str());
            }
        };
        fd->OpenDialog("FolderKey", "Default Folder", nullptr, cfg);
    }
    if (display_file_dialog(fd, "FolderKey")) {
        if (fd->IsOk()) {
            std::error_code ec;
            std::string d = fd->GetFilePathName();
            if (!std::filesystem::is_directory(d, ec)) d = fd->GetCurrentPath();
            if (std::filesystem::is_directory(d, ec)) {
                d = std::filesystem::path(d).lexically_normal().string();
                while (d.size() > 1 && (d.back() == '/' || d.back() == '\\')) d.pop_back();
                input.browse_default_dir = d;
                file_browser_note(d, false);
                std::snprintf(input.notification, sizeof(input.notification), "Default folder: %s",
                              d.c_str());
            }
        }
        fd->Close();
        input.browse_default_request = false;
    }
    if (!input.browse_default_request && fd->IsOpened("FolderKey"))
        fd->Close();
#endif // !__EMSCRIPTEN__

    // ---- Save project (.chisel) ----
    auto do_save_project = [&](const std::string& path) {
        ProjectData proj;
        app_build_project(a, proj);
        SaveResult sr = save_project(path.c_str(), proj);
        if (sr == SaveResult::OK && input.quit_after_save) {
            input.quit_after_save = false;
            // Save & New: the fresh sphere comes after this save, not a close.
            if (input.quit_to_new) { input.quit_to_new = false; input.new_project_requested = true; }
            else glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (sr != SaveResult::OK) { input.quit_after_save = false; input.quit_to_new = false; }
        if (sr == SaveResult::OK) {
            current_project_path = path;
            clean_edit_mark = scene.edit_mark();
            clean_retopo_rev = scene.retopo_rev();
            recents_add(path);
            autosave_drop_session();   // the real file holds this work now
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
        if (input.drop_recover) do_recover(input.drop_path);
        else                    do_import_path(input.drop_path);
        input.drop_recover = false;
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
    // Numbering rules live in next_incremental_path (file_browser.cpp), shared with the
    // dialogs' "+" button so both walk the same series.
    if (input.save_incremental_requested) {
        input.save_incremental_requested = false;
        if (current_project_path.empty()) {
            // Nothing to number off yet — ask for a name first, like plain save.
            input.save_dialog_active = true;
        } else {
            do_save_project(next_incremental_path(current_project_path, ".chisel"));
        }
    }

#ifndef __EMSCRIPTEN__
    if (input.save_dialog_active && !fd->IsOpened("SaveKey")) {
        IGFD::FileDialogConfig cfg;
        // A project that has a file starts in its folder, under its name.
        std::error_code ec;
        const std::filesystem::path cur(current_project_path);
        cfg.path = file_browser_start_dir(input.browse_default_dir, default_browse_path);
        if (!current_project_path.empty() && std::filesystem::is_directory(cur.parent_path(), ec)) {
            cfg.path = cur.parent_path().string();
            cfg.fileName = cur.filename().string();
        }
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite | ImGuiFileDialogFlags_DontShowHiddenFiles;
        fd->OpenDialog("SaveKey", "Save Project", ".chisel", cfg);
    }

    if (display_file_dialog(fd, "SaveKey")) {
        if (fd->IsOk()) {
            file_browser_note(fd->GetFilePathName());
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
    if (exp_mesh && !ImGui::IsPopupOpen("Export Mesh"))
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

    // UV mode's two. The UV map's name is made for the user: <mesh>-UVmap_<size>.png.
    auto web_get = [&](const std::vector<std::string>& files) {
        for (const std::string& out : files) web_download_file(out.c_str(), out.c_str() + 1);
        input.export_dialog_active = false;
        ImGui::CloseCurrentPopup();
    };
    if (exp_uvmap && !ImGui::IsPopupOpen("Export UV Map"))
        ImGui::OpenPopup("Export UV Map");
    if (ImGui::BeginPopupModal("Export UV Map", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!input.export_dialog_active) ImGui::CloseCurrentPopup();
        if (ImGui::IsWindowAppearing()) fill_uvx_avail();
        draw_uvmap_options(input);
        const std::string name = active_part_name() + "-UVmap_" + std::to_string(input.uvmap_size) + ".png";
        ImGui::Spacing();
        ImGui::TextDisabled("%s", name.c_str());
        if (ImGui::Button("Download")) web_get(export_uv_map("/" + name));
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { input.export_dialog_active = false; ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    static char web_uvx_name[128] = "";
    if (exp_uvx && !ImGui::IsPopupOpen("Export Model and Textures"))
        ImGui::OpenPopup("Export Model and Textures");
    if (ImGui::BeginPopupModal("Export Model and Textures", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!input.export_dialog_active) ImGui::CloseCurrentPopup();
        if (ImGui::IsWindowAppearing()) {
            fill_uvx_avail();
            std::snprintf(web_uvx_name, sizeof web_uvx_name, "%s", active_part_name().c_str());
            ImGui::SetKeyboardFocusHere();
        }
        ImGui::TextUnformatted("Download as:");
        ImGui::SetNextItemWidth(260);
        ImGui::InputText("##uvx_name", web_uvx_name, sizeof web_uvx_name);
        ImGui::Spacing();
        draw_uvx_options(input);
        ImGui::Spacing();
        if (ImGui::Button("Download")) {
            std::string n = web_uvx_name[0] ? web_uvx_name : "model";
            for (const char* e : {".glb", ".obj", ".stl"})
                if (n.size() > 4 && n.compare(n.size() - 4, 4, e) == 0) n.resize(n.size() - 4);
            web_get(export_uv_assets("/" + n + uvx_ext()));
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { input.export_dialog_active = false; ImGui::CloseCurrentPopup(); }
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
    if (!input.quit_requested && !input.quit_after_save) input.quit_to_new = false;

    // File > New (or the "New?" dialog's Discard / Save first, once the save landed).
    if (input.new_project_requested) {
        input.new_project_requested = false;
        do_new_project();
    }
}

// ---- Autosave (native) ----
// Called from main.cpp's idle-frame skip (and its hidden-window path) with how long
// nothing has happened. Two triggers, both only with work changed since the last
// autosave AND not saved for real:
//   idle: a full minute without input, the "went for coffee" case;
//   timer: 5 minutes of unsaved work, taken at the next 1-second pause, so a session
//          that never stops for a minute still gets a copy.
// The main thread only copies the scene (app_build_project); the writing happens on
// session_files' thread. The marks move on at the snapshot, so a failed write waits
// for the next edit instead of retrying every frame. Console lines only: a toast
// would wake the drawing loop, and nobody asked for this save.
void app_autosave_tick(App& a, double idle_secs) {
    if (!autosave_enabled()) return;
    static uint64_t saved_edit = ~0ull, saved_rev = ~0ull;
    static double dirty_since = -1.0;

    bool ok; double secs; unsigned long long bytes; std::string path;
    if (autosave_poll(ok, secs, bytes, path)) {
        if (ok) std::printf("[autosave] wrote %.1f MB in %.2f s: %s\n", bytes / 1048576.0, secs, path.c_str());
        else    std::printf("[autosave] FAILED after %.2f s: %s\n", secs, path.c_str());
        std::fflush(stdout);
    }

    Scene& scene = a.scene;
    const uint64_t edit = scene.edit_mark(), rev = scene.retopo_rev();
    const bool clean = edit == a.clean_edit_mark && rev == a.clean_retopo_rev;
    const bool changed = edit != saved_edit || rev != saved_rev;
    if (clean || !changed) { dirty_since = -1.0; return; }
    const double now = glfwGetTime();
    if (dirty_since < 0.0) dirty_since = now;

    const bool idle_rule  = idle_secs >= 60.0;
    const bool timer_rule = now - dirty_since >= 300.0 && idle_secs >= 1.0;
    if (!idle_rule && !timer_rule) return;
    const InputState& in = a.input;
    if (autosave_busy() || a.app_state != AppState::IDLE || a.vmerge_job
        || a.brush_stroke.is_active() || in.remesh_requested || in.remesh_in_progress
        || in.voxel_merge_requested || in.voxel_merge_in_progress) return;

    const double t0 = glfwGetTime();
    ProjectData* proj = new ProjectData;
    app_build_project(a, *proj);
    saved_edit = scene.edit_mark();
    saved_rev = scene.retopo_rev();
    dirty_since = -1.0;
    const double copy_ms = (glfwGetTime() - t0) * 1000.0;
    std::printf("[autosave] %s: snapshot of %zu mesh%s took %.0f ms (main thread), writing in the background\n",
                idle_rule ? "idle 60 s" : "5 min unsaved", proj->entities.size(),
                proj->entities.size() == 1 ? "" : "es", copy_ms);
    std::fflush(stdout);
    autosave_start(proj);
}
