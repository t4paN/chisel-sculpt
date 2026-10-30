#pragma once
#include <cstdint>
#include <string>
#include "imgui.h"
#include "ui_overlay.h"   // MultiresInfo

struct InputState;
struct AlphaLibrary;

// The modern ("default") skin from the 2026-10 UI redesign: everything on the
// screen edges, the middle left empty for the model.
//   left rail    modes 1-4 | brushes with key badges | Smooth, Mask | current alpha
//   top-left     file name (+ "unsaved") | Save, Save copy, Open, Export
//   top-right    Undo, Redo | subdiv stepper | Merge, Mirror X, paint visibility | ? | menu
//   bottom-left  brush readout: Size / Strength / Hardness / Spacing with key + bar
//   bottom-right one status line: verts, mirror, autosmooth, fps
// Panels share one material (Menu > Appearance): clear glass with a backdrop blur
// through to solid, plus optional springy motion and viewfinder corners.
//
// The Y2K skin (board E) has its own layout in ui_skin.cpp's draw_y2k_ui: opaque
// shelves docked on all four edges (title + slider shelf on top, brushes left,
// commands right, a status bar below), glossy two-tone tiles, orange for "on".
// Verdana where the system has it (not redistributable, so loaded from the OS),
// else DejaVu Sans, else Plex.
//
// The DOS skin (board D) is the same layout and code with a CGA theme swapped in:
// Chisel's 8x8 bitmap font at 16 px, the 16 CGA colours, black panels with grey
// frames and hard shadows, 16x16 one-bit icons, no blur and no springs. The
// classic islands + bitmap HUD stay available too (InputState::ui_skin).

enum class UiFont { Sans, SansSemibold, Mono, MonoMedium };

// Once, after ImGui::CreateContext and before the first NewFrame.
void ui_skin_load_fonts();
// Every frame, before ImGui::NewFrame: switches font + style when the skin changes.
void ui_skin_begin_frame(const InputState& input);
// The active skin's faces (all four are the bitmap font in DOS); nullptr while
// the classic skin is active.
ImFont* ui_skin_font(UiFont f);

// A fixed menu-item width designed for 13-14 px proportional text, adjusted for the
// active skin's font (doubled-and-then-some for DOS's 16 px cells).
float ui_skin_menu_item_width(float w);

// Backdrop blur for this frame (sigma in px, saturation). blur_px == 0 means the
// material is solid enough that no blur pass should run.
void ui_skin_backdrop_params(const InputState& input, float* blur_px, float* saturate);

struct SkinStats {
    const char* version = nullptr;    // CHISEL_VERSION (the Y2K title strip shows it)
    uint32_t    tris = 0;
    uint32_t    verts = 0;
    int         level = 0;            // live multires level
    float       fps = 0.0f;
    const char* project_path = nullptr;
};

// The whole modern UI, including its own notification toast and hold-key slider
// readout (the caller skips TextOverlay's draw_notification / draw_slider /
// draw_toolbar / draw_fps / draw_mode_indicator while this skin is active).
void draw_modern_ui(InputState& input, int win_w, int win_h, const AlphaLibrary* alpha_lib,
                    MultiresInfo mres, const SkinStats& stats);

// What the remesh / SDF-merge dialogs need from the scene, gathered in main.cpp.
struct DialogInfo {
    int   levels_to_bake = 0;    // see snapshot_levels_to_bake: -1 = too big to stash
    int   n_selected     = 0;    // meshes the merge welds
    int   n_unselected   = 0;    // red meshes a subtract merge would carve away
    bool  has_density    = false;
    float merge_progress = 0.0f; // 0..1 while a merge runs
};

// The remesh and SDF-merge prompts and progress panels, drawn as real dialogs in the
// active skin (Modern, DOS, Y2K). Classic keeps its bitmap prompts (ui_overlay.cpp).
// Call at the top level, after draw_modern_ui.
void draw_skin_dialogs(InputState& input, int win_w, int win_h, const DialogInfo& info);

// Menu > Appearance (skin, panel material, motion, viewfinder). Both skins' menus
// call it so Classic can always switch back.
void draw_appearance_menu_items(InputState& input);

// Inside BeginPopup of a popup pushed with a transparent PopupBg: paints the
// panel material under it.
void ui_skin_popup_background();

// Hover note for the menu item just drawn, in the skin's own tooltip chip (the one
// the brush dock uses), parked beside the menu. Classic keeps ImGui's tooltip, which
// is what its islands use. printf-style, like ImGui::SetTooltip.
void ui_skin_menu_tip(const char* fmt, ...) IM_FMTARGS(1);

// A dropdown opened inside such a popup inherits that transparent PopupBg and its
// list floats see-through over the menu. Push this around the Combo to give it the
// skin's popup colour back, fully opaque. Pop with ImGui::PopStyleColor().
void ui_skin_push_dropdown_bg();

// Retopo mode's "Polygon Layers" dock: movable, resizable, always up in mode 0 (the caller gates it).
// Drawn with this file's parts so it wears the live skin; call at the top level.
struct RetopoState;
void draw_retopo_dock(InputState& input, RetopoState& st, int win_w, int win_h);
// Sculpt / Select modes' "Meshes" dock: the scene's meshes as layers. The caller fills
// one row per mesh from the scene; the dock edits name and eye in place (and sets
// input.mesh_layers_changed), and asks for picks and deletes through input.mesh_layer_*.
struct MeshLayerRow {
    uint32_t     id = 0;
    std::string* name = nullptr;     // the mesh's own; empty = unnamed
    const char*  shown = nullptr;    // what the row says: the name, or "Mesh <id>"
    bool*        visible = nullptr;
    bool         active = false, selected = false;
    uint32_t     tris = 0;
    int          level = 0;
    uint32_t     lo_faces = 0;       // its retopo's built faces: 0 = no low-poly yet
};
void draw_mesh_layers_dock(InputState& input, MeshLayerRow* rows, int n, int win_w, int win_h);

// UV mode's panel (key 9): the layout, Unwrap, Re-pack, the meshes, Bake. Drawn instead of
// the retopo dock. One row per mesh, filled by the caller from the scene; the square shows
// the active mesh's layout and the other selected meshes' with it (an atlas when they
// were packed together). Clicks come back through input.mesh_layer_pick / uv_focus.
struct UvMeshRow {
    uint32_t     id = 0;
    const char*  shown = nullptr;    // the mesh's name, or "Mesh <id>"
    RetopoState* st = nullptr;       // its retopo (its layout lives there)
    bool         active = false, selected = false;
};
void draw_uv_dock(InputState& input, RetopoState& st, UvMeshRow* rows, int n, int win_w, int win_h);
// A UV island's colour (the UV panel's, and the model's tint in UV mode).
ImU32 uv_island_color(uint32_t island, float alpha);
