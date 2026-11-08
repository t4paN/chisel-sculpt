#pragma once
#include <cstdint>

struct TextOverlay;
struct InputState;
struct StippleSettings;
struct AlphaLibrary;
class Scene;

void draw_quit_dialog(TextOverlay& text, int win_w, int win_h);
void draw_drop_confirm(TextOverlay& text, const char* path, int win_w, int win_h);
// levels_to_bake: subdivision levels the rescue snapshot will bake down first to
// get the scene under its triangle cap (0 = none, -1 = cannot fit, no rescue).
// Shown in the prompt because baking is destructive and the Y is the consent.
void draw_remesh_confirm(TextOverlay& text, int levels_to_bake, bool mirror, int win_w, int win_h);
void draw_remesh_progress(TextOverlay& text, int win_w, int win_h, float progress);
void draw_voxel_merge_confirm(TextOverlay& text, int resolution, int close_holes, int n_selected,
                             int n_unselected, bool surface_nets,
                             bool has_density, bool adaptive, int levels_to_bake,
                             int win_w, int win_h);
void draw_voxel_merge_progress(TextOverlay& text, int win_w, int win_h, float progress);
// level is the multires stack's LIVE editing level, not InputState::subdiv_level —
// that one is the opening sphere's base-cage subdivision, is written into the save
// file, and never moves, so the readout sat frozen on it.
void draw_toolbar(TextOverlay& text, const InputState& input,
                  uint32_t tri_count, uint32_t vert_count, const char* ver,
                  int level, const char* project_path, int win_w, int win_h);
void draw_slider(TextOverlay& text, const InputState& input, int win_w, int win_h);
// Confirm for the burger menu's "delete highest subdiv" — destructive, wipes undo.
void draw_drop_level_confirm(TextOverlay& text, int level, int win_w, int win_h);
void draw_reset_confirm(TextOverlay& text, int win_w, int win_h);
void draw_notification(TextOverlay& text, InputState& input, int win_w, int win_h);
// Red flash for the one case where Chisel overrides a setting the user chose: the mesh
// cannot supply mirror pairs, so Topological was switched back to World Space. Sits at
// two thirds down, clear of draw_notification's line near the bottom, so the two can be
// on screen together — flipping the mode fires both at once.
void draw_mirror_unavailable(TextOverlay& text, InputState& input, int win_w, int win_h);
void draw_fps(TextOverlay& text, float fps, int win_w, int win_h);
void draw_mode_indicator(TextOverlay& text, const char* mode_text, int win_w, int win_h);

// Multires state the burger menu needs to label/enable its "delete highest
// subdiv" item. lmax = base_level + layer count; locked false = no stack yet.
struct MultiresInfo {
    bool locked = false;
    int  base_level = 0;
    int  lmax = 0;
};

// ImGui button islands: brush selection + ops (undo/redo, multires, save/load)
// Returns true if any button was clicked (caller should check input flags).
void draw_button_islands(InputState& input, int win_w, int win_h,
                         const AlphaLibrary* alpha_lib = nullptr,
                         MultiresInfo mres = MultiresInfo());

// Shared by both skins. The burger menu's body (call between BeginPopup/EndPopup;
// sync_tabs = true on the frame the menu opened, see the profile tabs), and the
// shortcut card (call inside the window that opens "##helppopup").
void draw_settings_menu_items(InputState& input, MultiresInfo mres, bool sync_tabs);
void draw_help_popup(InputState& input, int win_w, int win_h);
// Burger menu -> Key bindings: the centred rebinding window. Call once per frame at
// the top level (outside any ImGui window), after the skin's UI.
void draw_keybind_window(InputState& input, int win_w, int win_h);
// Brush properties (burger menu): same frame as the key bindings window.
void draw_brush_props_window(InputState& input, int win_w, int win_h);
// Stipple's knobs (rotation / size / depth jitter, Scatter + copies + copy size) as
// plain ImGui widgets, shared by every skin's right-click popup and the modern tool
// options dock. `row` lays them out on one line (the dock); otherwise stacked (popup).
// Returns true if anything changed.
// `paint` is paint mode's Stipple: its depth jitter is opacity, so the label says so.
bool draw_stipple_options(StippleSettings& st, bool row, bool paint = false);
// Paint layers docker (paint mode, colour target). Shared by every skin.
void draw_paint_layers_panel(InputState& input, Scene& scene, int win_w, int win_h);

// Eyedropper cursor while the colour picker is armed (tip at x,y; dimmed when
// the cursor is off the model). ImGui foreground draw list — identical on all
// backends, replaces the brush ring for that state.
void draw_pick_cursor(float x, float y, bool on_model, const float* swatch = nullptr);
void draw_bucket_cursor(float x, float y, bool on_model);
void draw_gradient_drag(float ax, float ay, float bx, float by);

// Pose brush hinge, drawn through the mesh (the handles sit inside it): the line,
// the anchor as a ring with the limb's girth round it, the lever as a dot. The
// handle under the cursor (hover: kPoseAnchor / kPoseLever, -1 none) is lit. With
// mirror, the twin the swing will also move shows as a ghost. Background draw list,
// so the docks stay on top. And the pose cursor: a dot.
struct PoseHinge;
struct Camera;
void draw_pose_hinge(const PoseHinge& hinge, const Camera& cam, int win_w, int win_h,
                     int hover, bool mirror);
void draw_pose_cursor(float x, float y, bool on_target);

// Retopo mode (key 0). The overlay draws every stroke as a translucent band at its
// width (donuts teal, strips amber; mirrored twins ghosted when mirror is on) and the
// built quads as a wireframe, hiding what faces away or sits behind the sculpt. The
// depth test reads the plane cache, so pass depth_fresh = false while the camera
// moves (only the facing test runs then). Background draw list: docks stay on top.
// Shared by every skin. (The dock that goes with it is ui_skin.h's draw_retopo_dock.)
struct RetopoState;
struct Renderer;
// uv: UV mode (faces tinted by island, in mesh `mesh`'s colours).
void draw_retopo_overlay(const RetopoState& st, Renderer& renderer, const Camera& cam,
                         int win_w, int win_h, bool depth_fresh, bool mirror, bool uv = false, uint32_t mesh = 0);
// UV mode: every mesh the square shows, marked in the 3D view in its own colour: corner
// brackets round its low on screen, and the low's outline (GPU, hidden where the scene
// is in front; only with `outline`), px wide. Off unless Viewfinder corners is on.
// Sculpt and Select modes' viewfinder: brackets round the active mesh (with `selection`,
// every selected one too) in its colour, only while two or more meshes are on screen (one
// alone needs no telling apart). No outline.
class Scene;
void draw_scene_brackets(const Scene& scene, const Camera& cam, int win_w, int win_h, float px, bool selection);
struct UvMeshRow;
void draw_uv_mesh_marks(const UvMeshRow* rows, int n, Renderer& renderer, const Camera& cam,
                        int win_w, int win_h, float px, bool outline);
