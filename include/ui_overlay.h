#pragma once
#include <cstdint>

struct TextOverlay;
struct InputState;
struct AlphaLibrary;

void draw_quit_dialog(TextOverlay& text, int win_w, int win_h);
void draw_drop_confirm(TextOverlay& text, const char* path, int win_w, int win_h);
// can_snapshot false = the scene is too big to stash a rescue copy of, so the
// prompt warns to save first. The op still runs if the user says yes.
void draw_remesh_confirm(TextOverlay& text, bool can_snapshot, int win_w, int win_h);
void draw_remesh_progress(TextOverlay& text, int win_w, int win_h);
void draw_voxel_merge_confirm(TextOverlay& text, int resolution, int n_selected,
                             int n_unselected, bool surface_nets,
                             bool has_density, bool adaptive, bool can_snapshot,
                             int win_w, int win_h);
void draw_voxel_merge_progress(TextOverlay& text, int win_w, int win_h, float progress);
void draw_toolbar(TextOverlay& text, const InputState& input,
                  uint32_t tri_count, uint32_t vert_count, const char* ver,
                  const char* project_path, int win_w, int win_h);
void draw_slider(TextOverlay& text, const InputState& input, int win_w, int win_h);
// Confirm for the burger menu's "delete highest subdiv" — destructive, wipes undo.
void draw_drop_level_confirm(TextOverlay& text, int level, int win_w, int win_h);
void draw_notification(TextOverlay& text, InputState& input, int win_w, int win_h);
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

// Eyedropper cursor while the colour picker is armed (tip at x,y; dimmed when
// the cursor is off the model). ImGui foreground draw list — identical on all
// backends, replaces the brush ring for that state.
void draw_pick_cursor(float x, float y, bool on_model);
