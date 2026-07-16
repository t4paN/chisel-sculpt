#pragma once
#include <cstdint>

struct TextOverlay;
struct InputState;
struct AlphaLibrary;

void draw_quit_dialog(TextOverlay& text, int win_w, int win_h);
void draw_drop_confirm(TextOverlay& text, const char* path, int win_w, int win_h);
void draw_remesh_confirm(TextOverlay& text, int win_w, int win_h);
void draw_remesh_progress(TextOverlay& text, int win_w, int win_h);
void draw_voxel_merge_confirm(TextOverlay& text, int resolution, int n_selected,
                             int n_unselected, bool surface_nets,
                             bool has_density, bool adaptive, int win_w, int win_h);
void draw_voxel_merge_progress(TextOverlay& text, int win_w, int win_h, float progress);
void draw_toolbar(TextOverlay& text, const InputState& input,
                  uint32_t tri_count, uint32_t vert_count, const char* ver,
                  const char* project_path, int win_w, int win_h);
void draw_slider(TextOverlay& text, const InputState& input, int win_w, int win_h);
void draw_notification(TextOverlay& text, InputState& input, int win_w, int win_h);
void draw_fps(TextOverlay& text, float fps, int win_w, int win_h);
void draw_mode_indicator(TextOverlay& text, const char* mode_text, int win_w, int win_h);

// ImGui button islands: brush selection + ops (undo/redo, multires, save/load)
// Returns true if any button was clicked (caller should check input flags).
void draw_button_islands(InputState& input, int win_w, int win_h,
                         const AlphaLibrary* alpha_lib = nullptr);
