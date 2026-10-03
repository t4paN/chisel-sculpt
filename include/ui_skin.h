#pragma once
#include <cstdint>
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
// through to solid, plus optional springy motion and viewfinder corners. The
// classic islands + bitmap HUD stay available as InputState::ui_classic.

enum class UiFont { Sans, SansSemibold, Mono, MonoMedium };

// Once, after ImGui::CreateContext and before the first NewFrame.
void ui_skin_load_fonts();
// Every frame, before ImGui::NewFrame: switches font + style when the skin changes.
void ui_skin_begin_frame(const InputState& input);
// The modern skin's faces; nullptr while the classic skin is active.
ImFont* ui_skin_font(UiFont f);

// Backdrop blur for this frame (sigma in px, saturation). blur_px == 0 means the
// material is solid enough that no blur pass should run.
void ui_skin_backdrop_params(const InputState& input, float* blur_px, float* saturate);

struct SkinStats {
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

// Menu > Appearance (skin, panel material, motion, viewfinder). Both skins' menus
// call it so Classic can always switch back.
void draw_appearance_menu_items(InputState& input);

// Inside BeginPopup of a popup pushed with a transparent PopupBg: paints the
// panel material under it.
void ui_skin_popup_background();
