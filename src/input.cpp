#include "input.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

InputState::InputState()
    : mouse_x(0), mouse_y(0)
    , prev_mouse_x(0), prev_mouse_y(0)
    , mouse1_down(false), mouse2_down(false), mouse3_down(false)
    , mouse1_just_pressed(false), mouse1_just_released(false)
    , shift_held(false), ctrl_held(false), alt_held(false)
    , current_brush(BrushType::DRAW)
    , smooth_locked(false), subtract_locked(false)
    , brush_size(50.0f), brush_strength(0.5f), brush_hardness(0.5f), brush_spacing(0.25f)
    , per_brush{}
    , slider_mode(SliderMode::NONE)
    , slider_start_x(0), slider_start_y(0), slider_start_value(0), slider_accum(0)
    , toolbar_visible(true)
    , sculpting(false), on_model(false)
    , drag_mode(DragMode::NONE)
    , quit_requested(false)
    , export_dialog_active(false)
    , import_dialog_active(false)
    , save_requested(false)
    , save_as_requested(false)
    , save_dialog_active(false)
    , notification()
    , notification_timer(0.0f)
    , focus_requested(false)
    , snap_view_requested(SnapView::NONE)
    , undo_requested(false)
    , redo_requested(false)
    , fullscreen_toggle_requested(false)
    , is_fullscreen(false)
    , cursor_nx(0), cursor_ny(0), cursor_nz(1.0f)
    , mesh_locked(false)
    , mirror_x(true)
    , autosmooth(true)
    , pressure_enabled(true)
    , facing_threshold(0.0f)
    , fast_normals(false)
    , current_lod(0)
    , subdiv_level(4)
    , level_switch_delta(0)
    , debug_multires_requested(false)
    , debug_stride_cycle_requested(false)
    , debug_pick_vertex_requested(false)
    , show_debug_mesh(false)
    , project_requested(false)
    , remesh_requested(false)
    , remesh_in_progress(false)
    , remesh_confirm_pending(false)
    , voxel_merge_requested(false)
    , voxel_merge_in_progress(false)
    , voxel_merge_confirm_pending(false)
    , voxel_merge_resolution(128)
    , mask_invert_requested(false)
    , mask_clear_requested(false)
    , interaction_mode(InteractionMode::EDIT)
    , delete_mesh_requested(false)
    , enter_pressed(false)
    , key_y_pressed(false)
    , key_n_pressed(false)
{
    for (int i = 0; i < (int)BrushType::COUNT; i++) {
        per_brush[i].strength = brush_strength;
        per_brush[i].hardness = brush_hardness;
    }
    per_brush[(int)BrushType::MASK].strength = 1.0f;      // 100% strength
    per_brush[(int)BrushType::MASK].hardness = 0.64f;     // 64% hardness
}

void InputState::switch_brush(BrushType to) {
    if (to == current_brush) return;
    per_brush[(int)current_brush].strength = brush_strength;
    per_brush[(int)current_brush].hardness = brush_hardness;
    current_brush = to;
    brush_strength = per_brush[(int)to].strength;
    brush_hardness = per_brush[(int)to].hardness;
}

void InputState::clear_smooth_lock() {
    if (!smooth_locked) return;
    per_brush[(int)BrushType::SMOOTH].strength = brush_strength;
    per_brush[(int)BrushType::SMOOTH].hardness = brush_hardness;
    brush_strength = per_brush[(int)current_brush].strength;
    brush_hardness = per_brush[(int)current_brush].hardness;
    smooth_locked = false;
}

void InputState::begin_frame() {
    mouse1_just_pressed = false;
    mouse1_just_released = false;
    enter_pressed = false;
    key_y_pressed = false;
    key_n_pressed = false;
    debug_stride_cycle_requested = false;
    debug_pick_vertex_requested  = false;
}

void InputState::end_frame() {
    prev_mouse_x = mouse_x;
    prev_mouse_y = mouse_y;
}

bool InputState::is_smooth_active() const {
    return smooth_locked || (shift_held && !ctrl_held);
}

bool InputState::is_subtract_active() const {
    return subtract_locked || (ctrl_held && !shift_held);
}

const char* InputState::brush_name() const {
    if (is_smooth_active()) return "Smooth";
    switch (current_brush) {
        case BrushType::DRAW:   return is_subtract_active() ? "Draw (-)"   : "Draw";
        case BrushType::CREASE: return is_subtract_active() ? "Crease (-)" : "Crease";
        case BrushType::PINCH:  return is_subtract_active() ? "Pinch (-)"  : "Pinch";
        case BrushType::MOVE:   return "Move";
        case BrushType::SMOOTH: return "Smooth";
        case BrushType::MASK:   return "Mask";
        default: return "Unknown";
    }
}

// ---- GLFW Callbacks ----

static InputState* g_input = nullptr;
static double g_last_shift_time = 0;
static double g_last_ctrl_time = 0;
static int g_shift_tap_count = 0;
static int g_ctrl_tap_count = 0;

static double g_slider_last_raw_x = 0;

static void cursor_pos_callback(GLFWwindow* w, double x, double y) {
    if (!g_input) return;

    // Handle slider drag: accumulate delta, don't update mouse position
    if (g_input->slider_mode != InputState::SliderMode::NONE) {
        float dx = (float)(x - g_slider_last_raw_x);
        g_slider_last_raw_x = x;
        g_input->slider_accum += dx;

        // Clamp accumulator to valid range so it doesn't overshoot
        switch (g_input->slider_mode) {
            case InputState::SliderMode::SIZE: {
                float scaled = g_input->slider_accum * 0.4f;
                float min_val = 5.0f - g_input->slider_start_value;
                float max_val = 500.0f - g_input->slider_start_value;
                scaled = std::max(min_val, std::min(max_val, scaled));
                g_input->slider_accum = scaled / 0.4f;
                g_input->brush_size = g_input->slider_start_value + scaled;
                break;
            }
            case InputState::SliderMode::STRENGTH: {
                float delta = g_input->slider_accum * 0.001f;
                float min_delta = 0.01f - g_input->slider_start_value;
                float max_delta = 1.0f - g_input->slider_start_value;
                delta = std::max(min_delta, std::min(max_delta, delta));
                g_input->slider_accum = delta / 0.001f;
                g_input->brush_strength = g_input->slider_start_value + delta;
                break;
            }
            case InputState::SliderMode::HARDNESS: {
                float delta = g_input->slider_accum * 0.001f;
                float min_delta = 0.01f - g_input->slider_start_value;
                float max_delta = 1.0f - g_input->slider_start_value;
                delta = std::max(min_delta, std::min(max_delta, delta));
                g_input->slider_accum = delta / 0.001f;
                g_input->brush_hardness = g_input->slider_start_value + delta;
                break;
            }
            case InputState::SliderMode::SPACING: {
                float delta = g_input->slider_accum * 0.001f;
                float min_delta = 0.05f - g_input->slider_start_value;
                float max_delta = 1.0f - g_input->slider_start_value;
                delta = std::max(min_delta, std::min(max_delta, delta));
                g_input->slider_accum = delta / 0.001f;
                g_input->brush_spacing = g_input->slider_start_value + delta;
                break;
            }
            default: break;
        }
        int save_brush = g_input->smooth_locked ? (int)BrushType::SMOOTH : (int)g_input->current_brush;
        g_input->per_brush[save_brush].strength = g_input->brush_strength;
        g_input->per_brush[save_brush].hardness = g_input->brush_hardness;
        // Don't update mouse_x/y — keep cursor visually locked
        return;
    }

    g_input->mouse_x = x;
    g_input->mouse_y = y;
}

static void mouse_button_callback(GLFWwindow* w, int button, int action, int mods) {
    if (!g_input) return;

    // Update modifiers from mouse event too
    g_input->shift_held = (mods & GLFW_MOD_SHIFT) != 0;
    g_input->ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;
    g_input->alt_held = (mods & GLFW_MOD_ALT) != 0;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        bool was_down = g_input->mouse1_down;
        g_input->mouse1_down = (action == GLFW_PRESS);
        if (action == GLFW_PRESS && !was_down) {
            g_input->mouse1_just_pressed = true;
            // Zero delta so the first sculpt/drag frame doesn't inherit pre-click cursor motion.
            g_input->prev_mouse_x = g_input->mouse_x;
            g_input->prev_mouse_y = g_input->mouse_y;
            // Latch drag mode on press
            if (g_input->slider_mode != InputState::SliderMode::NONE) {
                // Slider active, no drag mode
            } else if (g_input->alt_held) {
                g_input->drag_mode = InputState::DragMode::ORBIT;
            } else if (g_input->on_model) {
                g_input->drag_mode = InputState::DragMode::SCULPT;
            } else {
                g_input->drag_mode = InputState::DragMode::ORBIT;
            }
        }
        if (action == GLFW_RELEASE && was_down) {
            g_input->mouse1_just_released = true;
            g_input->drag_mode = InputState::DragMode::NONE;
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_input->mouse2_down = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        g_input->mouse3_down = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) {
            g_input->drag_mode = InputState::DragMode::PAN;
        }
        if (action == GLFW_RELEASE) {
            if (!g_input->mouse1_down)
                g_input->drag_mode = InputState::DragMode::NONE;
        }
    }
}

static void scroll_callback(GLFWwindow* w, double xoff, double yoff) {
    if (!g_input) return;
    // Scroll = zoom (stored externally, main loop reads it)
    // We'll use a simple approach: store scroll delta
    // Main loop handles camera zoom
    // For now, using a global to pass scroll delta
}

// Global scroll accumulator (read and cleared each frame in main loop)
static float g_scroll_accum = 0;

float input_consume_scroll() {
    float s = g_scroll_accum;
    g_scroll_accum = 0;
    return s;
}

static void scroll_cb_internal(GLFWwindow* w, double xoff, double yoff) {
    g_scroll_accum += (float)yoff;
}

static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (!g_input) return;
    double now = glfwGetTime();

    // Track modifiers
    g_input->shift_held = (mods & GLFW_MOD_SHIFT) != 0;
    g_input->ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;
    g_input->alt_held = (mods & GLFW_MOD_ALT) != 0;

    // Modal UI steals keypresses from sculpting. ImGui file dialogs handle
    // their own keyboard input via ImGui IO — only ESC passes through here
    // to close the dialog flag. Y/N dialogs (quit, remesh) still route keys.
    bool modal_active = g_input->quit_requested
                     || g_input->export_dialog_active
                     || g_input->import_dialog_active
                     || g_input->save_dialog_active
                     || g_input->remesh_confirm_pending
                     || g_input->voxel_merge_confirm_pending;
    if (modal_active) {
        bool is_yn_dialog = g_input->quit_requested
                         || g_input->remesh_confirm_pending
                         || g_input->voxel_merge_confirm_pending;
        // The voxel-merge dialog also takes [ / ] to nudge the resolution.
        bool res_keys = g_input->voxel_merge_confirm_pending
                     && (key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET);
        bool allow = (key == GLFW_KEY_ESCAPE)
                  || (is_yn_dialog && (key == GLFW_KEY_Y || key == GLFW_KEY_N))
                  || res_keys;
        if (!allow) return;
    }

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_D:
                if (g_input->ctrl_held && !g_input->shift_held && g_input->mesh_locked) {
                    g_input->level_switch_delta = +1;    // Ctrl+D: level up
                } else if (g_input->shift_held && !g_input->ctrl_held && g_input->mesh_locked) {
                    g_input->level_switch_delta = -1;    // Shift+D: level down
                } else if (!g_input->shift_held && !g_input->ctrl_held) {
                    g_input->clear_smooth_lock();
                    g_input->switch_brush(BrushType::DRAW);
                    g_input->subtract_locked = false;
                }
                break;

            case GLFW_KEY_C:
                g_input->clear_smooth_lock();
                g_input->switch_brush(BrushType::CREASE);
                g_input->subtract_locked = false;
                break;

            case GLFW_KEY_V:
                g_input->clear_smooth_lock();
                g_input->switch_brush(BrushType::PINCH);
                g_input->subtract_locked = false;
                break;

            case GLFW_KEY_M:
                g_input->clear_smooth_lock();
                g_input->switch_brush(BrushType::MASK);
                g_input->subtract_locked = false;
                break;

            case GLFW_KEY_Q:
                // Cycle brush backward
                {
                    g_input->clear_smooth_lock();
                    int b = (int)g_input->current_brush - 1;
                    if (b < 0) b = (int)BrushType::COUNT - 1;
                    g_input->switch_brush((BrushType)b);
                }
                break;

            case GLFW_KEY_E:
                if (g_input->ctrl_held) {
                    g_input->export_dialog_active = true;
                    g_input->quit_requested = false;
                } else {
                    // E: Cycle brush forward
                    g_input->clear_smooth_lock();
                    int b = (int)g_input->current_brush + 1;
                    if (b >= (int)BrushType::COUNT) b = 0;
                    g_input->switch_brush((BrushType)b);
                }
                break;

            case GLFW_KEY_S:
                if (g_input->ctrl_held && g_input->shift_held) {
                    g_input->save_as_requested = true;
                } else if (g_input->ctrl_held) {
                    g_input->save_requested = true;
                } else {
                    g_input->slider_mode = InputState::SliderMode::SIZE;
                    g_input->slider_start_x = g_input->mouse_x;
                    g_input->slider_start_y = g_input->mouse_y;
                    g_input->slider_start_value = g_input->brush_size;
                    g_input->slider_accum = 0;
                    g_slider_last_raw_x = g_input->mouse_x;
                }
                break;

            case GLFW_KEY_W:
                g_input->slider_mode = InputState::SliderMode::STRENGTH;
                g_input->slider_start_x = g_input->mouse_x;
                g_input->slider_start_y = g_input->mouse_y;
                g_input->slider_start_value = g_input->brush_strength;
                g_input->slider_accum = 0;
                g_slider_last_raw_x = g_input->mouse_x;
                break;

            case GLFW_KEY_A:
                if (g_input->ctrl_held && !g_input->shift_held) {
                    // Ctrl+A: clear mask (any brush)
                    g_input->mask_clear_requested = true;
                } else {
                    // A: hardness slider
                    g_input->slider_mode = InputState::SliderMode::HARDNESS;
                    g_input->slider_start_x = g_input->mouse_x;
                    g_input->slider_start_y = g_input->mouse_y;
                    g_input->slider_start_value = g_input->brush_hardness;
                    g_input->slider_accum = 0;
                    g_slider_last_raw_x = g_input->mouse_x;
                }
                break;

            case GLFW_KEY_O:
                if (g_input->ctrl_held) {
                    g_input->import_dialog_active = true;
                    g_input->quit_requested = false;
                    g_input->export_dialog_active = false;
                } else {
                    g_input->slider_mode = InputState::SliderMode::SPACING;
                    g_input->slider_start_x = g_input->mouse_x;
                    g_input->slider_start_y = g_input->mouse_y;
                    g_input->slider_start_value = g_input->brush_spacing;
                    g_input->slider_accum = 0;
                    g_slider_last_raw_x = g_input->mouse_x;
                }
                break;

            case GLFW_KEY_SPACE:
                g_input->fullscreen_toggle_requested = true;
                break;

            case GLFW_KEY_X:
                g_input->mirror_x = !g_input->mirror_x;
                break;

            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT:
                // Double-tap shift detection
                if (now - g_last_shift_time < 0.3) {
                    if (!g_input->smooth_locked) {
                        // Save current brush settings, load smooth's
                        g_input->per_brush[(int)g_input->current_brush].strength = g_input->brush_strength;
                        g_input->per_brush[(int)g_input->current_brush].hardness = g_input->brush_hardness;
                        g_input->brush_strength = g_input->per_brush[(int)BrushType::SMOOTH].strength;
                        g_input->brush_hardness = g_input->per_brush[(int)BrushType::SMOOTH].hardness;
                        g_input->smooth_locked = true;
                    } else {
                        g_input->clear_smooth_lock();
                    }
                    g_shift_tap_count = 0;
                }
                g_last_shift_time = now;
                break;

            case GLFW_KEY_LEFT_CONTROL:
            case GLFW_KEY_RIGHT_CONTROL:
                // Double-tap ctrl detection
                if (now - g_last_ctrl_time < 0.3) {
                    g_input->subtract_locked = !g_input->subtract_locked;
                    g_ctrl_tap_count = 0;
                }
                g_last_ctrl_time = now;
                break;

            case GLFW_KEY_SLASH:
                if (g_input->mesh_locked &&
                    !g_input->quit_requested &&
                    !g_input->export_dialog_active &&
                    !g_input->import_dialog_active &&
                    !g_input->remesh_confirm_pending &&
                    !g_input->remesh_in_progress) {
                    g_input->remesh_confirm_pending = true;
                }
                break;

            case GLFW_KEY_J:
                // Voxel merge (join selection for print). Same gating as remesh.
                if (g_input->mesh_locked &&
                    !g_input->quit_requested &&
                    !g_input->export_dialog_active &&
                    !g_input->import_dialog_active &&
                    !g_input->save_dialog_active &&
                    !g_input->remesh_confirm_pending &&
                    !g_input->voxel_merge_confirm_pending &&
                    !g_input->voxel_merge_in_progress) {
                    g_input->voxel_merge_confirm_pending = true;
                }
                break;

            case GLFW_KEY_LEFT_BRACKET:
                if (g_input->voxel_merge_confirm_pending &&
                    g_input->voxel_merge_resolution > 64)
                    g_input->voxel_merge_resolution -= 32;
                break;
            case GLFW_KEY_RIGHT_BRACKET:
                if (g_input->voxel_merge_confirm_pending &&
                    g_input->voxel_merge_resolution < 256)
                    g_input->voxel_merge_resolution += 32;
                break;

            case GLFW_KEY_ESCAPE:
                if (g_input->voxel_merge_confirm_pending) {
                    g_input->voxel_merge_confirm_pending = false;
                } else if (g_input->remesh_confirm_pending) {
                    g_input->remesh_confirm_pending = false;
                } else if (g_input->export_dialog_active) {
                    g_input->export_dialog_active = false;
                } else if (g_input->import_dialog_active) {
                    g_input->import_dialog_active = false;
                } else if (g_input->save_dialog_active) {
                    g_input->save_dialog_active = false;
                } else if (g_input->quit_requested) {
                    g_input->quit_requested = false;
                } else {
                    g_input->quit_requested = true;
                }
                break;

            case GLFW_KEY_F:
                g_input->focus_requested = true;
                break;

            case GLFW_KEY_F1:
                g_input->snap_view_requested = InputState::SnapView::FRONT;
                break;
            case GLFW_KEY_F2:
                g_input->snap_view_requested = InputState::SnapView::SIDE;
                break;
            case GLFW_KEY_F3:
                g_input->snap_view_requested = InputState::SnapView::TOP;
                break;
            case GLFW_KEY_F9:
                g_input->debug_stride_cycle_requested = true;
                break;
            case GLFW_KEY_F10:
                g_input->debug_pick_vertex_requested = true;
                break;
            case GLFW_KEY_F12:
                g_input->debug_multires_requested = true;
                break;

            case GLFW_KEY_P:
                if (g_input->mesh_locked &&
                    !g_input->shift_held && !g_input->ctrl_held && !g_input->alt_held) {
                    g_input->project_requested = true;
                }
                break;

            case GLFW_KEY_Z:
                if (g_input->ctrl_held) {
                    if (g_input->shift_held) g_input->redo_requested = true;
                    else g_input->undo_requested = true;
                }
                break;

            case GLFW_KEY_I:
                if (g_input->ctrl_held && !g_input->shift_held) {
                    // Ctrl+I: invert mask
                    g_input->mask_invert_requested = true;
                }
                break;

            case GLFW_KEY_G:
                g_input->clear_smooth_lock();
                g_input->switch_brush(BrushType::MOVE);
                g_input->subtract_locked = false;
                break;

            case GLFW_KEY_ENTER:
                g_input->enter_pressed = true;
                break;

            case GLFW_KEY_BACKSPACE:
                break;

            case GLFW_KEY_Y:
                if (g_input->voxel_merge_confirm_pending) {
                    g_input->voxel_merge_confirm_pending = false;
                    g_input->voxel_merge_requested = true;
                } else if (g_input->remesh_confirm_pending) {
                    g_input->remesh_confirm_pending = false;
                    g_input->remesh_requested = true;
                } else if (g_input->quit_requested) {
                    glfwSetWindowShouldClose(w, GLFW_TRUE);
                } else {
                    g_input->key_y_pressed = true;
                    if (g_input->interaction_mode != InputState::InteractionMode::INSERT)
                        g_input->show_debug_mesh = !g_input->show_debug_mesh;
                }
                break;

            case GLFW_KEY_B:
                g_input->autosmooth = !g_input->autosmooth;
                snprintf(g_input->notification, sizeof(g_input->notification),
                         g_input->autosmooth ? "Autosmooth ON" : "Autosmooth OFF");
                g_input->notification_timer = 1.5f;
                break;

            case GLFW_KEY_K:
                g_input->pressure_enabled = !g_input->pressure_enabled;
                snprintf(g_input->notification, sizeof(g_input->notification),
                         g_input->pressure_enabled ? "Pen pressure ON" : "Pen pressure OFF");
                g_input->notification_timer = 1.5f;
                break;

            case GLFW_KEY_N:
                if (g_input->voxel_merge_confirm_pending) {
                    g_input->voxel_merge_confirm_pending = false;
                } else if (g_input->remesh_confirm_pending) {
                    g_input->remesh_confirm_pending = false;
                } else if (g_input->quit_requested) {
                    g_input->quit_requested = false;
                } else {
                    g_input->key_n_pressed = true;
                    if (g_input->interaction_mode != InputState::InteractionMode::INSERT)
                        g_input->fast_normals = !g_input->fast_normals;
                }
                break;

            case GLFW_KEY_1:
                g_input->interaction_mode = InputState::InteractionMode::EDIT;
                snprintf(g_input->notification, sizeof(g_input->notification), "Edit");
                g_input->notification_timer = 1.0f;
                break;
            case GLFW_KEY_2:
                g_input->interaction_mode = InputState::InteractionMode::INSERT;
                snprintf(g_input->notification, sizeof(g_input->notification), "Insert mode");
                g_input->notification_timer = 1.5f;
                break;
            case GLFW_KEY_3:
                g_input->interaction_mode = InputState::InteractionMode::SELECT;
                snprintf(g_input->notification, sizeof(g_input->notification), "Select");
                g_input->notification_timer = 1.0f;
                break;
            case GLFW_KEY_DELETE:
                if (!g_input->ctrl_held && !g_input->shift_held)
                    g_input->delete_mesh_requested = true;
                break;

        }
    }

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_LEFT_BRACKET:
                g_input->facing_threshold = std::max(0.0f, g_input->facing_threshold - 0.01f);
                snprintf(g_input->notification, sizeof(g_input->notification),
                         "Facing threshold: %.2f", g_input->facing_threshold);
                g_input->notification_timer = 1.5f;
                break;
            case GLFW_KEY_RIGHT_BRACKET:
                g_input->facing_threshold = std::min(1.0f, g_input->facing_threshold + 0.01f);
                snprintf(g_input->notification, sizeof(g_input->notification),
                         "Facing threshold: %.2f", g_input->facing_threshold);
                g_input->notification_timer = 1.5f;
                break;
        }
    }

    if (action == GLFW_RELEASE) {
        switch (key) {
            case GLFW_KEY_S:
            case GLFW_KEY_W:
            case GLFW_KEY_A:
            case GLFW_KEY_O:
                if (g_input->slider_mode != InputState::SliderMode::NONE) {
                    // Warp cursor back to where slider started
                    glfwSetCursorPos(w, g_input->slider_start_x, g_input->slider_start_y);
                    g_input->mouse_x = g_input->slider_start_x;
                    g_input->mouse_y = g_input->slider_start_y;
                    g_input->prev_mouse_x = g_input->slider_start_x;
                    g_input->prev_mouse_y = g_input->slider_start_y;
                    g_input->slider_mode = InputState::SliderMode::NONE;
                }
                break;
        }
    }
}

void setup_input_callbacks(GLFWwindow* window, InputState* state) {
    g_input = state;
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_cb_internal);
    glfwSetKeyCallback(window, key_callback);
}

void char_callback(GLFWwindow* w, unsigned int codepoint) {
    (void)w;
    (void)codepoint;
}

void setup_char_callback(GLFWwindow* window) {
    glfwSetCharCallback(window, char_callback);
}
