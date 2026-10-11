#include "input.h"
#include "debug_console.h"
#include "tablet.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <algorithm>
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/html5.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#endif

// ---- Rebindable shortcuts ----------------------------------------------------------

namespace {
constexpr int C = GLFW_MOD_CONTROL, S = GLFW_MOD_SHIFT;
const KeyActionInfo kKeyActions[(int)KeyAction::COUNT] = {
    {"mode_sculpt",   "Modes",   "Sculpt mode",         nullptr, {GLFW_KEY_1, 0}},
    {"mode_insert",   "Modes",   "Insert mode",         nullptr, {GLFW_KEY_2, 0}},
    {"mode_select",   "Modes",   "Select mode",         nullptr, {GLFW_KEY_3, 0}},
    {"mode_paint",    "Modes",   "Paint mode",          nullptr, {GLFW_KEY_4, 0}},
    {"brush_draw",    "Brushes", "Draw",                nullptr, {GLFW_KEY_D, 0}},
    {"brush_clay",    "Brushes", "Clay",                nullptr, {GLFW_KEY_T, 0}},
    {"brush_inflate", "Brushes", "Inflate",             nullptr, {GLFW_KEY_I, 0}},
    {"brush_crease",  "Brushes", "Crease",              "In Paint mode: colour picker", {GLFW_KEY_C, 0}},
    {"brush_pinch",   "Brushes", "Pinch",               nullptr, {GLFW_KEY_V, 0}},
    {"brush_move",    "Brushes", "Move",                nullptr, {GLFW_KEY_G, 0}},
    {"brush_limb",    "Brushes", "Limb",                nullptr, {GLFW_KEY_H, 0}},
    {"brush_mask",    "Brushes", "Mask",                nullptr, {GLFW_KEY_M, 0}},
    {"brush_prev",    "Brushes", "Previous alpha",
        "Draw and Mask. In Paint mode: swap colours. In Select mode: hold to spin the selection.", {GLFW_KEY_Q, 0}},
    {"brush_next",    "Brushes", "Next alpha",
        "Draw and Mask. In Paint mode: swap colours. In Select mode: hold to spin the selection.", {GLFW_KEY_E, 0}},
    {"hold_size",     "Brush feel", "Size",             "Hold and drag", {GLFW_KEY_S, 0}},
    {"hold_strength", "Brush feel", "Strength",         "Hold and drag", {GLFW_KEY_W, 0}},
    {"hold_hardness", "Brush feel", "Hardness",         "Hold and drag", {GLFW_KEY_A, 0}},
    {"hold_spacing",  "Brush feel", "Spacing",          "Hold and drag", {GLFW_KEY_O, 0}},
    {"size_down",     "Brush feel", "Size down",        "Repeats while held", {GLFW_KEY_LEFT_BRACKET, 0}},
    {"size_up",       "Brush feel", "Size up",          "Repeats while held", {GLFW_KEY_RIGHT_BRACKET, 0}},
    {"autosmooth",    "Brush feel", "Autosmooth",       nullptr, {GLFW_KEY_B, 0}},
    {"level_up",      "Shape",   "Subdivision up",      nullptr, {GLFW_KEY_D, C}},
    {"level_down",    "Shape",   "Subdivision down",    nullptr, {GLFW_KEY_D, S}},
    {"project",       "Shape",   "Project detail down", nullptr, {GLFW_KEY_P, 0}},
    {"mirror",        "Shape",   "Mirror X",            nullptr, {GLFW_KEY_X, 0}},
    {"remesh",        "Shape",   "Remesh",              nullptr, {GLFW_KEY_SLASH, 0}},
    {"merge",         "Shape",   "SDF merge",           nullptr, {GLFW_KEY_J, 0}},
    {"delete_mesh",   "Shape",   "Delete object",       nullptr, {GLFW_KEY_DELETE, 0}},
    {"mask_invert",   "Mask",    "Invert mask",         nullptr, {GLFW_KEY_I, C}},
    {"mask_clear",    "Mask",    "Clear mask",          nullptr, {GLFW_KEY_A, C}},
    {"focus",         "View",    "Focus",               nullptr, {GLFW_KEY_F, 0}},
    {"view_front",    "View",    "Front view",          nullptr, {GLFW_KEY_F1, 0}},
    {"view_side",     "View",    "Side view",           nullptr, {GLFW_KEY_F2, 0}},
    {"view_top",      "View",    "Top view",            nullptr, {GLFW_KEY_F3, 0}},
    {"fullscreen",    "View",    "Fullscreen",          nullptr, {GLFW_KEY_TAB, 0}},
    {"wireframe",     "View",    "Wireframe overlay",   nullptr, {GLFW_KEY_Y, 0}},
    {"fast_normals",  "View",    "Fast normals",        nullptr, {GLFW_KEY_N, 0}},
    {"undo",          "Files",   "Undo",                nullptr, {GLFW_KEY_Z, C}},
    {"redo",          "Files",   "Redo",                nullptr, {GLFW_KEY_Z, C | S}},
    {"save",          "Files",   "Save",                nullptr, {GLFW_KEY_S, C}},
    {"save_as",       "Files",   "Save as",             nullptr, {GLFW_KEY_S, C | S}},
    {"open",          "Files",   "Open",                nullptr, {GLFW_KEY_O, C}},
    {"export",        "Files",   "Export",              nullptr, {GLFW_KEY_E, C}},
};
}  // namespace

const KeyActionInfo& key_action_info(KeyAction a) { return kKeyActions[(int)a]; }

// US names, matching the key's place on the keyboard: GLFW codes are layout-independent
// for letters and digits, and glfwGetKeyName would hand back 'δ' under a Greek layout
// or nothing at all for Tab.
static const char* key_name(int key) {
    static char one[2];
    if ((key >= GLFW_KEY_A && key <= GLFW_KEY_Z) || (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)) {
        one[0] = (char)key; one[1] = '\0';
        return one;
    }
    if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F25) {
        static char f[4];
        std::snprintf(f, sizeof(f), "F%d", key - GLFW_KEY_F1 + 1);
        return f;
    }
    if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_9) {
        static char kp[8];
        std::snprintf(kp, sizeof(kp), "Num%d", key - GLFW_KEY_KP_0);
        return kp;
    }
    switch (key) {
        case GLFW_KEY_SPACE:         return "Space";
        case GLFW_KEY_APOSTROPHE:    return "'";
        case GLFW_KEY_COMMA:         return ",";
        case GLFW_KEY_MINUS:         return "-";
        case GLFW_KEY_PERIOD:        return ".";
        case GLFW_KEY_SLASH:         return "/";
        case GLFW_KEY_SEMICOLON:     return ";";
        case GLFW_KEY_EQUAL:         return "=";
        case GLFW_KEY_LEFT_BRACKET:  return "[";
        case GLFW_KEY_BACKSLASH:     return "\\";
        case GLFW_KEY_RIGHT_BRACKET: return "]";
        case GLFW_KEY_GRAVE_ACCENT:  return "`";
        case GLFW_KEY_ESCAPE:        return "Esc";
        case GLFW_KEY_ENTER:         return "Enter";
        case GLFW_KEY_TAB:           return "Tab";
        case GLFW_KEY_BACKSPACE:     return "Backspace";
        case GLFW_KEY_INSERT:        return "Insert";
        case GLFW_KEY_DELETE:        return "Del";
        case GLFW_KEY_RIGHT:         return "Right";
        case GLFW_KEY_LEFT:          return "Left";
        case GLFW_KEY_DOWN:          return "Down";
        case GLFW_KEY_UP:            return "Up";
        case GLFW_KEY_PAGE_UP:       return "PgUp";
        case GLFW_KEY_PAGE_DOWN:     return "PgDn";
        case GLFW_KEY_HOME:          return "Home";
        case GLFW_KEY_END:           return "End";
        case GLFW_KEY_CAPS_LOCK:     return "CapsLock";
        case GLFW_KEY_PAUSE:         return "Pause";
        case GLFW_KEY_KP_DECIMAL:    return "Num.";
        case GLFW_KEY_KP_DIVIDE:     return "Num/";
        case GLFW_KEY_KP_MULTIPLY:   return "Num*";
        case GLFW_KEY_KP_SUBTRACT:   return "Num-";
        case GLFW_KEY_KP_ADD:        return "Num+";
        case GLFW_KEY_KP_ENTER:      return "NumEnter";
        case GLFW_KEY_MENU:          return "Menu";
        default:                     return "?";
    }
}

const char* format_keybind(const KeyBind& b, char* buf, int n) {
    if (!b.bound()) { std::snprintf(buf, n, "???"); return buf; }
    std::snprintf(buf, n, "%s%s%s%s",
                  (b.mods & GLFW_MOD_CONTROL) ? "Ctrl+"  : "",
                  (b.mods & GLFW_MOD_ALT)     ? "Alt+"   : "",
                  (b.mods & GLFW_MOD_SHIFT)   ? "Shift+" : "",
                  key_name(b.key));
    return buf;
}

const char* keybind_reserved(int key) {
    switch (key) {
        case GLFW_KEY_ESCAPE:       return "Esc closes dialogs";
        case GLFW_KEY_GRAVE_ACCENT: return "` is the console";
        case GLFW_KEY_F9: case GLFW_KEY_F10: case GLFW_KEY_F12:
        case GLFW_KEY_PAGE_UP: case GLFW_KEY_PAGE_DOWN:
                                    return "that's a debug key";
        default:                    return nullptr;
    }
}

const char* InputState::key_label(KeyAction a) const {
    static char bufs[(int)KeyAction::COUNT][32];
    return format_keybind(keybinds[(int)a], bufs[(int)a], (int)sizeof(bufs[0]));
}

InputState::InputState()
    : mouse_x(0), mouse_y(0)
    , prev_mouse_x(0), prev_mouse_y(0)
    , mouse1_down(false), mouse2_down(false), mouse3_down(false)
    , mouse1_just_pressed(false), mouse1_just_released(false)
    , shift_held(false), ctrl_held(false), alt_held(false)
    , current_brush(BrushType::DRAW)
    , smooth_locked(false), subtract_locked(false)
    // 70, not the historical 50: since 2026-08-17 the size is no longer restored from
    // the last session, so this number is what every session actually opens on rather
    // than a one-time first-run value — worth it being a comfortable working size.
    , brush_size(70.0f), brush_strength(0.5f), brush_hardness(0.5f), brush_spacing(0.25f)
    , per_brush{}
    , active_profile(InputProfile::MOUSE)
    , profiles{}
    , settings_menu_open(false)
    , help_popup_open(false)
    , help_seen(false)
    , slider_mode(SliderMode::NONE)
    , slider_start_x(0), slider_start_y(0), slider_start_value(0), slider_accum(0)
    , slider_slot(BrushType::DRAW)
    , toolbar_visible(true)
    , sculpting(false), on_model(false)
    , drag_mode(DragMode::NONE)
    , quit_requested(false)
    , export_dialog_active(false)
    , import_dialog_active(false)
    , import_append(false)
    , save_requested(false)
    , save_as_requested(false)
    , save_incremental_requested(false)
    , save_dialog_active(false)
    , drop_confirm_pending(false)
    , drop_open_requested(false)
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
    , mirror_topological(false)
    , autosmooth(false)
    , max_effect(0.6f)          // mouse default; the tablet profile is raised to 1.0 below
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
    , voxel_merge_mirror(false)
    , voxel_merge_surface_nets(false)
    , voxel_merge_subtract(false)
    , voxel_merge_adaptive(true)
    , voxel_merge_resolution(128)
    , mask_invert_requested(false)
    , mask_clear_requested(false)
    , interaction_mode(InteractionMode::EDIT)
    , insert_shape(InsertShape::SPHERE)
    , sphere_kind(SphereKind::ICO)
    , delete_mesh_requested(false)
    , enter_pressed(false)
    , key_y_pressed(false)
    , key_n_pressed(false)
{
    for (int i = 0; i < (int)BrushType::COUNT; i++) {
        per_brush[i].strength = brush_strength;
        per_brush[i].hardness = brush_hardness;
        per_brush[i].spacing  = brush_spacing;
    }
    seed_brush_sizes();
    per_brush[(int)BrushType::MASK].strength = 1.0f;      // 100% strength
    per_brush[(int)BrushType::MASK].hardness = 0.64f;     // 64% hardness
    per_brush[(int)BrushType::PAINT].strength = 0.5f;     // soft build-up by default
    per_brush[(int)BrushType::PAINT].hardness = 0.5f;
    // Clay's hardness is the square stamp's edge feather (clay_square in
    // draw_accum): 1 = crisp chisel rim, lower melts the rim toward the centre.
    // 0.3 = generous feather by default (user-tuned; crisper felt jittery).
    per_brush[(int)BrushType::CLAY].strength = 0.6f;
    per_brush[(int)BrushType::CLAY].hardness = 0.3f;
    // Tight spacing so the raking square turns smoothly through curves: at the
    // default 0.25 the between-dab angle steps are big enough that dabs read as
    // separate stamps; 0.10 stamps the in-between angles.
    per_brush[(int)BrushType::CLAY].spacing = 0.10f;

    drop_path[0] = '\0';
    paint_color[0] = 0.85f; paint_color[1] = 0.25f; paint_color[2] = 0.30f;      // warm red
    paint_color_alt[0] = 0.25f; paint_color_alt[1] = 0.45f; paint_color_alt[2] = 0.85f; // cool blue
    paint_visible = true;
    color_pick_active = false;
    color_pick_click = false;

    // Seed BOTH profiles from the same defaults rather than shipping invented pen
    // numbers — the whole point of the split is that the user tunes each to their
    // own hand, and a guessed tablet preset would just be a wrong starting point
    // that looks deliberate. flush_profile fills the active one; copy it across.
    flush_profile();
    for (int i = 0; i < (int)InputProfile::COUNT; i++)
        profiles[i] = profiles[(int)active_profile];

    // The one deliberate exception to that. max_effect is the ceiling of the strength
    // ramp, and the two devices genuinely arrive at it differently: a mouse dab is always
    // at full input (hence the 0.6 hold-back that keeps additive brushes from piling up),
    // while a pen only reaches the ceiling when you press it to the stop and spends most
    // of a stroke well below. 1.0 here is not a guess about tablets — it is the number
    // the pressure ramp already used implicitly, so pen feel is bit-for-bit unchanged.
    profiles[(int)InputProfile::TABLET].max_effect = 1.0f;

    for (int i = 0; i < (int)KeyAction::COUNT; i++) keybinds[i] = kKeyActions[i].def;
}

void InputState::seed_brush_sizes() {
    for (int i = 0; i < (int)BrushType::COUNT; i++) brush_size_of[i] = brush_size;
}

void InputState::switch_brush(BrushType to) {
    if (to != BrushType::PAINT) color_pick_active = false;
    if (to == current_brush) return;
    // Nothing to save on the way out: per_brush[] and brush_size_of[] are written at
    // the point of edit, so they are already current. This used to copy the live
    // fields into the OUTGOING brush's slot, which is fine only while the mirror is
    // guaranteed to be aimed at that slot — and it is not, because a held Shift aims
    // it at Smooth. Switching brushes mid-Shift therefore stamped Smooth's feel onto
    // whatever you were leaving. See CHANGES 2026-08-23.
    current_brush = to;
    sync_live_settings();
}

// Which per_brush slot the mirror reflects — and the single definition of "the brush
// under your hand". The HUD reads the mirror, the sliders drag it, and the dab loop
// reads per_brush[] directly, so all three have to name the same slot or you end up
// tuning one brush while sculpting with another.
//
// Transient shift-smooth COUNTS, and that is the fix. The dab loop has always treated a
// merely-held Shift as Smooth (see is_smooth in main.cpp); this used to key off
// smooth_locked alone, so with Shift down the mirror kept showing the underlying brush
// while the stroke used Smooth's numbers. Dragging a slider in that state wrote the
// value you were watching into the underlying brush's slot, silently — which is how
// Clay ended up holding a spacing nobody set on it.
BrushType InputState::live_brush_slot() const {
    return (is_smooth_active() || current_brush == BrushType::SMOOTH)
         ? BrushType::SMOOTH : current_brush;
}

BrushType InputState::size_slot() const {
    BrushType b = live_brush_slot();
    return (b == BrushType::SMOOTH) ? BrushType::DRAW : b;
}

// Re-aim the mirror. Idempotent and cheap (three floats), so it can be called from the
// top of every frame — which is what makes the invariant hold across Shift going down
// or up between frames, an edge no key handler observes.
void InputState::sync_live_settings() {
    const BrushSettings& b = per_brush[(int)live_brush_slot()];
    brush_strength = b.strength;
    brush_hardness = b.hardness;
    brush_spacing  = b.spacing;
    // Size is only mirrored when the per-brush toggle is on; off, one shared size is
    // the whole point. The stash is still kept current at the point of edit either way,
    // so flipping the toggle mid-session starts from the sizes you were actually using.
    if (per_brush_sizes) brush_size = brush_size_of[(int)size_slot()];
}

void InputState::flush_profile() {
    // No fold from the mirror: per_brush[] is written at the point of edit and is
    // already the truth. Folding a mirror that may be aimed at Smooth was the other
    // half of the slot-confusion bug — it ran four times a second from settings_tick.
    ProfileSettings& p = profiles[(int)active_profile];
    p.max_effect = max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) p.per_brush[i] = per_brush[i];
}

void InputState::switch_profile(InputProfile to) {
    if (to == active_profile || to == InputProfile::COUNT) return;
    // Deliberately does NOT clear the smooth lock. This runs on every device swap, and
    // silently dropping a sticky lock the user set by double-tapping Shift would read as
    // the app losing state. flush/restore route through live_brush_slot() instead, so the
    // lock rides across the switch untouched.
    flush_profile();

    active_profile = to;
    const ProfileSettings& p = profiles[(int)to];
    max_effect = p.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) per_brush[i] = p.per_brush[i];
    sync_live_settings();
}

void InputState::clear_smooth_lock() {
    if (!smooth_locked) return;
    smooth_locked = false;
    sync_live_settings();
}

void InputState::begin_frame() {
    // Re-aim the settings mirror before anything reads it. Shift going down or up is
    // what makes live_brush_slot() move, and no handler runs on that transition, so
    // this is the only place the invariant can be restored. Skipped mid-drag: a slider
    // owns the live value while it is being dragged and writes through to its latched
    // slot itself.
    if (slider_mode == SliderMode::NONE) sync_live_settings();

    // One-shot event flags are cleared in end_frame, NOT here. On the web,
    // GLFW callbacks fire from DOM handlers *between* rAF frames (glfwPollEvents
    // is a no-op), so a top-of-frame clear would eat every press/keydown before
    // any consumer ran — this is what killed insert-mesh placement and its Y/N
    // prompt in the browser build. Clearing at end-of-frame is correct on both
    // targets: native events land mid-frame via glfwPollEvents, web events land
    // between frames; either way they're set after the previous clear and read
    // before the next one.
}

void InputState::push_path(double x, double y) {
    const float fx = (float)x, fy = (float)y;
    if (path_n > 0 && path_x[path_n - 1] == fx && path_y[path_n - 1] == fy) return;
    // Full (a very long frame): keep the newest endpoint current rather than drop it.
    // Losing samples mid-path only straightens that stretch of the stroke.
    if (path_n == kPathMax) path_n--;
    path_x[path_n] = fx;
    path_y[path_n] = fy;
    path_n++;
}

void InputState::end_frame() {
    prev_mouse_x = mouse_x;
    prev_mouse_y = mouse_y;
    path_n = 0;
    mouse1_just_pressed = false;
    mouse1_just_released = false;
    enter_pressed = false;
    key_y_pressed = false;
    key_n_pressed = false;
    debug_stride_cycle_requested = false;
    debug_pick_vertex_requested  = false;
    color_pick_click = false;
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
        case BrushType::CLAY:   return is_subtract_active() ? "Clay (-)"   : "Clay";
        case BrushType::INFLATE:return is_subtract_active() ? "Inflate (-)": "Inflate";
        case BrushType::CREASE: return is_subtract_active() ? "Crease (-)" : "Crease";
        case BrushType::PINCH:  return is_subtract_active() ? "Pinch (-)"  : "Pinch";
        case BrushType::MOVE:   return "Move";
        case BrushType::LIMB:   return "Limb";
        case BrushType::SMOOTH: return "Smooth";
        case BrushType::MASK:   return "Mask";
        case BrushType::PAINT:  return "Paint";
        default: return "Unknown";
    }
}

// ---- GLFW Callbacks ----

static InputState* g_input = nullptr;

const char* live_key_label(KeyAction a) { return g_input ? g_input->key_label(a) : "?"; }
static double g_last_shift_time = 0;
static double g_last_ctrl_time = 0;
static int g_shift_tap_count = 0;
static int g_ctrl_tap_count = 0;

static double g_slider_last_raw_x = 0;
// GLFW's virtual position when the drag began. A pen drives a slider through a replay
// (tablet.cpp) without ever moving it, so an unchanged value at release means the pen,
// not the mouse, did the drag.
static double g_slider_begin_vx = 0, g_slider_begin_vy = 0;
static GLFWwindow* g_window = nullptr;

// Slider drag: accumulate a horizontal delta into the active slider, clamped so
// the accumulator doesn't overshoot the value range. Fed from the GLFW cursor
// callback natively, from the DOM pointermove listener (movementX) on web.
static void apply_slider_delta(float dx) {
    g_input->slider_accum += dx;

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
                // Ceiling is per-brush: clay's raking square needs overlapping dabs
                // and breaks up into a stamp lattice above 0.15 (see max_spacing_for).
                float min_delta = 0.05f - g_input->slider_start_value;
                float max_delta = max_spacing_for(g_input->slider_slot)
                                - g_input->slider_start_value;
                delta = std::max(min_delta, std::min(max_delta, delta));
                g_input->slider_accum = delta / 0.001f;
                g_input->brush_spacing = g_input->slider_start_value + delta;
                break;
            }
            default: break;
    }
    // Write through to the latched slot — per_brush[] is the truth, the live fields are
    // only the thing being dragged. slider_slot, not a fresh live_brush_slot(): see the
    // field's comment for why letting go of Shift mid-drag must not change the target.
    const int save_brush = (int)g_input->slider_slot;
    g_input->per_brush[save_brush].strength = g_input->brush_strength;
    g_input->per_brush[save_brush].hardness = g_input->brush_hardness;
    g_input->per_brush[save_brush].spacing  = g_input->brush_spacing;
    // Size lives outside BrushSettings but stashes the same way, and the SIZE drag used
    // to update only the live value — so with per-brush sizes on, every resize was
    // reverted by the next brush switch. Smooth folds onto Draw here for the same
    // reason it does in size_slot(): writing Smooth's own slot would be read back from
    // Draw's, and a size drag with Shift held would vanish on release.
    const int size_brush = (save_brush == (int)BrushType::SMOOTH)
                         ? (int)BrushType::DRAW : save_brush;
    g_input->brush_size_of[size_brush] = g_input->brush_size;
    // Don't update mouse_x/y — keep cursor visually locked
}

// Grab/release the browser pointer lock for the slider drag: the OS cursor stays
// frozen at the grab point (and Chrome restores it there on release) while
// pointermove keeps delivering movementX. No-op if the browser refuses the lock —
// the DOM movementX feed still works, only the visible cursor wanders.
#if defined(__EMSCRIPTEN__)
static void web_pointer_lock(bool grab) {
    if (grab) EM_ASM({ if (Module.canvas.requestPointerLock) Module.canvas.requestPointerLock(); });
    else      EM_ASM({ if (document.exitPointerLock) document.exitPointerLock(); });
}
#else
static void web_pointer_lock(bool) {}
#endif

// ---- Slider drag pointer capture ----
//
// A slider is a RELATIVE drag: only horizontal motion matters, and the pointer belongs
// back where it started when you let go.
//
// Wayland refuses glfwSetCursorPos outright unless the cursor is disabled — GLFW 3.5's
// own header says it "will only work when the cursor mode is GLFW_CURSOR_DISABLED,
// otherwise it will emit GLFW_FEATURE_UNAVAILABLE". So the warp-back on release
// silently did nothing there, while the bookkeeping underneath it ran anyway: the real
// pointer stayed where the drag ended and Chisel believed it was back at the start, so
// the ring and the cursor disagreed until the next mouse move. X11 has no such rule,
// which is why this went unnoticed.
//
// Capturing is the right mode regardless: the pointer is locked, so a drag is no longer
// bounded by the screen edge, and it is exactly what the web build already does with
// requestPointerLock. Raw motion is deliberately NOT enabled — it bypasses the OS
// pointer acceleration curve and would change the feel of every slider.
static void end_wrap_capture(GLFWwindow* w);
static void slider_capture(GLFWwindow* w, bool grab) {
#if defined(__EMSCRIPTEN__)
    (void)w;
    web_pointer_lock(grab);
#else
    if (!w) return;
    // GLFW's lock and Chisel's own wrap lock can't coexist on one surface.
    if (grab) end_wrap_capture(w);
    glfwSetInputMode(w, GLFW_CURSOR, grab ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
#endif
}

static void begin_slider_drag(GLFWwindow* w, InputState::SliderMode mode, float start_value) {
    g_input->slider_mode        = mode;
    g_input->slider_start_x     = g_input->mouse_x;
    g_input->slider_start_y     = g_input->mouse_y;
    g_input->slider_start_value = start_value;
    g_input->slider_slot        = g_input->live_brush_slot();
    g_input->slider_accum       = 0;
    slider_capture(w, true);
    // Seed the delta baseline AFTER capturing. Disabling the cursor switches GLFW to an
    // unbounded virtual position that need not start at the window coordinate, so
    // seeding from mouse_x would make the first motion event a jump of whatever the two
    // spaces happen to differ by — a visible lurch on the first pixel of the drag.
    g_slider_last_raw_x = g_input->mouse_x;
#if !defined(__EMSCRIPTEN__)
    if (w) {
        double cx = 0.0, cy = 0.0;
        glfwGetCursorPos(w, &cx, &cy);
        g_slider_last_raw_x = cx;
        g_slider_begin_vx = cx;
        g_slider_begin_vy = cy;
    }
#endif
}

// warp_back = false is the abandon path (focus loss): hand the cursor back where it
// actually is rather than teleporting it into a window the user has already left.
static void end_slider_drag(GLFWwindow* w, bool warp_back) {
    if (!g_input || g_input->slider_mode == InputState::SliderMode::NONE) return;
    bool mouse_dragged = true;
#if !defined(__EMSCRIPTEN__)
    if (w) {
        double vx = 0.0, vy = 0.0;
        glfwGetCursorPos(w, &vx, &vy);
        mouse_dragged = (vx != g_slider_begin_vx || vy != g_slider_begin_vy);
    }
#else
    (void)mouse_dragged;
#endif
    if (warp_back) {
        // Order matters. The position has to be set while the cursor is STILL disabled,
        // because that is the only mode Wayland accepts it in — there it records a
        // cursor-position hint on the locked pointer. Releasing the capture below is
        // what actually moves the pointer there. (On web the pointer-lock exit restores
        // the OS cursor to where it was grabbed, which is the same place.)
        if (w) glfwSetCursorPos(w, g_input->slider_start_x, g_input->slider_start_y);
        g_input->mouse_x      = g_input->slider_start_x;
        g_input->mouse_y      = g_input->slider_start_y;
        g_input->prev_mouse_x = g_input->slider_start_x;
        g_input->prev_mouse_y = g_input->slider_start_y;
    }
    g_input->slider_mode = InputState::SliderMode::NONE;
    slider_capture(w, false);
#if !defined(__EMSCRIPTEN__)
    // TEMPORARY, pairs with the trace in begin_slider_drag. If GLFW reports the asked-for
    // position here but the pointer still visibly jumps on the next move, then GLFW's
    // idea of the cursor and the compositor's have diverged — a different bug from the
    // warp being refused outright.
    // A pen can't be warped on Wayland, and it isn't the pointer this checks anyway —
    // after a pen-driven drag the "got" position is just the idle mouse.
    if (w && warp_back && mouse_dragged) {
        double ax = 0.0, ay = 0.0;
        glfwGetCursorPos(w, &ax, &ay);
        if (std::fabs(ax - g_input->slider_start_x) > 1.0) {
            std::printf("[slider] POINTER DID NOT RETURN: asked (%.1f, %.1f), got "
                        "(%.1f, %.1f). The capture was revoked before release — check "
                        "nothing else is setting GLFW_CURSOR behind the drag's back.\n",
                        g_input->slider_start_x, g_input->slider_start_y, ax, ay);
            std::fflush(stdout);
        }
    }
#endif
}

// ---- Edge wrap ----
//
// A held drag (strokes, Move/Limb, orbit, pan, zoom, object move/scale) that reaches the
// window edge carries on from the opposite edge. The callers skip the frame of the jump,
// so per-frame deltas never see it.
//
// X11 and Windows simply warp the pointer. Wayland and the browser can't, so there the
// pointer is captured and each wrap moves Chisel's idea of the position, not the
// pointer. Faking it without the capture is what this replaced: Chisel's cursor jumped
// across while the real one stayed at the edge, and the next event yanked it back, so a
// Move drag lurched to and fro.
//
// Native Wayland uses Chisel's own lock (tablet.cpp), taken on the drag's first frame:
// it is the only kind that can put the pointer back where the ring is when the drag
// ends. Where that lock is unavailable, GLFW's capture is the fallback — it works, but
// the pointer comes back where it was captured.
static double g_wrap_off_x = 0.0, g_wrap_off_y = 0.0;
static bool   g_wrap_captured = false;
static bool   g_wrap_own_lock = false;   // the capture is Chisel's own Wayland lock

static bool platform_can_warp() {
#if defined(__EMSCRIPTEN__)
    return false;
#elif GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4
    return glfwGetPlatform() != GLFW_PLATFORM_WAYLAND;
#else
    return true;
#endif
}

bool wrap_capture_active() { return g_wrap_captured; }

#if !defined(__EMSCRIPTEN__)
// Motion while Chisel's own lock holds: the pointer is frozen, so wl_pointer sends
// nothing and GLFW's cursor callback is silent. Same bookkeeping as that callback.
static void own_lock_motion(double dx, double dy) {
    if (!g_input) return;
    g_input->mouse_x += dx;
    g_input->mouse_y += dy;
    g_input->push_path(g_input->mouse_x, g_input->mouse_y);
}
#endif

bool wrap_cursor(GLFWwindow* window, InputState& input, int win_w, int win_h) {
    // A pen maps the tablet onto the screen: it can't travel past the edge and nothing
    // can warp it, so the next sample would just put it back.
    if (input.active_profile == InputProfile::TABLET) return false;
#if !defined(__EMSCRIPTEN__)
    // On the drag's first frame, while the pointer is still over the window — a lock
    // only takes hold there. Not while a slider holds GLFW's lock (two locks = protocol
    // error), and not with no button down, or nothing would end it.
    if (!platform_can_warp() && !g_wrap_captured &&
        (input.mouse1_down || input.mouse2_down || input.mouse3_down) &&
        input.slider_mode == InputState::SliderMode::NONE &&
        glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED &&
        wl_pointer_lock_begin(window, own_lock_motion)) {
        g_wrap_captured = true;
        g_wrap_own_lock = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);   // frozen arrow otherwise
    }
#endif
#if !defined(__EMSCRIPTEN__)
    // Not until KWin confirms the lock: before that the pointer still moves and still
    // reports absolute positions, and the first one after a wrap would yank Chisel's
    // cursor straight back across the window — a grab brush dragged its verts with it.
    if (g_wrap_own_lock && !wl_pointer_lock_active()) return false;
#endif
    double mx = input.mouse_x;
    double my = input.mouse_y;
    bool wrapped = false;
    if (mx <= 0) { mx = win_w - 2; wrapped = true; }
    else if (mx >= win_w - 1) { mx = 1; wrapped = true; }
    if (my <= 0) { my = win_h - 2; wrapped = true; }
    else if (my >= win_h - 1) { my = 1; wrapped = true; }
    if (!wrapped) return false;

    if (platform_can_warp()) {
        glfwSetCursorPos(window, mx, my);
    } else if (g_wrap_own_lock) {
        // Positions are integrated from deltas: moving them is the whole wrap.
    } else if (!g_wrap_captured) {
        g_wrap_captured = true;
#if defined(__EMSCRIPTEN__)
        // Positions stay continuous across the lock: the shell seeds its locked
        // position from the last free one, so the same offset carries over. If the
        // browser refuses the lock (it wants a recent click), the drag still gets the
        // room between the canvas and the screen edge.
        (void)window;
        web_pointer_lock(true);
        g_wrap_off_x += mx - input.mouse_x;
        g_wrap_off_y += my - input.mouse_y;
#else
        // GLFW switches to an unbounded virtual position that need not match the window
        // coordinate, so measure the offset against it after capturing.
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        double vx = 0.0, vy = 0.0;
        glfwGetCursorPos(window, &vx, &vy);
        g_wrap_off_x = mx - vx;
        g_wrap_off_y = my - vy;
#endif
    } else {
        g_wrap_off_x += mx - input.mouse_x;
        g_wrap_off_y += my - input.mouse_y;
    }
    input.mouse_x = mx;
    input.prev_mouse_x = mx;
    input.mouse_y = my;
    input.prev_mouse_y = my;
    return true;
}

// Drag over: hand the pointer back where the user sees the cursor. Chisel's own lock
// sends that as its position hint and KWin warps there as it ends. GLFW's capture
// can't: the pointer reappears where it was captured. A browser puts it back where the
// lock began; nothing can change that.
static void end_wrap_capture(GLFWwindow* w) {
    if (g_wrap_captured) {
#if defined(__EMSCRIPTEN__)
        (void)w;
        web_pointer_lock(false);
#else
        if (g_wrap_own_lock) {
            wl_pointer_lock_end(g_input->mouse_x, g_input->mouse_y);
        } else if (w) {
            glfwSetCursorPos(w, g_input->mouse_x, g_input->mouse_y);
        }
        // The main loop picks NORMAL or HIDDEN again next frame.
        if (w) glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
#endif
    }
    g_wrap_captured = false;
    g_wrap_own_lock = false;
    g_wrap_off_x = g_wrap_off_y = 0.0;
}

#ifndef __EMSCRIPTEN__
// A slider drag holds the pointer captured and hidden. If focus leaves while the key is
// still down the release never arrives, so without this the cursor would stay trapped
// in a window the user has already alt-tabbed away from.
static void window_focus_callback(GLFWwindow* w, int focused) {
    if (focused) return;
    end_slider_drag(w, false);
    end_wrap_capture(w);
}
#endif

static void cursor_pos_callback(GLFWwindow* w, double x, double y) {
    if (!g_input) return;
    (void)w;

#if defined(__EMSCRIPTEN__)
    // On web we ignore GLFW's pointer coords entirely: it reports them in canvas
    // backing-store space (scaled by canvas.width / boundingRect.width, and it
    // caches a bounding rect that goes stale across a resize), which desyncs from
    // the CSS-pixel space that rendering + picking use → the post-resize cursor
    // offset. Position AND slider deltas are fed instead by chisel_set_pointer(),
    // driven by a DOM pointermove listener that reads a *fresh*
    // getBoundingClientRect (and movementX) every event.
    (void)x; (void)y;
#else
    // Slider drag: accumulate delta, don't update mouse position.
    if (g_input->slider_mode != InputState::SliderMode::NONE) {
        float dx = (float)(x - g_slider_last_raw_x);
        g_slider_last_raw_x = x;
        apply_slider_delta(dx);
        return;
    }
    // Under Chisel's own lock, motion comes as deltas (own_lock_motion). An absolute
    // position here is the frozen pointer's, and would undo every wrap.
    if (g_wrap_own_lock && wl_pointer_lock_active()) return;
    x += g_wrap_off_x;
    y += g_wrap_off_y;
    g_input->mouse_x = x;
    g_input->mouse_y = y;
    g_input->push_path(x, y);
#endif
}

#if defined(__EMSCRIPTEN__)
// Called from the shell's DOM pointermove/pointerdown listener (see main.cpp) with
// the pointer position already in CSS-pixel space relative to the canvas top-left —
// i.e. the exact space the render surface, pick planes and cursor ring live in —
// plus the event's movementX. Bypasses GLFW's coordinate mangling entirely, so it's
// correct through any resize/DPR.
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_set_pointer(double x, double y, double dx) {
    if (!g_input) return;
    // While a slider drag is active the pointer is locked (or at least visually
    // frozen) — consume only the relative motion, don't move the cached position.
    if (g_input->slider_mode != InputState::SliderMode::NONE) {
        apply_slider_delta((float)dx);
        return;
    }
    x += g_wrap_off_x;
    y += g_wrap_off_y;
    g_input->mouse_x = x;
    g_input->mouse_y = y;
    g_input->push_path(x, y);
    // Feed ImGui the same CSS-pixel position through its own backend entry point.
    // Its GLFW cursor callback was taken back (input_web_take_cursor_callback), so
    // this is the only position source it sees — toolbar/dialog hit-testing stays
    // aligned with the real pointer instead of GLFW's desynced coords.
    if (g_window && ImGui::GetCurrentContext())
        ImGui_ImplGlfw_CursorPosCallback(g_window, x, y);
}

// The browser batches pointer moves to one event per frame; getCoalescedEvents()
// hands back the positions it merged. The shell feeds all but the last through here —
// path only, no ImGui, no slider delta — and the last through chisel_set_pointer.
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_path_sample(double x, double y) {
    if (!g_input || g_input->slider_mode != InputState::SliderMode::NONE) return;
    g_input->push_path(x + g_wrap_off_x, y + g_wrap_off_y);
}

// Fingers on the canvas, from the shell's touch tracker (main.cpp). `revert` is set
// on the call where a second finger lands soon after the first — see
// InputState::touch_revert_requested. The gesture ends once no finger is down AND
// GLFW has delivered the first finger's emulated release; whichever arrives last
// clears it (pointer events reach the shell before GLFW sees the touchend).
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_touch_count(int n, int revert) {
    if (!g_input) return;
    g_input->touch_count = n;
    if (n == 1 && !g_input->touch_gesture) g_input->touch_relatch = true;
    if (n == 0) g_input->touch_relatch = false;
    if (n >= 2 && !g_input->touch_gesture) {
        g_input->touch_gesture = true;
        g_input->touch_revert_requested = revert != 0;
        g_input->touch_cancelled_drag = (int)g_input->drag_mode;
        g_input->drag_mode = InputState::DragMode::NONE;   // ends the stroke / orbit
        g_input->mouse1_just_pressed = false;
        g_input->touch_zoom = 1.0f;
        g_input->touch_pan_x = g_input->touch_pan_y = 0.0f;
    }
    if (n == 0 && !g_input->mouse1_down)
        g_input->touch_gesture = false;
}

// Per two-finger move: distance ratio since the last call, and midpoint motion.
// Accumulated here, applied once per frame by main.cpp.
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_touch_pinch(double scale, double dx, double dy) {
    if (!g_input || !g_input->touch_gesture) return;
    g_input->touch_zoom  *= (float)scale;
    g_input->touch_pan_x += (float)dx;
    g_input->touch_pan_y += (float)dy;
}

// ImGui_ImplGlfw installs its own cursor-pos callback (chaining to ours) and would
// keep feeding ImGui the desynced GLFW coords the app already ignores. Take the
// callback back after ImGui init so GLFW motion reaches neither consumer;
// chisel_set_pointer above feeds both from the DOM.
void input_web_take_cursor_callback(GLFWwindow* window) {
    glfwSetCursorPosCallback(window, cursor_pos_callback);
}

// On web, ImGui_ImplGlfw drops GLFW scroll events on purpose (Emscripten's GLFW
// quantizes them) and reads the wheel from a canvas hook that only
// ImGui_ImplGlfw_InstallEmscriptenCallbacks registers. We can't call that one: it
// also takes over canvas resizing, which the shell and main loop own. So ImGui
// windows never scrolled in the browser. This is its wheel handler, alone. GLFW's
// own listener is separate and still drives camera zoom.
static EM_BOOL web_imgui_wheel(int, const EmscriptenWheelEvent* ev, void*) {
    if (!ImGui::GetCurrentContext()) return EM_FALSE;
    float k = ev->deltaMode == DOM_DELTA_PIXEL ? 1.0f / 100.0f    // same steps as ImGui's
            : ev->deltaMode == DOM_DELTA_LINE  ? 1.0f / 3.0f : 80.0f;
    ImGui::GetIO().AddMouseWheelEvent((float)ev->deltaX * -k, (float)ev->deltaY * -k);
    return EM_TRUE;
}
void input_web_feed_imgui_wheel() {
    emscripten_set_wheel_callback("#canvas", nullptr, false, web_imgui_wheel);
}
#endif

static void mouse_button_callback(GLFWwindow* w, int button, int action, int mods) {
    if (!g_input) return;

    // Update modifiers from mouse event too
    g_input->shift_held = (mods & GLFW_MOD_SHIFT) != 0;
    g_input->ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;
    g_input->alt_held = (mods & GLFW_MOD_ALT) != 0;

    // A two-finger gesture owns the touch until every finger is up: track the
    // emulated button so the state stays truthful, but latch nothing from it.
    if (button == GLFW_MOUSE_BUTTON_LEFT && g_input->touch_gesture) {
        g_input->mouse1_down = (action == GLFW_PRESS);
        if (action == GLFW_RELEASE && g_input->touch_count == 0)
            g_input->touch_gesture = false;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT) {
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
            } else if (g_input->interaction_mode == InputState::InteractionMode::SELECT) {
                // on_model only reflects the active entity's screen buffer (see
                // read_depth_at), so it can't tell us whether the cursor is over some
                // OTHER mesh. SELECT mode does its own scene-wide pick (main.cpp,
                // scene.render_pick over every entity) once drag_mode is SCULPT, so
                // latch unconditionally here and let that pick sort out hit vs. miss.
                g_input->drag_mode = InputState::DragMode::SCULPT;
            } else if (g_input->on_model && g_input->color_pick_active
                       && g_input->current_brush == BrushType::PAINT
                       && g_input->interaction_mode == InputState::InteractionMode::EDIT) {
                // Colour picker consumes this click: sample instead of sculpt
                // (main loop reads the albedo under the cursor and disarms).
                g_input->color_pick_click = true;
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
        bool was_down = g_input->mouse2_down;
        g_input->mouse2_down = (action == GLFW_PRESS);
        // SELECT mode: plain RMB drag scales the selected mesh (a modified RMB is zoom).
        if (action == GLFW_PRESS && !was_down
            && g_input->interaction_mode == InputState::InteractionMode::SELECT
            && !g_input->zoom_modifier_held()
            && g_input->slider_mode == InputState::SliderMode::NONE) {
            g_input->prev_mouse_x = g_input->mouse_x;
            g_input->prev_mouse_y = g_input->mouse_y;
            g_input->drag_mode = InputState::DragMode::SCALE_OBJECT;
        }
        if (action == GLFW_RELEASE && was_down
            && g_input->drag_mode == InputState::DragMode::SCALE_OBJECT) {
            g_input->drag_mode = InputState::DragMode::NONE;
        }
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

    if (action == GLFW_RELEASE && !g_input->mouse1_down && !g_input->mouse2_down
        && !g_input->mouse3_down)
        end_wrap_capture(w);
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

// The held modifiers as GLFW_MOD_* bits, from the per-event-corrected *_held flags
// (see the X11 note in key_callback) rather than the raw `mods` argument.
static int held_mods() {
    return (g_input->ctrl_held  ? GLFW_MOD_CONTROL : 0)
         | (g_input->shift_held ? GLFW_MOD_SHIFT   : 0)
         | (g_input->alt_held   ? GLFW_MOD_ALT     : 0);
}

static int mod_count(int mods) {
    return ((mods & GLFW_MOD_CONTROL) ? 1 : 0) + ((mods & GLFW_MOD_SHIFT) ? 1 : 0)
         + ((mods & GLFW_MOD_ALT) ? 1 : 0);
}

// Which shortcut a key press means. A binding matches when its key is the one pressed
// and every modifier it asks for is held; of those, the one asking for the MOST
// modifiers wins. That is how Shift+D reaches "Subdivision down" while plain D stays
// Draw — and why holding Shift for smooth doesn't stop T from picking Clay, which is
// how the hard-coded switch behaved before the keys became rebindable.
static int find_key_action(int key, int mods) {
    int best = -1, best_n = -1;
    for (int i = 0; i < (int)KeyAction::COUNT; i++) {
        const KeyBind& b = g_input->keybinds[i];
        if (!b.bound() || b.key != key || (b.mods & ~mods & kKeyModMask)) continue;
        const int n = mod_count(b.mods);
        if (n > best_n) { best = i; best_n = n; }
    }
    return best;
}

static void pick_brush(BrushType b) {
    g_input->clear_smooth_lock();
    g_input->switch_brush(b);
    g_input->subtract_locked = false;
}

static void notify(const char* text, float secs) {
    snprintf(g_input->notification, sizeof(g_input->notification), "%s", text);
    g_input->notification_timer = secs;
}

// What each shortcut does. `repeat` is a held key's auto-repeat: only the size nudge
// acts on those, everything else fires once per press.
static void run_key_action(GLFWwindow* w, KeyAction a, bool repeat) {
    InputState& in = *g_input;
    if (repeat && a != KeyAction::SIZE_DOWN && a != KeyAction::SIZE_UP) return;
    const bool select_mode = in.interaction_mode == InputState::InteractionMode::SELECT;
    const bool insert_mode = in.interaction_mode == InputState::InteractionMode::INSERT;

    switch (a) {
        case KeyAction::MODE_SCULPT:
            in.interaction_mode = InputState::InteractionMode::EDIT;
            // Leaving paint: drop back to the draw brush so this sculpts, not paints.
            if (in.current_brush == BrushType::PAINT) {
                in.clear_smooth_lock();
                in.switch_brush(BrushType::DRAW);
            }
            notify("Edit", 1.0f);
            break;
        case KeyAction::MODE_INSERT:
            in.interaction_mode = InputState::InteractionMode::INSERT;
            in.color_pick_active = false;
            notify("Insert mode", 1.5f);
            break;
        case KeyAction::MODE_SELECT:
            in.interaction_mode = InputState::InteractionMode::SELECT;
            in.color_pick_active = false;
            notify("Select", 1.0f);
            break;
        case KeyAction::MODE_PAINT:
            // Paint mode: sculpt-style interaction (EDIT) with the paint brush.
            in.interaction_mode = InputState::InteractionMode::EDIT;
            pick_brush(BrushType::PAINT);
            notify("Paint", 1.0f);
            break;

        case KeyAction::BRUSH_DRAW:    pick_brush(BrushType::DRAW);    break;
        case KeyAction::BRUSH_CLAY:    pick_brush(BrushType::CLAY);    break;
        case KeyAction::BRUSH_INFLATE: pick_brush(BrushType::INFLATE); break;
        case KeyAction::BRUSH_PINCH:   pick_brush(BrushType::PINCH);   break;
        case KeyAction::BRUSH_MOVE:    pick_brush(BrushType::MOVE);    break;
        // Limb (snakehook): pull + tangential redistribute
        case KeyAction::BRUSH_LIMB:    pick_brush(BrushType::LIMB);    break;
        case KeyAction::BRUSH_MASK:    pick_brush(BrushType::MASK);    break;
        case KeyAction::BRUSH_CREASE:
            if (in.current_brush == BrushType::PAINT) {
                // Paint mode: the crease key toggles the colour picker instead.
                // Density target has no colour to pick — a no-op there.
                if (in.paint_target_density) break;
                in.color_pick_active = !in.color_pick_active;
                notify(in.color_pick_active ? "Colour picker: click the model (C/ESC cancels)"
                                            : "Colour picker off", 1.5f);
            } else {
                pick_brush(BrushType::CREASE);
            }
            break;
        case KeyAction::BRUSH_PREV:
        case KeyAction::BRUSH_NEXT: {
            // In SELECT mode these keys spin the selection instead. The main loop polls
            // the bound keys directly for that (continuous, frame-rate independent), so
            // there is nothing to latch here — just don't step the alpha too.
            if (select_mode) break;
            if (in.interaction_mode != InputState::InteractionMode::EDIT || in.is_smooth_active()) break;
            if (in.current_brush == BrushType::PAINT) {
                // In paint mode both swap the active colour with the alternate.
                for (int c = 0; c < 3; c++) std::swap(in.paint_color[c], in.paint_color_alt[c]);
                break;
            }
            // Draw and Mask step through the alphas; the main loop owns the library,
            // so it applies the step and wraps it. Brushes with no alpha: nothing.
            if (in.current_brush == BrushType::DRAW || in.current_brush == BrushType::MASK)
                in.alpha_step += (a == KeyAction::BRUSH_NEXT ? 1 : -1);
            break;
        }

        case KeyAction::HOLD_SIZE:
            begin_slider_drag(w, InputState::SliderMode::SIZE, in.brush_size); break;
        case KeyAction::HOLD_STRENGTH:
            begin_slider_drag(w, InputState::SliderMode::STRENGTH, in.brush_strength); break;
        case KeyAction::HOLD_HARDNESS:
            begin_slider_drag(w, InputState::SliderMode::HARDNESS, in.brush_hardness); break;
        case KeyAction::HOLD_SPACING:
            begin_slider_drag(w, InputState::SliderMode::SPACING, in.brush_spacing); break;

        // Brush size by keyboard, repeating. The S-drag is still the fast way to cross
        // the whole 5..500 range; this is the path for machines without a comfortable
        // drag (trackpads) and for nudging a size you already like.
        //
        // The step is proportional, not fixed: 8% of the current size keeps the felt
        // increment constant across two decades, where a flat step would be a jump at
        // 5 px and imperceptible at 500. The 1 px floor keeps it moving at the bottom,
        // where 8% would round away under the "%.0f" readout.
        case KeyAction::SIZE_DOWN:
        case KeyAction::SIZE_UP: {
            float step = std::max(1.0f, in.brush_size * 0.08f);
            in.brush_size = std::max(5.0f, std::min(500.0f,
                in.brush_size + (a == KeyAction::SIZE_UP ? step : -step)));
            // Stash it too, or with per-brush sizes on the next switch_brush undoes this.
            in.brush_size_of[(int)in.size_slot()] = in.brush_size;
            snprintf(in.notification, sizeof(in.notification), "Size: %.0f", in.brush_size);
            in.notification_timer = 1.0f;
            break;
        }
        case KeyAction::AUTOSMOOTH:
            in.autosmooth = !in.autosmooth;
            notify(in.autosmooth ? "Autosmooth ON" : "Autosmooth OFF", 1.5f);
            break;

        case KeyAction::LEVEL_UP:   if (in.mesh_locked) in.level_switch_delta = +1; break;
        case KeyAction::LEVEL_DOWN: if (in.mesh_locked) in.level_switch_delta = -1; break;
        case KeyAction::PROJECT:    if (in.mesh_locked) in.project_requested = true; break;
        case KeyAction::MIRROR:     in.mirror_x = !in.mirror_x; break;
        case KeyAction::REMESH:
            if (in.mesh_locked && !in.remesh_in_progress) in.remesh_confirm_pending = true;
            break;
        case KeyAction::MERGE:
            // Voxel merge (join selection for print). Same gating as remesh.
            if (in.mesh_locked && !in.voxel_merge_in_progress) in.voxel_merge_confirm_pending = true;
            break;
        case KeyAction::DELETE_MESH: in.delete_mesh_requested = true; break;

        case KeyAction::MASK_INVERT: in.mask_invert_requested = true; break;
        case KeyAction::MASK_CLEAR:  in.mask_clear_requested = true; break;

        case KeyAction::FOCUS:      in.focus_requested = true; break;
        case KeyAction::VIEW_FRONT: in.snap_view_requested = InputState::SnapView::FRONT; break;
        case KeyAction::VIEW_SIDE:  in.snap_view_requested = InputState::SnapView::SIDE; break;
        case KeyAction::VIEW_TOP:   in.snap_view_requested = InputState::SnapView::TOP; break;
        case KeyAction::FULLSCREEN: in.fullscreen_toggle_requested = true; break;
        // Insert mode keeps Y and N for its symmetrize prompt (key_y/n_pressed, set in
        // key_callback), so these two toggles stand down there.
        case KeyAction::WIREFRAME:
            if (!insert_mode) in.show_debug_mesh = !in.show_debug_mesh;
            break;
        case KeyAction::FAST_NORMALS:
            if (!insert_mode) in.fast_normals = !in.fast_normals;
            break;

        case KeyAction::UNDO:    in.undo_requested = true; break;
        case KeyAction::REDO:    in.redo_requested = true; break;
        case KeyAction::SAVE:    in.save_requested = true; break;
        case KeyAction::SAVE_AS: in.save_as_requested = true; break;
        case KeyAction::OPEN:
            in.import_dialog_active = true;
            in.quit_requested = false;
            in.export_dialog_active = false;
            break;
        case KeyAction::EXPORT:
            in.export_dialog_active = true;
            in.quit_requested = false;
            break;
        case KeyAction::COUNT: break;
    }
}

// Keys while a dialog is up. These are fixed — they are how a dialog gets answered,
// so they live outside the rebindable table on purpose.
static void handle_modal_key(GLFWwindow* w, int key) {
    InputState& in = *g_input;
    const bool merge = in.voxel_merge_confirm_pending;
    switch (key) {
        case GLFW_KEY_ESCAPE:
            if      (in.drop_confirm_pending)       in.drop_confirm_pending = false;
            else if (in.voxel_merge_confirm_pending) in.voxel_merge_confirm_pending = false;
            else if (in.drop_level_confirm_pending) in.drop_level_confirm_pending = false;
            else if (in.reset_confirm_pending)      in.reset_confirm_pending = false;
            else if (in.remesh_confirm_pending)     in.remesh_confirm_pending = false;
            else if (in.export_dialog_active)       in.export_dialog_active = false;
            else if (in.import_dialog_active)       in.import_dialog_active = false;
            else if (in.save_dialog_active)         in.save_dialog_active = false;
            else if (in.quit_requested)             in.quit_requested = false;
            break;
        // `~` debug console. Native only — on web the shell.html overlay owns this key
        // and already mirrors console.log.
#ifndef __EMSCRIPTEN__
        case GLFW_KEY_GRAVE_ACCENT: debug_console::toggle(); break;
#endif
        case GLFW_KEY_Y:
            if (in.drop_confirm_pending) {
                // In the drop dialog, Y = open without saving.
                in.drop_confirm_pending = false;
                in.drop_open_requested = true;
            } else if (merge) {
                in.voxel_merge_confirm_pending = false;
                in.voxel_merge_mirror = false;   // faithful (asymmetric) merge
                in.voxel_merge_subtract = false;
                in.voxel_merge_requested = true;
            } else if (in.drop_level_confirm_pending) {
                in.drop_level_confirm_pending = false;
                in.drop_level_requested = true;
            } else if (in.reset_confirm_pending) {
                in.reset_confirm_pending = false;
                in.reset_settings_requested = true;
            } else if (in.remesh_confirm_pending) {
                in.remesh_confirm_pending = false;
                in.remesh_requested = true;
            } else if (in.quit_requested) {
                glfwSetWindowShouldClose(w, GLFW_TRUE);
            }
            break;
        case GLFW_KEY_N:
            if      (in.drop_confirm_pending)       in.drop_confirm_pending = false;
            else if (merge)                         in.voxel_merge_confirm_pending = false;
            else if (in.drop_level_confirm_pending) in.drop_level_confirm_pending = false;
            else if (in.reset_confirm_pending)      in.reset_confirm_pending = false;
            else if (in.remesh_confirm_pending)     in.remesh_confirm_pending = false;
            else if (in.quit_requested)             in.quit_requested = false;
            break;
        // The merge dialog's own keys: M = mirror-symmetric merge, '-' = subtract (union
        // the selection, then carve away every unselected red mesh), S flips the
        // extractor (Surface Nets / Marching Cubes), D flips the chained adaptive remesh,
        // [ / ] nudge the resolution.
        case GLFW_KEY_M:
            if (!merge) break;
            in.voxel_merge_confirm_pending = false;
            in.voxel_merge_mirror = true;
            in.voxel_merge_subtract = false;
            in.voxel_merge_requested = true;
            break;
        case GLFW_KEY_MINUS:
            if (!merge) break;
            in.voxel_merge_confirm_pending = false;
            in.voxel_merge_mirror = false;
            in.voxel_merge_subtract = true;
            in.voxel_merge_requested = true;
            break;
        case GLFW_KEY_S:
            if (merge) in.voxel_merge_surface_nets = !in.voxel_merge_surface_nets;
            else if (in.quit_requested) {   // Save & Quit
                in.quit_requested = false;
                in.quit_after_save = true;
                in.save_requested = true;
            }
            break;
        case GLFW_KEY_D: if (merge) in.voxel_merge_adaptive = !in.voxel_merge_adaptive; break;
        case GLFW_KEY_LEFT_BRACKET:
            if (merge && in.voxel_merge_resolution > 64) in.voxel_merge_resolution -= 32;
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            if (merge && in.voxel_merge_resolution < 256) in.voxel_merge_resolution += 32;
            break;
        default: break;
    }
}

void press_dialog_key(int glfw_key) {
    if (g_input) handle_modal_key(g_window, glfw_key);
}

static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    (void)scancode;
    if (!g_input) return;
    double now = glfwGetTime();

    // Track modifiers
    g_input->shift_held = (mods & GLFW_MOD_SHIFT) != 0;
    g_input->ctrl_held = (mods & GLFW_MOD_CONTROL) != 0;
    g_input->alt_held = (mods & GLFW_MOD_ALT) != 0;

    // ...except for a modifier key's OWN event, where `mods` is not just stale but
    // inverted. X11 hands GLFW the modifier state from BEFORE the event, so pressing
    // Shift reports mods&SHIFT = 0 and releasing it reports 1 (measured on GLFW
    // 3.3.10 X11). Taken at face value, merely holding Shift left shift_held false —
    // so the brush stayed visibly on the base brush until some *other* event arrived
    // to correct it, and the first one usually was the click. That is why the cursor
    // only flipped to Smooth once you pressed, instead of previewing it on the hold.
    // Ctrl (subtract) had the identical tell. The action is unambiguous where the
    // bitfield is not, so derive these three from it. REPEAT counts as held.
    bool modifier_key = true;
    switch (key) {
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
            g_input->shift_held = (action != GLFW_RELEASE); break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            g_input->ctrl_held = (action != GLFW_RELEASE); break;
        case GLFW_KEY_LEFT_ALT:
        case GLFW_KEY_RIGHT_ALT:
            g_input->alt_held = (action != GLFW_RELEASE); break;
        case GLFW_KEY_LEFT_SUPER:
        case GLFW_KEY_RIGHT_SUPER:
            break;
        default: modifier_key = false; break;
    }

    // Key bindings window: report keys to it, run nothing. A modifier on its own is
    // never a binding — it is held while the real key is pressed, and that press
    // carries it in held_mods().
    if (g_input->keybind_menu_open) {
        // GLFW_KEY_UNKNOWN is a key GLFW has no code for (some media keys) — it can't
        // be a binding, since that same value means "unbound".
        if (action != GLFW_PRESS || modifier_key || key == GLFW_KEY_UNKNOWN) return;
        if (key == GLFW_KEY_ESCAPE) {
            g_input->keybind_esc = true;
        } else if (g_input->keybind_capture >= 0) {
            g_input->keybind_captured = { key, held_mods() };
            g_input->keybind_has_capture = true;
        }
        return;
    }

    // Modal UI steals keypresses from sculpting. ImGui file dialogs handle
    // their own keyboard input via ImGui IO — only ESC passes through here
    // to close the dialog flag. Y/N dialogs (quit, remesh) still route keys.
    bool modal_active = g_input->quit_requested
                     || g_input->export_dialog_active
                     || g_input->import_dialog_active
                     || g_input->save_dialog_active
                     || g_input->remesh_confirm_pending
                     || g_input->voxel_merge_confirm_pending
                     || g_input->drop_level_confirm_pending
                     || g_input->reset_confirm_pending
                     || g_input->drop_confirm_pending;
    if (modal_active) {
        if (action == GLFW_PRESS) handle_modal_key(w, key);
        return;
    }

    // Fixed keys: the modifier gestures, Esc, the prompt answers and the debug keys.
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT:
                // Double-tap shift detection
                if (now - g_last_shift_time < 0.3) {
                    if (!g_input->smooth_locked) {
                        // Nothing to save: per_brush[] is written at the point of edit.
                        g_input->smooth_locked = true;
                        g_input->sync_live_settings();
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

            case GLFW_KEY_ESCAPE:
                if (g_input->color_pick_active) g_input->color_pick_active = false;
                else                            g_input->quit_requested = true;
                break;

            case GLFW_KEY_ENTER:
                g_input->enter_pressed = true;
                break;

            // Insert mode's symmetrize prompt (insert_controller.cpp) reads these.
            case GLFW_KEY_Y: g_input->key_y_pressed = true; break;
            case GLFW_KEY_N: g_input->key_n_pressed = true; break;

#ifndef __EMSCRIPTEN__
            case GLFW_KEY_GRAVE_ACCENT: debug_console::toggle(); break;
            case GLFW_KEY_PAGE_UP:      debug_console::scroll_lines(8); break;
            case GLFW_KEY_PAGE_DOWN:    debug_console::scroll_lines(-8); break;
#endif
            case GLFW_KEY_F9:  g_input->debug_stride_cycle_requested = true; break;
            case GLFW_KEY_F10: g_input->debug_pick_vertex_requested = true; break;
            case GLFW_KEY_F12: g_input->debug_multires_requested = true; break;
            default: break;
        }
    }

    // Everything else goes through the rebindable table.
    if ((action == GLFW_PRESS || action == GLFW_REPEAT) && !modifier_key &&
        !keybind_reserved(key)) {
        int a = find_key_action(key, held_mods());
        if (a >= 0) run_key_action(w, (KeyAction)a, action == GLFW_REPEAT);
    }

    // Letting go of a hold-and-drag key ends its drag, whatever modifiers are up now.
    if (action == GLFW_RELEASE) {
        const KeyAction holds[4] = { KeyAction::HOLD_SIZE, KeyAction::HOLD_STRENGTH,
                                     KeyAction::HOLD_HARDNESS, KeyAction::HOLD_SPACING };
        for (KeyAction h : holds) {
            const KeyBind& b = g_input->keybinds[(int)h];
            if (b.bound() && b.key == key) { end_slider_drag(w, true); break; }
        }
    }
}

#ifndef __EMSCRIPTEN__
static void drop_callback(GLFWwindow* w, int count, const char** paths) {
    (void)w;
    if (!g_input || count < 1) return;
    // One prompt at a time — ignore drops while any modal is already up.
    if (g_input->quit_requested || g_input->export_dialog_active ||
        g_input->import_dialog_active || g_input->save_dialog_active ||
        g_input->remesh_confirm_pending || g_input->voxel_merge_confirm_pending ||
        g_input->drop_level_confirm_pending || g_input->drop_confirm_pending ||
        g_input->reset_confirm_pending)
        return;
    // Only project/model files; anything else gets a notification, not a prompt.
    const char* path = paths[0];
    const char* dot = std::strrchr(path, '.');
    char ext[16] = {0};
    if (dot) {
        for (int i = 0; dot[i] && i < (int)sizeof(ext) - 1; i++)
            ext[i] = (char)std::tolower((unsigned char)dot[i]);
    }
    bool known = std::strcmp(ext, ".chisel") == 0 ||
                 std::strcmp(ext, ".obj") == 0 ||
                 std::strcmp(ext, ".ply") == 0;
    if (!known) {
        snprintf(g_input->notification, sizeof(g_input->notification),
                 "Can't open dropped file (want .chisel / .obj / .ply)");
        g_input->notification_timer = 3.0f;
        return;
    }
    snprintf(g_input->drop_path, sizeof(g_input->drop_path), "%s", path);
    g_input->drop_confirm_pending = true;
}
#endif

void setup_input_callbacks(GLFWwindow* window, InputState* state) {
    g_input = state;
    g_window = window;
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_cb_internal);
    glfwSetKeyCallback(window, key_callback);
#ifndef __EMSCRIPTEN__
    glfwSetWindowFocusCallback(window, window_focus_callback);
    glfwSetDropCallback(window, drop_callback);
    // The title-bar X (and a desktop's "close window", which KDE also sends its
    // windows at logout) never closes here: main.cpp decides, closing at once when
    // nothing is unsaved and otherwise raising the quit dialog. GLFW doesn't call
    // this back for the app's own glfwSetWindowShouldClose. A second X while the
    // dialog is up just keeps it up, and one during Save & Quit's Save window is
    // ignored.
    glfwSetWindowCloseCallback(window, [](GLFWwindow* w) {
        glfwSetWindowShouldClose(w, GLFW_FALSE);
        if (!g_input->quit_after_save) g_input->close_requested = true;
    });
#endif
}

void char_callback(GLFWwindow* w, unsigned int codepoint) {
    (void)w;
    (void)codepoint;
}

void setup_char_callback(GLFWwindow* window) {
    glfwSetCharCallback(window, char_callback);
}
