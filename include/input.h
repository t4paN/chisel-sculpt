#pragma once
#include <GLFW/glfw3.h>

enum class BrushType {
    DRAW,
    CLAY,
    INFLATE,
    CREASE,
    PINCH,
    MOVE,
    LIMB,
    SMOOTH,
    MASK,
    PAINT,
    COUNT
};

struct BrushSettings {
    float strength;
    float hardness;
    float spacing;
};

// Clay's stamp is a raking SQUARE, not a radial falloff, so it needs its dabs to
// overlap heavily: consecutive stamps have to blend into a continuous band and the
// frame has to turn a little at a time. Past ~0.15 the individual square footprints
// stop overlapping enough to merge and the stroke prints a lattice of separate
// stamps — diagonal, because a low-hardness clay_square is a max-norm pyramid whose
// creases run corner to corner. Clay's own default is 0.10 (see seed_brush_sizes).
//
// This is a CEILING, not a default, and it is enforced at the point of use as well
// as on the slider: the lattice shipped once already because a persistence path
// handed clay a spacing from another brush, and a value that only the UI clamps is
// one refactor away from coming back.
inline float max_spacing_for(BrushType b) {
    return b == BrushType::CLAY ? 0.15f : 1.0f;
}

// Which input device the brush-feel settings are tuned for. Switched by hand from
// the burger menu, NOT auto-detected: a pen and a mouse can both be live at once
// (and on web every pen is also a mouse), so any auto-switch has to guess, and
// guessing wrong mid-stroke changes the brush under your hand. See switch_profile.
enum class InputProfile {
    MOUSE,
    TABLET,
    COUNT
};

// The stash for one profile. Holds only the settings whose *right value depends on
// what you are holding* — a pen needs lighter strength and a lower ceiling than a
// mouse because pressure already modulates both. Everything else (brush size,
// autosmooth, matcap, colours, mirror, density mults) is a sculpting or look
// preference that has nothing to do with the input device, so it stays
// single-valued on InputState and is saved once.
//
// Brush size used to live here and was moved out on 2026-08-07: reaching for the
// mouse to orbit and finding a different-sized brush reads as the app losing your
// place, not as a thoughtful per-device default.
//
// These duplicate live InputState fields; the live copy is the truth for the ACTIVE
// profile and the stash is written back on switch — the same save/restore shape
// per_brush[] already uses for brush_strength (see switch_brush).
struct ProfileSettings {
    float max_effect;
    BrushSettings per_brush[(int)BrushType::COUNT];
};

// Rebindable keyboard shortcuts (burger menu -> Key bindings). One entry per thing a
// key can do outside a dialog; key_callback looks the pressed key up here instead of
// switching on fixed GLFW codes, and every on-screen key hint reads its label from the
// same table, so a rebind can never leave a tooltip or the shortcut card lying.
//
// NOT in here, on purpose: the dialog answers (Y/N, Esc, the merge dialog's M S D - [ ])
// and the modifier gestures (hold/double-tap Shift and Ctrl). Those are fixed — a
// dialog key you could unbind is a dialog you could no longer answer.
//
// The id strings are what settings.cpp writes to disk — never rename one, or every
// saved rebind of it is silently dropped. Append new actions anywhere; order is only
// the order the Key bindings window lists them in.
enum class KeyAction : int {
    MODE_SCULPT, MODE_INSERT, MODE_SELECT, MODE_PAINT,
    BRUSH_DRAW, BRUSH_CLAY, BRUSH_INFLATE, BRUSH_CREASE, BRUSH_PINCH,
    BRUSH_MOVE, BRUSH_LIMB, BRUSH_MASK, BRUSH_PREV, BRUSH_NEXT,
    PAINT_BUCKET, PAINT_PICKER,
    HOLD_SIZE, HOLD_STRENGTH, HOLD_HARDNESS, HOLD_SPACING, SIZE_DOWN, SIZE_UP, AUTOSMOOTH,
    LEVEL_UP, LEVEL_DOWN, PROJECT, MIRROR, REMESH, MERGE, DELETE_MESH,
    MASK_INVERT, MASK_CLEAR,
    FOCUS, VIEW_FRONT, VIEW_SIDE, VIEW_TOP, FULLSCREEN, WIREFRAME, FAST_NORMALS,
    UNDO, REDO, SAVE, SAVE_AS, OPEN, EXPORT,
    COUNT
};

// key = a GLFW_KEY_* code, or GLFW_KEY_UNKNOWN for "unbound" (shown as ???).
// mods = the GLFW_MOD_CONTROL / SHIFT / ALT bits that must be held with it.
struct KeyBind {
    int key;
    int mods;
    bool bound() const { return key != GLFW_KEY_UNKNOWN; }
    bool operator==(const KeyBind& o) const { return key == o.key && mods == o.mods; }
};
constexpr int kKeyModMask = GLFW_MOD_CONTROL | GLFW_MOD_SHIFT | GLFW_MOD_ALT;

// Where a shortcut is live. Sculpt and Paint each have their own dock, so each has its
// own keys: a Sculpt key does nothing in paint mode and the other way round, which is
// what lets one letter mean a sculpt brush in one and a paint tool in the other.
// Global keys (view, files, mask, brush feel) work everywhere, but in the mode that
// owns a scoped key, that key wins over a Global one bound to the same letter.
enum class KeyScope : uint8_t { Global, Sculpt, Paint };

struct KeyActionInfo {
    const char* id;      // on-disk name, stable forever
    const char* group;   // section heading in the Key bindings window
    const char* label;   // what it does, short
    const char* note;    // what it does in other modes, or nullptr
    KeyBind     def;     // factory binding
    KeyScope    scope = KeyScope::Global;
};
// Two scopes share a key only where both are live: Global overlaps everything.
inline bool key_scopes_overlap(KeyScope a, KeyScope b) {
    return a == KeyScope::Global || b == KeyScope::Global || a == b;
}
const KeyActionInfo& key_action_info(KeyAction a);
// "Ctrl+Shift+S", "F1", "Tab", "???" for unbound. Writes into buf and returns it.
const char* format_keybind(const KeyBind& b, char* buf, int n);
// Keys a binding may not use: they belong to the fixed set above (Esc closes every
// dialog, ` is the console, F9/F10/F12 and PgUp/PgDn are the debug keys). Returns a
// short reason, or nullptr when the key is free to bind.
const char* keybind_reserved(int key);
// The live label for an action, for code that has no InputState at hand (the bitmap
// dialogs). Same as InputState::key_label on the state input.cpp is wired to.
const char* live_key_label(KeyAction a);
// Exactly what pressing that key does while a dialog is up (Y, N, Esc, and the merge
// dialog's M S D - [ ]). The skinned dialogs' buttons call this, so a click and the
// keypress can never drift apart.
void press_dialog_key(int glfw_key);

struct InputState {
    // Mouse
    double mouse_x, mouse_y;
    double prev_mouse_x, prev_mouse_y;
    // Every pointer position reported since the last frame, oldest first; mouse_x/y
    // is only the newest of them. The OS reports motion at hundreds of Hz, but a heavy
    // mesh can hold the frame rate at 15-20, and a stroke laid through one position
    // per frame turns a fast circle into a polygon. The dab loop walks this path
    // instead. Fixed size, no allocation; cleared in end_frame.
    static constexpr int kPathMax = 512;
    float path_x[kPathMax], path_y[kPathMax];
    int   path_n = 0;
    void push_path(double x, double y);
    bool mouse1_down;
    bool mouse2_down;
    bool mouse3_down;  // middle
    bool mouse1_just_pressed;
    bool mouse1_just_released;

    // Modifiers
    bool shift_held;
    bool ctrl_held;
    bool alt_held;

    // Brush state
    BrushType current_brush;
    bool smooth_locked;
    bool subtract_locked;
    // Pixels. Shared across both profiles, and — unless per_brush_sizes is on — across
    // every brush. NOT persisted (dropped 2026-08-17): where the size wants to be is a
    // property of what you are sculpting right now, so restoring last session's number
    // hands you a brush scaled for a model you are no longer looking at.
    float brush_size;
    // These three are a READ-ONLY MIRROR of per_brush[live_brush_slot()], re-pointed
    // once a frame by sync_live_settings(). per_brush[] is the truth; the mirror exists
    // so the HUD and the sliders have a single scalar to read and drag. Anything that
    // edits a brush's feel writes per_brush[] and then re-syncs — never the mirror
    // alone, and never per_brush[] from the mirror. See sync_live_settings.
    float brush_strength;   // 0..1  (mirror of per_brush[live_brush_slot()].strength)
    float brush_hardness;   // 0..1  (mirror of ....hardness)
    float brush_spacing;    // fraction of brush radius between dabs, 0.05..1.0 (mirror)
    BrushSettings per_brush[(int)BrushType::COUNT];

    // "Individual brush sizes" (burger menu). On: brush_size saves/restores per brush
    // exactly the way brush_strength does, so Move can sit at 200px while Paint sits at
    // 40 and neither disturbs the other. Off (default): one shared size, the historical
    // behaviour.
    //
    // The TOGGLE persists but the sizes it stashes do not, for the same reason
    // brush_size itself does not. Deliberately NOT folded into BrushSettings: that
    // struct is what ProfileSettings stashes per input device, and size is not a device
    // trait (see the 2026-08-07 note above).
    bool  per_brush_sizes = true;
    float brush_size_of[(int)BrushType::COUNT];
    // Point every slot at the live brush_size. Called when the toggle flips and on
    // settings reset, so neither can make the brush under the cursor jump.
    void seed_brush_sizes();

    // Mouse/tablet brush-feel profiles. The live fields above are the active
    // profile's working copy; profiles[] holds the other one's parked values.
    // profiles[active_profile] is stale by design between switches — never read it
    // for the active profile, call flush_profile() first (settings save does).
    InputProfile     active_profile;
    ProfileSettings  profiles[(int)InputProfile::COUNT];
    // Set by the burger menu while its popup is up. Suspends the device auto-switch:
    // you tune the tablet profile with the mouse in your hand, and without this the
    // very act of reaching for a slider would swap you back to the mouse profile and
    // send the edit to the wrong one.
    bool             settings_menu_open;

    // Set by the (i) shortcut card while it is up. The card is a plain ImGui popup,
    // so the click that dismisses it lands on the canvas too — this makes main.cpp
    // swallow that click instead of laying down a stray dab or starting an orbit.
    bool             help_popup_open;

    // Two-finger touch (web on phones/tablets): pinch zooms, the fingers' midpoint
    // pans. Emscripten's GLFW turns the FIRST finger into a left click, so by the time
    // the second one lands the first has already started a stroke or an orbit. While
    // touch_gesture is set the left button is ignored entirely, from the moment the
    // second finger lands until every finger is up — the leftover finger must not
    // start a stroke of its own. Fed by chisel_touch_count / chisel_touch_pinch.
    bool  touch_gesture = false;
    int   touch_count   = 0;
    // Set when a gesture begins soon enough after the first finger that whatever it
    // started was a stub, not intent: main.cpp undoes the stroke (if it committed
    // anything) or puts the camera back (orbit). The drag mode it cancelled rides
    // along, since the callback has already cleared drag_mode by the time main runs.
    bool  touch_revert_requested = false;
    int   touch_cancelled_drag   = 0;   // (int)DragMode
    float touch_zoom  = 1.0f;           // accumulated finger-distance ratio, >1 = spread
    float touch_pan_x = 0.0f, touch_pan_y = 0.0f;   // CSS px, midpoint motion
    // A finger has no hover, so the press arrives before any frame has looked at what
    // is under it: the SCULPT-vs-ORBIT latch would read on_model from wherever the LAST
    // finger lifted. Set when a single finger lands; main.cpp re-reads the plane cache
    // at the press point on the next frame and re-latches before anything moves.
    bool  touch_relatch = false;
    // Touch mode: the last pointer was a finger (web only, fed by shell.html). Gates
    // the phone-only layout tweaks; desktop never touches the screen, never sees them.
    bool  touch_mode  = false;
    // On-screen modifier buttons (shell.html), OR'd into ctrl_held / shift_held.
    bool  touch_ctrl  = false;
    bool  touch_shift = false;

    // Live shortcut table (see KeyAction). Seeded with the factory bindings by the
    // constructor; settings.cpp persists only the ones that differ from those.
    KeyBind keybinds[(int)KeyAction::COUNT];
    // Label of the live binding, e.g. "Ctrl+D", or "???" when unbound. Each action has
    // its own buffer, so several can sit in one printf.
    const char* key_label(KeyAction a) const;

    // Key bindings window. While it is open key_callback runs no shortcuts at all: it
    // only reports keys to the window. keybind_capture is the action waiting for a new
    // key (-1 = none); the callback drops the next key press into keybind_captured
    // (key, with mods), or sets keybind_esc for Esc, and the window consumes both.
    bool    keybind_menu_open    = false;
    bool    keybind_menu_request = false;   // burger menu asks the window to open
    int     keybind_capture      = -1;
    bool    keybind_has_capture  = false;
    KeyBind keybind_captured     = { GLFW_KEY_UNKNOWN, 0 };
    bool    keybind_esc          = false;

    // False until the shortcut card has been shown once, then persisted. A first-time
    // user — especially on the itch build, where there is no install step to read a
    // readme during — has no way to guess that 1-4 are the modes, and no reason to
    // press a 28 px (i) disc. So the card opens itself exactly once, ever.
    bool             help_seen;

    // Brush-alpha (stamp) selection. Index into the AlphaLibrary pool; 0 = Round
    // (no stamp). One shared selection, but it only affects Draw, Mask and Paint —
    // every other brush forces the stamp off per dab (set_alpha_dab), except Clay,
    // which ignores the picker and always stamps the Square builtin. The main loop
    // uploads the effective bitmap to ComputeState when it changes; each alpha-
    // capable dab modulates its falloff by the sampled alpha. load_alpha_dialog_active
    // pops the custom-image file picker (mirrors import_dialog_active).
    int  active_alpha = 0;
    int  alpha_step = 0;   // pending Q/E steps through the alphas; main applies + wraps
    bool load_alpha_dialog_active = false;
    // Clay melt (0..1): how strongly clay pulls verts on the WRONG side of its plane
    // (proud on a build, sunk on a carve) back toward it. 0 (default) = ride over:
    // raised areas get nothing added and nothing cut, so a stroke crosses them
    // smoothly instead of terracing them down to the plane. 1 = fully two-sided
    // Blender-style clay (flattens as it builds — steps raised areas to the stroke
    // plane). Slider shows only while Clay is active.
    float clay_melt = 0.0f;

    // Matcap lighting dial (0..1): blends the viewport shading between the old flat
    // ramp (0, keyed off n.y alone) and the keyed/contrasty one (1, directional key
    // + deeper silhouette + sheen). A look preference, not a sculpting parameter —
    // the sun slider in the burger menu drives it. Purely display, nothing reads
    // it back.
    float matcap_contrast = 0.5f;

    // Faceted viewport shading (burger menu toggle, shown inverted as "Smooth
    // shading"). Off = the interpolated vertex normals; on = the true facet normal,
    // so the polygons read individually. Display only — nothing about the mesh or
    // the brush changes, it is a way to *see* the topology the smooth normals hide.
    // Defaults ON (i.e. "Smooth shading" ships unticked) — seeing the actual cage
    // while you work beats a prettier surface that hides where the density is.
    bool flat_shading = true;

    // FPS readout visibility (burger menu toggle). Display only.
    bool show_fps = true;

    // Appearance (burger menu). The modern skin is the default; Classic brings back
    // the original button islands and the bitmap-font HUD; DOS is the modern layout
    // dressed in Chisel's own 8x8 font, CGA colours and pixel icons; Y2K is the
    // early-2000s 3D-app homage (opaque docked shelves on every edge, glossy two-tone
    // tiles, an orange slider shelf). The material and
    // motion only affect the modern skin: panel material runs 0 = clear glass (thin
    // tint, strong backdrop blur) to 100 = solid (opaque, no blur pass at all); springy
    // motion is the press squash / menu pop / bar easing, and is also skipped when the
    // OS asks for reduced motion. Viewfinder corners (four thin L-marks framing the
    // work area) apply to Modern and DOS.
    enum class UiSkin { MODERN = 0, CLASSIC = 1, DOS = 2, Y2K = 3 };
    UiSkin ui_skin      = UiSkin::MODERN;
    bool  ui_classic() const { return ui_skin == UiSkin::CLASSIC; }
    float ui_material   = 45.0f;
    bool  ui_motion     = true;
    bool  ui_viewfinder = false;
    // Web, touch mode only: the whole page drawn at this percent (a CSS scale on the
    // canvas stage), like Chrome's page zoom but owned by the app. 50 = half-size UI
    // with twice the room and twice the pixels. Desktop always draws at 100.
    int   ui_scale_touch = 100;
    // Remesh snaps its new points back onto the original surface and refuses
    // flips that plane off a ridge. Session-only on purpose while it is being
    // A/B'd: untick, remesh, reload, tick, remesh — same file, two results.
    bool remesh_keep_detail = true;
    // Remesh detail offset, as a TRIANGLE-COUNT multiplier on the current mesh:
    // 1 = keep today's average, 2 = twice the tris, 0.5 = half. Stored as a tri
    // multiplier rather than an edge one because that is what you see and what
    // the memory guard has to reason about; the edge target is mean / sqrt(x).
    float remesh_detail = 1.0f;

    // Viewport projection. Orthographic is the default and stays that way: it is what
    // every version shipped and what the brush feel was tuned against. Perspective is
    // opt-in; camera_fov is its vertical FOV in degrees and is ignored while the
    // checkbox is off (ortho framing keeps the historical 45). A view preference, so
    // it lives in settings rather than the project file — main.cpp pushes both onto
    // the camera each frame, which also stops a project load stranding a stale one.
    bool  camera_perspective = false;
    float camera_fov = 45.0f;

    // Delete the highest subdivision level (burger menu). The menu item arms the
    // confirm; Y sets the request, which the main loop consumes once. Destructive
    // and not undoable — see the handler in main.cpp.
    bool drop_level_confirm_pending = false;
    // Menu > Reset settings: Y/N confirm first, then main applies settings_reset.
    bool reset_confirm_pending = false;
    bool reset_settings_requested = false;
    bool drop_level_requested = false;

    // Paint brush albedo (RGB, [0,1]). paint_color is the active colour used by
    // the brush; paint_color_alt is a stashed second colour. Q/E swap them while
    // the paint brush is active. RMB swatch / toolbar boxes edit both.
    float paint_color[3];
    float paint_color_alt[3];
    // Colour picker (C while the paint brush is active). While armed, LMB over
    // the model samples the stored vertex albedo under the cursor into
    // paint_color — pure colour mix, no shading — then disarms, leaving the
    // paint brush active. C again / ESC / leaving paint mode cancel it.
    // color_pick_click is the one-shot "sample now" event the main loop consumes.
    bool color_pick_active;
    bool color_pick_click;
    // Paint mode as a mode of its own, not just "the paint brush is current": its
    // dock (Paint, Bucket, Picker | Smooth, Mask) has to stay up while the Mask brush
    // is in hand, since masking is how you fence off the bucket's islands. Set by
    // MODE_PAINT; cleared by MODE_SCULPT and by picking any brush but Paint or Mask.
    bool paint_mode;
    // Fill bucket (paint mode). While armed, LMB on the model floods the clicked
    // unmasked island of the active layer with paint_color (Ctrl: erases it) instead
    // of starting a stroke. Unlike the picker it stays armed after a fill.
    // bucket_click is the one-shot event the main loop consumes; bucket_erase
    // latches Ctrl at the click, since it may be up again by the time that runs.
    bool bucket_active;
    bool bucket_click;
    bool bucket_erase;
    // What the bucket lays down: a fill_patterns() index (brush_alpha.h). Solid and the
    // tiling patterns fill on press; Gradient is a drag, press = A (main colour, and the
    // island it fills), release = B (secondary). bucket_a/b are those screen points, set
    // with bucket_click; bucket_dragging is "pressed, waiting for the release".
    int   fill_pattern;
    float fill_scale;    // pattern tile size, as a fraction of the model's radius
    float fill_angle;    // pattern rotation, degrees
    bool  bucket_dragging;
    float bucket_ax, bucket_ay, bucket_bx, bucket_by;
    // Show vertex paint in the viewport. Toggle (next to the Paint icon) lets you
    // hide albedo while sculpting; the paint brush always forces it visible.
    bool paint_visible;
    // Paint-brush target: false = albedo (colour), true = the remesh-density field.
    // While true the viewport shows colormap(density) instead of albedo (green =
    // coarse, red = dense); main.cpp owns the enter/exit colour-VBO swap.
    bool paint_target_density = false;
    // What the painted extremes mean to the adaptive remesher, as multipliers
    // on the auto target edge length: green (0) → coarse ×, red (1) → fine ×.
    // Defaults multiply to 1 so neutral 0.5 lands exactly on the uniform target.
    float density_coarse_mult = 2.0f;
    float density_fine_mult   = 0.5f;

    // Slider drag mode (S/W/A/O key held + dragging)
    enum class SliderMode { NONE, SIZE, STRENGTH, HARDNESS, SPACING };
    SliderMode slider_mode;
    double slider_start_x;
    double slider_start_y;
    float slider_start_value;
    float slider_accum;          // accumulated horizontal delta
    // Which brush slot this drag edits, latched at key-down. Latched rather than
    // re-derived per motion event because live_brush_slot() moves when Shift goes
    // down or up: without the latch, letting go of Shift mid-drag would carry a
    // start value belonging to Smooth into whatever brush is underneath.
    BrushType slider_slot;

    // UI
    bool toolbar_visible;
    bool sculpting;         // currently in a brush stroke
    bool on_model;          // cursor is over geometry

    // Interaction latch: locks mode on mouse-down until release
    enum class DragMode { NONE, ORBIT, SCULPT, PAN, ZOOM, MOVE_OBJECT, SCALE_OBJECT };
    DragMode drag_mode;

    // Quit confirmation
    bool quit_requested;    // ESC pressed, waiting for Y/N
    // Quit dialog's Save & Quit: the app closes once a save succeeds. If the project
    // has no file yet that save goes through the Save window, and cancelling it
    // drops this (main.cpp) — you are back in the app, nothing lost.
    bool quit_after_save = false;
    // A close asked from outside the app's own keys (title-bar X, the desktop's
    // "close window", which KDE sends at logout). main.cpp closes at once when
    // nothing is unsaved, else raises the window and asks.
    bool close_requested = false;

    // File dialogs (Ctrl+E export, Ctrl+O import, Ctrl+S save, Ctrl+Shift+S save-as)
    bool export_dialog_active;
    bool import_dialog_active;
    bool import_append;           // import dialog checkbox: add as new entity vs replace scene
    bool save_requested;
    bool save_as_requested;
    bool save_incremental_requested;   // "+" button: save the next numbered copy
    bool save_dialog_active;

    // Drag-and-drop open (native only): a file dropped on the window parks
    // here while the "open / cancel" prompt is up.
    bool drop_confirm_pending;
    bool drop_open_requested;       // Y: open, discarding current changes
    char drop_path[1024];

    // Notification (brief on-screen message)
    char notification[512];
    float notification_timer;  // seconds remaining
    // Set by event sources ImGui never hears about (a file drop), so the idle frame
    // skip in main.cpp still draws the frame that reacts to them. Cleared there.
    bool wake_requested = false;

    // Focus request
    bool focus_requested;   // F pressed, main loop handles reframe

    // Snap view requests (F1/F2/F3)
    enum class SnapView { NONE, FRONT, SIDE, TOP };
    SnapView snap_view_requested;

    // Undo/redo requests (Ctrl+Z / Ctrl+Shift+Z)
    bool undo_requested;
    bool redo_requested;

    // Fullscreen toggle request
    bool fullscreen_toggle_requested;
    bool is_fullscreen;

    // Cursor normal (updated per frame when on model, flat when off)
    float cursor_nx, cursor_ny, cursor_nz;

    bool mesh_locked;       // true after first brush contact

    // Mirror
    // X-axis symmetry. Session state, never persisted: every run starts with it ON
    // regardless of how the last one ended (settings.cpp neither writes nor reads it).
    // Reset still restores it, since that reads this default.
    bool mirror_x;

    // HOW the mirror works, not whether. Off (default) = GEOMETRIC: the dab is
    // reflected in the world x=0 plane and applied a second time against whatever
    // geometry is actually there, so symmetry needs nothing from the topology —
    // it works on an imported mesh, an asymmetrically remeshed one, or one whose
    // pair map never resolved. On = TOPOLOGICAL: the geometric dab still lands,
    // and the pair map (mirror_x_map) additionally forces the two sides to stay
    // byte-identical. Exact, but only meaningful on a mesh whose tessellation IS
    // mirror-symmetric, and it TEARS one that is not — hence the guard in
    // Mesh::mirror_world_symmetric(), which only gates this mode.
    // Persisted: unlike mirror_x, this is a preference, not session state.
    bool mirror_topological;

    // Counts down while the "Topological mirror not available" flash is on screen.
    // Set only by the enforcement in main.cpp, which reverts mirror_topological at the
    // same moment — the message and the state change are one event, never two.
    float mirror_unavailable_timer = 0.0f;

    // Autosmooth: light Laplacian pass on draw-brush strokes at pen-up.
    // Defaults OFF (was ON until 2026-10-08), toggled with B. Persisted since
    // 2026-08-06, global since 2026-08-07 — it is a sculpting preference, not a
    // device trait, so a mouse-side toggle must not flip back when you pick up the pen.
    bool autosmooth;

    // Max brush effect: the strength multiplier at FULL input — a mouse dab, or a pen
    // pressed to the stop. It is the ceiling of the pressure ramp, not a separate scale,
    // so one number means the same thing on both profiles (a per-profile copy that only
    // bit when pressure was off would be a dead knob on the Tablet tab).
    //
    // Applies to the additive-displacement brushes only (Draw/Inflate/Crease/Pinch) —
    // those accumulate without bound, so a ceiling is what stops full-strength dabs from
    // overshooting. Move/Limb read strength as a cursor-tracking ratio and
    // Smooth/Mask/Paint converge on a target, so capping those would just make them take
    // longer to reach the same place.
    //
    // Defaults differ by profile on purpose, and that is NOT a guessed preset: 0.6 mouse
    // / 1.0 tablet are exactly the constants this replaced (the old MOUSE_STRENGTH_SCALE
    // and the implicit 1.0 top of the pressure ramp), so the shipped feel is unchanged.
    float max_effect;

    // Fast draw mode: skip per-pixel normal interpolation, use vertex normals directly
    bool fast_normals;      // N key toggle, off by default

    // LOD
    int current_lod;

    // Icosphere subdivision level (pre-lock selector, frozen after first stroke)
    int subdiv_level;

    // Multires level switch: +1 = up (Shift+D post-lock), -1 = down (D post-lock), 0 = none
    int level_switch_delta;

    // Debug: print multires stack state to stdout (F12)
    bool debug_multires_requested;

    // Debug: F9 cycles stride override, F10 picks test vertex under cursor
    bool debug_stride_cycle_requested;
    bool debug_pick_vertex_requested;

    // Debug: Y key toggles mesh verts+edges overlay
    bool show_debug_mesh;

    // Manual projection trigger (P post-lock): projects current truth down onto current level
    bool project_requested;

    // Remesh (/ key)
    bool remesh_requested;
    bool remesh_in_progress;
    bool remesh_confirm_pending;

    // Voxel merge (SDF join-for-print) — Merge button / J key
    bool voxel_merge_requested;
    bool voxel_merge_in_progress;
    bool voxel_merge_confirm_pending;
    bool voxel_merge_mirror;       // M (vs Y): symmetrise the result about x=0
    bool voxel_merge_surface_nets; // S toggles: Surface Nets vs Marching Cubes extractor
    int  voxel_merge_resolution;   // target cells along longest axis (64..256)
    bool voxel_merge_subtract;     // '-' in merge dialog: carve unselected (red) meshes from the selected union
    bool voxel_merge_adaptive;     // D in merge dialog: chain the adaptive remesh after the merge (density field only)

    // Mask operations
    bool mask_invert_requested;   // Ctrl+I
    bool mask_clear_requested;    // Ctrl+A (when mask brush active)

    // Multi-mesh interaction modes (1/2/3 keys)
    enum class InteractionMode { EDIT, INSERT, SELECT };
    InteractionMode interaction_mode;

    // Insert-mode primitive selected in the shape picker (INSERT mode UI).
    enum class InsertShape { SPHERE, BOX, CYLINDER };
    InsertShape insert_shape;

    // Which sphere "sphere" means — persisted, and deliberately ONE setting for both
    // consumers: the ball the app opens with and the one the INSERT shape picker
    // spawns. Two knobs would let the two disagree, which is exactly the confusion
    // the user asked to avoid. Changing it re-aims insert immediately; the startup
    // mesh follows on the next launch (the current scene is not thrown away).
    // ICO is default: the icosphere has no pole pinch and is what brush feel was
    // tuned against. UV is 32x16 = Blender's stock sphere, for edge-loop workflows.
    enum class SphereKind { ICO, UV };
    SphereKind sphere_kind;
    bool delete_mesh_requested;   // Delete key
    bool enter_pressed;           // Enter key (consumed per-frame)
    bool key_y_pressed;           // Y key (consumed per-frame, for prompts)
    bool key_n_pressed;           // N key (consumed per-frame, for prompts)

    InputState();

    void begin_frame();
    void end_frame();

    void switch_brush(BrushType to);
    // The mode buttons and keys 1 and 4, one place for every skin. Paint: EDIT with the
    // plain paint brush in hand (no bucket, no picker, no smooth lock). Sculpt: EDIT
    // and out of paint mode; the paint brush gives way to Draw, a Mask brush stays.
    void enter_paint_mode();
    void enter_sculpt_mode();
    // Which mode's shortcuts are live: Paint while painting (EDIT in paint mode), else
    // Sculpt (Insert and Select keep the sculpt keys, as before the split).
    KeyScope key_scope() const {
        return paint_mode && interaction_mode == InteractionMode::EDIT ? KeyScope::Paint
                                                                      : KeyScope::Sculpt;
    }
    void clear_smooth_lock();  // drop the sticky smooth lock, re-aim the mirror
    void switch_profile(InputProfile to);  // park the live brush feel, load the other profile's
    void flush_profile();      // per_brush[] -> profiles[active_profile] (call before saving)
    // Re-point brush_strength/hardness/spacing (and the size, when per-brush sizes are
    // on) at per_brush[live_brush_slot()]. Idempotent; call it after anything that can
    // change which slot is live.
    void sync_live_settings();
    BrushType live_brush_slot() const;     // which per_brush slot the mirror reflects
    // Which brush_size_of[] slot to read and write. Same as live_brush_slot(), EXCEPT
    // that Smooth folds onto Draw so the two always share one size: you smooth what you
    // just drew, at the scale you drew it, and a held Shift must not resize the cursor
    // under your hand. Strength/hardness/spacing stay per-brush — only size is shared.
    BrushType size_slot() const;
    bool is_smooth_active() const;
    bool is_subtract_active() const;
    // Right-drag is zoom while either of these is held. A predicate rather than the
    // test written out three times, because every RMB consumer has to agree with it:
    // the zoom itself, SELECT mode's scale latch, and the paint swatch popup all have
    // to stand down, and one of them forgetting is a right-drag that zooms AND scales.
    bool zoom_modifier_held() const { return ctrl_held || alt_held; }
    const char* brush_name() const;
};

void setup_input_callbacks(GLFWwindow* window, InputState* state);
// Edge wrap for held drags (input.cpp). Returns true on the frame the cursor jumped to
// the far edge — the caller skips that frame's motion.
bool wrap_cursor(GLFWwindow* window, InputState& input, int win_w, int win_h);
// True while a wrap holds the pointer captured (Wayland); the per-frame cursor-mode
// write must leave it alone, as it does for a slider drag.
bool wrap_capture_active();
void setup_char_callback(GLFWwindow* window);
#if defined(__EMSCRIPTEN__)
// Re-take the GLFW cursor-pos callback from ImGui_ImplGlfw (call after ImGui init):
// on web both consumers are fed the DOM CSS-pixel position via chisel_set_pointer
// instead of GLFW's desynced backing-store coords.
void input_web_take_cursor_callback(GLFWwindow* window);
// Feed the mouse wheel to ImGui (call after ImGui init). Its GLFW backend ignores
// GLFW scroll on web and expects its own canvas wheel hook instead.
void input_web_feed_imgui_wheel();
#endif
