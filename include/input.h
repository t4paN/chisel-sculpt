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

struct InputState {
    // Mouse
    double mouse_x, mouse_y;
    double prev_mouse_x, prev_mouse_y;
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
    float brush_size;       // pixels (shared across all brushes AND both profiles)
    float brush_strength;   // 0..1 (per-brush, synced via switch_brush)
    float brush_hardness;   // 0..1 (per-brush, synced via switch_brush)
    float brush_spacing;    // fraction of brush radius between dabs (0.05..1.0, per-brush)
    BrushSettings per_brush[(int)BrushType::COUNT];

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

    // Brush-alpha (stamp) selection. Index into the AlphaLibrary pool; 0 = Round
    // (no stamp). One shared selection, but it only affects Draw, Mask and Paint —
    // every other brush forces the stamp off per dab (set_alpha_dab), except Clay,
    // which ignores the picker and always stamps the Square builtin. The main loop
    // uploads the effective bitmap to ComputeState when it changes; each alpha-
    // capable dab modulates its falloff by the sampled alpha. load_alpha_dialog_active
    // pops the custom-image file picker (mirrors import_dialog_active).
    int  active_alpha = 0;
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

    // FPS readout visibility (burger menu toggle). Display only.
    bool show_fps = true;

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

    // UI
    bool toolbar_visible;
    bool sculpting;         // currently in a brush stroke
    bool on_model;          // cursor is over geometry

    // Interaction latch: locks mode on mouse-down until release
    enum class DragMode { NONE, ORBIT, SCULPT, PAN, ZOOM, MOVE_OBJECT, SCALE_OBJECT };
    DragMode drag_mode;

    // Quit confirmation
    bool quit_requested;    // ESC pressed, waiting for Y/N

    // File dialogs (Ctrl+E export, Ctrl+O import, Ctrl+S save, Ctrl+Shift+S save-as)
    bool export_dialog_active;
    bool import_dialog_active;
    bool import_append;           // import dialog checkbox: add as new entity vs replace scene
    bool save_requested;
    bool save_as_requested;
    bool save_dialog_active;

    // Drag-and-drop open (native only): a file dropped on the window parks
    // here while the "open / cancel" prompt is up.
    bool drop_confirm_pending;
    bool drop_open_requested;       // Y: open, discarding current changes
    char drop_path[1024];

    // Notification (brief on-screen message)
    char notification[512];
    float notification_timer;  // seconds remaining

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
    bool mirror_x;          // X-axis symmetry, on by default

    // Autosmooth: light Laplacian pass on draw-brush strokes at pen-up.
    // Defaults ON, toggled with B. Persisted since 2026-08-06, global since
    // 2026-08-07 — it is a sculpting preference, not a device trait, so a
    // mouse-side toggle must not flip back when you pick up the pen.
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
    bool delete_mesh_requested;   // Delete key
    bool enter_pressed;           // Enter key (consumed per-frame)
    bool key_y_pressed;           // Y key (consumed per-frame, for prompts)
    bool key_n_pressed;           // N key (consumed per-frame, for prompts)

    InputState();

    void begin_frame();
    void end_frame();

    void switch_brush(BrushType to);
    void clear_smooth_lock();  // save smooth's settings, restore current brush's
    void switch_profile(InputProfile to);  // park the live brush feel, load the other profile's
    void flush_profile();      // live fields -> profiles[active_profile] (call before saving)
    BrushType live_brush_slot() const;     // which per_brush slot the live fields mirror
    bool is_smooth_active() const;
    bool is_subtract_active() const;
    const char* brush_name() const;
};

void setup_input_callbacks(GLFWwindow* window, InputState* state);
void setup_char_callback(GLFWwindow* window);
#if defined(__EMSCRIPTEN__)
// Re-take the GLFW cursor-pos callback from ImGui_ImplGlfw (call after ImGui init):
// on web both consumers are fed the DOM CSS-pixel position via chisel_set_pointer
// instead of GLFW's desynced backing-store coords.
void input_web_take_cursor_callback(GLFWwindow* window);
#endif
