#pragma once
#include <GLFW/glfw3.h>

enum class BrushType {
    DRAW,
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
    float brush_size;       // pixels (shared across all brushes)
    float brush_strength;   // 0..1 (per-brush, synced via switch_brush)
    float brush_hardness;   // 0..1 (per-brush, synced via switch_brush)
    float brush_spacing;    // fraction of brush radius between dabs (0.05..1.0, per-brush)
    BrushSettings per_brush[(int)BrushType::COUNT];

    // Brush-alpha (stamp) selection. Index into the AlphaLibrary pool; 0 = Round
    // (no stamp). One shared selection, but it only affects Draw, Mask and Paint —
    // every other brush forces the stamp off per dab (set_alpha_dab). The main loop
    // uploads the selected bitmap to ComputeState when this changes; each alpha-
    // capable dab modulates its falloff by the sampled alpha. load_alpha_dialog_active
    // pops the custom-image file picker (mirrors import_dialog_active).
    int  active_alpha = 0;
    bool load_alpha_dialog_active = false;

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
    // Per-session, defaults ON, toggled with B.
    bool autosmooth;

    // Pen-pressure response. Defaults ON; toggled with K. No effect without a tablet.
    bool pressure_enabled;

    // Facing threshold: skip brush pixels whose dot(normal, -view) < this value.
    // Prevents distortion near the silhouette. Adjusted with [ / ].
    float facing_threshold;

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
