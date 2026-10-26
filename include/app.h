#pragma once
// The main loop's state, for the parts of the frame that live outside main.cpp
// (app_files.cpp, app_ops.cpp). Every member is a REFERENCE to a local of main():
// main() still owns them, in the order it always built them, and on web
// emscripten_set_main_loop_arg's simulate_infinite_loop keeps that stack alive, so
// the references stay good for the life of the app.
//
// A moved block opens with aliases for only what it uses, so its text stays as it was
// in main.cpp. Always `auto& x = a.x;`: a plain `auto x = a.x;` copies (a Mesh, every
// frame) and its edits go nowhere. Check: grep -nE 'auto [a-z_]+ = a\.' src/app_*.cpp
#include <cstdint>
#include <string>
#include <vector>

enum class AppState { IDLE, SCULPTING };

struct GLFWwindow;
struct InputState;
class  Scene;
struct Mesh;
struct MultiresStack;
struct Camera;
struct ComputeState;
struct BrushStroke;
struct AlphaLibrary;
class  ObjectXformStack;
struct VoxelMergeJob;
struct Vec3;
struct Renderer;
struct SceneSnapshot;
struct ProjectionStats;
struct TextOverlay;

struct App {
    GLFWwindow*&      window;
    InputState&       input;
    Scene&            scene;
    Mesh*&            mesh;       // the active entity's; re-pointed after anything that rebuilds the scene
    MultiresStack*&   multires;   // likewise
    Camera&           camera;
    ComputeState&     compute;
    BrushStroke&      brush_stroke;
    AlphaLibrary&     alpha_lib;
    ObjectXformStack& xforms;
    VoxelMergeJob*&   vmerge_job;
    AppState&         app_state;
    Vec3&             mesh_center;
    float&            mesh_radius;
    bool&             screen_buffers_dirty;
    uint64_t&         clean_edit_mark;
    std::string&      cli_open_path;
    std::string&      default_browse_path;
    std::string&      current_project_path;
    std::string&      error_popup_msg;
    bool&             error_popup_trigger;
    Renderer&         renderer;
    SceneSnapshot&    rescue;
    int&              rescue_shown;
    int&              max_level;            // --max-level
    bool&             swallow_level_switch;
    bool&             remesh_chained_from_merge;
    TextOverlay&      text;
    float&            fps_display;
    Vec3&             rot_axis;     // Select-mode Q/E spin: view axis latched at spin start
    bool&             rot_active;
    double&           rot_last_time;
    std::vector<uint32_t>& rot_dirty;
    Vec3&             scale_pivot;  // latched on the first frame of an RMB scale drag
#ifdef __EMSCRIPTEN__
    char            (&web_save_name)[96];
    char            (&web_export_name)[96];
#endif
#ifdef CHISEL_BACKEND_WEBGPU
    int&              fbw;        // the size the surface was last configured at
    int&              fbh;
#endif
};

// app_ops.cpp: the frame's shared helpers, once lambdas inside main()'s frame (or, for
// report_projection_check, a static of main.cpp).
void report_projection_check(const ProjectionStats& ps);
void app_refresh_active_gpu_residency(App& a);
void app_print_undo_top(App& a, const char* tag);
void app_take_rescue(App& a, const char* op);
void app_arm_geometry_handoff(App& a, const Mesh& m);

// app_ops.cpp: the frame's one-shot user operations, each called where it ran in main.
void app_ops_paint_clicks(App& app, int win_w, int win_h);   // C colour picker, fill bucket
void app_ops_levels_and_views(App& a);   // D / Shift-D, delete top level, F focus, F1-F3 views
void app_ops_remesh_and_merge(App& a);   // remesh run, SDF merge kick-off + per-frame step

// app_files.cpp: open/import, the dev auto-import hooks, save, save-as, incremental
// save, export, and their dialogs (native file browser, web name prompts + picker).
void app_files_frame(App& a);

// app_render.cpp: cursor visibility, the scene pass, overlays and dialogs, then (after
// app_files_frame) the button islands, panels, error popup, ImGui render and present.
void app_render_frame(App& a, int win_w, int win_h, bool imgui_wants_mouse);

// app_idle.cpp: the IDLE state's update (camera drags, Select-mode move / scale / spin,
// mask ops, undo/redo, IDLE-only one-shots). dx, dy = this frame's mouse delta.
void app_idle_update(App& a, int win_w, int win_h, float dx, float dy);
