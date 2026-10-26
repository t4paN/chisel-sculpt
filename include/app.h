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
#ifdef __EMSCRIPTEN__
    char            (&web_save_name)[96];
    char            (&web_export_name)[96];
#endif
};

// app_ops.cpp: the frame's shared helpers, once lambdas inside main()'s frame.
void app_refresh_active_gpu_residency(App& a);

// app_ops.cpp: the frame's one-shot user operations, each called where it ran in main.
void app_ops_paint_clicks(App& app, int win_w, int win_h);   // C colour picker, fill bucket

// app_files.cpp: open/import, the dev auto-import hooks, save, save-as, incremental
// save, export, and their dialogs (native file browser, web name prompts + picker).
void app_files_frame(App& a);
