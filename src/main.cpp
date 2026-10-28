#ifdef CHISEL_BACKEND_WEBGPU
#if defined(__EMSCRIPTEN__)
// Web target: the browser IS the WebGPU implementation (emdawnwebgpu). No X11
// native handles, no wgpu-native extension header — the surface comes from a
// canvas CSS selector, and the event loop is the browser's.
#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#else
// A Linux desktop session is either X11 or Wayland and GLFW picks at runtime,
// so expose BOTH native handle sets and choose the matching surface source
// below. Exposing only X11 builds fine on a Wayland box and fails at runtime.
// Windows has the one platform: an HWND surface.
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #define GLFW_EXPOSE_NATIVE_WIN32
#else
  #define GLFW_EXPOSE_NATIVE_X11
  #define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpu-native extensions (wgpuDevicePoll / ProcessEvents)
#endif
#include "gpu/gpu.h"       // device_from_webgpu + surface-format/depth setters
#else
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#if defined(_WIN32)
#include <GLFW/glfw3native.h>   // glfwGetWin32Window, for MouseHistory
#endif
#endif
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cctype>
#include <algorithm>
#include <string>

#include "mesh.h"
#include "camera.h"
#include "renderer.h"
#include "input.h"
#include "settings.h"
#include "text_overlay.h"
#include "brush.h"
#include "tablet.h"
#include "undo.h"
#include "multires_stack.h"
#include "remesh.h"
#include "compute.h"
#include "chisel_debug.h"
#include "scene.h"
#include "sdf.h"
#include "insert_controller.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include "retopo.h"
#include "web_bridge.h"
#include "gpu/wgpu_surface.h"
#include "app.h"
#include "ui_backdrop.h"
#include "debug_console.h"
#include "bench.h"
#include "imgui.h"
#include "imgui_internal.h"   // InputEventsQueue: the idle frame skip reads it
#include "imgui_impl_glfw.h"
#ifdef CHISEL_BACKEND_WEBGPU
#include "imgui_impl_wgpu.h"
#else
#include "imgui_impl_opengl3.h"
#endif
#include "ImGuiFileDialog.h"
#include "brush_alpha.h"
#include "project_file.h"
#include "scene_snapshot.h"
#include "object_xform.h"
#include <string>
#include <filesystem>

// Pen-pressure response curve. Pressure (0..1) maps to independent multipliers for
// brush strength and size. Floors differ on purpose: a feather-light touch should
// fade strength to near-nothing while keeping a usable footprint, so a light pass
// smooths/shades broadly and a hard press digs in. GAMMA shapes the ramp (1.0 =
// linear; the xf86-input-wacom pressure curve is already applied driver-side).
static constexpr float PRESSURE_STR_FLOOR  = 0.05f;
static constexpr float PRESSURE_SIZE_FLOOR = 0.40f;
static constexpr float PRESSURE_GAMMA      = 1.0f;

// Speed falloff (InputState::speed_gain): strength x 1 / (1 + gain * speed / REF), so at
// full gain a stroke at SPEED_REF px/s lands at half strength. Speed is screen px/s,
// smoothed over SPEED_TAU so one jerky frame doesn't pulse the stroke; the floor keeps a
// flick from vanishing outright.
static constexpr float SPEED_REF   = 400.0f;
static constexpr float SPEED_TAU   = 0.08f;
static constexpr float SPEED_FLOOR = 0.10f;
static constexpr float SPEED_SIZE_FLOOR = 0.80f;   // a fast stroke thins, never by more than a fifth
struct StrokeSpeed { float px_s = 0.0f; float x = 0.0f; float y = 0.0f; double t = 0.0; };
static StrokeSpeed g_stroke_speed;
// The same smoothing run while hovering, so a stroke pressed mid-motion starts at the speed
// the hand already had instead of 0 (full strength). A pause before pressing decays it to
// ~0 on its own. A sample faster than SPEED_TELEPORT is a warp (slider return, window
// re-entry), not a hand; anything near that speed already sits at the floor.
static constexpr float SPEED_TELEPORT = 15000.0f;
static StrokeSpeed g_hover_speed;

// The old MOUSE_STRENGTH_SCALE (0.6) lived here. It is now the per-profile "max brush
// effect" slider (InputState::max_effect) — same number, same defaults, but reachable.
// Dab spacing is stepped in SCREEN pixels but every dab deposits into a WORLD-space
// sphere, so the step has to be foreshortening-corrected — see the block in the dab
// loop. MIN_COS floors the correction (at most 1/0.30 ≈ 3.3x the dab count near a
// silhouette); DAB_COUNT_MAX is a hard budget on top of that, since each dab is a
// full-vertex-array dispatch.
static constexpr float DAB_SPACING_MIN_COS = 0.30f;
static constexpr int   DAB_COUNT_MAX       = 64;

// The spacing the strength defaults were tuned at, and the fixed reference the dab-
// density compensation normalises against. Must track BrushSettings::spacing's
// constructor default: at s == SPACING_REF the compensation is exactly 1.0, so the
// shipped feel is the thing being preserved.
static constexpr float SPACING_REF = 0.25f;

// From input.cpp
extern float input_consume_scroll();
void setup_char_callback(GLFWwindow* window);

// From window_icon.cpp — sets the GLFW window icon from the embedded PNG.
void set_window_icon(GLFWwindow* window);



// Pointer samples per stroke frame, printed once per stroke. Says whether the path the
// dab loop walks is dense: a mouse at 125-1000 Hz gives several samples per frame at
// 20-60 fps, while a device (or browser) that reports once per frame gives 1, and then
// fast curves facet again. One line per stroke; remove once the numbers are known.
struct PathStats {
    uint32_t frames = 0, samples = 0, max_per_frame = 0;
    void frame(int n) {
        frames++;
        samples += (uint32_t)n;
        if ((uint32_t)n > max_per_frame) max_per_frame = (uint32_t)n;
    }
    void report() {
        if (frames)
            std::printf("[path] %u stroke frames, %.1f pointer samples/frame (max %u)\n",
                        frames, (double)samples / frames, max_per_frame);
        *this = PathStats();
    }
};
static PathStats g_path_stats;

#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
// Windows does not queue mouse motion: WM_MOUSEMOVE is synthesized once per message
// pump from the latest position, so glfwPollEvents hands the cursor callback ~1
// position per frame however fast the mouse reports (a 500 Hz mouse at 75 fps: 1 of
// ~6.6 positions), and InputState::path_x is as sparse as no path at all — fast
// small-brush strokes facet into corners. The system still records every reported
// position; GetMouseMovePointsEx returns that history (newest first, 64 deep), so after
// each poll the frame's path is rebuilt from it. Linux/X11/Wayland deliver every event
// and need none of this.
struct MouseHistory {
    MOUSEMOVEPOINT last = {};
    bool have_last = false;

    // Forget the history position (after a cursor teleport), so the next frame only
    // takes points reported after it.
    void forget() { have_last = false; }

    void pull(GLFWwindow* window, InputState& input) {
        if (input.slider_mode != InputState::SliderMode::NONE ||
            glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            have_last = false;
            return;
        }
        POINT cur;
        if (!GetCursorPos(&cur)) return;
        MOUSEMOVEPOINT in = {};
        in.x = cur.x & 0xFFFF;
        in.y = cur.y & 0xFFFF;
        MOUSEMOVEPOINT buf[64];
        const int n = GetMouseMovePointsEx(sizeof(MOUSEMOVEPOINT), &in, buf, 64,
                                           GMMP_USE_DISPLAY_POINTS);
        if (n <= 0) return;   // cursor not in the history (e.g. a SetCursorPos): keep GLFW's
        int fresh = 1;
        if (have_last) {
            fresh = 0;
            while (fresh < n && !(buf[fresh].x == last.x && buf[fresh].y == last.y &&
                                  buf[fresh].time == last.time))
                fresh++;
        }
        last = buf[0];
        have_last = true;
        if (fresh <= 1) return;   // nothing GLFW didn't already deliver

        HWND hwnd = glfwGetWin32Window(window);
        input.path_n = 0;
        for (int i = fresh - 1; i >= 0; --i) {
            // display points are 16-bit; sign-extend for monitors left of / above primary
            POINT p = { buf[i].x > 32767 ? buf[i].x - 65536 : buf[i].x,
                        buf[i].y > 32767 ? buf[i].y - 65536 : buf[i].y };
            ScreenToClient(hwnd, &p);
            input.push_path(p.x, p.y);
        }
    }
};
static MouseHistory g_mouse_history;
#endif

// Sample depth at a pixel to decide if the cursor is on the model. Returns false
// when no fresh sample is available this frame — the caller keeps the previous
// on_model value (matters on webgpu, where the plane cache lands a frame or two
// after the screen-buffer render; forcing false there would drop the sculpt latch
// between rapid consecutive strokes).
static bool sample_on_model(Renderer& renderer, int x, int y, int screen_h, bool* on_model) {
    // The on-model latch samples the screen-target's linear-depth attachment
    // (0, `-viewPos.z`, cleared to 1000) via the renderer's CPU plane cache — no
    // in-frame readback on either backend. (GL used to glReadPixels the swapchain
    // depth here: one more per-frame sync, gone with the shared plane cache.)
    (void)screen_h;
    float d = 1000.0f;
    if (!renderer.sample_depth(x, y, &d)) return false;
    *on_model = d < 500.0f;  // clear = 1000; geometry linear-depth is « far plane (~100)
    return true;
}

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
// Set by SIGTERM / SIGHUP, consumed by the frame loop (a handler may only touch a
// sig_atomic_t; the dialog is raised from the loop).
static volatile std::sig_atomic_t g_quit_signal = 0;
#endif

int main(int argc, char* argv[]) {
    debug_console::init();
    bench::init();
    bool cli_use_topology = true;
    int max_level = MULTIRES_MAX_LEVEL;
    std::string cli_open_path;
#ifdef CHISEL_DEBUG_MULTIRES
    // Debug builds default to a tiny GPU undo ring so the wrap/evict path gets
    // exercised in a few big strokes (no flags needed). --toaster/--ring-mb still override.
    UndoStack::ring_max_bytes = 4ull * 1024ull * 1024ull;
#endif
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--mirror=spatial") == 0)
            cli_use_topology = false;
        else if (std::strcmp(argv[i], "--toaster") == 0) {
            UndoStack::max_bytes      = 256ull * 1024ull * 1024ull;  // CPU history cap
            UndoStack::ring_max_bytes =  64ull * 1024ull * 1024ull;  // GPU ring cap
        }
        else if (std::strncmp(argv[i], "--ring-mb=", 10) == 0) {
            // debug/test: shrink the GPU undo ring to force a wrap+evict quickly
            unsigned long mb = std::strtoul(argv[i] + 10, nullptr, 10);
            if (mb > 0) UndoStack::ring_max_bytes = (size_t)mb * 1024ull * 1024ull;
        }
        else if (std::strncmp(argv[i], "--max-level=", 12) == 0) {
            // Raise (or lower) the subdivision-level cap. Levels past 9 are
            // CPU-heavy (switches/merge/remesh ~4x per level) and the GPU
            // guard still refuses what the device can't hold.
            long lv = std::strtol(argv[i] + 12, nullptr, 10);
            if (lv >= 1 && lv <= 12) max_level = (int)lv;
            else std::printf("[cli] ignoring --max-level=%s (want 1..12)\n", argv[i] + 12);
        }
        else if (argv[i][0] != '-')
            cli_open_path = argv[i];  // project/model to open once the scene is up
    }
    if (max_level != MULTIRES_MAX_LEVEL)
        std::printf("[multires] subdivision cap: L%d (--max-level)\n", max_level);
    if (!cli_use_topology)
        std::printf("[mirror] using spatial-hash fallback (--mirror=spatial)\n");
    std::printf("[undo] history budget: %zu MB CPU / %zu MB GPU ring%s\n",
                UndoStack::max_bytes / (1024 * 1024),
                UndoStack::ring_max_bytes / (1024 * 1024),
                UndoStack::max_bytes < 1024ull * 1024ull * 1024ull ? " (--toaster)" : "");

    // Init GLFW
    // GLFW errors went nowhere before this. That matters more than it sounds: several
    // GLFW calls fail by emitting an error and doing nothing, rather than returning a
    // status — glfwSetCursorPos under Wayland is exactly one of those. A silent no-op is
    // indistinguishable from a working call until the symptom shows up somewhere else.
    glfwSetErrorCallback([](int code, const char* desc) {
        std::printf("[glfw] error %d: %s\n", code, desc ? desc : "(none)");
        std::fflush(stdout);
    });

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }

// glfwGetPlatform is GLFW 3.4+; the Linux CI image (Ubuntu 20.04) ships 3.3.
#if !defined(__EMSCRIPTEN__) && (GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4)
    // Which platform GLFW actually picked decides what the window system will let the
    // app do — cursor warping above all. This was only ever printed on the wgpu path,
    // so the GL build gave no way to tell native Wayland from XWayland.
    {
        const int plat = glfwGetPlatform();
        const char* name = plat == GLFW_PLATFORM_WAYLAND ? "Wayland"
                         : plat == GLFW_PLATFORM_X11     ? "X11"
                         : plat == GLFW_PLATFORM_COCOA   ? "Cocoa"
                         : plat == GLFW_PLATFORM_WIN32   ? "Win32" : "unknown";
        std::printf("[win] GLFW platform: %s\n", name);
    }
#endif

#ifdef CHISEL_BACKEND_WEBGPU
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // WebGPU owns the surface; no GL context
#else
    // GL 4.3 is where compute shaders and SSBOs became core, and the kernels are
    // all #version 430. On a 3.3 context they ran through extensions, outside the
    // spec. 4.3 also makes KHR_debug core, which init below relies on. A driver
    // without 4.3 falls back to 3.3 at window creation (compute then gates off).
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #ifdef CHISEL_DEBUG
    const bool gl_debug_verbose = true;
  #else
    const char* gl_debug_env = std::getenv("CHISEL_GL_DEBUG");
    const bool gl_debug_verbose = gl_debug_env && gl_debug_env[0] == '1';
  #endif
    // A debug context makes some drivers (NVIDIA above all) report more, at a
    // speed cost, so only the verbose mode asks for one.
    if (gl_debug_verbose) glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

    // WM_CLASS must match StartupWMClass=Chisel in chisel.desktop so the running
    // window groups under the launcher entry in the taskbar/dock (X11). No window
    // manager in a browser — these hints don't exist in Emscripten's GLFW.
#if !defined(__EMSCRIPTEN__)
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "Chisel");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "Chisel");
  #ifdef GLFW_WAYLAND_APP_ID
    // Wayland ignores glfwSetWindowIcon; the compositor finds the titlebar/taskbar
    // icon by matching app_id to a .desktop file name. "chisel" = chisel.desktop
    // (the one AppRun installs). Without it KWin shows its generic "W" icon.
    glfwWindowHintString(GLFW_WAYLAND_APP_ID, "chisel");
  #endif
#endif

    // Initial window size.
#if defined(__EMSCRIPTEN__)
    // Emscripten's GLFW shim has no real monitor and returns a NULL video mode
    // (libglfw.js: glfwGetVideoMode => 0), so mode->width/height read 0 and
    // glfwCreateWindow(0,0) fails ("width <= 0 ... return 0"). Size from the browser
    // viewport instead; the in-loop resize handler tracks the canvas thereafter.
    int init_w = EM_ASM_INT({ return window.innerWidth  | 0; });
    int init_h = EM_ASM_INT({ return window.innerHeight | 0; });
    if (init_w <= 0) init_w = 1280;
    if (init_h <= 0) init_h = 720;
#else
    // Get primary monitor for fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int init_w = mode->width, init_h = mode->height;
#endif

    // Start windowed but maximized (easier for development)
#if !defined(__EMSCRIPTEN__)
    // Actually maximized: a window sized to the full video mode is taller than the work
    // area once the title bar and borders are added, so it hangs under the taskbar —
    // and in that state the Intel Vulkan driver on Windows reports the swapchain
    // suboptimal and shows black triangles while orbiting/sculpting (gone after a
    // maximize). GNOME auto-maximizes a screen-sized window and hid this on Linux;
    // KWin doesn't, so Plasma showed the overhang.
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(init_w, init_h, "Chisel", nullptr, nullptr);
#ifdef CHISEL_BACKEND_GL
    if (!window) {
        std::printf("[gl] no GL 4.3 context, retrying with 3.3 (GPU brushes will be off)\n");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        window = glfwCreateWindow(init_w, init_h, "Chisel", nullptr, nullptr);
    }
#endif
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }
    set_window_icon(window);

#ifdef CHISEL_BACKEND_WEBGPU
    // ---- WebGPU: instance -> surface(X11) -> adapter -> device -> seam ----
    // (mirrors src/gpu/wgpu_window.cpp; the seam owns resources, the window owns
    // surface/device creation + per-frame acquire/present.)
    int fbw = init_w, fbh = init_h;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    if (!wgpu_init(window, fbw, fbh)) return 1;   // gpu/wgpu_surface.cpp
    // CHISEL_VERSION is baked at CONFIGURE time from git describe, so a build tree
    // configured before a tag keeps reporting the old one — reconfigure, do not just rebuild.
    std::printf("Chisel %s (WebGPU)\n", CHISEL_VERSION);
#else
    glfwMakeContextCurrent(window);
    glfwSwapInterval(bench::vsync() ? 1 : 0); // vsync (off only for a CHISEL_PERF_VSYNC=0 run)

    // Load OpenGL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL\n");
        return 1;
    }

    std::printf("Chisel %s\n", CHISEL_VERSION);
    std::printf("OpenGL %s\n", glGetString(GL_VERSION));
    std::printf("Renderer: %s\n", glGetString(GL_RENDERER));

    chisel_init_gl_debug(gl_debug_verbose);   // driver error reports, see chisel_debug.h
    gpu::set_app_device(gpu::gl_device());
    // Capture SSBO size limits for the subdivision guard. GL 4.3 enum: on a plain
    // 3.3 context the query fails silently (GL_INVALID_ENUM) and the value keeps
    // its conservative fallback — matching compute being unavailable there anyway.
    {
        GLint64 ssbo_max = (GLint64)(128ll << 20);
        glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo_max);
        gpu::DeviceLimits dl;
        dl.max_buffer_size          = (uint64_t)ssbo_max;
        dl.max_storage_binding_size = (uint64_t)ssbo_max;
        gpu::set_device_limits(dl);
        dl = gpu::device_limits();  // re-read: the dev-hook clamp may have applied
        std::printf("[gpu] device limits: maxStorageBlock %llu MB\n",
                    (unsigned long long)(dl.max_storage_binding_size >> 20));
    }
#endif

    ComputeState compute;
    compute.init();
    if (compute.supported) {
        compute.init_draw_accum();
        compute.init_draw_apply();
        compute.init_draw_mirror_apply();
        compute.init_draw_accum_symmetrize();
        compute.init_smooth();
        compute.init_stroke_smooth();
        compute.init_mirror_project();
        compute.init_dirty_args();
        compute.init_normals_expand();   // after dirty_args: it dispatches indirect
        compute.init_touched_fold();     // after dirty_args: it dispatches indirect
        compute.init_block_cull();
        compute.init_crease();
        compute.init_pinch();
        compute.init_move();
        compute.init_limb();
        compute.init_pose();
        compute.init_mask();
        compute.init_density();
        compute.init_color();
        compute.init_compute_normals();
        compute.init_multires_diff();
        compute.init_multires_apply();
        compute.init_cascade();
        compute.init_alpha();   // shared brush-alpha stamp buffers (all dab kernels)
        compute.undo_ring_set_budget(UndoStack::ring_max_bytes);  // blood-moon 3b-iv (decoupled)
        compute.undo_ring_selftest();                        // no-op in release
        compute.init_remesh_select();
        compute.init_remesh_grow_selection();
        compute.init_remesh_mirror_selection();
        compute.init_remesh_find_pinned();
        compute.init_remesh_smooth_weights();
        compute.init_remesh_smooth();
        compute.init_remesh_seam_snap_weld();
    }

    // Init systems
    InputState input;
    // Before anything reads the brush fields. Missing/corrupt storage is a no-op that
    // leaves the constructor defaults standing, so there is nothing to check here.
    settings_load(input);
    // Dev: the phone layout on desktop, for layout work (touch mode is otherwise web-only).
    if (std::getenv("CHISEL_TOUCH_UI")) input.touch_mode = true;
#ifdef __EMSCRIPTEN__
    g_settings_input = &input;   // arms the pagehide flush above
    // A phone starts in touch mode rather than at the first finger, so its saved UI
    // scale (touch-only) is in force from the first frame instead of only after a tap.
    // Same media query the shell uses to show the modifier buttons up front.
    if (EM_ASM_INT({ return (window.matchMedia && matchMedia('(pointer: coarse)').matches) ? 1 : 0; }))
        input.touch_mode = true;
#endif
    setup_input_callbacks(window, &input);
    setup_char_callback(window);
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
    // Logout / shutdown / `kill` ask first, like the title-bar X. SIGINT (Ctrl+C in a
    // terminal) is left alone: that is the developer's hard stop.
    std::signal(SIGTERM, [](int) { g_quit_signal = 1; });
    std::signal(SIGHUP,  [](int) { g_quit_signal = 1; });
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Persist imgui.ini in the user's config dir (XDG) rather than the CWD, so the
    // window layout survives regardless of where the app is launched from and works
    // under a read-only AppImage mount. Static so the backing string outlives the
    // context — ImGui keeps the pointer and re-reads it on every settings save.
    static std::string imgui_ini;
    {
        const char* xdg  = std::getenv("XDG_CONFIG_HOME");
        const char* home = std::getenv("HOME");
        std::string base = (xdg && *xdg)   ? std::string(xdg)
                         : (home && *home) ? std::string(home) + "/.config"
                         :                   std::string();
#ifdef _WIN32
        // Windows has neither XDG_CONFIG_HOME nor HOME — use %APPDATA%.
        if (base.empty()) {
            const char* appdata = std::getenv("APPDATA");
            if (appdata && *appdata) base = appdata;
        }
#endif
        if (!base.empty()) {
            std::string dir = base + "/chisel";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);  // harmless if it already exists
            imgui_ini = dir + "/imgui.ini";
            ImGui::GetIO().IniFilename = imgui_ini.c_str();
        }
        // else (no config dir resolvable): leave ImGui's default ("imgui.ini" in the CWD).
    }
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    ImGui::GetStyle().HoverDelayNormal = 0.0f;
    ImGui::GetStyle().HoverDelayShort  = 0.0f;
    ui_skin_load_fonts();
#ifdef CHISEL_BACKEND_WEBGPU
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplWGPU_InitInfo imguiInit = {};
    imguiInit.Device = g_device;
    imguiInit.NumFramesInFlight = 3;
    imguiInit.RenderTargetFormat = g_surface_fmt;
    imguiInit.DepthStencilFormat = WGPUTextureFormat_Undefined;
    if (!ImGui_ImplWGPU_Init(&imguiInit)) {
        std::fprintf(stderr, "ImGui_ImplWGPU_Init failed\n"); return 1;
    }
#else
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
#endif
#ifdef __EMSCRIPTEN__
    // ImGui must not read GLFW's desynced pointer coords either — take the cursor
    // callback back; chisel_set_pointer feeds ImGui the DOM position instead.
    input_web_take_cursor_callback(window);
    input_web_feed_imgui_wheel();
#endif

    Renderer renderer;
    renderer.init();

    // Pen tablet: XInput2 on X11/XWayland, a Wacom's own hidraw reports where readable
    // (OpenTabletDriver setups), and on native Wayland the compositor's tablet protocol,
    // which also supplies the pen's position and clicks. No-op if absent. Detects hotplug.
    Tablet tablet;
    tablet.init(window);
    wl_pointer_warp_init(window);   // edge wrap on Wayland (tablet.cpp)
    if (tablet.available()) {
        std::snprintf(input.notification, sizeof(input.notification),
                      "Pen pressure: tablet detected");
        input.notification_timer = 2.0f;
    }
    bool prev_tablet_avail = tablet.available();
    // Device-profile arbiter state (see the switch block in the frame loop).
    unsigned long prev_pen_samples = tablet.sample_count();
    double last_pen_input_time = -1e9;   // "no pen ever" without special-casing

    // GPU sculpt shaders read the mask buffer to gate locked vertices; the paint
    // shader writes the color VBO directly. compute.mask_ssbo / color_ssbo alias the
    // renderer's owned vbo_mask / vbo_color and are (re)pointed by Scene::bind_active_
    // after every upload_mesh — a realloc makes a new handle (Step 3a), so a set-once
    // alias here would go stale on the first topology change.

    TextOverlay text;
    text.init(renderer.gpu_dev);

    Camera camera;

    // The opening ball. settings_load ran back at startup, so sphere_kind is the
    // user's persisted choice by now. A UV sphere is a base cage, not a subdivided
    // icosahedron: it locks at level 0 and subdiv_level has to say so, because this is
    // what Scene() hands multires_stack_init_from_lock as the stack's base_level.
    const bool uv_start = (input.sphere_kind == InputState::SphereKind::UV);
    if (uv_start) input.subdiv_level = 0;
    Scene scene(uv_start ? uv_sphere(32, 16) : icosphere(input.subdiv_level),
                renderer, compute, input.subdiv_level);
    scene.set_mirror_topology(cli_use_topology);
    // -1, not the level: that argument is a key into the icosphere mirror-map cache,
    // and handing it a UV sphere would park a 482-vert map in the level-0 slot.
    scene.refresh_mirror_map(uv_start ? -1 : input.subdiv_level);
    scene.sync();
    input.mesh_locked = true;

    // Phase 1 GPU residency: mirror the startup entity's locked level. Later
    // mutations refresh via refresh_active_gpu_residency() inside the loop.
    scene.active_entity().multires_gpu.supported = compute.supported;
    scene.active_entity().multires_gpu.dev       = &compute.gpu_dev;
    scene.active_entity().multires_gpu.upload_level(scene.active_multires(),
                                                    scene.active_multires().current_level);

    Mesh* mesh = &scene.active_mesh();
    MultiresStack* multires = &scene.active_multires();
    int perf_prev_level = multires->current_level;   // CHISEL_PERF: level-switch frames

    // Tick-driven voxel merge: non-null while a merge job is in flight (advanced one
    // budgeted step per frame so the window stays responsive). See sdf.h / CHANGES.
    VoxelMergeJob* vmerge_job = nullptr;

    // scene.edit_mark() as of the last save or load (or startup): when the two
    // match there is nothing to save, and a close from outside needn't ask.
    uint64_t clean_edit_mark = scene.edit_mark();

    // Pre-remesh rescue copy (scene_snapshot.h). Held only for the two ops you
    // cannot judge until the old model is gone: '/' and 'J'. rescue_shown is the
    // last countdown figure put on screen, -1 meaning "not announced yet".
    SceneSnapshot rescue;
    int rescue_shown = -1;
    // Q/E object spin (SELECT mode). Each entity's pivot is latched in its own
    // ObjectXformTarget on the gesture's first frame — a bounding centre is
    // rotation-invariant in exact arithmetic but not in floating point, and
    // recomputing it every frame lets the piece wander while it turns. rot_dirty is
    // reused rather than rebuilt per frame — the sync wants a full dirty list and
    // this runs every frame a key is down.
    // The view axis is latched for the same reason: nudging the
    // camera mid-turn would otherwise bend the axis under the spin, and the undo
    // record could no longer be one angle about one line.
    Vec3  rot_axis  = {0, 0, 1};
    bool  rot_active = false;
    double rot_last_time = 0.0;
    std::vector<uint32_t> rot_dirty;
    // Same latch, same reason, for the first frame of an RMB scale drag.
    Vec3  scale_pivot = {0, 0, 0};
    // Undo for the Select-mode object transforms (move / scale / Q-E spin). A
    // separate stack because UndoStack is per-entity and one gesture can turn
    // several selected meshes; interleaved with it by sequence number, not
    // siloed, because UndoEntry holds ABSOLUTE positions and undoing a stroke
    // that predates a transform would teleport its vertices. See object_xform.h.
    ObjectXformStack xforms;
    // Set when a merge chains its adaptive remesh: that remesh must not take its
    // own snapshot, or it would overwrite the merge's with the post-merge state,
    // and the merge is the step the user wants back.
    bool remesh_chained_from_merge = false;

    Vec3 mesh_center;
    float mesh_radius;
    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
    camera.set_target(mesh_center);
    camera.distance = mesh_radius * 2.5f;

    Vec3 last_sculpt_point = mesh_center;
    BrushStroke brush_stroke;
    brush_stroke.compute = &compute;

    // Brush-alpha (stamp) library — built-in pool + user-loaded images. The selected
    // entry's bitmap is uploaded to the compute state whenever input.active_alpha
    // changes; every falloff-computing dab kernel then modulates by it.
    AlphaLibrary alpha_lib;
    alpha_lib.init_builtins();
    int last_uploaded_alpha = -1;
    // Set after a level switch that froze the frame; the next frame's queued
    // level-switch input (D mashed during the freeze) is swallowed once, so a
    // slow build can't queue a pile of further builds.
    bool swallow_level_switch = false;
    // True while the viewport shows colormap(density) instead of albedo (paint
    // brush with the density target). Tracks the enter/exit edge for the restore.
    bool density_view_active = false;
    // Undo history is per-model: each MeshEntity owns its UndoStack. Undo/redo
    // always act on the active entity via scene.active_undo(), resolved fresh at
    // each use so a mid-frame selection change targets the right stack.
    AppState app_state = AppState::IDLE;
    InsertController insert_ctrl(scene, renderer);
    RetopoState retopo;   // mode 0: sketched strokes + the quads Build made from them
    uint64_t clean_retopo_rev = 0;   // retopo.rev as of the last save or load
    // Two-finger take-back (see InputState::touch_revert_requested).
    Camera   camera_at_press;
    bool     touch_undo_on_commit   = false;
    uint64_t pushes_at_stroke_start = 0;
    bool     pen_up_pending = false;   // the release frame laid its tail; finalize next

    std::string default_browse_path = ".";
    {
        namespace fs = std::filesystem;
        const char* home = nullptr;
#ifdef _WIN32
        home = std::getenv("USERPROFILE");
#else
        home = std::getenv("HOME");
#endif
        if (home) {
            fs::path p = fs::path(home) / "Desktop" / "chisel-sculpts";
            std::error_code ec;
            fs::create_directories(p, ec);
            default_browse_path = p.string();
        }
    }
    std::string current_project_path;
#ifdef __EMSCRIPTEN__
    // Filename fields for the web save/export prompts (remember the last name).
    char web_save_name[96]   = "sculpt";
    char web_export_name[96] = "sculpt";
#endif
    std::string error_popup_msg;
    bool error_popup_trigger = false;

    bool screen_buffers_dirty = true;

    // Idle frame skip (see the block before ImGui::NewFrame).
    double last_active_time = glfwGetTime();
    bool idle_last_frame = false;
    int idle_prev_w = 0, idle_prev_h = 0;
    unsigned long idle_prev_pen = tablet.sample_count();

    // FPS counter
    double fps_last_time = glfwGetTime();
    int fps_frame_count = 0;
    float fps_display = 0.0f;

    // Settings autosave clock. Its own timebase because the FPS counter resets its
    // reference every second, which would hand settings_tick a bogus delta.
    double settings_last_time = glfwGetTime();

    // Store windowed position/size for fullscreen restore
    int windowed_x = 0, windowed_y = 0, windowed_w = init_w, windowed_h = init_h;
    glfwGetWindowPos(window, &windowed_x, &windowed_y);
    glfwGetWindowSize(window, &windowed_w, &windowed_h);

    // Paint/mask persistence across cascades lives in the multires stack's
    // finest-level planes: multires_sync_paint (multires_stack.cpp) folds the
    // working arrays in before a cascade, cascade_to_level reads the prefix
    // back out. Full fidelity — fine-level detail survives level round-trips
    // exactly; a coarse repaint re-interpolates only the region it touched.

    // Main loop. The body is a lambda capturing every setup local by reference so the
    // same code drives both targets: native spins it in a blocking while-loop; the web
    // hands it to emscripten_set_main_loop_arg (below), since the browser owns the event
    // loop. simulate_infinite_loop=true throws to keep main()'s stack frame — and thus
    // every captured local and the lambda itself — alive across browser-driven frames.
    // What the frame's moved-out parts reach (app.h): references to the locals above.
    App app{window, input, scene, mesh, multires, camera, compute, brush_stroke, alpha_lib,
            xforms, vmerge_job, app_state, mesh_center, mesh_radius, screen_buffers_dirty,
            clean_edit_mark, cli_open_path, default_browse_path, current_project_path,
            error_popup_msg, error_popup_trigger, renderer, rescue, rescue_shown,
            max_level, swallow_level_switch, remesh_chained_from_merge, text, fps_display,
            rot_axis, rot_active, rot_last_time, rot_dirty, scale_pivot,
            retopo, clean_retopo_rev
#ifdef __EMSCRIPTEN__
            , web_save_name, web_export_name
#endif
#ifdef CHISEL_BACKEND_WEBGPU
            , fbw, fbh
#endif
    };
    auto frame = [&]() {
        input.begin_frame();

        // Resolve the window/drawable size and, on web, reconfigure the surface + depth
        // BEFORE polling input. glfwPollEvents scales the incoming pointer by the current
        // canvas backing (canvas.width / boundingRect.width); if the surface were
        // reconfigured *after* the poll, the first mouse event of a resize frame would
        // still be scaled by the stale backing → an out-of-range cursor for a frame
        // (seen as the ring jumping off the pointer right after a resize). Doing it first
        // means the pointer is already in the freshly-resized space when events arrive.
        int win_w, win_h;
#if defined(__EMSCRIPTEN__)
        // The custom shell's canvas is 100vw x 100vh, so its CSS (client) size is the
        // authoritative window size. glfwGetFramebufferSize can't be trusted here: the
        // canvas backing may be resized outside GLFW's knowledge, and stale GLFW size
        // vs. a resized surface produces attachment-size-mismatch validation errors.
        // UI scale (touch mode only): the shell lays the canvas stage out at 1/scale of
        // the window and CSS-scales it back down, so the app sees a bigger window — more
        // room, more pixels — exactly like page zoom. Set before clientWidth is read so
        // this frame's layout already has the new size. See shell.html #stage.
        {
            static int applied = -1;
            const int pct = input.touch_mode ? input.ui_scale_touch : 100;
            if (pct != applied) {
                applied = pct;
                EM_ASM({ document.documentElement.style.setProperty('--ui', String($0 / 100)); }, pct);
            }
        }
        win_w = EM_ASM_INT({ return (Module.canvas.clientWidth  || window.innerWidth)  | 0; });
        win_h = EM_ASM_INT({ return (Module.canvas.clientHeight || window.innerHeight) | 0; });
#else
        glfwGetFramebufferSize(window, &win_w, &win_h);
#endif
#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
        // SIGTERM / SIGHUP (logout, shutdown, `kill`) asks like the title-bar X.
        if (g_quit_signal) {
            g_quit_signal = 0;
            if (!input.quit_after_save) input.close_requested = true;
        }
#endif
#ifndef __EMSCRIPTEN__
        // A close from outside (title-bar X, logout, a signal). Nothing unsaved and
        // nothing mid-flight: just close. Otherwise bring up the quit dialog and pull
        // the window forward, since the request may have come from the taskbar or a
        // logout while the window sat behind others or minimised. On Wayland the
        // compositor decides whether to raise it or only flash its taskbar entry.
        // Checked before the minimised-window early-out below, so a minimised window
        // gets restored. systemd still sends SIGKILL if nobody answers before its
        // stop timeout. ESC stays a plain ask: a stray key shouldn't close the app.
        if (input.close_requested) {
            input.close_requested = false;
            const bool busy = app_state != AppState::IDLE || vmerge_job
                || input.remesh_requested || input.remesh_in_progress;
            if (scene.edit_mark() == clean_edit_mark && retopo.rev == clean_retopo_rev && !busy) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            } else {
                input.quit_requested = true;
                if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) glfwRestoreWindow(window);
                glfwFocusWindow(window);
                glfwRequestWindowAttention(window);
            }
        }
#endif
        if (win_w == 0 || win_h == 0) {
            glfwPollEvents();      // keep the browser event loop breathing on a hidden frame
            input.end_frame();
            return;   // skip this frame (lambda body; == `continue` in the native loop)
        }
#ifdef CHISEL_BACKEND_WEBGPU
        // Reconfigure the surface + depth on resize; per-pass viewport is set by the
        // seam (no global glViewport). On web, configureSurface also sets the canvas
        // backing (dpr=1), keeping color (surface) and depth attachments the same size.
        if (win_w != fbw || win_h != fbh || g_surface_reconfigure) {
            g_surface_reconfigure = false;
            fbw = win_w; fbh = win_h;
            configureSurface(fbw, fbh);
            makeDepth(fbw, fbh);
            // The pick planes (depth/normal/triid) only re-render on this flag. Without
            // it they stay at the old size + framing after a resize while the visible
            // model re-renders at the new size, so the cursor latch / normal / brush
            // anchor sample a stale-aspect plane → cursor offset until the camera moves.
            screen_buffers_dirty = true;
        }
#else
        glViewport(0, 0, win_w, win_h);
#endif

        debug_console::pump();
#ifndef __EMSCRIPTEN__
        // After an idle (skipped) frame, sleep until an event instead of spinning; the
        // timeout keeps the signal/close checks and settings autosave ticking. The web
        // can't block — the browser's rAF paces it and a skipped frame costs ~nothing.
        if (idle_last_frame) glfwWaitEventsTimeout(0.1);
        else
#endif
        glfwPollEvents();
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
        g_mouse_history.pull(window, input);
#endif
        wl_pointer_warp_dispatch();   // the pointer-enter serial a Wayland edge wrap needs
        tablet.poll(brush_stroke.is_active());
        if (tablet.available() && !prev_tablet_avail) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Pen pressure: tablet connected");
            input.notification_timer = 2.0f;
        }
        prev_tablet_avail = tablet.available();

        // Burger-menu projection settings → camera. Pushed every frame rather than on
        // edit, so a project load (which overwrites the whole camera, fov included)
        // cannot strand the view on a projection the menu no longer shows.
        {
            // Ortho keeps the historical 45: it is only a framing scale there, and
            // holding it fixed means toggling perspective at 45 changes nothing but
            // the convergence, which is what makes the two comparable at a glance.
            static float prev_eff_fov = 45.0f;
            float eff_fov = input.camera_perspective ? input.camera_fov : 45.0f;
            if (eff_fov != prev_eff_fov) {
                // Dolly-compensate. Framing is distance*tan(fov/2) in BOTH projections,
                // so holding that product fixed keeps the model the same size on screen
                // while the lens angle changes — a lens change, not a zoom. Without it
                // the slider just scales the view and reads as a duplicate of scroll.
                const float kDeg2Half = 3.14159265358979323846f / 360.0f;
                float ta = std::tan(prev_eff_fov * kDeg2Half);
                float tb = std::tan(eff_fov * kDeg2Half);
                camera.distance = std::max(0.05f, std::min(200.0f, camera.distance * ta / tb));
                prev_eff_fov = eff_fov;
            }
            // The projection changed, so the cached screen buffers (depth/normal/triid)
            // describe a view that no longer exists — same invalidation orbit/pan/zoom
            // do. Without this the next pen-down unprojects against a stale plane cache
            // and lands its first dab in the wrong place; it self-heals on any camera
            // move, which is exactly what makes it easy to miss in testing.
            if (camera.fov != eff_fov || camera.perspective != input.camera_perspective)
                screen_buffers_dirty = true;
            camera.perspective = input.camera_perspective;
            camera.fov = eff_fov;
        }

        // Device-driven profile switch. Both profiles stay loaded; whichever device is
        // actually producing input owns the live brush settings.
        //
        // The pen wins on evidence (its sample counter moved), the mouse only on evidence
        // PLUS a quiet window, because on X11 the stylus also drives the core pointer —
        // every pen motion is indistinguishable from a mouse motion at the GLFW layer, so
        // without the quiet window a pen stroke would flip to Mouse on its own cursor
        // movement. The web pen path has the same shape (chisel_set_pointer fires for pen
        // events too), so one rule covers both.
        //
        // Latching on the *other* device rather than on pen idleness is deliberate:
        // a timeout-to-mouse would re-tune strength/hardness/spacing every time you
        // lifted the pen to think, which reads as the app glitching. (Brush size used
        // to be the loudest symptom of that; it went global on 2026-08-07, but the
        // brush-feel fields it still swaps make the same argument.)
        if (!input.sculpting && !input.settings_menu_open &&
            input.drag_mode == InputState::DragMode::NONE &&
            input.slider_mode == InputState::SliderMode::NONE) {
            const double kPenHoldSec = 0.25;
            double now = glfwGetTime();
            unsigned long samples = tablet.sample_count();
            bool pen_reported = (samples != prev_pen_samples);
            if (pen_reported) { prev_pen_samples = samples; last_pen_input_time = now; }

            bool cursor_moved = (input.mouse_x != input.prev_mouse_x ||
                                 input.mouse_y != input.prev_mouse_y);
            InputProfile want = input.active_profile;
            if (pen_reported)
                want = InputProfile::TABLET;
            else if (cursor_moved && now - last_pen_input_time > kPenHoldSec)
                want = InputProfile::MOUSE;

            if (want != input.active_profile) {
                input.switch_profile(want);
                std::snprintf(input.notification, sizeof(input.notification),
                              want == InputProfile::TABLET ? "Tablet settings"
                                                           : "Mouse settings");
                input.notification_timer = 1.2f;
            }
        }

        // ---- Idle frame skip ----
        // Nothing on screen changes unless something happened, so a model you're only
        // looking at shouldn't cost a full scene + UI repaint every vsync (an idle web
        // tab held the GPU at ~100%). A frame is active when there's new input (ImGui's
        // queue sees every key/button/wheel/pointer event on both targets), the window
        // resized, the pen reported, or multi-frame work is in flight. Drawing then
        // continues for kIdleTail after the last active frame: that covers everything
        // that keeps moving on its own once input stops (button springs, tooltip
        // delays, the cursor ring's LERP, a request flag a finishing job chains into)
        // without a trigger per animation. A skipped frame presents nothing: the
        // browser keeps showing the last canvas, native sleeps in glfwWaitEvents.
        bool woke_from_idle = false;
        {
            const double kIdleTail = 1.0;
            const double now = glfwGetTime();
            bool active = !ImGui::GetCurrentContext()->InputEventsQueue.empty()
                || input.wake_requested
                || input.path_n > 0
                || input.mouse_x != input.prev_mouse_x || input.mouse_y != input.prev_mouse_y
                || tablet.sample_count() != idle_prev_pen
                || win_w != idle_prev_w || win_h != idle_prev_h;
#ifdef __EMSCRIPTEN__
            active = active || !g_web_import_path.empty() || !g_web_alpha_path.empty();
#endif
            const bool busy = bench::active()
                || app_state != AppState::IDLE
                || brush_stroke.is_active() || brush_stroke.reconciling()
                || brush_stroke.dab_readbacks_pending()
                || vmerge_job
                || input.remesh_requested || input.remesh_in_progress
                || input.voxel_merge_requested || input.voxel_merge_in_progress
                || input.drag_mode != InputState::DragMode::NONE
                || input.touch_gesture
                || input.slider_mode != InputState::SliderMode::NONE
                || input.notification_timer > 0.0f || input.mirror_unavailable_timer > 0.0f
                // Same condition the refresh below runs under, so it can't keep us awake
                // while the menu defers it.
                || (screen_buffers_dirty && !input.settings_menu_open)
                || ImGui::GetIO().WantTextInput;   // keep a text field's caret alive
            // A held key or button sends no events (ImGui drops key repeats), yet a held
            // rotate key turns the model every frame. ImGuiKey's named range includes
            // the mouse buttons.
            bool held = false;
            for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END && !held; ++k)
                held = ImGui::IsKeyDown((ImGuiKey)k);
            input.wake_requested = false;
            idle_prev_w = win_w; idle_prev_h = win_h;
            idle_prev_pen = tablet.sample_count();

            if (active || busy || held) last_active_time = now;
            if (now - last_active_time > kIdleTail) {
                // Settings still save on their debounce while nothing is drawn.
                settings_tick(input, (float)(now - settings_last_time));
                settings_last_time = now;
                idle_last_frame = true;
                input.end_frame();
                return;   // == `continue` in the native loop, like the hidden-frame skip
            }
            if (idle_last_frame) {
                woke_from_idle = true;
                idle_last_frame = false;
                // The FPS readout holds its last real value until a full drawn second.
                fps_last_time = now;
                fps_frame_count = 0;
            }
        }

        mesh = &scene.active_mesh();
        multires = &scene.active_multires();

#ifdef CHISEL_BACKEND_WEBGPU
        ImGui_ImplWGPU_NewFrame();
#else
        ImGui_ImplOpenGL3_NewFrame();
#endif
        ImGui_ImplGlfw_NewFrame();
        // ImGui measures DeltaTime from the last drawn frame, so the wake frame after an
        // idle stretch would see the whole gap: a toast raised by the waking keypress
        // would be counted down by minutes and vanish before it was ever drawn.
        if (woke_from_idle) ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
#ifdef __EMSCRIPTEN__
        // Re-take DisplaySize from the canvas, for the same reason win_w is read from
        // clientWidth up top: ImGui_ImplGlfw_NewFrame sources it from glfwGetWindowSize,
        // which tracks the backing GLFW *believes* it has, and the shell resizes the
        // canvas outside GLFW's knowledge. The stale value is what loses the burger menu
        // after a window resize — it is the only right-anchored island, positioned at
        // win_w - margin, and SetNextWindowPos(ImGuiCond_Always) counts as API-set, which
        // makes ImGui skip its clamp-into-viewport rescue. So once win_w outruns a stale
        // DisplaySize the island is placed past the edge with nothing to pull it back.
        // Scale is 1:1 because configureSurface sets the canvas backing at dpr=1.
        ImGui::GetIO().DisplaySize = ImVec2((float)win_w, (float)win_h);
        ImGui::GetIO().DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
#elif defined(CHISEL_BACKEND_WEBGPU)
        // The swapchain was configured from the size read at the top of the frame,
        // before glfwPollEvents; ImGui_ImplGlfw_NewFrame reads the size again *after*
        // it. On the frame a resize lands (un-maximize is the reliable trigger) the two
        // differ, ImGui's renderer sets a scissor taller than the swapchain texture
        // ("Scissor Rect 1920x1061 is not contained in the render target 1920x1017"),
        // and wgpu-native treats a submit-time validation error as fatal. Clamp ImGui
        // to the configured size; the next frame reconfigures and catches up.
        {
            ImGuiIO& io = ImGui::GetIO();
            const float sx = io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0f;
            const float sy = io.DisplayFramebufferScale.y > 0.0f ? io.DisplayFramebufferScale.y : 1.0f;
            io.DisplaySize.x = std::min(io.DisplaySize.x, (float)fbw / sx);
            io.DisplaySize.y = std::min(io.DisplaySize.y, (float)fbh / sy);
        }
#endif
        ui_skin_begin_frame(input);      // font + style follow the Appearance setting
        ImGui::NewFrame();

        // A drop on an untouched scene — the default sphere with an empty
        // undo record, never saved, nothing inserted — has nothing worth a
        // prompt: open it straight away. (mesh_locked is useless here: it is
        // set at startup. Converted before the overlay pass so no dialog
        // flashes.)
        if (input.drop_confirm_pending && current_project_path.empty() &&
            scene.alive_count() == 1 &&
            !scene.active_entity().undo.can_undo() &&
            !scene.active_entity().undo.can_redo()) {
            input.drop_confirm_pending = false;
            input.drop_open_requested = true;
        }

        bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
        bool dialog_open = input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.voxel_merge_confirm_pending || input.drop_confirm_pending || input.help_popup_open || input.keybind_menu_open || input.brush_props_open;
        if ((imgui_wants_mouse || dialog_open) && app_state != AppState::SCULPTING) {
            // A press on a panel over the model is the panel's, not a pick or a fill
            // (the bucket's Scale slider sits right over it). Only this frame's press:
            // a gradient dragged from the model may still be released over a panel.
            if (input.mouse1_just_pressed) {
                input.color_pick_click = false;
                input.bucket_click = false;
                input.bucket_dragging = false;
            }
            input.drag_mode = InputState::DragMode::NONE;
            input.mouse1_just_pressed = false;
            input_consume_scroll();
        }

        // Age the rescue snapshot. Purely a message — the countdown changes nothing
        // but when the copy is freed. Messages wait for a clear notification line
        // so they never stomp the remesh's own result line, which carries the
        // vertex/triangle counts the user is judging by.
        if (rescue.valid()) {
            int left = rescue.edits_left();
            if (left <= 0) {
                std::printf("[snapshot] %s rescue expired after %d edits\n",
                            rescue.op, SceneSnapshot::GRACE_EDITS);
                if (input.notification_timer <= 0.0f) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Undo can no longer reach the %s", rescue.op);
                    input.notification_timer = 2.0f;
                }
                rescue.clear();
                rescue_shown = -1;
            } else if (left != rescue_shown && input.notification_timer <= 0.0f) {
                rescue_shown = left;
                std::snprintf(input.notification, sizeof(input.notification),
                              "%s reverts the %s - %d stroke%s left",
                              input.key_label(KeyAction::UNDO),
                              rescue.op, left, left == 1 ? "" : "s");
                input.notification_timer = 2.5f;
            }
        }

        // FPS counter
        fps_frame_count++;
        double fps_now = glfwGetTime();
        if (fps_now - fps_last_time >= 1.0) {
            fps_display = (float)fps_frame_count / (float)(fps_now - fps_last_time);
            fps_frame_count = 0;
            fps_last_time = fps_now;
        }

        // Persisted settings: notice edits, write them out once they settle. Self-gated
        // to skip strokes entirely, so this never lands in the hot path.
        {
            double now = glfwGetTime();
            settings_tick(input, (float)(now - settings_last_time));
            settings_last_time = now;
        }

        // Fullscreen toggle (Space)
        if (input.fullscreen_toggle_requested) {
            input.fullscreen_toggle_requested = false;
#ifndef __EMSCRIPTEN__
            // Real fullscreen on the monitor, over the taskbar too. The old toggle only
            // dropped the titlebar (borderless), and the desktop's panel stayed on top.
            // Wayland can't report a window position (GLFW errors if asked), so only the
            // size is remembered there; the compositor places the restored window.
            bool wayland = false;
#if GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4
            wayland = glfwGetPlatform() == GLFW_PLATFORM_WAYLAND;
#endif
            if (!input.is_fullscreen) {
                if (!wayland) glfwGetWindowPos(window, &windowed_x, &windowed_y);
                glfwGetWindowSize(window, &windowed_w, &windowed_h);
                GLFWmonitor* mon = glfwGetPrimaryMonitor();
                const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
                if (vm) {
                    glfwSetWindowMonitor(window, mon, 0, 0, vm->width, vm->height,
                                         vm->refreshRate);
                    input.is_fullscreen = true;
                }
            } else {
                glfwSetWindowMonitor(window, nullptr, windowed_x, windowed_y,
                                     windowed_w, windowed_h, GLFW_DONT_CARE);
                input.is_fullscreen = false;
            }
#else
            // Browser fullscreen of the whole page (the canvas is position:fixed at
            // 100vw/100vh, so it follows; win_w/h re-read the size next frame).
            // navigationUI 'hide' also drops Android's nav bar. Must run with the
            // tap's user activation still live — it is: Chrome keeps it ~5 s, and
            // this runs on the frame after the click or keypress.
            EM_ASM({
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                } else if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen({ navigationUI: 'hide' })
                        .catch(function(e) { console.warn('[fullscreen] ' + e.message); });
                }
            });
#endif
        }
#ifdef __EMSCRIPTEN__
        // The browser owns this state: Esc, Android's back gesture or a refused
        // request all change it without us. Read it back every frame for the icon.
        input.is_fullscreen = EM_ASM_INT({ return document.fullscreenElement ? 1 : 0; }) != 0;
#endif

        // (win_w / win_h resolved and the surface reconfigured at the top of the frame,
        //  before glfwPollEvents — see the note there.)

        // Pump async GPU→CPU readbacks once per frame: deliver map callbacks (no-op
        // on GL) and land any finished screen-buffer plane reads. Everything that
        // samples the screen buffers inside the frame goes through these caches.
        gpu::process_events(renderer.gpu_dev);
        renderer.poll_plane_reads();

        // Update cursor normal: average surface normal under the brush circle.
        // Sampling contributes for any brush pixel that hits geometry — so as the
        // cursor slides past the silhouette, the remaining on-model pixels bias
        // the averaged normal and the cursor "slides along" the form. When the
        // full circle is off the mesh we target face-camera. A per-frame LERP
        // smooths the transition for a magnetic feel.
        // Touch press: correct the SCULPT/ORBIT latch from what is actually under the
        // finger (see InputState::touch_relatch). Runs before camera_moving and the
        // IDLE→SCULPTING transition, so the wrong latch never moves anything. Waits
        // while the emulated press hasn't arrived yet; a gesture or lift clears it.
        if (input.touch_relatch && (input.mouse1_down || input.touch_gesture)) {
            input.touch_relatch = false;
            int px = (int)input.mouse_x;
            int py = (int)input.mouse_y;
            bool om;
            if (!input.touch_gesture && !input.alt_held
                && input.interaction_mode == InputState::InteractionMode::EDIT
                && (input.drag_mode == InputState::DragMode::SCULPT
                    || input.drag_mode == InputState::DragMode::ORBIT)
                && px >= 0 && px < win_w && py >= 0 && py < win_h
                && sample_on_model(renderer, px, py, win_h, &om)) {
                input.on_model = om;
                input.drag_mode = om ? InputState::DragMode::SCULPT
                                     : InputState::DragMode::ORBIT;
            }
        }

        bool camera_moving = input.drag_mode == InputState::DragMode::ORBIT
                          || input.drag_mode == InputState::DragMode::PAN
                          || input.drag_mode == InputState::DragMode::ZOOM
                          || input.mouse2_down || input.mouse3_down
                          // Pinch/pan moves the camera with drag_mode NONE; without
                          // this, every gesture frame paid a full extra MRT pass.
                          || input.touch_gesture;

        // Refresh screen buffers whenever idle and dirty. The menu counts as not-idle:
        // the FOV slider dirties these every frame it moves, and unlike an orbit drag
        // (which camera_moving already defers) nothing else would hold the refresh off,
        // so a heavy mesh would eat a full extra MRT pass per slider frame. Nothing can
        // sculpt while the menu is up, so deferring to the frame after it closes is free.
        if (!camera_moving && !brush_stroke.is_active() && !input.settings_menu_open) {
            if (screen_buffers_dirty) {
                renderer.render_screen_buffers(camera, win_w, win_h);
                screen_buffers_dirty = false;
            }
        }

        // Compute target normal: sample 1 pixel from the screen FBO normal attachment.
        float target_nx, target_ny, target_nz;
        bool have_sample = false;

        if (!camera_moving && !screen_buffers_dirty) {
            int cx = (int)input.mouse_x;
            int cy = (int)input.mouse_y;
            if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
                // On-model latch (SCULPT vs ORBIT on left-press) reads the same fresh
                // screen buffer as the cursor normal. Gated with it so it only fires
                // when stationary over fresh buffers, which is exactly when the press
                // latch fires. Orbiting/mid-stroke — or while the webgpu plane cache
                // hasn't landed yet — on_model keeps its last value.
                bool om;
                if (sample_on_model(renderer, cx, cy, win_h, &om))
                    input.on_model = om;

                float norm_pixel[3];
                if (renderer.sample_normal(cx, cy, norm_pixel)) {
                    float len = std::sqrt(norm_pixel[0]*norm_pixel[0] + norm_pixel[1]*norm_pixel[1] + norm_pixel[2]*norm_pixel[2]);
                    if (len > 1e-6f) {
                        target_nx = norm_pixel[0] / len;
                        target_ny = norm_pixel[1] / len;
                        target_nz = norm_pixel[2] / len;
                        have_sample = true;
                    }
                }
            }
        }

        if (!have_sample) {
            Vec3 vd = camera.get_view_direction();
            target_nx = -vd.x; target_ny = -vd.y; target_nz = -vd.z;
        }

        input.cursor_nx = target_nx;
        input.cursor_ny = target_ny;
        input.cursor_nz = target_nz;

        // Consume scroll for camera zoom
        {
            float scroll = input_consume_scroll();
            // Over a pose sphere the wheel sizes the sphere instead of zooming.
            PoseRig& rig = scene.active_entity().pose;
            const int ph = input.pose_hover_joint;
            if (std::fabs(scroll) > 0.01f && input.pose_brush_on()
                && ph >= 0 && ph < (int)rig.joints.size()) {
                rig.joints[ph].radius = std::max(1e-4f,
                    rig.joints[ph].radius * std::pow(1.1f, scroll));
            } else if (std::fabs(scroll) > 0.01f) {
                camera.zoom(scroll);
                screen_buffers_dirty = true;
            }
        }

        // Pinch + two-finger pan. Distance divides by the finger ratio directly (not
        // through zoom()'s 10%-per-notch step) so the model stays stuck to the fingers.
        // Same clamp as zoom(). Pan is 1:1 too: the view spans 2*half_height() world
        // units over win_h pixels, so that is one pixel of finger travel. Camera::pan
        // runs at a fixed mouse rate of half_height()*0.0048284 per pixel instead —
        // fine for a middle-drag, but about twice the finger on a phone-sized view,
        // so the model slid away from under the fingers. Rescale into its units.
        if (input.touch_zoom != 1.0f || input.touch_pan_x != 0.0f || input.touch_pan_y != 0.0f) {
            if (input.touch_zoom > 0.0f)
                camera.distance = std::max(0.05f, std::min(200.0f, camera.distance / input.touch_zoom));
            float to_pan_units = (2.0f / (float)std::max(1, win_h)) / 0.0048284f;
            camera.pan(input.touch_pan_x * to_pan_units, input.touch_pan_y * to_pan_units,
                       win_w, win_h);
            input.touch_zoom = 1.0f;
            input.touch_pan_x = input.touch_pan_y = 0.0f;
            screen_buffers_dirty = true;
        }

        // Debug: print multires stack state (F12)
        if (input.debug_multires_requested) {
            input.debug_multires_requested = false;
            multires_stack_debug_print(*multires);
        }

        // Debug: F9 cycles stride override 0(adaptive)->1->2->3->0
        if (input.debug_stride_cycle_requested) {
            input.debug_stride_cycle_requested = false;
            BrushStroke::debug_stride_override = (BrushStroke::debug_stride_override + 1) % 4;
            printf("[debug] stride_override = %d (%s)\n",
                   BrushStroke::debug_stride_override,
                   BrushStroke::debug_stride_override == 0 ? "adaptive" : "forced");
        }

        // Debug: F10 picks the test vertex under the cursor (highest bary weight in cursor triangle)
        if (input.debug_pick_vertex_requested) {
            input.debug_pick_vertex_requested = false;
            int cx = (int)input.mouse_x;
            int cy = (int)input.mouse_y;
            if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
                uint32_t tid;
                renderer.read_triid_region(cx, cy, 1, 1, &tid);
                const Mesh& rm_pick = scene.active_mesh();  // screen FBO holds the active entity
                if (tid != 0xFFFFFFFF && tid < rm_pick.tri_count()) {
                    float bary[2];
                    renderer.read_bary_region(cx, cy, 1, 1, bary);
                    float bu = bary[0], bv = bary[1], bw = 1.0f - bu - bv;
                    uint32_t i0 = rm_pick.indices[tid*3+0];
                    uint32_t i1 = rm_pick.indices[tid*3+1];
                    uint32_t i2 = rm_pick.indices[tid*3+2];
                    uint32_t best = (bu >= bv && bu >= bw) ? i0 : (bv >= bw ? i1 : i2);
                    BrushStroke::debug_test_vertex = (int)best;
                    printf("[debug] test vertex = %u (tri %u, bary %.2f/%.2f/%.2f)\n",
                           best, tid, bu, bv, bw);
                } else {
                    printf("[debug] F10: no mesh under cursor\n");
                }
            }
        }

        app_ops_paint_clicks(app, win_w, win_h);   // app_ops.cpp: C colour picker, fill bucket

        auto print_undo_top = [&](const char* tag) { app_print_undo_top(app, tag); };

        // Phase 1 GPU residency: full re-upload of the active entity's mirrored
        // level after any wholesale CPU mutation (lock, level switch, projection,
        // cascade). Cheap no-op until the stack is locked / compute supported.
        // GEOMETRIC symmetry for a dab: the reflected-anchor second lobe every brush
        // kernel already carries. It consults no topology, so there is nothing to
        // check and nothing that can tear — the toggle is the whole answer.
        auto mirror_effective = [&]() -> bool {
            return input.mirror_x;
        };

        // TOPOLOGICAL symmetry for a dab: the pair-map passes layered on top
        // (draw_accum_symmetrize, mirror_project, smooth_mirror_apply, the undo-snap
        // twin push). Opt-in, and additionally gated on the mesh still being world-X
        // symmetric: two of those passes write absolute positions derived from the
        // x=0 plane, so on a mesh a Select-mode spin — or an asymmetric SDF merge —
        // has turned off that plane they do not mirror wrong, they TEAR.
        //
        // Failing the guard now costs only exactness: mirror_effective() above keeps
        // mirroring the stroke geometrically, so the message says what was lost, not
        // that symmetry stopped. Mesh::mirror_world_symmetric() caches against
        // topo_version, so this is a stamp compare on all but the first call after a
        // change.
        // ONE check, and it ENFORCES rather than reports: a mesh that cannot supply
        // pairs switches the setting back to World Space and flashes the notice, so the
        // button, the HUD and the dab loop can never disagree about what is happening.
        // Reverting the setting is also what makes this self-limiting — the early-out on
        // the next frame sees mirror_topological already false, so it fires exactly once
        // per transition into an unusable mesh rather than every frame.
        //
        // Run from the top of the frame rather than off the button, so it also catches
        // the ways a mesh becomes unpairable without anyone touching the control:
        // loading a project with the preference persisted on, importing, or spinning a
        // piece off the plane in Select mode.
        auto enforce_mirror_mode = [&]() {
            // Gated on mirror_x as well: with symmetry off there is nothing to override
            // and nothing to warn about, and the message would arrive detached from any
            // action. Held back this way it fires on the X keypress instead, which is
            // the moment it means something.
            if (!input.mirror_x || !input.mirror_topological) return;
            if (mesh->mirror_world_symmetric()) return;
            input.mirror_topological = false;
            input.mirror_unavailable_timer = 2.5f;
        };

        auto mirror_pairs_effective = [&]() -> bool {
            return input.mirror_x && input.mirror_topological;
        };
        enforce_mirror_mode();

        auto refresh_active_gpu_residency = [&]() { app_refresh_active_gpu_residency(app); };

        app_ops_levels_and_views(app);   // app_ops.cpp: D / Shift-D, delete top level, F, F1-F3

        // ---- Density-view colour swap ----
        // While the paint brush targets the density field, the colour VBO shows
        // colormap(density). Re-dispatched every frame while active: one fixed
        // micro-dispatch (no readback, no alloc) that self-heals every path that
        // rewrites the colour VBO under us — level switches, undo, entity rebinds.
        // On exit, albedo restores from mesh.color (authoritative outside strokes).
        {
            const bool want = input.current_brush == BrushType::PAINT
                           && input.paint_target_density
                           && compute.supported && compute.has_density_kernels();
            if (want) {
                compute.dispatch_density_colormap(mesh->vertex_count());
            } else if (density_view_active) {
                renderer.update_color(*mesh);
            }
            density_view_active = want;
        }

        // ---- Brush-alpha upload-on-change ----
        // Push the selected stamp to the compute state when it changes (cheap: only
        // on selection, not per frame). Index 0 (Round) uploads a null → alpha off.
        // Clay overrides the selection with the Square builtin (its stamp is fixed),
        // so switching Clay ↔ other brushes swaps the upload back and forth.
        if (input.active_alpha < 0 || input.active_alpha >= alpha_lib.count())
            input.active_alpha = 0;
        if (input.alpha_step && alpha_lib.count() > 0) {
            int n = alpha_lib.count();
            input.active_alpha = ((input.active_alpha + input.alpha_step) % n + n) % n;
            input.alpha_step = 0;
            std::snprintf(input.notification, sizeof(input.notification), "Alpha: %s",
                          alpha_lib.get(input.active_alpha).name.c_str());
            input.notification_timer = 1.0f;
        }
        int want_alpha = (input.current_brush == BrushType::CLAY)
                             ? AlphaLibrary::kSquareIndex : input.active_alpha;
        if (want_alpha != last_uploaded_alpha) {
            const AlphaEntry& ae = alpha_lib.get(want_alpha);
            if (ae.is_round || ae.data.empty())
                compute.upload_alpha(nullptr, 0, 0);
            else
                compute.upload_alpha(ae.data.data(), ae.w, ae.h);
            last_uploaded_alpha = want_alpha;
        }

        // ---- File dialogs (ImGuiFileDialog) ----
        // Handled in the ImGui widget section below (after rendering).

        app_ops_remesh_and_merge(app);   // app_ops.cpp: remesh, SDF merge start + step

        // Mouse delta (shared by camera and sculpt)
        float dx = (float)(input.mouse_x - input.prev_mouse_x);
        float dy = (float)(input.mouse_y - input.prev_mouse_y);

        // Two-finger touch landed right after the first finger: what that finger
        // started was a stub of the pinch, not a stroke. An orbit is put back here; a
        // stroke still has to finish committing (drag_mode is already NONE, so the
        // pen-up path below runs), and is undone from the commit.
        if (input.mouse1_just_pressed) camera_at_press = camera;
        if (input.touch_revert_requested) {
            input.touch_revert_requested = false;
            auto was = (InputState::DragMode)input.touch_cancelled_drag;
            if (was == InputState::DragMode::ORBIT) {
                camera.yaw   = camera_at_press.yaw;
                camera.pitch = camera_at_press.pitch;
                screen_buffers_dirty = true;
            } else if (was == InputState::DragMode::SCULPT && app_state == AppState::SCULPTING) {
                touch_undo_on_commit = true;
            }
        }

        // ---- State transitions ----

        // SELECT mode: intercept sculpt drag for mesh picking or object move
        if (app_state == AppState::IDLE
            && input.drag_mode == InputState::DragMode::SCULPT
            && input.interaction_mode == InputState::InteractionMode::SELECT) {

            // Entity-id pick pass: draw all entities into the FBO id buffer, then
            // read the id directly under the cursor (nearest non-zero in a small
            // window for robustness). No tri-walk, no vertex→owner lookup.
            scene.render_pick(camera, win_w, win_h);
            screen_buffers_dirty = true;  // pick overwrote the shared screen FBO

            uint32_t clicked_mesh = 0;
            int cx = (int)input.mouse_x;
            int cy = (int)input.mouse_y;
            if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
                constexpr int PICK_R = 5;
                constexpr int PICK_D = PICK_R * 2 + 1;
                int rx = std::max(cx - PICK_R, 0);
                int ry = std::max(cy - PICK_R, 0);
                int rw = std::min(PICK_D, win_w  - rx);
                int rh = std::min(PICK_D, win_h - ry);
                uint32_t pick_buf[PICK_D * PICK_D];
                renderer.read_id_region(rx, ry, rw, rh, pick_buf);
                int best_dist2 = INT_MAX;
                int ocx = cx - rx, ocy = cy - ry;
                for (int py = 0; py < rh; py++)
                    for (int px = 0; px < rw; px++) {
                        uint32_t id = pick_buf[py * rw + px];
                        if (id != 0) {
                            int ddx = px - ocx, ddy = py - ocy;
                            int d2 = ddx*ddx + ddy*ddy;
                            if (d2 < best_dist2) { best_dist2 = d2; clicked_mesh = id; }
                        }
                    }
            }

            bool already_selected = false;
            if (clicked_mesh != 0) {
                for (uint32_t sel_id : scene.selected_ids())
                    if (sel_id == clicked_mesh) { already_selected = true; break; }
            }

            if (already_selected && input.ctrl_held) {
                // Ctrl+click an already-selected mesh: remove it from the selection
                // instead of starting a move-drag. This works for the active entity
                // too now — it just drops out of the visible selection (active stays
                // bound), so peeling out the last one deselects the whole scene.
                input.drag_mode = InputState::DragMode::NONE;
                if (scene.toggle_selected(clicked_mesh)) {
                    screen_buffers_dirty = true;
                    uint32_t nsel = (uint32_t)scene.selected_ids().size();
                    if (nsel == 0)
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Deselected all");
                    else
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Selection: %u meshes", nsel);
                    input.notification_timer = 1.5f;
                }
            } else if (already_selected) {
                input.drag_mode = InputState::DragMode::MOVE_OBJECT;
            } else if (clicked_mesh == 0) {
                // Missed every entity — same as a normal empty-space click: orbit.
                // (The callback latches SCULPT unconditionally in SELECT mode so this
                // pick pass always gets to run; a miss has to fall back to ORBIT here
                // instead of going dead, or plain-drag orbiting breaks in SELECT mode.)
                input.drag_mode = InputState::DragMode::ORBIT;
            } else {
                input.drag_mode = InputState::DragMode::NONE;
                if (input.ctrl_held) {
                    if (scene.toggle_selected(clicked_mesh)) {
                        screen_buffers_dirty = true;
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Selection: %u meshes",
                                      (uint32_t)scene.selected_ids().size());
                        input.notification_timer = 1.5f;
                    }
                } else {
                    if (scene.select(clicked_mesh)) {
                        mesh = &scene.active_mesh();
                        multires = &scene.active_multires();
                        scene.refresh_mirror_map();
                        // Undo entries are per-entity (each carries entity_id and
                        // local indices), so history survives selection switches.
                        screen_buffers_dirty = true;
                        uint32_t mid = scene.active_mesh_id();
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Selected mesh %u", mid);
                        input.notification_timer = 1.5f;
                    }
                }
            }
        }

        // Set up object mask when entering EDIT mode: keep selected mesh
        // unmasked so only it is visually active and editable.
        static auto prev_interaction_mode = InputState::InteractionMode::EDIT;
        if (input.interaction_mode == InputState::InteractionMode::EDIT
            && prev_interaction_mode != InputState::InteractionMode::EDIT) {
            scene.refresh_for_edit_mode();
        }
        prev_interaction_mode = input.interaction_mode;

        // ---- INSERT mode state machine ----
        insert_ctrl.tick(input, camera, *multires, brush_stroke,
                         mesh_center, mesh_radius, dx,
                         app_state == AppState::IDLE,
                         win_w, win_h, window, screen_buffers_dirty);

        // Block sculpt drag in INSERT/SELECT modes (prevent falling through to SCULPTING)
        if (app_state == AppState::IDLE
            && input.drag_mode == InputState::DragMode::SCULPT
            && input.interaction_mode != InputState::InteractionMode::EDIT) {
            input.drag_mode = InputState::DragMode::NONE;
        }

        // ---- Retopo mode (retopo_input.cpp): pen, clicks, its own undo, Build ----
        retopo_frame(retopo, input, scene, *mesh, renderer, camera, tablet, win_w, win_h,
                     screen_buffers_dirty, imgui_wants_mouse);
        if (input.uv_bake_save_requested) {
            input.uv_bake_save_requested = false;
            retopo_bake_save(retopo, input, current_project_path);
        }
        if (input.retopo_export_requested) {
            input.retopo_export_requested = false;
            const std::string path = retopo_export(retopo, input, current_project_path);
#if defined(__EMSCRIPTEN__)
            if (!path.empty()) web_download_file(path.c_str(), "retopo.obj");
#endif
        }

        // Hover speed, for the stroke's running start (g_hover_speed). Orbit, pan and
        // slider drags aren't the hand heading into a stroke: re-anchor at zero there.
        if (app_state == AppState::IDLE) {
            const double now = glfwGetTime();
            const float hx = (float)input.mouse_x, hy = (float)input.mouse_y;
            const float dt = (float)(now - g_hover_speed.t);
            const bool hovering = input.drag_mode == InputState::DragMode::NONE
                               || input.drag_mode == InputState::DragMode::SCULPT;
            if (!hovering) {
                g_hover_speed = StrokeSpeed{0.0f, hx, hy, now};
            } else if (dt > 0.0f) {
                const float v = std::hypot(hx - g_hover_speed.x, hy - g_hover_speed.y) / dt;
                if (v < SPEED_TELEPORT)
                    g_hover_speed.px_s += (v - g_hover_speed.px_s) * (1.0f - std::exp(-dt / SPEED_TAU));
                g_hover_speed.x = hx;
                g_hover_speed.y = hy;
                g_hover_speed.t = now;
            }
        }

        // ---- Pose brush: hover, laying spheres, sizing them, clearing the rig ----
        // A sculpt stroke only starts when the press grabbed a sphere (input.cpp); this
        // block owns everything else the brush does. All of it is user-paced, so the
        // CPU ray casts here are fine (they never run inside a stroke).
        {
            PoseRig& rig = scene.active_entity().pose;
            struct PoseBuild {
                bool on = false, new_a = false;
                int a = -1, b = -1;
                float px = 0, py = 0, lx = -1, ly = -1;
            };
            static PoseBuild build;
            // Shift+drag relocation: which sphere, and where the cursor sat on it, so
            // the sphere keeps its offset under the hand instead of jumping to it.
            struct PoseRelocate { bool on = false; int j = -1; float ox = 0, oy = 0, lx = 0, ly = 0; };
            static PoseRelocate reloc;
            const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
            const bool pose_on = input.pose_brush_on();
            const bool laying = input.drag_mode == InputState::DragMode::POSE_EDIT
                             || input.drag_mode == InputState::DragMode::POSE_MOVE;

            if (!pose_on) {
                input.pose_hover_joint = -1;
                input.pose_remove_joint = -1;
            } else if (app_state == AppState::IDLE && !build.on && !reloc.on && !laying
                       && input.drag_mode != InputState::DragMode::SCULPT) {
                // Held through a grab: the stroke drives the sphere the press landed on.
                input.pose_hover_joint = pose_pick(rig, camera, mx, my, win_w, win_h);
            }

            // Press: a new pair (A at the press, B dragged out), or with Ctrl on a
            // sphere, a new sphere B grown off it.
            if (pose_on && app_state == AppState::IDLE && !build.on
                && input.drag_mode == InputState::DragMode::POSE_EDIT) {
                scene.materialize_active_cpu();   // the ray cast reads mesh.pos
                const int h = input.pose_hover_joint;
                build.new_a = h < 0;
                int a = h;
                if (a < 0) {
                    Vec3 c;
                    float r;
                    if (pose_volume_center(*mesh, camera, mx, my, win_w, win_h, c, r)) {
                        // An open mesh has no far wall to measure: fall back to the ring.
                        if (r <= 0.0f) r = input.brush_size / pose_px_per_world(camera, c, win_h);
                        a = rig.add_joint(c, r);
                    }
                }
                if (a >= 0) {
                    const PoseJoint ja = rig.joints[a];   // a copy: add_joint may reallocate
                    const int b = rig.add_joint(ja.pos, ja.radius);
                    if (rig.add_bone(a, b)) {
                        build.on = true;
                        build.a = a;
                        build.b = b;
                        build.px = mx; build.py = my;
                        build.lx = build.ly = -1.0f;
                    } else {
                        rig.pop_joint();
                        if (build.new_a) rig.pop_joint();
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "A pose set holds at most %d bones", kPoseMaxBones);
                        input.notification_timer = 2.0f;
                    }
                }
                if (!build.on) input.drag_mode = InputState::DragMode::NONE;
            }

            // Dragging B out: it sits in the middle of whatever volume is under the
            // cursor, sized to it; off the model it rides the plane through A.
            if (build.on && input.drag_mode == InputState::DragMode::POSE_EDIT
                && (mx != build.lx || my != build.ly)) {
                build.lx = mx; build.ly = my;
                const PoseJoint ja = rig.joints[build.a];
                PoseJoint& jb = rig.joints[build.b];
                Vec3 c;
                float r;
                if (pose_volume_center(*mesh, camera, mx, my, win_w, win_h, c, r)) {
                    jb.pos = c;
                    jb.radius = r > 0.0f ? r : ja.radius;
                } else {
                    Vec3 o, d;
                    pose_screen_ray(camera, mx, my, win_w, win_h, o, d);
                    jb.pos = o + d * camera.view_depth(ja.pos);
                    jb.radius = ja.radius;
                }
            }

            // Release: a click that never dragged makes nothing.
            if (build.on && input.drag_mode != InputState::DragMode::POSE_EDIT) {
                if (std::hypot(mx - build.px, my - build.py) < 4.0f) {
                    rig.pop_joint();
                    if (build.new_a) rig.pop_joint();
                }
                build.on = false;
                input.pose_hover_joint = -1;   // indices may have shifted; re-picked next frame
            }

            // Shift+drag: the sphere alone, re-centred in the volume under the cursor (its
            // size kept, it may have been tuned); off the model it rides its own depth.
            if (pose_on && app_state == AppState::IDLE && !reloc.on
                && input.drag_mode == InputState::DragMode::POSE_MOVE) {
                const int h = input.pose_hover_joint;
                if (h >= 0 && h < (int)rig.joints.size()) {
                    scene.materialize_active_cpu();   // the ray cast reads mesh.pos
                    float sx = mx, sy = my;
                    camera.world_to_screen(rig.joints[h].pos, win_w, win_h, sx, sy);
                    reloc = PoseRelocate{true, h, mx - sx, my - sy, mx, my};
                    rig.forget_moves(h);
                } else {
                    input.drag_mode = InputState::DragMode::NONE;
                }
            }
            if (reloc.on && input.drag_mode == InputState::DragMode::POSE_MOVE
                && (mx != reloc.lx || my != reloc.ly)) {
                reloc.lx = mx; reloc.ly = my;
                PoseJoint& jt = rig.joints[reloc.j];
                const float tx = mx - reloc.ox, ty = my - reloc.oy;
                Vec3 c;
                float r;
                if (pose_volume_center(*mesh, camera, tx, ty, win_w, win_h, c, r)) {
                    jt.pos = c;
                } else {
                    Vec3 o, d;
                    pose_screen_ray(camera, tx, ty, win_w, win_h, o, d);
                    jt.pos = o + d * camera.view_depth(jt.pos);
                }
            }
            if (reloc.on && input.drag_mode != InputState::DragMode::POSE_MOVE)
                reloc.on = false;

            // Right-click on a sphere.
            if (input.pose_remove_joint >= 0 && app_state == AppState::IDLE
                && !build.on && !reloc.on) {
                rig.remove_joint(input.pose_remove_joint);
                input.pose_remove_joint = -1;
                input.pose_hover_joint = -1;   // indices shifted; re-picked next frame
            }

            // [ and ] over a sphere (input.cpp routes them here instead of brush size).
            if (input.pose_resize_steps != 0) {
                const int h = input.pose_hover_joint;
                if (pose_on && h >= 0 && h < (int)rig.joints.size())
                    rig.joints[h].radius = std::max(1e-4f,
                        rig.joints[h].radius * std::pow(1.1f, (float)input.pose_resize_steps));
                input.pose_resize_steps = 0;
            }

            if (app_state == AppState::IDLE && !build.on && input.pose_delete_key) {
                input.pose_delete_key = false;
                if (rig.empty()) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "No pose spheres to delete");
                    input.notification_timer = 1.5f;
                } else {
                    input.pose_clear_confirm_pending = true;
                }
            }
            if (app_state == AppState::IDLE && input.pose_clear_requested) {
                input.pose_clear_requested = false;
                rig.clear();
                input.pose_hover_joint = -1;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Pose spheres deleted");
                input.notification_timer = 1.5f;
            }
        }

        // IDLE → SCULPTING
        if (app_state == AppState::IDLE && input.drag_mode == InputState::DragMode::SCULPT) {
            input.sculpting = true;
            // A pose grab binds on the CPU at its first dab (pose_bind reads mesh.pos).
            if (input.current_brush == BrushType::POSE && !input.is_smooth_active())
                scene.materialize_active_cpu();

            // Active entity lives in the working buffers at offset 0; the brush
            // dispatches over its full [0, vertex_count) range.
            brush_stroke.begin(renderer, camera,
                               (float)input.mouse_x, (float)input.mouse_y,
                               input.brush_size,
                               win_w, win_h, mesh->vertex_count(), *multires,
                               input.current_brush, scene.active_mesh_id(),
                               scene.active_entity().multires_gpu,
                               !screen_buffers_dirty);
            brush_stroke.walk_x = (float)input.mouse_x;
            brush_stroke.walk_y = (float)input.mouse_y;
            brush_stroke.walk_carry = 0.0f;
            g_stroke_speed = g_hover_speed;   // updated just above, this frame: a running start
            pushes_at_stroke_start = UndoStack::global_pushes;
            app_state = AppState::SCULPTING;
        }

        // SCULPTING → IDLE (pen-up). finalize is a small per-frame state machine:
        // it drains in-flight dab readbacks, kicks the pen-up buffer reads, and
        // commits when they land — one call on GL, a few frames on webgpu/web.
        // app_state stays SCULPTING while it runs, which blocks new strokes and the
        // IDLE-only ops (undo, delete, merge, …) exactly like an active stroke; a
        // re-press during the reconcile starts its stroke from the IDLE transition
        // a frame or two later.
        // The release frame first lays the path that arrived before the button came up
        // (the dab loop below), and pen-up runs the frame after. Finalizing straight
        // away ended the stroke wherever the PREVIOUS frame had left it, dropping the
        // last frame of motion: ~15 px a release, up to 80 on a quick flick, so short
        // strokes stopped short of where the button came up (2026-10-20).
        bool pen_up = brush_stroke.reconciling() || pen_up_pending
                   || input.drag_mode != InputState::DragMode::SCULPT;
        if (app_state == AppState::SCULPTING && pen_up && !pen_up_pending
            && !brush_stroke.reconciling()
            && input.release_tail && brush_stroke.is_active()) {
            pen_up_pending = true;
        } else if (app_state == AppState::SCULPTING && pen_up) {
            pen_up_pending = false;
            input.sculpting = false;
            DabContext fctx { renderer, camera, compute, *mesh, *multires, input,
                              win_w, win_h, brush_stroke.vertex_count, input.brush_size,
                              mirror_effective(), mirror_pairs_effective() };
            bool had_update = false;
            if (brush_stroke.finalize(fctx, *mesh, scene.active_undo(), *multires,
                                      scene.active_entity().multires_gpu,
                                      renderer, input.current_brush,
                                      input.autosmooth, had_update)) {
                print_undo_top("stroke-commit");
                g_path_stats.report();
                // Only if this stroke pushed an entry: a stub that never laid a dab
                // commits nothing, and a blind undo would eat the stroke BEFORE it.
                // A pose stroke ties its sphere's move to the entry it just pushed, so
                // Ctrl+Z takes the sphere back with the mesh (PoseRig::on_undo).
                if (brush_stroke.pose.joint >= 0) {
                    PoseRig& rig = scene.active_entity().pose;
                    const int j = brush_stroke.pose.joint;
                    brush_stroke.pose.joint = -1;
                    const UndoEntry* top = scene.active_undo().peek_undo();
                    if (top && j < (int)rig.joints.size()
                        && UndoStack::global_pushes > pushes_at_stroke_start)
                        rig.moves.push_back({top->seq, (uint16_t)j,
                                             brush_stroke.pose.start, rig.joints[j].pos});
                }
                if (touch_undo_on_commit) {
                    touch_undo_on_commit = false;
                    if (UndoStack::global_pushes > pushes_at_stroke_start)
                        input.undo_requested = true;
                }
                if (had_update)
                    renderer.update_screen_positions(*mesh);
                screen_buffers_dirty = true;
                app_state = AppState::IDLE;
            }
        }

        // ---- State-specific update ----

        if (app_state == AppState::IDLE) {
            app_idle_update(app, win_w, win_h, dx, dy);   // app_idle.cpp: camera, object move/scale/spin, undo/redo, one-shots
        }

        if (app_state == AppState::SCULPTING) {
            bool wrapped = wrap_cursor(window, input, win_w, win_h);
            // A wrap teleports the cursor to the far edge. Restart the path there, or
            // the next frame would lay a line of dabs straight across the screen.
            if (wrapped) {
                brush_stroke.walk_x = (float)input.mouse_x;
                brush_stroke.walk_y = (float)input.mouse_y;
                g_stroke_speed.x = (float)input.mouse_x;   // the jump isn't hand speed
                g_stroke_speed.y = (float)input.mouse_y;
                brush_stroke.walk_carry = 0.0f;
                input.path_n = 0;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
                g_mouse_history.forget();
#endif
            }

            if (brush_stroke.is_active() && !wrapped) {
                bool is_smooth = input.is_smooth_active() ||
                                 input.current_brush == BrushType::SMOOTH;
                bool is_move = input.current_brush == BrushType::MOVE;
                bool is_limb = input.current_brush == BrushType::LIMB;
                bool is_pose = input.current_brush == BrushType::POSE && !is_smooth;
                // The grab brushes capture once and drive per-frame: one dab at the
                // live cursor, no spline interpolation, no dab-spacing advance.
                bool is_grab = is_move || is_limb || is_pose;

                // live_brush_slot() rather than a second copy of the same rule: it is
                // now defined as exactly `is_smooth ? SMOOTH : current_brush`, and the
                // HUD and the sliders address that same slot. Two hand-kept copies of
                // this rule is what let the slider edit one brush while the dab loop
                // read another (CHANGES 2026-08-23).
                const BrushSettings& eff = input.per_brush[(int)input.live_brush_slot()];
                float eff_strength = eff.strength;
                float eff_hardness = eff.hardness;

                // Pen pressure → strength & size (independent floors). Synthetic 1.0
                // unless the PEN is the device currently driving input.
                //
                // tablet.available() alone is the wrong question, and asking it was a
                // real regression: it only says a tablet is plugged in. Tablet::pressure()
                // is sticky — it holds the last stylus sample — so with the pen resting on
                // the desk it reads ~0, and every mouse stroke came out at the ramp floor
                // (0.05x strength, all brushes). The K toggle used to be the escape hatch
                // for exactly that; removing K on 2026-08-07 exposed it.
                //
                // active_profile is the signal that was already correct: the arbiter
                // switches to TABLET on stylus samples and back to MOUSE on cursor motion
                // after a 250 ms quiet window, and it is frozen mid-stroke, so this cannot
                // flip under a stroke in progress.
                bool pressure_is_synthetic = !(tablet.available() &&
                                               input.active_profile == InputProfile::TABLET);
                float pressure = pressure_is_synthetic ? 1.0f : tablet.pressure();
                float p_shaped = std::pow(pressure, PRESSURE_GAMMA);

                // Max brush effect (per-profile slider) is the CEILING of this ramp, not
                // a scale applied after it: at full input the dab lands on exactly
                // max_effect × nominal strength, and the floor stays where it is so a
                // feather-light pen touch still fades to near-nothing.
                //
                // Only the additive-displacement brushes are capped. They accumulate
                // without bound, so an uncapped full-strength dab overshoots — that is
                // the whole reason the mouse path was held back to 0.6 in the first
                // place. Move/Limb read strength as a cursor-tracking ratio, and
                // Smooth/Mask/Paint converge on a target (neighbour average, mask=1, a
                // colour); capping those would only make them take longer to arrive at
                // the same result, so they keep the full 1.0 ceiling.
                BrushType bt = input.live_brush_slot();
                bool additive_brush = (bt == BrushType::DRAW  || bt == BrushType::INFLATE ||
                                       bt == BrushType::CREASE || bt == BrushType::PINCH);
                float ceiling = additive_brush ? input.max_effect : 1.0f;
                eff_strength *= PRESSURE_STR_FLOOR +
                                (ceiling - PRESSURE_STR_FLOOR) * p_shaped;

                float eff_brush_size = input.brush_size *
                    (PRESSURE_SIZE_FLOOR + (1.0f - PRESSURE_SIZE_FLOOR) * p_shaped);

                // Speed falloff: tracked every stroke frame so the speed is warm if the
                // slider moves mid-session; applied only without real pressure, and not
                // to Move/Limb, which read strength as how closely to track the cursor.
                {
                    const double now = glfwGetTime();
                    const float dt = (float)(now - g_stroke_speed.t);
                    if (dt > 0.0f) {
                        const float d = std::hypot((float)input.mouse_x - g_stroke_speed.x,
                                                   (float)input.mouse_y - g_stroke_speed.y);
                        const float a = 1.0f - std::exp(-dt / SPEED_TAU);
                        g_stroke_speed.px_s += (d / dt - g_stroke_speed.px_s) * a;
                        g_stroke_speed.x = (float)input.mouse_x;
                        g_stroke_speed.y = (float)input.mouse_y;
                        g_stroke_speed.t = now;
                    }
                    // Hardness softens at half the rate and the size shrinks at a tenth: a
                    // fast stroke lands lighter, a little more feathered and slightly
                    // thinner, the way a quick pen stroke does. Size never drops below 80%
                    // (SPEED_SIZE_FLOOR), and spacing and the density compensation below
                    // already follow eff_brush_size, as for a pen.
                    if (pressure_is_synthetic && !is_grab && input.speed_gain > 0.0f) {
                        const float k = input.speed_gain * g_stroke_speed.px_s / SPEED_REF;
                        eff_strength   *= std::max(SPEED_FLOOR, 1.0f / (1.0f + k));
                        const float hf  = std::max(SPEED_FLOOR, 1.0f / (1.0f + 0.5f * k));
                        // Paint feathers the softness instead: scaling hardness itself
                        // meant full hardness never arrived mid-stroke, and the hard,
                        // vertex-stepped edge at the top of the slider is the point there.
                        if (input.current_brush == BrushType::PAINT)
                            eff_hardness = std::max(0.0f, 1.0f - (1.0f - eff_hardness) / hf);
                        else
                            eff_hardness *= hf;
                        eff_brush_size *= std::max(SPEED_SIZE_FLOOR, 1.0f / (1.0f + 0.1f * k));
                    }
                }

                // This frame's pointer path: from where the last frame's path ended,
                // through every position reported since (InputState::path_x), to the
                // cursor. Dabs are laid along its arc length, carrying the distance left
                // over since the last dab into the next frame, so a fast circle at a low
                // frame rate still follows the hand instead of cutting corners between
                // one sample per frame. (A Catmull-Rom spline through those per-frame
                // samples used to stand in for the missing points; with the real points
                // there is nothing left to guess.)
                float cur_x = (float)input.mouse_x;
                float cur_y = (float)input.mouse_y;
                if (input.release_tail) {   // the release frame: stop where the button came up
                    input.path_n = input.release_path_n;
                    cur_x = input.release_x;
                    cur_y = input.release_y;
                }
                static float pts_x[InputState::kPathMax + 2], pts_y[InputState::kPathMax + 2];
                int np = 0;
                pts_x[np] = brush_stroke.walk_x;
                pts_y[np] = brush_stroke.walk_y;
                np++;
                for (int i = 0; i < input.path_n; i++) {
                    pts_x[np] = input.path_x[i];
                    pts_y[np] = input.path_y[i];
                    np++;
                }
                if (pts_x[np - 1] != cur_x || pts_y[np - 1] != cur_y) {
                    pts_x[np] = cur_x;
                    pts_y[np] = cur_y;
                    np++;
                }
                g_path_stats.frame(input.path_n);

                float dab_dist = brush_stroke.walk_carry;
                for (int i = 0; i + 1 < np; i++) {
                    float sx = pts_x[i + 1] - pts_x[i], sy = pts_y[i + 1] - pts_y[i];
                    dab_dist += std::sqrt(sx * sx + sy * sy);
                }
                // eff.spacing (not the live global): transient shift-smooth doesn't
                // swap the live globals, so this is what makes it use Smooth's spacing.
                //
                // Scale off eff_brush_size, NOT input.brush_size: the dab that gets
                // dispatched has the pressure-scaled radius, so pairing the step with
                // the raw slider left a light-pressure stroke stepping up to 1/0.40 =
                // 2.5x its own footprint — gaps at nothing but a soft touch.
                // Ceiling enforced HERE, not just on the slider: eff comes straight out
                // of per_brush[], which load, profile-switch and brush-switch all write.
                // Clay shipped a visible stamp lattice once because one of those paths
                // handed it another brush's spacing, so the invariant lives at the point
                // of use where nothing can route around it. See max_spacing_for.
                float eff_spacing = std::min(eff.spacing,
                                             max_spacing_for(input.live_brush_slot()));
                float spacing = eff_brush_size * eff_spacing;

                // Foreshortening. This step is in screen pixels, but the dab deposits
                // into a world-space SPHERE of a radius fixed by the orbit distance
                // (BrushStroke::anchor_world_radius). On a surface tilted t away from
                // the view plane, a d-pixel screen step lands d/cos(t) apart in 3D, so
                // a fixed screen step spreads the spheres until they stop overlapping
                // (past ~1.0 R) and a continuous stroke breaks into discrete beads —
                // the long-standing "choppy at oblique angles" bug. Multiply the step
                // by cos(t) so the WORLD spacing stays put.
                //
                // The point normal is enough here: it only scales the step, so texel
                // noise perturbs spacing slightly rather than placing anything. This
                // is the same cosine set_anchor already inverts for screen_slack — the
                // scan box was foreshortening-corrected all along, the step never was.
                // Falls back to 1.0 (old behaviour) whenever the plane cache can't
                // answer, so a not-yet-landed webgpu readback can't stall the stroke.
                {
                    float n[3];
                    if (renderer.sample_normal((int)cur_x, (int)cur_y, n)) {
                        float nl = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                        if (nl > 1e-6f) {
                            Vec3 vd = camera.get_view_direction();
                            float c = std::fabs((vd.x*n[0] + vd.y*n[1] + vd.z*n[2]) / nl);
                            spacing *= std::max(c, DAB_SPACING_MIN_COS);
                        }
                    }
                }
                spacing = std::max(spacing, 1.0f);   // never sub-pixel

                // Captured before the dab block flips BEGIN to ACTIVE.
                const bool first_dab = brush_stroke.phase == StrokePhase::BEGIN;
                int dab_count = 0;
                if (is_grab || first_dab) {
                    dab_count = 1;
                } else if (dab_dist >= spacing) {
                    dab_count = (int)(dab_dist / spacing);
                }

                // Budget. Every dab dispatches over the WHOLE vertex array, so the
                // count is linear GPU cost and the correction above can multiply it by
                // 3.3x. Stretch the step to fit rather than dropping dabs: dropping
                // them would leave last_dab behind the cursor and the stroke would
                // rubber-band for frames afterwards. Only bites on a fast flick, where
                // the spacing was already coarser than the tuning anyway.
                if (dab_count > DAB_COUNT_MAX) {
                    dab_count = DAB_COUNT_MAX;
                    spacing = dab_dist / (float)DAB_COUNT_MAX;
                }

                // Crease Build: while the brush rests (no dab due from movement), keep
                // dabbing where the last one landed, up to 20 a second at 100%. A pause
                // longer than a frame doesn't bank dabs: at most one per frame, no backlog.
                bool hold_dab = false;
                {
                    static double hold_prev = 0.0;
                    static float hold_acc = 0.0f;
                    const double now = glfwGetTime();
                    const bool building = input.current_brush == BrushType::CREASE && !is_smooth
                                       && !is_grab && input.crease_build > 0.0f;
                    if (first_dab || !building || dab_count > 0) {
                        hold_acc = 0.0f;
                    } else {
                        hold_acc += (float)std::min(now - hold_prev, 0.1);
                        const float interval = 1.0f / (20.0f * input.crease_build);
                        if (hold_acc >= interval) {
                            hold_acc = std::min(hold_acc - interval, interval);
                            dab_count = 1;
                            hold_dab = true;
                        }
                    }
                    hold_prev = now;
                }

                // Decouple stroke strength from dab DENSITY.
                //
                // The additive brushes deposit per dab, so material laid per unit of
                // stroke length is (per-dab amount)/spacing — which means every change
                // to spacing silently rescaled strength. That coupling predates all of
                // this, but the foreshortening fix made it bite: taking spacing off
                // eff_brush_size packs dabs 1/(0.40+0.60*pressure) tighter, up to 2.5x
                // at the pressure floor, so mid-pressure pen strokes started depositing
                // ~1.4-2.5x more per unit length than the defaults were tuned against.
                // It read as "the brush strength reduction got reverted".
                //
                // Normalise against a FIXED reference — face-on, full pressure, at the
                // default spacing — so material per unit length is what strength says it
                // is, independent of pressure-size, viewing angle AND the spacing slider.
                // A grazing stroke lays the same bead as a face-on one instead of the old
                // thin, gappy one.
                //
                // The reference used to be `input.brush_size * eff.spacing`, i.e. it
                // tracked the live spacing setting — so eff.spacing appeared in both the
                // actual step and the reference and cancelled itself out of the ratio.
                // The correction covered pressure and tilt but not the user's own slider,
                // leaving deposit per unit length scaling as 1/spacing: tighten the
                // spacing and the same nominal strength dug in visibly harder. Referencing
                // SPACING_REF instead leaves the ratio proportional to eff.spacing, which
                // is exactly what cancels the 1/spacing in dabs-per-unit-length.
                //
                // Same brush set as the max-effect ceiling above (additive_brush), for
                // the same reason: these accumulate without bound. Move/Limb track the
                // cursor, and Smooth/Mask/Paint/Clay converge on a target, so denser dabs
                // make those settle sooner rather than pile up.
                //
                // The clamp scales with the spacing setting rather than sitting at a flat
                // [0.20, 1.0]. Its job is unchanged — the ceiling stops the budget branch
                // above turning a fast flick into a gouge, the floor keeps a
                // near-silhouette dab registering — but a flat 1.0 ceiling would clip the
                // very boost that decouples wide spacing, undoing the fix above 0.25. At
                // s == SPACING_REF the window is [0.20, 1.0], i.e. exactly what it was.
                {
                    float ref_spacing = input.brush_size * SPACING_REF;
                    if (additive_brush && ref_spacing > 1e-6f) {
                        float window = eff.spacing / SPACING_REF;
                        float comp   = spacing / ref_spacing;
                        float lo = 0.20f * window, hi = window;
                        comp = comp < lo ? lo : (comp > hi ? hi : comp);
                        eff_strength *= comp;
                    }
                }

                if (dab_count > 0) {
                if (brush_stroke.phase == StrokePhase::BEGIN)
                    brush_stroke.phase = StrokePhase::ACTIVE;

                // Place this frame's dabs along the path: the first one `spacing` of arc
                // past the last dab (the carry has already covered part of that), then
                // every `spacing` after it. Grab brushes and a stroke's first dab sit on
                // the cursor itself.
                static float dab_px[DAB_COUNT_MAX], dab_py[DAB_COUNT_MAX];
                if (is_grab || first_dab) {
                    dab_px[0] = cur_x;
                    dab_py[0] = cur_y;
                } else if (hold_dab) {
                    dab_px[0] = brush_stroke.last_dab_x;
                    dab_py[0] = brush_stroke.last_dab_y;
                } else {
                    int k = 0;
                    float to_next = std::max(spacing - brush_stroke.walk_carry, 0.0f);
                    for (int i = 0; i + 1 < np && k < dab_count; i++) {
                        float ax = pts_x[i], ay = pts_y[i];
                        float sx = pts_x[i + 1] - ax, sy = pts_y[i + 1] - ay;
                        float len = std::sqrt(sx * sx + sy * sy);
                        float at = 0.0f;   // arc position within this segment
                        while (k < dab_count && len - at >= to_next) {
                            at += to_next;
                            float t = len > 1e-6f ? at / len : 1.0f;
                            dab_px[k] = ax + sx * t;
                            dab_py[k] = ay + sy * t;
                            k++;
                            to_next = spacing;
                        }
                        to_next -= len - at;
                    }
                    for (; k < dab_count; k++) {   // float rounding at the very end
                        dab_px[k] = cur_x;
                        dab_py[k] = cur_y;
                    }
                }

                brush_stroke.gpu_dirty.clear();

                for (int dab_i = 0; dab_i < dab_count; dab_i++) {
                    const float dab_x = dab_px[dab_i];
                    const float dab_y = dab_py[dab_i];

                    DabContext ctx { renderer, camera, compute, *mesh, *multires, input, win_w, win_h,
                                     brush_stroke.vertex_count, eff_brush_size,
                                     mirror_effective(), mirror_pairs_effective() };

                    if (is_smooth) {
                        // Smooth gesture while painting blends colours, not geometry.
                        // Same idea while masking: blend mask values, not geometry —
                        // otherwise the stroke moves verts but finalize (keyed on
                        // current_brush == MASK) records only mask deltas, leaving an
                        // un-undoable geometry edit on the mesh.
                        if (input.current_brush == BrushType::PAINT &&
                            input.paint_target_density &&
                            compute.supported && compute.has_density_smooth()) {
                            brush_stroke.apply_density_smooth_gpu(ctx, dab_x, dab_y, eff_strength, eff_hardness);
                        } else if (input.current_brush == BrushType::PAINT &&
                            compute.supported && compute.has_color_smooth()) {
                            brush_stroke.apply_color_smooth_gpu(ctx, dab_x, dab_y, eff_strength, eff_hardness);
                        } else if (input.current_brush == BrushType::MASK &&
                                   compute.supported && compute.has_mask_smooth()) {
                            brush_stroke.apply_mask_smooth_gpu(ctx, dab_x, dab_y, eff_strength, eff_hardness);
                        } else {
                            brush_stroke.apply_smooth(ctx, dab_x, dab_y, eff_strength, eff_hardness);
                        }
                    } else if (is_move) {
                        brush_stroke.apply_move_gpu(ctx, dx, dy, eff_strength, eff_hardness);
                    } else if (is_limb) {
                        brush_stroke.apply_limb_gpu(ctx, dx, dy, eff_strength, eff_hardness);
                    } else if (is_pose) {
                        brush_stroke.apply_pose_gpu(ctx, dx, dy, eff_hardness,
                                                    scene.active_entity().pose,
                                                    input.pose_hover_joint);
                    } else if (input.current_brush == BrushType::CREASE) {
                        brush_stroke.apply_crease(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                  input.is_subtract_active());
                    } else if (input.current_brush == BrushType::PINCH) {
                        brush_stroke.apply_pinch(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                  input.is_subtract_active());
                    } else if (input.current_brush == BrushType::MASK) {
                        if (compute.supported && compute.has_mask()) {
                            brush_stroke.apply_mask_gpu(ctx, dab_x, dab_y,
                                                        eff_strength, eff_hardness,
                                                        input.is_subtract_active());
                        } else {
                            brush_stroke.apply_mask(renderer, camera, *mesh,
                                                    dab_x, dab_y,
                                                    eff_brush_size, eff_strength,
                                                    eff_hardness,
                                                    input.is_subtract_active(),
                                                    ctx.mirror_pairs,  // no B lobe here — this fallback walks mirror_x_map
                                                    win_w, win_h);

                            std::vector<uint32_t> mask_dirty;
                            brush_stroke.apply_mask_changes(*mesh, mask_dirty);
                            if (!mask_dirty.empty()) {
                                renderer.update_mask_partial(*mesh, mask_dirty);
                            }
                        }
                    } else if (input.current_brush == BrushType::PAINT) {
                        if (input.paint_target_density) {
                            if (compute.supported && compute.has_density_kernels()) {
                                brush_stroke.apply_density_gpu(ctx, dab_x, dab_y,
                                                               eff_strength, eff_hardness,
                                                               input.is_subtract_active());
                            }
                        } else if (compute.supported && compute.has_color()) {
                            brush_stroke.apply_color_gpu(ctx, dab_x, dab_y,
                                                         eff_strength, eff_hardness,
                                                         input.is_subtract_active());
                        }
                    } else if (input.current_brush == BrushType::DRAW) {
                        brush_stroke.apply_draw(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                input.is_subtract_active());
                    } else if (input.current_brush == BrushType::CLAY) {
                        brush_stroke.apply_draw(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                input.is_subtract_active(), false, true);
                    } else if (input.current_brush == BrushType::INFLATE) {
                        brush_stroke.apply_draw(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                input.is_subtract_active(), true);
                    }

                    // Mask and paint write straight to their own VBO and never move
                    // geometry, so they skip the position/normal post-dab sync.
                    bool is_nongeo = input.current_brush == BrushType::MASK
                                  || input.current_brush == BrushType::PAINT;
                    if (!is_nongeo) {
                        brush_stroke.post_dab(ctx);
                    }
                }

                brush_stroke.last_dab_x = dab_px[dab_count - 1];
                brush_stroke.last_dab_y = dab_py[dab_count - 1];
                brush_stroke.needs_mesh_update = true;
                } // dab_count > 0

                // The next frame's path starts here. Whatever arc is left past the last
                // dab carries over; with no dab this frame, all of it does.
                if (is_grab || first_dab)
                    brush_stroke.walk_carry = 0.0f;
                else if (hold_dab)
                    brush_stroke.walk_carry = dab_dist;   // a rest dab spends none of the path
                else
                    brush_stroke.walk_carry = std::max(dab_dist - spacing * (float)dab_count, 0.0f);
                brush_stroke.walk_x = cur_x;
                brush_stroke.walk_y = cur_y;

                // Runs every stroke frame (not just dab frames): drains landed async
                // dab readbacks — snap/normals bookkeeping for dabs kicked 1–2 frames
                // ago — then dispatches partial normals for whatever landed.
                {
                    DabContext ctx { renderer, camera, compute, *mesh, *multires, input, win_w, win_h,
                                     brush_stroke.vertex_count, eff_brush_size,
                                     mirror_effective(), mirror_pairs_effective() };
                    brush_stroke.post_frame(ctx);
                }
            }
        }

        app_render_frame(app, win_w, win_h, imgui_wants_mouse);   // app_render.cpp: draw, UI, present
        if (bench::active()) {
            const int lvl = multires->current_level;
            bench::Phase ph = bench::Phase::IDLE;
            if (lvl != perf_prev_level)                          ph = bench::Phase::SWITCH;
            else if (brush_stroke.is_active())                   ph = bench::Phase::SCULPT;
            else if (app_state == AppState::SCULPTING
                     || brush_stroke.reconciling()
                     || brush_stroke.dab_readbacks_pending())    ph = bench::Phase::PENUP;
            bench::frame_end(lvl, ph);
            perf_prev_level = lvl;
        }
        input.end_frame();
    };

#ifdef __EMSCRIPTEN__
    web_install_dom_listeners();   // web_bridge.cpp

    // Browser drives the loop via requestAnimationFrame (fps=0). simulate_infinite_loop
    // =true unwinds out of main() but preserves its stack, so &frame and every captured
    // local stay valid; the trampoline (a captureless lambda → C function pointer)
    // forwards each tick to `frame`. Code after this line never runs on the web.
    emscripten_set_main_loop_arg(
        [](void* p) { (*static_cast<decltype(frame)*>(p))(); }, &frame, 0, true);
#else
    while (!glfwWindowShouldClose(window)) frame();
#endif

    ui_backdrop::shutdown();
#ifdef CHISEL_BACKEND_WEBGPU
    ImGui_ImplWGPU_Shutdown();
#else
    ImGui_ImplOpenGL3_Shutdown();
#endif
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Catch anything still inside the autosave debounce window on the way out. Native
    // only — on web this line is unreachable (see the main-loop note above), which is
    // what chisel_flush_settings/pagehide exists to cover.
    settings_save(input);

    tablet.shutdown();
    bench::shutdown();
    debug_console::shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
