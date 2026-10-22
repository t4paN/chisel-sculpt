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

#ifdef CHISEL_BACKEND_WEBGPU
// WebGPU windowing lives in the app (the seam owns resources, not the surface —
// see gpu.h). These mirror the proven setup in src/gpu/wgpu_window.cpp, which is a
// standalone probe (its own main) and so isn't linked into the app.
namespace {
struct AdapterResult { WGPUAdapter adapter = nullptr; bool done = false; bool ok = false; };
struct DeviceResult  { WGPUDevice  device  = nullptr; bool done = false; bool ok = false; };

void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
               WGPUStringView msg, void* ud1, void*) {
    auto* r = static_cast<AdapterResult*>(ud1);
    r->done = true;
    if (status == WGPURequestAdapterStatus_Success) { r->adapter = adapter; r->ok = true; }
    else std::printf("[win] adapter failed: status=%d msg=%.*s\n",
                     (int)status, (int)msg.length, msg.data ? msg.data : "");
}
void onDevice(WGPURequestDeviceStatus status, WGPUDevice device,
              WGPUStringView msg, void* ud1, void*) {
    auto* r = static_cast<DeviceResult*>(ud1);
    r->done = true;
    if (status == WGPURequestDeviceStatus_Success) { r->device = device; r->ok = true; }
    else std::printf("[win] device failed: status=%d msg=%.*s\n",
                     (int)status, (int)msg.length, msg.data ? msg.data : "");
}

WGPUSurface  g_surface = nullptr;
WGPUDevice   g_device  = nullptr;
WGPUAdapter  g_adapter = nullptr;   // kept for present-mode queries in configureSurface
WGPUTexture  g_depth_tex  = nullptr;
WGPUTextureView g_depth_view = nullptr;
WGPUTextureFormat g_surface_fmt = WGPUTextureFormat_BGRA8Unorm;
WGPUTextureUsage  g_surface_usage = WGPUTextureUsage_RenderAttachment;   // + CopySrc if offered
static const WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth24Plus;
// Set when a frame's surface texture first comes back suboptimal at a window size: that
// frame still presents, and the next frame reconfigures before acquiring.
// Reconfiguring while the texture is still acquired aborts inside wgpu-native.
bool g_surface_reconfigure = false;

void configureSurface(int w, int h) {
    WGPUSurfaceConfiguration cfg = {};
    cfg.device      = g_device;
    cfg.format      = g_surface_fmt;
    cfg.usage       = g_surface_usage;
    cfg.width       = (uint32_t)w;
    cfg.height      = (uint32_t)h;
    cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
    cfg.presentMode = WGPUPresentMode_Fifo;
#ifndef __EMSCRIPTEN__
    // CHISEL_PERF_VSYNC=0: present uncapped (Immediate, else Mailbox) if the surface offers it.
    if (!bench::vsync() && g_adapter) {
        WGPUSurfaceCapabilities caps = {};
        if (wgpuSurfaceGetCapabilities(g_surface, g_adapter, &caps) == WGPUStatus_Success) {
            for (size_t i = 0; i < caps.presentModeCount; ++i)
                if (caps.presentModes[i] == WGPUPresentMode_Immediate) cfg.presentMode = WGPUPresentMode_Immediate;
            if (cfg.presentMode == WGPUPresentMode_Fifo)
                for (size_t i = 0; i < caps.presentModeCount; ++i)
                    if (caps.presentModes[i] == WGPUPresentMode_Mailbox) cfg.presentMode = WGPUPresentMode_Mailbox;
            wgpuSurfaceCapabilitiesFreeMembers(caps);
        }
    }
#endif
    wgpuSurfaceConfigure(g_surface, &cfg);
}

// Stand-in colour target for a frame the swapchain didn't hand over (Timeout, or a
// lost surface that stayed lost after the reconfigure). Every default-screen pass
// still needs a real attachment: a pass begun on a null view fails validation and
// wgpu-native aborts the whole app (seen 2026-10-18, status 3 = Timeout on KWin).
// The frame is drawn here and thrown away; only allocated if that ever happens.
WGPUTexture     g_spare_tex  = nullptr;
WGPUTextureView g_spare_view = nullptr;
int g_spare_w = 0, g_spare_h = 0;

WGPUTextureView spareColorView(int w, int h) {
    if (g_spare_view && g_spare_w == w && g_spare_h == h) return g_spare_view;
    if (g_spare_view) { wgpuTextureViewRelease(g_spare_view); g_spare_view = nullptr; }
    if (g_spare_tex)  { wgpuTextureRelease(g_spare_tex);      g_spare_tex  = nullptr; }
    WGPUTextureDescriptor td = {};
    td.usage = WGPUTextureUsage_RenderAttachment;
    td.dimension = WGPUTextureDimension_2D;
    td.size = { (uint32_t)w, (uint32_t)h, 1 };
    td.format = g_surface_fmt;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    g_spare_tex  = wgpuDeviceCreateTexture(g_device, &td);
    g_spare_view = wgpuTextureCreateView(g_spare_tex, nullptr);
    g_spare_w = w; g_spare_h = h;
    return g_spare_view;
}

void makeDepth(int w, int h) {
    if (g_depth_view) { wgpuTextureViewRelease(g_depth_view); g_depth_view = nullptr; }
    if (g_depth_tex)  { wgpuTextureRelease(g_depth_tex);      g_depth_tex  = nullptr; }
    WGPUTextureDescriptor td = {};
    td.usage = WGPUTextureUsage_RenderAttachment;
    td.dimension = WGPUTextureDimension_2D;
    td.size = { (uint32_t)w, (uint32_t)h, 1 };
    td.format = kDepthFormat;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    g_depth_tex  = wgpuDeviceCreateTexture(g_device, &td);
    g_depth_view = wgpuTextureCreateView(g_depth_tex, nullptr);
    gpu::webgpu_set_default_depth(g_depth_view);
}
} // namespace
#endif // CHISEL_BACKEND_WEBGPU

#ifdef __EMSCRIPTEN__
// Browser save/open bridge. MEMFS is RAM that dies with the tab, and host files
// are invisible to it, so path-based dialogs are meaningless on web: save/export
// write through the normal writers into MEMFS and the bytes leave as a browser
// download; open/import goes through an <input type=file> picker whose bytes
// land back in MEMFS for the normal loaders. No IDBFS persistence by design.

// Settings autosave backstop. The frame loop's debounced write is not enough on its own
// here: a browser can discard a tab without ever running an exit path, and native's
// shutdown flush has no web equivalent (emscripten_set_main_loop_arg never returns).
// pagehide/visibilitychange are the last events that reliably fire — beforeunload does
// not on mobile. Points at main()'s InputState, which outlives every frame (the
// simulate_infinite_loop throw keeps main's stack frame alive).
static InputState* g_settings_input = nullptr;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_flush_settings() {
    if (g_settings_input) settings_save(*g_settings_input);
}

// Asked by the page's keydown hook whether the browser should be kept off this key:
// yes for anything while the Key bindings window is open (it is listening, and Ctrl+D
// must reach it rather than bookmark the page), else yes only when a binding wants
// exactly this key and these modifiers. Exact, not the subset rule key_callback uses,
// so a plain F binding does not also swallow the browser's own Ctrl+F.
extern "C" EMSCRIPTEN_KEEPALIVE int chisel_key_wanted(int glfw_key, int ctrl, int shift, int alt) {
    if (!g_settings_input) return 0;
    const InputState& in = *g_settings_input;
    if (in.keybind_menu_open) return 1;
    const int mods = (ctrl ? GLFW_MOD_CONTROL : 0) | (shift ? GLFW_MOD_SHIFT : 0)
                   | (alt ? GLFW_MOD_ALT : 0);
    for (int i = 0; i < (int)KeyAction::COUNT; i++)
        if (in.keybinds[i].bound() && in.keybinds[i].key == glfw_key && in.keybinds[i].mods == mods)
            return 1;
    return 0;
}

// Set async by the picker's JS callback; the frame loop consumes it. Safe to
// assign from JS mid-ASYNCIFY-suspend: plain reentry, nothing here suspends.
static std::string g_web_import_path;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_web_import_done(const char* path) {
    g_web_import_path = path ? path : "";
}

// Brush-alpha custom image: same MEMFS bridge as import, separate path so the two
// pickers can't clobber each other. The frame loop consumes it into the alpha pool.
static std::string g_web_alpha_path;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_web_alpha_done(const char* path) {
    g_web_alpha_path = path ? path : "";
}

// Read fs_path out of MEMFS and hand it to the browser, deleting the MEMFS copy.
// Preferred route is showSaveFilePicker (the browser's real save dialog: folder
// browsing + editable name, prefilled with download_name) — but Chrome blocks it
// in cross-origin iframes (itch embeds the game that way), so any refusal falls
// back to a plain anchor download. A user cancel (AbortError) saves nothing.
static void web_download_file(const char* fs_path, const char* download_name) {
    EM_ASM({
        var path = UTF8ToString($0);
        var name = UTF8ToString($1);
        var data = FS.readFile(path);
        FS.unlink(path);
        var fallback = function() {
            var a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([data]));
            a.download = name;
            a.click();
            setTimeout(function() { URL.revokeObjectURL(a.href); }, 10000);
        };
        if (window.showSaveFilePicker) {
            window.showSaveFilePicker({ suggestedName: name }).then(function(h) {
                return h.createWritable().then(function(w) {
                    return w.write(data).then(function() { return w.close(); });
                });
            }).catch(function(e) {
                if (e && e.name === 'AbortError') return;   // user cancelled
                fallback();   // SecurityError (cross-origin iframe) etc.
            });
        } else {
            fallback();       // Firefox/Safari: no File System Access API
        }
    }, fs_path, download_name);
}

// Fire the browser's file picker. Needs transient user activation, so it must
// run within a few seconds of the click/keypress that requested it (it does:
// the request flag is consumed the next frame). Result arrives whenever the
// user picks — chisel_web_import_done gets the MEMFS path.
static void web_open_file_picker() {
    EM_ASM({
        var inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = '.chisel,.obj,.ply';
        inp.onchange = function() {
            var f = inp.files && inp.files[0];
            if (!f) return;
            f.arrayBuffer().then(function(buf) {
                var path = '/' + f.name;
                FS.writeFile(path, new Uint8Array(buf));
                Module.ccall('chisel_web_import_done', null, ['string'], [path]);
            });
        };
        inp.click();
    });
}

// Brush-alpha image picker — twin of web_open_file_picker, image types, lands in
// MEMFS under /alpha_ (namespaced so it never collides with an import) and reports
// via chisel_web_alpha_done.
static void web_open_alpha_picker() {
    EM_ASM({
        var inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'image/png,image/jpeg,.png,.jpg,.jpeg,.tga,.bmp';
        inp.onchange = function() {
            var f = inp.files && inp.files[0];
            if (!f) return;
            f.arrayBuffer().then(function(buf) {
                var path = '/alpha_' + f.name;
                FS.writeFile(path, new Uint8Array(buf));
                Module.ccall('chisel_web_alpha_done', null, ['string'], [path]);
            });
        };
        inp.click();
    });
}
#endif // __EMSCRIPTEN__

enum class AppState { IDLE, SCULPTING };

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

// Reports the projection truth check when it ran (CHISEL_PROJECT_CHECK=1, or a
// CHISEL_DEBUG_MULTIRES build). project_down_to_level promises that cascading back to
// L_max reproduces the pre-projection surface EXACTLY, so any real distance here is a
// broken contract, not a tolerance to tune. Silent when the check did not run, so the
// normal log is unchanged.
//
// The 1e-4 line is a heuristic, not a spec: a few cascade steps of float arithmetic on
// O(1) coordinates land around 1e-6, while a mean edge at L9 is ~5e-3 — so anything at
// 1e-4 is already far outside rounding and heading for visible.
static void report_projection_check(const ProjectionStats& ps) {
    if (!ps.checked) return;
    const bool bad = ps.max_reconstruction_error > 1e-4;
    std::printf("[project] CHECK reconstruction max error %.6g (worst vert %u)%s\n",
                ps.max_reconstruction_error, ps.worst_vertex,
                bad ? "  <-- LOSSY, cascade does not reproduce the surface" : "  ok");
    std::fflush(stdout);
}

#ifndef __EMSCRIPTEN__
// Native file dialogs (Open, Save, Export, Load Brush Alpha) open centred on the
// screen rather than wherever ImGui last left them (often under the button that
// opened them). Appearing only, so the user can still drag one aside. The position
// is set only when this key's dialog is up: otherwise it would land on whatever
// window begins next.
static bool display_file_dialog(IGFD::FileDialog* fd, const char* key) {
    if (fd->IsOpened(key))
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                                ImVec2(0.5f, 0.5f));
    return fd->Display(key, ImGuiWindowFlags_NoCollapse, ImVec2(600, 400));
}
#endif

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
#if defined(__EMSCRIPTEN__)
    WGPUInstance instance = wgpuCreateInstance(nullptr);
#else
    // wgpu-native's own warnings and errors, so driver/surface trouble reaches the log.
    wgpuSetLogCallback([](WGPULogLevel level, WGPUStringView msg, void*) {
        std::printf("[wgpu] %s: %.*s\n", level == WGPULogLevel_Error ? "error" : "warn",
                    (int)msg.length, msg.data ? msg.data : "");
        std::fflush(stdout);
    }, nullptr);
    wgpuSetLogLevel(WGPULogLevel_Warn);
    // CHISEL_WGPU_BACKEND=dx12|vulkan|gl pins wgpu-native to one API (default: let it
    // choose). For telling a driver's presentation bug from ours: same window state,
    // other API.
    WGPUInstanceExtras inst_extras = {};
    inst_extras.chain.sType = (WGPUSType)WGPUSType_InstanceExtras;
    if (const char* be = std::getenv("CHISEL_WGPU_BACKEND")) {
        if      (!std::strcmp(be, "dx12"))   inst_extras.backends = WGPUInstanceBackend_DX12;
        else if (!std::strcmp(be, "vulkan")) inst_extras.backends = WGPUInstanceBackend_Vulkan;
        else if (!std::strcmp(be, "gl"))     inst_extras.backends = WGPUInstanceBackend_GL;
        else std::printf("[win] CHISEL_WGPU_BACKEND=%s not recognised (dx12|vulkan|gl)\n", be);
    }
    WGPUInstanceDescriptor inst_desc = {};
    inst_desc.nextInChain = &inst_extras.chain;
    WGPUInstance instance = wgpuCreateInstance(inst_extras.backends ? &inst_desc : nullptr);
#endif
    if (!instance) { std::fprintf(stderr, "wgpuCreateInstance failed\n"); return 1; }
    // The seam's web readbacks pump this instance's event loop (no-op on native).
    gpu::webgpu_set_instance(instance);
    WGPUSurfaceDescriptor sd = {};
#if defined(__EMSCRIPTEN__)
    // Browser: bind the surface to the page canvas by CSS selector. Emscripten's
    // GLFW (-sUSE_GLFW=3) drives that same "#canvas" element for input/resize.
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasSrc = {};
    canvasSrc.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
    canvasSrc.selector = WGPUStringView{ "#canvas", 7 };
    sd.nextInChain = &canvasSrc.chain;
#else
    // The surface source MUST match the platform GLFW actually chose. In a
    // Wayland session glfwGetX11Display() returns NULL, and wgpu-native does
    // not report that as an error — it panics inside the Vulkan backend with
    // "Display pointer is not set" and aborts the process, so there is nothing
    // to gate on after the fact. Branch up front instead. Both descriptors are
    // declared out here because the chain must outlive CreateSurface below.
  #if defined(_WIN32)
    WGPUSurfaceSourceWindowsHWND hwnd_src = {};
    hwnd_src.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
    hwnd_src.hinstance   = GetModuleHandleW(nullptr);
    hwnd_src.hwnd        = glfwGetWin32Window(window);
    sd.nextInChain = &hwnd_src.chain;
  #else
    // glfwGetPlatform is GLFW 3.4+; a distro 3.3 (Ubuntu/Mint 24.04) is X11-only.
    WGPUSurfaceSourceXlibWindow     x11 = {};
  #if GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4
    WGPUSurfaceSourceWaylandSurface wl  = {};
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        wl.chain.sType = WGPUSType_SurfaceSourceWaylandSurface;
        wl.display     = glfwGetWaylandDisplay();
        wl.surface     = glfwGetWaylandWindow(window);
        sd.nextInChain = &wl.chain;
    } else
  #endif
    {
        x11.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        x11.display     = glfwGetX11Display();
        x11.window      = (uint64_t)glfwGetX11Window(window);
        sd.nextInChain  = &x11.chain;
    }
  #endif
#endif
    g_surface = wgpuInstanceCreateSurface(instance, &sd);
    if (!g_surface) { std::fprintf(stderr, "createSurface failed\n"); return 1; }
#if defined(_WIN32)
    std::printf("[win] surface created (Win32)\n");
#elif !defined(__EMSCRIPTEN__) && (GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4)
    std::printf("[win] surface created (%s)\n",
                glfwGetPlatform() == GLFW_PLATFORM_WAYLAND ? "Wayland" : "X11");
#elif !defined(__EMSCRIPTEN__)
    std::printf("[win] surface created (X11)\n");
#endif

    AdapterResult ar;
    WGPURequestAdapterOptions aopt = {};
    aopt.compatibleSurface = g_surface;
    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode = WGPUCallbackMode_AllowProcessEvents;
    acb.callback = onAdapter;
    acb.userdata1 = &ar;
    wgpuInstanceRequestAdapter(instance, &aopt, acb);
    for (int i = 0; i < 200 && !ar.done; ++i) {
        wgpuInstanceProcessEvents(instance);
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);  // ASYNCIFY yield: requestAdapter is a JS promise that
                               // only resolves once control returns to the event loop.
#endif
    }
    if (!ar.ok) { std::fprintf(stderr, "no WebGPU adapter\n"); return 1; }

    DeviceResult dr;
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = onDevice;
    dcb.userdata1 = &dr;
    // Request an explicit, portable limit set — NOT the adapter's maxima verbatim.
    // Every compute kernel now fits the WebGPU baseline: storage buffers ≤8
    // (baseline 8, after remesh_smooth was consolidated), workgroups ≤256 invocations
    // (baseline 256), and no kernel uses workgroup shared memory. So we leave every
    // field at its baseline default (WGPU_LIMITS_INIT = all "undefined" → default) and
    // bump ONLY the two buffer-size fields — large sculpts can exceed the 128MB
    // baseline storage-buffer binding — up to whatever the adapter actually supports.
    // Result: the device creates on any conformant WebGPU implementation (baseline
    // browsers included), instead of demanding this GPU's high limits. If the web
    // build later trips a specific baseline field, bump that one field here.
    WGPULimits supported = WGPU_LIMITS_INIT;
    g_adapter = ar.adapter;
#if !defined(__EMSCRIPTEN__)
    {
        WGPUAdapterInfo info = {};
        if (wgpuAdapterGetInfo(ar.adapter, &info) == WGPUStatus_Success) {
            const char* api = info.backendType == WGPUBackendType_Vulkan ? "Vulkan"
                            : info.backendType == WGPUBackendType_D3D12  ? "D3D12"
                            : info.backendType == WGPUBackendType_Metal  ? "Metal"
                            : info.backendType == WGPUBackendType_OpenGL ||
                              info.backendType == WGPUBackendType_OpenGLES ? "GL" : "other";
            std::printf("[win] adapter: %.*s via %s\n", (int)info.device.length,
                        info.device.data ? info.device.data : "", api);
            wgpuAdapterInfoFreeMembers(info);
        }
    }
#endif
    wgpuAdapterGetLimits(ar.adapter, &supported);
    WGPULimits limits = WGPU_LIMITS_INIT;                     // all baseline defaults
    limits.maxBufferSize               = supported.maxBufferSize;
    limits.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
    WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    ddesc.requiredLimits = &limits;
#if !defined(__EMSCRIPTEN__)
    // Without a handler, wgpu-native turns any validation error into a panic, and a
    // panic across the C boundary aborts the process with no message (0xc0000409 in
    // wgpu_native.dll). Log the error and carry on instead.
    ddesc.uncapturedErrorCallbackInfo.callback =
        [](WGPUDevice const*, WGPUErrorType type, WGPUStringView msg, void*, void*) {
            std::printf("[wgpu] uncaptured error (type %d): %.*s\n", (int)type,
                        (int)msg.length, msg.data ? msg.data : "");
            std::fflush(stdout);
        };
#endif
    wgpuAdapterRequestDevice(ar.adapter, &ddesc, dcb);
    for (int i = 0; i < 200 && !dr.done; ++i) {
        wgpuInstanceProcessEvents(instance);
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);  // ASYNCIFY yield (see requestAdapter loop above).
#endif
    }
    if (!dr.ok) { std::fprintf(stderr, "no WebGPU device\n"); return 1; }
    g_device = dr.device;
    WGPUQueue queue = wgpuDeviceGetQueue(g_device);
    gpu::set_app_device(gpu::device_from_webgpu(g_device, queue));
    // Capture what the device actually granted (the request above raises the two
    // buffer-size fields to the adapter's maxima) — the subdivision guard sizes
    // predicted level buffers against these.
    {
        WGPULimits granted = WGPU_LIMITS_INIT;
        if (wgpuDeviceGetLimits(g_device, &granted) == WGPUStatus_Success) {
            gpu::DeviceLimits dl;
            dl.max_buffer_size          = granted.maxBufferSize;
            dl.max_storage_binding_size = granted.maxStorageBufferBindingSize;
            gpu::set_device_limits(dl);
        }
        gpu::DeviceLimits dl = gpu::device_limits();
        std::printf("[gpu] device limits: maxBuffer %llu MB, maxStorageBinding %llu MB\n",
                    (unsigned long long)(dl.max_buffer_size >> 20),
                    (unsigned long long)(dl.max_storage_binding_size >> 20));
    }

    WGPUSurfaceCapabilities caps = {};
    if (wgpuSurfaceGetCapabilities(g_surface, ar.adapter, &caps) != WGPUStatus_Success
        || caps.formatCount == 0) {
        std::fprintf(stderr, "surfaceGetCapabilities failed\n"); return 1;
    }
    // Prefer a non-sRGB (linear) surface format to match the GL backend, which writes
    // shader output straight to the default framebuffer with no sRGB encode. wgpu-native
    // usually lists the *sRGB* variant first (caps.formats[0] = BGRA8UnormSrgb), and
    // configuring that makes the GPU re-encode our already-display-ready colours →
    // everything washes out lighter. Pick the plain UNORM twin if the surface offers it.
    g_surface_fmt = caps.formats[0];
    for (size_t i = 0; i < caps.formatCount; ++i) {
        WGPUTextureFormat f = caps.formats[i];
        if (f == WGPUTextureFormat_BGRA8Unorm || f == WGPUTextureFormat_RGBA8Unorm) {
            g_surface_fmt = f;
            break;
        }
    }
    // CopySrc lets the modern UI skin copy the frame for its frosted-glass panels
    // (ui_backdrop). Optional: without it the panels are tinted but not blurred.
    if (caps.usages & WGPUTextureUsage_CopySrc)
        g_surface_usage = (WGPUTextureUsage)(g_surface_usage | WGPUTextureUsage_CopySrc);
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    gpu::webgpu_set_surface_format(g_surface_fmt);
    configureSurface(fbw, fbh);
    makeDepth(fbw, fbh);
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
            if (scene.edit_mark() == clean_edit_mark && !busy) {
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

        // Colour picker (C in paint mode): sample the stored albedo under the
        // click — tri + barycentrics from the screen FBO, colours from the CPU
        // mesh (synced at pen-up), so shading/matcap never leaks into the pick.
        if (input.color_pick_click) {
            input.color_pick_click = false;
            int cx = (int)input.mouse_x;
            int cy = (int)input.mouse_y;
            if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
                uint32_t tid;
                renderer.read_triid_region(cx, cy, 1, 1, &tid);
                const Mesh& pm = scene.active_mesh();   // screen FBO holds the active entity
                if (tid != 0xFFFFFFFF && tid < pm.tri_count()) {
                    float bary[2];
                    renderer.read_bary_region(cx, cy, 1, 1, bary);
                    float bu = bary[0], bv = bary[1], bw = 1.0f - bu - bv;
                    uint32_t i0 = pm.indices[tid*3+0];
                    uint32_t i1 = pm.indices[tid*3+1];
                    uint32_t i2 = pm.indices[tid*3+2];
                    auto vcol = [&](uint32_t v) {   // what you see: the layers' composite
                        return paint_composite_vertex(pm.paint_layers, pm.color, v);
                    };
                    uint32_t c0 = vcol(i0), c1 = vcol(i1), c2 = vcol(i2);
                    for (int ch = 0; ch < 3; ch++) {
                        float f0 = (float)((c0 >> (ch*8)) & 0xFF);
                        float f1 = (float)((c1 >> (ch*8)) & 0xFF);
                        float f2 = (float)((c2 >> (ch*8)) & 0xFF);
                        input.paint_color[ch] = (bu*f0 + bv*f1 + bw*f2) / 255.0f;
                    }
                    input.color_pick_active = false;
                    snprintf(input.notification, sizeof(input.notification),
                             "Picked #%02X%02X%02X",
                             (int)(input.paint_color[0]*255.0f + 0.5f),
                             (int)(input.paint_color[1]*255.0f + 0.5f),
                             (int)(input.paint_color[2]*255.0f + 0.5f));
                    input.notification_timer = 1.5f;
                }
            }
        }

        // Fill bucket (paint mode): flood the unmasked island under the click on the
        // active layer, flat or through a pattern (main colour to secondary); a gradient
        // runs from the press (A, main) to the release (B, secondary). Points come from the screen FBO exactly as the picker's
        // do (tri + barycentrics); mask and colour are the CPU copies, both synced at
        // pen-up. One PAINT undo entry, so Ctrl+Z takes the fill back.
        if (input.bucket_click) {
            input.bucket_click = false;
            Mesh& pm = scene.active_mesh();
            const int pattern = input.fill_pattern;
            const bool gradient = pattern == kFillGradient;
            const bool solid = pattern == kFillSolid || pattern < 0 || pattern >= fill_patterns().count();
            // Hardness below 1 feathers the fill off the mask, measured along the surface.
            const bool feathered = input.brush_hardness < 0.99f;
            // Patterns, gradients and the feather read positions/normals: pull the GPU's
            // back first.
            if (!solid || feathered) scene.materialize_active_cpu();
            // The surface under a screen pixel: nearest corner (the seed) and the point.
            float click_n[3] = {0.0f, 0.0f, 1.0f};   // the surface normal at the last surface_at hit
            auto surface_at = [&](float fx, float fy, uint32_t* seed, float out[3]) -> bool {
                const int cx = (int)fx, cy = (int)fy;
                if (cx < 0 || cx >= win_w || cy < 0 || cy >= win_h) return false;
                uint32_t tid = 0xFFFFFFFF;
                renderer.read_triid_region(cx, cy, 1, 1, &tid);
                if (tid == 0xFFFFFFFF || tid >= pm.tri_count()) return false;
                float bary[2];
                renderer.read_bary_region(cx, cy, 1, 1, bary);
                const float bw[3] = {bary[0], bary[1], 1.0f - bary[0] - bary[1]};
                const uint32_t ix[3] = {pm.indices[tid*3+0], pm.indices[tid*3+1], pm.indices[tid*3+2]};
                if (seed) *seed = (bw[0] >= bw[1] && bw[0] >= bw[2]) ? ix[0] : (bw[1] >= bw[2] ? ix[1] : ix[2]);
                if (out) {
                    out[0] = out[1] = out[2] = 0.0f;
                    for (int c = 0; c < 3; c++) {
                        out[0] += bw[c] * pm.pos_x[ix[c]];
                        out[1] += bw[c] * pm.pos_y[ix[c]];
                        out[2] += bw[c] * pm.pos_z[ix[c]];
                    }
                }
                if (pm.norm_x.size() >= pm.vertex_count()) {
                    Vec3 n{0, 0, 0};
                    for (int c = 0; c < 3; c++)
                        n = n + Vec3{pm.norm_x[ix[c]], pm.norm_y[ix[c]], pm.norm_z[ix[c]]} * bw[c];
                    if (n.length() > 1e-6f) {
                        n = n.normalized();
                        click_n[0] = n.x; click_n[1] = n.y; click_n[2] = n.z;
                    }
                }
                return true;
            };
            uint32_t seed = 0;
            float a[3], b[3];
            const float drag_px = std::hypot(input.bucket_bx - input.bucket_ax, input.bucket_by - input.bucket_ay);
            if (!surface_at(input.bucket_ax, input.bucket_ay, &seed, a)) {
                // Missed the model (on_model and the FBO can disagree by a pixel).
            } else if (gradient && drag_px < 4.0f) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "Drag across the model to set the gradient");
                input.notification_timer = 1.5f;
            } else {
                if (gradient && !surface_at(input.bucket_bx, input.bucket_by, nullptr, b)) {
                    // Released off the model: B on the view plane through A.
                    Vec3 cam_pos = camera.get_position();
                    Vec3 fwd = (camera.target - cam_pos).normalized();
                    Vec3 right = fwd.cross(Vec3{0, 1, 0}).normalized();
                    Vec3 up = right.cross(fwd).normalized();
                    const float d = camera.view_depth(Vec3{a[0], a[1], a[2]});
                    const float half_h = camera.half_height_at(d);
                    const float half_w = half_h * ((float)win_w / (float)win_h);
                    const float ndc_x = 2.0f * input.bucket_bx / (float)win_w - 1.0f;
                    const float ndc_y = 1.0f - 2.0f * input.bucket_by / (float)win_h;
                    Vec3 p = cam_pos + fwd * d + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                    b[0] = p.x; b[1] = p.y; b[2] = p.z;
                }
                // Patterns run from the main colour (black) to the secondary (white).
                auto pack = [](const float* rgb) {
                    uint32_t c = 0xFFu << 24;
                    for (int ch = 0; ch < 3; ch++)
                        c |= (uint32_t)(std::max(0.0f, std::min(1.0f, rgb[ch])) * 255.0f + 0.5f) << (ch * 8);
                    return c;
                };
                const uint32_t main_c = pack(input.paint_color), second_c = pack(input.paint_color_alt);
                const double t0 = glfwGetTime();
                static std::vector<uint32_t> island;
                static std::vector<float> value, edge;
                // World size of a screen pixel at the clicked spot: the brush ring there.
                const float wpp = 2.0f * camera.half_height_at(camera.view_depth(Vec3{a[0], a[1], a[2]}))
                                / (float)win_h;
                // Hardness = the edge against the walls; at 0 the fade is one brush radius
                // wide (the bucket's own slot: live_brush_slot() is BUCKET while armed).
                const float feather = feathered ? (1.0f - input.brush_hardness) * input.brush_size * wpp : 0.0f;
                // Repeat fills of one island with the same paint creep the edge further
                // across the walls, half a band per click. A repeat = the undo stack's top
                // is still the fill recorded at that depth, so Ctrl+Z steps the creep back
                // with it, and any other edit in between starts it over.
                // A run of repeats reuses its first fill's walls and island (the chain):
                // with paint as walls, that fill's own soft edge would otherwise read as
                // new walls, or new floor, on the next click and shift them.
                struct BucketRepeat {
                    uint32_t entity, chain, main_c, second_c;
                    int pattern; bool erase;
                    size_t depth; size_t top_n; uint32_t top_v, top_c;
                    int n;
                };
                static std::vector<BucketRepeat> repeats;
                static std::vector<float> chain_block;        // the chain's walls, per vertex
                static std::vector<uint32_t> chain_island;    // its flood, sorted
                static uint32_t chain_id = 0;
                BucketRepeat rep{scene.active_entity().id, 0, main_c, second_c, pattern,
                                 input.bucket_erase, 0, 0, 0, 0, 0};
                bool repeat = false;
                {
                    UndoStack& us = scene.active_undo();
                    while (!repeats.empty() && repeats.back().depth > us.undo_depth()) repeats.pop_back();
                    const UndoEntry* top = us.peek_undo();
                    if (!repeats.empty() && top && top->kind == UndoEntry::Kind::PAINT && !top->verts.empty()) {
                        const BucketRepeat& r = repeats.back();
                        repeat = r.depth == us.undo_depth() && r.top_n == top->verts.size()
                              && r.top_v == top->verts[0] && r.top_c == top->new_color[0]
                              && r.entity == rep.entity && r.chain == chain_id
                              && r.main_c == rep.main_c && r.second_c == rep.second_c
                              && r.pattern == rep.pattern && r.erase == rep.erase
                              && chain_block.size() == pm.vertex_count()
                              && std::binary_search(chain_island.begin(), chain_island.end(), seed);
                        if (repeat) rep.n = r.n + 1;
                    }
                }
                if (!repeat) {
                    // Walls: the mask, plus (Stop at paint) paint unlike the clicked colour,
                    // read off what the viewport shows, every visible layer.
                    static std::vector<uint32_t> shown;
                    if (input.bucket_paint_walls) {
                        shown.resize(pm.vertex_count());
                        paint_composite(pm.paint_layers, pm.color, pm.vertex_count(), shown.data());
                    }
                    constexpr float kPaintTolerance = 0.15f;   // largest channel difference that walls
                    paint_bucket_walls(pm, input.bucket_paint_walls ? shown.data() : nullptr, seed,
                                       kPaintTolerance, chain_block);
                    paint_bucket_island(pm, chain_block.data(), seed, chain_island);
                    std::sort(chain_island.begin(), chain_island.end());
                    chain_id++;
                }
                rep.chain = chain_id;
                island = chain_island;
                paint_bucket_edge(pm, chain_block.data(), island, input.brush_hardness, feather,
                                  0.5f * feather * (float)rep.n, edge);
                const float* cv = nullptr;
                if (gradient) {
                    paint_cover_gradient(pm, island, a, b, value);
                    cv = value.data();
                } else if (!solid) {
                    // The click's frame: its normal, and the camera's up laid onto the
                    // surface there, so Rotate 0 reads upright from where you look.
                    // (surface_at(A) ran last on this path, so click_n is A's.)
                    Vec3 n{click_n[0], click_n[1], click_n[2]};
                    Vec3 cam_fwd = (camera.target - camera.get_position()).normalized();
                    Vec3 cam_right = cam_fwd.cross(Vec3{0, 1, 0}).normalized();
                    Vec3 cam_up = cam_right.cross(cam_fwd).normalized();
                    Vec3 up = cam_up - n * n.dot(cam_up);
                    if (up.length() < 1e-3f) up = cam_right - n * n.dot(cam_right);
                    up = up.normalized();
                    Vec3 right = up.cross(n).normalized();
                    const float frame[9] = {right.x, right.y, right.z, up.x, up.y, up.z, n.x, n.y, n.z};
                    // Size = one tile across the brush circle at the clicked spot, so
                    // the ring shows the repeat before you click.
                    const AlphaEntry& pe = fill_patterns().get(pattern);
                    paint_cover_triplanar(pm, island, pe.data.data(), pe.w, pe.h,
                                          2.0f * input.brush_size * wpp,
                                          input.fill_angle * 3.14159265f / 180.0f, a, frame, value);
                    cv = value.data();
                }
                UndoEntry e;
                e.kind = UndoEntry::Kind::PAINT;
                e.paint_layer_id = paint_layers_active_id(pm.paint_layers);
                paint_bucket_apply(pm, island, main_c, second_c, cv, edge.data(), input.bucket_erase,
                                   input.brush_strength, e.verts, e.old_color, e.new_color);
                const double ms = (glfwGetTime() - t0) * 1000.0;
                std::printf("[bucket] seed %u, %s, hardness %.2f, repeat %d%s: reached %zu / %u verts, changed %zu, %.1f ms%s\n",
                            seed, fill_patterns().get(solid ? kFillSolid : pattern).name.c_str(),
                            input.brush_hardness, rep.n, input.bucket_paint_walls ? ", stops at paint" : "",
                            island.size(), pm.vertex_count(), e.verts.size(), ms,
                            input.bucket_erase ? " (erase)" : "");
                if (island.empty()) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "That spot is masked");
                    input.notification_timer = 1.5f;
                } else if (!e.verts.empty()) {
                    rep.top_n = e.verts.size(); rep.top_v = e.verts[0]; rep.top_c = e.new_color[0];
                    scene.sync_color_partial_entity(scene.active_entity().id, e.verts);
                    scene.active_undo().push(std::move(e));
                    rep.depth = scene.active_undo().undo_depth();
                    repeats.push_back(rep);
                    if (repeats.size() > 64) repeats.erase(repeats.begin());
                }
            }
        }

        // Debug helper: print undo stack top after key operations
        auto print_undo_top = [&](const char* tag) {
            UndoStack& undo_stack = scene.active_undo();
            const UndoEntry* e = undo_stack.peek_undo();
            if (e) {
                if (e->kind == UndoEntry::Kind::PROJECTION)
                    std::printf("[undo-trace][%s] depth=%zu top=PROJECTION target_level=%d\n",
                                tag, undo_stack.undo_depth(), e->target_level);
                else if (e->kind == UndoEntry::Kind::LEVEL)
                    std::printf("[undo-trace][%s] depth=%zu top=LEVEL from=%d to=%d proj=%d\n",
                                tag, undo_stack.undo_depth(), e->from_level, e->to_level,
                                (int)!e->before.empty());
                else if (e->kind == UndoEntry::Kind::MASK)
                    std::printf("[undo-trace][%s] depth=%zu top=MASK verts=%zu\n",
                                tag, undo_stack.undo_depth(), e->verts.size());
                else if (e->kind == UndoEntry::Kind::PAINT)
                    std::printf("[undo-trace][%s] depth=%zu top=PAINT verts=%zu\n",
                                tag, undo_stack.undo_depth(), e->verts.size());
                else
                    std::printf("[undo-trace][%s] depth=%zu top=STROKE level=%d targets_base=%d disp_index=%d\n",
                                tag, undo_stack.undo_depth(), e->level, (int)e->targets_base, e->disp_index);
            } else {
                std::printf("[undo-trace][%s] depth=%zu (empty)\n", tag, undo_stack.undo_depth());
            }
        };

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

        auto refresh_active_gpu_residency = [&]() {
            MeshEntity& ent = scene.active_entity();
            ent.multires_gpu.supported = compute.supported;
            ent.multires_gpu.dev       = &compute.gpu_dev;
            // Residency dedupe: right after a GPU cascade replay, the target
            // level's disp/frames still sit in the replay's layer scratch —
            // upload_level copies them device-local instead of re-sending from
            // CPU. The handoff is consumed here (one switch only) so no later
            // refresh can copy stale scratch.
            const gpu::Buffer* dsrc = nullptr;
            const gpu::Buffer* fsrc = nullptr;
            if (compute.cascade_out_valid &&
                compute.cascade_out_passes ==
                    ent.multires.current_level - ent.multires.base_level) {
                dsrc = &compute.cascade_disp_ssbo;
                fsrc = &compute.cascade_frames_ssbo;
                std::printf("[residency] disp/frames via gpu copy\n");
            }
            ent.multires_gpu.upload_level(ent.multires, ent.multires.current_level, dsrc, fsrc);
            compute.cascade_out_valid = false;
        };

        // Prepare + capture the rescue snapshot, shared by '/' and 'J'. Frees the
        // undo history, bakes the scene down to the triangle cap, then copies the
        // stacks. Lives here rather than outside the loop because it needs
        // refresh_active_gpu_residency, which is itself a per-frame lambda.
        auto take_rescue = [&](const char* op) {
            int baked = snapshot_prepare(scene, compute);
            if (baked < 0) { rescue.clear(); return; }   // over cap, unprotected
            if (baked > 0) {
                // snapshot_prepare rebuilt the scene through load_entities to
                // re-cascade the baked stacks, so every cached pointer is stale.
                mesh     = &scene.active_mesh();
                multires = &scene.active_multires();
                scene.refresh_mirror_map();
                scene.sync();
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
            }
            rescue.levels_baked = baked;
            snapshot_capture(rescue, scene, op);
            rescue_shown = -1;
        };
        // Arm the renderer's one-shot geometry handoff for the sync following a
        // GPU cascade replay: the working VBOs/EBO then fill by GPU→GPU copy from
        // the replay scratch (positions/normals) and the VRAM level tables
        // (indices). No-op when the replay didn't run (CPU fallback).
        auto arm_geometry_handoff = [&](const Mesh& m) {
            if (!compute.cascade_out_valid || !compute.cascade_out_pos) return;
            if (compute.cascade_out_passes <= 0 ||
                compute.cascade_out_passes >= (int)compute.cascade_levels.size()) return;
            renderer.set_geometry_source(compute.cascade_out_pos, &compute.cascade_norm_ssbo,
                                         &compute.cascade_levels[compute.cascade_out_passes].indices,
                                         m.vertex_count(), (uint32_t)m.indices.size());
        };

        // Multires level switch (D / Shift-D post-lock). Recorded on the undo
        // timeline as a LEVEL entry so Ctrl-Z retraces the literal action sequence
        // (…draw, subd-up, draw, subd-down…) with the view level following along.
        if (input.level_switch_delta != 0 && swallow_level_switch) {
            // Drop input queued while the previous switch froze the frame —
            // keymashing D must not stack further multi-second builds.
            input.level_switch_delta = 0;
        }
        swallow_level_switch = false;
        if (input.level_switch_delta != 0 && app_state != AppState::IDLE) {
            // Mid-stroke D: the cascade would rebuild the mesh under the live
            // brush (pen-down FBO cache, GPU accumulators, dirty lists all
            // reference the old buffers) — crash. Level switch is idle-only.
            input.level_switch_delta = 0;
        }
        if (input.level_switch_delta != 0) {
            int delta = input.level_switch_delta;
            input.level_switch_delta = 0;
            const int from   = multires->current_level;
            const int target = from + delta;
            // Subdivision guard: before going up a level, predict the largest
            // resulting GPU buffer against the limits the device actually granted.
            // Subdividing quadruples tris; the index SSBO and CSR adjacency both
            // weigh tris x 12 bytes and are the biggest allocations. Refusing here
            // beats attempting the build and losing the device (WebGPU treats an
            // over-limit buffer as a validation error -> device loss).
            if (delta > 0) {
                const uint64_t tris_next  = (uint64_t)mesh->tri_count() * 4ull;
                const uint64_t bytes_next = tris_next * 12ull;
                const gpu::DeviceLimits dl = gpu::device_limits();
                const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                                      ? dl.max_buffer_size : dl.max_storage_binding_size;
                if (bytes_next > budget) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Subdivision would exceed GPU limits (%.1fM tris)",
                                  (double)tris_next / 1e6);
                    input.notification_timer = 4.0f;
                    std::printf("[multires] refused L%d -> L%d: %.1fM tris needs %llu MB, device grants %llu MB\n",
                                from, target, (double)tris_next / 1e6,
                                (unsigned long long)(bytes_next >> 20),
                                (unsigned long long)(budget >> 20));
                    delta = 0;
                }
            }
            const double switch_t0 = glfwGetTime();
            if (delta != 0 && target >= multires->base_level && target <= max_level) {
                scene.materialize_active_cpu();  // 2b: projection/cascade read disp/base
                UndoEntry lvl_e;
                lvl_e.kind       = UndoEntry::Kind::LEVEL;
                lvl_e.from_level = from;
                lvl_e.to_level   = target;
                if (delta < 0) {
                    // Auto-project before descending: bake detail from L_max down to
                    // target. The snapshot rides on the LEVEL entry — undo restores it,
                    // redo replays the projection.
                    const int L_max = multires->base_level + (int)multires->disp.size();
                    capture_projection_snapshot(*multires, target, lvl_e.before);
                    ProjectionStats ps = project_down_to_level(*multires, target);
                    std::printf("[project] auto L%d -> L%d in %.2f ms\n", L_max, target, ps.elapsed_ms);
                    report_projection_check(ps);
                }
                scene.active_undo().push(std::move(lvl_e));
                multires->current_level = target;
                multires_sync_paint(*multires, *mesh);
                if (scene.alive_count() <= 1) {
                    // Paint/mask ride the cascade (finest-level planes) — no
                    // save/restore needed across the rebuild.
                    cascade_to_level(*multires, *mesh, target, &compute);
                    arm_geometry_handoff(*mesh);
                    scene.refresh_mirror_map();
                    scene.sync();  // rebinds active: rebuilds adjacency from new topology
                } else {
                    Mesh solo;
                    cascade_to_level(*multires, solo, target, &compute);
                    arm_geometry_handoff(solo);
                    scene.splice_active(solo);  // splice_active marks topo dirty
                    scene.refresh_mirror_map();
                }
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
                std::printf("[multires] switched to level %d (%u verts, %u tris) in %.1f ms\n",
                            target, mesh->vertex_count(), mesh->tri_count(),
                            (glfwGetTime() - switch_t0) * 1000.0);
                print_undo_top("level-switch");
                // A switch slow enough to freeze the frame has stale input queued
                // behind it (GLFW delivers it all in the next poll) — swallow it.
                if (glfwGetTime() - switch_t0 > 0.15)
                    swallow_level_switch = true;
            }
        }

        // Delete the highest subdivision level (burger menu -> Y/N confirm).
        // Destructive and deliberately NOT undoable: the whole point is to hand
        // the top layer's memory back, and an undo entry holding it would give
        // exactly none of that back. The history is cleared instead — its STROKE
        // entries carry vert ids from a level that no longer exists, and a LEVEL
        // entry replaying upward would resurrect the level as a flat zero layer.
        if (input.drop_level_requested) {
            input.drop_level_requested = false;
            const int L_max = multires->base_level + (int)multires->disp.size();
            if (!multires->locked || multires->disp.empty()) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "No subdivision level to delete");
                input.notification_timer = 2.5f;
            } else if (app_state != AppState::IDLE) {
                // Same reason the level switch is idle-only: the cascade would
                // rebuild the mesh out from under a live stroke.
                std::snprintf(input.notification, sizeof(input.notification),
                              "Finish the stroke first");
                input.notification_timer = 2.0f;
            } else {
                const double drop_t0 = glfwGetTime();
                scene.mark_edited();
                const int target = L_max - 1;
                scene.materialize_active_cpu();  // projection/cascade read disp/base

                // Bake first, then drop: the same inverse-Loop projection a level-down
                // does, so the form survives into the layer below and only the
                // residual detail dies with the layer.
                ProjectionStats ps = project_down_to_level(*multires, target);
                report_projection_check(ps);
                // Only forced down if the view was ON the layer being removed —
                // deleting L5 while editing at L3 must leave you at L3.
                if (multires->current_level > target) multires->current_level = target;
                const int view = multires->current_level;
                multires_sync_paint(*multires, *mesh);   // fold paint in BEFORE the planes shrink
                multires_drop_top_level(*multires);

                scene.active_undo().clear(&compute);
                xforms.clear();          // history wiped means history wiped
                // Grow-only GPU mirror still sized for the removed level; drop it
                // so the refresh below re-allocates at the new (smaller) level.
                scene.active_entity().multires_gpu.cleanup();

                if (scene.alive_count() <= 1) {
                    cascade_to_level(*multires, *mesh, view, &compute);
                    arm_geometry_handoff(*mesh);
                    scene.refresh_mirror_map();
                    scene.sync();
                } else {
                    Mesh solo;
                    cascade_to_level(*multires, solo, view, &compute);
                    arm_geometry_handoff(solo);
                    scene.splice_active(solo);
                    scene.refresh_mirror_map();
                }
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Deleted subdiv level %d - max is now L%d (%.2fM tris here)",
                              L_max, target, (double)mesh->tri_count() / 1e6);
                input.notification_timer = 3.0f;
                std::printf("[multires] delete L%d: projected in %.2f ms, max now L%d, "
                            "viewing L%d (%u verts, %u tris), total %.1f ms\n",
                            L_max, ps.elapsed_ms, target, view, mesh->vertex_count(),
                            mesh->tri_count(), (glfwGetTime() - drop_t0) * 1000.0);
            }
        }

        // Handle focus request (F key)
        if (input.focus_requested) {
            input.focus_requested = false;
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            if (brush_stroke.last_stroke_valid) {
                camera.set_target(brush_stroke.last_stroke_pos);
            } else {
                camera.set_target(mesh_center);
            }
            camera.distance = std::max(camera.distance * 0.675f, mesh_radius * 0.05f);
            screen_buffers_dirty = true;
        }

        // Snap views (F1/F2/F3)
        if (input.snap_view_requested != InputState::SnapView::NONE) {
            mesh->compute_bounding_sphere(mesh_center, mesh_radius);
            camera.set_target(mesh_center);
            camera.distance = mesh_radius * 2.5f;
            switch (input.snap_view_requested) {
                case InputState::SnapView::FRONT:
                    camera.yaw = 0.0f;
                    camera.pitch = 0.0f;
                    break;
                case InputState::SnapView::SIDE:
                    camera.yaw = (float)(M_PI / 2.0);
                    camera.pitch = 0.0f;
                    break;
                case InputState::SnapView::TOP:
                    camera.yaw = 0.0f;
                    camera.pitch = (float)(M_PI / 2.0 - 0.001);
                    break;
                default: break;
            }
            input.snap_view_requested = InputState::SnapView::NONE;
            screen_buffers_dirty = true;
        }

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

        // ---- Remesh execution ----
        // A remesh runs inside this one frame, so the "Remeshing..." panel would never
        // reach the screen: the first frame only raises remesh_in_progress (the panel
        // draws and presents), the next one does the work.
        static bool remesh_shown = false;
        if (input.remesh_requested && !remesh_shown) {
            remesh_shown = true;
            input.remesh_in_progress = true;
        } else if (input.remesh_requested) {
            remesh_shown = false;
            input.remesh_requested = false;
            input.remesh_in_progress = true;

            scene.materialize_active_cpu();  // 2b: remesh reads the live surface (mesh.pos)

            // Adaptive-remesh guard (density spec Q2): a red-heavy field at a
            // small fine-mult can explode tri counts the same way keymashed
            // subdivision does. Same predicted-size check, same refusal toast.
            bool remesh_refused = false;
            // The detail slider multiplies the count on either path, so it rides
            // the same guard — a x4 on a big mesh is the uniform version of the
            // red-heavy field this was written for.
            const float detail = input.remesh_detail;
            if (!mesh->density.empty() || detail > 1.0f) {
                const uint64_t base_pred = mesh->density.empty()
                    ? (uint64_t)mesh->tri_count()
                    : predict_adaptive_tris(*mesh, 0.0f, input.density_coarse_mult,
                                            input.density_fine_mult);
                const uint64_t tris_pred = (uint64_t)((double)base_pred * std::max(1.0f, detail));
                const uint64_t bytes_pred = tris_pred * 12ull;  // index/CSR SSBOs
                const gpu::DeviceLimits dl = gpu::device_limits();
                const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                                      ? dl.max_buffer_size : dl.max_storage_binding_size;
                if (bytes_pred > budget) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Remesh would exceed GPU limits (%.1fM tris)",
                                  (double)tris_pred / 1e6);
                    input.notification_timer = 4.0f;
                    std::printf("[remesh] refused: %.1fM predicted tris needs %llu MB, device grants %llu MB\n",
                                (double)tris_pred / 1e6,
                                (unsigned long long)(bytes_pred >> 20),
                                (unsigned long long)(budget >> 20));
                    remesh_refused = true;
                }
            }

            // Rescue copy before the topology goes. A second remesh replaces the
            // first's snapshot — one step back means the LAST remesh — but a
            // remesh chained off a merge leaves the merge's snapshot alone.
            if (!remesh_refused && !remesh_chained_from_merge) take_rescue("remesh");

            // Destructive remesh breaks topology mirror — switch to spatial
            // mode afterward. Mirror setting (input.mirror_x) is preserved.
            RemeshResult result;
            if (!remesh_refused)
                result = perform_remesh(*mesh, *multires, 0.0f, 10,
                                        compute.supported ? &compute : nullptr,
                                        input.density_coarse_mult,
                                        input.density_fine_mult,
                                        input.remesh_keep_detail,
                                        input.remesh_detail);

            if (result.success) {
                scene.mark_edited();
                mesh->mask.clear();
                // perform_remesh rebuilt the ACTIVE entity's mesh + multires base
                // in place. Leave every other entity untouched — only the active
                // one was remeshed. Its topology is fresh (base level 0), so reset
                // its subdiv level, refresh its mirror map, and re-sync the working
                // set. Spatial mirror mode: destructive remesh breaks topology mirror.
                scene.set_mirror_topology(false);
                scene.active_entity().subdiv_level = 0;
                // Remesh is single-entity: it only rebuilt the active mesh, so a
                // leftover multi-selection is stale. Collapse it to the active
                // entity now — clears the deselected tint and keeps downstream
                // ops (edit-mode entry, merge) from acting on a mixed set.
                scene.collapse_selection_to_active();
                scene.refresh_mirror_map();
                scene.sync();
                mesh = &scene.active_mesh();
                multires = &scene.active_multires();
                // Same resync as the voxel merge: the GPU copy of the base still
                // holds the PRE-remesh mesh, and strokes at the base level are
                // diffed against it and later materialized from it. Without this,
                // every stroke after a remesh landed as old-mesh positions at the
                // new mesh's ids — invisible until the next level change rebuilt
                // the model from them (2026-09-25).
                refresh_active_gpu_residency();
                screen_buffers_dirty = true;
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                if (result.selected_tris > 0) { scene.active_undo().clear(&compute); xforms.clear(); }
                std::snprintf(input.notification, sizeof(input.notification),
                              "Remesh: %u sel, %u/%u -> %u/%u v/t (spatial mirror)",
                              result.selected_tris,
                              result.old_verts, result.old_tris,
                              result.new_verts, result.new_tris);
                input.notification_timer = 4.0f;
            } else if (!remesh_refused) {
                // Nothing changed, so there is nothing this remesh can rescue —
                // but a chained one must leave the merge's snapshot standing.
                if (!remesh_chained_from_merge) rescue.clear();
                std::printf("[remesh] FAILED: %s\n", result.error.c_str());
                std::snprintf(input.notification, sizeof(input.notification),
                              "Remesh FAILED — check console");
                input.notification_timer = 4.0f;
            }

            remesh_chained_from_merge = false;
            input.remesh_in_progress = false;
        }

        // ---- Voxel merge execution (SDF join-for-print) ----
        // Kick off a merge: build the job (gather/grid/alloc/compile), then let the
        // per-frame advance below tick it to completion. The merge spans frames so the
        // window stays responsive and the progress HUD animates (the winding-sign pass
        // alone is seconds at R>=128).
        if (input.voxel_merge_requested) {
            input.voxel_merge_requested = false;
            if (!compute.supported) {
                std::snprintf(input.notification, sizeof(input.notification),
                              "SDF remesh needs GPU compute (unavailable)");
                input.notification_timer = 4.0f;
            } else if (!vmerge_job) {
                take_rescue("SDF remesh");
                scene.materialize_active_cpu();  // 2b: merge reads the live surface (mesh.pos)
                if (const char* dump = std::getenv("CHISEL_MERGE_SRC_DUMP"))   // dev: pre-merge mesh
                    scene.active_mesh().export_obj(dump);
                vmerge_job = voxel_merge_begin(scene, compute,
                                               input.voxel_merge_resolution,
                                               input.voxel_merge_mirror,
                                               input.voxel_merge_surface_nets,
                                               input.voxel_merge_subtract,
                                               input.remesh_keep_detail);
                input.voxel_merge_in_progress = true;
            }
        }

        // Advance an in-flight merge by one budgeted step.
        if (vmerge_job) {
            VoxelMergeResult vm;
            VoxelMergeStatus st = voxel_merge_tick(scene, compute, *vmerge_job, vm);
            if (st != VoxelMergeStatus::Working) {
                voxel_merge_destroy(vmerge_job);
                vmerge_job = nullptr;
                input.voxel_merge_in_progress = false;

                if (st == VoxelMergeStatus::Done && vm.success) {
                    scene.mark_edited();
                    // Mirror merge yields a tessellation-symmetric mesh → topology
                    // mirror gives an exact partner map. Faithful merge is generic
                    // geometry → spatial mirror. Either way refresh from the fresh mesh.
                    scene.set_mirror_topology(input.voxel_merge_mirror);
                    scene.refresh_mirror_map();
                    scene.sync();
                    mesh = &scene.active_mesh();
                    multires = &scene.active_multires();
                    // The merge replaces the active entity wholesale; resync the
                    // GPU-residency mirror to the fresh mesh's current level so the
                    // Phase-2 diff/apply shaders read a valid disp/base layer.
                    refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    screen_buffers_dirty = true;
                    brush_stroke.vertex_count = 0;
                    brush_stroke.phase = StrokePhase::NONE;
                    app_state = AppState::IDLE;
                    bool watertight = (vm.boundary_edges == 0 && vm.nonmanifold_edges == 0);
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "SDF remesh %u -> %u v / %u t (R=%u, %.0f ms) | %s: %u comp, %u bnd, %u nonmf",
                                  vm.in_entities, vm.out_verts, vm.out_tris,
                                  vm.R, vm.elapsed_ms,
                                  watertight ? "watertight" : "NOT watertight",
                                  vm.components, vm.boundary_edges, vm.nonmanifold_edges);
                    input.notification_timer = 6.0f;
                    std::printf("[voxel-merge] %s\n", input.notification);
                    // Dev hook: CHISEL_MERGE_DUMP=<path.obj> writes the result for offline checks.
                    if (const char* dump = std::getenv("CHISEL_MERGE_DUMP"))
                        mesh->export_obj(dump);
                    // Chain the adaptive remesh: the merge output is uniform-dense
                    // MC soup; when it carries a painted density field the user
                    // wants the topology to follow it right away (D in the merge
                    // dialog opts out). Runs through the normal remesh block next
                    // frame — same GPU-limit guard, same notification.
                    if (input.voxel_merge_adaptive && !mesh->density.empty()) {
                        std::printf("[voxel-merge] density field present -> chaining adaptive remesh\n");
                        input.remesh_requested = true;
                        remesh_chained_from_merge = true;
                    }
                } else {
                    rescue.clear();   // scene untouched, nothing to revert to
                    std::printf("[voxel-merge] FAILED: %s\n", vm.error.c_str());
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "SDF remesh failed: %.200s", vm.error.c_str());
                    input.notification_timer = 4.0f;
                }
            }
        }

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

        // ---- Retopo mode: stroke capture, its own undo, Build ----
        // Strokes are sampled off the pen-down plane cache like a brush anchor, so
        // capture never reads the GPU back. Retopo never edits the sculpt, so the
        // cache stays good for the whole stroke.
        {
            const bool retopo_on = input.interaction_mode == InputState::InteractionMode::RETOPO;
            // Smooth (Shift / double Shift) works on the built quads, not the strokes.
            // Decided at the press: letting go of Shift mid-drag doesn't start a stroke.
            static bool retopo_smoothing = false;
            static bool retopo_grabbing = false;
            const bool retopo_press = retopo_on && input.drag_mode == InputState::DragMode::RETOPO;
            if (retopo_press && !retopo.drawing
                && (retopo_smoothing || input.is_smooth_active())) {
                if (!retopo_smoothing) {
                    if (!retopo.built) {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Nothing to smooth: Build first (Space)");
                        input.notification_timer = 2.0f;
                        input.drag_mode = InputState::DragMode::NONE;
                    } else {
                        scene.materialize_active_cpu();   // the snap re-reads mesh.pos
                        retopo_edit_begin(retopo, *mesh);
                        retopo_smoothing = true;
                    }
                }
                if (retopo_smoothing)
                    retopo_smooth_dab(retopo, camera, win_w, win_h, (float)input.mouse_x,
                                      // Fixed for now: Size is the only knob (user).
                                      (float)input.mouse_y, input.brush_size,
                                      0.5f, input.mirror_x);
            } else if (retopo_press && !retopo.drawing
                       && (retopo_grabbing || input.retopo_tool == InputState::RetopoTool::GRAB)) {
                // Grab: the press takes what's under the ring, the drag carries it.
                static float grab_x = 0.0f, grab_y = 0.0f;
                if (!retopo_grabbing) {
                    if (!retopo.built) {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Nothing to grab: Build first (Space)");
                        input.notification_timer = 2.0f;
                        input.drag_mode = InputState::DragMode::NONE;
                    } else {
                        scene.materialize_active_cpu();
                        retopo_edit_begin(retopo, *mesh);
                        grab_x = (float)input.mouse_x;
                        grab_y = (float)input.mouse_y;
                        retopo_grab_begin(retopo, camera, win_w, win_h, grab_x, grab_y,
                                          input.brush_size, input.mirror_x);
                        retopo_grabbing = true;
                    }
                }
                if (retopo_grabbing)
                    retopo_grab_drag(retopo, camera, win_w, win_h,
                                     (float)input.mouse_x - grab_x, (float)input.mouse_y - grab_y);
            } else if (retopo_press && !screen_buffers_dirty) {
                const float mx = (float)input.mouse_x, my = (float)input.mouse_y;
                if (!retopo.drawing) {
                    retopo.drawing = true;
                    retopo.live = RetopoStroke{};
                    retopo.last_sx = mx - 1e3f;   // the press point always lands
                }
                float d = 0.0f;
                if (std::hypot(mx - retopo.last_sx, my - retopo.last_sy) >= 3.0f
                    && renderer.sample_depth_bilinear(mx, my, &d, 0.0f) && d < 500.0f) {
                    Vec3 cam_pos = camera.get_position();
                    Vec3 fwd = (camera.target - cam_pos).normalized();
                    Vec3 right = fwd.cross(Vec3(0, 1, 0)).normalized();
                    Vec3 up = right.cross(fwd).normalized();
                    const float half_h = camera.half_height_at(d);
                    const float half_w = half_h * ((float)win_w / (float)win_h);
                    const float ndc_x = 2.0f * mx / (float)win_w - 1.0f;
                    const float ndc_y = 1.0f - 2.0f * my / (float)win_h;
                    Vec3 p = cam_pos + fwd * d + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                    float nt[3];
                    Vec3 n = renderer.sample_normal((int)mx, (int)my, nt)
                           ? Vec3(nt[0], nt[1], nt[2]).normalized() : fwd * -1.0f;
                    // Band width = the brush ring, in world units where the pen is.
                    // Pen pressure thins it, so a stroke can taper.
                    const float world_per_px = 2.0f * half_h / (float)win_h;
                    // Same pen test as the brush: plugged in isn't enough, the pen has
                    // to be what's driving (a resting pen reads ~0 forever).
                    const bool pen = tablet.available()
                                  && input.active_profile == InputProfile::TABLET;
                    const float pr = pen ? std::max(0.2f, tablet.pressure()) : 1.0f;
                    retopo.live.pts.push_back(p);
                    retopo.live.nrm.push_back(n);
                    retopo.live.half_w.push_back(input.brush_size * world_per_px * pr);
                    retopo.last_sx = mx;
                    retopo.last_sy = my;
                }
            } else if (retopo.drawing) {
                // Pen up (or the mode changed under it).
                const bool kept = retopo.finish_live();
                if (kept) {
                    std::snprintf(input.notification, sizeof(input.notification), "%s - Space builds",
                                  retopo.strokes.back().name.c_str());
                    input.notification_timer = 1.5f;
                }
            }
            if (!retopo_press) retopo_smoothing = retopo_grabbing = false;
            // Ctrl+Z / Ctrl+Shift+Z walk the stroke list here, never the sculpt's
            // history: retopo doesn't touch the sculpt.
            if (retopo_on && (input.undo_requested || input.redo_requested)) {
                if (input.undo_requested) retopo.undo();
                else                      retopo.redo_one();
                input.undo_requested = input.redo_requested = false;
            }
            if (input.retopo_build_requested) {
                input.retopo_build_requested = false;
                scene.materialize_active_cpu();   // Build snaps onto mesh.pos
                const double ms = retopo_build(retopo, *mesh, input.mirror_x);
                if (retopo.built)
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Built %u quads (%.0f ms)", retopo.result.quad_count(), ms);
                else
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Nothing to build: sketch a stroke first");
                input.notification_timer = 2.5f;
            }
            if (input.retopo_export_requested) {
                input.retopo_export_requested = false;
                // No dialog: next to the project (or home), named after it. The
                // low-poly is a by-product to take elsewhere, not a document.
#if defined(__EMSCRIPTEN__)
                const std::string name = "retopo.obj", path = "/" + name;
#else
                std::string stem = current_project_path;
                if (stem.empty()) {
                    const char* home = std::getenv("HOME");
                    stem = std::string(home ? home : ".") + "/chisel";
                } else {
                    const auto dot = stem.find_last_of('.');
                    if (dot != std::string::npos && stem.find_last_of('/') < dot) stem.resize(dot);
                }
                const std::string path = stem + "-retopo.obj", name = path;
#endif
                if (retopo_export_obj(retopo.result, path.c_str())) {
#if defined(__EMSCRIPTEN__)
                    web_download_file(path.c_str(), name.c_str());
#endif
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Exported: %.400s", name.c_str());
                } else {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  retopo.built ? "Export failed: %.400s" : "Build first%.0s",
                                  name.c_str());
                }
                input.notification_timer = 3.0f;
            }
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
            // Camera controls
            if (input.drag_mode == InputState::DragMode::ORBIT) {
                camera.orbit(dx, dy);
                screen_buffers_dirty = true;
                wrap_cursor(window, input, win_w, win_h);
            }
            if (input.drag_mode == InputState::DragMode::PAN || input.mouse3_down) {
                camera.pan(dx, dy, win_w, win_h);
                if (dx != 0.0f || dy != 0.0f) screen_buffers_dirty = true;
                wrap_cursor(window, input, win_w, win_h);
            }
            if (input.zoom_modifier_held() && input.mouse2_down) {
                camera.zoom(-dy * 0.05f);
                if (dy != 0.0f) screen_buffers_dirty = true;
                wrap_cursor(window, input, win_w, win_h);
            }

            // Object move: drag selected meshes in view-plane
            if (input.drag_mode == InputState::DragMode::MOVE_OBJECT) {
                // Framing-relative, matching Camera::pan — see the note there for why
                // this is not keyed off camera.distance.
                float scale = camera.half_height() * 0.0048284f;
                Vec3 cam_pos = camera.get_position();
                Vec3 fwd = (camera.target - cam_pos).normalized();
                Vec3 world_up = {0, 1, 0};
                Vec3 right = fwd.cross(world_up).normalized();
                Vec3 up = right.cross(fwd).normalized();
                // Signs chosen so the mesh tracks the cursor 1:1: +dx (cursor right)
                // moves right; -dy (GLFW y grows downward, so cursor up is dy<0) moves
                // up. Both selectmove specs agree on this.
                Vec3 delta = right * (dx * scale) + up * (-dy * scale);

                if (delta.x != 0.0f || delta.y != 0.0f || delta.z != 0.0f) {
                    // 2c-iii: if the active entity was left stale by the flipped
                    // pen-up path, its mesh.pos must be fresh before move_mesh /
                    // compute_bounding_sphere read it (no-op when not dirty).
                    scene.materialize_active_cpu();
                    for (uint32_t sel_id : scene.selected_ids()) {
                        MeshEntity* e = scene.find_entity(sel_id);
                        if (!e) continue;
                        uint32_t vc = e->mesh.vertex_count();

                        // X-axis behaviour depends on WHAT sits on the mirror plane.
                        // Bounding centre at x=0 alone can't tell a centered single
                        // piece from a symmetrized pair — the pair's centre is also
                        // at 0 — so also test whether the geometry is continuous
                        // across the plane (some triangle touches/crosses x=0):
                        // - Centered single (spans the seam): lock X so it stays
                        //   centered, regardless of the symmetry toggle.
                        // - Symmetrized pair (two disjoint lobes, nothing crossing):
                        //   with symmetry on, move the lobes as exact mirrors
                        //   (spread/converge); with symmetry off, translate freely.
                        // - Off-centre pieces always translate freely on all axes.
                        Vec3 c; float r;
                        e->mesh.compute_bounding_sphere(c, r);
                        float rr = (r > 0.0f) ? r : 1.0f;
                        bool centered = std::fabs(c.x) < 1e-3f * rr;
                        float seam_eps = 1e-4f * rr;

                        bool spans_seam = false;
                        if (centered) {
                            const Mesh& sm = e->mesh;
                            float band = 1e-3f * rr;
                            for (size_t t = 0; t + 2 < sm.indices.size() && !spans_seam; t += 3) {
                                bool pos = false, neg = false;
                                for (int k = 0; k < 3; k++) {
                                    float x = sm.pos_x[sm.indices[t + k]];
                                    if (x > band) pos = true;
                                    else if (x < -band) neg = true;
                                    else { pos = true; neg = true; }  // in the seam band
                                }
                                spans_seam = pos && neg;
                            }
                        }
                        bool lock_x = centered && spans_seam;
                        bool mirror_lobes = centered && !spans_seam && input.mirror_x;

                        // Record it. The X rule is latched on this entity's first
                        // frame of the drag and reused for the rest of it, so the
                        // undo replays the same split it was moved with even if a
                        // lobe drifts close enough to the seam to re-classify.
                        xforms.begin(ObjectXform::Kind::MOVE, scene);
                        ObjectXformTarget& xt = xforms.target(sel_id);
                        if (xt.delta.x == 0.0f && xt.delta.y == 0.0f && xt.delta.z == 0.0f) {
                            xt.lock_x       = lock_x;
                            xt.mirror_lobes = mirror_lobes;
                            xt.seam_eps     = seam_eps;
                        } else {
                            lock_x       = xt.lock_x;
                            mirror_lobes = xt.mirror_lobes;
                            seam_eps     = xt.seam_eps;
                        }
                        xt.delta += delta;

                        auto move_mesh = [&](Mesh& m) {
                            uint32_t n = m.vertex_count();
                            for (uint32_t v = 0; v < n; v++) {
                                float sx = delta.x;
                                if (lock_x) sx = 0.0f;
                                else if (mirror_lobes) {
                                    float x = m.pos_x[v];
                                    sx = (x >  seam_eps) ?  delta.x
                                       : (x < -seam_eps) ? -delta.x : 0.0f;
                                }
                                m.pos_x[v] += sx;
                                m.pos_y[v] += delta.y;
                                m.pos_z[v] += delta.z;
                            }
                        };
                        move_mesh(e->mesh);
                        e->mesh.invalidate_mirror_symmetry();   // see the scale block
                        // The base cage follows for EVERY moved entity, not just
                        // the active one. It used to be active-only, which left a
                        // non-active locked mesh with a moved surface over a base
                        // still at the old spot — invisible until you selected it
                        // and switched level, at which point the cascade snapped it
                        // back. Spin already did it this way; move and scale now
                        // agree, which is also what makes their undo an honest
                        // inverse rather than an over-correction.
                        if (e->multires.locked) move_mesh(e->multires.base);

                        std::vector<uint32_t> local_dirty(vc);
                        for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                        scene.sync_partial_entity(sel_id, local_dirty);
                    }
                    screen_buffers_dirty = true;
                }
                wrap_cursor(window, input, win_w, win_h);
            }

            // Object scale: RMB drag in SELECT mode scales selected meshes about
            // their shared centroid. Drag right grows, left shrinks (exponential
            // so the feel is uniform regardless of current size).
            if (input.drag_mode == InputState::DragMode::SCALE_OBJECT && dx != 0.0f) {
                const auto& sel = scene.selected_ids();
                if (!sel.empty()) {
                    scene.materialize_active_cpu();

                    // Pivot = centroid of selected bounding centres. Invariant under
                    // uniform scale about itself, so recomputing per frame is stable.
                    Vec3 pivot = {0, 0, 0};
                    uint32_t np = 0;
                    for (uint32_t sel_id : sel) {
                        MeshEntity* e = scene.find_entity(sel_id);
                        if (!e) continue;
                        Vec3 c; float r;
                        e->mesh.compute_bounding_sphere(c, r);
                        pivot.x += c.x; pivot.y += c.y; pivot.z += c.z;
                        np++;
                    }
                    if (np > 0) {
                        pivot.x /= np; pivot.y /= np; pivot.z /= np;
                        // With symmetry on, scale X about the mirror plane so the
                        // piece (and its -x twin) stay exact mirrors.
                        if (input.mirror_x) pivot.x = 0.0f;

                        float f = std::exp(dx * 0.005f);

                        // Record it, and latch the pivot on the drag's first
                        // frame for the LIVE scale too: a bounding centre is
                        // invariant under a scale about itself in exact
                        // arithmetic only, so letting it drift per frame would
                        // both creep the object and put the undo somewhere the
                        // forward pass never was.
                        const bool scale_first = !(xforms.gesture_open()
                            && xforms.gesture_kind() == ObjectXform::Kind::SCALE);
                        xforms.begin(ObjectXform::Kind::SCALE, scene);
                        if (scale_first) scale_pivot = pivot;
                        else             pivot       = scale_pivot;

                        auto scale_mesh = [&](Mesh& m) {
                            uint32_t n = m.vertex_count();
                            for (uint32_t v = 0; v < n; v++) {
                                m.pos_x[v] = pivot.x + (m.pos_x[v] - pivot.x) * f;
                                m.pos_y[v] = pivot.y + (m.pos_y[v] - pivot.y) * f;
                                m.pos_z[v] = pivot.z + (m.pos_z[v] - pivot.z) * f;
                            }
                        };

                        for (uint32_t sel_id : sel) {
                            MeshEntity* e = scene.find_entity(sel_id);
                            if (!e) continue;
                            uint32_t vc = e->mesh.vertex_count();
                            ObjectXformTarget& xt = xforms.target(sel_id);
                            xt.pivot   = pivot;
                            xt.factor *= f;
                            scale_mesh(e->mesh);
                            if (e->multires.locked) scale_mesh(e->multires.base);   // see move
                            // Scaling a symmetric piece about an off-plane pivot
                            // takes it off the mirror plane, and no topology changed,
                            // so nothing else would re-measure. See the stroke gate.
                            e->mesh.invalidate_mirror_symmetry();

                            std::vector<uint32_t> local_dirty(vc);
                            for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                            scene.sync_partial_entity(sel_id, local_dirty);
                        }
                        screen_buffers_dirty = true;
                    }
                }
                wrap_cursor(window, input, win_w, win_h);
            }

            // Object spin: Q/E in SELECT mode turn the selection about the VIEW
            // axis, so the turn follows what you are looking at rather than a world
            // axis — face the model from any angle and Q/E always read as
            // anticlockwise/clockwise on screen.
            //
            // The key is POLLED rather than driven off key events: events give you
            // the OS auto-repeat, which is a delay followed by discrete jumps, and
            // no rate you pick makes that feel continuous. Polling with a real dt is
            // smooth at any frame rate and cannot get stuck on a release the modal
            // gate swallowed. Constant rate, no ramp. Like move and scale the whole
            // press-to-release is recorded as ONE step on the object-transform
            // stack (object_xform.h), which Ctrl+Z reaches from Select mode.
            {
                const bool modal = input.quit_requested || input.remesh_confirm_pending
                                || input.voxel_merge_confirm_pending
                                || input.drop_level_confirm_pending
                                || input.reset_confirm_pending
                                || input.pose_clear_confirm_pending
                                || input.drop_confirm_pending
                                || input.export_dialog_active || input.import_dialog_active
                                || input.save_dialog_active
                                || input.keybind_menu_open || input.brush_props_open;
                // The spin keys are whatever Previous/Next alpha are bound to (Q/E by
                // default) — the same keys step the alpha outside Select mode.
                const KeyBind& spin_cw  = input.keybinds[(int)KeyAction::BRUSH_NEXT];
                const KeyBind& spin_ccw = input.keybinds[(int)KeyAction::BRUSH_PREV];
                int dir = 0;
                if (input.interaction_mode == InputState::InteractionMode::SELECT
                    && !modal && !input.ctrl_held && !scene.selected_ids().empty()) {
                    if (spin_cw.bound()  && glfwGetKey(window, spin_cw.key)  == GLFW_PRESS) dir += 1;
                    if (spin_ccw.bound() && glfwGetKey(window, spin_ccw.key) == GLFW_PRESS) dir -= 1;
                }

                const double now = glfwGetTime();
                if (dir == 0) {
                    rot_active = false;
                } else {
                    const auto& sel = scene.selected_ids();
                    scene.materialize_active_cpu();

                    if (!rot_active) {
                        // The pivot is now PER ENTITY (latched below, in the record),
                        // not one centroid for the whole selection. A shared pivot
                        // made everything ORBIT it — select a head and a pair of ears
                        // and the ears swung around the head instead of turning where
                        // they sat, which is not what "spin" means.
                        //
                        // get_view_direction() runs camera -> target, i.e. INTO
                        // the screen. A right-handed positive turn about an axis
                        // pointing away from the viewer reads clockwise, so E is +.
                        rot_axis      = camera.get_view_direction().normalized();
                        rot_last_time = now;
                        rot_active    = true;
                        xforms.begin(ObjectXform::Kind::SPIN, scene);
                    }

                    // Clamped so a hitch (a cascade, a window drag) cannot bank up
                    // into one violent jump on the frame after it.
                    float dt = (float)(now - rot_last_time);
                    rot_last_time = now;
                    if (dt > 0.1f) dt = 0.1f;

                    const float ang = (float)dir * 1.5707963f * dt;   // 90 deg/sec
                    if (dt > 0.0f && ang != 0.0f) {
                        for (uint32_t sel_id : sel) {
                            MeshEntity* e = scene.find_entity(sel_id);
                            if (!e) continue;
                            uint32_t vc = e->mesh.vertex_count();
                            ObjectXformTarget& xt = xforms.target(sel_id);
                            // Latch this entity's pivot and lobe rule on its first
                            // frame and reuse them for the rest of the gesture. Both
                            // for the reason the axis is latched — recomputing a
                            // bounding centre per frame lets the piece wander as it
                            // turns, because the centre is only rotation-invariant
                            // in exact arithmetic — and so the undo replays the same
                            // split the turn was made with.
                            if (xt.angle == 0.0f)
                                spin_latch_target(e->mesh, input.mirror_x, xt);
                            xt.axis   = rot_axis;
                            xt.angle += ang;
                            auto rotate_mesh = [&](Mesh& m) {
                                spin_apply_mesh(m, xt.pivot, rot_axis, ang,
                                                xt.mirror_lobes);
                            };
                            rotate_mesh(e->mesh);
                            if (e->multires.locked) {
                                // The base cage has to turn with the surface or the
                                // next cascade regenerates the mesh unrotated.
                                rotate_mesh(e->multires.base);
                                // Tangent frames are DIRECTIONS on the base surface and
                                // the disp layers are expressed in them. Translation and
                                // uniform scale leave them valid — which is why move and
                                // scale ignore them — but a rotation does not. Mark every
                                // level stale so the next cascade rebuilds them on the
                                // turned surface; leaving them re-applies your detail in
                                // the old orientation, one subdiv step later. Sizes stay
                                // (frames.size() must track disp.size()), contents go.
                                for (auto& lvl : e->multires.frames) lvl.clear();
                                e->multires_gpu.cleanup();   // VRAM mirror holds stale frames
                            }
                            if (rot_dirty.size() != vc) {
                                rot_dirty.resize(vc);
                                for (uint32_t i = 0; i < vc; i++) rot_dirty[i] = i;
                            }
                            scene.sync_partial_entity(sel_id, rot_dirty);
                        }
                        refresh_active_gpu_residency();
                        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                        screen_buffers_dirty = true;
                    }
                }
            }

            // Close the object-transform gesture the frame the drag or the key
            // ends. One press-to-release is one step back: hold E for two
            // seconds and that is a single Ctrl+Z, not a hundred. Sited here so
            // all three paths share it and none can leak a half-open gesture.
            if (input.drag_mode != InputState::DragMode::MOVE_OBJECT
                && input.drag_mode != InputState::DragMode::SCALE_OBJECT
                && !rot_active)
                xforms.commit(scene);

            // One-shot actions (IDLE only)

            // Delete selected mesh (+ mirror pair if linked)
            if (input.delete_mesh_requested) {
                input.delete_mesh_requested = false;
                if (scene.selected()) {
                    auto r = scene.delete_selected();
                    if (r.blocked_only_mesh) {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Cannot delete the only mesh");
                        input.notification_timer = 2.0f;
                    } else if (r.deleted) {
                        mesh = &scene.active_mesh();
                        multires = &scene.active_multires();
                        // The deleted entity's undo history dies with it; the
                        // now-active entity keeps its own. No clear here.
                        brush_stroke.vertex_count = 0;
                        brush_stroke.phase = StrokePhase::NONE;
                        mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                        screen_buffers_dirty = true;
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Deleted mesh %u, selected mesh %u",
                                      r.deleted_id, r.new_selected_id);
                        input.notification_timer = 2.0f;
                    }
                }
            }

            // Mask invert (Ctrl+I)
            if (input.mask_invert_requested) {
                input.mask_invert_requested = false;
                if (mesh->mask.empty()) {
                    mesh->mask.assign(mesh->vertex_count(), 0.0f);
                } else if (mesh->mask.size() < mesh->vertex_count()) {
                    mesh->mask.resize(mesh->vertex_count(), 0.0f);
                }
                // Record the flip as an undoable MASK entry (mirrors mask-clear below),
                // capturing every vertex whose value actually changes (mask != 0.5) as
                // old->new. Without this the invert leaves nothing on the stack and
                // Ctrl+Z silently skips over it.
                UndoEntry mask_e;
                mask_e.kind = UndoEntry::Kind::MASK;
                for (uint32_t v = 0; v < mesh->vertex_count(); v++) {
                    float old_val = mesh->mask[v];
                    float new_val = 1.0f - old_val;
                    if (old_val != new_val) {
                        mask_e.verts.push_back(v);
                        mask_e.old_mask.push_back(old_val);
                        mask_e.new_mask.push_back(new_val);
                    }
                    mesh->mask[v] = new_val;
                }
                // Invert the finest-level plane too: midpoint averaging is
                // linear (avg(1-a,1-b) == 1-avg(a,b)), so an elementwise flip
                // keeps every level consistent — fine detail inverts in place
                // instead of being re-interpolated from the coarse prefix.
                for (float& mv : multires->mask) mv = 1.0f - mv;
                renderer.update_mask(*mesh);
                if (!mask_e.verts.empty())
                    scene.active_undo().push(std::move(mask_e));
            }

            // Mask clear (Ctrl+A when mask brush is active)
            if (input.mask_clear_requested) {
                input.mask_clear_requested = false;
                // Drop the finest-level plane outright: clearing only the
                // working prefix would leave fine-level detail to resurrect
                // on the next level switch.
                multires->mask.clear();
                if (!mesh->mask.empty()) {
                    UndoEntry mask_e;
                    mask_e.kind      = UndoEntry::Kind::MASK;
                    for (uint32_t v = 0; v < mesh->vertex_count() && v < (uint32_t)mesh->mask.size(); v++) {
                        if (mesh->mask[v] != 0.0f) {
                            mask_e.verts.push_back(v);
                            mask_e.old_mask.push_back(mesh->mask[v]);
                            mask_e.new_mask.push_back(0.0f);
                            mesh->mask[v] = 0.0f;
                        }
                    }
                    // Sync before move: mask_e.verts invalid after std::move.
                    scene.sync_mask_partial_entity(scene.active_mesh_id(), mask_e.verts);
                    scene.active_undo().push(std::move(mask_e));
                }
            }

            // Undo/redo act on the active entity's own per-model stack. When the
            // reverted layer sits below the current view level, rebuild the active
            // surface from its multires stack (the pre-multimesh single-mesh flow).
            auto cascade_active = [&](bool needs_cascade) {
                if (!needs_cascade) return;
                // An in-place GPU-resident undo (undo.cpp Case 1) writes the working
                // VBO + disp/base SSBO and only mark_cpu_dirty()s the touched verts —
                // CPU disp/base/pos stay stale. Flush them back NOW, while the surface
                // is still at the current (pre-cascade) level so multires_gpu's dirty
                // vert indices are in range: cascade_to_level below reads CPU disp/base
                // (else the descended surface is wrong), and splice_active shrinks the
                // mesh — a later materialize against the smaller mesh would index its
                // adjacency out of bounds (wasm traps; native was lenient). Same choke
                // every other rebuild path already runs first (level-switch/remesh/merge).
                scene.materialize_active_cpu();
                MeshEntity& ent = scene.active_entity();
                multires_sync_paint(ent.multires, ent.mesh);
                // Same-level cascade: restore the working mask verbatim after the
                // rebuild (exact — no fold/interpolate round-trip blur).
                auto saved_mask = std::move(ent.mesh.mask);
                Mesh solo;
                cascade_to_level(ent.multires, solo, ent.multires.current_level, &compute);
                if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
                    solo.mask = std::move(saved_mask);
                arm_geometry_handoff(solo);
                scene.splice_active(solo);  // reuses cascade CSR/normals + full sync
                mesh     = &scene.active_mesh();
                multires = &scene.active_multires();
                scene.refresh_mirror_map();
                refresh_active_gpu_residency();
            };
            // UndoStack is per-entity sculpt history, so outside Edit mode there is
            // still nothing IT can correctly revert — insert placement and selection
            // push no entries. Reverting an unrelated older stroke instead is worse
            // than doing nothing, so that side stays clamped to a no-op. Object
            // transforms are the exception and are handled below: they have their
            // own scene-level stack now, so Ctrl+Z reaches them from Select mode.
            // Revert the last remesh. Deliberately ahead of the Edit-mode clamp
            // below: that clamp exists because per-entity sculpt history cannot
            // correctly revert scene-level work, and this snapshot is exactly the
            // scene-level undo it is missing. A merge is usually driven from Select
            // mode, so gating it on Edit would put the rescue out of reach in the
            // one case it was built for. Only fires once the entity's own history
            // is exhausted, so it is always the LAST step back, never a shortcut.
            // A dab's dirty list lands ASYNCHRONOUSLY, and an undo can move — or, across
            // a level switch, RENUMBER — every vertex. Ids still in flight would then be
            // applied to a mesh they no longer describe, which puts arbitrary vertices
            // in arbitrary directions: the spikes reported on 2026-09-17. finalize
            // already refuses to commit while reads are outstanding; undo never did, and
            // rapid undo presses are exactly how the window gets hit.
            //
            // Deferred, not drained in place: on the web the callbacks that land these
            // reads come from the event loop, so blocking here would deadlock the frame.
            // The request stays set and fires on a later frame, once the stroke's own
            // finalize/post_frame drain has emptied the queue.
            const bool undo_reads_pending =
                (input.undo_requested || input.redo_requested)
                && brush_stroke.dab_readbacks_pending();

            if (input.undo_requested && !undo_reads_pending
                && rescue.valid() && rescue.edits_left() > 0
                && !scene.active_undo().can_undo() && !xforms.can_undo()) {
                input.undo_requested = false;
                const char* what = rescue.op;
                snapshot_restore(rescue, scene);
                scene.mark_edited();
                rescue_shown = -1;
                // Same fixups the .chisel load path runs: the scene was rebuilt
                // wholesale, so every cached pointer and GPU mirror is stale.
                // No level hint on the mirror refresh — that argument is only an
                // icosphere cache lookup and a restored scene is arbitrary geometry.
                scene.refresh_mirror_map();
                scene.sync();
                mesh     = &scene.active_mesh();
                multires = &scene.active_multires();
                refresh_active_gpu_residency();
                // Framing bounds only — the CAMERA is deliberately left where the
                // user put it while they were inspecting the remesh.
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                scene.active_undo().clear(&compute);   // ring cached the dead topology
                xforms.clear();                        // and this scene is a rebuilt one
                brush_stroke.vertex_count = 0;
                brush_stroke.phase = StrokePhase::NONE;
                app_state = AppState::IDLE;
                screen_buffers_dirty = true;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Reverted the %s (%u v, %u t) - no undo history",
                              what, mesh->vertex_count(), mesh->tri_count());
                input.notification_timer = 3.0f;
            }

            // Which of the two histories does this Ctrl+Z belong to? Both stamp
            // their entries from UndoStack's scene-wide clock, so undo takes
            // whichever top is NEWER and redo whichever is OLDER — strict
            // last-in-first-out across the pair. That ordering is not a nicety:
            // UndoEntry stores absolute positions, so undoing a stroke that
            // predates an object transform would drop those vertices back into
            // the pre-transform frame and tear the mesh. Interleaving makes it
            // impossible to undo *through* a transform.
            //
            // A push on either stack kills the other's redo arm, since the user
            // has branched off that timeline. The xform side clears the entity
            // stacks it touches at commit; this clock check covers the reverse.
            if (input.redo_requested && xforms.can_redo()
                && UndoStack::global_pushes != xforms.redo_clock)
                xforms.clear_redo();

            const UndoEntry* utop = scene.active_undo().peek_undo();
            const UndoEntry* rtop = scene.active_undo().peek_redo();
            const bool xform_undo_next = xforms.can_undo()
                && (!utop || xforms.undo_seq() > utop->seq);
            const bool xform_redo_next = xforms.can_redo()
                && (!rtop || xforms.redo_seq() < rtop->seq);

            // Per-entity sculpt history still can't correctly revert scene-level
            // work outside Edit mode, so the old clamp stands — but an object
            // transform IS scene-level, and it is driven from Select mode, so it
            // has to stay reachable there.
            bool undo_allowed = input.interaction_mode == InputState::InteractionMode::EDIT;
            if (input.undo_requested && !undo_allowed && xform_undo_next) undo_allowed = true;
            if (input.redo_requested && !undo_allowed && xform_redo_next) undo_allowed = true;
            if ((input.undo_requested || input.redo_requested) && !undo_allowed
                && !undo_reads_pending) {
                input.undo_requested = false;
                input.redo_requested = false;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Undo only works in Edit mode (1)");
                input.notification_timer = 1.5f;
            }
            if (input.undo_requested && !undo_reads_pending) {
                input.undo_requested = false;
                uint32_t xform_blocker = xform_undo_next ? xforms.newer_edit_entity(scene) : 0;
                if (xform_blocker) {
                    // See newer_edit_entity(): a sibling of this transform has been
                    // sculpted since, and its stroke positions are absolute.
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Undo the newer edits on mesh %u first", xform_blocker);
                    input.notification_timer = 2.5f;
                } else if (xform_undo_next) {
                    xforms.undo(scene);
                    if (xforms.last_was_spin()) refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Undid object %s", xforms.last_label());
                    input.notification_timer = 1.5f;
                } else {
                    const UndoEntry* top = scene.active_undo().peek_undo();
                    const uint64_t seq = top ? top->seq : 0;
                    cascade_active(scene.active_undo().undo(scene.active_entity(), scene));
                    scene.active_entity().pose.on_undo(seq);
                    print_undo_top("ctrl-z");
                }
                screen_buffers_dirty = true;
            }
            if (input.redo_requested && !undo_reads_pending) {
                input.redo_requested = false;
                if (xform_redo_next) {
                    xforms.redo(scene);
                    if (xforms.last_was_spin()) refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Redid object %s", xforms.last_label());
                    input.notification_timer = 1.5f;
                } else {
                    const UndoEntry* top = scene.active_undo().peek_redo();
                    const uint64_t seq = top ? top->seq : 0;
                    cascade_active(scene.active_undo().redo(scene.active_entity(), scene));
                    scene.active_entity().pose.on_redo(seq);
                    print_undo_top("ctrl-shift-z");
                }
                screen_buffers_dirty = true;
            }

            // Manual projection (P post-lock)
            if (input.project_requested) {
                input.project_requested = false;
                const int L_max = multires->base_level + (int)multires->disp.size();
                const int target = multires->current_level;
                if (target >= L_max) {
                    std::printf("[project] already at top level, nothing to project\n");
                } else {
                    std::printf("[project] projecting from level %d (truth) down to level %d (current)\n",
                                L_max, target);
                    std::printf("[project]   snapshotting levels %d..%d\n",
                                target, L_max);

                    UndoEntry e;
                    e.kind         = UndoEntry::Kind::PROJECTION;
                    e.target_level = target;
                    capture_projection_snapshot(*multires, target, e.before);

                    ProjectionStats ps = project_down_to_level(*multires, target);
                    report_projection_check(ps);

                    for (int k = L_max; k > target; k--)
                        std::printf("[project]   inverse-Loop %d -> %d ... done\n", k, k - 1);

                    int k_first = (target == multires->base_level) ? 0
                                    : (target - multires->base_level - 1);
                    std::printf("[project]   rewriting disp[%d..%d] ... done\n",
                                k_first, (int)multires->disp.size() - 1);

                    std::printf("[project] done in %.2f ms\n", ps.elapsed_ms);
#ifdef CHISEL_DEBUG_MULTIRES
                    std::printf("[project][debug] max reconstruction error at L%d: %.3e\n",
                                L_max, ps.max_reconstruction_error);
#endif

                    scene.active_undo().push(std::move(e));
                    print_undo_top("project");

                    multires_sync_paint(*multires, *mesh);
                    if (scene.alive_count() <= 1) {
                        auto saved_mask = std::move(mesh->mask);
                        cascade_to_level(*multires, *mesh, multires->current_level, &compute);
                        if (!saved_mask.empty() && saved_mask.size() == mesh->vertex_count())
                            mesh->mask = std::move(saved_mask);
                        arm_geometry_handoff(*mesh);
                        scene.refresh_mirror_map();
                        scene.sync();
                    } else {
                        // Same-level cascade — keep the exact working mask (see
                        // the undo cascade above).
                        auto saved_mask = std::move(mesh->mask);
                        Mesh solo;
                        cascade_to_level(*multires, solo, multires->current_level, &compute);
                        if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
                            solo.mask = std::move(saved_mask);
                        arm_geometry_handoff(solo);
                        scene.splice_active(solo);
                        scene.refresh_mirror_map();
                    }
                    refresh_active_gpu_residency();
                    screen_buffers_dirty = true;
                }
            }
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

        // ---- Cursor visibility ----
        // Retopo counts as an edit mode here: it shows the brush ring (the band width).
        const bool retopo_mode = input.interaction_mode == InputState::InteractionMode::RETOPO;
        bool non_edit_mode = input.interaction_mode != InputState::InteractionMode::EDIT
                          && !retopo_mode;
        // Hide the OS cursor while sculpting (we draw our own brush cursor).
        bool show_os_cursor = input.quit_requested || input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.remesh_confirm_pending || input.voxel_merge_confirm_pending || input.drop_confirm_pending || input.reset_confirm_pending || input.pose_clear_confirm_pending || input.help_popup_open || input.keybind_menu_open || input.brush_props_open || imgui_wants_mouse || non_edit_mode;
#ifndef __EMSCRIPTEN__
        // A slider drag owns the cursor mode while it runs: it holds GLFW_CURSOR_DISABLED
        // to capture the pointer. This line ran unconditionally every frame and put the
        // mode straight back to NORMAL/HIDDEN, which destroys the compositor's pointer
        // lock ~50us after the drag created it. Two owners of one piece of state, and
        // this one won sixty times a second — so the capture never actually held, the
        // pointer really did travel, and glfwSetCursorPos was refused for want of a lock.
        if (input.slider_mode == InputState::SliderMode::NONE)
            glfwSetInputMode(window, GLFW_CURSOR, show_os_cursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
#else
        // Emscripten's GLFW has no GLFW_CURSOR_HIDDEN (warns every frame) —
        // drive the canvas CSS cursor instead, only on state change.
        static bool css_cursor_hidden = false;
        if (css_cursor_hidden == show_os_cursor) {
            css_cursor_hidden = !show_os_cursor;
            EM_ASM({ Module['canvas'].style.cursor = $0 ? 'none' : 'auto'; }, css_cursor_hidden);
        }
#endif

        // ---- Render ----
#ifdef CHISEL_BACKEND_WEBGPU
        // Acquire this frame's swapchain colour view and inject it as the default
        // render target; default-screen passes render into it (see webgpu_backend).
        // The colour/depth CLEAR happens via the first pass's loadOp=Clear, owned by
        // renderer.draw_background (the GL build instead clears explicitly below).
        WGPUSurfaceTexture surfTex = {};
        wgpuSurfaceGetCurrentTexture(g_surface, &surfTex);
#ifndef __EMSCRIPTEN__
        // An outdated/lost surface (the window changed under the swapchain) hands back
        // no texture and must be reconfigured before it can present again; those are
        // logged.
        //
        // "Suboptimal" gets ONE reconfigure per window size, on the next frame. Intel's
        // Windows Vulkan driver reports it on nearly every frame after any resize, so it
        // isn't logged and isn't chased beyond that one retry — but the retry matters:
        // without it, a window restored (or created) hanging under the taskbar shows
        // black triangles while orbiting/sculpting; with it, it doesn't (user-tested
        // both ways). Presumably the reconfigure made at the resize lands while the
        // window is still mid-transition, and the second one sees it settled.
        {
            static int subopt_w = -1, subopt_h = -1;
            if (surfTex.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal &&
                (subopt_w != win_w || subopt_h != win_h)) {
                subopt_w = win_w; subopt_h = win_h;
                g_surface_reconfigure = true;
            }
            const bool lost = surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
                              surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Lost;
            if (lost || surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Timeout ||
                surfTex.status == WGPUSurfaceGetCurrentTextureStatus_Error)
                std::printf("[win] surface status %d (window %dx%d, configured %dx%d)\n",
                            (int)surfTex.status, win_w, win_h, fbw, fbh);
            if (!surfTex.texture && lost) {
                fbw = win_w; fbh = win_h;
                configureSurface(fbw, fbh);
                makeDepth(fbw, fbh);
                screen_buffers_dirty = true;
                wgpuSurfaceGetCurrentTexture(g_surface, &surfTex);
            }
        }
#endif
        WGPUTextureView frameView = surfTex.texture
            ? wgpuTextureCreateView(surfTex.texture, nullptr) : nullptr;
        bool spare_frame = false;
#ifndef __EMSCRIPTEN__
        // No texture this frame: draw into the stand-in and don't present (see
        // spareColorView). Said once per run of missed frames, not every frame.
        {
            static bool missing = false;
            if (!frameView) {
                if (!missing)
                    std::printf("[win] no swapchain frame (status %d): drawing offscreen "
                                "until it comes back\n", (int)surfTex.status);
                frameView   = spareColorView(fbw, fbh);
                spare_frame = true;
            }
            missing = spare_frame;
        }
#endif
        gpu::webgpu_set_default_color(frameView);
#else
        // glClear(DEPTH) is gated by the depth write-mask; force it TRUE so the clear
        // can never be silently no-op'd by whatever pipeline drew last (e.g. a HUD
        // pipeline leaving depthMask FALSE). Don't rely on every pipeline upholding it.
        glDepthMask(GL_TRUE);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#endif

        // Background gradient
        renderer.draw_background(win_w, win_h);

        // Paint stays visible while the paint brush is active regardless of the
        // toggle, so you can always see what you're laying down.
        renderer.paint_visible =
            (input.paint_visible || input.current_brush == BrushType::PAINT) ? 1.0f : 0.0f;
        renderer.matcap_contrast = input.matcap_contrast;
        renderer.flat_shading = input.flat_shading ? 1.0f : 0.0f;
        for (int i = 0; i < 3; i++) renderer.unselected_tint[i] = input.unselected_tint[i];

        // Mesh: N draws. The active entity is drawn from the working VAO; every
        // other alive entity from its static display VAO. Depth test composes them.
        {
            const std::vector<uint32_t>& sel = scene.selected_ids();
            uint32_t active_id = scene.active_mesh_id();
            for (auto& up : scene.entities()) {
                if (!up || !up->alive) continue;
                // An entity is tinted (deselected) unless it's in the selection set.
                // An empty set is a real, reachable state (the user deselected
                // everything) → every entity tinted: a fully-deselected scene.
                bool selected = false;
                for (uint32_t id : sel) if (id == up->id) { selected = true; break; }
                if (up->id == active_id)
                    renderer.draw_mesh(camera, win_w, win_h,
                                       (uint32_t)scene.active_mesh().indices.size(),
                                       selected);
                else
                    renderer.draw_display(camera, up->gpu, win_w, win_h, selected);
            }
        }

        // Debug mesh overlay (Y key toggle). Invalidate cached edge buffer on
        // every transition so the next "on" rebuilds against current topology.
        static bool prev_show_debug_mesh = false;
        if (input.show_debug_mesh != prev_show_debug_mesh) {
            renderer.invalidate_debug_mesh();
            prev_show_debug_mesh = input.show_debug_mesh;
        }
        if (input.show_debug_mesh) {
            renderer.draw_debug_mesh(camera, *mesh, win_w, win_h);
        }

        // Pose rig: shown only while the pose brush is in hand. The reach ring is the
        // farthest a bone's skin envelope may extend from it: two sphere radii (pose_bind).
        if (input.pose_brush_on() && !non_edit_mode) {
            draw_pose_rig(scene.active_entity().pose, camera, win_w, win_h,
                          input.pose_hover_joint, 1.0f,
                          input.ctrl_held && app_state == AppState::IDLE);
        }

        // Brush cursor — draw at locked position during slider, normal position otherwise
        if (!input.quit_requested && !input.export_dialog_active && !input.import_dialog_active && !input.save_dialog_active && !input.remesh_confirm_pending && !input.voxel_merge_confirm_pending && !input.drop_confirm_pending && !input.reset_confirm_pending && !input.pose_clear_confirm_pending && !input.help_popup_open && !input.keybind_menu_open && !input.brush_props_open && !imgui_wants_mouse && !non_edit_mode) {
            float cursor_x, cursor_y;
            if (input.slider_mode != InputState::SliderMode::NONE) {
                cursor_x = (float)input.slider_start_x;
                cursor_y = (float)input.slider_start_y;
            } else {
                cursor_x = (float)input.mouse_x;
                cursor_y = (float)input.mouse_y;
            }
            if (retopo_mode) {
                renderer.draw_cursor(camera, cursor_x, cursor_y, input.brush_size,
                                     input.cursor_nx, input.cursor_ny, input.cursor_nz,
                                     1.0f, 0.5f, win_w, win_h, input.on_model);
            } else if (input.color_pick_active) {
                // Armed colour picker: eyedropper glyph instead of the brush
                // ring — the ring's size/hardness reading is meaningless here.
                draw_pick_cursor(cursor_x, cursor_y, input.on_model);
            } else if (input.pose_brush_on() && !input.is_smooth_active()) {
                draw_pose_cursor(cursor_x, cursor_y,
                                 input.on_model || input.pose_hover_joint >= 0);
            } else if (input.bucket_active && input.current_brush == BrushType::PAINT
                       && !input.is_smooth_active()) {
                if (input.bucket_dragging) {
                    draw_gradient_drag(input.bucket_ax, input.bucket_ay, cursor_x, cursor_y);
                } else {
                    // A tiling pattern repeats once across the brush circle, and the
                    // mask feather (Hardness below 1) is up to a radius wide: show it.
                    if ((input.fill_pattern != kFillSolid && input.fill_pattern != kFillGradient)
                        || input.brush_hardness < 0.99f)
                        renderer.draw_cursor(camera, cursor_x, cursor_y, input.brush_size,
                                             input.cursor_nx, input.cursor_ny, input.cursor_nz,
                                             input.brush_hardness, input.brush_strength,
                                             win_w, win_h, input.on_model);
                    draw_bucket_cursor(cursor_x, cursor_y, input.on_model);
                }
            } else {
                renderer.draw_cursor(
                    camera,
                    cursor_x, cursor_y,
                    input.brush_size,
                    input.cursor_nx, input.cursor_ny, input.cursor_nz,
                    input.brush_hardness, input.brush_strength,
                    win_w, win_h, input.on_model);

                // Symmetry lobe marker. Where the second dab lands is a WORLD
                // fact (reflect the anchor through x=0, same as every brush's
                // anchor_b), not a screen one — mirroring the cursor pixel about
                // the window centre would only be right for a centred, axis-on
                // camera. So unproject the pen through the cached depth plane,
                // flip X, project back. Off-model there is no anchor to reflect.
                if (input.mirror_x && input.on_model) {
                    float hit_depth = 0.0f;
                    if (renderer.sample_depth_bilinear(cursor_x, cursor_y, &hit_depth, 0.0f)) {
                        Vec3 cam_pos = camera.get_position();
                        Vec3 fwd = (camera.target - cam_pos).normalized();
                        Vec3 wup = {0, 1, 0};
                        Vec3 right = fwd.cross(wup).normalized();
                        Vec3 up = right.cross(fwd).normalized();
                        float half_h = camera.half_height_at(hit_depth);
                        float half_w = half_h * ((float)win_w / (float)win_h);
                        float ndc_x = 2.0f * cursor_x / (float)win_w - 1.0f;
                        float ndc_y = 1.0f - 2.0f * cursor_y / (float)win_h;
                        Vec3 anchor = cam_pos + fwd * hit_depth
                                    + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                        anchor.x = -anchor.x;
                        float mx, my;
                        if (camera.world_to_screen(anchor, win_w, win_h, mx, my)) {
                            // Occlusion: the reflected anchor sits on the mesh's far
                            // side whenever it is deeper than whatever the depth plane
                            // already holds at that pixel. Fade rather than hide — you
                            // still want to know the lobe is there, just not to mistake
                            // it for something in front. The tolerance is a few pixels'
                            // worth of world depth, so a point on its OWN near-side
                            // surface (where the two depths agree to within the plane's
                            // own quantisation) never counts as hidden.
                            float alpha = 1.0f;
                            float front_depth = 0.0f;
                            if (renderer.sample_depth((int)mx, (int)my, &front_depth)) {
                                float world_per_pixel = (2.0f * half_h) / (float)win_h;
                                float tol = world_per_pixel * 6.0f;
                                if (camera.view_depth(anchor) > front_depth + tol)
                                    alpha = 0.30f;
                            }
                            renderer.draw_mirror_cursor(mx, my, input.brush_hardness,
                                                        alpha, win_w, win_h);
                        }
                    }
                }
            }
        }

        // ---- Overlays ----
        // Remesh and SDF-merge prompts: what they need from the scene is gathered once
        // here. The classic skin draws them as bitmap text right below; every other
        // skin draws real dialogs (draw_skin_dialogs, after the skin's UI).
        DialogInfo dlg;
        if (input.remesh_confirm_pending || input.voxel_merge_confirm_pending)
            dlg.levels_to_bake = snapshot_levels_to_bake(scene);
        if (input.voxel_merge_confirm_pending) {
            dlg.n_selected = (int)scene.selected_ids().size();
            // Count unselected (red) committed entities — candidate cutters for subtract.
            for (const auto& up : scene.entities()) {
                if (!up || !up->alive || up->preview) continue;
                bool sel = false;
                for (uint32_t sid : scene.selected_ids())
                    if (sid == up->id) { sel = true; break; }
                if (!sel) dlg.n_unselected++;
            }
            // Density field on any selected mesh → the merge output will carry
            // the heatmap; offer the chained adaptive remesh line.
            for (uint32_t sid : scene.selected_ids()) {
                const MeshEntity* e = scene.find_entity(sid);
                if (e && e->alive && !e->mesh.density.empty()) { dlg.has_density = true; break; }
            }
        }
        if (vmerge_job) dlg.merge_progress = voxel_merge_progress(*vmerge_job);
        const bool bitmap_dialogs = input.ui_classic();

        if (bitmap_dialogs && input.quit_requested)
            draw_quit_dialog(text, win_w, win_h);
        if (input.drop_confirm_pending)
            draw_drop_confirm(text, input.drop_path, win_w, win_h);
        if (bitmap_dialogs && input.remesh_confirm_pending)
            draw_remesh_confirm(text, dlg.levels_to_bake, win_w, win_h);
        if (input.drop_level_confirm_pending)
            draw_drop_level_confirm(text, multires->base_level + (int)multires->disp.size(),
                                    win_w, win_h);
        if (bitmap_dialogs && input.reset_confirm_pending)
            draw_reset_confirm(text, win_w, win_h);
        if (bitmap_dialogs && input.pose_clear_confirm_pending)
            draw_pose_clear_confirm(text, win_w, win_h);
        if (bitmap_dialogs && input.voxel_merge_confirm_pending)
            draw_voxel_merge_confirm(text, input.voxel_merge_resolution,
                                     dlg.n_selected, dlg.n_unselected,
                                     input.voxel_merge_surface_nets,
                                     dlg.has_density, input.voxel_merge_adaptive,
                                     dlg.levels_to_bake, win_w, win_h);
        if (bitmap_dialogs && input.remesh_in_progress)
            draw_remesh_progress(text, win_w, win_h);
        if (bitmap_dialogs && input.voxel_merge_in_progress)
            draw_voxel_merge_progress(text, win_w, win_h, dlg.merge_progress);
        // The bitmap HUD belongs to the classic skin; the modern skin draws its own
        // readouts, toast and slider in draw_modern_ui below.
        if (input.ui_classic()) {
            if (input.toolbar_visible)
                draw_toolbar(text, input, mesh->tri_count(), mesh->vertex_count(), CHISEL_VERSION,
                             multires->current_level,
                             current_project_path.c_str(), win_w, win_h);
            if (input.slider_mode != InputState::SliderMode::NONE)
                draw_slider(text, input, win_w, win_h);
            if (input.interaction_mode == InputState::InteractionMode::SELECT)
                draw_mode_indicator(text, "SELECT", win_w, win_h);
            else if (input.interaction_mode == InputState::InteractionMode::INSERT)
                draw_mode_indicator(text, "INSERT", win_w, win_h);
            draw_notification(text, input, win_w, win_h);
            draw_mirror_unavailable(text, input, win_w, win_h);
            if (input.show_fps)
                draw_fps(text, fps_display, win_w, win_h);
        }
        debug_console::draw(text, win_w, win_h);

        // ---- Open/import a path (shared by the native dialog and the web picker) ----
        auto do_import_path = [&](const std::string& path) {
            auto dot = path.find_last_of('.');
            std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
            for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);

            if (ext == "chisel") {
                ProjectData proj;
                LoadResult lr = load_project(path.c_str(), proj);
                if (lr == LoadResult::OK && !proj.entities.empty()) {
                    camera = proj.camera;
                    input.mirror_x = proj.mirror_x;
                    input.subdiv_level = proj.subdiv_level;
                    input.mesh_locked = true;
                    current_project_path = path;

                    // Rebuild the whole multimesh scene. load_entities does
                    // per-entity cascade/adjacency/normals/mirror and restores
                    // the saved selection; the active entity's mirror map is
                    // (re)cached by refresh_mirror_map before sync.
                    scene.set_mirror_topology(proj.mirror_use_topology);
                    scene.load_entities(proj.entities, proj.active_id,
                                        proj.selected_ids, proj.next_id);
                    xforms.clear();      // records point at the outgoing scene
                    scene.refresh_mirror_map(input.subdiv_level);
                    scene.sync();
                    clean_edit_mark = scene.edit_mark();

                    mesh = &scene.active_mesh();
                    multires = &scene.active_multires();
                    refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    brush_stroke.vertex_count = 0;
                    brush_stroke.phase = StrokePhase::NONE;
                    app_state = AppState::IDLE;
                    screen_buffers_dirty = true;

                    if (scene.load_flattened() > 0) {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Loaded with %d mesh(es) flattened: file is from "
                                      "another platform — re-save there to migrate",
                                      scene.load_flattened());
                        input.notification_timer = 6.0f;
                    } else {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Loaded: %.400s (%zu mesh%s, active %u v, %u t)",
                                      path.c_str(), proj.entities.size(),
                                      proj.entities.size() == 1 ? "" : "es",
                                      mesh->vertex_count(), mesh->tri_count());
                        input.notification_timer = 3.0f;
                    }
                } else {
                    error_popup_msg = std::string("Load failed: ") + result_string(lr) + "\n" + path;
                    error_popup_trigger = true;
                }
            } else {
                Mesh loaded;
                bool imported = (ext == "ply") ? Mesh::import_ply(path.c_str(), loaded)
                                               : Mesh::import_obj(path.c_str(), loaded);
                if (imported && input.import_append) {
                    // Append: add the mesh as a NEW scene entity at its authored
                    // scale, leaving existing entities, the camera, and their undo
                    // untouched. Same commit path insert uses (add_preview →
                    // commit_preview → multires/mirror init).
                    loaded.mask.clear();
                    uint32_t new_id = scene.add_preview(loaded, 0);
                    scene.commit_preview(new_id);
                    multires_stack_init_from_lock(scene.active_multires(),
                                                  scene.active_mesh(), 0);
                    scene.set_mirror_topology(false);
                    scene.refresh_mirror_map();
                    scene.sync();

                    input.mesh_locked = true;
                    mesh = &scene.active_mesh();
                    multires = &scene.active_multires();
                    refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    brush_stroke.vertex_count = 0;
                    brush_stroke.phase = StrokePhase::NONE;
                    app_state = AppState::IDLE;
                    screen_buffers_dirty = true;

                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Appended: %.400s (%u v, %u t)",
                                  path.c_str(), mesh->vertex_count(), mesh->tri_count());
                    input.notification_timer = 3.0f;
                } else if (imported) {
                    *mesh = std::move(loaded);
                    mesh->mask.clear();

                    scene.set_mirror_topology(false);
                    // build_adjacency FIRST: it bumps topo_version, and
                    // refresh_mirror_map stamps mirror_topo_version with whatever
                    // topo_version reads at the time. Refreshing first left the
                    // stamp permanently one behind, so the guard in
                    // refresh_mirror_map never short-circuited again and every later
                    // call rebuilt the map from *sculpted* positions — exactly the
                    // reclassification that guard exists to prevent.
                    mesh->build_adjacency();
                    scene.refresh_mirror_map();

                    input.mesh_locked = true;
                    multires_stack_init_from_lock(*multires, *mesh, 0);
                    scene.reset_to_single_mesh(0);
                    scene.sync();
                    mesh = &scene.active_mesh();
                    multires = &scene.active_multires();
                    refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    camera.set_target(mesh_center);
                    camera.distance = mesh_radius * 2.5f;
                    current_project_path.clear();

                    scene.active_undo().clear(&compute);
                    xforms.clear();
                    brush_stroke.vertex_count = 0;
                    brush_stroke.phase = StrokePhase::NONE;
                    app_state = AppState::IDLE;
                    screen_buffers_dirty = true;
                    clean_edit_mark = scene.edit_mark();   // the imported file still holds it

                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Imported: %.400s (%u v, %u t)",
                                  path.c_str(), mesh->vertex_count(), mesh->tri_count());
                    input.notification_timer = 3.0f;
                } else {
                    error_popup_msg = "Import failed: " + path;
                    error_popup_trigger = true;
                }
            }
        };

        // Dev hook: CHISEL_AUTO_IMPORT=<path> imports a file on the first frame
        // (headless load-path testing without driving the UI). No-op when unset.
        // A bare path on the command line ("Open With" / chisel foo.chisel)
        // rides the same first-frame import; CLI path wins over the env hook.
        {
            static bool auto_import_done = false;
            if (!auto_import_done) {
                auto_import_done = true;
                const char* auto_path = cli_open_path.empty()
                                      ? std::getenv("CHISEL_AUTO_IMPORT")
                                      : cli_open_path.c_str();
                if (auto_path) {
                    if (FILE* tf = std::fopen(auto_path, "rb")) {
                        std::fclose(tf);
                        std::printf("[auto-import] loading %s\n", auto_path);
                        do_import_path(auto_path);
                    } else {
                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Cannot open: %.400s", auto_path);
                        input.notification_timer = 4.0f;
                    }
                }
            }
        }

        // Dev hook: CHISEL_AUTO_SUBD=<n> requests one level-up per frame for the
        // first n frames — drives the subdivision guard without the UI (pair with
        // CHISEL_LIMITS_MB to watch it refuse). No-op when unset.
        {
            static int auto_subd_left = -1;
            if (auto_subd_left < 0) {
                const char* env = std::getenv("CHISEL_AUTO_SUBD");
                auto_subd_left = env ? std::atoi(env) : 0;
            }
            if (auto_subd_left > 0) {
                --auto_subd_left;
                input.level_switch_delta = +1;
            }
        }

        // Dev hook: CHISEL_REMESH_KEEP_DETAIL=0 starts with the Settings box
        // "Remesh keeps detail" unticked, so a scripted remesh can be run both
        // ways on the same file and the two DRIFT lines compared.
        {
            static bool keep_detail_env_done = false;
            if (!keep_detail_env_done) {
                keep_detail_env_done = true;
                if (const char* env = std::getenv("CHISEL_REMESH_KEEP_DETAIL"))
                    input.remesh_keep_detail = std::atoi(env) != 0;
                if (const char* env = std::getenv("CHISEL_REMESH_DETAIL"))
                    input.remesh_detail = std::min(4.0f, std::max(0.25f, (float)std::atof(env)));
                // SDF remesh knobs for CHISEL_AUTO_MERGE runs.
                if (const char* env = std::getenv("CHISEL_MERGE_RES"))
                    input.voxel_merge_resolution = std::min(256, std::max(64, std::atoi(env)));
                if (const char* env = std::getenv("CHISEL_MERGE_MIRROR"))
                    input.voxel_merge_mirror = std::atoi(env) != 0;
            }
        }

        // Dev hook: CHISEL_AUTO_REMESH=1 fires one plain remesh (no density
        // paint) once the scene has settled — pairs with CHISEL_AUTO_IMPORT and
        // the two env knobs above to A/B a file headless. No-op when unset.
        {
            static int auto_remesh_frames = -1;
            if (auto_remesh_frames < 0) {
                const char* env = std::getenv("CHISEL_AUTO_REMESH");
                auto_remesh_frames = (env && std::atoi(env) != 0) ? 1 : 0;
            }
            if (auto_remesh_frames > 0 && ++auto_remesh_frames > 12 &&
                app_state == AppState::IDLE && !input.remesh_in_progress) {
                input.remesh_requested = true;
                auto_remesh_frames = 0;
            }
        }

        // Dev hook: CHISEL_AUTO_MERGE=1 fires one SDF remesh of the selection the
        // same way, so CHISEL_REMESH_KEEP_DETAIL=0/1 A/Bs the merge's snap too.
        {
            static int auto_merge_frames = -1;
            if (auto_merge_frames < 0) {
                const char* env = std::getenv("CHISEL_AUTO_MERGE");
                auto_merge_frames = (env && std::atoi(env) != 0) ? 1 : 0;
            }
            if (auto_merge_frames > 0 && ++auto_merge_frames > 12 &&
                app_state == AppState::IDLE && !input.voxel_merge_in_progress) {
                input.voxel_merge_requested = true;
                auto_merge_frames = 0;
            }
        }

        // Dev hook: CHISEL_AUTO_DENSITY_REMESH=1 paints a hard half-and-half
        // density field on the active mesh (y above centroid ⇒ red 1.0, below
        // ⇒ green 0.0) and fires one remesh — drives the adaptive remesher
        // headless, then prints per-bucket edge-length means so the split can
        // be verified from the log (red mean ≈ fine_mult/coarse_mult × green
        // mean). CHISEL_AUTO_DENSITY_MERGE=1 paints the same field but fires a
        // voxel merge instead (mirror MC, default R), then prints the merged
        // mesh's mean density above/below the centroid — verifies the
        // nearest-source carry spatially (above ≈ 1, below ≈ 0). No-op when
        // both are unset.
        {
            static int  auto_density_state = -1;
            static int  auto_density_warmup = 0;
            static bool auto_density_merge = false;
            if (auto_density_state < 0) {
                const char* env = std::getenv("CHISEL_AUTO_DENSITY_REMESH");
                auto_density_state = (env && std::atoi(env) != 0) ? 1 : 0;
                const char* menv = std::getenv("CHISEL_AUTO_DENSITY_MERGE");
                if (menv && std::atoi(menv) != 0) {
                    auto_density_state = 1;
                    auto_density_merge = true;
                }
            }
            // Warmup lets a CHISEL_AUTO_SUBD ladder finish first (composable:
            // subdivide N levels, then paint + adaptive-remesh the result).
            if (auto_density_state == 1 && ++auto_density_warmup > 10 &&
                app_state == AppState::IDLE &&
                input.level_switch_delta == 0 && !input.remesh_in_progress &&
                !input.voxel_merge_in_progress && !vmerge_job) {
                const uint32_t vc = mesh->vertex_count();
                double cy = 0.0;
                for (uint32_t v = 0; v < vc; v++) cy += mesh->pos_y[v];
                cy = (vc > 0) ? cy / vc : 0.0;
                mesh->density.assign(vc, 0.0f);
                for (uint32_t v = 0; v < vc; v++)
                    if (mesh->pos_y[v] > cy) mesh->density[v] = 1.0f;
                std::printf("[auto-density] painted half/half field (centroid y=%.4f), requesting %s\n",
                            cy, auto_density_merge ? "voxel merge" : "remesh");
                if (auto_density_merge) {
                    input.voxel_merge_mirror       = true;
                    input.voxel_merge_subtract     = false;
                    input.voxel_merge_surface_nets = false;
                    input.voxel_merge_requested    = true;
                } else {
                    input.remesh_requested = true;
                }
                auto_density_state = 2;
            } else if (auto_density_state == 2 && auto_density_merge &&
                       !input.voxel_merge_requested && !input.voxel_merge_in_progress) {
                const uint32_t vc = mesh->vertex_count();
                const uint32_t dc = (uint32_t)mesh->density.size();
                double cy = 0.0;
                for (uint32_t v = 0; v < vc; v++) cy += mesh->pos_y[v];
                cy = (vc > 0) ? cy / vc : 0.0;
                double sum[2] = {0, 0}; uint64_t cnt[2] = {0, 0}; uint64_t mid = 0;
                for (uint32_t v = 0; v < vc && v < dc; v++) {
                    int b = (mesh->pos_y[v] > cy) ? 1 : 0;
                    sum[b] += mesh->density[v]; cnt[b]++;
                    if (mesh->density[v] > 0.25f && mesh->density[v] < 0.75f) mid++;
                }
                std::printf("[auto-density] post-merge density means: "
                            "above=%.3f (n=%llu) below=%.3f (n=%llu) mid-band=%llu, field %u/%u verts\n",
                            cnt[1] ? sum[1]/cnt[1] : -1.0, (unsigned long long)cnt[1],
                            cnt[0] ? sum[0]/cnt[0] : -1.0, (unsigned long long)cnt[0],
                            (unsigned long long)mid, dc, vc);
                auto_density_state = 3;
            } else if (auto_density_state == 2 && !auto_density_merge &&
                       !input.remesh_requested && !input.remesh_in_progress) {
                double sum[3] = {0, 0, 0};
                uint64_t cnt[3] = {0, 0, 0};
                const uint32_t tc = mesh->tri_count();
                const uint32_t dc = (uint32_t)mesh->density.size();
                for (uint32_t t = 0; t < tc; t++) {
                    for (int e = 0; e < 3; e++) {
                        uint32_t a = mesh->indices[t*3+e];
                        uint32_t b = mesh->indices[t*3+(e+1)%3];
                        float da = (a < dc) ? mesh->density[a] : 0.5f;
                        float db = (b < dc) ? mesh->density[b] : 0.5f;
                        float d = 0.5f * (da + db);
                        int bucket = (d > 0.75f) ? 2 : (d < 0.25f) ? 0 : 1;
                        float dx = mesh->pos_x[a] - mesh->pos_x[b];
                        float dy = mesh->pos_y[a] - mesh->pos_y[b];
                        float dz = mesh->pos_z[a] - mesh->pos_z[b];
                        sum[bucket] += std::sqrt(dx*dx + dy*dy + dz*dz);
                        cnt[bucket]++;
                    }
                }
                std::printf("[auto-density] post-remesh edge means: "
                            "green=%.5f (n=%llu) mid=%.5f (n=%llu) red=%.5f (n=%llu), green/red=%.2f\n",
                            cnt[0] ? sum[0]/cnt[0] : 0.0, (unsigned long long)cnt[0],
                            cnt[1] ? sum[1]/cnt[1] : 0.0, (unsigned long long)cnt[1],
                            cnt[2] ? sum[2]/cnt[2] : 0.0, (unsigned long long)cnt[2],
                            (cnt[0] && cnt[2] && sum[2] > 0.0)
                                ? (sum[0]/cnt[0]) / (sum[2]/cnt[2]) : 0.0);
                auto_density_state = 3;
            }
        }

#ifndef __EMSCRIPTEN__
        // ---- ImGui file dialogs (native: browse the real filesystem) ----
        auto* fd = IGFD::FileDialog::Instance();

        if (input.export_dialog_active && !fd->IsOpened("ExportKey")) {
            IGFD::FileDialogConfig cfg;
            cfg.path = default_browse_path;
            cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
            fd->OpenDialog("ExportKey", "Export Mesh", ".obj,.stl,.ply", cfg);
        }
        if (input.import_dialog_active && !fd->IsOpened("ImportKey")) {
            IGFD::FileDialogConfig cfg;
            cfg.path = default_browse_path;
            cfg.flags = ImGuiFileDialogFlags_None;
            cfg.sidePaneWidth = 190.0f;
            cfg.sidePane = [&](const char*, IGFD::UserDatas, bool*) {
                ImGui::Checkbox("Append to scene", &input.import_append);
                ImGui::Spacing();
                ImGui::TextWrapped("On: add the mesh as a new object at its "
                                   "own scale (OBJ/PLY only). Off: replace the "
                                   "scene. .chisel projects always replace.");
            };
            fd->OpenDialog("ImportKey", "Open File", ".chisel,.obj,.ply", cfg);
        }

        if (display_file_dialog(fd, "ExportKey")) {
            if (fd->IsOk()) {
                std::string path = fd->GetFilePathName();
                auto dot = path.find_last_of('.');
                std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
                for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
                // GPU-resident sculpting/undo leave mesh.pos stale on the CPU —
                // pull the live surface back first (save does the same).
                scene.materialize_active_cpu();
                bool ok = (ext == "stl") ? mesh->export_stl(path.c_str())
                        : (ext == "ply") ? mesh->export_ply(path.c_str())
                                         : mesh->export_obj(path.c_str());
                if (ok) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Exported: %.400s", path.c_str());
                    input.notification_timer = 3.0f;
                } else {
                    error_popup_msg = "Export failed: " + path;
                    error_popup_trigger = true;
                }
            }
            fd->Close();
            input.export_dialog_active = false;
        }
        if (!input.export_dialog_active && fd->IsOpened("ExportKey"))
            fd->Close();

        if (display_file_dialog(fd, "ImportKey")) {
            if (fd->IsOk())
                do_import_path(fd->GetFilePathName());
            fd->Close();
            input.import_dialog_active = false;
        }
        if (!input.import_dialog_active && fd->IsOpened("ImportKey"))
            fd->Close();

        // Brush-alpha custom image loader (mirrors ImportKey). Loads a grayscale
        // bitmap as a new pool entry and selects it.
        if (input.load_alpha_dialog_active && !fd->IsOpened("AlphaKey")) {
            IGFD::FileDialogConfig cfg;
            cfg.path = default_browse_path;
            cfg.flags = ImGuiFileDialogFlags_None;
            fd->OpenDialog("AlphaKey", "Load Brush Alpha", ".png,.jpg,.jpeg,.tga,.bmp", cfg);
        }
        if (display_file_dialog(fd, "AlphaKey")) {
            if (fd->IsOk()) {
                int idx = alpha_lib.load_custom(fd->GetFilePathName().c_str());
                if (idx > 0) {
                    input.active_alpha = idx;   // upload-on-change picks it up next frame
                } else {
                    error_popup_msg = "Failed to load alpha image: " + fd->GetFilePathName();
                    error_popup_trigger = true;
                }
            }
            fd->Close();
            input.load_alpha_dialog_active = false;
        }
        if (!input.load_alpha_dialog_active && fd->IsOpened("AlphaKey"))
            fd->Close();
#endif // !__EMSCRIPTEN__

        // ---- Save project (.chisel) ----
        auto do_save_project = [&](const std::string& path) {
            scene.materialize_active_cpu();  // 2b: copies the active entity's mesh + multires to disk
            ProjectData proj;
            // Persist the whole multimesh scene: every alive, committed entity
            // (skip transient INSERT previews) with its own mesh + multires.
            for (auto& up : scene.entities()) {
                if (!up || !up->alive || up->preview) continue;
                EntityRecord rec;
                rec.id           = up->id;
                rec.subdiv_level = up->subdiv_level;
                rec.mesh         = up->mesh;
                rec.multires     = up->multires;
                proj.entities.push_back(std::move(rec));
            }
            proj.active_id           = scene.active_mesh_id();
            proj.selected_ids        = scene.selected_ids();
            proj.next_id             = scene.next_id();
            proj.mirror_use_topology = scene.mirror_topology();
            proj.camera       = camera;
            proj.mirror_x     = input.mirror_x;
            proj.subdiv_level = input.subdiv_level;
            SaveResult sr = save_project(path.c_str(), proj);
            if (sr == SaveResult::OK && input.quit_after_save) {
                input.quit_after_save = false;
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            if (sr != SaveResult::OK) input.quit_after_save = false;   // stay; the error popup says why
            if (sr == SaveResult::OK) {
                current_project_path = path;
                clean_edit_mark = scene.edit_mark();
#ifdef __EMSCRIPTEN__
                // A MEMFS "save" only counts once the bytes leave as a download
                // (which also unlinks the MEMFS copy; Ctrl+S recreates it).
                web_download_file(path.c_str(),
                                  path.substr(path.find_last_of('/') + 1).c_str());
#endif
                std::snprintf(input.notification, sizeof(input.notification),
                              "Saved: %.400s", path.c_str());
                input.notification_timer = 2.0f;
            } else {
                error_popup_msg = std::string("Save failed: ") + result_string(sr) + "\n" + path;
                error_popup_trigger = true;
            }
        };

        // ---- Drag-and-drop open (prompt was answered in the key callback) ----
        if (input.drop_open_requested) {
            input.drop_open_requested = false;
            do_import_path(input.drop_path);
        }

        if (input.save_requested) {
            input.save_requested = false;
            if (current_project_path.empty()) {
                input.save_dialog_active = true;
            } else {
                do_save_project(current_project_path);
            }
        }
        if (input.save_as_requested) {
            input.save_as_requested = false;
            input.save_dialog_active = true;
        }

        // ---- Incremental save ("+"): sculpt.chisel -> sculpt_001.chisel -> _002 ----
        // A trailing _NNN is treated as the counter rather than part of the stem, so
        // repeated presses walk the series instead of nesting suffixes. Existing files
        // are skipped, so an incremental save can never overwrite a previous version
        // (on web nothing persists in MEMFS, so it just counts up from the last name).
        auto next_incremental_path = [&](const std::string& path) -> std::string {
            const std::string ext = ".chisel";
            std::string stem = path;
            if (stem.size() >= ext.size()
                && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
                stem.resize(stem.size() - ext.size());
            // A browser renames a colliding download to "bust_001 (1).chisel", and
            // opening that file makes it the current project. Strip the marker before
            // reading the counter below, or "bust_001 (1)" parses as an un-numbered
            // stem and the next save compounds into "bust_001 (1)_001".
            if (!stem.empty() && stem.back() == ')') {
                size_t op = stem.find_last_of('(');
                if (op != std::string::npos && op + 1 < stem.size() - 1) {
                    bool all_digits = true;
                    for (size_t i = op + 1; i + 1 < stem.size(); i++)
                        if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
                    if (all_digits) {
                        while (op > 0 && stem[op - 1] == ' ') op--;   // Chrome " (1)", Firefox "(1)"
                        stem.resize(op);
                    }
                }
            }
            int n = 0;
            size_t us = stem.find_last_of('_');
            if (us != std::string::npos && us + 1 < stem.size()) {
                bool all_digits = true;
                for (size_t i = us + 1; i < stem.size(); i++)
                    if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
                if (all_digits) {
                    n = std::atoi(stem.c_str() + us + 1);
                    stem.resize(us);
                }
            }
            char suffix[16];
            for (int i = n + 1; i <= 9999; i++) {
                std::snprintf(suffix, sizeof(suffix), "_%03d", i);
                std::string cand = stem + suffix + ext;
                std::error_code ec;
                if (!std::filesystem::exists(cand, ec)) return cand;
            }
            return stem + "_9999" + ext;   // series exhausted: overwrite the last one
        };

        if (input.save_incremental_requested) {
            input.save_incremental_requested = false;
            if (current_project_path.empty()) {
                // Nothing to number off yet — ask for a name first, like plain save.
                input.save_dialog_active = true;
            } else {
                do_save_project(next_incremental_path(current_project_path));
            }
        }

#ifndef __EMSCRIPTEN__
        if (input.save_dialog_active && !fd->IsOpened("SaveKey")) {
            IGFD::FileDialogConfig cfg;
            cfg.path = default_browse_path;
            cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
            fd->OpenDialog("SaveKey", "Save Project", ".chisel", cfg);
        }

        if (display_file_dialog(fd, "SaveKey")) {
            if (fd->IsOk()) {
                do_save_project(fd->GetFilePathName());
            }
            fd->Close();
            input.save_dialog_active = false;
        }
        if (!input.save_dialog_active && fd->IsOpened("SaveKey"))
            fd->Close();
#else
        // ---- Web dialogs: name prompt + browser download; picker for open ----
        // (same *_dialog_active flags as native, so the sculpt-input and hotkey
        // gating keeps working unchanged)

        // Export: pick a name, a format button writes via the normal exporter
        // into MEMFS, web_download_file hands the bytes to the browser.
        if (input.export_dialog_active && !ImGui::IsPopupOpen("Export Mesh"))
            ImGui::OpenPopup("Export Mesh");
        if (ImGui::BeginPopupModal("Export Mesh", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (!input.export_dialog_active)   // ESC cleared the flag
                ImGui::CloseCurrentPopup();
            ImGui::TextUnformatted("Download as:");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(280);
            ImGui::InputText("##export_name", web_export_name, sizeof(web_export_name));
            auto web_export = [&](const std::string& ext) {
                std::string name = web_export_name[0] ? web_export_name : "sculpt";
                // Don't double the extension if the user typed it.
                if (name.size() < ext.size()
                    || name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
                    name += ext;
                std::string fs_path = "/" + name;
                // GPU-resident sculpting/undo leave mesh.pos stale on the CPU —
                // pull the live surface back first (save does the same).
                scene.materialize_active_cpu();
                bool ok = (ext == ".stl") ? mesh->export_stl(fs_path.c_str())
                        : (ext == ".ply") ? mesh->export_ply(fs_path.c_str())
                                          : mesh->export_obj(fs_path.c_str());
                if (ok) {
                    web_download_file(fs_path.c_str(), name.c_str());
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Exported: %.400s", name.c_str());
                    input.notification_timer = 3.0f;
                } else {
                    error_popup_msg = "Export failed: " + name;
                    error_popup_trigger = true;
                }
                input.export_dialog_active = false;
                ImGui::CloseCurrentPopup();
            };
            if (ImGui::Button(".obj")) web_export(".obj");
            ImGui::SameLine();
            if (ImGui::Button(".stl")) web_export(".stl");
            ImGui::SameLine();
            if (ImGui::Button(".ply")) web_export(".ply");
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                input.export_dialog_active = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        // Save project: one name field, ".chisel" appended, straight to download.
        if (input.save_dialog_active && !ImGui::IsPopupOpen("Save Project"))
            ImGui::OpenPopup("Save Project");
        if (ImGui::BeginPopupModal("Save Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (!input.save_dialog_active)
                ImGui::CloseCurrentPopup();
            ImGui::TextUnformatted("Download project as:");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(220);
            bool commit = ImGui::InputText("##save_name", web_save_name, sizeof(web_save_name),
                                           ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::SameLine();
            ImGui::TextUnformatted(".chisel");
            if (ImGui::Button("Save") || commit) {
                std::string name = web_save_name[0] ? web_save_name : "sculpt";
                const std::string ext = ".chisel";
                if (name.size() >= ext.size()
                    && name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
                    name.resize(name.size() - ext.size());
                do_save_project("/" + name + ext);
                input.save_dialog_active = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                input.save_dialog_active = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        // Open/import: no ImGui — the browser's picker IS the dialog. The picked
        // file lands in MEMFS async (chisel_web_import_done); consume it here,
        // then free the RAM copy (current_project_path stays valid — a save
        // recreates the file before downloading it).
        if (input.import_dialog_active) {
            input.import_dialog_active = false;
            web_open_file_picker();
        }
        if (!g_web_import_path.empty()) {
            std::string path = g_web_import_path;
            g_web_import_path.clear();
            do_import_path(path);
            ::remove(path.c_str());
        }

        // Brush-alpha custom image: browser picker IS the dialog. Bytes land in MEMFS
        // (chisel_web_alpha_done); decode into the pool, select it, free the RAM copy.
        if (input.load_alpha_dialog_active) {
            input.load_alpha_dialog_active = false;
            web_open_alpha_picker();
        }
        if (!g_web_alpha_path.empty()) {
            std::string path = g_web_alpha_path;
            g_web_alpha_path.clear();
            int idx = alpha_lib.load_custom(path.c_str());
            if (idx > 0) input.active_alpha = idx;
            ::remove(path.c_str());
        }
#endif // __EMSCRIPTEN__

        // Menu > Reset settings, after its Y/N confirm.
        if (input.reset_settings_requested) {
            input.reset_settings_requested = false;
            settings_reset(input);
            std::snprintf(input.notification, sizeof(input.notification), "Settings reset to defaults");
            input.notification_timer = 2.0f;
        }

        // Save & Quit whose Save window was cancelled: forget the quit. A successful
        // save already closed the window inside do_save_project.
        if (input.quit_after_save && !input.save_dialog_active && !input.save_requested)
            input.quit_after_save = false;

        // ---- Button islands (brush selection + ops) ----
        MultiresInfo mres_info;
        mres_info.locked     = multires->locked;
        mres_info.base_level = multires->base_level;
        mres_info.lmax       = multires->base_level + (int)multires->disp.size();
        if (input.ui_classic()) {
            draw_button_islands(input, win_w, win_h, &alpha_lib, mres_info);
        } else {
            // Frosted glass: blur what has been drawn so far (the scene, cursor and any
            // bitmap dialog) for the panels to sit on. Only the ImGui pass is still to
            // come, so the copy is exactly what is behind them.
            float blur_px = 0.0f, saturate = 1.0f;
            ui_skin_backdrop_params(input, &blur_px, &saturate);
#ifdef CHISEL_BACKEND_WEBGPU
            ui_backdrop::capture(win_w, win_h, blur_px, saturate, surfTex.texture);
#else
            ui_backdrop::capture(win_w, win_h, blur_px, saturate, nullptr);
#endif
            SkinStats st;
            st.version = CHISEL_VERSION;
            st.tris = mesh->tri_count();
            st.verts = mesh->vertex_count();
            st.level = multires->current_level;
            st.fps = fps_display;
            st.project_path = current_project_path.c_str();
            draw_modern_ui(input, win_w, win_h, &alpha_lib, mres_info, st);
            draw_skin_dialogs(input, win_w, win_h, dlg);
        }

        // Key bindings window (burger menu). Top level, so every skin shares it.
        draw_keybind_window(input, win_w, win_h);
        draw_brush_props_window(input, win_w, win_h);
        draw_paint_layers_panel(input, scene, win_w, win_h);
        if (input.interaction_mode == InputState::InteractionMode::RETOPO) {
            const bool cam_moving = input.drag_mode == InputState::DragMode::ORBIT
                                 || input.drag_mode == InputState::DragMode::PAN
                                 || input.drag_mode == InputState::DragMode::ZOOM
                                 || input.mouse2_down || input.mouse3_down;
            // Bar first: it sets which stroke the cursor is over, and the overlay lights
            // it the same frame (background list, so it still draws under the bar).
            draw_retopo_bar(input, retopo, win_w, win_h);
            draw_retopo_overlay(retopo, renderer, camera, win_w, win_h,
                                !cam_moving && !screen_buffers_dirty, input.mirror_x);
        }

        // ---- Error popup ----
        if (error_popup_trigger) {
            ImGui::OpenPopup("Error");
            error_popup_trigger = false;
        }
        if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(error_popup_msg.c_str());
            if (ImGui::Button("OK", ImVec2(120, 0)))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        ImGui::Render();
#ifdef CHISEL_BACKEND_WEBGPU
        // UI pass: load the scene already drawn into the swapchain view, draw ImGui on
        // top (no depth), then present.
        if (frameView) {
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
            WGPURenderPassColorAttachment uiColor = {};
            uiColor.view       = frameView;
            uiColor.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            uiColor.loadOp     = WGPULoadOp_Load;
            uiColor.storeOp    = WGPUStoreOp_Store;
            WGPURenderPassDescriptor uiRp = {};
            uiRp.colorAttachmentCount = 1;
            uiRp.colorAttachments = &uiColor;
            WGPURenderPassEncoder uiPass = wgpuCommandEncoderBeginRenderPass(enc, &uiRp);
            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), uiPass);
            wgpuRenderPassEncoderEnd(uiPass);
            wgpuRenderPassEncoderRelease(uiPass);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(wgpuDeviceGetQueue(g_device), 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }
#ifndef __EMSCRIPTEN__
        // Native (wgpu-native) presents explicitly. On the web the browser composites
        // the canvas automatically when the requestAnimationFrame callback returns, and
        // emdawnwebgpu aborts if wgpuSurfacePresent is called at all — so skip it.
        if (surfTex.texture) wgpuSurfacePresent(g_surface);   // nothing acquired, nothing to show
#endif
        gpu::webgpu_set_default_color(nullptr);
        if (frameView && !spare_frame) wgpuTextureViewRelease(frameView);
        if (surfTex.texture) wgpuTextureRelease(surfTex.texture);
#else
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
#endif
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
    // Feed pointer position straight from the DOM in CSS-pixel space (canvas-relative),
    // bypassing GLFW's backing-store coords which desync from render/pick space across a
    // resize (the cursor-offset bug). A fresh getBoundingClientRect per event keeps it
    // exact through any resize/DPR change. chisel_set_pointer lives in input.cpp.
    EM_ASM({
        var c = Module.canvas;
        // Under pointer lock (slider drags only) clientX freezes, so the position is
        // rebuilt from movementX/Y, starting from the last free one, and stays
        // continuous when the lock lets go.
        var lx = 0;   // (separate statements: a bare comma would split EM_ASM's
        var ly = 0;   //  macro arguments)
        // Slider drags are fed their own delta, not the raw movementX: browsers on Linux
        // report spikes of hundreds of pixels in it (on lock and at random), which pinned
        // a slider to 0 or 100 with nothing between. Free pointer: the clientX step,
        // which can't spike. Locked: movementX, minus the event that crosses the lock
        // change and anything too big for one frame of a hand.
        // The ring stays where a slider drag began. The position is frozen at the grab
        // point for the whole drag. A held lock (Chromium) returns the pointer there, apart
        // from one ghost event at the unlock (see below). Firefox on Linux holds nothing,
        // so there the next real position gets an offset that continues from the grab
        // point instead of jumping to wherever the pointer went. The offset clears where
        // the hidden pointer must be honest: at the canvas edge (it can't go further),
        // over the UI (the OS arrow is showing there) and for pens and fingers (absolute).
        var sliding = false;
        var anchorX = 0;
        var anchorY = 0;
        var reanchor = false;
        var offX = 0;
        var offY = 0;
        var lastX = 0;
        var lastY = 0;
        // Where the real pointer was at the grab, whether a lock actually held during the
        // drag, and when the drag ended (see the ghost below).
        var grabX = 0;
        var grabY = 0;
        var slideLocked = false;
        var endT = 0;
        Module.chiselSliderBegin = function(x, y) {
            sliding = true; anchorX = x; anchorY = y;
            grabX = lx; grabY = ly; slideLocked = false;
        };
        Module.chiselSliderEnd = function(keep) {
            sliding = false;
            // A held lock puts the real pointer back on the grab point, so the ring already
            // continues from there; only a lockless drag (Firefox on Linux) needs the offset.
            reanchor = keep !== 0 && !slideLocked;
            endT = performance.now();
            lastX = anchorX;
            lastY = anchorY;
        };
        var sx0 = -1;
        var sLocked = false;
        var SPIKE = 400;
        document.addEventListener('pointerlockchange', function() {
            console.log('[slider] pointer lock ' + (document.pointerLockElement === c ? 'on' : 'off'));
        });
        document.addEventListener('pointerlockerror', function() {
            console.log('[slider] pointer lock refused (pointerlockerror)');
        });

        // Two-finger touch: pinch zooms, midpoint pans (applied in the main loop).
        // Only the first finger reaches the app as a click (Emscripten's GLFW), so
        // every finger is tracked here from pointer events. Registered BEFORE feed so
        // the gesture flag is already up when feed sees the second finger. Per-axis
        // maps, not {x, y} objects: same EM_ASM comma rule as above.
        var tpx = {};
        var tpy = {};
        var tn = 0;
        var tFirst = 0;       // when the first finger of this touch landed
        var tGesture = false;
        // The finger GLFW clicks with: the first one on the CANVAS. Not e.isPrimary —
        // that is the first finger on the whole page, which is the thumb whenever it
        // is already holding a modifier button (shell.html), and then the sculpting
        // finger would never move the cursor.
        var tPrimary = -1;
        var tDist = 0;
        var tMx = 0;
        var tMy = 0;
        // A second finger this soon after the first means the first one's stroke or
        // orbit was a stub of the pinch, and the app takes it back. Later than this,
        // whatever the first finger did is kept.
        var TAKE_BACK_MS = 300;
        var tPair = function() {
            var k = Object.keys(tpx);
            var dx = tpx[k[0]] - tpx[k[1]];
            var dy = tpy[k[0]] - tpy[k[1]];
            tDist = Math.sqrt(dx * dx + dy * dy);
            tMx = (tpx[k[0]] + tpx[k[1]]) * 0.5;
            tMy = (tpy[k[0]] + tpy[k[1]]) * 0.5;
        };
        c.addEventListener('pointerdown', function(e) {
            if (e.pointerType !== 'touch') return;
            if (tn === 0) { tFirst = performance.now(); tPrimary = e.pointerId; }
            tpx[e.pointerId] = e.clientX;
            tpy[e.pointerId] = e.clientY;
            tn++;
            var revert = 0;
            if (tn >= 2 && !tGesture) {
                tGesture = true;
                revert = (performance.now() - tFirst < TAKE_BACK_MS) ? 1 : 0;
            }
            if (tn >= 2) tPair();   // re-baseline: a new finger is not a pinch step
            Module._chisel_touch_count(tn, revert);
        });
        c.addEventListener('pointermove', function(e) {
            if (e.pointerType !== 'touch' || !(e.pointerId in tpx)) return;
            tpx[e.pointerId] = e.clientX;
            tpy[e.pointerId] = e.clientY;
            if (!tGesture || tn < 2) return;
            var d0 = tDist;
            var mx0 = tMx;
            var my0 = tMy;
            tPair();
            if (d0 > 1 && tDist > 1)
            {
                // Pan in the app's pixels: client px times the stage's CSS scale-down
                // (1 unless the touch UI scale is on, see shell.html #stage).
                var pk = c.clientWidth / (c.getBoundingClientRect().width || c.clientWidth || 1);
                Module._chisel_touch_pinch(tDist / d0, (tMx - mx0) * pk, (tMy - my0) * pk);
            }
        });
        var tUp = function(e) {
            if (e.pointerType !== 'touch' || !(e.pointerId in tpx)) return;
            delete tpx[e.pointerId];
            delete tpy[e.pointerId];
            tn--;
            if (tn >= 2) tPair();
            if (tn === 0) { tGesture = false; tPrimary = -1; }
            Module._chisel_touch_count(tn, 0);
        };
        c.addEventListener('pointerup', tUp);
        c.addEventListener('pointercancel', tUp);

        var feed = function(e) {
            // During a gesture no finger drives the cursor, and outside one only the
            // first canvas finger does (it is the one GLFW clicks with).
            if (e.pointerType === 'touch' && (tGesture || e.pointerId !== tPrimary)) return;
            var r = c.getBoundingClientRect();
            // Client px -> the app's pixels. 1 normally; under the touch UI scale the
            // canvas is laid out bigger than it shows (shell.html #stage), and the app
            // works in its layout size.
            var k = r.width > 0 ? c.clientWidth / r.width : 1;
            var locked = document.pointerLockElement === c;
            // Browsers deliver one pointermove per frame; the positions merged into it
            // are the stroke's real path (see InputState::path_x). The last coalesced
            // event is `e` itself, so it goes through the normal call below.
            if (e.type === 'pointermove' && e.getCoalescedEvents) {
                var list = e.getCoalescedEvents();
                var px = lx;
                var py = ly;
                for (var i = 0; i + 1 < list.length; i++) {
                    if (locked) {
                        if (!sliding) { px += (list[i].movementX || 0) * k; py += (list[i].movementY || 0) * k; }
                    } else {
                        px = (list[i].clientX - r.left) * k; py = (list[i].clientY - r.top) * k;
                    }
                    Module._chisel_path_sample(px + offX, py + offY);
                }
                if (locked) { lx = px; ly = py; }
            }
            if (locked) {
                if (!sliding) { lx += (e.movementX || 0) * k; ly += (e.movementY || 0) * k; }
                else slideLocked = true;
            } else {
                lx = (e.clientX - r.left) * k; ly = (e.clientY - r.top) * k;
                // Chromium's unlock first reports a ghost: the position the hand would have
                // taken the pointer to, with zero movement. The real pointer arrives back on
                // the grab point right after. Taking the ghost as real threw the ring a whole
                // drag's distance (traced 2026-10-21), so it is held on the grab point.
                if (slideLocked && !sliding && performance.now() - endT < 300
                    && !e.movementX && !e.movementY
                    && Math.abs(lx - grabX) + Math.abs(ly - grabY) > 2) {
                    lx = grabX; ly = grabY;
                }
            }
            var outX = anchorX;
            var outY = anchorY;
            if (!sliding) {
                // The first free position after a drag: continue from where the ring is.
                // (Still locked = Chromium hasn't let go yet; lx is still on the grab
                // point there, so the offset waits for the real position.)
                if (reanchor && !locked) {
                    reanchor = false;
                    offX = lastX - lx;
                    offY = lastY - ly;
                }
                var edge = lx <= 0.5 || ly <= 0.5
                        || lx >= c.clientWidth - 1.5 || ly >= c.clientHeight - 1.5;
                if (e.pointerType !== 'mouse' || c.style.cursor !== 'none' || edge) {
                    offX = 0;
                    offY = 0;
                }
                outX = lx + offX;
                outY = ly + offY;
            }
            lastX = outX;
            lastY = outY;
            var sdx = 0;
            if (locked !== sLocked) {
                sLocked = locked;
            } else if (locked) {
                sdx = (e.movementX || 0) * k;
                if (Math.abs(sdx) > SPIKE) {
                    console.log('[slider] dropped a movementX spike: ' + e.movementX);
                    sdx = 0;
                }
            } else if (sx0 >= 0) {
                sdx = (e.clientX - sx0) * k;
            }
            sx0 = e.clientX;
            Module._chisel_set_pointer(outX, outY, sdx);
        };
        c.addEventListener('pointermove', feed);
        c.addEventListener('pointerdown', feed);
        // Keep a mouse drag's moves coming once it leaves the canvas (or itch's
        // iframe), so an orbit or a stroke carries on past it. Released by the browser
        // on pointerup. Pens are left alone.
        c.addEventListener('pointerdown', function(e) {
            if (e.pointerType === 'mouse' && c.setPointerCapture) {
                try { c.setPointerCapture(e.pointerId); } catch (err) {}
            }
        });

        // Stylus force, which GLFW never surfaces — this is the web build's entire
        // tablet path (see the __EMSCRIPTEN__ branch of tablet.cpp). Guarded on
        // pointerType because a mouse reports a fixed 0.5 while held, which is
        // indistinguishable from a pen at half force.
        //
        // Bound to window in the CAPTURE phase, not to the canvas: capture runs
        // window -> target, so this sees every pointer event that exists, including
        // ones an upstream handler swallows with stopPropagation before they reach
        // the canvas (itch serves us inside its own iframe wrapper). It must stay a
        // SEPARATE handler from feed() above — feed passes e.movementX to
        // chisel_set_pointer, which *accumulates* into the slider drag, so running
        // the same event through both listeners would double every slider's speed.
        var feedPen = function(e) {
            if (e.pointerType === 'pen')
                Module._chisel_set_pen(1, e.pressure);
        };
        window.addEventListener('pointermove', feedPen, true);
        window.addEventListener('pointerdown', feedPen, true);

        // Last-chance settings write. visibilitychange->hidden is the reliable one on
        // mobile (a backgrounded tab may never get pagehide before it is discarded);
        // pagehide covers desktop navigation away. Both are cheap and idempotent.
        window.addEventListener('pagehide', function() { Module._chisel_flush_settings(); });
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'hidden') Module._chisel_flush_settings();
        });

        // Keep app-owned shortcuts away from the browser: Emscripten's GLFW only
        // preventDefault()s Backspace/Tab, so F1 opened help, Ctrl+D bookmarked,
        // Ctrl+S saved the page, … The app still receives these through GLFW's own
        // window-level listener (preventDefault stops the browser default, not
        // propagation). Ctrl+Shift combos stay untouched except Shift+Z (redo) so
        // devtools (Ctrl+Shift+I) keep working. Matched on e.code (physical key),
        // not e.key: under a non-Latin layout (e.g. Greek) Ctrl+D arrives as
        // e.key='δ' and a key-based match lets the browser bookmark anyway.
        //
        // Rebound shortcuts: Ctrl/Alt combos and F-keys are checked against the live
        // bindings (chisel_key_wanted), mapping e.code to the GLFW code — letters and
        // digits share their ASCII value, F1 is 290. Plain keys are left alone: keeping
        // the browser off them would also eat the keypress the Save dialog types with.
        // While the Key bindings window is open it wants every key.
        window.addEventListener('keydown', function(e) {
            var code = e.code || "";
            var ctrl = e.ctrlKey || e.metaKey;
            if (e.key === 'F1'
                || (ctrl && !e.altKey && (!e.shiftKey || code === 'KeyZ')
                    && ['KeyD','KeyS','KeyO','KeyI','KeyZ','KeyY','KeyP'].indexOf(code) !== -1)) {
                e.preventDefault();
                return;
            }
            var gk = -1, fkey = false;
            if (/^Key[A-Z]$/.test(code)) gk = code.charCodeAt(3);
            else if (/^Digit[0-9]$/.test(code)) gk = code.charCodeAt(5);
            else if (/^F([1-9]|1[0-2])$/.test(code)) { gk = 289 + parseInt(code.substring(1), 10); fkey = true; }
            // -2 is no key at all, so this is true only while the window is listening.
            var listening = Module._chisel_key_wanted(-2, 0, 0, 0);
            if (listening || ((ctrl || e.altKey || fkey) && gk !== -1
                    && Module._chisel_key_wanted(gk, ctrl ? 1 : 0, e.shiftKey ? 1 : 0, e.altKey ? 1 : 0)))
                e.preventDefault();
        }, true);
    });

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
