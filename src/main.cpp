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
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpu-native extensions (wgpuDevicePoll / ProcessEvents)
#endif
#include "gpu/gpu.h"       // device_from_webgpu + surface-format/depth setters
#else
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif
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
#include "imgui.h"
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
WGPUTexture  g_depth_tex  = nullptr;
WGPUTextureView g_depth_view = nullptr;
WGPUTextureFormat g_surface_fmt = WGPUTextureFormat_BGRA8Unorm;
static const WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth24Plus;

void configureSurface(int w, int h) {
    WGPUSurfaceConfiguration cfg = {};
    cfg.device      = g_device;
    cfg.format      = g_surface_fmt;
    cfg.usage       = WGPUTextureUsage_RenderAttachment;
    cfg.width       = (uint32_t)w;
    cfg.height      = (uint32_t)h;
    cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
    cfg.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(g_surface, &cfg);
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

// Wrap cursor at screen edges. Returns true if cursor was wrapped.
// Non-static: insert_controller.cpp forward-declares and calls it.
bool wrap_cursor(GLFWwindow* window, InputState& input, int win_w, int win_h) {
    double mx = input.mouse_x;
    double my = input.mouse_y;
    bool wrapped = false;
    if (mx <= 0) { mx = win_w - 2; wrapped = true; }
    else if (mx >= win_w - 1) { mx = 1; wrapped = true; }
    if (my <= 0) { my = win_h - 2; wrapped = true; }
    else if (my >= win_h - 1) { my = 1; wrapped = true; }
    if (wrapped) {
        glfwSetCursorPos(window, mx, my);
        input.mouse_x = mx;
        input.prev_mouse_x = mx;
        input.mouse_y = my;
        input.prev_mouse_y = my;
    }
    return wrapped;
}

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

int main(int argc, char* argv[]) {
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
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to init GLFW\n");
        return 1;
    }

#ifdef CHISEL_BACKEND_WEBGPU
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // WebGPU owns the surface; no GL context
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    // WM_CLASS must match StartupWMClass=Chisel in chisel.desktop so the running
    // window groups under the launcher entry in the taskbar/dock (X11). No window
    // manager in a browser — these hints don't exist in Emscripten's GLFW.
#if !defined(__EMSCRIPTEN__)
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "Chisel");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "Chisel");
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
    GLFWwindow* window = glfwCreateWindow(init_w, init_h, "Chisel", nullptr, nullptr);
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
    WGPUInstance instance = wgpuCreateInstance(nullptr);
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
    WGPUSurfaceSourceXlibWindow x11 = {};
    x11.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
    x11.display = glfwGetX11Display();
    x11.window  = (uint64_t)glfwGetX11Window(window);
    sd.nextInChain = &x11.chain;
#endif
    g_surface = wgpuInstanceCreateSurface(instance, &sd);
    if (!g_surface) { std::fprintf(stderr, "createSurface failed\n"); return 1; }

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
    wgpuAdapterGetLimits(ar.adapter, &supported);
    WGPULimits limits = WGPU_LIMITS_INIT;                     // all baseline defaults
    limits.maxBufferSize               = supported.maxBufferSize;
    limits.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
    WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    ddesc.requiredLimits = &limits;
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
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    gpu::webgpu_set_surface_format(g_surface_fmt);
    configureSurface(fbw, fbh);
    makeDepth(fbw, fbh);
    std::printf("Chisel v0.1 (WebGPU)\n");
#else
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // Load OpenGL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "Failed to load OpenGL\n");
        return 1;
    }

    std::printf("Chisel v0.1\n");
    std::printf("OpenGL %s\n", glGetString(GL_VERSION));
    std::printf("Renderer: %s\n", glGetString(GL_RENDERER));

    chisel_init_gl_debug();   // synchronous KHR_debug output (CHISEL_DEBUG builds only)
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
        compute.init_crease();
        compute.init_pinch();
        compute.init_move();
        compute.init_limb();
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
#ifdef __EMSCRIPTEN__
    g_settings_input = &input;   // arms the pagehide flush above
#endif
    setup_input_callbacks(window, &input);
    setup_char_callback(window);

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
#endif

    Renderer renderer;
    renderer.init();

    // Pen tablet (X11/XInput2, dlopen'd libXi). No-op if absent. Detects hotplug.
    Tablet tablet;
    tablet.init();
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
    // icosahedron: it locks at level 0 and subdiv_level has to say so, or the HUD
    // reports a level the multires stack never had.
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

    // Tick-driven voxel merge: non-null while a merge job is in flight (advanced one
    // budgeted step per frame so the window stays responsive). See sdf.h / CHANGES.
    VoxelMergeJob* vmerge_job = nullptr;

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
    // One-shot arming for the "symmetry is off" notice below, so a mesh that has
    // been turned off the mirror plane says so once rather than once per frame.
    bool mirror_warn_armed = true;
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
        win_w = EM_ASM_INT({ return (Module.canvas.clientWidth  || window.innerWidth)  | 0; });
        win_h = EM_ASM_INT({ return (Module.canvas.clientHeight || window.innerHeight) | 0; });
#else
        glfwGetFramebufferSize(window, &win_w, &win_h);
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
        if (win_w != fbw || win_h != fbh) {
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

        glfwPollEvents();
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

        mesh = &scene.active_mesh();
        multires = &scene.active_multires();

#ifdef CHISEL_BACKEND_WEBGPU
        ImGui_ImplWGPU_NewFrame();
#else
        ImGui_ImplOpenGL3_NewFrame();
#endif
        ImGui_ImplGlfw_NewFrame();
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
#endif
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
        bool dialog_open = input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.voxel_merge_confirm_pending || input.drop_confirm_pending || input.help_popup_open;
        if ((imgui_wants_mouse || dialog_open) && app_state != AppState::SCULPTING) {
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
                              "Ctrl+Z reverts the %s - %d stroke%s left",
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

        // Handle borderless toggle (Space)
        if (input.fullscreen_toggle_requested) {
            input.fullscreen_toggle_requested = false;
            if (!input.is_fullscreen) {
                // Remove titlebar, keep taskbar
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
                input.is_fullscreen = true;
            } else {
                // Restore titlebar
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);
                input.is_fullscreen = false;
            }
        }

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
        bool camera_moving = input.drag_mode == InputState::DragMode::ORBIT
                          || input.drag_mode == InputState::DragMode::PAN
                          || input.drag_mode == InputState::DragMode::ZOOM
                          || input.mouse2_down || input.mouse3_down;

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
            if (std::fabs(scroll) > 0.01f) {
                camera.zoom(scroll);
                screen_buffers_dirty = true;
            }
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
                    auto vcol = [&](uint32_t v) {
                        return v < pm.color.size() ? pm.color[v] : 0xFFFFFFFFu;
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
        // EFFECTIVE X symmetry for a dab on the active mesh. Every mirror pass in
        // brush.cpp reads DabContext::mirror_x, which comes from here, so this one
        // test disarms all four at once. Two of them (mirror_project,
        // smooth_mirror_apply) write absolute positions derived from the world x=0
        // plane; on a mesh a Select-mode spin has turned off that plane they do not
        // mirror wrong, they TEAR. So symmetry is dropped rather than applied, and
        // said once. Mesh::mirror_world_symmetric() caches against topo_version, so
        // this is a stamp compare on all but the first call after a change.
        auto mirror_effective = [&]() -> bool {
            if (!input.mirror_x) { mirror_warn_armed = true; return false; }
            if (mesh->mirror_world_symmetric()) { mirror_warn_armed = true; return true; }
            if (mirror_warn_armed) {
                mirror_warn_armed = false;
                std::snprintf(input.notification, sizeof(input.notification),
                              "X symmetry off — this mesh is turned off the mirror plane");
                input.notification_timer = 2.5f;
            }
            return false;
        };

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
                const int target = L_max - 1;
                scene.materialize_active_cpu();  // projection/cascade read disp/base

                // Bake first, then drop: the same inverse-Loop projection a level-down
                // does, so the form survives into the layer below and only the
                // residual detail dies with the layer.
                ProjectionStats ps = project_down_to_level(*multires, target);
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
        if (input.remesh_requested) {
            input.remesh_requested = false;
            input.remesh_in_progress = true;

            scene.materialize_active_cpu();  // 2b: remesh reads the live surface (mesh.pos)

            // Adaptive-remesh guard (density spec Q2): a red-heavy field at a
            // small fine-mult can explode tri counts the same way keymashed
            // subdivision does. Same predicted-size check, same refusal toast.
            bool remesh_refused = false;
            if (!mesh->density.empty()) {
                const uint64_t tris_pred = predict_adaptive_tris(
                    *mesh, 0.0f, input.density_coarse_mult, input.density_fine_mult);
                const uint64_t bytes_pred = tris_pred * 12ull;  // index/CSR SSBOs
                const gpu::DeviceLimits dl = gpu::device_limits();
                const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                                      ? dl.max_buffer_size : dl.max_storage_binding_size;
                if (bytes_pred > budget) {
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Adaptive remesh would exceed GPU limits (%.1fM tris)",
                                  (double)tris_pred / 1e6);
                    input.notification_timer = 4.0f;
                    std::printf("[remesh] refused adaptive: %.1fM predicted tris needs %llu MB, device grants %llu MB\n",
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
                                        input.density_fine_mult);

            if (result.success) {
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
                vmerge_job = voxel_merge_begin(scene, compute,
                                               input.voxel_merge_resolution,
                                               input.voxel_merge_mirror,
                                               input.voxel_merge_surface_nets,
                                               input.voxel_merge_subtract);
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

        // IDLE → SCULPTING
        if (app_state == AppState::IDLE && input.drag_mode == InputState::DragMode::SCULPT) {
            input.sculpting = true;

            // Active entity lives in the working buffers at offset 0; the brush
            // dispatches over its full [0, vertex_count) range.
            brush_stroke.begin(renderer, camera,
                               (float)input.mouse_x, (float)input.mouse_y,
                               input.brush_size,
                               win_w, win_h, mesh->vertex_count(), *multires,
                               input.current_brush, scene.active_mesh_id(),
                               scene.active_entity().multires_gpu,
                               !screen_buffers_dirty);
            brush_stroke.cursor_hist_count = 1;
            brush_stroke.cursor_hist_x[0] = (float)input.mouse_x;
            brush_stroke.cursor_hist_y[0] = (float)input.mouse_y;
            app_state = AppState::SCULPTING;
        }

        // SCULPTING → IDLE (pen-up). finalize is a small per-frame state machine:
        // it drains in-flight dab readbacks, kicks the pen-up buffer reads, and
        // commits when they land — one call on GL, a few frames on webgpu/web.
        // app_state stays SCULPTING while it runs, which blocks new strokes and the
        // IDLE-only ops (undo, delete, merge, …) exactly like an active stroke; a
        // re-press during the reconcile starts its stroke from the IDLE transition
        // a frame or two later.
        if (app_state == AppState::SCULPTING
            && (brush_stroke.reconciling()
                || input.drag_mode != InputState::DragMode::SCULPT)) {
            input.sculpting = false;
            DabContext fctx { renderer, camera, compute, *mesh, *multires, input,
                              win_w, win_h, brush_stroke.vertex_count, input.brush_size,
                              mirror_effective() };
            bool had_update = false;
            if (brush_stroke.finalize(fctx, *mesh, scene.active_undo(), *multires,
                                      scene.active_entity().multires_gpu,
                                      renderer, input.current_brush,
                                      input.autosmooth, had_update)) {
                print_undo_top("stroke-commit");
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
            }
            if (input.ctrl_held && input.mouse2_down) {
                camera.zoom(-dy * 0.05f);
                if (dy != 0.0f) screen_buffers_dirty = true;
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
                                || input.drop_confirm_pending
                                || input.export_dialog_active || input.import_dialog_active
                                || input.save_dialog_active;
                int dir = 0;
                if (input.interaction_mode == InputState::InteractionMode::SELECT
                    && !modal && !input.ctrl_held && !scene.selected_ids().empty()) {
                    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) dir += 1;
                    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) dir -= 1;
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
            if (input.undo_requested && rescue.valid() && rescue.edits_left() > 0
                && !scene.active_undo().can_undo() && !xforms.can_undo()) {
                input.undo_requested = false;
                const char* what = rescue.op;
                snapshot_restore(rescue, scene);
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
            if ((input.undo_requested || input.redo_requested) && !undo_allowed) {
                input.undo_requested = false;
                input.redo_requested = false;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Undo only works in Edit mode (1)");
                input.notification_timer = 1.5f;
            }
            if (input.undo_requested) {
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
                    cascade_active(scene.active_undo().undo(scene.active_entity(), scene));
                    print_undo_top("ctrl-z");
                }
                screen_buffers_dirty = true;
            }
            if (input.redo_requested) {
                input.redo_requested = false;
                if (xform_redo_next) {
                    xforms.redo(scene);
                    if (xforms.last_was_spin()) refresh_active_gpu_residency();
                    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Redid object %s", xforms.last_label());
                    input.notification_timer = 1.5f;
                } else {
                    cascade_active(scene.active_undo().redo(scene.active_entity(), scene));
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

            if (brush_stroke.is_active() && !wrapped) {
                bool is_smooth = input.is_smooth_active() ||
                                 input.current_brush == BrushType::SMOOTH;
                bool is_move = input.current_brush == BrushType::MOVE;
                bool is_limb = input.current_brush == BrushType::LIMB;
                // Both grab brushes capture once and drive per-frame: one dab at the
                // live cursor, no spline interpolation, no dab-spacing advance.
                bool is_grab = is_move || is_limb;

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

                // Push cursor position into history for spline interpolation
                float cur_x = (float)input.mouse_x;
                float cur_y = (float)input.mouse_y;
                if (brush_stroke.cursor_hist_count < BrushStroke::CURSOR_HIST_SIZE) {
                    int i = brush_stroke.cursor_hist_count++;
                    brush_stroke.cursor_hist_x[i] = cur_x;
                    brush_stroke.cursor_hist_y[i] = cur_y;
                } else {
                    for (int i = 0; i < 3; i++) {
                        brush_stroke.cursor_hist_x[i] = brush_stroke.cursor_hist_x[i+1];
                        brush_stroke.cursor_hist_y[i] = brush_stroke.cursor_hist_y[i+1];
                    }
                    brush_stroke.cursor_hist_x[3] = cur_x;
                    brush_stroke.cursor_hist_y[3] = cur_y;
                }

                float dab_dx = cur_x - brush_stroke.last_dab_x;
                float dab_dy = cur_y - brush_stroke.last_dab_y;
                float dab_dist = std::sqrt(dab_dx*dab_dx + dab_dy*dab_dy);
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

                int dab_count = 0;
                if (is_grab || brush_stroke.phase == StrokePhase::BEGIN) {
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

                bool use_spline = !is_grab && dab_count > 1
                    && brush_stroke.cursor_hist_count >= 3;

                float p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y;
                if (use_spline) {
                    int hc = brush_stroke.cursor_hist_count;
                    p0x = brush_stroke.cursor_hist_x[std::max(0, hc-3)];
                    p0y = brush_stroke.cursor_hist_y[std::max(0, hc-3)];
                    p1x = brush_stroke.last_dab_x;
                    p1y = brush_stroke.last_dab_y;
                    p2x = cur_x;
                    p2y = cur_y;
                    p3x = p2x + (p2x - p1x);
                    p3y = p2y + (p2y - p1y);
                }

                float step_x = 0.0f, step_y = 0.0f;
                if (!is_grab && dab_count > 0 && dab_dist > 1e-6f) {
                    step_x = dab_dx / dab_dist * spacing;
                    step_y = dab_dy / dab_dist * spacing;
                }

                brush_stroke.gpu_dirty.clear();

                for (int dab_i = 0; dab_i < dab_count; dab_i++) {
                    float dab_x, dab_y;
                    if (is_grab || dab_count == 1) {
                        dab_x = cur_x;
                        dab_y = cur_y;
                    } else if (use_spline) {
                        float t = (float)(dab_i + 1) / (float)dab_count;
                        float t2 = t * t;
                        float t3 = t2 * t;
                        dab_x = 0.5f * ((2.0f*p1x) +
                                 (-p0x + p2x) * t +
                                 (2.0f*p0x - 5.0f*p1x + 4.0f*p2x - p3x) * t2 +
                                 (-p0x + 3.0f*p1x - 3.0f*p2x + p3x) * t3);
                        dab_y = 0.5f * ((2.0f*p1y) +
                                 (-p0y + p2y) * t +
                                 (2.0f*p0y - 5.0f*p1y + 4.0f*p2y - p3y) * t2 +
                                 (-p0y + 3.0f*p1y - 3.0f*p2y + p3y) * t3);
                    } else {
                        dab_x = brush_stroke.last_dab_x + step_x * (dab_i + 1);
                        dab_y = brush_stroke.last_dab_y + step_y * (dab_i + 1);
                    }

                    DabContext ctx { renderer, camera, compute, *mesh, *multires, input, win_w, win_h,
                                     brush_stroke.vertex_count, eff_brush_size,
                                     mirror_effective() };

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
                                                    input.mirror_x,
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

                if (!is_grab) {
                    if (use_spline) {
                        brush_stroke.last_dab_x = cur_x;
                        brush_stroke.last_dab_y = cur_y;
                    } else {
                        brush_stroke.last_dab_x += step_x * dab_count;
                        brush_stroke.last_dab_y += step_y * dab_count;
                    }
                }

                brush_stroke.needs_mesh_update = true;
                } // dab_count > 0

                // Runs every stroke frame (not just dab frames): drains landed async
                // dab readbacks — snap/normals bookkeeping for dabs kicked 1–2 frames
                // ago — then dispatches partial normals for whatever landed.
                {
                    DabContext ctx { renderer, camera, compute, *mesh, *multires, input, win_w, win_h,
                                     brush_stroke.vertex_count, eff_brush_size,
                                     mirror_effective() };
                    brush_stroke.post_frame(ctx);
                }
            }
        }

        // ---- Cursor visibility ----
        bool non_edit_mode = input.interaction_mode != InputState::InteractionMode::EDIT;
        // Hide the OS cursor while sculpting (we draw our own brush cursor).
        bool show_os_cursor = input.quit_requested || input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.remesh_confirm_pending || input.voxel_merge_confirm_pending || input.drop_confirm_pending || input.help_popup_open || imgui_wants_mouse || non_edit_mode;
#ifndef __EMSCRIPTEN__
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
        WGPUTextureView frameView = surfTex.texture
            ? wgpuTextureCreateView(surfTex.texture, nullptr) : nullptr;
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

        // Brush cursor — draw at locked position during slider, normal position otherwise
        if (!input.quit_requested && !input.export_dialog_active && !input.import_dialog_active && !input.save_dialog_active && !input.remesh_confirm_pending && !input.voxel_merge_confirm_pending && !input.drop_confirm_pending && !input.help_popup_open && !imgui_wants_mouse && !non_edit_mode) {
            float cursor_x, cursor_y;
            if (input.slider_mode != InputState::SliderMode::NONE) {
                cursor_x = (float)input.slider_start_x;
                cursor_y = (float)input.slider_start_y;
            } else {
                cursor_x = (float)input.mouse_x;
                cursor_y = (float)input.mouse_y;
            }
            if (input.color_pick_active) {
                // Armed colour picker: eyedropper glyph instead of the brush
                // ring — the ring's size/hardness reading is meaningless here.
                draw_pick_cursor(cursor_x, cursor_y, input.on_model);
            } else {
                renderer.draw_cursor(
                    camera,
                    cursor_x, cursor_y,
                    input.brush_size,
                    input.cursor_nx, input.cursor_ny, input.cursor_nz,
                    input.brush_hardness,
                    win_w, win_h, input.on_model);
            }
        }

        // ---- Overlays ----
        if (input.quit_requested)
            draw_quit_dialog(text, win_w, win_h);
        if (input.drop_confirm_pending)
            draw_drop_confirm(text, input.drop_path, win_w, win_h);
        if (input.remesh_confirm_pending)
            draw_remesh_confirm(text, snapshot_levels_to_bake(scene), win_w, win_h);
        if (input.drop_level_confirm_pending)
            draw_drop_level_confirm(text, multires->base_level + (int)multires->disp.size(),
                                    win_w, win_h);
        if (input.voxel_merge_confirm_pending) {
            // Count unselected (red) committed entities — candidate cutters for subtract.
            int n_unselected = 0;
            for (const auto& up : scene.entities()) {
                if (!up || !up->alive || up->preview) continue;
                bool sel = false;
                for (uint32_t sid : scene.selected_ids())
                    if (sid == up->id) { sel = true; break; }
                if (!sel) n_unselected++;
            }
            // Density field on any selected mesh → the merge output will carry
            // the heatmap; offer the chained adaptive remesh line.
            bool merge_has_density = false;
            for (uint32_t sid : scene.selected_ids()) {
                const MeshEntity* e = scene.find_entity(sid);
                if (e && e->alive && !e->mesh.density.empty()) { merge_has_density = true; break; }
            }
            draw_voxel_merge_confirm(text, input.voxel_merge_resolution,
                                     (int)scene.selected_ids().size(), n_unselected,
                                     input.voxel_merge_surface_nets,
                                     merge_has_density, input.voxel_merge_adaptive,
                                     snapshot_levels_to_bake(scene), win_w, win_h);
        }
        if (input.remesh_in_progress)
            draw_remesh_progress(text, win_w, win_h);
        if (input.voxel_merge_in_progress)
            draw_voxel_merge_progress(text, win_w, win_h,
                                      vmerge_job ? voxel_merge_progress(*vmerge_job) : 0.0f);
        if (input.toolbar_visible)
            draw_toolbar(text, input, mesh->tri_count(), mesh->vertex_count(), CHISEL_VERSION,
                         current_project_path.c_str(), win_w, win_h);
        if (input.slider_mode != InputState::SliderMode::NONE)
            draw_slider(text, input, win_w, win_h);
        if (input.interaction_mode == InputState::InteractionMode::SELECT)
            draw_mode_indicator(text, "SELECT", win_w, win_h);
        else if (input.interaction_mode == InputState::InteractionMode::INSERT)
            draw_mode_indicator(text, "INSERT", win_w, win_h);
        draw_notification(text, input, win_w, win_h);
        if (input.show_fps)
            draw_fps(text, fps_display, win_w, win_h);

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

        if (fd->Display("ExportKey", ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
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

        if (fd->Display("ImportKey", ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
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
        if (fd->Display("AlphaKey", ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
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
            if (sr == SaveResult::OK) {
                current_project_path = path;
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

        if (fd->Display("SaveKey", ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
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

        // ---- Button islands (brush selection + ops) ----
        MultiresInfo mres_info;
        mres_info.locked     = multires->locked;
        mres_info.base_level = multires->base_level;
        mres_info.lmax       = multires->base_level + (int)multires->disp.size();
        draw_button_islands(input, win_w, win_h, &alpha_lib, mres_info);

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
        wgpuSurfacePresent(g_surface);
#endif
        gpu::webgpu_set_default_color(nullptr);
        if (frameView) wgpuTextureViewRelease(frameView);
        if (surfTex.texture) wgpuTextureRelease(surfTex.texture);
#else
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
#endif
        input.end_frame();
    };

#ifdef __EMSCRIPTEN__
    // Feed pointer position straight from the DOM in CSS-pixel space (canvas-relative),
    // bypassing GLFW's backing-store coords which desync from render/pick space across a
    // resize (the cursor-offset bug). A fresh getBoundingClientRect per event keeps it
    // exact through any resize/DPR change. chisel_set_pointer lives in input.cpp.
    EM_ASM({
        var c = Module.canvas;
        var feed = function(e) {
            var r = c.getBoundingClientRect();
            // movementX drives the slider drags (consistent CSS-scale deltas even
            // under pointer lock, where clientX freezes).
            Module._chisel_set_pointer(e.clientX - r.left, e.clientY - r.top,
                                       e.movementX || 0);
        };
        c.addEventListener('pointermove', feed);
        c.addEventListener('pointerdown', feed);

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
        window.addEventListener('keydown', function(e) {
            var code = e.code || "";
            var ctrl = e.ctrlKey || e.metaKey;
            if (e.key === 'F1'
                || (ctrl && !e.altKey && (!e.shiftKey || code === 'KeyZ')
                    && ['KeyD','KeyS','KeyO','KeyI','KeyZ','KeyY','KeyP'].indexOf(code) !== -1))
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
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
