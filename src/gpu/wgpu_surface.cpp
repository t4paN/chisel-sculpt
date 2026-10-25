// WebGPU startup and the window's surface: instance -> surface -> adapter -> device ->
// the gpu:: seam, plus the per-frame helpers main.cpp's frame calls (reconfigure,
// depth target, spare colour target). Moved out of main.cpp as is.
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
#include "gpu/gpu.h"
#include "gpu/wgpu_surface.h"
#include "bench.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
} // namespace

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
bool wgpu_init(GLFWwindow* window, int fbw, int fbh) {
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
    if (!instance) { std::fprintf(stderr, "wgpuCreateInstance failed\n"); return false; }
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
    if (!g_surface) { std::fprintf(stderr, "createSurface failed\n"); return false; }
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
    if (!ar.ok) { std::fprintf(stderr, "no WebGPU adapter\n"); return false; }

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
    if (!dr.ok) { std::fprintf(stderr, "no WebGPU device\n"); return false; }
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
        std::fprintf(stderr, "surfaceGetCapabilities failed\n"); return false;
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
    return true;
}
#endif // CHISEL_BACKEND_WEBGPU
