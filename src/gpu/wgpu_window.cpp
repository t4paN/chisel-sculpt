// Stage 2 Step 2 probe: GLFW window -> WGPUSurface (X11) -> device -> configure
// -> render pass that clears to a color -> present. No mesh, no pipeline yet;
// this only proves the on-screen swapchain path on wgpu-native v29.
//
// Run live (window stays open until closed) for a screenshot, or set
// CHISEL_PROBE_FRAMES=N to auto-exit after N presented frames (CI/headless-ish).
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <webgpu/webgpu.h>

#include <cstdio>
#include <cstdlib>

// ---- async adapter/device request, same pump pattern as the Step 1 probe ----
struct AdapterResult { WGPUAdapter adapter = nullptr; bool done = false; bool ok = false; };
struct DeviceResult  { WGPUDevice  device  = nullptr; bool done = false; bool ok = false; };

static void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                      WGPUStringView msg, void* ud1, void*) {
    auto* r = static_cast<AdapterResult*>(ud1);
    r->done = true;
    if (status == WGPURequestAdapterStatus_Success) { r->adapter = adapter; r->ok = true; }
    else std::printf("[win] adapter failed: status=%d msg=%.*s\n",
                     (int)status, (int)msg.length, msg.data ? msg.data : "");
}
static void onDevice(WGPURequestDeviceStatus status, WGPUDevice device,
                     WGPUStringView msg, void* ud1, void*) {
    auto* r = static_cast<DeviceResult*>(ud1);
    r->done = true;
    if (status == WGPURequestDeviceStatus_Success) { r->device = device; r->ok = true; }
    else std::printf("[win] device failed: status=%d msg=%.*s\n",
                     (int)status, (int)msg.length, msg.data ? msg.data : "");
}

// ---- live framebuffer size (resize -> reconfigure) ----
static int g_fbw = 1280, g_fbh = 720;
static bool g_resized = false;
static void onFramebufferSize(GLFWwindow*, int w, int h) {
    g_fbw = w; g_fbh = h; g_resized = true;
}

static void configureSurface(WGPUSurface surface, WGPUDevice device,
                             WGPUTextureFormat format, int w, int h) {
    WGPUSurfaceConfiguration cfg = {};
    cfg.device      = device;
    cfg.format      = format;
    cfg.usage       = WGPUTextureUsage_RenderAttachment;
    cfg.width       = (uint32_t)w;
    cfg.height      = (uint32_t)h;
    cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
    cfg.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(surface, &cfg);
}

int main() {
    if (!glfwInit()) { std::printf("[win] glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // no GL context — WebGPU owns the surface
    GLFWwindow* win = glfwCreateWindow(g_fbw, g_fbh, "Chisel WebGPU — Stage 2 Step 2", nullptr, nullptr);
    if (!win) { std::printf("[win] glfwCreateWindow failed\n"); glfwTerminate(); return 1; }
    glfwGetFramebufferSize(win, &g_fbw, &g_fbh);
    glfwSetFramebufferSizeCallback(win, onFramebufferSize);

    WGPUInstance instance = wgpuCreateInstance(nullptr);
    if (!instance) { std::printf("[win] wgpuCreateInstance failed\n"); return 1; }

    // ---- surface from the X11 window ----
    WGPUSurfaceSourceXlibWindow x11 = {};
    x11.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
    x11.display = glfwGetX11Display();
    x11.window  = (uint64_t)glfwGetX11Window(win);
    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = &x11.chain;
    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    if (!surface) { std::printf("[win] createSurface failed\n"); return 2; }
    std::printf("[win] surface created (X11 %dx%d)\n", g_fbw, g_fbh);

    // ---- adapter (compatible with this surface) ----
    AdapterResult ar;
    WGPURequestAdapterOptions aopt = {};
    aopt.compatibleSurface = surface;
    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode = WGPUCallbackMode_AllowProcessEvents;
    acb.callback = onAdapter;
    acb.userdata1 = &ar;
    wgpuInstanceRequestAdapter(instance, &aopt, acb);
    for (int i = 0; i < 200 && !ar.done; ++i) wgpuInstanceProcessEvents(instance);
    if (!ar.ok) { std::printf("[win] no adapter\n"); return 2; }

    // ---- device + queue ----
    DeviceResult dr;
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = onDevice;
    dcb.userdata1 = &dr;
    wgpuAdapterRequestDevice(ar.adapter, nullptr, dcb);
    for (int i = 0; i < 200 && !dr.done; ++i) wgpuInstanceProcessEvents(instance);
    if (!dr.ok) { std::printf("[win] no device\n"); return 2; }
    WGPUDevice device = dr.device;
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    std::printf("[win] device + queue ready\n");

    // ---- pick a surface format from capabilities, then configure ----
    WGPUSurfaceCapabilities caps = {};
    if (wgpuSurfaceGetCapabilities(surface, ar.adapter, &caps) != WGPUStatus_Success
        || caps.formatCount == 0) {
        std::printf("[win] surfaceGetCapabilities failed / no formats\n"); return 2;
    }
    WGPUTextureFormat format = caps.formats[0];  // preferred format
    std::printf("[win] surface format = %d (of %zu)\n", (int)format, caps.formatCount);
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    configureSurface(surface, device, format, g_fbw, g_fbh);

    // ---- frame loop ----
    const char* framesEnv = std::getenv("CHISEL_PROBE_FRAMES");
    const long maxFrames = framesEnv ? std::atol(framesEnv) : -1;  // -1 = run until closed
    long frame = 0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (g_resized) { configureSurface(surface, device, format, g_fbw, g_fbh); g_resized = false; }

        WGPUSurfaceTexture st = {};
        wgpuSurfaceGetCurrentTexture(surface, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
            && st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            // Outdated/Lost: reconfigure and try next frame.
            if (st.texture) wgpuTextureRelease(st.texture);
            configureSurface(surface, device, format, g_fbw, g_fbh);
            continue;
        }

        WGPUTextureView view = wgpuTextureCreateView(st.texture, nullptr);

        WGPURenderPassColorAttachment color = {};
        color.view       = view;
        color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        color.loadOp     = WGPULoadOp_Clear;
        color.storeOp    = WGPUStoreOp_Store;
        color.clearValue = WGPUColor{0.10, 0.45, 0.55, 1.0};  // Chisel teal — unambiguous in a screenshot

        WGPURenderPassDescriptor rp = {};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &color;

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuSurfacePresent(surface);

        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);

        if (++frame == 1) std::printf("[win] first frame presented\n");
        if (maxFrames > 0 && frame >= maxFrames) break;
    }
    std::printf("[win] presented %ld frame(s)\n", frame);

    wgpuQueueRelease(queue);
    wgpuSurfaceRelease(surface);
    wgpuAdapterRelease(ar.adapter);
    wgpuDeviceRelease(device);
    wgpuInstanceRelease(instance);
    glfwDestroyWindow(win);
    glfwTerminate();
    std::printf("[win] OK\n");
    return 0;
}
