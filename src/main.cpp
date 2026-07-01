#ifdef CHISEL_BACKEND_WEBGPU
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpu-native extensions (wgpuDevicePoll / ProcessEvents)
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
#include "project_file.h"
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

// Read depth buffer at pixel to determine if cursor is on model
static bool read_depth_at(Renderer& renderer, int x, int y, int screen_h, float* depth) {
#ifdef CHISEL_BACKEND_WEBGPU
    // No cheap read of the swapchain depth on WebGPU, so the on-model latch samples
    // the screen-target's linear-depth attachment (0, `-viewPos.z`, cleared to 1000)
    // — the same idle-refreshed offscreen buffer the cursor-normal sampling reads.
    // Freshness domain matches: the press latch only fires when stationary, when the
    // screen buffers are current (they refresh on the idle path before the next press).
    (void)screen_h;
    float d = 1000.0f;
    renderer.read_depth_region(x, y, 1, 1, &d);
    *depth = d;
    return d < 500.0f;  // clear = 1000; geometry linear-depth is « far plane (~100)
#else
    (void)renderer;
    float d;
    glReadPixels(x, screen_h - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    *depth = d;
    return d < 1.0f; // 1.0 = clear value = no geometry
#endif
}

int main(int argc, char* argv[]) {
    bool cli_use_topology = true;
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
    }
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
    // window groups under the launcher entry in the taskbar/dock (X11).
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "Chisel");
    glfwWindowHintString(GLFW_X11_INSTANCE_NAME, "Chisel");

    // Get primary monitor for fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // Start windowed but maximized (easier for development)
    GLFWwindow* window = glfwCreateWindow(
        mode->width, mode->height, "Chisel", nullptr, nullptr);
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
    int fbw = mode->width, fbh = mode->height;
    glfwGetFramebufferSize(window, &fbw, &fbh);
    WGPUInstance instance = wgpuCreateInstance(nullptr);
    if (!instance) { std::fprintf(stderr, "wgpuCreateInstance failed\n"); return 1; }
    WGPUSurfaceSourceXlibWindow x11 = {};
    x11.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
    x11.display = glfwGetX11Display();
    x11.window  = (uint64_t)glfwGetX11Window(window);
    WGPUSurfaceDescriptor sd = {};
    sd.nextInChain = &x11.chain;
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
    for (int i = 0; i < 200 && !ar.done; ++i) wgpuInstanceProcessEvents(instance);
    if (!ar.ok) { std::fprintf(stderr, "no WebGPU adapter\n"); return 1; }

    DeviceResult dr;
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = onDevice;
    dcb.userdata1 = &dr;
    // Request the adapter's full supported limits verbatim. The default device
    // limits cap max_storage_buffers_per_shader_stage at 8, but our compute kernels
    // bind up to 9 SSBOs — without this the first such pipeline aborts the device.
    // Copying adapter limits is always valid (required == supported for every field,
    // alignment limits included). NOTE web target: baseline WebGPU only guarantees 8
    // storage buffers/stage, so the 9-SSBO kernels need consolidation for Emscripten.
    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(ar.adapter, &limits);
    WGPUDeviceDescriptor ddesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    ddesc.requiredLimits = &limits;
    wgpuAdapterRequestDevice(ar.adapter, &ddesc, dcb);
    for (int i = 0; i < 200 && !dr.done; ++i) wgpuInstanceProcessEvents(instance);
    if (!dr.ok) { std::fprintf(stderr, "no WebGPU device\n"); return 1; }
    g_device = dr.device;
    WGPUQueue queue = wgpuDeviceGetQueue(g_device);
    gpu::set_app_device(gpu::device_from_webgpu(g_device, queue));

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
        compute.init_crease();
        compute.init_pinch();
        compute.init_move();
        compute.init_limb();
        compute.init_mask();
        compute.init_color();
        compute.init_compute_normals();
        compute.init_multires_diff();
        compute.init_multires_apply();
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

    // GPU sculpt shaders read the mask buffer to gate locked vertices; the paint
    // shader writes the color VBO directly. compute.mask_ssbo / color_ssbo alias the
    // renderer's owned vbo_mask / vbo_color and are (re)pointed by Scene::bind_active_
    // after every upload_mesh — a realloc makes a new handle (Step 3a), so a set-once
    // alias here would go stale on the first topology change.

    TextOverlay text;
    text.init(renderer.gpu_dev);

    Camera camera;

    Scene scene(icosphere(input.subdiv_level), renderer, compute, input.subdiv_level);
    scene.set_mirror_topology(cli_use_topology);
    scene.refresh_mirror_map(input.subdiv_level);
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

    Vec3 mesh_center;
    float mesh_radius;
    mesh->compute_bounding_sphere(mesh_center, mesh_radius);
    camera.set_target(mesh_center);
    camera.distance = mesh_radius * 2.5f;

    Vec3 last_sculpt_point = mesh_center;
    BrushStroke brush_stroke;
    brush_stroke.compute = &compute;
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
    std::string error_popup_msg;
    bool error_popup_trigger = false;

    bool screen_buffers_dirty = true;

    // FPS counter
    double fps_last_time = glfwGetTime();
    int fps_frame_count = 0;
    float fps_display = 0.0f;

    // Store windowed position/size for fullscreen restore
    int windowed_x = 0, windowed_y = 0, windowed_w = mode->width, windowed_h = mode->height;
    glfwGetWindowPos(window, &windowed_x, &windowed_y);
    glfwGetWindowSize(window, &windowed_w, &windowed_h);

    // Fold the working mesh's vertex paint into the multires base before a
    // cascade. cascade_to_level rebuilds the surface from stack.base, so paint
    // only survives a level change if base.color carries it. Loop keeps the
    // original verts as the [0, base.vcount) prefix at every level, so the
    // prefix of the working colour IS the base colour (model B: single array,
    // interpolate up — fine-level midpoint paint blurs to base on round-trip).
    auto sync_color_to_base = [](const Mesh& working, MultiresStack& stk) {
        if (working.color.empty()) return;
        uint32_t vb = stk.base.vertex_count();
        stk.base.color.assign(
            working.color.begin(),
            working.color.begin() + std::min<size_t>(vb, working.color.size()));
        stk.base.color.resize(vb, 0xFFFFFFFFu);
    };

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        input.begin_frame();
        glfwPollEvents();
        tablet.poll(brush_stroke.is_active());
        if (tablet.available() && !prev_tablet_avail) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Pen pressure: tablet connected");
            input.notification_timer = 2.0f;
        }
        prev_tablet_avail = tablet.available();
        mesh = &scene.active_mesh();
        multires = &scene.active_multires();

#ifdef CHISEL_BACKEND_WEBGPU
        ImGui_ImplWGPU_NewFrame();
#else
        ImGui_ImplOpenGL3_NewFrame();
#endif
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
        bool dialog_open = input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.voxel_merge_confirm_pending;
        if ((imgui_wants_mouse || dialog_open) && app_state != AppState::SCULPTING) {
            input.drag_mode = InputState::DragMode::NONE;
            input.mouse1_just_pressed = false;
            input_consume_scroll();
        }

        // FPS counter
        fps_frame_count++;
        double fps_now = glfwGetTime();
        if (fps_now - fps_last_time >= 1.0) {
            fps_display = (float)fps_frame_count / (float)(fps_now - fps_last_time);
            fps_frame_count = 0;
            fps_last_time = fps_now;
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

        int win_w, win_h;
        glfwGetFramebufferSize(window, &win_w, &win_h);
        if (win_w == 0 || win_h == 0) {
            input.end_frame();
            continue;
        }
#ifdef CHISEL_BACKEND_WEBGPU
        // Reconfigure the surface + depth on resize; per-pass viewport is set by the
        // seam (no global glViewport).
        if (win_w != fbw || win_h != fbh) {
            fbw = win_w; fbh = win_h;
            configureSurface(fbw, fbh);
            makeDepth(fbw, fbh);
        }
#else
        glViewport(0, 0, win_w, win_h);
#endif

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

        // Refresh screen buffers whenever idle and dirty.
        if (!camera_moving && !brush_stroke.is_active()) {
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
                // screen buffer as the cursor normal. Gated with it so the depth
                // readback — a wasted glReadPixels on GL, a full GPU sync on WebGPU —
                // only fires when stationary over fresh buffers, which is exactly when
                // the press latch fires. Orbiting/mid-stroke, on_model keeps its last
                // value (not consumed then).
                float depth;
                input.on_model = read_depth_at(renderer, cx, cy, win_h, &depth);

                float norm_pixel[3];
                renderer.read_normal_region(cx, cy, 1, 1, norm_pixel);
                float len = std::sqrt(norm_pixel[0]*norm_pixel[0] + norm_pixel[1]*norm_pixel[1] + norm_pixel[2]*norm_pixel[2]);
                if (len > 1e-6f) {
                    target_nx = norm_pixel[0] / len;
                    target_ny = norm_pixel[1] / len;
                    target_nz = norm_pixel[2] / len;
                    have_sample = true;
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
        auto refresh_active_gpu_residency = [&]() {
            MeshEntity& ent = scene.active_entity();
            ent.multires_gpu.supported = compute.supported;
            ent.multires_gpu.dev       = &compute.gpu_dev;
            ent.multires_gpu.upload_level(ent.multires, ent.multires.current_level);
        };

        // Multires level switch (D / Shift-D post-lock). Recorded on the undo
        // timeline as a LEVEL entry so Ctrl-Z retraces the literal action sequence
        // (…draw, subd-up, draw, subd-down…) with the view level following along.
        if (input.level_switch_delta != 0) {
            int delta = input.level_switch_delta;
            input.level_switch_delta = 0;
            const int from   = multires->current_level;
            const int target = from + delta;
            if (target >= multires->base_level && target <= MULTIRES_MAX_LEVEL) {
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
                sync_color_to_base(*mesh, *multires);
                if (scene.alive_count() <= 1) {
                    auto saved_mask = std::move(mesh->mask);
                    cascade_to_level(*multires, *mesh, target);
                    if (!saved_mask.empty() && saved_mask.size() == mesh->vertex_count())
                        mesh->mask = std::move(saved_mask);
                    scene.refresh_mirror_map();
                    scene.sync();  // rebinds active: rebuilds adjacency from new topology
                } else {
                    Mesh solo;
                    cascade_to_level(*multires, solo, target);
                    scene.splice_active(solo);  // splice_active marks topo dirty
                    scene.refresh_mirror_map();
                }
                refresh_active_gpu_residency();
                mesh->compute_bounding_sphere(mesh_center, mesh_radius);
                screen_buffers_dirty = true;
                std::printf("[multires] switched to level %d (%u verts, %u tris)\n",
                            target, mesh->vertex_count(), mesh->tri_count());
                print_undo_top("level-switch");
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

        // ---- File dialogs (ImGuiFileDialog) ----
        // Handled in the ImGui widget section below (after rendering).

        // ---- Remesh execution ----
        if (input.remesh_requested) {
            input.remesh_requested = false;
            input.remesh_in_progress = true;

            scene.materialize_active_cpu();  // 2b: remesh reads the live surface (mesh.pos)
            // Destructive remesh breaks topology mirror — switch to spatial
            // mode afterward. Mirror setting (input.mirror_x) is preserved.
            auto result = perform_remesh(*mesh, *multires, 0.0f, 10,
                                         compute.supported ? &compute : nullptr);

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
                if (result.selected_tris > 0) scene.active_undo().clear(&compute);
                std::snprintf(input.notification, sizeof(input.notification),
                              "Remesh: %u sel, %u/%u -> %u/%u v/t (spatial mirror)",
                              result.selected_tris,
                              result.old_verts, result.old_tris,
                              result.new_verts, result.new_tris);
                input.notification_timer = 4.0f;
            } else {
                std::printf("[remesh] FAILED: %s\n", result.error.c_str());
                std::snprintf(input.notification, sizeof(input.notification),
                              "Remesh FAILED — check console");
                input.notification_timer = 4.0f;
            }

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
                              "Voxel merge needs GPU compute (unavailable)");
                input.notification_timer = 4.0f;
            } else if (!vmerge_job) {
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
                                  "Merged %u -> %u v / %u t (R=%u, %.0f ms) | %s: %u comp, %u bnd, %u nonmf",
                                  vm.in_entities, vm.out_verts, vm.out_tris,
                                  vm.R, vm.elapsed_ms,
                                  watertight ? "watertight" : "NOT watertight",
                                  vm.components, vm.boundary_edges, vm.nonmanifold_edges);
                    input.notification_timer = 6.0f;
                } else {
                    std::printf("[voxel-merge] FAILED: %s\n", vm.error.c_str());
                    std::snprintf(input.notification, sizeof(input.notification),
                                  "Voxel merge failed: %.200s", vm.error.c_str());
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

            if (already_selected) {
                input.drag_mode = InputState::DragMode::MOVE_OBJECT;
            } else {
                input.drag_mode = InputState::DragMode::NONE;
                if (clicked_mesh != 0) {
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
                               scene.active_entity().multires_gpu);
            brush_stroke.cursor_hist_count = 1;
            brush_stroke.cursor_hist_x[0] = (float)input.mouse_x;
            brush_stroke.cursor_hist_y[0] = (float)input.mouse_y;
            app_state = AppState::SCULPTING;
        }

        // SCULPTING → IDLE (pen-up)
        if (app_state == AppState::SCULPTING && input.drag_mode != InputState::DragMode::SCULPT) {
            bool had_update = brush_stroke.finalize(*mesh, scene.active_undo(), *multires,
                                                     scene.active_entity().multires_gpu,
                                                     renderer, input.current_brush,
                                                     input.autosmooth);
            print_undo_top("stroke-commit");
            input.sculpting = false;
            if (had_update)
                renderer.update_screen_positions(*mesh);
            screen_buffers_dirty = true;
            app_state = AppState::IDLE;
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
                float scale = camera.distance * 0.002f;
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

                        // X-axis behaviour for a *centered* (symmetric) piece — its
                        // bounding centre sits on the mirror plane. With symmetry on,
                        // mirror the motion per-lobe: +x verts move +delta.x, -x verts
                        // -delta.x, seam pinned at 0 -> the lobes (the mesh and its
                        // -x twin) spread/converge as exact mirrors, stays symmetric.
                        // With symmetry off, lock X so the piece stays centered.
                        // Off-centre pieces always translate freely on all axes.
                        Vec3 c; float r;
                        e->mesh.compute_bounding_sphere(c, r);
                        float rr = (r > 0.0f) ? r : 1.0f;
                        bool centered = std::fabs(c.x) < 1e-3f * rr;
                        bool mirror_lobes = centered && input.mirror_x;
                        bool lock_x = centered && !input.mirror_x;
                        float seam_eps = 1e-4f * rr;

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
                        // Also shift multires base if this is the active entity
                        if (sel_id == scene.active_mesh_id() && multires->locked)
                            move_mesh(multires->base);

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
                            scale_mesh(e->mesh);
                            if (sel_id == scene.active_mesh_id() && multires->locked)
                                scale_mesh(multires->base);

                            std::vector<uint32_t> local_dirty(vc);
                            for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                            scene.sync_partial_entity(sel_id, local_dirty);
                        }
                        screen_buffers_dirty = true;
                    }
                }
                wrap_cursor(window, input, win_w, win_h);
            }

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
                for (uint32_t v = 0; v < mesh->vertex_count(); v++) {
                    mesh->mask[v] = 1.0f - mesh->mask[v];
                }
                renderer.update_mask(*mesh);
            }

            // Mask clear (Ctrl+A when mask brush is active)
            if (input.mask_clear_requested) {
                input.mask_clear_requested = false;
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
                MeshEntity& ent = scene.active_entity();
                sync_color_to_base(ent.mesh, ent.multires);
                auto saved_mask = std::move(ent.mesh.mask);
                Mesh solo;
                cascade_to_level(ent.multires, solo, ent.multires.current_level);
                if (!saved_mask.empty() && saved_mask.size() == solo.vertex_count())
                    solo.mask = std::move(saved_mask);
                scene.splice_active(solo);  // rebuilds adjacency + full sync
                mesh     = &scene.active_mesh();
                multires = &scene.active_multires();
                scene.refresh_mirror_map();
                refresh_active_gpu_residency();
            };
            if (input.undo_requested) {
                input.undo_requested = false;
                cascade_active(scene.active_undo().undo(scene.active_entity(), scene));
                print_undo_top("ctrl-z");
                screen_buffers_dirty = true;
            }
            if (input.redo_requested) {
                input.redo_requested = false;
                cascade_active(scene.active_undo().redo(scene.active_entity(), scene));
                print_undo_top("ctrl-shift-z");
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

                    sync_color_to_base(*mesh, *multires);
                    if (scene.alive_count() <= 1) {
                        auto saved_mask = std::move(mesh->mask);
                        cascade_to_level(*multires, *mesh, multires->current_level);
                        if (!saved_mask.empty() && saved_mask.size() == mesh->vertex_count())
                            mesh->mask = std::move(saved_mask);
                        scene.refresh_mirror_map();
                        scene.sync();
                    } else {
                        Mesh solo;
                        cascade_to_level(*multires, solo, multires->current_level);
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

                const BrushSettings& eff = is_smooth
                    ? input.per_brush[(int)BrushType::SMOOTH]
                    : input.per_brush[(int)input.current_brush];
                float eff_strength = eff.strength;
                float eff_hardness = eff.hardness;

                // Pen pressure → strength & size (independent floors). 1.0 when no
                // tablet / disabled, so mouse behaviour is unchanged.
                float pressure = (input.pressure_enabled && tablet.available())
                                 ? tablet.pressure() : 1.0f;
                float p_shaped = std::pow(pressure, PRESSURE_GAMMA);
                eff_strength *= PRESSURE_STR_FLOOR + (1.0f - PRESSURE_STR_FLOOR) * p_shaped;
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
                float spacing = input.brush_size * input.brush_spacing;

                int dab_count = 0;
                if (is_grab || brush_stroke.phase == StrokePhase::BEGIN) {
                    dab_count = 1;
                } else if (dab_dist >= spacing) {
                    dab_count = (int)(dab_dist / spacing);
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
                                     brush_stroke.vertex_count, eff_brush_size };

                    if (is_smooth) {
                        // Smooth gesture while painting blends colours, not geometry.
                        if (input.current_brush == BrushType::PAINT &&
                            compute.supported && compute.has_color_smooth()) {
                            brush_stroke.apply_color_smooth_gpu(ctx, dab_x, dab_y, eff_strength, eff_hardness);
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
                        if (compute.supported && compute.has_color()) {
                            brush_stroke.apply_color_gpu(ctx, dab_x, dab_y,
                                                         eff_strength, eff_hardness,
                                                         input.is_subtract_active());
                        }
                    } else if (input.current_brush == BrushType::DRAW) {
                        brush_stroke.apply_draw(ctx, dab_x, dab_y, eff_strength, eff_hardness,
                                                input.is_subtract_active());
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

                {
                    DabContext ctx { renderer, camera, compute, *mesh, *multires, input, win_w, win_h,
                                     brush_stroke.vertex_count, eff_brush_size };
                    brush_stroke.post_frame(ctx);
                }

                brush_stroke.needs_mesh_update = true;
                } // dab_count > 0
            }
        }

        // ---- Cursor visibility ----
        bool non_edit_mode = input.interaction_mode != InputState::InteractionMode::EDIT;
        if (input.quit_requested || input.export_dialog_active || input.import_dialog_active || input.save_dialog_active || input.remesh_confirm_pending || input.voxel_merge_confirm_pending || imgui_wants_mouse || non_edit_mode) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        }

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

        // Mesh: N draws. The active entity is drawn from the working VAO; every
        // other alive entity from its static display VAO. Depth test composes them.
        {
            const std::vector<uint32_t>& sel = scene.selected_ids();
            uint32_t active_id = scene.active_mesh_id();
            for (auto& up : scene.entities()) {
                if (!up || !up->alive) continue;
                // Empty selection = all editable (single-mesh) → no tint. Otherwise
                // an entity is tinted (deselected) unless it's in the selection set.
                bool selected = sel.empty();
                for (uint32_t id : sel) if (id == up->id) { selected = true; break; }
                if (up->id == active_id)
                    renderer.draw_mesh(camera, win_w, win_h,
                                       (uint32_t)scene.active_mesh().indices.size(),
                                       input.facing_threshold, selected);
                else
                    renderer.draw_display(camera, up->gpu, win_w, win_h,
                                          input.facing_threshold, selected);
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
        if (!input.quit_requested && !input.export_dialog_active && !input.import_dialog_active && !input.save_dialog_active && !input.remesh_confirm_pending && !input.voxel_merge_confirm_pending && !imgui_wants_mouse && !non_edit_mode) {
            float cursor_x, cursor_y;
            if (input.slider_mode != InputState::SliderMode::NONE) {
                cursor_x = (float)input.slider_start_x;
                cursor_y = (float)input.slider_start_y;
            } else {
                cursor_x = (float)input.mouse_x;
                cursor_y = (float)input.mouse_y;
            }
            renderer.draw_cursor(
                camera,
                cursor_x, cursor_y,
                input.brush_size,
                input.cursor_nx, input.cursor_ny, input.cursor_nz,
                input.brush_hardness,
                win_w, win_h, input.on_model);
        }

        // ---- Overlays ----
        if (input.quit_requested)
            draw_quit_dialog(text, win_w, win_h);
        if (input.remesh_confirm_pending)
            draw_remesh_confirm(text, win_w, win_h);
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
            draw_voxel_merge_confirm(text, input.voxel_merge_resolution,
                                     (int)scene.selected_ids().size(), n_unselected,
                                     input.voxel_merge_surface_nets, win_w, win_h);
        }
        if (input.remesh_in_progress)
            draw_remesh_progress(text, win_w, win_h);
        if (input.voxel_merge_in_progress)
            draw_voxel_merge_progress(text, win_w, win_h,
                                      vmerge_job ? voxel_merge_progress(*vmerge_job) : 0.0f);
        if (input.toolbar_visible)
            draw_toolbar(text, input, mesh->tri_count(), mesh->vertex_count(), win_w, win_h);
        if (input.slider_mode != InputState::SliderMode::NONE)
            draw_slider(text, input, win_w, win_h);
        if (input.interaction_mode == InputState::InteractionMode::SELECT)
            draw_mode_indicator(text, "SELECT", win_w, win_h);
        else if (input.interaction_mode == InputState::InteractionMode::INSERT)
            draw_mode_indicator(text, "INSERT", win_w, win_h);
        draw_notification(text, input, win_w, win_h);
        draw_fps(text, fps_display, win_w, win_h);
        draw_version(text, CHISEL_VERSION, win_w, win_h);

        // ---- ImGui file dialogs ----
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
            fd->OpenDialog("ImportKey", "Open File", ".chisel,.obj,.ply", cfg);
        }

        if (fd->Display("ExportKey", ImGuiWindowFlags_NoCollapse, ImVec2(600, 400))) {
            if (fd->IsOk()) {
                std::string path = fd->GetFilePathName();
                auto dot = path.find_last_of('.');
                std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";
                for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
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
            if (fd->IsOk()) {
                std::string path = fd->GetFilePathName();
                auto dot = path.find_last_of('.');
                std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "";

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

                        std::snprintf(input.notification, sizeof(input.notification),
                                      "Loaded: %.400s (%zu mesh%s, active %u v, %u t)",
                                      path.c_str(), proj.entities.size(),
                                      proj.entities.size() == 1 ? "" : "es",
                                      mesh->vertex_count(), mesh->tri_count());
                        input.notification_timer = 3.0f;
                    } else {
                        error_popup_msg = std::string("Load failed: ") + result_string(lr) + "\n" + path;
                        error_popup_trigger = true;
                    }
                } else {
                    Mesh loaded;
                    bool imported = (ext == "ply") ? Mesh::import_ply(path.c_str(), loaded)
                                                   : Mesh::import_obj(path.c_str(), loaded);
                    if (imported) {
                        *mesh = std::move(loaded);
                        mesh->mask.clear();

                        scene.set_mirror_topology(false);
                        scene.refresh_mirror_map();
                        mesh->build_adjacency();

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
            }
            fd->Close();
            input.import_dialog_active = false;
        }
        if (!input.import_dialog_active && fd->IsOpened("ImportKey"))
            fd->Close();

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
                std::snprintf(input.notification, sizeof(input.notification),
                              "Saved: %.400s", path.c_str());
                input.notification_timer = 2.0f;
            } else {
                error_popup_msg = std::string("Save failed: ") + result_string(sr) + "\n" + path;
                error_popup_trigger = true;
            }
        };

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

        // ---- Button islands (brush selection + ops) ----
        draw_button_islands(input, win_w, win_h);

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
        wgpuSurfacePresent(g_surface);
        gpu::webgpu_set_default_color(nullptr);
        if (frameView) wgpuTextureViewRelease(frameView);
        if (surfTex.texture) wgpuTextureRelease(surfTex.texture);
#else
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
#endif
        input.end_frame();
    }

#ifdef CHISEL_BACKEND_WEBGPU
    ImGui_ImplWGPU_Shutdown();
#else
    ImGui_ImplOpenGL3_Shutdown();
#endif
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    tablet.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
