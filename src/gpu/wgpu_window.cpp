// Stage 3+4 probe: GLFW window -> WGPUSurface (X11) -> device -> configure ->
// render a real mesh (icosphere from the gl tree's sphere.cpp) transformed by a
// Camera UBO, depth-tested, with the procedural matcap-grey shading ported from
// renderer.cpp, plus line/disc overlays and a full Dear ImGui pass (Stage 3).
// Stage 4 adds the picking FBO: an offscreen MRT pass writing {linear depth,
// world normal, triangle-id} (renderer.cpp's screen_buf_* shaders) and an async
// MAP_READ readback of the brush-point triangle id — the groundwork every brush
// stage needs to seed a stroke at pen-down.
// Stage 5 (start) adds the first compute brush: the mask-paint kernel
// (shaders/wgsl/mask_paint.wgsl) dispatched against a mask SSBO that doubles as the
// mesh's @loc2 tint attribute. Pick triid -> centroid anchor -> compute dab ->
// visible mask tint, all on WebGPU. Headless auto-paints center; LMB paints live.
//
// Run live (window stays open until closed) for a screenshot, or set
// CHISEL_PROBE_FRAMES=N to auto-exit after N presented frames (CI/headless-ish).
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpu-native extensions: wgpuDevicePoll for the readback pump

#include "mesh.h"
#include "camera.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// WGSL string -> WGPUStringView (explicit length; wgpu-native v29 wants a sized view).
static WGPUStringView sv(const char* s) { return WGPUStringView{ s, s ? std::strlen(s) : 0 }; }

// Load a canonical .wgsl kernel from the source tree (CHISEL_WGSL_DIR is injected by
// CMake). Keeps the compute kernels as the single-source .wgsl files — no inlined
// copy to drift. Returns "" on failure (caller checks).
#ifndef CHISEL_WGSL_DIR
#define CHISEL_WGSL_DIR "shaders/wgsl"
#endif
static std::string loadWgsl(const char* name) {
    std::string path = std::string(CHISEL_WGSL_DIR) + "/" + name;
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("[win] failed to open wgsl: %s\n", path.c_str()); return ""; }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Procedural matcap-grey + camera, ported 1:1 from renderer.cpp's matcap shader.
// The "matcap" is view-space-normal shading (no texture), so it reproduces exactly.
// Camera is two column-major mat4 (view, proj); proj already carries the GL->WebGPU
// z-clip correction (see makeProjCorrected), so clip-space z lands in [0,1].
static const char* kMeshWGSL = R"(
struct Camera { view: mat4x4<f32>, proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) mask: f32,
};

@vertex
fn vs_main(@location(0) aPos: vec3<f32>, @location(1) aNorm: vec3<f32>,
           @location(2) aMask: f32) -> VSOut {
    var o: VSOut;
    // view-space normal: upper-left 3x3 of the (column-major) view matrix
    let nm = mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);
    o.nrm = normalize(nm * aNorm);
    o.mask = aMask;
    o.pos = cam.proj * cam.view * vec4<f32>(aPos, 1.0);
    return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.nrm);
    let rim = 1.0 - abs(n.z);
    let top = n.y * 0.5 + 0.5;
    let base = 0.35 + 0.45 * top + 0.15 * (1.0 - rim * rim);
    let cavity = 1.0 - rim * rim * 0.3;
    var val = base * cavity;
    // Sculpt mask: unmasked (0) = normal, masked (1) = darkened — same as renderer.cpp.
    val = val * mix(1.0, 0.4, clamp(in.mask, 0.0, 1.0));
    return vec4<f32>(vec3<f32>(val), 1.0);
}
)";

// Overlay 1: world-space debug lines (wireframe). Reuses the same Camera UBO /
// bind group as the mesh; constant cyan, ported from renderer.cpp's debug pass.
static const char* kLineWGSL = R"(
struct Camera { view: mat4x4<f32>, proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;
@vertex
fn vs_line(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    return cam.proj * cam.view * vec4<f32>(p, 1.0);
}
@fragment
fn fs_line() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 1.0, 1.0, 1.0); }
)";

// Overlay 2: screen-space alpha-blended disc (the brush "shadow" footprint).
// Vertex data is (ndc.x, ndc.y, dist), dist 0 at center -> 1 at rim. Falloff
// ported from renderer.cpp's cursor_shadow_frag. No camera; no depth write.
static const char* kDiscWGSL = R"(
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) d: f32 };
@vertex
fn vs_disc(@location(0) v: vec3<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(v.xy, 0.0, 1.0);
    o.d = v.z;
    return o;
}
@fragment
fn fs_disc(in: VOut) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(0.72, 1.0, in.d);
    let core = 0.55 + 0.45 * smoothstep(0.0, 0.55, in.d);
    let a = 0.6 * edge * core;
    return vec4<f32>(0.95, 0.55, 0.15, a);  // Chisel orange, straight alpha
}
)";

// Stage 4: picking-FBO MRT shader. Ports renderer.cpp's screen_buf_{vert,frag}:
// writes linear depth (@loc0, R32F), WORLD normal (@loc1, RGBA16F — not view-space,
// the brush back-projection wants world dirs), and the flat per-triangle id
// (@loc2, R32U). Camera UBO is the same {view, proj} bind group as the mesh pass.
// Bary (renderer's @loc3) is deferred until the mask brush needs it.
static const char* kPickWGSL = R"(
struct Camera { view: mat4x4<f32>, proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) nrmWorld: vec3<f32>,
    @location(1) depth: f32,
    @location(2) @interpolate(flat) triid: u32,
};

@vertex
fn vs_pick(@location(0) aPos: vec3<f32>, @location(1) aNorm: vec3<f32>,
           @location(2) aTriID: u32) -> VSOut {
    var o: VSOut;
    let viewPos = cam.view * vec4<f32>(aPos, 1.0);
    o.nrmWorld = normalize(aNorm);   // world-space, do NOT transform by view
    o.depth = -viewPos.z;            // linear depth, positive into screen
    o.triid = aTriID;
    o.pos = cam.proj * viewPos;
    return o;
}

struct FSOut {
    @location(0) depth: f32,
    @location(1) nrm: vec4<f32>,
    @location(2) triid: u32,
};

@fragment
fn fs_pick(in: VSOut) -> FSOut {
    var o: FSOut;
    o.depth = in.depth;
    o.nrm = vec4<f32>(normalize(in.nrmWorld), 1.0);
    o.triid = in.triid;
    return o;
}
)";

// De-index the SOA mesh into a flat per-triangle-vertex layout (3 unique verts per
// triangle), each carrying its triangle index — exactly what renderer.cpp's
// screen_expand compute kernel builds so the flat triid attribute is well-defined
// (a shared indexed vertex can't carry one triangle's id). pos/norm xyz + u32 triid.
static void buildExpanded(const Mesh& mesh,
                          std::vector<float>& pos, std::vector<float>& nrm,
                          std::vector<uint32_t>& triid) {
    const uint32_t flatVc = (uint32_t)mesh.indices.size();
    pos.resize(flatVc * 3);
    nrm.resize(flatVc * 3);
    triid.resize(flatVc);
    for (uint32_t k = 0; k < flatVc; ++k) {
        uint32_t v = mesh.indices[k];
        pos[k*3+0] = mesh.pos_x[v]; pos[k*3+1] = mesh.pos_y[v]; pos[k*3+2] = mesh.pos_z[v];
        nrm[k*3+0] = mesh.norm_x[v]; nrm[k*3+1] = mesh.norm_y[v]; nrm[k*3+2] = mesh.norm_z[v];
        triid[k] = k / 3;
    }
}

// ---- column-major 4x4 multiply: out = a * b (m[col*4 + row]) ----
static void mat_mul(float* out, const float* a, const float* b) {
    float r[16];
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k*4 + row] * b[c*4 + k];
            r[c*4 + row] = s;
        }
    std::memcpy(out, r, sizeof r);
}

// GL clip space is z in [-1,1]; WebGPU is z in [0,1]. This column-major matrix
// remaps z' = 0.5*z + 0.5*w, leaving x/y/w untouched. Pre-multiply the GL proj.
static void makeProjCorrected(float* out, const float* glProj) {
    const float C[16] = {
        1, 0, 0,   0,
        0, 1, 0,   0,
        0, 0, 0.5f, 0,
        0, 0, 0.5f, 1,
    };
    mat_mul(out, C, glProj);  // out = C * glProj
}

// Wireframe cube edges (world space) around the sphere: 8 corners, 12 edges,
// 24 line-list vertices. r > sphere radius so front edges show, back edges get
// occluded by the depth-tested mesh (proves multi-pipeline depth compositing).
static std::vector<float> buildCubeEdges(float r) {
    const float c[8][3] = {
        {-r,-r,-r},{ r,-r,-r},{-r, r,-r},{ r, r,-r},
        {-r,-r, r},{ r,-r, r},{-r, r, r},{ r, r, r},
    };
    const int e[12][2] = {
        {0,1},{1,3},{3,2},{2,0},   // -z face
        {4,5},{5,7},{7,6},{6,4},   // +z face
        {0,4},{1,5},{2,6},{3,7},   // verticals
    };
    std::vector<float> v;
    v.reserve(12 * 2 * 3);
    for (auto& edge : e)
        for (int k = 0; k < 2; ++k) {
            v.push_back(c[edge[k]][0]); v.push_back(c[edge[k]][1]); v.push_back(c[edge[k]][2]);
        }
    return v;
}

// Screen-space disc as a triangle fan expanded to a triangle list (no fan
// topology in WebGPU). Each vertex is (ndc.x, ndc.y, dist). Aspect-corrected so
// it reads round, not elliptical. Rebuilt on resize.
static const int kDiscSegments = 48;
static std::vector<float> buildDisc(int w, int h) {
    const float ry = 0.25f;
    const float rx = ry * (float)h / (float)w;
    std::vector<float> v;
    v.reserve(kDiscSegments * 3 * 3);
    auto push = [&](float x, float y, float d) { v.push_back(x); v.push_back(y); v.push_back(d); };
    const float TAU = 6.28318530718f;
    for (int i = 0; i < kDiscSegments; ++i) {
        float a0 = (float)i / kDiscSegments * TAU;
        float a1 = (float)(i + 1) / kDiscSegments * TAU;
        push(0.0f, 0.0f, 0.0f);
        push(rx * std::cos(a0), ry * std::sin(a0), 1.0f);
        push(rx * std::cos(a1), ry * std::sin(a1), 1.0f);
    }
    return v;
}

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

// ---- live framebuffer size (resize -> reconfigure surface + depth) ----
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

static const WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth24Plus;

// Create (or recreate) the depth texture + view at the given size. Releases the
// previous one. Returns the new view via out params.
static void makeDepth(WGPUDevice device, int w, int h,
                      WGPUTexture* tex, WGPUTextureView* view) {
    if (*view) { wgpuTextureViewRelease(*view); *view = nullptr; }
    if (*tex)  { wgpuTextureRelease(*tex);       *tex  = nullptr; }
    WGPUTextureDescriptor td = {};
    td.usage = WGPUTextureUsage_RenderAttachment;
    td.dimension = WGPUTextureDimension_2D;
    td.size = { (uint32_t)w, (uint32_t)h, 1 };
    td.format = kDepthFormat;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    *tex  = wgpuDeviceCreateTexture(device, &td);
    *view = wgpuTextureCreateView(*tex, nullptr);
}

// Stage 4 picking FBO: the GL screen-buffer MRT set. Three color targets matching
// renderer.cpp's create_screen_buffers (depth R32F / normal RGBA16F / triid R32U,
// usage RenderAttachment|CopySrc so we can read them back) plus a private depth
// buffer for z-testing the pick pass. Recreated on resize like the main depth.
static const WGPUTextureFormat kPickDepthFmt  = WGPUTextureFormat_R32Float;
static const WGPUTextureFormat kPickNormalFmt = WGPUTextureFormat_RGBA16Float;
static const WGPUTextureFormat kPickTriidFmt  = WGPUTextureFormat_R32Uint;

struct PickTargets {
    WGPUTexture depthTex = nullptr, normalTex = nullptr, triidTex = nullptr, zTex = nullptr;
    WGPUTextureView depthView = nullptr, normalView = nullptr, triidView = nullptr, zView = nullptr;
};

static void makePickTargets(WGPUDevice device, int w, int h, PickTargets* p) {
    WGPUTextureView* views[4] = { &p->depthView, &p->normalView, &p->triidView, &p->zView };
    WGPUTexture*     texs[4]  = { &p->depthTex,  &p->normalTex,  &p->triidTex,  &p->zTex  };
    for (int i = 0; i < 4; ++i) {
        if (*views[i]) { wgpuTextureViewRelease(*views[i]); *views[i] = nullptr; }
        if (*texs[i])  { wgpuTextureRelease(*texs[i]);       *texs[i]  = nullptr; }
    }
    auto make = [&](WGPUTextureFormat fmt, WGPUTextureUsage usage,
                    WGPUTexture* tex, WGPUTextureView* view) {
        WGPUTextureDescriptor td = {};
        td.usage = usage;
        td.dimension = WGPUTextureDimension_2D;
        td.size = { (uint32_t)w, (uint32_t)h, 1 };
        td.format = fmt;
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        *tex  = wgpuDeviceCreateTexture(device, &td);
        *view = wgpuTextureCreateView(*tex, nullptr);
    };
    WGPUTextureUsage colorUsage =
        (WGPUTextureUsage)(WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    make(kPickDepthFmt,  colorUsage, &p->depthTex,  &p->depthView);
    make(kPickNormalFmt, colorUsage, &p->normalTex, &p->normalView);
    make(kPickTriidFmt,  colorUsage, &p->triidTex,  &p->triidView);
    make(kDepthFormat,   WGPUTextureUsage_RenderAttachment, &p->zTex, &p->zView);
}

static void releasePickTargets(PickTargets* p) {
    WGPUTextureView views[4] = { p->depthView, p->normalView, p->triidView, p->zView };
    WGPUTexture     texs[4]  = { p->depthTex,  p->normalTex,  p->triidTex,  p->zTex  };
    for (int i = 0; i < 4; ++i) {
        if (views[i]) wgpuTextureViewRelease(views[i]);
        if (texs[i])  wgpuTextureRelease(texs[i]);
    }
    *p = PickTargets{};
}

// One-shot readback state: the map callback flips done/status. Legal here because
// the probe (like the real app) only reads back at a discrete moment, never mid-stroke.
struct MapResult { bool done = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; };
static void onMap(WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
    auto* r = static_cast<MapResult*>(ud1);
    r->done = true; r->status = status;
}

// Synchronous one-shot: map a MAP_READ staging buffer, busy-wait the poll until the
// callback fires, copy `count` u32s out, unmap. Legal only at the discrete readback
// points (pen-down/up/one-shot) the architecture already restricts us to.
static void readU32(WGPUDevice device, WGPUBuffer staging, uint32_t count, uint32_t* out) {
    const uint32_t bytes = count * 4;
    MapResult mr;
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode = WGPUCallbackMode_AllowProcessEvents;
    mcb.callback = onMap; mcb.userdata1 = &mr;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, bytes, mcb);
    while (!mr.done) wgpuDevicePoll(device, true, nullptr);
    if (mr.status == WGPUMapAsyncStatus_Success) {
        const uint32_t* p = (const uint32_t*)wgpuBufferGetMappedRange(staging, 0, bytes);
        if (p) std::memcpy(out, p, bytes); else for (uint32_t i=0;i<count;++i) out[i]=0xFFFFFFFFu;
        wgpuBufferUnmap(staging);
    } else {
        for (uint32_t i=0;i<count;++i) out[i]=0xFFFFFFFFu;
        wgpuBufferUnmap(staging);
    }
}

// Mask-paint Params UBO — byte-identical to mask_paint.wgsl's Params (std140/uniform:
// vec3 occupies 16 bytes, a trailing f32 packs into its slot). 48 bytes total.
struct MaskParamsGPU {
    float anchor_a[3];      // 0..11
    float world_radius;     // 12  (rides in anchor_a's 16-byte slot)
    float anchor_b[3];      // 16..27
    float hardness;         // 28
    float paint_strength;   // 32
    uint32_t use_b;         // 36
    uint32_t vertex_count;  // 40
    uint32_t _pad0;         // 44  (rounds to 48)
};
static_assert(sizeof(MaskParamsGPU) == 48, "Params UBO must be 48 bytes to match WGSL");

int main() {
    if (!glfwInit()) { std::printf("[win] glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // no GL context — WebGPU owns the surface
    GLFWwindow* win = glfwCreateWindow(g_fbw, g_fbh, "Chisel WebGPU — Stage 5 mask brush", nullptr, nullptr);
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

    // ---- depth texture ----
    WGPUTexture depthTex = nullptr; WGPUTextureView depthView = nullptr;
    makeDepth(device, g_fbw, g_fbh, &depthTex, &depthView);

    // ---- Stage 4 pick FBO targets (MRT) ----
    PickTargets pick;
    makePickTargets(device, g_fbw, g_fbh, &pick);

    // ---- mesh: icosphere from the gl tree's sphere.cpp; normals via mesh.cpp ----
    Mesh mesh = icosphere(4);            // 4 subdivisions ~ 2562 verts / 5120 tris
    mesh.build_adjacency();
    mesh.recompute_normals();
    const uint32_t vcount = mesh.vertex_count();
    const uint32_t icount = (uint32_t)mesh.indices.size();
    std::printf("[win] icosphere: %u verts, %u tris\n", vcount, icount / 3);

    // Repack SOA (separate pos_x/y/z, norm_x/y/z) into two interleaved xyz buffers,
    // one per vertex attribute — closest mapping to the SOA layout (two @location).
    std::vector<float> posBuf(vcount * 3), nrmBuf(vcount * 3);
    for (uint32_t i = 0; i < vcount; ++i) {
        posBuf[i*3+0] = mesh.pos_x[i]; posBuf[i*3+1] = mesh.pos_y[i]; posBuf[i*3+2] = mesh.pos_z[i];
        nrmBuf[i*3+0] = mesh.norm_x[i]; nrmBuf[i*3+1] = mesh.norm_y[i]; nrmBuf[i*3+2] = mesh.norm_z[i];
    }

    auto makeBuf = [&](const void* data, size_t bytes, WGPUBufferUsage usage) {
        WGPUBufferDescriptor bd = {};
        bd.usage = usage | WGPUBufferUsage_CopyDst;
        bd.size  = bytes;
        WGPUBuffer b = wgpuDeviceCreateBuffer(device, &bd);
        wgpuQueueWriteBuffer(queue, b, 0, data, bytes);
        return b;
    };
    // posVB doubles as the mask kernel's positions SSBO (binding 0) → needs Storage.
    WGPUBuffer posVB = makeBuf(posBuf.data(), posBuf.size()*sizeof(float),
                               (WGPUBufferUsage)(WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage));
    WGPUBuffer nrmVB = makeBuf(nrmBuf.data(), nrmBuf.size()*sizeof(float), WGPUBufferUsage_Vertex);
    WGPUBuffer idxB  = makeBuf(mesh.indices.data(), icount*sizeof(uint32_t), WGPUBufferUsage_Index);

    // ---- Stage 5 mask brush buffers (indexed, one entry per mesh vertex) ----
    // mask SSBO doubles as the mesh's @loc2 vertex attribute (Storage write + Vertex read),
    // so a paint dab is visible on the very next draw with no copy.
    std::vector<float> zeroMask(vcount, 0.0f);
    WGPUBuffer maskBuf = makeBuf(zeroMask.data(), vcount*sizeof(float),
                                 (WGPUBufferUsage)(WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage
                                                   | WGPUBufferUsage_CopySrc));  // CopySrc: final mask readback
    // dirty list: { u32 count; u32 ids[vcount] }. CopySrc so we can read the count back.
    std::vector<uint32_t> zeroDirty(vcount + 1, 0u);
    WGPUBuffer dirtyBuf = makeBuf(zeroDirty.data(), (vcount+1)*sizeof(uint32_t),
                                  (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc));
    // Params UBO (binding 63), uploaded per dab.
    WGPUBufferDescriptor mpd = {};
    mpd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    mpd.size  = sizeof(MaskParamsGPU);
    WGPUBuffer maskParamsUBO = wgpuDeviceCreateBuffer(device, &mpd);
    // staging buffers for the dirty-count readback and the final whole-mask readback.
    WGPUBufferDescriptor dsbd = {};
    dsbd.usage = (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    dsbd.size  = 256;
    WGPUBuffer dirtyStaging = wgpuDeviceCreateBuffer(device, &dsbd);

    // Stage 4: de-indexed mesh for the pick pass (flat per-triangle id needs unique
    // verts). 3 buffers: pos @loc0, norm @loc1, triid @loc2 (Uint32).
    std::vector<float> exPos, exNrm; std::vector<uint32_t> exTriid;
    buildExpanded(mesh, exPos, exNrm, exTriid);
    const uint32_t exVcount = (uint32_t)exTriid.size();
    WGPUBuffer pickPosVB   = makeBuf(exPos.data(),   exPos.size()*sizeof(float),     WGPUBufferUsage_Vertex);
    WGPUBuffer pickNrmVB   = makeBuf(exNrm.data(),   exNrm.size()*sizeof(float),     WGPUBufferUsage_Vertex);
    WGPUBuffer pickTriidVB = makeBuf(exTriid.data(), exTriid.size()*sizeof(uint32_t), WGPUBufferUsage_Vertex);

    // ---- camera UBO: { mat4 view; mat4 proj; } = 128 bytes ----
    Camera cam;
    cam.distance = 3.0f;
    cam.pitch = 0.35f;
    cam.yaw = 0.6f;
    WGPUBufferDescriptor ubd = {};
    ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    ubd.size  = 128;
    WGPUBuffer cameraUBO = wgpuDeviceCreateBuffer(device, &ubd);

    // ---- bind group layout / pipeline layout / bind group ----
    WGPUBindGroupLayoutEntry bglEntry = {};
    bglEntry.binding = 0;
    bglEntry.visibility = WGPUShaderStage_Vertex;
    bglEntry.buffer.type = WGPUBufferBindingType_Uniform;
    bglEntry.buffer.minBindingSize = 128;
    WGPUBindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 1;
    bglDesc.entries = &bglEntry;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

    WGPUPipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

    WGPUBindGroupEntry bgEntry = {};
    bgEntry.binding = 0;
    bgEntry.buffer = cameraUBO;
    bgEntry.offset = 0;
    bgEntry.size = 128;
    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout = bgl;
    bgDesc.entryCount = 1;
    bgDesc.entries = &bgEntry;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    // ---- render pipeline ----
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(kMeshWGSL);
    WGPUShaderModuleDescriptor smDesc = {};
    smDesc.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smDesc);
    if (!module) { std::printf("[win] shader module failed\n"); return 2; }

    // two vertex buffers: position @location 0, normal @location 1 (SOA-style)
    WGPUVertexAttribute posAttr = {};
    posAttr.format = WGPUVertexFormat_Float32x3;
    posAttr.offset = 0;
    posAttr.shaderLocation = 0;
    WGPUVertexAttribute nrmAttr = {};
    nrmAttr.format = WGPUVertexFormat_Float32x3;
    nrmAttr.offset = 0;
    nrmAttr.shaderLocation = 1;
    WGPUVertexAttribute maskAttr = {};
    maskAttr.format = WGPUVertexFormat_Float32;   // per-vertex sculpt mask @loc2
    maskAttr.offset = 0;
    maskAttr.shaderLocation = 2;
    WGPUVertexBufferLayout vbl[3] = {};
    vbl[0].arrayStride = 3 * sizeof(float);
    vbl[0].stepMode = WGPUVertexStepMode_Vertex;
    vbl[0].attributeCount = 1;
    vbl[0].attributes = &posAttr;
    vbl[1].arrayStride = 3 * sizeof(float);
    vbl[1].stepMode = WGPUVertexStepMode_Vertex;
    vbl[1].attributeCount = 1;
    vbl[1].attributes = &nrmAttr;
    vbl[2].arrayStride = sizeof(float);
    vbl[2].stepMode = WGPUVertexStepMode_Vertex;
    vbl[2].attributeCount = 1;
    vbl[2].attributes = &maskAttr;

    WGPUColorTargetState colorTarget = {};
    colorTarget.format    = format;
    colorTarget.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState frag = {};
    frag.module = module;
    frag.entryPoint = sv("fs_main");
    frag.targetCount = 1;
    frag.targets = &colorTarget;

    WGPUDepthStencilState depth = {};
    depth.format = kDepthFormat;
    depth.depthWriteEnabled = WGPUOptionalBool_True;
    depth.depthCompare = WGPUCompareFunction_Less;
    depth.stencilFront.compare = WGPUCompareFunction_Always;
    depth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor pd = {};
    pd.layout = pipelineLayout;
    pd.vertex.module = module;
    pd.vertex.entryPoint = sv("vs_main");
    pd.vertex.bufferCount = 3;
    pd.vertex.buffers = vbl;
    pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pd.primitive.frontFace = WGPUFrontFace_CCW;
    pd.primitive.cullMode = WGPUCullMode_None;  // matcap draws both sides; depth sorts it
    pd.depthStencil = &depth;
    pd.multisample.count = 1;
    pd.multisample.mask = 0xFFFFFFFFu;
    pd.fragment = &frag;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pd);
    if (!pipeline) { std::printf("[win] render pipeline failed\n"); return 2; }
    std::printf("[win] render pipeline ready\n");

    // ======================= overlay pass 1: debug lines =======================
    std::vector<float> cubeVerts = buildCubeEdges(1.25f);
    const uint32_t lineCount = (uint32_t)(cubeVerts.size() / 3);
    WGPUBuffer lineVB = makeBuf(cubeVerts.data(), cubeVerts.size()*sizeof(float), WGPUBufferUsage_Vertex);

    WGPUShaderSourceWGSL lineWgsl = {};
    lineWgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    lineWgsl.code = sv(kLineWGSL);
    WGPUShaderModuleDescriptor lineSm = {}; lineSm.nextInChain = &lineWgsl.chain;
    WGPUShaderModule lineModule = wgpuDeviceCreateShaderModule(device, &lineSm);

    WGPUVertexAttribute lineAttr = {};
    lineAttr.format = WGPUVertexFormat_Float32x3; lineAttr.shaderLocation = 0;
    WGPUVertexBufferLayout lineVbl = {};
    lineVbl.arrayStride = 3 * sizeof(float);
    lineVbl.attributeCount = 1; lineVbl.attributes = &lineAttr;

    WGPUColorTargetState lineColor = {};
    lineColor.format = format; lineColor.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState lineFrag = {};
    lineFrag.module = lineModule; lineFrag.entryPoint = sv("fs_line");
    lineFrag.targetCount = 1; lineFrag.targets = &lineColor;
    // depth-test against the mesh, but don't write (overlay shouldn't occlude itself)
    WGPUDepthStencilState lineDepth = {};
    lineDepth.format = kDepthFormat;
    lineDepth.depthWriteEnabled = WGPUOptionalBool_False;
    lineDepth.depthCompare = WGPUCompareFunction_Less;
    lineDepth.stencilFront.compare = WGPUCompareFunction_Always;
    lineDepth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor lpd = {};
    lpd.layout = pipelineLayout;             // reuse the camera bind-group layout
    lpd.vertex.module = lineModule; lpd.vertex.entryPoint = sv("vs_line");
    lpd.vertex.bufferCount = 1; lpd.vertex.buffers = &lineVbl;
    lpd.primitive.topology = WGPUPrimitiveTopology_LineList;
    lpd.depthStencil = &lineDepth;
    lpd.multisample.count = 1; lpd.multisample.mask = 0xFFFFFFFFu;
    lpd.fragment = &lineFrag;
    WGPURenderPipeline linePipeline = wgpuDeviceCreateRenderPipeline(device, &lpd);
    if (!linePipeline) { std::printf("[win] line pipeline failed\n"); return 2; }

    // ================= overlay pass 2: alpha-blended screen disc =================
    std::vector<float> discVerts = buildDisc(g_fbw, g_fbh);
    const uint32_t discCount = (uint32_t)(discVerts.size() / 3);
    WGPUBufferDescriptor discBd = {};
    discBd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    discBd.size  = discVerts.size() * sizeof(float);
    WGPUBuffer discVB = wgpuDeviceCreateBuffer(device, &discBd);
    wgpuQueueWriteBuffer(queue, discVB, 0, discVerts.data(), discBd.size);

    WGPUShaderSourceWGSL discWgsl = {};
    discWgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    discWgsl.code = sv(kDiscWGSL);
    WGPUShaderModuleDescriptor discSm = {}; discSm.nextInChain = &discWgsl.chain;
    WGPUShaderModule discModule = wgpuDeviceCreateShaderModule(device, &discSm);

    WGPUVertexAttribute discAttr = {};
    discAttr.format = WGPUVertexFormat_Float32x3; discAttr.shaderLocation = 0;
    WGPUVertexBufferLayout discVbl = {};
    discVbl.arrayStride = 3 * sizeof(float);
    discVbl.attributeCount = 1; discVbl.attributes = &discAttr;

    WGPUBlendState discBlend = {};
    discBlend.color.operation = WGPUBlendOperation_Add;
    discBlend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    discBlend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    discBlend.alpha.operation = WGPUBlendOperation_Add;
    discBlend.alpha.srcFactor = WGPUBlendFactor_One;
    discBlend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    WGPUColorTargetState discColor = {};
    discColor.format = format; discColor.writeMask = WGPUColorWriteMask_All;
    discColor.blend = &discBlend;
    WGPUFragmentState discFrag = {};
    discFrag.module = discModule; discFrag.entryPoint = sv("fs_disc");
    discFrag.targetCount = 1; discFrag.targets = &discColor;
    // render pass carries a depth attachment, so every pipeline must declare one;
    // the disc ignores it (Always, no write).
    WGPUDepthStencilState discDepth = {};
    discDepth.format = kDepthFormat;
    discDepth.depthWriteEnabled = WGPUOptionalBool_False;
    discDepth.depthCompare = WGPUCompareFunction_Always;
    discDepth.stencilFront.compare = WGPUCompareFunction_Always;
    discDepth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor dpd = {};
    dpd.layout = nullptr;                    // no bindings -> auto layout
    dpd.vertex.module = discModule; dpd.vertex.entryPoint = sv("vs_disc");
    dpd.vertex.bufferCount = 1; dpd.vertex.buffers = &discVbl;
    dpd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    dpd.depthStencil = &discDepth;
    dpd.multisample.count = 1; dpd.multisample.mask = 0xFFFFFFFFu;
    dpd.fragment = &discFrag;
    WGPURenderPipeline discPipeline = wgpuDeviceCreateRenderPipeline(device, &dpd);
    if (!discPipeline) { std::printf("[win] disc pipeline failed\n"); return 2; }
    std::printf("[win] overlay pipelines ready (lines + blended disc)\n");

    // ==================== Stage 4: pick MRT pipeline ====================
    WGPUShaderSourceWGSL pickWgsl = {};
    pickWgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    pickWgsl.code = sv(kPickWGSL);
    WGPUShaderModuleDescriptor pickSm = {}; pickSm.nextInChain = &pickWgsl.chain;
    WGPUShaderModule pickModule = wgpuDeviceCreateShaderModule(device, &pickSm);
    if (!pickModule) { std::printf("[win] pick shader module failed\n"); return 2; }

    WGPUVertexAttribute pPosAttr = {};
    pPosAttr.format = WGPUVertexFormat_Float32x3; pPosAttr.shaderLocation = 0;
    WGPUVertexAttribute pNrmAttr = {};
    pNrmAttr.format = WGPUVertexFormat_Float32x3; pNrmAttr.shaderLocation = 1;
    WGPUVertexAttribute pIdAttr = {};
    pIdAttr.format = WGPUVertexFormat_Uint32; pIdAttr.shaderLocation = 2;
    WGPUVertexBufferLayout pickVbl[3] = {};
    pickVbl[0].arrayStride = 3 * sizeof(float); pickVbl[0].attributeCount = 1; pickVbl[0].attributes = &pPosAttr;
    pickVbl[1].arrayStride = 3 * sizeof(float); pickVbl[1].attributeCount = 1; pickVbl[1].attributes = &pNrmAttr;
    pickVbl[2].arrayStride = sizeof(uint32_t);  pickVbl[2].attributeCount = 1; pickVbl[2].attributes = &pIdAttr;

    // three color targets, no blending (depth/normal/id are data, not composited)
    WGPUColorTargetState pickTargets[3] = {};
    pickTargets[0].format = kPickDepthFmt;  pickTargets[0].writeMask = WGPUColorWriteMask_All;
    pickTargets[1].format = kPickNormalFmt; pickTargets[1].writeMask = WGPUColorWriteMask_All;
    pickTargets[2].format = kPickTriidFmt;  pickTargets[2].writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState pickFrag = {};
    pickFrag.module = pickModule; pickFrag.entryPoint = sv("fs_pick");
    pickFrag.targetCount = 3; pickFrag.targets = pickTargets;

    WGPUDepthStencilState pickDepth = {};
    pickDepth.format = kDepthFormat;
    pickDepth.depthWriteEnabled = WGPUOptionalBool_True;
    pickDepth.depthCompare = WGPUCompareFunction_Less;
    pickDepth.stencilFront.compare = WGPUCompareFunction_Always;
    pickDepth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor ppd = {};
    ppd.layout = pipelineLayout;             // same camera bind-group layout
    ppd.vertex.module = pickModule; ppd.vertex.entryPoint = sv("vs_pick");
    ppd.vertex.bufferCount = 3; ppd.vertex.buffers = pickVbl;
    ppd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    ppd.primitive.frontFace = WGPUFrontFace_CCW;
    ppd.primitive.cullMode = WGPUCullMode_None;
    ppd.depthStencil = &pickDepth;
    ppd.multisample.count = 1; ppd.multisample.mask = 0xFFFFFFFFu;
    ppd.fragment = &pickFrag;
    WGPURenderPipeline pickPipeline = wgpuDeviceCreateRenderPipeline(device, &ppd);
    if (!pickPipeline) { std::printf("[win] pick pipeline failed\n"); return 2; }

    // Readback staging buffer: one R32U pixel, but bytesPerRow must be 256-aligned,
    // so the staging row is 256 bytes (the id sits in the first 4).
    const uint32_t kReadbackBytes = 256;
    WGPUBufferDescriptor sbd = {};
    sbd.usage = (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    sbd.size  = kReadbackBytes;
    WGPUBuffer readbackBuf = wgpuDeviceCreateBuffer(device, &sbd);
    std::printf("[win] pick pipeline ready (MRT depth/normal/triid + readback buffer)\n");

    // ==================== Stage 5: mask-paint compute pipeline ====================
    std::string maskSrc = loadWgsl("mask_paint.wgsl");
    if (maskSrc.empty()) { std::printf("[win] mask wgsl missing\n"); return 2; }
    WGPUShaderSourceWGSL maskWgsl = {};
    maskWgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    maskWgsl.code = sv(maskSrc.c_str());
    WGPUShaderModuleDescriptor maskSm = {}; maskSm.nextInChain = &maskWgsl.chain;
    WGPUShaderModule maskModule = wgpuDeviceCreateShaderModule(device, &maskSm);
    if (!maskModule) { std::printf("[win] mask shader module failed\n"); return 2; }

    // bindings mirror the ComputeBinding enum: positions(0,ro), mask(12,rw),
    // dirty(6,rw), params(63,uniform) — all @group(0), all compute-visible.
    WGPUBindGroupLayoutEntry me[4] = {};
    me[0].binding = 0;  me[0].visibility = WGPUShaderStage_Compute; me[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    me[1].binding = 12; me[1].visibility = WGPUShaderStage_Compute; me[1].buffer.type = WGPUBufferBindingType_Storage;
    me[2].binding = 6;  me[2].visibility = WGPUShaderStage_Compute; me[2].buffer.type = WGPUBufferBindingType_Storage;
    me[3].binding = 63; me[3].visibility = WGPUShaderStage_Compute; me[3].buffer.type = WGPUBufferBindingType_Uniform;
    me[3].buffer.minBindingSize = sizeof(MaskParamsGPU);
    WGPUBindGroupLayoutDescriptor mbgld = {}; mbgld.entryCount = 4; mbgld.entries = me;
    WGPUBindGroupLayout maskBgl = wgpuDeviceCreateBindGroupLayout(device, &mbgld);

    WGPUPipelineLayoutDescriptor mpld = {};
    mpld.bindGroupLayoutCount = 1; mpld.bindGroupLayouts = &maskBgl;
    WGPUPipelineLayout maskPL = wgpuDeviceCreatePipelineLayout(device, &mpld);

    WGPUBindGroupEntry mbe[4] = {};
    mbe[0].binding = 0;  mbe[0].buffer = posVB;         mbe[0].size = posBuf.size()*sizeof(float);
    mbe[1].binding = 12; mbe[1].buffer = maskBuf;       mbe[1].size = (uint64_t)vcount*sizeof(float);
    mbe[2].binding = 6;  mbe[2].buffer = dirtyBuf;      mbe[2].size = (uint64_t)(vcount+1)*sizeof(uint32_t);
    mbe[3].binding = 63; mbe[3].buffer = maskParamsUBO; mbe[3].size = sizeof(MaskParamsGPU);
    WGPUBindGroupDescriptor mbgd = {}; mbgd.layout = maskBgl; mbgd.entryCount = 4; mbgd.entries = mbe;
    WGPUBindGroup maskBindGroup = wgpuDeviceCreateBindGroup(device, &mbgd);

    WGPUComputePipelineDescriptor mcpd = {};
    mcpd.layout = maskPL;
    mcpd.compute.module = maskModule;
    mcpd.compute.entryPoint = sv("main");
    WGPUComputePipeline maskPipeline = wgpuDeviceCreateComputePipeline(device, &mcpd);
    if (!maskPipeline) { std::printf("[win] mask compute pipeline failed\n"); return 2; }
    std::printf("[win] mask compute pipeline ready\n");

    // Brush tuning + CPU-side anchor reconstruction. The icosphere is ~unit radius,
    // so a 0.45 world radius paints a generous cap. Anchor = picked triangle's
    // centroid (exact world-space, no depth unprojection needed).
    const float kBrushRadius = 0.45f;
    auto triCentroid = [&](uint32_t t, float* out) {
        uint32_t i0 = mesh.indices[t*3+0], i1 = mesh.indices[t*3+1], i2 = mesh.indices[t*3+2];
        out[0] = (mesh.pos_x[i0] + mesh.pos_x[i1] + mesh.pos_x[i2]) / 3.0f;
        out[1] = (mesh.pos_y[i0] + mesh.pos_y[i1] + mesh.pos_y[i2]) / 3.0f;
        out[2] = (mesh.pos_z[i0] + mesh.pos_z[i1] + mesh.pos_z[i2]) / 3.0f;
    };

    // ---- Dear ImGui (GLFW platform + WebGPU renderer) ----
    // GLFW callbacks were installed above; InitForOther chains to them. The UI is
    // drawn in its own loadOp=Load pass with no depth attachment, so the renderer
    // pipeline carries no depth-stencil state (DepthStencilFormat = Undefined).
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(win, true);
    ImGui_ImplWGPU_InitInfo imguiInit = {};
    imguiInit.Device = device;
    imguiInit.NumFramesInFlight = 3;
    imguiInit.RenderTargetFormat = format;
    imguiInit.DepthStencilFormat = WGPUTextureFormat_Undefined;
    if (!ImGui_ImplWGPU_Init(&imguiInit)) { std::printf("[win] ImGui_ImplWGPU_Init failed\n"); return 2; }
    std::printf("[win] ImGui ready\n");

    // ---- frame loop ----
    const char* framesEnv = std::getenv("CHISEL_PROBE_FRAMES");
    const long maxFrames = framesEnv ? std::atol(framesEnv) : -1;  // -1 = run until closed
    long frame = 0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (g_resized) {
            configureSurface(surface, device, format, g_fbw, g_fbh);
            makeDepth(device, g_fbw, g_fbh, &depthTex, &depthView);
            makePickTargets(device, g_fbw, g_fbh, &pick);
            // disc is aspect-corrected in NDC -> rebuild on resize
            discVerts = buildDisc(g_fbw, g_fbh);
            wgpuQueueWriteBuffer(queue, discVB, 0, discVerts.data(), discVerts.size()*sizeof(float));
            g_resized = false;
        }

        // ---- ImGui frame ----
        ImGui_ImplWGPU_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        static uint32_t uiTriid = 0xFFFFFFFFu;
        static uint32_t uiMasked = 0;
        bool clearMask = false;
        {
            ImGui::Begin("Chisel WebGPU");
            ImGui::Text("Stage 5: mask brush (compute) — hold LMB to paint");
            ImGui::Text("backend: wgpu-native v29  (%s)", IMGUI_VERSION);
            ImGui::Text("icosphere: %u verts / %u tris", vcount, icount / 3);
            ImGui::Text("%.1f FPS", (double)ImGui::GetIO().Framerate);
            ImGui::Separator();
            if (uiTriid == 0xFFFFFFFFu) ImGui::Text("under cursor: (background)");
            else                        ImGui::Text("under cursor: tri %u", uiTriid);
            ImGui::Text("last dab masked %u verts", uiMasked);
            clearMask = ImGui::Button("Clear mask");
            ImGui::SliderFloat("orbit pitch", &cam.pitch, -1.5f, 1.5f);
            ImGui::End();
        }
        ImGui::Render();
        ImGuiIO& io = ImGui::GetIO();

        // slow orbit so a screenshot reads as a 3D sphere, not a flat disc
        cam.yaw += 0.01f;
        float view[16], glProj[16], proj[16];
        cam.get_view_matrix(view);
        cam.get_projection_matrix(glProj, (float)g_fbw / (float)g_fbh);
        makeProjCorrected(proj, glProj);
        wgpuQueueWriteBuffer(queue, cameraUBO, 0,   view, 64);
        wgpuQueueWriteBuffer(queue, cameraUBO, 64,  proj, 64);

        if (clearMask) {
            std::vector<float> z(vcount, 0.0f);
            wgpuQueueWriteBuffer(queue, maskBuf, 0, z.data(), vcount * sizeof(float));
        }

        // ---- brush point: screen center for the headless auto-paint, else the cursor ----
        const bool headless = (maxFrames > 0);
        bool painting = false;
        uint32_t bx = (uint32_t)(g_fbw / 2), by = (uint32_t)(g_fbh / 2);
        if (headless) {
            painting = true;  // auto-paint the center each frame (verifiable without input)
        } else if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS
                   && !io.WantCaptureMouse) {
            int ww = 1, wh = 1; glfwGetWindowSize(win, &ww, &wh);
            double mxp = 0, myp = 0; glfwGetCursorPos(win, &mxp, &myp);
            if (ww > 0 && wh > 0) {
                long ix = (long)(mxp * g_fbw / ww), iy = (long)(myp * g_fbh / wh);
                ix = ix < 0 ? 0 : (ix > g_fbw - 1 ? g_fbw - 1 : ix);
                iy = iy < 0 ? 0 : (iy > g_fbh - 1 ? g_fbh - 1 : iy);
                bx = (uint32_t)ix; by = (uint32_t)iy;
                painting = true;
            }
        }

        // ========== pen-down seed: pick MRT pass + brush-point triid readback ==========
        // Render {depth, normal, triid} offscreen, copy the triangle id under the brush
        // point, read it back synchronously — exactly how the app seeds a stroke. Its own
        // submit so the CPU has the id before the mask dispatch below.
        {
            WGPUCommandEncoder penc = wgpuDeviceCreateCommandEncoder(device, nullptr);
            WGPURenderPassColorAttachment pickAtt[3] = {};
            pickAtt[0].view = pick.depthView;  pickAtt[0].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            pickAtt[0].loadOp = WGPULoadOp_Clear; pickAtt[0].storeOp = WGPUStoreOp_Store;
            pickAtt[0].clearValue = WGPUColor{0,0,0,0};
            pickAtt[1].view = pick.normalView; pickAtt[1].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            pickAtt[1].loadOp = WGPULoadOp_Clear; pickAtt[1].storeOp = WGPUStoreOp_Store;
            pickAtt[1].clearValue = WGPUColor{0,0,0,0};
            pickAtt[2].view = pick.triidView; pickAtt[2].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            pickAtt[2].loadOp = WGPULoadOp_Clear; pickAtt[2].storeOp = WGPUStoreOp_Store;
            pickAtt[2].clearValue = WGPUColor{4294967295.0,0,0,0};  // 0xFFFFFFFF = "no triangle"

            WGPURenderPassDepthStencilAttachment pickZ = {};
            pickZ.view = pick.zView;
            pickZ.depthLoadOp = WGPULoadOp_Clear; pickZ.depthStoreOp = WGPUStoreOp_Store;
            pickZ.depthClearValue = 1.0f;

            WGPURenderPassDescriptor pickRp = {};
            pickRp.colorAttachmentCount = 3; pickRp.colorAttachments = pickAtt;
            pickRp.depthStencilAttachment = &pickZ;

            WGPURenderPassEncoder pickPass = wgpuCommandEncoderBeginRenderPass(penc, &pickRp);
            wgpuRenderPassEncoderSetPipeline(pickPass, pickPipeline);
            wgpuRenderPassEncoderSetBindGroup(pickPass, 0, bindGroup, 0, nullptr);
            wgpuRenderPassEncoderSetVertexBuffer(pickPass, 0, pickPosVB, 0, WGPU_WHOLE_SIZE);
            wgpuRenderPassEncoderSetVertexBuffer(pickPass, 1, pickNrmVB, 0, WGPU_WHOLE_SIZE);
            wgpuRenderPassEncoderSetVertexBuffer(pickPass, 2, pickTriidVB, 0, WGPU_WHOLE_SIZE);
            wgpuRenderPassEncoderDraw(pickPass, exVcount, 1, 0, 0);
            wgpuRenderPassEncoderEnd(pickPass);
            wgpuRenderPassEncoderRelease(pickPass);

            // copy the brush-point pixel's triangle id into the staging buffer (256-aligned)
            WGPUTexelCopyTextureInfo copySrc = {};
            copySrc.texture = pick.triidTex; copySrc.mipLevel = 0;
            copySrc.origin = { bx, by, 0 }; copySrc.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferInfo copyDst = {};
            copyDst.buffer = readbackBuf;
            copyDst.layout.offset = 0;
            copyDst.layout.bytesPerRow = kReadbackBytes;
            copyDst.layout.rowsPerImage = 1;
            WGPUExtent3D copyExtent = { 1, 1, 1 };
            wgpuCommandEncoderCopyTextureToBuffer(penc, &copySrc, &copyDst, &copyExtent);

            WGPUCommandBuffer pcmd = wgpuCommandEncoderFinish(penc, nullptr);
            wgpuQueueSubmit(queue, 1, &pcmd);
            wgpuCommandBufferRelease(pcmd);
            wgpuCommandEncoderRelease(penc);
        }
        uint32_t triid = 0xFFFFFFFFu;
        readU32(device, readbackBuf, 1, &triid);
        uiTriid = triid;

        // ========== one mask dab: dispatch mask_paint over all verts ==========
        uint32_t masked = 0;
        if (painting && triid != 0xFFFFFFFFu && triid < (icount / 3)) {
            float anchor[3]; triCentroid(triid, anchor);
            MaskParamsGPU mp = {};
            mp.anchor_a[0] = anchor[0]; mp.anchor_a[1] = anchor[1]; mp.anchor_a[2] = anchor[2];
            mp.world_radius   = kBrushRadius;
            mp.hardness       = 0.5f;
            mp.paint_strength = 1.0f;
            mp.use_b          = 0u;
            mp.vertex_count   = vcount;
            wgpuQueueWriteBuffer(queue, maskParamsUBO, 0, &mp, sizeof(mp));
            uint32_t zero = 0;
            wgpuQueueWriteBuffer(queue, dirtyBuf, 0, &zero, sizeof(zero));  // reset dirty count

            WGPUCommandEncoder cenc = wgpuDeviceCreateCommandEncoder(device, nullptr);
            WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(cenc, nullptr);
            wgpuComputePassEncoderSetPipeline(cpass, maskPipeline);
            wgpuComputePassEncoderSetBindGroup(cpass, 0, maskBindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(cpass, (vcount + 255u) / 256u, 1, 1);
            wgpuComputePassEncoderEnd(cpass);
            wgpuComputePassEncoderRelease(cpass);
            wgpuCommandEncoderCopyBufferToBuffer(cenc, dirtyBuf, 0, dirtyStaging, 0, 4);
            WGPUCommandBuffer ccmd = wgpuCommandEncoderFinish(cenc, nullptr);
            wgpuQueueSubmit(queue, 1, &ccmd);
            wgpuCommandBufferRelease(ccmd);
            wgpuCommandEncoderRelease(cenc);
            readU32(device, dirtyStaging, 1, &masked);
            uiMasked = masked;
        }

        if (frame < 8) {
            if (triid == 0xFFFFFFFFu)
                std::printf("[win] frame %ld brush@(%u,%u): (background)\n", frame, bx, by);
            else
                std::printf("[win] frame %ld brush@(%u,%u): tri %u, masked %u verts this dab\n",
                            frame, bx, by, triid, masked);
        }

        // ===================== visible frame: mesh (with mask tint) + overlays + UI =====================
        WGPUSurfaceTexture st = {};
        wgpuSurfaceGetCurrentTexture(surface, &st);
        if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
            && st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            if (st.texture) wgpuTextureRelease(st.texture);
            configureSurface(surface, device, format, g_fbw, g_fbh);
            makeDepth(device, g_fbw, g_fbh, &depthTex, &depthView);
            continue;
        }

        WGPUTextureView view0 = wgpuTextureCreateView(st.texture, nullptr);

        WGPURenderPassColorAttachment color = {};
        color.view       = view0;
        color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        color.loadOp     = WGPULoadOp_Clear;
        color.storeOp    = WGPUStoreOp_Store;
        color.clearValue = WGPUColor{0.10, 0.45, 0.55, 1.0};  // Chisel teal

        WGPURenderPassDepthStencilAttachment depthAtt = {};
        depthAtt.view = depthView;
        depthAtt.depthLoadOp = WGPULoadOp_Clear;
        depthAtt.depthStoreOp = WGPUStoreOp_Store;
        depthAtt.depthClearValue = 1.0f;

        WGPURenderPassDescriptor rp = {};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &color;
        rp.depthStencilAttachment = &depthAtt;

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, posVB, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 1, nrmVB, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 2, maskBuf, 0, WGPU_WHOLE_SIZE);  // mask tint @loc2
        wgpuRenderPassEncoderSetIndexBuffer(pass, idxB, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(pass, icount, 1, 0, 0, 0);

        // overlay 1: world-space wireframe cube (reuses camera bind group, depth-tested)
        wgpuRenderPassEncoderSetPipeline(pass, linePipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, lineVB, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDraw(pass, lineCount, 1, 0, 0);

        // overlay 2: screen-space alpha-blended disc (no camera, no depth write)
        wgpuRenderPassEncoderSetPipeline(pass, discPipeline);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, discVB, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDraw(pass, discCount, 1, 0, 0);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        // ---- UI pass: load the rendered scene, draw ImGui on top, no depth ----
        WGPURenderPassColorAttachment uiColor = {};
        uiColor.view       = view0;
        uiColor.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        uiColor.loadOp     = WGPULoadOp_Load;   // keep the 3D scene already drawn
        uiColor.storeOp    = WGPUStoreOp_Store;
        WGPURenderPassDescriptor uiRp = {};
        uiRp.colorAttachmentCount = 1;
        uiRp.colorAttachments = &uiColor;
        WGPURenderPassEncoder uiPass = wgpuCommandEncoderBeginRenderPass(enc, &uiRp);
        ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), uiPass);
        wgpuRenderPassEncoderEnd(uiPass);
        wgpuRenderPassEncoderRelease(uiPass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuSurfacePresent(surface);

        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuTextureViewRelease(view0);
        wgpuTextureRelease(st.texture);

        if (++frame == 1) std::printf("[win] first frame presented\n");
        if (maxFrames > 0 && frame >= maxFrames) break;
    }
    std::printf("[win] presented %ld frame(s)\n", frame);

    // Final whole-mask readback (one-shot): count how many vertices ended up masked.
    // Proves the compute writes persisted in the SSBO end-to-end, not just per-dab.
    {
        WGPUBufferDescriptor msbd = {};
        msbd.usage = (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
        msbd.size  = (uint64_t)vcount * sizeof(float);
        WGPUBuffer maskStaging = wgpuDeviceCreateBuffer(device, &msbd);
        WGPUCommandEncoder menc = wgpuDeviceCreateCommandEncoder(device, nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(menc, maskBuf, 0, maskStaging, 0, msbd.size);
        WGPUCommandBuffer mcmd = wgpuCommandEncoderFinish(menc, nullptr);
        wgpuQueueSubmit(queue, 1, &mcmd);
        wgpuCommandBufferRelease(mcmd);
        wgpuCommandEncoderRelease(menc);

        MapResult mr;
        WGPUBufferMapCallbackInfo mcb = {};
        mcb.mode = WGPUCallbackMode_AllowProcessEvents; mcb.callback = onMap; mcb.userdata1 = &mr;
        wgpuBufferMapAsync(maskStaging, WGPUMapMode_Read, 0, msbd.size, mcb);
        while (!mr.done) wgpuDevicePoll(device, true, nullptr);
        if (mr.status == WGPUMapAsyncStatus_Success) {
            const float* m = (const float*)wgpuBufferGetMappedRange(maskStaging, 0, msbd.size);
            uint32_t nMasked = 0; float maxv = 0.0f;
            for (uint32_t i = 0; i < vcount; ++i) { if (m[i] > 0.5f) ++nMasked; if (m[i] > maxv) maxv = m[i]; }
            std::printf("[win] final mask: %u / %u verts masked (max value %.3f)\n", nMasked, vcount, maxv);
            wgpuBufferUnmap(maskStaging);
        } else {
            wgpuBufferUnmap(maskStaging);
        }
        wgpuBufferRelease(maskStaging);
    }

    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    wgpuComputePipelineRelease(maskPipeline);
    wgpuPipelineLayoutRelease(maskPL);
    wgpuBindGroupRelease(maskBindGroup);
    wgpuBindGroupLayoutRelease(maskBgl);
    wgpuShaderModuleRelease(maskModule);
    wgpuBufferRelease(dirtyStaging);
    wgpuBufferRelease(maskParamsUBO);
    wgpuBufferRelease(dirtyBuf);
    wgpuBufferRelease(maskBuf);
    wgpuBufferRelease(readbackBuf);
    wgpuRenderPipelineRelease(pickPipeline);
    wgpuShaderModuleRelease(pickModule);
    wgpuBufferRelease(pickTriidVB);
    wgpuBufferRelease(pickNrmVB);
    wgpuBufferRelease(pickPosVB);
    releasePickTargets(&pick);
    wgpuRenderPipelineRelease(discPipeline);
    wgpuShaderModuleRelease(discModule);
    wgpuBufferRelease(discVB);
    wgpuRenderPipelineRelease(linePipeline);
    wgpuShaderModuleRelease(lineModule);
    wgpuBufferRelease(lineVB);
    wgpuBindGroupRelease(bindGroup);
    wgpuPipelineLayoutRelease(pipelineLayout);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuBufferRelease(cameraUBO);
    wgpuBufferRelease(idxB);
    wgpuBufferRelease(nrmVB);
    wgpuBufferRelease(posVB);
    if (depthView) wgpuTextureViewRelease(depthView);
    if (depthTex)  wgpuTextureRelease(depthTex);
    wgpuRenderPipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
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
