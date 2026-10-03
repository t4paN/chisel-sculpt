#include "ui_backdrop.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(CHISEL_BACKEND_WEBGPU)
#include <webgpu/webgpu.h>
#include "gpu/gpu.h"
#else
#include <glad/glad.h>
#endif

// Three steps, same on both backends:
//   1. copy the frame into F (full resolution);
//   2. F -> A at quarter resolution: four bilinear taps = a 4x4 box, so a moving
//      model doesn't shimmer through the glass the way a plain 4x shrink would;
//   3. A -> B horizontal Gaussian, B -> A vertical Gaussian + saturation.
// A is what the panels sample. A and B are grow-only (with headroom) so a window
// drag-resize doesn't reallocate every frame; the used corner is tracked and the
// UVs scaled to it. Rows run bottom-up on GL and top-down on WebGPU; only uv_for
// has to know.

namespace ui_backdrop {
namespace {

int   g_win_w = 0, g_win_h = 0;     // frame size this capture was taken at
int   g_qw = 0, g_qh = 0;           // used quarter-res region of A/B
int   g_cap_w = 0, g_cap_h = 0;     // allocated A/B size
bool  g_ok = false;                 // a blurred frame exists this frame
bool  g_failed = false;             // init failed once: stop trying

int grow(int need, int have) {
    if (need <= have) return have;
    return need + need / 4 + 16;    // headroom: a resize drag reallocates rarely
}

struct PassParams {
    float scale[2];   // dest uv [0,1] -> source uv (used / allocated)
    float maxuv[2];   // clamp: never sample past the source's used region
    float texel[2];   // one source texel in uv
    float dir[2];     // blur direction (1,0) / (0,1)
    float sigma;      // in source texels
    float mode;       // 0 downsample, 1 blur, 2 blur + saturate
    float sat;
    float pad;
};

PassParams make_params(int src_used_w, int src_used_h, int src_cap_w, int src_cap_h,
                       float dx, float dy, float sigma, float mode, float sat) {
    PassParams p{};
    p.scale[0] = (float)src_used_w / (float)src_cap_w;
    p.scale[1] = (float)src_used_h / (float)src_cap_h;
    p.maxuv[0] = ((float)src_used_w - 0.5f) / (float)src_cap_w;
    p.maxuv[1] = ((float)src_used_h - 0.5f) / (float)src_cap_h;
    p.texel[0] = 1.0f / (float)src_cap_w;
    p.texel[1] = 1.0f / (float)src_cap_h;
    p.dir[0] = dx; p.dir[1] = dy;
    p.sigma = sigma; p.mode = mode; p.sat = sat;
    return p;
}

#if !defined(CHISEL_BACKEND_WEBGPU)
// ============================== OpenGL =======================================

const char* kVS = R"(#version 330 core
out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* kFS = R"(#version 330 core
uniform sampler2D u_src;
uniform vec2  u_scale, u_max, u_texel, u_dir;
uniform float u_sigma, u_mode, u_sat;
in vec2 v_uv;
out vec4 o;
vec3 tap(vec2 uv) { return texture(u_src, min(uv, u_max)).rgb; }
void main() {
    vec2 uv = v_uv * u_scale;
    vec3 c;
    if (u_mode < 0.5) {
        c = 0.25 * (tap(uv + u_texel * vec2(-1.0, -1.0)) + tap(uv + u_texel * vec2(1.0, -1.0)) +
                    tap(uv + u_texel * vec2(-1.0,  1.0)) + tap(uv + u_texel * vec2(1.0,  1.0)));
    } else {
        float s = max(u_sigma, 0.5);
        int r = min(int(ceil(s * 2.5)), 24);
        float wsum = 0.0;
        c = vec3(0.0);
        for (int i = -r; i <= r; i++) {
            float w = exp(-0.5 * float(i * i) / (s * s));
            c += w * tap(uv + u_dir * u_texel * float(i));
            wsum += w;
        }
        c /= wsum;
        if (u_mode > 1.5) {
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
            c = clamp(mix(vec3(l), c, u_sat), 0.0, 1.0);
        }
    }
    o = vec4(c, 1.0);
}
)";

GLuint g_prog = 0, g_vao = 0;
GLint  u_scale = -1, u_max = -1, u_texel = -1, u_dir = -1, u_sigma = -1, u_mode = -1, u_sat = -1;
GLuint g_texF = 0, g_fboF = 0;  int g_fw = 0, g_fh = 0;
GLuint g_tex[2] = {0, 0}, g_fbo[2] = {0, 0};   // [0] = A (result), [1] = B

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "[ui_backdrop] shader compile failed: %s\n", log);
        glDeleteShader(s); return 0;
    }
    return s;
}

bool init_gl() {
    GLuint vs = compile(GL_VERTEX_SHADER, kVS), fs = compile(GL_FRAGMENT_SHADER, kFS);
    if (!vs || !fs) return false;
    g_prog = glCreateProgram();
    glAttachShader(g_prog, vs); glAttachShader(g_prog, fs);
    glLinkProgram(g_prog);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint ok = 0; glGetProgramiv(g_prog, GL_LINK_STATUS, &ok);
    if (!ok) { std::fprintf(stderr, "[ui_backdrop] link failed\n"); return false; }
    u_scale = glGetUniformLocation(g_prog, "u_scale");
    u_max   = glGetUniformLocation(g_prog, "u_max");
    u_texel = glGetUniformLocation(g_prog, "u_texel");
    u_dir   = glGetUniformLocation(g_prog, "u_dir");
    u_sigma = glGetUniformLocation(g_prog, "u_sigma");
    u_mode  = glGetUniformLocation(g_prog, "u_mode");
    u_sat   = glGetUniformLocation(g_prog, "u_sat");
    glUseProgram(g_prog);
    glUniform1i(glGetUniformLocation(g_prog, "u_src"), 0);
    glUseProgram(0);
    glGenVertexArrays(1, &g_vao);
    return true;
}

void make_target(GLuint& tex, GLuint& fbo, int w, int h) {
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (!fbo) glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
}

void run_pass(GLuint dst_fbo, int vw, int vh, GLuint src_tex, const PassParams& p) {
    glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
    glViewport(0, 0, vw, vh);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, src_tex);
    glUniform2f(u_scale, p.scale[0], p.scale[1]);
    glUniform2f(u_max,   p.maxuv[0], p.maxuv[1]);
    glUniform2f(u_texel, p.texel[0], p.texel[1]);
    glUniform2f(u_dir,   p.dir[0], p.dir[1]);
    glUniform1f(u_sigma, p.sigma);
    glUniform1f(u_mode,  p.mode);
    glUniform1f(u_sat,   p.sat);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

bool capture_gl(float sigma_q, float saturate) {
    if (!g_prog && !init_gl()) return false;

    // 1. frame -> F (full res, grow-only)
    if (g_win_w > g_fw || g_win_h > g_fh) {
        g_fw = grow(g_win_w, g_fw); g_fh = grow(g_win_h, g_fh);
        make_target(g_texF, g_fboF, g_fw, g_fh);
    }
    if (g_qw > g_cap_w || g_qh > g_cap_h) {
        g_cap_w = grow(g_qw, g_cap_w); g_cap_h = grow(g_qh, g_cap_h);
        make_target(g_tex[0], g_fbo[0], g_cap_w, g_cap_h);
        make_target(g_tex[1], g_fbo[1], g_cap_w, g_cap_h);
    }

    GLboolean depth = glIsEnabled(GL_DEPTH_TEST), blend = glIsEnabled(GL_BLEND);
    GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST), cull = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_SCISSOR_TEST); glDisable(GL_CULL_FACE);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_fboF);
    glBlitFramebuffer(0, 0, g_win_w, g_win_h, 0, 0, g_win_w, g_win_h,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glUseProgram(g_prog);
    glBindVertexArray(g_vao);
    // 2. F -> A, 4x4 box
    run_pass(g_fbo[0], g_qw, g_qh, g_texF,
             make_params(g_win_w, g_win_h, g_fw, g_fh, 0, 0, 0, 0.0f, 1.0f));
    // 3. A -> B -> A
    run_pass(g_fbo[1], g_qw, g_qh, g_tex[0],
             make_params(g_qw, g_qh, g_cap_w, g_cap_h, 1, 0, sigma_q, 1.0f, 1.0f));
    run_pass(g_fbo[0], g_qw, g_qh, g_tex[1],
             make_params(g_qw, g_qh, g_cap_w, g_cap_h, 0, 1, sigma_q, 2.0f, saturate));

    glBindVertexArray(0);
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, g_win_w, g_win_h);
    if (depth) glEnable(GL_DEPTH_TEST);
    if (blend) glEnable(GL_BLEND);
    if (scissor) glEnable(GL_SCISSOR_TEST);
    if (cull) glEnable(GL_CULL_FACE);
    return true;
}

void shutdown_gl() {
    if (g_prog) glDeleteProgram(g_prog);
    if (g_vao) glDeleteVertexArrays(1, &g_vao);
    if (g_texF) glDeleteTextures(1, &g_texF);
    if (g_fboF) glDeleteFramebuffers(1, &g_fboF);
    glDeleteTextures(2, g_tex);
    glDeleteFramebuffers(2, g_fbo);
    g_prog = g_vao = g_texF = g_fboF = 0;
    g_tex[0] = g_tex[1] = g_fbo[0] = g_fbo[1] = 0;
}

#else
// ============================== WebGPU =======================================

const char* kWGSL = R"(
struct P { scale: vec2f, maxuv: vec2f, texel: vec2f, dir: vec2f,
           sigma: f32, mode: f32, sat: f32, pad: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VO {
    let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VO;
    o.pos = vec4f(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0, 0.0, 1.0);
    o.uv = q;
    return o;
}
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, smp, min(uv, p.maxuv), 0.0).rgb; }
@fragment fn fs_main(v: VO) -> @location(0) vec4f {
    let uv = v.uv * p.scale;
    var c = vec3f(0.0);
    if (p.mode < 0.5) {
        c = 0.25 * (tap(uv + p.texel * vec2f(-1.0, -1.0)) + tap(uv + p.texel * vec2f(1.0, -1.0)) +
                    tap(uv + p.texel * vec2f(-1.0,  1.0)) + tap(uv + p.texel * vec2f(1.0,  1.0)));
    } else {
        let s = max(p.sigma, 0.5);
        let r = min(i32(ceil(s * 2.5)), 24);
        var wsum = 0.0;
        for (var i = -r; i <= r; i++) {
            let w = exp(-0.5 * f32(i * i) / (s * s));
            c += w * tap(uv + p.dir * p.texel * f32(i));
            wsum += w;
        }
        c /= wsum;
        if (p.mode > 1.5) {
            let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
            c = clamp(mix(vec3f(l), c, p.sat), vec3f(0.0), vec3f(1.0));
        }
    }
    return vec4f(c, 1.0);
}
)";

WGPUStringView sv(const char* s) { return WGPUStringView{ s, s ? std::strlen(s) : 0 }; }

struct Tex { WGPUTexture tex = nullptr; WGPUTextureView view = nullptr; };

WGPUTextureFormat g_fmt = WGPUTextureFormat_Undefined;
WGPURenderPipeline g_pipe = nullptr;
WGPUBindGroupLayout g_bgl = nullptr;
WGPUPipelineLayout g_pl = nullptr;
WGPUSampler g_smp = nullptr;
WGPUBuffer g_ubo[3] = {nullptr, nullptr, nullptr};
WGPUBindGroup g_bg[3] = {nullptr, nullptr, nullptr};
Tex g_F, g_A, g_B;
int g_fw = 0, g_fh = 0;
// ImGui's WebGPU renderer caches one bind group per ImTextureID (the view pointer)
// and never evicts it. Releasing A's view on growth could let a later allocation
// reuse the address and inherit that stale bind group, so retired views are kept.
std::vector<WGPUTextureView> g_retired;

Tex make_tex(WGPUDevice dev, int w, int h, WGPUTextureUsage usage) {
    Tex t;
    WGPUTextureDescriptor td = {};
    td.usage = usage;
    td.dimension = WGPUTextureDimension_2D;
    td.size = { (uint32_t)w, (uint32_t)h, 1 };
    td.format = g_fmt;
    td.mipLevelCount = 1; td.sampleCount = 1;
    t.tex = wgpuDeviceCreateTexture(dev, &td);
    t.view = t.tex ? wgpuTextureCreateView(t.tex, nullptr) : nullptr;
    return t;
}

void drop_tex(Tex& t, bool retire_view) {
    if (t.view) { if (retire_view) g_retired.push_back(t.view); else wgpuTextureViewRelease(t.view); }
    if (t.tex) wgpuTextureRelease(t.tex);
    t = Tex{};
}

void drop_bind_groups() {
    for (auto& g : g_bg) if (g) { wgpuBindGroupRelease(g); g = nullptr; }
}

bool init_wgpu(WGPUDevice dev) {
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(kWGSL);
    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(dev, &smd);
    if (!mod) return false;

    WGPUBindGroupLayoutEntry e[3] = {};
    e[0].binding = 0;
    e[0].visibility = WGPUShaderStage_Fragment;
    e[0].buffer.type = WGPUBufferBindingType_Uniform;
    e[0].buffer.minBindingSize = sizeof(PassParams);
    e[1].binding = 1;
    e[1].visibility = WGPUShaderStage_Fragment;
    e[1].texture.sampleType = WGPUTextureSampleType_Float;
    e[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    e[2].binding = 2;
    e[2].visibility = WGPUShaderStage_Fragment;
    e[2].sampler.type = WGPUSamplerBindingType_Filtering;
    WGPUBindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 3; bgld.entries = e;
    g_bgl = wgpuDeviceCreateBindGroupLayout(dev, &bgld);
    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &g_bgl;
    g_pl = wgpuDeviceCreatePipelineLayout(dev, &pld);

    WGPUColorTargetState target = {};
    target.format = g_fmt;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState frag = {};
    frag.module = mod; frag.entryPoint = sv("fs_main");
    frag.targetCount = 1; frag.targets = &target;
    WGPURenderPipelineDescriptor pd = {};
    pd.layout = g_pl;
    pd.vertex.module = mod; pd.vertex.entryPoint = sv("vs_main");
    pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pd.primitive.frontFace = WGPUFrontFace_CCW;
    pd.primitive.cullMode = WGPUCullMode_None;
    pd.multisample.count = 1; pd.multisample.mask = 0xFFFFFFFFu;
    pd.fragment = &frag;
    g_pipe = wgpuDeviceCreateRenderPipeline(dev, &pd);
    wgpuShaderModuleRelease(mod);
    if (!g_pipe) return false;

    WGPUSamplerDescriptor sd = {};
    sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = WGPUFilterMode_Linear;
    sd.minFilter = WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.maxAnisotropy = 1;
    g_smp = wgpuDeviceCreateSampler(dev, &sd);

    for (auto& b : g_ubo) {
        WGPUBufferDescriptor bd = {};
        bd.size = sizeof(PassParams);
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        b = wgpuDeviceCreateBuffer(dev, &bd);
    }
    return true;
}

WGPUBindGroup make_bg(WGPUDevice dev, WGPUBuffer ubo, WGPUTextureView src) {
    WGPUBindGroupEntry e[3] = {};
    e[0].binding = 0; e[0].buffer = ubo; e[0].size = sizeof(PassParams);
    e[1].binding = 1; e[1].textureView = src;
    e[2].binding = 2; e[2].sampler = g_smp;
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout = g_bgl; bgd.entryCount = 3; bgd.entries = e;
    return wgpuDeviceCreateBindGroup(dev, &bgd);
}

void encode_pass(WGPUCommandEncoder enc, WGPUTextureView dst, int vw, int vh, WGPUBindGroup bg) {
    WGPURenderPassColorAttachment ca = {};
    ca.view = dst;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = WGPULoadOp_Load;
    ca.storeOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor rp = {};
    rp.colorAttachmentCount = 1; rp.colorAttachments = &ca;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)vw, (float)vh, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, (uint32_t)vw, (uint32_t)vh);
    wgpuRenderPassEncoderSetPipeline(pass, g_pipe);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

bool capture_wgpu(WGPUTexture frame, float sigma_q, float saturate) {
    if (!frame) return false;
    if (!(wgpuTextureGetUsage(frame) & WGPUTextureUsage_CopySrc)) return false;
    gpu::Device d = gpu::app_device();
    WGPUDevice dev = d.device;
    if (!dev) return false;

    WGPUTextureFormat fmt = wgpuTextureGetFormat(frame);
    if (fmt != g_fmt) {                               // first use (or a format change)
        shutdown();
        g_fmt = fmt;
        if (!init_wgpu(dev)) return false;
    }

    bool rebind = false;
    if (g_win_w > g_fw || g_win_h > g_fh) {
        g_fw = grow(g_win_w, g_fw); g_fh = grow(g_win_h, g_fh);
        drop_tex(g_F, false);
        g_F = make_tex(dev, g_fw, g_fh,
                       (WGPUTextureUsage)(WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding));
        rebind = true;
    }
    if (g_qw > g_cap_w || g_qh > g_cap_h) {
        g_cap_w = grow(g_qw, g_cap_w); g_cap_h = grow(g_qh, g_cap_h);
        WGPUTextureUsage u = (WGPUTextureUsage)(WGPUTextureUsage_RenderAttachment |
                                                WGPUTextureUsage_TextureBinding);
        drop_tex(g_A, true);
        drop_tex(g_B, false);
        g_A = make_tex(dev, g_cap_w, g_cap_h, u);
        g_B = make_tex(dev, g_cap_w, g_cap_h, u);
        rebind = true;
    }
    if (!g_F.view || !g_A.view || !g_B.view) return false;
    if (rebind || !g_bg[0]) {
        drop_bind_groups();
        g_bg[0] = make_bg(dev, g_ubo[0], g_F.view);
        g_bg[1] = make_bg(dev, g_ubo[1], g_A.view);
        g_bg[2] = make_bg(dev, g_ubo[2], g_B.view);
    }

    PassParams p0 = make_params(g_win_w, g_win_h, g_fw, g_fh, 0, 0, 0, 0.0f, 1.0f);
    PassParams p1 = make_params(g_qw, g_qh, g_cap_w, g_cap_h, 1, 0, sigma_q, 1.0f, 1.0f);
    PassParams p2 = make_params(g_qw, g_qh, g_cap_w, g_cap_h, 0, 1, sigma_q, 2.0f, saturate);
    wgpuQueueWriteBuffer(d.queue, g_ubo[0], 0, &p0, sizeof p0);
    wgpuQueueWriteBuffer(d.queue, g_ubo[1], 0, &p1, sizeof p1);
    wgpuQueueWriteBuffer(d.queue, g_ubo[2], 0, &p2, sizeof p2);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev, nullptr);
    WGPUTexelCopyTextureInfo src = {};
    src.texture = frame; src.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyTextureInfo dst = {};
    dst.texture = g_F.tex; dst.aspect = WGPUTextureAspect_All;
    WGPUExtent3D ext = { (uint32_t)g_win_w, (uint32_t)g_win_h, 1 };
    wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &ext);
    encode_pass(enc, g_A.view, g_qw, g_qh, g_bg[0]);
    encode_pass(enc, g_B.view, g_qw, g_qh, g_bg[1]);
    encode_pass(enc, g_A.view, g_qw, g_qh, g_bg[2]);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(d.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    return true;
}
#endif

} // namespace

void capture(int win_w, int win_h, float blur_px, float saturate, void* frame_texture) {
    g_ok = false;
    if (g_failed || blur_px < 0.5f || win_w < 8 || win_h < 8) return;
    g_win_w = win_w; g_win_h = win_h;
    g_qw = std::max(1, (win_w + 3) / 4);
    g_qh = std::max(1, (win_h + 3) / 4);
    const float sigma_q = blur_px / 4.0f;
#if defined(CHISEL_BACKEND_WEBGPU)
    g_ok = capture_wgpu((WGPUTexture)frame_texture, sigma_q, saturate);
#else
    (void)frame_texture;
    g_ok = capture_gl(sigma_q, saturate);
    if (!g_ok && !g_prog) g_failed = true;
#endif
}

bool available() { return g_ok; }

ImTextureID texture() {
#if defined(CHISEL_BACKEND_WEBGPU)
    return (ImTextureID)(intptr_t)g_A.view;
#else
    return (ImTextureID)(intptr_t)g_tex[0];
#endif
}

void uv_for(ImVec2 p0, ImVec2 p1, ImVec2* uv0, ImVec2* uv1) {
    const float su = (float)g_qw / (float)std::max(1, g_cap_w);
    const float sv_ = (float)g_qh / (float)std::max(1, g_cap_h);
    const float W = (float)std::max(1, g_win_w), H = (float)std::max(1, g_win_h);
    uv0->x = p0.x / W * su;
    uv1->x = p1.x / W * su;
#if defined(CHISEL_BACKEND_WEBGPU)
    uv0->y = p0.y / H * sv_;
    uv1->y = p1.y / H * sv_;
#else
    uv0->y = (1.0f - p0.y / H) * sv_;      // GL textures are bottom-up
    uv1->y = (1.0f - p1.y / H) * sv_;
#endif
}

void shutdown() {
    g_ok = false;
#if defined(CHISEL_BACKEND_WEBGPU)
    drop_bind_groups();
    for (auto& b : g_ubo) if (b) { wgpuBufferRelease(b); b = nullptr; }
    if (g_smp)  { wgpuSamplerRelease(g_smp); g_smp = nullptr; }
    if (g_pipe) { wgpuRenderPipelineRelease(g_pipe); g_pipe = nullptr; }
    if (g_pl)   { wgpuPipelineLayoutRelease(g_pl); g_pl = nullptr; }
    if (g_bgl)  { wgpuBindGroupLayoutRelease(g_bgl); g_bgl = nullptr; }
    drop_tex(g_F, false);
    drop_tex(g_A, true);
    drop_tex(g_B, false);
    g_fw = g_fh = g_cap_w = g_cap_h = 0;
    g_fmt = WGPUTextureFormat_Undefined;
#else
    shutdown_gl();
    g_fw = g_fh = g_cap_w = g_cap_h = 0;
#endif
}

} // namespace ui_backdrop
