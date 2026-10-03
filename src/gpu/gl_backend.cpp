// src/gpu/gl_backend.cpp
// OpenGL 4.3 compute implementation of the gpu:: compute seam (gpu/gpu.h). The
// native sibling of webgpu_backend.cpp: same interface, immediate-mode underneath.
// A current GL context must already exist (the windowing/app code owns it).
#include "gpu/gpu.h"

#include <glad/glad.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace gpu {

static GLenum gl_target(Bind b) {
    return (b == Bind::Uniform) ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER;
}

Device gl_device() { return Device{}; }  // GL has no device object

static Device g_app_device;
void set_app_device(const Device& d) { g_app_device = d; }
Device app_device() { return g_app_device; }

static DeviceLimits g_device_limits;
void set_device_limits(const DeviceLimits& l) {
    g_device_limits = l;
    // Dev hook: CHISEL_LIMITS_MB clamps both limits — lets sizing guards (the
    // subdivision guard) be exercised on hardware whose real limits are too
    // big to hit. No effect when unset.
    if (const char* env = std::getenv("CHISEL_LIMITS_MB")) {
        const uint64_t cap = (uint64_t)std::strtoull(env, nullptr, 10) << 20;
        if (cap > 0) {
            if (g_device_limits.max_buffer_size          > cap) g_device_limits.max_buffer_size          = cap;
            if (g_device_limits.max_storage_binding_size > cap) g_device_limits.max_storage_binding_size = cap;
            std::printf("[gpu] CHISEL_LIMITS_MB dev clamp: %llu MB\n",
                        (unsigned long long)(cap >> 20));
        }
    }
}
DeviceLimits device_limits() { return g_device_limits; }

Buffer create_buffer(Device&, const void* data, uint64_t size, Usage /*usage*/) {
    // GL buffers carry no fixed usage role; the bind point (SSBO vs UBO) is decided
    // at bind time from the pipeline layout, so the Usage flags are advisory here.
    Buffer b; b.size = size;
    GLuint h = 0;
    glGenBuffers(1, &h);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, h);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    b.handle = h;
    return b;
}

void write_buffer(Device&, Buffer& b, uint64_t offset, const void* data, uint64_t size) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.handle);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)offset, (GLsizeiptr)size, data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void release_buffer(Buffer& b) {
    if (b.handle) { GLuint h = b.handle; glDeleteBuffers(1, &h); b.handle = 0; }
    b.size = 0;
}

// The ComputeBinding ids are global and run up to ~50, but GL only guarantees
// GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS >= 8 (Intel Arc on Windows reports 16), and
// a declared binding >= the limit fails to compile. No kernel uses more than a
// handful of SSBOs, so pack each kernel's SSBO bindings into slots 0..k-1 in
// declaration order and keep the logical -> slot map on the pipeline. UBOs are left
// alone (the limit there is 84, well above the 61-63 the shared blocks use).
static bool is_ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static std::string compact_ssbo_bindings(const char* glsl, ComputePipeline& p) {
    std::string s(glsl);
    size_t pos = 0;
    while ((pos = s.find("layout", pos)) != std::string::npos) {
        if (pos > 0 && is_ident_char(s[pos - 1])) { pos += 6; continue; }
        size_t open = s.find_first_not_of(" \t", pos + 6);
        if (open == std::string::npos || s[open] != '(') { pos += 6; continue; }
        size_t close = s.find(')', open);
        if (close == std::string::npos) break;
        // skip memory qualifiers between the layout and the block keyword
        size_t q = close + 1;
        std::string word;
        for (;;) {
            q = s.find_first_not_of(" \t\r\n", q);
            if (q == std::string::npos) break;
            size_t e = q;
            while (e < s.size() && is_ident_char(s[e])) ++e;
            word = s.substr(q, e - q);
            if (word == "readonly" || word == "writeonly" || word == "restrict" ||
                word == "coherent" || word == "volatile") { q = e; continue; }
            break;
        }
        if (word != "buffer") { pos = close; continue; }

        size_t b = s.find("binding", open);
        if (b == std::string::npos || b > close) { pos = close; continue; }
        size_t eq = s.find('=', b);
        size_t d0 = (eq == std::string::npos) ? std::string::npos : s.find_first_of("0123456789", eq);
        if (d0 == std::string::npos || d0 > close) { pos = close; continue; }
        size_t d1 = d0;
        while (d1 < close && s[d1] >= '0' && s[d1] <= '9') ++d1;
        uint32_t logical = (uint32_t)std::strtoul(s.c_str() + d0, nullptr, 10);

        uint32_t slot = p.ssbo_count;
        for (uint32_t k = 0; k < p.ssbo_count; ++k)
            if (p.ssbo_logical[k] == logical) { slot = k; break; }
        if (slot == p.ssbo_count) {
            if (p.ssbo_count == kMaxBindings) { pos = close; continue; }  // leave as-is; compile reports it
            p.ssbo_logical[p.ssbo_count++] = logical;
        }
        std::string repl = std::to_string(slot);
        s.replace(d0, d1 - d0, repl);
        pos = d0 + repl.size();
    }
    return s;
}

static uint32_t ssbo_slot(const ComputePipeline& p, uint32_t logical) {
    for (uint32_t k = 0; k < p.ssbo_count; ++k)
        if (p.ssbo_logical[k] == logical) return k;
    return logical;  // not declared by this kernel: bind where the caller asked
}

ComputePipeline create_compute_pipeline(Device&, const ShaderSources& src,
                                        const BindEntry* entries, uint32_t n,
                                        const char* /*entry_point*/) {
    ComputePipeline p;
    if (!src.glsl) { std::printf("[gpu] no GLSL source for compute pipeline\n"); return p; }

    const std::string glsl = compact_ssbo_bindings(src.glsl, p);
    const char* glsl_c = glsl.c_str();
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &glsl_c, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        std::printf("[gpu] compute shader compile failed:\n%s\n", log);
        glDeleteShader(sh); return p;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glDeleteShader(sh);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        std::printf("[gpu] compute program link failed:\n%s\n", log);
        glDeleteProgram(prog); return p;
    }
    p.handle = prog;
    // cache the bind layout (GL has no bind-group-layout object)
    p.binding_count = (n < kMaxBindings) ? n : kMaxBindings;
    for (uint32_t i = 0; i < p.binding_count; ++i) {
        p.binding_id[i]   = entries[i].binding;
        p.binding_type[i] = entries[i].type;
    }
    // A declared SSBO the layout doesn't list would read whatever is left in its
    // packed slot by the previous dispatch — flag it rather than sculpt garbage.
    for (uint32_t k = 0; k < p.ssbo_count; ++k) {
        bool listed = false;
        for (uint32_t i = 0; i < p.binding_count; ++i)
            if (p.binding_id[i] == p.ssbo_logical[k]) { listed = true; break; }
        if (!listed)
            std::printf("[gpu] warning: kernel declares SSBO binding %u but its layout doesn't list it\n",
                        p.ssbo_logical[k]);
    }
    return p;
}

void release_compute_pipeline(ComputePipeline& p) {
    if (p.handle) { glDeleteProgram(p.handle); p.handle = 0; }
    p.binding_count = 0;
}

BindGroup create_bind_group(Device&, ComputePipeline& pipe,
                            const BindBufferEntry* entries, uint32_t n) {
    BindGroup g;
    g.count = (n < kMaxBindings) ? n : kMaxBindings;
    for (uint32_t i = 0; i < g.count; ++i) {
        g.buffer[i]  = entries[i].buffer->handle;
        // resolve the GL target from the pipeline's layout for this binding
        GLenum tgt = GL_SHADER_STORAGE_BUFFER;
        for (uint32_t k = 0; k < pipe.binding_count; ++k)
            if (pipe.binding_id[k] == entries[i].binding) { tgt = gl_target(pipe.binding_type[k]); break; }
        g.target[i] = tgt;
        // SSBOs go to the kernel's packed slot (see compact_ssbo_bindings)
        g.binding[i] = (tgt == GL_SHADER_STORAGE_BUFFER) ? ssbo_slot(pipe, entries[i].binding)
                                                         : entries[i].binding;
    }
    return g;
}

void release_bind_group(BindGroup& g) { g.count = 0; }

ComputeBatch begin_compute(Device& dev) {
    ComputeBatch b; b.dev = &dev; return b;
}

void dispatch(ComputeBatch&, ComputePipeline& pipe, BindGroup& group, uint32_t groups_x, uint32_t groups_y) {
    glUseProgram(pipe.handle);
    for (uint32_t i = 0; i < group.count; ++i)
        glBindBufferBase(group.target[i], group.binding[i], group.buffer[i]);
    glDispatchCompute(groups_x, groups_y, 1);
    // make this dispatch's storage writes visible to the next dispatch / copy in the
    // batch (WebGPU does this implicitly between passes; GL needs the barrier).
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void dispatch_indirect(ComputeBatch&, ComputePipeline& pipe, BindGroup& group,
                       const Buffer& args, uint64_t offset) {
    glUseProgram(pipe.handle);
    for (uint32_t i = 0; i < group.count; ++i)
        glBindBufferBase(group.target[i], group.binding[i], group.buffer[i]);
    // The args were written by a compute dispatch in this same batch; GL needs an
    // explicit barrier before the command processor reads them (WebGPU does it).
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, args.handle);
    glDispatchComputeIndirect((GLintptr)offset);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

void end_compute_pass(ComputeBatch&) { /* no GL pass object */ }

void copy_buffer(ComputeBatch&, const Buffer& src, uint64_t src_off,
                 const Buffer& dst, uint64_t dst_off, uint64_t size) {
    glBindBuffer(GL_COPY_READ_BUFFER,  src.handle);
    glBindBuffer(GL_COPY_WRITE_BUFFER, dst.handle);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        (GLintptr)src_off, (GLintptr)dst_off, (GLsizeiptr)size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void submit(ComputeBatch&) {
    // dispatches/copies already executed; ensure subsequent readback + vertex use see them.
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void map_read(Device&, const Buffer& staging, uint64_t size, void* out) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, staging.handle);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)size, out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// ---- One-shot buffer ops -----------------------------------------------------

void read_buffer(Device&, const Buffer& src, uint64_t offset, uint64_t size, void* out) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, src.handle);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)offset, (GLsizeiptr)size, out);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void clear_buffer(Device&, Buffer& b, uint32_t fill_word) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.handle);
    // R32UI internal format clears the whole store to the repeated 32-bit pattern
    // (fill_word==0 is the common zero-clear; non-zero writes the SDF band sentinel).
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER,
                      GL_UNSIGNED_INT, &fill_word);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void copy_buffer(Device&, const Buffer& src, uint64_t src_off,
                 const Buffer& dst, uint64_t dst_off, uint64_t size) {
    glBindBuffer(GL_COPY_READ_BUFFER,  src.handle);
    glBindBuffer(GL_COPY_WRITE_BUFFER, dst.handle);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        (GLintptr)src_off, (GLintptr)dst_off, (GLsizeiptr)size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void resize_buffer(Device&, Buffer& b, uint64_t new_size, Usage /*usage*/) {
    // GL keeps the handle; an in-place glBufferData reallocates its store. (The
    // Usage role is advisory on GL — see create_buffer.)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.handle);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)new_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    b.size = new_size;
}

void barrier(Device&) {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}

// ============================ Render-side seam ============================
// Immediate-mode GL impl of the render primitives (gpu.h "Render-side seam").
// A current GL context must already exist. The pipeline owns a VAO; its vertex
// layout is replayed onto that VAO at draw time from the buffers bound on the pass
// (mirroring WebGPU's "no VAO, vertex buffers bound on the pass" model).

static GLuint compile_raster_stage(GLenum type, const char* src, const char* tag) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        std::printf("[gpu] %s shader compile failed:\n%s\n", tag, log);
        glDeleteShader(sh); return 0;
    }
    return sh;
}

// VertexFormat -> (component type, component count, normalized, is-integer-attr).
static void gl_vertex_format(VertexFormat f, GLenum& type, GLint& size,
                             GLboolean& norm, bool& is_int) {
    is_int = false; norm = GL_FALSE;
    switch (f) {
        case VertexFormat::F32:       type = GL_FLOAT; size = 1; break;
        case VertexFormat::F32x2:     type = GL_FLOAT; size = 2; break;
        case VertexFormat::F32x3:     type = GL_FLOAT; size = 3; break;
        case VertexFormat::F32x4:     type = GL_FLOAT; size = 4; break;
        case VertexFormat::U8x4_norm: type = GL_UNSIGNED_BYTE; size = 4; norm = GL_TRUE; break;
        default:                      type = GL_FLOAT; size = 1; break;
    }
}

static GLenum gl_topology(uint8_t t) {
    switch ((Topology)t) {
        case Topology::TriangleStrip: return GL_TRIANGLE_STRIP;
        case Topology::Lines:         return GL_LINES;
        case Topology::Points:        return GL_POINTS;
        case Topology::Triangles:
        default:                      return GL_TRIANGLES;
    }
}

RenderPipeline create_render_pipeline(Device&, const RenderPipelineDesc& d) {
    RenderPipeline p;
    if (!d.shaders.vert_glsl || !d.shaders.frag_glsl) {
        std::printf("[gpu] no GLSL vert/frag source for render pipeline\n");
        return p;
    }
    GLuint vs = compile_raster_stage(GL_VERTEX_SHADER,   d.shaders.vert_glsl, "vertex");
    GLuint fs = compile_raster_stage(GL_FRAGMENT_SHADER, d.shaders.frag_glsl, "fragment");
    if (!vs || !fs) { if (vs) glDeleteShader(vs); if (fs) glDeleteShader(fs); return p; }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        std::printf("[gpu] render program link failed:\n%s\n", log);
        glDeleteProgram(prog); return p;
    }
    p.handle = prog;
    glGenVertexArrays(1, &p.vao);

    // cache the vertex layout (replayed at draw) — strides indexed by slot
    p.attr_count = (d.attr_count < kMaxVertexAttrs) ? d.attr_count : kMaxVertexAttrs;
    for (uint32_t i = 0; i < p.attr_count; ++i) p.attrs[i] = d.attrs[i];
    for (uint32_t s = 0; s < d.slot_count && s < kMaxVertexAttrs; ++s)
        p.slot_stride[s] = d.slots[s].stride;

    // cache the UBO bind layout (GL has no BGL object)
    p.binding_count = (d.bind_count < kMaxBindings) ? d.bind_count : kMaxBindings;
    for (uint32_t i = 0; i < p.binding_count; ++i) {
        p.binding_id[i]   = d.binds[i].binding;
        p.binding_type[i] = d.binds[i].type;
    }

    p.topology    = (uint8_t)d.topology;
    p.depth_test  = d.depth_test  ? 1 : 0;
    p.depth_write = d.depth_write ? 1 : 0;
    p.blend       = d.blend       ? 1 : 0;
    return p;
}

void release_render_pipeline(RenderPipeline& p) {
    if (p.handle) { glDeleteProgram(p.handle); p.handle = 0; }
    if (p.vao)    { glDeleteVertexArrays(1, &p.vao); p.vao = 0; }
    p.attr_count = 0; p.binding_count = 0;
}

BindGroup create_bind_group(Device&, RenderPipeline& pipe,
                            const BindBufferEntry* entries, uint32_t n, const Texture* tex) {
    BindGroup g;
    g.count = (n < kMaxBindings) ? n : kMaxBindings;
    for (uint32_t i = 0; i < g.count; ++i) {
        g.binding[i] = entries[i].binding;
        g.buffer[i]  = entries[i].buffer->handle;
        GLenum tgt = GL_UNIFORM_BUFFER;  // render UBOs (storage only on the compute path)
        for (uint32_t k = 0; k < pipe.binding_count; ++k)
            if (pipe.binding_id[k] == entries[i].binding) { tgt = gl_target(pipe.binding_type[k]); break; }
        g.target[i] = tgt;
    }
    g.tex_handle = tex ? tex->handle : 0;
    return g;
}

RenderPass begin_render_pass(Device& dev, const RenderTarget& t) {
    RenderPass rp; rp.dev = &dev;
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(0, 0, t.width, t.height);
    if (t.clear) {
        glClearColor(t.clear_color[0], t.clear_color[1], t.clear_color[2], t.clear_color[3]);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    return rp;
}

void set_pipeline(RenderPass& rp, RenderPipeline& pipe) {
    rp.pipe = &pipe;
    glUseProgram(pipe.handle);
    if (pipe.depth_test) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    glDepthMask(pipe.depth_write ? GL_TRUE : GL_FALSE);
    if (pipe.blend) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glDisable(GL_BLEND);
    }
}

void set_bind_group(RenderPass&, RenderPipeline&, BindGroup& g) {
    for (uint32_t i = 0; i < g.count; ++i)
        glBindBufferBase(g.target[i], g.binding[i], g.buffer[i]);
    if (g.tex_handle) {                       // sampled_texture pipeline (font atlas)
        glActiveTexture(GL_TEXTURE0 + kTextureBinding);
        glBindTexture(GL_TEXTURE_2D, g.tex_handle);
    }
}

void set_vertex_buffer(RenderPass& rp, uint32_t slot, const Buffer& b) {
    if (slot < kMaxVertexAttrs) rp.vbuf[slot] = b.handle;
}

void set_index_buffer(RenderPass& rp, const Buffer& b) { rp.ibuf = b.handle; }

// Replay the pipeline's vertex layout onto its VAO from the buffers bound on the
// pass, then issue the draw. (GL needs the VAO + attrib pointers; WebGPU bakes the
// layout into the pipeline and binds vertex buffers on the pass directly.)
static void gl_setup_vao(RenderPass& rp) {
    RenderPipeline& pipe = *rp.pipe;
    glBindVertexArray(pipe.vao);
    for (uint32_t i = 0; i < pipe.attr_count; ++i) {
        const VertexAttr& a = pipe.attrs[i];
        glBindBuffer(GL_ARRAY_BUFFER, rp.vbuf[a.slot]);
        GLenum type; GLint size; GLboolean norm; bool is_int;
        gl_vertex_format(a.format, type, size, norm, is_int);
        glEnableVertexAttribArray(a.location);
        glVertexAttribPointer(a.location, size, type, norm,
                              (GLsizei)pipe.slot_stride[a.slot], (const void*)(size_t)a.offset);
    }
}

void draw(RenderPass& rp, uint32_t vertex_count, uint32_t first_vertex) {
    gl_setup_vao(rp);
    glDrawArrays(gl_topology(rp.pipe->topology), (GLint)first_vertex, (GLsizei)vertex_count);
    glBindVertexArray(0);
}

void draw_indexed(RenderPass& rp, uint32_t index_count) {
    gl_setup_vao(rp);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rp.ibuf);
    glDrawElements(gl_topology(rp.pipe->topology), (GLsizei)index_count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void end_render_pass(RenderPass& rp) {
    // Default-target passes leave their FBO bound for the frame; an offscreen pass
    // rebinds the default framebuffer so the next default pass / readback is correct.
    if (rp.offscreen) glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ---- Sampled texture (the font atlas) ----------------------------------------

Texture create_sampled_texture(Device&, int w, int h, const void* data) {
    Texture t; t.width = w; t.height = h;
    glGenTextures(1, &t.handle);
    glBindTexture(GL_TEXTURE_2D, t.handle);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   // R8 rows are unaligned
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

void release_texture(Texture& t) {
    if (t.handle) { glDeleteTextures(1, &t.handle); t.handle = 0; }
}

// ---- Offscreen MRT render target + readback ----------------------------------

// TexFormat -> (sized internal format, client format, client type) for glTexImage2D
// and glReadPixels.
static void gl_tex_format(TexFormat f, GLint& internal, GLenum& fmt, GLenum& type) {
    switch (f) {
        case TexFormat::R32F:  internal = GL_R32F;  fmt = GL_RED;         type = GL_FLOAT;        break;
        case TexFormat::RGB16F:internal = GL_RGB16F;fmt = GL_RGB;         type = GL_FLOAT;        break;
        case TexFormat::RG16F: internal = GL_RG16F; fmt = GL_RG;          type = GL_FLOAT;        break;
        case TexFormat::R32UI: internal = GL_R32UI; fmt = GL_RED_INTEGER; type = GL_UNSIGNED_INT; break;
        default:               internal = GL_R32F;  fmt = GL_RED;         type = GL_FLOAT;        break;
    }
}

OffscreenTarget create_offscreen_target(Device&, int w, int h,
                                        const TexFormat* fmts, uint32_t color_count) {
    OffscreenTarget t;
    t.width = w; t.height = h;
    t.color_count = (color_count < kMaxColorAttachments) ? color_count : kMaxColorAttachments;

    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);

    GLenum draw_bufs[kMaxColorAttachments];
    for (uint32_t i = 0; i < t.color_count; ++i) {
        t.color_fmt[i] = fmts[i];
        GLint internal; GLenum fmt, type;
        gl_tex_format(fmts[i], internal, fmt, type);
        glGenTextures(1, &t.color_tex[i]);
        glBindTexture(GL_TEXTURE_2D, t.color_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                               GL_TEXTURE_2D, t.color_tex[i], 0);
        draw_bufs[i] = GL_COLOR_ATTACHMENT0 + i;
    }

    glGenRenderbuffers(1, &t.depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, t.depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, t.depth_rbo);

    glDrawBuffers((GLsizei)t.color_count, draw_bufs);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
        std::printf("[gpu] offscreen target incomplete: 0x%x\n", status);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return t;
}

void resize_offscreen_target(Device&, OffscreenTarget& t, int w, int h) {
    if (w == t.width && h == t.height) return;
    t.width = w; t.height = h;
    for (uint32_t i = 0; i < t.color_count; ++i) {
        GLint internal; GLenum fmt, type;
        gl_tex_format(t.color_fmt[i], internal, fmt, type);
        glBindTexture(GL_TEXTURE_2D, t.color_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, type, nullptr);
    }
    glBindRenderbuffer(GL_RENDERBUFFER, t.depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
}

void release_offscreen_target(OffscreenTarget& t) {
    for (uint32_t i = 0; i < t.color_count; ++i)
        if (t.color_tex[i]) glDeleteTextures(1, &t.color_tex[i]);
    if (t.depth_rbo) glDeleteRenderbuffers(1, &t.depth_rbo);
    if (t.fbo) glDeleteFramebuffers(1, &t.fbo);
    t = OffscreenTarget();
}

RenderPass begin_offscreen_pass(Device& dev, OffscreenTarget& t, const OffscreenPassDesc& d) {
    RenderPass rp; rp.dev = &dev; rp.offscreen = true;
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(0, 0, t.width, t.height);

    // Draw-buffer mask: an enabled attachment maps location i → attachment i; a
    // disabled one maps to GL_NONE (the pick pass writes only depth + id).
    GLenum bufs[kMaxColorAttachments];
    for (uint32_t i = 0; i < t.color_count; ++i)
        bufs[i] = d.color[i].enabled ? (GL_COLOR_ATTACHMENT0 + i) : GL_NONE;
    glDrawBuffers((GLsizei)t.color_count, bufs);

    for (uint32_t i = 0; i < t.color_count; ++i) {
        if (!d.color[i].enabled || !d.color[i].clear) continue;
        if (d.color[i].is_uint) glClearBufferuiv(GL_COLOR, (GLint)i, d.color[i].u);
        else                    glClearBufferfv (GL_COLOR, (GLint)i, d.color[i].f);
    }
    if (d.clear_depth) glClear(GL_DEPTH_BUFFER_BIT);
    return rp;
}

void read_target_region(Device&, OffscreenTarget& t, uint32_t attachment,
                        int x, int y, int w, int h, void* out) {
    GLint internal; GLenum fmt, type;
    gl_tex_format(t.color_fmt[attachment], internal, fmt, type);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment);
    glReadPixels(x, t.height - y - h, w, h, fmt, type, out);  // GL origin is bottom-left
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    // glReadPixels fills rows bottom-up; the seam contract (and the webgpu backend)
    // is top-down, row 0 = the requested y. Swap in place so multi-row consumers
    // (the plane cache, the CPU mask fallback) can index both backends identically.
    if (h > 1) {
        size_t row = (size_t)texformat_out_bpp(t.color_fmt[attachment]) * (size_t)w;
        static std::vector<uint8_t> tmp;
        tmp.resize(row);
        uint8_t* p = (uint8_t*)out;
        for (int i = 0; i < h / 2; ++i) {
            uint8_t* a = p + (size_t)i * row;
            uint8_t* b = p + (size_t)(h - 1 - i) * row;
            std::memcpy(tmp.data(), a, row);
            std::memcpy(a, b, row);
            std::memcpy(b, tmp.data(), row);
        }
    }
}

// ---- Async readback tickets ----------------------------------------------------
// Buffer reads are genuinely asynchronous here too: the kick copies the range into a
// staging buffer on the GPU's own timeline and drops a fence behind it; ticket_ready
// polls the fence without waiting; ticket_take then copies out of a buffer the GPU has
// already finished with.
//
// This used to be a plain glGetBufferSubData at kick, which drains the whole GPU
// pipeline and then copies — once per brush dab, since every dab reads its dirty
// list back. The app side was already written for asynchronous tickets (the web
// build needs them), so only the timing changes: dab lists now land a frame or so
// later on GL, exactly as they always have on WebGPU.
//
// Texture reads (read_target_region_async, the screen-buffer plane cache) are, on
// Windows only, asynchronous the same way (elsewhere they stay synchronous, see there): glReadPixels into a staging buffer bound as the pixel
// pack buffer, then a fence. A synchronous full-screen glReadPixels of the three
// planes cost ~235 ms on Intel Arc / Windows (177 MB/s, tools/glbench --readback) —
// the "freeze" after every orbit and pen-up; through a pack buffer the kick is
// ~0.1 ms and the copy-out ~13 ms, a frame or two later.
//
// Copy-out maps the staging buffer and memcpys. On that same driver
// glGetBufferSubData (and mapping a CLIENT_STORAGE buffer) reads at ~15 MB/s, while
// mapping a STREAM_READ buffer reads at ~3 GB/s.

namespace {
struct GlTicket {
    std::vector<uint8_t> data;    // synchronous tickets: the bytes, already here
    GLuint   staging = 0;         // asynchronous tickets: the copy's destination
    uint64_t staging_cap = 0;
    GLsync   fence = nullptr;     // ...and the point the GPU must pass first
    uint64_t size = 0;
    // texture reads: glReadPixels packs rows bottom-up; ticket_take flips them to the
    // seam's top-down order while copying out
    uint32_t flip_rows = 0;
    size_t   row_bytes = 0;
};
std::unordered_map<uint32_t, GlTicket> g_gl_tickets;
uint32_t g_gl_next_ticket = 1;

// Staging buffers are recycled: dabs kick a read every few milliseconds, and a fresh
// glBufferData per read would allocate driver memory in the stroke loop. The pool is
// bounded in bytes so one whole-mesh read (pen-up fallback, 120 MB at L10) is not
// kept alive forever.
struct Staging { GLuint buf; uint64_t cap; };
std::vector<Staging> g_gl_staging_pool;
constexpr uint64_t kStagingPoolBytes = 256ull << 20;

Staging staging_acquire(uint64_t size) {
    int best = -1;
    for (int i = 0; i < (int)g_gl_staging_pool.size(); i++) {
        if (g_gl_staging_pool[i].cap < size) continue;
        if (best < 0 || g_gl_staging_pool[i].cap < g_gl_staging_pool[best].cap) best = i;
    }
    if (best >= 0) {
        Staging st = g_gl_staging_pool[best];
        g_gl_staging_pool.erase(g_gl_staging_pool.begin() + best);
        return st;
    }
    Staging st = { 0, size };
    glGenBuffers(1, &st.buf);
    glBindBuffer(GL_COPY_WRITE_BUFFER, st.buf);
    // STREAM_READ: written once by the GPU, read once by the CPU — the hint that puts
    // it where a CPU read after the fence is a memcpy, not another GPU round trip.
    glBufferData(GL_COPY_WRITE_BUFFER, (GLsizeiptr)size, nullptr, GL_STREAM_READ);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    return st;
}

void staging_release(GLuint buf, uint64_t cap) {
    if (!buf) return;
    uint64_t pooled = 0;
    for (const Staging& st : g_gl_staging_pool) pooled += st.cap;
    if (pooled + cap > kStagingPoolBytes) { glDeleteBuffers(1, &buf); return; }
    g_gl_staging_pool.push_back({ buf, cap });
}

void ticket_free(GlTicket& tk) {
    if (tk.fence) glDeleteSync(tk.fence);
    tk.fence = nullptr;
    staging_release(tk.staging, tk.staging_cap);
    tk.staging = 0;
}

uint32_t ticket_new_id() {
    uint32_t id = g_gl_next_ticket++;
    if (!g_gl_next_ticket) g_gl_next_ticket = 1;
    return id;
}
}  // namespace

ReadTicket read_buffer_async(Device&, const Buffer& src, uint64_t offset, uint64_t size) {
    if (!src.handle || size == 0) return 0;
    Staging st = staging_acquire(size);
    glBindBuffer(GL_COPY_READ_BUFFER,  src.handle);
    glBindBuffer(GL_COPY_WRITE_BUFFER, st.buf);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                        (GLintptr)offset, 0, (GLsizeiptr)size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    GlTicket tk;
    tk.staging = st.buf;
    tk.staging_cap = st.cap;
    tk.size = size;
    tk.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    // Without a flush the fence can sit in the driver's queue indefinitely and every
    // non-waiting poll in ticket_ready would say "not yet".
    glFlush();
    uint32_t id = ticket_new_id();
    g_gl_tickets.emplace(id, std::move(tk));
    return id;
}

ReadTicket read_target_region_async(Device& dev, OffscreenTarget& t, uint32_t attachment,
                                    int x, int y, int w, int h) {
    // Mirror the webgpu backend's bounds guard: a stale/straddling request yields a
    // failed ticket rather than an out-of-bounds read.
    if (attachment >= t.color_count ||
        x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > t.width || y + h > t.height)
        return 0;
#ifndef _WIN32
    // Everywhere but Windows the read stays synchronous. On Mesa the full three-plane
    // read is 14.5 ms and the planes are usable the same frame; through a pack buffer
    // they land ~60-80 ms later (tools/glbench --readback, Arc B570, Mesa 26.2), and
    // for those 5-6 frames after every pen-up and orbit the press latch (sculpt vs
    // orbit) and the cursor normal run on stale data — it felt "cranky". The async path
    // below only pays off on Intel's Windows driver, where the sync read is 235 ms.
    {
        GlTicket tk;
        tk.data.resize((size_t)texformat_out_bpp(t.color_fmt[attachment]) * w * h);
        tk.size = tk.data.size();
        read_target_region(dev, t, attachment, x, y, w, h, tk.data.data());
        uint32_t id = ticket_new_id();
        g_gl_tickets.emplace(id, std::move(tk));
        return id;
    }
#endif
    (void)dev;
    GLint internal; GLenum fmt, type;
    gl_tex_format(t.color_fmt[attachment], internal, fmt, type);
    const size_t row = (size_t)texformat_out_bpp(t.color_fmt[attachment]) * (size_t)w;
    const uint64_t size = (uint64_t)row * (uint64_t)h;

    Staging st = staging_acquire(size);
    GLint pack_align = 4;
    glGetIntegerv(GL_PACK_ALIGNMENT, &pack_align);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);   // rows tightly packed, as the seam expects
    glBindFramebuffer(GL_READ_FRAMEBUFFER, t.fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, st.buf);
    glReadPixels(x, t.height - y - h, w, h, fmt, type, nullptr);  // GL origin is bottom-left
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, pack_align);

    GlTicket tk;
    tk.staging = st.buf;
    tk.staging_cap = st.cap;
    tk.size = size;
    tk.flip_rows = (uint32_t)h;
    tk.row_bytes = row;
    tk.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();   // see read_buffer_async
    uint32_t id = ticket_new_id();
    g_gl_tickets.emplace(id, std::move(tk));
    return id;
}

void process_events(Device&) {}

bool ticket_ready(Device&, ReadTicket t) {
    if (t == 0) return true;                        // invalid tickets are "ready" (failed)
    auto it = g_gl_tickets.find(t);
    if (it == g_gl_tickets.end()) return true;      // unknown: ready, and take will fail
    GlTicket& tk = it->second;
    if (!tk.fence) return true;
    GLenum r = glClientWaitSync(tk.fence, 0, 0);    // poll, never wait
    return r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED;
}

bool ticket_take(Device&, ReadTicket t, void* out, uint64_t out_size) {
    auto it = g_gl_tickets.find(t);
    if (it == g_gl_tickets.end() || it->second.size != out_size) {
        std::memset(out, 0, (size_t)out_size);
        if (it != g_gl_tickets.end()) { ticket_free(it->second); g_gl_tickets.erase(it); }
        return false;
    }
    GlTicket& tk = it->second;
    bool ok = true;
    if (tk.fence) {
        // Callers poll ticket_ready first, so this normally returns at once. If one
        // didn't, waiting is still correct here — just not free.
        GLenum r = glClientWaitSync(tk.fence, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000000ull);
        ok = (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED);
        if (ok) {
            glBindBuffer(GL_COPY_READ_BUFFER, tk.staging);
            const uint8_t* src = (const uint8_t*)glMapBufferRange(
                GL_COPY_READ_BUFFER, 0, (GLsizeiptr)out_size, GL_MAP_READ_BIT);
            if (src) {
                uint8_t* dst = (uint8_t*)out;
                if (tk.flip_rows > 1) {
                    for (uint32_t r = 0; r < tk.flip_rows; ++r)
                        std::memcpy(dst + (size_t)r * tk.row_bytes,
                                    src + (size_t)(tk.flip_rows - 1 - r) * tk.row_bytes, tk.row_bytes);
                } else {
                    std::memcpy(dst, src, (size_t)out_size);
                }
                glUnmapBuffer(GL_COPY_READ_BUFFER);
            } else {
                ok = false;
                std::memset(out, 0, (size_t)out_size);
            }
            glBindBuffer(GL_COPY_READ_BUFFER, 0);
        } else {
            std::memset(out, 0, (size_t)out_size);
        }
    } else {
        std::memcpy(out, tk.data.data(), (size_t)out_size);
    }
    ticket_free(tk);
    g_gl_tickets.erase(it);
    return ok;
}

void ticket_drop(Device&, ReadTicket t) {
    auto it = g_gl_tickets.find(t);
    if (it == g_gl_tickets.end()) return;
    ticket_free(it->second);
    g_gl_tickets.erase(it);
}

} // namespace gpu
