// src/gpu/gl_backend.cpp
// OpenGL 4.3 compute implementation of the gpu:: compute seam (gpu/gpu.h). The
// native sibling of webgpu_backend.cpp: same interface, immediate-mode underneath.
// A current GL context must already exist (the windowing/app code owns it).
#include "gpu/gpu.h"

#include <glad/glad.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gpu {

static GLenum gl_target(Bind b) {
    return (b == Bind::Uniform) ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER;
}

Device gl_device() { return Device{}; }  // GL has no device object

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

ComputePipeline create_compute_pipeline(Device&, const ShaderSources& src,
                                        const BindEntry* entries, uint32_t n,
                                        const char* /*entry_point*/) {
    ComputePipeline p;
    if (!src.glsl) { std::printf("[gpu] no GLSL source for compute pipeline\n"); return p; }

    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src.glsl, nullptr);
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
        g.binding[i] = entries[i].binding;
        g.buffer[i]  = entries[i].buffer->handle;
        // resolve the GL target from the pipeline's layout for this binding
        GLenum tgt = GL_SHADER_STORAGE_BUFFER;
        for (uint32_t k = 0; k < pipe.binding_count; ++k)
            if (pipe.binding_id[k] == entries[i].binding) { tgt = gl_target(pipe.binding_type[k]); break; }
        g.target[i] = tgt;
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
                            const BindBufferEntry* entries, uint32_t n) {
    BindGroup g;
    g.count = (n < kMaxBindings) ? n : kMaxBindings;
    for (uint32_t i = 0; i < g.count; ++i) {
        g.binding[i] = entries[i].binding;
        g.buffer[i]  = entries[i].buffer->handle;
        GLenum tgt = GL_UNIFORM_BUFFER;  // render groups are UBO-only
        for (uint32_t k = 0; k < pipe.binding_count; ++k)
            if (pipe.binding_id[k] == entries[i].binding) { tgt = gl_target(pipe.binding_type[k]); break; }
        g.target[i] = tgt;
    }
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

void end_render_pass(RenderPass&) { /* no GL pass object; FBO stays bound for the frame */ }

} // namespace gpu
