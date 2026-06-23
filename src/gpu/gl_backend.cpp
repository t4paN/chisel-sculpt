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

void dispatch(ComputeBatch&, ComputePipeline& pipe, BindGroup& group, uint32_t groups_x) {
    glUseProgram(pipe.handle);
    for (uint32_t i = 0; i < group.count; ++i)
        glBindBufferBase(group.target[i], group.binding[i], group.buffer[i]);
    glDispatchCompute(groups_x, 1, 1);
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

} // namespace gpu
