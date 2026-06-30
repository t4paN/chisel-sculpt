#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// --- Chisel debug assertions ---
// Active when CHISEL_DEBUG is defined (cmake -DCHISEL_DEBUG=ON).
// Zero overhead in release builds.

#ifdef CHISEL_DEBUG

#define CHISEL_ASSERT(cond, fmt, ...) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "\n[CHISEL ASSERT] %s:%d: " fmt "\n", \
                     __FILE__, __LINE__, ##__VA_ARGS__); \
        std::fflush(stderr); \
        std::abort(); \
    } } while(0)

#define CHISEL_WARN(fmt, ...) \
    std::fprintf(stderr, "[CHISEL WARN] %s:%d: " fmt "\n", \
                 __FILE__, __LINE__, ##__VA_ARGS__)

#else

#define CHISEL_ASSERT(cond, fmt, ...) ((void)0)
#define CHISEL_WARN(fmt, ...) ((void)0)

#endif

// --- Undo/multires index validation helpers ---

inline void chisel_check_local_index(uint32_t v, uint32_t voff, uint32_t vc,
                                     const char* ctx) {
#ifdef CHISEL_DEBUG
    if (v < voff) {
        std::fprintf(stderr, "[CHISEL ASSERT] %s: vertex %u below offset %u\n", ctx, v, voff);
        std::abort();
    }
    uint32_t lv = v - voff;
    if (lv >= vc) {
        std::fprintf(stderr, "[CHISEL ASSERT] %s: local index %u (v=%u, off=%u) >= count %u\n",
                     ctx, lv, v, voff, vc);
        std::abort();
    }
#else
    (void)v; (void)voff; (void)vc; (void)ctx;
#endif
}

inline void chisel_check_undo_entry(uint32_t vertex_offset, uint32_t mesh_vc,
                                    uint32_t base_vc, int disp_index,
                                    int disp_count, bool targets_base,
                                    const char* ctx) {
#ifdef CHISEL_DEBUG
    if (targets_base) {
        CHISEL_ASSERT(base_vc > 0,
            "%s: targets_base but base has 0 verts", ctx);
    } else {
        CHISEL_ASSERT(disp_index >= 0 && disp_index < disp_count,
            "%s: disp_index=%d out of range [0, %d)", ctx, disp_index, disp_count);
    }
#else
    (void)vertex_offset; (void)mesh_vc; (void)base_vc;
    (void)disp_index; (void)disp_count; (void)targets_base; (void)ctx;
#endif
}

// --- GL debug output ---
// Call chisel_init_gl_debug() after gladLoadGL. Requires GL_KHR_debug or GL 4.3+.
// Falls back to no-op on GL 3.3 without the extension, and on the WebGPU backend.

#if defined(CHISEL_DEBUG) && defined(CHISEL_BACKEND_GL)
#include <glad/glad.h>

inline void GLAPIENTRY chisel_gl_debug_callback(
    GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
{
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;

    const char* src_str = "?";
    switch (source) {
        case GL_DEBUG_SOURCE_API:             src_str = "API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   src_str = "Window"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: src_str = "Shader"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     src_str = "3rdParty"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     src_str = "App"; break;
    }
    const char* type_str = "?";
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               type_str = "ERROR"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: type_str = "DEPRECATED"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  type_str = "UNDEFINED"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         type_str = "PORTABILITY"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         type_str = "PERF"; break;
        case GL_DEBUG_TYPE_MARKER:              type_str = "MARKER"; break;
    }
    const char* sev_str = "?";
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:   sev_str = "HIGH"; break;
        case GL_DEBUG_SEVERITY_MEDIUM: sev_str = "MED"; break;
        case GL_DEBUG_SEVERITY_LOW:    sev_str = "LOW"; break;
    }
    std::fprintf(stderr, "[GL %s][%s][%s] id=%u: %s\n",
                 sev_str, src_str, type_str, id, message);
}

inline void chisel_init_gl_debug() {
    if (GLAD_GL_KHR_debug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(chisel_gl_debug_callback, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE,
                              GL_DEBUG_SEVERITY_NOTIFICATION,
                              0, nullptr, GL_FALSE);
        std::printf("[debug] GL debug output enabled (KHR_debug)\n");
    } else {
        std::printf("[debug] GL_KHR_debug not available, GL debug output disabled\n");
    }
}

inline void chisel_gl_label(GLenum type, GLuint obj, const char* name) {
    if (GLAD_GL_KHR_debug)
        glObjectLabel(type, obj, -1, name);
}

inline void chisel_gl_push_group(const char* name) {
    if (GLAD_GL_KHR_debug)
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, name);
}

inline void chisel_gl_pop_group() {
    if (GLAD_GL_KHR_debug)
        glPopDebugGroup();
}

#else

inline void chisel_init_gl_debug() {}
inline void chisel_gl_label(unsigned, unsigned, const char*) {}
inline void chisel_gl_push_group(const char*) {}
inline void chisel_gl_pop_group() {}

#endif
