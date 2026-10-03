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
// Call chisel_init_gl_debug(verbose) after gladLoadGL. Needs GL_KHR_debug, which
// every GL 4.3 context has and most 3.3 ones advertise; without it this is a no-op.
//
// ON IN EVERY GL BUILD, not just CHISEL_DEBUG ones: we have no NVIDIA or Windows
// AMD tester, and the driver's own "that's an error / that's undefined" messages
// are the nearest thing to one. The default level is chosen to stay silent on a
// healthy run: errors, undefined behaviour, portability and deprecation only,
// asynchronous (synchronous output stalls the driver), and each message id prints
// at most kGlDebugRepeat times so a per-frame fault can't flood the console.
// verbose (CHISEL_GL_DEBUG=1, or a CHISEL_DEBUG build) adds performance/other
// messages and makes output synchronous so it lands next to the call that caused it.

#if defined(CHISEL_BACKEND_GL)
#include <glad/glad.h>
#include <mutex>

static constexpr int kGlDebugRepeat = 3;

inline void GLAPIENTRY chisel_gl_debug_callback(
    GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
{
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;

    // Cap repeats per distinct MESSAGE, not per id: Mesa reuses one id for
    // unrelated errors, so an id cap hid the second fault behind the first.
    // FNV-1a of the text, fixed table, no allocation. Async output can arrive on
    // a driver thread, hence the lock. Once the table fills, new messages still
    // print, they just aren't capped.
    uint32_t key = 2166136261u ^ id;
    for (const GLchar* c = message; c && *c; ++c) key = (key ^ (uint8_t)*c) * 16777619u;
    static std::mutex mtx;
    static uint32_t seen_key[64];
    static int      seen_n[64];
    static int      seen_count = 0;
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        int k = 0;
        while (k < seen_count && seen_key[k] != key) ++k;
        if (k == seen_count && seen_count < 64) { seen_key[k] = key; seen_n[k] = 0; ++seen_count; }
        if (k < seen_count) n = ++seen_n[k];
    }
    if (n > kGlDebugRepeat) return;

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
    if (n == kGlDebugRepeat)
        std::fprintf(stderr, "[GL] that message repeated %d times, muting it\n", n);
}

inline void chisel_init_gl_debug(bool verbose) {
    if (!GLAD_GL_KHR_debug) {
        std::printf("[debug] GL_KHR_debug not available, GL debug output disabled\n");
        return;
    }
    glEnable(GL_DEBUG_OUTPUT);
    if (verbose) glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(chisel_gl_debug_callback, nullptr);
    // Filter at the driver, so muted classes cost nothing: start from nothing,
    // then enable the classes we want.
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    const GLenum always[] = { GL_DEBUG_TYPE_ERROR, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR,
                              GL_DEBUG_TYPE_PORTABILITY, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR };
    for (GLenum t : always)
        glDebugMessageControl(GL_DONT_CARE, t, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    if (verbose) {
        glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
        glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_OTHER,       GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }
    std::printf("[debug] GL debug output on (%s)\n",
                verbose ? "verbose, synchronous" : "errors + undefined behaviour");
}

#else

inline void chisel_init_gl_debug(bool) {}

#endif

#if defined(CHISEL_DEBUG) && defined(CHISEL_BACKEND_GL)

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

inline void chisel_gl_label(unsigned, unsigned, const char*) {}
inline void chisel_gl_push_group(const char*) {}
inline void chisel_gl_pop_group() {}

#endif
