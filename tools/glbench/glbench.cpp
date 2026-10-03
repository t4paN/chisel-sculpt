// glbench — how much does one "dab" of small param uploads + compute dispatches cost?
//
// Models chisel's per-dab GL traffic (gl_backend.cpp: write_buffer = glBufferSubData
// into a small GL_DYNAMIC_DRAW buffer, dispatch = bind + glDispatchCompute +
// glMemoryBarrier) without any chisel code, so the same binary can be timed on the
// Windows Intel driver and on Mesa and the difference attributed to the upload path
// (H1), the barriers (H2), or neither. See the small-brush handoff for the plan.
//
// One dab = K x { upload a 112-byte param block -> bind -> dispatch 64x64 threads
// that read the params and add into a 1M-float SSBO -> barrier }. A run is N dabs
// then glFinish; each variant does one warm-up run plus R timed runs and reports
// the median per-dab cost:
//   wall    CPU time from the first upload to glFinish returning
//   issue   CPU time to *issue* the N dabs (before glFinish) — if this is close to
//           wall, the CPU was blocked inside GL calls rather than queueing work
//   upload  CPU time inside the upload calls alone (part of issue)
//   gpu     GL_TIME_ELAPSED across the run
// Every variant's SSBO is checked against a CPU replay afterwards; variant C (no
// barriers) is allowed to fail that check, the others are not.
//
// Usage: glbench [--dabs N] [--k K] [--runs R] [--only A,B1,...] [--debug]
//   --debug  creates a debug context and prints the driver's distinct messages per
//            variant (Intel's driver reports stalls as PERFORMANCE messages).
//            Timings under --debug are not comparable with normal runs.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSsboFloats = 1u << 20;      // 1M floats
constexpr uint32_t kMask       = kSsboFloats - 1;
constexpr uint32_t kLocal      = 64;            // local_size_x
constexpr uint32_t kGroups     = 64;            // 64 groups x 64 = 4096 threads
constexpr uint32_t kThreads    = kLocal * kGroups;
constexpr uint32_t kParamBytes = 112;
constexpr GLsizeiptr kRingBytes = 4 << 20;      // 4 MB ring for B1/B2
constexpr int kRingSegments    = 4;             // B2: one fence per ring quarter

// std140 layout of the kernel's uniform block; exactly 112 bytes like chisel's.
struct Params {
    uint32_t base;
    uint32_t pad[3];
    float    v[6][4];
};
static_assert(sizeof(Params) == kParamBytes, "param block must be 112 bytes");

const char* kKernel = R"(#version 430
layout(local_size_x = 64) in;
layout(std140, binding = 0) uniform P { uint base; uint p0, p1, p2; vec4 v[6]; } p;
layout(std430, binding = 0) buffer D { float d[]; };
void main() {
    uint i = (gl_GlobalInvocationID.x + p.base) & 0xFFFFFu;
    d[i] += p.v[0].x;
}
)";

// Each upload moves the kernel's window, so a stale or misplaced param block shows
// up in the CPU replay instead of passing silently.
uint32_t base_for(uint64_t seq) { return (uint32_t)((seq * 5003u) & kMask); }

enum class Upload { SubDataSame, RingSubData, RingPersistent, Orphan };

const char* upload_name(Upload u) {
    switch (u) {
    case Upload::SubDataSame:    return "BufferSubData, same buffer";
    case Upload::RingSubData:    return "ring, BufferSubData + BindBufferRange";
    case Upload::RingPersistent: return "ring, persistent coherent map";
    case Upload::Orphan:         return "orphan (BufferData NULL) + SubData";
    }
    return "?";
}

struct Variant {
    std::string name;
    Upload upload;
    bool barriers;
};

struct Stats {
    double wall = 0, issue = 0, upload = 0, gpu = 0, wall_min = 0;
    bool verified = false;
    size_t mismatches = 0;
    bool ran = false;
};

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n == 0 ? 0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

// ---- debug messages (--debug) ------------------------------------------------
std::map<std::string, int> g_debug_msgs;

void APIENTRY on_debug(GLenum /*source*/, GLenum type, GLuint /*id*/, GLenum severity,
                       GLsizei /*length*/, const GLchar* msg, const void* /*user*/) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION && type == GL_DEBUG_TYPE_OTHER) return;
    const char* t = type == GL_DEBUG_TYPE_PERFORMANCE ? "PERFORMANCE"
                  : type == GL_DEBUG_TYPE_ERROR       ? "ERROR"
                                                      : "OTHER";
    g_debug_msgs[std::string(t) + ": " + msg]++;
}

// ---- the benchmark -----------------------------------------------------------
struct Bench {
    GLuint prog = 0, ssbo = 0, param = 0, ring = 0, query = 0;
    GLint ubo_align = 256;
    int dabs = 2000, k = 6, runs = 5;

    // ring state (B1/B2)
    GLuint pring = 0;            // persistent ring buffer
    uint8_t* pring_ptr = nullptr;
    GLintptr ring_off = 0, pring_off = 0;
    int pring_seg = 0;           // segment pring_off is writing into
    GLsync seg_fence[kRingSegments] = {};
    double fence_wait_us = 0;

    bool init() {
        GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(sh, 1, &kKernel, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048]; glGetShaderInfoLog(sh, sizeof log, nullptr, log);
            std::fprintf(stderr, "kernel compile failed:\n%s\n", log);
            return false;
        }
        prog = glCreateProgram();
        glAttachShader(prog, sh);
        glLinkProgram(prog);
        glDeleteShader(sh);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) { std::fprintf(stderr, "kernel link failed\n"); return false; }

        glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &ubo_align);

        glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, kSsboFloats * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        // created exactly like gpu::create_buffer
        glGenBuffers(1, &param);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, param);
        glBufferData(GL_SHADER_STORAGE_BUFFER, kParamBytes, nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &ring);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ring);
        glBufferData(GL_SHADER_STORAGE_BUFFER, kRingBytes, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // chisel's glad is generated for 4.3, so glBufferStorage comes via the ARB
        // extension (advertised by any 4.4+ driver)
        if (GLAD_GL_ARB_buffer_storage && glBufferStorage) {
            const GLbitfield f = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
            glGenBuffers(1, &pring);
            glBindBuffer(GL_UNIFORM_BUFFER, pring);
            glBufferStorage(GL_UNIFORM_BUFFER, kRingBytes, nullptr, f);
            pring_ptr = (uint8_t*)glMapBufferRange(GL_UNIFORM_BUFFER, 0, kRingBytes, f);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }
        glGenQueries(1, &query);
        return true;
    }

    bool has_persistent() const { return pring_ptr != nullptr; }

    GLintptr align_up(GLintptr x) const { return (x + ubo_align - 1) / ubo_align * ubo_align; }

    // Uploads the block and binds it at UBO 0, the way the variant prescribes.
    void upload(Upload u, const Params& p) {
        switch (u) {
        case Upload::SubDataSame:
            // gpu::write_buffer, then the bind done by gpu::dispatch
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, param);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kParamBytes, &p);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, param);
            break;
        case Upload::RingSubData: {
            if (ring_off + kParamBytes > kRingBytes) ring_off = 0;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ring);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, ring_off, kParamBytes, &p);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, ring, ring_off, kParamBytes);
            ring_off = align_up(ring_off + kParamBytes);
            break;
        }
        case Upload::RingPersistent: {
            const GLintptr seg_bytes = kRingBytes / kRingSegments;
            GLintptr off = pring_off;
            if (off + kParamBytes > kRingBytes) off = 0;
            const int seg = (int)(off / seg_bytes);
            if (seg != pring_seg) {
                // fence the segment we just left; before writing into the next one,
                // wait for the fence placed when it was last left
                if (seg_fence[pring_seg]) glDeleteSync(seg_fence[pring_seg]);
                seg_fence[pring_seg] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                pring_seg = seg;
                if (seg_fence[seg]) {
                    auto t0 = Clock::now();
                    glClientWaitSync(seg_fence[seg], GL_SYNC_FLUSH_COMMANDS_BIT, ~0ull);
                    fence_wait_us += us_since(t0);
                    glDeleteSync(seg_fence[seg]);
                    seg_fence[seg] = nullptr;
                }
            }
            std::memcpy(pring_ptr + off, &p, kParamBytes);
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, pring, off, kParamBytes);
            pring_off = align_up(off + kParamBytes);
            break;
        }
        case Upload::Orphan:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, param);
            glBufferData(GL_SHADER_STORAGE_BUFFER, kParamBytes, nullptr, GL_DYNAMIC_DRAW);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kParamBytes, &p);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, param);
            break;
        }
    }

    void reset_ssbo() {
        const float zero = 0.0f;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glFinish();
    }

    struct RunResult { double wall, issue, upload, gpu; };

    RunResult run_once(const Variant& v, uint64_t& seq) {
        Params p{};
        p.v[0][0] = 1.0f;
        double upload_us = 0;

        glUseProgram(prog);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
        glFinish();

        const auto t0 = Clock::now();
        glBeginQuery(GL_TIME_ELAPSED, query);
        for (int d = 0; d < dabs; ++d) {
            for (int r = 0; r < k; ++r) {
                p.base = base_for(seq++);
                const auto tu = Clock::now();
                upload(v.upload, p);
                upload_us += us_since(tu);
                glUseProgram(prog);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
                glDispatchCompute(kGroups, 1, 1);
                if (v.barriers)
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
            }
        }
        glEndQuery(GL_TIME_ELAPSED);
        const double issue_us = us_since(t0);
        glFinish();
        const double wall_us = us_since(t0);

        GLuint64 gpu_ns = 0;
        glGetQueryObjectui64v(query, GL_QUERY_RESULT, &gpu_ns);
        return {wall_us, issue_us, upload_us, gpu_ns / 1000.0};
    }

    // Replays the bases on the CPU and compares with the SSBO.
    size_t verify(uint64_t seq_count) {
        std::vector<float> want(kSsboFloats, 0.0f), got(kSsboFloats);
        for (uint64_t s = 0; s < seq_count; ++s) {
            const uint32_t b = base_for(s);
            for (uint32_t t = 0; t < kThreads; ++t) want[(t + b) & kMask] += 1.0f;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kSsboFloats * sizeof(float), got.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        size_t bad = 0;
        for (uint32_t i = 0; i < kSsboFloats; ++i) bad += want[i] != got[i];
        return bad;
    }

    Stats run_variant(const Variant& v) {
        Stats s;
        reset_ssbo();
        ring_off = 0;
        fence_wait_us = 0;
        uint64_t seq = 0;
        run_once(v, seq);  // warm-up: shader caches, buffer residency, clocks
        std::vector<double> wall, issue, up, gpu;
        for (int r = 0; r < runs; ++r) {
            RunResult rr = run_once(v, seq);
            wall.push_back(rr.wall / dabs);
            issue.push_back(rr.issue / dabs);
            up.push_back(rr.upload / dabs);
            gpu.push_back(rr.gpu / dabs);
        }
        s.wall = median(wall);
        s.wall_min = *std::min_element(wall.begin(), wall.end());
        s.issue = median(issue);
        s.upload = median(up);
        s.gpu = median(gpu);
        s.mismatches = verify(seq);
        s.verified = s.mismatches == 0;
        s.ran = true;
        return s;
    }
};

// ---- --readback: the pick-plane refresh ---------------------------------------
// Chisel's GL backend refreshes three full-screen planes (render_screen_buffers):
// depth R32F, normal RGB16F (read back as RGB float), triid R32UI, each with a
// synchronous glReadPixels. This times the alternatives on a frame the GPU has just
// rendered: a sphere-ish blob in the middle, cleared background around it, like the
// real planes.
//   kick  CPU time issuing render + reads (what a frame pays when reads are async)
//   wait  fence wait until the copies are done (overlaps later frames when async)
//   copy  CPU time getting the bytes into client memory (incl. any conversion)
namespace rb {

const char* kVs = R"(#version 430
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
const char* kFs = R"(#version 430
uniform vec2 size;
layout(location = 0) out float o_depth;
layout(location = 1) out vec3  o_normal;
layout(location = 2) out uint  o_triid;
void main() {
    vec2 c = (gl_FragCoord.xy - 0.5 * size) / (0.4 * size.y);
    float r2 = dot(c, c);
    if (r2 > 1.0) { o_depth = 1000.0; o_normal = vec3(0.0); o_triid = 0xFFFFFFFFu; return; }
    float z = sqrt(1.0 - r2);
    o_depth = 5.0 - z;
    o_normal = vec3(c, z);
    uvec2 q = uvec2(gl_FragCoord.xy) / 6u;
    o_triid = q.y * 512u + q.x;
}
)";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, sizeof log, nullptr, log); std::fprintf(stderr, "%s\n", log); }
    return s;
}

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { exp = 127 - 14; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) f = sign | 0x7F800000 | (man << 13);
    else f = sign | ((exp + 112) << 23) | (man << 13);
    float out; std::memcpy(&out, &f, 4); return out;
}

struct Plane { GLenum fmt, type; int bpp; };

struct ReadBench {
    int w = 1920, h = 1080;
    GLuint fbo = 0, tex[3] = {}, depth_rb = 0, prog = 0, vao = 0;
    GLint loc_size = -1;

    std::vector<uint8_t> client[3];

    bool init() {
        prog = glCreateProgram();
        GLuint vs = compile(GL_VERTEX_SHADER, kVs), fs = compile(GL_FRAGMENT_SHADER, kFs);
        glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog);
        glDeleteShader(vs); glDeleteShader(fs);
        GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) { std::fprintf(stderr, "readback program link failed\n"); return false; }
        loc_size  = glGetUniformLocation(prog, "size");
        glGenVertexArrays(1, &vao);

        // same internal formats as Renderer's screen_target attachments 0..2
        const GLint internal[3] = { GL_R32F, GL_RGB16F, GL_R32UI };
        const GLenum fmt[3] = { GL_RED, GL_RGB, GL_RED_INTEGER };
        const GLenum type[3] = { GL_FLOAT, GL_FLOAT, GL_UNSIGNED_INT };
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(3, tex);
        for (int i = 0; i < 3; ++i) {
            glBindTexture(GL_TEXTURE_2D, tex[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, internal[i], w, h, 0, fmt[i], type[i], nullptr);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
        }
        glGenRenderbuffers(1, &depth_rb);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_rb);
        const GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        glDrawBuffers(3, bufs);
        GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (st != GL_FRAMEBUFFER_COMPLETE) { std::fprintf(stderr, "FBO incomplete 0x%x\n", st); return false; }

        // what the driver says it reads natively, per attachment
        for (int i = 0; i < 3; ++i) {
            glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
            GLint f = 0, t = 0;
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &f);
            glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &t);
            std::printf("attachment %d: implementation read format 0x%04x type 0x%04x\n", i, f, t);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return true;
    }

    void render() {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, w, h);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);
        glUniform2f(loc_size, (float)w, (float)h);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
    }

    // The three planes as Chisel reads them today, or with the normal plane in a
    // narrower form (RGBA half: 8 B/px instead of 12, converted on the CPU).
    void planes(bool half_normal, Plane out[3]) const {
        out[0] = { GL_RED, GL_FLOAT, 4 };
        out[1] = half_normal ? Plane{ GL_RGBA, GL_HALF_FLOAT, 8 } : Plane{ GL_RGB, GL_FLOAT, 12 };
        out[2] = { GL_RED_INTEGER, GL_UNSIGNED_INT, 4 };
    }

    uint64_t checksum() const {   // depth + triid, to compare variants
        uint64_t s = 1469598103934665603ull;
        for (int i : {0, 2})
            for (size_t k = 0; k < client[i].size(); k += 4093) s = (s ^ client[i][k]) * 1099511628211ull;
        return s;
    }

    struct Result { double kick, wait, copy, total; uint64_t sum; double mb; };

    enum class Mode { Sync, PboMap, PboGet, PboClient };

    Result run(Mode mode, bool half_normal, int iters) {
        Plane pl[3]; planes(half_normal, pl);
        size_t bytes[3]; double mb = 0;
        for (int i = 0; i < 3; ++i) {
            bytes[i] = (size_t)pl[i].bpp * w * h;
            client[i].resize(i == 1 ? (size_t)12 * w * h : bytes[i]);
            mb += bytes[i] / 1e6;
        }
        std::vector<uint8_t> half_tmp(half_normal ? bytes[1] : 0);

        GLuint pbo[3] = {};
        void* mapped[3] = {};
        if (mode != Mode::Sync) {
            glGenBuffers(3, pbo);
            for (int i = 0; i < 3; ++i) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
                if (mode == Mode::PboClient) {
                    const GLbitfield f = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                                         GL_CLIENT_STORAGE_BIT;
                    glBufferStorage(GL_PIXEL_PACK_BUFFER, bytes[i], nullptr, f);
                    mapped[i] = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes[i],
                                                 GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
                } else {
                    glBufferData(GL_PIXEL_PACK_BUFFER, bytes[i], nullptr, GL_STREAM_READ);
                }
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }

        std::vector<double> kick, wait, copy, total;
        Result r{};
        for (int it = 0; it < iters + 1; ++it) {   // +1 warm-up
            glFinish();
            const auto t0 = Clock::now();
            render();
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            if (mode == Mode::Sync) {
                for (int i = 0; i < 3; ++i) {
                    glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
                    void* dst = (i == 1 && half_normal) ? (void*)half_tmp.data() : (void*)client[i].data();
                    glReadPixels(0, 0, w, h, pl[i].fmt, pl[i].type, dst);
                }
            } else {
                for (int i = 0; i < 3; ++i) {
                    glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
                    glReadPixels(0, 0, w, h, pl[i].fmt, pl[i].type, nullptr);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            glFlush();
            const double k_us = us_since(t0);

            const auto t1 = Clock::now();
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, ~0ull);
            glDeleteSync(fence);
            const double w_us = us_since(t1);

            const auto t2 = Clock::now();
            if (mode != Mode::Sync) {
                for (int i = 0; i < 3; ++i) {
                    void* dst = (i == 1 && half_normal) ? (void*)half_tmp.data() : (void*)client[i].data();
                    if (mode == Mode::PboClient) {
                        std::memcpy(dst, mapped[i], bytes[i]);
                    } else {
                        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]);
                        if (mode == Mode::PboGet) {
                            glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 0, bytes[i], dst);
                        } else {
                            void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes[i], GL_MAP_READ_BIT);
                            std::memcpy(dst, p, bytes[i]);
                            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                        }
                    }
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }
            if (half_normal) {
                const uint16_t* src = (const uint16_t*)half_tmp.data();
                float* dst = (float*)client[1].data();
                const size_t px = (size_t)w * h;
                for (size_t p = 0; p < px; ++p) {
                    dst[p * 3 + 0] = half_to_float(src[p * 4 + 0]);
                    dst[p * 3 + 1] = half_to_float(src[p * 4 + 1]);
                    dst[p * 3 + 2] = half_to_float(src[p * 4 + 2]);
                }
            }
            const double c_us = us_since(t2);
            if (it == 0) continue;
            kick.push_back(k_us / 1000); wait.push_back(w_us / 1000); copy.push_back(c_us / 1000);
            total.push_back((k_us + w_us + c_us) / 1000);
        }
        if (mode != Mode::Sync) {
            if (mode == Mode::PboClient)
                for (int i = 0; i < 3; ++i) { glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[i]); glUnmapBuffer(GL_PIXEL_PACK_BUFFER); }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glDeleteBuffers(3, pbo);
        }
        r.kick = median(kick); r.wait = median(wait); r.copy = median(copy); r.total = median(total);
        r.sum = checksum(); r.mb = mb;
        return r;
    }
};

int run_readback(int iters, int w, int h) {
    ReadBench b; b.w = w; b.h = h;
    if (!b.init()) return 1;
    std::printf("planes %dx%d: depth R32F, normal RGB16F, triid R32UI; median of %d refreshes\n\n", w, h, iters);
    std::printf("%-4s %-46s %8s %8s %8s %8s %8s %9s  %s\n", "", "path", "MB", "kick", "wait", "copy", "total",
                "MB/s", "checksum");
    std::printf("%-4s %-46s %8s %8s %8s %8s %8s\n", "", "", "", "ms", "ms", "ms", "ms");
    struct V { const char* name; const char* desc; ReadBench::Mode mode; bool half; };
    const V vs[] = {
        {"S",   "sync glReadPixels (= Chisel today)",          ReadBench::Mode::Sync,      false},
        {"S-h", "sync, normal as RGBA half + CPU convert",     ReadBench::Mode::Sync,      true},
        {"P",   "PBO STREAM_READ, map + memcpy",               ReadBench::Mode::PboMap,    false},
        {"P-g", "PBO STREAM_READ, glGetBufferSubData",         ReadBench::Mode::PboGet,    false},
        {"C",   "PBO CLIENT_STORAGE persistent map, memcpy",   ReadBench::Mode::PboClient, false},
        {"C-h", "C, normal as RGBA half + CPU convert",        ReadBench::Mode::PboClient, true},
        {"P-h", "P, normal as RGBA half + CPU convert",        ReadBench::Mode::PboMap,    true},
    };
    uint64_t ref = 0;
    for (const V& v : vs) {
        ReadBench::Result r = b.run(v.mode, v.half, iters);
        if (!ref) ref = r.sum;
        std::printf("%-4s %-46s %8.1f %8.2f %8.2f %8.2f %8.2f %9.0f  %s\n", v.name, v.desc, r.mb, r.kick, r.wait,
                    r.copy, r.total, r.mb / (r.total / 1000), r.sum == ref ? "same" : "DIFFERS");
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace rb

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

void usage() {
    std::fprintf(stderr,
        "usage: glbench [--dabs N] [--k K] [--runs R] [--only A,B1,B2,B3,C,D] [--debug]\n"
        "       glbench --readback [--size WxH] [--runs R]   (pick-plane readback paths)\n");
}

}  // namespace

int main(int argc, char** argv) {
    int dabs = 2000, k = 6, runs = 5;
    bool debug = false, readback = false;
    int rb_w = 1920, rb_h = 1080;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--dabs") dabs = std::atoi(next());
        else if (a == "--k") k = std::atoi(next());
        else if (a == "--runs") runs = std::atoi(next());
        else if (a == "--only") only = split_csv(next());
        else if (a == "--debug") debug = true;
        else if (a == "--readback") readback = true;
        else if (a == "--size") {
            if (std::sscanf(next(), "%dx%d", &rb_w, &rb_h) != 2) { usage(); return 2; }
        }
        else { usage(); return 2; }
    }
    if (dabs <= 0 || k <= 0 || runs <= 0) { usage(); return 2; }

    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (debug) glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFWwindow* win = glfwCreateWindow(64, 64, "glbench", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "could not create a GL 4.3 core context\n"); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "glad failed to load GL\n");
        return 1;
    }
    if (debug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(on_debug, nullptr);
    }

    if (readback) {
        std::printf("GL_RENDERER  %s\nGL_VERSION   %s\n", (const char*)glGetString(GL_RENDERER),
                    (const char*)glGetString(GL_VERSION));
        int rc = rb::run_readback(runs < 10 ? 20 : runs, rb_w, rb_h);
        glfwDestroyWindow(win);
        glfwTerminate();
        return rc;
    }

    Bench b;
    b.dabs = dabs; b.k = k; b.runs = runs;
    if (!b.init()) return 1;

    std::printf("GL_VENDOR    %s\n", (const char*)glGetString(GL_VENDOR));
    std::printf("GL_RENDERER  %s\n", (const char*)glGetString(GL_RENDERER));
    std::printf("GL_VERSION   %s\n", (const char*)glGetString(GL_VERSION));
    std::printf("UBO offset alignment %d, persistent mapping %s\n", b.ubo_align,
                b.has_persistent() ? "yes" : "NO (B2 skipped)");
    std::printf("dab = %d x (upload %u B -> bind -> dispatch %u threads -> barrier), "
                "%d dabs/run, median of %d runs%s\n\n",
                k, kParamBytes, kThreads, dabs, runs, debug ? "  [--debug: timings not comparable]" : "");

    auto wanted = [&](const char* n) {
        return only.empty() || std::find(only.begin(), only.end(), n) != only.end();
    };

    std::vector<Variant> order = {
        {"A",  Upload::SubDataSame,    true},
        {"B1", Upload::RingSubData,    true},
        {"B2", Upload::RingPersistent, true},
        {"B3", Upload::Orphan,         true},
        {"C",  Upload::SubDataSame,    false},
    };
    std::vector<std::pair<Variant, Stats>> results;

    std::printf("%-3s %-40s %-8s %9s %9s %9s %9s %9s  %s\n", "", "upload", "barrier",
                "wall", "min", "issue", "upload", "gpu", "check");
    std::printf("%-3s %-40s %-8s %9s %9s %9s %9s %9s\n", "", "", "", "us/dab", "us/dab",
                "us/dab", "us/dab", "us/dab");

    auto report = [&](const Variant& v, const Stats& s) {
        std::printf("%-3s %-40s %-8s %9.2f %9.2f %9.2f %9.2f %9.2f  %s",
                    v.name.c_str(), upload_name(v.upload), v.barriers ? "each" : "none",
                    s.wall, s.wall_min, s.issue, s.upload, s.gpu,
                    s.verified ? "ok" : (v.barriers ? "MISMATCH" : "mismatch (expected w/o barriers)"));
        if (!s.verified) std::printf(" [%zu floats]", s.mismatches);
        if (v.upload == Upload::RingPersistent && b.fence_wait_us > 0)
            std::printf(" (fence waits %.0f us total)", b.fence_wait_us);
        std::printf("\n");
        if (debug) {
            for (auto& [msg, n] : g_debug_msgs) std::printf("      x%-6d %s\n", n, msg.c_str());
            g_debug_msgs.clear();
        }
        std::fflush(stdout);
    };

    for (const Variant& v : order) {
        if (!wanted(v.name.c_str())) continue;
        if (v.upload == Upload::RingPersistent && !b.has_persistent()) continue;
        Stats s = b.run_variant(v);
        report(v, s);
        results.push_back({v, s});
    }

    if (wanted("D")) {
        // D = the fastest B upload, without barriers. If no B ran, use B2 (else B1).
        Variant best{"D", b.has_persistent() ? Upload::RingPersistent : Upload::RingSubData, false};
        double best_wall = 1e300;
        for (auto& [v, s] : results)
            if ((v.name == "B1" || v.name == "B2" || v.name == "B3") && s.wall < best_wall) {
                best_wall = s.wall;
                best.upload = v.upload;
            }
        Stats s = b.run_variant(best);
        report(best, s);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
