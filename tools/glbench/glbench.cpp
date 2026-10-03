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
        "usage: glbench [--dabs N] [--k K] [--runs R] [--only A,B1,B2,B3,C,D] [--debug]\n");
}

}  // namespace

int main(int argc, char** argv) {
    int dabs = 2000, k = 6, runs = 5;
    bool debug = false;
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
