// src/bench.cpp — passive frame-time recorder (see bench.h).
#include "bench.h"
#include "gpu/gpu.h"
#include "debug_console.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace bench {
namespace {

using Clock = std::chrono::steady_clock;

struct State {
    bool inited = false, on = false, vsync = true;
    FILE* csv = nullptr;
    Clock::time_point t_start, t_last;
    bool have_last = false;
    unsigned long long frame = 0;
    std::vector<std::pair<std::string, std::vector<float>>> samples;
};
State S;

const char* phase_name(Phase p) {
    switch (p) {
    case Phase::IDLE:   return "idle";
    case Phase::SCULPT: return "sculpt";
    case Phase::PENUP:  return "penup";
    case Phase::SWITCH: return "switch";
    }
    return "?";
}

std::vector<float>& bucket(const std::string& name) {
    for (auto& p : S.samples) if (p.first == name) return p.second;
    S.samples.emplace_back(name, std::vector<float>{});
    S.samples.back().second.reserve(4096);
    return S.samples.back().second;
}

}  // namespace

void init() {
    if (S.inited) return;
    S.inited = true;
    const char* on = std::getenv("CHISEL_PERF");
    S.on = on && *on && std::strcmp(on, "0") != 0;
    if (!S.on) return;
    if (const char* v = std::getenv("CHISEL_PERF_VSYNC")) S.vsync = std::atoi(v) != 0;
    if (const char* path = std::getenv("CHISEL_PERF_CSV")) {
        S.csv = std::fopen(path, "w");
        if (S.csv) std::fprintf(S.csv, "frame,t_s,ms,level,phase\n");
        else std::printf("[perf] cannot write CSV %s\n", path);
    }
    S.t_start = Clock::now();
    std::printf("[perf] recording frame times (vsync %s)%s\n",
                S.vsync ? "on" : "off", S.csv ? ", CSV on" : "");
}

bool active() { return S.on; }
bool vsync()  { return !S.on || S.vsync; }

void frame_end(int level, Phase phase) {
    if (!S.on) return;
    const Clock::time_point now = Clock::now();
    if (S.have_last) {
        const float ms = std::chrono::duration<float, std::milli>(now - S.t_last).count();
        char name[32];
        std::snprintf(name, sizeof name, "L%d %s", level, phase_name(phase));
        bucket(name).push_back(ms);
        if (S.csv)
            std::fprintf(S.csv, "%llu,%.4f,%.3f,%d,%s\n", S.frame,
                         std::chrono::duration<double>(now - S.t_start).count(), ms, level,
                         phase_name(phase));
    }
    ++S.frame;
    S.t_last = now;
    S.have_last = true;
}

void shutdown() {
    if (!S.on) return;
    if (S.csv) { std::fclose(S.csv); S.csv = nullptr; }
    // Level order, then idle/sculpt/penup/switch, so runs line up in a diff.
    std::sort(S.samples.begin(), S.samples.end(), [](const auto& a, const auto& b) {
        int la = std::atoi(a.first.c_str() + 1), lb = std::atoi(b.first.c_str() + 1);
        return la != lb ? la < lb : a.first < b.first;
    });
    std::printf("[perf] ---- frame times (ms), %.0f s session ----\n",
                std::chrono::duration<double>(Clock::now() - S.t_start).count());
    std::printf("[perf] %-12s %7s %8s %8s %8s %8s %8s %8s\n",
                "phase", "frames", "mean", "p50", "p95", "p99", "max", "fps");
    for (auto& p : S.samples) {
        std::vector<float>& v = p.second;
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        double sum = 0; for (float x : v) sum += x;
        const double mean = sum / v.size();
        auto pct = [&](double q) { return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))]; };
        std::printf("[perf] %-12s %7zu %8.2f %8.2f %8.2f %8.2f %8.2f %8.1f\n",
                    p.first.c_str(), v.size(), mean, pct(0.50), pct(0.95), pct(0.99), v.back(),
                    mean > 0 ? 1000.0 / mean : 0.0);
    }
    std::fflush(stdout);
}

// ---- Pen-down probe (see bench.h) -------------------------------------------------
namespace {
struct PenDown {
    int on = -1;                 // env, read once
    bool armed = false;
    int frame = 0;
    gpu::Device* dev = nullptr;
    gpu::Buffer sync;            // 16 bytes; reading it back drains the queue
    int frames_left = 0;
    Clock::time_point t_mark, t_frame;
};
PenDown P;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
}  // namespace

void pd_arm(gpu::Device& dev) {
    if (P.on < 0) { const char* e = std::getenv("CHISEL_PENDOWN"); P.on = (e && *e && *e != '0') ? 1 : 0; }
    if (!P.on) return;
    P.dev = &dev;
    if (!P.sync.handle) P.sync = gpu::create_buffer(dev, nullptr, 16, gpu::Usage::Storage);
    uint32_t junk[4];
    gpu::read_buffer(dev, P.sync, 0, sizeof(junk), junk);   // start from an empty queue
    P.armed = true;
    P.frame = 0;
    P.frames_left = 3;
    P.t_mark = P.t_frame = Clock::now();
    std::printf("[pendown] ---- pen down ----\n");
}

// CHISEL_LEVELTRACE=1: the same probe, armed when a subdivision level switch starts and
// kept on for the six frames after it (2026-11-11, the 63M-tri engine reset).
void pd_arm_level(gpu::Device& dev) {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("CHISEL_LEVELTRACE"); on = (e && *e && *e != '0') ? 1 : 0; }
    if (!on) return;
    P.on = 1;
    pd_arm(dev);
    P.frames_left = 6;
    std::printf("[pendown] ---- (level switch, not a pen-down) ----\n");
}

void pd_mark(const char* label) {
    if (!P.armed) return;
    const double cpu = ms_since(P.t_mark);
    const Clock::time_point t = Clock::now();
    uint32_t junk[4];
    gpu::read_buffer(*P.dev, P.sync, 0, sizeof(junk), junk);
    std::printf("[pendown] f%d %-34s cpu %7.2f ms | gpu wait %7.2f ms\n",
                P.frame, label, cpu, ms_since(t));
    P.t_mark = Clock::now();
}

void pd_frame() {
    if (!P.armed) return;
    std::printf("[pendown] f%d frame total %.1f ms\n", P.frame, ms_since(P.t_frame));
    std::fflush(stdout);
    P.t_mark = P.t_frame = Clock::now();
    ++P.frame;
    if (--P.frames_left <= 0) P.armed = false;
}


void copy_test(gpu::Device& dev, const gpu::Buffer& src_in, uint64_t bytes, int level) {
    static int want = -2, stable = 0;
    if (want == -2) { const char* e = std::getenv("CHISEL_COPY_TEST"); want = e ? std::atoi(e) : -1; }
    if (want < 0 || level != want) { stable = 0; return; }
    if (++stable < 150) return;

    // CHISEL_COPY_TEST_MB=<n>: a synthetic source of that size instead of the mesh's,
    // to tell "big copies are slow" apart from "copies are slow once the mesh is big".
    gpu::Buffer own;
    if (const char* e = std::getenv("CHISEL_COPY_TEST_MB")) {
        bytes = (uint64_t)std::atoi(e) * 1024ull * 1024ull;
        if (const char* x = std::getenv("CHISEL_COPY_TEST_EXTRA")) bytes += (uint64_t)std::atoi(x);
        own = gpu::create_buffer(dev, nullptr, bytes, gpu::Usage::Storage);
    }
    const gpu::Buffer& src = own.handle ? own : src_in;
    gpu::Buffer dst = gpu::create_buffer(dev, nullptr, bytes, gpu::Usage::Storage);
    gpu::Buffer sync = gpu::create_buffer(dev, nullptr, 16, gpu::Usage::Storage);
    uint32_t junk[4];
    auto drain = [&] { gpu::read_buffer(dev, sync, 0, sizeof(junk), junk); };
    const uint64_t MB = 1024ull * 1024ull;
    std::printf("[copytest] source %.1f MB at L%d\n", (double)bytes / MB, level);

    // 0 = one copy; otherwise chunk size in bytes. `split` = one submit per chunk.
    auto run = [&](const char* label, uint64_t total, uint64_t chunk, bool split) {
        for (int rep = 0; rep < 3; rep++) {
            drain();
            const Clock::time_point t0 = Clock::now();
            if (chunk == 0) {
                gpu::copy_buffer(dev, src, 0, dst, 0, total);
            } else if (split) {
                for (uint64_t o = 0; o < total; o += chunk)
                    gpu::copy_buffer(dev, src, o, dst, o, std::min(chunk, total - o));
            } else {
                gpu::ComputeBatch b = gpu::begin_compute(dev);
                for (uint64_t o = 0; o < total; o += chunk)
                    gpu::copy_buffer(b, src, o, dst, o, std::min(chunk, total - o));
                gpu::submit(b);
            }
            const double cpu = ms_since(t0);
            const Clock::time_point t1 = Clock::now();
            drain();
            std::printf("[copytest] %-34s %6.1f MB  cpu %7.2f ms | gpu wait %7.2f ms\n",
                        label, (double)total / MB, cpu, ms_since(t1));
        }
    };
    // Range copies: (offset, size), one submit.
    auto range = [&](const char* label, uint64_t off, uint64_t size) {
        for (int rep = 0; rep < 3; rep++) {
            drain();
            const Clock::time_point t0 = Clock::now();
            gpu::copy_buffer(dev, src, off, dst, off, size);
            const double cpu = ms_since(t0);
            const Clock::time_point t1 = Clock::now();
            drain();
            std::printf("[copytest] %-34s %6.1f MB  cpu %7.2f ms | gpu wait %7.2f ms\n",
                        label, (double)size / MB, cpu, ms_since(t1));
        }
    };
    std::printf("[copytest] bytes %llu (= %llu MiB + %llu)\n", (unsigned long long)bytes,
                (unsigned long long)(bytes / MB), (unsigned long long)(bytes % MB));
    if (bytes > 81 * MB) {
        range("prefix 80 MiB + 24 bytes", 0, 80 * MB + 24);
        range("prefix 80 MiB + 4 MiB", 0, 84 * MB);
        range("prefix 88 MiB", 0, 88 * MB);
        range("all but last 24 bytes", 0, bytes - 24);
        range("all but last 64 KiB", 0, (bytes - 65536) & ~65535ull);
        range("last 64 KiB only", bytes - 65536, 65536);
        range("last 24 bytes only", bytes - 24, 24);
        range("second half", (bytes / 2) & ~3ull, bytes - ((bytes / 2) & ~3ull));
        range("from 24 bytes in, to the end", 24, bytes - 24);
        range("middle 80 MiB (4 MiB in)", 4 * MB, 80 * MB);
    }
    for (uint64_t mb : {8ull, 16ull, 24ull, 32ull, 48ull, 64ull, 80ull}) {
        if (mb * MB > bytes) break;
        run("single copy", mb * MB, 0, false);
    }
    run("single copy, whole buffer", bytes, 0, false);
    run("16 MB chunks, one submit", bytes, 16 * MB, false);
    run("16 MB chunks, a submit each", bytes, 16 * MB, true);
    run("4 MB chunks, one submit", bytes, 4 * MB, false);
    std::fflush(stdout);
    debug_console::pump();   // stdout is piped through the ~ console; _Exit skips its drain
    std::_Exit(0);
}

}  // namespace bench
