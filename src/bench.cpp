// src/bench.cpp — passive frame-time recorder (see bench.h).
#include "bench.h"

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

}  // namespace bench
