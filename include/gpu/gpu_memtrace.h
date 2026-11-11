// CHISEL_GPU_MEM=1: a ledger of every live GPU buffer and the source line that made
// it. Included by the two backends only. Off (the default) costs one cached getenv.
//
// Why: graphics memory is the limit at tens of millions of triangles, and the driver's
// own figure (/proc/<pid>/fdinfo) says how much is held but not by what. 2026-11-11:
// stepping down from a 62.9M-tri level left gigabytes of buffers sized for it alive,
// and nothing could name them.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace gpu { namespace memtrace {

struct Entry { uint64_t size; const char* file; int line; };

inline bool on() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("CHISEL_GPU_MEM"); v = (e && *e && *e != '0') ? 1 : 0; }
    return v == 1;
}
inline std::unordered_map<uintptr_t, Entry>& ledger() {
    static std::unordered_map<uintptr_t, Entry> m;
    return m;
}
inline void add(uintptr_t handle, uint64_t size, const char* file, int line) {
    if (on() && handle) ledger()[handle] = { size, file, line };
}
inline void remove(uintptr_t handle) {
    if (on() && handle) ledger().erase(handle);
}
// Live buffers summed per creating line, biggest first, lines under min_bytes folded.
inline void dump(const char* tag, uint64_t min_bytes) {
    if (!on()) return;
    struct Site { const char* file; int line; uint64_t bytes; int n; };
    std::vector<Site> sites;
    uint64_t total = 0;
    for (const auto& kv : ledger()) {
        const Entry& e = kv.second;
        total += e.size;
        bool found = false;
        for (Site& s : sites)
            if (s.line == e.line && std::strcmp(s.file, e.file) == 0) { s.bytes += e.size; s.n++; found = true; break; }
        if (!found) sites.push_back({ e.file, e.line, e.size, 1 });
    }
    std::sort(sites.begin(), sites.end(), [](const Site& a, const Site& b) { return a.bytes > b.bytes; });
    std::printf("[gpu-mem] %s: %.0f MB in %zu buffers\n", tag, (double)total / 1048576.0, ledger().size());
    uint64_t rest = 0;
    for (const Site& s : sites) {
        if (s.bytes < min_bytes) { rest += s.bytes; continue; }
        const char* base = std::strrchr(s.file, '/');
        std::printf("[gpu-mem]   %7.0f MB  x%-3d %s:%d\n", (double)s.bytes / 1048576.0, s.n,
                    base ? base + 1 : s.file, s.line);
    }
    if (rest) std::printf("[gpu-mem]   %7.0f MB  (everything smaller)\n", (double)rest / 1048576.0);
    std::fflush(stdout);
}

}}  // namespace gpu::memtrace
