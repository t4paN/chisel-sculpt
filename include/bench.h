#pragma once
#include <cstdint>

// Passive frame-time recorder for manual performance runs (CHISEL_PERF=1). Books
// every frame's wall time to a phase — "L<level> idle|sculpt|penup|switch" — prints
// per-phase percentiles at exit and, with CHISEL_PERF_CSV=<path>, writes one row per
// frame. It only watches: input, settings and the scene are untouched. Inert unless
// the env var is set.
//
//   CHISEL_PERF=1            enable
//   CHISEL_PERF_CSV=<path>   per-frame CSV (frame, t_s, ms, level, phase)
//   CHISEL_PERF_VSYNC=0      vsync off, to measure throughput instead of the 60 Hz cap

namespace gpu { struct Device; struct Buffer; }

namespace bench {

void init();          // read the env once; call before the swap interval / surface is set
bool active();
bool vsync();         // true unless recording with CHISEL_PERF_VSYNC=0

enum class Phase { IDLE, SCULPT, PENUP, SWITCH };

// Once per frame, after present. `level` is the active multires level.
void frame_end(int level, Phase phase);
void shutdown();      // print the summary, close the CSV

// Pen-down probe (CHISEL_PENDOWN=1). The stroke-stage timers open at the first dab,
// so they cannot see work done at pen-down, and a CPU clock alone cannot either: the
// GPU runs it later and the wait surfaces at present. Each mark therefore blocks until
// the queue has drained (a 16-byte read) and prints the CPU time since the previous
// mark and the GPU wait after it. That serialises the frame, so the parts are honest
// but their sum overstates the frame; never leave it on for a feel test.
void pd_arm(gpu::Device& dev);     // at pen-down; marks are no-ops until then
void pd_arm_level(gpu::Device& dev);   // CHISEL_LEVELTRACE=1: at a level switch, six frames
void pd_mark(const char* label);
void pd_frame();                   // after present; disarms three frames into the stroke

// CHISEL_COPY_TEST=<level>: once that level has been up for two seconds, time buffer
// copies out of `src` at a range of sizes and chunkings, print the table and exit.
// Pair with CHISEL_AUTO_SUBD. Exists to find why a whole-mesh copy costs CPU time.
void copy_test(gpu::Device& dev, const gpu::Buffer& src, uint64_t bytes, int level);

}  // namespace bench
