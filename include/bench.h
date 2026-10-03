#pragma once

// Passive frame-time recorder for manual performance runs (CHISEL_PERF=1). Books
// every frame's wall time to a phase — "L<level> idle|sculpt|penup|switch" — prints
// per-phase percentiles at exit and, with CHISEL_PERF_CSV=<path>, writes one row per
// frame. It only watches: input, settings and the scene are untouched. Inert unless
// the env var is set.
//
//   CHISEL_PERF=1            enable
//   CHISEL_PERF_CSV=<path>   per-frame CSV (frame, t_s, ms, level, phase)
//   CHISEL_PERF_VSYNC=0      vsync off, to measure throughput instead of the 60 Hz cap

namespace bench {

void init();          // read the env once; call before the swap interval / surface is set
bool active();
bool vsync();         // true unless recording with CHISEL_PERF_VSYNC=0

enum class Phase { IDLE, SCULPT, PENUP, SWITCH };

// Once per frame, after present. `level` is the active multires level.
void frame_end(int level, Phase phase);
void shutdown();      // print the summary, close the CSV

}  // namespace bench
