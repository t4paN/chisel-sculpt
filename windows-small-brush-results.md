# Windows small-brush slowness — glbench results (2026-10-03)

Answers Test 2 of `windows-small-brush-handoff.md` (on the SHARED partition).
Benchmark: `tools/glbench/` (standalone; build notes at the top of its CMakeLists.txt).

Machine: Intel Arc B570, Windows 10 IoT LTSC, driver 32.0.101.9033 (GL_VERSION reports
4.3.0). UBO offset alignment 16. One dab = 6 × (112-byte param upload → bind → dispatch
4096 threads → barrier), 2000 dabs per run, median of 5 runs. Two full runs, agreeing to
within a few µs; second run below.

| | upload | barrier | wall µs/dab | issue µs/dab | gpu µs/dab |
|---|---|---|---|---|---|
| **A** (= Chisel today) | BufferSubData, same buffer | each | **72.5** | 0.8 | 9.8 |
| B1 | ring, BufferSubData + BindBufferRange | each | 52.5 | 0.9 | 21.6 |
| **B2** | ring, persistent coherent map | each | **11.3** | 3.0 | 8.5 |
| B3 | orphan + SubData | each | 69.6 | 0.9 | 16.0 |
| C | as A | none | 60.2 | 0.9 | 28.6 |
| D | B2 | none | 29.8 | 3.0 | 28.1 |

All variants passed the CPU-replay check, C and D included.

## Reading

- **H1 confirmed on the Windows side.** A costs ~72 µs per dab, but the GPU only works
  ~10 µs of it. B2, with no `glBufferSubData` at all, drops to 11 µs/dab, with wall ≈ GPU
  time, so it's GPU-bound. That's **6.4× faster per dab**, from the upload path alone.
- The cost is hidden from the app thread in normal runs. `issue` is < 1 µs/dab because
  Intel's driver queues calls to a worker thread, and the time shows up inside
  `glFinish`/the next sync instead. With `--debug` (synchronous debug output, which
  serializes the driver onto the app thread) the upload calls themselves cost
  **~60 µs/dab under A (~10 µs per `glBufferSubData`) vs 0.4 µs under B2**. That's direct
  evidence the driver blocks on writes into a buffer the GPU still uses.
- **B1 only half-helps** (52 µs). As the handoff predicted, the driver tracks busy-ness
  per buffer, not per range, so sub-allocating one big buffer with `glBufferSubData`
  still syncs. **B3 (orphaning) doesn't help** (~65–70 µs). Only the persistent map
  avoids the driver's sync.
- **H2 not supported.** Dropping barriers helps A a little (72 → 60) but makes B2 *worse*
  (11 → 30), and GPU time rises without barriers. Overlapping dispatches on the same SSBO
  probably cost more than the barriers do. Barriers are not the problem; keep them.
- The driver printed no PERFORMANCE messages under `--debug`.

## Suggested fix

Per the handoff: a **param ring in `gl_backend.cpp`**. One `glBufferStorage`
(MAP_WRITE|PERSISTENT|COHERENT) buffer with a few MB of space. `write_buffer` on small
param buffers does a memcpy into the next aligned slice and records the offset. Dispatch
binds it with `glBindBufferRange`. Add a fence per ring segment (glbench uses 4 quarters;
it waited < 10 µs in total). Keep the per-dispatch barriers.

## Still to do

- **Linux numbers** for the same binary, to complete the "A ≫ B* on Windows but A ≈ B* on
  Linux" comparison: `cmake -S tools/glbench -B build-glbench && cmake --build
  build-glbench && ./build-glbench/glbench`.
- Test 1 (GL release vs Chrome WebGPU by hand) has not been reported from the Windows side.

## Build note (Windows)

Visual Studio's instance registration is currently missing on this box: `vswhere` finds
nothing, so CMake's "Visual Studio 17 2022" generator fails, although the compiler is
installed. glbench was built with `vcvars64.bat` + the Build Tools' bundled Ninja instead.
