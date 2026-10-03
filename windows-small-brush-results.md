# Windows small-brush slowness — results (2026-10-03)

## Update: the main cause was input, not the GPU (fixed)

The user's complaint, made precise: with the **mouse**, fast small-brush strokes come
out as **corners** (straight segments between samples) on Windows, on GL and WebGPU
alike, while Linux is smooth. Frame rate was a steady 75 Hz (vsync) in every run, so this
was never frame time.

Cause: Windows doesn't queue mouse motion. `WM_MOUSEMOVE` is synthesized once per
message pump from the latest position, so `glfwPollEvents` delivers ~1 cursor position
per frame. `InputState::path_x` (built for exactly this problem, assuming the OS delivers
every position as X11/Wayland do) was therefore as sparse as no path at all. Measured:

| | positions/s | per frame |
|---|---|---|
| GLFW cursor callbacks (what Chisel got) | 65–79 | ~1.0 (`[path]` log: 1.1) |
| `GetMouseMovePointsEx` history (what the 500 Hz mouse reported) | 400–500 | ~6.6 |

Fix (commit on `chisel-windows`): `MouseHistory` in `main.cpp`, Windows-only. After each
`glfwPollEvents` it pulls the positions reported since the last frame from
`GetMouseMovePointsEx` and rebuilds the frame's path from them. It stands down during
slider drags and disabled-cursor modes, and forgets its position across a sculpt-mode
cursor wrap. Result: `[path]` 6.6–6.7 samples/frame on both backends, frame times
unchanged. The user confirmed small fast strokes are smooth now.

Known limit: the history is 64 deep, so a frame longer than ~130 ms (at 500 Hz) loses
its oldest points. That only bites during GL's ~240 ms pick-readback freezes, which are
still open (26 of them in 34 s on the v0.2.25 GL release; none on WebGPU).

**Web build (itch) on Windows is not affected.** Measured with a 500 Hz injected mouse
(`SendInput`; calibrated against the native probe, where it reproduces GLFW ~1/frame vs
history ~5–6.6/frame) on a test page counting `getCoalescedEvents()` per animation frame:
Chrome 154 6.67 positions/frame (499/s), Firefox 157 6.63/frame (496/s), at 75 fps, both
with 1 `pointermove` per frame. Both browsers recover the full rate on Windows, and the web
shell already feeds the coalesced positions into `path_x`. Only the native build had the
gap.

Pen/WinTab not yet tested. The WinTab context already receives packets at full rate,
but only pressure is used. Positions from WinTab may be needed if the pen also arrives
as one coalesced position per frame.

---

## glbench (GL upload path)

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

Apex Legends (DX12) was running in the background during those runs. Re-run twice with
the GPU idle: A 67.5 / 68.0, B1 52, B2 11.7 / 12.1, B3 60–65, C 54–56, D 27–30 µs/dab.
Same picture, so the conclusions below stand.

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
