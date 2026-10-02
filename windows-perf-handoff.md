# Windows performance handoff — 2026-10-02

Branch `chisel-windows` (local only, fetch it off the Windows disk). Same machine
as the Linux dev setup: **Intel Arc B570**, 75 Hz monitor, dual boot. So every
Windows-vs-Linux difference below is **driver/OS, not hardware**: Windows runs
Intel's proprietary GL driver (32.0.101.9033) and wgpu-native on D3D12/Vulkan;
Linux runs Mesa.

## TL;DR

1. **v0.2.24 crashed at launch on every Windows box** — a CRT difference, not a
   perf problem. Fixed (`bd5af6c`).
2. **On the GL build, most compute kernels were dead on Windows** — Intel's
   Windows driver caps SSBO bindings at 16 and our ids go to 50. Brushes,
   cascade and remesh fell back to the CPU or did nothing. That was most of the
   "feels off" in v0.2.20. Fixed (`bd5af6c`).
3. **Still open: GL freezes one frame for ~250 ms on every pen-up/press/level
   switch.** The full-screen pick-plane readback is synchronous `glReadPixels`,
   and Intel's Windows GL driver does it at ~160 MB/s. WebGPU does the same read
   async and is smooth — the user confirmed pen-up "felt much better" on WebGPU.
4. **Sculpting itself is fine on both backends**: both held 75 fps (vsync cap)
   through L7–L8 strokes. Headroom above the cap is not measured yet.

## 1. Startup crash (fixed)

`debug_console::init` did `setvbuf(stdout, nullptr, _IOLBF, 0)`. glibc accepts
size 0; the MS CRT rejects size 0 for `_IOLBF`/`_IOFBF` via its invalid-parameter
handler → `__fastfail` → `0xC0000409` before the first log line (stdout is
captured, so nothing printed). MSVC also has no real line buffering. Fix: `_IONBF`
on `_WIN32`. Came in with the `~` console (`21f106d`), which is why v0.2.20 ran
and v0.2.24 didn't. **Why not on Linux:** glibc-only behaviour.

## 2. SSBO binding limit (fixed)

| | Windows Intel GL | Linux Mesa |
|---|---|---|
| `GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS` | **16** (same for 3.3 / 4.3 / 4.6 contexts) | high enough for ids up to 50 (it worked) — confirm with `glxinfo -l \| grep SHADER_STORAGE` |
| `GL_MAX_UNIFORM_BUFFER_BINDINGS` | 84 | — |

The GL spec only guarantees 8. Any kernel declaring `binding >= 16` failed to
compile (`[compute] ... pipeline failed to compile`). The app kept running with
those paths dead: brush accum kernels, mask/color/density paint, multires
diff/apply, the cascade (`level switch stays CPU`), and remesh select/smooth.

Fix lives entirely in `gl_backend.cpp`. `create_compute_pipeline` packs each
kernel's SSBO bindings into slots `0..k-1` by rewriting the GLSL source before
compiling, and stores the id → slot map on the pipeline. `create_bind_group`
binds to the packed slot. Shader files and ids are unchanged. UBOs (61–63) are
untouched. It logs if a kernel declares an SSBO its layout doesn't list; none do.
All 38 pipelines now compile on both backends.

Side note: the main window still asks for a **3.3** context and gets compute
through extensions. It works, but `#version 430` kernels on a 3.3 context are
outside the spec. Requesting 4.3 doesn't change any limit on this driver.

## 3. GL pick-plane readback stall (OPEN)

After every render that dirties the screen buffers (pen-up, press, level switch,
camera settle), `Renderer` reads three full-screen planes back
(`renderer.cpp:~2105`, `read_target_region_async`). The GL backend does this
**synchronously**, despite the name.

Measured at 1920×1061, `glFinish` before each read so only the read is timed:

| plane | format | bytes | sync `glReadPixels` |
|---|---|---|---|
| depth | R32F | 8.1 MB | ~50 ms |
| normal | RGB16F → float RGB | 24.4 MB | **~135 ms** |
| tri id | R32UI | 8.1 MB | ~50 ms |
| total | | 40.7 MB | **~240 ms** → one frozen frame |

Manual sessions, vsync on (75 Hz = 13.33 ms):

| | GL | WebGPU |
|---|---|---|
| sculpt frames p50 / p95 / max | 13.4 / 13.7 / 13.9 ms | 13.3 / 13.4 / 20 ms |
| pen-up commit (`[penup]`) | median 14 ms | median 13 ms |
| hitches > 100 ms in a 10 s session | **9, all 245–260 ms** | **0** (worst idle frame 23 ms) |

Each GL hitch lands on the frame right after a release, or the press frame,
or the frame after a level switch.

**Why not (or less) on Linux — inference, not measured:** the code was designed
around this being cheap. The old renderer comment said GL tickets "resolved
synchronously at kick, so ... a pen-down press never sees a not-ready gap".
Mesa's iris path does `glReadPixels` as a GPU blit into linear memory before the
CPU copy. Intel's Windows driver appears to de-tile and convert on the CPU (the
3-channel RGB16F → float plane is 3x slower per byte). **Please measure on Linux**
with the recorder (§5) before relying on this.

### What I tried on Windows (reverted, not on the branch)

| attempt | result |
|---|---|
| `glReadPixels` into a PBO + fence, take after the fence signals | kick < 3 ms, fence signals at once, then `glGetBufferSubData` takes **540 ms per 8 MB, 1.7 s for 24 MB** (~15 MB/s). The driver defers the de-tile to the first CPU touch. Frames went to ~3 s. |
| same, staging via `glBufferStorage(MAP_READ \| CLIENT_STORAGE)` | no change |
| compute kernel `texelFetch` → SSBO → `glCopyBufferSubData` to staging → fence | no change, ~3 s frames |

So on this driver, **CPU copy-out of large GPU-written buffers is ~15 MB/s**
however the data got there. Not yet tested: `glMapBufferRange` (instead of
`glGetBufferSubData`) on a persistent-mapped `MAP_READ|MAP_PERSISTENT|MAP_COHERENT`
buffer, and RGBA16F/RGBA32F instead of RGB16F.

**Risk worth checking:** the regular buffer tickets go through the same staging
path. Dab lists are small, and pen-up at L7 was a fine 14 ms. But the
whole-buffer pen-up reads at L9/L10 (normals are ~126 MB at L10) could hit
15 MB/s on Windows GL. Not measured.

### Directions (for the Linux side to choose)

- **Don't read full-screen planes.** Read a small region around the cursor on
  demand, or downsample: one plane at half resolution is ¼ the bytes.
- **Shrink the normal plane:** octahedral RG16F or RGBA8 instead of RGB16F
  removes the worst plane and the 3-channel conversion.
- **Make WebGPU the Windows default.** It already handles this path async and
  showed none of the hitches. CI's Windows job only builds GL today; the WebGPU
  zip also needs `wgpu_native.dll` (the branch now copies it next to the exe).

## 4. Smaller notes

- The `[stage]` stroke report says `over 0 dabs (0.0 ms/dab)` on both backends,
  even though strokes clearly landed. The dab counter looks unwired on this path.
- `[cascade] L8 slow 534 ms` in the old v0.2.20 Windows log was the CPU fallback
  caused by item 2. Re-measure level switches post-fix if they matter; L8 was
  679 ms on WebGPU in this session.

## 5. Tools on the branch

`5d9e783` adds `CHISEL_PERF=1`, a passive per-frame recorder. It's
cross-platform C++, so it works on Linux too:

```bash
CHISEL_PERF=1 CHISEL_DIRTY_HIST=1 CHISEL_PERF_CSV=frames.csv ./build-gl/chisel
# CHISEL_PERF_VSYNC=0 to measure headroom above the refresh cap
```

At exit it prints a p50/p95/p99/max table per `L<n> idle|sculpt|penup|switch`
phase. `CHISEL_DIRTY_HIST=1` turns on the app's existing `[stage]`/`[penup]`
timers. `tools/perf.ps1` is the Windows runner and comparer, and has no Linux
equivalent yet. The two most useful next numbers: the same manual session on
Linux GL, to confirm the readback is cheap under Mesa, and a `-NoVsync` run on
each Windows backend.
