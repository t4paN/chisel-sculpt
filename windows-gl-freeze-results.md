# Windows GL ~240 ms freezes — cause and fix (2026-10-03)

For the Linux side. Follows `windows-small-brush-results.md` (also on SHARED).

## Symptom

On the v0.2.25 GL release (Intel Arc B570, Windows driver 32.0.101.9033), frames of
~240 ms came about once a second during normal use: 26 in a 34 s session. They followed
every orbit/pan/zoom end, view snap and pen-up. WebGPU had none.

## Cause

`Renderer::render_screen_buffers` refreshes the three pick planes (depth R32F, normal
RGB16F read as RGB float, triid R32UI) whenever the view or mesh changes and the app is
idle. On GL, `gpu::read_target_region_async` was really synchronous: a full-screen
`glReadPixels` into client memory per plane, about 41.5 MB at 1920×1080. On this driver
that path runs at 177 MB/s, so 234 ms with the CPU blocked.

`tools/glbench --readback` (new) times the alternatives on the same three formats, all
returning identical data:

| path | CPU blocked at kick | GPU wait | copy-out | total |
|---|---|---|---|---|
| sync `glReadPixels` (Chisel before) | **234 ms** | — | — | 234 ms |
| PBO `STREAM_READ` → fence → map + memcpy | 0.14 ms | 5 ms | 12.9 ms | **18 ms** |
| PBO `STREAM_READ` → fence → `glGetBufferSubData` | 0.17 ms | 5 ms | **2818 ms** | 2824 ms |
| PBO `CLIENT_STORAGE` persistent map → memcpy | 0.17 ms | 5 ms | **2818 ms** | 2824 ms |

Reading the normal plane as RGBA half instead of RGB float saves little (S-h 193 ms,
P-h 18 ms). The format conversion isn't the cost; the sync path is.

The earlier note that "PBO attempts made it worse (~15 MB/s copy-out)" was the
copy-out method, not PBOs. **`glGetBufferSubData` and `CLIENT_STORAGE` mappings read at
~15 MB/s on this driver; mapping a `STREAM_READ` buffer reads at ~3 GB/s.**

The same matters for every buffer ticket: `ticket_take` used `glGetBufferSubData`, so
the per-dab dirty-list reads for mask/colour/density strokes and the pen-up reads all
copied at 15 MB/s.

## Fix (`gl_backend.cpp`, on `chisel-windows`)

- `read_target_region_async` is now asynchronous on GL too: `glReadPixels` into a pooled
  `STREAM_READ` staging buffer bound as `GL_PIXEL_PACK_BUFFER` (pack alignment 1), then
  a fence and a flush. The plane cache lands via `poll_plane_reads` a frame or two later,
  exactly as on WebGPU. The app already tolerates that gap: sample_* return false and
  callers keep their last value or skip the dab.
- `ticket_take` copies out with `glMapBufferRange(READ)` + memcpy instead of
  `glGetBufferSubData`. Texture tickets flip rows to top-down during that copy (one pass,
  no separate swap).
- Comments in `renderer.h`, `renderer.cpp` and `brush.cpp` that promised synchronous GL
  planes were updated.

## Verified (hands-off)

Each build launched with `CHISEL_PERF`; F1/F2/F3 view snaps sent 6 times, 1.2 s apart:

| build | frames > 20 ms | worst frame | p99 |
|---|---|---|---|
| v0.2.25 release | 6 (238–242 ms, one per snap) | 241.8 ms | 15.2 ms |
| fixed | **0** | **17.1 ms** | 13.9 ms |

Not yet verified by hand: sculpting feel, cursor latch/normal right after an orbit, and
mask/paint strokes (now on the faster copy-out). The user will test.

## Still open / noticed

- **`chisel-gl-compute-test` fails before and after this change** (GPU touched 37 vs CPU
  135 for the mask kernel), alongside `[gpu] warning: kernel declares SSBO binding N but its
  layout doesn't list it` for 38/39/40/45. It predates today's work (same numbers on
  `5a0e331`). Probably related to the SSBO slot packing in bd5af6c. The live app's mask
  works, so it may be the test's layout list, but it needs a look.
- The ~13 ms copy-out lands in one frame. If it ever shows, land the three planes on
  separate frames, or sample straight from the mapped buffer.
- The CPU mask fallback (`brush.cpp`, only without a GPU mask program) still does a
  synchronous full-screen triid+bary read at pen-down, ~140 ms by the same math.
