# GLSL 430 → WGSL translation conventions (Chisel WebGPU port)

Every compute kernel currently lives as an inline GLSL `#version 430` string in `src/compute*.cpp`
(and the raster/SDF shaders in `renderer.cpp` / `sdf.cpp`). The port moves each into a `.wgsl` file
here, one-to-one. `mask_paint.wgsl` is the worked reference; follow it. These are the rules that
make the translation mechanical.

## Bindings
- Keep the **`ComputeBinding` enum values** (`include/compute.h`) as `@group(0) @binding(N)`. A GL
  `layout(std430, binding = 12) buffer MaskBuf` becomes `@group(0) @binding(12) var<storage, ...>`.
- **Reserved binding 63 = `BIND_PARAMS`**, the per-dispatch uniform buffer (the old loose uniforms).
  Nothing in the enum uses 63; keep it the params slot everywhere.
- Everything sits in **`@group(0)`** for now. WebGPU guarantees only 4 bind groups but no hard cap
  on bindings per group within device limits; if a kernel exceeds `maxStorageBuffersPerShaderStage`
  (commonly 8–10 on the browser), split by lifetime into group(1)/group(2). The high-water mark is
  **`remesh_smooth` at 9 storage buffers** (right at the default cap) — watch it on a low-cap browser.
  **SDF needs no split** (checked when porting it): its 9 distinct `BIND_SDF_*` slots are spread
  across 5 kernels and never co-bound — no single SDF kernel binds more than 5 storage buffers.

## Storage buffers
- `readonly buffer X { float a[]; }` → `@group(0) @binding(N) var<storage, read> a : array<f32>;`
- read/write → `var<storage, read_write>`.
- A buffer with a header + tail array (`buffer Dirty { uint count; uint ids[]; }`) → a WGSL struct
  with the runtime array **last**: `struct Dirty { count : atomic<u32>, ids : array<u32> };`
- SOA float arrays stay `array<f32>` indexed `v*3u + k` exactly as in GLSL.

## Uniforms (the big change)
- GL loose uniforms (`uniform vec3 u_anchor_a; ...`) → **one `Params` UBO struct** at binding 63,
  uploaded per dispatch. The matching C++ `*Params` struct in `compute.h` becomes the upload payload.
- **std140/uniform layout is strict:** `vec3` aligns to 16 bytes and occupies 16. Pack a scalar
  after a `vec3` to fill its slot (see `mask_paint.wgsl`: `world_radius` rides in `anchor_a`'s slot).
  Lay every Params struct out explicitly and keep the C++ struct byte-identical — mismatches here are
  silent garbage, not compile errors. When in doubt, pad to 16-byte boundaries.
- For storage buffers `std430` is laxer (vec3 still aligns to 16); the accum buffers use `array<u32>`
  4-per-vertex which is already 16-byte friendly.

## Atomics
- `atomicAdd(counter, 1u)` → `atomicAdd(&dirty.count, 1u)` (returns the old value, same semantics).
- **No float atomics in WGSL.** GL's `GL_NV_shader_atomic_float` fast path is dropped; keep the
  portable **uint-bits + `atomicCompareExchangeWeak`** accumulation already written for the
  `!has_native_float_atomics` path. Translate that CAS loop, not the native-atomic branch.

## Builtins & types
- `gl_GlobalInvocationID.x` → `@builtin(global_invocation_id) gid : vec3<u32>` param, use `gid.x`.
- `layout(local_size_x = 256) in;` → `@compute @workgroup_size(256)`.
- `uint`→`u32`, `int`→`i32`, `float`→`f32`, `vec3`→`vec3<f32>`, `uvec3`→`vec3<u32>`.
- Integer literals need the `u`/`i` suffix where typed (`1u`, `255u`). Float compares stay `0.0`.
- `mix`, `clamp`, `length`, `normalize`, `dot`, `cross` all exist with the same names.
- WGSL has no implicit int↔uint↔float conversion — cast explicitly (`f32(x)`, `u32(x)`, `i32(x)`).
- No `return value;` shorthand differences; `if` needs braces.

## Memory barriers
- `glMemoryBarrier(...)` between dispatches → **delete**. WebGPU makes storage writes from one pass
  visible to the next pass in the same submit automatically. Only `workgroupBarrier()` matters, and
  only inside a kernel that uses `var<workgroup>` shared memory (none of Chisel's do yet).

## Readback
- `glGetBufferSubData` / `glReadBuffer` → async: copy into a `MAP_READ` staging buffer, submit, then
  `mapAsync` + read in the callback. Only legal at pen-down/pen-up/one-shot (never mid-stroke) — the
  existing architecture already restricts readback to exactly those points, so each site maps 1:1.

## Render shaders
- GLSL `in`/`out` varyings → WGSL `struct VertexOutput { @builtin(position) pos; @location(0) ... }`.
- `gl_Position` → the `@builtin(position)` field. Vertex attributes → `@location(N)` inputs matched
  to the render pipeline's vertex buffer layout (VAOs are gone).
- The 4-attachment picking FBO → a render pass with 4 color targets; fragment returns a struct with
  `@location(0..3)`. Texture formats must be render-attachment + copyable for the picking readback.

## File naming
One `.wgsl` per current GLSL block, named after the program: `mask_paint`, `draw_accum`,
`draw_apply`, `smooth_accum`, `crease_accum`, `pinch_accum`, `move_capture`, `compute_normals`,
`matcap`, `picking`, `sdf_mc`, … Shaders are embedded at build time (CMake-generated header on
native; preloaded on web).
