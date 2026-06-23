// draw_apply.wgsl
// Port of src/compute_draw.cpp (draw_apply_src). Pass 2 of the draw/inflate brush:
// each thread reads its vertex's accumulated displacement, scales by (1 - mask),
// adds it to the position, and appends the vertex to the compact dirty list.
// REFERENCE TRANSLATION — see CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0  positions (read_write)   3 accum (read)   6 dirty list (read_write)
//   12 mask (read)              63 params UBO (BIND_PARAMS)
//
// accum is read here as plain u32 (the atomic writes from draw_accum already
// landed in the prior dispatch); bitcast back to float-bits to recover the sum.

struct Params {
    vertex_count : u32,   // byte 0
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,   // struct rounds to 16
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read_write> positions : array<f32>;
@group(0) @binding(3)  var<storage, read>       accum     : array<u32>;
@group(0) @binding(6)  var<storage, read_write> dirty     : Dirty;
@group(0) @binding(12) var<storage, read>       mask      : array<f32>;
@group(0) @binding(63) var<uniform>             P         : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let base = v * 4u;
    let w = bitcast<f32>(accum[base + 3u]);
    if (w <= 0.0) {
        return;
    }

    let scale = 1.0 - mask[v];
    if (scale <= 0.0) {
        return;
    }

    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;

    let inv_w = 1.0 / w;
    let dx = bitcast<f32>(accum[base + 0u]) * inv_w * scale;
    let dy = bitcast<f32>(accum[base + 1u]) * inv_w * scale;
    let dz = bitcast<f32>(accum[base + 2u]) * inv_w * scale;

    let pidx = v * 3u;
    positions[pidx + 0u] = positions[pidx + 0u] + dx;
    positions[pidx + 1u] = positions[pidx + 1u] + dy;
    positions[pidx + 2u] = positions[pidx + 2u] + dz;
}
