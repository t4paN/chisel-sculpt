// limb_drag.wgsl
// Port of src/compute_limb.cpp (limb_drag_src). Pass 1 of the limb (snakehook)
// brush, once per dab: INCREMENTAL grab — positions[v] += this-dab world delta *
// falloff weight (mirror lobe summed, X-negated), damped by (1 - mask). Incremental
// (vs move's absolute init+total) so the per-dab tangential relax accumulates.
// Reuses the move capture's affected/weights buffers. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   8 affected (read)   9 weights (read)
//   12 mask (read)             63 params UBO (BIND_PARAMS)

struct Params {
    delta : vec3<f32>,   // struct rounds to 16
    _pad0 : f32,
};

struct Affected {
    count : u32,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read_write> positions    : array<f32>;
@group(0) @binding(8)  var<storage, read>       affected     : Affected;
@group(0) @binding(9)  var<storage, read>       move_weights : array<vec2<f32>>;
@group(0) @binding(12) var<storage, read>       mask         : array<f32>;
@group(0) @binding(63) var<uniform>             P            : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= affected.count) {
        return;
    }
    let v = affected.ids[idx];

    let w = move_weights[v];
    // Mirror lobe (w.y) is X-negated; w.y is 0 when mirror is off, so the term vanishes.
    let mdelta = vec3<f32>(-P.delta.x, P.delta.y, P.delta.z);
    let mscale = 1.0 - mask[v];
    let d = (P.delta * w.x + mdelta * w.y) * mscale;

    positions[v*3u + 0u] = positions[v*3u + 0u] + d.x;
    positions[v*3u + 1u] = positions[v*3u + 1u] + d.y;
    positions[v*3u + 2u] = positions[v*3u + 2u] + d.z;
}
