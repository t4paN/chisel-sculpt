// move_apply.wgsl
// Port of src/compute_move.cpp (move_apply_src). Pass 3 of the grab/move brush, run
// once per dab: each affected vertex is placed at its captured init position plus the
// accumulated stroke total, weighted by its (primary, mirror) falloff and damped by
// (1 - mask). The mirror lobe is the X-negated total summed in-place, so a symmetric
// grab is a single pass. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   8 affected (read)   9 weights (read)
//   11 init pos (read)         12 mask (read)       63 params UBO (BIND_PARAMS)

struct Params {
    total : vec3<f32>,   // struct rounds to 16
    _pad0 : f32,
};

struct Affected {
    count : u32,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read_write> positions    : array<f32>;
@group(0) @binding(8)  var<storage, read>       affected     : Affected;
@group(0) @binding(9)  var<storage, read>       move_weights : array<vec2<f32>>;
@group(0) @binding(11) var<storage, read>       move_init    : array<f32>;
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
    let init = vec3<f32>(move_init[v*3u], move_init[v*3u+1u], move_init[v*3u+2u]);
    let mirror_total = vec3<f32>(-P.total.x, P.total.y, P.total.z);
    let mscale = 1.0 - mask[v];
    let dst = init + (P.total * w.x + mirror_total * w.y) * mscale;

    positions[v*3u]      = dst.x;
    positions[v*3u + 1u] = dst.y;
    positions[v*3u + 2u] = dst.z;
}
