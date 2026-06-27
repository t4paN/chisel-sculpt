// smooth_apply.wgsl
// Port of src/compute_smooth.cpp (smooth_apply). Pass 2 of the smooth brush, run once
// per iteration: a uniform Laplacian over each touched vertex's 1-ring. The mirror
// seam crease is handled by re-imposing the X reflection after every iteration in
// dispatch_smooth (not by a normal projection in here — that variant pinched ridges/
// bumps under prolonged smoothing). Tail UBO slots are padding kept for shape parity.
// See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   2 indices (read)        3 accum (read)
//   4 adj_offset (read)        5 adj_list (read)        12 mask (read)
//   63 params UBO (BIND_PARAMS)

struct Params {
    vertex_count : u32,   // 0
    strength     : f32,   // 4
    _pad0        : u32,   // 8
    _pad1        : u32,   // 12
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(3)  var<storage, read>       accum      : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(12) var<storage, read>       mask       : array<f32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let w = bitcast<f32>(accum[v * 4u + 3u]);
    if (w <= 0.0) {
        return;
    }

    let mscale = 1.0 - mask[v];
    if (mscale <= 0.0) {
        return;
    }

    let cur_x = positions[v * 3u];
    let cur_y = positions[v * 3u + 1u];
    let cur_z = positions[v * 3u + 2u];

    var sum_x = 0.0;
    var sum_y = 0.0;
    var sum_z = 0.0;
    var count = 0.0;

    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];

        if (v == i0) {
            sum_x = sum_x + positions[i1*3u]      + positions[i2*3u];
            sum_y = sum_y + positions[i1*3u + 1u] + positions[i2*3u + 1u];
            sum_z = sum_z + positions[i1*3u + 2u] + positions[i2*3u + 2u];
        } else if (v == i1) {
            sum_x = sum_x + positions[i0*3u]      + positions[i2*3u];
            sum_y = sum_y + positions[i0*3u + 1u] + positions[i2*3u + 1u];
            sum_z = sum_z + positions[i0*3u + 2u] + positions[i2*3u + 2u];
        } else {
            sum_x = sum_x + positions[i0*3u]      + positions[i1*3u];
            sum_y = sum_y + positions[i0*3u + 1u] + positions[i1*3u + 1u];
            sum_z = sum_z + positions[i0*3u + 2u] + positions[i1*3u + 2u];
        }
        count = count + 2.0;
    }

    if (count <= 0.0) {
        return;
    }

    let inv_c = 1.0 / count;
    let blend = w * P.strength * mscale;
    positions[v*3u]      = positions[v*3u]      + (sum_x * inv_c - cur_x) * blend;
    positions[v*3u + 1u] = positions[v*3u + 1u] + (sum_y * inv_c - cur_y) * blend;
    positions[v*3u + 2u] = positions[v*3u + 2u] + (sum_z * inv_c - cur_z) * blend;
}
