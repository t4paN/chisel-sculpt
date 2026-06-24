// move_weight_smooth.wgsl
// Port of src/compute_move.cpp (move_weight_smooth_src). Pass 2 of the grab/move
// brush: ping-pong Laplacian over the captured affected set, smoothing the
// (primary, mirror) falloff weights so the grab pulls a soft, blended island instead
// of a hard-edged disc. Self-weighted x4 (matches the CPU path). See CONVENTIONS.md.
//
// No Params UBO: the dispatch is sized to the affected set and gated on affected.count.
// Ping-pong is driven host-side by swapping which buffer is bound at slots 9/10.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   2 indices (read)   4 adj_offset (read)   5 adj_list (read)
//   8 affected (read)  9 weights in (read)    10 weights out (read_write)

struct Affected {
    count : u32,
    ids   : array<u32>,
};

@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(8)  var<storage, read>       affected   : Affected;
@group(0) @binding(9)  var<storage, read>       w_in       : array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> w_out      : array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= affected.count) {
        return;
    }
    let v = affected.ids[idx];

    var sum = w_in[v] * 4.0;
    var count = 4u;

    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];
        if (i0 != v) { sum = sum + w_in[i0]; count = count + 1u; }
        if (i1 != v) { sum = sum + w_in[i1]; count = count + 1u; }
        if (i2 != v) { sum = sum + w_in[i2]; count = count + 1u; }
    }

    w_out[v] = sum / f32(count);
}
