// smooth_mirror_apply.wgsl
// Port of src/compute_smooth.cpp (smooth_mirror_apply_src). Re-imposes the X-mirror
// reflection after each smooth_apply iteration: each touched anchor-side vertex writes
// its negated-X position onto its mirror twin so the two lobes relax in lockstep and
// the seam stays a byte-exact reflection. Gated on accum weight > 0 AND the vert
// sharing the anchor's x sign. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   3 accum (read)   7 mirror_map (read)
//   12 mask (read)             63 params UBO (BIND_PARAMS)

struct Params {
    vertex_count : u32,   // 0
    anchor_x     : f32,   // 4
    _pad0        : u32,   // 8
    _pad1        : u32,   // 12
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(3)  var<storage, read>       accum      : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
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

    let vx = positions[v * 3u];
    if (vx * P.anchor_x < 0.0) {
        return;
    }

    let mv = mirror_map[v];
    if (mv == v) {
        return;
    }

    let mscale = 1.0 - mask[mv];
    if (mscale <= 0.0) {
        return;
    }

    let src = v * 3u;
    let dst = mv * 3u;
    positions[dst + 0u] = positions[dst + 0u] + (-positions[src + 0u] - positions[dst + 0u]) * mscale;
    positions[dst + 1u] = positions[dst + 1u] + ( positions[src + 1u] - positions[dst + 1u]) * mscale;
    positions[dst + 2u] = positions[dst + 2u] + ( positions[src + 2u] - positions[dst + 2u]) * mscale;
}
