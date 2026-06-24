// smooth_accum.wgsl
// Port of src/compute_smooth.cpp (smooth_accum_src). Pass 1 of the smooth brush:
// world-distance gate. Each vertex inside the brush sphere (anchor-side only when
// mirror_x) stamps its falloff weight into accum.w and appends itself to the compact
// dirty list. Buffer-only (the triid/bary pick read is CPU-side back-projection in
// brush.cpp, not this kernel). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   3 accum (read_write)   6 dirty list (read_write)
//   63 params UBO (BIND_PARAMS)

struct Params {
    anchor       : vec3<f32>,   // world_radius packs into .w slot
    world_radius : f32,
    hardness     : f32,
    mirror_x     : u32,
    vertex_count : u32,
    _pad0        : u32,
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<u32>;
@group(0) @binding(6)  var<storage, read_write> dirty     : Dirty;
@group(0) @binding(63) var<uniform>             P         : Params;

fn brush_falloff(dist : f32, radius : f32) -> f32 {
    let t = dist / radius;
    let inner = 0.15 + P.hardness * 0.55;
    if (t <= inner) {
        return 1.0;
    }
    var blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let p = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    if (P.mirror_x != 0u && P.anchor.x * p.x < 0.0) {
        return;
    }

    let d = distance(p, P.anchor);
    if (d >= P.world_radius) {
        return;
    }

    let w = brush_falloff(d, P.world_radius);
    if (w <= 0.0) {
        return;
    }

    accum[v * 4u + 3u] = bitcast<u32>(w);
    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;
}
