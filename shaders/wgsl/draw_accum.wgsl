// draw_accum.wgsl
// Port of src/compute_draw.cpp (draw_accum_src). Pass 1 of the draw/inflate brush:
// each thread owns one vertex, tests it against the dab sphere, and deposits a
// displacement vector + unit weight into the accum buffer. draw_apply consumes it.
// REFERENCE TRANSLATION — see CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0  positions (read)   1 stroke normals (read)   3 accum (read_write)
//   63 params UBO (BIND_PARAMS)
//
// accum is 4 u32 per vertex: { disp.x, disp.y, disp.z, weight } as float-bits.
// WGSL has no float atomics, so we accumulate with the portable uint-bits +
// atomicCompareExchangeWeak CAS loop (the GL !has_native_float_atomics path).

// std140/uniform layout — every vec3 occupies a 16-byte slot, scalars pack into
// the trailing space. The C++ upload struct MUST match this byte-for-byte (112 B).
struct Params {
    anchor_a         : vec3<f32>,   //   0..11
    world_radius     : f32,         //  12      (packs into anchor_a's slot)
    anchor_b         : vec3<f32>,   //  16..27
    disp_amount      : f32,         //  28
    view_a           : vec3<f32>,   //  32..43
    hardness         : f32,         //  44
    view_b           : vec3<f32>,   //  48..59
    facing_threshold : f32,         //  60
    anchor_normal_a  : vec3<f32>,   //  64..75
    use_b            : u32,         //  76
    anchor_normal_b  : vec3<f32>,   //  80..91
    inflate          : u32,         //  92
    vertex_count     : u32,         //  96
    _pad0            : u32,         // 100
    _pad1            : u32,         // 104
    _pad2            : u32,         // 108     (struct rounds to 112)
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals   : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<atomic<u32>>;
@group(0) @binding(63) var<uniform>             P         : Params;

// Portable float accumulate: CAS the float-bits until our add lands. Matches the
// atomicAddFloat in compute_draw.cpp (atomicCompSwap loop), capped the same way.
fn atomicAddFloat(idx : u32, val : f32) {
    var expected = atomicLoad(&accum[idx]);
    for (var i = 0; i < 128; i = i + 1) {
        let desired = bitcast<u32>(bitcast<f32>(expected) + val);
        let res = atomicCompareExchangeWeak(&accum[idx], expected, desired);
        if (res.exchanged) {
            return;
        }
        expected = res.old_value;
    }
}

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

fn deposit(v : u32, anchor : vec3<f32>, view : vec3<f32>, anchor_n : vec3<f32>,
           vp : vec3<f32>, vn : vec3<f32>) {
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    // Draw only deposits on verts facing the viewer (one-sided bump). Inflate is
    // a volumetric swell: every vert in the dab pushes along its OWN normal
    // regardless of facing, so the surface explodes outward in all directions.
    if (P.inflate == 0u) {
        let facing = -dot(vn, view);
        if (facing < P.facing_threshold) {
            return;
        }
    }
    let w = brush_falloff(dist, P.world_radius);
    if (w <= 0.0) {
        return;
    }
    // Draw pushes the whole dab along the cursor's surface normal; inflate pushes
    // each vert along its own normal so the surface swells outward locally.
    var dir = anchor_n;
    if (P.inflate != 0u) {
        dir = vn;
    }
    let d = dir * (P.disp_amount * w);
    let base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
    // Idempotent weight: any deposit sets weight to 1.0 (max of identical 1.0s),
    // so two anchors summing X/Y/Z into one vert don't halve the amplitude.
    atomicMax(&accum[base + 3u], bitcast<u32>(1.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }
    let vp = vec3<f32>(positions[v * 3u], positions[v * 3u + 1u], positions[v * 3u + 2u]);
    let vn = vec3<f32>(normals[v * 3u], normals[v * 3u + 1u], normals[v * 3u + 2u]);
    deposit(v, P.anchor_a, P.view_a, P.anchor_normal_a, vp, vn);
    if (P.use_b != 0u) {
        deposit(v, P.anchor_b, P.view_b, P.anchor_normal_b, vp, vn);
    }
}
