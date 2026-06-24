// color_paint.wgsl
// Port of src/compute_color.cpp (color_paint_src). Paint (vpaint) brush: per-dab
// world-distance check, dual anchor (mirror), lerps a packed RGBA8 albedo toward the
// brush colour and writes the colour buffer directly, appending touched verts to the
// compact dirty list. One owning invocation per vertex → no atomics on the colour
// buffer (the dirty counter is the only atomic). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   6 dirty (read_write)   12 mask (read)
//   28 color (read_write)                        63 params UBO (BIND_PARAMS)

struct Params {
    anchor_a       : vec3<f32>,   // world_radius packs into .w
    world_radius   : f32,
    anchor_b       : vec3<f32>,   // hardness packs into .w
    hardness       : f32,
    paint_color    : vec3<f32>,   // paint_strength packs into .w
    paint_strength : f32,
    use_b          : u32,
    vertex_count   : u32,
    _pad0          : u32,
    _pad1          : u32,
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(6)  var<storage, read_write> dirty     : Dirty;
@group(0) @binding(12) var<storage, read>       mask_buf  : array<f32>;
@group(0) @binding(28) var<storage, read_write> color_buf : array<u32>;
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

fn try_paint(v : u32, anchor : vec3<f32>, vp : vec3<f32>) {
    if (P.use_b != 0u && anchor.x * vp.x < 0.0) {
        return;
    }
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    var w = brush_falloff(dist, P.world_radius);
    if (w <= 0.0) {
        return;
    }

    w = w * (1.0 - clamp(mask_buf[v], 0.0, 1.0));   // mask shields paint
    if (w <= 0.0) {
        return;
    }

    let a = clamp(P.paint_strength * w, 0.0, 1.0);
    let old_rgba = unpack4x8unorm(color_buf[v]);
    let new_rgb  = mix(old_rgba.rgb, P.paint_color, a);   // lerp-to-color
    let pk_rgba  = pack4x8unorm(vec4<f32>(new_rgb, 1.0)); // alpha forced opaque
    if (pk_rgba == color_buf[v]) {
        return;
    }

    color_buf[v] = pk_rgba;
    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let vp = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    try_paint(v, P.anchor_a, vp);
    if (P.use_b != 0u) {
        try_paint(v, P.anchor_b, vp);
    }
}
