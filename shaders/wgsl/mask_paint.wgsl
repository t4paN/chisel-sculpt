// mask_paint.wgsl
// Port of src/compute_mask.cpp (mask brush). Per-vertex world-distance check,
// dual anchor (mirror), writes the mask buffer directly and appends touched
// vertex ids to a compact dirty list. REFERENCE TRANSLATION — see CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0  positions (read)   12 mask (read_write)   6 dirty list (read_write)
//   63 params UBO         (reserved per conventions: BIND_PARAMS)

// Per-dispatch parameters. Mirrors MaskPaintParams in compute.h, but laid out
// for std140/uniform rules: vec3 occupies 16 bytes, so a trailing f32 packs into
// the same 16-byte slot. The C++ upload struct MUST match this layout byte-for-byte.
struct Params {
    anchor_a       : vec3<f32>,   // bytes  0..11
    world_radius   : f32,         // byte  12   (packs into anchor_a's slot)
    anchor_b       : vec3<f32>,   // bytes 16..27
    hardness       : f32,         // byte  28
    paint_strength : f32,         // byte  32
    use_b          : u32,         // byte  36
    vertex_count   : u32,         // byte  40
    _pad0          : u32,         // byte  44   (struct rounds up to 48)
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(12) var<storage, read_write> mask_buf  : array<f32>;
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

fn try_paint(v : u32, anchor : vec3<f32>, vp : vec3<f32>) {
    if (P.use_b != 0u && anchor.x * vp.x < 0.0) {
        return;
    }
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    let w = brush_falloff(dist, P.world_radius);
    if (w <= 0.0) {
        return;
    }

    let delta = P.paint_strength * w;
    let old_val = mask_buf[v];
    let new_val = clamp(old_val + delta, 0.0, 1.0);
    if (new_val == old_val) {
        return;
    }

    mask_buf[v] = new_val;
    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let vp = vec3<f32>(positions[v * 3u], positions[v * 3u + 1u], positions[v * 3u + 2u]);

    try_paint(v, P.anchor_a, vp);
    if (P.use_b != 0u) {
        try_paint(v, P.anchor_b, vp);
    }
}
