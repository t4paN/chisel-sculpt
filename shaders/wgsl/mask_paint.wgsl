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

// --- shared brush-alpha stamp (keep byte-identical across every dab kernel) ---
struct AlphaParams {
    tangent      : vec3<f32>,
    inv_diameter : f32,
    bitangent    : vec3<f32>,
    enabled      : u32,
    tex_w        : u32,
    tex_h        : u32,
    _apad0       : u32,
    _apad1       : u32,
};
@group(0) @binding(40) var<storage, read> alpha_tex : array<f32>;
@group(0) @binding(62) var<uniform>       AP        : AlphaParams;
fn alpha_bilinear(u : f32, v : f32) -> f32 {
    let W = i32(AP.tex_w);
    let H = i32(AP.tex_h);
    let fx = u * f32(W) - 0.5;
    let fy = v * f32(H) - 0.5;
    let x0i = i32(floor(fx));
    let y0i = i32(floor(fy));
    let tx = fx - f32(x0i);
    let ty = fy - f32(y0i);
    let x0 = clamp(x0i, 0, W - 1);
    let y0 = clamp(y0i, 0, H - 1);
    let x1 = clamp(x0i + 1, 0, W - 1);
    let y1 = clamp(y0i + 1, 0, H - 1);
    let c00 = alpha_tex[y0 * W + x0];
    let c10 = alpha_tex[y0 * W + x1];
    let c01 = alpha_tex[y1 * W + x0];
    let c11 = alpha_tex[y1 * W + x1];
    return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}
fn sample_alpha(rel : vec3<f32>, mirrored : u32) -> f32 {
    if (AP.enabled == 0u) {
        return 1.0;
    }
    var tang = AP.tangent;
    var bitan = AP.bitangent;
    if (mirrored != 0u) {
        tang.x = -tang.x;
        bitan.x = -bitan.x;
    }
    let u = dot(rel, tang) * AP.inv_diameter + 0.5;
    let v = dot(rel, bitan) * AP.inv_diameter + 0.5;
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return 0.0;
    }
    return alpha_bilinear(u, v);
}
// --- end shared brush-alpha stamp ---

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

fn try_paint(v : u32, anchor : vec3<f32>, vp : vec3<f32>, mirrored : u32) {
    if (P.use_b != 0u && anchor.x * vp.x < 0.0) {
        return;
    }
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    var w = brush_falloff(dist, P.world_radius);
    w = w * sample_alpha(vp - anchor, mirrored);
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

    try_paint(v, P.anchor_a, vp, 0u);
    if (P.use_b != 0u) {
        try_paint(v, P.anchor_b, vp, 1u);
    }
}
