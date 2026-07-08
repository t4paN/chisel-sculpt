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

    try_paint(v, P.anchor_a, vp, 0u);
    if (P.use_b != 0u) {
        try_paint(v, P.anchor_b, vp, 1u);
    }
}
