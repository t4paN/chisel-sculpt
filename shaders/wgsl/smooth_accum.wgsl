// smooth_accum.wgsl
// Port of src/compute_smooth.cpp (smooth_accum_src). Pass 1 of the smooth brush:
// world-distance gate. Each vertex inside the brush sphere stamps its falloff weight
// into accum.w and appends itself to the compact dirty list.
//
// Mirror comes in two flavours, matching every other dab kernel. GEOMETRIC (use_b):
// a second gate at the reflected anchor, so the far side is smoothed because it lies
// under the mirrored brush — no pairing, works on any tessellation. TOPOLOGICAL
// (mirror_clip): gate only the anchor's own side and let smooth_mirror_apply reflect
// the result onto the twins between iterations. Buffer-only (the triid/bary pick read is CPU-side back-projection in
// brush.cpp, not this kernel). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   3 accum (read_write)   6 dirty list (read_write)
//   63 params UBO (BIND_PARAMS)

struct Params {
    anchor       : vec3<f32>,   //  0..11  (world_radius packs into .w slot)
    world_radius : f32,         // 12
    hardness     : f32,         // 16
    mirror_clip  : u32,         // 20
    vertex_count : u32,         // 24
    _pad0        : u32,         // 28
    anchor_b     : vec3<f32>,   // 32..43
    use_b        : u32,         // 44     (struct rounds to 48)
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<u32>;
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

fn lobe_weight(p : vec3<f32>, anchor : vec3<f32>, mirrored : u32) -> f32 {
    let d = distance(p, anchor);
    if (d >= P.world_radius) {
        return 0.0;
    }
    return brush_falloff(d, P.world_radius) * sample_alpha(p - anchor, mirrored);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let p = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    if (P.mirror_clip != 0u && P.anchor.x * p.x < 0.0) {
        return;
    }

    var w = lobe_weight(p, P.anchor, 0u);
    if (P.use_b != 0u) {
        // max, not sum: this weight is how hard the vert gets smoothed, and a vert
        // in the overlap where the two lobes meet at x=0 must not be smoothed twice
        // as hard as its neighbours — that is a crease down the seam.
        w = max(w, lobe_weight(p, P.anchor_b, 1u));
    }
    if (w <= 0.0) {
        return;
    }

    accum[v * 4u + 3u] = bitcast<u32>(w);
    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;
}
