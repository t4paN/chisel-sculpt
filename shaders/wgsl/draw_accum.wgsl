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
    clay             : u32,         // 100
    _pad1            : u32,         // 104
    _pad2            : u32,         // 108     (struct rounds to 112)
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals   : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<atomic<u32>>;
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

// Ramp width for the facing gate, in cosine terms. See deposit().
const FACING_BAND : f32 = 0.25;

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
           vp : vec3<f32>, vn : vec3<f32>, mirrored : u32) {
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    // Draw only deposits on verts facing the viewer (one-sided bump). Inflate is
    // a volumetric swell: every vert in the dab pushes along its OWN normal
    // regardless of facing, so the surface explodes outward in all directions.
    //
    // The gate is a RAMP, not a wall. facing == 0 is exactly the silhouette, so a
    // hard cutoff puts a full-strength/zero cliff along the horizon of the form. Head
    // on that sits harmlessly out at the model's edge, but tilt the camera against a
    // deep ridge and the horizon runs down the middle of the stroke: the flank facing
    // you takes the whole dab, the flank whose normals have swung toward the horizon
    // takes nothing, and the far wall is left behind as a one-vertex-wide sharp pit.
    // Verts below the threshold still deposit nothing (no back-face sculpting); the
    // band only fades the ones just above it in. FACING_BAND is in cosine terms —
    // 0.25 is roughly 15 degrees of ramp either side of the silhouette.
    var facing_w = 1.0;
    if (P.inflate == 0u) {
        let facing = -dot(vn, view);
        facing_w = smoothstep(P.facing_threshold, P.facing_threshold + FACING_BAND, facing);
        if (facing_w <= 0.0) {
            return;
        }
    }
    // Clay's stamp IS its edge: skip the radial falloff so the layer face is flat
    // and the square's corners don't fade out toward the dab rim (the radial fade
    // is what rounded them into a squircle). Every other brush keeps falloff*alpha.
    var w = 1.0;
    if (P.clay == 0u || AP.enabled == 0u) {
        w = brush_falloff(dist, P.world_radius);
    }
    w = w * sample_alpha(vp - anchor, mirrored);
    w = w * facing_w;
    if (w <= 0.0) {
        return;
    }
    // Draw pushes the whole dab along the cursor's surface normal; inflate pushes
    // each vert along its own normal so the surface swells outward locally.
    var dir = anchor_n;
    if (P.inflate != 0u) {
        dir = vn;
    }

    // Clay displaces TO a plane instead of BY a fixed amount. The plane sits
    // disp_amount above the anchor along anchor_n; a vert's signed height above it
    // is h, so (target - h) is the gap left to fill. Clamping that to the stroke's
    // direction is what makes clay build in layers: verts below the plane rise to
    // meet it (hollows fill), verts already proud of it don't move (detail survives
    // instead of being amplified), and repeat strokes settle at the plane rather
    // than growing without bound.
    if (P.clay != 0u) {
        // NB: not `target` — that's a WGSL reserved keyword and Tint rejects the
        // whole module for it (naga/glslang don't, so GL builds stay silent).
        let target_h = P.disp_amount;
        let h = dot(vp - anchor, dir);
        var delta = (target_h - h) * w;
        if (target_h >= 0.0) {
            delta = max(delta, 0.0);
        } else {
            delta = min(delta, 0.0);
        }
        let dc = dir * delta;
        let basec = v * 4u;
        atomicAddFloat(basec + 0u, dc.x);
        atomicAddFloat(basec + 1u, dc.y);
        atomicAddFloat(basec + 2u, dc.z);
        atomicMax(&accum[basec + 3u], bitcast<u32>(1.0));
        return;
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
    deposit(v, P.anchor_a, P.view_a, P.anchor_normal_a, vp, vn, 0u);
    if (P.use_b != 0u) {
        deposit(v, P.anchor_b, P.view_b, P.anchor_normal_b, vp, vn, 1u);
    }
}
