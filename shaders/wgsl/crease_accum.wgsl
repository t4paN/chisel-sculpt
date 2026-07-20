// crease_accum.wgsl
// Port of src/compute_crease_pinch.cpp (crease_accum_src). Crease brush accumulate
// pass: like draw, but each deposit adds a tangential pull toward the anchor
// (pinch_amount) on top of the normal displacement, folding a ridge/valley.
// Deposits into the shared accum buffer; the apply side reuses draw_apply /
// symmetrize / mirror_apply. Lockstep with crease_accum.comp. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum:
//   0 positions (read)   1 stroke normals (read)   3 accum (read_write)
//   63 params UBO (BIND_PARAMS)

// std140/uniform layout — every vec3 occupies a 16-byte slot, scalars pack into the
// trailing space. The C++ upload struct MUST match this byte-for-byte (112 B).
struct Params {
    anchor_a         : vec3<f32>,   //   0..11
    world_radius     : f32,         //  12
    anchor_b         : vec3<f32>,   //  16..27
    disp_amount      : f32,         //  28
    view_a           : vec3<f32>,   //  32..43
    pinch_amount     : f32,         //  44
    view_b           : vec3<f32>,   //  48..59
    hardness         : f32,         //  60
    anchor_normal_a  : vec3<f32>,   //  64..75
    facing_threshold : f32,         //  76
    anchor_normal_b  : vec3<f32>,   //  80..91
    use_b            : u32,         //  92
    vertex_count     : u32,         //  96
    stroke_dir_x     : f32,         // 100  \ loose scalars, not a vec3: a vec3 would
    stroke_dir_y     : f32,         // 104   > need 16-byte alignment and push this
    stroke_dir_z     : f32,         // 108  / block to 128. Zero length = no axis yet.
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
           vp : vec3<f32>, vn : vec3<f32>, mirrored : u32) {
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }

    let facing = -dot(vn, view);
    if (facing < P.facing_threshold) {
        return;
    }

    var w = brush_falloff(dist, P.world_radius);
    w = w * sample_alpha(vp - anchor, mirrored);
    if (w <= 0.0) {
        return;
    }

    var d = anchor_n * (P.disp_amount * w);

    if (dist > 1e-6) {
        let to_anchor = anchor - vp;
        var tangent = to_anchor - dot(to_anchor, anchor_n) * anchor_n;

        // Squeeze across the stroke, not toward the dab centre. Pulling at the anchor
        // point means each successive dab yanks the same vertices further down the
        // path, which drags topology along the stroke. Removing the stroke-parallel
        // component leaves a pure pinch onto the ridge line.
        var sdir = vec3<f32>(P.stroke_dir_x, P.stroke_dir_y, P.stroke_dir_z);
        if (mirrored != 0u) {
            sdir.x = -sdir.x;
        }
        if (dot(sdir, sdir) > 0.25) {
            tangent = tangent - dot(tangent, sdir) * sdir;
        }

        d = d + tangent * (w * P.pinch_amount / P.world_radius);
    }

    let base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
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
