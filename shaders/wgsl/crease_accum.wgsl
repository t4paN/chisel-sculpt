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

struct DirtyRegion {
    base       : u32,
    cap        : u32,
    block_mode : u32,   // 1 = dispatch is one workgroup per active block
    _p1        : u32,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals   : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<atomic<u32>>;
@group(0) @binding(45) var<storage, read>       block_list : array<u32>;
@group(0) @binding(61) var<uniform>             DR         : DirtyRegion;
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

fn deposit(v : u32, anchor : vec3<f32>, view : vec3<f32>, anchor_n : vec3<f32>,
           vp : vec3<f32>, vn : vec3<f32>, mirrored : u32) {
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }

    // Position-based gate, ported from draw_accum (which got it 2026-07-21; crease
    // and pinch were left on the old test as the control, and this is that control
    // coming back positive). The binary `facing < facing_threshold` cutoff tore
    // grazing strokes apart two ways: vertex normals FLIP across a crease line
    // (+0.4 → −0.6 between neighbours), so an angular test accepts one flank and
    // rejects the other mid-dab; and with the default threshold at 0.0 — exactly
    // the silhouette — the accepted SET changes discontinuously from dab to dab as
    // the surface turns away, which is what makes an oblique stroke wander.
    // Gate on position instead: reject verts significantly BEHIND the anchor
    // surface along the view (that's a thin sheet's far side), feathered in
    // position space so bulgy forms don't hit a wall. A crease's far flank sits at
    // ≈ the anchor's depth, so it deposits. Far-backfacing verts are still cut as
    // belt-and-suspenders for sheets thinner than the window.
    // P.facing_threshold is unused now — kept in the block so the 112 B std140
    // layout stays byte-identical to the C++ upload struct.
    // Band widens with tilt — see draw_accum.wgsl. Fixed thresholds amputate the
    // receding half of a grazing footprint, dragging the deposit centroid toward the
    // camera by an amount that varies along a curved stroke.
    let ct = max(abs(dot(view, anchor_n)), 0.30);
    let behind = dot(vp - anchor, view);
    var gate_w = 1.0 - smoothstep(0.25 * P.world_radius / ct,
                                  0.45 * P.world_radius / ct, behind);
    gate_w = gate_w * smoothstep(-0.6, -0.4, -dot(vn, view));
    if (gate_w <= 0.0) {
        return;
    }

    // The profile is a function of the distance ACROSS the stroke, not to the dab
    // centre. A radial dab, however sharp, blurs into a rounded groove once a stroke
    // of them overlaps; shaping the cross-section directly keeps the tip a V. Along
    // the stroke each dab is just a smooth bump, so neighbours blend. Zero-length
    // stroke_dir = the stroke's first dab, no axis yet: radial for that one dab.
    var sdir = vec3<f32>(P.stroke_dir_x, P.stroke_dir_y, P.stroke_dir_z);
    if (mirrored != 0u) {
        sdir.x = -sdir.x;
    }
    let rel = vp - anchor;
    let rel_t = rel - dot(rel, anchor_n) * anchor_n;
    var across = rel_t;
    var along = 0.0;
    if (dot(sdir, sdir) > 0.25) {
        along = dot(rel_t, sdir);
        across = rel_t - along * sdir;
    }
    let xa = length(across) / P.world_radius;
    if (xa >= 1.0) {
        return;
    }
    let ya = min(abs(along) / P.world_radius, 1.0);
    let mask = sample_alpha(rel, mirrored) * gate_w * (1.0 - smoothstep(0.0, 1.0, ya));
    if (mask <= 0.0) {
        return;
    }

    // Depth: a wide V with a sharp point on the stroke line. The body is (1 - x)^k with
    // k near 1, so the flanks run nearly straight; hardness raises k toward 2 for a
    // narrower V (it used to widen a flat-topped plateau, which is what made the tip
    // flat and the falloff wide). A fifth of the depth is a very narrow (1 - x)^10 cut
    // that sharpens the point itself without narrowing the flanks.
    let k = 1.0 + P.hardness;
    let shape = 0.8 * pow(1.0 - xa, k) + 0.2 * pow(1.0 - xa, 10.0);
    var d = anchor_n * (P.disp_amount * shape * mask);

    // Pinch: squeeze onto the stroke line, never along it (pulling at the dab centre
    // drags topology down the path). Weight (1 - x) makes the pull x(1 - x): zero on
    // the line and at the rim, strongest half-way, so it gathers the flanks into a
    // tight band and puts vertices in the tip. Order-preserving as long as the per-dab
    // fraction pinch_amount / R stays below 1 (it is at most 0.35): neighbours
    // contract toward the line and can't cross.
    d = d - across * ((1.0 - xa) * mask * P.pinch_amount / P.world_radius);

    let base = v * 4u;
    atomicAddFloat(base + 0u, d.x);
    atomicAddFloat(base + 1u, d.y);
    atomicAddFloat(base + 2u, d.z);
    atomicMax(&accum[base + 3u], bitcast<u32>(1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    // Dispatch culling: one workgroup per ACTIVE block; entry 0 is the count, hence +1.
    let v = select(wg.x * 64u + lid.x,
                   block_list[1u + wg.x] * 64u + lid.x,
                   DR.block_mode == 1u);
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
