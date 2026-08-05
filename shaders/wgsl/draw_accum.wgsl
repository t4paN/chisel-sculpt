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
    clay_sign        : i32,         // 104     (+1 build / -1 carve; clay's fill-clamp direction)
    clay_melt        : f32,         // 108     (0 = phase-1 preserve; >0 melts proud verts)
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals   : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<atomic<u32>>;
// Dispatch culling: one workgroup per ACTIVE 256-vertex block, so this maps
// workgroup -> block. See VertexBlocks in mesh.h.
@group(0) @binding(45) var<storage, read>       block_list : array<u32>;
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

// Clay-only: analytic square coverage in the stamp frame, replacing the bitmap
// sample. The stamp IS clay's brush edge, so hardness lives here: 1 = crisp
// chisel rim (the bitmap's old fixed AA band), 0 = the rim feathers all the way
// to the centre. Analytic rather than baked into the bitmap so the slider
// answers per dab — no texture re-upload, no resolution tie. Outer extent 0.705
// keeps the corners inside the kernels' dist < radius gate (inscribed square,
// see brush_alpha.cpp).
fn clay_square(rel : vec3<f32>, mirrored : u32) -> f32 {
    var tang = AP.tangent;
    var bitan = AP.bitangent;
    if (mirrored != 0u) {
        tang.x = -tang.x;
        bitan.x = -bitan.x;
    }
    let u = dot(rel, tang) * AP.inv_diameter * 2.0;
    let v = dot(rel, bitan) * AP.inv_diameter * 2.0;
    let m = max(abs(u), abs(v));
    let inner = min(0.705 * P.hardness, 0.70);
    return 1.0 - smoothstep(inner, 0.705, m);
}

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
           vp : vec3<f32>, vn : vec3<f32>, mirrored : u32) {
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return;
    }
    // Position-based gate (draw/clay; inflate self-limits by pushing every vert
    // along its OWN normal). The old facing gate dug one-vertex pits along deep
    // creases: vertex normals FLIP across a valley line (facing +0.4 → −0.6 between
    // neighbours), and no angular test — hard or feathered — survives that
    // discontinuity. Gate on position instead: reject verts significantly BEHIND
    // the anchor surface along the view (that's a thin sheet's far side), feathered
    // in position space (0.25R..0.45R) so bulgy forms don't get a hard wall. A
    // crease's far flank sits at ≈ the anchor's depth, so it deposits — no pit.
    // Far-backfacing verts (facing < ~−0.5, well below anything visible, crease
    // flanks included) are still cut as belt-and-suspenders for sheets thinner
    // than the window. P.facing_threshold is unused here for now.
    var gate_w = 1.0;
    if (P.inflate == 0u) {
        // The band widens as the surface tilts away. Face-on the whole footprint sits
        // at behind ~= 0 and 0.25R..0.45R is right; edge-on the LEGITIMATE surface
        // inside the dab spans nearly +-R of `behind`, so a fixed band amputates its
        // receding half. That missing half drags the deposit centroid toward the
        // camera, and since tilt varies along a curved stroke the drag varies with it
        // — which is the groove wandering sideways. Floor matches the CPU-side dab
        // spacing correction (DAB_SPACING_MIN_COS).
        let ct = max(abs(dot(view, anchor_n)), 0.30);
        let behind = dot(vp - anchor, view);
        gate_w = 1.0 - smoothstep(0.25 * P.world_radius / ct,
                                  0.45 * P.world_radius / ct, behind);
        gate_w = gate_w * smoothstep(-0.6, -0.4, -dot(vn, view));
        if (gate_w <= 0.0) {
            return;
        }
    }
    // Clay's stamp IS its edge: skip the radial falloff (it rounded the square's
    // corners into a squircle) and weight by the analytic square instead — that's
    // where clay's hardness applies (edge feather, see clay_square). Every other
    // brush keeps falloff*alpha.
    var w = 1.0;
    if (P.clay != 0u && AP.enabled != 0u) {
        w = clay_square(vp - anchor, mirrored);
    } else {
        w = brush_falloff(dist, P.world_radius) * sample_alpha(vp - anchor, mirrored);
    }
    w = w * gate_w;
    if (w <= 0.0) {
        return;
    }
    // Draw pushes the whole dab along the cursor's surface normal; inflate pushes
    // each vert along its own normal so the surface swells outward locally.
    var dir = anchor_n;
    if (P.inflate != 0u) {
        dir = vn;
    }

    // Clay displaces TOWARD a plane instead of BY a fixed amount. The plane sits
    // disp_amount above the anchor along anchor_n; a vert's signed height above it
    // is h, so (target - h) is the gap left to fill. Pulling toward the plane from
    // both sides is what makes clay build in flat layers: hollows fill, ridges past
    // the plane settle (melt-scaled), and repeat strokes converge at the plane
    // rather than growing without bound.
    if (P.clay != 0u) {
        // NB: not `target` — that's a WGSL reserved keyword and Tint rejects the
        // whole module for it (naga/glslang don't, so GL builds stay silent).
        let target_h = P.disp_amount;
        let h = dot(vp - anchor, dir);
        // Standard-clay approach (Blender-style): each dab moves verts a FRACTION of
        // the remaining gap to the plane instead of teleporting them onto it. The
        // overlapping dabs of a stroke still converge onto a flat layer behind the
        // leading edge, but each individual dab lays a soft increment — a spinning
        // or fast-moving stamp deposits blended steps instead of full-height cliffs.
        let raw = (target_h - h) * w * 0.5;
        // "Wrong side" = past the plane (proud on a build, sunk on a carve). Standard
        // clay pulls BOTH sides toward the plane — that's what makes it flatten as it
        // builds and even out crossings. clay_melt scales the wrong-side pull:
        // 1 = fully two-sided (Blender's clay), 0 = freeze proud detail (preserve
        // mode, the old phase-1 behavior). Direction test uses clay_sign, not raw's
        // sign: the area-plane bias can put the plane below the anchor (target_h < 0)
        // mid-build.
        // NB: keep each comparison on its own line. Inlining `raw < 0.0` as a
        // select() arg makes Tint read the `<` as a template-list opener and the
        // `>` in `clay_sign >= 0` as its close — the whole module fails to parse
        // (naga/glslang don't disambiguate, so GL/native stay silent). See
        // webgpudrawbug.md — this is the same trap as the `target` keyword.
        let proud = raw > 0.0;
        let sunk  = raw < 0.0;
        let build = P.clay_sign >= 0;
        let wrong = select(proud, sunk, build);
        var delta = select(raw, raw * P.clay_melt, wrong);
        // Plane trim (Blender's plane_trim): cap the per-dab move so a footprint
        // hanging over a deep hollow or a tall old stroke can't yank verts in one
        // violent step.
        let trim = 0.5 * P.world_radius;
        delta = clamp(delta, -trim, trim);
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
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    let v = block_list[wg.x] * 256u + lid.x;
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
