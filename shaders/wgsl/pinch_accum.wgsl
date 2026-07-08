// pinch_accum.wgsl
// Port of src/compute_crease_pinch.cpp (pinch_accum_src). Pinch brush accumulate
// pass: pinch_amount >= 0 pulls verts tangentially toward the anchor (pinch); < 0
// is an asymmetric flatten — shaves peaks fully, fills valleys partially (biased
// subtractive). Deposits into the
// shared accum buffer; the apply side reuses draw_apply / symmetrize / mirror_apply.
// Lockstep with pinch_accum.comp. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum:
//   0 positions (read)   1 stroke normals (read)   3 accum (read_write)
//   63 params UBO (BIND_PARAMS)

// std140/uniform layout — the C++ upload struct MUST match this byte-for-byte (96 B).
struct Params {
    anchor_a         : vec3<f32>,   //   0..11
    world_radius     : f32,         //  12
    anchor_b         : vec3<f32>,   //  16..27
    pinch_amount     : f32,         //  28
    view_a           : vec3<f32>,   //  32..43
    hardness         : f32,         //  44
    view_b           : vec3<f32>,   //  48..59
    facing_threshold : f32,         //  60
    anchor_normal_a  : vec3<f32>,   //  64..75
    use_b            : u32,         //  76
    anchor_normal_b  : vec3<f32>,   //  80..91
    vertex_count     : u32,         //  92  (struct is 96 bytes)
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals   : array<f32>;
@group(0) @binding(3)  var<storage, read_write> accum     : array<atomic<u32>>;
@group(0) @binding(63) var<uniform>             P         : Params;

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

    let facing = -dot(vn, view);
    if (facing < P.facing_threshold) {
        return;
    }

    let w = brush_falloff(dist, P.world_radius);
    if (w <= 0.0) {
        return;
    }

    var d : vec3<f32>;
    if (P.pinch_amount >= 0.0) {
        if (dist < 1e-6) {
            return;
        }
        let to_anchor = anchor - vp;
        let tangent = to_anchor - dot(to_anchor, anchor_n) * anchor_n;
        d = tangent * (w * P.pinch_amount / P.world_radius);
    } else {
        // Reverse pinch: an asymmetric flatten biased subtractive — a middle
        // ground between the old symmetric flatten (raised valleys as much as it
        // cut peaks) and a pure scrape (never filled valleys at all). The cut
        // plane sits a touch below the anchor; peaks above it are shaved at full
        // strength, valleys below get only a fraction of the fill.
        let depth = P.world_radius * 0.08;
        let height = dot(vp - anchor, anchor_n) + depth;
        var factor = 1.0;
        if (height < 0.0) {
            factor = 0.4;
        }
        d = -anchor_n * (height * factor * w * 1.5 * (-P.pinch_amount) / P.world_radius);
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

    deposit(v, P.anchor_a, P.view_a, P.anchor_normal_a, vp, vn);
    if (P.use_b != 0u) {
        deposit(v, P.anchor_b, P.view_b, P.anchor_normal_b, vp, vn);
    }
}
