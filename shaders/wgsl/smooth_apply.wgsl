// smooth_apply.wgsl
// Port of src/compute_smooth.cpp (smooth_apply). Pass 2 of the smooth brush, run once
// per iteration: a Laplacian over each touched vertex's 1-ring, each neighbour weighted
// by the surface it owns, squared (see owned()). The mirror seam crease is handled by
// re-imposing the X reflection after every iteration in dispatch_smooth (not by a
// normal projection in here — that variant pinched ridges/bumps under prolonged
// smoothing). Tail UBO slots are padding kept for shape parity.
// See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   2 indices (read)        3 accum (read)
//   4 adj_offset (read)        5 adj_list (read)        12 mask (read)
//   63 params UBO (BIND_PARAMS)

struct Params {
    vertex_count : u32,   // 0
    strength     : f32,   // 4
    _pad0        : u32,   // 8
    _pad1        : u32,   // 12
};

struct DirtyRegion {
    base       : u32,
    cap        : u32,
    block_mode : u32,   // 1 = dispatch is one workgroup per active block
    _p1        : u32,
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(3)  var<storage, read>       accum      : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(12) var<storage, read>       mask       : array<f32>;
@group(0) @binding(45) var<storage, read>       block_list : array<u32>;
@group(0) @binding(61) var<uniform>             DR         : DirtyRegion;
@group(0) @binding(63) var<uniform>             P          : Params;

fn pos3(i : u32) -> vec3<f32> {
    return vec3<f32>(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
}

// Surface a vertex owns: the summed (doubled) area of its triangles.
// A plain neighbour average packs the tris round a 5-edge vertex tight and spreads
// those round a 7-edge vertex wide, so every stroke used to pinch the poles the SDF
// remesh had just evened out. Weighting each neighbour by owned()^2 is that remesh's
// own rule (relax_to_field, even_area): a vertex is drawn toward neighbours holding
// too much surface and lets go of those holding too little. Where the tris are
// already even the weights are equal and this IS the plain average.
fn owned(u : u32) -> f32 {
    var s = 0.0;
    let b = adj_offset[u];
    let e = adj_offset[u + 1u];
    for (var j = b; j < e; j = j + 1u) {
        let t = adj_list[j];
        let p0 = pos3(indices[t * 3u]);
        let p1 = pos3(indices[t * 3u + 1u]);
        let p2 = pos3(indices[t * 3u + 2u]);
        s = s + length(cross(p1 - p0, p2 - p0));
    }
    return s;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    // Dispatch culling: one workgroup per ACTIVE block; entry 0 is the count, hence +1.
    // Neighbour positions are read by direct index, so a neighbour outside the
    // selection is still read correctly — it simply is not itself updated, which is
    // right, because it is outside the brush.
    let v = select(wg.x * 64u + lid.x,
                   block_list[1u + wg.x] * 64u + lid.x,
                   DR.block_mode == 1u);
    if (v >= P.vertex_count) {
        return;
    }

    let w = bitcast<f32>(accum[v * 4u + 3u]);
    if (w <= 0.0) {
        return;
    }

    let mscale = 1.0 - mask[v];
    if (mscale <= 0.0) {
        return;
    }

    let cur = pos3(v);

    // Own surface, only to bring the neighbours' into the range of 1 before squaring:
    // on a dense mesh a raw area squared is small enough to lose bits.
    let own = owned(v);
    if (own <= 0.0) {
        return;
    }
    let inv_own = 1.0 / own;

    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var wsum = 0.0;

    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];

        // The two corners that aren't v. Each neighbour turns up in two tris, so it
        // is counted twice, the same for all of them.
        var na = i0;
        var nb = i1;
        if (v == i0) {
            na = i1;
            nb = i2;
        } else if (v == i1) {
            nb = i2;
        }

        let ra = owned(na) * inv_own;
        let rb = owned(nb) * inv_own;
        let wa = ra * ra;
        let wb = rb * rb;
        sum = sum + pos3(na) * wa + pos3(nb) * wb;
        wsum = wsum + wa + wb;
    }

    if (wsum <= 0.0) {
        return;
    }

    let blend = w * P.strength * mscale;
    let moved = cur + (sum / wsum - cur) * blend;
    positions[v * 3u]      = moved.x;
    positions[v * 3u + 1u] = moved.y;
    positions[v * 3u + 2u] = moved.z;
}
