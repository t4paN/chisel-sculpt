// clay_relax.wgsl — WGSL sibling of shaders/glsl/clay_relax.comp. Keep lockstep.
// Pen-up wall-fill for the clay brush: a normal-stripped (tangential) Laplacian over
// the stroke's dirty-vert id list, gated to STRETCHED triangles only. Clay raises
// slabs without adding verts, so the near-vertical walls between the raised top and
// the base are spanned by a thin ring of stretched verts → long skinny triangles.
// This slides verts down into those walls to even the spacing, without deflating the
// form (normal component stripped, like limb_relax) or rounding the crisp square rim
// (the anisotropy gate skips isotropic top/rim verts). Redistribution only — no new
// triangles. In-place RW like the autosmooth pass. 16-byte std140 Params UBO.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)  1 normals (read)  2 indices (read)
//   4 adj_offset (read)  5 adj_list (read)  6 dirty_verts (read)  12 mask (read)
//   63 params UBO (BIND_PARAMS)

struct Params {
    dirty_count : u32,    // 0
    strength    : f32,    // 4
    aniso_lo    : f32,    // 8
    fill_bias   : f32,    // 12  >0 up-weights long edges → verts drift INTO the wall
};

@group(0) @binding(0)  var<storage, read_write> positions    : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals      : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices      : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset   : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list     : array<u32>;
@group(0) @binding(6)  var<storage, read>       dirty_verts  : array<u32>;
@group(0) @binding(12) var<storage, read>       mask         : array<f32>;
@group(0) @binding(63) var<uniform>             P            : Params;

fn getp(i : u32) -> vec3<f32> {
    return vec3<f32>(positions[i*3u], positions[i*3u+1u], positions[i*3u+2u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let di = gid.x;
    if (di >= P.dirty_count) {
        return;
    }
    let v = dirty_verts[di];

    let mscale = 1.0 - mask[v];
    if (mscale <= 0.0) {
        return;
    }

    let p = getp(v);
    let start = adj_offset[v];
    let end   = adj_offset[v + 1u];

    // Pass 1: edge stats. emean feeds the length-weighting; emax/emin the gate.
    var emin = 1e30;
    var emax = 0.0;
    var esum = 0.0;
    var ecount = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t*3u]; let i1 = indices[t*3u+1u]; let i2 = indices[t*3u+2u];
        var a = i0; if (i0 == v) { a = i1; }
        var b = i2; if (i2 == v) { b = i1; }
        let la = length(getp(a) - p); let lb = length(getp(b) - p);
        emin = min(emin, min(la, lb)); emax = max(emax, max(la, lb));
        esum = esum + la + lb; ecount = ecount + 2.0;
    }
    if (ecount <= 0.0 || emin >= 1e29) {
        return;
    }

    // Anisotropy gate: stretched wall triangles have a long (vertical) and a short
    // (horizontal) edge → high max/min. Flat top / base / rim triangles are near
    // isotropic → ~1. The gate ramps over a fixed span above aniso_lo, so verts on
    // the stretched walls (and the rim quads bordering them) relax while the flat
    // crisp faces stay put.
    let aniso = emax / max(emin, 1e-8);
    let gate = smoothstep(P.aniso_lo, P.aniso_lo + 1.5, aniso);
    if (gate <= 0.0) {
        return;
    }

    // Pass 2: length-weighted centroid. Neighbours across LONGER-than-average edges
    // (the sparse, stretched direction — down the wall) pull harder, so the vert
    // drifts toward the sparse side instead of just evening in place. That's what
    // actually feeds verts into the wall. fill_bias == 0 → plain umbrella Laplacian.
    let emean = esum / ecount;
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t*3u]; let i1 = indices[t*3u+1u]; let i2 = indices[t*3u+2u];
        var a = i0; if (i0 == v) { a = i1; }
        var b = i2; if (i2 == v) { b = i1; }
        let qa = getp(a); let qb = getp(b);
        let wa = 1.0 + P.fill_bias * max(0.0, length(qa - p) / emean - 1.0);
        let wb = 1.0 + P.fill_bias * max(0.0, length(qb - p) / emean - 1.0);
        sum = sum + qa * wa + qb * wb; wsum = wsum + wa + wb;
    }
    if (wsum <= 0.0) {
        return;
    }

    var delta = sum / wsum - p;
    // Strip the normal component → slide along the surface only (a full Laplacian
    // would deflate the slab, the same way the smooth brush shrinks form).
    var n = vec3<f32>(normals[v*3u], normals[v*3u+1u], normals[v*3u+2u]);
    let nl = length(n);
    if (nl > 1e-8) {
        n = n / nl;
        delta = delta - n * dot(delta, n);
    }

    let np = p + P.strength * mscale * gate * delta;
    positions[v*3u] = np.x; positions[v*3u+1u] = np.y; positions[v*3u+2u] = np.z;
}
