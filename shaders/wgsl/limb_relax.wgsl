// limb_relax.wgsl
// Port of src/compute_limb.cpp (limb_relax_src). Pass 2 of the limb brush: ping-pong
// tangential (normal-stripped) Laplacian over the captured set. Evens vertex spacing
// along the stretching shaft without deflating the form — a tip-biased centroid drifts
// verts up-shaft so the cap densifies as the limb grows. Reads the src snapshot
// (BIND_LIMB_POS_SRC), writes the live positions; the host swaps the two each
// iteration. Dispatched over the whole vertex range — verts outside the captured set
// (w==0) pass through unchanged so both buffers stay valid for the next iteration's
// neighbour reads. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   29 pos src (read)    0 pos dst (read_write)   1 normals (read)   2 indices (read)
//   4 adj_offset (read)  5 adj_list (read)        9 weights (read)   12 mask (read)
//   63 params UBO (BIND_PARAMS)

struct Params {
    vertex_count : u32,    // 0
    lambda       : f32,    // 4
    tip_bias     : f32,    // 8
    _pad0        : f32,    // 12
    tip_dir      : vec3<f32>,  // 16 (struct rounds to 32)
    _pad1        : f32,    // 28
};

@group(0) @binding(29) var<storage, read>       src          : array<f32>;
@group(0) @binding(0)  var<storage, read_write> dst          : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals      : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices      : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset   : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list     : array<u32>;
@group(0) @binding(9)  var<storage, read>       move_weights : array<vec2<f32>>;
@group(0) @binding(12) var<storage, read>       mask         : array<f32>;
@group(0) @binding(63) var<uniform>             P            : Params;

// Accumulate one neighbour into the (tip-biased) centroid. Neighbours toward the tip
// are up-weighted, so each vert's relax target shifts up the shaft: the distribution
// migrates toward the leading end, densifying the cap. The target stays a convex
// combination of the 1-ring → a bounded redistribution, not an extra translation.
fn accum_neighbour(vi : u32, p : vec3<f32>, tdir : vec3<f32>,
                   sum : ptr<function, vec3<f32>>, wsum : ptr<function, f32>) {
    let q = vec3<f32>(src[vi*3u], src[vi*3u+1u], src[vi*3u+2u]);
    let e = q - p;
    let el = length(e);
    var b = 1.0;
    if (el > 1e-8) {
        b = b + P.tip_bias * max(0.0, dot(e / el, tdir));
    }
    *sum = *sum + q * b;
    *wsum = *wsum + b;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let p = vec3<f32>(src[v*3u], src[v*3u+1u], src[v*3u+2u]);

    let w2 = move_weights[v];
    let w = max(w2.x, w2.y);
    if (w <= 0.0) {                       // outside the brush: pass through
        dst[v*3u] = p.x; dst[v*3u+1u] = p.y; dst[v*3u+2u] = p.z;
        return;
    }

    // Verts the mirror lobe owns (w.y dominant) drift toward the X-flipped pull,
    // matching the drag's mdelta = (-dx, dy, dz) decomposition.
    var tdir = P.tip_dir;
    if (w2.y > w2.x) {
        tdir = vec3<f32>(-P.tip_dir.x, P.tip_dir.y, P.tip_dir.z);
    }

    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    let start = adj_offset[v];
    let end   = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t*3u]; let i1 = indices[t*3u+1u]; let i2 = indices[t*3u+2u];
        if (i0 != v) { accum_neighbour(i0, p, tdir, &sum, &wsum); }
        if (i1 != v) { accum_neighbour(i1, p, tdir, &sum, &wsum); }
        if (i2 != v) { accum_neighbour(i2, p, tdir, &sum, &wsum); }
    }
    if (wsum <= 0.0) {
        dst[v*3u] = p.x; dst[v*3u+1u] = p.y; dst[v*3u+2u] = p.z;
        return;
    }

    var delta = sum / wsum - p;
    // Strip the normal component → tangential move only (a full Laplacian would shrink
    // the form, the same reason the smooth brush deflates).
    var n = vec3<f32>(normals[v*3u], normals[v*3u+1u], normals[v*3u+2u]);
    let nl = length(n);
    if (nl > 1e-8) {
        n = n / nl;
        delta = delta - n * dot(delta, n);
    }

    let mscale = 1.0 - mask[v];
    let np = p + P.lambda * w * mscale * delta;
    dst[v*3u] = np.x; dst[v*3u+1u] = np.y; dst[v*3u+2u] = np.z;
}
