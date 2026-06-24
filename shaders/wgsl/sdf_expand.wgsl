// sdf_expand.wgsl — WGSL sibling of shaders/glsl/sdf_expand.comp. See CONVENTIONS.md.
// SDF voxel-merge Pass C (expand + splat): one thread per work item; binary-search the
// owner tri in off[], unravel local index → voxel, one distance, atomicMin into the
// field. dist is the uint-bits distance grid (monotonic atomicMin for dd >= 0).
//
//   21 pos (read)  22 idx (read)  23 dist (atomic rw)  36 box (read)  37 off (read)  63 params

struct Params {
    u_origin    : vec3<f32>,  // 0
    u_voxel     : f32,        // 12
    u_R         : i32,        // 16
    u_triCount  : u32,        // 20
    u_workCount : u32,        // 24
    _pad0       : u32,        // 28
};

@group(0) @binding(21) var<storage, read>       p   : array<f32>;
@group(0) @binding(22) var<storage, read>       ix  : array<u32>;
@group(0) @binding(23) var<storage, read_write> d   : array<atomic<u32>>;
@group(0) @binding(36) var<storage, read>       box : array<u32>;
@group(0) @binding(37) var<storage, read>       off : array<u32>;
@group(0) @binding(63) var<uniform>             P   : Params;

fn tri_vert(t : u32, k : u32) -> vec3<f32> {
    let v = ix[3u*t + k];
    return vec3<f32>(p[3u*v], p[3u*v+1u], p[3u*v+2u]);
}

// Ericson closest-point-on-triangle Euclidean distance (degenerates filtered upstream).
fn pt_tri_dist(pt : vec3<f32>, a : vec3<f32>, b : vec3<f32>, c : vec3<f32>) -> f32 {
    let ab = b - a; let ac = c - a; let ap = pt - a;
    let d1 = dot(ab, ap); let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
    let bp = pt - b;
    let d3 = dot(ab, bp); let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
    let vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return length(pt - (a + v*ab));
    }
    let cp = pt - c;
    let d5 = dot(ab, cp); let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
    let vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return length(pt - (a + w*ac));
    }
    let va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return length(pt - (b + w*(c - b)));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom; let w = vc * denom;
    return length(pt - (a + ab*v + ac*w));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups)        nwg : vec3<u32>) {
    let g = gid.x + gid.y * nwg.x * 64u;
    if (g >= P.u_workCount) { return; }
    // Owner = upper_bound(off, g) - 1 over [0, u_triCount): first i with off[i] > g.
    var lo = 0u; var hi = P.u_triCount;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        if (off[mid] <= g) { lo = mid + 1u; } else { hi = mid; }
    }
    let t = lo - 1u;
    let j = g - off[t];                          // local work index within triangle t
    let o = 6u*t;
    let g0   = vec3<i32>(i32(box[o]), i32(box[o+1u]), i32(box[o+2u]));
    let span = vec3<u32>(box[o+3u], box[o+4u], box[o+5u]);
    let dx =  j % span.x;
    let dy = (j / span.x) % span.y;
    let dz =  j / (span.x * span.y);
    let gi = g0.x + i32(dx); let gj = g0.y + i32(dy); let gk = g0.z + i32(dz);
    let a = tri_vert(t,0u); let b = tri_vert(t,1u); let c = tri_vert(t,2u);
    let q = P.u_origin + P.u_voxel*vec3<f32>(f32(gi), f32(gj), f32(gk));
    let dd = pt_tri_dist(q, a, b, c);
    let R1 = P.u_R + 1;
    let ci = u32(gi + R1*(gj + R1*gk));
    atomicMin(&d[ci], bitcast<u32>(dd));          // monotonic for dd >= 0
}
