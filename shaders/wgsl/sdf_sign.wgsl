// sdf_sign.wgsl — WGSL sibling of shaders/glsl/sdf_sign.comp. See CONVENTIONS.md.
// SDF voxel-merge winding sign at band corners only (off-band → bandFar sentinel for
// the CPU flood fill). Chunked via u_cornerOffset so the host budgets the dominant
// pass across frames. dist read non-atomically here (written atomically in expand).
//
// Winding via the Fast Winding Number tree (Barill et al. 2018): far nodes use a
// dipole/order-1 expansion, near leaves are summed exactly. Host: sdf.cpp build_fwn_tree.
//
//  21 pos  22 idx  23 dist  24 field(rw)  38 nodes  39 triOrder  63 params

struct Params {
    u_origin       : vec3<f32>,  // 0
    u_voxel        : f32,        // 12
    u_R            : i32,        // 16
    u_triCount     : u32,        // 20
    u_cornerCount  : u32,        // 24
    u_bandFar      : f32,        // 28
    u_cornerOffset : u32,        // 32
    u_beta         : f32,        // 36  FWN far-field acceptance threshold
    _pad1          : u32,        // 40
    _pad2          : u32,        // 44
};

// FWN tree node — mirrors FwnNodeGPU in sdf.cpp (96 bytes, vec4-aligned).
struct FwnNode {
    p_r  : vec4<f32>,   // xyz expansion point, w subtree radius
    g0   : vec4<f32>,   // xyz Σ area-weighted normals (order 0)
    t0   : vec4<f32>,   // T row 0
    t1   : vec4<f32>,   // T row 1
    t2   : vec4<f32>,   // T row 2
    meta : vec4<i32>,   // x left, y right (<0 ⇒ leaf), z triStart, w triCount
};

@group(0) @binding(21) var<storage, read>       p     : array<f32>;
@group(0) @binding(22) var<storage, read>       ix    : array<u32>;
@group(0) @binding(23) var<storage, read>       d     : array<u32>;
@group(0) @binding(24) var<storage, read_write> f     : array<f32>;
@group(0) @binding(38) var<storage, read>       nodes : array<FwnNode>;
@group(0) @binding(39) var<storage, read>       triOrder : array<u32>;
@group(0) @binding(63) var<uniform>             P     : Params;

const FOUR_PI : f32 = 12.566370614359172;

fn nonfinite(v : vec3<f32>) -> bool {
    return !(v.x == v.x && v.y == v.y && v.z == v.z)
        || abs(v.x) > 3.4e38 || abs(v.y) > 3.4e38 || abs(v.z) > 3.4e38;
}
fn tri_degenerate(a : vec3<f32>, b : vec3<f32>, c : vec3<f32>) -> bool {
    if (nonfinite(a) || nonfinite(b) || nonfinite(c)) { return true; }
    return length(cross(b - a, c - a)) < 1e-20;
}
fn tri_vert(t : u32, k : u32) -> vec3<f32> {
    let v = ix[3u*t + k];
    return vec3<f32>(p[3u*v], p[3u*v+1u], p[3u*v+2u]);
}
fn tri_solid_angle(t : u32, q : vec3<f32>) -> f32 {
    let va = tri_vert(t,0u); let vb = tri_vert(t,1u); let vc = tri_vert(t,2u);
    if (tri_degenerate(va,vb,vc)) { return 0.0; }
    let A = va - q; let B = vb - q; let C = vc - q;
    let la = length(A); let lb = length(B); let lc = length(C);
    let num = dot(A, cross(B,C));
    let den = la*lb*lc + dot(A,B)*lc + dot(B,C)*la + dot(C,A)*lb;
    return 2.0 * atan2(num, den);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups)        nwg : vec3<u32>) {
    let ci = gid.x + gid.y * nwg.x * 64u + P.u_cornerOffset;
    if (ci >= P.u_cornerCount) { return; }
    let dist = bitcast<f32>(d[ci]);
    if (dist >= P.u_bandFar) { f[ci] = P.u_bandFar; return; }   // off-band → CPU flood

    let R1 = P.u_R + 1;
    let i =  i32(ci) % R1;
    let j = (i32(ci) / R1) % R1;
    let k =  i32(ci) / (R1*R1);
    let q = P.u_origin + P.u_voxel*vec3<f32>(f32(i), f32(j), f32(k));

    // Barnes-Hut traversal of the FWN tree; accumulate winding number w directly.
    var w = 0.0;
    var stack : array<i32, 64>;
    var sp = 0;
    stack[sp] = 0; sp = sp + 1;                 // root
    while (sp > 0) {
        sp = sp - 1;
        let nd = nodes[stack[sp]];
        let dvec = nd.p_r.xyz - q;
        let r2 = dot(dvec, dvec);
        let bound = P.u_beta * nd.p_r.w;
        if (r2 > bound*bound) {
            // Far field: dipole (order 0) + order-1 expansion of (x-q)/|x-q|^3.
            let r   = sqrt(r2);
            let ir3 = 1.0 / (r2 * r);
            let ir5 = ir3 / r2;
            let w0  = dot(dvec, nd.g0.xyz) * ir3;
            let trT = nd.t0.x + nd.t1.y + nd.t2.z;
            let Td  = vec3<f32>(dot(nd.t0.xyz, dvec), dot(nd.t1.xyz, dvec), dot(nd.t2.xyz, dvec));
            let dTd = dot(dvec, Td);
            w = w + (w0 + trT*ir3 - 3.0*dTd*ir5) / FOUR_PI;
        } else if (nd.meta.x < 0) {
            // Near leaf: exact solid-angle sum.
            for (var n = 0; n < nd.meta.w; n = n + 1) {
                w = w + tri_solid_angle(triOrder[u32(nd.meta.z + n)], q) / FOUR_PI;
            }
        } else if (sp <= 62) {                  // recurse (guard the fixed stack)
            stack[sp] = nd.meta.x; sp = sp + 1;
            stack[sp] = nd.meta.y; sp = sp + 1;
        }
    }
    f[ci] = select(1.0, -1.0, w > 0.5) * dist;        // signed real distance (band)
}
