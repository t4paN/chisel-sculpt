// sdf_sign.wgsl — WGSL sibling of shaders/glsl/sdf_sign.comp. See CONVENTIONS.md.
// SDF voxel-merge winding sign at band corners only (off-band → bandFar sentinel for
// the CPU flood fill). Chunked via u_cornerOffset so the host budgets the dominant
// pass across frames. dist read non-atomically here (written atomically in expand).
//
//   21 pos (read)  22 idx (read)  23 dist (read)  24 field (read_write)  63 params

struct Params {
    u_origin       : vec3<f32>,  // 0
    u_voxel        : f32,        // 12
    u_R            : i32,        // 16
    u_triCount     : u32,        // 20
    u_cornerCount  : u32,        // 24
    u_bandFar      : f32,        // 28
    u_cornerOffset : u32,        // 32
    _pad0          : u32,        // 36
    _pad1          : u32,        // 40
    _pad2          : u32,        // 44
};

@group(0) @binding(21) var<storage, read>       p  : array<f32>;
@group(0) @binding(22) var<storage, read>       ix : array<u32>;
@group(0) @binding(23) var<storage, read>       d  : array<u32>;
@group(0) @binding(24) var<storage, read_write> f  : array<f32>;
@group(0) @binding(63) var<uniform>             P  : Params;

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

    var omega = 0.0;
    for (var t = 0u; t < P.u_triCount; t = t + 1u) {
        let va = tri_vert(t,0u); let vb = tri_vert(t,1u); let vc = tri_vert(t,2u);
        if (tri_degenerate(va,vb,vc)) { continue; }   // keep degenerates out of the sum
        let A = va - q; let B = vb - q; let C = vc - q;
        let la = length(A); let lb = length(B); let lc = length(C);
        let num = dot(A, cross(B,C));
        let den = la*lb*lc + dot(A,B)*lc + dot(B,C)*la + dot(C,A)*lb;
        omega = omega + 2.0 * atan2(num, den);
    }
    let w = omega / FOUR_PI;
    f[ci] = select(1.0, -1.0, w > 0.5) * dist;        // signed real distance (band)
}
