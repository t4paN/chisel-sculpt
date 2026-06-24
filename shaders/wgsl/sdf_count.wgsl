// sdf_count.wgsl — WGSL sibling of shaders/glsl/sdf_count.comp. See CONVENTIONS.md.
// SDF voxel-merge Pass A (count): one thread per triangle; store the clamped band-AABB
// (g0.xyz + span.xyz) into box[] so Pass C reuses the exact integers. Degenerate tris
// store span 0 → skipped downstream.
//
//   21 soup pos (read)  22 soup idx (read)  36 box (read_write)  63 params UBO

struct Params {
    u_origin   : vec3<f32>,   // 0
    u_voxel    : f32,         // 12
    u_R        : i32,         // 16
    u_band     : i32,         // 20
    u_triCount : u32,         // 24
    _pad0      : u32,         // 28
};

@group(0) @binding(21) var<storage, read>       p   : array<f32>;
@group(0) @binding(22) var<storage, read>       ix  : array<u32>;
@group(0) @binding(36) var<storage, read_write> box : array<u32>;   // 6 per tri
@group(0) @binding(63) var<uniform>             P   : Params;

// WGSL has no isNan/isInf: NaN via x != x, inf via magnitude past finite max.
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
    let t = gid.x + gid.y * nwg.x * 64u;
    if (t >= P.u_triCount) { return; }
    let o = 6u*t;
    let a = tri_vert(t,0u); let b = tri_vert(t,1u); let c = tri_vert(t,2u);
    if (tri_degenerate(a,b,c)) {
        box[o]=0u; box[o+1u]=0u; box[o+2u]=0u; box[o+3u]=0u; box[o+4u]=0u; box[o+5u]=0u;
        return;
    }
    let lo = min(min(a,b),c);
    let hi = max(max(a,b),c);
    var g0 = vec3<i32>(floor((lo - P.u_origin)/P.u_voxel)) - P.u_band;
    var g1 = vec3<i32>(ceil ((hi - P.u_origin)/P.u_voxel)) + P.u_band;
    g0 = clamp(g0, vec3<i32>(0), vec3<i32>(P.u_R));
    g1 = clamp(g1, vec3<i32>(0), vec3<i32>(P.u_R));
    let span = vec3<u32>(g1 - g0 + vec3<i32>(1));   // g0 clamped to [0,R] ⇒ non-negative
    box[o]=u32(g0.x); box[o+1u]=u32(g0.y); box[o+2u]=u32(g0.z);
    box[o+3u]=span.x; box[o+4u]=span.y;   box[o+5u]=span.z;
}
