// sdf_mc.wgsl — WGSL sibling of shaders/glsl/sdf_mc.comp. See CONVENTIONS.md.
// SDF voxel-merge Marching Cubes extractor (default). Two passes driven by
// u_count_only: pass 1 tallies tris (reserve once), pass 2 emits the 18-floats/tri
// soup guarded by u_cap. Shared edges canonicalized by global corner coords for a
// bit-identical shared vertex; output winding oriented to the field gradient.
//
//   24 field (read)  25 mc_out (read_write)  26 mc_cnt (atomic)  27 tritable (read)  63 params

struct Params {
    u_origin     : vec3<f32>,  // 0
    u_voxel      : f32,        // 12
    u_R          : i32,        // 16
    u_cellCount  : u32,        // 20
    u_count_only : i32,        // 24
    u_cap        : u32,        // 28
};

@group(0) @binding(24) var<storage, read>       f   : array<f32>;
@group(0) @binding(25) var<storage, read_write> o   : array<f32>;   // 18 floats / tri
@group(0) @binding(26) var<storage, read_write> cnt : array<atomic<u32>>;
@group(0) @binding(27) var<storage, read>       tri : array<i32>;   // 256*16
@group(0) @binding(63) var<uniform>             P   : Params;

const CORNER = array<vec3<i32>, 8>(
    vec3<i32>(0,0,0), vec3<i32>(1,0,0), vec3<i32>(1,1,0), vec3<i32>(0,1,0),
    vec3<i32>(0,0,1), vec3<i32>(1,0,1), vec3<i32>(1,1,1), vec3<i32>(0,1,1));
const EDGE = array<vec2<i32>, 12>(
    vec2<i32>(0,1), vec2<i32>(1,2), vec2<i32>(2,3), vec2<i32>(3,0),
    vec2<i32>(4,5), vec2<i32>(5,6), vec2<i32>(6,7), vec2<i32>(7,4),
    vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7));

fn fld(cc : vec3<i32>, R1 : i32) -> f32 {
    let c = clamp(cc, vec3<i32>(0), vec3<i32>(P.u_R));
    return f[c.x + R1*(c.y + R1*c.z)];
}
fn grad(c : vec3<i32>, R1 : i32) -> vec3<f32> {
    return vec3<f32>(fld(c+vec3<i32>(1,0,0),R1) - fld(c-vec3<i32>(1,0,0),R1),
                     fld(c+vec3<i32>(0,1,0),R1) - fld(c-vec3<i32>(0,1,0),R1),
                     fld(c+vec3<i32>(0,0,1),R1) - fld(c-vec3<i32>(0,0,1),R1));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups)        nwg : vec3<u32>) {
    let cell = gid.x + gid.y * nwg.x * 64u;
    if (cell >= P.u_cellCount) { return; }
    let R1 = P.u_R + 1;
    let cx =  i32(cell) % P.u_R;
    let cy = (i32(cell) / P.u_R) % P.u_R;
    let cz =  i32(cell) / (P.u_R*P.u_R);
    let base = vec3<i32>(cx, cy, cz);

    var val : array<f32, 8>;
    var cubeindex = 0;
    for (var s = 0; s < 8; s = s + 1) {
        let c = base + CORNER[s];
        val[s] = f[c.x + R1*(c.y + R1*c.z)];
        if (val[s] < 0.0) { cubeindex = cubeindex | (1 << u32(s)); }
    }
    if (cubeindex == 0 || cubeindex == 255) { return; }

    let row = cubeindex * 16;

    // Count pass: count fully determined by the case table; one atomic per active cell.
    if (P.u_count_only == 1) {
        var ntri = 0u;
        for (var i = 0; i < 15 && tri[row+i] != -1; i = i + 3) { ntri = ntri + 1u; }
        if (ntri > 0u) { atomicAdd(&cnt[0], ntri); }
        return;
    }

    var vpos : array<vec3<f32>, 12>;
    var vnrm : array<vec3<f32>, 12>;
    for (var e = 0; e < 12; e = e + 1) {
        let ai = EDGE[e].x; let bi = EDGE[e].y;
        var fa = val[ai]; var fb = val[bi];
        if ((fa < 0.0) == (fb < 0.0)) { continue; }   // edge not crossed
        var ca = base + CORNER[ai];
        var cb = base + CORNER[bi];
        // Canonicalize the edge by GLOBAL corner coords (bit-identical shared vertex).
        var swap = false;
        if (ca.x != cb.x)      { swap = ca.x > cb.x; }
        else if (ca.y != cb.y) { swap = ca.y > cb.y; }
        else                   { swap = ca.z > cb.z; }
        if (swap) {
            let tc = ca; ca = cb; cb = tc;
            let tf = fa; fa = fb; fb = tf;
        }
        let tt = fa / (fa - fb);
        vpos[e] = P.u_origin + P.u_voxel * (vec3<f32>(ca) + tt * vec3<f32>(cb - ca));
        let nn = mix(grad(ca, R1), grad(cb, R1), vec3<f32>(tt));
        if (dot(nn,nn) > 0.0) { vnrm[e] = normalize(nn); } else { vnrm[e] = vec3<f32>(0.0, 0.0, 1.0); }
    }

    for (var i = 0; i < 15 && tri[row+i] != -1; i = i + 3) {
        let e0 = tri[row+i]; let e1 = tri[row+i+1]; let e2 = tri[row+i+2];
        var p0 = vpos[e0]; var p1 = vpos[e1]; var p2 = vpos[e2];
        var n0 = vnrm[e0]; var n1 = vnrm[e1]; var n2 = vnrm[e2];
        // Orient so face winding matches the field gradient (outward).
        if (dot(cross(p1 - p0, p2 - p0), n0 + n1 + n2) < 0.0) {
            let tp = p1; p1 = p2; p2 = tp;
            let tn = n1; n1 = n2; n2 = tn;
        }
        let slot = atomicAdd(&cnt[0], 1u);
        if (slot >= P.u_cap) { continue; }   // overflow guard (exact in practice)
        let b0 = slot * 18u;
        o[b0+0u]=p0.x;  o[b0+1u]=p0.y;  o[b0+2u]=p0.z;  o[b0+3u]=n0.x;  o[b0+4u]=n0.y;  o[b0+5u]=n0.z;
        o[b0+6u]=p1.x;  o[b0+7u]=p1.y;  o[b0+8u]=p1.z;  o[b0+9u]=n1.x;  o[b0+10u]=n1.y; o[b0+11u]=n1.z;
        o[b0+12u]=p2.x; o[b0+13u]=p2.y; o[b0+14u]=p2.z; o[b0+15u]=n2.x; o[b0+16u]=n2.y; o[b0+17u]=n2.z;
    }
}
