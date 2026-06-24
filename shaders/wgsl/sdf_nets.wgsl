// sdf_nets.wgsl — WGSL sibling of shaders/glsl/sdf_nets.comp. See CONVENTIONS.md.
// SDF voxel-merge Surface Nets extractor (naive dual contouring). Same Field-in /
// McOut 18-floats/tri soup / McCnt counter interface and the same u_count_only / u_cap
// two-pass protocol as MC. One dual vertex per active cell; the four cells around each
// sign-changing grid edge form a quad (shortest-diagonal split, per-quad winding).
//
//   24 field (read)  25 mc_out (read_write)  26 mc_cnt (atomic)  63 params

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
@group(0) @binding(63) var<uniform>             P   : Params;

const CORNER = array<vec3<i32>, 8>(
    vec3<i32>(0,0,0), vec3<i32>(1,0,0), vec3<i32>(1,1,0), vec3<i32>(0,1,0),
    vec3<i32>(0,0,1), vec3<i32>(1,0,1), vec3<i32>(1,1,1), vec3<i32>(0,1,1));
const EDGE = array<vec2<i32>, 12>(
    vec2<i32>(0,1), vec2<i32>(1,2), vec2<i32>(2,3), vec2<i32>(3,0),
    vec2<i32>(4,5), vec2<i32>(5,6), vec2<i32>(6,7), vec2<i32>(7,4),
    vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7));

struct Dual { wpos : vec3<f32>, nrm : vec3<f32> };

fn fld(cc : vec3<i32>, R1 : i32) -> f32 {
    let c = clamp(cc, vec3<i32>(0), vec3<i32>(P.u_R));
    return f[c.x + R1*(c.y + R1*c.z)];
}
fn grad(c : vec3<i32>, R1 : i32) -> vec3<f32> {
    return vec3<f32>(fld(c+vec3<i32>(1,0,0),R1) - fld(c-vec3<i32>(1,0,0),R1),
                     fld(c+vec3<i32>(0,1,0),R1) - fld(c-vec3<i32>(0,1,0),R1),
                     fld(c+vec3<i32>(0,0,1),R1) - fld(c-vec3<i32>(0,0,1),R1));
}

// Dual vertex of cell `base`: world position + orientation normal. Edges in a FIXED
// order so the average is recomputed bit-identically from every referencing thread.
fn cell_vertex(base : vec3<i32>, R1 : i32) -> Dual {
    var val : array<f32, 8>;
    for (var s = 0; s < 8; s = s + 1) {
        let c = base + CORNER[s];
        val[s] = f[c.x + R1*(c.y + R1*c.z)];
    }
    var sum = vec3<f32>(0.0);
    var n = 0;
    for (var e = 0; e < 12; e = e + 1) {
        let a = EDGE[e].x; let b = EDGE[e].y;
        let fa = val[a]; let fb = val[b];
        if ((fa < 0.0) == (fb < 0.0)) { continue; }   // edge not crossed
        let tt = fa / (fa - fb);
        sum = sum + mix(vec3<f32>(base + CORNER[a]), vec3<f32>(base + CORNER[b]), vec3<f32>(tt));
        n = n + 1;
    }
    var out : Dual;
    out.wpos = P.u_origin + P.u_voxel * (sum / f32(max(n, 1)));
    let g = grad(base, R1);                            // outward (off-band side is +/-FAR)
    if (dot(g,g) > 0.0) { out.nrm = normalize(g); } else { out.nrm = vec3<f32>(0.0, 0.0, 1.0); }
    return out;
}

// Emit one triangle with an explicit, caller-decided winding flip.
fn put_tri(ip0 : vec3<f32>, ip1 : vec3<f32>, ip2 : vec3<f32>,
           in0 : vec3<f32>, in1 : vec3<f32>, in2 : vec3<f32>, flip : bool) {
    var p0 = ip0; var p1 = ip1; var p2 = ip2;
    var n0 = in0; var n1 = in1; var n2 = in2;
    if (flip) {
        let tp = p1; p1 = p2; p2 = tp;
        let tn = n1; n1 = n2; n2 = tn;
    }
    let slot = atomicAdd(&cnt[0], 1u);
    if (slot >= P.u_cap) { return; }                   // overflow guard (exact in practice)
    let b0 = slot * 18u;
    o[b0+0u]=p0.x;  o[b0+1u]=p0.y;  o[b0+2u]=p0.z;  o[b0+3u]=n0.x;  o[b0+4u]=n0.y;  o[b0+5u]=n0.z;
    o[b0+6u]=p1.x;  o[b0+7u]=p1.y;  o[b0+8u]=p1.z;  o[b0+9u]=n1.x;  o[b0+10u]=n1.y; o[b0+11u]=n1.z;
    o[b0+12u]=p2.x; o[b0+13u]=p2.y; o[b0+14u]=p2.z; o[b0+15u]=n2.x; o[b0+16u]=n2.y; o[b0+17u]=n2.z;
}

// Quad over the four cells around one sign-changing grid edge, split on its shortest
// diagonal with a single per-quad winding flip (so a sheared quad can't fold).
fn emit_quad(c0 : vec3<i32>, c1 : vec3<i32>, c2 : vec3<i32>, c3 : vec3<i32>, R1 : i32) {
    let d0 = cell_vertex(c0, R1); let d1 = cell_vertex(c1, R1);
    let d2 = cell_vertex(c2, R1); let d3 = cell_vertex(c3, R1);
    let p0 = d0.wpos; let p1 = d1.wpos; let p2 = d2.wpos; let p3 = d3.wpos;
    let n0 = d0.nrm;  let n1 = d1.nrm;  let n2 = d2.nrm;  let n3 = d3.nrm;

    let flip = dot(cross(p2 - p0, p3 - p1), n0 + n1 + n2 + n3) < 0.0;

    if (dot(p2 - p0, p2 - p0) <= dot(p3 - p1, p3 - p1)) {   // split on p0-p2
        put_tri(p0,p1,p2, n0,n1,n2, flip);
        put_tri(p0,p2,p3, n0,n2,n3, flip);
    } else {                                                // split on p1-p3
        put_tri(p1,p2,p3, n1,n2,n3, flip);
        put_tri(p1,p3,p0, n1,n3,n0, flip);
    }
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

    // Surface from the three grid edges leaving this cell's min-corner toward +x/+y/+z.
    let sc = (f[cx     + R1*(cy     + R1* cz   )] < 0.0);   // sign at the min corner
    let active_x = (sc != (f[(cx+1) + R1*(cy + R1*cz)] < 0.0)) && cy >= 1 && cz >= 1;
    let active_y = (sc != (f[cx + R1*((cy+1) + R1*cz)] < 0.0)) && cx >= 1 && cz >= 1;
    let active_z = (sc != (f[cx + R1*(cy + R1*(cz+1))] < 0.0)) && cx >= 1 && cy >= 1;

    if (P.u_count_only == 1) {
        var ntri = 0u;
        if (active_x) { ntri = ntri + 2u; }
        if (active_y) { ntri = ntri + 2u; }
        if (active_z) { ntri = ntri + 2u; }
        if (ntri > 0u) { atomicAdd(&cnt[0], ntri); }
        return;
    }

    if (active_x) { emit_quad(vec3<i32>(cx,cy-1,cz-1), vec3<i32>(cx,cy,cz-1),
                              vec3<i32>(cx,cy,cz),     vec3<i32>(cx,cy-1,cz),   R1); }
    if (active_y) { emit_quad(vec3<i32>(cx-1,cy,cz-1), vec3<i32>(cx,cy,cz-1),
                              vec3<i32>(cx,cy,cz),     vec3<i32>(cx-1,cy,cz),   R1); }
    if (active_z) { emit_quad(vec3<i32>(cx-1,cy-1,cz), vec3<i32>(cx,cy-1,cz),
                              vec3<i32>(cx,cy,cz),     vec3<i32>(cx-1,cy,cz),   R1); }
}
