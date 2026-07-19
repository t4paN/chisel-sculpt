// cascade_midpoint.wgsl — WGSL sibling of shaders/glsl/cascade_midpoint.comp.
// GPU level-switch replay, pass 2 of 3: midpoint vertices from the cached
// SubdivStencil mid table (4 ids per edge: v0, v1, opp0, opp1; opp1 ==
// 0xFFFFFFFF on a boundary edge — kept for safety, the GPU path only runs on
// closed meshes). Bindings: 42 src pos, 43 dst pos, 44 mid table, 63 params.

struct Params {
    edge_count : u32,   // number of midpoint verts this pass
    vcoarse    : u32,   // coarse vertex count (write offset)
    _pad0      : u32,
    _pad1      : u32,   // struct rounds to 16
};

@group(0) @binding(42) var<storage, read>       src_pos : array<f32>;
@group(0) @binding(43) var<storage, read_write> dst_pos : array<f32>;
@group(0) @binding(44) var<storage, read>       mid     : array<u32>;
@group(0) @binding(63) var<uniform>             P       : Params;

fn fetch(v : u32) -> vec3<f32> {
    return vec3<f32>(src_pos[v * 3u], src_pos[v * 3u + 1u], src_pos[v * 3u + 2u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let e = gid.x;
    if (e >= P.edge_count) {
        return;
    }

    let s0 = mid[e * 4u];
    let s1 = mid[e * 4u + 1u];
    let s2 = mid[e * 4u + 2u];
    let s3 = mid[e * 4u + 3u];

    let p0 = fetch(s0);
    let p1 = fetch(s1);
    var pos : vec3<f32>;
    if (s3 != 0xFFFFFFFFu) {
        pos = (p0 + p1) * (3.0 / 8.0) + (fetch(s2) + fetch(s3)) * (1.0 / 8.0);
    } else {
        pos = (p0 + p1) * 0.5;
    }

    let o = (P.vcoarse + e) * 3u;
    dst_pos[o] = pos.x;
    dst_pos[o + 1u] = pos.y;
    dst_pos[o + 2u] = pos.z;
}
