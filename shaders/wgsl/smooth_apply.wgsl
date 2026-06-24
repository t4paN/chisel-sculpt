// smooth_apply.wgsl
// Port of src/compute_smooth.cpp (smooth_apply_src). Pass 2 of the smooth brush, run
// once per iteration: a uniform Laplacian over each touched vertex's 1-ring, with the
// tangential component stripped in the bulk (kills drift on irregular meshes) but
// faded back inside the mirror seam band (keeps the x=0 seam from pinching). The
// area-weighted vertex normal is accumulated inline from the same 1-ring walk.
// (The original u_iteration uniform was unused and is dropped.) See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   2 indices (read)        3 accum (read)
//   4 adj_offset (read)        5 adj_list (read)        12 mask (read)
//   63 params UBO (BIND_PARAMS)

struct Params {
    vertex_count : u32,   // 0
    strength     : f32,   // 4
    mirror_x     : u32,   // 8
    seam_band    : f32,   // 12
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(3)  var<storage, read>       accum      : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(12) var<storage, read>       mask       : array<f32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
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

    let cur = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    var sum = vec3<f32>(0.0);
    var nrm = vec3<f32>(0.0);   // area-weighted vertex normal, accumulated inline
    var count = 0.0;

    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];

        let p0 = vec3<f32>(positions[i0*3u], positions[i0*3u+1u], positions[i0*3u+2u]);
        let p1 = vec3<f32>(positions[i1*3u], positions[i1*3u+1u], positions[i1*3u+2u]);
        let p2 = vec3<f32>(positions[i2*3u], positions[i2*3u+1u], positions[i2*3u+2u]);

        // un-normalized cross = 2*area*unit_normal → area-weighted vertex normal
        nrm = nrm + cross(p1 - p0, p2 - p0);

        if (v == i0)      { sum = sum + p1 + p2; }
        else if (v == i1) { sum = sum + p0 + p2; }
        else              { sum = sum + p0 + p1; }
        count = count + 2.0;
    }

    if (count <= 0.0) {
        return;
    }

    var move = sum * (1.0 / count) - cur;

    let nlen = length(nrm);
    if (nlen > 1e-12) {
        let n = nrm / nlen;
        let move_n = dot(move, n) * n;
        var t = 1.0;
        if (P.mirror_x != 0u) {
            t = smoothstep(0.0, P.seam_band, abs(cur.x));
        }
        move = mix(move, move_n, t);
    }

    let blend = w * P.strength * mscale;
    positions[v*3u]      = positions[v*3u]      + move.x * blend;
    positions[v*3u + 1u] = positions[v*3u + 1u] + move.y * blend;
    positions[v*3u + 2u] = positions[v*3u + 2u] + move.z * blend;
}
