// seam_snap.wgsl
// Port of src/compute_remesh.cpp (seam_snap_src). Post-remesh seam snap: pull near-x=0
// verts onto the mirror plane. Verts already within seam_tol snap unconditionally; verts
// within snap_tol snap only if a topological test finds neighbours on both +x and -x
// sides (a true straddling seam vert). Fully-masked verts are skipped. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   13 pos (read_write)  4 adj_off (read)  5 adj_list (read)
//   2 indices (read)     12 mask (read)    63 params UBO

struct Params {
    vertex_count : u32,
    mask_size    : u32,
    seam_tol     : f32,
    snap_tol     : f32,
};

@group(0) @binding(13) var<storage, read_write> pos      : array<f32>;
@group(0) @binding(4)  var<storage, read>       adj_off  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list : array<u32>;
@group(0) @binding(2)  var<storage, read>       indices  : array<u32>;
@group(0) @binding(12) var<storage, read>       mask     : array<f32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    // Skip fully masked
    if (P.mask_size > 0u && v < P.mask_size && mask[v] >= 1.0) {
        return;
    }

    let ax = abs(pos[v * 3u]);

    // Tight snap: already within seam_tol
    if (ax < P.seam_tol) {
        pos[v * 3u] = 0.0;
        return;
    }

    // Outside snap zone
    if (ax >= P.snap_tol) {
        return;
    }

    // Topological test: neighbors on both +x and -x sides
    var has_pos = false;
    var has_neg = false;
    let t_start = adj_off[v];
    let t_end   = adj_off[v + 1u];
    for (var j = t_start; j < t_end && !(has_pos && has_neg); j = j + 1u) {
        let tri = adj_list[j];
        for (var k = 0u; k < 3u; k = k + 1u) {
            let nv = indices[tri * 3u + k];
            if (nv == v) { continue; }
            let nx = pos[nv * 3u];
            if (nx > P.seam_tol) { has_pos = true; }
            else if (nx < -P.seam_tol) { has_neg = true; }
        }
    }

    if (has_pos && has_neg) {
        pos[v * 3u] = 0.0;
    }
}
