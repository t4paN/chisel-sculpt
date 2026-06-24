// find_pinned.wgsl
// Port of src/compute_remesh.cpp (find_pinned_src). Remesh per-vertex pinned-boundary
// detection: a vert is pinned (immovable during the tangential smooth) if it sits on
// the mirror seam (|x| < seam_tol) or on the boundary of the selected region (has both
// selected and unselected adjacent tris). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   13 in pos (read)   4 adj_off (read)   5 adj_list (read)
//   16 tri_sel (read)  15 pinned (read_write)   63 params UBO

struct Params {
    vertex_count : u32,
    seam_tol     : f32,
    _pad0        : u32,
    _pad1        : u32,
};

@group(0) @binding(13) var<storage, read>       pos      : array<f32>;
@group(0) @binding(4)  var<storage, read>       adj_off  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list : array<u32>;
@group(0) @binding(16) var<storage, read>       tri_sel  : array<u32>;
@group(0) @binding(15) var<storage, read_write> pinned   : array<u32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    if (abs(pos[v*3u]) < P.seam_tol) {
        pinned[v] = 1u;
        return;
    }

    let t_start = adj_off[v];
    let t_end   = adj_off[v + 1u];
    var has_sel = false;
    var has_unsel = false;
    for (var j = t_start; j < t_end; j = j + 1u) {
        if (tri_sel[adj_list[j]] != 0u) { has_sel = true; }
        else                            { has_unsel = true; }
        if (has_sel && has_unsel) { pinned[v] = 1u; return; }
    }
    pinned[v] = 0u;
}
