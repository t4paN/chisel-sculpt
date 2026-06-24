// grow_selection.wgsl
// Port of src/compute_remesh.cpp (grow_selection_src). Remesh per-tri ring-grow (BFS
// by shared vertex, one ring per dispatch): an unselected tri becomes selected if any
// vertex-adjacent tri is in the input snapshot. The host snapshots tri_sel → in_sel
// (binding 18) before each ring; this writes out_sel (binding 16) in place. See
// CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   18 in_sel (read)   16 out_sel (read_write)   2 indices (read)
//   4 adj_off (read)   5 adj_list (read)         63 params UBO

struct Params {
    tri_count : u32,
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
};

@group(0) @binding(18) var<storage, read>       in_sel   : array<u32>;
@group(0) @binding(16) var<storage, read_write> out_sel  : array<u32>;
@group(0) @binding(2)  var<storage, read>       indices  : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_off  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list : array<u32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let t = gid.x;
    if (t >= P.tri_count) {
        return;
    }
    if (in_sel[t] != 0u) {               // already in selection, leave the copy
        return;
    }

    let v0 = indices[t*3u + 0u];
    let v1 = indices[t*3u + 1u];
    let v2 = indices[t*3u + 2u];
    var vs = array<u32, 3>(v0, v1, v2);
    for (var i = 0; i < 3; i = i + 1) {
        let v = vs[i];
        let s = adj_off[v];
        let e = adj_off[v + 1u];
        for (var j = s; j < e; j = j + 1u) {
            if (in_sel[adj_list[j]] != 0u) {
                out_sel[t] = 1u;
                return;
            }
        }
    }
}
