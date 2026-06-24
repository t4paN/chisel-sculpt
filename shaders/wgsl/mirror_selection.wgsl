// mirror_selection.wgsl
// Port of src/compute_remesh.cpp (mirror_selection_src). Remesh per-tri mirror-spread:
// each selected tri ORs the vertex-adjacent tris of its verts' mirror twins into the
// selection, so a one-sided selection becomes symmetric. Reads the snapshot in_sel(18),
// atomic-ORs into out_sel(16). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   18 in_sel (read)   16 out_sel (read_write, atomic)   2 indices (read)
//   4 adj_off (read)   5 adj_list (read)   7 mirror (read)   63 params UBO

struct Params {
    tri_count    : u32,
    vertex_count : u32,
    _pad0        : u32,
    _pad1        : u32,
};

@group(0) @binding(18) var<storage, read>       in_sel   : array<u32>;
@group(0) @binding(16) var<storage, read_write> out_sel  : array<atomic<u32>>;
@group(0) @binding(2)  var<storage, read>       indices  : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_off  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror   : array<u32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let t = gid.x;
    if (t >= P.tri_count) {
        return;
    }
    if (in_sel[t] == 0u) {              // only selected tris seed mirror spread
        return;
    }

    let v0 = indices[t*3u + 0u];
    let v1 = indices[t*3u + 1u];
    let v2 = indices[t*3u + 2u];
    var vs = array<u32, 3>(v0, v1, v2);
    for (var i = 0; i < 3; i = i + 1) {
        let v = vs[i];
        if (v >= P.vertex_count) { continue; }
        let mv = mirror[v];
        if (mv == v || mv >= P.vertex_count) { continue; }
        let s = adj_off[mv];
        let e = adj_off[mv + 1u];
        for (var j = s; j < e; j = j + 1u) {
            atomicOr(&out_sel[adj_list[j]], 1u);
        }
    }
}
