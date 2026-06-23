// compute_normals.wgsl
// Port of src/compute_smooth.cpp (compute_normals_src). After a deforming brush
// moves positions, recompute the vertex normals for the touched vertices so the
// matcap shading tracks the new surface. One thread per dirty vertex: sum the
// (area-weighted) face normals of its adjacent triangles via CSR adjacency, then
// normalize. REFERENCE TRANSLATION — see CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0  positions (read)   1 normals (read_write)   2 indices (read)
//   4  adj_offset (read)  5 adj_list (read)         6 dirty verts (read)
//   63 params UBO (BIND_PARAMS)
//
// binding 6 here is the plain id list (no count header) — the caller passes the
// count in the params UBO, exactly like the GL u_dirty_count uniform.

struct Params {
    dirty_count : u32,   // byte 0
    _pad0       : u32,
    _pad1       : u32,
    _pad2       : u32,   // struct rounds to 16
};

@group(0) @binding(0)  var<storage, read>       positions   : array<f32>;
@group(0) @binding(1)  var<storage, read_write> normals     : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices     : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list    : array<u32>;
@group(0) @binding(6)  var<storage, read>       dirty_verts : array<u32>;
@group(0) @binding(63) var<uniform>             P           : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let di = gid.x;
    if (di >= P.dirty_count) {
        return;
    }

    let v = dirty_verts[di];

    var n = vec3<f32>(0.0, 0.0, 0.0);

    let start = adj_offset[v];
    let end = adj_offset[v + 1u];

    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];

        let p0 = vec3<f32>(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
        let p1 = vec3<f32>(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]);
        let p2 = vec3<f32>(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]);

        // un-normalized cross = 2*area*unit_normal → summing area-weights for free
        let fn_ = cross(p1 - p0, p2 - p0);
        if (length(fn_) < 1e-7) {
            continue;
        }
        n = n + fn_;
    }

    let len = length(n);
    if (len > 1e-8) {
        n = n / len;
    } else {
        n = vec3<f32>(0.0, 0.0, 0.0);
    }

    let base = v * 3u;
    normals[base] = n.x;
    normals[base + 1u] = n.y;
    normals[base + 2u] = n.z;
}
