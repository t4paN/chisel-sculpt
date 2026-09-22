// compute_normals.wgsl
// Port of src/compute_smooth.cpp (compute_normals_src). After a deforming brush
// moves positions, recompute the vertex normals for the touched vertices so the
// matcap shading tracks the new surface. One thread per dirty vertex: sum the
// (Max-weighted) face normals of its adjacent triangles via CSR adjacency, then
// normalize. REFERENCE TRANSLATION — see CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0  positions (read)   1 normals (read_write)   2 indices (read)
//   4  adj_offset (read)  5 adj_list (read)         6 dirty verts (read)
//   63 params UBO (BIND_PARAMS)
//
// binding 6 is either the plain id list with the count in the params UBO (list_mode
// 0, CPU-built lists), or a {count, ids[]} list whose count only exists on the GPU
// (list_mode 1, normals_expand's output — dispatched indirect, count clamped to cap).

struct Params {
    dirty_count : u32,   // byte 0 — list_mode 0 only
    list_mode   : u32,   // 0 = plain ids, 1 = {count, ids[]} header
    header_cap  : u32,   // list_mode 1: id capacity of the list
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
    var count = P.dirty_count;
    var first = 0u;
    if (P.list_mode == 1u) {
        count = min(dirty_verts[0], P.header_cap);
        first = 1u;
    }
    if (di >= count) {
        return;
    }

    let v = dirty_verts[first + di];

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

        // cross(next-k, prev-k) is the SAME vector at all three corners, so one face
        // normal serves the triangle; only the weight depends on which corner v is.
        let fn_ = cross(p1 - p0, p2 - p0);
        if (length(fn_) < 1e-7) {
            continue;
        }

        // Nelson Max weighting: w = 1/(|a|^2 * |b|^2) on the two edges leaving v.
        // Plain area weighting biases a vertex normal toward its largest neighbour,
        // which on an anisotropic mesh (a UV sphere near the poles) tilts the normal
        // off-radial by degrees. MUST match max_corner_weight() in src/mesh.cpp — a
        // GPU-touched vertex and a CPU-touched one sit side by side on the same
        // surface. Plain if/else, not select(): Tint reads an inline `<` in select()
        // as a template-list opener (see CONVENTIONS.md).
        var ea = vec3<f32>(0.0, 0.0, 0.0);
        var eb = vec3<f32>(0.0, 0.0, 0.0);
        if (v == i0) {
            ea = p1 - p0; eb = p2 - p0;
        } else if (v == i1) {
            ea = p2 - p1; eb = p0 - p1;
        } else {
            ea = p0 - p2; eb = p1 - p2;
        }
        let denom = dot(ea, ea) * dot(eb, eb);
        if (denom <= 0.0) {
            continue;
        }
        n = n + fn_ * (1.0 / denom);
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
