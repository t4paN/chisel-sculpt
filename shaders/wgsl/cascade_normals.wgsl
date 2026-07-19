// cascade_normals.wgsl — WGSL sibling of shaders/glsl/cascade_normals.comp.
// Full-mesh vertex normals after a GPU cascade: the sequential-id variant of
// compute_normals.wgsl (no dirty list — every vertex is "dirty" after a level
// switch). Bindings: 42 final cascade positions (read), 1 normals (read_write),
// 2 indices, 4 adj_offset, 5 adj_list, 63 params.

struct Params {
    vcount : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,   // struct rounds to 16
};

@group(0) @binding(42) var<storage, read>       positions  : array<f32>;
@group(0) @binding(1)  var<storage, read_write> normals    : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vcount) {
        return;
    }

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

    let o = v * 3u;
    normals[o] = n.x;
    normals[o + 1u] = n.y;
    normals[o + 2u] = n.z;
}
