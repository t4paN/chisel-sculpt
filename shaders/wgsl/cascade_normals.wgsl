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

// Sliver guard: MUST match guard_normal() in src/mesh.cpp and the other normals
// shaders. Max weighting lets one crushed, wrong-facing sliver (a fold where verts
// crossed over) outvote a whole ring; where the Max and area normals disagree by more
// than 15 deg, blend to the area normal (fully by 35 deg). Clean meshes never trip it.
fn guard_normal(max_sum : vec3<f32>, area_sum : vec3<f32>) -> vec3<f32> {
    var na = vec3<f32>(0.0, 0.0, 0.0);
    let la = length(area_sum);
    if (la > 1e-8) {
        na = area_sum / la;
    }
    let lm = length(max_sum);
    if (lm <= 1e-8) {
        return na;
    }
    let nm = max_sum / lm;
    let d = dot(nm, na);
    let t = smoothstep(0.81915204, 0.96592583, d);
    let b = na * (1.0 - t) + nm * t;
    let lb = length(b);
    if (lb > 1e-8) {
        return b / lb;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vcount) {
        return;
    }

    var n = vec3<f32>(0.0, 0.0, 0.0);
    var a = vec3<f32>(0.0, 0.0, 0.0);

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

        let fn_ = cross(p1 - p0, p2 - p0);
        if (length(fn_) < 1e-7) {
            continue;
        }
        // Same Max weighting + guard as compute_normals.wgsl. This shader stayed on
        // plain area weighting when v0.2.10 moved the other sites to Max, so a level
        // switch shaded differently from a stroke on the same surface.
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
        a = a + fn_;
    }

    n = guard_normal(n, a);

    let o = v * 3u;
    normals[o] = n.x;
    normals[o + 1u] = n.y;
    normals[o + 2u] = n.z;
}
