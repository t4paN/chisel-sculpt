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
// Degenerate means truly zero in f32 (1e-30), never "small": the sums carry length^2
// units, so a fixed 1e-8 zeroed the normal of every fan finer than ~1e-4 — the valence-4
// verts of the startup ball at L8, crease-pinched slivers — which then drew black and gave
// brushes no direction (2026-11-23). With no area direction, the Max normal stands alone.
fn guard_normal(max_sum : vec3<f32>, area_sum : vec3<f32>) -> vec3<f32> {
    var na = vec3<f32>(0.0, 0.0, 0.0);
    let la = length(area_sum);
    if (la > 1e-30) {
        na = area_sum / la;
    }
    let lm = length(max_sum);
    if (lm <= 1e-30) {
        return na;
    }
    let nm = max_sum / lm;
    if (la <= 1e-30) {
        return nm;
    }
    let d = dot(nm, na);
    let t = smoothstep(0.81915204, 0.96592583, d);
    let b = na * (1.0 - t) + nm * t;
    let lb = length(b);
    if (lb > 1e-30) {
        return b / lb;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups)        nwg : vec3<u32>) {
    let v = gid.x + gid.y * nwg.x * 256u;   // linear id: gpu::dispatch_1d may fold the count into rows
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
        // No size cut-off here (was length < 1e-7): |cross| is 2x area, so a fixed floor
        // drops every face of a fine fan -> zero normal -> black streaks at ~60M tris
        // (startup-ball pole slivers, pinched creases). A flat face adds zero anyway and
        // the denom check below guards the divide. Matches Mesh::recompute_normals.
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
