// cascade_frames.wgsl — WGSL sibling of shaders/glsl/cascade_frames.comp.
// GPU twin of compute_frames (multires_stack.cpp): per-vertex tangent frame on
// the pre-displacement subdivided surface, for levels whose CPU frames cache a
// sculpt below invalidated. Bindings: 42 pre-disp positions (read), 1 pre-disp
// normals (read), 2 indices, 4 adj_offset, 5 adj_list, 31 frames out
// (read_write), 63 params.

struct Params {
    vcount : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,   // struct rounds to 16
};

@group(0) @binding(42) var<storage, read>       positions  : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals    : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(31) var<storage, read_write> frames     : array<f32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vcount) {
        return;
    }

    // Mesh normals are ~unit already; re-normalize like the CPU (Vec3::normalized
    // zero-guards at 1e-8).
    let nr = vec3<f32>(normals[v * 3u], normals[v * 3u + 1u], normals[v * 3u + 2u]);
    let nl = length(nr);
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (nl >= 1e-8) {
        n = nr / nl;
    }

    // Lowest-indexed neighbor via CSR for a deterministic tangent.
    var best_nbr = 0xFFFFFFFFu;
    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t * 3u];
        let i1 = indices[t * 3u + 1u];
        let i2 = indices[t * 3u + 2u];
        if (i0 != v && i0 < best_nbr) { best_nbr = i0; }
        if (i1 != v && i1 < best_nbr) { best_nbr = i1; }
        if (i2 != v && i2 < best_nbr) { best_nbr = i2; }
    }

    var t_raw = vec3<f32>(1.0, 0.0, 0.0);
    if (best_nbr != 0xFFFFFFFFu) {
        t_raw = vec3<f32>(positions[best_nbr * 3u]      - positions[v * 3u],
                          positions[best_nbr * 3u + 1u] - positions[v * 3u + 1u],
                          positions[best_nbr * 3u + 2u] - positions[v * 3u + 2u]);
    }

    // Project onto tangent plane.
    let dn = dot(t_raw, n);
    var t_proj = t_raw - n * dn;
    var t_len = length(t_proj);
    if (t_len < 1e-6) {
        // Axis-aligned fallback: world axis most perpendicular to n.
        let ax = abs(n.x);
        let ay = abs(n.y);
        let az = abs(n.z);
        var axis = vec3<f32>(0.0, 0.0, 1.0);
        if (ax <= ay && ax <= az) {
            axis = vec3<f32>(1.0, 0.0, 0.0);
        } else if (ay <= az) {
            axis = vec3<f32>(0.0, 1.0, 0.0);
        }
        let d2 = dot(axis, n);
        t_proj = axis - n * d2;
        t_len = length(t_proj);
    }

    let t_final = t_proj / t_len;
    let b_raw = cross(n, t_final);
    let bl = length(b_raw);
    var b_final = vec3<f32>(0.0, 0.0, 0.0);
    if (bl >= 1e-8) {
        b_final = b_raw / bl;
    }

    let f = v * 9u;
    frames[f]      = t_final.x; frames[f + 1u] = t_final.y; frames[f + 2u] = t_final.z;
    frames[f + 3u] = b_final.x; frames[f + 4u] = b_final.y; frames[f + 5u] = b_final.z;
    frames[f + 6u] = n.x;       frames[f + 7u] = n.y;       frames[f + 8u] = n.z;
}
