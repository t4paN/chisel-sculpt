// seam_weld.wgsl
// Port of src/compute_remesh.cpp (seam_weld_src). Post-remesh seam weld: for each vert
// sitting exactly on the seam (x==0), find the lowest-index seam vert within weld_tol in
// the (y,z) plane and record it as the merge target; otherwise the vert maps to itself.
// The host then chases merge chains to roots. Fully-masked verts are skipped. See
// CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   13 pos (read)   12 mask (read)   19 merge_map (read_write)   63 params UBO

struct Params {
    vertex_count : u32,
    mask_size    : u32,
    weld_tol     : f32,
    _pad0        : u32,
};

@group(0) @binding(13) var<storage, read>       pos       : array<f32>;
@group(0) @binding(12) var<storage, read>       mask      : array<f32>;
@group(0) @binding(19) var<storage, read_write> merge_map : array<u32>;
@group(0) @binding(63) var<uniform>             P         : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    merge_map[v] = v;

    // Only process verts sitting on seam (x == 0)
    if (pos[v * 3u] != 0.0) {
        return;
    }

    // Skip fully masked
    if (P.mask_size > 0u && v < P.mask_size && mask[v] >= 1.0) {
        return;
    }

    let py = pos[v * 3u + 1u];
    let pz = pos[v * 3u + 2u];
    let weld_sq = P.weld_tol * P.weld_tol;

    var best = v;
    for (var u = 0u; u < v; u = u + 1u) {
        if (pos[u * 3u] != 0.0) { continue; }
        if (P.mask_size > 0u && u < P.mask_size && mask[u] >= 1.0) {
            continue;
        }
        let dy = pos[u * 3u + 1u] - py;
        let dz = pos[u * 3u + 2u] - pz;
        let d2 = dy * dy + dz * dz;
        if (d2 < weld_sq && u < best) {
            best = u;
        }
    }
    merge_map[v] = best;
}
