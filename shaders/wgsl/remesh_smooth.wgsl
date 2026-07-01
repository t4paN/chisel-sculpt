// remesh_smooth.wgsl
// Port of src/compute_remesh.cpp (remesh_smooth_src). Remesh tangential Laplacian (GPU
// ping-pong): each in-region, non-pinned, weighted vertex moves toward its 1-ring
// centroid with the normal component stripped (tangential only — a full Laplacian
// shrinks the form). Seam x is clamped so non-pinned verts can't drift into / across the
// seam band. Pinned / out-of-region / weightless / degenerate-normal verts pass through.
// See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 out_pos (read_write)  1 normals (read)  2 indices (read)  4 adj_csr (read)
//   13 in_pos (read)  14 weights (read)  15 pinned (read)  16 tri_sel (read)
//   63 params UBO
// 8 storage buffers. adj_csr is a CSR-concatenated copy of the shared adjacency
// offset+list buffers (offsets in [0, vertex_count], list appended starting at
// element vertex_count+1 — see remesh_adj_csr_ssbo in compute.h) built by the C++
// side each dispatch; this merge is what keeps the kernel under the 8/stage web
// baseline (it was 9 with adj_off/adj_list as separate bindings).

struct Params {
    lambda       : f32,
    seam_tol     : f32,
    vertex_count : u32,
    _pad0        : u32,
};

@group(0) @binding(0)  var<storage, read_write> out_pos  : array<f32>;
@group(0) @binding(1)  var<storage, read>       normals  : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices  : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_csr  : array<u32>;
@group(0) @binding(13) var<storage, read>       in_pos   : array<f32>;
@group(0) @binding(14) var<storage, read>       weights  : array<f32>;
@group(0) @binding(15) var<storage, read>       pinned   : array<u32>;
@group(0) @binding(16) var<storage, read>       tri_sel  : array<u32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    out_pos[v*3u+0u] = in_pos[v*3u+0u];
    out_pos[v*3u+1u] = in_pos[v*3u+1u];
    out_pos[v*3u+2u] = in_pos[v*3u+2u];

    if (pinned[v] != 0u) { return; }

    let w = weights[v];
    if (w < 1e-6) { return; }

    // adj_csr[0 .. vertex_count] is the offset array; the list starts right after.
    let list_base = P.vertex_count + 1u;
    let t_start = adj_csr[v];
    let t_end   = adj_csr[v + 1u];

    var in_region = false;
    for (var j = t_start; j < t_end; j = j + 1u) {
        if (tri_sel[adj_csr[list_base + j]] != 0u) { in_region = true; break; }
    }
    if (!in_region) { return; }

    var nbrs : array<u32, 48>;
    var nbr_count = 0u;
    for (var j = t_start; j < t_end; j = j + 1u) {
        let t = adj_csr[list_base + j];
        for (var k = 0u; k < 3u; k = k + 1u) {
            let n = indices[t*3u + k];
            if (n == v) { continue; }
            var found = false;
            for (var l = 0u; l < nbr_count; l = l + 1u) {
                if (nbrs[l] == n) { found = true; break; }
            }
            if (!found && nbr_count < 48u) {
                nbrs[nbr_count] = n;
                nbr_count = nbr_count + 1u;
            }
        }
    }
    if (nbr_count == 0u) { return; }

    var nbr_centroid = vec3<f32>(0.0);
    for (var l = 0u; l < nbr_count; l = l + 1u) {
        nbr_centroid.x = nbr_centroid.x + in_pos[nbrs[l]*3u+0u];
        nbr_centroid.y = nbr_centroid.y + in_pos[nbrs[l]*3u+1u];
        nbr_centroid.z = nbr_centroid.z + in_pos[nbrs[l]*3u+2u];
    }
    nbr_centroid = nbr_centroid / f32(nbr_count);

    let pos = vec3<f32>(in_pos[v*3u+0u], in_pos[v*3u+1u], in_pos[v*3u+2u]);
    // A degenerate (zero-length) vertex normal has no tangent plane to project onto.
    // normalize() of a zero vector is 0/0 -> NaN, which poisons the post-remesh mirror
    // rebuild. Bail out leaving the vertex put when that happens.
    let nrm_raw = vec3<f32>(normals[v*3u+0u], normals[v*3u+1u], normals[v*3u+2u]);
    let nrm_len = length(nrm_raw);
    if (nrm_len < 1e-12) {
        out_pos[v*3u+0u] = pos.x;
        out_pos[v*3u+1u] = pos.y;
        out_pos[v*3u+2u] = pos.z;
        return;
    }
    let nrm = nrm_raw / nrm_len;
    let d   = nbr_centroid - pos;
    let td  = d - nrm * dot(d, nrm);

    var new_x = pos.x + td.x * (P.lambda * w);
    // Non-pinned verts are outside the seam band by construction. Keep them there:
    // don't allow smoothing to drift |x| into the band or flip sides.
    new_x = sign(pos.x) * max(P.seam_tol, abs(new_x));

    out_pos[v*3u+0u] = new_x;
    out_pos[v*3u+1u] = pos.y + td.y * (P.lambda * w);
    out_pos[v*3u+2u] = pos.z + td.z * (P.lambda * w);
}
