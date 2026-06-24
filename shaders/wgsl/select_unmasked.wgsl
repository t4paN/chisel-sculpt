// select_unmasked.wgsl
// Port of src/compute_remesh.cpp (select_unmasked_src). Remesh per-tri selection:
// flags triangles whose three verts are all unmasked (mask < 0.5, or out of mask
// range, or no mask) — those are free to be remeshed. Writes 0/1 into the per-tri
// selection buffer. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   14 mask (read)   2 indices (read)   16 tri_sel (read_write)   63 params UBO

struct Params {
    tri_count : u32,
    mask_size : u32,
    _pad0     : u32,
    _pad1     : u32,
};

@group(0) @binding(14) var<storage, read>       mask    : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices : array<u32>;
@group(0) @binding(16) var<storage, read_write> tri_sel : array<u32>;
@group(0) @binding(63) var<uniform>             P       : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let t = gid.x;
    if (t >= P.tri_count) {
        return;
    }

    let i0 = indices[t*3u+0u];
    let i1 = indices[t*3u+1u];
    let i2 = indices[t*3u+2u];

    let u0 = (P.mask_size == 0u || i0 >= P.mask_size || mask[i0] < 0.5);
    let u1 = (P.mask_size == 0u || i1 >= P.mask_size || mask[i1] < 0.5);
    let u2 = (P.mask_size == 0u || i2 >= P.mask_size || mask[i2] < 0.5);

    tri_sel[t] = select(0u, 1u, u0 && u1 && u2);
}
