// select_stretched.wgsl
// Port of src/compute_remesh.cpp (select_stretched_src). Remesh per-tri selection:
// flags triangles that are over-long, high edge-ratio, or sliver (smallest angle
// below ~15°) so the remesher targets them. Writes 0/1 into the per-tri selection
// buffer. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   13 in pos (read)   2 indices (read)   16 tri_sel (read_write)   63 params UBO

struct Params {
    target_len : f32,
    tri_count  : u32,
    _pad0      : u32,
    _pad1      : u32,
};

@group(0) @binding(13) var<storage, read>       pos     : array<f32>;
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

    let p0 = vec3<f32>(pos[i0*3u], pos[i0*3u+1u], pos[i0*3u+2u]);
    let p1 = vec3<f32>(pos[i1*3u], pos[i1*3u+1u], pos[i1*3u+2u]);
    let p2 = vec3<f32>(pos[i2*3u], pos[i2*3u+1u], pos[i2*3u+2u]);

    let e01 = length(p1 - p0);
    let e12 = length(p2 - p1);
    let e20 = length(p0 - p2);

    let longest  = max(max(e01, e12), e20);
    let shortest = min(min(e01, e12), e20);

    // Sliver test: smallest angle below ~15° (cos > 0.966). Law of cosines on all
    // three corners; clamp guards against float drift past ±1.
    var sliver = false;
    if (shortest > 1e-8) {
        let e01_2 = e01*e01; let e12_2 = e12*e12; let e20_2 = e20*e20;
        let c0 = clamp((e12_2 + e20_2 - e01_2) / (2.0 * e12 * e20), -1.0, 1.0);
        let c1 = clamp((e01_2 + e20_2 - e12_2) / (2.0 * e01 * e20), -1.0, 1.0);
        let c2 = clamp((e01_2 + e12_2 - e20_2) / (2.0 * e01 * e12), -1.0, 1.0);
        let max_cos = max(max(c0, c1), c2);
        sliver = max_cos > 0.966;  // cos(15°)
    }

    let sel = longest > 1.2 * P.target_len ||
              (shortest > 1e-8 && longest / shortest > 1.5) ||
              sliver;
    tri_sel[t] = select(0u, 1u, sel);
}
