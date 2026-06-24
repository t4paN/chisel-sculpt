// draw_accum_symmetrize.wgsl
// Port of src/compute_draw.cpp (draw_accum_symmetrize_src). For each vertex v with
// mirror twin mv != v, fold the X-mirrored accum[mv] into v's accum and write to a
// separate output buffer, so out[mv] is strictly the X-mirror of out[v]. Self-paired/
// seam verts (mv==v) and out-of-range partners copy through unchanged. Reading and
// writing different buffers avoids the read-after-write race an in-place version has.
// Lockstep with draw_accum_symmetrize.comp. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum:
//   3 accum_in (read)   20 accum_out (write)   7 mirror_map (read)   63 params UBO

struct Params {
    vertex_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,   // struct rounds to 16
};

@group(0) @binding(3)  var<storage, read>       accum_in   : array<u32>;
@group(0) @binding(20) var<storage, read_write> accum_out  : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let base_v = v * 4u;
    let vx = bitcast<f32>(accum_in[base_v + 0u]);
    let vy = bitcast<f32>(accum_in[base_v + 1u]);
    let vz = bitcast<f32>(accum_in[base_v + 2u]);
    let vw = bitcast<f32>(accum_in[base_v + 3u]);

    let mv = mirror_map[v];
    if (mv == v || mv >= P.vertex_count) {
        accum_out[base_v + 0u] = bitcast<u32>(vx);
        accum_out[base_v + 1u] = bitcast<u32>(vy);
        accum_out[base_v + 2u] = bitcast<u32>(vz);
        accum_out[base_v + 3u] = bitcast<u32>(vw);
        return;
    }

    let base_m = mv * 4u;
    let mx = bitcast<f32>(accum_in[base_m + 0u]);
    let my = bitcast<f32>(accum_in[base_m + 1u]);
    let mz = bitcast<f32>(accum_in[base_m + 2u]);
    let mw = bitcast<f32>(accum_in[base_m + 3u]);

    accum_out[base_v + 0u] = bitcast<u32>(vx + (-mx));
    accum_out[base_v + 1u] = bitcast<u32>(vy +  my);
    accum_out[base_v + 2u] = bitcast<u32>(vz +  mz);
    accum_out[base_v + 3u] = bitcast<u32>(vw +  mw);
}
