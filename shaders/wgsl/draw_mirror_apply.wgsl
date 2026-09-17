// draw_mirror_apply.wgsl
// Port of src/compute_draw.cpp (draw_mirror_apply_src). Writes negated-X
// displacements to mirror-twin vertices: for each vertex v the brush touched
// (accum weight > 0), apply the X-mirror of v's averaged displacement to its twin
// mv (scaled by 1 - mask[mv]). Lockstep with draw_mirror_apply.comp. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum:
//   0 positions (read_write)   3 accum (read)   7 mirror_map (read)
//   12 mask (read)             63 params UBO

struct Params {
    vertex_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,   // struct rounds to 16
};

struct DirtyRegion {
    base       : u32,
    cap        : u32,
    block_mode : u32,   // 1 = dispatch is one workgroup per active block
    _p1        : u32,
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(3)  var<storage, read>       accum      : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
@group(0) @binding(12) var<storage, read>       mask       : array<f32>;
@group(0) @binding(45) var<storage, read>       block_list : array<u32>;
@group(0) @binding(61) var<uniform>             DR         : DirtyRegion;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    // Culled by the block of the vertex it READS (v, which carries the accum weight),
    // not the twin it writes — the write is a direct index, so the twin's block need
    // not be dispatched. Safe because mirror_pairs implies mirror_x (see brush.h), so
    // the dab's selection always carries the reflected lobe and the twin's block was
    // cleared this dab too.
    let v = select(wg.x * 64u + lid.x,
                   block_list[1u + wg.x] * 64u + lid.x,
                   DR.block_mode == 1u);
    if (v >= P.vertex_count) {
        return;
    }

    let base = v * 4u;
    let w = bitcast<f32>(accum[base + 3u]);
    if (w <= 0.0) {
        return;
    }

    let mv = mirror_map[v];
    if (mv == v || mv >= P.vertex_count) {   // seam / MIRROR_UNPAIRED sentinel
        return;
    }

    let scale = 1.0 - mask[mv];
    if (scale <= 0.0) {
        return;
    }

    let inv_w = 1.0 / w;
    let dx = bitcast<f32>(accum[base + 0u]) * inv_w * scale;
    let dy = bitcast<f32>(accum[base + 1u]) * inv_w * scale;
    let dz = bitcast<f32>(accum[base + 2u]) * inv_w * scale;

    let pidx = mv * 3u;
    positions[pidx + 0u] = positions[pidx + 0u] + (-dx);
    positions[pidx + 1u] = positions[pidx + 1u] +  dy;
    positions[pidx + 2u] = positions[pidx + 2u] +  dz;
}
