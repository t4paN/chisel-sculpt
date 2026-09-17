// accum_clear_blocks.wgsl — lockstep with shaders/glsl/accum_clear_blocks.comp.
// Zeroes the accum buffer for the ACTIVE blocks only.
//
// The whole accum buffer used to be cleared once per dab — 40 MB at 5M tris, for a
// dab that writes a few hundred KB of it. With dispatch culling that clear became the
// dominant per-dab cost by a wide margin (a 3155-dab stroke was moving ~126 GB of
// zeroes). Only the blocks this dab is about to write need clearing: apply reads back
// exactly the blocks accum wrote, so stale accum in an unselected block is never seen.
struct Params { vertex_count : u32, _p0 : u32, _p1 : u32, _p2 : u32 };

@group(0) @binding(3)  var<storage, read_write> accum      : array<u32>;
@group(0) @binding(45) var<storage, read>       block_list : array<u32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    let v = block_list[1u + wg.x] * 64u + lid.x;
    if (v >= P.vertex_count) {
        return;
    }
    let b = v * 4u;
    accum[b + 0u] = 0u;
    accum[b + 1u] = 0u;
    accum[b + 2u] = 0u;
    accum[b + 3u] = 0u;
}
