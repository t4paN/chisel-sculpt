// block_boxes.wgsl — lockstep with shaders/glsl/block_boxes.comp.
// Builds one world AABB per VERTEX_BLOCK (64 consecutive vertex indices), read by
// block_select to cull brush dispatches down to the blocks a dab can reach.
//
// Built on the GPU, from the positions the GPU actually holds. The CPU copy is
// pen-down-stale by design (GPU-resident undo defers the writeback), and a box built
// from stale positions is too SMALL — the one error that silently drops vertices out
// of a dab. Building here removes that whole class, and is cheap enough to redo every
// frame, which is what lets selection be exact instead of sticky-for-a-whole-stroke.
//
// One thread per block, looping its own 64 vertices: no workgroup shared memory and
// no barriers, and the total vertex reads are the same as one full-mesh pass.
//
// Bindings:  0 positions (read)   47 boxes (read_write)   63 params

struct Params {
    vertex_count : u32,
    block_count  : u32,
    _pad0        : u32,
    _pad1        : u32,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(47) var<storage, read_write> boxes     : array<f32>;
@group(0) @binding(63) var<uniform>             P         : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let b = gid.x;
    if (b >= P.block_count) {
        return;
    }
    let v0 = b * 64u;
    var v1 = v0 + 64u;
    if (v1 > P.vertex_count) { v1 = P.vertex_count; }

    var lo = vec3<f32>( 1e30,  1e30,  1e30);
    var hi = vec3<f32>(-1e30, -1e30, -1e30);
    for (var v = v0; v < v1; v = v + 1u) {
        let p = vec3<f32>(positions[v * 3u], positions[v * 3u + 1u], positions[v * 3u + 2u]);
        lo = min(lo, p);
        hi = max(hi, p);
    }
    // An empty block (past the end) keeps its inverted box, which intersects nothing.
    let o = b * 6u;
    boxes[o + 0u] = lo.x; boxes[o + 1u] = lo.y; boxes[o + 2u] = lo.z;
    boxes[o + 3u] = hi.x; boxes[o + 4u] = hi.y; boxes[o + 5u] = hi.z;
}
