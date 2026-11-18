// block_args.wgsl — lockstep with shaders/glsl/block_args.comp.
// Turns the active-block count into an indirect dispatch triple. One workgroup per
// active block, so the count IS the workgroup count (the brush kernels use a
// workgroup size of 64, which is exactly one block). Past the device's per-dimension
// limit (65535 on the web: a brush covering >4.2M verts) the count folds into rows;
// the kernels rebuild the linear id and stop at the list's count.
struct Params { block_count : u32, max_groups : u32, _p1 : u32, _p2 : u32 };

@group(0) @binding(45) var<storage, read>       blocks : array<u32>;
@group(0) @binding(46) var<storage, read_write> args   : array<u32>;
@group(0) @binding(63) var<uniform>             P      : Params;

@compute @workgroup_size(1)
fn main() {
    var n = blocks[0];
    if (n > P.block_count) { n = P.block_count; }
    var mg = P.max_groups;
    if (mg == 0u) { mg = 65535u; }
    if (n <= mg) {
        args[0] = n;
        args[1] = 1u;
    } else {
        args[0] = mg;
        args[1] = (n + mg - 1u) / mg;
    }
    args[2] = 1u;
}
