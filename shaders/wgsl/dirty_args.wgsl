// dirty_args.wgsl
// Writes the workgroup counts for an indirect dispatch over a dab's dirty list.
// Lockstep with shaders/glsl/dirty_args.comp.
//
// The dab's touched-vertex count only exists on the GPU. Without this, every kernel
// that consumes the list had to dispatch worst-case threads over the WHOLE mesh and
// let the overshoot early-out — the count could not reach the CPU in time to size a
// dispatch, and reading it back mid-stroke is forbidden. One thread turns that count
// into a (ceil(n / 256), 1, 1) triple the command processor reads directly.
//
// Bindings mirror the ComputeBinding enum:
//   6 dirty arena (read)   46 indirect args (read_write)   61 dirty region UBO

struct DirtyRegion {
    base : u32,
    cap  : u32,
    _p0  : u32,
    _p1  : u32,
};

@group(0) @binding(6)  var<storage, read>       list : array<u32>;
@group(0) @binding(46) var<storage, read_write> args : array<u32>;
@group(0) @binding(61) var<uniform>             DR   : DirtyRegion;

@compute @workgroup_size(1)
fn main() {
    // Clamp: the region counter deliberately runs past cap when a dab overflowed,
    // and the ids past cap were never written. Dispatching for them would read
    // garbage — the CPU handles that case with a whole-mesh snapshot instead.
    let n = min(list[DR.base], DR.cap);
    args[0] = (n + 255u) / 256u;
    args[1] = 1u;
    args[2] = 1u;
}
