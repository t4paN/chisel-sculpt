// block_select.wgsl — lockstep with shaders/glsl/block_select.comp.
// Marks every block whose AABB a dab can reach, and appends the newly marked ones to
// the compact dispatch list the brush kernels run over.
//
// STICKY WITHIN A FRAME, not within a stroke. Boxes are rebuilt at the top of each
// frame, so they are exact for every vertex that has not moved since; the only
// vertices that have moved are ones an earlier dab THIS frame displaced, and their
// blocks are already in the list. So the union of a frame's dabs is conservative and
// correct, and the list is reset next frame instead of growing for a whole stroke —
// which is what made the earlier attempt's culling decay as a stroke went on.
//
// Both lobes are tested at once: the geometric mirror's second anchor writes its own
// vertices, so its blocks must be dispatched too.
//
// Bindings: 45 block list (read_write)  47 boxes (read)  48 sticky (read_write)  63 params

struct Params {
    anchor_a  : vec3<f32>,
    radius_a  : f32,
    anchor_b  : vec3<f32>,
    radius_b  : f32,   // <= 0 disables the second lobe
    block_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(45) var<storage, read_write> blocks : array<atomic<u32>>;
@group(0) @binding(47) var<storage, read>       boxes  : array<f32>;
@group(0) @binding(48) var<storage, read_write> sticky : array<atomic<u32>>;
@group(0) @binding(63) var<uniform>             P      : Params;

fn hits(lo : vec3<f32>, hi : vec3<f32>, c : vec3<f32>, r : f32) -> bool {
    if (r <= 0.0) { return false; }
    // Squared distance from the sphere centre to the closest point on the box.
    let d = max(lo - c, max(vec3<f32>(0.0), c - hi));
    return dot(d, d) < r * r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let b = gid.x;
    if (b >= P.block_count) {
        return;
    }
    if (atomicLoad(&sticky[b]) != 0u) {
        return;                     // already in the list this frame
    }
    let o = b * 6u;
    let lo = vec3<f32>(boxes[o + 0u], boxes[o + 1u], boxes[o + 2u]);
    let hi = vec3<f32>(boxes[o + 3u], boxes[o + 4u], boxes[o + 5u]);
    if (!hits(lo, hi, P.anchor_a, P.radius_a) && !hits(lo, hi, P.anchor_b, P.radius_b)) {
        return;
    }
    // Claim it exactly once — two dabs in the same frame must not list it twice.
    let was = atomicExchange(&sticky[b], 1u);
    if (was != 0u) {
        return;
    }
    let slot = atomicAdd(&blocks[0], 1u);
    atomicStore(&blocks[1u + slot], b);
}
