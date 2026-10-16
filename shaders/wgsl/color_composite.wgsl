// color_composite.wgsl
// Paint layers -> the display colour buffer. Lockstep with shaders/glsl/color_composite.comp;
// the CPU reference is paint_blend() in src/paint_layers.cpp — all three must agree.
//
// Starts from opaque white (unpainted clay) and applies every visible layer bottom to
// top. Layer values are PREMULTIPLIED RGBA8, alpha = coverage. Each blend mode is one
// B(dst, src) formula on straight colour; the result is mix(dst, B, alpha * opacity).
// New mode = one more case in blend_channel(), here, in the .comp and on the CPU.
//
// Where each layer's values live (the table's .w >> 8):
//   0 PACKED  in the pool: ids at .x (ascending), values at .y, .z entries; binary search
//   1 DENSE   in the pool: values at .y, indexed by vertex
//   2 ACTIVE  the buffer the brush writes (binding 28), indexed by vertex
// Bit 16 of .w = CLIP (clipping mask): the layer's coverage is scaled by the
// coverage of the nearest unclipped layer before it in the table (its base). Only
// layers the composite actually draws are in the table — the CPU drops hidden layers
// and clipped layers whose base is hidden, and clears CLIP when there is no base.
//
// list_mode 0 runs one thread per vertex (a full rebuild). list_mode 1 runs one thread
// per entry of the dab's dirty list, dispatched indirectly off dirty_args (256 per
// workgroup), so a dab re-blends exactly the verts it painted.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   6 dirty arena (read)   28 active layer (read)   51 display colour (read_write)
//   52 layer pool (read)   61 dirty region UBO      63 params UBO

struct Params {
    vertex_count : u32,
    layer_count  : u32,
    list_mode    : u32,
    _pad0        : u32,
    opacity      : array<vec4<f32>, 4>,    // layer i -> opacity[i / 4][i % 4]
    layers       : array<vec4<u32>, 16>,   // x ids off, y values off, z count, w mode | source << 8 | clip << 16
};

struct DirtyRegion {
    base : u32,
    cap  : u32,
    _p0  : u32,
    _p1  : u32,
};

@group(0) @binding(6)  var<storage, read>       list    : array<u32>;
@group(0) @binding(28) var<storage, read>       active_layer : array<u32>;
@group(0) @binding(51) var<storage, read_write> display : array<u32>;
@group(0) @binding(52) var<storage, read>       pool    : array<u32>;
@group(0) @binding(61) var<uniform>             DR      : DirtyRegion;
@group(0) @binding(63) var<uniform>             P       : Params;

const SRC_PACKED : u32 = 0u;
const SRC_DENSE  : u32 = 1u;

const MODE_MULTIPLY : u32 = 1u;

fn blend_channel(d : vec3<f32>, s : vec3<f32>, mode : u32) -> vec3<f32> {
    if (mode == MODE_MULTIPLY) {
        return d * s;
    }
    return s;   // MODE_NORMAL (0) and anything unknown
}

// Lower-bound search over a packed layer's ids; 0 = transparent where it has nothing.
fn sample_packed(ids_off : u32, vals_off : u32, count : u32, v : u32) -> u32 {
    var lo = 0u;
    var hi = count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        let below = pool[ids_off + mid] < v;
        if (below) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let inside = lo < count;
    if (inside) {
        if (pool[ids_off + lo] == v) {
            return pool[vals_off + lo];
        }
    }
    return 0u;
}

fn sample_layer(e : vec4<u32>, v : u32) -> u32 {
    let src = (e.w >> 8u) & 0xFFu;   // mask: bit 16 is the clip flag
    if (src == SRC_PACKED) {
        return sample_packed(e.x, e.y, e.z, v);
    }
    if (src == SRC_DENSE) {
        return pool[e.y + v];
    }
    return active_layer[v];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    var v = gid.x;
    if (P.list_mode == 1u) {
        // Clamp: the region counter runs past cap when a dab overflowed, and the ids
        // past cap were never written (see CONVENTIONS.md, the dirty list).
        let n = min(list[DR.base], DR.cap);
        if (gid.x >= n) {
            return;
        }
        v = list[DR.base + 1u + gid.x];
    }
    if (v >= P.vertex_count) {
        return;
    }

    var d = vec3<f32>(1.0, 1.0, 1.0);
    var base_a = 1.0;
    for (var i = 0u; i < P.layer_count; i = i + 1u) {
        let e = P.layers[i];
        let c = unpack4x8unorm(sample_layer(e, v));
        let clipped = ((e.w >> 16u) & 1u) == 1u;
        var op = P.opacity[i / 4u][i % 4u];
        if (clipped) {
            op = op * base_a;
        } else {
            base_a = c.a;    // set even when transparent: an empty base clips to nothing
        }
        if (c.a <= 0.0) {
            continue;
        }
        let ao = c.a * op;
        let s = c.rgb / c.a;                      // un-premultiply
        d = mix(d, blend_channel(d, s, e.w & 0xFFu), ao);
        // Round to 8 bits after every layer, as the CPU composite does (it stores
        // RGBA8 between layers). Without this the two drift apart by a level or two
        // per layer and a CPU-composited undo would visibly disagree with the stroke.
        d = unpack4x8unorm(pack4x8unorm(vec4<f32>(d, 1.0))).rgb;
    }
    display[v] = pack4x8unorm(vec4<f32>(d, 1.0));
}
