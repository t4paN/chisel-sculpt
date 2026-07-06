// mirror_project.wgsl
// Symmetry constraint projection — lockstep with shaders/glsl/mirror_project.comp.
// After a dab (or the pen-up autosmooth) has moved vertices, re-impose exact
// X-mirror symmetry over the touched set instead of trusting every producer
// kernel to be symmetric. Per touched vertex v:
//   seam   (map[v] == v)          → snap pos.x to 0 (y/z free)
//   paired (map[v] == twin)       → canonical side (v < twin) averages v with the
//                                   reflected twin and writes both sides
//   unpaired (sentinel >= count)  → untouched
// Pairs with either side fully masked are skipped. Idempotent, race-free: only
// the list entry whose id is the smaller pair index writes.
// The touched list is either the per-dab dirty buffer {count, ids[]} (list_mode 0)
// or a plain id array with the count in Params (list_mode 1, stroke_smooth path).
//
// Bindings mirror the ComputeBinding enum:
//   0 positions (read_write)   6 touched list (read)   7 mirror_map (read)
//   12 mask (read)             63 params UBO

struct Params {
    vertex_count : u32,   // 0
    list_mode    : u32,   // 4   0 = {count, ids[]} header, 1 = plain ids + list_count
    list_count   : u32,   // 8   entry count for list_mode 1
    _pad0        : u32,   // 12  (struct rounds to 16)
};

@group(0) @binding(0)  var<storage, read_write> positions  : array<f32>;
@group(0) @binding(6)  var<storage, read>       list       : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
@group(0) @binding(12) var<storage, read>       mask       : array<f32>;
@group(0) @binding(63) var<uniform>             P          : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    var n : u32;
    if (P.list_mode == 0u) { n = list[0]; } else { n = P.list_count; }
    if (i >= n) {
        return;
    }
    var v : u32;
    if (P.list_mode == 0u) { v = list[i + 1u]; } else { v = list[i]; }
    if (v >= P.vertex_count) {
        return;
    }

    let mv = mirror_map[v];
    if (mv >= P.vertex_count) {   // MIRROR_UNPAIRED sentinel
        return;
    }

    if (mv == v) {
        // Seam vertex: constrained to the mirror plane.
        if (mask[v] >= 1.0) {
            return;
        }
        positions[v * 3u] = 0.0;
        return;
    }

    if (v > mv) {                 // one projection per pair
        return;
    }
    if (mask[v] >= 1.0 || mask[mv] >= 1.0) {
        return;
    }

    let a = v * 3u;
    let b = mv * 3u;
    let px = 0.5 * (positions[a + 0u] - positions[b + 0u]);
    let py = 0.5 * (positions[a + 1u] + positions[b + 1u]);
    let pz = 0.5 * (positions[a + 2u] + positions[b + 2u]);
    positions[a + 0u] =  px;
    positions[a + 1u] =  py;
    positions[a + 2u] =  pz;
    positions[b + 0u] = -px;
    positions[b + 1u] =  py;
    positions[b + 2u] =  pz;
}
