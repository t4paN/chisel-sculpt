// multires_diff.wgsl — WGSL sibling of shaders/glsl/multires_diff.comp.
// GPU-resident undo, pen-up multires diff. For each touched vert form the
// world-space stroke delta d = live_pos - snap_pos and, for a displacement-level
// stroke, reproject it into the vert's tangent frame, accumulating onto the
// pen-down disp; for a base-level stroke store the live position as the new base.
// Optionally captures (old,new) into the persistent undo ring (3b-iii).
// See CONVENTIONS.md. Bindings mirror ComputeBinding in include/compute.h:
//   0 live_pos (read)  6 dirty (read)  30 disp (read_write)  31 frames (read)
//   32 snap_pos (read) 33 base_pos (read_write) 35 ring (read_write)  63 Params

struct Params {
    count          : u32,   // 0
    writes_to_base : i32,   // 4
    ring_base      : u32,   // 8
    ring_on        : i32,   // 12  (struct = 16)
};

@group(0) @binding(0)  var<storage, read>       live_pos : array<f32>;
@group(0) @binding(6)  var<storage, read>       dirty    : array<u32>;
@group(0) @binding(30) var<storage, read_write> disp     : array<f32>;
@group(0) @binding(31) var<storage, read>       frames   : array<f32>;
@group(0) @binding(32) var<storage, read>       snap_pos : array<f32>;
@group(0) @binding(33) var<storage, read_write> base_pos : array<f32>;
@group(0) @binding(35) var<storage, read_write> ring     : array<f32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let di = gid.x;
    if (di >= P.count) {
        return;
    }
    let v = dirty[di];

    let lp = vec3<f32>(live_pos[v*3u], live_pos[v*3u+1u], live_pos[v*3u+2u]);
    let sp = vec3<f32>(snap_pos[v*3u], snap_pos[v*3u+1u], snap_pos[v*3u+2u]);

    var old_v : vec3<f32>;
    var new_v : vec3<f32>;

    if (P.writes_to_base != 0) {
        // Base-level stroke: the surface IS the base cage, so old = pen-down world
        // pos (snap), new = live pos.
        old_v = sp;
        new_v = lp;
        base_pos[v*3u]      = lp.x;
        base_pos[v*3u+1u]   = lp.y;
        base_pos[v*3u+2u]   = lp.z;
    } else {
        let d = lp - sp;
        let t = vec3<f32>(frames[v*9u],      frames[v*9u+1u], frames[v*9u+2u]);
        let b = vec3<f32>(frames[v*9u+3u],   frames[v*9u+4u], frames[v*9u+5u]);
        let n = vec3<f32>(frames[v*9u+6u],   frames[v*9u+7u], frames[v*9u+8u]);

        // disp[] still holds the pen-down disp on entry — that's old.
        old_v = vec3<f32>(disp[v*3u], disp[v*3u+1u], disp[v*3u+2u]);
        new_v = old_v + vec3<f32>(dot(t, d), dot(b, d), dot(n, d));
        disp[v*3u]      = new_v.x;
        disp[v*3u+1u]   = new_v.y;
        disp[v*3u+2u]   = new_v.z;
    }

    if (P.ring_on != 0) {
        // Pack (old,new) at this stroke's reserved span, indexed by local di so the
        // ring is densely packed per stroke (3b-iii undo capture).
        let ro = P.ring_base + di * 6u;
        ring[ro]      = old_v.x;  ring[ro+1u] = old_v.y;  ring[ro+2u] = old_v.z;
        ring[ro+3u]   = new_v.x;  ring[ro+4u] = new_v.y;  ring[ro+5u] = new_v.z;
    }
}
