// multires_apply.wgsl — WGSL sibling of shaders/glsl/multires_apply.comp.
// GPU-resident undo, undo/redo apply (same-level STROKE). For a disp layer it
// scatters the target disp into disp[] and reprojects (target - source) through
// the tangent frame into the working VBO: pos += dx*t + dy*b + dz*n. For the base
// cage the surface IS the base, so it writes the absolute target into base[] and
// the VBO. The (target,source) pair comes from the transient stage buffer, or in
// ring mode (3b-iv) from the persistent undo ring (old,new) with forward picking
// undo vs redo direction. See CONVENTIONS.md. Bindings mirror ComputeBinding:
//   0 pos (read_write)  6 dirty (read)  30 disp (read_write)  31 frames (read)
//   33 base (read_write) 34 stage (read) 35 ring (read)  63 Params

struct Params {
    count        : u32,   // 0
    targets_base : i32,   // 4
    ring_mode    : i32,   // 8
    ring_base    : u32,   // 12  (slot fills to 16)
    forward      : i32,   // 16
    _pad0        : i32,   // 20
    _pad1        : i32,   // 24
    _pad2        : i32,   // 28  (struct = 32)
};

@group(0) @binding(0)  var<storage, read_write> pos      : array<f32>;
@group(0) @binding(6)  var<storage, read>       dirty    : array<u32>;
@group(0) @binding(30) var<storage, read_write> disp     : array<f32>;
@group(0) @binding(31) var<storage, read>       frames   : array<f32>;
@group(0) @binding(33) var<storage, read_write> base_pos : array<f32>;
@group(0) @binding(34) var<storage, read>       stage    : array<f32>;
@group(0) @binding(35) var<storage, read>       ring     : array<f32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let di = gid.x;
    if (di >= P.count) {
        return;
    }
    let v = dirty[di];

    var tgt : vec3<f32>;
    var src : vec3<f32>;
    if (P.ring_mode != 0) {
        let ro = P.ring_base + di * 6u;
        let old_v = vec3<f32>(ring[ro],      ring[ro+1u], ring[ro+2u]);
        let new_v = vec3<f32>(ring[ro+3u],   ring[ro+4u], ring[ro+5u]);
        tgt = select(old_v, new_v, P.forward != 0);
        src = select(new_v, old_v, P.forward != 0);
    } else {
        tgt = vec3<f32>(stage[di*6u],      stage[di*6u+1u], stage[di*6u+2u]);
        src = vec3<f32>(stage[di*6u+3u],   stage[di*6u+4u], stage[di*6u+5u]);
    }

    if (P.targets_base != 0) {
        base_pos[v*3u]    = tgt.x;
        base_pos[v*3u+1u] = tgt.y;
        base_pos[v*3u+2u] = tgt.z;
        pos[v*3u]    = tgt.x;
        pos[v*3u+1u] = tgt.y;
        pos[v*3u+2u] = tgt.z;
        return;
    }

    let d = tgt - src;

    disp[v*3u]    = tgt.x;
    disp[v*3u+1u] = tgt.y;
    disp[v*3u+2u] = tgt.z;

    let t = vec3<f32>(frames[v*9u],      frames[v*9u+1u], frames[v*9u+2u]);
    let b = vec3<f32>(frames[v*9u+3u],   frames[v*9u+4u], frames[v*9u+5u]);
    let n = vec3<f32>(frames[v*9u+6u],   frames[v*9u+7u], frames[v*9u+8u]);

    pos[v*3u]    = pos[v*3u]    + d.x*t.x + d.y*b.x + d.z*n.x;
    pos[v*3u+1u] = pos[v*3u+1u] + d.x*t.y + d.y*b.y + d.z*n.y;
    pos[v*3u+2u] = pos[v*3u+2u] + d.x*t.z + d.y*b.z + d.z*n.z;
}
