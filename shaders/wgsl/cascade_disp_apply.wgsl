// cascade_disp_apply.wgsl — WGSL sibling of shaders/glsl/cascade_disp_apply.comp.
// GPU level-switch replay, pass 3 of 3: apply this layer's displacements in
// local tangent frames, in place on the fine position buffer. Bindings:
// 43 fine pos (read_write), 30 disp (float3/vert), 31 frames (float9/vert),
// 63 params.

struct Params {
    vcount : u32,   // fine vertex count
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,   // struct rounds to 16
};

@group(0) @binding(43) var<storage, read_write> pos    : array<f32>;
@group(0) @binding(30) var<storage, read>       disp   : array<f32>;
@group(0) @binding(31) var<storage, read>       frames : array<f32>;
@group(0) @binding(63) var<uniform>             P      : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vcount) {
        return;
    }

    let dx = disp[v * 3u];
    let dy = disp[v * 3u + 1u];
    let dz = disp[v * 3u + 2u];

    let f = v * 9u;    // Frame = t(3) b(3) n(3)
    // Same per-component expression order as the CPU loop.
    pos[v * 3u]      = pos[v * 3u]      + frames[f]      * dx + frames[f + 3u] * dy + frames[f + 6u] * dz;
    pos[v * 3u + 1u] = pos[v * 3u + 1u] + frames[f + 1u] * dx + frames[f + 4u] * dy + frames[f + 7u] * dz;
    pos[v * 3u + 2u] = pos[v * 3u + 2u] + frames[f + 2u] * dx + frames[f + 5u] * dy + frames[f + 8u] * dz;
}
