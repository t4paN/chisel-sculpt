// density_colormap.wgsl
// Lockstep sibling of shaders/glsl/density_colormap.comp. Writes
// colormap(density) into the display colour VBO: the density target in Paint
// mode shows the field instead of albedo (green 0 = coarse → yellow 0.5 =
// neutral → red 1 = dense). Full pass over all verts — micro work even at
// millions — dispatched on entering the view and after every density dab; the
// app restores albedo (update_colors) when leaving the view.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   41 density (read)   28 colour (read_write)   63 params UBO

struct Params {
    vertex_count : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(41) var<storage, read>       density_buf : array<f32>;
@group(0) @binding(28) var<storage, read_write> color_buf   : array<u32>;
@group(0) @binding(63) var<uniform>             P           : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }
    let t = clamp(density_buf[v], 0.0, 1.0);
    let lo  = vec3<f32>(0.10, 0.75, 0.15);
    let mid = vec3<f32>(0.92, 0.86, 0.10);
    let hi  = vec3<f32>(0.90, 0.10, 0.08);
    var rgb : vec3<f32>;
    if (t < 0.5) {
        rgb = mix(lo, mid, t * 2.0);
    } else {
        rgb = mix(mid, hi, (t - 0.5) * 2.0);
    }
    color_buf[v] = pack4x8unorm(vec4<f32>(rgb, 1.0));
}
