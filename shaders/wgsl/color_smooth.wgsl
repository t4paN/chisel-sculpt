// color_smooth.wgsl
// Port of src/compute_color.cpp (color_smooth_src). Paint-smooth: the smooth gesture
// while a paint brush is active. Blends each in-radius vertex's colour toward the
// average of its 1-ring neighbours (CSR adjacency over incident tris) by
// strength*falloff. Same dual-anchor mirror + mask-shield as paint; in-place
// neighbour reads are Gauss-Seidel-racy across invocations — harmless for a colour
// blur (a u32 read is never torn). See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   2 indices (read)   4 adj_offset (read)   5 adj_list (read)
//   6 dirty (read_write) 12 mask (read)     28 color (read_write) 63 params UBO

struct Params {
    anchor_a     : vec3<f32>,   // world_radius packs into .w
    world_radius : f32,
    anchor_b     : vec3<f32>,   // hardness packs into .w
    hardness     : f32,
    strength     : f32,
    use_b        : u32,
    vertex_count : u32,
    _pad0        : u32,
};

struct Dirty {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions  : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(6)  var<storage, read_write> dirty      : Dirty;
@group(0) @binding(12) var<storage, read>       mask_buf   : array<f32>;
@group(0) @binding(28) var<storage, read_write> color_buf  : array<u32>;
@group(0) @binding(63) var<uniform>             P          : Params;

fn brush_falloff(dist : f32, radius : f32) -> f32 {
    let t = dist / radius;
    let inner = 0.15 + P.hardness * 0.55;
    if (t <= inner) {
        return 1.0;
    }
    var blend = (t - inner) / (1.0 - inner + 1e-6);
    blend = blend * blend * (3.0 - 2.0 * blend);
    return 1.0 - blend;
}

fn anchor_weight(anchor : vec3<f32>, vp : vec3<f32>) -> f32 {
    if (P.use_b != 0u && anchor.x * vp.x < 0.0) {
        return 0.0;
    }
    let dist = length(vp - anchor);
    if (dist >= P.world_radius) {
        return 0.0;
    }
    return brush_falloff(dist, P.world_radius);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let vp = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);
    var w = anchor_weight(P.anchor_a, vp);
    if (P.use_b != 0u) {
        w = max(w, anchor_weight(P.anchor_b, vp));
    }
    if (w <= 0.0) {
        return;
    }

    w = w * (1.0 - clamp(mask_buf[v], 0.0, 1.0));   // mask shields paint-smooth too
    if (w <= 0.0) {
        return;
    }

    var sum = vec3<f32>(0.0);
    var count = 0.0;
    let start = adj_offset[v];
    let end   = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let i0 = indices[t*3u];
        let i1 = indices[t*3u+1u];
        let i2 = indices[t*3u+2u];
        var n0 : u32;
        var n1 : u32;
        if      (v == i0) { n0 = i1; n1 = i2; }
        else if (v == i1) { n0 = i0; n1 = i2; }
        else              { n0 = i0; n1 = i1; }
        sum = sum + unpack4x8unorm(color_buf[n0]).rgb;
        sum = sum + unpack4x8unorm(color_buf[n1]).rgb;
        count = count + 2.0;
    }
    if (count <= 0.0) {
        return;
    }

    let avg     = sum / count;
    let old_rgb = unpack4x8unorm(color_buf[v]).rgb;
    let a       = clamp(P.strength * w, 0.0, 1.0);
    let pk_rgba = pack4x8unorm(vec4<f32>(mix(old_rgb, avg, a), 1.0));
    if (pk_rgba == color_buf[v]) {
        return;
    }

    color_buf[v] = pk_rgba;
    let idx = atomicAdd(&dirty.count, 1u);
    dirty.ids[idx] = v;
}
