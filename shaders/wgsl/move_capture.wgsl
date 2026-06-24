// move_capture.wgsl
// Port of src/compute_move.cpp (move_capture_src). Pass 1 of the grab/move brush
// (capture, once at pen-down): brute-force per-vertex world-distance gate. For each
// vertex inside the brush sphere (and its X-mirror twin when mirror_x), write the
// falloff weights (primary, mirror), snapshot the init position, and append the
// vertex id to the compact affected list. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   8 affected list (read_write)   9 weights (read_write)
//   11 init pos (read_write)                            63 params UBO (BIND_PARAMS)

struct Params {
    anchor       : vec3<f32>,   // world_radius packs into .w slot
    world_radius : f32,
    hardness     : f32,
    mirror_x     : u32,
    vertex_count : u32,
    _pad0        : u32,
};

struct Affected {
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions    : array<f32>;
@group(0) @binding(8)  var<storage, read_write> affected     : Affected;
@group(0) @binding(9)  var<storage, read_write> move_weights : array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> move_init    : array<f32>;
@group(0) @binding(63) var<uniform>             P            : Params;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    let p = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    let d1 = distance(p, P.anchor);
    var w1 = 0.0;
    if (d1 < P.world_radius) {
        w1 = brush_falloff(d1, P.world_radius);
    }

    var w2 = 0.0;
    if (P.mirror_x != 0u) {
        let ma = vec3<f32>(-P.anchor.x, P.anchor.y, P.anchor.z);
        let d2 = distance(p, ma);
        if (d2 < P.world_radius) {
            w2 = brush_falloff(d2, P.world_radius);
        }
    }

    if (w1 <= 0.0 && w2 <= 0.0) {
        return;
    }

    move_weights[v]      = vec2<f32>(w1, w2);
    move_init[v*3u]      = p.x;
    move_init[v*3u + 1u] = p.y;
    move_init[v*3u + 2u] = p.z;

    let idx = atomicAdd(&affected.count, 1u);
    affected.ids[idx] = v;
}
