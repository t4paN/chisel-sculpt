// pose_apply.wgsl
// Pass 2 of the pose brush, run once per dab. Each captured vertex is placed from
// its pen-down position: every bone that reaches it gives a weight, the moving
// bones also give where they would carry it (swung about their fixed sphere and
// stretched along their length), and the vertex takes the weighted average of
// those moves, scaled by its strongest weight. Bones that stay put add weight but
// no motion, so they hold the region around an anchored sphere still. Damped by
// (1 - mask), like move.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   8 affected (read)   11 init pos (read)
//   12 mask (read)             63 params UBO (BIND_PARAMS)

struct Bone {
    a  : vec4<f32>,   // joint A xyz at pen-down, .w = its radius
    b  : vec4<f32>,   // joint B xyz at pen-down, .w = its radius
    f  : vec4<f32>,   // the end that stays put, xyz; .w = 1 if this bone moves
    r0 : vec4<f32>,   // rotation, column 0; .w = bone length at pen-down
    r1 : vec4<f32>,   // rotation, column 1; .w = stretch along the bone
    r2 : vec4<f32>,   // rotation, column 2
    u  : vec4<f32>,   // unit axis, fixed end -> moving end, at pen-down
};

struct Params {
    bone_count   : u32,
    vertex_count : u32,
    soft         : f32,   // falloff past a sphere's skin, in radii
    _pad0        : u32,
    bones        : array<Bone, 32>,
};

struct Affected {
    count : u32,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read_write> positions : array<f32>;
@group(0) @binding(8)  var<storage, read>       affected  : Affected;
@group(0) @binding(11) var<storage, read>       move_init : array<f32>;
@group(0) @binding(12) var<storage, read>       mask      : array<f32>;
@group(0) @binding(63) var<uniform>             P         : Params;

fn bone_weight(p : vec3<f32>, i : u32) -> f32 {
    let a = P.bones[i].a.xyz;
    let ab = P.bones[i].b.xyz - a;
    let len2 = max(dot(ab, ab), 1e-12);
    let t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    let r = mix(P.bones[i].a.w, P.bones[i].b.w, t);
    let d = distance(p, a + ab * t);
    let edge = r * (1.0 + P.soft);
    if (d >= edge) {
        return 0.0;
    }
    if (d <= r) {
        return 1.0;
    }
    var x = (d - r) / max(edge - r, 1e-8);
    x = x * x * (3.0 - 2.0 * x);
    return 1.0 - x;
}

// Where moving bone i carries p. The stretch applies along the bone's own length
// only: behind the fixed sphere nothing stretches, and past the moving one the
// region rides along rigidly.
fn bone_carry(p : vec3<f32>, i : u32) -> vec3<f32> {
    let f = P.bones[i].f.xyz;
    let u = P.bones[i].u.xyz;
    let len = P.bones[i].r0.w;
    let k = P.bones[i].r1.w - 1.0;
    let q = p - f;
    let along = dot(q, u);
    var extra = 0.0;
    if (along > len) {
        extra = k * len;
    } else if (along > 0.0) {
        extra = k * along;
    }
    let s = q + u * extra;
    return f + P.bones[i].r0.xyz * s.x + P.bones[i].r1.xyz * s.y + P.bones[i].r2.xyz * s.z;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= affected.count) {
        return;
    }
    let v = affected.ids[idx];
    let p = vec3<f32>(move_init[v*3u], move_init[v*3u+1u], move_init[v*3u+2u]);

    var wsum = 0.0;
    var wmax = 0.0;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < P.bone_count; i = i + 1u) {
        let w = bone_weight(p, i);
        if (w <= 0.0) {
            continue;
        }
        wsum = wsum + w;
        wmax = max(wmax, w);
        if (P.bones[i].f.w > 0.5) {
            acc = acc + (bone_carry(p, i) - p) * w;
        }
    }
    var moved = vec3<f32>(0.0, 0.0, 0.0);
    if (wsum > 0.0) {
        moved = acc * (wmax / wsum);
    }
    let dst = p + moved * (1.0 - mask[v]);

    positions[v*3u]      = dst.x;
    positions[v*3u + 1u] = dst.y;
    positions[v*3u + 2u] = dst.z;
}
