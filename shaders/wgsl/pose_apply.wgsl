// pose_apply.wgsl
// The pose brush, run once per dab. Who moves and how much is decided once at
// pen-down on the CPU (pose_bind in src/pose.cpp: the skin envelope around each
// bone, kept to what is connected to it) and uploaded compact: the affected ids,
// their pen-down positions and, per vertex, one weight per moving bone. Here each
// moving bone carries the vertex (swung about its fixed sphere, stretched along its
// length) and the vertex takes the weighted sum of those moves. The weights already
// include the share of the bones that hold still, so this is all that is left to do.
// With mirror on, the second four weights carry the vertex through the rig
// reflected in x = 0 (reflect, carry, reflect back). Damped by (1 - mask), like move.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read_write)   8 affected (read)   9 weights, 2 vec4 per affected (read)
//   11 init pos, 3 per affected (read)   12 mask (read)   63 params UBO (BIND_PARAMS)

struct Bone {
    f  : vec4<f32>,   // the end that stays put, xyz
    r0 : vec4<f32>,   // rotation, column 0; .w = bone length at pen-down
    r1 : vec4<f32>,   // rotation, column 1; .w = stretch along the bone
    r2 : vec4<f32>,   // rotation, column 2
    u  : vec4<f32>,   // unit axis, fixed end -> moving end, at pen-down
};

struct Params {
    bone_count     : u32,   // moving bones, at most 4 (one per weight lane)
    affected_count : u32,
    _pad0          : u32,
    _pad1          : u32,
    bones          : array<Bone, 4>,
};

struct Affected {
    count : u32,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read_write> positions : array<f32>;
@group(0) @binding(8)  var<storage, read>       affected  : Affected;
@group(0) @binding(9)  var<storage, read>       weights   : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>       init_pos  : array<f32>;
@group(0) @binding(12) var<storage, read>       mask      : array<f32>;
@group(0) @binding(63) var<uniform>             P         : Params;

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
    if (idx >= P.affected_count) {
        return;
    }
    let v = affected.ids[idx];
    let p = vec3<f32>(init_pos[idx*3u], init_pos[idx*3u+1u], init_pos[idx*3u+2u]);
    let w = weights[idx * 2u];
    let wm = weights[idx * 2u + 1u];
    let flip = vec3<f32>(-1.0, 1.0, 1.0);

    var moved = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < P.bone_count; i = i + 1u) {
        moved = moved + (bone_carry(p, i) - p) * w[i];
        moved = moved + (bone_carry(p * flip, i) * flip - p) * wm[i];
    }
    let dst = p + moved * (1.0 - mask[v]);

    positions[v*3u]      = dst.x;
    positions[v*3u + 1u] = dst.y;
    positions[v*3u + 2u] = dst.z;
}
