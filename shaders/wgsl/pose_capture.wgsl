// pose_capture.wgsl
// Pass 1 of the pose brush (once at pen-down). A vertex joins the stroke when any
// bone that moves (one end is the grabbed sphere) reaches it; its position is
// snapshotted and its id appended to the compact affected list, the same list the
// move brush builds, so undo, normals and the mirror sink read it unchanged.
// Bones that stay put are skipped here: on their own they never move a vertex.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   0 positions (read)   8 affected list (read_write)   11 init pos (read_write)
//   63 params UBO (BIND_PARAMS)
//
// The bone struct and bone_weight() are copied verbatim into pose_apply.wgsl;
// keep the two (and the .comp twins) identical.

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
    count : atomic<u32>,
    ids   : array<u32>,
};

@group(0) @binding(0)  var<storage, read>       positions : array<f32>;
@group(0) @binding(8)  var<storage, read_write> affected  : Affected;
@group(0) @binding(11) var<storage, read_write> move_init : array<f32>;
@group(0) @binding(63) var<uniform>             P         : Params;

// 1 inside the capsule swept between the two spheres (radius blends along it),
// smoothstep down to 0 at `soft` radii past its skin.
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }
    let p = vec3<f32>(positions[v*3u], positions[v*3u+1u], positions[v*3u+2u]);

    var hit = false;
    for (var i = 0u; i < P.bone_count; i = i + 1u) {
        if (P.bones[i].f.w > 0.5 && bone_weight(p, i) > 0.0) {
            hit = true;
            break;
        }
    }
    if (!hit) {
        return;
    }

    move_init[v*3u]      = p.x;
    move_init[v*3u + 1u] = p.y;
    move_init[v*3u + 2u] = p.z;
    let idx = atomicAdd(&affected.count, 1u);
    affected.ids[idx] = v;
}
