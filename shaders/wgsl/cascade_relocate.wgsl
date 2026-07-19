// cascade_relocate.wgsl — WGSL sibling of shaders/glsl/cascade_relocate.comp.
// GPU level-switch replay, pass 1 of 3: Loop relocation of coarse vertices via
// CSR adjacency (see the GLSL header comment for the full story). Closed meshes
// only. Bindings mirror ComputeBinding (include/compute.h):
//   42 cascade src pos (read)   43 cascade dst pos (read_write)
//   2 indices   4 adj_offset   5 adj_list   63 params UBO

struct Params {
    vcount : u32,   // coarse vertex count
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,   // struct rounds to 16
};

@group(0) @binding(42) var<storage, read>       src_pos    : array<f32>;
@group(0) @binding(43) var<storage, read_write> dst_pos    : array<f32>;
@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(63) var<uniform>             P          : Params;

fn fetch(v : u32) -> vec3<f32> {
    return vec3<f32>(src_pos[v * 3u], src_pos[v * 3u + 1u], src_pos[v * 3u + 2u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= P.vcount) {
        return;
    }

    let p = fetch(i);

    let start = adj_offset[i];
    let end = adj_offset[i + 1u];
    let n = end - start;
    var beta : f32;
    if (n == 3u) {
        beta = 3.0 / 16.0;
    } else {
        beta = 3.0 / (8.0 * f32(n));
    }

    // Each neighbor vertex appears in two adjacent tris, hence the 0.5 factor
    // on the sum (same trick as the CPU loop).
    var nbr_sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        let a = indices[t * 3u];
        let b = indices[t * 3u + 1u];
        let c = indices[t * 3u + 2u];
        if (a == i) {
            nbr_sum = nbr_sum + fetch(b);
            nbr_sum = nbr_sum + fetch(c);
        } else if (b == i) {
            nbr_sum = nbr_sum + fetch(a);
            nbr_sum = nbr_sum + fetch(c);
        } else {
            nbr_sum = nbr_sum + fetch(a);
            nbr_sum = nbr_sum + fetch(b);
        }
    }

    let np = p * (1.0 - f32(n) * beta) + nbr_sum * (beta * 0.5);
    dst_pos[i * 3u] = np.x;
    dst_pos[i * 3u + 1u] = np.y;
    dst_pos[i * 3u + 2u] = np.z;
}
