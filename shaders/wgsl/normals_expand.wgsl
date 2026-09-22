// normals_expand.wgsl
// GPU twin of Mesh::expand_dirty_to_affected. Lockstep with shaders/glsl/normals_expand.comp.
//
// A moved vertex changes the face normals it shares with its un-moved neighbours, so
// normals must be recomputed over the one-ring of every moved vertex, not just the
// moved set. That expansion used to run on the CPU: read the dab's dirty list back,
// walk the adjacency, sort/unique, upload the result — a GPU->CPU->GPU round trip
// worth 37-60% of a big-brush L10 stroke, whose only consumer was the GPU. The list
// and the adjacency both already live here, so the expansion does too.
//
// One thread per id in a {count, ids[]} source list (a dab's arena region, or the
// move/limb affected list). For each triangle around the id, each corner is claimed
// with atomicExchange on a per-vertex frame stamp: the first claim this frame appends
// the vertex to the affected list, every later one is a no-op. That dedupes across
// all of a frame's dabs for free, which the CPU needed a sort for.
//
// Bindings mirror the ComputeBinding enum:
//   2 indices   4 adj_offset   5 adj_list   6 source list (read)   7 mirror map
//   49 mark (per-vertex stamp)   50 affected list {count, ids[]}
//   61 source region UBO {base, cap}   63 params

struct Region {
    base : u32,
    cap  : u32,
    _p0  : u32,
    _p1  : u32,
};

struct Params {
    stamp        : u32,   // this frame's claim value; never 0 (0 = never claimed)
    vertex_count : u32,
    use_mirror   : u32,   // 1 = also expand each id's pair-map twin
    out_cap      : u32,   // id capacity of the affected list
};

@group(0) @binding(2)  var<storage, read>       indices    : array<u32>;
@group(0) @binding(4)  var<storage, read>       adj_offset : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list   : array<u32>;
@group(0) @binding(6)  var<storage, read>       src        : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
@group(0) @binding(49) var<storage, read_write> mark       : array<atomic<u32>>;
@group(0) @binding(50) var<storage, read_write> affected   : array<atomic<u32>>;
@group(0) @binding(61) var<uniform>             R          : Region;
@group(0) @binding(63) var<uniform>             P          : Params;

fn claim(w : u32) {
    if (w >= P.vertex_count) {
        return;
    }
    let prev = atomicExchange(&mark[w], P.stamp);
    if (prev == P.stamp) {
        return;
    }
    let slot = atomicAdd(&affected[0], 1u);
    if (slot < P.out_cap) {
        atomicStore(&affected[1u + slot], w);
    }
}

fn expand(v : u32) {
    let start = adj_offset[v];
    let end = adj_offset[v + 1u];
    for (var j = start; j < end; j = j + 1u) {
        let t = adj_list[j];
        claim(indices[t * 3u]);
        claim(indices[t * 3u + 1u]);
        claim(indices[t * 3u + 2u]);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Clamp: an overflowed region's counter runs past cap and those ids were never
    // written (same rule as dirty_args).
    let n = min(src[R.base], R.cap);
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let v = src[R.base + 1u + i];
    if (v >= P.vertex_count) {
        return;
    }
    expand(v);

    // Pair-map mirror: mirror_project moved v's twin without listing it, so its
    // one-ring needs normals too. Over-including a twin only costs a normal.
    if (P.use_mirror == 1u) {
        let mv = mirror_map[v];
        let is_other = mv != v;
        let in_range = mv < P.vertex_count;
        if (is_other && in_range) {
            expand(mv);
        }
    }
}
