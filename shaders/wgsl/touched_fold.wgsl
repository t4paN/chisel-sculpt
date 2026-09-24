// touched_fold.wgsl
// Folds a {count, ids[]} source list into a deduped, GPU-resident touched list.
// Lockstep with shaders/glsl/touched_fold.comp.
//
// Every geometry dab used to read its dirty list back to the CPU so the CPU could
// keep the stroke's touched set for undo — up to ~980 MB of bus traffic on one big L10
// stroke. The set's only consumers at pen-up are GPU kernels (autosmooth, the multires
// diff, the undo ring), so it is built here instead and never crosses unless a CPU
// reader genuinely needs it (materialize, at a level switch / save / remesh).
//
// Same dedupe as normals_expand: a per-vertex stamp claimed with atomicExchange. The
// stamp advances per stroke (or per materialize, for the stale list), so the mark
// buffer never needs clearing except when the stamp wraps.
//
// Destination layout: [0] overflow flag, [1] count, [2..] ids. The count sits at word
// 1 so the list reads as a {count, ids[]} region at base 1 — the layout dirty_args,
// normals_expand and mirror_project already take. The flag is set when a SOURCE region
// overflowed: its surplus ids were never written, so the fold cannot recover them and
// the CPU must fall back to a whole-mesh entry.
//
// Bindings mirror the ComputeBinding enum:
//   6 source list (read)   7 mirror map (read)   49 mark (per-vertex stamp)
//   50 destination list    61 source region UBO {base, cap}   63 params

struct Region {
    base : u32,
    cap  : u32,
    _p0  : u32,
    _p1  : u32,
};

struct Params {
    stamp        : u32,   // this list's claim value; never 0 (0 = never claimed)
    vertex_count : u32,   // also the destination's id capacity
    use_mirror   : u32,   // 1 = also claim each id's pair-map twin
    _pad         : u32,
};

@group(0) @binding(6)  var<storage, read>       src        : array<u32>;
@group(0) @binding(7)  var<storage, read>       mirror_map : array<u32>;
@group(0) @binding(49) var<storage, read_write> mark       : array<atomic<u32>>;
@group(0) @binding(50) var<storage, read_write> dst        : array<atomic<u32>>;
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
    let slot = atomicAdd(&dst[1], 1u);
    if (slot < P.vertex_count) {
        atomicStore(&dst[2u + slot], w);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let raw = src[R.base];
    let i = gid.x;
    if (i == 0u) {
        let overflowed = raw > R.cap;
        if (overflowed) {
            atomicMax(&dst[0], 1u);
        }
    }
    let n = min(raw, R.cap);
    if (i >= n) {
        return;
    }
    let v = src[R.base + 1u + i];
    claim(v);

    // Pair-map mirror: mirror_project moved v's twin without listing it, so the twin
    // changed too and undo must cover it. A twin that did not actually move only costs
    // a slot — the diff records old == new for it.
    if (P.use_mirror == 1u) {
        let in_range = v < P.vertex_count;
        if (in_range) {
            let mv = mirror_map[v];
            let is_other = mv != v;
            if (is_other) {
                claim(mv);
            }
        }
    }
}
