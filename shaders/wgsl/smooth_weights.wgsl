// smooth_weights.wgsl
// Port of src/compute_remesh.cpp (smooth_weights_src). Remesh per-vertex smooth weight:
// 0 if pinned or no adjacent tri is in the full selection; 1 if every adjacent tri is in
// the core (pre-grow) selection; otherwise a ring-falloff (1 ring out if it touches the
// core, else 2) over support_rings+1. See CONVENTIONS.md.
//
// Bindings mirror the ComputeBinding enum in include/compute.h:
//   4 adj_off (read)   5 adj_list (read)   14 weights (read_write)
//   15 pinned (read)   16 full_sel (read)  17 core_sel (read)   63 params UBO

struct Params {
    vertex_count  : u32,
    support_rings : i32,
    _pad0         : u32,
    _pad1         : u32,
};

@group(0) @binding(4)  var<storage, read>       adj_off  : array<u32>;
@group(0) @binding(5)  var<storage, read>       adj_list : array<u32>;
@group(0) @binding(14) var<storage, read_write> weights  : array<f32>;
@group(0) @binding(15) var<storage, read>       pinned   : array<u32>;
@group(0) @binding(16) var<storage, read>       full_sel : array<u32>;
@group(0) @binding(17) var<storage, read>       core_sel : array<u32>;
@group(0) @binding(63) var<uniform>             P        : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let v = gid.x;
    if (v >= P.vertex_count) {
        return;
    }

    if (pinned[v] != 0u) { weights[v] = 0.0; return; }

    let t_start = adj_off[v];
    let t_end   = adj_off[v + 1u];

    var any_full = false;
    var all_core = true;
    var any_core = false;
    for (var j = t_start; j < t_end; j = j + 1u) {
        let t = adj_list[j];
        if (full_sel[t] != 0u) { any_full = true; }
        if (core_sel[t] != 0u) { any_core = true; }
        else                   { all_core = false; }
    }

    if (!any_full) { weights[v] = 0.0; return; }

    if (all_core) {
        weights[v] = 1.0;
    } else {
        var ring = 2.0;
        if (any_core) { ring = 1.0; }
        weights[v] = max(0.0, 1.0 - ring / f32(P.support_rings + 1));
    }
}
