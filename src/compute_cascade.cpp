#include "compute.h"
#include "mesh.h"
#include "multires_stack.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("cascade_*")
#include <atomic>
#include <chrono>
#include <cstdio>

// ---------------------------------------------------------------------------
// GPU level-switch cascade replay (subdiv-switch-perf-spec.md #4).
//
// The cd47a6e topology cache reduced a warm level switch to flat position math
// over static tables — exactly the shape a compute kernel wants. These four
// kernels are the GPU twin of cascade_replay_fast's loops (multires_stack.cpp):
// per pass, relocate coarse verts (Loop weights via CSR), fill midpoints from
// the cached stencil, apply the layer's displacements through its tangent
// frames; after the last pass, full-mesh normals; then ONE blocking readback
// of positions+normals (level switch is a one-shot user-paced op — the
// no-readback rule is stroke-only).
//
// Data contract: the CPU stays the storage authority. Static topology tables
// (indices, CSR, stencil mid) are VRAM-resident per level, keyed by the
// stack's lock_stamp — immutable for the locked lifetime, exactly like
// topo_cache itself. disp/frames/base re-upload on every replay, so
// projection, undo, and load need no GPU invalidation hooks; the upload is
// the price of never being stale. Dirty-tracking those uploads is the next
// lever if they ever dominate the [cascade-gpu] breakdown.
//
// Numeric contract: same per-thread accumulation order as the CPU loops, but
// GPU rounding (FMA contraction) means the result matches to ~float ulps, not
// bit-exact. CHISEL_DEBUG_MULTIRES cross-checks against the CPU fast replay
// with an epsilon (see cascade_to_level).
// ---------------------------------------------------------------------------

namespace {

// 16-byte std140 blocks, byte-identical to the cascade_*.{comp,wgsl} Params.
struct CascadeRelocateParamsGPU { uint32_t vcount, _p0, _p1, _p2; };
struct CascadeMidpointParamsGPU { uint32_t edge_count, vcoarse, _p0, _p1; };
struct CascadeDispParamsGPU     { uint32_t vcount, _p0, _p1, _p2; };
struct CascadeNormalsParamsGPU  { uint32_t vcount, _p0, _p1, _p2; };
struct CascadeFramesParamsGPU   { uint32_t vcount, _p0, _p1, _p2; };
static_assert(sizeof(CascadeRelocateParamsGPU) == 16, "cascade Params UBOs must be 16 bytes");
static_assert(sizeof(CascadeMidpointParamsGPU) == 16, "cascade Params UBOs must be 16 bytes");

// lock_stamp source: unique per lock event across all entities, assigned
// lazily on the first GPU replay (0 = unassigned; init_from_lock resets it).
std::atomic<uint64_t> g_next_lock_stamp{1};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

void release_level(ComputeState::CascadeLevelGPU& lv) {
    gpu::release_buffer(lv.indices);
    gpu::release_buffer(lv.adj_offset);
    gpu::release_buffer(lv.adj_list);
    gpu::release_buffer(lv.mid);
    lv.vcount = 0;
}

} // namespace

bool ComputeState::init_cascade() {
    if (!supported) return false;

    {
        const gpu::BindEntry layout[] = {
            { BIND_CASCADE_SRC,      gpu::Bind::StorageRead,      0 },
            { BIND_CASCADE_DST,      gpu::Bind::StorageReadWrite, 0 },
            { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
            { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(CascadeRelocateParamsGPU) },
        };
        cascade_relocate_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                        gpu::embedded_shader("cascade_relocate"), layout, 6);
    }
    {
        const gpu::BindEntry layout[] = {
            { BIND_CASCADE_SRC, gpu::Bind::StorageRead,      0 },
            { BIND_CASCADE_DST, gpu::Bind::StorageReadWrite, 0 },
            { BIND_CASCADE_MID, gpu::Bind::StorageRead,      0 },
            { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(CascadeMidpointParamsGPU) },
        };
        cascade_midpoint_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                        gpu::embedded_shader("cascade_midpoint"), layout, 4);
    }
    {
        const gpu::BindEntry layout[] = {
            { BIND_CASCADE_DST,     gpu::Bind::StorageReadWrite, 0 },
            { BIND_MULTIRES_DISP,   gpu::Bind::StorageRead,      0 },
            { BIND_MULTIRES_FRAMES, gpu::Bind::StorageRead,      0 },
            { BIND_PARAMS,          gpu::Bind::Uniform,          sizeof(CascadeDispParamsGPU) },
        };
        cascade_disp_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                    gpu::embedded_shader("cascade_disp_apply"), layout, 4);
    }
    {
        const gpu::BindEntry layout[] = {
            { BIND_CASCADE_SRC,      gpu::Bind::StorageRead,      0 },
            { BIND_NORMALS,          gpu::Bind::StorageReadWrite, 0 },
            { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
            { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(CascadeNormalsParamsGPU) },
        };
        cascade_normals_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                       gpu::embedded_shader("cascade_normals"), layout, 6);
    }
    {
        const gpu::BindEntry layout[] = {
            { BIND_CASCADE_SRC,      gpu::Bind::StorageRead,      0 },
            { BIND_NORMALS,          gpu::Bind::StorageRead,      0 },
            { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
            { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
            { BIND_MULTIRES_FRAMES,  gpu::Bind::StorageReadWrite, 0 },
            { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(CascadeFramesParamsGPU) },
        };
        cascade_frames_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                      gpu::embedded_shader("cascade_frames"), layout, 7);
    }

    if (!cascade_relocate_pipeline.handle || !cascade_midpoint_pipeline.handle ||
        !cascade_disp_pipeline.handle || !cascade_normals_pipeline.handle ||
        !cascade_frames_pipeline.handle) {
        std::printf("[compute] cascade pipelines failed to compile — level switch stays CPU\n");
        gpu::release_compute_pipeline(cascade_relocate_pipeline);
        gpu::release_compute_pipeline(cascade_midpoint_pipeline);
        gpu::release_compute_pipeline(cascade_disp_pipeline);
        gpu::release_compute_pipeline(cascade_normals_pipeline);
        gpu::release_compute_pipeline(cascade_frames_pipeline);
        return false;
    }

    cascade_relocate_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(CascadeRelocateParamsGPU), gpu::Usage::Uniform);
    cascade_midpoint_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(CascadeMidpointParamsGPU), gpu::Usage::Uniform);
    cascade_disp_ubo     = gpu::create_buffer(gpu_dev, nullptr, sizeof(CascadeDispParamsGPU), gpu::Usage::Uniform);
    cascade_normals_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(CascadeNormalsParamsGPU), gpu::Usage::Uniform);
    cascade_frames_ubo   = gpu::create_buffer(gpu_dev, nullptr, sizeof(CascadeFramesParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] cascade pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::cleanup_cascade() {
    gpu::release_compute_pipeline(cascade_relocate_pipeline);
    gpu::release_compute_pipeline(cascade_midpoint_pipeline);
    gpu::release_compute_pipeline(cascade_disp_pipeline);
    gpu::release_compute_pipeline(cascade_normals_pipeline);
    gpu::release_compute_pipeline(cascade_frames_pipeline);
    gpu::release_buffer(cascade_relocate_ubo);
    gpu::release_buffer(cascade_midpoint_ubo);
    gpu::release_buffer(cascade_disp_ubo);
    gpu::release_buffer(cascade_normals_ubo);
    gpu::release_buffer(cascade_frames_ubo);
    for (auto& lv : cascade_levels) release_level(lv);
    cascade_levels.clear();
    cascade_stamp = 0;
    gpu::release_buffer(cascade_pos_a);
    gpu::release_buffer(cascade_pos_b);
    gpu::release_buffer(cascade_norm_ssbo);
    cascade_pos_capacity = 0;
    gpu::release_buffer(cascade_disp_ssbo);
    gpu::release_buffer(cascade_frames_ssbo);
    cascade_layer_capacity = 0;
}

bool ComputeState::gpu_cascade_replay(MultiresStack& stack, Mesh& out, int passes) {
    if (!has_cascade() || passes <= 0) return false;

    // Base CSR must exist (immutable post-lock; built once — same guard as the
    // CPU fast replay).
    if (stack.base.vert_tri_offset.size() != stack.base.vertex_count() + 1)
        stack.base.build_adjacency();

    const uint32_t V_base = stack.base.vertex_count();
    const uint32_t V_top  = stack.topo_cache[passes - 1].vcount;

    // ---- device-limit guard: refuse before creating anything over-budget ----
    // (WebGPU treats an over-limit buffer as a validation error -> device loss.)
    const gpu::DeviceLimits dl = gpu::device_limits();
    const uint64_t budget = dl.max_buffer_size < dl.max_storage_binding_size
                          ? dl.max_buffer_size : dl.max_storage_binding_size;
    {
        uint64_t biggest = (uint64_t)V_top * 9ull * sizeof(float);   // frames scratch
        const uint64_t idx_top = (uint64_t)stack.topo_cache[passes - 1].indices.size() * sizeof(uint32_t);
        if (idx_top > biggest) biggest = idx_top;   // indices == CSR list size
        if (biggest > budget) {
            std::printf("[cascade-gpu] fallback: buffer %llu MB over device budget %llu MB\n",
                        (unsigned long long)(biggest >> 20), (unsigned long long)(budget >> 20));
            return false;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();

    // ---- static per-level tables, keyed by lock_stamp ----
    if (stack.lock_stamp == 0)
        stack.lock_stamp = g_next_lock_stamp.fetch_add(1);
    if (cascade_stamp != stack.lock_stamp) {
        for (auto& lv : cascade_levels) release_level(lv);
        cascade_levels.clear();
        cascade_stamp = stack.lock_stamp;
    }
    if (cascade_levels.empty()) {
        CascadeLevelGPU base_lv;
        base_lv.vcount  = V_base;
        base_lv.indices = gpu::create_buffer(gpu_dev, stack.base.indices.data(),
                              (uint64_t)stack.base.indices.size() * sizeof(uint32_t), gpu::Usage::Storage);
        base_lv.adj_offset = gpu::create_buffer(gpu_dev, stack.base.vert_tri_offset.data(),
                              (uint64_t)stack.base.vert_tri_offset.size() * sizeof(uint32_t), gpu::Usage::Storage);
        base_lv.adj_list = gpu::create_buffer(gpu_dev, stack.base.vert_tri_list.data(),
                              (uint64_t)stack.base.vert_tri_list.size() * sizeof(uint32_t), gpu::Usage::Storage);
        cascade_levels.push_back(base_lv);
    }
    while ((int)cascade_levels.size() < passes + 1) {
        const MultiresStack::LevelTopo& tc = stack.topo_cache[cascade_levels.size() - 1];
        CascadeLevelGPU lv;
        lv.vcount  = tc.vcount;
        lv.indices = gpu::create_buffer(gpu_dev, tc.indices.data(),
                          (uint64_t)tc.indices.size() * sizeof(uint32_t), gpu::Usage::Storage);
        lv.adj_offset = gpu::create_buffer(gpu_dev, tc.vt_offset.data(),
                          (uint64_t)tc.vt_offset.size() * sizeof(uint32_t), gpu::Usage::Storage);
        lv.adj_list = gpu::create_buffer(gpu_dev, tc.vt_list.data(),
                          (uint64_t)tc.vt_list.size() * sizeof(uint32_t), gpu::Usage::Storage);
        lv.mid = gpu::create_buffer(gpu_dev, tc.stencil.mid.data(),
                          (uint64_t)tc.stencil.mid.size() * sizeof(uint32_t), gpu::Usage::Storage);
        cascade_levels.push_back(lv);
    }
    // Paranoia: a stale table set under a matching stamp would be silent garbage.
    for (int i = 0; i < passes; i++) {
        if (cascade_levels[i + 1].vcount != stack.topo_cache[i].vcount) {
            std::printf("[cascade-gpu] fallback: table vcount mismatch at pass %d (%u != %u) — evicting\n",
                        i, cascade_levels[i + 1].vcount, stack.topo_cache[i].vcount);
            for (auto& lv : cascade_levels) release_level(lv);
            cascade_levels.clear();
            cascade_stamp = 0;
            return false;   // CPU replay this switch; tables rebuild next time
        }
    }
    const double t_tables = ms_since(t0);

    // ---- scratch buffers (grow-only) ----
    if (cascade_pos_capacity < V_top || !cascade_pos_a.handle) {
        gpu::release_buffer(cascade_pos_a);
        gpu::release_buffer(cascade_pos_b);
        gpu::release_buffer(cascade_norm_ssbo);
        const uint64_t bytes = (uint64_t)V_top * 3ull * sizeof(float);
        cascade_pos_a     = gpu::create_buffer(gpu_dev, nullptr, bytes, gpu::Usage::Storage);
        cascade_pos_b     = gpu::create_buffer(gpu_dev, nullptr, bytes, gpu::Usage::Storage);
        cascade_norm_ssbo = gpu::create_buffer(gpu_dev, nullptr, bytes, gpu::Usage::Storage);
        cascade_pos_capacity = V_top;
    }
    if (cascade_layer_capacity < V_top || !cascade_disp_ssbo.handle) {
        gpu::release_buffer(cascade_disp_ssbo);
        gpu::release_buffer(cascade_frames_ssbo);
        cascade_disp_ssbo   = gpu::create_buffer(gpu_dev, nullptr,
                                  (uint64_t)V_top * 3ull * sizeof(float), gpu::Usage::Storage);
        cascade_frames_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                                  (uint64_t)V_top * 9ull * sizeof(float), gpu::Usage::Storage);
        cascade_layer_capacity = V_top;
    }

    // ---- base positions: SOA -> interleaved -> pos_a ----
    cascade_stage.resize((size_t)V_base * 3);
    for (uint32_t v = 0; v < V_base; v++) {
        cascade_stage[(size_t)v * 3 + 0] = stack.base.pos_x[v];
        cascade_stage[(size_t)v * 3 + 1] = stack.base.pos_y[v];
        cascade_stage[(size_t)v * 3 + 2] = stack.base.pos_z[v];
    }
    gpu::write_buffer(gpu_dev, cascade_pos_a, 0, cascade_stage.data(),
                      (uint64_t)V_base * 3ull * sizeof(float));

    // ---- replay: one batch per dispatch (the SDF idiom — submit boundaries
    // are the cross-backend ordering guarantee, and queue writes between
    // submits stay ordered on WebGPU) ----
    double t_layers = 0.0;
    int frames_gpu_count = 0;
    gpu::Buffer* src = &cascade_pos_a;
    gpu::Buffer* dst = &cascade_pos_b;
    for (int i = 0; i < passes; i++) {
        const CascadeLevelGPU& coarse = cascade_levels[i];
        const CascadeLevelGPU& fine   = cascade_levels[i + 1];
        const uint32_t Vc = coarse.vcount;
        const uint32_t Vf = fine.vcount;
        const uint32_t E  = Vf - Vc;

        {   // relocate: coarse verts -> Loop positions on the subdivided surface
            CascadeRelocateParamsGPU u = {}; u.vcount = Vc;
            gpu::write_buffer(gpu_dev, cascade_relocate_ubo, 0, &u, sizeof(u));
            const gpu::BindBufferEntry bg[] = {
                { BIND_CASCADE_SRC,      src, src->size },
                { BIND_CASCADE_DST,      dst, dst->size },
                { BIND_INDICES,          &coarse.indices,    coarse.indices.size },
                { BIND_ADJACENCY_OFFSET, &coarse.adj_offset, coarse.adj_offset.size },
                { BIND_ADJACENCY_LIST,   &coarse.adj_list,   coarse.adj_list.size },
                { BIND_PARAMS,           &cascade_relocate_ubo, sizeof(CascadeRelocateParamsGPU) },
            };
            gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_relocate_pipeline, bg, 6);
            gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
            gpu::dispatch(b, cascade_relocate_pipeline, grp, (Vc + 255u) / 256u);
            gpu::submit(b);
            gpu::release_bind_group(grp);
        }
        {   // midpoints, canonical numbering order
            CascadeMidpointParamsGPU u = {}; u.edge_count = E; u.vcoarse = Vc;
            gpu::write_buffer(gpu_dev, cascade_midpoint_ubo, 0, &u, sizeof(u));
            const gpu::BindBufferEntry bg[] = {
                { BIND_CASCADE_SRC, src, src->size },
                { BIND_CASCADE_DST, dst, dst->size },
                { BIND_CASCADE_MID, &fine.mid, fine.mid.size },
                { BIND_PARAMS,      &cascade_midpoint_ubo, sizeof(CascadeMidpointParamsGPU) },
            };
            gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_midpoint_pipeline, bg, 4);
            gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
            gpu::dispatch(b, cascade_midpoint_pipeline, grp, (E + 255u) / 256u);
            gpu::submit(b);
            gpu::release_bind_group(grp);
        }
        // Frames for this layer: fresh CPU cache uploads as-is (bit-consistent
        // with the frames its disp was encoded against). A stale cache (sculpt
        // below cleared it — the core "block low, ascend for detail" flow)
        // rebuilds ON the GPU from the pre-disp surface just written to dst:
        // normals pass, then the compute_frames twin. Only the target level's
        // frames read back to CPU (the brush residency uploads them at rebind);
        // lower stale levels stay lazy, exactly like the CPU paths expect.
        const bool frames_stale = stack.frames[i].size() != (size_t)Vf;
        if (frames_stale) {
            frames_gpu_count++;
            {   // pre-disp normals of this level, into the normals scratch
                CascadeNormalsParamsGPU u = {}; u.vcount = Vf;
                gpu::write_buffer(gpu_dev, cascade_normals_ubo, 0, &u, sizeof(u));
                const gpu::BindBufferEntry bg[] = {
                    { BIND_CASCADE_SRC,      dst, dst->size },
                    { BIND_NORMALS,          &cascade_norm_ssbo, cascade_norm_ssbo.size },
                    { BIND_INDICES,          &fine.indices,    fine.indices.size },
                    { BIND_ADJACENCY_OFFSET, &fine.adj_offset, fine.adj_offset.size },
                    { BIND_ADJACENCY_LIST,   &fine.adj_list,   fine.adj_list.size },
                    { BIND_PARAMS,           &cascade_normals_ubo, sizeof(CascadeNormalsParamsGPU) },
                };
                gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_normals_pipeline, bg, 6);
                gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
                gpu::dispatch(b, cascade_normals_pipeline, grp, (Vf + 255u) / 256u);
                gpu::submit(b);
                gpu::release_bind_group(grp);
            }
            {   // tangent frames from pre-disp surface + normals
                CascadeFramesParamsGPU u = {}; u.vcount = Vf;
                gpu::write_buffer(gpu_dev, cascade_frames_ubo, 0, &u, sizeof(u));
                const gpu::BindBufferEntry bg[] = {
                    { BIND_CASCADE_SRC,      dst, dst->size },
                    { BIND_NORMALS,          &cascade_norm_ssbo, cascade_norm_ssbo.size },
                    { BIND_INDICES,          &fine.indices,    fine.indices.size },
                    { BIND_ADJACENCY_OFFSET, &fine.adj_offset, fine.adj_offset.size },
                    { BIND_ADJACENCY_LIST,   &fine.adj_list,   fine.adj_list.size },
                    { BIND_MULTIRES_FRAMES,  &cascade_frames_ssbo, cascade_frames_ssbo.size },
                    { BIND_PARAMS,           &cascade_frames_ubo, sizeof(CascadeFramesParamsGPU) },
                };
                gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_frames_pipeline, bg, 7);
                gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
                gpu::dispatch(b, cascade_frames_pipeline, grp, (Vf + 255u) / 256u);
                gpu::submit(b);
                gpu::release_bind_group(grp);
            }
            if (i == passes - 1) {
                // Target level: brushes encode disp against these frames, so the
                // CPU cache must hold exactly what the surface was built with.
                stack.frames[i].resize(Vf);
                gpu::read_buffer(gpu_dev, cascade_frames_ssbo, 0,
                                 (uint64_t)Vf * 9ull * sizeof(float), stack.frames[i].data());
            }
        }

        {   // this layer's disp (CPU authority), frames upload when cache fresh
            const auto tl = std::chrono::steady_clock::now();
            gpu::write_buffer(gpu_dev, cascade_disp_ssbo, 0, stack.disp[i].data(),
                              (uint64_t)Vf * 3ull * sizeof(float));
            if (!frames_stale)
                gpu::write_buffer(gpu_dev, cascade_frames_ssbo, 0, stack.frames[i].data(),
                                  (uint64_t)Vf * 9ull * sizeof(float));
            t_layers += ms_since(tl);

            CascadeDispParamsGPU u = {}; u.vcount = Vf;
            gpu::write_buffer(gpu_dev, cascade_disp_ubo, 0, &u, sizeof(u));
            const gpu::BindBufferEntry bg[] = {
                { BIND_CASCADE_DST,     dst, dst->size },
                { BIND_MULTIRES_DISP,   &cascade_disp_ssbo,   cascade_disp_ssbo.size },
                { BIND_MULTIRES_FRAMES, &cascade_frames_ssbo, cascade_frames_ssbo.size },
                { BIND_PARAMS,          &cascade_disp_ubo, sizeof(CascadeDispParamsGPU) },
            };
            gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_disp_pipeline, bg, 4);
            gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
            gpu::dispatch(b, cascade_disp_pipeline, grp, (Vf + 255u) / 256u);
            gpu::submit(b);
            gpu::release_bind_group(grp);
        }

        gpu::Buffer* tmp = src; src = dst; dst = tmp;
    }

    {   // full-mesh normals on the final surface (src holds it after the swap)
        const CascadeLevelGPU& top = cascade_levels[passes];
        CascadeNormalsParamsGPU u = {}; u.vcount = V_top;
        gpu::write_buffer(gpu_dev, cascade_normals_ubo, 0, &u, sizeof(u));
        const gpu::BindBufferEntry bg[] = {
            { BIND_CASCADE_SRC,      src, src->size },
            { BIND_NORMALS,          &cascade_norm_ssbo, cascade_norm_ssbo.size },
            { BIND_INDICES,          &top.indices,    top.indices.size },
            { BIND_ADJACENCY_OFFSET, &top.adj_offset, top.adj_offset.size },
            { BIND_ADJACENCY_LIST,   &top.adj_list,   top.adj_list.size },
            { BIND_PARAMS,           &cascade_normals_ubo, sizeof(CascadeNormalsParamsGPU) },
        };
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cascade_normals_pipeline, bg, 6);
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, cascade_normals_pipeline, grp, (V_top + 255u) / 256u);
        gpu::submit(b);
        gpu::release_bind_group(grp);
    }
    const double t_kernels = ms_since(t0) - t_tables - t_layers;

    // ---- readback: interleaved -> SOA (positions from src, normals) ----
    const auto tr = std::chrono::steady_clock::now();
    out.pos_x.resize(V_top); out.pos_y.resize(V_top); out.pos_z.resize(V_top);
    out.norm_x.resize(V_top); out.norm_y.resize(V_top); out.norm_z.resize(V_top);
    cascade_stage.resize((size_t)V_top * 3);
    gpu::read_buffer(gpu_dev, *src, 0, (uint64_t)V_top * 3ull * sizeof(float), cascade_stage.data());
    for (uint32_t v = 0; v < V_top; v++) {
        out.pos_x[v] = cascade_stage[(size_t)v * 3 + 0];
        out.pos_y[v] = cascade_stage[(size_t)v * 3 + 1];
        out.pos_z[v] = cascade_stage[(size_t)v * 3 + 2];
    }
    gpu::read_buffer(gpu_dev, cascade_norm_ssbo, 0, (uint64_t)V_top * 3ull * sizeof(float), cascade_stage.data());
    for (uint32_t v = 0; v < V_top; v++) {
        out.norm_x[v] = cascade_stage[(size_t)v * 3 + 0];
        out.norm_y[v] = cascade_stage[(size_t)v * 3 + 1];
        out.norm_z[v] = cascade_stage[(size_t)v * 3 + 2];
    }
    const double t_readback = ms_since(tr);

    std::printf("[cascade-gpu] %d passes to %u verts (%d frames on gpu): tables %.1f, layers %.1f, kernels %.1f, readback %.1f ms\n",
                passes, V_top, frames_gpu_count, t_tables, t_layers, t_kernels, t_readback);
    return true;
}
