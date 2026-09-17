#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("mirror_project")
#include <cstdio>
#include <cstring>
#include <cstdlib>

ComputeState::ComputeState()
    : supported(false)
    , has_native_float_atomics(false)
    , max_workgroup_size(0)
    , max_workgroup_invocations(0)
    , max_ssbo_bindings(0)
    , accum_vertex_count(0)
    , stroke_norm_capacity(0)
    , move_buffers_capacity(0)
    , limb_scratch_capacity(0)
    , multires_stage_capacity(0)
    , undo_ring_cap_bytes(1024ull * 1024ull * 1024ull)
    , undo_ring_bytes(0)
    , undo_ring_head(0)
    , adjacency_vertex_count(0)
    , mirror_map_vertex_count(0)
    , dirty_verts_capacity(0)
    , smooth_dirty_capacity(0)
    , dirty_arena_words(0)
    , dirty_arena_head(0)
    , dirty_arena_tail(0)
    , dirty_arena_live(0)
    , remesh_vert_capacity(0)
    , remesh_tri_capacity(0)
{}

bool ComputeState::init() {
    supported = false;

    // The gpu:: seam device is set once at startup by the windowing code.
    gpu_dev = gpu::app_device();

#if defined(CHISEL_BACKEND_WEBGPU)
    // WebGPU: no raw-GL capability probe. The seam's pipeline creation guarantees
    // compute support; advertise sane limits and the CAS float-atomic path (the same
    // emulation the GL fallback uses — the WGSL kernels implement it identically).
    supported = true;
    has_native_float_atomics  = false;
    max_workgroup_size        = 256;
    max_workgroup_invocations = 256;
    max_ssbo_bindings         = 8;
    std::printf("[compute] WebGPU backend: seam pipelines, CAS float atomics\n");
    return true;
#else
    if (!GLAD_GL_ARB_compute_shader) {
        std::printf("[compute] GL_ARB_compute_shader not available\n");
        return false;
    }
    if (!GLAD_GL_ARB_shader_storage_buffer_object) {
        std::printf("[compute] GL_ARB_shader_storage_buffer_object not available\n");
        return false;
    }
    if (!GLAD_GL_ARB_shader_image_load_store) {
        std::printf("[compute] GL_ARB_shader_image_load_store not available\n");
        return false;
    }

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &max_workgroup_size);
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_workgroup_invocations);
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max_ssbo_bindings);

    has_native_float_atomics = GLAD_GL_NV_shader_atomic_float != 0;

    supported = true;

    std::printf("[compute] available: workgroup_size=%d invocations=%d ssbo_bindings=%d float_atomics=%s\n",
                max_workgroup_size, max_workgroup_invocations, max_ssbo_bindings,
                has_native_float_atomics ? "native" : "CAS emulation");

    static const char* test_src = R"(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer TestBuf { uint data[]; };
void main() { data[gl_GlobalInvocationID.x] = 42u; }
)";
    GLuint test_prog = compile_program(test_src);
    if (!test_prog) {
        std::printf("[compute] validation shader failed to compile\n");
        supported = false;
        return false;
    }

    GLuint test_ssbo;
    glGenBuffers(1, &test_ssbo);
    uint32_t init_val = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, test_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &init_val, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, test_ssbo);

    glUseProgram(test_prog);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    uint32_t result = 0;
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &result);
    glDeleteProgram(test_prog);

    if (result != 42) {
        std::printf("[compute] validation failed: expected 42, got %u\n", result);
        glDeleteBuffers(1, &test_ssbo);
        supported = false;
        return false;
    }
    std::printf("[compute] validation passed\n");

    static const char* cas_test_src = R"(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer AccumBuf { uint accum[]; };

void atomicAddFloat(uint idx, float val) {
    uint expected, desired;
    expected = accum[idx];
    for (int i = 0; i < 128; i++) {
        desired = floatBitsToUint(uintBitsToFloat(expected) + val);
        uint old = atomicCompSwap(accum[idx], expected, desired);
        if (old == expected) return;
        expected = old;
    }
}

void main() { atomicAddFloat(0, 1.0); }
)";
    GLuint cas_prog = compile_program(cas_test_src);
    if (!cas_prog) {
        std::printf("[compute] CAS float atomic shader failed to compile\n");
        glDeleteBuffers(1, &test_ssbo);
        supported = false;
        return false;
    }

    init_val = 0;
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &init_val, GL_DYNAMIC_COPY);

    glUseProgram(cas_prog);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &result);
    float cas_result = *reinterpret_cast<float*>(&result);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &test_ssbo);
    glDeleteProgram(cas_prog);

    if (cas_result < 63.5f || cas_result > 64.5f) {
        std::printf("[compute] CAS float atomic failed: expected 64.0, got %.1f\n", cas_result);
        supported = false;
        return false;
    }
    std::printf("[compute] CAS float atomic validated (64 threads -> %.1f)\n", cas_result);

    return true;
#endif // CHISEL_BACKEND_GL vs WEBGPU
}

GLuint ComputeState::compile_program(const char* src) const {
#if defined(CHISEL_BACKEND_WEBGPU)
    // GLSL compute programs aren't used on WebGPU — kernels run as WGSL via the seam.
    (void)src;
    return 0;
#else
    if (!supported) return 0;

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[compute] compile error: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[compute] link error: %s\n", log);
        glDeleteProgram(program);
        return 0;
    }

    return program;
#endif
}

void ComputeState::cleanup() {
    gpu::release_buffer(accum_ssbo);
    gpu::release_buffer(accum_sym_ssbo);
    gpu::release_buffer(stroke_norm_ssbo);
    stroke_norm_capacity = 0;
    gpu::release_compute_pipeline(draw_accum_pipeline);
    gpu::release_compute_pipeline(draw_symmetrize_pipeline);
    gpu::release_compute_pipeline(draw_apply_pipeline);
    gpu::release_compute_pipeline(draw_mirror_apply_pipeline);
    gpu::release_buffer(draw_accum_ubo);
    gpu::release_buffer(draw_vcount_ubo);
    gpu::release_buffer(mirror_map_ssbo);
    gpu::release_compute_pipeline(smooth_accum_pipeline);
    gpu::release_compute_pipeline(smooth_apply_pipeline);
    gpu::release_compute_pipeline(smooth_mirror_apply_pipeline);
    gpu::release_buffer(smooth_accum_ubo);
    gpu::release_buffer(smooth_apply_ubo);
    gpu::release_buffer(smooth_mirror_ubo);
    gpu::release_compute_pipeline(stroke_smooth_apply_pipeline);
    gpu::release_buffer(stroke_smooth_ubo);
    gpu::release_compute_pipeline(mirror_project_pipeline);
    gpu::release_compute_pipeline(dirty_args_pipeline);
    gpu::release_buffer(dispatch_args_ssbo);
    gpu::release_compute_pipeline(block_boxes_pipeline);
    gpu::release_compute_pipeline(block_select_pipeline);
    gpu::release_compute_pipeline(block_args_pipeline);
    gpu::release_compute_pipeline(accum_clear_blocks_pipeline);
    gpu::release_buffer(block_boxes_ssbo);
    gpu::release_buffer(block_list_ssbo);
    gpu::release_buffer(block_sticky_ssbo);
    gpu::release_buffer(block_args_ssbo);
    gpu::release_buffer(block_ubo);
    block_count = block_capacity = 0;
    gpu::release_buffer(mirror_project_ubo);
    gpu::release_compute_pipeline(crease_accum_pipeline);
    gpu::release_compute_pipeline(pinch_accum_pipeline);
    gpu::release_buffer(crease_ubo);
    gpu::release_buffer(pinch_ubo);
    gpu::release_compute_pipeline(mask_pipeline);
    gpu::release_compute_pipeline(mask_smooth_pipeline);
    gpu::release_buffer(mask_params_ubo);
    gpu::release_buffer(mask_smooth_ubo);
    gpu::release_compute_pipeline(color_paint_pipeline);
    gpu::release_compute_pipeline(color_smooth_pipeline);
    gpu::release_buffer(color_paint_ubo);
    gpu::release_buffer(color_smooth_ubo);
    gpu::release_compute_pipeline(move_capture_pipeline);
    gpu::release_compute_pipeline(move_weight_smooth_pipeline);
    gpu::release_compute_pipeline(move_apply_pipeline);
    gpu::release_buffer(move_capture_ubo);
    gpu::release_buffer(move_apply_ubo);
    gpu::release_buffer(move_affected_ssbo);
    gpu::release_buffer(move_weights_ssbo);
    gpu::release_buffer(move_weights_pong_ssbo);
    gpu::release_buffer(move_init_ssbo);
    gpu::release_compute_pipeline(limb_drag_pipeline);
    gpu::release_compute_pipeline(limb_relax_pipeline);
    gpu::release_buffer(limb_drag_ubo);
    gpu::release_buffer(limb_relax_ubo);
    gpu::release_buffer(limb_pos_scratch_ssbo);
    move_buffers_capacity = 0;
    gpu::release_compute_pipeline(compute_normals_pipeline);
    gpu::release_buffer(compute_normals_ubo);
    gpu::release_compute_pipeline(multires_diff_pipeline);
    gpu::release_compute_pipeline(multires_apply_pipeline);
    gpu::release_buffer(multires_diff_ubo);
    gpu::release_buffer(multires_apply_ubo);
    gpu::release_buffer(multires_stage_ssbo);
    multires_stage_capacity = 0;
    cleanup_cascade();
    gpu::release_buffer(undo_ring_ssbo);
    undo_ring_bytes = 0;
    undo_ring_head  = 0;
    gpu::release_buffer(adjacency_offset_ssbo);
    gpu::release_buffer(adjacency_list_ssbo);
    gpu::release_buffer(dirty_verts_ssbo);
    gpu::release_buffer(smooth_dirty_ssbo);
    gpu::release_compute_pipeline(remesh_select_stretched_pipeline);
    gpu::release_compute_pipeline(remesh_select_unmasked_pipeline);
    gpu::release_compute_pipeline(remesh_grow_selection_pipeline);
    gpu::release_compute_pipeline(remesh_mirror_selection_pipeline);
    gpu::release_compute_pipeline(remesh_find_pinned_pipeline);
    gpu::release_compute_pipeline(remesh_smooth_weights_pipeline);
    gpu::release_compute_pipeline(remesh_seam_snap_pipeline);
    gpu::release_compute_pipeline(remesh_seam_weld_pipeline);
    gpu::release_compute_pipeline(remesh_smooth_pipeline);
    gpu::release_buffer(remesh_select_stretched_ubo);
    gpu::release_buffer(remesh_select_unmasked_ubo);
    gpu::release_buffer(remesh_grow_selection_ubo);
    gpu::release_buffer(remesh_mirror_selection_ubo);
    gpu::release_buffer(remesh_find_pinned_ubo);
    gpu::release_buffer(remesh_smooth_weights_ubo);
    gpu::release_buffer(remesh_seam_snap_ubo);
    gpu::release_buffer(remesh_seam_weld_ubo);
    gpu::release_buffer(remesh_smooth_ubo);
    gpu::release_buffer(remesh_core_sel_ssbo);
    gpu::release_buffer(remesh_trisel_pong_ssbo);
    gpu::release_buffer(seam_weld_map_ssbo);
    gpu::release_buffer(remesh_ping_ssbo);
    gpu::release_buffer(remesh_pong_ssbo);
    gpu::release_buffer(remesh_norm_ssbo);
    gpu::release_buffer(remesh_weights_ssbo);
    gpu::release_buffer(remesh_pinned_ssbo);
    gpu::release_buffer(remesh_trisel_ssbo);
    gpu::release_buffer(remesh_indices_ssbo);
    gpu::release_buffer(remesh_adj_csr_ssbo);
    remesh_vert_capacity = remesh_tri_capacity = 0;
    accum_vertex_count = 0;
    adjacency_vertex_count = 0;
    gpu::release_buffer(dirty_region_ubo);
    dirty_verts_capacity = 0;
    smooth_dirty_capacity = 0;
    dirty_arena_words = 0;
    dirty_arena_reset();
}

// ---------------------------------------------------------------------------
// Async count+list readbacks (dirty list / move-affected list)
// ---------------------------------------------------------------------------

gpu::ReadTicket ComputeState::kick_count_list_read(const gpu::Buffer& buf,
                                                   uint32_t capacity, uint32_t& words,
                                                   uint32_t word_offset) {
    if (!buf.handle || capacity == 0) { words = 0; return 0; }
    words = capacity + 1;   // [count, id0, id1, ...]
    return gpu::read_buffer_async(gpu_dev, buf,
                                  (uint64_t)word_offset * sizeof(uint32_t),
                                  (uint64_t)words * sizeof(uint32_t));
}

bool ComputeState::take_count_list_read(gpu::ReadTicket t, uint32_t words,
                                        std::vector<uint32_t>& out, uint32_t* out_total) {
    out.clear();
    if (out_total) *out_total = 0;
    if (!t || words == 0) return true;
    count_list_scratch.resize(words);
    if (!gpu::ticket_take(gpu_dev, t, count_list_scratch.data(),
                          (uint64_t)words * sizeof(uint32_t)))
        return false;
    uint32_t count = count_list_scratch[0];
    // The counter is GPU-written and deliberately unbounded, so bound it by something
    // real before it is used as a length: no dab can touch more than the whole mesh.
    if (count > smooth_dirty_capacity) count = smooth_dirty_capacity;
    // The kernels let the counter run past the region's cap on purpose: that overrun
    // is the ONLY evidence the region was too small, and the surplus ids were never
    // written. Report the true total, then clamp to what is actually readable.
    if (out_total) *out_total = count;
    if (count > words - 1) count = words - 1;
    out.assign(count_list_scratch.begin() + 1, count_list_scratch.begin() + 1 + count);
    return true;
}

gpu::ReadTicket ComputeState::kick_dirty_read(uint32_t base, uint32_t cap, uint32_t& words) {
    return kick_count_list_read(smooth_dirty_ssbo, cap, words, base);
}

// ---------------------------------------------------------------------------
// Dirty-list arena
// ---------------------------------------------------------------------------
// A ring of per-dab regions. The accounting is four numbers and one invariant:
//
//     tail + live == head   (mod dirty_arena_words)
//
// which is checked in alloc() rather than reasoned about, because this exact
// bookkeeping has been got wrong before: releasing a bailed dab's region through
// retire() (which frees the OLDEST region) told the ring that a region still being
// read was free. Once the ring wrapped, two dabs shared one region, a dab's ids were
// not the ids its own kernels wrote, and undo restored the wrong vertices — surfacing
// a mile from its cause as corruption after deep undo across subdiv levels. An
// assertion here turns that whole class into a loud refusal.

void ComputeState::dirty_arena_reset() {
    dirty_arena_head = dirty_arena_tail = dirty_arena_live = 0;
}

// Ring positions are logical: position p lives at word 1 + p, because word 0 is the
// bit bucket reserved for dabs that could not get a region.
uint32_t ComputeState::dirty_arena_ring_words() const {
    return dirty_arena_words > 1 ? dirty_arena_words - 1 : 0;
}

uint32_t ComputeState::dirty_arena_max_cap() const {
    uint32_t w = dirty_arena_ring_words();
    return w > 1 ? w - 1 : 0;
}

bool ComputeState::dirty_arena_alloc(uint32_t cap, uint32_t& base_out,
                                     uint32_t& footprint_out) {
    const uint32_t W = dirty_arena_ring_words();
    if (W == 0) return false;
    const uint32_t need = cap + 1;                 // counter word + ids
    if (need > W) return false;

    if ((dirty_arena_tail + dirty_arena_live) % W != dirty_arena_head) {
        std::printf("[arena] INVARIANT BROKEN: tail %u + live %u != head %u (ring %u) — "
                    "refusing to allocate\n",
                    dirty_arena_tail, dirty_arena_live, dirty_arena_head, W);
        return false;
    }

    // A region must be contiguous (the shader indexes base + 1 + i with no wrap), so
    // if it does not fit before the end of the ring the gap is padded and charged to
    // this region's footprint — that keeps the invariant arithmetic a plain sum.
    uint32_t base = dirty_arena_head;
    uint32_t foot = need;
    if (dirty_arena_head + need > W) {
        foot = (W - dirty_arena_head) + need;
        base = 0;
    }
    if (foot > W - dirty_arena_live) return false;   // full; caller drains and retries

    dirty_arena_head = (dirty_arena_head + foot) % W;
    dirty_arena_live += foot;
    base_out = base + 1;                             // logical -> word
    footprint_out = foot;
    return true;
}

void ComputeState::dirty_arena_retire(uint32_t footprint) {
    const uint32_t W = dirty_arena_ring_words();
    if (W == 0 || footprint == 0) return;
    if (footprint > dirty_arena_live) { dirty_arena_reset(); return; }
    dirty_arena_tail = (dirty_arena_tail + footprint) % W;
    dirty_arena_live -= footprint;
}

void ComputeState::dirty_arena_unalloc(uint32_t footprint) {
    // For a dab that reserved a region and then bailed before kicking its read. It
    // gives back the NEWEST region, so it rolls the bump cursor back and leaves the
    // tail alone — the opposite end from retire().
    const uint32_t W = dirty_arena_ring_words();
    if (W == 0 || footprint == 0) return;
    if (footprint > dirty_arena_live) { dirty_arena_reset(); return; }
    dirty_arena_head = (dirty_arena_head + W - footprint) % W;
    dirty_arena_live -= footprint;
}

void ComputeState::set_dirty_region(uint32_t base, uint32_t cap) {
    if (!dirty_region_ubo.handle)
        dirty_region_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DirtyRegionGPU),
                                              gpu::Usage::Uniform);
    dirty_region_base = base;
    dirty_region_cap  = cap;
    ++dab_serial;            // a selection made for an earlier dab is now stale
    dirty_block_mode  = 0;   // a new dab has no selection yet; only a path that runs
                             // block_select for THIS dab may turn it on
    DirtyRegionGPU dr = { base, cap, dirty_block_mode, 0 };
    gpu::write_buffer(gpu_dev, dirty_region_ubo, 0, &dr, sizeof(dr));
    uint32_t zero = 0;
    if (smooth_dirty_ssbo.handle)
        gpu::write_buffer(gpu_dev, smooth_dirty_ssbo,
                          (uint64_t)base * sizeof(uint32_t), &zero, sizeof(zero));
}

gpu::ReadTicket ComputeState::kick_move_affected_read(uint32_t& words) {
    return kick_count_list_read(move_affected_ssbo, move_buffers_capacity, words);
}

// ---------------------------------------------------------------------------
// Mirror constraint projection (single symmetry sink for all brushes)
// ---------------------------------------------------------------------------

// 16-byte std140 block, byte-identical to mirror_project.{comp,wgsl}'s Params.
struct MirrorProjectParamsGPU {
    uint32_t vertex_count;
    uint32_t list_mode;    // 0 = {count, ids[]} header, 1 = plain ids + list_count
    uint32_t list_count;
    uint32_t _pad0;
};
static_assert(sizeof(MirrorProjectParamsGPU) == 16, "mirror_project Params UBO must be 16 bytes");

bool ComputeState::init_dirty_args() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_DIRTY_VERTS,   gpu::Bind::StorageRead,      0 },
        { BIND_DISPATCH_ARGS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_REGION,  gpu::Bind::Uniform,          sizeof(DirtyRegionGPU) },
    };
    dirty_args_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("dirty_args"), layout, 3);
    if (!dirty_args_pipeline.handle) {
        std::printf("[compute] dirty_args pipeline failed to compile\n");
        return false;
    }
    dispatch_args_ssbo = gpu::create_buffer(gpu_dev, nullptr, 3 * sizeof(uint32_t),
                                            gpu::Usage::Storage | gpu::Usage::Indirect);
    std::printf("[compute] dirty_args pipeline compiled (indirect dispatch)\n");
    return true;
}

bool ComputeState::init_mirror_project() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP,  gpu::Bind::StorageRead,      0 },
        { BIND_MASK,        gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_REGION, gpu::Bind::Uniform,         sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(MirrorProjectParamsGPU) },
    };
    mirror_project_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                  gpu::embedded_shader("mirror_project"), layout, 6);
    if (!mirror_project_pipeline.handle) {
        std::printf("[compute] mirror_project pipeline failed to compile\n");
        return false;
    }
    mirror_project_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(MirrorProjectParamsGPU),
                                            gpu::Usage::Uniform);
    std::printf("[compute] mirror_project pipeline compiled (gpu:: seam)\n");
    return true;
}

static void dispatch_mirror_project_impl(ComputeState& cs, const gpu::Buffer& pos_vbo,
                                         uint32_t vertex_count, const gpu::Buffer& list_buf,
                                         uint32_t list_mode, uint32_t list_count,
                                         uint32_t groups) {
    if (!cs.has_mirror_project() || !cs.mask_ssbo.handle) return;
    if (!cs.mirror_map_ssbo.handle || cs.mirror_map_vertex_count != vertex_count) return;
    if (!list_buf.handle || groups == 0) return;

    MirrorProjectParamsGPU u = { vertex_count, list_mode, list_count, 0 };
    gpu::write_buffer(cs.gpu_dev, cs.mirror_project_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,   &pos_vbo,               (uint64_t)vertex_count * 3u * sizeof(float) },
        { BIND_DIRTY_VERTS, &list_buf,              list_buf.size },
        { BIND_MIRROR_MAP,  &cs.mirror_map_ssbo,    (uint64_t)cs.mirror_map_vertex_count * sizeof(uint32_t) },
        { BIND_MASK,        &cs.mask_ssbo,          (uint64_t)vertex_count * sizeof(float) },
        { BIND_DIRTY_REGION, &cs.dirty_region_ubo,  sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,      &cs.mirror_project_ubo, sizeof(MirrorProjectParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(cs.gpu_dev, cs.mirror_project_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
    gpu::dispatch(b, cs.mirror_project_pipeline, grp, groups);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

// Same bindings as dispatch_mirror_project_impl in list_mode 0, but the workgroup
// count comes from dirty_args writing the dab's own count into dispatch_args_ssbo.
// Both dispatches go in ONE batch: the args must be written and visible before the
// command processor reads them, which is free inside a pass on WebGPU and needs the
// GL_COMMAND_BARRIER the GL backend issues in dispatch_indirect.
static void dispatch_mirror_project_indirect(ComputeState& cs, const gpu::Buffer& pos_vbo,
                                             uint32_t vertex_count,
                                             const gpu::Buffer& list_buf) {
    if (!cs.has_mirror_project() || !cs.mask_ssbo.handle) return;
    if (!cs.mirror_map_ssbo.handle || cs.mirror_map_vertex_count != vertex_count) return;
    if (!list_buf.handle) return;

    MirrorProjectParamsGPU u = { vertex_count, 0u, 0u, 0 };
    gpu::write_buffer(cs.gpu_dev, cs.mirror_project_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry args_bg[] = {
        { BIND_DIRTY_VERTS,   &list_buf,               list_buf.size },
        { BIND_DISPATCH_ARGS, &cs.dispatch_args_ssbo,  cs.dispatch_args_ssbo.size },
        { BIND_DIRTY_REGION,  &cs.dirty_region_ubo,    sizeof(DirtyRegionGPU) },
    };
    gpu::BindGroup args_grp = gpu::create_bind_group(cs.gpu_dev, cs.dirty_args_pipeline,
                                                     args_bg, 3);

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,    &pos_vbo,               (uint64_t)vertex_count * 3u * sizeof(float) },
        { BIND_DIRTY_VERTS,  &list_buf,              list_buf.size },
        { BIND_MIRROR_MAP,   &cs.mirror_map_ssbo,    (uint64_t)cs.mirror_map_vertex_count * sizeof(uint32_t) },
        { BIND_MASK,         &cs.mask_ssbo,          (uint64_t)vertex_count * sizeof(float) },
        { BIND_DIRTY_REGION, &cs.dirty_region_ubo,   sizeof(DirtyRegionGPU) },
        { BIND_PARAMS,       &cs.mirror_project_ubo, sizeof(MirrorProjectParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(cs.gpu_dev, cs.mirror_project_pipeline, bg, 6);

    gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
    gpu::dispatch(b, cs.dirty_args_pipeline, args_grp, 1);
    gpu::dispatch_indirect(b, cs.mirror_project_pipeline, grp, cs.dispatch_args_ssbo, 0);
    gpu::submit(b);
    gpu::release_bind_group(args_grp);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_mirror_project_header(const gpu::Buffer& pos_vbo,
                                                  uint32_t vertex_count,
                                                  const gpu::Buffer& list_buf) {
    // The entry count lives in the dab's region header on the GPU. With indirect
    // dispatch we size the dispatch from that count directly; without it, the only
    // option was worst-case threads over the whole mesh with the overshoot early-
    // outing — one full-mesh pass per dab to touch a few hundred vertices.
    if (has_dirty_args() && &list_buf == &smooth_dirty_ssbo) {
        dispatch_mirror_project_indirect(*this, pos_vbo, vertex_count, list_buf);
        return;
    }
    dispatch_mirror_project_impl(*this, pos_vbo, vertex_count, list_buf,
                                 0u, 0u, (vertex_count + 255u) / 256u);
}

void ComputeState::dispatch_mirror_project_ids(const gpu::Buffer& pos_vbo,
                                               uint32_t vertex_count, uint32_t id_count) {
    if (id_count == 0) return;
    dispatch_mirror_project_impl(*this, pos_vbo, vertex_count, dirty_verts_ssbo,
                                 1u, id_count, (id_count + 255u) / 256u);
}

// ---------------------------------------------------------------------------
// Brush dispatch culling — see the contract on ComputeState's block_* members.
// ---------------------------------------------------------------------------

namespace {
struct BlockBoxParamsGPU { uint32_t vertex_count, block_count, _p0, _p1; };
static_assert(sizeof(BlockBoxParamsGPU) == 16, "block box Params UBO must be 16 bytes");

struct BlockSelectParamsGPU {
    float    anchor_a[3]; float radius_a;            // 16
    float    anchor_b[3]; float radius_b;            // 16
    uint32_t block_count; uint32_t _p0, _p1, _p2;    // 16
};
static_assert(sizeof(BlockSelectParamsGPU) == 48, "block select Params UBO must be 48 bytes");
}  // namespace

bool ComputeState::init_block_cull() {
    if (!supported) return false;
    // On by default; CHISEL_BLOCK_CULL=0 turns it off for an A/B without a rebuild.
    const char* e = getenv("CHISEL_BLOCK_CULL");
    block_cull_on = !(e && *e == '0');
    if (!block_cull_on) {
        std::printf("[compute] block culling DISABLED (CHISEL_BLOCK_CULL=0)\n");
        return false;
    }

    const gpu::BindEntry box_layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageRead,      0 },
        { BIND_BLOCK_BOXES, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(BlockBoxParamsGPU) },
    };
    block_boxes_pipeline = gpu::create_compute_pipeline(gpu_dev,
                               gpu::embedded_shader("block_boxes"), box_layout, 3);

    const gpu::BindEntry sel_layout[] = {
        { BIND_BLOCK_LIST,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_BLOCK_BOXES,  gpu::Bind::StorageRead,      0 },
        { BIND_BLOCK_STICKY, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,       gpu::Bind::Uniform,          sizeof(BlockSelectParamsGPU) },
    };
    block_select_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                gpu::embedded_shader("block_select"), sel_layout, 4);

    const gpu::BindEntry args_layout[] = {
        { BIND_BLOCK_LIST,    gpu::Bind::StorageRead,      0 },
        { BIND_DISPATCH_ARGS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(BlockBoxParamsGPU) },
    };
    block_args_pipeline = gpu::create_compute_pipeline(gpu_dev,
                              gpu::embedded_shader("block_args"), args_layout, 3);

    const gpu::BindEntry clr_layout[] = {
        { BIND_ACCUM,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_BLOCK_LIST, gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,     gpu::Bind::Uniform,          sizeof(BlockBoxParamsGPU) },
    };
    accum_clear_blocks_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                      gpu::embedded_shader("accum_clear_blocks"),
                                      clr_layout, 3);

    if (!block_boxes_pipeline.handle || !block_select_pipeline.handle
        || !block_args_pipeline.handle || !accum_clear_blocks_pipeline.handle) {
        std::printf("[compute] block-cull pipelines failed to compile — culling off\n");
        block_cull_on = false;
        return false;
    }
    block_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(BlockSelectParamsGPU),
                                   gpu::Usage::Uniform);
    block_args_ssbo = gpu::create_buffer(gpu_dev, nullptr, 3 * sizeof(uint32_t),
                                         gpu::Usage::Storage | gpu::Usage::Indirect);
    std::printf("[compute] block-cull pipelines compiled (%u-vertex blocks)\n", kVertexBlock);
    return true;
}

void ComputeState::ensure_block_buffers(uint32_t vertex_count) {
    block_count = (vertex_count + kVertexBlock - 1u) / kVertexBlock;
    if (block_count <= block_capacity && block_boxes_ssbo.handle) return;

    gpu::release_buffer(block_boxes_ssbo);
    gpu::release_buffer(block_list_ssbo);
    gpu::release_buffer(block_sticky_ssbo);
    uint32_t alloc = block_count < 64u ? 64u : block_count;
    block_boxes_ssbo  = gpu::create_buffer(gpu_dev, nullptr,
                            (uint64_t)alloc * 6u * sizeof(float), gpu::Usage::Storage);
    // +1 for the count word the list carries in slot 0.
    block_list_ssbo   = gpu::create_buffer(gpu_dev, nullptr,
                            (uint64_t)(alloc + 1u) * sizeof(uint32_t), gpu::Usage::Storage);
    block_sticky_ssbo = gpu::create_buffer(gpu_dev, nullptr,
                            (uint64_t)alloc * sizeof(uint32_t), gpu::Usage::Storage);
    block_capacity = alloc;
}

void ComputeState::dispatch_block_boxes(const gpu::Buffer& pos_vbo, uint32_t vertex_count) {
    if (!has_block_cull()) return;
    ensure_block_buffers(vertex_count);
    if (!block_boxes_ssbo.handle || block_count == 0) return;

    BlockBoxParamsGPU u = { vertex_count, block_count, 0, 0 };
    gpu::write_buffer(gpu_dev, block_ubo, 0, &u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,   &pos_vbo,          (uint64_t)vertex_count * 3u * sizeof(float) },
        { BIND_BLOCK_BOXES, &block_boxes_ssbo, block_boxes_ssbo.size },
        { BIND_PARAMS,      &block_ubo,        sizeof(BlockBoxParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, block_boxes_pipeline, bg, 3);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, block_boxes_pipeline, grp, (block_count + 63u) / 64u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::begin_block_frame() {
    if (!has_block_cull() || !block_list_ssbo.handle) return;
    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, block_list_ssbo, 0, &zero, sizeof(zero));
    gpu::clear_buffer(gpu_dev, block_sticky_ssbo, 0);
}

void ComputeState::dispatch_block_select(float ax, float ay, float az, float ra,
                                         float bx, float by, float bz, float rb) {
    if (!has_block_cull() || !block_boxes_ssbo.handle || block_count == 0) return;

    BlockSelectParamsGPU u = {};
    u.anchor_a[0] = ax; u.anchor_a[1] = ay; u.anchor_a[2] = az; u.radius_a = ra;
    u.anchor_b[0] = bx; u.anchor_b[1] = by; u.anchor_b[2] = bz; u.radius_b = rb;
    u.block_count = block_count;
    gpu::write_buffer(gpu_dev, block_ubo, 0, &u, sizeof(u));
    block_sel_serial = dab_serial;   // this selection belongs to THIS dab

    const gpu::BindBufferEntry bg[] = {
        { BIND_BLOCK_LIST,   &block_list_ssbo,   block_list_ssbo.size },
        { BIND_BLOCK_BOXES,  &block_boxes_ssbo,  block_boxes_ssbo.size },
        { BIND_BLOCK_STICKY, &block_sticky_ssbo, block_sticky_ssbo.size },
        { BIND_PARAMS,       &block_ubo,         sizeof(BlockSelectParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, block_select_pipeline, bg, 4);

    const gpu::BindBufferEntry abg[] = {
        { BIND_BLOCK_LIST,    &block_list_ssbo,  block_list_ssbo.size },
        { BIND_DISPATCH_ARGS, &block_args_ssbo,  block_args_ssbo.size },
        { BIND_PARAMS,        &block_ubo,        16 },
    };

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, block_select_pipeline, grp, (block_count + 63u) / 64u);
    gpu::submit(b);
    gpu::release_bind_group(grp);

    // block_args reads block_count from the FIRST word of its own 16-byte view of this
    // UBO, where BlockSelectParams keeps anchor_a.x — so rewrite it as a BlockBoxParams
    // before the args dispatch. Getting this wrong once made workgroup count come out
    // zero and the draw brush silently do nothing.
    uint32_t a[4] = { block_count, 0, 0, 0 };
    gpu::write_buffer(gpu_dev, block_ubo, 0, a, sizeof(a));
    gpu::BindGroup agrp = gpu::create_bind_group(gpu_dev, block_args_pipeline, abg, 3);
    gpu::ComputeBatch b2 = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b2, block_args_pipeline, agrp, 1);
    gpu::submit(b2);
    gpu::release_bind_group(agrp);

    // CHISEL_BLOCK_CULL=2: synchronous peek at what the selection produced. A stroke
    // sync, so debug only — it exists because "nothing was selected" and "nothing was
    // dispatched" look identical from the dirty counts alone.
    static int dbg = -1;
    if (dbg < 0) { const char* e = getenv("CHISEL_BLOCK_CULL"); dbg = (e && *e == '2') ? 24 : 0; }
    if (dbg > 0) {
        uint32_t n = 0, args3[3] = {0,0,0};
        float box6[6] = {0,0,0,0,0,0};
        gpu::read_buffer(gpu_dev, block_list_ssbo, 0, sizeof(uint32_t), &n);
        gpu::read_buffer(gpu_dev, block_args_ssbo, 0, sizeof(args3), args3);
        gpu::read_buffer(gpu_dev, block_boxes_ssbo, 0, sizeof(box6), box6);
        std::printf("[blockdbg] sel=%u of %u  args=(%u,%u,%u)  anchor=(%.3f,%.3f,%.3f) r=%.4f  box0=[%.3f %.3f %.3f]..[%.3f %.3f %.3f]\n",
                    n, block_count, args3[0], args3[1], args3[2], ax, ay, az, ra,
                    box6[0], box6[1], box6[2], box6[3], box6[4], box6[5]);
        dbg--;
    }
}

void ComputeState::clear_accum_blocks(uint32_t vertex_count) {
    if (!has_block_cull() || !accum_clear_blocks_pipeline.handle) return;
    if (!accum_ssbo.handle || !block_args_ssbo.handle) return;

    uint32_t u[4] = { vertex_count, 0, 0, 0 };
    gpu::write_buffer(gpu_dev, block_ubo, 0, u, sizeof(u));

    const gpu::BindBufferEntry bg[] = {
        { BIND_ACCUM,      &accum_ssbo,      (uint64_t)vertex_count * 4u * sizeof(uint32_t) },
        { BIND_BLOCK_LIST, &block_list_ssbo, block_list_ssbo.size },
        { BIND_PARAMS,     &block_ubo,       16 },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, accum_clear_blocks_pipeline, bg, 3);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch_indirect(b, accum_clear_blocks_pipeline, grp, block_args_ssbo, 0);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::set_block_mode(bool on) {
    uint32_t want = on ? 1u : 0u;
    if (want == dirty_block_mode) return;
    dirty_block_mode = want;
    if (!dirty_region_ubo.handle) return;
    DirtyRegionGPU dr = { dirty_region_base, dirty_region_cap, dirty_block_mode, 0 };
    gpu::write_buffer(gpu_dev, dirty_region_ubo, 0, &dr, sizeof(dr));
}
