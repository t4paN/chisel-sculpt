#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("mirror_project")
#include <cstdio>
#include <cstring>

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
    dirty_verts_capacity = 0;
    smooth_dirty_capacity = 0;
}

// ---------------------------------------------------------------------------
// Async count+list readbacks (dirty list / move-affected list)
// ---------------------------------------------------------------------------

gpu::ReadTicket ComputeState::kick_count_list_read(const gpu::Buffer& buf,
                                                   uint32_t capacity, uint32_t& words) {
    if (!buf.handle || capacity == 0) { words = 0; return 0; }
    words = capacity + 1;   // [count, id0, id1, ...]
    return gpu::read_buffer_async(gpu_dev, buf, 0, (uint64_t)words * sizeof(uint32_t));
}

bool ComputeState::take_count_list_read(gpu::ReadTicket t, uint32_t words,
                                        std::vector<uint32_t>& out) {
    out.clear();
    if (!t || words == 0) return true;
    count_list_scratch.resize(words);
    if (!gpu::ticket_take(gpu_dev, t, count_list_scratch.data(),
                          (uint64_t)words * sizeof(uint32_t)))
        return false;
    uint32_t count = count_list_scratch[0];
    if (count > words - 1) count = words - 1;
    out.assign(count_list_scratch.begin() + 1, count_list_scratch.begin() + 1 + count);
    return true;
}

gpu::ReadTicket ComputeState::kick_dirty_read(uint32_t& words) {
    return kick_count_list_read(smooth_dirty_ssbo, smooth_dirty_capacity, words);
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

bool ComputeState::init_mirror_project() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP,  gpu::Bind::StorageRead,      0 },
        { BIND_MASK,        gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(MirrorProjectParamsGPU) },
    };
    mirror_project_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                  gpu::embedded_shader("mirror_project"), layout, 5);
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
        { BIND_PARAMS,      &cs.mirror_project_ubo, sizeof(MirrorProjectParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(cs.gpu_dev, cs.mirror_project_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(cs.gpu_dev);
    gpu::dispatch(b, cs.mirror_project_pipeline, grp, groups);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_mirror_project_header(const gpu::Buffer& pos_vbo,
                                                  uint32_t vertex_count,
                                                  const gpu::Buffer& list_buf) {
    // Entry count lives in list_buf[0] on the GPU; dispatch worst-case threads
    // (same cost class as the apply kernels, which run full-range per dab) and
    // let overshoot threads early-out on the count.
    dispatch_mirror_project_impl(*this, pos_vbo, vertex_count, list_buf,
                                 0u, 0u, (vertex_count + 255u) / 256u);
}

void ComputeState::dispatch_mirror_project_ids(const gpu::Buffer& pos_vbo,
                                               uint32_t vertex_count, uint32_t id_count) {
    if (id_count == 0) return;
    dispatch_mirror_project_impl(*this, pos_vbo, vertex_count, dirty_verts_ssbo,
                                 1u, id_count, (id_count + 255u) / 256u);
}
