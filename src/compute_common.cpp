#include "compute.h"
#include <cstdio>
#include <cstring>

ComputeState::ComputeState()
    : supported(false)
    , has_native_float_atomics(false)
    , max_workgroup_size(0)
    , max_workgroup_invocations(0)
    , max_ssbo_bindings(0)
    , accum_ssbo(0)
    , accum_vertex_count(0)
    , accum_sym_ssbo(0)
    , draw_accum_program(0)
    , draw_accum_symmetrize_program(0)
    , stroke_norm_ssbo(0)
    , stroke_norm_capacity(0)
    , draw_apply_program(0)
    , draw_mirror_apply_program(0)
    , smooth_accum_program(0)
    , smooth_apply_program(0)
    , smooth_mirror_apply_program(0)
    , stroke_smooth_apply_program(0)
    , crease_accum_program(0)
    , pinch_accum_program(0)
    , mask_paint_program(0)
    , color_paint_program(0)
    , move_capture_program(0)
    , move_weight_smooth_program(0)
    , move_apply_program(0)
    , move_affected_ssbo(0)
    , move_weights_ssbo(0)
    , move_weights_pong_ssbo(0)
    , move_init_ssbo(0)
    , move_buffers_capacity(0)
    , compute_normals_program(0)
    , adjacency_offset_ssbo(0)
    , adjacency_list_ssbo(0)
    , adjacency_vertex_count(0)
    , mirror_map_ssbo(0)
    , mirror_map_vertex_count(0)
    , dirty_verts_ssbo(0)
    , dirty_verts_capacity(0)
    , smooth_dirty_ssbo(0)
    , smooth_dirty_capacity(0)
    , mask_ssbo(0)
    , color_ssbo(0)
    , remesh_select_stretched_program(0)
    , remesh_select_unmasked_program(0)
    , remesh_grow_selection_program(0)
    , remesh_mirror_selection_program(0)
    , remesh_find_pinned_program(0)
    , remesh_smooth_weights_program(0)
    , remesh_core_sel_ssbo(0)
    , remesh_trisel_pong_ssbo(0)
    , remesh_smooth_program(0)
    , remesh_ping_ssbo(0)
    , remesh_pong_ssbo(0)
    , remesh_norm_ssbo(0)
    , remesh_weights_ssbo(0)
    , remesh_pinned_ssbo(0)
    , remesh_trisel_ssbo(0)
    , remesh_indices_ssbo(0)
    , remesh_seam_snap_program(0)
    , remesh_seam_weld_program(0)
    , seam_weld_map_ssbo(0)
    , remesh_vert_capacity(0)
    , remesh_tri_capacity(0)
{}

bool ComputeState::init() {
    supported = false;

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
}

GLuint ComputeState::compile_program(const char* src) const {
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
}

void ComputeState::cleanup() {
    if (accum_ssbo) { glDeleteBuffers(1, &accum_ssbo); accum_ssbo = 0; }
    if (accum_sym_ssbo) { glDeleteBuffers(1, &accum_sym_ssbo); accum_sym_ssbo = 0; }
    if (stroke_norm_ssbo) { glDeleteBuffers(1, &stroke_norm_ssbo); stroke_norm_ssbo = 0; }
    stroke_norm_capacity = 0;
    if (draw_accum_program) { glDeleteProgram(draw_accum_program); draw_accum_program = 0; }
    if (draw_accum_symmetrize_program) { glDeleteProgram(draw_accum_symmetrize_program); draw_accum_symmetrize_program = 0; }
    if (draw_apply_program) { glDeleteProgram(draw_apply_program); draw_apply_program = 0; }
    if (draw_mirror_apply_program) { glDeleteProgram(draw_mirror_apply_program); draw_mirror_apply_program = 0; }
    if (mirror_map_ssbo) { glDeleteBuffers(1, &mirror_map_ssbo); mirror_map_ssbo = 0; }
    if (smooth_accum_program) { glDeleteProgram(smooth_accum_program); smooth_accum_program = 0; }
    if (smooth_apply_program) { glDeleteProgram(smooth_apply_program); smooth_apply_program = 0; }
    if (smooth_mirror_apply_program) { glDeleteProgram(smooth_mirror_apply_program); smooth_mirror_apply_program = 0; }
    if (stroke_smooth_apply_program) { glDeleteProgram(stroke_smooth_apply_program); stroke_smooth_apply_program = 0; }
    if (crease_accum_program) { glDeleteProgram(crease_accum_program); crease_accum_program = 0; }
    if (pinch_accum_program) { glDeleteProgram(pinch_accum_program); pinch_accum_program = 0; }
    if (mask_paint_program) { glDeleteProgram(mask_paint_program); mask_paint_program = 0; }
    if (color_paint_program) { glDeleteProgram(color_paint_program); color_paint_program = 0; }
    if (move_capture_program)       { glDeleteProgram(move_capture_program);       move_capture_program       = 0; }
    if (move_weight_smooth_program) { glDeleteProgram(move_weight_smooth_program); move_weight_smooth_program = 0; }
    if (move_apply_program)         { glDeleteProgram(move_apply_program);         move_apply_program         = 0; }
    if (move_affected_ssbo)         { glDeleteBuffers(1, &move_affected_ssbo);     move_affected_ssbo         = 0; }
    if (move_weights_ssbo)          { glDeleteBuffers(1, &move_weights_ssbo);      move_weights_ssbo          = 0; }
    if (move_weights_pong_ssbo)     { glDeleteBuffers(1, &move_weights_pong_ssbo); move_weights_pong_ssbo     = 0; }
    if (move_init_ssbo)             { glDeleteBuffers(1, &move_init_ssbo);         move_init_ssbo             = 0; }
    move_buffers_capacity = 0;
    if (compute_normals_program) { glDeleteProgram(compute_normals_program); compute_normals_program = 0; }
    if (adjacency_offset_ssbo) { glDeleteBuffers(1, &adjacency_offset_ssbo); adjacency_offset_ssbo = 0; }
    if (adjacency_list_ssbo) { glDeleteBuffers(1, &adjacency_list_ssbo); adjacency_list_ssbo = 0; }
    if (dirty_verts_ssbo) { glDeleteBuffers(1, &dirty_verts_ssbo); dirty_verts_ssbo = 0; }
    if (smooth_dirty_ssbo) { glDeleteBuffers(1, &smooth_dirty_ssbo); smooth_dirty_ssbo = 0; }
    if (remesh_select_stretched_program) { glDeleteProgram(remesh_select_stretched_program); remesh_select_stretched_program = 0; }
    if (remesh_select_unmasked_program)  { glDeleteProgram(remesh_select_unmasked_program);  remesh_select_unmasked_program  = 0; }
    if (remesh_grow_selection_program)   { glDeleteProgram(remesh_grow_selection_program);   remesh_grow_selection_program   = 0; }
    if (remesh_mirror_selection_program) { glDeleteProgram(remesh_mirror_selection_program); remesh_mirror_selection_program = 0; }
    if (remesh_find_pinned_program)      { glDeleteProgram(remesh_find_pinned_program);      remesh_find_pinned_program      = 0; }
    if (remesh_smooth_weights_program)   { glDeleteProgram(remesh_smooth_weights_program);   remesh_smooth_weights_program   = 0; }
    if (remesh_core_sel_ssbo)            { glDeleteBuffers(1, &remesh_core_sel_ssbo);         remesh_core_sel_ssbo            = 0; }
    if (remesh_trisel_pong_ssbo)         { glDeleteBuffers(1, &remesh_trisel_pong_ssbo);     remesh_trisel_pong_ssbo         = 0; }
    if (remesh_seam_snap_program) { glDeleteProgram(remesh_seam_snap_program); remesh_seam_snap_program = 0; }
    if (remesh_seam_weld_program) { glDeleteProgram(remesh_seam_weld_program); remesh_seam_weld_program = 0; }
    if (seam_weld_map_ssbo)       { glDeleteBuffers(1, &seam_weld_map_ssbo);  seam_weld_map_ssbo       = 0; }
    if (remesh_smooth_program) { glDeleteProgram(remesh_smooth_program); remesh_smooth_program = 0; }
    if (remesh_ping_ssbo)     { glDeleteBuffers(1, &remesh_ping_ssbo);     remesh_ping_ssbo     = 0; }
    if (remesh_pong_ssbo)     { glDeleteBuffers(1, &remesh_pong_ssbo);     remesh_pong_ssbo     = 0; }
    if (remesh_norm_ssbo)     { glDeleteBuffers(1, &remesh_norm_ssbo);     remesh_norm_ssbo     = 0; }
    if (remesh_weights_ssbo)  { glDeleteBuffers(1, &remesh_weights_ssbo);  remesh_weights_ssbo  = 0; }
    if (remesh_pinned_ssbo)   { glDeleteBuffers(1, &remesh_pinned_ssbo);   remesh_pinned_ssbo   = 0; }
    if (remesh_trisel_ssbo)   { glDeleteBuffers(1, &remesh_trisel_ssbo);   remesh_trisel_ssbo   = 0; }
    if (remesh_indices_ssbo)  { glDeleteBuffers(1, &remesh_indices_ssbo);  remesh_indices_ssbo  = 0; }
    remesh_vert_capacity = remesh_tri_capacity = 0;
    accum_vertex_count = 0;
    adjacency_vertex_count = 0;
    dirty_verts_capacity = 0;
    smooth_dirty_capacity = 0;
}
