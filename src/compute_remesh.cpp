#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("select_stretched" / ...)
#include <cstdio>
#include <algorithm>

// ---------------------------------------------------------------------------
// Remesh GPU helper kernels — ported onto the gpu:: seam (Seam Step 2b).
//
// These are the isotropic-remesh support passes (per-tri selection, ring grow,
// mirror spread, pinned-boundary detection, smooth weights, the tangential smooth
// ping-pong, and the post-remesh seam snap/weld). The actual edge split/collapse/flip
// topology ops are CPU (remesh.cpp); these run around them. Unlike the brushes these
// are one-shot / user-paced, so CPU readback is allowed (the no-readback rule is
// stroke-only).
//
// Kernel logic lives in the canonical shaders/{glsl,wgsl}/<stem>.* (embedded at build
// time). Each kernel's loose uniforms became a std140 Params UBO at binding 63. The
// remesh-specific scratch SSBOs stay GL-owned (wrapped in gpu::Buffer views at
// dispatch); their alloc/upload/copy/readback stay raw GL.
// ---------------------------------------------------------------------------

namespace {
// All Params blocks are padded to 16 bytes (std140 uniform-block min alignment).
struct SelectStretchedParamsGPU { float target_len; uint32_t tri_count; uint32_t _p0, _p1; };
struct SelectUnmaskedParamsGPU  { uint32_t tri_count, mask_size, _p0, _p1; };
struct GrowSelectionParamsGPU   { uint32_t tri_count, _p0, _p1, _p2; };
struct MirrorSelectionParamsGPU { uint32_t tri_count, vertex_count, _p0, _p1; };
struct FindPinnedParamsGPU      { uint32_t vertex_count; float seam_tol; uint32_t _p0, _p1; };
struct SmoothWeightsParamsGPU   { uint32_t vertex_count; int32_t support_rings; uint32_t _p0, _p1; };
struct SeamSnapParamsGPU        { uint32_t vertex_count, mask_size; float seam_tol, snap_tol; };
struct SeamWeldParamsGPU        { uint32_t vertex_count, mask_size; float weld_tol; uint32_t _p0; };
struct RemeshSmoothParamsGPU    { float lambda, seam_tol; uint32_t vertex_count, _p0; };
static_assert(sizeof(SelectStretchedParamsGPU) == 16, "");
static_assert(sizeof(SelectUnmaskedParamsGPU)  == 16, "");
static_assert(sizeof(GrowSelectionParamsGPU)   == 16, "");
static_assert(sizeof(MirrorSelectionParamsGPU) == 16, "");
static_assert(sizeof(FindPinnedParamsGPU)      == 16, "");
static_assert(sizeof(SmoothWeightsParamsGPU)   == 16, "");
static_assert(sizeof(SeamSnapParamsGPU)        == 16, "");
static_assert(sizeof(SeamWeldParamsGPU)        == 16, "");
static_assert(sizeof(RemeshSmoothParamsGPU)    == 16, "");
}

// ---------------------------------------------------------------------------
// Init — compile each pipeline + allocate its Params UBO.
// ---------------------------------------------------------------------------

bool ComputeState::init_remesh_select() {
    const gpu::BindEntry stretched_layout[] = {
        { BIND_REMESH_IN_POS, gpu::Bind::StorageRead,      0 },
        { BIND_INDICES,       gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL, gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,        gpu::Bind::Uniform,          sizeof(SelectStretchedParamsGPU) },
    };
    remesh_select_stretched_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("select_stretched"), stretched_layout, 4);

    const gpu::BindEntry unmasked_layout[] = {
        { BIND_REMESH_WEIGHTS, gpu::Bind::StorageRead,      0 },
        { BIND_INDICES,        gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,         gpu::Bind::Uniform,          sizeof(SelectUnmaskedParamsGPU) },
    };
    remesh_select_unmasked_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("select_unmasked"), unmasked_layout, 4);

    if (!remesh_select_stretched_pipeline.handle || !remesh_select_unmasked_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh select shaders failed\n");
        return false;
    }
    remesh_select_stretched_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(SelectStretchedParamsGPU), gpu::Usage::Uniform);
    remesh_select_unmasked_ubo  = gpu::create_buffer(gpu_dev, nullptr, sizeof(SelectUnmaskedParamsGPU),  gpu::Usage::Uniform);
    std::printf("[compute] remesh select pipelines compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_grow_selection() {
    const gpu::BindEntry layout[] = {
        { BIND_REMESH_TRISEL_PONG, gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,            gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET,   gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,     gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,             gpu::Bind::Uniform,          sizeof(GrowSelectionParamsGPU) },
    };
    remesh_grow_selection_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("grow_selection"), layout, 6);
    if (!remesh_grow_selection_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh grow_selection shader failed\n");
        return false;
    }
    remesh_grow_selection_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(GrowSelectionParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh grow_selection pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_mirror_selection() {
    const gpu::BindEntry layout[] = {
        { BIND_REMESH_TRISEL_PONG, gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,      gpu::Bind::StorageReadWrite, 0 },
        { BIND_INDICES,            gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET,   gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,     gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP,         gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,             gpu::Bind::Uniform,          sizeof(MirrorSelectionParamsGPU) },
    };
    remesh_mirror_selection_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("mirror_selection"), layout, 7);
    if (!remesh_mirror_selection_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh mirror_selection shader failed\n");
        return false;
    }
    remesh_mirror_selection_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(MirrorSelectionParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh mirror_selection pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_find_pinned() {
    const gpu::BindEntry layout[] = {
        { BIND_REMESH_IN_POS,    gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,    gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_PINNED,    gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(FindPinnedParamsGPU) },
    };
    remesh_find_pinned_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("find_pinned"), layout, 6);
    if (!remesh_find_pinned_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh find_pinned shader failed\n");
        return false;
    }
    remesh_find_pinned_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(FindPinnedParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh find_pinned pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_smooth_weights() {
    const gpu::BindEntry layout[] = {
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_WEIGHTS,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_REMESH_PINNED,    gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,    gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_CORE_SEL,  gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(SmoothWeightsParamsGPU) },
    };
    remesh_smooth_weights_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("smooth_weights"), layout, 7);
    if (!remesh_smooth_weights_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh smooth_weights shader failed\n");
        return false;
    }
    remesh_smooth_weights_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(SmoothWeightsParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh smooth_weights pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_smooth() {
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,        gpu::Bind::StorageReadWrite, 0 },
        { BIND_NORMALS,          gpu::Bind::StorageRead,      0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_IN_POS,    gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_WEIGHTS,   gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_PINNED,    gpu::Bind::StorageRead,      0 },
        { BIND_REMESH_TRISEL,    gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(RemeshSmoothParamsGPU) },
    };
    remesh_smooth_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("remesh_smooth"), layout, 10);
    if (!remesh_smooth_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh smooth shader failed\n");
        return false;
    }
    remesh_smooth_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(RemeshSmoothParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh smooth pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_remesh_seam_snap_weld() {
    const gpu::BindEntry snap_layout[] = {
        { BIND_REMESH_IN_POS,    gpu::Bind::StorageReadWrite, 0 },
        { BIND_ADJACENCY_OFFSET, gpu::Bind::StorageRead,      0 },
        { BIND_ADJACENCY_LIST,   gpu::Bind::StorageRead,      0 },
        { BIND_INDICES,          gpu::Bind::StorageRead,      0 },
        { BIND_MASK,             gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,           gpu::Bind::Uniform,          sizeof(SeamSnapParamsGPU) },
    };
    remesh_seam_snap_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("seam_snap"), snap_layout, 6);

    const gpu::BindEntry weld_layout[] = {
        { BIND_REMESH_IN_POS,  gpu::Bind::StorageRead,      0 },
        { BIND_MASK,           gpu::Bind::StorageRead,      0 },
        { BIND_SEAM_WELD_MAP,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,         gpu::Bind::Uniform,          sizeof(SeamWeldParamsGPU) },
    };
    remesh_seam_weld_pipeline = gpu::create_compute_pipeline(gpu_dev,
        gpu::embedded_shader("seam_weld"), weld_layout, 4);

    if (!remesh_seam_snap_pipeline.handle || !remesh_seam_weld_pipeline.handle) {
        std::fprintf(stderr, "[compute] remesh seam snap/weld shaders failed\n");
        return false;
    }
    remesh_seam_snap_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(SeamSnapParamsGPU), gpu::Usage::Uniform);
    remesh_seam_weld_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(SeamWeldParamsGPU), gpu::Usage::Uniform);
    std::printf("[compute] remesh seam snap/weld pipelines compiled (gpu:: seam)\n");
    return true;
}

void ComputeState::ensure_remesh_smooth_buffers(uint32_t vc, uint32_t tc) {
    auto alloc_ssbo = [](GLuint& ssbo, GLsizeiptr bytes) {
        if (!ssbo) glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
    };
    if (vc > remesh_vert_capacity) {
        remesh_vert_capacity = vc + vc / 4 + 256;
        alloc_ssbo(remesh_ping_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_pong_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_norm_ssbo,    (GLsizeiptr)remesh_vert_capacity * 3 * sizeof(float));
        alloc_ssbo(remesh_weights_ssbo, (GLsizeiptr)remesh_vert_capacity     * sizeof(float));
        alloc_ssbo(remesh_pinned_ssbo,  (GLsizeiptr)remesh_vert_capacity     * sizeof(uint32_t));
    }
    if (tc > remesh_tri_capacity) {
        remesh_tri_capacity = tc + tc / 4 + 256;
        alloc_ssbo(remesh_trisel_ssbo,     (GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
        alloc_ssbo(remesh_indices_ssbo,    (GLsizeiptr)remesh_tri_capacity * 3 * sizeof(uint32_t));
        alloc_ssbo(remesh_core_sel_ssbo,   (GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
        alloc_ssbo(remesh_trisel_pong_ssbo,(GLsizeiptr)remesh_tri_capacity     * sizeof(uint32_t));
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// ---------------------------------------------------------------------------
// Dispatch — uploads/copies/readback stay raw GL; the compute step runs through
// the seam. Wrap each GL-owned SSBO in a transient gpu::Buffer view at dispatch.
// (sizes are best-effort for the WebGPU min-size guard later; the GL backend
// whole-buffer-binds and ignores them. 0 = unguarded.)
// ---------------------------------------------------------------------------

void ComputeState::dispatch_select_stretched(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float target_len)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    SelectStretchedParamsGPU u = {};
    u.target_len = target_len; u.tri_count = tri_count;
    gpu::write_buffer(gpu_dev, remesh_select_stretched_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{ (uint64_t)vertex_count*3u*sizeof(float),    remesh_ping_ssbo };
    gpu::Buffer idxView{ (uint64_t)tri_count*3u*sizeof(uint32_t),    remesh_indices_ssbo };
    gpu::Buffer selView{ (uint64_t)tri_count*sizeof(uint32_t),       remesh_trisel_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_REMESH_IN_POS, &posView, posView.size },
        { BIND_INDICES,       &idxView, idxView.size },
        { BIND_REMESH_TRISEL, &selView, selView.size },
        { BIND_PARAMS,        &remesh_select_stretched_ubo, sizeof(SelectStretchedParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_select_stretched_pipeline, bg, 4);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, remesh_select_stretched_pipeline, grp, (tri_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_select_unmasked(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* mask, uint32_t mask_size)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);
    if (mask && mask_size > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mask_size*sizeof(float), mask);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    SelectUnmaskedParamsGPU u = {};
    u.tri_count = tri_count; u.mask_size = mask_size;
    gpu::write_buffer(gpu_dev, remesh_select_unmasked_ubo, 0, &u, sizeof(u));

    gpu::Buffer maskView{ (uint64_t)(mask_size ? mask_size : 1u)*sizeof(float), remesh_weights_ssbo };
    gpu::Buffer idxView{  (uint64_t)tri_count*3u*sizeof(uint32_t),              remesh_indices_ssbo };
    gpu::Buffer selView{  (uint64_t)tri_count*sizeof(uint32_t),                 remesh_trisel_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_REMESH_WEIGHTS, &maskView, maskView.size },
        { BIND_INDICES,        &idxView,  idxView.size },
        { BIND_REMESH_TRISEL,  &selView,  selView.size },
        { BIND_PARAMS,         &remesh_select_unmasked_ubo, sizeof(SelectUnmaskedParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_select_unmasked_pipeline, bg, 4);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, remesh_select_unmasked_pipeline, grp, (tri_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_grow_selection(
    uint32_t vertex_count, uint32_t tri_count, int rings)
{
    if (rings <= 0 || tri_count == 0 || !remesh_grow_selection_pipeline.handle) return;
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    GrowSelectionParamsGPU u = {};
    u.tri_count = tri_count;
    gpu::write_buffer(gpu_dev, remesh_grow_selection_ubo, 0, &u, sizeof(u));

    gpu::Buffer pongView{ (uint64_t)tri_count*sizeof(uint32_t),     remesh_trisel_pong_ssbo };
    gpu::Buffer selView{  (uint64_t)tri_count*sizeof(uint32_t),     remesh_trisel_ssbo };
    gpu::Buffer idxView{  (uint64_t)tri_count*3u*sizeof(uint32_t),  remesh_indices_ssbo };
    gpu::Buffer offView{  0, adjacency_offset_ssbo };
    gpu::Buffer listView{ 0, adjacency_list_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_REMESH_TRISEL_PONG, &pongView, pongView.size },
        { BIND_REMESH_TRISEL,      &selView,  selView.size },
        { BIND_INDICES,            &idxView,  idxView.size },
        { BIND_ADJACENCY_OFFSET,   &offView,  offView.size },
        { BIND_ADJACENCY_LIST,     &listView, listView.size },
        { BIND_PARAMS,             &remesh_grow_selection_ubo, sizeof(GrowSelectionParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_grow_selection_pipeline, bg, 6);

    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);
    for (int r = 0; r < rings; r++) {
        // Snapshot current selection into pong (the kernel's input) — raw GL copy.
        glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
        glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_trisel_pong_ssbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, remesh_grow_selection_pipeline, grp, (tri_count + 255u) / 256u);
        gpu::submit(b);
    }
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_mirror_selection(uint32_t vertex_count, uint32_t tri_count)
{
    if (tri_count == 0 || !remesh_mirror_selection_pipeline.handle) return;
    if (mirror_map_vertex_count == 0 || !mirror_map_ssbo.handle) return;
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);
    glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_trisel_pong_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    MirrorSelectionParamsGPU u = {};
    u.tri_count = tri_count; u.vertex_count = vertex_count;
    gpu::write_buffer(gpu_dev, remesh_mirror_selection_ubo, 0, &u, sizeof(u));

    gpu::Buffer pongView{   (uint64_t)tri_count*sizeof(uint32_t),    remesh_trisel_pong_ssbo };
    gpu::Buffer selView{    (uint64_t)tri_count*sizeof(uint32_t),    remesh_trisel_ssbo };
    gpu::Buffer idxView{    (uint64_t)tri_count*3u*sizeof(uint32_t), remesh_indices_ssbo };
    gpu::Buffer offView{    0, adjacency_offset_ssbo };
    gpu::Buffer listView{   0, adjacency_list_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_REMESH_TRISEL_PONG, &pongView,   pongView.size },
        { BIND_REMESH_TRISEL,      &selView,    selView.size },
        { BIND_INDICES,            &idxView,    idxView.size },
        { BIND_ADJACENCY_OFFSET,   &offView,    offView.size },
        { BIND_ADJACENCY_LIST,     &listView,   listView.size },
        { BIND_MIRROR_MAP,         &mirror_map_ssbo, 0 },
        { BIND_PARAMS,             &remesh_mirror_selection_ubo, sizeof(MirrorSelectionParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_mirror_selection_pipeline, bg, 7);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, remesh_mirror_selection_pipeline, grp, (tri_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::snapshot_core_sel(uint32_t tri_count)
{
    if (tri_count == 0) return;
    GLsizeiptr bytes = (GLsizeiptr)tri_count * sizeof(uint32_t);
    glBindBuffer(GL_COPY_READ_BUFFER,  remesh_trisel_ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, remesh_core_sel_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::readback_trisel(uint32_t tri_count, std::vector<uint32_t>& out)
{
    out.resize(tri_count);
    if (tri_count == 0) return;
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_trisel_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*sizeof(uint32_t), out.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_find_pinned(
    uint32_t vertex_count, uint32_t tri_count,
    const float* pos_x, const float* pos_y, const float* pos_z,
    float seam_tol,
    std::vector<uint32_t>& out_pinned)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    FindPinnedParamsGPU u = {};
    u.vertex_count = vertex_count; u.seam_tol = seam_tol;
    gpu::write_buffer(gpu_dev, remesh_find_pinned_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{    (uint64_t)vertex_count*3u*sizeof(float), remesh_ping_ssbo };
    gpu::Buffer offView{    0, adjacency_offset_ssbo };
    gpu::Buffer listView{   0, adjacency_list_ssbo };
    gpu::Buffer selView{    (uint64_t)tri_count*sizeof(uint32_t),    remesh_trisel_ssbo };
    gpu::Buffer pinnedView{ (uint64_t)vertex_count*sizeof(uint32_t), remesh_pinned_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_REMESH_IN_POS,    &posView,    posView.size },
        { BIND_ADJACENCY_OFFSET, &offView,    offView.size },
        { BIND_ADJACENCY_LIST,   &listView,   listView.size },
        { BIND_REMESH_TRISEL,    &selView,    selView.size },
        { BIND_REMESH_PINNED,    &pinnedView, pinnedView.size },
        { BIND_PARAMS,           &remesh_find_pinned_ubo, sizeof(FindPinnedParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_find_pinned_pipeline, bg, 6);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, remesh_find_pinned_pipeline, grp, (vertex_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);

    out_pinned.resize(vertex_count);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_pinned_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*sizeof(uint32_t), out_pinned.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::dispatch_compute_smooth_weights(
    uint32_t vertex_count, uint32_t tri_count,
    int support_rings)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    SmoothWeightsParamsGPU u = {};
    u.vertex_count = vertex_count; u.support_rings = support_rings;
    gpu::write_buffer(gpu_dev, remesh_smooth_weights_ubo, 0, &u, sizeof(u));

    gpu::Buffer offView{     0, adjacency_offset_ssbo };
    gpu::Buffer listView{    0, adjacency_list_ssbo };
    gpu::Buffer weightView{  (uint64_t)vertex_count*sizeof(float),    remesh_weights_ssbo };
    gpu::Buffer pinnedView{  (uint64_t)vertex_count*sizeof(uint32_t), remesh_pinned_ssbo };
    gpu::Buffer fullSelView{ (uint64_t)tri_count*sizeof(uint32_t),    remesh_trisel_ssbo };
    gpu::Buffer coreSelView{ (uint64_t)tri_count*sizeof(uint32_t),    remesh_core_sel_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_ADJACENCY_OFFSET, &offView,     offView.size },
        { BIND_ADJACENCY_LIST,   &listView,    listView.size },
        { BIND_REMESH_WEIGHTS,   &weightView,  weightView.size },
        { BIND_REMESH_PINNED,    &pinnedView,  pinnedView.size },
        { BIND_REMESH_TRISEL,    &fullSelView, fullSelView.size },
        { BIND_REMESH_CORE_SEL,  &coreSelView, coreSelView.size },
        { BIND_PARAMS,           &remesh_smooth_weights_ubo, sizeof(SmoothWeightsParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, remesh_smooth_weights_pipeline, bg, 7);
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, remesh_smooth_weights_pipeline, grp, (vertex_count + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_remesh_smooth(
    uint32_t vertex_count, uint32_t tri_count,
    const uint32_t* mesh_indices,
    const float* pos_x, const float* pos_y, const float* pos_z,
    const float* norm_x, const float* norm_y, const float* norm_z,
    const std::vector<float>& weights,
    const std::vector<uint32_t>& pinned,
    float lambda,
    float seam_tol,
    float* out_pos_x, float* out_pos_y, float* out_pos_z,
    bool weights_on_gpu)
{
    ensure_remesh_smooth_buffers(vertex_count, tri_count);

    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = norm_x[i];
        readback_buf[i*3+1] = norm_y[i];
        readback_buf[i*3+2] = norm_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_norm_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());

    if (!weights_on_gpu) {
        std::vector<float> w_buf(vertex_count, 1.0f);
        if (!weights.empty()) {
            uint32_t n = (uint32_t)weights.size();
            for (uint32_t i = 0; i < vertex_count; i++)
                w_buf[i] = (i < n) ? weights[i] : 1.0f;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*sizeof(float), w_buf.data());
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_pinned_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    std::min<uint32_t>(vertex_count, (uint32_t)pinned.size()) * sizeof(uint32_t),
                    pinned.data());

    // tri_sel is assumed already in remesh_trisel_ssbo (set by select+grow+mirror).

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_indices_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, tri_count*3*sizeof(uint32_t), mesh_indices);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    RemeshSmoothParamsGPU u = {};
    u.lambda = lambda; u.seam_tol = seam_tol; u.vertex_count = vertex_count;
    gpu::write_buffer(gpu_dev, remesh_smooth_ubo, 0, &u, sizeof(u));

    // Constant inputs across the ping-pong (slots 0 / 13 swap between ping/pong).
    gpu::Buffer normView{ (uint64_t)vertex_count*3u*sizeof(float), remesh_norm_ssbo };
    gpu::Buffer idxView{  (uint64_t)tri_count*3u*sizeof(uint32_t), remesh_indices_ssbo };
    gpu::Buffer offView{  0, adjacency_offset_ssbo };
    gpu::Buffer listView{ 0, adjacency_list_ssbo };
    gpu::Buffer wView{    (uint64_t)vertex_count*sizeof(float),    remesh_weights_ssbo };
    gpu::Buffer pinView{  (uint64_t)vertex_count*sizeof(uint32_t), remesh_pinned_ssbo };
    gpu::Buffer selView{  (uint64_t)tri_count*sizeof(uint32_t),    remesh_trisel_ssbo };
    gpu::Buffer pingView{ (uint64_t)vertex_count*3u*sizeof(float), remesh_ping_ssbo };
    gpu::Buffer pongView{ (uint64_t)vertex_count*3u*sizeof(float), remesh_pong_ssbo };

    auto make_bg = [&](gpu::Buffer& inBuf, gpu::Buffer& outBuf) {
        const gpu::BindBufferEntry bg[] = {
            { BIND_POSITIONS,        &outBuf,  outBuf.size },   // out_pos (read_write)
            { BIND_NORMALS,          &normView, normView.size },
            { BIND_INDICES,          &idxView,  idxView.size },
            { BIND_ADJACENCY_OFFSET, &offView,  offView.size },
            { BIND_ADJACENCY_LIST,   &listView, listView.size },
            { BIND_REMESH_IN_POS,    &inBuf,   inBuf.size },    // in_pos (read)
            { BIND_REMESH_WEIGHTS,   &wView,    wView.size },
            { BIND_REMESH_PINNED,    &pinView,  pinView.size },
            { BIND_REMESH_TRISEL,    &selView,  selView.size },
            { BIND_PARAMS,           &remesh_smooth_ubo, sizeof(RemeshSmoothParamsGPU) },
        };
        return gpu::create_bind_group(gpu_dev, remesh_smooth_pipeline, bg, 10);
    };
    gpu::BindGroup grp_a = make_bg(pingView, pongView);  // in=ping, out=pong
    gpu::BindGroup grp_b = make_bg(pongView, pingView);  // in=pong, out=ping

    // Multi-step smoothing with ping-pong. The shader copies in_pos→out_pos for
    // pinned / out-of-region / weightless verts, so swapping is safe — every vertex
    // is written each step. After the loop, `out_buf` holds the final result.
    static constexpr int NUM_SMOOTH_STEPS = 3;
    uint32_t groups = (vertex_count + 255u) / 256u;
    GLuint out_buf_gl = remesh_ping_ssbo;  // tracks which GL buffer holds the result
    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    for (int step = 0; step < NUM_SMOOTH_STEPS; step++) {
        // step even: in=ping → out=pong (grp_a); step odd: in=pong → out=ping (grp_b)
        gpu::dispatch(b, remesh_smooth_pipeline, (step & 1) == 0 ? grp_a : grp_b, groups);
        out_buf_gl = (step & 1) == 0 ? remesh_pong_ssbo : remesh_ping_ssbo;
    }
    gpu::submit(b);
    gpu::release_bind_group(grp_a);
    gpu::release_bind_group(grp_b);

    readback_buf.resize(vertex_count * 3);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, out_buf_gl);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count*3*sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    for (uint32_t i = 0; i < vertex_count; i++) {
        out_pos_x[i] = readback_buf[i*3+0];
        out_pos_y[i] = readback_buf[i*3+1];
        out_pos_z[i] = readback_buf[i*3+2];
    }
}

void ComputeState::dispatch_seam_snap_weld(
    uint32_t vertex_count,
    float* pos_x, const float* pos_y, const float* pos_z,
    const float* mask_data, uint32_t mask_size,
    float seam_tol, float snap_tol, float weld_tol,
    std::vector<uint32_t>& out_merge_map)
{
    ensure_remesh_smooth_buffers(vertex_count, 1);

    // Interleave positions into ping SSBO
    readback_buf.resize(vertex_count * 3);
    for (uint32_t i = 0; i < vertex_count; i++) {
        readback_buf[i*3+0] = pos_x[i];
        readback_buf[i*3+1] = pos_y[i];
        readback_buf[i*3+2] = pos_z[i];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count * 3 * sizeof(float), readback_buf.data());

    // Upload mask (reuse remesh_weights_ssbo as scratch)
    if (mask_data && mask_size > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_weights_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mask_size * sizeof(float), mask_data);
    }

    // Allocate weld merge map SSBO
    if (!seam_weld_map_ssbo) glGenBuffers(1, &seam_weld_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seam_weld_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertex_count * sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    uint32_t groups = (vertex_count + 255u) / 256u;

    gpu::Buffer posView{    (uint64_t)vertex_count*3u*sizeof(float),               remesh_ping_ssbo };
    gpu::Buffer maskView{   (uint64_t)(mask_size ? mask_size : 1u)*sizeof(float),  remesh_weights_ssbo };
    gpu::Buffer offView{    0, adjacency_offset_ssbo };
    gpu::Buffer listView{   0, adjacency_list_ssbo };
    gpu::Buffer idxView{    0, remesh_indices_ssbo };
    gpu::Buffer mergeView{  (uint64_t)vertex_count*sizeof(uint32_t), seam_weld_map_ssbo };

    // --- Pass 1: seam snap ---
    SeamSnapParamsGPU su = {};
    su.vertex_count = vertex_count; su.mask_size = mask_size;
    su.seam_tol = seam_tol; su.snap_tol = snap_tol;
    gpu::write_buffer(gpu_dev, remesh_seam_snap_ubo, 0, &su, sizeof(su));
    const gpu::BindBufferEntry snap_bg[] = {
        { BIND_REMESH_IN_POS,    &posView,  posView.size },
        { BIND_ADJACENCY_OFFSET, &offView,  offView.size },
        { BIND_ADJACENCY_LIST,   &listView, listView.size },
        { BIND_INDICES,          &idxView,  idxView.size },
        { BIND_MASK,             &maskView, maskView.size },
        { BIND_PARAMS,           &remesh_seam_snap_ubo, sizeof(SeamSnapParamsGPU) },
    };
    gpu::BindGroup snap_grp = gpu::create_bind_group(gpu_dev, remesh_seam_snap_pipeline, snap_bg, 6);
    {
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, remesh_seam_snap_pipeline, snap_grp, groups);
        gpu::submit(b);
    }
    gpu::release_bind_group(snap_grp);

    // --- Pass 2: seam weld --- (reads the snapped positions)
    SeamWeldParamsGPU wu = {};
    wu.vertex_count = vertex_count; wu.mask_size = mask_size; wu.weld_tol = weld_tol;
    gpu::write_buffer(gpu_dev, remesh_seam_weld_ubo, 0, &wu, sizeof(wu));
    const gpu::BindBufferEntry weld_bg[] = {
        { BIND_REMESH_IN_POS, &posView,   posView.size },
        { BIND_MASK,          &maskView,  maskView.size },
        { BIND_SEAM_WELD_MAP, &mergeView, mergeView.size },
        { BIND_PARAMS,        &remesh_seam_weld_ubo, sizeof(SeamWeldParamsGPU) },
    };
    gpu::BindGroup weld_grp = gpu::create_bind_group(gpu_dev, remesh_seam_weld_pipeline, weld_bg, 4);
    {
        gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
        gpu::dispatch(b, remesh_seam_weld_pipeline, weld_grp, groups);
        gpu::submit(b);
    }
    gpu::release_bind_group(weld_grp);

    // Readback snapped positions and count how many changed
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, remesh_ping_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count * 3 * sizeof(float), readback_buf.data());
    uint32_t n_snapped = 0;
    for (uint32_t i = 0; i < vertex_count; i++) {
        float new_x = readback_buf[i*3+0];
        if (new_x == 0.0f && pos_x[i] != 0.0f) n_snapped++;
        pos_x[i] = new_x;
    }

    // Readback merge map
    out_merge_map.resize(vertex_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, seam_weld_map_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_count * sizeof(uint32_t), out_merge_map.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Chase merge chains to roots
    for (uint32_t v = 0; v < vertex_count; v++) {
        uint32_t root = v;
        while (out_merge_map[root] != root) root = out_merge_map[root];
        out_merge_map[v] = root;
    }

    uint32_t n_welded = 0;
    for (uint32_t v = 0; v < vertex_count; v++)
        if (out_merge_map[v] != v) n_welded++;

    if (n_snapped > 0 || n_welded > 0)
        std::printf("[mirror] GPU seam: snapped %u, welded %u verts\n", n_snapped, n_welded);
}
