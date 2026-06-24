#include "compute.h"
#include "gpu_shaders_generated.h"   // gpu::embedded_shader("draw_*")
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Draw / inflate brush — ported onto the gpu:: seam (Seam Step 2b). Kernel logic
// lives in the canonical shaders/{glsl,wgsl}/draw_* (embedded at build time); this
// file drives the seam (pipelines, Params UBOs, bind groups, dispatch). The accum /
// stroke-normal / mirror-map buffers remain GL-owned for now (they migrate to
// gpu::Buffer with the buffer-ownership pass), wrapped in views at dispatch.
// ---------------------------------------------------------------------------

namespace {
// 112-byte std140 block, byte-identical to draw_accum.{comp,wgsl}'s Params.
struct DrawAccumParamsGPU {
    float    anchor_a[3];        float world_radius;
    float    anchor_b[3];        float disp_amount;
    float    view_a[3];          float hardness;
    float    view_b[3];          float facing_threshold;
    float    anchor_normal_a[3]; uint32_t use_b;
    float    anchor_normal_b[3]; uint32_t inflate;
    uint32_t vertex_count;       uint32_t _pad0; uint32_t _pad1; uint32_t _pad2;
};
static_assert(sizeof(DrawAccumParamsGPU) == 112, "draw_accum Params UBO must be 112 bytes");

// 16-byte std140 block shared by draw_apply / symmetrize / mirror_apply.
struct VCountParamsGPU {
    uint32_t vertex_count; uint32_t _pad0; uint32_t _pad1; uint32_t _pad2;
};
static_assert(sizeof(VCountParamsGPU) == 16, "vcount Params UBO must be 16 bytes");

// Lazily create the shared 16-byte vcount UBO (whichever vcount kernel inits first).
void ensure_vcount_ubo(ComputeState& cs) {
    if (!cs.draw_vcount_ubo.handle)
        cs.draw_vcount_ubo = gpu::create_buffer(cs.gpu_dev, nullptr,
                                                sizeof(VCountParamsGPU), gpu::Usage::Uniform);
}
}

// ---------------------------------------------------------------------------
// Buffer management (raw GL — these buffers are still GL-owned; see header note)
// ---------------------------------------------------------------------------

void ComputeState::ensure_accum_buffer(uint32_t vertex_count) {
    if (accum_ssbo && accum_vertex_count >= vertex_count) return;
    if (accum_ssbo) glDeleteBuffers(1, &accum_ssbo);
    glGenBuffers(1, &accum_ssbo);
    GLsizeiptr size = (GLsizeiptr)vertex_count * 4 * sizeof(uint32_t);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

    if (accum_sym_ssbo) glDeleteBuffers(1, &accum_sym_ssbo);
    glGenBuffers(1, &accum_sym_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_sym_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    accum_vertex_count = vertex_count;
}

void ComputeState::clear_accum_buffer() {
    if (!accum_ssbo || accum_vertex_count == 0) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    uint32_t zero = 0;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::snapshot_stroke_normals(GLuint norm_vbo, uint32_t vertex_count) {
    // Working buffer holds only the active entity at offset 0 — copy [0, vertex_count).
    if (stroke_norm_capacity < vertex_count || !stroke_norm_ssbo) {
        if (stroke_norm_ssbo) { glDeleteBuffers(1, &stroke_norm_ssbo); stroke_norm_ssbo = 0; }
        glGenBuffers(1, &stroke_norm_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, stroke_norm_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)vertex_count * 3 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        stroke_norm_capacity = vertex_count;
    }
    GLsizeiptr byte_size = (GLsizeiptr)vertex_count * 3 * sizeof(float);
    glBindBuffer(GL_COPY_READ_BUFFER,  norm_vbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, stroke_norm_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, byte_size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void ComputeState::readback_accum(uint32_t vertex_count) {
    readback_buf.resize(vertex_count * 4);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       vertex_count * 4 * sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::upload_accum(const float* disp_x, const float* disp_y,
                                 const float* disp_z, const float* disp_weight,
                                 uint32_t vertex_count) {
    readback_buf.resize(vertex_count * 4);
    for (uint32_t v = 0; v < vertex_count; v++) {
        readback_buf[v * 4 + 0] = disp_x[v];
        readback_buf[v * 4 + 1] = disp_y[v];
        readback_buf[v * 4 + 2] = disp_z[v];
        readback_buf[v * 4 + 3] = disp_weight[v];
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, accum_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    vertex_count * 4 * sizeof(float), readback_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ComputeState::upload_mirror_map(const std::vector<uint32_t>& map) {
    if (map.empty()) {
        mirror_map_vertex_count = 0;
        return;
    }

    if (!mirror_map_ssbo) glGenBuffers(1, &mirror_map_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mirror_map_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, map.size() * sizeof(uint32_t), map.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    mirror_map_vertex_count = map.size();

    std::printf("[compute] mirror_map uploaded: %u vertices\n", mirror_map_vertex_count);
}

// ---------------------------------------------------------------------------
// Pipeline init (gpu:: seam)
// ---------------------------------------------------------------------------

bool ComputeState::init_draw_accum() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS, gpu::Bind::StorageRead,      0 },
        { BIND_NORMALS,   gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM,     gpu::Bind::StorageReadWrite, 0 },
        { BIND_PARAMS,    gpu::Bind::Uniform,          sizeof(DrawAccumParamsGPU) },
    };
    draw_accum_pipeline = gpu::create_compute_pipeline(gpu_dev, gpu::embedded_shader("draw_accum"),
                                                       layout, 4);
    if (!draw_accum_pipeline.handle) {
        std::printf("[compute] draw_accum pipeline failed to compile\n");
        return false;
    }
    draw_accum_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DrawAccumParamsGPU),
                                        gpu::Usage::Uniform);
    std::printf("[compute] draw_accum pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_draw_apply() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_ACCUM,       gpu::Bind::StorageRead,      0 },
        { BIND_DIRTY_VERTS, gpu::Bind::StorageReadWrite, 0 },
        { BIND_MASK,        gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(VCountParamsGPU) },
    };
    draw_apply_pipeline = gpu::create_compute_pipeline(gpu_dev, gpu::embedded_shader("draw_apply"),
                                                       layout, 5);
    if (!draw_apply_pipeline.handle) {
        std::printf("[compute] draw_apply pipeline failed to compile\n");
        return false;
    }
    ensure_vcount_ubo(*this);
    std::printf("[compute] draw_apply pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_draw_accum_symmetrize() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_ACCUM,       gpu::Bind::StorageRead,      0 },
        { BIND_ACCUM_SYM,   gpu::Bind::StorageReadWrite, 0 },
        { BIND_MIRROR_MAP,  gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,      gpu::Bind::Uniform,          sizeof(VCountParamsGPU) },
    };
    draw_symmetrize_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                   gpu::embedded_shader("draw_accum_symmetrize"), layout, 4);
    if (!draw_symmetrize_pipeline.handle) {
        std::printf("[compute] draw_accum_symmetrize pipeline failed to compile\n");
        return false;
    }
    ensure_vcount_ubo(*this);
    std::printf("[compute] draw_accum_symmetrize pipeline compiled (gpu:: seam)\n");
    return true;
}

bool ComputeState::init_draw_mirror_apply() {
    if (!supported) return false;
    const gpu::BindEntry layout[] = {
        { BIND_POSITIONS,  gpu::Bind::StorageReadWrite, 0 },
        { BIND_ACCUM,      gpu::Bind::StorageRead,      0 },
        { BIND_MIRROR_MAP, gpu::Bind::StorageRead,      0 },
        { BIND_MASK,       gpu::Bind::StorageRead,      0 },
        { BIND_PARAMS,     gpu::Bind::Uniform,          sizeof(VCountParamsGPU) },
    };
    draw_mirror_apply_pipeline = gpu::create_compute_pipeline(gpu_dev,
                                     gpu::embedded_shader("draw_mirror_apply"), layout, 5);
    if (!draw_mirror_apply_pipeline.handle) {
        std::printf("[compute] draw_mirror_apply pipeline failed to compile\n");
        return false;
    }
    ensure_vcount_ubo(*this);
    std::printf("[compute] draw_mirror_apply pipeline compiled (gpu:: seam)\n");
    return true;
}

// ---------------------------------------------------------------------------
// Dispatch (gpu:: seam). The seam's GL backend issues the storage / buffer-update /
// vertex-attrib barriers (per dispatch + at submit), a superset of the old explicit
// glMemoryBarrier calls, so ordering is preserved (extra barriers are always safe).
// ---------------------------------------------------------------------------

void ComputeState::dispatch_draw_accum(const DrawAccumParams& p, GLuint pos_vbo) {
    if (!has_draw() || !accum_ssbo || !stroke_norm_ssbo) return;
    const uint32_t vc = p.vertex_count;

    DrawAccumParamsGPU u = {};
    u.anchor_a[0] = p.anchor_a_x; u.anchor_a[1] = p.anchor_a_y; u.anchor_a[2] = p.anchor_a_z;
    u.world_radius = p.world_radius;
    u.anchor_b[0] = p.anchor_b_x; u.anchor_b[1] = p.anchor_b_y; u.anchor_b[2] = p.anchor_b_z;
    u.disp_amount = p.disp_amount;
    u.view_a[0] = p.view_a_x; u.view_a[1] = p.view_a_y; u.view_a[2] = p.view_a_z;
    u.hardness = p.hardness;
    u.view_b[0] = p.view_b_x; u.view_b[1] = p.view_b_y; u.view_b[2] = p.view_b_z;
    u.facing_threshold = p.facing_threshold;
    u.anchor_normal_a[0] = p.anchor_normal_a_x; u.anchor_normal_a[1] = p.anchor_normal_a_y; u.anchor_normal_a[2] = p.anchor_normal_a_z;
    u.use_b = (uint32_t)p.use_b;
    u.anchor_normal_b[0] = p.anchor_normal_b_x; u.anchor_normal_b[1] = p.anchor_normal_b_y; u.anchor_normal_b[2] = p.anchor_normal_b_z;
    u.inflate = (uint32_t)p.inflate;
    u.vertex_count = vc;
    gpu::write_buffer(gpu_dev, draw_accum_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{  (uint64_t)vc * 3u * sizeof(float),     pos_vbo };
    gpu::Buffer normView{ (uint64_t)vc * 3u * sizeof(float),     stroke_norm_ssbo };
    gpu::Buffer accumView{(uint64_t)vc * 4u * sizeof(uint32_t),  accum_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS, &posView,        posView.size },
        { BIND_NORMALS,   &normView,       normView.size },
        { BIND_ACCUM,     &accumView,      accumView.size },
        { BIND_PARAMS,    &draw_accum_ubo, sizeof(DrawAccumParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, draw_accum_pipeline, bg, 4);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, draw_accum_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_draw_accum_symmetrize(uint32_t vertex_count) {
    if (!has_draw_symmetrize() || !accum_ssbo || !accum_sym_ssbo) return;
    if (!mirror_map_ssbo || mirror_map_vertex_count == 0) return;
    const uint32_t vc = vertex_count;

    VCountParamsGPU u = { vc, 0, 0, 0 };
    gpu::write_buffer(gpu_dev, draw_vcount_ubo, 0, &u, sizeof(u));

    gpu::Buffer accumView{ (uint64_t)vc * 4u * sizeof(uint32_t),                accum_ssbo };
    gpu::Buffer symView{   (uint64_t)vc * 4u * sizeof(uint32_t),                accum_sym_ssbo };
    gpu::Buffer mirrorView{(uint64_t)mirror_map_vertex_count * sizeof(uint32_t), mirror_map_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_ACCUM,      &accumView,       accumView.size },
        { BIND_ACCUM_SYM,  &symView,         symView.size },
        { BIND_MIRROR_MAP, &mirrorView,      mirrorView.size },
        { BIND_PARAMS,     &draw_vcount_ubo, sizeof(VCountParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, draw_symmetrize_pipeline, bg, 4);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, draw_symmetrize_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_draw_apply(GLuint pos_vbo, uint32_t vertex_count,
                                        GLuint accum_override) {
    if (!draw_apply_pipeline.handle || !mask_ssbo) return;
    const uint32_t vc = vertex_count;

    ensure_smooth_dirty_buffer(vc);

    GLuint accum_src = accum_override ? accum_override : accum_ssbo;
    if (!accum_src) return;

    VCountParamsGPU u = { vc, 0, 0, 0 };
    gpu::write_buffer(gpu_dev, draw_vcount_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{   (uint64_t)vc * 3u * sizeof(float),        pos_vbo };
    gpu::Buffer accumView{ (uint64_t)vc * 4u * sizeof(uint32_t),     accum_src };
    gpu::Buffer dirtyView{ (uint64_t)(vc + 1u) * sizeof(uint32_t),   smooth_dirty_ssbo };
    gpu::Buffer maskView{  (uint64_t)vc * sizeof(float),             mask_ssbo };

    uint32_t zero = 0;
    gpu::write_buffer(gpu_dev, dirtyView, 0, &zero, sizeof(zero));

    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,   &posView,         posView.size },
        { BIND_ACCUM,       &accumView,       accumView.size },
        { BIND_DIRTY_VERTS, &dirtyView,       dirtyView.size },
        { BIND_MASK,        &maskView,        maskView.size },
        { BIND_PARAMS,      &draw_vcount_ubo, sizeof(VCountParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, draw_apply_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, draw_apply_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}

void ComputeState::dispatch_draw_mirror_apply(GLuint pos_vbo, uint32_t vertex_count) {
    if (!draw_mirror_apply_pipeline.handle || mirror_map_vertex_count == 0) return;
    if (!accum_ssbo || !mask_ssbo) return;
    const uint32_t vc = vertex_count;

    VCountParamsGPU u = { vc, 0, 0, 0 };
    gpu::write_buffer(gpu_dev, draw_vcount_ubo, 0, &u, sizeof(u));

    gpu::Buffer posView{   (uint64_t)vc * 3u * sizeof(float),                  pos_vbo };
    gpu::Buffer accumView{ (uint64_t)vc * 4u * sizeof(uint32_t),               accum_ssbo };
    gpu::Buffer mirrorView{(uint64_t)mirror_map_vertex_count * sizeof(uint32_t), mirror_map_ssbo };
    gpu::Buffer maskView{  (uint64_t)vc * sizeof(float),                       mask_ssbo };
    const gpu::BindBufferEntry bg[] = {
        { BIND_POSITIONS,  &posView,         posView.size },
        { BIND_ACCUM,      &accumView,       accumView.size },
        { BIND_MIRROR_MAP, &mirrorView,      mirrorView.size },
        { BIND_MASK,       &maskView,        maskView.size },
        { BIND_PARAMS,     &draw_vcount_ubo, sizeof(VCountParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, draw_mirror_apply_pipeline, bg, 5);

    gpu::ComputeBatch b = gpu::begin_compute(gpu_dev);
    gpu::dispatch(b, draw_mirror_apply_pipeline, grp, (vc + 255u) / 256u);
    gpu::submit(b);
    gpu::release_bind_group(grp);
}
