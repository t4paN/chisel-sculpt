#include "multires_gpu.h"
#include "multires_stack.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// Coalesce an unsorted vert-ID list into contiguous runs, bridging gaps <= GAP.
// Same rationale as brush.cpp's banded readback: mirror twins and remesh scatter
// touched verts across wide index spans, so one glBufferSubData per run beats both
// a per-vert call storm and a single span that drags the gaps across the bus.
namespace {
struct Run { uint32_t first, count; };
constexpr uint32_t RUN_GAP = 256;

void coalesce(const std::vector<uint32_t>& verts, std::vector<Run>& out) {
    out.clear();
    if (verts.empty()) return;
    static std::vector<uint32_t> sorted;
    sorted.assign(verts.begin(), verts.end());
    std::sort(sorted.begin(), sorted.end());
    uint32_t first = sorted[0], last = sorted[0];
    for (size_t i = 1; i < sorted.size(); i++) {
        uint32_t v = sorted[i];
        if (v == last) continue;
        if (v - last > RUN_GAP) { out.push_back({first, last - first + 1}); first = v; }
        last = v;
    }
    out.push_back({first, last - first + 1});
}

// Interleave a contiguous run of SOA base positions into an AOS float3 scratch.
void interleave_base(const MultiresStack& stack, uint32_t first, uint32_t count,
                     std::vector<float>& scratch) {
    scratch.resize((size_t)count * 3);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v = first + i;
        scratch[i*3+0] = stack.base.pos_x[v];
        scratch[i*3+1] = stack.base.pos_y[v];
        scratch[i*3+2] = stack.base.pos_z[v];
    }
}
} // namespace

void MultiresGPU::ensure(uint32_t vertex_count, uint32_t base_vertex_count) {
    if (!supported) return;

    if (capacity < vertex_count || !disp_ssbo || !frames_ssbo) {
        if (disp_ssbo)   { glDeleteBuffers(1, &disp_ssbo);   disp_ssbo = 0; }
        if (frames_ssbo) { glDeleteBuffers(1, &frames_ssbo); frames_ssbo = 0; }

        glGenBuffers(1, &disp_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, disp_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)vertex_count * 3 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);

        glGenBuffers(1, &frames_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, frames_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)vertex_count * 9 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);

        capacity = vertex_count;
    }

    if (base_capacity < base_vertex_count || !base_ssbo) {
        if (base_ssbo) { glDeleteBuffers(1, &base_ssbo); base_ssbo = 0; }
        glGenBuffers(1, &base_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, base_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)base_vertex_count * 3 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);
        base_capacity = base_vertex_count;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void MultiresGPU::upload_level(const MultiresStack& stack, int abs_level) {
    if (!supported || !stack.locked) return;

    uint32_t base_vc = stack.base.vertex_count();

    if (abs_level == stack.base_level) {
        ensure(1, base_vc);   // disp/frames unused at base; keep them non-null
        static std::vector<float> scratch;
        interleave_base(stack, 0, base_vc, scratch);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, base_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        (GLsizeiptr)base_vc * 3 * sizeof(float), scratch.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        level = abs_level;
        return;
    }

    int k = abs_level - stack.base_level - 1;
    if (k < 0 || k >= (int)stack.disp.size()) return;

    uint32_t vc = (uint32_t)stack.disp[k].size();
    ensure(vc, base_vc);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, disp_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)vc * 3 * sizeof(float), stack.disp[k].data());

    // frames[k] is lazily populated by cascade; upload only when present.
    if (k < (int)stack.frames.size() && stack.frames[k].size() == vc) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, frames_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        (GLsizeiptr)vc * 9 * sizeof(float), stack.frames[k].data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    level = abs_level;

#ifdef CHISEL_DEBUG_MULTIRES
    // Validate the disp mirror against CPU truth (Phase 1 has no consumer yet, so
    // this readback is the only thing proving the SSBO copy is correct).
    {
        static std::vector<float> chk;
        chk.resize((size_t)vc * 3);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, disp_ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           (GLsizeiptr)vc * 3 * sizeof(float), chk.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        double maxe = 0.0;
        for (uint32_t v = 0; v < vc; v++) {
            maxe = std::max(maxe, (double)std::fabs(chk[v*3+0] - stack.disp[k][v].x));
            maxe = std::max(maxe, (double)std::fabs(chk[v*3+1] - stack.disp[k][v].y));
            maxe = std::max(maxe, (double)std::fabs(chk[v*3+2] - stack.disp[k][v].z));
        }
        std::printf("[mgpu][debug] upload_level L%d disp mirror max|err|=%.3e (%u verts)\n",
                    abs_level, maxe, vc);
    }
#endif
}

void MultiresGPU::upload_disp_partial(const MultiresStack& stack, int abs_level,
                                      const std::vector<uint32_t>& verts) {
    if (!supported || abs_level != level || verts.empty()) return;

    static std::vector<Run> runs;
    coalesce(verts, runs);

    if (abs_level == stack.base_level) {
        static std::vector<float> scratch;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, base_ssbo);
        for (const Run& r : runs) {
            if (r.first + r.count > stack.base.vertex_count()) continue;
            interleave_base(stack, r.first, r.count, scratch);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            (GLintptr)r.first * 3 * sizeof(float),
                            (GLsizeiptr)r.count * 3 * sizeof(float), scratch.data());
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return;
    }

    int k = abs_level - stack.base_level - 1;
    if (k < 0 || k >= (int)stack.disp.size()) return;
    const std::vector<Vec3>& disp = stack.disp[k];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, disp_ssbo);
    for (const Run& r : runs) {
        if (r.first + r.count > disp.size()) continue;
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        (GLintptr)r.first * 3 * sizeof(float),
                        (GLsizeiptr)r.count * 3 * sizeof(float), &disp[r.first]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void MultiresGPU::snapshot_positions(GLuint pos_vbo, uint32_t vertex_count) {
    if (!supported || vertex_count == 0) return;

    // Lazy grow-only resize, then a GPU→GPU copy of the active entity's working
    // VBO range. Mirrors ComputeState::snapshot_stroke_normals.
    if (snap_pos_capacity < vertex_count || !snap_pos_ssbo) {
        if (snap_pos_ssbo) { glDeleteBuffers(1, &snap_pos_ssbo); snap_pos_ssbo = 0; }
        glGenBuffers(1, &snap_pos_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, snap_pos_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     (GLsizeiptr)vertex_count * 3 * sizeof(float),
                     nullptr, GL_DYNAMIC_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        snap_pos_capacity = vertex_count;
    }

    GLsizeiptr byte_size = (GLsizeiptr)vertex_count * 3 * sizeof(float);
    glBindBuffer(GL_COPY_READ_BUFFER,  pos_vbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, snap_pos_ssbo);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, byte_size);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void MultiresGPU::mark_cpu_dirty(const std::vector<uint32_t>& verts) {
    if (!supported || verts.empty()) return;
    cpu_dirty = true;
    dirty_verts.insert(dirty_verts.end(), verts.begin(), verts.end());
}

void MultiresGPU::materialize_cpu(MultiresStack& stack, Mesh& mesh, GLuint vbo_pos) {
    if (!supported || !cpu_dirty) return;

    // Inverse of upload_disp_partial: read the mirrored level's GPU storage back
    // into CPU disp/base for the dirty verts, in coalesced runs. After this the CPU
    // copy matches the GPU again, so cascade/projection/save can read CPU truth.
    static std::vector<Run> runs;
    coalesce(dirty_verts, runs);

    // Surface sync (2c-i): pull the working VBO positions back into mesh.pos for the
    // same runs — the GPU brush wrote the VBO in place and the pen-up readback no
    // longer copies it down (2c-iv), so the live surface readers need it here.
    // mesh.pos is SOA, the VBO is interleaved float3, so de-interleave per run.
    if (vbo_pos) {
        static std::vector<float> pos_scratch;
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
        uint32_t vcount = mesh.vertex_count();
        for (const Run& r : runs) {
            if (r.first + r.count > vcount) continue;
            pos_scratch.resize((size_t)r.count * 3);
            glGetBufferSubData(GL_ARRAY_BUFFER,
                               (GLintptr)r.first * 3 * sizeof(float),
                               (GLsizeiptr)r.count * 3 * sizeof(float), pos_scratch.data());
            for (uint32_t i = 0; i < r.count; i++) {
                uint32_t v = r.first + i;
                mesh.pos_x[v] = pos_scratch[i*3+0];
                mesh.pos_y[v] = pos_scratch[i*3+1];
                mesh.pos_z[v] = pos_scratch[i*3+2];
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Normals must track the positions we just pulled. The flip leaves
        // mesh.norm_* covering only the moved (snap_list) verts and skips the
        // 1-ring, so a later upload_mesh / upload_display (both read mesh.norm_*
        // directly, renderer.cpp) would shade the surface with stale normals.
        // Recompute the affected (1-ring) normals from the fresh CPU positions so
        // this choke hands back a fully self-consistent mesh. Partial + no heap
        // churn, matching the no-full-iterate rule.
        mesh.recompute_normals_partial(dirty_verts);
    }

    if (level == stack.base_level) {
        static std::vector<float> scratch;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, base_ssbo);
        for (const Run& r : runs) {
            if (r.first + r.count > stack.base.vertex_count()) continue;
            scratch.resize((size_t)r.count * 3);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                               (GLintptr)r.first * 3 * sizeof(float),
                               (GLsizeiptr)r.count * 3 * sizeof(float), scratch.data());
            for (uint32_t i = 0; i < r.count; i++) {
                uint32_t v = r.first + i;
                stack.base.pos_x[v] = scratch[i*3+0];
                stack.base.pos_y[v] = scratch[i*3+1];
                stack.base.pos_z[v] = scratch[i*3+2];
            }
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    } else {
        int k = level - stack.base_level - 1;
        if (k >= 0 && k < (int)stack.disp.size()) {
            std::vector<Vec3>& disp = stack.disp[k];
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, disp_ssbo);
            for (const Run& r : runs) {
                if (r.first + r.count > disp.size()) continue;
                // Vec3 is 3 contiguous floats; &disp[first] is a float3*count span
                // (mirrors upload_disp_partial's write side).
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                                   (GLintptr)r.first * 3 * sizeof(float),
                                   (GLsizeiptr)r.count * 3 * sizeof(float), &disp[r.first]);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
    }

    cpu_dirty = false;
    dirty_verts.clear();
}

void MultiresGPU::cleanup() {
    if (disp_ssbo)     { glDeleteBuffers(1, &disp_ssbo);     disp_ssbo = 0; }
    if (frames_ssbo)   { glDeleteBuffers(1, &frames_ssbo);   frames_ssbo = 0; }
    if (base_ssbo)     { glDeleteBuffers(1, &base_ssbo);     base_ssbo = 0; }
    if (snap_pos_ssbo) { glDeleteBuffers(1, &snap_pos_ssbo); snap_pos_ssbo = 0; }
    capacity = base_capacity = snap_pos_capacity = 0;
    level = -1;
    cpu_dirty = false;
    dirty_verts.clear();
}
