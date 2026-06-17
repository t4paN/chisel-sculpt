#include "undo.h"
#include "scene.h"
#include "mesh_entity.h"
#include "renderer.h"
#include "compute.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// Default undo budget: 1 GB. Overridden at startup by --toaster (256 MB).
size_t UndoStack::max_bytes = 1024ull * 1024ull * 1024ull;

void UndoStack::push(UndoEntry&& e) {
    if (e.kind == UndoEntry::Kind::STROKE && e.verts.empty()) return;
    if (e.kind == UndoEntry::Kind::MASK && e.verts.empty()) return;
    if (e.kind == UndoEntry::Kind::PAINT && e.verts.empty()) return;
    if (e.kind == UndoEntry::Kind::PROJECTION && e.before.empty()) return;
    for (const auto& r : redo_stack) total_bytes -= entry_bytes(r);
    redo_stack.clear();
    total_bytes += entry_bytes(e);
    undo_stack.push_back(std::move(e));
    evict_to_budget();
}

void UndoStack::evict_to_budget() {
    while (total_bytes > max_bytes && undo_stack.size() > 1) {
        total_bytes -= entry_bytes(undo_stack.front());
        undo_stack.pop_front();
    }
}

void UndoStack::clear() {
    undo_stack.clear();
    redo_stack.clear();
    total_bytes = 0;
}

// Applies the entry to its storage layer (base or disp[k]), then updates the view
// based on the relationship between entry.level and multires.current_level:
//   entry.level <  current_level: revert storage, return true so caller cascades.
//   entry.level == current_level: revert storage, partial-recompute mesh in place.
//   entry.level >  current_level: revert storage only; view does not depend on this layer.
// The base/disp distinction only matters for *where* storage is written.
//
// This is the pre-multimesh logic verbatim. The only multimesh adaptation is the
// GPU sync: where the single-mesh version called renderer.update_mesh_partial /
// update_mask against the lone VBO, this routes through Scene's entity-aware
// partial sync so writes land at the active entity's slice of the shared buffer.
bool UndoStack::apply_projection(const UndoEntry& e, MeshEntity& ent, bool forward) {
    MultiresStack& multires = ent.multires;
    if (forward) {
        // Redo: re-run projection from the restored pre-state. Deterministic.
        project_down_to_level(multires, e.target_level);
    } else {
        // Undo: restore captured pre-state.
        restore_projection_snapshot(multires, e.before);
    }
    // View depends on the full stack — always cascade.
    return true;
}

// A multires view-level change recorded on the undo timeline. Undo/redo move
// current_level (so cascade_active rebuilds the surface at the right level) and,
// for a descend that auto-projected, restore/replay the projection exactly like
// apply_projection. An ascend carries no snapshot: pure view move.
bool UndoStack::apply_level(const UndoEntry& e, MeshEntity& ent, bool forward) {
    MultiresStack& multires = ent.multires;
    if (!e.before.empty()) {
        if (forward) project_down_to_level(multires, e.to_level);
        else         restore_projection_snapshot(multires, e.before);
    }
    multires.current_level = forward ? e.to_level : e.from_level;
    return true;  // view level changed — caller cascades to current_level
}

bool UndoStack::apply(const UndoEntry& e, MeshEntity& ent, Scene& scene, bool forward) {
    if (e.kind == UndoEntry::Kind::PROJECTION) {
        return apply_projection(e, ent, forward);
    }
    if (e.kind == UndoEntry::Kind::LEVEL) {
        return apply_level(e, ent, forward);
    }

    Mesh&          mesh     = ent.mesh;
    MultiresStack& multires = ent.multires;

    if (e.kind == UndoEntry::Kind::MASK) {
        const std::vector<float>& target = forward ? e.new_mask : e.old_mask;
        if (mesh.mask.empty()) mesh.mask.assign(mesh.vertex_count(), 0.0f);
        for (size_t k = 0; k < e.verts.size(); ++k) {
            uint32_t v = e.verts[k];
            if (v < (uint32_t)mesh.mask.size()) mesh.mask[v] = target[k];
        }
        scene.sync_mask_partial_entity(ent.id, e.verts);
        return false;
    }

    if (e.kind == UndoEntry::Kind::PAINT) {
        const std::vector<uint32_t>& target = forward ? e.new_color : e.old_color;
        if (mesh.color.empty()) mesh.color.assign(mesh.vertex_count(), 0xFFFFFFFFu);
        else if (mesh.color.size() < mesh.vertex_count())
            mesh.color.resize(mesh.vertex_count(), 0xFFFFFFFFu);
        for (size_t k = 0; k < e.verts.size(); ++k) {
            uint32_t v = e.verts[k];
            if (v < (uint32_t)mesh.color.size()) mesh.color[v] = target[k];
        }
        scene.sync_color_partial_entity(ent.id, e.verts);
        return false;
    }

    const std::vector<float>& tx = forward ? e.new_x : e.old_x;
    const std::vector<float>& ty = forward ? e.new_y : e.old_y;
    const std::vector<float>& tz = forward ? e.new_z : e.old_z;
    const std::vector<float>& sx = forward ? e.old_x : e.new_x;
    const std::vector<float>& sy = forward ? e.old_y : e.new_y;
    const std::vector<float>& sz = forward ? e.old_z : e.new_z;

    // 1) Write to storage (base cage or disp layer).
    if (e.targets_base) {
        for (size_t k = 0; k < e.verts.size(); ++k) {
            uint32_t v = e.verts[k];
            multires.base.pos_x[v] = tx[k];
            multires.base.pos_y[v] = ty[k];
            multires.base.pos_z[v] = tz[k];
        }
    } else {
        auto& layer = multires.disp[e.disp_index];
        for (size_t k = 0; k < e.verts.size(); ++k) {
            uint32_t v = e.verts[k];
            layer[v].x = tx[k];
            layer[v].y = ty[k];
            layer[v].z = tz[k];
        }
    }

    // Phase 1 GPU residency: mirror the storage edit. No-ops unless e.level is the
    // currently mirrored layer (the common in-place case); the cascade paths below
    // refresh the whole level via refresh_active_gpu_residency() in main.
    ent.multires_gpu.upload_disp_partial(multires, e.level, e.verts);

    // 2) Update view and invalidate frame caches based on view relationship.
    const int cur = multires.current_level;

    if (e.level < cur) {
        // Reverted layer sits below the view. Caller must cascade.
        if (e.targets_base) {
            for (auto& fv : multires.frames) fv.clear();
        } else {
            // frames[k] is pre-disp[k], so disp[k]'s own frames are still valid;
            // layers above depend on the surface that disp[k] contributes to.
            for (int i = e.disp_index + 1; i < (int)multires.frames.size(); i++)
                multires.frames[i].clear();
        }
        return true;
    }

    if (e.level == cur) {
        // View equals the reverted layer. Partial-recompute mesh in place.

        // Phase 2c (GPU-resident undo): run the revert on the GPU too — scatter the
        // target disp into disp_ssbo and reproject the storage delta into the
        // working VBO (the twin of the CPU loop just below) + compute_normals.
        // CPU stays authoritative in 2c: it recomputes mesh.pos and re-syncs the
        // VBO at the end of this block, overwriting the GPU result, so this is
        // behavior-neutral. Under CHISEL_DEBUG_MULTIRES we snapshot the GPU VBO
        // here and compare to the CPU mesh.pos before that re-sync. Active entity
        // only (its mesh occupies the working set at offset 0).
        bool gpu_apply_ran = false;
        bool gpu_apply_used_ring = false;
        if (scene.compute().supported && ent.multires_gpu.supported
            && ent.multires_gpu.level == e.level
            && scene.active_mesh_id() == ent.id) {
            Renderer&     r = scene.renderer();
            ComputeState& c = scene.compute();
            // 3b-iv: if this stroke's (old,new) is still resident in the GPU undo
            // ring, the apply shader reads it straight from there — no CPU stage
            // build. Else fall back to the transient stage (2c path). The vcount
            // guard catches any verts/ring-slot misalignment (e.g. a filtered entry).
            bool use_ring = e.ring_offset != SIZE_MAX && c.undo_ring_ssbo
                            && e.ring_vcount == (uint32_t)e.verts.size();
            gpu_apply_used_ring = use_ring;
            if (use_ring) {
                c.dispatch_multires_apply(r.vbo_pos, ent.multires_gpu.disp_ssbo,
                                          ent.multires_gpu.frames_ssbo,
                                          ent.multires_gpu.base_ssbo,
                                          e.verts.data(), nullptr,
                                          (uint32_t)e.verts.size(), e.targets_base,
                                          c.undo_ring_ssbo,
                                          (uint32_t)e.ring_offset, forward);
            } else {
                static std::vector<float> stage;   // 6 floats/vert: target xyz, source xyz
                stage.resize(e.verts.size() * 6);
                for (size_t k = 0; k < e.verts.size(); ++k) {
                    stage[k*6+0] = tx[k]; stage[k*6+1] = ty[k]; stage[k*6+2] = tz[k];
                    stage[k*6+3] = sx[k]; stage[k*6+4] = sy[k]; stage[k*6+5] = sz[k];
                }
                c.dispatch_multires_apply(r.vbo_pos, ent.multires_gpu.disp_ssbo,
                                          ent.multires_gpu.frames_ssbo,
                                          ent.multires_gpu.base_ssbo,
                                          e.verts.data(), stage.data(),
                                          (uint32_t)e.verts.size(), e.targets_base);
            }
            c.dispatch_compute_normals(e.verts.data(), (uint32_t)e.verts.size(),
                                       r.vbo_pos, r.vbo_norm, r.ebo);
            gpu_apply_ran = true;
        }
#ifdef CHISEL_DEBUG_MULTIRES
        static std::vector<float> gpu_apply_chk;
        if (gpu_apply_ran) {
            gpu_apply_chk.resize(e.verts.size() * 3);
            glBindBuffer(GL_ARRAY_BUFFER, scene.renderer().vbo_pos);
            for (size_t k = 0; k < e.verts.size(); ++k) {
                glGetBufferSubData(GL_ARRAY_BUFFER,
                    (GLintptr)e.verts[k] * 3 * sizeof(float),
                    (GLsizeiptr)3 * sizeof(float), &gpu_apply_chk[k*3]);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
#endif
        if (e.targets_base) {
            // cur == base_level here: mesh surface == base cage directly.
            for (size_t k = 0; k < e.verts.size(); ++k) {
                uint32_t v = e.verts[k];
                mesh.pos_x[v] = tx[k];
                mesh.pos_y[v] = ty[k];
                mesh.pos_z[v] = tz[k];
            }
            for (auto& fv : multires.frames) fv.clear();
        } else {
            // mesh.pos = subdivided_base_pos + frame * disp[k]
            // delta_mesh = frame * (target_disp - source_disp)
            // frames[e.disp_index] is pre-disp at this layer, so it is unchanged by
            // this storage edit and was populated when the user cascaded to cur.
            const auto& frames = multires.frames[e.disp_index];
            for (size_t k = 0; k < e.verts.size(); ++k) {
                uint32_t v = e.verts[k];
                float dx = tx[k] - sx[k];
                float dy = ty[k] - sy[k];
                float dz = tz[k] - sz[k];
                const Frame& f = frames[v];
                mesh.pos_x[v] += f.t.x * dx + f.b.x * dy + f.n.x * dz;
                mesh.pos_y[v] += f.t.y * dx + f.b.y * dy + f.n.y * dz;
                mesh.pos_z[v] += f.t.z * dx + f.b.z * dy + f.n.z * dz;
            }
            for (int i = e.disp_index + 1; i < (int)multires.frames.size(); i++)
                multires.frames[i].clear();
        }
#ifdef CHISEL_DEBUG_MULTIRES
        // Phase 2c validation: GPU-applied VBO (snapshotted above) vs the CPU
        // mesh.pos we just recomputed. Both should agree to float noise. Runs
        // before sync_partial_entity re-uploads CPU positions over the GPU result.
        if (gpu_apply_ran) {
            double maxe = 0.0;
            for (size_t k = 0; k < e.verts.size(); ++k) {
                uint32_t v = e.verts[k];
                maxe = std::max(maxe, (double)std::fabs(gpu_apply_chk[k*3+0] - mesh.pos_x[v]));
                maxe = std::max(maxe, (double)std::fabs(gpu_apply_chk[k*3+1] - mesh.pos_y[v]));
                maxe = std::max(maxe, (double)std::fabs(gpu_apply_chk[k*3+2] - mesh.pos_z[v]));
            }
            std::printf("[mgpu][debug] multires_apply L%d %s %s max|err|=%.3e (%zu verts)\n",
                        e.level, e.targets_base ? "base" : "disp",
                        gpu_apply_used_ring ? "ring" : "stage", maxe, e.verts.size());
        }
#endif
        scratch_dirty = e.verts;
        scratch_gpu.clear();
        mesh.recompute_normals_partial(scratch_dirty, &scratch_gpu);
        // sync_partial_entity patches the active entity's working VBO (offset 0)
        // or, if inactive, its display VAO; either way only the affected verts.
        // Screen-space positions are refreshed by the main loop (screen_buffers_dirty).
        if (!scratch_gpu.empty()) scene.sync_partial_entity(ent.id, scratch_gpu);
        return false;
    }

    // e.level > cur: reverted layer is above the view. No mesh update needed.
    if (e.targets_base) {
        for (auto& fv : multires.frames) fv.clear();
    } else {
        for (int i = e.disp_index + 1; i < (int)multires.frames.size(); i++)
            multires.frames[i].clear();
    }
    return false;
}

bool UndoStack::undo(MeshEntity& ent, Scene& scene) {
    if (undo_stack.empty()) return false;
    UndoEntry e = std::move(undo_stack.back());
    undo_stack.pop_back();
    bool needs_cascade = apply(e, ent, scene, false);
    redo_stack.push_back(std::move(e));
    return needs_cascade;
}

bool UndoStack::redo(MeshEntity& ent, Scene& scene) {
    if (redo_stack.empty()) return false;
    UndoEntry e = std::move(redo_stack.back());
    redo_stack.pop_back();
    bool needs_cascade = apply(e, ent, scene, true);
    undo_stack.push_back(std::move(e));
    return needs_cascade;
}
