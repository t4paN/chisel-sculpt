#include "undo.h"
#include "scene.h"
#include "mesh_entity.h"
#include "renderer.h"
#include "compute.h"
#include <cmath>
#include <cstdio>
#include <algorithm>

// Default CPU undo budget: 1 GB. Overridden at startup by --toaster (256 MB).
// A level-descend snapshot is enormous: it carries disp AND frames for every layer the
// projection rewrites, measured at ~165 MB on an L9 model (frames alone are 76% of it).
// Ten descends in one session overran the old 1 GB cap — and eviction is SILENT and
// drops the OLDEST entries, so undoing "everything" lands on the oldest state still in
// memory rather than the model you started with. That is what made a trashed model
// un-rescuable by Ctrl+Z. 4 GB buys roughly 25 descends.
//
// Emscripten's size_t is 32 bits and the whole WASM heap is smaller than this, so the
// 4 GB literal would wrap to ZERO there and evict the history down to one entry. The
// web build keeps the old cap.
#if defined(__EMSCRIPTEN__)
size_t UndoStack::max_bytes = 1024ull * 1024ull * 1024ull;
#else
size_t UndoStack::max_bytes = 4096ull * 1024ull * 1024ull;
#endif
// GPU undo ring budget: 256 MB. Overridden at startup by --toaster (64 MB).
// Raised from 256 MB 2026-09-22. The ring grows lazily (copy-preserving doubling)
// toward this cap, so a bigger ceiling costs nothing until the history actually needs
// it; what it buys is more strokes staying GPU-resident instead of spilling to the CPU
// arrays. The B570 reports maxStorageBlock 2047 MB, so 1 GB is a legal single binding.
//
// Web keeps 256 MB: browsers commonly cap maxStorageBufferBindingSize at 128 MB, and
// while the ring already degrades gracefully past that (undo_ring_reserve returns
// SIZE_MAX and the entry falls back to its CPU arrays), there is nothing to gain by
// asking for a gigabyte the device will never grant.
#if defined(__EMSCRIPTEN__)
size_t UndoStack::ring_max_bytes = 256ull * 1024ull * 1024ull;
#else
size_t UndoStack::ring_max_bytes = 1024ull * 1024ull * 1024ull;
#endif
uint64_t UndoStack::global_pushes = 0;

// Spill a resident STROKE entry's (old,new) out of the GPU ring into its CPU arrays.
// The ring packs 6 floats per vert at slot k: old.xyz then new.xyz, aligned with
// e.verts[k]. This is the one amortized readback 2c-iii moved off the pen-up path —
// it runs only when the ring evicts/parks/cross-level-applies the entry, never per
// stroke. After it the entry is fully CPU-backed and applies via the CPU stage.
// Pull just the vertex ids of a ring_has_ids entry into e.verts. They are raw u32
// bits in a float array, so they are copied as bytes, never through a float.
static void read_ids_from_ring(UndoEntry& e, ComputeState& c) {
    const uint32_t n = e.ring_vcount;
    if (!e.ring_has_ids || n == 0 || e.ring_offset == SIZE_MAX || !e.verts.empty()) return;
    e.verts.resize(n);
    c.undo_ring_read((e.ring_offset + (size_t)n * 6) * sizeof(float), n,
                     reinterpret_cast<float*>(e.verts.data()));
}

static void spill_entry_from_ring(UndoEntry& e, ComputeState& c) {
    const uint32_t n = e.ring_vcount;
    if (n == 0 || e.ring_offset == SIZE_MAX) return;
    read_ids_from_ring(e, c);
    e.old_x.resize(n); e.old_y.resize(n); e.old_z.resize(n);
    e.new_x.resize(n); e.new_y.resize(n); e.new_z.resize(n);
    static std::vector<float> sp;
    sp.resize((size_t)n * 6);
    c.undo_ring_read(e.ring_offset * sizeof(float), sp.size(), sp.data());
    for (uint32_t k = 0; k < n; ++k) {
        e.old_x[k] = sp[k*6+0]; e.old_y[k] = sp[k*6+1]; e.old_z[k] = sp[k*6+2];
        e.new_x[k] = sp[k*6+3]; e.new_y[k] = sp[k*6+4]; e.new_z[k] = sp[k*6+5];
    }
}

// Drop an entry's ring residency. After a spill it is fully CPU-backed.
static void unring(UndoEntry& e) {
    e.ring_offset = SIZE_MAX;
    e.ring_vcount = 0;
    e.ring_has_ids = false;
}

void UndoStack::ring_evict_overlap(size_t byte_off, size_t byte_len, ComputeState& c) {
    if (byte_len == 0) return;

    auto evict = [&](UndoEntry& e) {
        if (e.kind != UndoEntry::Kind::STROKE || e.ring_offset == SIZE_MAX) return;
        const size_t so = e.ring_offset * sizeof(float);
        const size_t sl = e.ring_span_floats() * sizeof(float);
        // Half-open interval intersection in the circular ring's byte space. Spans
        // never straddle the end (reserve wraps to 0 first), so this is a plain test.
        if (!(so < byte_off + byte_len && byte_off < so + sl)) return;
        // 2c-iii: the ring bytes are about to be overwritten by a new stroke — spill
        // (old,new) back into the CPU entry so it can still apply via the CPU stage.
        spill_entry_from_ring(e, c);
        unring(e);   // fall back to the CPU stage on a future apply
    };

    for (auto& e : undo_stack) evict(e);
    for (auto& e : redo_stack) evict(e);
}

void UndoStack::push(UndoEntry&& e) {
    if (e.kind == UndoEntry::Kind::STROKE && e.stroke_verts() == 0) return;
    if (e.kind == UndoEntry::Kind::MASK && e.verts.empty()) return;
    if (e.kind == UndoEntry::Kind::PAINT && e.verts.empty()) return;
    if (e.kind == UndoEntry::Kind::PROJECTION && e.before.empty()) return;
    for (const auto& r : redo_stack) total_bytes -= entry_bytes(r);
    redo_stack.clear();
    total_bytes += entry_bytes(e);
    e.seq = next_seq();       // scene-wide edit clock; SceneSnapshot ages against it
    undo_stack.push_back(std::move(e));
    evict_to_budget();
}

void UndoStack::evict_to_budget() {
    size_t dropped = 0, freed = 0;
    while (total_bytes > max_bytes && undo_stack.size() > 1) {
        const size_t b = entry_bytes(undo_stack.front());
        total_bytes -= b;
        undo_stack.pop_front();
        dropped++;
        freed += b;
    }
    // This used to be silent, which is why it cost a model. Losing the OLDEST entries
    // means Ctrl+Z can no longer reach the state you started from — and when what goes
    // is a LEVEL entry, the destructive projection it snapshotted can never be undone.
    if (dropped)
        std::printf("[undo] TRIPWIRE: evicted %zu oldest entr%s (%zu MB) to stay under "
                    "the %zu MB budget — undo can no longer reach back that far\n",
                    dropped, dropped == 1 ? "y" : "ies",
                    freed >> 20, max_bytes >> 20);
}

void UndoStack::clear_redo() {
    for (const auto& r : redo_stack) total_bytes -= entry_bytes(r);
    redo_stack.clear();
}

void UndoStack::clear(ComputeState* c) {
    undo_stack.clear();
    redo_stack.clear();
    total_bytes = 0;
    // Active-entity clear → the ring's contents are this entity's history, now gone.
    // Reset it so the (still-active) entity starts capturing from head 0. Guarded:
    // callers pass nullptr when clearing a non-active stack (don't wipe the live ring).
    if (c) c->undo_ring_reset();
}

void UndoStack::ring_park_all(ComputeState& c) {
    // Mark every resident STROKE entry non-resident so a future apply uses the CPU
    // stage (the entries stay in this entity's history; the ring is just their cache).
    // 2c-iii: spill each resident entry's (old,new) from the ring into its CPU arrays
    // first — post-flip the CPU arrays are placeholders, and these entries are the
    // outgoing entity's history that must survive the switch.
    auto park = [&](UndoEntry& e) {
        if (e.kind == UndoEntry::Kind::STROKE && e.ring_offset != SIZE_MAX) {
            spill_entry_from_ring(e, c);
            unring(e);
        }
    };
    for (auto& e : undo_stack) park(e);
    for (auto& e : redo_stack) park(e);
    c.undo_ring_reset();
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

bool UndoStack::apply(UndoEntry& e, MeshEntity& ent, Scene& scene, bool forward) {
    // PROJECTION and LEVEL entries rewrite CPU disp/base (snapshot restore, or a
    // re-run projection), so any GPU-resident edits must come down FIRST. The caller's
    // cascade_active materializes too, but only after this returns — and a GPU stroke
    // undo leaves the active layer's disp_ssbo holding the post-projection values, so
    // flushing after the restore overwrote the restored layer at every vert those
    // strokes touched. The layer below then disagreed with the detail above it, and
    // the next ascend threw it as spikes exactly where the user had sculpted.
    if ((e.kind == UndoEntry::Kind::PROJECTION || e.kind == UndoEntry::Kind::LEVEL)
        && scene.active_mesh_id() == ent.id) {
        if (ent.multires_gpu.cpu_dirty)
            std::printf("[undo] %s %s: flushing %zu GPU-resident verts before the restore\n",
                        e.kind == UndoEntry::Kind::LEVEL ? "LEVEL" : "PROJECTION",
                        forward ? "redo" : "undo", ent.multires_gpu.dirty_verts.size());
        scene.materialize_active_cpu();
    }
    if (e.kind == UndoEntry::Kind::PROJECTION) {
        return apply_projection(e, ent, forward);
    }
    if (e.kind == UndoEntry::Kind::LEVEL) {
        return apply_level(e, ent, forward);
    }

    Mesh&          mesh     = ent.mesh;
    MultiresStack& multires = ent.multires;

    // [paint-audit] a max id >= current vcount means this entry's ids belong to
    // another level — the writes below clamp, but the entry itself is misrouted.
    if (e.kind == UndoEntry::Kind::MASK || e.kind == UndoEntry::Kind::PAINT) {
        uint32_t max_id = 0;
        for (uint32_t v : e.verts) if (v > max_id) max_id = v;
        std::printf("[paint-audit] undo %s %s: %zu verts, max id %u / vcount %u%s\n",
                    e.kind == UndoEntry::Kind::MASK ? "MASK" : "PAINT",
                    forward ? "redo" : "undo", e.verts.size(), max_id,
                    mesh.vertex_count(),
                    max_id >= mesh.vertex_count() ? "  ** OOB — Lead 1 **" : "");
    }

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

    if (e.kind == UndoEntry::Kind::DENSITY) {
        // Same float pair as MASK entries; neutral default is 0.5, not 0.
        const std::vector<float>& target = forward ? e.new_mask : e.old_mask;
        if (mesh.density.empty()) mesh.density.assign(mesh.vertex_count(), 0.5f);
        else if (mesh.density.size() < mesh.vertex_count())
            mesh.density.resize(mesh.vertex_count(), 0.5f);
        for (size_t k = 0; k < e.verts.size(); ++k) {
            uint32_t v = e.verts[k];
            if (v < (uint32_t)mesh.density.size()) mesh.density[v] = target[k];
        }
        scene.sync_density_partial_entity(ent.id, e.verts);
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

    const int cur = multires.current_level;
    const bool gpu_avail = scene.compute().supported && ent.multires_gpu.supported
                           && scene.active_mesh_id() == ent.id;

    // Case 1 (2c-iii): in-place undo/redo with this stroke still resident in the GPU
    // undo ring → fully on the GPU. The apply shader writes the working VBO +
    // disp_ssbo/base_ssbo straight from the ring's (old,new); CPU storage and mesh.pos
    // stay stale and are marked dirty for the next reader. SKIP every CPU arm (storage
    // write, upload_disp_partial, mesh.pos recompute, sync_partial_entity) — they read/
    // write the placeholder old_*/new_* arrays and would clobber the correct GPU VBO.
    const bool inplace_gpu_ring =
        gpu_avail && ent.multires_gpu.level == e.level && e.level == cur
        && e.ring_offset != SIZE_MAX && scene.compute().undo_ring_ssbo.handle
        && (e.ring_has_ids || e.ring_vcount == (uint32_t)e.verts.size());
    // A stroke whose ids live only in the ring: stay entirely on the GPU. The ids are
    // copied out of the ring GPU-side; nothing crosses to the CPU.
    if (inplace_gpu_ring && e.ring_has_ids && e.verts.empty()) {
        ComputeState& c = scene.compute();
        if (c.has_gpu_touched() && c.has_gpu_normals()
            && c.adjacency_vertex_count == mesh.vertex_count()) {
            Renderer& r = scene.renderer();
            const uint32_t n = e.ring_vcount;
            const uint64_t ids_off = (e.ring_offset + (uint64_t)n * 6) * sizeof(float);
            c.ensure_dirty_verts(n);
            gpu::copy_buffer(c.gpu_dev, c.undo_ring_ssbo, ids_off, c.dirty_verts_ssbo, 0,
                             (uint64_t)n * sizeof(uint32_t));
            c.dispatch_multires_apply(r.vbo_pos, ent.multires_gpu.disp_ssbo,
                                      ent.multires_gpu.frames_ssbo, ent.multires_gpu.base_ssbo,
                                      nullptr, nullptr, n, e.targets_base, true,
                                      (uint32_t)e.ring_offset, forward);
            // The same ids as a {count, ids[]} list: normals over their one-ring, and
            // the entity's stale set for the next materialize.
            c.load_touched_ids(c.undo_ring_ssbo, ids_off, n, mesh.vertex_count());
            c.expand_normals_from(c.touched_list_ssbo, 1, n, mesh.vertex_count(), false, r.ebo);
            c.flush_gpu_normals(mesh.vertex_count(), r.vbo_pos, r.vbo_norm, r.ebo);
            MultiresGPU& mg = ent.multires_gpu;
            mg.ensure_stale(mesh.vertex_count());
            c.dispatch_touched_fold(c.touched_list_ssbo, 1, n, mg.stale_mark_ssbo,
                                    mg.stale_list_ssbo, mg.stale_stamp, mesh.vertex_count(),
                                    false);
            mg.mark_gpu_dirty();
            if (e.targets_base) {
                for (auto& fv : multires.frames) fv.clear();
            } else {
                for (int i = e.disp_index + 1; i < (int)multires.frames.size(); i++)
                    multires.frames[i].clear();
            }
            return false;
        }
        // No GPU expansion available: fetch the ids and take the arm below.
        read_ids_from_ring(e, c);
    }
    if (inplace_gpu_ring) {
        Renderer&     r = scene.renderer();
        ComputeState& c = scene.compute();
        c.dispatch_multires_apply(r.vbo_pos, ent.multires_gpu.disp_ssbo,
                                  ent.multires_gpu.frames_ssbo, ent.multires_gpu.base_ssbo,
                                  e.verts.data(), nullptr, (uint32_t)e.verts.size(),
                                  e.targets_base, c.undo_ring_ssbo.handle != 0,
                                  (uint32_t)e.ring_offset, forward);
        // Normals must cover the one-ring of the reverted verts, not just the moved
        // set: a moved vert changes the face normals it shares with its un-moved
        // neighbours, so those neighbours' vertex normals shift too. The live brush
        // expands via expand_dirty_to_affected before compute_normals (post_dab); this
        // fully-GPU undo arm took the early return without that expansion, leaving the
        // stroke's boundary ring stale-shaded after a full undo. Topology-only read, so
        // it's safe with mesh.pos still stale on this path.
        static std::vector<uint32_t> ring;
        mesh.expand_dirty_to_affected(e.verts, ring);
        const uint32_t* nrm_v = ring.empty() ? e.verts.data() : ring.data();
        uint32_t        nrm_n = ring.empty() ? (uint32_t)e.verts.size() : (uint32_t)ring.size();
        c.dispatch_compute_normals(nrm_v, nrm_n,
                                   r.vbo_pos, r.vbo_norm, r.ebo);
        ent.multires_gpu.mark_cpu_dirty(e.verts);
        // Same frame-cache invalidation the in-place CPU path does below.
        if (e.targets_base) {
            for (auto& fv : multires.frames) fv.clear();
        } else {
            for (int i = e.disp_index + 1; i < (int)multires.frames.size(); i++)
                multires.frames[i].clear();
        }
#ifdef CHISEL_DEBUG_MULTIRES
        // Oracle: force the choke to materialize, then check mesh.pos == VBO for the
        // touched verts — validates the stale→materialize indexing (the apply shader
        // math itself was validated by 3b-iv part 1). Release stays lazy.
        scene.materialize_active_cpu();
        {
            static std::vector<float> chk;
            chk.resize(e.verts.size() * 3);
            for (size_t k = 0; k < e.verts.size(); ++k)
                gpu::read_buffer(c.gpu_dev, scene.renderer().vbo_pos,
                                 (uint64_t)e.verts[k] * 3 * sizeof(float),
                                 3 * sizeof(float), &chk[k * 3]);
            double maxe = 0.0;
            for (size_t k = 0; k < e.verts.size(); ++k) {
                uint32_t v = e.verts[k];
                maxe = std::max(maxe, (double)std::fabs(chk[k*3+0] - mesh.pos_x[v]));
                maxe = std::max(maxe, (double)std::fabs(chk[k*3+1] - mesh.pos_y[v]));
                maxe = std::max(maxe, (double)std::fabs(chk[k*3+2] - mesh.pos_z[v]));
            }
            std::printf("[mgpu][debug] apply-ring materialize L%d %s max|err|=%.3e (%zu verts)\n",
                        e.level, e.targets_base ? "base" : "disp", maxe, e.verts.size());
        }
#endif
        return false;
    }

    // Cases 2 & 3 are CPU-authoritative. If a prior flipped stroke left the active
    // entity's CPU copy stale, materialize it through the choke before we read/write
    // CPU storage (no-op when not dirty).
    scene.materialize_active_cpu();
    // Every arm below reads this entry's CPU verts/old/new — empty or placeholders if
    // it is still ring-resident. Spill it out of the ring first so they hold the real
    // values. (This used to spill only cross-level; a same-level entry reaching here
    // without the GPU arm would have applied placeholder zeros.)
    if (e.ring_offset != SIZE_MAX && scene.compute().undo_ring_ssbo.handle) {
        spill_entry_from_ring(e, scene.compute());
        unring(e);
    }

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
    // (cur computed above for the Case 1 gate.)

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
            bool use_ring = e.ring_offset != SIZE_MAX && c.undo_ring_ssbo.handle
                            && e.ring_vcount == (uint32_t)e.verts.size();
            gpu_apply_used_ring = use_ring;
            if (use_ring) {
                c.dispatch_multires_apply(r.vbo_pos, ent.multires_gpu.disp_ssbo,
                                          ent.multires_gpu.frames_ssbo,
                                          ent.multires_gpu.base_ssbo,
                                          e.verts.data(), nullptr,
                                          (uint32_t)e.verts.size(), e.targets_base,
                                          c.undo_ring_ssbo.handle != 0,
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
            for (size_t k = 0; k < e.verts.size(); ++k) {
                gpu::read_buffer(scene.compute().gpu_dev, scene.renderer().vbo_pos,
                    (uint64_t)e.verts[k] * 3 * sizeof(float),
                    3 * sizeof(float), &gpu_apply_chk[k*3]);
            }
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
