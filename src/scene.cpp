#include "scene.h"
#include "renderer.h"
#include "compute.h"
#include "multires_stack.h"
#include <cstdio>
#include <algorithm>
#include <cassert>

// ---- Construction ----

Scene::Scene(Mesh initial, Renderer& renderer, ComputeState& compute,
             uint32_t initial_subdiv_level)
    : renderer_(renderer), compute_(compute), icosphere_mirror_cache_(8)
{
    initial.recompute_normals();
    initial.build_adjacency();
    MeshEntity* e = new_entity_(std::move(initial), initial_subdiv_level);
    multires_stack_init_from_lock(e->multires, e->mesh, (int)initial_subdiv_level);
    active_id_ = e->id;
    selected_ids_.push_back(e->id);
}

// ---- Private helpers ----

MeshEntity* Scene::new_entity_(Mesh m, uint32_t subdiv_level) {
    auto up = std::make_unique<MeshEntity>();
    up->id           = next_id_++;
    up->subdiv_level = subdiv_level;
    up->mesh         = std::move(m);
    up->alive        = true;
    MeshEntity* ptr  = up.get();
    entities_.push_back(std::move(up));
    return ptr;
}

void Scene::remove_entity_(uint32_t id) {
    for (auto& up : entities_) {
        if (up && up->id == id) {
            up->alive = false;
            renderer_.free_display(up->gpu);  // release its display VAO/VBOs
            if (bound_active_id_ == id) bound_active_id_ = 0;
            // Compact: remove dead entries
            entities_.erase(
                std::remove_if(entities_.begin(), entities_.end(),
                    [](const std::unique_ptr<MeshEntity>& p){ return p && !p->alive; }),
                entities_.end());
            return;
        }
    }
}

MeshEntity* Scene::active_entity_() {
    for (auto& up : entities_)
        if (up && up->id == active_id_ && up->alive) return up.get();
    return nullptr;
}

const MeshEntity* Scene::active_entity_() const {
    for (auto& up : entities_)
        if (up && up->id == active_id_ && up->alive) return up.get();
    return nullptr;
}

MeshEntity* Scene::find_entity(uint32_t id) {
    for (auto& up : entities_)
        if (up && up->id == id && up->alive) return up.get();
    return nullptr;
}

const MeshEntity* Scene::find_entity(uint32_t id) const {
    for (auto& up : entities_)
        if (up && up->id == id && up->alive) return up.get();
    return nullptr;
}

// ---- Active entity accessors ----

Mesh& Scene::active_mesh() {
    MeshEntity* e = active_entity_();
    assert(e && "no active entity");
    return e->mesh;
}

MultiresStack& Scene::active_multires() {
    MeshEntity* e = active_entity_();
    assert(e && "no active entity");
    return e->multires;
}

MeshEntity& Scene::active_entity() {
    MeshEntity* e = active_entity_();
    assert(e && "no active entity");
    return *e;
}

UndoStack& Scene::active_undo() {
    MeshEntity* e = active_entity_();
    assert(e && "no active entity");
    return e->undo;
}

// ---- Mirror map ----

void Scene::materialize_active_cpu() {
    // blood-moon 3b-iv part 2b: choke point ahead of every CPU read of the active
    // entity's disp/base/pos. No-op until 2c marks the CPU copy dirty (the pen-up
    // readback still keeps it fresh today), so behavior-neutral. 2c extends this to
    // also pull renderer_.vbo_pos back into mesh.pos for the surface readers.
    MeshEntity* e = active_entity_();
    if (!e) return;
    // Active entity is the working set (offset 0), so its surface lives in the
    // renderer's working position VBO. 2c-i syncs both storage (disp/base) and
    // surface (mesh.pos) from the GPU.
    e->multires_gpu.materialize_cpu(e->multires, e->mesh, renderer_.vbo_pos);
}

void Scene::refresh_mirror_map(int icosphere_level) {
    MeshEntity* e = active_entity_();
    if (!e) return;
    materialize_active_cpu();  // 2b: build_mirror_x_map reads mesh.pos
    Mesh& m = e->mesh;

    if (mirror_use_topology_) {
        if (!e->multires.locked) {
            if (icosphere_level >= 0 && icosphere_level < (int)icosphere_mirror_cache_.size()) {
                auto& cached = icosphere_mirror_cache_[icosphere_level];
                if (!cached.empty() && cached.size() == m.vertex_count()) {
                    m.mirror_x_map = cached;
                    m.mirror_topo_version = m.topo_version;
                    return;
                }
                m.build_mirror_x_map();
                if (m.mirror_x_map.size() == m.vertex_count())
                    cached = m.mirror_x_map;
                return;
            }
        } else if (m.mirror_x_map.size() == m.vertex_count()) {
            return;
        }
    }
    // Persistent map: rebuild only when topology actually changed. Rebuilding
    // from sculpted positions reclassifies drifted verts (paired → unpaired),
    // which bakes seam asymmetry in permanently — never do it for
    // position-only edits.
    if (m.mirror_x_map.size() == m.vertex_count()
        && m.mirror_topo_version == m.topo_version)
        return;
    m.build_mirror_x_map();
    sync_mirror_map();
}

void Scene::sync_mirror_map() {
    // The working buffer holds only the active entity; upload its mirror map at
    // offset 0 (no +base_v). The brush symmetrize gate compares
    // mirror_map_vertex_count against the active vertex count.
    MeshEntity* e = active_entity_();
    if (!e || !compute_.supported) return;
    const std::vector<uint32_t>& m = e->mesh.mirror_x_map;
    if (!m.empty()) compute_.upload_mirror_map(m);
}

// ---- bind-on-select: swap the working set to a given entity ----

void Scene::bind_active_(uint32_t id) {
    // 1. Flush the outgoing active entity's working buffers into its own display
    //    VAO so its viewport/pick copy stays current. Its CPU mesh is already
    //    authoritative (finalize reads back pos+normals at pen-up).
    if (bound_active_id_ && bound_active_id_ != id) {
        MeshEntity* out = find_entity(bound_active_id_);
        if (out) {
            // 2c: the working VBO (which holds `out`'s surface) is about to be
            // overwritten by the incoming entity. Pull `out`'s GPU-resident edits
            // back to CPU first (no-op until 2c-iii flips), then park its undo ring
            // so the incoming entity captures from a fresh ring — the ring is a hot
            // cache for the active entity only.
            out->multires_gpu.materialize_cpu(out->multires, out->mesh, renderer_.vbo_pos);
            out->undo.ring_park_all(compute_);
            renderer_.upload_display(out->gpu, out->mesh);
        }
    }

    MeshEntity* in = find_entity(id);
    if (!in) return;

    // Same-id rebind (e.g. sync() right after add_preview, where the active entity
    // is unchanged): the working VBO already holds this entity's surface, and a
    // flipped GPU stroke may have left its edits resident on the GPU with mesh.pos
    // still stale (mark_cpu_dirty, brush.cpp). Pull them back to CPU *before*
    // upload_mesh re-reads the CPU mesh — otherwise the re-upload overwrites the
    // working buffer with pre-stroke geometry and the strokes vanish until undo.
    // No-op when the CPU copy is already authoritative (materialize_cpu self-guards
    // on cpu_dirty), so the splice/cascade paths that legitimately push CPU→GPU are
    // untouched.
    if (id == bound_active_id_)
        in->multires_gpu.materialize_cpu(in->multires, in->mesh, renderer_.vbo_pos);

    // Working VBOs hold the incoming entity at offset 0.
    renderer_.upload_mesh(in->mesh);
    // Refresh the mask/color aliases: upload_mesh may have released+recreated
    // vbo_mask/vbo_color (new handle on a vertex-count change — the WebGPU-correct
    // path has no in-place resize), so the compute alias must be re-pointed here
    // rather than cached once at startup (Step 3a).
    compute_.mask_ssbo  = renderer_.vbo_mask;
    compute_.color_ssbo = renderer_.vbo_color;
    // Brush/pick FBO mesh.
    renderer_.upload_screen_mesh(in->mesh);

    // Per-entity compute SSBOs for the active entity (offset 0). Adjacency and
    // mirror map live on the entity's own mesh; re-uploaded on every bind so a
    // re-selection (or post-cascade sync) reflects the entity's current topology.
    if (compute_.supported) {
        if (compute_.has_normals() && !in->mesh.vert_tri_offset.empty())
            compute_.upload_adjacency(in->mesh.vert_tri_offset.data(),
                                      (uint32_t)in->mesh.vert_tri_offset.size(),
                                      in->mesh.vert_tri_list.data(),
                                      (uint32_t)in->mesh.vert_tri_list.size());
        if (!in->mesh.mirror_x_map.empty())
            compute_.upload_mirror_map(in->mesh.mirror_x_map);
        compute_.ensure_accum_buffer(in->mesh.vertex_count());
    }

    bound_active_id_ = id;
}

// ---- GPU sync ----

void Scene::sync() {
    // No concatenation, no GPU-normal readback: a stroke writes only the active
    // working VBO; inactive display VBOs are static until reselected/edited, so
    // there is nothing to protect. Bind the active entity into the working set,
    // then refresh any inactive entity whose display VAO is stale or unbuilt.
    bind_active_(active_id_);
    for (auto& up : entities_) {
        if (!up || !up->alive || up->id == active_id_) continue;
        if (up->gpu.dirty || !up->gpu.vbo_pos.handle)
            renderer_.upload_display(up->gpu, up->mesh);
    }
}

void Scene::sync_partial_entity(uint32_t eid,
                                const std::vector<uint32_t>& local_dirty) {
    MeshEntity* e = find_entity(eid);
    if (!e || local_dirty.empty()) return;

    if (eid == active_id_) {
        // Active entity lives in the working buffer at offset 0 — patch it
        // directly; the screen mesh re-expands from the working VBO on the GPU.
        renderer_.update_mesh_verts(e->mesh, local_dirty);
        renderer_.update_screen_positions(e->mesh);
    } else {
        // Inactive entity (object-move drag): re-upload its static display VAO.
        renderer_.upload_display(e->gpu, e->mesh);
    }
}

void Scene::sync_mask_partial_entity(uint32_t eid,
                                     const std::vector<uint32_t>& local_dirty) {
    MeshEntity* e = find_entity(eid);
    if (!e || local_dirty.empty()) return;

    if (eid == active_id_)
        renderer_.update_mask_verts(e->mesh, local_dirty);
    else
        renderer_.upload_display(e->gpu, e->mesh);
}

void Scene::sync_color_partial_entity(uint32_t eid,
                                      const std::vector<uint32_t>& local_dirty) {
    MeshEntity* e = find_entity(eid);
    if (!e || local_dirty.empty()) return;

    if (eid == active_id_)
        renderer_.update_color_verts(e->mesh, local_dirty);
    else
        renderer_.upload_display(e->gpu, e->mesh);
}

// ---- Selection ----

void Scene::set_active(uint32_t entity_id) {
    active_id_ = entity_id;
    selected_ids_.clear();
    if (entity_id != 0) selected_ids_.push_back(entity_id);
    if (preview_id_ != 0) selected_ids_.push_back(preview_id_);
    // Swap the working set to the new active entity. id 0 (single-mesh
    // refresh_for_edit_mode) is a no-op: the working buffer keeps its content.
    if (entity_id != 0) bind_active_(entity_id);
}

void Scene::collapse_selection_to_active() {
    selected_ids_.clear();
    if (active_id_  != 0) selected_ids_.push_back(active_id_);
    if (preview_id_ != 0) selected_ids_.push_back(preview_id_);
}

bool Scene::toggle_selected(uint32_t entity_id) {
    if (entity_id == 0) return false;
    // The active entity may be toggled out too: this only removes it from the
    // *visible* selection set (the highlight/group), not from active_id_ — the
    // working buffers stay bound so the always-a-valid-active invariant holds.
    // Peeling out the last member yields a legitimately empty selection (a
    // fully-deselected scene); a plain click re-selects via select().

    for (auto it = selected_ids_.begin(); it != selected_ids_.end(); ++it) {
        if (*it == entity_id) {
            selected_ids_.erase(it);
            return true;
        }
    }

    selected_ids_.push_back(entity_id);
    return true;
}

MeshEntity* Scene::selected() {
    return find_entity(active_id_);
}

const MeshEntity* Scene::selected() const {
    return find_entity(active_id_);
}

uint32_t Scene::alive_count() const {
    uint32_t n = 0;
    for (auto& up : entities_) if (up && up->alive) n++;
    return n;
}

bool Scene::select(uint32_t entity_id) {
    if (entity_id == 0) return false;
    MeshEntity* e = find_entity(entity_id);
    if (!e) return false;
    // Note: no early-out when entity_id == active_id_. After a full deselect the
    // active entity stays bound but drops out of the visible selection; a plain
    // click on it must re-highlight it. set_active rebuilds selected_ids_ to just
    // this entity, and bind_active_ already no-ops when it's already bound (so no
    // working-buffer churn on the common re-click-the-active case).
    set_active(entity_id);
    return true;
}

void Scene::refresh_for_edit_mode() {
    // Make the active entity authoritative for editing. The old code did
    // set_active(0) for the single-mesh case, but entity ids start at 1 — id 0
    // means "no active entity", so that stranded active_id_ and the next
    // active_mesh() aborted. It only stayed latent because a lone *original*
    // mesh is rarely re-entered via SELECT; a merge (which collapses to one
    // entity with a nonzero id) then remesh, then EDIT, hit it every time.
    // Resolve to a live entity: the current active if it survived, else the
    // first alive entity (e.g. the lone survivor of a merge).
    MeshEntity* e = active_entity_();
    if (!e) {
        for (auto& up : entities_)
            if (up && up->alive) { e = up.get(); break; }
    }
    if (e) set_active(e->id);
}

void Scene::render_pick(const Camera& cam, int w, int h) {
    // Draw every alive entity into the pick FBO writing its id: the active entity
    // from the working VAO (its indices start at 0), inactive ones from their
    // display VAOs. Caller reads back with renderer.read_id_region.
    renderer_.pick_begin(cam, w, h);
    for (auto& up : entities_) {
        if (!up || !up->alive) continue;
        if (up->id == active_id_)
            renderer_.pick_draw(up->id, renderer_.vbo_pos, renderer_.ebo,
                                (uint32_t)up->mesh.indices.size());
        else
            renderer_.pick_draw(up->id, up->gpu.vbo_pos, up->gpu.ebo, up->gpu.index_count);
    }
    renderer_.pick_end();
}

// ---- Project load (multimesh) ----

void Scene::load_entities(std::vector<EntityRecord>& records,
                          uint32_t active_id,
                          const std::vector<uint32_t>& selected,
                          uint32_t next_id) {
    // Tear down the current scene, releasing each entity's display VAO.
    for (auto& up : entities_)
        if (up) renderer_.free_display(up->gpu);
    entities_.clear();
    bound_active_id_ = 0;
    preview_id_      = 0;
    load_flattened_  = 0;

    for (EntityRecord& rec : records) {
        auto up = std::make_unique<MeshEntity>();
        up->id           = rec.id;
        up->subdiv_level = rec.subdiv_level;
        up->alive        = true;
        up->preview      = false;
        up->mesh         = std::move(rec.mesh);
        up->multires     = std::move(rec.multires);

        // A locked stack stores base + displacement layers; the on-disk mesh is
        // only the cached current-level surface. Regenerate it from the stack
        // (matching the legacy single-mesh load), preserving the saved mask.
        if (up->multires.locked) {
            auto saved_mask  = std::move(up->mesh.mask);
            auto saved_color = std::move(up->mesh.color);
            bool stack_ok = true;
            if (rec.legacy_numbering && !up->multires.disp.empty())
                stack_ok = migrate_legacy_numbering(up->multires, up->mesh);
            if (stack_ok) {
                cascade_to_level(up->multires, up->mesh, up->multires.current_level);
            } else {
                // v<=3 file saved on a different platform: its disp layers can't
                // be decoded here, but the cached surface is ground truth. Keep
                // it and relock the stack flat at the current level — geometry
                // correct, lower levels lost. (Re-saving on the origin platform
                // migrates the file properly to v4.)
                std::printf("[load] entity %u: foreign legacy multires — "
                            "flattened at level %d\n",
                            up->id, up->multires.current_level);
                int lvl = up->multires.current_level;
                up->mesh.build_adjacency();
                multires_stack_init_from_lock(up->multires, up->mesh, lvl);
                load_flattened_++;
            }
            if (!saved_mask.empty() && saved_mask.size() == up->mesh.vertex_count())
                up->mesh.mask = std::move(saved_mask);
            if (!saved_color.empty() && saved_color.size() == up->mesh.vertex_count())
                up->mesh.color = std::move(saved_color);
        }
        up->mesh.recompute_normals();
        up->mesh.build_adjacency();
        // Per-entity mirror map so any entity is ready to symmetrize when it
        // becomes active (bind_active_ re-uploads it). The active entity's map
        // is refreshed/cached again by the caller's refresh_mirror_map.
        up->mesh.build_mirror_x_map();

        entities_.push_back(std::move(up));
    }

    // Restore bookkeeping, then sanitize selection against what actually loaded.
    auto exists = [&](uint32_t id) {
        for (auto& up : entities_) if (up && up->id == id) return true;
        return false;
    };

    active_id_ = active_id;
    if (!exists(active_id_) && !entities_.empty())
        active_id_ = entities_.front()->id;

    selected_ids_.clear();
    for (uint32_t id : selected)
        if (exists(id)) selected_ids_.push_back(id);
    if (selected_ids_.empty() && active_id_ != 0)
        selected_ids_.push_back(active_id_);

    // Never hand back a next_id that could collide with a loaded entity.
    next_id_ = next_id;
    for (auto& up : entities_)
        if (up && up->id >= next_id_) next_id_ = up->id + 1;
    // caller must sync()
}

// ---- Reset ----

uint32_t Scene::reset_to_single_mesh(uint32_t subdiv_level) {
    // Keep only the active entity, or if none, the first alive entity.
    // Caller has already set active_entity()->mesh to the desired state.
    MeshEntity* keep = active_entity_();
    if (!keep && !entities_.empty()) keep = entities_.front().get();

    // Remove all others
    std::vector<uint32_t> to_remove;
    for (auto& up : entities_)
        if (up && up->alive && up.get() != keep)
            to_remove.push_back(up->id);
    for (uint32_t rid : to_remove) remove_entity_(rid);

    if (keep) {
        keep->subdiv_level = subdiv_level;
        keep->preview = false;
        active_id_ = keep->id;
        selected_ids_ = { keep->id };
        preview_id_ = 0;
        return keep->id;  // caller must sync()
    }
    return 0;
}

// ---- Voxel merge (SDF) ----

uint32_t Scene::merge_selected_into(const Mesh& welded, uint32_t subdiv_level) {
    if (selected_ids_.empty()) return 0;

    // Carrier = the active entity if it is part of the selection, else the
    // first selected entity. Its mesh becomes the welded result.
    uint32_t keep_id = 0;
    for (uint32_t id : selected_ids_)
        if (id == active_id_) { keep_id = id; break; }
    if (!keep_id) keep_id = selected_ids_.front();

    MeshEntity* keep = find_entity(keep_id);
    if (!keep) return 0;

    // Drop every OTHER selected entity (unselected entities are preserved).
    std::vector<uint32_t> drop;
    for (uint32_t id : selected_ids_)
        if (id != keep_id) drop.push_back(id);
    for (uint32_t id : drop) remove_entity_(id);

    // Replace the carrier's mesh; topology changed entirely → drop its multires
    // stack + mask + per-model undo and reinit a fresh level-0 lock.
    keep->mesh = welded;
    keep->mesh.mask.clear();
    keep->mesh.build_adjacency();
    keep->mesh.recompute_normals();
    keep->multires = MultiresStack{};
    multires_stack_init_from_lock(keep->multires, keep->mesh, (int)subdiv_level);
    keep->subdiv_level = subdiv_level;
    keep->preview      = false;
    keep->undo.clear(&compute_);  // keep becomes active; reset the ring for the fresh topology

    active_id_    = keep_id;
    selected_ids_ = { keep_id };
    preview_id_   = 0;
    return keep_id;   // caller does mirror refresh + sync()
}

// ---- Deletion ----

Scene::DeleteResult Scene::delete_selected() {
    DeleteResult r;
    MeshEntity* sel = selected();
    if (!sel) return r;

    if (alive_count() <= 1) {
        r.blocked_only_mesh = true;
        return r;
    }

    uint32_t del_id = sel->id;
    remove_entity_(del_id);

    // Select the first surviving entity
    uint32_t new_sel = 0;
    for (auto& up : entities_)
        if (up && up->alive) { new_sel = up->id; break; }

    if (new_sel != 0) set_active(new_sel);
    sync();

    r.deleted          = true;
    r.deleted_id       = del_id;
    r.new_selected_id  = new_sel;
    return r;
}

// ---- INSERT preview lifecycle ----

uint32_t Scene::add_preview(const Mesh& src, uint32_t subdiv_level) {
    // New independent entity — NO appending into the active mesh's arrays.
    MeshEntity* e = new_entity_(src, subdiv_level);
    e->mesh.build_adjacency();
    e->preview = true;
    preview_id_ = e->id;
    selected_ids_.push_back(e->id);
    sync();
    return e->id;
}

void Scene::remove_preview(uint32_t entity_id) {
    if (preview_id_ == entity_id) preview_id_ = 0;
    for (auto it = selected_ids_.begin(); it != selected_ids_.end(); ++it) {
        if (*it == entity_id) { selected_ids_.erase(it); break; }
    }
    remove_entity_(entity_id);
    sync();
}

void Scene::commit_preview(uint32_t entity_id) {
    MeshEntity* e = find_entity(entity_id);
    if (!e) return;
    e->preview = false;
    if (preview_id_ == entity_id) preview_id_ = 0;
    // Make it the active entity
    set_active(entity_id);
    refresh_for_edit_mode();
}

// ---- Multires splice (active entity only) ----

void Scene::splice_active(const Mesh& replacement) {
    // Multires cascade and undo always target the ACTIVE entity (twins are gone),
    // so this replaces active_entity()->mesh in place and re-syncs — the
    // pre-multimesh single-mesh path pointed at the active entity. No offset
    // shift, no render_ patch.
    MeshEntity* e = active_entity_();
    if (!e) return;
    e->mesh.pos_x   = replacement.pos_x;
    e->mesh.pos_y   = replacement.pos_y;
    e->mesh.pos_z   = replacement.pos_z;
    e->mesh.norm_x  = replacement.norm_x;
    e->mesh.norm_y  = replacement.norm_y;
    e->mesh.norm_z  = replacement.norm_z;
    e->mesh.indices = replacement.indices;
    if (!replacement.mask.empty())
        e->mesh.mask = replacement.mask;
    e->mesh.build_adjacency();
    e->mesh.recompute_normals();
    // sync() → bind_active_ re-uploads the working buffers + adjacency/mirror
    // SSBOs from the new topology, and refreshes inactive display VAOs.
    sync();
}
