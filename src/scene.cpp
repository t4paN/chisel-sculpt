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

void Scene::refresh_mirror_map(int icosphere_level) {
    MeshEntity* e = active_entity_();
    if (!e) return;
    Mesh& m = e->mesh;

    if (mirror_use_topology_) {
        if (!e->multires.locked) {
            if (icosphere_level >= 0 && icosphere_level < (int)icosphere_mirror_cache_.size()) {
                auto& cached = icosphere_mirror_cache_[icosphere_level];
                if (!cached.empty() && cached.size() == m.vertex_count()) {
                    m.mirror_x_map = cached;
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
        if (out) renderer_.upload_display(out->gpu, out->mesh);
    }

    MeshEntity* in = find_entity(id);
    if (!in) return;

    // Working VBOs hold the incoming entity at offset 0.
    renderer_.upload_mesh(in->mesh);
    // Brush/pick FBO mesh.
    renderer_.upload_screen_mesh(in->mesh);

    // Per-entity compute SSBOs for the active entity (offset 0). Adjacency and
    // mirror map live on the entity's own mesh; re-uploaded on every bind so a
    // re-selection (or post-cascade sync) reflects the entity's current topology.
    if (compute_.supported) {
        if (compute_.compute_normals_program && !in->mesh.vert_tri_offset.empty())
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
        if (up->gpu.dirty || !up->gpu.vao)
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

bool Scene::toggle_selected(uint32_t entity_id) {
    if (entity_id == 0) return false;
    if (entity_id == active_id_) return false;

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
    std::printf("[select] entity_id=%u active=%u\n", entity_id, active_id_);
    if (entity_id == 0) return false;
    if (entity_id == active_id_) return false;
    MeshEntity* e = find_entity(entity_id);
    if (!e) return false;
    set_active(entity_id);
    return true;
}

void Scene::refresh_for_edit_mode() {
    if (alive_count() > 1) {
        MeshEntity* e = selected();
        if (e) set_active(e->id);
    } else {
        set_active(0);
    }
}

void Scene::render_pick(const Camera& cam, int w, int h) {
    // Draw every alive entity into the pick FBO writing its id: the active entity
    // from the working VAO (its indices start at 0), inactive ones from their
    // display VAOs. Caller reads back with renderer.read_id_region.
    renderer_.pick_begin(cam, w, h);
    for (auto& up : entities_) {
        if (!up || !up->alive) continue;
        if (up->id == active_id_)
            renderer_.pick_draw(up->id, renderer_.vao,
                                (uint32_t)up->mesh.indices.size());
        else
            renderer_.pick_draw(up->id, up->gpu.vao, up->gpu.index_count);
    }
    renderer_.pick_end();
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
    keep->undo.clear();

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
