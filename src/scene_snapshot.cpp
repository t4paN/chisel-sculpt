#include "scene_snapshot.h"
#include "scene.h"
#include "mesh_entity.h"
#include "undo.h"
#include <cstdio>

// ---- Sizing -------------------------------------------------------------
// Walks container sizes only, never elements, so this is O(levels) and safe to
// call every frame while a confirm prompt is up.

static size_t vec_bytes(size_t count, size_t elem) { return count * elem; }

static size_t mesh_bytes(const Mesh& m) {
    size_t b = 0;
    b += vec_bytes(m.pos_x.size() + m.pos_y.size() + m.pos_z.size(), sizeof(float));
    b += vec_bytes(m.norm_x.size() + m.norm_y.size() + m.norm_z.size(), sizeof(float));
    b += vec_bytes(m.indices.size(), sizeof(uint32_t));
    b += vec_bytes(m.mirror_x_map.size(), sizeof(uint32_t));
    b += vec_bytes(m.vert_tri_offset.size(), sizeof(uint32_t));
    b += vec_bytes(m.vert_tri_list.size(), sizeof(uint32_t));
    b += vec_bytes(m.mask.size(), sizeof(float));
    b += vec_bytes(m.color.size(), sizeof(uint32_t));
    b += vec_bytes(m.density.size(), sizeof(float));
    return b;
}

static size_t stencil_bytes(const SubdivStencil& s) {
    return vec_bytes(s.mid.size(), sizeof(uint32_t))
         + vec_bytes(s.is_bnd.size(), sizeof(uint8_t))
         + vec_bytes(s.bnd_a.size() + s.bnd_b.size(), sizeof(uint32_t));
}

static size_t stack_bytes(const MultiresStack& s) {
    size_t b = mesh_bytes(s.base);
    for (const auto& d : s.disp)    b += vec_bytes(d.size(), sizeof(Vec3));
    for (const auto& f : s.frames)  b += vec_bytes(f.size(), sizeof(Frame));
    for (const auto& m : s.mirror)  b += vec_bytes(m.size(), sizeof(uint32_t));
    for (const auto& p : s.midpoint_parents) b += vec_bytes(p.size(), sizeof(uint32_t));
    b += vec_bytes(s.base_mirror.size(), sizeof(uint32_t));
    b += vec_bytes(s.color.size(), sizeof(uint32_t));
    b += vec_bytes(s.mask.size(), sizeof(float));
    b += vec_bytes(s.density.size(), sizeof(float));
    for (const auto& t : s.topo_cache) {
        b += vec_bytes(t.indices.size() + t.vt_offset.size() + t.vt_list.size(),
                       sizeof(uint32_t));
        b += stencil_bytes(t.stencil);
    }
    return b;
}

size_t snapshot_bytes(const Scene& scene) {
    size_t b = 0;
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        b += mesh_bytes(up->mesh) + stack_bytes(up->multires);
    }
    return b;
}

size_t snapshot_budget() {
#ifdef __EMSCRIPTEN__
    return 96ull * 1024ull * 1024ull;
#else
    return 512ull * 1024ull * 1024ull;
#endif
}

bool snapshot_affordable(const Scene& scene) {
    return snapshot_bytes(scene) <= snapshot_budget();
}

// ---- Lifetime -----------------------------------------------------------

int SceneSnapshot::edits_left() const {
    if (!valid()) return 0;
    uint64_t spent = UndoStack::global_pushes - edits_at_capture;
    if (spent >= (uint64_t)GRACE_EDITS) return 0;
    return GRACE_EDITS - (int)spent;
}

void SceneSnapshot::clear() {
    entities.clear();
    selected_ids.clear();
    active_id = 0;
    next_id   = 1;
    op        = "";
    edits_at_capture = 0;
}

// ---- Capture / restore --------------------------------------------------
// Both halves mirror the project save/load paths (main.cpp do_save_project and
// the .chisel branch of do_import_path) minus the file. EntityRecord is already
// the GPU-free, undo-free description of an entity that both of those share, so
// there is no new format here and nothing to keep in sync with the file version.

void snapshot_capture(SceneSnapshot& snap, Scene& scene, const char* op) {
    snap.clear();
    scene.materialize_active_cpu();   // the active entity may be living on the GPU

    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        EntityRecord rec;
        rec.id           = up->id;
        rec.subdiv_level = up->subdiv_level;
        rec.mesh         = up->mesh;
        rec.multires     = up->multires;
        snap.entities.push_back(std::move(rec));
    }
    snap.active_id           = scene.active_mesh_id();
    snap.selected_ids        = scene.selected_ids();
    snap.next_id             = scene.next_id();
    snap.mirror_use_topology = scene.mirror_topology();
    snap.op                  = op;
    snap.edits_at_capture    = UndoStack::global_pushes;

    std::printf("[snapshot] captured %zu entit%s before %s (%.1f MB, %d edits of grace)\n",
                snap.entities.size(), snap.entities.size() == 1 ? "y" : "ies", op,
                (double)snapshot_bytes(scene) / (1024.0 * 1024.0),
                SceneSnapshot::GRACE_EDITS);
}

void snapshot_restore(SceneSnapshot& snap, Scene& scene) {
    if (!snap.valid()) return;
    std::printf("[snapshot] reverting %s: restoring %zu entit%s\n",
                snap.op, snap.entities.size(),
                snap.entities.size() == 1 ? "y" : "ies");

    // Mirror mode first: a destructive remesh switched the scene to spatial
    // mirror, and load_entities builds each entity's map under whatever mode is
    // set. Restoring geometry without this leaves you silently off topology mirror.
    scene.set_mirror_topology(snap.mirror_use_topology);
    // Consumes the records (load_entities moves from them), so a restore spends
    // the snapshot: one step back, no redo.
    scene.load_entities(snap.entities, snap.active_id, snap.selected_ids, snap.next_id);
    snap.clear();
}
