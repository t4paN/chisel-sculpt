#include "scene_snapshot.h"
#include "scene.h"
#include "mesh_entity.h"
#include "undo.h"
#include <cstdio>

// ---- Sizing -------------------------------------------------------------

// Triangles this entity carries at its top level. Loop subdivision quadruples
// exactly, so the stack alone answers this — no cascade, no working mesh. That
// matters during a bake, when the surfaces are deliberately stale.
static uint64_t entity_top_tris(const MeshEntity& e) {
    if (!e.multires.locked) return e.mesh.tri_count();
    uint64_t t = e.multires.base.tri_count();
    for (size_t k = 0; k < e.multires.disp.size(); k++) t *= 4;
    return t;
}

uint64_t snapshot_scene_tris(const Scene& scene) {
    uint64_t t = 0;
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        t += entity_top_tris(*up);
    }
    return t;
}

// Walks the scene the way the bake loop does, without touching it.
int snapshot_levels_to_bake(const Scene& scene) {
    struct Sim { uint64_t tris; size_t layers; };
    std::vector<Sim> sim;
    uint64_t total = 0;
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        uint64_t t = entity_top_tris(*up);
        sim.push_back({ t, up->multires.locked ? up->multires.disp.size() : 0 });
        total += t;
    }
    int baked = 0;
    while (total > SNAPSHOT_TRI_CAP) {
        // Same choice the real loop makes: always the heaviest entity that still
        // has a layer to give, so the cap is reached in the fewest levels.
        int pick = -1;
        for (size_t i = 0; i < sim.size(); i++)
            if (sim[i].layers > 0 && (pick < 0 || sim[i].tris > sim[pick].tris))
                pick = (int)i;
        if (pick < 0) return -1;          // nothing left to drop, still over
        total -= sim[pick].tris - sim[pick].tris / 4;
        sim[pick].tris /= 4;
        sim[pick].layers--;
        baked++;
    }
    return baked;
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
    levels_baked = 0;
    edits_at_capture = 0;
}

// ---- Lean records -------------------------------------------------------

// One entity, stripped to what cannot be recomputed. For a locked entity that is
// the stack and nothing else: load_entities cascades the surface straight back
// out of it, and cascade_to_level refills mask and colour from the stack's own
// planes — so the working mesh is pure derived data and is dropped whole. An
// unlocked entity has no stack to rebuild from, so its mesh IS the truth and is
// kept, minus the arrays load_entities regenerates anyway.
static EntityRecord lean_record(const MeshEntity& e) {
    EntityRecord rec;
    rec.id           = e.id;
    rec.subdiv_level = e.subdiv_level;
    rec.multires     = e.multires;

    // Derived caches inside the stack: frames are recomputed on the
    // pre-displacement surface at every cascade, the topo cache is a pure
    // function of base topology, mirror maps and midpoint parents are rebuilt
    // lazily. All four are big and all four come back on their own.
    rec.multires.frames.clear();
    rec.multires.topo_cache.clear();
    rec.multires.midpoint_parents.clear();
    rec.multires.mirror.clear();
    rec.multires.base_mirror.clear();

    if (!e.multires.locked) {
        rec.mesh = e.mesh;
        rec.mesh.norm_x.clear(); rec.mesh.norm_y.clear(); rec.mesh.norm_z.clear();
        rec.mesh.vert_tri_offset.clear();
        rec.mesh.vert_tri_list.clear();
        rec.mesh.mirror_x_map.clear();
    }
    return rec;
}

static void collect_lean(const Scene& scene, std::vector<EntityRecord>& out) {
    out.clear();
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        out.push_back(lean_record(*up));
    }
}

// ---- Prepare ------------------------------------------------------------

int snapshot_prepare(Scene& scene, ComputeState& compute) {
    scene.materialize_active_cpu();   // the active entity may be living on the GPU

    // Fold each entity's working paint and mask into its stack planes FIRST. The
    // lean copy keeps the planes and throws the surface away, so anything still
    // only on the surface would be lost — and a bake shrinks the planes, so this
    // has to happen before the first drop, not after.
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive || up->preview) continue;
        multires_sync_paint(up->multires, up->mesh);
    }

    // Spend the undo history before allocating anything. The op about to run
    // wipes it regardless, so this is free, and it takes the single largest
    // block (up to UndoStack::max_bytes, 1 GB) out of the peak. Only the active
    // entity's clear may touch the GPU ring — the ring caches the active entity,
    // and wiping it from a non-active clear would corrupt it.
    const uint32_t active = scene.active_mesh_id();
    for (const auto& up : scene.entities()) {
        if (!up || !up->alive) continue;
        up->undo.clear(up->id == active ? &compute : nullptr);
    }

    int baked = 0;
    while (snapshot_scene_tris(scene) > SNAPSHOT_TRI_CAP) {
        MeshEntity* pick = nullptr;
        uint64_t    best = 0;
        for (const auto& up : scene.entities()) {
            if (!up || !up->alive || up->preview) continue;
            if (!up->multires.locked || up->multires.disp.empty()) continue;
            uint64_t t = entity_top_tris(*up);
            if (!pick || t > best) { pick = up.get(); best = t; }
        }
        if (!pick) {
            std::printf("[snapshot] %.2fM tris and no layer left to bake — "
                        "running unprotected\n",
                        (double)snapshot_scene_tris(scene) / 1e6);
            return -1;
        }

        // The same two steps the burger menu's "delete highest subdiv" runs: bake
        // the top layer's form down into the one below (inverse Loop), then drop
        // it. Only the residual detail dies; the shape survives.
        MultiresStack& st = pick->multires;
        const int target = st.base_level + (int)st.disp.size() - 1;
        project_down_to_level(st, target);
        if (st.current_level > target) st.current_level = target;
        multires_drop_top_level(st);
        pick->multires_gpu.cleanup();   // grow-only mirror still sized for the dead level
        baked++;
        std::printf("[snapshot] baked entity %u down to L%d (%.2fM tris scene-wide)\n",
                    pick->id, target, (double)snapshot_scene_tris(scene) / 1e6);
    }

    // Every surface is stale now — they were cascaded from stacks that just lost
    // a layer. Rebuilding through load_entities is the one path that reliably
    // re-cascades, re-normals, re-adjacencies and re-uploads EVERY entity, not
    // just the active one.
    if (baked > 0) {
        std::vector<EntityRecord> recs;
        collect_lean(scene, recs);
        // COPY the selection: load_entities clears the scene's own selected_ids_
        // and then iterates the vector it was handed. Passing scene.selected_ids()
        // straight through aliases those two and walks a cleared container.
        const std::vector<uint32_t> sel = scene.selected_ids();
        scene.load_entities(recs, scene.active_mesh_id(), sel, scene.next_id());
    }
    return baked;
}

void snapshot_capture(SceneSnapshot& snap, Scene& scene, const char* op) {
    const int baked = snap.levels_baked;
    snap.clear();
    collect_lean(scene, snap.entities);
    snap.active_id           = scene.active_mesh_id();
    snap.selected_ids        = scene.selected_ids();
    snap.next_id             = scene.next_id();
    snap.mirror_use_topology = scene.mirror_topology();
    snap.op                  = op;
    snap.levels_baked        = baked;
    snap.edits_at_capture    = UndoStack::global_pushes;

    size_t bytes = 0;
    for (const EntityRecord& r : snap.entities) {
        for (const auto& d : r.multires.disp) bytes += d.size() * sizeof(Vec3);
        bytes += r.multires.color.size() * 4 + r.multires.mask.size() * 4
               + r.multires.density.size() * 4;
        bytes += (r.mesh.pos_x.size() * 3 + r.mesh.indices.size()) * 4;
    }
    std::printf("[snapshot] %zu entit%s before %s: %.1f MB, %.2fM tris, %d level%s baked\n",
                snap.entities.size(), snap.entities.size() == 1 ? "y" : "ies", op,
                (double)bytes / (1024.0 * 1024.0),
                (double)snapshot_scene_tris(scene) / 1e6,
                baked, baked == 1 ? "" : "s");
}

void snapshot_restore(SceneSnapshot& snap, Scene& scene) {
    if (!snap.valid()) return;
    std::printf("[snapshot] reverting %s: %zu entit%s from stacks\n",
                snap.op, snap.entities.size(),
                snap.entities.size() == 1 ? "y" : "ies");

    // Mirror mode first: a destructive remesh switched the scene to spatial
    // mirror, and load_entities builds each entity's map under whatever mode is
    // set. Restoring geometry without this leaves you silently off topology mirror.
    scene.set_mirror_topology(snap.mirror_use_topology);
    // Consumes the records, so a restore spends the snapshot: one step back, no redo.
    scene.load_entities(snap.entities, snap.active_id, snap.selected_ids, snap.next_id);
    snap.clear();
}
