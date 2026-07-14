#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include "mesh.h"
#include "mesh_entity.h"
#include "entity_record.h"

class Renderer;
struct ComputeState;
struct Camera;

// Scene owns a collection of independent MeshEntity objects. Each entity has
// its own Mesh and MultiresStack — no shared flat buffer, no cross-entity
// normal or adjacency contamination.
//
// Per-entity GPU spine: there is no concatenated render_ buffer. The ACTIVE
// entity lives in the renderer's working buffers (vao/vbo_*/ebo + screen FBO
// mesh) at offset 0; every other alive entity owns a lightweight static display
// VAO (EntityGpu) for the viewport draw and the entity-id pick pass. Selection
// swaps the working set via bind_active_ (user-paced, once per select).
class Scene {
public:
    // Takes ownership of the initial mesh; creates the first MeshEntity.
    // multires_stack_init_from_lock is called internally.
    Scene(Mesh initial, Renderer& renderer, ComputeState& compute,
          uint32_t initial_subdiv_level);

    // ---- Active entity accessors (stable refs via unique_ptr) ----
    Mesh&          active_mesh();
    MultiresStack& active_multires();
    MeshEntity&    active_entity();     // the whole active entity (asserts one exists)
    UndoStack&     active_undo();       // active entity's per-model undo history

    // ---- Mirror map ----
    void set_mirror_topology(bool use_topology) { mirror_use_topology_ = use_topology; }
    bool mirror_topology() const { return mirror_use_topology_; }
    void refresh_mirror_map(int icosphere_level = -1);
    void sync_mirror_map();

    // ---- GPU-resident undo: deferred CPU writeback (blood-moon 3b-iv part 2b) ----
    // Pull the active entity's GPU-resident truth back into its CPU arrays before a
    // CPU consumer reads them. The single choke point for every disp/base/pos reader
    // (save, projection/cascade, remesh, voxel-merge, mirror rebuild). Currently
    // syncs the multires *storage* (disp/base) via MultiresGPU::materialize_cpu();
    // both that and the eventual working-VBO→mesh.pos readback (part 2c, via
    // renderer_) are no-ops until 2c starts marking the CPU copy dirty — so this is
    // behavior-neutral groundwork: the call sites are placed now, the pos body lands
    // with 2c without touching any consumer again.
    void materialize_active_cpu();

    // ---- GPU sync ----
    // Bind the active entity into the working set (bind_active_) and refresh any
    // dirty inactive entity's display VAO. Called only at user-paced boundaries
    // (init, select, insert-commit, delete, remesh, load, undo cascade) where the
    // CPU meshes are authoritative — never mid-stroke.
    void sync();

    // Partial: caller wrote into entity->mesh local indices. Active entity →
    // patch the working buffer (offset 0). Inactive entity (object-move) →
    // re-upload its display VAO.
    void sync_partial_entity(uint32_t entity_id,
                             const std::vector<uint32_t>& local_dirty);

    // Partial mask sync: active → patch working mask buffer; inactive → display.
    void sync_density_partial_entity(uint32_t entity_id,
                                     const std::vector<uint32_t>& local_dirty);
    void sync_mask_partial_entity(uint32_t entity_id,
                                  const std::vector<uint32_t>& local_dirty);

    // Partial color sync: active → patch working color buffer; inactive → display.
    void sync_color_partial_entity(uint32_t entity_id,
                                   const std::vector<uint32_t>& local_dirty);

    // ---- Selection / active ----
    void     set_active(uint32_t entity_id);
    uint32_t active_mesh_id()  const { return active_id_; }

    bool toggle_selected(uint32_t entity_id);
    // Collapse a multi-selection down to just the active entity (+ live preview),
    // without touching GPU buffers. Use after a destructive single-entity op
    // (remesh) so the stale selection set — and its deselected tint — clears.
    void collapse_selection_to_active();
    const std::vector<uint32_t>& selected_ids() const { return selected_ids_; }
    uint32_t next_id() const { return next_id_; }

    uint32_t preview_mesh_id() const { return preview_id_; }

    MeshEntity*       selected();
    const MeshEntity* selected() const;
    uint32_t          alive_count() const;

    bool select(uint32_t entity_id);
    void refresh_for_edit_mode();

    // Entity-id pick pass: draw all alive entities (active working VAO + inactive
    // display VAOs) into the screen FBO, each writing its id. Read back with
    // renderer.read_id_region. Used by select and insert raycast.
    void render_pick(const Camera& cam, int w, int h);

    // ---- Project load (multimesh) ----
    // Replace the entire scene with the given saved entities. Tears down every
    // current entity (and its GPU display VAO), recreates each record with its
    // saved id/subdiv_level/mesh/multires, rebuilds adjacency + normals + mirror
    // map per entity (cascading any locked multires stack up to its
    // current_level, preserving the saved mask), and restores
    // active/selected/next-id bookkeeping. Per-entity undo histories start
    // empty. Selection is sanitized against the loaded ids. Records are moved
    // from. Does NOT call sync() — caller refreshes the active mirror map + syncs
    // after, exactly like reset_to_single_mesh.
    void load_entities(std::vector<EntityRecord>& records,
                       uint32_t active_id,
                       const std::vector<uint32_t>& selected,
                       uint32_t next_id);

    // Entities whose legacy (v<=3) multires stack could not be decoded on this
    // platform during the last load_entities — surfaces kept, stacks flattened.
    int load_flattened() const { return load_flattened_; }

    // ---- Reset ----
    // Replace scene with a single mesh (e.g. after remesh/load).
    // Does NOT call sync() — caller must do adjacency/mirror/sync after.
    uint32_t reset_to_single_mesh(uint32_t subdiv_level);

    // ---- Voxel merge (SDF) ----
    // Consume the current selection: replace it with one welded watertight mesh.
    // Keeps the active entity (if selected, else the first selected) as the
    // carrier, drops every OTHER selected entity, and preserves all unselected
    // entities. Rebuilds adjacency/normals and reinits a fresh multires lock;
    // the inputs' multires stacks are dropped (topology changed entirely).
    // Returns the merged entity id, or 0 if there is nothing to merge.
    // Does NOT call sync() — caller does mirror refresh + sync after.
    uint32_t merge_selected_into(const Mesh& welded, uint32_t subdiv_level);

    // ---- Deletion ----
    struct DeleteResult {
        bool     deleted            = false;
        bool     blocked_only_mesh  = false;
        uint32_t deleted_id         = 0;
        uint32_t new_selected_id    = 0;
    };
    DeleteResult delete_selected();

    // ---- INSERT preview lifecycle ----
    uint32_t add_preview(const Mesh& src, uint32_t subdiv_level);
    void     remove_preview(uint32_t entity_id);
    void     commit_preview(uint32_t entity_id);

    // ---- Entity access ----
    MeshEntity*       find_entity(uint32_t id);
    const MeshEntity* find_entity(uint32_t id) const;

    // Iterate all entities (alive and dead — caller checks ->alive). Used by the
    // viewport draw loop (N draws) and the pick pass.
    const std::vector<std::unique_ptr<MeshEntity>>& entities() const { return entities_; }

    // ---- Multires splice (active entity only) ----
    // Replace active_entity()->mesh with `replacement` (indices local/0-based),
    // rebuild adjacency/normals, and re-sync. Used by multires cascade and undo,
    // which always target the active entity.
    void splice_active(const Mesh& replacement);

    // Legacy alias — prefer find_entity().
    MeshEntity* find(uint32_t id) { return find_entity(id); }

    // GPU spine accessors. Scoped for the GPU-resident-undo path (undo.cpp), which
    // is already coupled to Scene's per-entity GPU sync: the active entity's
    // working buffers live in the renderer, and the multires apply shader runs on
    // ComputeState. Both are valid only for the active entity (working set offset 0).
    Renderer&     renderer() { return renderer_; }
    ComputeState& compute()  { return compute_; }

private:
    Renderer&     renderer_;
    ComputeState& compute_;

    std::vector<std::unique_ptr<MeshEntity>> entities_;
    uint32_t next_id_        = 1;
    uint32_t active_id_      = 0;
    uint32_t preview_id_     = 0;
    int      load_flattened_ = 0;
    std::vector<uint32_t> selected_ids_;

    bool mirror_use_topology_ = true;
    std::vector<std::vector<uint32_t>> icosphere_mirror_cache_;

    MeshEntity*       active_entity_();
    const MeshEntity* active_entity_() const;
    MeshEntity*       new_entity_(Mesh m, uint32_t subdiv_level);
    void              remove_entity_(uint32_t id);

    // Swap the working set to entity `id`: flush the outgoing active entity to
    // its display VAO, then upload `id`'s mesh into the renderer working buffers
    // + screen mesh + compute SSBOs (adjacency/mirror/mesh-id) at offset 0.
    void              bind_active_(uint32_t id);
    uint32_t          bound_active_id_ = 0;  // entity currently in the working set
};
