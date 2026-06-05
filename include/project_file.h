#pragma once
#include <vector>
#include <cstdint>
#include "camera.h"
#include "entity_record.h"

// A saved Chisel project is the whole multimesh scene: every alive, committed
// entity (each with its own mesh + multires stack), plus the selection state
// and a few global options so a load restores the scene exactly as it was at
// save time. Per-entity undo history is intentionally NOT serialized — undo
// stacks reset on load.
struct ProjectData {
    std::vector<EntityRecord> entities;

    // Selection / id bookkeeping, restored verbatim.
    uint32_t              active_id = 0;       // which entity is active on load
    std::vector<uint32_t> selected_ids;        // current selection set
    uint32_t              next_id   = 1;        // scene's next-entity counter
    bool                  mirror_use_topology = true;

    // Global (scene-wide) state.
    Camera camera;
    bool   mirror_x     = true;
    int    subdiv_level = 4;
};

enum class SaveResult { OK, ERR_OPEN, ERR_WRITE };
enum class LoadResult { OK, ERR_OPEN, ERR_READ, ERR_MAGIC, ERR_VERSION, ERR_CORRUPT };

SaveResult save_project(const char* path, const ProjectData& data);
LoadResult load_project(const char* path, ProjectData& data);

const char* result_string(SaveResult r);
const char* result_string(LoadResult r);
