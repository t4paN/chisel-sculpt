#pragma once
#include <glad/glad.h>
#include <cstdint>

// Static display GL resources for an INACTIVE entity. The active entity is
// drawn from the renderer's working buffers (vao/vbo_*/ebo); every other alive
// entity owns one of these lightweight indexed VAOs for the viewport draw and
// the entity-id pick pass. Refreshed (dirty) on geometry change or when the
// entity is flushed out of the working set on selection change.
struct EntityGpu {
    GLuint vao        = 0;
    GLuint vbo_pos    = 0;
    GLuint vbo_norm   = 0;
    GLuint vbo_mask   = 0;
    GLuint vbo_color  = 0;
    GLuint ebo        = 0;
    uint32_t index_count = 0;
    bool dirty = true;       // needs re-upload from entity->mesh
};
