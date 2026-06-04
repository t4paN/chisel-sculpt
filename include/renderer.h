#pragma once
#include <glad/glad.h>
#include "mesh.h"
#include "camera.h"
#include "entity_gpu.h"

struct Renderer {
    // Mesh GPU buffers
    GLuint vao;
    GLuint vbo_pos;
    GLuint vbo_norm;
    GLuint vbo_mask;      // per-vertex sculpt mask values (0..1)
    GLuint vbo_tri_id;    // per-vertex triangle ID (flat, 3 verts per tri)
    GLuint vbo_bary;     // per-vertex barycentric coord
    GLuint ebo;

    // Matcap shader
    GLuint matcap_program;

    // Entity-id pick shader (writes linear depth + entity id into the screen FBO,
    // reusing the depth attachment and the triid attachment as an id buffer).
    GLuint pick_program = 0;

    // Background shader (gradient)
    GLuint bg_program;
    GLuint bg_vao;

    // Brush cursor shader
    GLuint cursor_program;
    GLuint cursor_shadow_program;
    GLuint crosshair_program;

    // Debug visualization
    GLuint debug_vert_program;     // vertices as points
    GLuint debug_edge_program;     // edges as lines
    GLuint debug_vert_vao;
    GLuint debug_edge_vao;
    GLuint debug_edge_vbo;
    uint32_t debug_edge_count;

    // Screen buffer FBO for brush pipeline
    GLuint screen_fbo;
    GLuint screen_depth_tex;    // GL_R32F - linear depth
    GLuint screen_normal_tex;   // GL_RGB16F - view-space normals
    GLuint screen_triid_tex;    // GL_R32UI - triangle index
    GLuint screen_bary_tex;     // GL_RG16F - barycentric u,v (w = 1-u-v)
    GLuint screen_depth_rbo;    // actual depth/stencil for z-test
    int screen_buf_w, screen_buf_h;

    // Screen buffer shader (MRT output)
    GLuint screen_buf_program;

    // Expanded mesh VAO for screen buffer pass (no index sharing, flat tris)
    GLuint screen_vao;
    GLuint screen_vbo_pos;
    GLuint screen_vbo_norm;
    GLuint screen_vbo_triid;
    GLuint screen_vbo_bary;
    uint32_t screen_tri_count;

    // Compute shader for GPU-side indexed→flat expansion
    GLuint screen_expand_program;

    bool initialized;

    Renderer();
    ~Renderer();

    void init();
    void upload_mesh(const Mesh& mesh);
    void update_mask(const Mesh& mesh);
    void update_mask_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts);

    // Partial update: sync dirty vertex positions+normals from CPU mesh to VBOs
    void update_mesh_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts);

    // Scattered update: upload ONLY the listed verts (coalesced into runs of
    // consecutive indices), never the span between them. Used by undo so it
    // touches exactly the affected verts and leaves all others — and their
    // GPU-computed normals — untouched.
    void update_mesh_verts(const Mesh& mesh, const std::vector<uint32_t>& verts);
    void update_mask_verts(const Mesh& mesh, const std::vector<uint32_t>& verts);

    // Screen buffer FBO management
    void create_screen_buffers(int w, int h);
    void resize_screen_buffers(int w, int h);
    void upload_screen_mesh(const Mesh& mesh);
    void update_screen_positions(const Mesh& mesh);

    // GPU-side screen mesh position/normal update (no CPU mesh read)
    void update_screen_mesh_gpu();

    // Render to screen buffers (call once on pen-down or per frame during stroke)
    void render_screen_buffers(const Camera& cam, int w, int h);

    // Read back screen buffer data for brush operations
    void read_depth_region(int x, int y, int w, int h, float* out);
    void read_normal_region(int x, int y, int w, int h, float* out);
    void read_triid_region(int x, int y, int w, int h, uint32_t* out);
    void read_bary_region(int x, int y, int w, int h, float* out);

    void draw_background(int w, int h);
    void draw_mesh(const Camera& cam, int w, int h, uint32_t index_count,
                    float facing_threshold, bool selected);

    // Per-entity static display buffers (inactive entities).
    void upload_display(EntityGpu& g, const Mesh& mesh);
    void free_display(EntityGpu& g);
    void draw_display(const Camera& cam, EntityGpu& g, int w, int h,
                      float facing_threshold, bool selected);

    // Entity-id pick pass. Draw all entities (active working VAO + inactive
    // display VAOs) into the screen FBO writing linear depth (attachment 0) and
    // a per-draw entity id (attachment 2). Read back with read_id_region (id)
    // and read_depth_region (depth → unproject for insert).
    void pick_begin(const Camera& cam, int w, int h);
    void pick_draw(uint32_t entity_id, GLuint vao, uint32_t index_count);
    void pick_end();
    void read_id_region(int x, int y, int w, int h, uint32_t* out);
    void draw_cursor(const Camera& cam, float cx, float cy, float radius,
                     float nx, float ny, float nz, float hardness,
                     int w, int h, bool on_model);

    // Debug visualization
    void draw_debug_mesh(const Camera& cam, const Mesh& mesh, int w, int h);
    void invalidate_debug_mesh() { debug_edge_count = 0; }
};

GLuint compile_shader(GLenum type, const char* src);
GLuint link_program(GLuint vert, GLuint frag);
