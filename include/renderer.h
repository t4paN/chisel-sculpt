#pragma once
#include <glad/glad.h>
#include "mesh.h"
#include "camera.h"
#include "entity_gpu.h"
#include "gpu/gpu.h"

struct Renderer {
    // gpu:: seam device (== gpu::gl_device() on GL). Render programs migrate off
    // raw GL onto the seam one at a time (mirrors the compute Seam 2b port); the
    // still-raw programs below keep their GLuint members until ported.
    gpu::Device gpu_dev;

    // Mesh GPU buffers
    GLuint vao;
    // Shared render+compute mesh buffers — owned seam buffers (buffer-ownership
    // migration Step 1). pos/norm carry Vertex|Storage, ebo Index|Storage: the
    // render path binds them as vertex/index buffers, the brush kernels bind them
    // as storage. Passed by `.handle` into the still-GLuint dispatch_* signatures
    // until those flip to gpu::Buffer in later steps.
    gpu::Buffer vbo_pos;
    gpu::Buffer vbo_norm;
    GLuint vbo_mask;      // per-vertex sculpt mask values (0..1) — Step 3 (aliased by compute.mask_ssbo)
    GLuint vbo_color;     // per-vertex albedo, packed RGBA8 (paint) — Step 3 (aliased by compute.color_ssbo)
    GLuint vbo_tri_id;    // per-vertex triangle ID (flat, 3 verts per tri)
    GLuint vbo_bary;     // per-vertex barycentric coord
    gpu::Buffer ebo;

    // Matcap shader — on the gpu:: seam. Loose uniforms (uView/uProj/facing/
    // objMask/paintVisible) become a std140 Params UBO uploaded per draw.
    gpu::RenderPipeline matcap_pipeline;
    gpu::Buffer         matcap_ubo;

    // Vertex-paint visibility (1 = show albedo, 0 = plain matcap). Set per frame.
    float paint_visible = 1.0f;

    // Entity-id pick pass — on the gpu:: seam. Renders into the shared screen
    // offscreen target writing linear depth (attachment 0) + entity id (attachment
    // 2). The pass spans pick_begin → N×pick_draw → pick_end, so the RenderPass +
    // bind group are held as members across those calls.
    gpu::RenderPipeline pick_pipeline;
    gpu::Buffer         pick_ubo;        // view/proj + per-draw entity id
    gpu::RenderPass     pick_pass;       // live only between pick_begin and pick_end
    gpu::BindGroup      pick_bg;

    // Background shader (gradient) — on the gpu:: seam.
    gpu::RenderPipeline bg_pipeline;
    gpu::Buffer         bg_vbuf;

    // Brush cursor overlay — on the gpu:: seam. Three render pipelines (ring /
    // footprint disc / centre crosshair), each with static camera-independent
    // geometry built once and a std140 Params UBO updated per draw.
    gpu::RenderPipeline cursor_pipeline;
    gpu::RenderPipeline shadow_pipeline;
    gpu::RenderPipeline crosshair_pipeline;
    gpu::Buffer cursor_vbuf;     // unit ring (TriangleStrip)
    gpu::Buffer shadow_vbuf;     // unit disc (Triangles, fan converted to a list)
    gpu::Buffer crosshair_vbuf;  // fixed centre X (Lines)
    gpu::Buffer cursor_ubo;
    gpu::Buffer shadow_ubo;
    gpu::Buffer crosshair_ubo;

    // Debug visualization (wireframe edge overlay) — on the gpu:: seam. Lines
    // pipeline + a std140 Params UBO (view/proj); the GL-owned edge index buffer
    // is built lazily and bound through the seam.
    gpu::RenderPipeline debug_edge_pipeline;
    gpu::Buffer         debug_edge_ubo;
    GLuint debug_edge_vbo;         // edge index buffer (GL-owned, built lazily)
    uint32_t debug_edge_count;

    // Screen-buffer MRT for the brush pipeline — on the gpu:: seam. A 4-attachment
    // offscreen target (R32F depth / RGB16F normal / R32UI triid / RG16F bary) +
    // depth RBO, shared with the pick pass. The MRT render pipeline writes all four;
    // loose uView/uProj become a std140 view/proj UBO.
    gpu::OffscreenTarget screen_target;
    gpu::RenderPipeline  screen_pipeline;
    gpu::Buffer          screen_ubo;

    // Flat triangle-soup vertex buffers for the screen pass (no index sharing).
    // pos/norm are filled by the GPU-side expand compute; triid/bary are static per
    // topology. GL-owned (also bound as SSBOs by the expand kernel), wrapped in
    // gpu::Buffer views at draw/dispatch — same staged pattern as the brush SSBOs.
    GLuint screen_vbo_pos;
    GLuint screen_vbo_norm;
    GLuint screen_vbo_triid;
    GLuint screen_vbo_bary;
    uint32_t screen_tri_count;

    // GPU-side indexed→flat expansion — on the gpu:: compute seam.
    gpu::ComputePipeline screen_expand_pipeline;
    gpu::Buffer          screen_expand_ubo;

    // Screen MRT colour-texture handles (triid = attachment 2, bary = attachment 3)
    // — still passed to the smooth dispatch's (currently unused) texture params.
    GLuint screen_triid_tex() const { return screen_target.color_tex[2]; }
    GLuint screen_bary_tex()  const { return screen_target.color_tex[3]; }

    bool initialized;

    Renderer();
    ~Renderer();

    void init();
    void upload_mesh(const Mesh& mesh);
    void update_mask(const Mesh& mesh);
    void update_mask_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts);
    void update_color(const Mesh& mesh);
    void update_color_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts);

    // Partial update: sync dirty vertex positions+normals from CPU mesh to VBOs
    void update_mesh_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts);

    // Scattered update: upload ONLY the listed verts (coalesced into runs of
    // consecutive indices), never the span between them. Used by undo so it
    // touches exactly the affected verts and leaves all others — and their
    // GPU-computed normals — untouched.
    void update_mesh_verts(const Mesh& mesh, const std::vector<uint32_t>& verts);
    void update_mask_verts(const Mesh& mesh, const std::vector<uint32_t>& verts);
    void update_color_verts(const Mesh& mesh, const std::vector<uint32_t>& verts);

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
    // Fill + upload the matcap Params UBO (camera + per-draw flags); shared by
    // draw_mesh and draw_display.
    void upload_matcap_params(const Camera& cam, int w, int h,
                              float facing_threshold, bool selected);
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
    void pick_draw(uint32_t entity_id, GLuint pos_vbo, GLuint ebo, uint32_t index_count);
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
