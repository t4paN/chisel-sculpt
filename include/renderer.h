#pragma once
#include <glad/glad.h>
#include <vector>
#include "mesh.h"
#include "camera.h"
#include "entity_gpu.h"
#include "gpu/gpu.h"

struct Renderer {
    // gpu:: seam device (== gpu::gl_device() on GL). All render programs and their
    // buffers are now on the seam (mirrors the compute Seam 2b port).
    gpu::Device gpu_dev;

    // Shared render+compute mesh buffers — owned seam buffers (buffer-ownership
    // migration Step 1). pos/norm carry Vertex|Storage, ebo Index|Storage: the
    // render path binds them as vertex/index buffers, the brush kernels bind them
    // as storage. (The draw-time VAO lives on the render pipeline — no renderer VAO.)
    gpu::Buffer vbo_pos;
    gpu::Buffer vbo_norm;
    // Owned seam buffers (Step 3a). Both carry Vertex|Storage: render binds them as
    // vertex attributes, the brush kernels write them as storage (mask gate / paint).
    // compute.mask_ssbo / color_ssbo alias these; the alias is *refreshed* after every
    // (re)allocation in Scene::bind_active_ (a WebGPU grow makes a new handle — the old
    // set-once GL alias relied on stable-handle realloc, which WebGPU doesn't have).
    gpu::Buffer vbo_mask;      // per-vertex sculpt mask values (0..1)
    gpu::Buffer vbo_color;     // per-vertex albedo, packed RGBA8 (paint)
    gpu::Buffer ebo;

    // Matcap shader — on the gpu:: seam. Loose uniforms (uView/uProj/facing/
    // objMask/paintVisible) become a std140 Params UBO uploaded per draw.
    gpu::RenderPipeline matcap_pipeline;
    gpu::Buffer         matcap_ubo;

    // Vertex-paint visibility (1 = show albedo, 0 = plain matcap). Set per frame.
    float paint_visible = 1.0f;

    // Entity-id pick pass — on the gpu:: seam. Renders into the shared screen
    // offscreen target writing linear depth (attachment 0) + entity id (attachment
    // 2). The pass spans pick_begin → N×pick_draw → pick_end.
    //
    // Each draw uses its OWN UBO + bind group (a growable pool, indexed by pick_slot).
    // A distinct buffer per draw is mandatory on WebGPU: writeBuffer runs on the queue
    // timeline at submit, so N writes into ONE shared UBO all collapse to the last
    // value — every entity would then read the final id (the old single-pick_ubo bug).
    // The pool persists across picks (picking is user-paced, not a hot path).
    gpu::RenderPipeline         pick_pipeline;
    gpu::RenderPass             pick_pass;   // live only between pick_begin and pick_end
    std::vector<gpu::Buffer>    pick_ubos;   // view/proj + entity id, one per draw slot
    std::vector<gpu::BindGroup> pick_bgs;    // parallel to pick_ubos
    uint32_t                    pick_slot = 0;
    float                       pick_view[16];
    float                       pick_proj[16];

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

    // Debug visualization (wireframe edge overlay) — on the gpu:: seam. Fat-line
    // Triangles pipeline (screen-space ribbon expansion in the vertex stage, AA
    // feather in the fragment stage) + a std140 Params UBO; the edge-pair SSBO
    // is built lazily and bound through the seam.
    gpu::RenderPipeline debug_edge_pipeline;
    gpu::Buffer         debug_edge_ubo;
    gpu::Buffer debug_edge_vbo;    // unique edge-pair SSBO (seam-owned, built lazily)
    uint32_t debug_edge_count;     // u32 entries in it (2 per unique edge)
    uint32_t debug_edge_src_tris;  // tri count it was built from (stale-cache check)
    float    debug_mesh_radius;    // bounding radius at build (zoom-adaptive width)

    // Screen-buffer MRT for the brush pipeline — on the gpu:: seam. A 4-attachment
    // offscreen target (R32F depth / RGB16F normal / R32UI triid / RG16F bary) +
    // depth RBO, shared with the pick pass. The MRT render pipeline writes all four;
    // loose uView/uProj become a std140 view/proj UBO.
    gpu::OffscreenTarget screen_target;
    gpu::RenderPipeline  screen_pipeline;
    gpu::Buffer          screen_ubo;

    // Flat triangle-soup vertex buffers for the screen pass (no index sharing).
    // pos/norm are filled by the GPU-side expand compute (so they carry
    // Vertex|Storage); triid/bary are static per topology (Vertex). Seam-owned.
    gpu::Buffer screen_vbo_pos;
    gpu::Buffer screen_vbo_norm;
    gpu::Buffer screen_vbo_triid;
    gpu::Buffer screen_vbo_bary;
    uint32_t screen_tri_count;

    // GPU-side indexed→flat expansion — on the gpu:: compute seam.
    gpu::ComputePipeline screen_expand_pipeline;
    gpu::Buffer          screen_expand_ubo;

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

    // Cursor-sample plane cache — the in-frame read path for the screen buffers.
    // On WebGPU a 1×1 readback is a full GPU sync (and a fatal suspend on web), so
    // render_screen_buffers kicks an async full-plane read of depth/normal/triid,
    // poll_plane_reads() lands it a frame or two later, and sample_* index the CPU
    // copy (false = cache not landed yet / out of bounds — caller keeps its last
    // value or skips the dab). On GL sample_* are the same cheap immediate 1×1
    // glReadPixels as before and the cache machinery is a no-op.
    void poll_plane_reads();                          // call once per frame
    bool sample_depth(int x, int y, float* out);      // attachment 0, linear distance
    bool sample_normal(int x, int y, float out[3]);   // attachment 1, raw normal texel
    bool sample_triid(int x, int y, uint32_t* out);   // attachment 2
#if defined(CHISEL_BACKEND_WEBGPU)
    std::vector<float>    plane_depth;
    std::vector<float>    plane_norm;                 // 3 floats per pixel
    std::vector<uint32_t> plane_triid;
    int  plane_w = 0, plane_h = 0;                    // dims of the landed planes
    int  plane_kick_w = 0, plane_kick_h = 0;          // dims of the in-flight kick
    gpu::ReadTicket plane_tk[3] = {0, 0, 0};
    bool plane_pending = false;
    bool plane_valid   = false;
#endif

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
    void pick_draw(uint32_t entity_id, const gpu::Buffer& pos_vbo, const gpu::Buffer& ebo, uint32_t index_count);
    void pick_end();
    void read_id_region(int x, int y, int w, int h, uint32_t* out);
    void draw_cursor(const Camera& cam, float cx, float cy, float radius,
                     float nx, float ny, float nz, float hardness,
                     int w, int h, bool on_model);

    // Debug visualization
    void draw_debug_mesh(const Camera& cam, const Mesh& mesh, int w, int h);
    void invalidate_debug_mesh() { debug_edge_count = 0; debug_edge_src_tris = 0; }
};
