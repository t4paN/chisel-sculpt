#pragma once
#include <glad/glad.h>
#include <cstdint>
#include <vector>

// SSBO binding points (shared between C++ and GLSL)
enum ComputeBinding : GLuint {
    BIND_POSITIONS   = 0,
    BIND_NORMALS     = 1,
    BIND_INDICES     = 2,
    BIND_ACCUM       = 3,
    BIND_ADJACENCY_OFFSET = 4,
    BIND_ADJACENCY_LIST   = 5,
    BIND_DIRTY_VERTS = 6,
    BIND_MIRROR_MAP  = 7,
    BIND_MOVE_AFFECTED   = 8,
    BIND_MOVE_WEIGHTS    = 9,
    BIND_MOVE_WEIGHTS_PONG = 10,
    BIND_MOVE_INIT       = 11,
    BIND_MASK            = 12,
    BIND_REMESH_IN_POS   = 13,   // remesh smooth: ping positions (input)
    BIND_REMESH_WEIGHTS  = 14,   // remesh smooth: per-vertex weights
    BIND_REMESH_PINNED   = 15,   // remesh smooth: per-vertex pinned flags
    BIND_REMESH_TRISEL   = 16,   // remesh smooth: per-tri selection flags (full/grown)
    BIND_REMESH_CORE_SEL = 17,   // remesh smooth_weights: per-tri core selection
    BIND_REMESH_TRISEL_PONG = 18,// remesh grow/mirror selection: per-tri input snapshot
    BIND_SEAM_WELD_MAP      = 19,// seam weld: per-vertex merge target
    BIND_ACCUM_SYM          = 20,// draw symmetrize: output (apply reads as accum when bound at BIND_ACCUM)
    // Voxel-merge (SDF) bindings — see sdf.h / sdf-remesh-spec.md.
    BIND_SDF_SOUP_POS = 21,  // float3 soup vertex positions (world)
    BIND_SDF_SOUP_IDX = 22,  // uint   soup triangle indices (3 per tri)
    BIND_SDF_DIST     = 23,  // uint   band distance grid (float bits, atomicMin)
    BIND_SDF_FIELD    = 24,  // float  signed field grid  (MC input)
    BIND_SDF_MC_OUT   = 25,  // float  MC output soup (pos+normal interleaved)
    BIND_SDF_MC_COUNT = 26,  // uint   MC output triangle counter (atomic)
    BIND_SDF_TRITABLE = 27,  // int    MC 256x16 triangle table
    BIND_COLOR        = 28,  // uint   per-vertex packed RGBA8 albedo (paint)
};

struct DrawAccumParams {
    float anchor_a_x, anchor_a_y, anchor_a_z;
    float anchor_b_x, anchor_b_y, anchor_b_z;  // ignored when use_b == 0
    int   use_b;                               // 0 or 1
    float world_radius;
    float disp_amount;
    float hardness;
    float facing_threshold;
    float view_a_x, view_a_y, view_a_z;
    float view_b_x, view_b_y, view_b_z;
    // Anchor surface normal at the cursor pixel. Each dab blends the vert's
    // frozen normal toward this so crossing ridges don't fold geometry along
    // the flank normals.
    float anchor_normal_a_x, anchor_normal_a_y, anchor_normal_a_z;
    float anchor_normal_b_x, anchor_normal_b_y, anchor_normal_b_z;
    uint32_t vertex_count;
    int inflate;  // 0 = draw (push along cursor normal), 1 = inflate (per-vertex normal)
};

struct SmoothAccumParams {
    float anchor_x, anchor_y, anchor_z;
    float world_radius;
    float hardness;
    int region_x, region_y, region_w, region_h;
    int screen_h;
    uint32_t tri_count;
    uint32_t vertex_count;
    float strength;
    int iterations;
    bool mirror_x;
};

struct CreaseAccumParams {
    float anchor_a_x, anchor_a_y, anchor_a_z;
    float anchor_b_x, anchor_b_y, anchor_b_z;
    int   use_b;
    float world_radius;
    float disp_amount;
    float pinch_amount;
    float hardness;
    float facing_threshold;
    float view_a_x, view_a_y, view_a_z;
    float view_b_x, view_b_y, view_b_z;
    float anchor_normal_a_x, anchor_normal_a_y, anchor_normal_a_z;
    float anchor_normal_b_x, anchor_normal_b_y, anchor_normal_b_z;
    uint32_t vertex_count;
};

struct PinchAccumParams {
    float anchor_a_x, anchor_a_y, anchor_a_z;
    float anchor_b_x, anchor_b_y, anchor_b_z;
    int   use_b;
    float world_radius;
    float pinch_amount;
    float hardness;
    float facing_threshold;
    float view_a_x, view_a_y, view_a_z;
    float view_b_x, view_b_y, view_b_z;
    float anchor_normal_a_x, anchor_normal_a_y, anchor_normal_a_z;
    float anchor_normal_b_x, anchor_normal_b_y, anchor_normal_b_z;
    uint32_t vertex_count;
};

struct MaskPaintParams {
    float anchor_a_x, anchor_a_y, anchor_a_z;
    float anchor_b_x, anchor_b_y, anchor_b_z;
    int   use_b;
    float world_radius;
    float hardness;
    float paint_strength;
    uint32_t vertex_count;
};

struct ColorPaintParams {
    float anchor_a_x, anchor_a_y, anchor_a_z;
    float anchor_b_x, anchor_b_y, anchor_b_z;
    int   use_b;
    float world_radius;
    float hardness;
    float paint_strength;       // lerp factor toward paint_color (0..1, * falloff)
    float paint_r, paint_g, paint_b;  // brush albedo, linear [0,1]
    uint32_t vertex_count;
};

struct MoveCaptureParams {
    float anchor_x, anchor_y, anchor_z;
    float world_radius;
    float hardness;
    bool mirror_x;
    uint32_t vertex_count;
};

struct MoveApplyParams {
    float total_dx, total_dy, total_dz;
    bool mirror_x;
    uint32_t vertex_count;  // for safety bounds (unused inside shader, kept for symmetry)
};

struct ComputeState {
    bool supported;
    bool has_native_float_atomics;  // GL_NV_shader_atomic_float
    int max_workgroup_size;
    int max_workgroup_invocations;
    int max_ssbo_bindings;

    // Accumulation buffer: 4 uints per vertex (dx, dy, dz, weight as float-bits)
    GLuint accum_ssbo;
    uint32_t accum_vertex_count;

    // Symmetrized accumulation buffer (same shape as accum_ssbo). Populated by
    // the draw symmetrize pass when mirror_x is on; each paired (v, mv) gets
    // out[v] = accum[v] + (-mx, my, mz, mw), so apply produces byte-for-byte
    // mirrored displacements regardless of tessellation drift between twins.
    GLuint accum_sym_ssbo;

    // Draw accumulate compute shader
    GLuint draw_accum_program;

    // Draw symmetrize-accum compute shader: folds mirror twin's accum into v's
    // own accum so paired displacements are strictly X-mirror symmetric.
    GLuint draw_accum_symmetrize_program;

    // Stroke-begin snapshot of vertex normals. Draw_accum reads displacement
    // direction from this frozen buffer so each dab moves verts along the
    // surface normal as it was at pen-down, not the live (already-perturbed)
    // VBO normal. Without this, every dab biases toward the camera and long
    // strokes near silhouettes fold triangles.
    GLuint stroke_norm_ssbo;
    uint32_t stroke_norm_capacity;

    // Draw apply compute shader (normalizes accum → writes positions)
    GLuint draw_apply_program;

    // Draw mirror apply compute shader (writes negated-X displacements to mirror twins)
    GLuint draw_mirror_apply_program;

    // Smooth brush compute shaders
    GLuint smooth_accum_program;
    GLuint smooth_apply_program;
    GLuint smooth_mirror_apply_program;

    // Stroke autosmooth: one-pass Laplacian over a vert-ID list at fixed strength.
    // Runs at pen-up over snap_list verts on draw strokes when autosmooth is enabled.
    GLuint stroke_smooth_apply_program;

    // Crease brush compute shader
    GLuint crease_accum_program;

    // Pinch brush compute shader
    GLuint pinch_accum_program;

    // Mask brush compute shader (writes directly to mask VBO/SSBO)
    GLuint mask_paint_program;

    // Paint brush compute shader (writes directly to color VBO/SSBO)
    GLuint color_paint_program;

    // Move brush compute shaders (stateful: capture-once weights + per-dab apply)
    GLuint move_capture_program;        // brute-force per-vertex world-distance gate: sets weight, init pos, appends affected
    GLuint move_weight_smooth_program;  // ping-pong Laplacian over affected list
    GLuint move_apply_program;          // per-dab: pos = init + total*w.x + mirror_total*w.y (mirror summed in-place)

    // Move stroke buffers (allocated lazily, persist across strokes; resized as vertex_count grows)
    GLuint move_affected_ssbo;       // [count, v0, v1, ...]
    GLuint move_weights_ssbo;        // vec2 per vertex: (primary, mirror) brush weights
    GLuint move_weights_pong_ssbo;   // vec2 per vertex (ping-pong scratch)
    GLuint move_init_ssbo;           // 3 floats per vertex (snapshotted at capture)
    uint32_t move_buffers_capacity;

    // Compute normals shader
    GLuint compute_normals_program;

    // Adjacency CSR SSBOs (uploaded once at mesh init)
    GLuint adjacency_offset_ssbo;
    GLuint adjacency_list_ssbo;
    uint32_t adjacency_vertex_count;

    // Mirror map SSBO (uploaded once at mesh init, maps vertex -> twin vertex)
    GLuint mirror_map_ssbo;
    uint32_t mirror_map_vertex_count;

    // Mask SSBO: alias of renderer.vbo_mask. Set once after renderer init so the
    // apply shaders can scale per-vertex displacement by (1 - mask[v]). The CPU
    // mask brush keeps mesh.mask authoritative and uploads via vbo_mask, so this
    // GPU view is always in sync with what the user painted.
    GLuint mask_ssbo;

    // Color SSBO: alias of renderer.vbo_color. Set once after renderer init so
    // the paint shader writes packed RGBA8 directly into the display VBO.
    GLuint color_ssbo;

    // Dirty vertex list SSBO (uploaded per dispatch, used by compute_normals)
    GLuint dirty_verts_ssbo;
    uint32_t dirty_verts_capacity;

    // Smooth compact dirty list SSBO: [count, v0, v1, ...] written by smooth_accum
    GLuint smooth_dirty_ssbo;
    uint32_t smooth_dirty_capacity;  // max vertex IDs (excludes counter slot)

    // Remesh per-tri selection GPU passes
    GLuint remesh_select_stretched_program;
    GLuint remesh_select_unmasked_program;
    // Remesh per-tri grow + mirror selection (keeps tri_sel GPU-resident)
    GLuint remesh_grow_selection_program;
    GLuint remesh_mirror_selection_program;
    // Remesh per-vertex pinned-boundary detection
    GLuint remesh_find_pinned_program;
    // Remesh per-vertex smooth weight computation
    GLuint remesh_smooth_weights_program;
    GLuint remesh_core_sel_ssbo;    // uint[T] — core (pre-grow) tri selection
    GLuint remesh_trisel_pong_ssbo; // uint[T] — input snapshot for grow/mirror

    // Remesh seam snap + weld GPU passes
    GLuint remesh_seam_snap_program;
    GLuint remesh_seam_weld_program;
    GLuint seam_weld_map_ssbo;       // uint[V] — merge target per vert (v or lower match)

    // Remesh tangential smooth GPU pass (ping-pong, no hot-path allocation)
    GLuint remesh_smooth_program;
    GLuint remesh_ping_ssbo;       // float[V*3] SOA — positions input
    GLuint remesh_pong_ssbo;       // float[V*3] SOA — positions output
    GLuint remesh_norm_ssbo;       // float[V*3] SOA — normals
    GLuint remesh_weights_ssbo;    // float[V]   — smooth weights (1.0 if non-mask)
    GLuint remesh_pinned_ssbo;     // uint[V]    — 0/1 pinned flags
    GLuint remesh_trisel_ssbo;     // uint[T]    — 0/1 tri selected flags
    GLuint remesh_indices_ssbo;    // uint[T*3]  — triangle indices snapshot
    uint32_t remesh_vert_capacity;
    uint32_t remesh_tri_capacity;

    // Readback scratch buffer (persistent to avoid per-dab allocation)
    std::vector<float> readback_buf;

    ComputeState();

    // Probe GL extensions after GLAD loads. Returns true if compute is usable.
    bool init();

    // Compile a compute shader source into a linked program. Returns 0 on failure.
    GLuint compile_program(const char* src) const;

    // Allocate/resize the accumulation buffer for the given vertex count.
    void ensure_accum_buffer(uint32_t vertex_count);

    // Clear the accumulation buffer to zero. Call before each dab.
    void clear_accum_buffer();

    // Compile the draw-brush accumulation compute shader. Called once at init.
    bool init_draw_accum();

    // Compile the draw-brush apply compute shader. Called once at init.
    bool init_draw_apply();

    // Compile the draw-brush mirror apply compute shader. Called once at init.
    bool init_draw_mirror_apply();

    // Snapshot the current per-vertex normals from norm_vbo into stroke_norm_ssbo.
    // Call once at the start of a draw/crease/pinch stroke. The working buffer holds
    // only the active entity at offset 0, so this copies [0, vertex_count) straight
    // across. dispatch_draw_accum reads normals[v*3].
    void snapshot_stroke_normals(GLuint norm_vbo, uint32_t vertex_count);

    // Dispatch the draw accumulate shader once over all vertices.
    // Reads positions from pos_vbo and normals from stroke_norm_ssbo (must be
    // populated by snapshot_stroke_normals before calling). Caller must clear
    // accum buffer beforehand. Issues memory barrier.
    void dispatch_draw_accum(const DrawAccumParams& params, GLuint pos_vbo);

    // Read back accum buffer into readback_buf (sized vertex_count * 4).
    void readback_accum(uint32_t vertex_count);

    // Upload finalized displacement data (after target-height + Laplacian) to accum SSBO.
    // Packs disp_x/y/z/weight into the 4-uint-per-vertex layout expected by the apply shader.
    void upload_accum(const float* disp_x, const float* disp_y, const float* disp_z,
                      const float* disp_weight, uint32_t vertex_count);

    // Dispatch the apply shader: normalizes accum (xyz/w) and adds to position SSBO.
    // pos_vbo is bound as SSBO at BIND_POSITIONS. If accum_override != 0, that
    // buffer is used as the accum source instead of accum_ssbo (used by the draw
    // brush to read from accum_sym_ssbo after symmetrize).
    void dispatch_draw_apply(GLuint pos_vbo, uint32_t vertex_count,
                              GLuint accum_override = 0);

    // Compile the draw-brush symmetrize-accum compute shader. Called once at init.
    bool init_draw_accum_symmetrize();

    // Dispatch symmetrize: reads accum_ssbo + mirror_map_ssbo, writes
    // accum_sym_ssbo. Caller must have populated accum via dispatch_draw_accum
    // and issued a SHADER_STORAGE_BARRIER first.
    void dispatch_draw_accum_symmetrize(uint32_t vertex_count);

    // Upload mirror map to GPU. Call at mesh init and after remesh.
    void upload_mirror_map(const std::vector<uint32_t>& map);

    // Dispatch mirror apply shader: writes negated-X displacements to mirror twin vertices.
    // Only called when mirror_x is active and mirror_map is uploaded.
    void dispatch_draw_mirror_apply(GLuint pos_vbo, uint32_t vertex_count);

    // Compile and dispatch the smooth brush compute shaders.
    bool init_smooth();
    void dispatch_smooth(const SmoothAccumParams& params,
                         GLuint triid_tex, GLuint bary_tex,
                         GLuint pos_vbo, GLuint index_ebo);

    // Ensure the smooth compact dirty list SSBO is large enough for max_verts IDs.
    void ensure_smooth_dirty_buffer(uint32_t max_verts);

    // Read back the compact dirty list written by smooth_accum.
    // Returns the count; fills out with vertex IDs.
    uint32_t readback_smooth_dirty(std::vector<uint32_t>& out);

    // Scan accum buffer for vertices where w > 0 (touched by the brush).
    // Returns the count; fills out with vertex IDs (not positions).
    uint32_t readback_accum_dirty(uint32_t vertex_count, std::vector<uint32_t>& out);

    // Mirror smoothed primary-side positions onto twins (reflect x).
    // Gated on accum weight > 0 AND vert sharing anchor's x sign, so straddle
    // triangles near the seam don't overwrite anchor-side twins from the
    // opposite side.
    void dispatch_smooth_mirror_apply(GLuint pos_vbo, uint32_t vertex_count, float anchor_x);

    // Compile the stroke autosmooth shader. Called once at init.
    bool init_stroke_smooth();

    // Dispatch a single Laplacian smoothing pass over the given vertex IDs.
    // Reads positions in-place from pos_vbo; mask gates per-vertex.
    void dispatch_stroke_smooth_apply(const uint32_t* vert_ids, uint32_t count,
                                      float strength,
                                      GLuint pos_vbo, GLuint index_ebo);

    // Compile the crease brush accumulate compute shader. Called once at init.
    bool init_crease();

    // Dispatch the crease accumulate shader over all vertices.
    // Reads frozen normals from stroke_norm_ssbo (must be populated by snapshot_stroke_normals).
    void dispatch_crease_accum(const CreaseAccumParams& params, GLuint pos_vbo);

    // Compile the pinch brush accumulate compute shader. Called once at init.
    bool init_pinch();

    // Dispatch the pinch accumulate shader over all vertices.
    // Reads frozen normals from stroke_norm_ssbo (must be populated by snapshot_stroke_normals).
    void dispatch_pinch_accum(const PinchAccumParams& params, GLuint pos_vbo);

    // Compile the mask brush compute shader. Called once at init.
    bool init_mask();
    bool init_color();

    // Dispatch the mask paint shader: per-vertex distance check, writes mask VBO
    // directly. Uses smooth_dirty_ssbo for the compact dirty list. Caller reads
    // back dirty list via readback_smooth_dirty.
    void dispatch_mask_paint(const MaskPaintParams& params, GLuint pos_vbo);
    void dispatch_color_paint(const ColorPaintParams& params, GLuint pos_vbo);

    // Compile the four move-brush compute shaders. Called once at init.
    bool init_move();

    // Allocate/resize move stroke buffers (weights, init pos, affected list).
    void ensure_move_buffers(uint32_t vertex_count);

    // Capture pass: brute-force over vertex_count threads. Gates by world-distance
    // to anchor (and mirror x-sign if mirror_x). Writes weight, init position,
    // appends to affected list. Clears affected count + weights up front.
    void dispatch_move_capture(const MoveCaptureParams& params, GLuint pos_vbo);

    // Smooth weights: ping-pong Laplacian iterations over the affected set.
    void dispatch_move_weight_smooth(uint32_t vertex_count, int iterations, GLuint index_ebo);

    // Apply pass: dispatched over affected list. positions[v] = init[v] + total*w.
    // Sums primary + mirrored brush contributions per vertex (mirror handled in-shader).
    void dispatch_move_apply(const MoveApplyParams& params, GLuint pos_vbo);

    // Readback affected list. Returns count; fills out[].
    uint32_t readback_move_affected(std::vector<uint32_t>& out);

    // Compile the compute normals shader. Called once at init.
    bool init_compute_normals();

    // Upload adjacency CSR to GPU. Call at mesh init and after remesh.
    void upload_adjacency(const uint32_t* offsets, uint32_t offset_count,
                          const uint32_t* list, uint32_t list_count);

    // Dispatch compute normals for a set of affected vertices.
    // Reads positions from pos_vbo, writes normals to norm_vbo.
    void dispatch_compute_normals(const uint32_t* dirty_verts, uint32_t dirty_count,
                                  GLuint pos_vbo, GLuint norm_vbo, GLuint index_ebo);

    // Compile remesh GPU shaders. Called once at init.
    bool init_remesh_select();
    bool init_remesh_grow_selection();
    bool init_remesh_mirror_selection();
    bool init_remesh_find_pinned();
    bool init_remesh_smooth_weights();
    bool init_remesh_smooth();
    bool init_remesh_seam_snap_weld();

    // Grow remesh SSBOs to fit the given mesh dimensions if needed.
    void ensure_remesh_smooth_buffers(uint32_t vertex_count, uint32_t tri_count);

    // GPU per-tri stretched-triangle classification.
    // Uploads positions + indices and writes remesh_trisel_ssbo on GPU.
    // No readback — caller pulls the result via readback_trisel after the full
    // select → grow → mirror chain so tri_sel stays GPU-resident.
    void dispatch_select_stretched(
        uint32_t vertex_count,
        uint32_t tri_count,
        const uint32_t* mesh_indices,
        const float* pos_x, const float* pos_y, const float* pos_z,
        float target_len);

    // GPU per-tri unmasked-triangle classification.
    // mask may be null (empty mesh.mask → all unmasked). Writes remesh_trisel_ssbo;
    // no readback (see dispatch_select_stretched).
    void dispatch_select_unmasked(
        uint32_t vertex_count,
        uint32_t tri_count,
        const uint32_t* mesh_indices,
        const float* mask,
        uint32_t mask_size);

    // GPU per-tri ring-grow on remesh_trisel_ssbo (in place). N rings of
    // BFS-by-shared-vertex via the already-uploaded adjacency SSBOs.
    void dispatch_grow_selection(uint32_t vertex_count,
                                 uint32_t tri_count,
                                 int rings);

    // GPU per-tri mirror-spread on remesh_trisel_ssbo (in place). For each
    // selected tri, ORs its 1 into every tri adjacent to the mirror of any of
    // its vertices. Uses the already-uploaded mirror_map_ssbo.
    void dispatch_mirror_selection(uint32_t vertex_count, uint32_t tri_count);

    // Snapshot remesh_trisel_ssbo into remesh_core_sel_ssbo (GPU-side copy).
    // Mask path uses this to preserve the pre-grow selection for smooth_weights.
    void snapshot_core_sel(uint32_t tri_count);

    // Readback remesh_trisel_ssbo into a CPU vector (resized to tri_count).
    void readback_trisel(uint32_t tri_count, std::vector<uint32_t>& out);

    // GPU per-vertex smooth weight computation.
    // Reads pinned + full_sel + core_sel from already-uploaded remesh SSBOs.
    // Writes weights to remesh_weights_ssbo — no CPU readback; smooth reads it directly.
    void dispatch_compute_smooth_weights(
        uint32_t vertex_count,
        uint32_t tri_count,
        int support_rings);

    // GPU boundary-vertex detection for remesh.
    // Uploads positions; tri_sel is assumed already in remesh_trisel_ssbo.
    // Seam snap (pos_x = 0) must be applied CPU-side after readback using the returned pinned flags.
    void dispatch_find_pinned(
        uint32_t vertex_count,
        uint32_t tri_count,
        const float* pos_x, const float* pos_y, const float* pos_z,
        float seam_tol,
        std::vector<uint32_t>& out_pinned);

    // GPU tangential smooth for remesh. Reads pos/norm from CPU arrays, dispatches,
    // readbacks into out_pos_x/y/z. Caller must upload_adjacency first; tri_sel is
    // assumed already in remesh_trisel_ssbo (set by prior select/grow/mirror).
    // weights may be empty (non-mask path) — shader uses 1.0 for all verts in that case.
    void dispatch_remesh_smooth(
        uint32_t vertex_count,
        uint32_t tri_count,
        const uint32_t* mesh_indices,
        const float* pos_x, const float* pos_y, const float* pos_z,
        const float* norm_x, const float* norm_y, const float* norm_z,
        const std::vector<float>& weights,
        const std::vector<uint32_t>& pinned,
        float lambda,
        float seam_tol,
        float* out_pos_x, float* out_pos_y, float* out_pos_z,
        bool weights_on_gpu = false);  // skip weights upload when GPU already computed them

    // GPU seam snap + weld for post-remesh mirror. Snaps verts near x=0 (within
    // snap_tol) using topological detection (neighbors on both sides), then welds
    // spatially-close verts at x=0 within weld_tol. Writes snapped positions back
    // to pos_x and fills out_merge_map (same-index = no merge, lower index = merge target).
    // Requires adjacency already uploaded.
    void dispatch_seam_snap_weld(
        uint32_t vertex_count,
        float* pos_x, const float* pos_y, const float* pos_z,
        const float* mask, uint32_t mask_size,
        float seam_tol, float snap_tol, float weld_tol,
        std::vector<uint32_t>& out_merge_map);

    void cleanup();
};
