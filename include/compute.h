#pragma once
#include <glad/glad.h>
#include "gpu/gpu.h"      // the GPU seam (RHI); backend chosen at build time
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
    BIND_LIMB_POS_SRC = 29,  // float3 read-side position snapshot for limb relax ping-pong
    // GPU-resident undo: pen-up multires diff (see multires_gpu.h, Phase 2b).
    BIND_MULTIRES_DISP     = 30, // float3 disp layer of the active level (in: snap, out: new)
    BIND_MULTIRES_FRAMES   = 31, // float9 tangent frame (t,b,n) per vert of the active level
    BIND_MULTIRES_SNAP_POS = 32, // float3 pen-down world position snapshot
    BIND_MULTIRES_BASE     = 33, // float3 base positions (out, base-level strokes)
    BIND_MULTIRES_STAGE    = 34, // float6 per touched vert: target(3) + source(3) disp (undo apply ring)
    BIND_UNDO_RING         = 35, // float6 per touched vert: old(3) + new(3) — persistent undo history (3b-iii)
    BIND_SDF_SPLAT_BOX     = 36, // uint6 per-tri voxel box: g0.xyz + span.xyz (SDF splat Pass A→C)
    BIND_SDF_SPLAT_OFFSET  = 37, // uint exclusive scan of footprint (SDF splat Pass C: owner search)
    BIND_SDF_FWN_NODES     = 38, // FwnNode[] BVH+dipole tree for the fast winding-number sign pass
    BIND_SDF_FWN_TRIORDER  = 39, // uint   leaf-contiguous triangle order (sign pass exact-leaf fetch)
    // Reserved high slot for the per-dispatch std140 Params UBO that replaces loose
    // GL uniforms on the gpu:: seam (see webgpu-port-plan.md / CONVENTIONS.md). Every
    // ported kernel binds its *ParamsGPU block here.
    BIND_PARAMS            = 63,
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

// Limb (snakehook) brush. Reuses the move capture (BIND_MOVE_WEIGHTS / affected /
// mask). drag is INCREMENTAL (pos += this-dab delta * w) rather than move's
// absolute init+total, so the per-dab tangential relax persists and accumulates.
struct LimbDragParams {
    float dx, dy, dz;       // this dab's world-space drag increment
    bool mirror_x;
    uint32_t vertex_count;
};

struct ComputeState {
    bool supported;
    bool has_native_float_atomics;  // GL_NV_shader_atomic_float

    // GPU seam device handle. On the GL backend this is an empty handle (a current
    // GL context must already exist); on WebGPU it carries the device+queue. Kernels
    // ported onto gpu:: (Seam Step 2b, mask first) dispatch through this.
    gpu::Device gpu_dev;
    int max_workgroup_size;
    int max_workgroup_invocations;
    int max_ssbo_bindings;

    // Accumulation buffer: 4 uints per vertex (dx, dy, dz, weight as float-bits)
    gpu::Buffer accum_ssbo;
    uint32_t accum_vertex_count;

    // Symmetrized accumulation buffer (same shape as accum_ssbo). Populated by
    // the draw symmetrize pass when mirror_x is on; each paired (v, mv) gets
    // out[v] = accum[v] + (-mx, my, mz, mw), so apply produces byte-for-byte
    // mirrored displacements regardless of tessellation drift between twins.
    gpu::Buffer accum_sym_ssbo;

    // Draw brush — ported onto the gpu:: seam (Seam Step 2b). draw_accum carries a
    // 112-byte std140 Params block; apply/symmetrize/mirror only need vertex_count,
    // so they share a 16-byte UBO. has_draw()/has_draw_symmetrize() report readiness.
    gpu::ComputePipeline draw_accum_pipeline;
    gpu::Buffer          draw_accum_ubo;       // 112-byte DrawAccumParams block
    gpu::Buffer          draw_vcount_ubo;      // 16-byte {vertex_count} — shared by apply/sym/mirror

    // Draw symmetrize-accum compute shader: folds mirror twin's accum into v's
    // own accum so paired displacements are strictly X-mirror symmetric.
    gpu::ComputePipeline draw_symmetrize_pipeline;

    // Stroke-begin snapshot of vertex normals. Draw_accum reads displacement
    // direction from this frozen buffer so each dab moves verts along the
    // surface normal as it was at pen-down, not the live (already-perturbed)
    // VBO normal. Without this, every dab biases toward the camera and long
    // strokes near silhouettes fold triangles.
    gpu::Buffer stroke_norm_ssbo;
    uint32_t stroke_norm_capacity;

    // Draw apply compute shader (normalizes accum → writes positions)
    gpu::ComputePipeline draw_apply_pipeline;

    // Draw mirror apply compute shader (writes negated-X displacements to mirror twins)
    gpu::ComputePipeline draw_mirror_apply_pipeline;

    // Smooth brush compute shaders — ported onto the gpu:: seam (Seam Step 2b).
    // Buffer-only (the triid/bary pick read is CPU-side back-projection in brush.cpp,
    // NOT a compute input), so it fits the existing seam with no texture-bind work.
    // accum carries a 32-byte Params block, apply a 16-byte one, mirror_apply a
    // 16-byte one. has_smooth() reports readiness.
    gpu::ComputePipeline smooth_accum_pipeline;
    gpu::ComputePipeline smooth_apply_pipeline;
    gpu::ComputePipeline smooth_mirror_apply_pipeline;
    gpu::Buffer          smooth_accum_ubo;        // 32-byte SmoothAccumParams block
    gpu::Buffer          smooth_apply_ubo;        // 16-byte {vcount, strength, mirror_x, seam_band}
    gpu::Buffer          smooth_mirror_ubo;       // 16-byte {vcount, anchor_x}

    // Stroke autosmooth: one-pass Laplacian over a vert-ID list at fixed strength.
    // Runs at pen-up over snap_list verts on draw strokes when autosmooth is enabled.
    gpu::ComputePipeline stroke_smooth_apply_pipeline;
    gpu::Buffer          stroke_smooth_ubo;   // 16-byte {dirty_count, strength} block

    // Crease + pinch brushes — ported onto the gpu:: seam (Seam Step 2b). Both are
    // accum-only kernels (deposit into the shared accum buffer); the apply side
    // reuses draw_apply / symmetrize / mirror_apply. Crease carries a 112-byte Params
    // block, pinch a 96-byte one.
    gpu::ComputePipeline crease_accum_pipeline;
    gpu::Buffer          crease_ubo;
    gpu::ComputePipeline pinch_accum_pipeline;
    gpu::Buffer          pinch_ubo;

    // Mask brush — ported onto the gpu:: seam (Seam Step 2b). Pipeline compiled from
    // the embedded mask_paint kernel; the std140 Params block (binding 63) replaces
    // the old loose uniforms and is uploaded per dab. has_mask() reports availability.
    gpu::ComputePipeline mask_pipeline;
    gpu::Buffer          mask_params_ubo;

    // Paint brush — ported onto the gpu:: seam (Seam Step 2b). Buffer-only: writes the
    // colour VBO/SSBO directly (lerp-to-colour) and the paint-smooth blends each vertex
    // colour toward its 1-ring neighbour average. paint carries a 64-byte std140 Params
    // block, smooth a 48-byte one. has_color()/has_color_smooth() report readiness.
    gpu::ComputePipeline color_paint_pipeline;
    gpu::ComputePipeline color_smooth_pipeline;
    gpu::Buffer          color_paint_ubo;     // 64-byte ColorPaintParams block
    gpu::Buffer          color_smooth_ubo;    // 48-byte ColorSmoothParams block

    // Move brush compute shaders — ported onto the gpu:: seam (Seam Step 2b).
    // Stateful: capture-once weights + per-dab apply. Capture carries a 32-byte
    // std140 Params block, apply a 16-byte one; weight_smooth takes no UBO (it is
    // sized to the affected set and gated on affected_count). The move buffers below
    // stay GL-owned (wrapped in views at dispatch). has_move() reports availability.
    gpu::ComputePipeline move_capture_pipeline;       // brute-force per-vertex world-distance gate: sets weight, init pos, appends affected
    gpu::ComputePipeline move_weight_smooth_pipeline; // ping-pong Laplacian over affected list
    gpu::ComputePipeline move_apply_pipeline;         // per-dab: pos = init + total*w.x + mirror_total*w.y (mirror summed in-place)
    gpu::Buffer          move_capture_ubo;            // 32-byte MoveCaptureParams block
    gpu::Buffer          move_apply_ubo;              // 16-byte {total} block

    // Move stroke buffers (allocated lazily, persist across strokes; resized as vertex_count grows)
    // Seam-owned gpu::Buffers (Step 2 cont): grow-only scratch; clears/copies/readback via .handle.
    gpu::Buffer move_affected_ssbo;       // [count, v0, v1, ...]
    gpu::Buffer move_weights_ssbo;        // vec2 per vertex: (primary, mirror) brush weights
    gpu::Buffer move_weights_pong_ssbo;   // vec2 per vertex (ping-pong scratch)
    gpu::Buffer move_init_ssbo;           // 3 floats per vertex (snapshotted at capture)
    uint32_t move_buffers_capacity;

    // Limb (snakehook) brush — ported onto the gpu:: seam (Seam Step 2b). Incremental
    // drag + tangential redistribution. Shares the move capture (weights/affected/init);
    // adds a relax pass. has_limb() reports readiness. The scratch SSBO stays GL-owned
    // (wrapped in a view at dispatch; the copy snapshot/copy-back stay raw GL).
    gpu::ComputePipeline limb_drag_pipeline;   // per-dab: pos += (delta*w.x + mirror_delta*w.y)*(1-mask)
    gpu::ComputePipeline limb_relax_pipeline;  // tangential (normal-stripped) Laplacian over the captured set
    gpu::Buffer          limb_drag_ubo;        // 16-byte {delta} block
    gpu::Buffer          limb_relax_ubo;       // 32-byte {vertex_count,lambda,tip_bias,tip_dir} block
    gpu::Buffer limb_pos_scratch_ssbo;    // 3 floats per vertex: read-side snapshot for relax ping-pong (seam-owned, Step 2 cont)
    uint32_t limb_scratch_capacity;

    // Compute normals — ported onto the gpu:: seam (Seam Step 2b). Shared per-stroke
    // recompute: one thread per dirty vertex, area-weighted 1-ring face normals. The
    // dirty-vert id list (dirty_verts_ssbo) stays GL-owned (uploaded raw GL, shared
    // with stroke_smooth); the count rides in a 16-byte Params UBO. has_normals() reports
    // readiness.
    gpu::ComputePipeline compute_normals_pipeline;
    gpu::Buffer          compute_normals_ubo;   // 16-byte {dirty_count} block

    // GPU-resident undo (Phase 2b): pen-up multires diff — ported onto the gpu::
    // seam (Seam Step 2b). Reprojects the world-space stroke delta (live VBO -
    // pen-down snapshot) into the active level's tangent frames and accumulates
    // onto the resident disp layer — the GPU twin of the CPU readback loop in
    // brush.cpp finalize(). Base-level strokes write the live position straight
    // into the base buffer; optionally captures (old,new) into the undo ring.
    // has_multires_diff() reports readiness.
    gpu::ComputePipeline multires_diff_pipeline;
    gpu::Buffer          multires_diff_ubo;   // 16-byte {count,writes_to_base,ring_base,ring_on} block

    // GPU-resident undo (Phase 2c): undo/redo apply — on the gpu:: seam. Scatters a
    // per-vert (target,source) disp pair from the stage buffer (or the undo ring in
    // ring mode) into the resident disp layer and reprojects the (target - source)
    // delta through the tangent frames into the working VBO — the GPU twin of the
    // CPU revert loop in undo.cpp apply() (same-level STROKE). Base-level strokes
    // write the absolute target straight into the base buffer and the VBO. Caller
    // follows with compute_normals. has_multires_apply() reports readiness.
    gpu::ComputePipeline multires_apply_pipeline;
    gpu::Buffer          multires_apply_ubo;  // 32-byte {count,targets_base,ring_mode,ring_base,forward} block
    gpu::Buffer multires_stage_ssbo;     // float6 * V scratch (target+source), grow-only — seam-owned (Step 3b)
    uint32_t multires_stage_capacity;

    // GPU-resident undo ring (blood-moon 3b-ii). PERSISTENT history of per-vert
    // (old,new) STROKE deltas — float6 per recorded vert — so pen-up can capture
    // undo data on the GPU without a CPU readback. Bump-allocated, grow-only
    // (copy-preserving resize) up to undo_ring_cap_bytes (set from
    // UndoStack::max_bytes / --toaster). Wrap + FIFO eviction land in 3b-iv when a
    // consumer exists to validate against; 3b-ii is append/read/reset + selftest,
    // no stroke-path consumer. Distinct from multires_stage_ssbo (transient scratch).
    gpu::Buffer undo_ring_ssbo;  // persistent float6-per-vert history buffer — seam-owned (Step 3b)
    size_t undo_ring_cap_bytes;  // hard ceiling (from UndoStack::max_bytes)
    size_t undo_ring_bytes;      // currently allocated buffer size
    size_t undo_ring_head;       // next free byte offset (bump allocator)

    // Adjacency CSR SSBOs (uploaded once at mesh init) — seam-owned gpu::Buffers (Step 2 cont)
    gpu::Buffer adjacency_offset_ssbo;
    gpu::Buffer adjacency_list_ssbo;
    uint32_t adjacency_vertex_count;

    // Mirror map SSBO (uploaded once at mesh init, maps vertex -> twin vertex)
    gpu::Buffer mirror_map_ssbo;
    uint32_t mirror_map_vertex_count;

    // Mask SSBO: alias (non-owning copy) of renderer.vbo_mask, so the apply shaders
    // can scale per-vertex displacement by (1 - mask[v]). The CPU mask brush keeps
    // mesh.mask authoritative and uploads via vbo_mask, so this GPU view stays in sync.
    // Refreshed by Scene::bind_active_ after every upload_mesh — a WebGPU grow makes a
    // new handle, so the old set-once alias is no longer valid (Step 3a). NOT owned
    // here (renderer owns it); never released through ComputeState.
    gpu::Buffer mask_ssbo;

    // Color SSBO: alias (non-owning copy) of renderer.vbo_color so the paint shader
    // writes packed RGBA8 directly into the display VBO. Same refresh contract as
    // mask_ssbo; not owned here.
    gpu::Buffer color_ssbo;

    // Dirty vertex list SSBO (uploaded per dispatch, used by compute_normals) — seam-owned (Step 2 cont)
    gpu::Buffer dirty_verts_ssbo;
    uint32_t dirty_verts_capacity;

    // Smooth compact dirty list SSBO: [count, v0, v1, ...] written by smooth_accum — seam-owned (Step 2 cont)
    gpu::Buffer smooth_dirty_ssbo;
    uint32_t smooth_dirty_capacity;  // max vertex IDs (excludes counter slot)

    // Remesh GPU passes — ported onto the gpu:: seam (Seam Step 2b). One-shot /
    // user-paced ops (not strokes), so CPU readback is allowed. The remesh-specific
    // SSBOs below stay GL-owned (wrapped in views at dispatch; uploads/copies/readback
    // stay raw GL). Each kernel's loose uniforms → a std140 Params UBO at binding 63.
    // has_remesh_smooth() reports the smooth kernel's availability (gates the CPU path).
    gpu::ComputePipeline remesh_select_stretched_pipeline;
    gpu::ComputePipeline remesh_select_unmasked_pipeline;
    gpu::ComputePipeline remesh_grow_selection_pipeline;
    gpu::ComputePipeline remesh_mirror_selection_pipeline;
    gpu::ComputePipeline remesh_find_pinned_pipeline;
    gpu::ComputePipeline remesh_smooth_weights_pipeline;
    gpu::ComputePipeline remesh_seam_snap_pipeline;
    gpu::ComputePipeline remesh_seam_weld_pipeline;
    gpu::ComputePipeline remesh_smooth_pipeline;
    gpu::Buffer remesh_select_stretched_ubo;   // 16-byte {target_len, tri_count}
    gpu::Buffer remesh_select_unmasked_ubo;    // 16-byte {tri_count, mask_size}
    gpu::Buffer remesh_grow_selection_ubo;     // 16-byte {tri_count}
    gpu::Buffer remesh_mirror_selection_ubo;   // 16-byte {tri_count, vertex_count}
    gpu::Buffer remesh_find_pinned_ubo;        // 16-byte {vertex_count, seam_tol}
    gpu::Buffer remesh_smooth_weights_ubo;     // 16-byte {vertex_count, support_rings}
    gpu::Buffer remesh_seam_snap_ubo;          // 16-byte {vertex_count, mask_size, seam_tol, snap_tol}
    gpu::Buffer remesh_seam_weld_ubo;          // 16-byte {vertex_count, mask_size, weld_tol}
    gpu::Buffer remesh_smooth_ubo;             // 16-byte {lambda, seam_tol, vertex_count}

    gpu::Buffer remesh_core_sel_ssbo;    // uint[T] — core (pre-grow) tri selection
    gpu::Buffer remesh_trisel_pong_ssbo; // uint[T] — input snapshot for grow/mirror
    gpu::Buffer seam_weld_map_ssbo;      // uint[V] — merge target per vert (v or lower match)

    // Remesh seam-owned scratch SSBOs (uploads/copies/readback stay raw GL via .handle):
    gpu::Buffer remesh_ping_ssbo;       // float[V*3] SOA — positions input
    gpu::Buffer remesh_pong_ssbo;       // float[V*3] SOA — positions output
    gpu::Buffer remesh_norm_ssbo;       // float[V*3] SOA — normals
    gpu::Buffer remesh_weights_ssbo;    // float[V]   — smooth weights (1.0 if non-mask)
    gpu::Buffer remesh_pinned_ssbo;     // uint[V]    — 0/1 pinned flags
    gpu::Buffer remesh_trisel_ssbo;     // uint[T]    — 0/1 tri selected flags
    gpu::Buffer remesh_indices_ssbo;    // uint[T*3]  — triangle indices snapshot
    // remesh_smooth is the one kernel that needs all of adjacency + its own scratch
    // at once (9 storage buffers — one over the 8/stage web baseline). This is a
    // CSR-concatenated copy of adjacency_offset_ssbo+adjacency_list_ssbo (offsets
    // first, list appended at element vertex_count+1) rebuilt each dispatch, so the
    // two shared adjacency buffers collapse into a single binding just for that
    // kernel — see dispatch_remesh_smooth.
    gpu::Buffer remesh_adj_csr_ssbo;
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
    void snapshot_stroke_normals(const gpu::Buffer& norm_vbo, uint32_t vertex_count);

    // Dispatch the draw accumulate shader once over all vertices.
    // Reads positions from pos_vbo and normals from stroke_norm_ssbo (must be
    // populated by snapshot_stroke_normals before calling). Caller must clear
    // accum buffer beforehand. Issues memory barrier.
    void dispatch_draw_accum(const DrawAccumParams& params, const gpu::Buffer& pos_vbo);

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
    void dispatch_draw_apply(const gpu::Buffer& pos_vbo, uint32_t vertex_count,
                              const gpu::Buffer& accum_override = {});

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
    void dispatch_draw_mirror_apply(const gpu::Buffer& pos_vbo, uint32_t vertex_count);

    // Compile and dispatch the smooth brush compute shaders.
    bool init_smooth();
    void dispatch_smooth(const SmoothAccumParams& params,
                         const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo);

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
    void dispatch_smooth_mirror_apply(const gpu::Buffer& pos_vbo, uint32_t vertex_count, float anchor_x);

    // Compile the stroke autosmooth shader. Called once at init.
    bool init_stroke_smooth();

    // Dispatch a single Laplacian smoothing pass over the given vertex IDs.
    // Reads positions in-place from pos_vbo; mask gates per-vertex.
    void dispatch_stroke_smooth_apply(const uint32_t* vert_ids, uint32_t count,
                                      float strength,
                                      const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo);

    // Compile the crease brush accumulate compute shader. Called once at init.
    bool init_crease();

    // Dispatch the crease accumulate shader over all vertices.
    // Reads frozen normals from stroke_norm_ssbo (must be populated by snapshot_stroke_normals).
    void dispatch_crease_accum(const CreaseAccumParams& params, const gpu::Buffer& pos_vbo);

    // Compile the pinch brush accumulate compute shader. Called once at init.
    bool init_pinch();

    // Dispatch the pinch accumulate shader over all vertices.
    // Reads frozen normals from stroke_norm_ssbo (must be populated by snapshot_stroke_normals).
    void dispatch_pinch_accum(const PinchAccumParams& params, const gpu::Buffer& pos_vbo);

    // Compile the mask brush compute shader. Called once at init.
    bool init_mask();
    bool init_color();

    // Is the mask kernel compiled and ready? (replaces the old `mask_paint_program`
    // truthiness check now that the program lives behind the gpu:: seam.)
    bool has_mask() const { return mask_pipeline.handle != 0; }

    // Draw-brush readiness (replaces draw_*_program truthiness checks). has_draw()
    // gates the whole draw path; has_draw_symmetrize() gates the mirror-symmetry pass.
    bool has_draw() const { return draw_accum_pipeline.handle != 0; }
    bool has_draw_symmetrize() const { return draw_symmetrize_pipeline.handle != 0; }

    // Crease / pinch readiness (replaces crease_accum_program / pinch_accum_program checks).
    bool has_crease() const { return crease_accum_pipeline.handle != 0; }
    bool has_pinch() const { return pinch_accum_pipeline.handle != 0; }

    // Move/grab readiness (replaces move_*_program truthiness checks). Gates the
    // whole grab path; limb reuses the capture/weight-smooth pieces.
    bool has_move() const { return move_capture_pipeline.handle != 0; }

    // Limb (snakehook) readiness (replaces limb_*_program truthiness checks). Needs
    // both its own kernels plus the move capture pieces it rides on (has_move()).
    bool has_limb() const { return limb_drag_pipeline.handle != 0 && limb_relax_pipeline.handle != 0; }

    // Smooth readiness (replaces smooth_accum_program truthiness checks).
    bool has_smooth() const { return smooth_accum_pipeline.handle != 0; }

    // Paint readiness (replaces color_paint_program / color_smooth_program checks).
    bool has_color() const { return color_paint_pipeline.handle != 0; }
    bool has_color_smooth() const { return color_smooth_pipeline.handle != 0; }

    // Compute-normals readiness (replaces compute_normals_program truthiness checks).
    bool has_normals() const { return compute_normals_pipeline.handle != 0; }

    // Remesh tangential-smooth readiness (replaces remesh_smooth_program checks).
    bool has_remesh_smooth() const { return remesh_smooth_pipeline.handle != 0; }

    // Stroke-autosmooth readiness (replaces stroke_smooth_apply_program checks).
    bool has_stroke_smooth() const { return stroke_smooth_apply_pipeline.handle != 0; }

    // Dispatch the mask paint shader: per-vertex distance check, writes mask VBO
    // directly. Uses smooth_dirty_ssbo for the compact dirty list. Caller reads
    // back dirty list via readback_smooth_dirty.
    void dispatch_mask_paint(const MaskPaintParams& params, const gpu::Buffer& pos_vbo);
    void dispatch_color_paint(const ColorPaintParams& params, const gpu::Buffer& pos_vbo);
    // Paint-smooth: reuses ColorPaintParams (paint_strength = blend amount,
    // paint_r/g/b ignored). Averages neighbour colours via CSR adjacency.
    void dispatch_color_smooth(const ColorPaintParams& params, const gpu::Buffer& pos_vbo, const gpu::Buffer& index_ebo);

    // Compile the three move-brush compute pipelines (capture / weight-smooth /
    // apply) + their Params UBOs, through the gpu:: seam. Called once at init.
    bool init_move();

    // Allocate/resize move stroke buffers (weights, init pos, affected list).
    void ensure_move_buffers(uint32_t vertex_count);

    // Capture pass: brute-force over vertex_count threads. Gates by world-distance
    // to anchor (and mirror x-sign if mirror_x). Writes weight, init position,
    // appends to affected list. Clears affected count + weights up front.
    void dispatch_move_capture(const MoveCaptureParams& params, const gpu::Buffer& pos_vbo);

    // Smooth weights: ping-pong Laplacian iterations over the affected set.
    void dispatch_move_weight_smooth(uint32_t vertex_count, int iterations, const gpu::Buffer& index_ebo);

    // Apply pass: dispatched over affected list. positions[v] = init[v] + total*w.
    // Sums primary + mirrored brush contributions per vertex (mirror handled in-shader).
    void dispatch_move_apply(const MoveApplyParams& params, const gpu::Buffer& pos_vbo);

    // Readback affected list. Returns count; fills out[].
    uint32_t readback_move_affected(std::vector<uint32_t>& out);

    // Limb (snakehook) brush. Reuses the move capture/weights/affected/mask.
    bool init_limb();
    void ensure_limb_buffers(uint32_t vertex_count);
    // Incremental drag: positions[v] += this-dab world delta * w (mirror summed).
    void dispatch_limb_drag(const LimbDragParams& params, const gpu::Buffer& pos_vbo);
    // Tangential redistribution: N Laplacian iterations over the captured set,
    // normal component stripped so it evens spacing without deflating the form.
    void dispatch_limb_relax(uint32_t vertex_count, int iterations, float lambda,
                             float tip_dx, float tip_dy, float tip_dz, float tip_bias,
                             const gpu::Buffer& pos_vbo, const gpu::Buffer& norm_vbo, const gpu::Buffer& index_ebo);

    // Compile the compute normals shader. Called once at init.
    bool init_compute_normals();

    // Compile the pen-up multires diff shader. Called once at init.
    bool init_multires_diff();
    bool has_multires_diff() const { return multires_diff_pipeline.handle != 0; }

    // Dispatch the multires diff over the stroke's touched verts (`verts`/`count`).
    // disp/frames/snap_pos/base are the MultiresGPU SSBOs for the active level;
    // pos_vbo is the live working VBO at pen-up. When writes_to_base, the live
    // position is written to base_ssbo; otherwise the world delta is reprojected
    // into the frames and accumulated onto disp_ssbo (which holds the pen-down
    // disp). Issues a buffer-update barrier so a debug readback sees the result.
    // `ring_ssbo`/`ring_base_floats`: when ring_ssbo != 0, the shader also writes
    // (old,new) deltas for each touched vert into the undo ring at ring_base_floats
    // + di*6 (3b-iii capture). Pass ring_ssbo=0 to skip (compute/ring unavailable).
    void dispatch_multires_diff(const gpu::Buffer& pos_vbo, const gpu::Buffer& disp_ssbo,
                                const gpu::Buffer& frames_ssbo, const gpu::Buffer& snap_pos_ssbo,
                                const gpu::Buffer& base_ssbo,
                                const uint32_t* verts, uint32_t count,
                                bool writes_to_base,
                                bool ring_ssbo, uint32_t ring_base_floats);

    // Compile the undo/redo multires apply shader. Called once at init.
    bool init_multires_apply();
    bool has_multires_apply() const { return multires_apply_pipeline.handle != 0; }

    // Dispatch the multires apply over `verts`/`count`. `stage` is count*6 floats
    // (target xyz, source xyz per vert). For a disp-level revert it scatters target
    // into disp_ssbo and adds frame*(target-source) to pos_vbo; for a base revert
    // it writes the absolute target into base_ssbo and pos_vbo. Caller must run
    // compute_normals afterward. Issues a buffer-update barrier for a debug readback.
    // `ring_ssbo`/`ring_base_floats`/`forward` (3b-iv): when ring_ssbo != 0 the (old,
    // new) pair is read from the persistent undo ring at ring_base_floats + di*6
    // instead of from `stage` (which may be null), with target=new on redo (forward),
    // target=old on undo. Pass ring_ssbo=0 to use the CPU stage.
    void dispatch_multires_apply(const gpu::Buffer& pos_vbo, const gpu::Buffer& disp_ssbo,
                                 const gpu::Buffer& frames_ssbo, const gpu::Buffer& base_ssbo,
                                 const uint32_t* verts,
                                 const float* stage, uint32_t count, bool targets_base,
                                 bool ring_ssbo = false, uint32_t ring_base_floats = 0,
                                 bool forward = false);

    // ---- GPU-resident undo ring (blood-moon 3b-ii) ----------------------------
    // Set the ring's hard ceiling. Call once at startup from UndoStack::ring_max_bytes.
    void undo_ring_set_budget(size_t cap_bytes);
    // Drop all history (keep the buffer). Call on project load / undo clear.
    void undo_ring_reset();
    // Append float_count floats to the ring (reserve + upload). Returns the BYTE
    // offset of the appended span, or SIZE_MAX if the span is larger than the whole
    // ring. Wraps circularly at the cap (3b-iv part 2).
    size_t undo_ring_append(const float* data, size_t float_count);
    // Reserve float_count floats WITHOUT uploading (a compute shader fills them).
    // Grows copy-preserving toward the cap, then wraps circularly. Returns the
    // FLOAT offset of the reserved span, or SIZE_MAX if it can't fit even an empty
    // ring. Used by the pen-up diff shader (3b-iii) to write (old,new) deltas; the
    // consumer must invalidate overlapped entries (UndoStack::ring_evict_overlap).
    size_t undo_ring_reserve(size_t float_count);
    // Read float_count floats back from byte_offset into out (spill / validation).
    void undo_ring_read(size_t byte_offset, size_t float_count, float* out);
    // CHISEL_DEBUG_MULTIRES round-trip self-test (append → read → compare). No-op
    // in release. Prints [undo-ring][debug].
    void undo_ring_selftest();

    // Upload adjacency CSR to GPU. Call at mesh init and after remesh.
    void upload_adjacency(const uint32_t* offsets, uint32_t offset_count,
                          const uint32_t* list, uint32_t list_count);

    // Dispatch compute normals for a set of affected vertices.
    // Reads positions from pos_vbo, writes normals to norm_vbo.
    void dispatch_compute_normals(const uint32_t* dirty_verts, uint32_t dirty_count,
                                  const gpu::Buffer& pos_vbo, const gpu::Buffer& norm_vbo, const gpu::Buffer& index_ebo);

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
