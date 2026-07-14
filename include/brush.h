#pragma once
#include "mesh.h"
#include "renderer.h"
#include "camera.h"
#include "input.h"
#include "undo.h"
#include "multires_stack.h"
#include "compute.h"
#include <vector>
#include <cmath>
#include <algorithm>

struct MeshEntity;
struct Scene;
struct MultiresGPU;

enum class StrokePhase { NONE, BEGIN, ACTIVE, END };

struct BrushRegion {
    int x, y, w, h;
    bool valid() const { return w > 0 && h > 0; }
};

struct DabContext {
    Renderer&         renderer;
    const Camera&     cam;
    ComputeState&     compute;
    Mesh&             mesh;
    MultiresStack&    multires;
    const InputState& input;
    int               win_w, win_h;
    uint32_t          vertex_count;  // active entity vertex count (working buffer at offset 0)
    float             eff_brush_size; // input.brush_size * pen-pressure size multiplier (px)
};

BrushRegion compute_brush_region(float dab_x, float dab_y,
                                  float brush_size, float screen_slack,
                                  int win_w, int win_h);

int populate_disp_from_readback(const float* readback_buf,
                                 uint32_t vertex_count,
                                 std::vector<float>& disp_x,
                                 std::vector<float>& disp_y,
                                 std::vector<float>& disp_z,
                                 std::vector<float>& disp_weight);

struct MoveState {
    std::vector<uint32_t> affected_list;
    std::vector<float>    weights;
    std::vector<bool>     affected_flag;
    std::vector<float>    init_x, init_y, init_z;
    float total_dx, total_dy, total_dz;
    bool  captured;

    // In-flight async readback of the GPU capture's affected list. The apply
    // dispatches don't need the CPU list (weights live GPU-side), so dabs keep
    // landing while this is pending; the snap/normals bookkeeping starts once it
    // lands (a frame or two).
    gpu::ReadTicket capture_tk;
    uint32_t        capture_words;
    bool            capture_booked;   // first landed-dab bookkeeping done

    MoveState();
    void reset(uint32_t vert_count);
    void clear();
};

struct MaskState {
    std::vector<float>    delta;
    std::vector<float>    snap;
    std::vector<bool>     snap_flag;
    std::vector<uint32_t> snap_list;

    void reset();
    void clear();
};

// Paint snapshot: packed RGBA8 pre-stroke color per first-touched vertex, so the
// pen-up readback knows which verts to pull and undo can diff old vs new.
struct ColorState {
    std::vector<uint32_t> snap;        // pre-stroke packed color, indexed by vert
    std::vector<bool>     snap_flag;
    std::vector<uint32_t> snap_list;   // first-touch order

    void reset();
    void clear();
};

struct BrushStroke {
    // Screen-space brush region (bounding box around brush circle)
    int region_x, region_y, region_w, region_h;

    // RETAINED: Mask brush CPU cache (delete when Mask gets GPU shader)
    std::vector<uint32_t> cached_triid;  // screen_w * screen_h
    std::vector<float> cached_bary;      // screen_w * screen_h * 2
    int cached_w, cached_h;

    int screen_w, screen_h;  // screen dimensions, set at begin()

    // Per-frame anchor: 3D surface point at the cursor pixel, set at the start of each
    // apply_* call. Used by walk_brush_region to gate pixels by world-space proximity,
    // so the brush footprint is a surface patch near the cursor rather than a
    // screen-space wedge through the mesh (prevents ear-tip stroke hitting the skull).
    Vec3 anchor_pos;
    float anchor_world_radius;
    bool anchor_valid;
    float cyl_axis_x, cyl_axis_y, cyl_axis_z;
    float screen_slack;  // screen bbox expansion for grazing angles (1.0 facing, up to 3.0 at silhouette)

    // Accumulated displacement per vertex (sparse: only affected verts)
    std::vector<float> disp_x, disp_y, disp_z;
    std::vector<float> disp_weight;  // total weight accumulated per vertex

    // Persistent scratch buffers (avoid per-frame allocation)
    std::vector<uint32_t> dirty_verts;
    std::vector<uint32_t> gpu_dirty;

    // In-flight per-dab dirty-list readbacks. Each dab kicks one ticket right after
    // its dispatches; drain_dab_readbacks (from post_frame, and finalize before the
    // pen-up commit) consumes them in FIFO order a frame or two later and runs the
    // snap/normals bookkeeping that used to happen synchronously per dab. The CPU
    // mesh stays pen-down-stale for the whole stroke, so the delayed snapshot
    // captures the same pre-stroke values it always did.
    enum : uint8_t { DAB_GEO = 0, DAB_MASK = 1, DAB_COLOR = 2, DAB_DENSITY = 3 };
    struct PendingDab {
        gpu::ReadTicket tk = 0;
        uint32_t words = 0;
        float anchor_x = 0.0f;   // this dab's anchor (mirror-side snap filter)
        uint8_t kind = DAB_GEO;
    };
    std::vector<PendingDab> pending_dabs;
    void kick_dab_readback(DabContext& ctx, uint8_t kind);
    void drain_dab_readbacks(DabContext& ctx);
    bool dab_readbacks_pending() const { return !pending_dabs.empty() || move.capture_tk != 0; }


    // Last successfully anchored brush position — persists across strokes for F-focus
    Vec3 last_stroke_pos;
    bool last_stroke_valid;

    // Dab spacing: only apply when cursor has moved far enough since last dab
    float last_dab_x, last_dab_y;

    // Cursor history for Catmull-Rom spline interpolation between frames
    static constexpr int CURSOR_HIST_SIZE = 4;
    float cursor_hist_x[4], cursor_hist_y[4];
    int cursor_hist_count;

    // Move and Mask sub-structs
    MoveState move;
    MaskState mask;
    ColorState color;
    MaskState density;   // density-field stroke state — same scalar shape as mask

    // RETAINED: Adjacency flood for Move capture + Mask (delete when both get GPU shaders)

    // Undo snapshot: records each vertex's pre-stroke position on its first touch
    std::vector<bool> snap_flag;
    std::vector<uint32_t> snap_list;
    std::vector<float> snap_x, snap_y, snap_z;

    // Stroke-level metadata: captured at begin(), fixed for the entire stroke
    int  stroke_level;
    int  stroke_disp_index;     // -1 if writing to base, else level - base_level - 1
    bool stroke_writes_to_base;
    int  stroke_sign;           // +1 or -1: locked from first displacement, prevents sign flips at silhouette edges
    size_t stroke_ring_base = SIZE_MAX;  // float offset of this stroke's undo-ring span (3b-iv); SIZE_MAX if not captured

    // Compute shader state (nullptr = CPU-only fallback)
    ComputeState* compute;

    // Set true when GPU normals path was used (CPU mesh.norm deferred to pen-up)
    bool gpu_normals_deferred;

    // Set true when GPU smooth wrote positions to VBO but mesh.pos_* / multires.base
    // / cpu_pos have NOT been synced yet.  Readback deferred to finalize (pen-up).
    bool gpu_positions_deferred;

    // Set true when GPU mask shader wrote to vbo_mask directly. mesh.mask is stale
    // until finalize reads back the changed values for undo commit.
    bool gpu_mask_deferred;

    // Set true when GPU paint shader wrote to vbo_color directly. mesh.color is
    // stale until finalize reads back the changed values (display VAO + undo).
    bool gpu_color_deferred;

    // Set true when the GPU density brush wrote to vbo_density directly. mesh.density
    // is stale until finalize reads back the changed values for undo commit.
    bool gpu_density_deferred;

    StrokePhase phase = StrokePhase::NONE;
    bool needs_mesh_update = false;
    uint32_t vertex_count;
    uint32_t stroke_entity_id = 0;       // entity active at pen-down; stamped onto undo entries

    bool is_active() const { return phase == StrokePhase::BEGIN || phase == StrokePhase::ACTIVE; }

    BrushStroke();

    // Begin a new stroke: capture screen buffers and stroke-level metadata.
    // The active entity lives in the renderer's working buffers at offset 0, so
    // the brush dispatches over [0, vert_count). entity_id is the active entity
    // at pen-down; stamped onto undo entries. screen_buffers_fresh = the idle path
    // already rendered the screen buffers for this camera/mesh (the usual case) —
    // begin() then reuses them and the renderer's landed plane cache instead of
    // re-rendering, which would re-kick the async plane reads and cost the first
    // dab(s) on webgpu.
    void begin(Renderer& renderer, const Camera& cam,
               float screen_x, float screen_y, float brush_radius,
               int screen_w, int screen_h, uint32_t vert_count,
               const MultiresStack& multires, BrushType brush_type,
               uint32_t entity_id, MultiresGPU& mgpu,
               bool screen_buffers_fresh);

    // GPU dispatch methods (absorb code from main.cpp)
    void apply_smooth(DabContext& ctx, float dab_x, float dab_y,
                      float strength, float hardness);

    void apply_crease(DabContext& ctx, float dab_x, float dab_y,
                      float strength, float hardness, bool subtract);

    void apply_pinch(DabContext& ctx, float dab_x, float dab_y,
                     float strength, float hardness, bool subtract = false);

    // inflate=false: push the dab along the cursor's surface normal (Draw).
    // inflate=true:  push each vert along its own normal (Inflate) — reuses the
    // entire draw accum/symmetrize/apply/mirror/mask/undo pipeline.
    void apply_draw(DabContext& ctx, float dab_x, float dab_y,
                    float strength, float hardness, bool subtract,
                    bool inflate = false);

    void apply_move_gpu(DabContext& ctx, float cursor_dx, float cursor_dy,
                        float strength, float hardness);

    // Limb (snakehook): sticky-capture grab (move) + incremental drag + per-dab
    // tangential redistribution. hardness controls capture falloff; strength
    // scales the drag (and the relax amount rides along).
    void apply_limb_gpu(DabContext& ctx, float cursor_dx, float cursor_dy,
                        float strength, float hardness);

    // Post-dispatch methods
    void post_dab(DabContext& ctx);
    void post_frame(DabContext& ctx);

    // Mask brush: paints/erases mask with accumulation blending
    void apply_mask(Renderer& renderer, const Camera& cam, const Mesh& mesh,
                    float screen_x, float screen_y,
                    float brush_radius, float strength, float hardness,
                    bool invert, bool mirror_x,
                    int screen_w, int screen_h);

    // GPU mask brush: compute shader writes directly to mask VBO.
    void apply_mask_gpu(DabContext& ctx, float dab_x, float dab_y,
                        float strength, float hardness, bool invert);

    // GPU mask-smooth: the smooth gesture (shift / double-shift) while the mask
    // brush is active — blends mask values toward their neighbour average
    // instead of smoothing geometry. Shares the mask undo-snap path.
    void apply_mask_smooth_gpu(DabContext& ctx, float dab_x, float dab_y,
                               float strength, float hardness);

    // GPU paint brush: compute shader lerps vertex albedo in the color VBO.
    // erase = paint toward white instead of the brush color.
    void apply_color_gpu(DabContext& ctx, float dab_x, float dab_y,
                         float strength, float hardness, bool erase);

    // GPU paint-smooth: the smooth gesture (shift / double-shift) while a paint
    // brush is active — blends vertex colours toward their neighbour average
    // instead of smoothing geometry. Shares the paint undo-snap path.
    void apply_density_gpu(DabContext& ctx, float dab_x, float dab_y,
                           float strength, float hardness, bool invert);

    void apply_density_smooth_gpu(DabContext& ctx, float dab_x, float dab_y,
                                  float strength, float hardness);

    void apply_color_smooth_gpu(DabContext& ctx, float dab_x, float dab_y,
                                float strength, float hardness);

    // Pen-up reconcile — tick once per frame from main once the stroke ends, until
    // it returns true (the commit completed this call; had_update = geometry was
    // modified, caller should update_screen_positions). Stage 1 drains in-flight
    // dab readbacks, runs autosmooth + the GPU undo-ring diff, and kicks the pen-up
    // buffer reads; stage 2 lands them, syncs mesh/multires, and commits the undo
    // entry. Multi-frame only when async readbacks are in flight (webgpu); on GL it
    // completes on the first call, like the old synchronous finalize.
    bool finalize(DabContext& ctx, Mesh& mesh, UndoStack& stack,
                  MultiresStack& multires, MultiresGPU& mgpu, Renderer& renderer,
                  BrushType brush_type, bool autosmooth, bool& had_update);

    // Pen-up reconcile state (see finalize).
    enum class FinState : uint8_t { IDLE, DRAIN, READS };
    FinState fin_state = FinState::IDLE;
    bool reconciling() const { return fin_state != FinState::IDLE; }
    bool fin_ring_captured = false;
    BrushType fin_brush_type = BrushType::DRAW;   // captured at the first tick —
    bool      fin_autosmooth = false;             // input can change mid-reconcile
    gpu::ReadTicket fin_pos_tk = 0, fin_mask_tk = 0, fin_color_tk = 0, fin_norm_tk = 0,
                    fin_density_tk = 0;
    std::vector<float>    fin_pos_buf, fin_mask_buf, fin_norm_buf, fin_density_buf;  // persistent
    std::vector<uint32_t> fin_color_buf;
    std::vector<float>    fin_gpu_diff_chk;   // CHISEL_DEBUG_MULTIRES stage-1 snapshot

    // Clear stroke buffers (called by finalize)
    void end();

    // Commit this stroke's per-vertex deltas as an undo entry, then clear snapshots.
    // Call before end(). No-op if nothing was touched.
    void commit_undo(const Mesh& mesh, UndoStack& stack, const MultiresStack& multires);

    // Commit mask stroke as an undo entry. Call before end().
    void commit_mask_undo(const Mesh& mesh, UndoStack& stack);

    // Commit paint stroke as an undo entry. Call before end().
    void commit_color_undo(const Mesh& mesh, UndoStack& stack);
    void commit_density_undo(const Mesh& mesh, UndoStack& stack);

    // Apply accumulated mask changes to the mesh, return list of dirty vertices
    void apply_mask_changes(Mesh& mesh, std::vector<uint32_t>& dirty_verts_out);

    // RETAINED: only used by apply_mask and apply_move capture pass.
    // TODO: GPU shaders for mask + move capture, then delete.
    // Shared brush region walker. Invokes callback(i0, i1, i2, bu, bv, bw,
    // dist, eff_radius, stride, pixel_idx) for each screen pixel inside the brush
    // that hits a valid triangle. When anchor_valid, dist and eff_radius are
    // world-space (surface footprint is round on the mesh). When anchor is
    // unavailable, falls back to screen-space dist/radius (cursor off-mesh).
    // Also updates region_x/y/w/h as a side effect.
    template<typename Fn>
    void walk_brush_region(const Mesh& mesh, float cx, float cy, float radius,
                           int screen_w, int screen_h, Fn&& callback) {
        float r_screen = anchor_valid ? radius * screen_slack : radius;
        int r = (int)std::ceil(r_screen);
        region_x = std::max(0, (int)cx - r);
        region_y = std::max(0, (int)cy - r);
        int x2 = std::min(screen_w, (int)cx + r + 1);
        int y2 = std::min(screen_h, (int)cy + r + 1);
        region_w = x2 - region_x;
        region_h = y2 - region_y;
        if (region_w <= 0 || region_h <= 0) return;

        int stride = (debug_stride_override >= 1) ? debug_stride_override : 1;

        int cached_base_row = cached_h - region_y - region_h;
        uint32_t tc  = mesh.tri_count();

        for (int py = 0; py < region_h; py += stride) {
            int cached_row = cached_base_row + py;
            for (int px = 0; px < region_w; px += stride) {
                int idx = cached_row * cached_w + (region_x + px);
                uint32_t tri_id = cached_triid[idx];
                // Active entity's tris start at 0 in the working buffer / pick FBO.
                if (tri_id == 0xFFFFFFFF || tri_id >= tc) continue;
                uint32_t local_tid = tri_id;

                float bu = std::max(0.0f, std::min(1.0f, cached_bary[idx*2+0]));
                float bv = std::max(0.0f, std::min(1.0f, cached_bary[idx*2+1]));
                float bw = std::max(0.0f, std::min(1.0f, 1.0f - bu - bv));

                uint32_t i0 = mesh.indices[local_tid*3+0];
                uint32_t i1 = mesh.indices[local_tid*3+1];
                uint32_t i2 = mesh.indices[local_tid*3+2];

                float wx = mesh.pos_x[i0]*bu + mesh.pos_x[i1]*bv + mesh.pos_x[i2]*bw;
                float wy = mesh.pos_y[i0]*bu + mesh.pos_y[i1]*bv + mesh.pos_y[i2]*bw;
                float wz = mesh.pos_z[i0]*bu + mesh.pos_z[i1]*bv + mesh.pos_z[i2]*bw;

                float world_dist;
                float effective_radius;

                if (anchor_valid) {
                    float dwx = wx - anchor_pos.x;
                    float dwy = wy - anchor_pos.y;
                    float dwz = wz - anchor_pos.z;
                    world_dist = std::sqrt(dwx*dwx + dwy*dwy + dwz*dwz);
                    if (world_dist >= anchor_world_radius) continue;
                    effective_radius = anchor_world_radius;
                } else {
                    float pxf = (float)(region_x + px);
                    float pyf = (float)(region_y + py);
                    float ddx = pxf - cx;
                    float ddy = pyf - cy;
                    float screen_dist = std::sqrt(ddx*ddx + ddy*ddy);
                    if (screen_dist >= radius) continue;
                    world_dist = screen_dist;
                    effective_radius = radius;
                }

                callback(i0, i1, i2, bu, bv, bw, world_dist, effective_radius, stride, idx);
            }
        }
    }

    // DEBUG: if >= 1, overrides adaptive stride selection in walk_brush_region.
    // Set to 0 to disable override and use adaptive logic. Cycled by F9.
    static int debug_stride_override;

    // DEBUG: if >= 0, apply_draw logs the normalized deposit for this vertex
    // after each accumulation pass. Set by F10 (picks vertex under cursor). -1 = disabled.
    static int debug_test_vertex;

    void set_anchor(const Mesh& mesh, const Camera& cam,
                    float cursor_x, float cursor_y, float brush_radius,
                    int screen_h, Renderer& renderer);

    // Build this dab's brush-alpha stamp frame (screen-aligned tangent/bitangent in
    // the surface tangent plane) and push it to the compute state's shared alpha UBO.
    // A no-op-equivalent (enabled=false) is written when no alpha is selected or the
    // calling brush doesn't support alphas (`allow` = false; only draw/mask/paint do),
    // so the round brush is preserved. Call once per dab, after set_anchor().
    void set_alpha_dab(DabContext& ctx, bool allow);

private:
    float falloff(float dist, float radius, float hardness) const;

};
