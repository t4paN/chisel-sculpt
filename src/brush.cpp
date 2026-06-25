#include "brush.h"
#include "mesh_entity.h"
#include "scene.h"
#include <glad/glad.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cstdint>

int BrushStroke::debug_stride_override = 0;
int BrushStroke::debug_test_vertex     = -1;

// --- Banded pen-up readback ---
// snap_list / mask.snap_list / color.snap_list are deduped but UNSORTED and
// sparse in index space: mirror twins land in a far index cluster, and remesh
// scatters spatially-adjacent verts across wide index spans. A single
// glGetBufferSubData over [min,max] therefore drags the gaps across the bus for
// a handful of useful verts (worst case: the whole VBO for one mirrored stroke).
// Coalesce the touched indices into contiguous runs and read one run at a time.
// Verts inside a bridged gap are untouched this stroke, so VBO == mesh for them
// and re-applying their value is a no-op (base: same value; disp: delta 0;
// mask/color/normals: same value) — hence we can iterate runs, not the list.
struct ReadRun { uint32_t first, count; };

// Merge runs whose index gap is <= this. Reading ~GAP filler verts (a few KB)
// is far cheaper than an extra blocking GL round-trip, so bridge small holes.
static constexpr uint32_t READBACK_RUN_GAP = 256;

static void coalesce_snap_runs(const std::vector<uint32_t>& verts,
                               std::vector<ReadRun>& runs_out) {
    runs_out.clear();
    if (verts.empty()) return;
    static std::vector<uint32_t> sorted;
    sorted.assign(verts.begin(), verts.end());
    std::sort(sorted.begin(), sorted.end());
    uint32_t first = sorted[0], last = sorted[0];
    for (size_t i = 1; i < sorted.size(); i++) {
        uint32_t v = sorted[i];
        if (v == last) continue;                 // already deduped; cheap guard
        if (v - last > READBACK_RUN_GAP) {
            runs_out.push_back({first, last - first + 1});
            first = v;
        }
        last = v;
    }
    runs_out.push_back({first, last - first + 1});
}

// --- Shared helpers ---

BrushRegion compute_brush_region(float dab_x, float dab_y,
                                  float brush_size, float screen_slack,
                                  int win_w, int win_h) {
    float r_screen = brush_size * screen_slack;
    int r = (int)std::ceil(r_screen);
    int x = std::max(0, (int)dab_x - r);
    int y = std::max(0, (int)dab_y - r);
    int x2 = std::min(win_w, (int)dab_x + r + 1);
    int y2 = std::min(win_h, (int)dab_y + r + 1);
    return { x, y, x2 - x, y2 - y };
}

// --- Sub-struct methods ---

MoveState::MoveState()
    : total_dx(0.0f), total_dy(0.0f), total_dz(0.0f)
    , captured(false)
{}

void MoveState::reset(uint32_t vert_count) {
    affected_flag.assign(vert_count, false);
    weights.assign(vert_count, 0.0f);
    affected_list.clear();
    init_x.clear();
    init_y.clear();
    init_z.clear();
    total_dx = total_dy = total_dz = 0.0f;
    captured = false;
}

void MoveState::clear() {
    affected_list.clear();
    weights.clear();
    affected_flag.clear();
    captured = false;
}

void MaskState::reset() {
    delta.clear();
    snap.clear();
    snap_flag.clear();
    snap_list.clear();
}

void MaskState::clear() {
    delta.clear();
    snap.clear();
    snap_flag.clear();
    snap_list.clear();
}

void ColorState::reset() {
    snap.clear();
    snap_flag.clear();
    snap_list.clear();
}
void ColorState::clear() {
    snap.clear();
    snap_flag.clear();
    snap_list.clear();
}

template<typename P>
static void set_area_normal(P& p, float ax, float ay, float az) {
    p.anchor_normal_a_x =  ax;  p.anchor_normal_a_y = ay;  p.anchor_normal_a_z = az;
    p.anchor_normal_b_x = -ax;  p.anchor_normal_b_y = ay;  p.anchor_normal_b_z = az;
}

// --- BrushStroke ---

static void snap_and_mirror_dirty(BrushStroke& bs, DabContext& ctx) {
    uint32_t vc = ctx.mesh.vertex_count();
    for (uint32_t v : bs.dirty_verts) {
        if (v >= vc) {
            std::printf("[CRASH-GUARD] dirty vert %u >= vc %u\n", v, vc);
            continue;
        }
        if (!bs.snap_flag[v]) {
            bs.snap_flag[v] = true;
            if (bs.stroke_writes_to_base) {
                bs.snap_x[v] = ctx.mesh.pos_x[v];
                bs.snap_y[v] = ctx.mesh.pos_y[v];
                bs.snap_z[v] = ctx.mesh.pos_z[v];
            } else {
                bs.snap_x[v] = ctx.multires.disp[bs.stroke_disp_index][v].x;
                bs.snap_y[v] = ctx.multires.disp[bs.stroke_disp_index][v].y;
                bs.snap_z[v] = ctx.multires.disp[bs.stroke_disp_index][v].z;
            }
            bs.snap_list.push_back(v);
        }
    }
    if (ctx.input.mirror_x && !ctx.mesh.mirror_x_map.empty()) {
        size_t n = bs.dirty_verts.size();
        for (size_t i = 0; i < n; i++) {
            uint32_t v = bs.dirty_verts[i];
            if (v >= vc) continue;
            uint32_t mv = ctx.mesh.mirror_x_map[v];
            if (mv >= vc) {
                std::printf("[CRASH-GUARD] mirror %u -> %u >= vc %u\n", v, mv, vc);
                continue;
            }
            if (mv == v) continue;
            if (ctx.mesh.pos_x[v] * bs.anchor_pos.x < 0.0f) continue;
            if (!bs.snap_flag[mv]) {
                bs.snap_flag[mv] = true;
                if (bs.stroke_writes_to_base) {
                    bs.snap_x[mv] = ctx.mesh.pos_x[mv];
                    bs.snap_y[mv] = ctx.mesh.pos_y[mv];
                    bs.snap_z[mv] = ctx.mesh.pos_z[mv];
                } else {
                    bs.snap_x[mv] = ctx.multires.disp[bs.stroke_disp_index][mv].x;
                    bs.snap_y[mv] = ctx.multires.disp[bs.stroke_disp_index][mv].y;
                    bs.snap_z[mv] = ctx.multires.disp[bs.stroke_disp_index][mv].z;
                }
                bs.snap_list.push_back(mv);
            }
            bs.dirty_verts.push_back(mv);
        }
    }
}

static uint32_t mirror_of(uint32_t v, const Mesh& m) {
    if (!m.mirror_x_map.empty() && v < (uint32_t)m.mirror_x_map.size()) {
        uint32_t mv = m.mirror_x_map[v];
        if (mv < (uint32_t)m.mirror_x_map.size()) return mv;
    }
    return v;
}

BrushStroke::BrushStroke()
    : region_x(0), region_y(0), region_w(0), region_h(0)
    , cached_w(0), cached_h(0)
    , screen_w(0), screen_h(0)
    , anchor_valid(false)
    , anchor_world_radius(0.0f)
    , cyl_axis_x(0.0f), cyl_axis_y(0.0f), cyl_axis_z(0.0f)
    , screen_slack(1.0f)
    , last_stroke_valid(false)
    , cursor_hist_count(0)
    , stroke_level(0), stroke_disp_index(-1), stroke_writes_to_base(true), stroke_sign(0)
    , compute(nullptr)
    , gpu_normals_deferred(false)
    , gpu_positions_deferred(false)
    , gpu_mask_deferred(false)
    , gpu_color_deferred(false)
    , phase(StrokePhase::NONE), needs_mesh_update(false), vertex_count(0)
{}

void BrushStroke::set_anchor(const Mesh& mesh, const Camera& cam,
                              float cursor_x, float cursor_y, float brush_radius,
                              int screen_h, Renderer& renderer) {
    anchor_valid = false;
    int cx = (int)cursor_x;
    int cy = (int)cursor_y;
    if (cx < 0 || cx >= screen_w || cy < 0 || cy >= screen_h) return;

    uint32_t tid;
    float nx, ny, nz;

    renderer.read_triid_region(cx, cy, 1, 1, &tid);
    float norm_pixel[3];
    renderer.read_normal_region(cx, cy, 1, 1, norm_pixel);
    nx = norm_pixel[0];
    ny = norm_pixel[1];
    nz = norm_pixel[2];

    // Active entity's tris start at 0 in the pick FBO. tid is just the on-model hit test.
    if (tid == 0xFFFFFFFF || tid >= mesh.tri_count()) return;

    // Anchor world position from the FRESH GPU depth at the cursor pixel, unprojected
    // through the camera (same reconstruction as insert_controller's on-model hit).
    // Sourcing this from the pick FBO instead of interpolating CPU mesh.pos keeps the
    // anchor correct once the pen-up readback is deferred and mesh.pos goes stale
    // (blood-moon 3b-iv 2c): the depth buffer is rendered from the live VBO every
    // pen-down, so it tracks the GPU surface; mesh.pos no longer will. Behavior-neutral
    // while mesh.pos is still fresh — depth and mesh.pos describe the same surface point.
    //
    // NOTE: this camera is ORTHOGRAPHIC (see Camera::get_projection_matrix /
    // world_to_screen) — the lateral screen scale is fixed by the orbit `distance`, NOT
    // by the per-pixel depth. So the in-plane offset uses `distance`; `hit_depth` only
    // places the point along the view axis. (linear depth = fwd · (world − cam_pos).)
    float hit_depth = 0.0f;
    renderer.read_depth_region(cx, cy, 1, 1, &hit_depth);
    float ndc_x  = 2.0f * (float)cx / (float)screen_w - 1.0f;
    float ndc_y  = 1.0f - 2.0f * (float)cy / (float)screen_h;
    float aspect = (float)screen_w / (float)screen_h;
    Vec3 fwd      = cam.get_view_direction();
    Vec3 world_up = {0.0f, 1.0f, 0.0f};
    Vec3 right    = fwd.cross(world_up).normalized();
    Vec3 up       = right.cross(fwd).normalized();
    float half_h  = cam.distance * std::tan(cam.fov * 3.14159265358979323846f / 360.0f);
    float half_w  = half_h * aspect;
    Vec3 cam_pos  = cam.get_position();
    anchor_pos = cam_pos + fwd * hit_depth
               + right * (ndc_x * half_w) + up * (ndc_y * half_h);

    float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nlen > 1e-6f) {
        cyl_axis_x = nx / nlen;
        cyl_axis_y = ny / nlen;
        cyl_axis_z = nz / nlen;
    } else {
        cyl_axis_x = 0.0f; cyl_axis_y = 1.0f; cyl_axis_z = 0.0f;
    }

    float fov_rad = cam.fov * 3.14159265358979323846f / 180.0f;
    float view_plane_h = 2.0f * cam.distance * std::tan(fov_rad * 0.5f);
    float world_per_pixel = view_plane_h / (float)screen_h;
    anchor_world_radius = brush_radius * world_per_pixel;

    Vec3 vd = cam.get_view_direction();
    float cos_theta = std::abs(vd.x * cyl_axis_x + vd.y * cyl_axis_y + vd.z * cyl_axis_z);
    screen_slack = std::min(3.0f, 1.0f / std::max(cos_theta, 0.33f));

    anchor_valid = true;
    last_stroke_pos = anchor_pos;
    last_stroke_valid = true;
}

float BrushStroke::falloff(float dist, float radius, float hardness) const {
    if (dist >= radius) return 0.0f;
    float t = dist / radius;
    float inner = 0.15f + hardness * 0.55f;
    if (t <= inner) return 1.0f;
    float blend = (t - inner) / (1.0f - inner + 1e-6f);
    blend = blend * blend * (3.0f - 2.0f * blend);
    return 1.0f - blend;
}


void BrushStroke::begin(Renderer& renderer, const Camera& cam,
                        float screen_x, float screen_y, float brush_radius,
                        int screen_w_in, int screen_h_in, uint32_t vert_count,
                        const MultiresStack& multires, BrushType brush_type,
                        uint32_t entity_id, MultiresGPU& mgpu) {
    phase = StrokePhase::BEGIN;
    needs_mesh_update = false;
    vertex_count = vert_count;
    stroke_entity_id = entity_id;
    last_dab_x = screen_x;
    last_dab_y = screen_y;

    screen_w = screen_w_in;
    screen_h = screen_h_in;

    stroke_level = multires.current_level;
    stroke_writes_to_base = (stroke_level == multires.base_level);
    stroke_disp_index = stroke_writes_to_base ? -1 : (stroke_level - multires.base_level - 1);
    stroke_sign = 0;

    dirty_verts.clear();
    gpu_dirty.clear();

    anchor_valid = false;

    snap_flag.assign(vert_count, false);
    snap_x.assign(vert_count, 0.0f);
    snap_y.assign(vert_count, 0.0f);
    snap_z.assign(vert_count, 0.0f);
    snap_list.clear();

    mask.reset();
    color.reset();

    renderer.render_screen_buffers(cam, screen_w, screen_h);

    // Freeze per-vertex normals for the entire stroke. Draw_accum reads
    // displacement direction from this snapshot — without it, each dab would
    // displace along the live vbo_norm, which compute_normals has already
    // tilted toward the camera. That feedback loop produces folded triangles
    // on long strokes near silhouettes.
    if (compute && compute->supported
        && (brush_type == BrushType::DRAW
            || brush_type == BrushType::INFLATE
            || brush_type == BrushType::CREASE
            || brush_type == BrushType::PINCH)) {
        compute->snapshot_stroke_normals(renderer.vbo_norm.handle, vert_count);
    }

    // Phase 2a (GPU-resident undo): snapshot the pen-down VBO positions so the
    // pen-up diff shader can reproject world deltas into disp/base on the GPU.
    // Geometry brushes only — MASK/PAINT don't touch positions. Pure capture for
    // now (no consumer yet), gated on compute support so it's behavior-neutral.
    if (compute && compute->supported && mgpu.supported
        && brush_type != BrushType::MASK
        && brush_type != BrushType::PAINT) {
        mgpu.snapshot_positions(renderer.vbo_pos.handle, vert_count);
    }

    // Only the CPU mask fallback (apply_mask → walk_brush_region) ever reads these
    // full-screen buffers. MOVE gets its anchor from set_anchor's per-dab 1x1 reads,
    // and GPU mask (apply_mask_gpu) paints directly on the VBO — neither consumes
    // cached_triid/cached_bary. So gate the ~100MB blocking read to exactly the case
    // that uses it: MASK with no GPU mask program available. (matches main.cpp:958)
    bool need_fullscreen = (brush_type == BrushType::MASK)
                           && !(compute && compute->supported && compute->has_mask());
    if (need_fullscreen) {
        int total_pixels = screen_w * screen_h;
        cached_triid.resize(total_pixels);
        cached_bary.resize(total_pixels * 2);
        renderer.read_triid_region(0, 0, screen_w, screen_h, cached_triid.data());
        renderer.read_bary_region(0, 0, screen_w, screen_h, cached_bary.data());
    } else {
        cached_triid.clear();
        cached_bary.clear();
    }

    cached_w = screen_w;
    cached_h = screen_h;

    // Move-only CPU fallback buffers
    if (brush_type == BrushType::MOVE) {
        disp_x.assign(vert_count, 0.0f);
        disp_y.assign(vert_count, 0.0f);
        disp_z.assign(vert_count, 0.0f);
        disp_weight.assign(vert_count, 0.0f);
    } else {
        disp_x.clear();
        disp_y.clear();
        disp_z.clear();
        disp_weight.clear();
    }
    // Both grab brushes capture-once; clear the sticky state at stroke start.
    if (brush_type == BrushType::MOVE || brush_type == BrushType::LIMB) {
        move.reset(vert_count);
    }
}


// --- GPU dispatch methods ---

void BrushStroke::apply_smooth(DabContext& ctx, float dab_x, float dab_y,
                                float strength, float hardness) {
    dirty_verts.clear();

    if (!ctx.compute.supported || !ctx.compute.has_smooth()
        || ctx.compute.adjacency_vertex_count == 0) {
        return;
    }

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!anchor_valid) return;

    ctx.compute.ensure_accum_buffer(ctx.vertex_count);

    SmoothAccumParams sp;
    sp.anchor_x = anchor_pos.x;
    sp.anchor_y = anchor_pos.y;
    sp.anchor_z = anchor_pos.z;
    sp.world_radius = anchor_world_radius;
    sp.hardness = hardness;
    sp.strength = strength * 0.64f;
    sp.iterations = 3;
    sp.vertex_count = ctx.mesh.vertex_count();
    sp.tri_count = ctx.mesh.tri_count();
    sp.mirror_x = ctx.input.mirror_x;

    BrushRegion rgn = compute_brush_region(dab_x, dab_y, ctx.eff_brush_size, screen_slack, ctx.win_w, ctx.win_h);
    sp.region_x = rgn.x;
    sp.region_y = rgn.y;
    sp.region_w = rgn.w;
    sp.region_h = rgn.h;
    sp.screen_h = ctx.win_h;

    ctx.compute.dispatch_smooth(sp,
        ctx.renderer.screen_triid_tex(), ctx.renderer.screen_bary_tex(),
        ctx.renderer.vbo_pos.handle, ctx.renderer.ebo.handle);

    ctx.compute.readback_smooth_dirty(dirty_verts);

    // Mirror reflection is now re-imposed after every smoothing iteration inside
    // dispatch_smooth (so the seam band relaxes in lockstep), making a separate
    // end-of-dab dispatch_smooth_mirror_apply redundant.

    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

void BrushStroke::apply_crease(DabContext& ctx, float dab_x, float dab_y,
                                float strength, float hardness, bool subtract) {
    dirty_verts.clear();

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!ctx.compute.supported || !ctx.compute.has_crease() || !anchor_valid) {
        return;
    }

    float sign = subtract ? 1.0f : -1.0f;
    float base_disp = anchor_world_radius;
    float disp_amount = base_disp * strength * sign * 0.15f;
    float pinch_amount = base_disp * strength * 0.35f;
    Vec3 view_dir = ctx.cam.get_view_direction();

    ctx.compute.ensure_accum_buffer(ctx.vertex_count);

    CreaseAccumParams params{};
    params.anchor_a_x = anchor_pos.x;
    params.anchor_a_y = anchor_pos.y;
    params.anchor_a_z = anchor_pos.z;
    params.anchor_b_x = -anchor_pos.x;
    params.anchor_b_y =  anchor_pos.y;
    params.anchor_b_z =  anchor_pos.z;
    params.use_b = ctx.input.mirror_x ? 1 : 0;
    params.world_radius = anchor_world_radius;
    params.disp_amount = disp_amount;
    params.pinch_amount = pinch_amount;
    params.hardness = hardness;
    params.facing_threshold = ctx.input.facing_threshold;
    params.view_a_x = view_dir.x;
    params.view_a_y = view_dir.y;
    params.view_a_z = view_dir.z;
    params.view_b_x = -view_dir.x;
    params.view_b_y =  view_dir.y;
    params.view_b_z =  view_dir.z;
    set_area_normal(params, cyl_axis_x, cyl_axis_y, cyl_axis_z);
    params.vertex_count = ctx.mesh.vertex_count();

    ctx.compute.dispatch_crease_accum(params, ctx.renderer.vbo_pos.handle);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    bool use_sym = ctx.input.mirror_x
                   && ctx.compute.has_draw_symmetrize()
                   && ctx.compute.mirror_map_vertex_count == ctx.vertex_count;
    if (use_sym) {
        ctx.compute.dispatch_draw_accum_symmetrize(ctx.mesh.vertex_count());
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count(),
                                         ctx.compute.accum_sym_ssbo);
    } else {
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count());
    }

    ctx.compute.readback_smooth_dirty(dirty_verts);
    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

void BrushStroke::apply_pinch(DabContext& ctx, float dab_x, float dab_y,
                               float strength, float hardness, bool subtract) {
    dirty_verts.clear();

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!ctx.compute.supported || !ctx.compute.has_pinch() || !anchor_valid) {
        return;
    }

    float sign = subtract ? -1.0f : 1.0f;
    float pinch_amount = anchor_world_radius * 0.5f;
    Vec3 view_dir = ctx.cam.get_view_direction();

    ctx.compute.ensure_accum_buffer(ctx.vertex_count);

    PinchAccumParams params{};
    params.anchor_a_x = anchor_pos.x;
    params.anchor_a_y = anchor_pos.y;
    params.anchor_a_z = anchor_pos.z;
    params.anchor_b_x = -anchor_pos.x;
    params.anchor_b_y =  anchor_pos.y;
    params.anchor_b_z =  anchor_pos.z;
    params.use_b = ctx.input.mirror_x ? 1 : 0;
    params.world_radius = anchor_world_radius;
    params.pinch_amount = pinch_amount * strength * sign;
    params.hardness = hardness;
    params.facing_threshold = ctx.input.facing_threshold;
    params.view_a_x = view_dir.x;
    params.view_a_y = view_dir.y;
    params.view_a_z = view_dir.z;
    params.view_b_x = -view_dir.x;
    params.view_b_y =  view_dir.y;
    params.view_b_z =  view_dir.z;
    set_area_normal(params, cyl_axis_x, cyl_axis_y, cyl_axis_z);
    params.vertex_count = ctx.mesh.vertex_count();

    ctx.compute.dispatch_pinch_accum(params, ctx.renderer.vbo_pos.handle);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    bool use_sym = ctx.input.mirror_x
                   && ctx.compute.has_draw_symmetrize()
                   && ctx.compute.mirror_map_vertex_count == ctx.vertex_count;
    if (use_sym) {
        ctx.compute.dispatch_draw_accum_symmetrize(ctx.mesh.vertex_count());
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count(),
                                         ctx.compute.accum_sym_ssbo);
    } else {
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count());
    }

    ctx.compute.readback_smooth_dirty(dirty_verts);
    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

void BrushStroke::apply_draw(DabContext& ctx, float dab_x, float dab_y,
                              float strength, float hardness, bool subtract,
                              bool inflate) {
    dirty_verts.clear();

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!ctx.compute.supported || !ctx.compute.has_draw() || !anchor_valid) {
        return;
    }

    if (stroke_sign == 0)
        stroke_sign = subtract ? -1 : 1;
    float base_disp = anchor_world_radius;
    float disp_amount = base_disp * strength * (float)stroke_sign * 0.5f;

    Vec3 view_dir = ctx.cam.get_view_direction();

    ctx.compute.ensure_accum_buffer(ctx.vertex_count);
    ctx.compute.clear_accum_buffer();

    DrawAccumParams params{};
    params.anchor_a_x = anchor_pos.x;
    params.anchor_a_y = anchor_pos.y;
    params.anchor_a_z = anchor_pos.z;
    params.anchor_b_x = -anchor_pos.x;
    params.anchor_b_y =  anchor_pos.y;
    params.anchor_b_z =  anchor_pos.z;
    params.use_b = ctx.input.mirror_x ? 1 : 0;
    params.world_radius = anchor_world_radius;
    params.disp_amount = disp_amount;
    params.hardness = hardness;
    params.facing_threshold = ctx.input.facing_threshold;
    params.view_a_x = view_dir.x;
    params.view_a_y = view_dir.y;
    params.view_a_z = view_dir.z;
    params.view_b_x = -view_dir.x;
    params.view_b_y =  view_dir.y;
    params.view_b_z =  view_dir.z;
    set_area_normal(params, cyl_axis_x, cyl_axis_y, cyl_axis_z);
    params.vertex_count = ctx.mesh.vertex_count();
    params.inflate = inflate ? 1 : 0;

    ctx.compute.dispatch_draw_accum(params, ctx.renderer.vbo_pos.handle);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Symmetrize accum across mirror pairs before apply. Each paired (v, mv)
    // gets out[v] = accum[v] + (-mx, my, mz, mw), so apply produces strictly
    // X-mirror displacements regardless of small tessellation drift between
    // twins. Orphan/seam verts copy through unchanged.
    bool use_sym = ctx.input.mirror_x
                   && ctx.compute.has_draw_symmetrize()
                   && ctx.compute.mirror_map_vertex_count == ctx.vertex_count;
    if (use_sym) {
        ctx.compute.dispatch_draw_accum_symmetrize(ctx.mesh.vertex_count());
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count(),
                                         ctx.compute.accum_sym_ssbo);
    } else {
        ctx.compute.dispatch_draw_apply(ctx.renderer.vbo_pos.handle, ctx.mesh.vertex_count());
    }

    ctx.compute.readback_smooth_dirty(dirty_verts);
    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

void BrushStroke::apply_move_gpu(DabContext& ctx, float cursor_dx, float cursor_dy,
                                  float strength, float hardness) {
    dirty_verts.clear();

    if (!is_active()) return;

    float dab_x = (float)ctx.input.mouse_x;
    float dab_y = (float)ctx.input.mouse_y;

    if (!move.captured) {
        set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
        if (!anchor_valid) return;

        uint32_t vc = ctx.mesh.vertex_count();
        ctx.compute.ensure_accum_buffer(ctx.vertex_count);
        ctx.compute.ensure_move_buffers(ctx.vertex_count);

        // Capture over the active entity's range. mirror_x reflects the anchor
        // within this same range, so on a symmetric mesh both lobes displace in
        // one pass (the w.x/w.y weight decomposition produces the X-mirror).
        MoveCaptureParams cp;
        cp.anchor_x = anchor_pos.x;
        cp.anchor_y = anchor_pos.y;
        cp.anchor_z = anchor_pos.z;
        cp.world_radius = anchor_world_radius;
        cp.hardness = hardness;
        cp.mirror_x = ctx.input.mirror_x;
        cp.vertex_count = vc;
        ctx.compute.dispatch_move_capture(cp, ctx.renderer.vbo_pos.handle);

        // 3 Laplacian iterations on weights (matches CPU)
        ctx.compute.dispatch_move_weight_smooth(vc, 3, ctx.renderer.ebo.handle);

        // Capture writes only the active entity's range [0, vc) — no offset filter.
        ctx.compute.readback_move_affected(move.affected_list);

        move.total_dx = move.total_dy = move.total_dz = 0.0f;
        move.captured = true;
    }

    if (move.affected_list.empty()) return;

    // Accumulate cursor delta → world-space total
    float fov_rad = ctx.cam.fov * 3.14159265358979323846f / 180.0f;
    float view_plane_h = 2.0f * ctx.cam.distance * std::tan(fov_rad * 0.5f);
    float scale = view_plane_h / (float)ctx.win_h;

    Vec3 pos = ctx.cam.get_position();
    Vec3 fwd = (ctx.cam.target - pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 right = fwd.cross(world_up).normalized();
    Vec3 up = right.cross(fwd).normalized();

    float wdx = cursor_dx * scale * strength;
    float wdy = -cursor_dy * scale * strength;

    move.total_dx += (right.x * wdx + up.x * wdy);
    move.total_dy += (right.y * wdx + up.y * wdy);
    move.total_dz += (right.z * wdx + up.z * wdy);

    MoveApplyParams ap;
    ap.total_dx = move.total_dx;
    ap.total_dy = move.total_dy;
    ap.total_dz = move.total_dz;
    ap.mirror_x = ctx.input.mirror_x;
    ap.vertex_count = ctx.vertex_count;
    ctx.compute.dispatch_move_apply(ap, ctx.renderer.vbo_pos.handle);

    dirty_verts = move.affected_list;
    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

void BrushStroke::apply_limb_gpu(DabContext& ctx, float cursor_dx, float cursor_dy,
                                  float strength, float hardness) {
    // Tangential redistribution per dab. Fixed for now; cheap (a few sweeps over
    // the captured set). lambda = per-iter step, iters = sweeps.
    const float LIMB_RELAX_LAMBDA = 0.5f;
    const int   LIMB_RELAX_ITERS  = 6;
    // Tip bias: up-weights tipward neighbours in the relax centroid so the vertex
    // distribution drifts toward the leading end → a denser cap ("top-heavy" limb).
    // 0 = old symmetric relax. Tuning knob; "slightly more tris at the tip".
    const float LIMB_TIP_BIAS     = 0.5f;

    dirty_verts.clear();
    if (!is_active()) return;

    float dab_x = (float)ctx.input.mouse_x;
    float dab_y = (float)ctx.input.mouse_y;

    // Capture once at pen-down — identical to the move brush (sticky set + falloff
    // weights + mirror decomposition). This is what lets the pulled limb keep
    // being driven no matter how far it travels from the anchor.
    if (!move.captured) {
        set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
        if (!anchor_valid) return;

        uint32_t vc = ctx.mesh.vertex_count();
        ctx.compute.ensure_move_buffers(ctx.vertex_count);
        ctx.compute.ensure_limb_buffers(ctx.vertex_count);

        MoveCaptureParams cp;
        cp.anchor_x = anchor_pos.x;
        cp.anchor_y = anchor_pos.y;
        cp.anchor_z = anchor_pos.z;
        cp.world_radius = anchor_world_radius;
        cp.hardness = hardness;
        cp.mirror_x = ctx.input.mirror_x;
        cp.vertex_count = vc;
        ctx.compute.dispatch_move_capture(cp, ctx.renderer.vbo_pos.handle);
        ctx.compute.dispatch_move_weight_smooth(vc, 3, ctx.renderer.ebo.handle);
        ctx.compute.readback_move_affected(move.affected_list);

        move.captured = true;
    }

    if (move.affected_list.empty()) return;

    // This dab's world-space drag increment (screen delta → view plane). Unlike
    // move this is applied incrementally so the relax accumulates frame to frame.
    float fov_rad = ctx.cam.fov * 3.14159265358979323846f / 180.0f;
    float view_plane_h = 2.0f * ctx.cam.distance * std::tan(fov_rad * 0.5f);
    float scale = view_plane_h / (float)ctx.win_h;

    Vec3 pos = ctx.cam.get_position();
    Vec3 fwd = (ctx.cam.target - pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 right = fwd.cross(world_up).normalized();
    Vec3 up = right.cross(fwd).normalized();

    float wdx = cursor_dx * scale * strength;
    float wdy = -cursor_dy * scale * strength;

    LimbDragParams dp;
    dp.dx = right.x * wdx + up.x * wdy;
    dp.dy = right.y * wdx + up.y * wdy;
    dp.dz = right.z * wdx + up.z * wdy;
    dp.mirror_x = ctx.input.mirror_x;
    dp.vertex_count = ctx.vertex_count;
    ctx.compute.dispatch_limb_drag(dp, ctx.renderer.vbo_pos.handle);

    // Accumulate the world drag → stable tip direction (the leading end the limb has
    // been pulled toward). total_d* is reset by move.reset() at pen-down (begin()).
    move.total_dx += dp.dx;
    move.total_dy += dp.dy;
    move.total_dz += dp.dz;
    float tlen = std::sqrt(move.total_dx*move.total_dx + move.total_dy*move.total_dy +
                           move.total_dz*move.total_dz);
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;
    if (tlen > 1e-8f) { tx = move.total_dx/tlen; ty = move.total_dy/tlen; tz = move.total_dz/tlen; }

    ctx.compute.dispatch_limb_relax(ctx.vertex_count, LIMB_RELAX_ITERS, LIMB_RELAX_LAMBDA,
                                    tx, ty, tz, LIMB_TIP_BIAS,
                                    ctx.renderer.vbo_pos.handle, ctx.renderer.vbo_norm.handle, ctx.renderer.ebo.handle);

    dirty_verts = move.affected_list;
    snap_and_mirror_dirty(*this, ctx);

    gpu_positions_deferred = true;
}

// --- Post-dispatch methods ---

void BrushStroke::post_dab(DabContext& ctx) {
    if (!dirty_verts.empty()) {
        static std::vector<uint32_t> dab_affected;
        ctx.mesh.expand_dirty_to_affected(dirty_verts, dab_affected);
        gpu_dirty.insert(gpu_dirty.end(), dab_affected.begin(), dab_affected.end());
    }
}

void BrushStroke::post_frame(DabContext& ctx) {
    if (gpu_dirty.empty()) return;

    std::sort(gpu_dirty.begin(), gpu_dirty.end());
    gpu_dirty.erase(std::unique(gpu_dirty.begin(), gpu_dirty.end()), gpu_dirty.end());
    ctx.compute.dispatch_compute_normals(gpu_dirty.data(),
                                          (uint32_t)gpu_dirty.size(),
                                          ctx.renderer.vbo_pos.handle, ctx.renderer.vbo_norm.handle,
                                          ctx.renderer.ebo.handle);
    gpu_normals_deferred = true;
    gpu_dirty.clear();
}


// --- Existing methods with member reference updates ---

void BrushStroke::apply_mask_gpu(DabContext& ctx, float dab_x, float dab_y,
                                 float strength, float hardness, bool invert) {
    if (!is_active()) return;
    if (!ctx.compute.supported || !ctx.compute.has_mask()) return;

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!anchor_valid) return;

    float paint_strength = invert ? -strength : strength;

    MaskPaintParams p{};
    p.anchor_a_x = anchor_pos.x;
    p.anchor_a_y = anchor_pos.y;
    p.anchor_a_z = anchor_pos.z;
    p.anchor_b_x = -anchor_pos.x;
    p.anchor_b_y =  anchor_pos.y;
    p.anchor_b_z =  anchor_pos.z;
    p.use_b = ctx.input.mirror_x ? 1 : 0;
    p.world_radius = anchor_world_radius;
    p.hardness = hardness;
    p.paint_strength = paint_strength;
    p.vertex_count = ctx.mesh.vertex_count();

    ctx.compute.dispatch_mask_paint(p, ctx.renderer.vbo_pos.handle);

    std::vector<uint32_t> dirty;
    ctx.compute.readback_smooth_dirty(dirty);

    if (mask.snap_flag.empty()) {
        mask.snap_flag.assign(vertex_count, false);
        mask.snap.assign(vertex_count, 0.0f);
    }
    for (uint32_t v : dirty) {
        if (!mask.snap_flag[v]) {
            mask.snap_flag[v] = true;
            mask.snap[v] = (v < (uint32_t)ctx.mesh.mask.size()) ? ctx.mesh.mask[v] : 0.0f;
            mask.snap_list.push_back(v);
        }
    }

    gpu_mask_deferred = true;
    needs_mesh_update = true;
}

void BrushStroke::apply_color_gpu(DabContext& ctx, float dab_x, float dab_y,
                                  float strength, float hardness, bool erase) {
    if (!is_active()) return;
    if (!ctx.compute.supported || !ctx.compute.has_color()) return;

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!anchor_valid) return;

    ColorPaintParams p{};
    p.anchor_a_x = anchor_pos.x;
    p.anchor_a_y = anchor_pos.y;
    p.anchor_a_z = anchor_pos.z;
    p.anchor_b_x = -anchor_pos.x;
    p.anchor_b_y =  anchor_pos.y;
    p.anchor_b_z =  anchor_pos.z;
    p.use_b = ctx.input.mirror_x ? 1 : 0;
    p.world_radius = anchor_world_radius;
    p.hardness = hardness;
    p.paint_strength = strength;
    if (erase) {
        p.paint_r = p.paint_g = p.paint_b = 1.0f;   // erase = lerp toward white
    } else {
        p.paint_r = ctx.input.paint_color[0];
        p.paint_g = ctx.input.paint_color[1];
        p.paint_b = ctx.input.paint_color[2];
    }
    p.vertex_count = ctx.mesh.vertex_count();

    ctx.compute.dispatch_color_paint(p, ctx.renderer.vbo_pos.handle);

    std::vector<uint32_t> dirty;
    ctx.compute.readback_smooth_dirty(dirty);

    if (color.snap_flag.empty()) {
        color.snap_flag.assign(vertex_count, false);
        color.snap.assign(vertex_count, 0xFFFFFFFFu);
    }
    for (uint32_t v : dirty) {
        if (!color.snap_flag[v]) {
            color.snap_flag[v] = true;
            color.snap[v] = (v < (uint32_t)ctx.mesh.color.size()) ? ctx.mesh.color[v] : 0xFFFFFFFFu;
            color.snap_list.push_back(v);
        }
    }

    gpu_color_deferred = true;
    needs_mesh_update = true;
}

void BrushStroke::apply_color_smooth_gpu(DabContext& ctx, float dab_x, float dab_y,
                                         float strength, float hardness) {
    if (!is_active()) return;
    if (!ctx.compute.supported || !ctx.compute.has_color_smooth()) return;

    set_anchor(ctx.mesh, ctx.cam, dab_x, dab_y, ctx.eff_brush_size, ctx.win_h, ctx.renderer);
    if (!anchor_valid) return;

    ColorPaintParams p{};
    p.anchor_a_x = anchor_pos.x;
    p.anchor_a_y = anchor_pos.y;
    p.anchor_a_z = anchor_pos.z;
    p.anchor_b_x = -anchor_pos.x;
    p.anchor_b_y =  anchor_pos.y;
    p.anchor_b_z =  anchor_pos.z;
    p.use_b = ctx.input.mirror_x ? 1 : 0;
    p.world_radius = anchor_world_radius;
    p.hardness = hardness;
    p.paint_strength = strength;        // blend amount toward neighbour average
    p.vertex_count = ctx.mesh.vertex_count();

    ctx.compute.dispatch_color_smooth(p, ctx.renderer.vbo_pos.handle, ctx.renderer.ebo.handle);

    std::vector<uint32_t> dirty;
    ctx.compute.readback_smooth_dirty(dirty);

    if (color.snap_flag.empty()) {
        color.snap_flag.assign(vertex_count, false);
        color.snap.assign(vertex_count, 0xFFFFFFFFu);
    }
    for (uint32_t v : dirty) {
        if (!color.snap_flag[v]) {
            color.snap_flag[v] = true;
            color.snap[v] = (v < (uint32_t)ctx.mesh.color.size()) ? ctx.mesh.color[v] : 0xFFFFFFFFu;
            color.snap_list.push_back(v);
        }
    }

    gpu_color_deferred = true;
    needs_mesh_update = true;
}

void BrushStroke::apply_mask(Renderer& renderer, const Camera& cam,
                              const Mesh& mesh,
                              float screen_x, float screen_y,
                              float brush_radius, float strength, float hardness,
                              bool invert, bool mirror_x,
                              int screen_w, int screen_h) {
    if (!is_active()) return;

    set_anchor(mesh, cam, screen_x, screen_y, brush_radius, screen_h, renderer);

    if (mask.delta.empty()) mask.delta.assign(vertex_count, 0.0f);

    float paint_strength = invert ? -strength : strength;
    bool use_mirror = mirror_x && !mesh.mirror_x_map.empty();

    walk_brush_region(mesh, screen_x, screen_y, brush_radius, screen_w, screen_h,
        [&](uint32_t i0, uint32_t i1, uint32_t i2,
            float /*bu*/, float /*bv*/, float /*bw*/, float dist, float eff_radius, int /*stride*/, int /*pixel_idx*/) {
            float w = falloff(dist, eff_radius, hardness);
            if (w <= 0.0f) return;

            float contrib = paint_strength * w;
            mask.delta[i0] += contrib;
            mask.delta[i1] += contrib;
            mask.delta[i2] += contrib;

            if (use_mirror) {
                uint32_t mi0 = mirror_of(i0, mesh);
                uint32_t mi1 = mirror_of(i1, mesh);
                uint32_t mi2 = mirror_of(i2, mesh);
                mask.delta[mi0] += contrib;
                mask.delta[mi1] += contrib;
                mask.delta[mi2] += contrib;
            }
        });
}

void BrushStroke::commit_undo(const Mesh& mesh, UndoStack& stack, const MultiresStack& multires) {
    if (snap_list.empty()) return;

    UndoEntry e;
    e.level        = stroke_level;
    e.targets_base = stroke_writes_to_base;
    e.disp_index   = stroke_disp_index;

    size_t n = snap_list.size();

    // When the pen-up diff captured (old,new) into the GPU undo ring, the ring is
    // packed densely by snap_list index and now OWNS the (old,new) values (2c-iii).
    const bool ring = (stroke_ring_base != SIZE_MAX);

    if (ring) {
        // The flip dropped the pen-up readback, so mesh.pos/disp are stale here — do
        // NOT read them. Record the unfiltered snap_list (verts[k] aligns with ring
        // slot k) and SIZE old_*/new_* to vcount as placeholders so entry_bytes counts
        // them; the evict/park/cross-level spill writes the real values from the ring.
        e.verts = snap_list;
        e.old_x.assign(n, 0.0f); e.old_y.assign(n, 0.0f); e.old_z.assign(n, 0.0f);
        e.new_x.assign(n, 0.0f); e.new_y.assign(n, 0.0f); e.new_z.assign(n, 0.0f);
        e.ring_offset = stroke_ring_base;
        e.ring_vcount = (uint32_t)n;
    } else {
        // Degrade path (no ring capture): the readback ran, mesh.pos/disp are fresh.
        e.verts.reserve(n);
        e.old_x.reserve(n); e.old_y.reserve(n); e.old_z.reserve(n);
        e.new_x.reserve(n); e.new_y.reserve(n); e.new_z.reserve(n);
        for (uint32_t v : snap_list) {
            float ox = snap_x[v], oy = snap_y[v], oz = snap_z[v];
            float nx, ny, nz;
            if (stroke_writes_to_base) {
                nx = mesh.pos_x[v]; ny = mesh.pos_y[v]; nz = mesh.pos_z[v];
            } else {
                nx = multires.disp[stroke_disp_index][v].x;
                ny = multires.disp[stroke_disp_index][v].y;
                nz = multires.disp[stroke_disp_index][v].z;
            }
            if (ox == nx && oy == ny && oz == nz) continue;
            e.verts.push_back(v);
            e.old_x.push_back(ox); e.old_y.push_back(oy); e.old_z.push_back(oz);
            e.new_x.push_back(nx); e.new_y.push_back(ny); e.new_z.push_back(nz);
        }
    }

    stack.push(std::move(e));
}

void BrushStroke::commit_mask_undo(const Mesh& mesh, UndoStack& stack) {
    if (mask.snap_list.empty()) return;

    UndoEntry e;
    e.kind      = UndoEntry::Kind::MASK;

    for (uint32_t v : mask.snap_list) {
        float old_val = mask.snap[v];
        float new_val = (v < (uint32_t)mesh.mask.size()) ? mesh.mask[v] : 0.0f;
        if (old_val == new_val) continue;
        e.verts.push_back(v);
        e.old_mask.push_back(old_val);
        e.new_mask.push_back(new_val);
    }

    stack.push(std::move(e));
}

void BrushStroke::commit_color_undo(const Mesh& mesh, UndoStack& stack) {
    if (color.snap_list.empty()) return;

    UndoEntry e;
    e.kind = UndoEntry::Kind::PAINT;

    for (uint32_t v : color.snap_list) {
        uint32_t old_val = color.snap[v];
        uint32_t new_val = (v < (uint32_t)mesh.color.size()) ? mesh.color[v] : 0xFFFFFFFFu;
        if (old_val == new_val) continue;
        e.verts.push_back(v);
        e.old_color.push_back(old_val);
        e.new_color.push_back(new_val);
    }

    if (e.verts.empty()) return;
    stack.push(std::move(e));
}

void BrushStroke::apply_mask_changes(Mesh& mesh, std::vector<uint32_t>& dirty_verts_out) {
    dirty_verts_out.clear();

    if (mask.delta.empty()) return;

    if (mesh.mask.empty()) {
        mesh.mask.assign(mesh.vertex_count(), 0.0f);
    } else if (mesh.mask.size() < mesh.vertex_count()) {
        mesh.mask.resize(mesh.vertex_count(), 0.0f);
    }

    if (mask.snap_flag.empty()) {
        mask.snap_flag.assign(vertex_count, false);
        mask.snap.assign(vertex_count, 0.0f);
    }

    for (uint32_t v = 0; v < vertex_count; v++) {
        if (v >= mask.delta.size() || mask.delta[v] == 0.0f) continue;

        if (!mask.snap_flag[v]) {
            mask.snap_flag[v] = true;
            mask.snap[v] = mesh.mask[v];
            mask.snap_list.push_back(v);
        }

        float new_val = std::max(0.0f, std::min(1.0f, mesh.mask[v] + mask.delta[v]));
        mesh.mask[v] = new_val;
        dirty_verts_out.push_back(v);
    }

    std::fill(mask.delta.begin(), mask.delta.end(), 0.0f);
}

bool BrushStroke::finalize(Mesh& mesh, UndoStack& stack, MultiresStack& multires,
                           MultiresGPU& mgpu, Renderer& renderer,
                           BrushType brush_type, bool autosmooth) {
    stroke_ring_base = SIZE_MAX;   // set below iff the pen-up diff captured into the ring (3b-iv)
    // Autosmooth: one Laplacian pass over the stroke's affected verts before
    // we read positions back. The smoothed positions become the "after" state
    // recorded in the undo entry. Draw-only — other brushes have their own
    // tuning. Mask gating is built into the shader.
    static constexpr float AUTOSMOOTH_STRENGTH   = 0.3f;
    static constexpr int   AUTOSMOOTH_ITERATIONS = 1;
    if (autosmooth
        && brush_type == BrushType::DRAW
        && !snap_list.empty()
        && compute && compute->supported
        && compute->has_stroke_smooth()
        && compute->adjacency_vertex_count > 0) {
        for (int it = 0; it < AUTOSMOOTH_ITERATIONS; it++) {
            compute->dispatch_stroke_smooth_apply(snap_list.data(),
                                                   (uint32_t)snap_list.size(),
                                                   AUTOSMOOTH_STRENGTH,
                                                   renderer.vbo_pos.handle, renderer.ebo.handle);
        }
        // Refresh normals for the smoothed verts so the deferred normal
        // readback below picks up post-smoothing values.
        if (compute->has_normals()) {
            compute->dispatch_compute_normals(snap_list.data(),
                                               (uint32_t)snap_list.size(),
                                               renderer.vbo_pos.handle, renderer.vbo_norm.handle,
                                               renderer.ebo.handle);
            gpu_normals_deferred = true;
        }
        // Smoothing changed positions on the VBO; force the deferred-readback
        // path below so mesh.pos_* / multires sync to the new state.
        gpu_positions_deferred = true;
    }

    // Deferred position readback: GPU wrote positions to VBO during the stroke.
    // Sync mesh.pos_* and multires at pen-up before commit_undo. The active
    // entity is the working buffer, so vert index == VBO offset (offset 0).
    if (gpu_positions_deferred && !snap_list.empty()) {
        // Phase 2b (GPU-resident undo): reproject the stroke delta into the
        // resident disp/base layer on the GPU, from the pen-down snapshots. This
        // is the twin of the CPU readback loop below. The CPU path stays
        // authoritative in 2b (it re-syncs disp_ssbo via upload_disp_partial at
        // the end); under CHISEL_DEBUG_MULTIRES we snapshot the GPU result here
        // and compare it to the CPU result before that re-sync overwrites it.
        bool gpu_diff_ran = false;
        size_t ring_base = SIZE_MAX;   // float offset of this stroke's span in the undo ring (3b-iii)
        if (compute && compute->supported && mgpu.supported
            && mgpu.level == stroke_level) {
            // Reserve float6 per touched vert; the diff shader fills (old,new). On
            // cap overflow reserve returns SIZE_MAX and we skip ring capture (the
            // CPU path stays authoritative — dual-bookkeeping in 3b-iii).
            ring_base = compute->undo_ring_reserve(snap_list.size() * 6);
            stroke_ring_base = ring_base;   // recorded on the UndoEntry by commit_undo (3b-iv)
            // Circular ring (3b-iv part 2): the span we just reserved may overwrite
            // older entries' bytes. Invalidate them (→ CPU-stage fallback) before the
            // diff fills the span. No-op until the ring wraps at the cap.
            if (ring_base != SIZE_MAX)
                stack.ring_evict_overlap(ring_base * sizeof(float),
                                         snap_list.size() * 6 * sizeof(float), *compute);
            GLuint ring_ssbo = (ring_base != SIZE_MAX) ? compute->undo_ring_ssbo.handle : 0;
            compute->dispatch_multires_diff(renderer.vbo_pos, mgpu.disp_ssbo,
                                            mgpu.frames_ssbo, mgpu.snap_pos_ssbo,
                                            mgpu.base_ssbo,
                                            snap_list.data(), (uint32_t)snap_list.size(),
                                            stroke_writes_to_base,
                                            ring_ssbo,
                                            (uint32_t)(ring_base == SIZE_MAX ? 0 : ring_base));
            gpu_diff_ran = true;
        }
#ifdef CHISEL_DEBUG_MULTIRES
        static std::vector<float> gpu_diff_chk;   // 3 floats per snap_list entry
        if (gpu_diff_ran) {
            gpu_diff_chk.resize(snap_list.size() * 3);
            GLuint src = (stroke_writes_to_base ? mgpu.base_ssbo : mgpu.disp_ssbo).handle;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, src);
            for (size_t i = 0; i < snap_list.size(); i++) {
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    (GLintptr)snap_list[i] * 3 * sizeof(float),
                    (GLsizeiptr)3 * sizeof(float), &gpu_diff_chk[i * 3]);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
#else
        (void)gpu_diff_ran;
#endif

        // THE FLIP (2c-iii): when the pen-up diff captured (old,new) into the GPU undo
        // ring, the GPU now owns this stroke — the working VBO holds the new surface and
        // disp_ssbo/base_ssbo the new storage. Drop the CPU position readback; just mark
        // the CPU copy dirty so the next CPU reader (save/cascade/remesh/mirror) pulls it
        // down through the Scene materialize choke. The degrade path (ring overflow OR no
        // compute / level mismatch → ring_captured false) keeps the authoritative readback:
        // those cases have no GPU truth (architecture rule: always degrade gracefully).
        const bool ring_captured = (ring_base != SIZE_MAX);
        if (!ring_captured) {
            // Invariant: this baseline (mesh.pos) is the pre-stroke surface. Staleness
            // only ever comes from a prior flipped stroke, and the flip requires
            // compute — so the only reachable !ring_captured case is !compute->supported
            // (or a never-flipped session), where no prior stroke went stale. Hence
            // mesh.pos is fresh here. (A single stroke can't exceed the ring cap on any
            // real mesh, so the overflow branch of reserve is unreachable in practice.)
            static std::vector<ReadRun> runs;
            coalesce_snap_runs(snap_list, runs);
            static std::vector<float> pos_readback;
            glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo_pos.handle);
            if (stroke_writes_to_base) {
                for (const ReadRun& r : runs) {
                    pos_readback.resize(r.count * 3);
                    glGetBufferSubData(GL_ARRAY_BUFFER,
                        (GLintptr)r.first * 3 * sizeof(float),
                        (GLsizeiptr)r.count * 3 * sizeof(float),
                        pos_readback.data());
                    for (uint32_t v = r.first; v < r.first + r.count; v++) {
                        uint32_t ri = (v - r.first) * 3;
                        mesh.pos_x[v] = pos_readback[ri + 0];
                        mesh.pos_y[v] = pos_readback[ri + 1];
                        mesh.pos_z[v] = pos_readback[ri + 2];
                        multires.base.pos_x[v] = mesh.pos_x[v];
                        multires.base.pos_y[v] = mesh.pos_y[v];
                        multires.base.pos_z[v] = mesh.pos_z[v];
                    }
                }
            } else {
                const auto& frames = multires.frames[stroke_disp_index];
                auto& disp = multires.disp[stroke_disp_index];
                for (const ReadRun& r : runs) {
                    pos_readback.resize(r.count * 3);
                    glGetBufferSubData(GL_ARRAY_BUFFER,
                        (GLintptr)r.first * 3 * sizeof(float),
                        (GLsizeiptr)r.count * 3 * sizeof(float),
                        pos_readback.data());
                    for (uint32_t v = r.first; v < r.first + r.count; v++) {
                        uint32_t ri = (v - r.first) * 3;
                        float new_x = pos_readback[ri + 0];
                        float new_y = pos_readback[ri + 1];
                        float new_z = pos_readback[ri + 2];
                        float dx = new_x - mesh.pos_x[v];
                        float dy = new_y - mesh.pos_y[v];
                        float dz = new_z - mesh.pos_z[v];
                        const Frame& f = frames[v];
                        disp[v].x += f.t.x*dx + f.t.y*dy + f.t.z*dz;
                        disp[v].y += f.b.x*dx + f.b.y*dy + f.b.z*dz;
                        disp[v].z += f.n.x*dx + f.n.y*dy + f.n.z*dz;
                        mesh.pos_x[v] = new_x;
                        mesh.pos_y[v] = new_y;
                        mesh.pos_z[v] = new_z;
                    }
                }
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        } else {
            // mesh.pos + disp/base now hold stale (pen-down) values; the ring holds
            // (old,new) and the VBO/SSBO hold the new state. Defer the sync.
            mgpu.mark_cpu_dirty(snap_list);
#ifdef CHISEL_DEBUG_MULTIRES
            // Truth-check build: exercise the real flip, then immediately materialize
            // through the same choke a CPU reader would use, so the diff/capture
            // err-compares below read freshly-materialized mesh.pos/disp (validates the
            // materialize indexing + the ring capture together). Release stays lazy.
            mgpu.materialize_cpu(multires, mesh, renderer.vbo_pos.handle);
#endif
        }
        gpu_positions_deferred = false;

#ifdef CHISEL_DEBUG_MULTIRES
        // Phase 2b validation: GPU diff (snapshotted above) vs the CPU result we
        // just computed. Both should agree to float noise. Runs before the
        // upload_disp_partial re-sync below clobbers the GPU diff with CPU truth.
        if (gpu_diff_ran) {
            double maxe = 0.0;
            if (stroke_writes_to_base) {
                for (size_t i = 0; i < snap_list.size(); i++) {
                    uint32_t v = snap_list[i];
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+0] - multires.base.pos_x[v]));
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+1] - multires.base.pos_y[v]));
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+2] - multires.base.pos_z[v]));
                }
            } else if (stroke_disp_index >= 0
                       && stroke_disp_index < (int)multires.disp.size()) {
                const auto& disp = multires.disp[stroke_disp_index];
                for (size_t i = 0; i < snap_list.size(); i++) {
                    uint32_t v = snap_list[i];
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+0] - disp[v].x));
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+1] - disp[v].y));
                    maxe = std::max(maxe, (double)std::fabs(gpu_diff_chk[i*3+2] - disp[v].z));
                }
            }
            std::printf("[mgpu][debug] multires_diff L%d %s max|err|=%.3e (%zu verts)\n",
                        stroke_level, stroke_writes_to_base ? "base" : "disp",
                        maxe, snap_list.size());
        }

        // Phase 3b-iii validation: the (old,new) the diff shader wrote into the undo
        // ring vs the CPU truth — old should match the pen-down snapshot (snap_*),
        // new should match the post-readback CPU disp/pos. Runs after the CPU
        // readback above has materialized the new values. Confirms the GPU ring
        // capture is faithful before 3b-iv trusts it (and drops the CPU readback).
        if (gpu_diff_ran && ring_base != SIZE_MAX) {
            static std::vector<float> ring_chk;   // 6 floats/vert: old(3) + new(3)
            ring_chk.resize(snap_list.size() * 6);
            compute->undo_ring_read(ring_base * sizeof(float),
                                    snap_list.size() * 6, ring_chk.data());
            double maxe_old = 0.0, maxe_new = 0.0;
            for (size_t i = 0; i < snap_list.size(); i++) {
                uint32_t v = snap_list[i];
                maxe_old = std::max(maxe_old, (double)std::fabs(ring_chk[i*6+0] - snap_x[v]));
                maxe_old = std::max(maxe_old, (double)std::fabs(ring_chk[i*6+1] - snap_y[v]));
                maxe_old = std::max(maxe_old, (double)std::fabs(ring_chk[i*6+2] - snap_z[v]));
                float nx, ny, nz;
                if (stroke_writes_to_base) {
                    nx = mesh.pos_x[v]; ny = mesh.pos_y[v]; nz = mesh.pos_z[v];
                } else {
                    nx = multires.disp[stroke_disp_index][v].x;
                    ny = multires.disp[stroke_disp_index][v].y;
                    nz = multires.disp[stroke_disp_index][v].z;
                }
                maxe_new = std::max(maxe_new, (double)std::fabs(ring_chk[i*6+3] - nx));
                maxe_new = std::max(maxe_new, (double)std::fabs(ring_chk[i*6+4] - ny));
                maxe_new = std::max(maxe_new, (double)std::fabs(ring_chk[i*6+5] - nz));
            }
            std::printf("[undo-ring][debug] capture L%d %s old|err|=%.3e new|err|=%.3e (%zu verts)\n",
                        stroke_level, stroke_writes_to_base ? "base" : "disp",
                        maxe_old, maxe_new, snap_list.size());
        }
#endif

        // Phase 1: keep the GPU residency mirror in sync with the disp/base edit we
        // just wrote to CPU storage. Active entity is offset 0, so vert index == SSBO
        // index. In the flipped ring path the diff shader already wrote disp_ssbo/
        // base_ssbo and CPU storage is stale, so re-uploading would clobber the GPU
        // truth — skip it (the SSBO is already current there).
        if (!ring_captured)
            mgpu.upload_disp_partial(multires, stroke_level, snap_list);
    }

    if (gpu_mask_deferred && !mask.snap_list.empty()) {
        if (mesh.mask.empty()) mesh.mask.assign(mesh.vertex_count(), 0.0f);
        static std::vector<ReadRun> runs;
        coalesce_snap_runs(mask.snap_list, runs);
        static std::vector<float> mask_readback;
        glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo_mask.handle);
        for (const ReadRun& r : runs) {
            mask_readback.resize(r.count);
            glGetBufferSubData(GL_ARRAY_BUFFER,
                (GLintptr)r.first * sizeof(float),
                (GLsizeiptr)r.count * sizeof(float),
                mask_readback.data());
            for (uint32_t v = r.first; v < r.first + r.count; v++)
                mesh.mask[v] = mask_readback[v - r.first];
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        gpu_mask_deferred = false;
    }

    if (gpu_color_deferred && !color.snap_list.empty()) {
        if (mesh.color.empty()) mesh.color.assign(mesh.vertex_count(), 0xFFFFFFFFu);
        else if (mesh.color.size() < mesh.vertex_count())
            mesh.color.resize(mesh.vertex_count(), 0xFFFFFFFFu);
        static std::vector<ReadRun> runs;
        coalesce_snap_runs(color.snap_list, runs);
        static std::vector<uint32_t> color_readback;
        glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo_color.handle);
        for (const ReadRun& r : runs) {
            color_readback.resize(r.count);
            glGetBufferSubData(GL_ARRAY_BUFFER,
                (GLintptr)r.first * sizeof(uint32_t),
                (GLsizeiptr)r.count * sizeof(uint32_t),
                color_readback.data());
            for (uint32_t v = r.first; v < r.first + r.count; v++)
                mesh.color[v] = color_readback[v - r.first];
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        gpu_color_deferred = false;
    }

    if (brush_type == BrushType::MASK) {
        commit_mask_undo(mesh, stack);
    } else if (brush_type == BrushType::PAINT) {
        commit_color_undo(mesh, stack);
    } else {
        for (int fi = stroke_disp_index + 1; fi < (int)multires.frames.size(); fi++)
            multires.frames[fi].clear();
        commit_undo(mesh, stack, multires);
    }

    if (gpu_normals_deferred) {
        if (!snap_list.empty()) {
            static std::vector<ReadRun> runs;
            coalesce_snap_runs(snap_list, runs);
            static std::vector<float> norm_readback;
            glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo_norm.handle);
            for (const ReadRun& r : runs) {
                norm_readback.resize(r.count * 3);
                glGetBufferSubData(GL_ARRAY_BUFFER,
                    (GLintptr)r.first * 3 * sizeof(float),
                    (GLsizeiptr)r.count * 3 * sizeof(float),
                    norm_readback.data());
                for (uint32_t v = r.first; v < r.first + r.count; v++) {
                    uint32_t ri = (v - r.first) * 3;
                    mesh.norm_x[v] = norm_readback[ri + 0];
                    mesh.norm_y[v] = norm_readback[ri + 1];
                    mesh.norm_z[v] = norm_readback[ri + 2];
                }
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        gpu_normals_deferred = false;
    }

    bool had_mesh_update = needs_mesh_update;
    end();
    return had_mesh_update;
}

void BrushStroke::end() {
    phase = StrokePhase::NONE;
    needs_mesh_update = false;
    cached_triid.clear();
    cached_bary.clear();
    cached_w = 0;
    cached_h = 0;
    disp_x.clear();
    disp_y.clear();
    disp_z.clear();
    disp_weight.clear();
    dirty_verts.clear();
    gpu_dirty.clear();
    move.clear();
    snap_flag.clear();
    snap_x.clear();
    snap_y.clear();
    snap_z.clear();
    snap_list.clear();
    mask.clear();
    color.clear();
}
