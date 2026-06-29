#pragma once
#include <cstdint>
#include <string>
#include "mesh.h"   // Vec3

class Scene;
struct ComputeState;

// SDF voxel-merge: join an arbitrary selection of mesh entities into ONE
// watertight manifold for export / 3D printing. GPU-native (three compute
// dispatches, flat SSBOs); the only CPU touch is a hash-weld on the final
// readback. Robust to non-watertight input via the generalized winding number.
// See sdf-remesh-spec.md for the full derivation.

// Marching-cubes corner grid spanning the soup AABB (with padding).
// Cell (i,j,k) corner index = i + (R+1)*(j + (R+1)*k).
struct SdfGrid {
    Vec3     origin;        // world pos of corner (0,0,0)
    float    voxel = 0.0f;  // edge length of one cell
    uint32_t R = 0;         // cells per axis (corners = R+1)
    uint32_t corners() const { return (R+1)*(R+1)*(R+1); }
    uint32_t cells()   const { return R*R*R; }
};

struct VoxelMergeResult {
    bool        success = false;
    std::string error;
    double      elapsed_ms = 0.0;
    uint32_t    in_entities = 0, in_tris = 0, out_verts = 0, out_tris = 0;
    uint32_t    R = 0;
    // Manifold report (chunk 4): surfaced so a bad merge is visible before print.
    uint32_t    boundary_edges = 0, nonmanifold_edges = 0, components = 0;
};

// Merge the current selection (scene.selected_ids()) into one watertight mesh,
// splicing the welded result back into the scene as the merged entity (it
// consumes the selection; unselected entities are preserved). Caller is
// responsible for the post-merge mirror refresh + scene.sync().
// `resolution` = target cells along the longest AABB axis (clamped to [16,256]).
// `mirror` = extract only the +x half and reflect it across x=0 (swapped winding)
// so the result is tessellation-symmetric about the app's mirror plane — exact
// vertex partner map, mirror-editable. Default (false) keeps the faithful,
// possibly-asymmetric union (correct for a pure print export).
//
// Synchronous convenience: drives the tick-based job below to completion in one
// call (blocks for the whole merge). Still used by any caller that wants the old
// one-shot behaviour; the interactive path uses the begin/tick split instead.
// `surface_nets` = extract with Surface Nets (smoother, more uniform, quad-dominant,
// fewer slivers) instead of the default Marching Cubes. MC stays the default because
// the naive Surface Nets vertex-per-cell can be non-manifold on thin/ambiguous cells.
// Works with mirror too: rather than MC's keep-+x-and-reflect seam (SN has no vertex
// on the plane), the SN mirror path symmetrises the field about x=0 and extracts the
// whole surface, yielding exact mirror-paired verts continuous through the plane.
// `subtract` = treat every UNSELECTED (red) committed entity as a cutter: the
// merge unions the selection, then carves those cutters out (boolean A − B),
// achieved by feeding the cutter triangles into the soup with reversed winding.
// Cutters are not consumed (only the selection is spliced out for the result).
VoxelMergeResult voxel_merge_selected(Scene& scene, ComputeState& cs,
                                      int resolution, bool mirror = false,
                                      bool surface_nets = false,
                                      bool subtract = false);

// ---- Tick-driven (multi-frame) merge --------------------------------------
// The merge is a multi-second GPU job (the winding-sign pass alone is seconds at
// R≥128). Running it in one frame freezes the window and, unsliced, trips the GPU
// watchdog. So it is structured as an enqueue (`begin`) + per-frame advance
// (`tick`) job: the dominant winding-sign pass is budgeted across frames so the
// window keeps pumping events and the progress HUD animates. This split is also
// the structural prerequisite for the WebGPU port, where the readbacks become
// async `mapAsync` promises and the merge MUST span frames regardless.
struct VoxelMergeJob;   // opaque; defined in sdf.cpp (owns the cross-frame GL state)

enum class VoxelMergeStatus { Working, Done, Failed };

// Gather the soup, build the grid, allocate SSBOs and compile the programs, then
// return a heap-owned job (never null). Setup failures don't throw — the job's
// first `tick` reports Failed with the error in the result, so all error handling
// funnels through one path. Caller owns the job; free it with voxel_merge_destroy.
VoxelMergeJob* voxel_merge_begin(Scene& scene, ComputeState& cs,
                                 int resolution, bool mirror, bool surface_nets,
                                 bool subtract = false);

// Advance one budgeted step. Returns Working until the job completes; on the final
// step it runs the CPU tail (weld → relax → mirror seam → manifold gate → scene
// splice) and fills `out`. Returns Done (out.success true) or Failed (out.error).
// On Done/Failed the caller should run its post-merge refresh and then destroy the
// job. GL state is fully re-established each tick (the render loop clobbers it).
VoxelMergeStatus voxel_merge_tick(Scene& scene, ComputeState& cs,
                                  VoxelMergeJob& job, VoxelMergeResult& out);

// Progress in [0,1] for the HUD.
float voxel_merge_progress(const VoxelMergeJob& job);

void voxel_merge_destroy(VoxelMergeJob* job);
