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
VoxelMergeResult voxel_merge_selected(Scene& scene, ComputeState& cs,
                                      int resolution, bool mirror = false);
