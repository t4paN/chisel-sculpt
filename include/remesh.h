#pragma once
#include <string>
#include <cstdint>
#include <functional>
#include <vector>
#include "mesh.h"
#include "multires_stack.h"

struct ComputeState;  // optional GPU acceleration

struct RemeshResult {
    bool success = false;
    std::string error;
    double elapsed_ms = 0.0;
    uint32_t old_verts = 0, old_tris = 0;
    uint32_t new_verts = 0, new_tris = 0;
    uint32_t selected_tris = 0;
};

RemeshResult perform_remesh(Mesh& mesh, MultiresStack& stack,
                            float target_edge_length = 0.0f,
                            int iterations = 10,
                            ComputeState* cs = nullptr,
                            float density_coarse_mult = 2.0f,
                            float density_fine_mult = 0.5f,
                            bool keep_detail = true,   // snap to the original surface + guard flips
                            float detail = 1.0f);      // tri-count multiplier on the auto target

// Predicted post-remesh triangle count when a painted density field drives
// adaptive sizing (each tri refines toward its local target edge length).
// Used by the caller to refuse a remesh that would exceed GPU buffer limits,
// same as the subdivision guard. Ignores collapses (conservative).
uint64_t predict_adaptive_tris(const Mesh& mesh, float target_edge_length,
                               float coarse_mult, float fine_mult);

// SDF merge: pull an extracted surface back onto the flat source soup (xyz per
// vertex, 3 indices per tri) along each vertex normal, at most max_step. `accept`
// vetoes hits that aren't on the result's real skin (buried halves at a join).
// seam_tol > 0 = mirror +x half: verts under it are left alone, none move below
// it. snap=false only measures. Prints the [sdf] DRIFT line either way.
uint32_t snap_mesh_to_soup(Mesh& m, const std::vector<float>& pos,
                           const std::vector<uint32_t>& idx,
                           float cell, float max_step, bool snap,
                           const std::vector<uint8_t>* pinned, float seam_tol,
                           const std::function<bool(Vec3, Vec3)>& accept);
