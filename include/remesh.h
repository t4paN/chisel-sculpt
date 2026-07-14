#pragma once
#include <string>
#include <cstdint>
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
                            float density_fine_mult = 0.5f);

// Predicted post-remesh triangle count when a painted density field drives
// adaptive sizing (each tri refines toward its local target edge length).
// Used by the caller to refuse a remesh that would exceed GPU buffer limits,
// same as the subdivision guard. Ignores collapses (conservative).
uint64_t predict_adaptive_tris(const Mesh& mesh, float target_edge_length,
                               float coarse_mult, float fine_mult);
