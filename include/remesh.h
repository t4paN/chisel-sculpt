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
                            ComputeState* cs = nullptr);
