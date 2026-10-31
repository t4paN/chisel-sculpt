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

// The remesh runs as a job across frames, one step per frame, on a private copy
// of `mesh`; the last step writes the result back into `mesh` and resets `stack`.
// Both must outlive the job. Tick once a frame until it returns true; `out` then
// holds the result. Destroying an unfinished job leaves `mesh` as it was.
struct RemeshJob;
RemeshJob* remesh_begin(Mesh& mesh, MultiresStack& stack,
                        float target_edge_length = 0.0f,
                        int iterations = 10,
                        ComputeState* cs = nullptr,
                        float density_coarse_mult = 2.0f,
                        float density_fine_mult = 0.5f,
                        bool keep_detail = true,   // snap to the original surface + guard flips
                        float detail = 1.0f);      // tri-count multiplier on the auto target
bool  remesh_tick(RemeshJob& job, RemeshResult& out);
float remesh_progress(const RemeshJob& job);       // 0..1
void  remesh_destroy(RemeshJob* job);

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

// Textured meshes: give dst (positions final) per-corner UVs from the nearest
// source surface (src_pos xyz per vertex, src_idx 3 per tri, src_uv 6 per tri).
// Each dst tri reads all three corners off ONE source UV island, so a tri that
// straddles a seam never smears across the atlas. `cell` = how far dst may sit
// from the source (one remesh edge). Replaces dst.uv; leaves dst.texture alone.
// skip (optional): source tris to avoid while anything else is in reach (the SDF
// merge's buried halves). tri_island (optional): per dst tri, the source island
// it read from, named by one of its tris (UINT32_MAX = none in reach).
void transfer_uv(Mesh& dst, const std::vector<float>& src_pos,
                 const std::vector<uint32_t>& src_idx,
                 const std::vector<float>& src_uv, float cell,
                 const std::vector<uint8_t>* skip = nullptr,
                 std::vector<uint32_t>* tri_island = nullptr);

// SDF merge paint carry: for each vertex of m (normals current), the soup vertex
// to take paint from — nearest corner of the closest same-facing soup tri that
// `accept` puts on the skin. UINT32_MAX where none lies within `cell`.
void pick_soup_sources(const Mesh& m, const std::vector<float>& pos,
                       const std::vector<uint32_t>& idx, float cell,
                       const std::function<bool(Vec3, Vec3)>& accept,
                       std::vector<uint32_t>& pick);
