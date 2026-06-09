#include "mesh.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cassert>

void Mesh::recompute_normals() {
    uint32_t vc = vertex_count();
    uint32_t tc = tri_count();

    // Zero normals
    std::memset(norm_x.data(), 0, vc * sizeof(float));
    std::memset(norm_y.data(), 0, vc * sizeof(float));
    std::memset(norm_z.data(), 0, vc * sizeof(float));

    // Accumulate face normals (area-weighted by cross product magnitude)
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = indices[t*3+0];
        uint32_t i1 = indices[t*3+1];
        uint32_t i2 = indices[t*3+2];

        Vec3 v0 = get_pos(i0);
        Vec3 v1 = get_pos(i1);
        Vec3 v2 = get_pos(i2);

        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 fn = e1.cross(e2); // not normalized = area weighted

        norm_x[i0] += fn.x; norm_y[i0] += fn.y; norm_z[i0] += fn.z;
        norm_x[i1] += fn.x; norm_y[i1] += fn.y; norm_z[i1] += fn.z;
        norm_x[i2] += fn.x; norm_y[i2] += fn.y; norm_z[i2] += fn.z;
    }

    // Normalize
    for (uint32_t i = 0; i < vc; i++) {
        Vec3 n = {norm_x[i], norm_y[i], norm_z[i]};
        n = n.normalized();
        norm_x[i] = n.x;
        norm_y[i] = n.y;
        norm_z[i] = n.z;
    }
}

void Mesh::compute_bounding_sphere(Vec3& center, float& radius) const {
    uint32_t vc = vertex_count();
    if (vc == 0) { center = {0,0,0}; radius = 1.0f; return; }

    // Compute centroid
    float cx = 0, cy = 0, cz = 0;
    for (uint32_t i = 0; i < vc; i++) {
        cx += pos_x[i]; cy += pos_y[i]; cz += pos_z[i];
    }
    float inv = 1.0f / vc;
    center = {cx*inv, cy*inv, cz*inv};

    // Find max distance from center
    radius = 0;
    for (uint32_t i = 0; i < vc; i++) {
        Vec3 d = get_pos(i) - center;
        float dist = d.length();
        if (dist > radius) radius = dist;
    }
}

void build_mirror_spatial(const Mesh& m, std::vector<uint32_t>& out) {
    uint32_t vc = m.vertex_count();
    out.resize(vc);
    for (uint32_t i = 0; i < vc; i++) out[i] = i;

    // Adaptive tolerance from mean edge length
    float edge_sum = 0.0f;
    uint32_t edge_count = 0;
    uint32_t tc = m.tri_count();
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t i0 = m.indices[t*3+0], i1 = m.indices[t*3+1], i2 = m.indices[t*3+2];
        float dx, dy, dz;
        dx = m.pos_x[i1]-m.pos_x[i0]; dy = m.pos_y[i1]-m.pos_y[i0]; dz = m.pos_z[i1]-m.pos_z[i0];
        edge_sum += std::sqrt(dx*dx+dy*dy+dz*dz);
        dx = m.pos_x[i2]-m.pos_x[i1]; dy = m.pos_y[i2]-m.pos_y[i1]; dz = m.pos_z[i2]-m.pos_z[i1];
        edge_sum += std::sqrt(dx*dx+dy*dy+dz*dz);
        dx = m.pos_x[i0]-m.pos_x[i2]; dy = m.pos_y[i0]-m.pos_y[i2]; dz = m.pos_z[i0]-m.pos_z[i2];
        edge_sum += std::sqrt(dx*dx+dy*dy+dz*dz);
        edge_count += 3;
    }
    float mean_edge = (edge_count > 0) ? edge_sum / (float)edge_count : 1e-3f;
    float tol = mean_edge * 0.5f;
    float tol_sq = tol * tol;
    float seam_tol = std::max(1e-5f, mean_edge * 0.01f);

    // Split vertices into +x, -x, and seam buckets
    std::vector<uint32_t> pos_side, neg_side;
    pos_side.reserve(vc / 2);
    neg_side.reserve(vc / 2);
    for (uint32_t i = 0; i < vc; i++) {
        if (std::fabs(m.pos_x[i]) < seam_tol) { out[i] = i; continue; }
        if (m.pos_x[i] > 0.0f) pos_side.push_back(i);
        else neg_side.push_back(i);
    }

    // Spatial grid for fast nearest-neighbor search. Grid cell size = 2*tol
    // covers search radius with some overlap. Vertices binned by (y, z) since
    // we're matching symmetrically across x.
    float grid_size = 2.0f * tol;
    if (grid_size < 1e-6f) grid_size = 1e-3f;
    float inv_grid = 1.0f / grid_size;

    // Build grid for neg_side
    std::unordered_map<uint64_t, std::vector<uint32_t>> neg_grid;
    auto grid_key = [](int gy, int gz) -> uint64_t {
        return ((uint64_t)(uint32_t)gy << 32) | (uint32_t)gz;
    };
    for (uint32_t ni = 0; ni < neg_side.size(); ni++) {
        uint32_t j = neg_side[ni];
        int gy = (int)std::floor(m.pos_y[j] * inv_grid);
        int gz = (int)std::floor(m.pos_z[j] * inv_grid);
        neg_grid[grid_key(gy, gz)].push_back(j);
    }

    // For each +x vertex, find nearest -x match via grid
    std::vector<uint32_t> pos_best(pos_side.size());
    for (uint32_t pi = 0; pi < pos_side.size(); pi++) {
        uint32_t i = pos_side[pi];
        float mx = -m.pos_x[i], my = m.pos_y[i], mz = m.pos_z[i];
        int gy = (int)std::floor(my * inv_grid);
        int gz = (int)std::floor(mz * inv_grid);

        float best_sq = tol_sq;
        uint32_t best = i;

        // Search 3x3x3 grid neighborhood
        for (int dgy = -1; dgy <= 1; dgy++) {
            for (int dgz = -1; dgz <= 1; dgz++) {
                auto it = neg_grid.find(grid_key(gy + dgy, gz + dgz));
                if (it == neg_grid.end()) continue;
                for (uint32_t j : it->second) {
                    float dx = m.pos_x[j]-mx, dy = m.pos_y[j]-my, dz = m.pos_z[j]-mz;
                    float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < best_sq) { best_sq = d2; best = j; }
                }
            }
        }
        pos_best[pi] = best;
    }

    // Build grid for pos_side
    std::unordered_map<uint64_t, std::vector<uint32_t>> pos_grid;
    for (uint32_t pi = 0; pi < pos_side.size(); pi++) {
        uint32_t i = pos_side[pi];
        int gy = (int)std::floor(m.pos_y[i] * inv_grid);
        int gz = (int)std::floor(m.pos_z[i] * inv_grid);
        pos_grid[grid_key(gy, gz)].push_back(i);
    }

    // For each -x vertex, find nearest +x match via grid
    std::vector<uint32_t> neg_best(neg_side.size());
    for (uint32_t ni = 0; ni < neg_side.size(); ni++) {
        uint32_t j = neg_side[ni];
        float mx = -m.pos_x[j], my = m.pos_y[j], mz = m.pos_z[j];
        int gy = (int)std::floor(my * inv_grid);
        int gz = (int)std::floor(mz * inv_grid);

        float best_sq = tol_sq;
        uint32_t best = j;

        // Search 3x3x3 grid neighborhood
        for (int dgy = -1; dgy <= 1; dgy++) {
            for (int dgz = -1; dgz <= 1; dgz++) {
                auto it = pos_grid.find(grid_key(gy + dgy, gz + dgz));
                if (it == pos_grid.end()) continue;
                for (uint32_t i : it->second) {
                    float dx = m.pos_x[i]-mx, dy = m.pos_y[i]-my, dz = m.pos_z[i]-mz;
                    float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < best_sq) { best_sq = d2; best = i; }
                }
            }
        }
        neg_best[ni] = best;
    }

    // Only commit mutual pairs: A→B and B→A
    // Build reverse lookup: neg vertex index → its position in neg_side
    std::unordered_map<uint32_t, uint32_t> neg_idx_map;
    neg_idx_map.reserve(neg_side.size());
    for (uint32_t ni = 0; ni < neg_side.size(); ni++)
        neg_idx_map[neg_side[ni]] = ni;

    for (uint32_t pi = 0; pi < pos_side.size(); pi++) {
        uint32_t i = pos_side[pi];
        uint32_t j = pos_best[pi];
        if (j == i) continue;
        auto it = neg_idx_map.find(j);
        if (it == neg_idx_map.end()) continue;
        if (neg_best[it->second] == i) {
            out[i] = j;
            out[j] = i;
        }
    }

    uint32_t paired = 0, seam = 0, unpaired = 0;
    for (uint32_t i = 0; i < vc; i++) {
        if (out[i] == i) {
            if (std::fabs(m.pos_x[i]) < seam_tol) seam++;
            else unpaired++;
        } else paired++;
    }
    std::printf("[mirror] spatial rebuild: %u paired, %u seam, %u unpaired (tol=%.4f, edge=%.4f, "
                "+x=%zu, -x=%zu)\n",
                paired, seam, unpaired, tol, mean_edge,
                pos_side.size(), neg_side.size());
}

void Mesh::build_mirror_x_map() {
    build_mirror_spatial(*this, mirror_x_map);
}

void Mesh::build_adjacency() {
    uint32_t vc = vertex_count();
    uint32_t tc = tri_count();

    // Count triangles per vertex
    vert_tri_offset.assign(vc + 1, 0);
    for (uint32_t t = 0; t < tc; t++) {
        vert_tri_offset[indices[t*3+0] + 1]++;
        vert_tri_offset[indices[t*3+1] + 1]++;
        vert_tri_offset[indices[t*3+2] + 1]++;
    }

    // Prefix sum to get offsets
    for (uint32_t i = 1; i <= vc; i++) {
        vert_tri_offset[i] += vert_tri_offset[i-1];
    }

    // Fill adjacency list
    vert_tri_list.resize(vert_tri_offset[vc]);
    std::vector<uint32_t> cursor(vc, 0);
    for (uint32_t t = 0; t < tc; t++) {
        for (int k = 0; k < 3; k++) {
            uint32_t v = indices[t*3+k];
            vert_tri_list[vert_tri_offset[v] + cursor[v]] = t;
            cursor[v]++;
        }
    }
}

void Mesh::expand_dirty_to_affected(const std::vector<uint32_t>& dirty_verts,
                                     std::vector<uint32_t>& affected_out) {
    if (vert_tri_offset.empty()) return;

    uint32_t vc = vertex_count();
    uint32_t tc = tri_count();

    static std::vector<bool> tri_flag;
    static std::vector<bool> vert_flag;
    static std::vector<uint32_t> affected_tri_list;

    if (tri_flag.size() < tc) tri_flag.resize(tc, false);
    if (vert_flag.size() < vc) vert_flag.resize(vc, false);
    affected_tri_list.clear();
    affected_out.clear();

    for (uint32_t v : dirty_verts) {
        if (v >= vc) {
            std::printf("[CRASH-GUARD] expand_dirty v=%u >= vc=%u\n", v, vc);
            continue;
        }
        uint32_t start = vert_tri_offset[v];
        uint32_t end = vert_tri_offset[v + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t t = vert_tri_list[j];
            if (!tri_flag[t]) {
                tri_flag[t] = true;
                affected_tri_list.push_back(t);
            }
        }
    }

    for (uint32_t t : affected_tri_list) {
        uint32_t i0 = indices[t*3+0];
        uint32_t i1 = indices[t*3+1];
        uint32_t i2 = indices[t*3+2];
        if (!vert_flag[i0]) { vert_flag[i0] = true; affected_out.push_back(i0); }
        if (!vert_flag[i1]) { vert_flag[i1] = true; affected_out.push_back(i1); }
        if (!vert_flag[i2]) { vert_flag[i2] = true; affected_out.push_back(i2); }
    }

    for (uint32_t t : affected_tri_list) tri_flag[t] = false;
    for (uint32_t v : affected_out) vert_flag[v] = false;
}

void Mesh::recompute_normals_partial(const std::vector<uint32_t>& dirty_verts,
                                     std::vector<uint32_t>* affected_verts_out) {
    if (vert_tri_offset.empty()) {
        // Fallback if adjacency not built
        recompute_normals();
        return;
    }

    // Collect all triangles touching any dirty vertex, and all vertices of those triangles
    // We need to recompute normals for all vertices of affected triangles,
    // not just the dirty ones, because a face normal change affects all 3 verts

    uint32_t vc = vertex_count();
    uint32_t tc = tri_count();

    // Static flat flag arrays — persist across calls, no heap churn
    static std::vector<bool> tri_flag;
    static std::vector<bool> vert_flag;
    static std::vector<bool> recompute_tri_flag;
    static std::vector<uint32_t> affected_tri_list;
    static std::vector<uint32_t> affected_vert_list;
    static std::vector<uint32_t> recompute_tri_list;

    if (tri_flag.size() < tc) tri_flag.resize(tc, false);
    if (vert_flag.size() < vc) vert_flag.resize(vc, false);
    if (recompute_tri_flag.size() < tc) recompute_tri_flag.resize(tc, false);
    affected_tri_list.clear();
    affected_vert_list.clear();
    recompute_tri_list.clear();

    // Collect all triangles touching any dirty vertex
    for (uint32_t v : dirty_verts) {
        uint32_t start = vert_tri_offset[v];
        uint32_t end = vert_tri_offset[v + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t t = vert_tri_list[j];
            if (!tri_flag[t]) {
                tri_flag[t] = true;
                affected_tri_list.push_back(t);
            }
        }
    }

    // Collect all vertices of affected triangles
    for (uint32_t t : affected_tri_list) {
        uint32_t i0 = indices[t*3+0];
        uint32_t i1 = indices[t*3+1];
        uint32_t i2 = indices[t*3+2];
        if (!vert_flag[i0]) { vert_flag[i0] = true; affected_vert_list.push_back(i0); }
        if (!vert_flag[i1]) { vert_flag[i1] = true; affected_vert_list.push_back(i1); }
        if (!vert_flag[i2]) { vert_flag[i2] = true; affected_vert_list.push_back(i2); }
    }

    // Zero normals for affected verts
    for (uint32_t v : affected_vert_list) {
        norm_x[v] = 0.0f;
        norm_y[v] = 0.0f;
        norm_z[v] = 0.0f;
    }

    // Collect all triangles touching affected verts (superset for boundary correctness)
    for (uint32_t v : affected_vert_list) {
        uint32_t start = vert_tri_offset[v];
        uint32_t end = vert_tri_offset[v + 1];
        for (uint32_t j = start; j < end; j++) {
            uint32_t t = vert_tri_list[j];
            if (!recompute_tri_flag[t]) {
                recompute_tri_flag[t] = true;
                recompute_tri_list.push_back(t);
            }
        }
    }

    for (uint32_t t : recompute_tri_list) {
        uint32_t i0 = indices[t*3+0];
        uint32_t i1 = indices[t*3+1];
        uint32_t i2 = indices[t*3+2];

        Vec3 v0 = get_pos(i0);
        Vec3 v1 = get_pos(i1);
        Vec3 v2 = get_pos(i2);

        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 fn = e1.cross(e2);

        if (vert_flag[i0]) {
            norm_x[i0] += fn.x; norm_y[i0] += fn.y; norm_z[i0] += fn.z;
        }
        if (vert_flag[i1]) {
            norm_x[i1] += fn.x; norm_y[i1] += fn.y; norm_z[i1] += fn.z;
        }
        if (vert_flag[i2]) {
            norm_x[i2] += fn.x; norm_y[i2] += fn.y; norm_z[i2] += fn.z;
        }
    }

    // Normalize
    for (uint32_t v : affected_vert_list) {
        Vec3 n = {norm_x[v], norm_y[v], norm_z[v]};
        n = n.normalized();
        norm_x[v] = n.x;
        norm_y[v] = n.y;
        norm_z[v] = n.z;
    }

    // Output
    if (affected_verts_out) {
        *affected_verts_out = affected_vert_list;
    }

    // Clean up flags (only touch entries we set)
    for (uint32_t t : affected_tri_list) tri_flag[t] = false;
    for (uint32_t v : affected_vert_list) vert_flag[v] = false;
    for (uint32_t t : recompute_tri_list) recompute_tri_flag[t] = false;
}

bool Mesh::export_obj(const char* filename) const {
    FILE* f = std::fopen(filename, "w");
    if (!f) return false;

    std::fprintf(f, "# OBJ export from Chisel\n");
    std::fprintf(f, "# vertices: %u\n", vertex_count());
    std::fprintf(f, "# triangles: %u\n", tri_count());

    for (uint32_t i = 0; i < vertex_count(); i++) {
        std::fprintf(f, "v %f %f %f\n", pos_x[i], pos_y[i], pos_z[i]);
    }

    for (uint32_t i = 0; i < vertex_count(); i++) {
        std::fprintf(f, "vn %f %f %f\n", norm_x[i], norm_y[i], norm_z[i]);
    }

    for (uint32_t t = 0; t < tri_count(); t++) {
        uint32_t i0 = indices[t*3+0];
        uint32_t i1 = indices[t*3+1];
        uint32_t i2 = indices[t*3+2];
        std::fprintf(f, "f %u//%u %u//%u %u//%u\n",
                    i0+1, i0+1,
                    i1+1, i1+1,
                    i2+1, i2+1);
    }

    std::fclose(f);
    return true;
}

bool Mesh::export_stl(const char* filename, float scale) const {
    FILE* f = std::fopen(filename, "wb");
    if (!f) return false;

    // 80-byte header (must NOT begin with "solid" or parsers read it as ASCII).
    char header[80];
    std::memset(header, 0, sizeof(header));
    std::snprintf(header, sizeof(header), "Chisel binary STL");
    std::fwrite(header, 1, sizeof(header), f);

    uint32_t nt = tri_count();
    std::fwrite(&nt, sizeof(uint32_t), 1, f);

    const uint16_t attr = 0;
    for (uint32_t t = 0; t < nt; t++) {
        uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
        Vec3 a = get_pos(i0) * scale;
        Vec3 b = get_pos(i1) * scale;
        Vec3 c = get_pos(i2) * scale;
        Vec3 n = (b - a).cross(c - a).normalized();   // per-tri facet normal
        float rec[12] = {
            n.x, n.y, n.z,
            a.x, a.y, a.z,
            b.x, b.y, b.z,
            c.x, c.y, c.z,
        };
        std::fwrite(rec, sizeof(float), 12, f);
        std::fwrite(&attr, sizeof(uint16_t), 1, f);
    }

    std::fclose(f);
    return true;
}

bool Mesh::export_ply(const char* filename) const {
    FILE* f = std::fopen(filename, "w");
    if (!f) return false;

    uint32_t nv = vertex_count(), nt = tri_count();
    bool have_color = (color.size() == nv);

    std::fprintf(f, "ply\n");
    std::fprintf(f, "format ascii 1.0\n");
    std::fprintf(f, "comment Chisel PLY export\n");
    std::fprintf(f, "element vertex %u\n", nv);
    std::fprintf(f, "property float x\n");
    std::fprintf(f, "property float y\n");
    std::fprintf(f, "property float z\n");
    std::fprintf(f, "property float nx\n");
    std::fprintf(f, "property float ny\n");
    std::fprintf(f, "property float nz\n");
    std::fprintf(f, "property uchar red\n");
    std::fprintf(f, "property uchar green\n");
    std::fprintf(f, "property uchar blue\n");
    std::fprintf(f, "element face %u\n", nt);
    std::fprintf(f, "property list uchar uint vertex_indices\n");
    std::fprintf(f, "end_header\n");

    for (uint32_t i = 0; i < nv; i++) {
        uint32_t c = have_color ? color[i] : 0xFFFFFFFFu;
        unsigned r = c & 0xFF, g = (c >> 8) & 0xFF, b = (c >> 16) & 0xFF;
        std::fprintf(f, "%f %f %f %f %f %f %u %u %u\n",
                     pos_x[i], pos_y[i], pos_z[i],
                     norm_x[i], norm_y[i], norm_z[i], r, g, b);
    }

    for (uint32_t t = 0; t < nt; t++) {
        std::fprintf(f, "3 %u %u %u\n",
                     indices[t*3+0], indices[t*3+1], indices[t*3+2]);
    }

    std::fclose(f);
    return true;
}

bool Mesh::import_obj(const char* filename, Mesh& out) {
    FILE* f = std::fopen(filename, "r");
    if (!f) return false;

    std::vector<float> vx, vy, vz;
    std::vector<float> nx, ny, nz;
    std::vector<uint32_t> idxs;

    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                vx.push_back(x); vy.push_back(y); vz.push_back(z);
            }
        } else if (line[0] == 'v' && line[1] == 'n' && line[2] == ' ') {
            float x, y, z;
            if (std::sscanf(line + 3, "%f %f %f", &x, &y, &z) == 3) {
                nx.push_back(x); ny.push_back(y); nz.push_back(z);
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            const char* p = line + 2;
            uint32_t face_verts[16];
            int count = 0;
            while (*p && *p != '\n' && *p != '\r' && count < 16) {
                while (*p == ' ') p++;
                if (!*p || *p == '\n' || *p == '\r') break;
                int vi = 0;
                while (*p >= '0' && *p <= '9') { vi = vi * 10 + (*p - '0'); p++; }
                face_verts[count++] = (uint32_t)(vi - 1);
                if (*p == '/') {
                    p++;
                    while (*p >= '0' && *p <= '9') p++;
                    if (*p == '/') {
                        p++;
                        while (*p >= '0' && *p <= '9') p++;
                    }
                }
            }
            for (int i = 1; i + 1 < count; i++) {
                idxs.push_back(face_verts[0]);
                idxs.push_back(face_verts[i]);
                idxs.push_back(face_verts[i + 1]);
            }
        }
    }
    std::fclose(f);

    if (vx.empty() || idxs.empty()) return false;

    out.pos_x = std::move(vx);
    out.pos_y = std::move(vy);
    out.pos_z = std::move(vz);
    out.indices = std::move(idxs);
    out.norm_x.clear(); out.norm_y.clear(); out.norm_z.clear();
    out.mirror_x_map.clear();
    out.vert_tri_offset.clear();
    out.vert_tri_list.clear();

    uint32_t nv = out.vertex_count();
    if (nx.size() == nv) {
        out.norm_x = std::move(nx);
        out.norm_y = std::move(ny);
        out.norm_z = std::move(nz);
    } else {
        out.norm_x.resize(nv, 0.0f);
        out.norm_y.resize(nv, 0.0f);
        out.norm_z.resize(nv, 1.0f);
        out.recompute_normals();
    }

    out.build_adjacency();

    std::printf("[import] loaded %u verts, %u tris from %s\n",
                out.vertex_count(), out.tri_count(), filename);
    return true;
}

bool Mesh::import_ply(const char* filename, Mesh& out) {
    FILE* f = std::fopen(filename, "rb");
    if (!f) return false;

    auto type_size = [](const std::string& t) -> int {
        if (t=="char"||t=="uchar"||t=="int8"||t=="uint8") return 1;
        if (t=="short"||t=="ushort"||t=="int16"||t=="uint16") return 2;
        if (t=="int"||t=="uint"||t=="int32"||t=="uint32"||t=="float"||t=="float32") return 4;
        if (t=="double"||t=="float64") return 8;
        return 0;
    };
    auto is_float_type = [](const std::string& t) {
        return t=="float"||t=="float32"||t=="double"||t=="float64";
    };

    struct Prop { std::string name; int size; bool is_float; };
    std::vector<Prop> vprops;
    uint32_t n_vert = 0, n_face = 0;
    int format = -1;                 // 0 ascii, 1 binary_le
    int face_count_size = 1, face_idx_size = 4;
    enum { EL_NONE, EL_VERTEX, EL_FACE } cur = EL_NONE;

    char line[512];
    if (!std::fgets(line, sizeof(line), f) || std::strncmp(line, "ply", 3) != 0) {
        std::fclose(f); return false;
    }
    while (std::fgets(line, sizeof(line), f)) {
        char a[64], b[64], c[64];
        if (std::sscanf(line, "%63s", a) != 1) continue;
        if (std::strcmp(a, "end_header") == 0) break;
        if (std::strcmp(a, "format") == 0) {
            std::sscanf(line, "format %63s", b);
            if      (std::strcmp(b, "ascii") == 0)                format = 0;
            else if (std::strcmp(b, "binary_little_endian") == 0) format = 1;
            // binary_big_endian intentionally unsupported (format stays -1).
        } else if (std::strcmp(a, "element") == 0) {
            unsigned long n = 0;
            if (std::sscanf(line, "element %63s %lu", b, &n) == 2) {
                if      (std::strcmp(b, "vertex") == 0) { cur = EL_VERTEX; n_vert = (uint32_t)n; }
                else if (std::strcmp(b, "face")   == 0) { cur = EL_FACE;   n_face = (uint32_t)n; }
                else cur = EL_NONE;
            }
        } else if (std::strcmp(a, "property") == 0) {
            if (cur == EL_VERTEX && std::sscanf(line, "property %63s %63s", b, c) == 2) {
                vprops.push_back({ c, type_size(b), is_float_type(b) });
            } else if (cur == EL_FACE && std::sscanf(line, "property list %63s %63s %63s", a, b, c) == 3) {
                face_count_size = type_size(a);
                face_idx_size   = type_size(b);
            }
        }
    }
    if (format < 0 || n_vert == 0) { std::fclose(f); return false; }

    // Locate the properties we care about and their byte offsets within a record.
    int ix=-1,iy=-1,iz=-1, inx=-1,iny=-1,inz=-1, ir=-1,ig=-1,ib=-1;
    std::vector<int> off(vprops.size());
    int stride = 0;
    for (size_t i = 0; i < vprops.size(); i++) {
        off[i] = stride; stride += vprops[i].size;
        const std::string& nm = vprops[i].name;
        if      (nm=="x")  ix=(int)i;  else if (nm=="y")  iy=(int)i;  else if (nm=="z")  iz=(int)i;
        else if (nm=="nx") inx=(int)i; else if (nm=="ny") iny=(int)i; else if (nm=="nz") inz=(int)i;
        else if (nm=="red"||nm=="r") ir=(int)i;
        else if (nm=="green"||nm=="g") ig=(int)i;
        else if (nm=="blue"||nm=="b") ib=(int)i;
    }
    if (ix<0 || iy<0 || iz<0) { std::fclose(f); return false; }
    bool have_n = (inx>=0 && iny>=0 && inz>=0);
    bool have_c = (ir>=0  && ig>=0  && ib>=0);

    std::vector<float> px(n_vert), py(n_vert), pz(n_vert), nx, ny, nz;
    std::vector<uint32_t> col;
    if (have_n) { nx.resize(n_vert); ny.resize(n_vert); nz.resize(n_vert); }
    if (have_c) col.resize(n_vert);

    // Reads property `pidx` from an in-memory record as a double (ints are
    // unsigned little-endian; floats as stored).
    auto rd = [&](const unsigned char* base, int pidx) -> double {
        const unsigned char* q = base + off[pidx];
        const Prop& pr = vprops[pidx];
        if (pr.is_float) {
            if (pr.size == 8) { double d; std::memcpy(&d, q, 8); return d; }
            float fv; std::memcpy(&fv, q, 4); return fv;
        }
        uint64_t u = 0;
        for (int k = 0; k < pr.size; k++) u |= (uint64_t)q[k] << (8*k);
        return (double)u;
    };
    auto to_byte = [&](double val, int pidx) -> uint32_t {
        double s = vprops[pidx].is_float ? val * 255.0 : val;   // float colour is 0..1
        if (s < 0) s = 0; if (s > 255) s = 255;
        return (uint32_t)(s + 0.5);
    };

    std::vector<unsigned char> rec(stride);
    for (uint32_t v = 0; v < n_vert; v++) {
        if (format == 0) {
            if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); return false; }
            // Tokenize the line into the record buffer is overkill; read doubles
            // directly in property order, then pick out the ones we need.
            double vals[64]; char* p = line; int got = 0;
            while (got < (int)vprops.size()) {
                while (*p==' '||*p=='\t') p++;
                if (*p=='\0'||*p=='\n'||*p=='\r') break;
                vals[got++] = std::strtod(p, &p);
            }
            if (got < (int)vprops.size()) { std::fclose(f); return false; }
            px[v]=(float)vals[ix]; py[v]=(float)vals[iy]; pz[v]=(float)vals[iz];
            if (have_n) { nx[v]=(float)vals[inx]; ny[v]=(float)vals[iny]; nz[v]=(float)vals[inz]; }
            if (have_c) {
                uint32_t r=to_byte(vals[ir],ir), g=to_byte(vals[ig],ig), b=to_byte(vals[ib],ib);
                col[v] = r | (g<<8) | (b<<16) | (0xFFu<<24);
            }
        } else {
            if (std::fread(rec.data(), 1, stride, f) != (size_t)stride) { std::fclose(f); return false; }
            px[v]=(float)rd(rec.data(),ix); py[v]=(float)rd(rec.data(),iy); pz[v]=(float)rd(rec.data(),iz);
            if (have_n) { nx[v]=(float)rd(rec.data(),inx); ny[v]=(float)rd(rec.data(),iny); nz[v]=(float)rd(rec.data(),inz); }
            if (have_c) {
                uint32_t r=to_byte(rd(rec.data(),ir),ir), g=to_byte(rd(rec.data(),ig),ig), b=to_byte(rd(rec.data(),ib),ib);
                col[v] = r | (g<<8) | (b<<16) | (0xFFu<<24);
            }
        }
    }

    std::vector<uint32_t> idxs;
    for (uint32_t fc = 0; fc < n_face; fc++) {
        uint32_t fv[32]; int got = 0;
        if (format == 0) {
            if (!std::fgets(line, sizeof(line), f)) break;
            char* p = line;
            long cnt = std::strtol(p, &p, 10);
            for (long k = 0; k < cnt; k++) {
                uint32_t idx = (uint32_t)std::strtol(p, &p, 10);
                if (got < 32) fv[got++] = idx;
            }
        } else {
            unsigned char cb[8] = {0};
            if (std::fread(cb, 1, face_count_size, f) != (size_t)face_count_size) break;
            uint64_t cnt = 0;
            for (int k = 0; k < face_count_size; k++) cnt |= (uint64_t)cb[k] << (8*k);
            for (uint64_t k = 0; k < cnt; k++) {
                unsigned char ibuf[8] = {0};
                if (std::fread(ibuf, 1, face_idx_size, f) != (size_t)face_idx_size) { std::fclose(f); return false; }
                uint64_t idx = 0;
                for (int b = 0; b < face_idx_size; b++) idx |= (uint64_t)ibuf[b] << (8*b);
                if (got < 32) fv[got++] = (uint32_t)idx;
            }
        }
        for (int k = 1; k + 1 < got; k++) {            // fan-triangulate
            idxs.push_back(fv[0]); idxs.push_back(fv[k]); idxs.push_back(fv[k+1]);
        }
    }
    std::fclose(f);
    if (idxs.empty()) return false;

    out.pos_x = std::move(px); out.pos_y = std::move(py); out.pos_z = std::move(pz);
    out.indices = std::move(idxs);
    out.mirror_x_map.clear();
    out.vert_tri_offset.clear(); out.vert_tri_list.clear();
    out.mask.clear();

    uint32_t nv = out.vertex_count();
    if (have_n && nx.size() == nv) {
        out.norm_x = std::move(nx); out.norm_y = std::move(ny); out.norm_z = std::move(nz);
    } else {
        out.norm_x.assign(nv, 0.0f); out.norm_y.assign(nv, 0.0f); out.norm_z.assign(nv, 1.0f);
        out.recompute_normals();
    }

    // Keep colour only if something is actually painted; an all-white import is
    // "unpainted" (matches the internal empty-means-unpainted convention).
    out.color.clear();
    if (have_c && col.size() == nv) {
        bool any = false;
        for (uint32_t c : col) if (c != 0xFFFFFFFFu) { any = true; break; }
        if (any) out.color = std::move(col);
    }

    out.build_adjacency();
    std::printf("[import] loaded %u verts, %u tris%s from %s\n",
                out.vertex_count(), out.tri_count(),
                out.color.empty() ? "" : " (+paint)", filename);
    return true;
}
