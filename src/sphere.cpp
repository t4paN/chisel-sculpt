#include "mesh.h"
#include <cmath>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Mesh loop_subdivide(const Mesh& input) {
    const uint32_t V = input.vertex_count();
    const uint32_t F = input.tri_count();

    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        if (a > b) { uint32_t t = a; a = b; b = t; }
        return ((uint64_t)a << 32) | b;
    };

    struct EdgeData {
        uint32_t opp[2];
        uint32_t opp_count;
        uint32_t new_vert;
    };

    std::unordered_map<uint64_t, EdgeData> edge_map;
    edge_map.reserve(F * 2);

    // Pass 1: collect edges and their opposite vertices
    for (uint32_t t = 0; t < F; t++) {
        uint32_t a = input.indices[t*3+0];
        uint32_t b = input.indices[t*3+1];
        uint32_t c = input.indices[t*3+2];
        uint32_t ev[3][2] = {{a,b},{b,c},{c,a}};
        uint32_t eo[3]    = {c, a, b};
        for (int e = 0; e < 3; e++) {
            uint64_t k = edge_key(ev[e][0], ev[e][1]);
            EdgeData& d = edge_map[k];  // zero-initialised on first insert
            if (d.opp_count < 2) d.opp[d.opp_count++] = eo[e];
        }
    }

    const uint32_t E    = (uint32_t)edge_map.size();
    const uint32_t outV = V + E;
    const uint32_t outF = F * 4;

    Mesh out;
    out.pos_x.resize(outV); out.pos_y.resize(outV); out.pos_z.resize(outV);
    out.norm_x.resize(outV, 0.0f); out.norm_y.resize(outV, 0.0f); out.norm_z.resize(outV, 0.0f);
    out.indices.resize(outF * 3);

    // Carry vertex paint across the subdivision: original verts keep their
    // colour (Loop preserves them as the [0,V) prefix), edge midpoints take the
    // average of their two endpoints. Empty when unpainted (renders white).
    const bool has_color = !input.color.empty();
    if (has_color) {
        out.color.resize(outV, 0xFFFFFFFFu);
        for (uint32_t i = 0; i < V; i++) out.color[i] = input.color[i];
    }

    // Pass 2: assign indices and positions to edge-midpoint vertices
    uint32_t next_vert = V;
    for (auto& [key, d] : edge_map) {
        uint32_t v0 = (uint32_t)(key >> 32);
        uint32_t v1 = (uint32_t)(key & 0xFFFFFFFF);
        d.new_vert = next_vert++;
        Vec3 p0 = input.get_pos(v0);
        Vec3 p1 = input.get_pos(v1);
        Vec3 pos;
        if (d.opp_count == 2) {
            Vec3 p2 = input.get_pos(d.opp[0]);
            Vec3 p3 = input.get_pos(d.opp[1]);
            pos = (p0 + p1) * (3.0f/8.0f) + (p2 + p3) * (1.0f/8.0f);
        } else {
            pos = (p0 + p1) * 0.5f;
        }
        out.set_pos(d.new_vert, pos);
        if (has_color) out.color[d.new_vert] = color_avg(input.color[v0], input.color[v1]);
    }

    // Pass 3: move original vertices with Loop weights
    // Boundary detection from edges with one adjacent triangle
    std::vector<bool>     is_boundary(V, false);
    std::vector<uint32_t> bnd_a(V, 0), bnd_b(V, 0), bnd_cnt(V, 0);
    for (auto& [key, d] : edge_map) {
        if (d.opp_count != 1) continue;
        uint32_t v0 = (uint32_t)(key >> 32);
        uint32_t v1 = (uint32_t)(key & 0xFFFFFFFF);
        is_boundary[v0] = true; is_boundary[v1] = true;
        if (bnd_cnt[v0] == 0) { bnd_a[v0] = v1; bnd_cnt[v0]++; }
        else if (bnd_cnt[v0] == 1) { bnd_b[v0] = v1; bnd_cnt[v0]++; }
        if (bnd_cnt[v1] == 0) { bnd_a[v1] = v0; bnd_cnt[v1]++; }
        else if (bnd_cnt[v1] == 1) { bnd_b[v1] = v0; bnd_cnt[v1]++; }
    }

    for (uint32_t i = 0; i < V; i++) {
        Vec3 p = input.get_pos(i);
        Vec3 new_pos;
        if (is_boundary[i]) {
            Vec3 pb0 = input.get_pos(bnd_a[i]);
            Vec3 pb1 = input.get_pos(bnd_b[i]);
            new_pos = p * 0.75f + (pb0 + pb1) * (1.0f/8.0f);
        } else {
            // n = valence = adjacent tri count for interior manifold vertex
            uint32_t n = input.vert_tri_offset[i+1] - input.vert_tri_offset[i];
            float beta = (n == 3) ? (3.0f/16.0f) : (3.0f / (8.0f * (float)n));
            // Each of n distinct neighbours appears in exactly 2 adjacent tris,
            // so the raw sum double-counts → multiply beta by 0.5
            Vec3 nbr_sum = {0, 0, 0};
            for (uint32_t tj = input.vert_tri_offset[i]; tj < input.vert_tri_offset[i+1]; tj++) {
                uint32_t tri = input.vert_tri_list[tj];
                uint32_t a = input.indices[tri*3+0];
                uint32_t b = input.indices[tri*3+1];
                uint32_t c = input.indices[tri*3+2];
                if      (a == i) { nbr_sum += input.get_pos(b); nbr_sum += input.get_pos(c); }
                else if (b == i) { nbr_sum += input.get_pos(a); nbr_sum += input.get_pos(c); }
                else             { nbr_sum += input.get_pos(a); nbr_sum += input.get_pos(b); }
            }
            new_pos = p * (1.0f - (float)n * beta) + nbr_sum * (beta * 0.5f);
        }
        out.set_pos(i, new_pos);
    }

    // Pass 4: emit 4 output triangles per input triangle (preserve winding).
    // INVARIANT depended on by build_fine_mirror (multires_stack.cpp): for each
    // coarse face t, the 12 output indices at out.indices[t*12 + 0..11] follow
    // this exact layout — [1]=midpoint(a,b), [4]=midpoint(b,c), [7]=midpoint(c,a).
    // Do not reorder the four sub-triangles without updating build_fine_mirror.
    uint32_t idx = 0;
    for (uint32_t t = 0; t < F; t++) {
        uint32_t a  = input.indices[t*3+0];
        uint32_t b  = input.indices[t*3+1];
        uint32_t c  = input.indices[t*3+2];
        uint32_t ab = edge_map.at(edge_key(a, b)).new_vert;
        uint32_t bc = edge_map.at(edge_key(b, c)).new_vert;
        uint32_t ca = edge_map.at(edge_key(c, a)).new_vert;
        out.indices[idx++] = a;  out.indices[idx++] = ab; out.indices[idx++] = ca;
        out.indices[idx++] = b;  out.indices[idx++] = bc; out.indices[idx++] = ab;
        out.indices[idx++] = c;  out.indices[idx++] = ca; out.indices[idx++] = bc;
        out.indices[idx++] = ab; out.indices[idx++] = bc; out.indices[idx++] = ca;
    }

    return out;
}

Mesh icosahedron() {
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;

    const float raw[12][3] = {
        {-1,  phi,  0}, { 1,  phi,  0}, {-1, -phi,  0}, { 1, -phi,  0},
        { 0, -1,  phi}, { 0,  1,  phi}, { 0, -1, -phi}, { 0,  1, -phi},
        { phi,  0, -1}, { phi,  0,  1}, {-phi,  0, -1}, {-phi,  0,  1},
    };

    const uint32_t faces[20][3] = {
        { 0, 11,  5}, { 0,  5,  1}, { 0,  1,  7}, { 0,  7, 10}, { 0, 10, 11},
        { 1,  5,  9}, { 5, 11,  4}, {11, 10,  2}, {10,  7,  6}, { 7,  1,  8},
        { 3,  9,  4}, { 3,  4,  2}, { 3,  2,  6}, { 3,  6,  8}, { 3,  8,  9},
        { 4,  9,  5}, { 2,  4, 11}, { 6,  2, 10}, { 8,  6,  7}, { 9,  8,  1},
    };

    Mesh mesh;
    mesh.pos_x.resize(12);
    mesh.pos_y.resize(12);
    mesh.pos_z.resize(12);
    mesh.norm_x.resize(12, 0.0f);
    mesh.norm_y.resize(12, 0.0f);
    mesh.norm_z.resize(12, 0.0f);

    for (int i = 0; i < 12; i++) {
        Vec3 v = Vec3(raw[i][0], raw[i][1], raw[i][2]).normalized();
        mesh.pos_x[i] = v.x;
        mesh.pos_y[i] = v.y;
        mesh.pos_z[i] = v.z;
    }

    mesh.indices.reserve(60);
    for (int i = 0; i < 20; i++) {
        mesh.indices.push_back(faces[i][0]);
        mesh.indices.push_back(faces[i][1]);
        mesh.indices.push_back(faces[i][2]);
    }

    return mesh;
}

Mesh icosphere(int subdivisions) {
    Mesh m = icosahedron();
    for (int i = 0; i < subdivisions; i++) {
        m.build_adjacency();  // loop_subdivide reads CSR for vertex relocation
        m = loop_subdivide(m);
        uint32_t vc = m.vertex_count();
        for (uint32_t v = 0; v < vc; v++) {
            m.set_pos(v, m.get_pos(v).normalized());
        }
    }
    return m;
}
