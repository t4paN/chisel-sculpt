#include "mesh.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Mesh loop_subdivide(const Mesh& input, bool legacy_numbering, SubdivStencil* stencil) {
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

    // Same ride for the sculpt mask (may be shorter than V — zero-pad).
    const bool has_mask = !input.mask.empty();
    if (has_mask) {
        out.mask.resize(outV, 0.0f);
        for (uint32_t i = 0; i < V && i < (uint32_t)input.mask.size(); i++)
            out.mask[i] = input.mask[i];
    }

    // And the remesh-density field (neutral 0.5 where absent).
    const bool has_density = !input.density.empty();
    if (has_density) {
        out.density.resize(outV, 0.5f);
        for (uint32_t i = 0; i < V && i < (uint32_t)input.density.size(); i++)
            out.density[i] = input.density[i];
    }

    // Pass 2: assign indices and positions to edge-midpoint vertices.
    // Canonical numbering: edges in sorted-key order — bit-identical on every
    // platform. Legacy numbering (v<=3 files) is the raw unordered_map iteration
    // order this platform's stdlib produces; see the header note in mesh.h.
    auto assign_midpoint = [&](uint64_t key, EdgeData& d, uint32_t vert) {
        uint32_t v0 = (uint32_t)(key >> 32);
        uint32_t v1 = (uint32_t)(key & 0xFFFFFFFF);
        d.new_vert = vert;
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
        if (has_mask) {
            float m0 = (v0 < (uint32_t)input.mask.size()) ? input.mask[v0] : 0.0f;
            float m1 = (v1 < (uint32_t)input.mask.size()) ? input.mask[v1] : 0.0f;
            out.mask[d.new_vert] = 0.5f * (m0 + m1);
        }
        if (has_density) {
            float d0 = (v0 < (uint32_t)input.density.size()) ? input.density[v0] : 0.5f;
            float d1 = (v1 < (uint32_t)input.density.size()) ? input.density[v1] : 0.5f;
            out.density[d.new_vert] = 0.5f * (d0 + d1);
        }
    };
    uint32_t next_vert = V;
    if (legacy_numbering) {
        stencil = nullptr;  // stencil capture is canonical-numbering only
        for (auto& [key, d] : edge_map)
            assign_midpoint(key, d, next_vert++);
    } else {
        std::vector<uint64_t> sorted_keys;
        sorted_keys.reserve(edge_map.size());
        for (auto& [key, d] : edge_map) sorted_keys.push_back(key);
        std::sort(sorted_keys.begin(), sorted_keys.end());
        if (stencil) {
            stencil->mid.clear();
            stencil->mid.reserve((size_t)E * 4);
        }
        for (uint64_t key : sorted_keys) {
            EdgeData& d = edge_map.at(key);
            assign_midpoint(key, d, next_vert++);
            if (stencil) {
                stencil->mid.push_back((uint32_t)(key >> 32));
                stencil->mid.push_back((uint32_t)(key & 0xFFFFFFFF));
                stencil->mid.push_back(d.opp[0]);
                stencil->mid.push_back(d.opp_count == 2 ? d.opp[1] : UINT32_MAX);
            }
        }
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

    // Capture the RESOLVED boundary tables: which neighbor landed in bnd_a vs
    // bnd_b depends on hash-map iteration order, so a replay must reuse these
    // exact tables to reproduce this pass bit-for-bit. Closed mesh → all empty.
    if (stencil) {
        stencil->is_bnd.clear();
        stencil->bnd_a.clear();
        stencil->bnd_b.clear();
        bool any_bnd = false;
        for (uint32_t i = 0; i < V; i++) if (is_boundary[i]) { any_bnd = true; break; }
        if (any_bnd) {
            stencil->is_bnd.resize(V);
            for (uint32_t i = 0; i < V; i++) stencil->is_bnd[i] = is_boundary[i] ? 1 : 0;
            stencil->bnd_a = bnd_a;
            stencil->bnd_b = bnd_b;
        }
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

// Scale every vertex so the furthest from the origin lands at distance 1, then
// bake face-averaged normals. Shared final step for the box/cylinder builders so
// all insert primitives live on the same unit bounding sphere as the icosphere.
static void finalize_primitive(Mesh& m) {
    float max_d2 = 0.0f;
    uint32_t vc = m.vertex_count();
    for (uint32_t i = 0; i < vc; i++) {
        Vec3 p = m.get_pos(i);
        float d2 = p.x*p.x + p.y*p.y + p.z*p.z;
        if (d2 > max_d2) max_d2 = d2;
    }
    float inv = (max_d2 > 0.0f) ? 1.0f / std::sqrt(max_d2) : 1.0f;
    for (uint32_t i = 0; i < vc; i++) m.set_pos(i, m.get_pos(i) * inv);

    m.norm_x.assign(vc, 0.0f);
    m.norm_y.assign(vc, 0.0f);
    m.norm_z.assign(vc, 0.0f);
    m.recompute_normals();
}

Mesh box_primitive(int seg) {
    if (seg < 1) seg = 1;
    Mesh m;

    // Weld surface grid points across faces: a packed (i,j,k) key with each axis in
    // [0,seg] maps to a single vertex, so shared edges/corners aren't duplicated.
    std::unordered_map<uint64_t, uint32_t> vmap;
    vmap.reserve((size_t)(seg + 1) * (seg + 1) * 6);
    auto key = [&](int i, int j, int k) -> uint64_t {
        return ((uint64_t)(uint32_t)i << 42) | ((uint64_t)(uint32_t)j << 21) | (uint32_t)k;
    };
    auto vert = [&](int i, int j, int k) -> uint32_t {
        uint64_t kk = key(i, j, k);
        auto it = vmap.find(kk);
        if (it != vmap.end()) return it->second;
        uint32_t id = m.vertex_count();
        vmap.emplace(kk, id);
        m.pos_x.push_back(-1.0f + 2.0f * (float)i / (float)seg);
        m.pos_y.push_back(-1.0f + 2.0f * (float)j / (float)seg);
        m.pos_z.push_back(-1.0f + 2.0f * (float)k / (float)seg);
        return id;
    };

    // Each face: one axis fixed, the other two swept over the (u,v) grid. u/v axes
    // are ordered so the emitted winding gives an outward normal.
    auto emit_face = [&](int fixed_axis, int fixed_val, int u_axis, int v_axis) {
        auto coord = [&](int u, int v, int c[3]) {
            c[0] = c[1] = c[2] = 0;
            c[fixed_axis] = fixed_val; c[u_axis] = u; c[v_axis] = v;
        };
        for (int u = 0; u < seg; u++) {
            for (int v = 0; v < seg; v++) {
                int c00[3], c10[3], c11[3], c01[3];
                coord(u,   v,   c00); coord(u+1, v,   c10);
                coord(u+1, v+1, c11); coord(u,   v+1, c01);
                uint32_t a = vert(c00[0], c00[1], c00[2]);
                uint32_t b = vert(c10[0], c10[1], c10[2]);
                uint32_t c = vert(c11[0], c11[1], c11[2]);
                uint32_t d = vert(c01[0], c01[1], c01[2]);
                m.indices.push_back(a); m.indices.push_back(b); m.indices.push_back(c);
                m.indices.push_back(a); m.indices.push_back(c); m.indices.push_back(d);
            }
        }
    };
    emit_face(0, seg, 1, 2);  // +X: sweep Y,Z
    emit_face(0, 0,   2, 1);  // -X: sweep Z,Y
    emit_face(1, seg, 2, 0);  // +Y: sweep Z,X
    emit_face(1, 0,   0, 2);  // -Y: sweep X,Z
    emit_face(2, seg, 0, 1);  // +Z: sweep X,Y
    emit_face(2, 0,   1, 0);  // -Z: sweep Y,X

    finalize_primitive(m);
    return m;
}

Mesh cylinder_primitive(int radial, int height_seg) {
    if (radial < 3) radial = 3;
    if (height_seg < 1) height_seg = 1;
    Mesh m;

    // Side rings: (height_seg+1) rings of `radial` verts. The seam welds naturally
    // because index a is taken mod radial — no duplicated seam column.
    for (int r = 0; r <= height_seg; r++) {
        float y = -1.0f + 2.0f * (float)r / (float)height_seg;
        for (int a = 0; a < radial; a++) {
            float ang = 2.0f * (float)M_PI * (float)a / (float)radial;
            m.pos_x.push_back(std::cos(ang));
            m.pos_y.push_back(y);
            m.pos_z.push_back(std::sin(ang));
        }
    }
    auto ring = [&](int r, int a) -> uint32_t { return (uint32_t)(r * radial + (a % radial)); };

    // Side quads (outward-facing winding, angle increases CCW seen from +Y).
    for (int r = 0; r < height_seg; r++) {
        for (int a = 0; a < radial; a++) {
            uint32_t v00 = ring(r,   a),   v10 = ring(r,   a+1);
            uint32_t v11 = ring(r+1, a+1), v01 = ring(r+1, a);
            m.indices.push_back(v00); m.indices.push_back(v01); m.indices.push_back(v11);
            m.indices.push_back(v00); m.indices.push_back(v11); m.indices.push_back(v10);
        }
    }

    // Cap centres, fanned to the existing rim rings so the mesh stays watertight.
    uint32_t top_c = m.vertex_count();
    m.pos_x.push_back(0.0f); m.pos_y.push_back( 1.0f); m.pos_z.push_back(0.0f);
    uint32_t bot_c = m.vertex_count();
    m.pos_x.push_back(0.0f); m.pos_y.push_back(-1.0f); m.pos_z.push_back(0.0f);
    for (int a = 0; a < radial; a++) {
        // Top (+Y outward): reversed sweep so the fan normal is +Y.
        m.indices.push_back(top_c); m.indices.push_back(ring(height_seg, a+1)); m.indices.push_back(ring(height_seg, a));
        // Bottom (-Y outward).
        m.indices.push_back(bot_c); m.indices.push_back(ring(0, a)); m.indices.push_back(ring(0, a+1));
    }

    finalize_primitive(m);
    return m;
}
