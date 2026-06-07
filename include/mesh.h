#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const {
        float l = length();
        if (l < 1e-8f) return {0,0,0};
        return {x/l, y/l, z/l};
    }
};

// SOA triangle mesh
struct Mesh {
    std::vector<float> pos_x, pos_y, pos_z;   // vertex positions
    std::vector<float> norm_x, norm_y, norm_z; // vertex normals
    std::vector<uint32_t> indices;             // triangle indices (3 per tri)

    // Mirror map: mirror_x_map[i] = index of vertex mirrored across X axis
    // A vertex maps to itself if it sits on the mirror plane
    std::vector<uint32_t> mirror_x_map;

    // Vertex -> triangle adjacency (CSR format)
    // vert_tri_offset[i] .. vert_tri_offset[i+1] indexes into vert_tri_list
    std::vector<uint32_t> vert_tri_offset;  // size = vertex_count + 1
    std::vector<uint32_t> vert_tri_list;    // flat list of triangle indices

    // Mask: per-vertex mask values (0..1), sparse storage
    std::vector<float> mask;

    // Vertex paint: per-vertex albedo, packed RGBA8 (one uint32 per vertex,
    // little-endian r|g<<8|b<<16|a<<24). Empty = unpainted (renders as white).
    // Persists across subdivide/remesh (carried by interpolation); unlike mask
    // it is creative work, not a transient selection. 0xFFFFFFFF = white.
    std::vector<uint32_t> color;

    uint32_t vertex_count() const { return (uint32_t)pos_x.size(); }
    uint32_t tri_count() const { return (uint32_t)(indices.size() / 3); }

    Vec3 get_pos(uint32_t i) const { return {pos_x[i], pos_y[i], pos_z[i]}; }
    void set_pos(uint32_t i, Vec3 p) { pos_x[i]=p.x; pos_y[i]=p.y; pos_z[i]=p.z; }

    Vec3 get_normal(uint32_t i) const { return {norm_x[i], norm_y[i], norm_z[i]}; }
    void set_normal(uint32_t i, Vec3 n) { norm_x[i]=n.x; norm_y[i]=n.y; norm_z[i]=n.z; }

    void recompute_normals();
    void recompute_normals_partial(const std::vector<uint32_t>& dirty_verts,
                                   std::vector<uint32_t>* affected_verts_out = nullptr);
    void expand_dirty_to_affected(const std::vector<uint32_t>& dirty_verts,
                                  std::vector<uint32_t>& affected_out);
    void build_adjacency();
    void compute_bounding_sphere(Vec3& center, float& radius) const;
    void build_mirror_x_map();

    bool export_obj(const char* filename) const;
    // Binary STL for 3D printing: per-tri facet normal + 3 verts, 80-byte header,
    // uint32 count. `scale` multiplies every coordinate so the print lands at
    // real-world size (e.g. mm). Winding order is preserved from the mesh.
    bool export_stl(const char* filename, float scale = 1.0f) const;
    static bool import_obj(const char* filename, Mesh& out);
};

// Spatial-hash mirror builder — same logic as Mesh::build_mirror_x_map but
// operates as a free function so multires_stack can call it on the base cage.
void build_mirror_spatial(const Mesh& m, std::vector<uint32_t>& out);

Mesh loop_subdivide(const Mesh& input);
Mesh icosahedron();
Mesh icosphere(int subdivisions);
