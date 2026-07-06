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

    // Mirror map: mirror_x_map[i] = index of vertex mirrored across X axis.
    // Three explicit classes: paired (twin index), seam (maps to itself —
    // constrained to the x=0 plane), unpaired (MIRROR_UNPAIRED sentinel — the
    // mirror machinery leaves it alone). The sentinel is >= vertex_count, which
    // kernels already treat as "copy through".
    static constexpr uint32_t MIRROR_UNPAIRED = 0xFFFFFFFFu;
    std::vector<uint32_t> mirror_x_map;

    // Topology stamp: bumped by build_adjacency() (called at every topology
    // change, never during strokes). The mirror map records the stamp it was
    // built at, so position-only edits can never trigger a rebuild that would
    // reclassify drifted verts.
    uint32_t topo_version = 0;
    uint32_t mirror_topo_version = 0xFFFFFFFFu;

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
    // ASCII PLY with per-vertex position, normal, and RGB (from packed `color`).
    // PLY is the only export carrying paint; unpainted verts write white.
    bool export_ply(const char* filename) const;
    static bool import_obj(const char* filename, Mesh& out);
    // ASCII or binary-little-endian PLY in. Reads position (required), normal
    // (recomputed if absent), and per-vertex RGB into packed `color` if present.
    static bool import_ply(const char* filename, Mesh& out);
};

// Spatial-hash mirror builder — same logic as Mesh::build_mirror_x_map but
// operates as a free function so multires_stack can call it on the base cage.
void build_mirror_spatial(const Mesh& m, std::vector<uint32_t>& out);

// legacy_numbering: pre-v4 midpoint-vertex numbering (stdlib hash-map iteration
// order — platform-specific). Only the v<=3 project loader may pass true; every
// live path uses the canonical sorted-edge-key numbering, identical on all
// platforms. Multires disp layers are indexed by these vertex numbers and
// persist in .chisel files, so the numbering is effectively on-disk ABI.
Mesh loop_subdivide(const Mesh& input, bool legacy_numbering = false);
Mesh icosahedron();
Mesh icosphere(int subdivisions);

// Packed-RGBA8 vertex-colour blending used to carry paint across topology
// changes (subdivide / remesh). Alpha is forced to 0xFF — the paint model
// keeps alpha at 1.0, and unpainted verts are white 0xFFFFFFFF.
inline uint32_t color_lerp(uint32_t a, uint32_t b, float t) {
    float ar = (float)( a        & 0xFF), ag = (float)((a >> 8) & 0xFF), ab = (float)((a >> 16) & 0xFF);
    float br = (float)( b        & 0xFF), bg = (float)((b >> 8) & 0xFF), bb = (float)((b >> 16) & 0xFF);
    uint32_t r  = (uint32_t)(ar + t * (br - ar) + 0.5f);
    uint32_t g  = (uint32_t)(ag + t * (bg - ag) + 0.5f);
    uint32_t bl = (uint32_t)(ab + t * (bb - ab) + 0.5f);
    return r | (g << 8) | (bl << 16) | (0xFFu << 24);
}
inline uint32_t color_avg(uint32_t a, uint32_t b) { return color_lerp(a, b, 0.5f); }
