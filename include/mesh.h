#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <memory>
#include "paint_layers.h"

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

// An image a mesh wears through its corner UVs (an imported glTF base colour). Never
// edited in place, so every level and copy of the mesh shares one set of pixels: a 2K
// texture is 16 MB, and the web heap is 32-bit.
struct MeshTexture {
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;   // w * h * 4, row 0 = the image's top
    std::string name;
    // The image file's own bytes (PNG / JPEG), kept so a save stores them as they came:
    // a fraction of the pixels' size, and no re-encode. A texture made in the app (an
    // SDF merge's repack) has none until its first save PNG-encodes it here, once —
    // hence mutable: the pixels are shared const, the cache rides along.
    mutable std::vector<uint8_t> encoded;
    // Bilinear, wrapping (glTF's default sampler), at OBJ-convention (u, v): v = 1 is
    // the top row. Packed RGBA8 like Mesh::color. The bake and the colour picker share it.
    uint32_t sample(float u, float v) const;
};

// PNG / JPEG bytes -> a texture, halved past the platform's cap (glb.cpp). Null when the
// bytes don't decode; `note` then (or when it was scaled down) says so.
std::shared_ptr<const MeshTexture> decode_mesh_texture(std::vector<uint8_t> bytes,
                                                       const std::string& name, std::string* note);

// SOA triangle mesh
struct ObjPart;

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

    // Is this mesh still an exact mirror of itself about the WORLD x=0 plane?
    // The whole mirror machinery assumes it is: two of the four passes write
    // ABSOLUTE positions derived from that assumption (mirror_project averages
    // each pair onto the plane and snaps seam verts to x=0; smooth_mirror_apply
    // writes the twin outright), so on a mesh that has been turned off the plane
    // by a Select-mode spin they do not merely paint in the wrong place — they
    // tear it. Measured lazily off the pair map, cached against topo_version;
    // object transforms call invalidate_mirror_symmetry() because a turn changes
    // no topology and so would never re-trigger the measurement on its own.
    uint32_t mirror_sym_topo = 0xFFFFFFFFu;
    bool     mirror_sym_ok   = true;
    // The mean edge length build_mirror_spatial measured when it built the map, kept
    // because the symmetry test has to be judged against the SAME number the pairing
    // used. Its tolerance is 0.5 * mean_edge, so any threshold looser than that is
    // arithmetically unfailable — which is exactly the bug this field exists to close.
    float    mirror_mean_edge = 0.0f;

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

    // Paint layers. `color` above IS the active layer's pixels (premultiplied RGBA8,
    // alpha = coverage); this holds every other layer plus the stack's order, modes and
    // flags. Empty = the pre-layer world: `color` is the only paint. See paint_layers.h.
    PaintLayerStack paint_layers;

    // Remesh-density field: per-vertex 0..1 (0 = coarse, 1 = dense), painted in
    // Paint mode (density target) and consumed by the adaptive remesher. Empty =
    // unpainted = implicit 0.5 everywhere (neutral: remesh behaves exactly as a
    // uniform target). Carried by interpolation like color.
    std::vector<float> density;

    // Per-corner UVs, 2 floats per entry of `indices` (OBJ convention: v points up,
    // origin bottom-left), and the texture they look into. Per corner, not per vertex,
    // because a sculpt is welded and its UV seams run through shared vertices. Anything
    // that rebuilds the triangles without carrying these leaves `uv` the wrong size, so
    // has_uv() is the test, never uv.empty().
    std::vector<float> uv;
    std::shared_ptr<const MeshTexture> texture;
    bool has_uv() const { return !indices.empty() && uv.size() == indices.size() * 2; }

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
    // True while the mesh is still world-X symmetric to within a fraction of its
    // own size. Compares the MEAN residual over a sample of mirror pairs against
    // 1% of the bounding radius: honest sculpting drift (which mirror_project
    // exists to correct, every dab) is orders of magnitude under that, while even
    // a 5 deg turn puts it an order of magnitude over. Cheap enough to be lazy —
    // it only ever recomputes after a topology change or an object transform.
    bool mirror_world_symmetric();
    void invalidate_mirror_symmetry() { mirror_sym_topo = 0xFFFFFFFFu; }

    bool export_obj(const char* filename) const;
    // One mesh as an "o" block of an OBJ that may hold several: v / vn indices start
    // after `base` vertices already in the file. Returns the vertices it wrote.
    uint32_t write_obj(std::FILE* f, const char* object, uint32_t base) const;
    // Binary STL for 3D printing: per-tri facet normal + 3 verts, 80-byte header,
    // uint32 count. `scale` multiplies every coordinate so the print lands at
    // real-world size (e.g. mm). Winding order is preserved from the mesh.
    bool export_stl(const char* filename, float scale = 1.0f) const;
    // ASCII PLY with per-vertex position, normal, and RGB (from packed `color`).
    // PLY is the only export carrying paint; unpainted verts write white.
    bool export_ply(const char* filename) const;
    // `out` is the whole file. With `parts`, a file holding two or more `o` objects also
    // comes back one mesh per object (empty otherwise). Only `o` splits: `g` groups are
    // often polygroups of one sculpt, which must stay one mesh.
    static bool import_obj(const char* filename, Mesh& out, std::vector<ObjPart>* parts = nullptr);
    // ASCII or binary-little-endian PLY in. Reads position (required), normal
    // (recomputed if absent), and per-vertex RGB into packed `color` if present.
    static bool import_ply(const char* filename, Mesh& out);
    // glTF 2.0 in (.glb, or .gltf with its files beside it), in glb.cpp. Same contract as
    // import_obj: `out` is the whole file; `parts` one per mesh node when there are two or
    // more. Vertices are welded by position (other apps split them at UV seams and hard
    // edges, which a sculpt can't have), UVs kept per corner, the base colour texture
    // decoded. On failure `err` says why when it's worth telling (compressed meshes).
    static bool import_glb(const char* filename, Mesh& out, std::vector<ObjPart>* parts,
                           std::string* err);
};

struct ObjPart {
    std::string name;   // the `o` line's, "" for faces before any
    Mesh        mesh;
    // The faces as written (n-gons kept, not triangulated), in `mesh`'s vertex numbering:
    // face f is poly[poly_off[f] .. poly_off[f + 1]). Their `vt` per corner, 2 each, or
    // empty unless every corner had one. A retopo is rebuilt from these (retopo_from_obj).
    std::vector<uint32_t> poly, poly_off;
    std::vector<float>    uv;
};

// Spatial-hash mirror builder — same logic as Mesh::build_mirror_x_map but
// operates as a free function so multires_stack can call it on the base cage.
void build_mirror_spatial(const Mesh& m, std::vector<uint32_t>& out,
                          float* out_mean_edge = nullptr);

// Topology stencil captured by loop_subdivide under canonical numbering:
// everything needed to recompute the subdivided level's positions from new
// coarse positions without re-extracting edges. `mid` holds 4 ids per midpoint
// vertex, in midpoint numbering order: v0, v1, opp0, opp1 (opp1 == UINT32_MAX
// on a boundary edge). The bnd_* tables are the coarse verts' resolved
// boundary neighbors for the Pass-3 relocation rule; all three stay empty when
// the coarse mesh is closed (the common case). Relocation of interior verts
// additionally reads the coarse mesh's indices + CSR adjacency — cache those
// alongside this struct.
struct SubdivStencil {
    std::vector<uint32_t> mid;              // 4 per midpoint: v0, v1, opp0, opp1
    std::vector<uint8_t>  is_bnd;           // per coarse vert; empty = closed mesh
    std::vector<uint32_t> bnd_a, bnd_b;     // boundary neighbors, valid where is_bnd
};

// legacy_numbering: pre-v4 midpoint-vertex numbering (stdlib hash-map iteration
// order — platform-specific). Only the v<=3 project loader may pass true; every
// live path uses the canonical sorted-edge-key numbering, identical on all
// platforms. Multires disp layers are indexed by these vertex numbers and
// persist in .chisel files, so the numbering is effectively on-disk ABI.
// stencil: when non-null (canonical numbering only), receives the subdivision
// stencil so callers can replay this level's positions without a re-subdivide.
Mesh loop_subdivide(const Mesh& input, bool legacy_numbering = false,
                    SubdivStencil* stencil = nullptr);
Mesh icosahedron();
Mesh icosphere(int subdivisions);

// Procedural insert-mesh primitives. Both are watertight, welded (one normal per
// vertex), triangulated, and normalised so the furthest vertex sits at distance 1
// from the origin — i.e. they share the icosphere's unit bounding-sphere radius so
// the insert-mode radial scale is consistent across shapes.
//   box:      subdivided cube, `seg` quads per edge (seg >= 1).
//   cylinder: axis along Y, `radial` segments around, `height_seg` along the axis,
//             triangle-fan caps sharing the rim ring.
//   uv sphere: latitude/longitude sphere, poles on Y. (32, 16) reproduces
//             Blender's stock UV sphere vertex-for-vertex.
Mesh box_primitive(int seg);
Mesh cylinder_primitive(int radial, int height_seg);
Mesh uv_sphere(int segments, int rings);

// Packed-RGBA8 vertex-colour blending used to carry paint across topology
// changes (subdivide / remesh). All four channels lerp: paint layers store
// PREMULTIPLIED colour with alpha = coverage, so a soft stroke edge is
// "dark and half covered". Forcing alpha to 0xFF (the pre-layer rule) turned
// every new midpoint on an edge or an erased patch into opaque black dots.
// Legacy opaque paint (alpha 0xFF both ends) lerps to 0xFF exactly as before.
inline uint32_t color_lerp(uint32_t a, uint32_t b, float t) {
    uint32_t out = 0;
    for (int ch = 0; ch < 32; ch += 8) {
        float x = (float)((a >> ch) & 0xFF), y = (float)((b >> ch) & 0xFF);
        out |= (uint32_t)(x + t * (y - x) + 0.5f) << ch;
    }
    return out;
}
inline uint32_t color_avg(uint32_t a, uint32_t b) { return color_lerp(a, b, 0.5f); }
