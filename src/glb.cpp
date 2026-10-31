// glTF 2.0 import (.glb / .gltf): the way AI-generated models arrive (image-to-3D tools
// hand out a dense triangle mesh with UVs and a baked base colour texture). Geometry
// goes the same way an OBJ does, one mesh per node; the texture rides on the mesh as
// per-corner UVs + shared pixels (Mesh::uv / Mesh::texture).
#define CGLTF_IMPLEMENTATION
#include "cgltf/cgltf.h"
#include "stb/stb_image.h"   // the implementation is in window_icon.cpp
#include "mesh.h"
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace {

// The biggest side a texture keeps. The web heap is 32-bit and a texture lives twice
// (CPU for the bake, GPU for the screen): 4K RGBA is 64 MB each, so the browser halves
// anything past 2K. Halving keeps every texel an even blend of four, no resampling.
#ifdef __EMSCRIPTEN__
constexpr int kMaxTexSide = 2048;
#else
constexpr int kMaxTexSide = 4096;
#endif

void halve(MeshTexture& t) {
    const int w = std::max(1, t.w / 2), h = std::max(1, t.h / 2);
    std::vector<uint8_t> out((size_t)w * h * 4);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 4; c++) {
                int sum = 0;
                for (int k = 0; k < 4; k++) {
                    const int sx = std::min(t.w - 1, 2 * x + (k & 1));
                    const int sy = std::min(t.h - 1, 2 * y + (k >> 1));
                    sum += t.rgba[((size_t)sy * t.w + sx) * 4 + c];
                }
                out[((size_t)y * w + x) * 4 + c] = (uint8_t)((sum + 2) / 4);
            }
    t.w = w; t.h = h;
    t.rgba = std::move(out);
}

// The image's encoded bytes (PNG/JPEG): inside the GLB, a data: URI, or a file beside
// a .gltf. stb_image decodes them; webp/basis/KTX textures it can't are turned away.
std::shared_ptr<const MeshTexture> load_image(const cgltf_options& opt, const cgltf_image* img,
                                              const std::string& dir, std::string* note) {
    std::vector<uint8_t> bytes;
    if (img->buffer_view) {
        const uint8_t* p = cgltf_buffer_view_data(img->buffer_view);
        if (p) bytes.assign(p, p + img->buffer_view->size);
    } else if (img->uri && std::strncmp(img->uri, "data:", 5) == 0) {
        const char* comma = std::strchr(img->uri, ',');
        if (comma && std::strstr(img->uri, ";base64")) {
            const size_t n = std::strlen(comma + 1);
            size_t size = n / 4 * 3;
            if (n >= 1 && comma[n] == '=') size--;
            if (n >= 2 && comma[n - 1] == '=') size--;
            void* out = nullptr;
            if (cgltf_load_buffer_base64(&opt, size, comma + 1, &out) == cgltf_result_success) {
                bytes.assign((uint8_t*)out, (uint8_t*)out + size);
                std::free(out);
            }
        }
    } else if (img->uri) {
        std::string uri = img->uri;
        cgltf_decode_uri(&uri[0]);
        uri.resize(std::strlen(uri.c_str()));
        if (std::FILE* f = std::fopen((dir + uri).c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (n > 0) {
                bytes.resize((size_t)n);
                if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
            }
            std::fclose(f);
        }
    }
    if (bytes.empty()) { if (note) *note = "texture not found"; return nullptr; }
    int w = 0, h = 0, comp = 0;
    uint8_t* px = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &comp, 4);
    if (!px) { if (note) *note = "texture format not readable"; return nullptr; }
    auto t = std::make_shared<MeshTexture>();
    t->w = w; t->h = h;
    t->rgba.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    t->name = img->name ? img->name : (img->uri && std::strncmp(img->uri, "data:", 5) ? img->uri : "");
    while (t->w > kMaxTexSide || t->h > kMaxTexSide) halve(*t);
    if (t->w != w || t->h != h) {
        std::printf("[glb] texture %dx%d halved to %dx%d (cap %d)\n", w, h, t->w, t->h, kMaxTexSide);
        if (note) *note = "texture scaled down";
    }
    return t;
}

// One part under construction: welded vertices, triangles, corner UVs.
struct Build {
    std::string name;
    const cgltf_node*  node  = nullptr;
    const cgltf_image* image = nullptr;   // the base colour image its triangles look into
    std::vector<float> x, y, z;
    std::vector<uint32_t> color;          // COLOR_0 per welded vertex, kept sized once on
    bool has_color = false;
    std::vector<uint32_t> idx;
    std::vector<float> uv;                // 2 per corner, OBJ convention
    bool all_uv = true;                   // every corner had a TEXCOORD
    std::unordered_map<uint64_t, std::vector<uint32_t>> weld;   // position hash -> verts
};

uint32_t bits(float f) {
    if (f == 0.0f) f = 0.0f;   // -0 and +0 are one place
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

uint32_t weld_vertex(Build& b, float px, float py, float pz) {
    const uint64_t h = ((uint64_t)bits(px) * 73856093u) ^ ((uint64_t)bits(py) * 19349663u << 21)
                     ^ ((uint64_t)bits(pz) * 83492791u << 42);
    std::vector<uint32_t>& bucket = b.weld[h];
    for (uint32_t v : bucket)
        if (b.x[v] == px && b.y[v] == py && b.z[v] == pz) return v;
    const uint32_t v = (uint32_t)b.x.size();
    b.x.push_back(px); b.y.push_back(py); b.z.push_back(pz);
    if (b.has_color) b.color.push_back(0xFFFFFFFFu);
    bucket.push_back(v);
    return v;
}

uint32_t pack_rgba(const float* c, int n) {
    auto q = [](float f) { return (uint32_t)std::lround(std::min(1.0f, std::max(0.0f, f)) * 255.0f); };
    return q(c[0]) | q(c[1]) << 8 | q(c[2]) << 16 | (n == 4 ? q(c[3]) : 255u) << 24;
}

// One primitive into its part, through the node's world matrix.
void add_primitive(Build& b, const cgltf_primitive& prim, const float M[16], int* skipped) {
    const cgltf_accessor *pos = nullptr, *col = nullptr;
    const cgltf_accessor* tex[4] = {};
    for (cgltf_size i = 0; i < prim.attributes_count; i++) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) pos = a.data;
        else if (a.type == cgltf_attribute_type_color && a.index == 0) col = a.data;
        else if (a.type == cgltf_attribute_type_texcoord && a.index >= 0 && a.index < 4) tex[a.index] = a.data;
    }
    if (!pos || pos->count == 0) { (*skipped)++; return; }
    int set = 0;
    if (prim.material && prim.material->has_pbr_metallic_roughness)
        set = prim.material->pbr_metallic_roughness.base_color_texture.texcoord;
    const cgltf_accessor* uva = (set >= 0 && set < 4) ? tex[set] : nullptr;

    const size_t n = pos->count;
    std::vector<float> p(n * 3), t, c;
    cgltf_accessor_unpack_floats(pos, p.data(), n * 3);
    if (uva && uva->count == n) {
        t.resize(n * 2);
        cgltf_accessor_unpack_floats(uva, t.data(), n * 2);
    }
    int ncol = 0;
    if (col && col->count == n) {
        ncol = (int)cgltf_num_components(col->type);
        c.resize(n * ncol);
        cgltf_accessor_unpack_floats(col, c.data(), n * ncol);
        if (!b.has_color) { b.color.assign(b.x.size(), 0xFFFFFFFFu); b.has_color = true; }
    }

    // Weld through the world matrix (column-major). A mirroring matrix turns the
    // triangles inside out, so it flips their winding back.
    std::vector<uint32_t> remap(n);
    for (size_t i = 0; i < n; i++) {
        const float* q = &p[3 * i];
        const float wx = M[0] * q[0] + M[4] * q[1] + M[8] * q[2] + M[12];
        const float wy = M[1] * q[0] + M[5] * q[1] + M[9] * q[2] + M[13];
        const float wz = M[2] * q[0] + M[6] * q[1] + M[10] * q[2] + M[14];
        remap[i] = weld_vertex(b, wx, wy, wz);
        if (ncol) b.color[remap[i]] = pack_rgba(&c[ncol * i], ncol);
    }
    const float det = M[0] * (M[5] * M[10] - M[9] * M[6]) - M[4] * (M[1] * M[10] - M[9] * M[2])
                    + M[8] * (M[1] * M[6] - M[5] * M[2]);
    const bool flip = det < 0.0f;

    auto index = [&](size_t k) -> size_t {
        return prim.indices ? cgltf_accessor_read_index(prim.indices, k) : k;
    };
    const size_t ni = prim.indices ? prim.indices->count : n;
    auto tri = [&](size_t i0, size_t i1, size_t i2) {
        if (i0 >= n || i1 >= n || i2 >= n) return;
        if (flip) std::swap(i1, i2);
        const uint32_t a = remap[i0], bb = remap[i1], cc = remap[i2];
        if (a == bb || bb == cc || cc == a) return;   // welded flat: a sculpt can't keep it
        b.idx.push_back(a); b.idx.push_back(bb); b.idx.push_back(cc);
        for (size_t k : {i0, i1, i2}) {
            // glTF puts the UV origin top-left; Chisel (like OBJ) bottom-left.
            b.uv.push_back(t.empty() ? 0.0f : t[2 * k]);
            b.uv.push_back(t.empty() ? 0.0f : 1.0f - t[2 * k + 1]);
        }
        if (t.empty()) b.all_uv = false;
    };
    switch (prim.type) {
    case cgltf_primitive_type_triangles:
        for (size_t k = 0; k + 2 < ni; k += 3) tri(index(k), index(k + 1), index(k + 2));
        break;
    case cgltf_primitive_type_triangle_strip:
        for (size_t k = 0; k + 2 < ni; k++) {
            if (k & 1) tri(index(k + 1), index(k), index(k + 2));
            else       tri(index(k), index(k + 1), index(k + 2));
        }
        break;
    case cgltf_primitive_type_triangle_fan:
        for (size_t k = 1; k + 1 < ni; k++) tri(index(0), index(k), index(k + 1));
        break;
    default:
        (*skipped)++;   // points and lines: nothing to sculpt
    }
}

}  // namespace

bool Mesh::import_glb(const char* filename, Mesh& out, std::vector<ObjPart>* parts, std::string* err) {
    cgltf_options opt = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opt, filename, &data) != cgltf_result_success) {
        if (err) *err = "Not a readable glTF file";
        return false;
    }
    struct Free { cgltf_data* d; ~Free() { cgltf_free(d); } } free_data{data};

    // Compressed geometry: cgltf reads the file but can't unpack the vertices. Draco
    // needs a decoder bigger than this whole importer; say so instead of half-loading.
    for (cgltf_size i = 0; i < data->extensions_required_count; i++) {
        const char* e = data->extensions_required[i];
        if (!std::strcmp(e, "KHR_draco_mesh_compression") || !std::strcmp(e, "EXT_meshopt_compression")) {
            if (err) *err = std::string("Compressed glTF (") + e + ") isn't supported: "
                            "export it again without mesh compression";
            return false;
        }
    }
    if (cgltf_load_buffers(&opt, data, filename) != cgltf_result_success) {
        if (err) *err = "glTF buffers missing (a .gltf needs its .bin beside it)";
        return false;
    }

    std::string dir = filename;
    dir.resize(dir.find_last_of("/\\") == std::string::npos ? 0 : dir.find_last_of("/\\") + 1);

    // Walk the scene's nodes (or every root when there is no scene), each mesh node
    // becoming one part per base colour image it uses: one texture per sculpt.
    std::vector<Build> builds;
    int skipped = 0;
    auto add_node = [&](const cgltf_node* node, auto& self) -> void {
        if (node->mesh) {
            float M[16];
            cgltf_node_transform_world(node, M);
            const char* nm = node->name ? node->name : node->mesh->name ? node->mesh->name : "";
            for (cgltf_size i = 0; i < node->mesh->primitives_count; i++) {
                const cgltf_primitive& prim = node->mesh->primitives[i];
                const cgltf_image* img = nullptr;
                if (prim.material && prim.material->has_pbr_metallic_roughness) {
                    const cgltf_texture* tx = prim.material->pbr_metallic_roughness.base_color_texture.texture;
                    if (tx) img = tx->image;
                }
                Build* b = nullptr;
                int of_node = 0;
                for (Build& c : builds)
                    if (c.node == node) { of_node++; if (c.image == img) b = &c; }
                if (!b) {
                    builds.push_back(Build{});
                    b = &builds.back();
                    b->node = node;
                    b->image = img;
                    b->name = of_node ? std::string(nm) + "_" + std::to_string(of_node + 1) : nm;
                }
                add_primitive(*b, prim, M, &skipped);
            }
        }
        for (cgltf_size i = 0; i < node->children_count; i++) self(node->children[i], self);
    };
    const cgltf_scene* scene = data->scene ? data->scene : data->scenes_count ? &data->scenes[0] : nullptr;
    if (scene) {
        for (cgltf_size i = 0; i < scene->nodes_count; i++) add_node(scene->nodes[i], add_node);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; i++)
            if (!data->nodes[i].parent) add_node(&data->nodes[i], add_node);
    }

    // Decode each image once, however many parts share it.
    std::string note;
    std::vector<std::pair<const cgltf_image*, std::shared_ptr<const MeshTexture>>> images;
    std::vector<ObjPart> made;
    for (Build& b : builds) {
        if (b.idx.empty()) continue;
        ObjPart part;
        part.name = b.name;
        Mesh& m = part.mesh;
        m.pos_x = std::move(b.x); m.pos_y = std::move(b.y); m.pos_z = std::move(b.z);
        m.indices = std::move(b.idx);
        m.color = std::move(b.color);
        if (b.all_uv) {
            m.uv = std::move(b.uv);
            if (b.image) {
                auto it = std::find_if(images.begin(), images.end(), [&](auto& e) { return e.first == b.image; });
                if (it == images.end()) {
                    images.push_back({b.image, load_image(opt, b.image, dir, &note)});
                    it = images.end() - 1;
                }
                m.texture = it->second;
            }
        }
        const uint32_t nv = m.vertex_count();
        m.norm_x.assign(nv, 0.0f); m.norm_y.assign(nv, 0.0f); m.norm_z.assign(nv, 1.0f);
        m.recompute_normals();   // welded: the split, hard-edged normals don't apply
        m.build_adjacency();
        // A retopo is rebuilt from faces + per-corner UVs (retopo_from_obj): here they're
        // the triangles, since glTF has no quads.
        part.poly = m.indices;
        part.poly_off.resize(m.tri_count() + 1);
        for (uint32_t f = 0; f <= m.tri_count(); f++) part.poly_off[f] = 3 * f;
        if (m.has_uv()) part.uv = m.uv;
        std::printf("[glb]   '%s': %u verts, %u tris, %s, texture %s\n", part.name.c_str(),
                    nv, m.tri_count(), m.has_uv() ? "UVs" : "no UVs",
                    m.texture ? (std::to_string(m.texture->w) + "x" + std::to_string(m.texture->h)).c_str() : "none");
        made.push_back(std::move(part));
    }
    if (made.empty()) {
        if (err) *err = "No triangles in this glTF file";
        return false;
    }

    // `out` is the whole file: the one part, or all of them side by side (it frames the
    // camera; the parts become the meshes).
    if (made.size() == 1) {
        out = std::move(made[0].mesh);
    } else {
        out = Mesh{};
        for (const ObjPart& p : made) {
            const uint32_t base = out.vertex_count();
            const Mesh& m = p.mesh;
            out.pos_x.insert(out.pos_x.end(), m.pos_x.begin(), m.pos_x.end());
            out.pos_y.insert(out.pos_y.end(), m.pos_y.begin(), m.pos_y.end());
            out.pos_z.insert(out.pos_z.end(), m.pos_z.begin(), m.pos_z.end());
            out.norm_x.insert(out.norm_x.end(), m.norm_x.begin(), m.norm_x.end());
            out.norm_y.insert(out.norm_y.end(), m.norm_y.begin(), m.norm_y.end());
            out.norm_z.insert(out.norm_z.end(), m.norm_z.begin(), m.norm_z.end());
            for (uint32_t i : m.indices) out.indices.push_back(base + i);
        }
        out.build_adjacency();
    }
    if (parts) {
        parts->clear();
        if (made.size() >= 2) *parts = std::move(made);
    }
    std::printf("[glb] loaded %u verts, %u tris from %s (%d primitive(s) skipped%s%s)\n",
                out.vertex_count(), out.tri_count(), filename, skipped,
                note.empty() ? "" : ", ", note.c_str());
    if (err) *err = note;
    return true;
}
