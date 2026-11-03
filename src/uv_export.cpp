#include "uv_export.h"
#include "retopo.h"
#include "ui_skin.h"          // uv_island_color: the picture's islands wear the panel's shades
#include "stb/stb_image_write.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>

// ---- The UV map picture ---------------------------------------------------------------

std::vector<uint8_t> uv_map_rgba(const std::vector<UvExportPart>& parts, int size,
                                 float wire_px, bool fill) {
    size = std::max(64, std::min(8192, size));
    const size_t n = (size_t)size * size;
    std::vector<uint8_t> out(n * 4, 0);
    const float S = (float)size;
    auto px = [&](const float* uv, float& x, float& y) { x = uv[0] * S; y = (1.0f - uv[1]) * S; };

    // Islands: each face fanned, a pixel filled where its centre falls inside. They don't
    // overlap in a packed layout, so the last one simply wins where they would.
    if (fill) {
        for (const UvExportPart& p : parts) {
            if (!p.uv || !p.uv->fits(*p.rm)) continue;
            const RetopoMesh& rm = *p.rm;
            const UvLayout& L = *p.uv;
            for (uint32_t f = 0; f < rm.face_count(); f++) {
                const uint32_t chart = L.face_chart[f];
                if (chart >= L.charts) continue;
                const uint32_t col = uv_island_color(p.id, chart, 0.55f);   // RGBA8, R lowest
                const uint32_t o = rm.face_off[f], k = rm.face_size(f);
                for (uint32_t j = 1; j + 1 < k; j++) {
                    float x[3], y[3];
                    px(&L.uv[2 * o], x[0], y[0]);
                    px(&L.uv[2 * (o + j)], x[1], y[1]);
                    px(&L.uv[2 * (o + j + 1)], x[2], y[2]);
                    const float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
                    if (std::fabs(area) < 1e-12f) continue;
                    const int x0 = std::max(0, (int)std::floor(std::min({x[0], x[1], x[2]})));
                    const int x1 = std::min(size - 1, (int)std::ceil(std::max({x[0], x[1], x[2]})));
                    const int y0 = std::max(0, (int)std::floor(std::min({y[0], y[1], y[2]})));
                    const int y1 = std::min(size - 1, (int)std::ceil(std::max({y[0], y[1], y[2]})));
                    const float inv = 1.0f / area;
                    for (int yy = y0; yy <= y1; yy++)
                        for (int xx = x0; xx <= x1; xx++) {
                            const float cx = xx + 0.5f, cy = yy + 0.5f;
                            const float b0 = ((x[1] - cx) * (y[2] - cy) - (x[2] - cx) * (y[1] - cy)) * inv;
                            const float b1 = ((x[2] - cx) * (y[0] - cy) - (x[0] - cx) * (y[2] - cy)) * inv;
                            if (b0 < 0.0f || b1 < 0.0f || b0 + b1 > 1.0f) continue;
                            std::memcpy(&out[4 * ((size_t)yy * size + xx)], &col, 4);
                        }
                }
            }
        }
    }

    // Wire: every face edge, antialiased by distance to the segment. Coverage keeps the
    // most any edge gave a pixel, so an edge shared by two faces isn't drawn darker.
    std::vector<uint8_t> cov(n, 0);
    const float half = std::max(0.25f, wire_px * 0.5f);
    for (const UvExportPart& p : parts) {
        if (!p.uv || !p.uv->fits(*p.rm)) continue;
        const RetopoMesh& rm = *p.rm;
        const UvLayout& L = *p.uv;
        for (uint32_t f = 0; f < rm.face_count(); f++) {
            if (L.face_chart[f] >= L.charts) continue;
            const uint32_t o = rm.face_off[f], k = rm.face_size(f);
            for (uint32_t j = 0; j < k; j++) {
                float ax, ay, bx, by;
                px(&L.uv[2 * (o + j)], ax, ay);
                px(&L.uv[2 * (o + (j + 1) % k)], bx, by);
                const float dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
                const int x0 = std::max(0, (int)std::floor(std::min(ax, bx) - half - 1.0f));
                const int x1 = std::min(size - 1, (int)std::ceil(std::max(ax, bx) + half + 1.0f));
                const int y0 = std::max(0, (int)std::floor(std::min(ay, by) - half - 1.0f));
                const int y1 = std::min(size - 1, (int)std::ceil(std::max(ay, by) + half + 1.0f));
                for (int yy = y0; yy <= y1; yy++)
                    for (int xx = x0; xx <= x1; xx++) {
                        const float cx = xx + 0.5f - ax, cy = yy + 0.5f - ay;
                        const float t = len2 > 0.0f ? std::max(0.0f, std::min(1.0f, (cx * dx + cy * dy) / len2)) : 0.0f;
                        const float ex = cx - dx * t, ey = cy - dy * t;
                        const float c = std::max(0.0f, std::min(1.0f, half + 0.5f - std::sqrt(ex * ex + ey * ey)));
                        uint8_t& w = cov[(size_t)yy * size + xx];
                        w = std::max(w, (uint8_t)std::lround(c * 255.0f));
                    }
            }
        }
    }
    // The wire over the islands (straight alpha "over").
    const float wc = 22.0f;
    for (size_t i = 0; i < n; i++) {
        if (!cov[i]) continue;
        uint8_t* q = &out[4 * i];
        const float aw = cov[i] / 255.0f, ab = q[3] / 255.0f;
        const float ao = aw + ab * (1.0f - aw);
        for (int c = 0; c < 3; c++)
            q[c] = (uint8_t)std::lround((wc * aw + q[c] * ab * (1.0f - aw)) / ao);
        q[3] = (uint8_t)std::lround(ao * 255.0f);
    }
    return out;
}

bool uv_map_write_png(const std::vector<UvExportPart>& parts, int size, float wire_px,
                      bool fill, const char* path) {
    size = std::max(64, std::min(8192, size));
    const std::vector<uint8_t> px = uv_map_rgba(parts, size, wire_px, fill);
    return stbi_write_png(path, size, size, 4, px.data(), size * 4) != 0;
}

// ---- GLB ------------------------------------------------------------------------------

namespace {
struct View { size_t off, len; int target; };   // target 0 = none (an image)
struct Acc { int view; int comp; uint32_t count; const char* type; bool minmax; float lo[3], hi[3]; };

void pad4(std::vector<uint8_t>& b, uint8_t with) { while (b.size() % 4) b.push_back(with); }

int add_view(std::vector<uint8_t>& bin, std::vector<View>& views, const void* p, size_t len, int target) {
    pad4(bin, 0);
    views.push_back({bin.size(), len, target});
    bin.insert(bin.end(), (const uint8_t*)p, (const uint8_t*)p + len);
    return (int)views.size() - 1;
}

// JSON string body: the names come from the user.
std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) o += ' ';
        else o += (char)c;
    }
    return o;
}
}  // namespace

bool glb_write_lows(const char* path, const std::vector<UvExportPart>& parts,
                    const std::vector<uint8_t>* color_png, const std::vector<uint8_t>* normal_png) {
    std::vector<uint8_t> bin;
    std::vector<View> views;
    std::vector<Acc> accs;
    std::string meshes, nodes, scene_nodes;
    if (color_png && color_png->empty()) color_png = nullptr;
    if (normal_png && normal_png->empty()) normal_png = nullptr;
    const bool material = color_png || normal_png;

    int written = 0;
    for (const UvExportPart& p : parts) {
        const RetopoMesh& rm = *p.rm;
        if (rm.faces.empty()) continue;
        const UvLayout* L = p.uv && p.uv->fits(rm) ? p.uv : nullptr;
        std::vector<Vec3> vn;
        retopo_smooth_normals(rm, vn);
        // One glTF vertex per (vertex, UV) pair: a seam vertex sits in two islands.
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> at;
        std::vector<float> pos, nrm, tex;
        std::vector<uint32_t> idx;
        auto corner = [&](uint32_t k) {
            const uint32_t v = rm.faces[k];
            uint32_t bu = 0, bv = 0;
            if (L) {
                std::memcpy(&bu, &L->uv[2 * k], 4);
                std::memcpy(&bv, &L->uv[2 * k + 1], 4);
            }
            const auto key = std::make_tuple(v, bu, bv);
            auto it = at.find(key);
            if (it != at.end()) return it->second;
            const uint32_t i = (uint32_t)(pos.size() / 3);
            at.emplace(key, i);
            pos.insert(pos.end(), {rm.v[v].x, rm.v[v].y, rm.v[v].z});
            nrm.insert(nrm.end(), {vn[v].x, vn[v].y, vn[v].z});
            if (L) tex.insert(tex.end(), {L->uv[2 * k], 1.0f - L->uv[2 * k + 1]});   // glTF's v runs down
            return i;
        };
        for (uint32_t f = 0; f < rm.face_count(); f++) {
            const uint32_t o = rm.face_off[f], k = rm.face_size(f);
            for (uint32_t j = 1; j + 1 < k; j++) {
                idx.push_back(corner(o));
                idx.push_back(corner(o + j));
                idx.push_back(corner(o + j + 1));
            }
        }
        if (idx.empty()) continue;
        const uint32_t nvtx = (uint32_t)(pos.size() / 3);
        Acc a{};
        a.view = add_view(bin, views, pos.data(), pos.size() * 4, 34962);
        a.comp = 5126; a.count = nvtx; a.type = "VEC3"; a.minmax = true;
        for (int c = 0; c < 3; c++) { a.lo[c] = 1e30f; a.hi[c] = -1e30f; }
        for (uint32_t i = 0; i < nvtx; i++)
            for (int c = 0; c < 3; c++) { a.lo[c] = std::min(a.lo[c], pos[3 * i + c]); a.hi[c] = std::max(a.hi[c], pos[3 * i + c]); }
        const int a_pos = (int)accs.size(); accs.push_back(a);
        a = Acc{};
        a.view = add_view(bin, views, nrm.data(), nrm.size() * 4, 34962);
        a.comp = 5126; a.count = nvtx; a.type = "VEC3";
        const int a_nrm = (int)accs.size(); accs.push_back(a);
        int a_tex = -1;
        if (L) {
            a = Acc{};
            a.view = add_view(bin, views, tex.data(), tex.size() * 4, 34962);
            a.comp = 5126; a.count = nvtx; a.type = "VEC2";
            a_tex = (int)accs.size(); accs.push_back(a);
        }
        a = Acc{};
        a.view = add_view(bin, views, idx.data(), idx.size() * 4, 34963);
        a.comp = 5125; a.count = (uint32_t)idx.size(); a.type = "SCALAR";
        const int a_idx = (int)accs.size(); accs.push_back(a);

        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "%s{\"name\":\"%s-LO\",\"primitives\":[{\"attributes\":{\"POSITION\":%d,\"NORMAL\":%d",
                      written ? "," : "", esc(p.name).c_str(), a_pos, a_nrm);
        meshes += buf;
        if (a_tex >= 0) { std::snprintf(buf, sizeof buf, ",\"TEXCOORD_0\":%d", a_tex); meshes += buf; }
        std::snprintf(buf, sizeof buf, "},\"indices\":%d,\"mode\":4%s}]}", a_idx, material ? ",\"material\":0" : "");
        meshes += buf;
        std::snprintf(buf, sizeof buf, "%s{\"name\":\"%s-LO\",\"mesh\":%d}", written ? "," : "", esc(p.name).c_str(), written);
        nodes += buf;
        std::snprintf(buf, sizeof buf, "%s%d", written ? "," : "", written);
        scene_nodes += buf;
        written++;
    }
    if (!written) return false;

    // The textures ride in the binary chunk as PNGs.
    std::string images, textures, mat;
    int n_img = 0, tex_color = -1, tex_normal = -1;
    const std::vector<uint8_t>* pngs[2] = {color_png, normal_png};
    for (int which = 0; which < 2; which++) {
        const std::vector<uint8_t>* png = pngs[which];
        if (!png) continue;
        const int v = add_view(bin, views, png->data(), png->size(), 0);
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s{\"bufferView\":%d,\"mimeType\":\"image/png\"}", n_img ? "," : "", v);
        images += buf;
        std::snprintf(buf, sizeof buf, "%s{\"sampler\":0,\"source\":%d}", n_img ? "," : "", n_img);
        textures += buf;
        (which == 0 ? tex_color : tex_normal) = n_img++;
    }
    if (material) {
        char buf[320];
        mat = "{\"name\":\"baked\",\"pbrMetallicRoughness\":{";
        if (tex_color >= 0) { std::snprintf(buf, sizeof buf, "\"baseColorTexture\":{\"index\":%d},", tex_color); mat += buf; }
        mat += "\"metallicFactor\":0,\"roughnessFactor\":1}";
        if (tex_normal >= 0) { std::snprintf(buf, sizeof buf, ",\"normalTexture\":{\"index\":%d}", tex_normal); mat += buf; }
        mat += "}";
    }

    std::string json = "{\"asset\":{\"version\":\"2.0\",\"generator\":\"Chisel\"},\"scene\":0,\"scenes\":[{\"nodes\":[" + scene_nodes
                     + "]}],\"nodes\":[" + nodes + "],\"meshes\":[" + meshes + "]";
    char buf[320];
    json += ",\"accessors\":[";
    for (size_t i = 0; i < accs.size(); i++) {
        const Acc& a = accs[i];
        std::snprintf(buf, sizeof buf, "%s{\"bufferView\":%d,\"componentType\":%d,\"count\":%u,\"type\":\"%s\"",
                      i ? "," : "", a.view, a.comp, a.count, a.type);
        json += buf;
        if (a.minmax) {
            std::snprintf(buf, sizeof buf, ",\"min\":[%.7g,%.7g,%.7g],\"max\":[%.7g,%.7g,%.7g]",
                          a.lo[0], a.lo[1], a.lo[2], a.hi[0], a.hi[1], a.hi[2]);
            json += buf;
        }
        json += "}";
    }
    json += "],\"bufferViews\":[";
    for (size_t i = 0; i < views.size(); i++) {
        std::snprintf(buf, sizeof buf, "%s{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu", i ? "," : "",
                      views[i].off, views[i].len);
        json += buf;
        if (views[i].target) { std::snprintf(buf, sizeof buf, ",\"target\":%d", views[i].target); json += buf; }
        json += "}";
    }
    pad4(bin, 0);
    std::snprintf(buf, sizeof buf, "],\"buffers\":[{\"byteLength\":%zu}]", bin.size());
    json += buf;
    if (material) {
        json += ",\"materials\":[" + mat + "],\"images\":[" + images + "],\"textures\":[" + textures
              + "],\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}]";
    }
    json += "}";
    while (json.size() % 4) json += ' ';

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    auto u32 = [&](uint32_t x) { std::fwrite(&x, 4, 1, f); };
    u32(0x46546C67u);   // "glTF"
    u32(2);
    u32((uint32_t)(12 + 8 + json.size() + 8 + bin.size()));
    u32((uint32_t)json.size());
    u32(0x4E4F534Au);   // "JSON"
    std::fwrite(json.data(), 1, json.size(), f);
    u32((uint32_t)bin.size());
    u32(0x004E4942u);   // "BIN"
    std::fwrite(bin.data(), 1, bin.size(), f);
    return std::fclose(f) == 0;
}
