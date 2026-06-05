#include "project_file_v1.h"
#include <fstream>
#include <cstring>
#include <vector>

// --- Legacy (.chisel VERSION 1) loader — REMOVABLE MODULE ---
//
// Fully self-contained: duplicates the small chunk parser instead of sharing
// helpers with project_file.cpp, so deleting this file removes v1 support with
// no ripple. See project_file_v1.h for the removal checklist.

namespace {

constexpr uint32_t V1_MAGIC   = 0x4C534843; // "CHSL"
constexpr uint32_t V1_VERSION = 1;

struct ChunkHeader {
    char     tag[4];
    uint32_t size;
};

class ChunkReader {
    const char* p;
    const char* end;
public:
    ChunkReader(const std::vector<char>& buf)
        : p(buf.data()), end(buf.data() + buf.size()) {}

    size_t remaining() const { return (size_t)(end - p); }

    bool read_raw(void* dst, size_t n) {
        if (remaining() < n) return false;
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }
    bool read_u32(uint32_t& v) { return read_raw(&v, 4); }
    bool read_i32(int32_t& v)  { return read_raw(&v, 4); }
    bool read_u8(uint8_t& v)   { return read_raw(&v, 1); }

    bool read_float_vec(std::vector<float>& out, size_t n) {
        out.resize(n);
        return n == 0 || read_raw(out.data(), n * sizeof(float));
    }
    bool read_u32_vec(std::vector<uint32_t>& out, size_t n) {
        out.resize(n);
        return n == 0 || read_raw(out.data(), n * sizeof(uint32_t));
    }
};

bool read_mesh_chunk(ChunkReader& r, Mesh& m) {
    uint32_t vc, tc;
    if (!r.read_u32(vc) || !r.read_u32(tc)) return false;
    if (!r.read_float_vec(m.pos_x, vc)) return false;
    if (!r.read_float_vec(m.pos_y, vc)) return false;
    if (!r.read_float_vec(m.pos_z, vc)) return false;
    if (!r.read_u32_vec(m.indices, (size_t)tc * 3)) return false;
    uint32_t mc;
    if (!r.read_u32(mc)) return false;
    if (mc > 0) {
        if (!r.read_float_vec(m.mask, mc)) return false;
    } else {
        m.mask.clear();
    }
    m.norm_x.resize(vc, 0.0f); m.norm_y.resize(vc, 0.0f); m.norm_z.resize(vc, 0.0f);
    m.mirror_x_map.clear();
    m.vert_tri_offset.clear(); m.vert_tri_list.clear();
    return true;
}

bool read_camr_chunk(ChunkReader& r, Camera& c) {
    float buf[7];
    if (!r.read_raw(buf, sizeof(buf))) return false;
    c.target = { buf[0], buf[1], buf[2] };
    c.distance = buf[3];
    c.yaw   = buf[4];
    c.pitch = buf[5];
    c.fov   = buf[6];
    return true;
}

bool read_opts_chunk(ChunkReader& r, bool& mirror_x, int& subdiv_level) {
    uint8_t mx;
    int32_t sl;
    if (!r.read_u8(mx) || !r.read_i32(sl)) return false;
    mirror_x = (mx != 0);
    subdiv_level = sl;
    return true;
}

bool read_mres_chunk(ChunkReader& r, MultiresStack& s) {
    uint8_t locked_byte;
    if (!r.read_u8(locked_byte)) return false;
    s.locked = (locked_byte != 0);

    int32_t bl, cl;
    if (!r.read_i32(bl) || !r.read_i32(cl)) return false;
    s.base_level = bl;
    s.current_level = cl;

    uint32_t bvc, btc;
    if (!r.read_u32(bvc) || !r.read_u32(btc)) return false;
    if (!r.read_float_vec(s.base.pos_x, bvc)) return false;
    if (!r.read_float_vec(s.base.pos_y, bvc)) return false;
    if (!r.read_float_vec(s.base.pos_z, bvc)) return false;
    if (!r.read_u32_vec(s.base.indices, (size_t)btc * 3)) return false;
    s.base.norm_x.resize(bvc, 0.0f); s.base.norm_y.resize(bvc, 0.0f); s.base.norm_z.resize(bvc, 0.0f);
    s.base.mask.clear();
    s.base.mirror_x_map.clear();
    s.base.vert_tri_offset.clear(); s.base.vert_tri_list.clear();

    uint32_t num_layers;
    if (!r.read_u32(num_layers)) return false;
    s.disp.resize(num_layers);
    s.frames.resize(num_layers);
    s.mirror.resize(num_layers);
    for (uint32_t k = 0; k < num_layers; k++) {
        uint32_t nv;
        if (!r.read_u32(nv)) return false;
        s.disp[k].resize(nv);
        if (nv > 0 && !r.read_raw(s.disp[k].data(), nv * sizeof(Vec3))) return false;
        s.frames[k].clear();
        s.mirror[k].clear();
    }
    s.base_mirror.clear();
    return true;
}

} // namespace

LoadResult load_project_v1(const char* path, ProjectData& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return LoadResult::ERR_OPEN;

    auto file_size = f.tellg();
    if (file_size < 8) return LoadResult::ERR_CORRUPT;
    f.seekg(0);

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != V1_MAGIC)     return LoadResult::ERR_MAGIC;
    if (version != V1_VERSION)  return LoadResult::ERR_VERSION;

    // Single entity rehydrated from the v1 chunks.
    EntityRecord rec;
    bool has_mesh = false;
    bool has_mres = false;

    while (f.tellg() < file_size) {
        ChunkHeader hdr;
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f.good()) break;

        if ((std::streamoff)f.tellg() + hdr.size > file_size)
            return LoadResult::ERR_CORRUPT;

        std::vector<char> payload(hdr.size);
        if (hdr.size > 0) {
            f.read(payload.data(), hdr.size);
            if (!f.good()) return LoadResult::ERR_READ;
        }
        ChunkReader cr(payload);

        if (std::memcmp(hdr.tag, "MESH", 4) == 0) {
            if (!read_mesh_chunk(cr, rec.mesh)) return LoadResult::ERR_CORRUPT;
            has_mesh = true;
        } else if (std::memcmp(hdr.tag, "CAMR", 4) == 0) {
            if (!read_camr_chunk(cr, data.camera)) return LoadResult::ERR_CORRUPT;
        } else if (std::memcmp(hdr.tag, "OPTS", 4) == 0) {
            if (!read_opts_chunk(cr, data.mirror_x, data.subdiv_level)) return LoadResult::ERR_CORRUPT;
        } else if (std::memcmp(hdr.tag, "MRES", 4) == 0) {
            if (!read_mres_chunk(cr, rec.multires)) return LoadResult::ERR_CORRUPT;
            has_mres = true;
        }
        // Unknown chunks: silently skipped.
    }

    if (!has_mesh) return LoadResult::ERR_CORRUPT;
    (void)has_mres;

    rec.id           = 1;
    rec.subdiv_level = (uint32_t)data.subdiv_level;

    data.entities.clear();
    data.entities.push_back(std::move(rec));
    data.active_id    = 1;
    data.selected_ids = { 1 };
    data.next_id      = 2;
    // v1 single-mesh load drove mirror topology from the multires lock state.
    data.mirror_use_topology = data.entities.front().multires.locked;
    return LoadResult::OK;
}
