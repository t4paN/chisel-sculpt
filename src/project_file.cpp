#include "project_file.h"
#include <fstream>
#include <cstring>
#include <vector>

static constexpr uint32_t MAGIC   = 0x4C534843; // "CHSL" little-endian
static constexpr uint32_t VERSION = 1;

struct ChunkHeader {
    char     tag[4];
    uint32_t size;
};

static bool write_raw(std::ofstream& f, const void* data, size_t n) {
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)n);
    return f.good();
}

static bool write_u32(std::ofstream& f, uint32_t v) { return write_raw(f, &v, 4); }
static bool write_i32(std::ofstream& f, int32_t v)  { return write_raw(f, &v, 4); }
static bool write_u8(std::ofstream& f, uint8_t v)   { return write_raw(f, &v, 1); }
static bool write_floats(std::ofstream& f, const float* p, size_t n) {
    return write_raw(f, p, n * sizeof(float));
}

static void write_chunk_begin(std::ofstream& f, const char tag[4], std::streampos& size_pos) {
    f.write(tag, 4);
    size_pos = f.tellp();
    uint32_t placeholder = 0;
    f.write(reinterpret_cast<const char*>(&placeholder), 4);
}

static void write_chunk_end(std::ofstream& f, std::streampos size_pos) {
    auto end = f.tellp();
    uint32_t size = (uint32_t)(end - size_pos - 4);
    f.seekp(size_pos);
    f.write(reinterpret_cast<const char*>(&size), 4);
    f.seekp(end);
}

static void write_mesh_chunk(std::ofstream& f, const Mesh& m) {
    std::streampos sp;
    write_chunk_begin(f, "MESH", sp);
    write_u32(f, m.vertex_count());
    write_u32(f, m.tri_count());
    write_floats(f, m.pos_x.data(), m.vertex_count());
    write_floats(f, m.pos_y.data(), m.vertex_count());
    write_floats(f, m.pos_z.data(), m.vertex_count());
    write_raw(f, m.indices.data(), m.indices.size() * sizeof(uint32_t));
    uint32_t mc = (uint32_t)m.mask.size();
    write_u32(f, mc);
    if (mc > 0) write_floats(f, m.mask.data(), mc);
    write_chunk_end(f, sp);
}

static void write_camr_chunk(std::ofstream& f, const Camera& c) {
    std::streampos sp;
    write_chunk_begin(f, "CAMR", sp);
    float buf[7] = { c.target.x, c.target.y, c.target.z,
                     c.distance, c.yaw, c.pitch, c.fov };
    write_raw(f, buf, sizeof(buf));
    write_chunk_end(f, sp);
}

static void write_opts_chunk(std::ofstream& f, bool mirror_x, int subdiv_level) {
    std::streampos sp;
    write_chunk_begin(f, "OPTS", sp);
    write_u8(f, mirror_x ? 1 : 0);
    write_i32(f, subdiv_level);
    write_chunk_end(f, sp);
}

static void write_mres_chunk(std::ofstream& f, const MultiresStack& s) {
    if (!s.locked) return;
    std::streampos sp;
    write_chunk_begin(f, "MRES", sp);
    write_u8(f, 1);
    write_i32(f, s.base_level);
    write_i32(f, s.current_level);
    write_u32(f, s.base.vertex_count());
    write_u32(f, s.base.tri_count());
    write_floats(f, s.base.pos_x.data(), s.base.vertex_count());
    write_floats(f, s.base.pos_y.data(), s.base.vertex_count());
    write_floats(f, s.base.pos_z.data(), s.base.vertex_count());
    write_raw(f, s.base.indices.data(), s.base.indices.size() * sizeof(uint32_t));
    uint32_t num_layers = (uint32_t)s.disp.size();
    write_u32(f, num_layers);
    for (uint32_t k = 0; k < num_layers; k++) {
        uint32_t nv = (uint32_t)s.disp[k].size();
        write_u32(f, nv);
        if (nv > 0)
            write_raw(f, s.disp[k].data(), nv * sizeof(Vec3));
    }
    write_chunk_end(f, sp);
}

SaveResult save_project(const char* path, const ProjectData& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return SaveResult::ERR_OPEN;

    write_raw(f, &MAGIC, 4);
    write_raw(f, &VERSION, 4);

    write_mesh_chunk(f, data.mesh);
    write_camr_chunk(f, data.camera);
    write_opts_chunk(f, data.mirror_x, data.subdiv_level);
    write_mres_chunk(f, data.multires);

    if (!f.good()) return SaveResult::ERR_WRITE;
    return SaveResult::OK;
}

// --- Loading ---

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

static bool read_mesh_chunk(ChunkReader& r, Mesh& m) {
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

static bool read_camr_chunk(ChunkReader& r, Camera& c) {
    float buf[7];
    if (!r.read_raw(buf, sizeof(buf))) return false;
    c.target = { buf[0], buf[1], buf[2] };
    c.distance = buf[3];
    c.yaw   = buf[4];
    c.pitch = buf[5];
    c.fov   = buf[6];
    return true;
}

static bool read_opts_chunk(ChunkReader& r, bool& mirror_x, int& subdiv_level) {
    uint8_t mx;
    int32_t sl;
    if (!r.read_u8(mx) || !r.read_i32(sl)) return false;
    mirror_x = (mx != 0);
    subdiv_level = sl;
    return true;
}

static bool read_mres_chunk(ChunkReader& r, MultiresStack& s) {
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

LoadResult load_project(const char* path, ProjectData& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return LoadResult::ERR_OPEN;

    auto file_size = f.tellg();
    if (file_size < 8) return LoadResult::ERR_CORRUPT;
    f.seekg(0);

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != MAGIC) return LoadResult::ERR_MAGIC;
    if (version != VERSION) return LoadResult::ERR_VERSION;

    bool has_mesh = false, has_camr = false, has_opts = false, has_mres = false;

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
            if (!read_mesh_chunk(cr, data.mesh)) return LoadResult::ERR_CORRUPT;
            has_mesh = true;
        } else if (std::memcmp(hdr.tag, "CAMR", 4) == 0) {
            if (!read_camr_chunk(cr, data.camera)) return LoadResult::ERR_CORRUPT;
            has_camr = true;
        } else if (std::memcmp(hdr.tag, "OPTS", 4) == 0) {
            if (!read_opts_chunk(cr, data.mirror_x, data.subdiv_level)) return LoadResult::ERR_CORRUPT;
            has_opts = true;
        } else if (std::memcmp(hdr.tag, "MRES", 4) == 0) {
            if (!read_mres_chunk(cr, data.multires)) return LoadResult::ERR_CORRUPT;
            has_mres = true;
        }
        // Unknown chunks: silently skipped
    }

    if (!has_mesh) return LoadResult::ERR_CORRUPT;
    (void)has_camr; (void)has_opts; (void)has_mres;
    return LoadResult::OK;
}

const char* result_string(SaveResult r) {
    switch (r) {
        case SaveResult::OK:        return "OK";
        case SaveResult::ERR_OPEN:  return "Could not open file for writing";
        case SaveResult::ERR_WRITE: return "Write error";
    }
    return "Unknown error";
}

const char* result_string(LoadResult r) {
    switch (r) {
        case LoadResult::OK:          return "OK";
        case LoadResult::ERR_OPEN:    return "Could not open file";
        case LoadResult::ERR_READ:    return "Read error";
        case LoadResult::ERR_MAGIC:   return "Not a .chisel file";
        case LoadResult::ERR_VERSION: return "Unsupported file version";
        case LoadResult::ERR_CORRUPT: return "File is corrupt or truncated";
    }
    return "Unknown error";
}
