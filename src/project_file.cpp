#include "project_file.h"
#include "project_file_v1.h"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// Current on-disk format. Bump VERSION whenever the v2 layout below changes;
// older versions are handled by their own decoupled reader modules (see
// project_file_v1.*) so legacy support can be deleted cleanly later.
static constexpr uint32_t MAGIC   = 0x4C534843; // "CHSL" little-endian
// v3 adds per-vertex paint colour after the mask in every mesh body (working +
// multires base). v2 files are still read by the same loader with has_color=false.
// v4: identical byte layout to v3; the version bump marks that multires disp
// layers are encoded under the CANONICAL (sorted-edge-key) midpoint numbering
// of loop_subdivide, which is platform-independent. v<=3 disp data used the
// stdlib hash-map iteration order of whatever platform saved the file.
// v5 appends the finest-level paint planes (colour count+data, mask
// count+data) to each locked multires body, after the disp layers — full
// paint/mask fidelity across levels survives save/load. The v3 base-colour
// field remains in the layout but is written empty (paint lives in the
// planes now). Older files load with empty planes; scene.cpp initialises
// them from the cached working surface (current-level fidelity).
// v6 appends the remesh-density field: per-vertex f32 in each mesh body and a
// finest-level density plane in each locked multires body (both count+data,
// empty when unpainted). Older files load with no field (= neutral 0.5).
// v7 stops writing data the loader throws away. Each mesh body now opens with
// a u8 `derived` flag: 1 means the positions and indices are omitted because
// the entity's locked multires stack regenerates them (scene.cpp cascades the
// stack over the loaded mesh, which never reads the incoming surface). Counts
// are still written either way. Independently, arrays whose every element is
// the neutral value are written as count 0 — an all-zero mask is exactly an
// absent mask, and an all-0.5 density field is exactly an absent one. Together
// these cut typical files by 45-73% losslessly. Older files are unaffected;
// old builds refuse v7 cleanly via the version check.
//
// Paint layers (2026-10-18) did NOT bump the version. A layered entity's colour slots
// still hold the blended picture, so any v7 reader shows it right, and the layers ride
// in an optional LAYR chunk written after the entity's ENTY. Readers that predate it
// skip the unknown chunk and load the flattened picture; this one swaps the layers
// back in (see write_layr_chunk). The chunk carries its own format byte for growth.
static constexpr uint32_t VERSION = 7;
static constexpr uint8_t  LAYR_FORMAT = 1;
// Retopo (2026-10-23) didn't bump VERSION either: it's an optional RTPO chunk after
// the entities, skipped by readers that predate it (the sculpt loads, the retopo is
// just not there). Its own format byte is for growth: a reader takes formats up to
// its own and fills what an older one lacks with defaults. Format 2 (2026-10-25) adds
// the vertex mask; a format 1 file loads with nothing masked.
static constexpr uint8_t  RTPO_FORMAT = 2;

struct ChunkHeader {
    char     tag[4];
    uint32_t size;
};

// ------------------------------------------------------------------ writing --

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

// v7: an array every element of which is the field's neutral value carries no
// information — the loader treats absent and neutral identically — so it is
// written as count 0. Exact compare, no epsilon: we wrote these values.
static bool all_equal(const std::vector<float>& a, float v) {
    for (float x : a) if (x != v) return false;
    return true;
}

// Mesh / multires *bodies*: the raw field layout with no chunk header, so they
// can be inlined inside an ENTY payload (and reused field-for-field by the v1
// reader's expectations).
//
// v7: `derived` omits the positions and indices — set only when the entity's
// multires stack is locked, in which case scene.cpp regenerates the surface by
// cascading that stack and the stored copy would be dropped on the floor.
// `may_drop_mask` gates the blank-mask elision (see write_enty_chunk).
// A layered entity saves its COMPOSITE in every colour slot, so a reader that doesn't
// know LAYR still shows the right picture (as one flattened layer). The layers
// themselves, and the active layer's own pixels for these same three slots, go in the
// LAYR chunk. Every colour array here is a prefix of one vertex numbering (working
// level, base cage, finest plane), so one stack composites them all.
static const std::vector<uint32_t>& paint_for_file(const PaintLayerStack& layers,
                                                   const std::vector<uint32_t>& act,
                                                   std::vector<uint32_t>& scratch) {
    if (layers.empty() || act.empty()) return act;
    scratch.resize(act.size());
    paint_composite(layers, act, (uint32_t)act.size(), scratch.data());
    return scratch;
}

static void write_mesh_body(std::ofstream& f, const Mesh& m, bool derived,
                            bool may_drop_mask) {
    write_u8(f, derived ? 1 : 0);
    write_u32(f, m.vertex_count());
    write_u32(f, m.tri_count());
    if (!derived) {
        write_floats(f, m.pos_x.data(), m.vertex_count());
        write_floats(f, m.pos_y.data(), m.vertex_count());
        write_floats(f, m.pos_z.data(), m.vertex_count());
        write_raw(f, m.indices.data(), m.indices.size() * sizeof(uint32_t));
    }
    uint32_t mc = (uint32_t)m.mask.size();
    if (may_drop_mask && all_equal(m.mask, 0.0f)) mc = 0;
    write_u32(f, mc);
    if (mc > 0) write_floats(f, m.mask.data(), mc);
    // v3: packed-RGBA8 paint colour (count then data, empty when unpainted).
    // Never elided: "unpainted" and "painted white" are not the same thing to
    // every consumer (matcap tinting), and the measured files show no gain.
    std::vector<uint32_t> flat;
    const std::vector<uint32_t>& col = paint_for_file(m.paint_layers, m.color, flat);
    uint32_t cc = (uint32_t)col.size();
    write_u32(f, cc);
    if (cc > 0) write_raw(f, col.data(), cc * sizeof(uint32_t));
    // v6: remesh-density field (count then data, empty when unpainted).
    // Neutral is 0.5, not 0 — an all-zero density field is real paint.
    uint32_t dc = (uint32_t)m.density.size();
    if (all_equal(m.density, 0.5f)) dc = 0;
    write_u32(f, dc);
    if (dc > 0) write_floats(f, m.density.data(), dc);
}

static void write_multires_body(std::ofstream& f, const MultiresStack& s,
                                const PaintLayerStack& layers) {
    std::vector<uint32_t> flat;
    write_u8(f, s.locked ? 1 : 0);
    if (!s.locked) return;
    write_i32(f, s.base_level);
    write_i32(f, s.current_level);
    write_u32(f, s.base.vertex_count());
    write_u32(f, s.base.tri_count());
    write_floats(f, s.base.pos_x.data(), s.base.vertex_count());
    write_floats(f, s.base.pos_y.data(), s.base.vertex_count());
    write_floats(f, s.base.pos_z.data(), s.base.vertex_count());
    write_raw(f, s.base.indices.data(), s.base.indices.size() * sizeof(uint32_t));
    // v3: base-cage paint colour (count then data), so level switches after a
    // load re-derive painted colour up the cascade. Empty when unpainted.
    const std::vector<uint32_t>& bcol = paint_for_file(layers, s.base.color, flat);
    uint32_t bcc = (uint32_t)bcol.size();
    write_u32(f, bcc);
    if (bcc > 0) write_raw(f, bcol.data(), bcc * sizeof(uint32_t));
    uint32_t num_layers = (uint32_t)s.disp.size();
    write_u32(f, num_layers);
    for (uint32_t k = 0; k < num_layers; k++) {
        uint32_t nv = (uint32_t)s.disp[k].size();
        write_u32(f, nv);
        if (nv > 0)
            write_raw(f, s.disp[k].data(), nv * sizeof(Vec3));
    }
    // v5: finest-level paint planes (count then data, empty when unused).
    const std::vector<uint32_t>& pcol = paint_for_file(layers, s.color, flat);
    uint32_t pcc = (uint32_t)pcol.size();
    write_u32(f, pcc);
    if (pcc > 0) write_raw(f, pcol.data(), pcc * sizeof(uint32_t));
    // v7: a fully-zero mask plane is an unmasked plane — write nothing. Load
    // rebuilds an all-zero plane from the working mesh (or leaves it empty,
    // which cascades to the same unmasked surface).
    uint32_t pmc = (uint32_t)s.mask.size();
    if (all_equal(s.mask, 0.0f)) pmc = 0;
    write_u32(f, pmc);
    if (pmc > 0) write_floats(f, s.mask.data(), pmc);
    // v6: finest-level density plane. v7: elided when uniformly neutral.
    uint32_t pdc = (uint32_t)s.density.size();
    if (all_equal(s.density, 0.5f)) pdc = 0;
    write_u32(f, pdc);
    if (pdc > 0) write_floats(f, s.density.data(), pdc);
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

static void write_scen_chunk(std::ofstream& f, const ProjectData& d) {
    std::streampos sp;
    write_chunk_begin(f, "SCEN", sp);
    write_u32(f, d.active_id);
    write_u32(f, d.next_id);
    write_u8(f, d.mirror_use_topology ? 1 : 0);
    write_u32(f, (uint32_t)d.selected_ids.size());
    for (uint32_t id : d.selected_ids) write_u32(f, id);
    write_chunk_end(f, sp);
}

static void write_enty_chunk(std::ofstream& f, const EntityRecord& e) {
    std::streampos sp;
    write_chunk_begin(f, "ENTY", sp);
    write_u32(f, e.id);
    write_u32(f, e.subdiv_level);
    // A blank working mask may only be dropped when the finest-level mask
    // plane is blank too. On load an *absent* working mask means "mask
    // cleared" to multires_sync_paint, which wipes the whole plane; an
    // all-zero array that is present folds in as a no-op diff and leaves the
    // plane alone. The difference is real: masking a region finer than the
    // coarse vertex spacing, then switching down a level, leaves an all-zero
    // working mask over a plane that still holds the mask up top.
    const bool may_drop_mask = all_equal(e.multires.mask, 0.0f);
    write_mesh_body(f, e.mesh, e.multires.locked, may_drop_mask);
    write_multires_body(f, e.multires, e.mesh.paint_layers);
    write_chunk_end(f, sp);
}

// One layer's pixels: u8 dense, u32 vcount, u32 n, [n ids if packed], n rgba.
// vcount 0 = an empty array (unpainted), which is not the same as all-zero.
// `empty_slot` (the active layer) writes the flags with no pixels: they're elsewhere.
static void write_layer_pixels(std::ofstream& f, const PaintLayer& L, bool empty_slot = false) {
    const uint32_t n = empty_slot ? 0u : (uint32_t)L.rgba.size();
    write_u8(f, L.dense ? 1 : 0);
    write_u32(f, L.vcount);
    write_u32(f, n);
    if (n && !L.dense) write_raw(f, L.ids.data(), n * sizeof(uint32_t));
    if (n)             write_raw(f, L.rgba.data(), n * sizeof(uint32_t));
}

// A dense colour array, packed on the way out when that's smaller (paint is usually
// sparse on an upper layer). Packing only drops zero-alpha entries, which premultiplied
// paint makes all-zero, so unpacking gives back the same array.
static void write_dense_pixels(std::ofstream& f, const std::vector<uint32_t>& a) {
    PaintLayer L;
    L.dense  = true;
    L.vcount = (uint32_t)a.size();
    L.rgba   = a;
    paint_layer_pack(L);
    write_layer_pixels(f, L);
}

// LAYR: u32 entity id, u8 format, i32 active, u32 next_id, u32 layer count, then per
// layer (bottom first) u32 id, u32 name length + bytes, u8 visible, f32 opacity,
// u8 mode, u8 clip, pixels. The active layer's stack slot is empty by design (its
// pixels live in the colour arrays), so it's written empty and followed, after the
// last layer, by the active layer's raw pixels for the three colour slots the ENTY
// body filled with the composite: working mesh, finest plane, base cage.
static void write_layr_chunk(std::ofstream& f, const EntityRecord& e) {
    const PaintLayerStack& s = e.mesh.paint_layers;
    std::streampos sp;
    write_chunk_begin(f, "LAYR", sp);
    write_u32(f, e.id);
    write_u8(f, LAYR_FORMAT);
    write_i32(f, s.active);
    write_u32(f, s.next_id);
    write_u32(f, (uint32_t)s.layers.size());
    for (int i = 0; i < (int)s.layers.size(); i++) {
        const PaintLayer& L = s.layers[i];
        write_u32(f, L.id);
        write_u32(f, (uint32_t)L.name.size());
        write_raw(f, L.name.data(), L.name.size());
        write_u8(f, L.visible ? 1 : 0);
        write_raw(f, &L.opacity, 4);
        write_u8(f, (uint8_t)L.mode);
        write_u8(f, L.clip ? 1 : 0);
        write_layer_pixels(f, L, i == s.active);
    }
    write_dense_pixels(f, e.mesh.color);
    write_dense_pixels(f, e.multires.color);
    write_dense_pixels(f, e.multires.base.color);
    write_chunk_end(f, sp);
}

// RTPO: u8 format, u32 next_id, u8 built, then the strokes and the result.
// Stroke: u32 id, u32 name length + bytes, u8 flags (1 closed, 2 visible, 4 frozen),
// i32 rings, f32 density, f32 width, u32 point count, then xyz points, xyz normals,
// half widths. Result: u32 vertex count, xyz positions, xyz normals, u32 owner and u32
// mirror twin per vertex; u32 face count, u32 face_off (count + 1), u32 corner count +
// corners, u32 owner per face; (format 2) f32 mask per vertex. Edges, neighbours, outlines and holes are derived: the
// loader works them out again. The built faces are saved as they are, never re-derived
// from the strokes, so a later change to Build can't change an old file's low-poly.
static void write_vec3s(std::ofstream& f, const std::vector<Vec3>& v) {
    for (const Vec3& p : v) { write_raw(f, &p.x, 4); write_raw(f, &p.y, 4); write_raw(f, &p.z, 4); }
}
static void write_u32s(std::ofstream& f, const std::vector<uint32_t>& v) {
    if (!v.empty()) write_raw(f, v.data(), v.size() * 4);
}
static void write_rtpo_chunk(std::ofstream& f, const RetopoSnapshot& r) {
    std::streampos sp;
    write_chunk_begin(f, "RTPO", sp);
    write_u8(f, RTPO_FORMAT);
    write_u32(f, r.next_id);
    write_u8(f, r.built ? 1 : 0);
    write_u32(f, (uint32_t)r.strokes.size());
    for (const RetopoStroke& s : r.strokes) {
        write_u32(f, s.id);
        write_u32(f, (uint32_t)s.name.size());
        write_raw(f, s.name.data(), s.name.size());
        write_u8(f, (s.closed ? 1 : 0) | (s.visible ? 2 : 0) | (s.frozen ? 4 : 0));
        write_i32(f, s.rings);
        write_raw(f, &s.density, 4);
        write_raw(f, &s.width, 4);
        const size_t n = std::min({s.pts.size(), s.nrm.size(), s.half_w.size()});
        write_u32(f, (uint32_t)n);
        for (size_t i = 0; i < n; i++) { write_raw(f, &s.pts[i].x, 4); write_raw(f, &s.pts[i].y, 4); write_raw(f, &s.pts[i].z, 4); }
        for (size_t i = 0; i < n; i++) { write_raw(f, &s.nrm[i].x, 4); write_raw(f, &s.nrm[i].y, 4); write_raw(f, &s.nrm[i].z, 4); }
        write_floats(f, s.half_w.data(), n);
    }
    const RetopoMesh& m = r.result;
    write_u32(f, (uint32_t)m.v.size());
    write_vec3s(f, m.v);
    write_vec3s(f, m.n);
    write_u32s(f, m.v_owner);
    write_u32s(f, m.mirror);
    write_u32(f, m.face_count());
    write_u32s(f, m.face_off);
    write_u32(f, (uint32_t)m.faces.size());
    write_u32s(f, m.faces);
    write_u32s(f, m.face_owner);
    for (uint32_t i = 0; i < (uint32_t)m.v.size(); i++) { const float k = m.mask_at(i); write_raw(f, &k, 4); }
    write_chunk_end(f, sp);
}

SaveResult save_project(const char* path, const ProjectData& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return SaveResult::ERR_OPEN;

    write_raw(f, &MAGIC, 4);
    write_raw(f, &VERSION, 4);

    write_camr_chunk(f, data.camera);
    write_opts_chunk(f, data.mirror_x, data.subdiv_level);
    write_scen_chunk(f, data);
    for (const EntityRecord& e : data.entities) {
        write_enty_chunk(f, e);
        if (!e.mesh.paint_layers.empty()) write_layr_chunk(f, e);
    }
    if (!data.retopo.strokes.empty() || data.retopo.result.face_count() > 0)
        write_rtpo_chunk(f, data.retopo);

    if (!f.good()) return SaveResult::ERR_WRITE;
    return SaveResult::OK;
}

// ------------------------------------------------------------------ reading --

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

// v7+ writes a leading `derived` byte (has_derived); when set, the surface is
// absent from the stream and the caller must regenerate it from the multires
// stack. `stored_vc` reports the vertex count the writer recorded — the only
// cross-check left on a derived body.
static bool read_mesh_body(ChunkReader& r, Mesh& m, bool has_color, bool has_density,
                           bool has_derived, bool& derived, uint32_t& stored_vc) {
    derived = false;
    if (has_derived) {
        uint8_t d;
        if (!r.read_u8(d)) return false;
        derived = (d != 0);
    }
    uint32_t vc, tc;
    if (!r.read_u32(vc) || !r.read_u32(tc)) return false;
    stored_vc = vc;
    if (derived) {
        // Leave the surface empty rather than zero-filling it: cascade_to_level
        // overwrites these arrays wholesale, vertex_count() reads pos_x.size(),
        // and a phantom-sized mesh would only hide bugs and waste RAM.
        m.pos_x.clear(); m.pos_y.clear(); m.pos_z.clear();
        m.indices.clear();
    } else {
        if (!r.read_float_vec(m.pos_x, vc)) return false;
        if (!r.read_float_vec(m.pos_y, vc)) return false;
        if (!r.read_float_vec(m.pos_z, vc)) return false;
        if (!r.read_u32_vec(m.indices, (size_t)tc * 3)) return false;
    }
    uint32_t mc;
    if (!r.read_u32(mc)) return false;
    if (mc > 0) {
        if (!r.read_float_vec(m.mask, mc)) return false;
    } else {
        m.mask.clear();
    }
    m.color.clear();
    if (has_color) {
        uint32_t cc;
        if (!r.read_u32(cc)) return false;
        if (cc > 0 && !r.read_u32_vec(m.color, cc)) return false;
    }
    m.density.clear();
    if (has_density) {
        uint32_t dc;
        if (!r.read_u32(dc)) return false;
        if (dc > 0 && !r.read_float_vec(m.density, dc)) return false;
    }
    const uint32_t nv = derived ? 0u : vc;
    m.norm_x.resize(nv, 0.0f); m.norm_y.resize(nv, 0.0f); m.norm_z.resize(nv, 0.0f);
    m.mirror_x_map.clear();
    m.vert_tri_offset.clear(); m.vert_tri_list.clear();
    return true;
}

static bool read_multires_body(ChunkReader& r, MultiresStack& s, bool has_color,
                               bool has_planes, bool has_density) {
    uint8_t locked_byte;
    if (!r.read_u8(locked_byte)) return false;
    s.locked = (locked_byte != 0);
    if (!s.locked) return true;

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
    s.base.color.clear();
    if (has_color) {
        uint32_t bcc;
        if (!r.read_u32(bcc)) return false;
        if (bcc > 0 && !r.read_u32_vec(s.base.color, bcc)) return false;
    }
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
    s.midpoint_parents.clear();
    // v5: finest-level paint planes. Non-empty planes must be sized for L_max
    // (= last disp layer's vertex count, or the base when no layers); anything
    // else is dropped — scene.cpp re-initialises planes from the cached
    // working surface, same as a pre-v5 load.
    s.color.clear();
    s.mask.clear();
    s.density.clear();
    if (has_planes) {
        const size_t v_top = num_layers ? s.disp[num_layers - 1].size() : (size_t)bvc;
        uint32_t pcc, pmc;
        if (!r.read_u32(pcc)) return false;
        if (pcc > 0 && !r.read_u32_vec(s.color, pcc)) return false;
        if (!r.read_u32(pmc)) return false;
        if (pmc > 0 && !r.read_float_vec(s.mask, pmc)) return false;
        if (!s.color.empty() && s.color.size() != v_top) s.color.clear();
        if (!s.mask.empty()  && s.mask.size()  != v_top) s.mask.clear();
        // v6: density plane, same size validation.
        if (has_density) {
            uint32_t pdc;
            if (!r.read_u32(pdc)) return false;
            if (pdc > 0 && !r.read_float_vec(s.density, pdc)) return false;
            if (!s.density.empty() && s.density.size() != v_top) s.density.clear();
        }
    }
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

static bool read_scen_chunk(ChunkReader& r, ProjectData& d) {
    uint8_t topo;
    uint32_t nsel;
    if (!r.read_u32(d.active_id) || !r.read_u32(d.next_id) ||
        !r.read_u8(topo) || !r.read_u32(nsel))
        return false;
    d.mirror_use_topology = (topo != 0);
    d.selected_ids.clear();
    d.selected_ids.reserve(nsel);
    for (uint32_t i = 0; i < nsel; i++) {
        uint32_t id;
        if (!r.read_u32(id)) return false;
        d.selected_ids.push_back(id);
    }
    return true;
}

static bool read_enty_chunk(ChunkReader& r, EntityRecord& e, bool has_color,
                            bool has_planes, bool has_density, bool has_derived) {
    if (!r.read_u32(e.id) || !r.read_u32(e.subdiv_level)) return false;
    bool derived = false;
    if (!read_mesh_body(r, e.mesh, has_color, has_density, has_derived,
                        derived, e.stored_vertex_count)) return false;
    if (!read_multires_body(r, e.multires, has_color, has_planes, has_density)) return false;
    // The honesty valve: a derived body promises the stack can regenerate it.
    // No stack, nothing to regenerate — corrupt rather than silently empty. A
    // future writer that can't make that promise just writes derived = 0.
    if (derived && !e.multires.locked) return false;
    return true;
}

// Pixels as write_layer_pixels laid them out. Every count is checked against what's
// left in the chunk BEFORE anything is allocated, so a damaged file can't ask for 4 GB.
static bool read_layer_pixels(ChunkReader& r, PaintLayer& L, bool empty_slot = false) {
    uint8_t dense;
    uint32_t vcount, n;
    if (!r.read_u8(dense) || !r.read_u32(vcount) || !r.read_u32(n)) return false;
    L.dense  = dense != 0;
    L.vcount = vcount;
    L.ids.clear();
    L.rgba.clear();
    if (empty_slot) return n == 0;
    if (n > vcount) return false;
    if (L.dense && n != vcount) return false;
    const size_t need = (size_t)n * (L.dense ? 4 : 8);
    if (need > r.remaining()) return false;
    if (!L.dense) {
        if (!r.read_u32_vec(L.ids, n)) return false;
        for (uint32_t k = 0; k < n; k++)                      // ascending, in range
            if (L.ids[k] >= vcount || (k && L.ids[k] <= L.ids[k - 1])) return false;
    }
    return r.read_u32_vec(L.rgba, n);
}

// The active layer's raw pixels for one colour slot. They must describe the same array
// the ENTY body filled with the composite; a slot the body reader dropped (a v5 plane
// of the wrong size) drops its raw copy too.
static bool read_raw_slot(ChunkReader& r, const std::vector<uint32_t>& slot,
                          std::vector<uint32_t>& out) {
    PaintLayer L;
    if (!read_layer_pixels(r, L)) return false;
    if (slot.empty()) { out.clear(); return true; }
    if (L.vcount != slot.size()) return false;
    paint_layer_unpack(L, L.vcount);
    out = std::move(L.rgba);
    return true;
}

// LAYR (see write_layr_chunk). Anything off and the entity keeps the flattened
// picture its colour slots already hold: a damaged layer section costs the layers,
// never the model. Returns false only to say so on the console.
static bool read_layr_chunk(ChunkReader& r, ProjectData& data) {
    uint32_t eid;
    uint8_t  format;
    if (!r.read_u32(eid) || !r.read_u8(format)) return false;
    EntityRecord* e = nullptr;
    for (EntityRecord& c : data.entities) if (c.id == eid) e = &c;
    if (!e || format != LAYR_FORMAT) return false;

    PaintLayerStack s;
    int32_t  active;
    uint32_t count;
    if (!r.read_i32(active) || !r.read_u32(s.next_id) || !r.read_u32(count)) return false;
    if (count == 0 || count > (uint32_t)kMaxPaintLayers) return false;
    if (active < 0 || active >= (int32_t)count) return false;
    s.active = active;
    s.layers.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        PaintLayer& L = s.layers[i];
        uint32_t len;
        uint8_t  vis, mode, clip;
        if (!r.read_u32(L.id) || !r.read_u32(len)) return false;
        if (len > 256 || len > r.remaining()) return false;
        L.name.resize(len);
        if (!r.read_raw(&L.name[0], len)) return false;
        if (!r.read_u8(vis) || !r.read_raw(&L.opacity, 4) ||
            !r.read_u8(mode) || !r.read_u8(clip)) return false;
        if (mode >= (uint8_t)BlendMode::Count) return false;
        if (!(L.opacity >= 0.0f && L.opacity <= 1.0f)) return false;   // also catches NaN
        L.visible = vis != 0;
        L.mode    = (BlendMode)mode;
        L.clip    = clip != 0;
        if (!read_layer_pixels(r, L, (int32_t)i == active)) return false;
        for (uint32_t j = 0; j < i; j++) if (s.layers[j].id == L.id) return false;
        if (L.id >= s.next_id) return false;
    }
    std::vector<uint32_t> work, plane, base;
    if (!read_raw_slot(r, e->mesh.color, work) ||
        !read_raw_slot(r, e->multires.color, plane) ||
        !read_raw_slot(r, e->multires.base.color, base)) return false;

    e->mesh.paint_layers  = std::move(s);
    e->mesh.color         = std::move(work);
    e->multires.color     = std::move(plane);
    e->multires.base.color = std::move(base);
    std::printf("[paint-layers] entity %u: %zu layers loaded, \"%s\" active\n",
                e->id, e->mesh.paint_layers.layers.size(),
                e->mesh.paint_layers.layers[active].name.c_str());
    return true;
}

// RTPO (see write_rtpo_chunk). Everything is checked before it's kept: counts against
// what's left of the chunk, every corner and face offset in range, arrays the sizes the
// counts say. Anything off and the project loads without its retopo (never without
// its sculpt). Returns false only to say so on the console.
static bool read_vec3s(ChunkReader& r, std::vector<Vec3>& out, size_t n) {
    if (r.remaining() / 12 < n) return false;
    out.resize(n);
    for (Vec3& p : out) if (!r.read_raw(&p.x, 4) || !r.read_raw(&p.y, 4) || !r.read_raw(&p.z, 4)) return false;
    return true;
}
static bool read_rtpo_chunk(ChunkReader& r, ProjectData& data) {
    uint8_t format, built;
    RetopoSnapshot s;
    uint32_t ns;
    if (!r.read_u8(format) || format == 0 || format > RTPO_FORMAT) return false;
    if (!r.read_u32(s.next_id) || !r.read_u8(built) || !r.read_u32(ns)) return false;
    if (ns > r.remaining() / 25) return false;   // a stroke takes at least 25 bytes
    s.built = built != 0;
    s.strokes.resize(ns);
    for (RetopoStroke& k : s.strokes) {
        uint32_t len, npts;
        uint8_t flags;
        if (!r.read_u32(k.id) || !r.read_u32(len) || len > r.remaining()) return false;
        k.name.resize(len);
        if (len && !r.read_raw(&k.name[0], len)) return false;
        if (!r.read_u8(flags) || !r.read_i32(k.rings) || !r.read_raw(&k.density, 4)
            || !r.read_raw(&k.width, 4) || !r.read_u32(npts)) return false;
        k.closed = flags & 1;
        k.visible = (flags & 2) != 0;
        k.frozen = (flags & 4) != 0;
        k.selected = false;
        k.rings = std::max(1, std::min(8, k.rings));
        if (!(k.density >= 0.1f && k.density <= 2.0f)) k.density = 1.0f;   // NaN lands here too
        if (!(k.width >= 0.1f && k.width <= 2.0f))     k.width = 1.0f;
        if (npts > r.remaining() / 28) return false;
        if (!read_vec3s(r, k.pts, npts) || !read_vec3s(r, k.nrm, npts) || !r.read_float_vec(k.half_w, npts))
            return false;
    }
    RetopoMesh& m = s.result;
    uint32_t nv, nf, nc;
    if (!r.read_u32(nv) || nv > r.remaining() / 32) return false;
    if (!read_vec3s(r, m.v, nv) || !read_vec3s(r, m.n, nv) || !r.read_u32_vec(m.v_owner, nv)
        || !r.read_u32_vec(m.mirror, nv)) return false;
    if (!r.read_u32(nf) || nf > r.remaining() / 8) return false;
    if (!r.read_u32_vec(m.face_off, (size_t)nf + 1)) return false;
    if (!r.read_u32(nc) || nc > r.remaining() / 4 || !r.read_u32_vec(m.faces, nc)) return false;
    if (!r.read_u32_vec(m.face_owner, nf)) return false;
    if (m.face_off[0] != 0 || m.face_off[nf] != nc) return false;
    for (uint32_t i = 0; i < nf; i++) if (m.face_off[i + 1] < m.face_off[i]) return false;
    for (uint32_t c : m.faces) if (c >= nv) return false;
    for (uint32_t& t : m.mirror) if (t >= nv) t = ~0u;   // no twin, rather than a wild one
    if (format >= 2) {
        if (!r.read_float_vec(m.mask, nv)) return false;
        for (float& k : m.mask) if (!(k >= 0.0f && k <= 1.0f)) k = 0.0f;   // NaN lands here too
    }
    data.retopo = std::move(s);
    std::printf("[retopo] loaded: %zu stroke(s), %u faces, %u verts\n",
                data.retopo.strokes.size(), nf, nv);
    return true;
}

// v2+ reader: file already validated (magic + version ok) and `f` positioned
// just past the 8-byte header. Layout differences are feature-gated: v3+
// carries per-vertex paint colour in each mesh body (has_color), v5+ carries
// finest-level paint planes in each locked multires body (has_planes), v6+
// the remesh-density field in both (has_density).
static LoadResult load_project_v2(std::ifstream& f, std::streamoff file_size,
                                  ProjectData& data, bool has_color,
                                  bool has_planes, bool has_density,
                                  bool has_derived) {
    bool has_scen = false;

    while ((std::streamoff)f.tellg() < file_size) {
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

        if (std::memcmp(hdr.tag, "ENTY", 4) == 0) {
            EntityRecord e;
            if (!read_enty_chunk(cr, e, has_color, has_planes, has_density, has_derived))
                return LoadResult::ERR_CORRUPT;
            data.entities.push_back(std::move(e));
        } else if (std::memcmp(hdr.tag, "LAYR", 4) == 0) {
            if (!read_layr_chunk(cr, data))
                std::printf("[load] a paint layer section didn't read back: that model "
                            "keeps its flattened picture\n");
        } else if (std::memcmp(hdr.tag, "RTPO", 4) == 0) {
            if (!read_rtpo_chunk(cr, data)) {
                data.retopo = RetopoSnapshot{};
                std::printf("[load] the retopo section didn't read back: the project "
                            "loads without it\n");
            }
        } else if (std::memcmp(hdr.tag, "SCEN", 4) == 0) {
            if (!read_scen_chunk(cr, data)) return LoadResult::ERR_CORRUPT;
            has_scen = true;
        } else if (std::memcmp(hdr.tag, "CAMR", 4) == 0) {
            if (!read_camr_chunk(cr, data.camera)) return LoadResult::ERR_CORRUPT;
        } else if (std::memcmp(hdr.tag, "OPTS", 4) == 0) {
            if (!read_opts_chunk(cr, data.mirror_x, data.subdiv_level)) return LoadResult::ERR_CORRUPT;
        }
        // Unknown chunks: silently skipped.
    }

    if (data.entities.empty()) return LoadResult::ERR_CORRUPT;

    // SCEN is optional defensively: fall back to a sane selection if missing.
    if (!has_scen) {
        data.active_id = data.entities.front().id;
        data.selected_ids = { data.active_id };
        uint32_t mx = 0;
        for (const EntityRecord& e : data.entities) mx = std::max(mx, e.id);
        data.next_id = mx + 1;
    }
    return LoadResult::OK;
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

    // Legacy versions live in their own removable modules. Delete the branch +
    // the module together when support is dropped.
    LoadResult lr;
    if (version == 1) {
        lr = load_project_v1(path, data);
    } else {
        if (version < 2 || version > VERSION) return LoadResult::ERR_VERSION;
        lr = load_project_v2(f, (std::streamoff)file_size, data,
                             version >= 3, version >= 5, version >= 6,
                             version >= 7);
    }

    // v<=3 multires disp layers are indexed under the saving platform's legacy
    // (hash-order) midpoint numbering; flag them for migration on scene rebuild.
    // v4+ files are canonical by definition.
    if (lr == LoadResult::OK && version <= 3)
        for (EntityRecord& e : data.entities)
            e.legacy_numbering = true;
    return lr;
}

// ------------------------------------------------------------------ strings --

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
