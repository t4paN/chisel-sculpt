#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// Bake (UV mode, key 9): the sculpt the retopo mesh was drawn on, written into textures
// laid out by the retopo mesh's UVs. The pair is implicit: the retopo mesh is the low,
// the sculpt under it the high, so there is nothing to match up by name.
//
// One ray per texel, from the low surface along its smooth normal, to the nearest piece
// of sculpt that faces the same way within `reach` either side; the sculpt's triangles
// sit in a bounding-volume tree so a ray only tests a handful of them. CPU, and run a
// slice per frame (bake_step): the web build has no threads, and a few seconds of
// responsive progress bar beats a few seconds of frozen window.
//
// The normal map is tangent-space in the MikkTSpace convention (external/mikktspace,
// zlib): the frame every common engine and DCC rebuilds from the low's normals and UVs,
// so a map baked here reads the same there. That only holds if they get the same
// normals: OBJ export writes them (retopo_smooth_normals), and faces are fanned from
// their first corner, as OBJ importers triangulate them.

struct Mesh;
struct RetopoMesh;
struct UvLayout;

struct BakeSettings {
    int   resolution = 1024;
    float reach = 1.0f;         // how far either side of the low a ray looks, in mean low edges
    bool  normal = true, color = true;
    bool  flip_green = false;   // green down (DirectX-style engines); default is green up
};

// A finished bake: the maps, plus the low mesh as the triangle soup they were baked
// against (corner tangent frames included), which is exactly what the preview draws.
struct BakeResult {
    int  res = 0;
    bool has_normal = false, has_color = false, flip_green = false;
    std::vector<uint8_t> normal_rgba, color_rgba;   // res * res * 4; row 0 = v 1 (the top, as a PNG)
    std::vector<float> pos, nrm, tan, uv;           // 3, 3, 4, 2 floats per corner, 3 corners per triangle
    uint64_t key = 0;        // bake_key of the low it was baked for
    uint32_t serial = 0;     // unique per bake in a run: the renderer re-uploads when it changes
    float    missed = 0.0f;  // share of the islands' texels whose ray found no sculpt
    double   ms = 0.0;
    uint32_t high_tris = 0;
    bool empty() const { return res == 0; }
    uint32_t corners() const { return (uint32_t)(pos.size() / 3); }
};

struct BakeWork;   // a bake in progress (bake.cpp)

// Lives in RetopoState: a new or loaded project drops it with the retopo mesh it was
// baked from. Not saved: the PNGs are the keepers.
struct BakeState {
    BakeSettings settings;
    BakeResult   result;
    std::shared_ptr<BakeWork> job;   // set while a bake runs
    bool preview = true;             // UV mode shows the low wearing the maps
    // Is `result` still the low as it is? Checked by bake_refresh once per retopo change.
    bool fresh = false;
    uint64_t checked_rev = ~0ull;
    bool running() const { return (bool)job; }
    bool showing() const { return preview && fresh && !result.empty(); }
};

// A hash of what a bake depends on in the low: faces, positions and UVs. Different now =
// the bake is out of date (the preview stops drawing it).
uint64_t bake_key(const RetopoMesh& rm, const UvLayout& L);

// Once a frame: re-hash the low only when `rev` (RetopoState::rev) moved since the last look.
void bake_refresh(BakeState& b, const RetopoMesh& rm, const UvLayout& L, uint64_t rev);

// Start a bake of `high` onto rm's layout L (both copied, the sculpt is read in place).
// nullptr = started, else why not.
const char* bake_start(BakeState& b, const RetopoMesh& rm, const UvLayout& L, const Mesh& high);
// Work for about budget_ms. 0 = still going, 1 = done (b.result filled), -1 = stopped
// (why says why). `high` must be the sculpt it started on; anything else stops it.
int  bake_step(BakeState& b, const Mesh& high, double budget_ms, const char** why);
void bake_cancel(BakeState& b);
float bake_progress(const BakeState& b);   // 0..1
const char* bake_phase(const BakeState& b);

// One map as an RGB PNG: which 0 = normal, 1 = colour. False = no such map or not written.
bool bake_write_png(const BakeResult& r, int which, const char* path);
