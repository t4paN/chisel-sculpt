#pragma once
#include <cstdint>
#include <string>
#include <vector>

// UV mode's exports (key 9): the layout as a picture to paint over, and the lows as a
// model with their baked textures (OBJ + MTL, STL, or a GLB that carries them inside).
// Every export takes the meshes the UV panel shows: the active one and the rest of the
// selection, each with a layout made for it as it is now.

struct RetopoMesh;
struct UvLayout;
struct BakeResult;

struct UvExportPart {
    uint32_t          id;
    std::string       name;   // the mesh's name as a file / object name part
    const RetopoMesh* rm;
    const UvLayout*   uv;     // nullptr = no current layout (the model goes without UVs)
};

// The layout at size x size, RGBA, transparent where no island is. With fill, each
// island in its mesh's shade (as the UV panel shows it), half see-through; the wire is
// every face edge, wire_px thick, near-black. Row 0 = v 1 (the top, as a PNG).
std::vector<uint8_t> uv_map_rgba(const std::vector<UvExportPart>& parts, int size,
                                 float wire_px, bool fill);
bool uv_map_write_png(const std::vector<UvExportPart>& parts, int size, float wire_px,
                      bool fill, const char* path);

// The lows as one binary glTF: one node + mesh each (named <name>-LO, so Chisel reads it
// back as their retopos), smooth normals as the bake used them, UVs where the layout is
// current, faces fanned from their first corner (as the bake and OBJ readers do). With
// colour / normal PNGs, one material wears them (an atlas: every mesh shares it).
bool glb_write_lows(const char* path, const std::vector<UvExportPart>& parts,
                    const std::vector<uint8_t>* color_png, const std::vector<uint8_t>* normal_png);
