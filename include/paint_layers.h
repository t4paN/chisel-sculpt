#pragma once
// Vertex-paint layers: the data model, the blend math, and the CPU composite.
//
// One layer = one RGBA8 value per vertex, stored PREMULTIPLIED (rgb already scaled by
// alpha). Alpha is the layer's coverage: 0 = this layer leaves the vertex alone.
// Premultiplied is what lets every existing colour path keep working unchanged:
// interpolation (subdivide, remesh, multires carry), the paint lerp and the colour
// smooth are all plain per-channel lerps, and a lerp of premultiplied colour is the
// correct one. Straight alpha would bleed the rgb of transparent verts into every
// average. An opaque layer is bit-identical in both forms, so pre-layer paint (which
// was always opaque) loads as-is.
//
// Storage follows the plan in ~/Projects/CHISEL/paint-layers-plan.md: the ACTIVE
// layer is dense (the brush kernels write it on the GPU, unchanged in shape); every
// other layer is packed to (vertex id, rgba) pairs, sorted by id, holding only the
// verts it actually covers. A packed layer that covers more than half the mesh would
// be bigger than dense, so packing keeps it dense instead.
//
// The composite (what the renderer shows) starts from opaque white, the look of
// unpainted clay, and applies every visible layer bottom to top. The GPU twin is
// shaders/{wgsl,glsl}/color_composite.*, and paint_blend() below is the reference it
// must match: change one, change the others.

#include <cstdint>
#include <string>
#include <vector>

// Separable blend modes, W3C-compositing style: each mode is one B(dst, src) formula
// on straight colour, and the result is mix(dst, B, src_alpha * opacity). Adding a
// mode = one enum value + one case in paint_blend_channel() + the same case in both
// shader twins. Values are written to project files: append, never renumber.
enum class BlendMode : uint8_t {
    Normal   = 0,
    Multiply = 1,
    Count
};
const char* blend_mode_name(BlendMode m);

// Hard cap. The GPU composite carries the visible-layer table in one uniform block,
// and it is sized for this many entries.
constexpr int kMaxPaintLayers = 16;

struct PaintLayer {
    std::string name;
    bool      visible = true;
    float     opacity = 1.0f;          // 0..1, scales coverage at composite time
    BlendMode mode    = BlendMode::Normal;

    // dense:  rgba has one entry per vertex, ids is empty.
    // packed: rgba[k] belongs to vertex ids[k]; ids ascending; no zero-alpha entries.
    bool dense = false;
    std::vector<uint32_t> ids;
    std::vector<uint32_t> rgba;

    size_t bytes() const { return (ids.size() + rgba.size()) * sizeof(uint32_t); }
};

struct PaintLayerStack {
    std::vector<PaintLayer> layers;    // bottom first
    int active = -1;                   // index into layers, -1 = no layers

    bool empty() const { return layers.empty(); }
    PaintLayer*       active_layer()       { return active >= 0 ? &layers[active] : nullptr; }
    const PaintLayer* active_layer() const { return active >= 0 ? &layers[active] : nullptr; }
};

// --- per-vertex math (the reference the shader twins follow) -----------------------

// Blend one layer value onto an opaque composite colour. dst is opaque RGBA8 (the
// composite so far); src is the layer's premultiplied RGBA8. Returns opaque RGBA8.
uint32_t paint_blend(uint32_t dst, uint32_t src, BlendMode mode, float opacity);

// --- layer storage -----------------------------------------------------------------

// Value of a layer at vertex v; 0 (transparent) where the layer has nothing.
// Packed lookup is a binary search: O(log n), for one-off reads (eyedropper, tests),
// not for loops over the mesh — those walk the packed list directly.
uint32_t paint_layer_sample(const PaintLayer& L, uint32_t v);

// Dense -> packed (no-op if already packed, or if packing would not save memory).
void paint_layer_pack(PaintLayer& L);
// Packed -> dense, sized vcount, uncovered verts = 0.
void paint_layer_unpack(PaintLayer& L, uint32_t vcount);

// --- stack edits (all keep the "only the active layer is forced dense" invariant) ---

// The single-array paint every pre-layer file and mesh carries becomes one opaque
// Normal layer named "Base". Empty colour = a stack with one blank Base layer.
void paint_layers_init(PaintLayerStack& s, std::vector<uint32_t>&& legacy_color,
                       uint32_t vcount);

// New empty layer directly above the active one, made active. Returns its index, or
// -1 at the cap.
int  paint_layer_add(PaintLayerStack& s, const std::string& name, uint32_t vcount);
// Packs the previous active layer, unpacks the new one.
void paint_layer_set_active(PaintLayerStack& s, int idx, uint32_t vcount);
// Removes a layer. The last layer can't be removed: it's cleared instead.
void paint_layer_remove(PaintLayerStack& s, int idx, uint32_t vcount);
// Moves a layer from one stack position to another; the active layer follows itself.
void paint_layer_move(PaintLayerStack& s, int from, int to);
// Bakes every visible layer into one opaque Normal "Base" layer (hidden layers are
// dropped). Exact, unlike merge-down: a merged pair can't, in general, be expressed as
// one layer with one mode.
void paint_layers_flatten(PaintLayerStack& s, uint32_t vcount);

// --- composite ---------------------------------------------------------------------

// Whole mesh. out is resized to vcount.
void paint_composite(const PaintLayerStack& s, uint32_t vcount, std::vector<uint32_t>& out);
// Only the listed verts, written into an existing vcount-sized out. Packed layers are
// searched per vertex, so this is for small lists (undo, a single edit); the stroke
// path runs the GPU composite instead.
void paint_composite_verts(const PaintLayerStack& s, const uint32_t* verts, size_t n,
                           uint32_t* out);

// Total bytes the stack holds (for budgets and the console).
size_t paint_layers_bytes(const PaintLayerStack& s);
