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
// unpainted clay, and applies every visible layer bottom to top.
//
// A conventional paint-app layer stack, kept minimal: there is no special mask layer
// type. A "mask" is an ordinary layer (greyscale or not) with a blend mode, and its
// `clip` switch (a clipping mask) confines it to where the
// nearest unclipped layer below it is painted. A clipped layer whose base is hidden
// is hidden too; a clipped layer with nothing below it behaves as unclipped.
// The sculpting mask is a separate thing entirely and is not a layer. The GPU twin is
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
    uint32_t    id = 0;                // stable across reorders; paint undo finds its layer by it. Base = 0
    std::string name;
    bool      visible = true;
    float     opacity = 1.0f;          // 0..1, scales coverage at composite time
    BlendMode mode    = BlendMode::Normal;
    bool      clip    = false;         // only where the nearest unclipped layer below is painted

    // dense:  rgba has one entry per vertex, ids is empty.
    // packed: rgba[k] belongs to vertex ids[k]; ids ascending; no zero-alpha entries.
    bool dense = false;
    std::vector<uint32_t> ids;
    std::vector<uint32_t> rgba;

    size_t bytes() const { return (ids.size() + rgba.size()) * sizeof(uint32_t); }
};

struct PaintLayerStack {
    std::vector<PaintLayer> layers;    // bottom first
    int active = -1;                   // index into layers, -1 = no layers yet
    uint32_t next_id = 1;              // Base is id 0

    bool empty() const { return layers.empty(); }
};

// THE ACTIVE LAYER'S PIXELS DO NOT LIVE IN THE STACK. They live in the caller's array
// (`act` below — in practice Mesh::color), so every pre-layer path that paints, undoes,
// subdivides or remeshes Mesh::color keeps working on the active layer untouched. The
// stack's own slot for the active layer stays empty between calls; edits that change
// which layer is active swap pixels in and out of `act`.
//
// An EMPTY stack is the pre-layer world: one implicit Base layer (Normal, opacity 1)
// whose pixels are `act`, where an empty `act` means unpainted white.

// --- per-vertex math (the reference the shader twins follow) -----------------------

// Blend one layer value onto an opaque composite colour. dst is opaque RGBA8 (the
// composite so far); src is the layer's premultiplied RGBA8. Returns opaque RGBA8.
uint32_t paint_blend(uint32_t dst, uint32_t src, BlendMode mode, float opacity);

// --- layer storage -----------------------------------------------------------------

// Value of a stored (non-active) layer at vertex v; 0 where it has nothing. Packed
// lookup is a binary search, for one-off reads — loops over the mesh walk the list.
uint32_t paint_layer_sample(const PaintLayer& L, uint32_t v);
// Value of stack layer i at v, reading the active layer from `act`.
uint32_t paint_layers_value(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                            int i, uint32_t v);

// Dense -> packed (no-op if already packed, or if packing would not save memory).
void paint_layer_pack(PaintLayer& L);
// Packed -> dense, sized vcount, uncovered verts = 0.
void paint_layer_unpack(PaintLayer& L, uint32_t vcount);

// --- stack edits -------------------------------------------------------------------

// Makes the implicit Base explicit (no-op once the stack has layers). Unpainted Base
// becomes opaque white, like a paint app's background layer: so a clipped layer above it shows
// everywhere, as it would over a white background.
void paint_layers_ensure(PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vcount);
// New empty layer directly above the active one, made active. Returns its index, or
// -1 at the cap.
int  paint_layer_add(PaintLayerStack& s, std::vector<uint32_t>& act,
                     const std::string& name, uint32_t vcount);
// Packs the previous active layer into the stack, unpacks the new one into act.
void paint_layer_set_active(PaintLayerStack& s, std::vector<uint32_t>& act, int idx,
                            uint32_t vcount);
// Removes a layer. The last layer can't be removed: it's cleared instead.
void paint_layer_remove(PaintLayerStack& s, std::vector<uint32_t>& act, int idx,
                        uint32_t vcount);
// Moves a layer from one stack position to another; the active layer follows itself.
void paint_layer_move(PaintLayerStack& s, int from, int to);
// Merge Visible: every shown layer becomes one Normal layer at the lowest
// merged layer's position, made active; hidden layers are kept, in order. Composited
// onto TRANSPARENT (full W3C source-over with the blend modes), so unpainted verts stay
// uncovered. Over the white base it reproduces the display exactly for Normal and
// Multiply; a future mode with B(white, s) != s (Screen, ...) would shift.
void paint_layers_merge_visible(PaintLayerStack& s, std::vector<uint32_t>& act,
                                uint32_t vcount);

// Stable id of the active layer (0 = Base, also for the implicit one), and the index
// of a layer by id (-1 if it no longer exists).
uint32_t paint_layers_active_id(const PaintLayerStack& s);
int      paint_layers_index_of(const PaintLayerStack& s, uint32_t id);

// Index of the layer a clipped layer i clips to (nearest unclipped layer below), or
// -1 if there is none. Unclipped layers return themselves.
int  paint_layer_clip_base(const PaintLayerStack& s, int i);
// Visible AND, if clipped, its base is visible: what the composite actually draws.
bool paint_layer_shown(const PaintLayerStack& s, int i);

// --- composite ---------------------------------------------------------------------

// Whole mesh into out[0..vcount).
void paint_composite(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                     uint32_t vcount, uint32_t* out);
// One vertex. Packed layers are binary-searched, so this is for short lists (undo, a
// single edit); the stroke path runs the GPU composite instead.
uint32_t paint_composite_vertex(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                                uint32_t v);

// Bytes the stack holds, not counting act (for budgets and the console).
size_t paint_layers_bytes(const PaintLayerStack& s);
