#include "paint_layers.h"

#include <algorithm>
#include <cmath>

// Unorm conversions written to match WGSL unpack4x8unorm / pack4x8unorm exactly
// (x/255 in, floor(0.5 + 255*clamp(x)) out), so the CPU composite and the GPU one
// agree to the bit wherever the float maths does.
static inline float unorm(uint32_t c, int ch) { return (float)((c >> (8 * ch)) & 0xFFu) / 255.0f; }
static inline uint32_t to_unorm(float x) {
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    return (uint32_t)std::floor(0.5f + 255.0f * x);
}

const char* blend_mode_name(BlendMode m) {
    switch (m) {
        case BlendMode::Normal:   return "Normal";
        case BlendMode::Multiply: return "Multiply";
        default:                  return "?";
    }
}

// B(dst, src) on straight colour, one channel. The ONLY place a mode's look is
// defined on the CPU; the shader twins carry the same switch.
static inline float paint_blend_channel(float d, float s, BlendMode mode) {
    switch (mode) {
        case BlendMode::Multiply: return d * s;
        case BlendMode::Normal:
        default:                  return s;
    }
}

uint32_t paint_blend(uint32_t dst, uint32_t src, BlendMode mode, float opacity) {
    const uint32_t sa8 = src >> 24;
    if (sa8 == 0) return dst;
    const float sa = (float)sa8 / 255.0f;
    const float ao = sa * opacity;
    uint32_t out = 0xFF000000u;
    for (int ch = 0; ch < 3; ch++) {
        const float d = unorm(dst, ch);
        const float s = unorm(src, ch) / sa;          // un-premultiply
        const float b = paint_blend_channel(d, s, mode);
        out |= to_unorm(d * (1.0f - ao) + b * ao) << (8 * ch);   // = WGSL mix()
    }
    return out;
}

uint32_t paint_layer_sample(const PaintLayer& L, uint32_t v) {
    if (L.dense) return v < L.rgba.size() ? L.rgba[v] : 0u;
    auto it = std::lower_bound(L.ids.begin(), L.ids.end(), v);
    if (it == L.ids.end() || *it != v) return 0u;
    return L.rgba[(size_t)(it - L.ids.begin())];
}

void paint_layer_pack(PaintLayer& L) {
    if (!L.dense) return;
    size_t covered = 0;
    for (uint32_t c : L.rgba) covered += (c >> 24) != 0;
    // Packed costs 8 B per covered vertex, dense 4 B per vertex: past half coverage
    // dense is smaller, so the layer stays dense even while inactive.
    if (covered * 2 > L.rgba.size()) return;
    std::vector<uint32_t> ids, rgba;
    ids.reserve(covered);
    rgba.reserve(covered);
    for (uint32_t v = 0; v < (uint32_t)L.rgba.size(); v++) {
        if ((L.rgba[v] >> 24) == 0) continue;
        ids.push_back(v);
        rgba.push_back(L.rgba[v]);
    }
    L.ids  = std::move(ids);
    L.rgba = std::move(rgba);
    L.dense = false;
}

void paint_layer_unpack(PaintLayer& L, uint32_t vcount) {
    if (L.dense) {
        L.rgba.resize(vcount, 0u);
        return;
    }
    std::vector<uint32_t> dense(vcount, 0u);
    for (size_t k = 0; k < L.ids.size(); k++)
        if (L.ids[k] < vcount) dense[L.ids[k]] = L.rgba[k];
    L.ids.clear();
    L.ids.shrink_to_fit();
    L.rgba  = std::move(dense);
    L.dense = true;
}

void paint_layers_init(PaintLayerStack& s, std::vector<uint32_t>&& legacy_color,
                       uint32_t vcount) {
    s.layers.clear();
    PaintLayer base;
    base.name  = "Base";
    base.dense = true;
    base.rgba  = std::move(legacy_color);
    base.rgba.resize(vcount, 0u);
    s.layers.push_back(std::move(base));
    s.active = 0;
}

int paint_layer_add(PaintLayerStack& s, const std::string& name, uint32_t vcount) {
    if ((int)s.layers.size() >= kMaxPaintLayers) return -1;
    if (PaintLayer* prev = s.active_layer()) paint_layer_pack(*prev);
    PaintLayer L;
    L.name  = name;
    L.dense = true;
    L.rgba.assign(vcount, 0u);
    const int idx = s.active + 1;      // directly above the active one (0 on an empty stack)
    s.layers.insert(s.layers.begin() + idx, std::move(L));
    s.active = idx;
    return idx;
}

void paint_layer_set_active(PaintLayerStack& s, int idx, uint32_t vcount) {
    if (idx < 0 || idx >= (int)s.layers.size() || idx == s.active) return;
    if (PaintLayer* prev = s.active_layer()) paint_layer_pack(*prev);
    s.active = idx;
    paint_layer_unpack(s.layers[idx], vcount);
}

void paint_layer_remove(PaintLayerStack& s, int idx, uint32_t vcount) {
    if (idx < 0 || idx >= (int)s.layers.size()) return;
    if (s.layers.size() == 1) {
        PaintLayer& L = s.layers[0];
        L.ids.clear();
        L.rgba.assign(vcount, 0u);
        L.dense = true;
        return;
    }
    s.layers.erase(s.layers.begin() + idx);
    if (idx == s.active) {
        // The layer below takes over (or the new bottom one).
        s.active = -1;
        paint_layer_set_active(s, std::max(0, idx - 1), vcount);
    } else if (idx < s.active) {
        s.active--;
    }
}

void paint_layer_move(PaintLayerStack& s, int from, int to) {
    const int n = (int)s.layers.size();
    if (from < 0 || from >= n || to < 0 || to >= n || from == to) return;
    PaintLayer L = std::move(s.layers[from]);
    s.layers.erase(s.layers.begin() + from);
    s.layers.insert(s.layers.begin() + to, std::move(L));
    if (s.active == from)                     s.active = to;
    else if (from < s.active && to >= s.active) s.active--;
    else if (from > s.active && to <= s.active) s.active++;
}

void paint_layers_flatten(PaintLayerStack& s, uint32_t vcount) {
    std::vector<uint32_t> flat;
    paint_composite(s, vcount, flat);
    paint_layers_init(s, std::move(flat), vcount);
}

void paint_composite(const PaintLayerStack& s, uint32_t vcount, std::vector<uint32_t>& out) {
    out.assign(vcount, 0xFFFFFFFFu);
    // Layer-major: each layer touches only what it covers, so a packed layer costs its
    // own size, not the mesh's. Order is still bottom to top at every vertex.
    for (const PaintLayer& L : s.layers) {
        if (!L.visible || L.opacity <= 0.0f) continue;
        if (L.dense) {
            const uint32_t n = std::min<uint32_t>(vcount, (uint32_t)L.rgba.size());
            for (uint32_t v = 0; v < n; v++)
                out[v] = paint_blend(out[v], L.rgba[v], L.mode, L.opacity);
        } else {
            for (size_t k = 0; k < L.ids.size(); k++) {
                const uint32_t v = L.ids[k];
                if (v < vcount) out[v] = paint_blend(out[v], L.rgba[k], L.mode, L.opacity);
            }
        }
    }
}

void paint_composite_verts(const PaintLayerStack& s, const uint32_t* verts, size_t n,
                           uint32_t* out) {
    for (size_t i = 0; i < n; i++) {
        const uint32_t v = verts[i];
        uint32_t c = 0xFFFFFFFFu;
        for (const PaintLayer& L : s.layers) {
            if (!L.visible || L.opacity <= 0.0f) continue;
            c = paint_blend(c, paint_layer_sample(L, v), L.mode, L.opacity);
        }
        out[v] = c;
    }
}

size_t paint_layers_bytes(const PaintLayerStack& s) {
    size_t b = 0;
    for (const PaintLayer& L : s.layers) b += L.bytes();
    return b;
}
