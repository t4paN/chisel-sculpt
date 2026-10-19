#include "paint_layers.h"
#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
    // Opaque Normal at full opacity is a copy — and it is the whole pre-layer world
    // (one opaque Base), so the common case skips the float maths entirely.
    if (sa8 == 255 && opacity >= 1.0f && mode == BlendMode::Normal) return src;
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

uint32_t paint_layers_value(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                            int i, uint32_t v) {
    if (i == s.active || s.empty()) {
        if (act.empty()) return 0xFFFFFFFFu;     // implicit / never-painted Base
        return v < act.size() ? act[v] : 0u;
    }
    return paint_layer_sample(s.layers[i], v);
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
    L.vcount = vcount;
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

// Edits work on the whole stack with the active layer's pixels parked in its own slot
// (stash), then hand the active layer's pixels back to act (unstash).
static void stash(PaintLayerStack& s, std::vector<uint32_t>& act) {
    if (s.active < 0) return;
    PaintLayer& L = s.layers[s.active];
    L.ids.clear();
    L.rgba   = std::move(act);
    L.dense  = true;
    L.vcount = (uint32_t)L.rgba.size();
    act.clear();
}
static void unstash(PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vcount) {
    if (s.active < 0) return;
    PaintLayer& L = s.layers[s.active];
    paint_layer_unpack(L, vcount);
    act = std::move(L.rgba);
    L.rgba.clear();
}

void paint_layers_ensure(PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vcount) {
    if (!s.empty()) return;
    PaintLayer base;
    base.name  = "Base";
    base.dense = true;
    s.layers.push_back(std::move(base));
    s.active = 0;
    if (act.empty()) act.assign(vcount, 0xFFFFFFFFu);
    else             act.resize(vcount, 0xFFFFFFFFu);
}

static void set_active_stashed(PaintLayerStack& s, int idx) {
    if (s.active >= 0) paint_layer_pack(s.layers[s.active]);
    s.active = idx;
}

int paint_layer_add(PaintLayerStack& s, std::vector<uint32_t>& act,
                    const std::string& name, uint32_t vcount) {
    paint_layers_ensure(s, act, vcount);
    if ((int)s.layers.size() >= kMaxPaintLayers) return -1;
    stash(s, act);
    PaintLayer L;
    L.id     = s.next_id++;
    L.name   = name;
    L.dense  = true;
    L.vcount = vcount;
    L.rgba.assign(vcount, 0u);
    const int idx = s.active + 1;      // directly above the active one
    s.layers.insert(s.layers.begin() + idx, std::move(L));
    set_active_stashed(s, idx);
    unstash(s, act, vcount);
    return idx;
}

void paint_layer_set_active(PaintLayerStack& s, std::vector<uint32_t>& act, int idx,
                            uint32_t vcount) {
    if (idx < 0 || idx >= (int)s.layers.size() || idx == s.active) return;
    stash(s, act);
    set_active_stashed(s, idx);
    unstash(s, act, vcount);
}

void paint_layer_remove(PaintLayerStack& s, std::vector<uint32_t>& act, int idx,
                        uint32_t vcount) {
    if (idx < 0 || idx >= (int)s.layers.size()) return;
    if (s.layers.size() == 1) {
        act.assign(vcount, 0u);
        return;
    }
    stash(s, act);
    s.layers.erase(s.layers.begin() + idx);
    if (idx == s.active) {
        // The layer below takes over (or the new bottom one).
        s.active = -1;
        set_active_stashed(s, std::max(0, idx - 1));
    } else if (idx < s.active) {
        s.active--;
    }
    unstash(s, act, vcount);
}

void paint_layer_move(PaintLayerStack& s, int from, int to) {
    const int n = (int)s.layers.size();
    if (from < 0 || from >= n || to < 0 || to >= n || from == to) return;
    PaintLayer L = std::move(s.layers[from]);
    s.layers.erase(s.layers.begin() + from);
    s.layers.insert(s.layers.begin() + to, std::move(L));
    if (s.active == from)                       s.active = to;
    else if (from < s.active && to >= s.active) s.active--;
    else if (from > s.active && to <= s.active) s.active++;
}

uint32_t paint_layers_active_id(const PaintLayerStack& s) {
    return s.active >= 0 ? s.layers[s.active].id : 0u;
}

int paint_layers_index_of(const PaintLayerStack& s, uint32_t id) {
    if (s.empty()) return id == 0 ? 0 : -1;
    for (int i = 0; i < (int)s.layers.size(); i++)
        if (s.layers[i].id == id) return i;
    return -1;
}

int paint_layer_clip_base(const PaintLayerStack& s, int i) {
    if (s.empty() || !s.layers[i].clip) return i;
    for (int b = i - 1; b >= 0; b--)
        if (!s.layers[b].clip) return b;
    return -1;
}

bool paint_layer_shown(const PaintLayerStack& s, int i) {
    if (s.empty()) return true;
    if (!s.layers[i].visible) return false;
    const int b = paint_layer_clip_base(s, i);
    return b < 0 || s.layers[b].visible;
}

// A clipped layer's coverage is scaled by its base's coverage at the vertex. Applied
// as a multiplier on opacity so paint_blend() stays the single per-vertex reference.
static inline float layer_opacity(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                                  int i, uint32_t v) {
    if (s.empty()) return 1.0f;
    const float op = s.layers[i].opacity;
    const int b = paint_layer_clip_base(s, i);
    if (b == i || b < 0) return op;
    return op * ((float)(paint_layers_value(s, act, b, v) >> 24) / 255.0f);
}
static inline BlendMode layer_mode(const PaintLayerStack& s, int i) {
    return s.empty() ? BlendMode::Normal : s.layers[i].mode;
}

void paint_layers_merge_visible(PaintLayerStack& s, std::vector<uint32_t>& act,
                                uint32_t vcount) {
    paint_layers_ensure(s, act, vcount);
    std::vector<int> merged;
    for (int i = 0; i < (int)s.layers.size(); i++)
        if (paint_layer_shown(s, i)) merged.push_back(i);
    if (merged.empty()) return;

    // W3C source-over with a separable blend, onto a transparent backdrop, in
    // premultiplied form:  co = as*(1-ab)*cs + as*ab*B(cb,cs) + (1-as)*cb_premult,
    //                      ao = as + ab*(1-as).  Float across layers, rounded once.
    std::vector<uint32_t> out(vcount, 0u);
    for (uint32_t v = 0; v < vcount; v++) {
        float cr[3] = {0, 0, 0}, ar = 0.0f;
        for (int i : merged) {
            const uint32_t c = paint_layers_value(s, act, i, v);
            const uint32_t ca8 = c >> 24;
            if (ca8 == 0) continue;
            const float ca = (float)ca8 / 255.0f;
            const float as = ca * layer_opacity(s, act, i, v);
            if (as <= 0.0f) continue;
            for (int ch = 0; ch < 3; ch++) {
                const float cs = unorm(c, ch) / ca;
                const float cb = ar > 0.0f ? cr[ch] / ar : 0.0f;
                cr[ch] = as * (1.0f - ar) * cs + as * ar * paint_blend_channel(cb, cs, s.layers[i].mode)
                       + (1.0f - as) * cr[ch];
            }
            ar = as + ar * (1.0f - as);
        }
        if (ar <= 0.0f) continue;
        out[v] = to_unorm(cr[0]) | to_unorm(cr[1]) << 8 | to_unorm(cr[2]) << 16 | to_unorm(ar) << 24;
    }

    stash(s, act);
    PaintLayer M;
    M.id     = s.next_id++;
    M.name   = "Merged";
    M.dense  = true;
    M.vcount = vcount;
    M.rgba  = std::move(out);
    const int at = merged.front();
    std::vector<PaintLayer> kept;
    for (int i = 0; i < (int)s.layers.size(); i++) {
        if (i == at) kept.push_back(std::move(M));
        else if (std::find(merged.begin(), merged.end(), i) == merged.end()) {
            paint_layer_pack(s.layers[i]);
            kept.push_back(std::move(s.layers[i]));
        }
    }
    s.layers = std::move(kept);
    s.active = at;   // everything below `at` was kept, so the merged layer lands at `at`
    unstash(s, act, vcount);
}

void paint_composite(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                     uint32_t vcount, uint32_t* out) {
    for (uint32_t v = 0; v < vcount; v++) out[v] = 0xFFFFFFFFu;
    const int n = s.empty() ? 1 : (int)s.layers.size();
    // Layer-major: each stored layer touches only what it covers, so a packed layer
    // costs its own size, not the mesh's. Order is still bottom to top at every vertex.
    for (int i = 0; i < n; i++) {
        if (!paint_layer_shown(s, i)) continue;
        const BlendMode mode = layer_mode(s, i);
        const bool is_act = s.empty() || i == s.active;
        const PaintLayer* L = is_act ? nullptr : &s.layers[i];
        if (is_act || L->dense) {
            if (is_act && act.empty()) continue;            // unpainted Base = the white we start from
            const std::vector<uint32_t>& px = is_act ? act : L->rgba;
            const uint32_t m = std::min<uint32_t>(vcount, (uint32_t)px.size());
            for (uint32_t v = 0; v < m; v++)
                out[v] = paint_blend(out[v], px[v], mode, layer_opacity(s, act, i, v));
        } else {
            for (size_t k = 0; k < L->ids.size(); k++) {
                const uint32_t v = L->ids[k];
                if (v < vcount) out[v] = paint_blend(out[v], L->rgba[k], mode, layer_opacity(s, act, i, v));
            }
        }
    }
}

uint32_t paint_composite_vertex(const PaintLayerStack& s, const std::vector<uint32_t>& act,
                                uint32_t v) {
    uint32_t c = 0xFFFFFFFFu;
    const int n = s.empty() ? 1 : (int)s.layers.size();
    for (int i = 0; i < n; i++) {
        if (!paint_layer_shown(s, i)) continue;
        c = paint_blend(c, paint_layers_value(s, act, i, v), layer_mode(s, i),
                        layer_opacity(s, act, i, v));
    }
    return c;
}

size_t paint_layers_bytes(const PaintLayerStack& s) {
    size_t b = 0;
    for (const PaintLayer& L : s.layers) b += L.bytes();
    return b;
}

void paint_layers_begin_carry(PaintLayerStack& s, uint32_t vcount) {
    s.carrying = !s.empty();
    for (int i = 0; i < (int)s.layers.size(); i++) {
        if (i == s.active) continue;
        PaintLayer& L = s.layers[i];
        paint_layer_unpack(L, vcount);     // dense; a longer layer keeps its prefix
    }
}

void paint_layers_end_carry(PaintLayerStack& s, uint32_t vcount) {
    if (!s.carrying) return;
    s.carrying = false;
    for (int i = 0; i < (int)s.layers.size(); i++) {
        if (i == s.active) continue;
        PaintLayer& L = s.layers[i];
        L.rgba.resize(vcount, 0u);
        L.vcount = vcount;
        paint_layer_pack(L);
    }
}

int paint_layers_drop_stale(PaintLayerStack& s, uint32_t vcount) {
    int dropped = 0;
    for (int i = 0; i < (int)s.layers.size(); i++) {
        if (i == s.active) continue;
        PaintLayer& L = s.layers[i];
        if (L.vcount == vcount) continue;
        std::printf("[paint-layers] layer \"%s\" was stored for %u verts, mesh has %u: cleared\n",
                    L.name.c_str(), L.vcount, vcount);
        L.ids.clear();
        L.rgba.clear();
        L.dense  = false;
        L.vcount = vcount;
        dropped++;
    }
    return dropped;
}

// ---- fill bucket ----

static inline float bucket_mask_at(const Mesh& mesh, uint32_t v) {
    float m = v < mesh.mask.size() ? mesh.mask[v] : 0.0f;
    return m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m);
}
static constexpr float kBucketWall = 0.5f;

void paint_bucket_walls(const Mesh& mesh, const uint32_t* shown, uint32_t seed, float tolerance,
                        std::vector<float>& block) {
    const uint32_t vc = mesh.vertex_count();
    block.resize(vc);
    for (uint32_t v = 0; v < vc; v++) block[v] = bucket_mask_at(mesh, v);
    if (!shown || seed >= vc) return;
    // Paint unlike the clicked colour walls the fill like a mask: the largest channel
    // difference, ramped so it crosses the wall (0.5) at `tolerance`. A ramp, not a
    // cut: a soft stroke's fringe holds partly, the way a soft mask edge does.
    const float tol = std::max(tolerance, 1e-3f);
    const uint32_t ref = shown[seed];
    for (uint32_t v = 0; v < vc; v++) {
        float d = 0.0f;
        for (int ch = 0; ch < 3; ch++) d = std::max(d, std::fabs(unorm(shown[v], ch) - unorm(ref, ch)));
        float w = (d - 0.5f * tol) / tol;
        w = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
        if (w > block[v]) block[v] = w;
    }
}

void paint_bucket_island(const Mesh& mesh, const float* block, uint32_t seed,
                         std::vector<uint32_t>& island) {
    island.clear();
    const uint32_t vc = mesh.vertex_count();
    if (seed >= vc || mesh.vert_tri_offset.size() != (size_t)vc + 1) return;
    if (block[seed] >= kBucketWall) return;
    // One-shot and user-paced, so it may walk the whole mesh. Kept across calls to
    // skip re-allocating it per fill.
    static std::vector<uint8_t> seen;
    seen.assign(vc, 0);
    island.push_back(seed);
    seen[seed] = 1;
    for (size_t head = 0; head < island.size(); head++) {
        const uint32_t v = island[head];
        if (block[v] >= kBucketWall) continue;
        for (uint32_t k = mesh.vert_tri_offset[v]; k < mesh.vert_tri_offset[v + 1]; k++) {
            const uint32_t t = mesh.vert_tri_list[k];
            for (int c = 0; c < 3; c++) {
                const uint32_t u = mesh.indices[t * 3 + c];
                if (!seen[u]) { seen[u] = 1; island.push_back(u); }
            }
        }
    }
}

// Bilinear, wrapping both ways: patterns tile.
static float sample_wrap(const float* data, uint32_t w, uint32_t h, float u, float v) {
    const float fx = (u - std::floor(u)) * w - 0.5f, fy = (v - std::floor(v)) * h - 0.5f;
    const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    const float tx = fx - x0, ty = fy - y0;
    auto at = [&](int x, int y) {
        x %= (int)w; if (x < 0) x += w;
        y %= (int)h; if (y < 0) y += h;
        return data[(size_t)y * w + x];
    };
    const float a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
    return (a + (b - a) * tx) * (1.0f - ty) + (c + (d - c) * tx) * ty;
}

void paint_cover_triplanar(const Mesh& mesh, const std::vector<uint32_t>& island,
                           const float* data, uint32_t w, uint32_t h, float tile,
                           float angle, const float origin[3], const float frame[9],
                           std::vector<float>& cover) {
    cover.resize(island.size());
    const float inv = 1.0f / std::max(tile, 1e-6f);
    const float ca = std::cos(angle), sa = std::sin(angle);
    const bool has_n = mesh.norm_x.size() >= mesh.vertex_count();
    const float* R = frame;       // right
    const float* U = frame + 3;   // up
    const float* N = frame + 6;   // the clicked spot's normal
    auto dot = [](const float* a, float x, float y, float z) { return a[0] * x + a[1] * y + a[2] * z; };
    // Facing weights are cos^4; the main one is boosted by 6.5 so the side projections
    // only win past tan(theta) = 6.5^(1/4) ~ 1.6, i.e. ~58 degrees off the click.
    constexpr float kMainBias = 6.5f;
    for (size_t k = 0; k < island.size(); k++) {
        const uint32_t v = island[k];
        const float rx = mesh.pos_x[v] - origin[0], ry = mesh.pos_y[v] - origin[1],
                    rz = mesh.pos_z[v] - origin[2];
        const float x = dot(R, rx, ry, rz) * inv, y = dot(U, rx, ry, rz) * inv,
                    z = dot(N, rx, ry, rz) * inv;
        float wx = 1, wy = 1, wz = 1;
        if (has_n) {
            const float nx = mesh.norm_x[v], ny = mesh.norm_y[v], nz = mesh.norm_z[v];
            wx = dot(R, nx, ny, nz); wx *= wx; wx *= wx;
            wy = dot(U, nx, ny, nz); wy *= wy; wy *= wy;
            wz = dot(N, nx, ny, nz); wz *= wz; wz *= wz * kMainBias;
        }
        const float sum = wx + wy + wz;
        if (sum <= 0.0f) { wx = wy = 0.0f; wz = 1.0f; } else { wx /= sum; wy /= sum; wz /= sum; }
        // Each axis sees the other two as its (across, up) plane; image rows run down.
        auto plane = [&](float pu, float pv) {
            const float u = ca * pu - sa * pv, t = sa * pu + ca * pv;
            return sample_wrap(data, w, h, u, -t);
        };
        cover[k] = wx * plane(z, y) + wy * plane(x, z) + wz * plane(x, y);
    }
}

void paint_cover_gradient(const Mesh& mesh, const std::vector<uint32_t>& island,
                          const float a[3], const float b[3], std::vector<float>& cover) {
    cover.resize(island.size());
    const float dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const float len2 = std::max(dx * dx + dy * dy + dz * dz, 1e-12f);
    for (size_t k = 0; k < island.size(); k++) {
        const uint32_t v = island[k];
        float t = ((mesh.pos_x[v] - a[0]) * dx + (mesh.pos_y[v] - a[1]) * dy +
                   (mesh.pos_z[v] - a[2]) * dz) / len2;
        cover[k] = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    }
}

void paint_bucket_edge(const Mesh& mesh, const float* block, std::vector<uint32_t>& island,
                       float hardness, float feather, float creep, std::vector<float>& weight) {
    weight.resize(island.size());
    const float h = hardness < 0.0f ? 0.0f : (hardness > 1.0f ? 1.0f : hardness);
    // The mask, ramped toward a hard split at the wall as hardness rises: 0 keeps a soft
    // mask as soft as it was painted, 1 paints everything short of the wall in full.
    const float half = std::max(0.5f * (1.0f - h), 0.004f);
    for (size_t k = 0; k < island.size(); k++) {
        float m = (block[island[k]] - 0.5f) / (2.0f * half) + 0.5f;
        weight[k] = 1.0f - (m < 0.0f ? 0.0f : (m > 1.0f ? 1.0f : m));
    }
    if (feather <= 0.0f) return;
    // The hard edge is one triangle wide, which on a coarse mesh reads as a jagged gap.
    // The feather is a band straddling the mask outline instead: 60% of it on the fill
    // side, 40% reaching into the mask, so the fill melts into what the mask protects
    // rather than stopping short of it. Distances run along the surface from the wall
    // vertices (Dijkstra, only as far as the band reaches, so a hard edge costs nothing).
    // `creep` slides the whole band that much further into the mask (repeat fills).
    const float reach_in = 0.6f * feather, reach_out = 0.4f * feather + creep;
    const float band_out = 0.4f * feather;
    const uint32_t vc = mesh.vertex_count();
    static std::vector<float> dist_in, dist_out;
    static std::vector<uint8_t> listed;
    static std::vector<std::pair<float, uint32_t>> heap;
    dist_in.assign(vc, reach_in);
    dist_out.assign(vc, reach_out);
    listed.assign(vc, 0);
    for (uint32_t v : island) listed[v] = 1;
    auto later = [](const std::pair<float, uint32_t>& x, const std::pair<float, uint32_t>& y) {
        return x.first > y.first;
    };
    // From every wall vertex at once, through the unmasked side or through the mask.
    auto spread = [&](std::vector<float>& dist, bool into_mask) {
        heap.clear();
        for (uint32_t v : island)
            if (block[v] >= kBucketWall) { dist[v] = 0.0f; heap.push_back({0.0f, v}); }
        std::make_heap(heap.begin(), heap.end(), later);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), later);
            const auto [d, v] = heap.back();
            heap.pop_back();
            if (d > dist[v]) continue;   // a shorter path got here first
            for (uint32_t k = mesh.vert_tri_offset[v]; k < mesh.vert_tri_offset[v + 1]; k++) {
                const uint32_t t = mesh.vert_tri_list[k];
                for (int c = 0; c < 3; c++) {
                    const uint32_t u = mesh.indices[t * 3 + c];
                    if ((block[u] >= kBucketWall) != into_mask) continue;
                    const float dx = mesh.pos_x[u] - mesh.pos_x[v], dy = mesh.pos_y[u] - mesh.pos_y[v],
                                dz = mesh.pos_z[u] - mesh.pos_z[v];
                    const float nd = d + std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (nd >= dist[u]) continue;
                    dist[u] = nd;
                    heap.push_back({nd, u});
                    std::push_heap(heap.begin(), heap.end(), later);
                }
            }
        }
    };
    spread(dist_in, false);
    spread(dist_out, true);
    // 1 at the band's inner end, 0 at its outer end; s = signed distance, + into the mask.
    auto ramp = [&](float s) {
        float t = (s - creep + reach_in) / (reach_in + band_out);
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        return 1.0f - t * t * (3.0f - 2.0f * t);
    };
    for (size_t k = 0; k < island.size(); k++) {
        const uint32_t v = island[k];
        // The smaller of the two, not their product: a soft mask's own ramp times the
        // band dipped just inside the wall and read as a pale ring there. The soft
        // mask's hold never drops below what the band gives the wall itself, so the
        // inside can't fall under the line as creep lifts it (a ring again, otherwise).
        // A vertex short of the wall holds at least 0.5, above the uncrept 0.35 there,
        // so without creep this is just the mask.
        if (block[v] >= kBucketWall) weight[k] = ramp(dist_out[v]);
        else weight[k] = std::min(std::max(weight[k], ramp(0.0f)), ramp(-dist_in[v]));
    }
    // Masked vertices the band reaches past the first wall ring join the fill. The flood
    // above only entered the mask, so it can't cross a thin mask line into the next island.
    for (uint32_t v = 0; v < vc; v++) {
        if (listed[v] || dist_out[v] >= reach_out) continue;
        island.push_back(v);
        weight.push_back(ramp(dist_out[v]));
    }
}

void paint_bucket_apply(Mesh& mesh, const std::vector<uint32_t>& island, uint32_t main,
                        uint32_t second, const float* value, const float* edge, bool erase,
                        float strength,
                        std::vector<uint32_t>& verts,
                        std::vector<uint32_t>& old_c, std::vector<uint32_t>& new_c) {
    const uint32_t vc = mesh.vertex_count();
    // An unpainted layer reads as what the renderer shows for it (paint_layer_px).
    const uint32_t blank = mesh.paint_layers.empty() ? 0xFFFFFFFFu : 0u;
    if (mesh.color.size() < vc) mesh.color.resize(vc, blank);
    for (size_t k = 0; k < island.size(); k++) {
        const uint32_t v = island[k];
        if (v >= vc) continue;
        const float t = value ? value[k] : 0.0f;
        const uint32_t old = mesh.color[v];
        float o[4], tg[4];
        for (int ch = 0; ch < 4; ch++) {
            o[ch] = unorm(old, ch);
            tg[ch] = erase ? 0.0f : unorm(main, ch) + (unorm(second, ch) - unorm(main, ch)) * t;
        }
        // The brush's lerp of all four channels (color_paint.*).
        float w = edge[k] * strength;
        if (erase) {
            w *= 1.0f - t;
        } else {
            // The edge is a ceiling, not a step: how much of this target the vertex
            // already has (its place on the blank-to-target line, when it sits on that
            // line) is its coverage, and each fill takes `strength` of the gap up to the
            // edge. Otherwise every repeat fill lerped the soft edge again, steepening it
            // and creeping it outward while the full middle stayed put.
            const uint32_t under = blank;
            float num = 0.0f, den = 0.0f;
            for (int ch = 0; ch < 4; ch++) {
                const float d = tg[ch] - unorm(under, ch);
                num += (o[ch] - unorm(under, ch)) * d;
                den += d * d;
            }
            float cov = den > 1e-6f ? num / den : 0.0f;
            cov = cov < 0.0f ? 0.0f : (cov > 1.0f ? 1.0f : cov);
            for (int ch = 0; ch < 4; ch++) {
                const float on_line = unorm(under, ch) + (tg[ch] - unorm(under, ch)) * cov;
                if (std::fabs(o[ch] - on_line) > 0.02f) { cov = 0.0f; break; }   // other paint: plain lerp
            }
            if (cov >= edge[k]) continue;                       // already at (or past) the edge
            if (cov > 0.0f) w = (edge[k] - cov) * strength / (1.0f - cov);
        }
        uint32_t out = 0;
        for (int ch = 0; ch < 4; ch++)
            out |= to_unorm(o[ch] + (tg[ch] - o[ch]) * w) << (8 * ch);
        if (out == old) continue;
        mesh.color[v] = out;
        verts.push_back(v);
        old_c.push_back(old);
        new_c.push_back(out);
    }
}
