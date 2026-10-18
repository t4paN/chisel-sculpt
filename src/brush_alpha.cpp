#include "brush_alpha.h"
#include "stb/stb_image.h"   // stbi_load — already vendored (used by window_icon too)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Build the 16x16 preview by nearest-sampling the full bitmap.
void build_preview(AlphaEntry& e) {
    if (e.data.empty() || e.w == 0 || e.h == 0) {
        for (float& p : e.preview) p = 0.0f;
        return;
    }
    for (int py = 0; py < 16; py++) {
        for (int px = 0; px < 16; px++) {
            uint32_t sx = (uint32_t)((px + 0.5f) / 16.0f * e.w);
            uint32_t sy = (uint32_t)((py + 0.5f) / 16.0f * e.h);
            if (sx >= e.w) sx = e.w - 1;
            if (sy >= e.h) sy = e.h - 1;
            e.preview[py * 16 + px] = e.data[sy * e.w + sx];
        }
    }
}

// A procedural preset generated from a normalized (u,v) in [-1,1]^2 (stamp square).
template <typename F>
AlphaEntry make_preset(const char* name, uint32_t res, F fn) {
    AlphaEntry e;
    e.name = name;
    e.w = res;
    e.h = res;
    e.data.resize((size_t)res * res);
    for (uint32_t y = 0; y < res; y++) {
        for (uint32_t x = 0; x < res; x++) {
            float u = ((x + 0.5f) / res) * 2.0f - 1.0f;
            float v = ((y + 0.5f) / res) * 2.0f - 1.0f;
            float a = fn(u, v);
            e.data[y * res + x] = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
        }
    }
    build_preview(e);
    return e;
}

float smoothstep01(float a, float b, float t) {
    if (b - a < 1e-6f) return t >= b ? 1.0f : 0.0f;
    t = (t - a) / (b - a);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

// Cheap hash-based value noise for the "Grunge" preset.
float hash2(int x, int y) {
    uint32_t h = (uint32_t)(x * 374761393 + y * 668265263);
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xFFFFFF) / (float)0xFFFFFF;
}
float value_noise(float u, float v) {
    float fx = u * 6.0f, fy = v * 6.0f;
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    float tx = fx - x0, ty = fy - y0;
    float a = hash2(x0, y0), b = hash2(x0 + 1, y0);
    float c = hash2(x0, y0 + 1), d = hash2(x0 + 1, y0 + 1);
    float sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

}  // namespace

void AlphaLibrary::init_builtins() {
    entries.clear();

    // Index 0: Round — no stamp, classic radial brush.
    AlphaEntry round;
    round.name = "Round";
    round.is_round = true;
    for (float& p : round.preview) p = 0.0f;
    // A filled disc preview so the swatch reads as "the default round brush".
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            float u = (x + 0.5f) / 16.0f * 2 - 1, v = (y + 0.5f) / 16.0f * 2 - 1;
            round.preview[y * 16 + x] = std::sqrt(u * u + v * v) < 0.85f ? 1.0f : 0.0f;
        }
    entries.push_back(std::move(round));

    // Soft: radial smooth falloff (a gentler bump than Round's hard-ish core).
    entries.push_back(make_preset("Soft", 128, [](float u, float v) {
        float r = std::sqrt(u * u + v * v);
        return 1.0f - smoothstep01(0.0f, 1.0f, r);
    }));

    // Ring: annulus — 0 at centre and edge, peak at mid radius.
    entries.push_back(make_preset("Ring", 128, [](float u, float v) {
        float r = std::sqrt(u * u + v * v);
        float d = std::fabs(r - 0.6f);
        return 1.0f - smoothstep01(0.0f, 0.28f, d);
    }));

    // Square: filled square (chisel/stamp shape), inscribed in the dab circle so the
    // corners survive the kernels' dist < radius gate (corner distance is m*sqrt(2) of
    // the half-diameter — outer extent must stay under 0.707). Edge band is ~2.5 texels
    // at 256; bilinear sampling in the kernel turns that into clean anti-aliasing.
    entries.push_back(make_preset("Square", 256, [](float u, float v) {
        float m = std::fmax(std::fabs(u), std::fabs(v));
        return 1.0f - smoothstep01(0.685f, 0.705f, m);
    }));

    // Grunge: value noise, edge-faded so it stays inside the dab.
    entries.push_back(make_preset("Grunge", 128, [](float u, float v) {
        float r = std::sqrt(u * u + v * v);
        float edge = 1.0f - smoothstep01(0.7f, 1.0f, r);
        float n = value_noise(u, v);
        n = smoothstep01(0.35f, 0.75f, n);
        return n * edge;
    }));
}

int AlphaLibrary::load_custom(const char* path, const char* display_name) {
    int w = 0, h = 0, ch = 0;
    // Force single-channel: stb converts to grayscale (luminance) for us.
    unsigned char* pixels = stbi_load(path, &w, &h, &ch, 1);
    if (!pixels || w <= 0 || h <= 0) {
        std::printf("[alpha] load failed: %s\n", path ? path : "(null)");
        if (pixels) stbi_image_free(pixels);
        return -1;
    }

    AlphaEntry e;
    if (display_name && display_name[0]) {
        e.name = display_name;
    } else {
        std::string s = path;
        auto slash = s.find_last_of("/\\");
        e.name = (slash == std::string::npos) ? s : s.substr(slash + 1);
    }
    e.w = (uint32_t)w;
    e.h = (uint32_t)h;
    e.data.resize((size_t)w * h);
    for (size_t i = 0; i < e.data.size(); i++)
        e.data[i] = pixels[i] / 255.0f;
    stbi_image_free(pixels);
    build_preview(e);

    std::string logged_name = e.name;
    entries.push_back(std::move(e));
    std::printf("[alpha] loaded %s (%dx%d) as index %d\n",
                logged_name.c_str(), w, h, count() - 1);
    return count() - 1;
}

// ---- fill-bucket patterns ----

namespace {

// One tile on (u,v) in [0,1)^2. Edges are smoothed over about a texel (e) so bilinear
// sampling across the wrap has no hard seam; every shape here repeats exactly at 1.
template <typename F>
AlphaEntry make_tile(const char* name, uint32_t res, F fn) {
    AlphaEntry e;
    e.name = name;
    e.w = res;
    e.h = res;
    e.data.resize((size_t)res * res);
    const float px = 1.0f / res;
    for (uint32_t y = 0; y < res; y++)
        for (uint32_t x = 0; x < res; x++) {
            float a = fn((x + 0.5f) * px, (y + 0.5f) * px, px);
            e.data[y * res + x] = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
        }
    build_preview(e);
    return e;
}

float fract(float x) { return x - std::floor(x); }
float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
// The tiles are stored SOFT: a ramp with 0.5 exactly on each shape's edge, as wide as
// the shape allows. The bucket's Hardness sharpens that ramp at fill time
// (fill_pattern_hardness), so one tile serves every edge from smooth to crisp.
// Signed band: +1 mid-band, 0 on its edges, -1 halfway between bands.
float band_s(float x) { return 1.0f - 4.0f * std::fabs(fract(x) - 0.5f); }

AlphaLibrary build_fill_patterns() {
    AlphaLibrary lib;
    AlphaEntry solid;
    solid.name = "Solid";
    solid.is_round = true;
    for (float& p : solid.preview) p = 0.0f;   // all main colour (black)
    lib.entries.push_back(std::move(solid));

    AlphaEntry grad;   // no bitmap: the bucket lays it along the drag
    grad.name = "Gradient (drag)";
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) grad.preview[y * 16 + x] = (x + 0.5f) / 16.0f;
    lib.entries.push_back(std::move(grad));

    lib.entries.push_back(make_tile("Stripes", 128, [](float u, float, float) {
        return 0.5f + 0.5f * band_s(u);
    }));
    lib.entries.push_back(make_tile("Checks", 128, [](float u, float v, float) {
        return 0.5f - 0.5f * band_s(u) * band_s(v);   // inside one band, outside the other
    }));
    lib.entries.push_back(make_tile("Grid", 128, [](float u, float v, float) {
        // Lines 0.12 wide, ramping out to 0 at twice that.
        const float du = std::fabs(fract(u + 0.5f) - 0.5f), dv = std::fabs(fract(v + 0.5f) - 0.5f);
        return clamp01(0.5f + (0.06f - std::fmin(du, dv)) * (0.5f / 0.06f));
    }));
    lib.entries.push_back(make_tile("Dots", 128, [](float u, float v, float) {
        const float du = fract(u) - 0.5f, dv = fract(v) - 0.5f;
        return clamp01(0.5f + (0.27f - std::sqrt(du * du + dv * dv)) * (0.5f / 0.27f));
    }));
    lib.entries.push_back(make_tile("Zigzag", 128, [](float u, float v, float) {
        // Two chevron bands per tile; the triangle wave keeps both ends of u level.
        const float tri = std::fabs(fract(u) - 0.5f);
        return 0.5f + 0.5f * band_s(2.0f * (v + tri));
    }));
    // The swatches show a crisp version, the way the default hardness lays it down.
    for (AlphaEntry& e : lib.entries)
        if (!e.data.empty())
            for (float& p : e.preview) p = fill_pattern_hardness(p, 0.85f);
    return lib;
}

}  // namespace

float fill_pattern_hardness(float v, float hardness) {
    // Half-width of the ramp around 0.5: the whole range at 0, a sliver at 1.
    const float half = std::fmax(0.5f * (1.0f - clamp01(hardness)), 0.004f);
    return clamp01((v - 0.5f) / (2.0f * half) + 0.5f);
}

const AlphaLibrary& fill_patterns() {
    static const AlphaLibrary lib = build_fill_patterns();
    return lib;
}
