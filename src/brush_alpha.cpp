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
