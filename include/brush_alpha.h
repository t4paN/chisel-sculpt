#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Brush-alpha (stamp) library. A flat, growable pool of grayscale bitmaps that
// modulate the brush falloff. Index 0 is always "Round" (no stamp — the classic
// radial brush); the rest are procedural built-ins plus any user-loaded images.
// The selected entry's `data` is uploaded to ComputeState::upload_alpha and sampled
// per-dab by every falloff-computing kernel. Shares the "pool" UI pattern with the
// insert-mesh shape picker.

struct AlphaEntry {
    std::string        name;
    uint32_t           w = 0, h = 0;
    std::vector<float> data;          // w*h, 0..1, row-major (empty when is_round)
    float              preview[256];  // 16x16 downsample for the toolbar swatch
    bool               is_round = false;
};

struct AlphaLibrary {
    std::vector<AlphaEntry> entries;

    // Populate index 0 (Round) + the procedural presets. Call once at startup.
    void init_builtins();

    // Load a grayscale bitmap (any stb-supported format) as a new appended entry.
    // Colour images are averaged to luminance. Returns the new index, or -1 on
    // failure. `display_name` labels it in the picker (falls back to the filename).
    int load_custom(const char* path, const char* display_name = nullptr);

    int count() const { return (int)entries.size(); }
    const AlphaEntry& get(int i) const { return entries[(i >= 0 && i < count()) ? i : 0]; }
};
