#pragma once
#include "gpu/gpu.h"
#include <cstdint>

// Bitmap-font HUD overlay — on the gpu:: seam. Two render pipelines:
//   text_pipeline  — textured glyph quads sampling the R8 font atlas (the one
//                    sampled texture in all of Chisel; see gpu.h Texture note),
//   panel_pipeline — solid translucent quad behind HUD blocks (UBO-only).
// Loose GL uniforms become std140 Params UBOs. (WGSL for these two — like every
// render pipeline — lands with the Stage-4 render-WGSL promotion; GLSL only today.)
struct TextOverlay {
    gpu::Device gpu_dev;

    gpu::RenderPipeline text_pipeline;
    gpu::Buffer         text_ubo;        // std140 { vec2 uScreen; vec4 uColor; }
    gpu::Buffer         text_vbuf;       // streamed glyph quads (grown as needed)
    gpu::Texture        font_texture;

    gpu::RenderPipeline panel_pipeline;
    gpu::Buffer         panel_ubo;       // std140 { vec2 uOffset,uSize,uScreen; vec4 uColor; }
    gpu::Buffer         panel_vbuf;      // unit quad (TriangleStrip)

    bool initialized = false;

    TextOverlay();
    ~TextOverlay();

    void init(gpu::Device& dev);
    void draw_text(const char* text, float x, float y, float scale,
                   int screen_w, int screen_h,
                   float r, float g, float b, float a);
    void draw_panel(float x, float y, float w, float h,
                    int screen_w, int screen_h,
                    float r, float g, float b, float a);
};
