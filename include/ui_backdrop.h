#pragma once
#include "imgui.h"

// Frosted-glass backdrop for the modern skin's panels. ImGui has no backdrop
// blur, so once per frame — after the scene is drawn, before the ImGui pass — the
// frame is copied, shrunk to quarter resolution and Gaussian-blurred (two passes).
// Panels then draw their screen rect of that texture under their tint.
//
// Cost scales with the material dial: at Solid (blur 0) nothing runs at all.
// Backend-native (GL / WebGPU) rather than going through gpu::, because the result
// has to be an ImTextureID the ImGui renderer can sample, and the WebGPU side needs
// the swapchain texture, which gpu:: never sees.
namespace ui_backdrop {

// `frame_texture`: the current swapchain WGPUTexture on WebGPU (must have been
// configured with CopySrc); ignored on GL, which reads the default framebuffer.
// `blur_px` is the Gaussian sigma in screen pixels (CSS blur() semantics);
// `saturate` boosts colour like CSS saturate(). blur_px < 0.5 skips the work and
// makes available() false for this frame.
void capture(int win_w, int win_h, float blur_px, float saturate, void* frame_texture);

bool available();            // a blurred frame exists for the current frame
ImTextureID texture();
// UVs of the blurred texture that cover the screen rect [p0, p1].
void uv_for(ImVec2 p0, ImVec2 p1, ImVec2* uv0, ImVec2* uv1);

void shutdown();

} // namespace ui_backdrop
