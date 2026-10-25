#pragma once
// WebGPU startup and the window's surface (src/gpu/wgpu_surface.cpp). WebGPU builds only.
#ifdef CHISEL_BACKEND_WEBGPU
#include <webgpu/webgpu.h>

struct GLFWwindow;

extern WGPUSurface       g_surface;
extern WGPUDevice        g_device;
extern WGPUTextureFormat g_surface_fmt;
// Set when a frame's surface texture comes back suboptimal: the next frame
// reconfigures before acquiring.
extern bool              g_surface_reconfigure;

void configureSurface(int w, int h);
WGPUTextureView spareColorView(int w, int h);
void makeDepth(int w, int h);
// Instance, surface, adapter, device, the seam's device + limits, surface format,
// first configure + depth. False (with a message on stderr) if any step fails.
bool wgpu_init(GLFWwindow* window, int fbw, int fbh);
#endif
