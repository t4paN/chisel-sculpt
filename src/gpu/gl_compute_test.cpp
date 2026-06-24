// src/gpu/gl_compute_test.cpp
// Headless GL proof for the gpu:: seam's GL backend — the native sibling of the
// WebGPU probe's compute check. Runs the mask kernel (GLSL) through gpu:: on a
// real GL 4.3 context, reads the result back, and compares against a CPU reference
// of the same kernel. Same gpu:: calls the WebGPU probe uses; only the backend and
// the shader source differ. Exits 0 on match, 1 on mismatch.
#include <glad/glad.h>       // must precede any GL header
#define GLFW_INCLUDE_NONE     // don't let GLFW pull its own GL header
#include <GLFW/glfw3.h>

#include "mesh.h"
#include "gpu/gpu.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#ifndef CHISEL_GLSL_DIR
#define CHISEL_GLSL_DIR "shaders/glsl"
#endif
static std::string loadGlsl(const char* name) {
    std::string path = std::string(CHISEL_GLSL_DIR) + "/" + name;
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("[gltest] failed to open glsl: %s\n", path.c_str()); return ""; }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// 48-byte std140 Params, byte-identical to mask_paint.{wgsl,comp} and the probe's
// MaskParamsGPU (vec3 fills a 16-byte slot; a trailing f32 packs in).
struct MaskParamsGPU {
    float anchor_a[3]; float world_radius;
    float anchor_b[3]; float hardness;
    float paint_strength; uint32_t use_b; uint32_t vertex_count; uint32_t _pad0;
};
static_assert(sizeof(MaskParamsGPU) == 48, "Params UBO must be 48 bytes");

static float falloff(float dist, float radius, float hardness) {
    float t = dist / radius;
    float inner = 0.15f + hardness * 0.55f;
    if (t <= inner) return 1.0f;
    float blend = (t - inner) / (1.0f - inner + 1e-6f);
    blend = blend * blend * (3.0f - 2.0f * blend);
    return 1.0f - blend;
}

int main() {
    if (!glfwInit()) { std::printf("[gltest] glfwInit failed\n"); return 2; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // headless: hidden window
    GLFWwindow* win = glfwCreateWindow(64, 64, "chisel gl compute test", nullptr, nullptr);
    if (!win) { std::printf("[gltest] window/context create failed\n"); glfwTerminate(); return 2; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::printf("[gltest] gladLoadGLLoader failed\n"); return 2;
    }
    std::printf("[gltest] GL %s\n", (const char*)glGetString(GL_VERSION));

    gpu::Device dev = gpu::gl_device();

    Mesh mesh = icosphere(4);
    mesh.build_adjacency();
    mesh.recompute_normals();
    const uint32_t vcount = mesh.vertex_count();
    const uint32_t icount = (uint32_t)mesh.indices.size();
    std::printf("[gltest] icosphere: %u verts, %u tris\n", vcount, icount / 3);

    std::vector<float> posBuf(vcount * 3);
    for (uint32_t i = 0; i < vcount; ++i) {
        posBuf[i*3+0] = mesh.pos_x[i]; posBuf[i*3+1] = mesh.pos_y[i]; posBuf[i*3+2] = mesh.pos_z[i];
    }

    const uint64_t kPosBytes   = (uint64_t)vcount*3*sizeof(float);
    const uint64_t kMaskBytes  = (uint64_t)vcount*sizeof(float);
    const uint64_t kDirtyBytes = (uint64_t)(vcount+1)*sizeof(uint32_t);

    gpu::Buffer posVB    = gpu::create_buffer(dev, posBuf.data(), kPosBytes, gpu::Usage::Storage);
    std::vector<float> zeroMask(vcount, 0.0f);
    gpu::Buffer maskBuf  = gpu::create_buffer(dev, zeroMask.data(), kMaskBytes, gpu::Usage::Storage | gpu::Usage::CopySrc);
    std::vector<uint32_t> zeroDirty(vcount + 1, 0u);
    gpu::Buffer dirtyBuf = gpu::create_buffer(dev, zeroDirty.data(), kDirtyBytes, gpu::Usage::Storage | gpu::Usage::CopySrc);
    gpu::Buffer paramsUBO = gpu::create_buffer(dev, nullptr, sizeof(MaskParamsGPU), gpu::Usage::Uniform);
    gpu::Buffer staging   = gpu::create_buffer(dev, nullptr, kPosBytes, gpu::Usage::MapRead);  // sized for the largest readback

    std::string glsl = loadGlsl("mask_paint.comp");
    if (glsl.empty()) return 2;
    gpu::BindEntry maskLayout[] = {
        { 0,  gpu::Bind::StorageRead,      kPosBytes },
        { 12, gpu::Bind::StorageReadWrite, kMaskBytes },
        { 6,  gpu::Bind::StorageReadWrite, kDirtyBytes },
        { 63, gpu::Bind::Uniform,          sizeof(MaskParamsGPU) },
    };
    gpu::ComputePipeline maskPipeline = gpu::create_compute_pipeline(dev, gpu::ShaderSources{ nullptr, glsl.c_str() }, maskLayout, 4);
    if (!maskPipeline.handle) { std::printf("[gltest] mask pipeline failed\n"); return 2; }
    gpu::BindBufferEntry maskBg[] = {
        { 0,  &posVB,     kPosBytes },
        { 12, &maskBuf,   kMaskBytes },
        { 6,  &dirtyBuf,  kDirtyBytes },
        { 63, &paramsUBO, sizeof(MaskParamsGPU) },
    };
    gpu::BindGroup maskBindGroup = gpu::create_bind_group(dev, maskPipeline, maskBg, 4);

    // Fixed dab: triangle 0's centroid, radius 0.45, hardness 0.5, strength 1.
    const float radius = 0.45f, hardness = 0.5f, strength = 1.0f;
    uint32_t i0 = mesh.indices[0], i1 = mesh.indices[1], i2 = mesh.indices[2];
    float anchor[3] = {
        (mesh.pos_x[i0] + mesh.pos_x[i1] + mesh.pos_x[i2]) / 3.0f,
        (mesh.pos_y[i0] + mesh.pos_y[i1] + mesh.pos_y[i2]) / 3.0f,
        (mesh.pos_z[i0] + mesh.pos_z[i1] + mesh.pos_z[i2]) / 3.0f,
    };

    MaskParamsGPU mp = {};
    mp.anchor_a[0] = anchor[0]; mp.anchor_a[1] = anchor[1]; mp.anchor_a[2] = anchor[2];
    mp.world_radius = radius; mp.hardness = hardness; mp.paint_strength = strength;
    mp.use_b = 0u; mp.vertex_count = vcount;
    gpu::write_buffer(dev, paramsUBO, 0, &mp, sizeof(mp));
    uint32_t zero = 0;
    gpu::write_buffer(dev, dirtyBuf, 0, &zero, sizeof(zero));

    uint32_t groups = (vcount + 255u) / 256u;
    gpu::ComputeBatch b = gpu::begin_compute(dev);
    gpu::dispatch(b, maskPipeline, maskBindGroup, groups);
    gpu::copy_buffer(b, dirtyBuf, 0, staging, 0, 4);
    gpu::submit(b);
    uint32_t gpuTouched = 0;
    gpu::map_read(dev, staging, 4, &gpuTouched);

    // read the whole mask back
    std::vector<float> gpuMask(vcount);
    gpu::ComputeBatch b2 = gpu::begin_compute(dev);
    gpu::copy_buffer(b2, maskBuf, 0, staging, 0, kMaskBytes);
    gpu::submit(b2);
    gpu::map_read(dev, staging, kMaskBytes, gpuMask.data());

    uint32_t gpuMasked = 0; float gpuMax = 0.0f;
    for (uint32_t i = 0; i < vcount; ++i) { if (gpuMask[i] > 0.5f) ++gpuMasked; if (gpuMask[i] > gpuMax) gpuMax = gpuMask[i]; }

    // ---- CPU reference of the same kernel ----
    uint32_t cpuTouched = 0, cpuMasked = 0; float cpuMax = 0.0f;
    for (uint32_t v = 0; v < vcount; ++v) {
        float dx = posBuf[v*3+0]-anchor[0], dy = posBuf[v*3+1]-anchor[1], dz = posBuf[v*3+2]-anchor[2];
        float dist = std::sqrt(dx*dx+dy*dy+dz*dz);
        if (dist >= radius) continue;
        float w = falloff(dist, radius, hardness);
        if (w <= 0.0f) continue;
        float val = strength * w; if (val > 1.0f) val = 1.0f;
        if (val == 0.0f) continue;
        ++cpuTouched;
        if (val > 0.5f) ++cpuMasked;
        if (val > cpuMax) cpuMax = val;
    }

    std::printf("[gltest] GPU: touched %u, masked(>0.5) %u, max %.3f\n", gpuTouched, gpuMasked, gpuMax);
    std::printf("[gltest] CPU: touched %u, masked(>0.5) %u, max %.3f\n", cpuTouched, cpuMasked, cpuMax);

    bool pass = (gpuTouched == cpuTouched) && (gpuMasked == cpuMasked) && (std::fabs(gpuMax - cpuMax) < 1e-4f);
    std::printf("[gltest] %s — gpu:: GL backend reproduces the mask kernel\n", pass ? "PASS" : "FAIL");

    // ---- SDF kernel compile smoke check (GLSL only) ----
    // Validates that the 5 self-contained sdf_*.comp compile + link through the seam
    // (the main porting risk: the inlined snippets + the anonymous Params UBO). Full
    // numeric verification is a live in-app mirror-merge. Bind layouts mirror sdf.cpp.
    {
        using B = gpu::Bind;
        struct Stem { const char* name; std::vector<gpu::BindEntry> layout; };
        std::vector<Stem> stems = {
            { "sdf_count",  { {21,B::StorageRead,0},{22,B::StorageRead,0},{36,B::StorageReadWrite,0},{63,B::Uniform,32} } },
            { "sdf_expand", { {21,B::StorageRead,0},{22,B::StorageRead,0},{23,B::StorageReadWrite,0},{36,B::StorageRead,0},{37,B::StorageRead,0},{63,B::Uniform,32} } },
            { "sdf_sign",   { {21,B::StorageRead,0},{22,B::StorageRead,0},{23,B::StorageRead,0},{24,B::StorageReadWrite,0},{63,B::Uniform,48} } },
            { "sdf_mc",     { {24,B::StorageRead,0},{25,B::StorageReadWrite,0},{26,B::StorageReadWrite,0},{27,B::StorageRead,0},{63,B::Uniform,32} } },
            { "sdf_nets",   { {24,B::StorageRead,0},{25,B::StorageReadWrite,0},{26,B::StorageReadWrite,0},{63,B::Uniform,32} } },
        };
        for (auto& s : stems) {
            std::string src = loadGlsl((std::string(s.name) + ".comp").c_str());
            bool ok = false;
            if (!src.empty()) {
                gpu::ComputePipeline p = gpu::create_compute_pipeline(
                    dev, gpu::ShaderSources{ nullptr, src.c_str() }, s.layout.data(), (uint32_t)s.layout.size());
                ok = (p.handle != 0);
                gpu::release_compute_pipeline(p);
            }
            std::printf("[gltest] sdf shader %-10s %s\n", s.name, ok ? "compiled" : "FAILED");
            pass = pass && ok;
        }
    }

    gpu::release_compute_pipeline(maskPipeline);
    gpu::release_bind_group(maskBindGroup);
    gpu::release_buffer(posVB); gpu::release_buffer(maskBuf); gpu::release_buffer(dirtyBuf);
    gpu::release_buffer(paramsUBO); gpu::release_buffer(staging);
    glfwDestroyWindow(win);
    glfwTerminate();
    return pass ? 0 : 1;
}
