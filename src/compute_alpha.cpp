#include "compute.h"
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Brush alpha (stamp) — shared GPU residency for every falloff-computing dab
// kernel. The selected grayscale bitmap lives in a single SSBO (BIND_ALPHA_TEX);
// the per-dab stamp frame rides in a 48-byte UBO (BIND_ALPHA_PARAMS). Each dab
// kernel binds both, samples sample_alpha(), and multiplies its falloff weight.
// Buffers are ALWAYS allocated (1x1 dummy when off) so bind groups stay complete.
// Keep the AlphaParams layout byte-identical to the `AlphaParams` block copied
// into every dab .comp/.wgsl (see the shared snippet in those shaders).
// ---------------------------------------------------------------------------

namespace {
// 48-byte std140 block, byte-identical to the `AlphaParams` block in the dab shaders.
struct AlphaParamsGPU {
    float    tangent[3];   float    inv_diameter;  //  0..15
    float    bitangent[3]; uint32_t enabled;       // 16..31
    uint32_t tex_w;        uint32_t tex_h;
    uint32_t _pad0;        uint32_t _pad1;          // 32..47
};
static_assert(sizeof(AlphaParamsGPU) == 48, "AlphaParams UBO must be 48 bytes");
}

bool ComputeState::init_alpha() {
    if (!supported) return false;
    // 1x1 dummy bitmap so the binding is always valid before any alpha is chosen.
    float dummy = 1.0f;
    alpha_tex_ssbo = gpu::create_buffer(gpu_dev, &dummy, sizeof(float), gpu::Usage::Storage);
    alpha_tex_w = 1;
    alpha_tex_h = 1;
    alpha_params_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(AlphaParamsGPU),
                                          gpu::Usage::Uniform);
    if (!alpha_tex_ssbo.handle || !alpha_params_ubo.handle) {
        std::printf("[compute] alpha buffers failed to allocate\n");
        return false;
    }
    // Start disabled (round brush) until the UI selects a stamp.
    AlphaParamsGPU u = {};
    u.tangent[0] = 1.0f; u.bitangent[1] = 1.0f;
    u.inv_diameter = 0.0f; u.enabled = 0u; u.tex_w = 1; u.tex_h = 1;
    gpu::write_buffer(gpu_dev, alpha_params_ubo, 0, &u, sizeof(u));
    alpha_enabled = false;
    std::printf("[compute] alpha stamp buffers allocated (shared dab binding)\n");
    return true;
}

void ComputeState::upload_alpha(const float* data, uint32_t w, uint32_t h) {
    if (!supported || !alpha_tex_ssbo.handle) return;

    if (!data || w == 0 || h == 0) {
        // Revert to a 1x1 dummy; frames set enabled=false so the round brush is used.
        float dummy = 1.0f;
        gpu::release_buffer(alpha_tex_ssbo);
        alpha_tex_ssbo = gpu::create_buffer(gpu_dev, &dummy, sizeof(float), gpu::Usage::Storage);
        alpha_tex_w = 1;
        alpha_tex_h = 1;
        return;
    }

    uint64_t count = (uint64_t)w * (uint64_t)h;
    gpu::release_buffer(alpha_tex_ssbo);
    alpha_tex_ssbo = gpu::create_buffer(gpu_dev, data, count * sizeof(float), gpu::Usage::Storage);
    alpha_tex_w = w;
    alpha_tex_h = h;
    std::printf("[compute] alpha stamp uploaded: %ux%u\n", w, h);
}

void ComputeState::set_alpha_frame(const float tangent[3], const float bitangent[3],
                                   float inv_diameter, bool enabled) {
    if (!alpha_params_ubo.handle) return;
    AlphaParamsGPU u = {};
    u.tangent[0]   = tangent[0];   u.tangent[1]   = tangent[1];   u.tangent[2]   = tangent[2];
    u.bitangent[0] = bitangent[0]; u.bitangent[1] = bitangent[1]; u.bitangent[2] = bitangent[2];
    u.inv_diameter = inv_diameter;
    u.enabled      = enabled ? 1u : 0u;
    u.tex_w        = alpha_tex_w;
    u.tex_h        = alpha_tex_h;
    gpu::write_buffer(gpu_dev, alpha_params_ubo, 0, &u, sizeof(u));
    alpha_enabled = enabled;
}
