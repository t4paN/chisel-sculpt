// src/gpu/webgpu_backend.cpp
// WebGPU (wgpu-native v29) implementation of the gpu:: compute seam (gpu/gpu.h).
// Seam Step 1 — compute path only. The render seam is added with the Stage 3 port.
#include "gpu/gpu.h"

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpuDevicePoll for the readback busy-wait
#include <cstring>
#include <cstdio>

namespace gpu {

// WGSL string -> sized WGPUStringView (v29 wants explicit length).
static WGPUStringView sv(const char* s) { return WGPUStringView{ s, s ? std::strlen(s) : 0 }; }

static WGPUBufferUsage to_wgpu_usage(Usage u) {
    uint32_t f = (uint32_t)WGPUBufferUsage_CopyDst;   // always writable (matches makeBuf)
    if (has(u, Usage::Vertex))  f |= WGPUBufferUsage_Vertex;
    if (has(u, Usage::Index))   f |= WGPUBufferUsage_Index;
    if (has(u, Usage::Storage)) f |= WGPUBufferUsage_Storage;
    if (has(u, Usage::Uniform)) f |= WGPUBufferUsage_Uniform;
    if (has(u, Usage::CopySrc)) f |= WGPUBufferUsage_CopySrc;
    if (has(u, Usage::MapRead)) f |= WGPUBufferUsage_MapRead;
    return (WGPUBufferUsage)f;
}

static WGPUBufferBindingType to_wgpu_bind(Bind b) {
    switch (b) {
        case Bind::StorageRead:      return WGPUBufferBindingType_ReadOnlyStorage;
        case Bind::StorageReadWrite: return WGPUBufferBindingType_Storage;
        case Bind::Uniform:          return WGPUBufferBindingType_Uniform;
    }
    return WGPUBufferBindingType_Storage;
}

Device device_from_webgpu(WGPUDevice device, WGPUQueue queue) {
    Device d;
    d.device = device;
    d.queue  = queue;
    return d;
}

Buffer create_buffer(Device& dev, const void* data, uint64_t size, Usage usage) {
    Buffer b;
    b.size = size;
    WGPUBufferDescriptor bd = {};
    bd.usage = to_wgpu_usage(usage);
    bd.size  = size;
    b.handle = wgpuDeviceCreateBuffer(dev.device, &bd);
    if (b.handle && data) wgpuQueueWriteBuffer(dev.queue, b.handle, 0, data, size);
    return b;
}

void write_buffer(Device& dev, Buffer& b, uint64_t offset, const void* data, uint64_t size) {
    wgpuQueueWriteBuffer(dev.queue, b.handle, offset, data, size);
}

void release_buffer(Buffer& b) {
    if (b.handle) { wgpuBufferRelease(b.handle); b.handle = nullptr; }
    b.size = 0;
}

ComputePipeline create_compute_pipeline(Device& dev, const ShaderSources& src,
                                        const BindEntry* entries, uint32_t n,
                                        const char* entry_point) {
    ComputePipeline p;
    if (!src.wgsl) { std::printf("[gpu] no WGSL source for compute pipeline\n"); return p; }

    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(src.wgsl);
    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl.chain;
    p.module = wgpuDeviceCreateShaderModule(dev.device, &smd);
    if (!p.module) { std::printf("[gpu] compute shader module failed\n"); return p; }

    // bind-group layout from the declared entries (all compute-visible)
    WGPUBindGroupLayoutEntry stackEnt[16] = {};
    for (uint32_t i = 0; i < n && i < 16; ++i) {
        stackEnt[i].binding = entries[i].binding;
        stackEnt[i].visibility = WGPUShaderStage_Compute;
        stackEnt[i].buffer.type = to_wgpu_bind(entries[i].type);
        stackEnt[i].buffer.minBindingSize = entries[i].min_size;
    }
    WGPUBindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = n;
    bgld.entries = stackEnt;
    p.bgl = wgpuDeviceCreateBindGroupLayout(dev.device, &bgld);

    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &p.bgl;
    p.pl = wgpuDeviceCreatePipelineLayout(dev.device, &pld);

    WGPUComputePipelineDescriptor cpd = {};
    cpd.layout = p.pl;
    cpd.compute.module = p.module;
    cpd.compute.entryPoint = sv(entry_point);
    p.handle = wgpuDeviceCreateComputePipeline(dev.device, &cpd);
    if (!p.handle) std::printf("[gpu] compute pipeline failed\n");
    return p;
}

void release_compute_pipeline(ComputePipeline& p) {
    if (p.handle) { wgpuComputePipelineRelease(p.handle); p.handle = nullptr; }
    if (p.pl)     { wgpuPipelineLayoutRelease(p.pl);      p.pl     = nullptr; }
    if (p.bgl)    { wgpuBindGroupLayoutRelease(p.bgl);    p.bgl    = nullptr; }
    if (p.module) { wgpuShaderModuleRelease(p.module);    p.module = nullptr; }
}

BindGroup create_bind_group(Device& dev, ComputePipeline& pipe,
                            const BindBufferEntry* entries, uint32_t n) {
    BindGroup g;
    WGPUBindGroupEntry stackEnt[16] = {};
    for (uint32_t i = 0; i < n && i < 16; ++i) {
        stackEnt[i].binding = entries[i].binding;
        stackEnt[i].buffer  = entries[i].buffer->handle;
        stackEnt[i].offset  = 0;
        stackEnt[i].size    = entries[i].size;
    }
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout = pipe.bgl;
    bgd.entryCount = n;
    bgd.entries = stackEnt;
    g.handle = wgpuDeviceCreateBindGroup(dev.device, &bgd);
    return g;
}

void release_bind_group(BindGroup& g) {
    if (g.handle) { wgpuBindGroupRelease(g.handle); g.handle = nullptr; }
}

ComputeBatch begin_compute(Device& dev) {
    ComputeBatch b;
    b.dev  = &dev;
    b.enc  = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    b.pass = wgpuCommandEncoderBeginComputePass(b.enc, nullptr);
    return b;
}

void dispatch(ComputeBatch& b, ComputePipeline& pipe, BindGroup& group, uint32_t groups_x) {
    wgpuComputePassEncoderSetPipeline(b.pass, pipe.handle);
    wgpuComputePassEncoderSetBindGroup(b.pass, 0, group.handle, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(b.pass, groups_x, 1, 1);
}

void end_compute_pass(ComputeBatch& b) {
    if (b.pass) {
        wgpuComputePassEncoderEnd(b.pass);
        wgpuComputePassEncoderRelease(b.pass);
        b.pass = nullptr;
    }
}

void copy_buffer(ComputeBatch& b, const Buffer& src, uint64_t src_off,
                 const Buffer& dst, uint64_t dst_off, uint64_t size) {
    // copies are encoder-level commands — the compute pass must be ended first.
    if (b.pass) end_compute_pass(b);
    wgpuCommandEncoderCopyBufferToBuffer(b.enc, src.handle, src_off, dst.handle, dst_off, size);
}

void submit(ComputeBatch& b) {
    if (b.pass) end_compute_pass(b);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(b.enc, nullptr);
    wgpuQueueSubmit(b.dev->queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(b.enc);
    b.enc = nullptr;
}

// one-shot map-read busy-wait (same pattern as the probe's readU32 / final maps)
struct MapState { bool done = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; };
static void onMap(WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*) {
    auto* s = static_cast<MapState*>(ud1);
    s->done = true; s->status = status;
}

void map_read(Device& dev, const Buffer& staging, uint64_t size, void* out) {
    MapState ms;
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode = WGPUCallbackMode_AllowProcessEvents;
    mcb.callback = onMap; mcb.userdata1 = &ms;
    wgpuBufferMapAsync(staging.handle, WGPUMapMode_Read, 0, size, mcb);
    while (!ms.done) wgpuDevicePoll(dev.device, true, nullptr);
    if (ms.status == WGPUMapAsyncStatus_Success) {
        const void* p = wgpuBufferGetMappedRange(staging.handle, 0, size);
        if (p) std::memcpy(out, p, size); else std::memset(out, 0, size);
        wgpuBufferUnmap(staging.handle);
    } else {
        std::memset(out, 0, size);
        wgpuBufferUnmap(staging.handle);
    }
}

} // namespace gpu
