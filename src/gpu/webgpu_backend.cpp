// src/gpu/webgpu_backend.cpp
// WebGPU (wgpu-native v29) implementation of the gpu:: compute seam (gpu/gpu.h).
// Seam Step 1 — compute path only. The render seam is added with the Stage 3 port.
#include "gpu/gpu.h"

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>   // wgpuDevicePoll for the readback busy-wait
#include <cstring>
#include <cstdio>
#include <vector>

namespace gpu {

// WGSL string -> sized WGPUStringView (v29 wants explicit length).
static WGPUStringView sv(const char* s) { return WGPUStringView{ s, s ? std::strlen(s) : 0 }; }

static WGPUBufferUsage to_wgpu_usage(Usage u) {
    // MAP_READ is exclusive in WebGPU: it may only pair with COPY_DST (used by the
    // internal readback staging). Any other flag on a mappable buffer is rejected.
    if (has(u, Usage::MapRead))
        return (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

    uint32_t f = (uint32_t)WGPUBufferUsage_CopyDst;   // always writable (matches makeBuf)
    if (has(u, Usage::Vertex))  f |= WGPUBufferUsage_Vertex;
    if (has(u, Usage::Index))   f |= WGPUBufferUsage_Index;
    if (has(u, Usage::Storage)) f |= WGPUBufferUsage_Storage;
    if (has(u, Usage::Uniform)) f |= WGPUBufferUsage_Uniform;
    // Symmetric with the always-on CopyDst: any non-mappable buffer may be a copy or
    // read_buffer source. The gpu:: seam copies freely between buffers (stroke-normal
    // snapshot, undo ring, remesh ping-pong, limb scratch, SDF readbacks, …) and
    // WebGPU aborts a copy whose source lacks COPY_SRC. GL ignores usage flags, so
    // this only bites here; the explicit Usage::CopySrc bit becomes a no-op.
    f |= WGPUBufferUsage_CopySrc;
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

static Device g_app_device;
void set_app_device(const Device& d) { g_app_device = d; }
Device app_device() { return g_app_device; }

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
        // size 0 means "whole buffer" on the seam (GL binds the whole range); WebGPU
        // rejects a zero-sized binding, so map it to the WHOLE_SIZE sentinel.
        stackEnt[i].size    = entries[i].size ? entries[i].size : WGPU_WHOLE_SIZE;
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

void dispatch(ComputeBatch& b, ComputePipeline& pipe, BindGroup& group, uint32_t groups_x, uint32_t groups_y) {
    wgpuComputePassEncoderSetPipeline(b.pass, pipe.handle);
    wgpuComputePassEncoderSetBindGroup(b.pass, 0, group.handle, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(b.pass, groups_x, groups_y, 1);
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

// ---- One-shot buffer ops -----------------------------------------------------

// A single growable readback staging buffer (MapRead|CopyDst) reused by read_buffer.
// The app is single-threaded and readbacks are discrete one-shot points, so one
// scratch buffer that grows to the largest request is enough.
static WGPUBuffer g_read_staging = nullptr;
static uint64_t   g_read_staging_size = 0;

void read_buffer(Device& dev, const Buffer& src, uint64_t offset, uint64_t size, void* out) {
    if (g_read_staging_size < size) {
        if (g_read_staging) wgpuBufferRelease(g_read_staging);
        WGPUBufferDescriptor bd = {};
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        bd.size  = size;
        g_read_staging = wgpuDeviceCreateBuffer(dev.device, &bd);
        g_read_staging_size = size;
    }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, offset, g_read_staging, 0, size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    MapState ms;
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode = WGPUCallbackMode_AllowProcessEvents;
    mcb.callback = onMap; mcb.userdata1 = &ms;
    wgpuBufferMapAsync(g_read_staging, WGPUMapMode_Read, 0, size, mcb);
    while (!ms.done) wgpuDevicePoll(dev.device, true, nullptr);
    if (ms.status == WGPUMapAsyncStatus_Success) {
        const void* p = wgpuBufferGetMappedRange(g_read_staging, 0, size);
        if (p) std::memcpy(out, p, size); else std::memset(out, 0, size);
    } else {
        std::memset(out, 0, size);
    }
    wgpuBufferUnmap(g_read_staging);
}

void clear_buffer(Device& dev, Buffer& b, uint32_t fill_word) {
    if (fill_word == 0) {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
        wgpuCommandEncoderClearBuffer(enc, b.handle, 0, b.size);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(dev.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        return;
    }
    // Non-zero pattern (the SDF band sentinel): WebGPU's ClearBuffer only zeroes, so
    // upload a CPU-filled vector. This is a one-shot merge path (readback-legal), not
    // a hot path; a fill compute kernel is the web-stage fast follow-up if needed.
    uint64_t words = b.size / 4;
    std::vector<uint32_t> fill((size_t)words, fill_word);
    wgpuQueueWriteBuffer(dev.queue, b.handle, 0, fill.data(), (size_t)words * 4);
}

void copy_buffer(Device& dev, const Buffer& src, uint64_t src_off,
                 const Buffer& dst, uint64_t dst_off, uint64_t size) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src.handle, src_off, dst.handle, dst_off, size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
}

void resize_buffer(Device& dev, Buffer& b, uint64_t new_size, Usage usage) {
    // WebGPU buffers are immutable in size — release and recreate. The handle changes,
    // so callers must rebuild any bind group that referenced the old handle (see gpu.h).
    if (b.handle) wgpuBufferRelease(b.handle);
    WGPUBufferDescriptor bd = {};
    bd.usage = to_wgpu_usage(usage);
    bd.size  = new_size;
    b.handle = wgpuDeviceCreateBuffer(dev.device, &bd);
    b.size   = new_size;
}

void barrier(Device&) { /* no-op: submit ordering serialises dependent compute work */ }

// ============================ Render-side seam ============================
// WebGPU impl of gpu.h's render primitives, generalized from the proven probe
// (wgpu_window.cpp). Model: ENCODER-PER-PASS — begin_render_pass creates an encoder
// + render pass; end_render_pass ends, finishes and submits it. A frame issues
// several passes against the swapchain view (bg, matcap, cursor, …); all but the
// first use loadOp=Load to accumulate. The windowing code acquires the surface view
// into RenderTarget.color and presents (wgpuSurfacePresent) after the last pass.

// Backend globals the windowing code injects (see gpu.h webgpu setters).
static WGPUTextureFormat g_surface_format = WGPUTextureFormat_BGRA8Unorm;
static WGPUTextureView   g_default_depth  = nullptr;
static WGPUTextureView   g_default_color  = nullptr;
static const WGPUTextureFormat kDepthFormat = WGPUTextureFormat_Depth24Plus;

void webgpu_set_surface_format(WGPUTextureFormat f) { g_surface_format = f; }
void webgpu_set_default_depth(WGPUTextureView v)    { g_default_depth = v; }
void webgpu_set_default_color(WGPUTextureView v)    { g_default_color = v; }
WGPUTextureFormat webgpu_depth_format()             { return kDepthFormat; }

static WGPUVertexFormat to_wgpu_vformat(VertexFormat f) {
    switch (f) {
        case VertexFormat::F32:       return WGPUVertexFormat_Float32;
        case VertexFormat::F32x2:     return WGPUVertexFormat_Float32x2;
        case VertexFormat::F32x3:     return WGPUVertexFormat_Float32x3;
        case VertexFormat::F32x4:     return WGPUVertexFormat_Float32x4;
        case VertexFormat::U8x4_norm: return WGPUVertexFormat_Unorm8x4;
    }
    return WGPUVertexFormat_Float32;
}

static WGPUPrimitiveTopology to_wgpu_topology(Topology t) {
    switch (t) {
        case Topology::Triangles:     return WGPUPrimitiveTopology_TriangleList;
        case Topology::TriangleStrip: return WGPUPrimitiveTopology_TriangleStrip;
        case Topology::Lines:         return WGPUPrimitiveTopology_LineList;
        case Topology::Points:        return WGPUPrimitiveTopology_PointList;
    }
    return WGPUPrimitiveTopology_TriangleList;
}

// TexFormat -> WGPUTextureFormat. NOTE: WebGPU has no renderable 3-channel 16F, so
// RGB16F (the GL world-normal attachment) maps to RGBA16Float (4ch) here; the extra
// channel is padding. read_target_region reconciles the readback layout.
static WGPUTextureFormat to_wgpu_texformat(TexFormat f) {
    switch (f) {
        case TexFormat::R32F:   return WGPUTextureFormat_R32Float;
        case TexFormat::RGB16F: return WGPUTextureFormat_RGBA16Float;
        case TexFormat::RG16F:  return WGPUTextureFormat_RG16Float;
        case TexFormat::R32UI:  return WGPUTextureFormat_R32Uint;
    }
    return WGPUTextureFormat_R32Float;
}

RenderPipeline create_render_pipeline(Device& dev, const RenderPipelineDesc& d) {
    RenderPipeline p;
    if (!d.shaders.wgsl) { std::printf("[gpu] no WGSL source for render pipeline\n"); return p; }

    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(d.shaders.wgsl);
    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl.chain;
    p.module = wgpuDeviceCreateShaderModule(dev.device, &smd);
    if (!p.module) { std::printf("[gpu] render shader module failed\n"); return p; }

    // Bind-group layout: the Params UBO(s) — visible to both stages, some fragment
    // shaders read the same Params block — plus, for a sampled_texture pipeline (the
    // font atlas), a fragment-stage texture + sampler at kTextureBinding/kSamplerBinding.
    if (d.bind_count > 0 || d.sampled_texture) {
        WGPUBindGroupLayoutEntry stackEnt[kMaxBindings + 2] = {};
        uint32_t n = d.bind_count < kMaxBindings ? d.bind_count : kMaxBindings;
        for (uint32_t i = 0; i < n; ++i) {
            stackEnt[i].binding = d.binds[i].binding;
            stackEnt[i].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
            stackEnt[i].buffer.type = to_wgpu_bind(d.binds[i].type);
            stackEnt[i].buffer.minBindingSize = d.binds[i].min_size;
        }
        if (d.sampled_texture) {
            stackEnt[n].binding = kTextureBinding;
            stackEnt[n].visibility = WGPUShaderStage_Fragment;
            stackEnt[n].texture.sampleType = WGPUTextureSampleType_Float;   // R8Unorm is filterable
            stackEnt[n].texture.viewDimension = WGPUTextureViewDimension_2D;
            ++n;
            stackEnt[n].binding = kSamplerBinding;
            stackEnt[n].visibility = WGPUShaderStage_Fragment;
            stackEnt[n].sampler.type = WGPUSamplerBindingType_Filtering;
            ++n;
        }
        WGPUBindGroupLayoutDescriptor bgld = {};
        bgld.entryCount = n; bgld.entries = stackEnt;
        p.bgl = wgpuDeviceCreateBindGroupLayout(dev.device, &bgld);
        WGPUPipelineLayoutDescriptor pld = {};
        pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &p.bgl;
        p.pl = wgpuDeviceCreatePipelineLayout(dev.device, &pld);
    }

    // Vertex buffer layouts: one WGPUVertexBufferLayout per slot, gathering the
    // attributes whose .slot == that slot. Buffer index == slot (matches the seam's
    // set_vertex_buffer(slot, ...) and the GL backend's slot model).
    WGPUVertexAttribute   vattr[kMaxVertexAttrs] = {};
    WGPUVertexBufferLayout vbl[kMaxVertexAttrs]  = {};
    uint32_t slot_n = d.slot_count < kMaxVertexAttrs ? d.slot_count : kMaxVertexAttrs;
    uint32_t attr_n = d.attr_count < kMaxVertexAttrs ? d.attr_count : kMaxVertexAttrs;
    // attribute storage is laid out per slot; track each slot's count + base.
    uint32_t slot_base[kMaxVertexAttrs] = {}, slot_cnt[kMaxVertexAttrs] = {};
    uint32_t cursor = 0;
    for (uint32_t s = 0; s < slot_n; ++s) {
        slot_base[s] = cursor;
        for (uint32_t i = 0; i < attr_n; ++i) {
            if (d.attrs[i].slot != s) continue;
            vattr[cursor].format = to_wgpu_vformat(d.attrs[i].format);
            vattr[cursor].offset = d.attrs[i].offset;
            vattr[cursor].shaderLocation = d.attrs[i].location;
            ++cursor; ++slot_cnt[s];
        }
        vbl[s].arrayStride = d.slots[s].stride;
        vbl[s].stepMode = WGPUVertexStepMode_Vertex;
        vbl[s].attributeCount = slot_cnt[s];
        vbl[s].attributes = &vattr[slot_base[s]];
    }

    // Colour targets: null color_targets => one swapchain target (surface format,
    // honour blend); else the listed offscreen formats (data targets, no blend).
    WGPUColorTargetState targets[kMaxColorAttachments] = {};
    WGPUBlendState blend = {};
    uint32_t tcount;
    if (!d.color_targets || d.color_target_count == 0) {
        tcount = 1;
        targets[0].format = g_surface_format;
        targets[0].writeMask = WGPUColorWriteMask_All;
        if (d.blend) {
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            targets[0].blend = &blend;
        }
    } else {
        tcount = d.color_target_count < kMaxColorAttachments ? d.color_target_count : kMaxColorAttachments;
        for (uint32_t i = 0; i < tcount; ++i) {
            targets[i].format = to_wgpu_texformat(d.color_targets[i]);
            targets[i].writeMask = WGPUColorWriteMask_All;
        }
    }
    WGPUFragmentState frag = {};
    frag.module = p.module; frag.entryPoint = sv("fs_main");
    frag.targetCount = tcount; frag.targets = targets;

    // Every pass carries depth, so every pipeline declares the depth format.
    // depth_test off => Always (no cull); depth_write maps directly.
    WGPUDepthStencilState depth = {};
    depth.format = kDepthFormat;
    depth.depthWriteEnabled = d.depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    depth.depthCompare = d.depth_test ? WGPUCompareFunction_Less : WGPUCompareFunction_Always;
    depth.stencilFront.compare = WGPUCompareFunction_Always;
    depth.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor pd = {};
    pd.layout = p.pl;  // null => auto layout (pipelines with no UBO)
    pd.vertex.module = p.module; pd.vertex.entryPoint = sv("vs_main");
    pd.vertex.bufferCount = slot_n; pd.vertex.buffers = vbl;
    pd.primitive.topology = to_wgpu_topology(d.topology);
    pd.primitive.frontFace = WGPUFrontFace_CCW;
    pd.primitive.cullMode = WGPUCullMode_None;
    pd.depthStencil = &depth;
    pd.multisample.count = 1; pd.multisample.mask = 0xFFFFFFFFu;
    pd.fragment = &frag;
    p.handle = wgpuDeviceCreateRenderPipeline(dev.device, &pd);
    if (!p.handle) std::printf("[gpu] render pipeline failed\n");
    return p;
}

void release_render_pipeline(RenderPipeline& p) {
    if (p.handle) { wgpuRenderPipelineRelease(p.handle); p.handle = nullptr; }
    if (p.pl)     { wgpuPipelineLayoutRelease(p.pl);     p.pl     = nullptr; }
    if (p.bgl)    { wgpuBindGroupLayoutRelease(p.bgl);   p.bgl    = nullptr; }
    if (p.module) { wgpuShaderModuleRelease(p.module);   p.module = nullptr; }
}

BindGroup create_bind_group(Device& dev, RenderPipeline& pipe,
                            const BindBufferEntry* entries, uint32_t n, const Texture* tex) {
    BindGroup g;
    WGPUBindGroupEntry stackEnt[kMaxBindings + 2] = {};
    uint32_t cnt = n < kMaxBindings ? n : kMaxBindings;
    for (uint32_t i = 0; i < cnt; ++i) {
        stackEnt[i].binding = entries[i].binding;
        stackEnt[i].buffer  = entries[i].buffer->handle;
        stackEnt[i].offset  = 0;
        stackEnt[i].size    = entries[i].size ? entries[i].size : WGPU_WHOLE_SIZE;
    }
    if (tex) {                                   // sampled_texture pipeline (font atlas)
        stackEnt[cnt].binding = kTextureBinding;
        stackEnt[cnt].textureView = tex->view;
        ++cnt;
        stackEnt[cnt].binding = kSamplerBinding;
        stackEnt[cnt].sampler = tex->sampler;
        ++cnt;
    }
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout = pipe.bgl;
    bgd.entryCount = cnt; bgd.entries = stackEnt;
    g.handle = wgpuDeviceCreateBindGroup(dev.device, &bgd);
    return g;
}

RenderPass begin_render_pass(Device& dev, const RenderTarget& t) {
    RenderPass rp; rp.dev = &dev;
    rp.enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);

    WGPURenderPassColorAttachment color = {};
    color.view = t.color ? t.color : g_default_color;  // default-screen pass = swapchain
    color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    color.loadOp  = t.clear ? WGPULoadOp_Clear : WGPULoadOp_Load;
    color.storeOp = WGPUStoreOp_Store;
    color.clearValue = WGPUColor{ t.clear_color[0], t.clear_color[1],
                                  t.clear_color[2], t.clear_color[3] };

    // The first pass of a frame clears (target.clear) — clear depth there too; later
    // passes Load both. The default depth view is injected by the windowing code.
    WGPURenderPassDepthStencilAttachment depthAtt = {};
    depthAtt.view = g_default_depth;
    depthAtt.depthLoadOp  = t.clear ? WGPULoadOp_Clear : WGPULoadOp_Load;
    depthAtt.depthStoreOp = WGPUStoreOp_Store;
    depthAtt.depthClearValue = 1.0f;

    WGPURenderPassDescriptor pd = {};
    pd.colorAttachmentCount = 1; pd.colorAttachments = &color;
    if (g_default_depth) pd.depthStencilAttachment = &depthAtt;
    rp.pass = wgpuCommandEncoderBeginRenderPass(rp.enc, &pd);
    return rp;
}

void set_pipeline(RenderPass& rp, RenderPipeline& pipe) {
    wgpuRenderPassEncoderSetPipeline(rp.pass, pipe.handle);
}

void set_bind_group(RenderPass& rp, RenderPipeline&, BindGroup& g) {
    wgpuRenderPassEncoderSetBindGroup(rp.pass, 0, g.handle, 0, nullptr);
}

void set_vertex_buffer(RenderPass& rp, uint32_t slot, const Buffer& b) {
    wgpuRenderPassEncoderSetVertexBuffer(rp.pass, slot, b.handle, 0, WGPU_WHOLE_SIZE);
}

void set_index_buffer(RenderPass& rp, const Buffer& b) {
    wgpuRenderPassEncoderSetIndexBuffer(rp.pass, b.handle, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
}

void draw(RenderPass& rp, uint32_t vertex_count, uint32_t first_vertex) {
    wgpuRenderPassEncoderDraw(rp.pass, vertex_count, 1, first_vertex, 0);
}

void draw_indexed(RenderPass& rp, uint32_t index_count) {
    wgpuRenderPassEncoderDrawIndexed(rp.pass, index_count, 1, 0, 0, 0);
}

void end_render_pass(RenderPass& rp) {
    if (rp.pass) { wgpuRenderPassEncoderEnd(rp.pass); wgpuRenderPassEncoderRelease(rp.pass); rp.pass = nullptr; }
    if (rp.enc) {
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(rp.enc, nullptr);
        wgpuQueueSubmit(rp.dev->queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(rp.enc);
        rp.enc = nullptr;
    }
}

// ---- Sampled texture (the font atlas) ----------------------------------------

Texture create_sampled_texture(Device& dev, int w, int h, const void* data) {
    Texture t; t.width = w; t.height = h;
    WGPUTextureDescriptor td = {};
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    td.dimension = WGPUTextureDimension_2D;
    td.size = { (uint32_t)w, (uint32_t)h, 1 };
    td.format = WGPUTextureFormat_R8Unorm;
    td.mipLevelCount = 1; td.sampleCount = 1;
    t.handle = wgpuDeviceCreateTexture(dev.device, &td);
    t.view   = wgpuTextureCreateView(t.handle, nullptr);

    if (data) {
        WGPUTexelCopyTextureInfo dst = {};
        dst.texture = t.handle;
        dst.mipLevel = 0;
        dst.origin = { 0, 0, 0 };
        dst.aspect = WGPUTextureAspect_All;
        WGPUTexelCopyBufferLayout layout = {};
        layout.bytesPerRow  = (uint32_t)w;       // R8: one byte per texel, no row padding
        layout.rowsPerImage = (uint32_t)h;
        WGPUExtent3D ext = { (uint32_t)w, (uint32_t)h, 1 };
        wgpuQueueWriteTexture(dev.queue, &dst, data, (size_t)w * h, &layout, &ext);
    }

    WGPUSamplerDescriptor sd = {};
    sd.addressModeU = WGPUAddressMode_ClampToEdge;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = WGPUFilterMode_Nearest;
    sd.minFilter = WGPUFilterMode_Nearest;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.maxAnisotropy = 1;
    t.sampler = wgpuDeviceCreateSampler(dev.device, &sd);
    return t;
}

void release_texture(Texture& t) {
    if (t.sampler) { wgpuSamplerRelease(t.sampler);     t.sampler = nullptr; }
    if (t.view)    { wgpuTextureViewRelease(t.view);    t.view    = nullptr; }
    if (t.handle)  { wgpuTextureRelease(t.handle);      t.handle  = nullptr; }
}

// ---- Offscreen MRT render target + readback ----------------------------------

OffscreenTarget create_offscreen_target(Device& dev, int w, int h,
                                        const TexFormat* fmts, uint32_t color_count) {
    OffscreenTarget t;
    t.width = w; t.height = h;
    t.color_count = color_count < kMaxColorAttachments ? color_count : kMaxColorAttachments;
    for (uint32_t i = 0; i < t.color_count; ++i) {
        t.color_fmt[i] = fmts[i];
        WGPUTextureDescriptor td = {};
        td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
        td.dimension = WGPUTextureDimension_2D;
        td.size = { (uint32_t)w, (uint32_t)h, 1 };
        td.format = to_wgpu_texformat(fmts[i]);
        td.mipLevelCount = 1; td.sampleCount = 1;
        t.color_tex[i]  = wgpuDeviceCreateTexture(dev.device, &td);
        t.color_view[i] = wgpuTextureCreateView(t.color_tex[i], nullptr);
    }
    WGPUTextureDescriptor zd = {};
    zd.usage = WGPUTextureUsage_RenderAttachment;
    zd.dimension = WGPUTextureDimension_2D;
    zd.size = { (uint32_t)w, (uint32_t)h, 1 };
    zd.format = kDepthFormat;
    zd.mipLevelCount = 1; zd.sampleCount = 1;
    t.depth_tex  = wgpuDeviceCreateTexture(dev.device, &zd);
    t.depth_view = wgpuTextureCreateView(t.depth_tex, nullptr);
    return t;
}

void resize_offscreen_target(Device& dev, OffscreenTarget& t, int w, int h) {
    if (w == t.width && h == t.height) return;
    TexFormat fmts[kMaxColorAttachments];
    uint32_t cc = t.color_count;
    for (uint32_t i = 0; i < cc; ++i) fmts[i] = t.color_fmt[i];
    release_offscreen_target(t);
    t = create_offscreen_target(dev, w, h, fmts, cc);
}

void release_offscreen_target(OffscreenTarget& t) {
    for (uint32_t i = 0; i < t.color_count; ++i) {
        if (t.color_view[i]) wgpuTextureViewRelease(t.color_view[i]);
        if (t.color_tex[i])  wgpuTextureRelease(t.color_tex[i]);
    }
    if (t.depth_view) wgpuTextureViewRelease(t.depth_view);
    if (t.depth_tex)  wgpuTextureRelease(t.depth_tex);
    if (t.readback)   wgpuBufferRelease(t.readback);
    t = OffscreenTarget();
}

RenderPass begin_offscreen_pass(Device& dev, OffscreenTarget& t, const OffscreenPassDesc& d) {
    RenderPass rp; rp.dev = &dev;
    rp.enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);

    // All attachments are always provided (the bound pipeline declares them all);
    // ColorOp.enabled/clear drives loadOp + the clear value. is_uint selects the
    // integer clear (R32UI id buffers). storeOp=Store preserves unwritten ones.
    WGPURenderPassColorAttachment att[kMaxColorAttachments] = {};
    for (uint32_t i = 0; i < t.color_count; ++i) {
        const ColorOp& c = d.color[i];
        att[i].view = t.color_view[i];
        att[i].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        att[i].loadOp  = c.clear ? WGPULoadOp_Clear : WGPULoadOp_Load;
        att[i].storeOp = WGPUStoreOp_Store;
        if (c.is_uint) att[i].clearValue = WGPUColor{ (double)c.u[0], (double)c.u[1], (double)c.u[2], (double)c.u[3] };
        else           att[i].clearValue = WGPUColor{ c.f[0], c.f[1], c.f[2], c.f[3] };
    }
    WGPURenderPassDepthStencilAttachment z = {};
    z.view = t.depth_view;
    z.depthLoadOp  = d.clear_depth ? WGPULoadOp_Clear : WGPULoadOp_Load;
    z.depthStoreOp = WGPUStoreOp_Store;
    z.depthClearValue = 1.0f;

    WGPURenderPassDescriptor pd = {};
    pd.colorAttachmentCount = t.color_count; pd.colorAttachments = att;
    pd.depthStencilAttachment = &z;
    rp.pass = wgpuCommandEncoderBeginRenderPass(rp.enc, &pd);
    return rp;
}

// half (IEEE 754 binary16) -> float, for the 16F readback expansion below.
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else { // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf/nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

// Format readback descriptor: bytes-per-texel in the WGPU texture, and how to
// reconcile to the GL readback contract the callers expect.
static void texformat_readback(TexFormat f, uint32_t& wgpu_bpt, uint32_t& halfs,
                               uint32_t& out_bpt) {
    switch (f) {
        // GL reads RGB16F as GL_RGB/GL_FLOAT (3 floats); WGPU stores RGBA16Float (4
        // halfs). Expand the first 3 halfs -> 3 floats.
        case TexFormat::RGB16F: wgpu_bpt = 8; halfs = 3; out_bpt = 12; break;
        // GL reads RG16F as GL_RG/GL_FLOAT (2 floats); WGPU stores 2 halfs.
        case TexFormat::RG16F:  wgpu_bpt = 4; halfs = 2; out_bpt = 8;  break;
        case TexFormat::R32F:   wgpu_bpt = 4; halfs = 0; out_bpt = 4;  break;
        case TexFormat::R32UI:  wgpu_bpt = 4; halfs = 0; out_bpt = 4;  break;
        default:                wgpu_bpt = 4; halfs = 0; out_bpt = 4;  break;
    }
}

void read_target_region(Device& dev, OffscreenTarget& t, uint32_t attachment,
                        int x, int y, int w, int h, void* out) {
    // Target not built yet (queried before the first offscreen render), or the region
    // straddles the edge on the frame after a resize: leave `out` as the caller set it
    // (the on-model depth read pre-inits to the 1000 "no geometry" clear) rather than
    // issuing an out-of-bounds copyTextureToBuffer, which wgpu-native aborts on.
    if (attachment >= t.color_count || !t.color_tex[attachment] ||
        x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > t.width || y + h > t.height)
        return;

    uint32_t wgpu_bpt, halfs, out_bpt;
    texformat_readback(t.color_fmt[attachment], wgpu_bpt, halfs, out_bpt);

    // copyTextureToBuffer requires bytesPerRow a multiple of 256.
    uint32_t row_bytes = (uint32_t)w * wgpu_bpt;
    uint32_t padded_row = (row_bytes + 255u) & ~255u;
    uint64_t staging_size = (uint64_t)padded_row * h;
    if (t.readback_size < staging_size) {
        if (t.readback) wgpuBufferRelease(t.readback);
        WGPUBufferDescriptor bd = {};
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        bd.size = staging_size;
        t.readback = wgpuDeviceCreateBuffer(dev.device, &bd);
        t.readback_size = staging_size;
    }

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(dev.device, nullptr);
    WGPUTexelCopyTextureInfo src = {};
    src.texture = t.color_tex[attachment]; src.mipLevel = 0;
    src.origin = { (uint32_t)x, (uint32_t)y, 0 };  // gpu.h: y is top-left (WGPU origin too)
    src.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferInfo dst = {};
    dst.buffer = t.readback;
    dst.layout.offset = 0; dst.layout.bytesPerRow = padded_row; dst.layout.rowsPerImage = h;
    WGPUExtent3D ext = { (uint32_t)w, (uint32_t)h, 1 };
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ext);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(dev.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    // map the (padded) staging, then compact rows into `out` (out has no row padding,
    // and 16F formats expand half->float to match GL's float readback).
    MapState ms;
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode = WGPUCallbackMode_AllowProcessEvents;
    mcb.callback = onMap; mcb.userdata1 = &ms;
    wgpuBufferMapAsync(t.readback, WGPUMapMode_Read, 0, staging_size, mcb);
    while (!ms.done) wgpuDevicePoll(dev.device, true, nullptr);
    if (ms.status != WGPUMapAsyncStatus_Success) {
        std::memset(out, 0, (size_t)out_bpt * w * h);
        wgpuBufferUnmap(t.readback);
        return;
    }
    const uint8_t* mapped = (const uint8_t*)wgpuBufferGetMappedRange(t.readback, 0, staging_size);
    uint8_t* o = (uint8_t*)out;
    for (int row = 0; row < h; ++row) {
        const uint8_t* srow = mapped + (size_t)row * padded_row;
        if (halfs == 0) {
            std::memcpy(o + (size_t)row * w * out_bpt, srow, (size_t)w * out_bpt);
        } else {
            float* of = (float*)(o + (size_t)row * w * out_bpt);
            for (int px = 0; px < w; ++px) {
                const uint16_t* sh = (const uint16_t*)(srow + (size_t)px * wgpu_bpt);
                for (uint32_t c = 0; c < halfs; ++c) of[px * halfs + c] = half_to_float(sh[c]);
            }
        }
    }
    wgpuBufferUnmap(t.readback);
}

} // namespace gpu
