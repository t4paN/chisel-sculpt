// include/gpu/gpu.h
// The Chisel GPU seam (RHI). A thin backend-agnostic layer sized to *exactly* what
// Chisel does — not a general abstraction (see webgpu-port-plan.md §"The seam").
// The backend is chosen at BUILD time (CMake CHISEL_GPU_BACKEND → one of the
// CHISEL_BACKEND_* macros), so there is no runtime polymorphism: each gpu:: type
// holds raw backend handles as members (guarded by the backend macro) and its
// methods are compiled from the matching src/gpu/<backend>_backend.cpp.
//
// Seam Step 1 covers the COMPUTE path only (buffers, compute pipelines, bind
// groups, a compute batch, readback) — enough to route the probe's mask + draw
// brushes through the seam. Render-side primitives land with Stage 3's render port.
#pragma once
#include <cstdint>
#include <cstddef>

#if defined(CHISEL_BACKEND_WEBGPU)
#include <webgpu/webgpu.h>
#endif
// GL backend stores handles as plain unsigned int (== GLuint) so this header does
// not need to pull in glad — only src/gpu/gl_backend.cpp includes the GL loader.

namespace gpu {

// Buffer usage bit-flags, mapped to the backend's native flags by create_buffer.
// CopyDst is always added by the backend (every buffer can be written), matching
// the probe's makeBuf convention.
// (no "None" member — X11's glfw3native.h #defines None, which would clobber the token)
enum class Usage : uint32_t {
    Vertex  = 1u << 0,
    Index   = 1u << 1,
    Storage = 1u << 2,
    Uniform = 1u << 3,
    CopySrc = 1u << 4,
    MapRead = 1u << 5,
};
inline Usage  operator|(Usage a, Usage b) { return (Usage)((uint32_t)a | (uint32_t)b); }
inline bool   has(Usage set, Usage bit)   { return ((uint32_t)set & (uint32_t)bit) != 0; }

// How a buffer is seen by a compute bind-group slot. Mirrors the std430 access in
// the WGSL: readonly storage, read_write storage, or a uniform block.
enum class Bind : uint32_t { StorageRead, StorageReadWrite, Uniform };

// ---- Device: wraps an already-created backend device + queue. Surface/adapter/
// device creation stays in the windowing code (Stage 2); the seam owns resources.
// (GL has no device object — gl_device() returns an empty handle; a current GL
// context must already exist.) ----
struct Device {
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUDevice device = nullptr;
    WGPUQueue  queue  = nullptr;
#endif
};
#if defined(CHISEL_BACKEND_WEBGPU)
Device device_from_webgpu(WGPUDevice, WGPUQueue);
#elif defined(CHISEL_BACKEND_GL)
Device gl_device();
#endif

struct Buffer {
    uint64_t size = 0;
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUBuffer handle = nullptr;   // exposed so the still-raw render path can bind it
#elif defined(CHISEL_BACKEND_GL)
    unsigned int handle = 0;       // GLuint
#endif
};

// Create a buffer of `size` bytes with `usage` (+ CopyDst). If `data` is non-null it
// is uploaded immediately. release_buffer frees the backend handle.
Buffer create_buffer(Device&, const void* data, uint64_t size, Usage usage);
void   write_buffer(Device&, Buffer&, uint64_t offset, const void* data, uint64_t size);
void   release_buffer(Buffer&);

// One compute bind-group slot's declaration (layout) — binding number mirrors the
// ComputeBinding enum; min_size guards the bound range.
struct BindEntry { uint32_t binding; Bind type; uint64_t min_size; };

// Per-backend shader sources for one kernel. The WebGPU backend compiles `wgsl`,
// the GL backend compiles `glsl` (#version 430 compute). A dual-backend kernel
// supplies both; each backend ignores the other. (The two must agree on the same
// ComputeBinding bindings + the std140 Params UBO at binding 63.)
struct ShaderSources {
    const char* wgsl = nullptr;
    const char* glsl = nullptr;
};

static const uint32_t kMaxBindings = 16;

struct ComputePipeline {
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUComputePipeline handle = nullptr;
    WGPUBindGroupLayout  bgl    = nullptr;
    WGPUPipelineLayout   pl     = nullptr;
    WGPUShaderModule     module = nullptr;
#elif defined(CHISEL_BACKEND_GL)
    unsigned int handle = 0;                 // GL program (0 = failed)
    uint32_t  binding_count = 0;             // bind-layout cache (GL has no BGL object):
    uint32_t  binding_id[kMaxBindings] = {}; //   binding number ...
    Bind      binding_type[kMaxBindings] = {};//   ... and its access type
#endif
};
// Compile a compute pipeline from the backend's shader source + its bind-group
// layout. Returns a pipeline with handle==null/0 on failure (caller checks).
ComputePipeline create_compute_pipeline(Device&, const ShaderSources&,
                                        const BindEntry* entries, uint32_t entry_count,
                                        const char* entry_point = "main");
void release_compute_pipeline(ComputePipeline&);

// One filled bind-group slot: the buffer bound at `binding` for `size` bytes.
struct BindBufferEntry { uint32_t binding; const Buffer* buffer; uint64_t size; };

struct BindGroup {
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUBindGroup handle = nullptr;
#elif defined(CHISEL_BACKEND_GL)
    // GL has no bind-group object: remember each binding's GL target + buffer and
    // re-bind at dispatch. Targets resolved from the pipeline layout at create time.
    uint32_t     count = 0;
    unsigned int target[kMaxBindings] = {};  // GL_SHADER_STORAGE_BUFFER / GL_UNIFORM_BUFFER
    uint32_t     binding[kMaxBindings] = {};
    unsigned int buffer[kMaxBindings]  = {};
#endif
};
BindGroup create_bind_group(Device&, ComputePipeline&,
                            const BindBufferEntry* entries, uint32_t entry_count);
void release_bind_group(BindGroup&);

// ---- Compute batch: record dispatches into one compute pass, optionally end the
// pass and append buffer copies, then submit. Maps 1:1 to a WebGPU encoder; a GL
// backend implements begin/end-pass as no-ops and dispatch/copy immediately. ----
struct ComputeBatch {
    Device* dev = nullptr;
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUCommandEncoder     enc  = nullptr;
    WGPUComputePassEncoder pass = nullptr;
#endif
    // GL needs no encoder — dispatch/copy execute immediately (with glMemoryBarrier
    // between dependent steps); begin/end-pass/submit are barrier bookkeeping.
};
ComputeBatch begin_compute(Device&);
// groups_y defaults to 1 (the common 1D case). A 2D group grid is only needed when a
// 1D count would exceed the backend's 65535 per-dimension limit (the SDF passes at
// R>=128); the kernel recovers the linear index from num_workgroups.x itself.
void dispatch(ComputeBatch&, ComputePipeline&, BindGroup&, uint32_t groups_x, uint32_t groups_y = 1);
void end_compute_pass(ComputeBatch&);                       // call before copy_buffer
void copy_buffer(ComputeBatch&, const Buffer& src, uint64_t src_off,
                 const Buffer& dst, uint64_t dst_off, uint64_t size);
void submit(ComputeBatch&);                                  // finish + submit + release

// Synchronous read of a MapRead staging buffer that has already been populated
// (e.g. via copy_buffer in a submitted batch). Busy-waits the backend until the map
// resolves, copies `size` bytes into `out`. Legal only at the discrete readback
// points (pen-down/up/one-shot) the architecture restricts us to — never mid-stroke.
void map_read(Device&, const Buffer& staging, uint64_t size, void* out);

// ========================== Render-side seam ==============================
// Added for the render-path port (renderer.cpp off raw GL). GL backend lands
// first; the WebGPU render backend lands with the Emscripten web target (the
// app isn't compiled under CHISEL_BACKEND_WEBGPU yet, so only the GL side is
// implemented this stage — nothing references these under webgpu).
//
// KEY SIMPLIFIER: none of Chisel's render shaders SAMPLE a texture — matcap, the
// background gradient and the cursor are procedural; the picking MRT textures are
// written then read back to CPU, never sampled. So render bind groups stay
// UBO-only and reuse the compute BindGroup / create_bind_group verbatim. Loose GL
// uniforms become a std140 UBO (the same move the compute kernels made), so both
// backends share one upload struct. (Sampled textures would need a seam extension;
// they don't exist in this renderer, so we don't build one.)

static const uint32_t kMaxVertexAttrs = 8;

// How a vertex attribute's bytes are interpreted. Maps to a GL type+size+normalize
// triple and to a WGPUVertexFormat.
enum class VertexFormat : uint32_t {
    F32, F32x2, F32x3, F32x4, // float[1..4]
    U8x4_norm,                // RGBA8 -> [0,1] (packed vertex colour)
};
enum class Topology : uint32_t { Triangles, TriangleStrip, Lines, Points };

// One vertex attribute: shader input `location`, read from vertex-buffer `slot`
// at byte `offset`, decoded as `format`. The per-vertex stride lives on the slot.
struct VertexAttr { uint32_t location; VertexFormat format; uint32_t slot; uint32_t offset; };
struct VertexSlot { uint32_t stride; };  // bytes between consecutive vertices

// Shader sources for a raster pipeline. GL compiles vert+frag GLSL; WebGPU one
// WGSL module (entry points vs_main/fs_main). Mirrors ShaderSources for compute.
struct RenderShaderSources {
    const char* wgsl      = nullptr;  // WebGPU: one module (vs_main/fs_main)
    const char* vert_glsl = nullptr;  // GL: #version 330 vertex stage
    const char* frag_glsl = nullptr;  // GL: #version 330 fragment stage
};

struct RenderPipelineDesc {
    RenderShaderSources shaders;
    const VertexAttr* attrs = nullptr; uint32_t attr_count = 0;
    const VertexSlot* slots = nullptr; uint32_t slot_count = 0;
    const BindEntry*  binds = nullptr; uint32_t bind_count = 0; // UBO layout (e.g. binding 63)
    Topology topology    = Topology::Triangles;
    bool     depth_test  = false;
    bool     depth_write = false;
    bool     blend       = false;  // straight alpha: src_alpha / one_minus_src_alpha
};

struct RenderPipeline {
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPURenderPipeline  handle = nullptr;
    WGPUBindGroupLayout bgl    = nullptr;
    WGPUPipelineLayout  pl     = nullptr;
    WGPUShaderModule    module = nullptr;
#elif defined(CHISEL_BACKEND_GL)
    unsigned int handle = 0;                 // GL program (0 = failed)
    unsigned int vao    = 0;                 // GL needs a VAO to draw; the pipeline owns one
    // vertex layout cache (replayed onto `vao` at draw time from the bound buffers)
    uint32_t   attr_count = 0;
    VertexAttr attrs[kMaxVertexAttrs] = {};
    uint32_t   slot_stride[kMaxVertexAttrs] = {};
    // UBO bind layout (GL has no BGL object — resolve target at bind, like compute)
    uint32_t binding_count = 0;
    uint32_t binding_id[kMaxBindings]   = {};
    Bind     binding_type[kMaxBindings] = {};
    // pipeline render state, applied at set_pipeline
    uint8_t topology = 0, depth_test = 0, depth_write = 0, blend = 0;
#endif
};
RenderPipeline create_render_pipeline(Device&, const RenderPipelineDesc&);
void release_render_pipeline(RenderPipeline&);

// UBO bind group for a render pipeline (same BindGroup type as compute; resolves
// each slot's GL target from the render pipeline's layout). Render groups are
// UBO-only — see the texture note above.
BindGroup create_bind_group(Device&, RenderPipeline&, const BindBufferEntry* entries, uint32_t entry_count);

// A render target: the default framebuffer (swapchain) for now; offscreen MRT
// (the picking FBO) grows this struct in the picking slice. `clear` clears colour
// to `clear_color` at begin; otherwise the existing contents load.
struct RenderTarget {
    int   width = 0, height = 0;
    bool  clear = false;
    float clear_color[4] = {0, 0, 0, 1};
#if defined(CHISEL_BACKEND_GL)
    unsigned int fbo = 0;                 // 0 = default framebuffer
#elif defined(CHISEL_BACKEND_WEBGPU)
    WGPUTextureView color = nullptr;      // surface view, acquired per frame
#endif
};

// One render pass = bind target + viewport, record draws, end. Maps 1:1 to a
// WebGPU render pass encoder; the GL backend binds the FBO/viewport at begin and
// the vertex/index buffers are recorded for replay at draw.
struct RenderPass {
    Device* dev = nullptr;
#if defined(CHISEL_BACKEND_WEBGPU)
    WGPUCommandEncoder      enc  = nullptr;
    WGPURenderPassEncoder   pass = nullptr;
#elif defined(CHISEL_BACKEND_GL)
    unsigned int vbuf[kMaxVertexAttrs] = {}; // vertex buffer per slot (set_vertex_buffer)
    unsigned int ibuf = 0;                   // index buffer (set_index_buffer)
    RenderPipeline* pipe = nullptr;          // current pipeline (set_pipeline)
#endif
};
RenderPass begin_render_pass(Device&, const RenderTarget&);
void set_pipeline(RenderPass&, RenderPipeline&);
void set_bind_group(RenderPass&, RenderPipeline&, BindGroup&);          // UBO group
void set_vertex_buffer(RenderPass&, uint32_t slot, const Buffer&);
void set_index_buffer(RenderPass&, const Buffer&);                       // 32-bit indices
void draw(RenderPass&, uint32_t vertex_count, uint32_t first_vertex = 0);
void draw_indexed(RenderPass&, uint32_t index_count);
void end_render_pass(RenderPass&);

} // namespace gpu
