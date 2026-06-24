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

} // namespace gpu
