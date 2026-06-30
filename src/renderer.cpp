#include "renderer.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// Matcap Params UBO payload — byte-identical to the std140 `Params` block in the
// matcap shaders (two mat4 = 128 B, then three floats + 4 B pad = 16 B).
struct MatcapParamsGPU {
    float view[16];
    float proj[16];
    float facing_threshold;
    float obj_mask;
    float paint_visible;
    float _pad0;
};
static_assert(sizeof(MatcapParamsGPU) == 144, "matcap Params UBO must be 144 bytes (std140)");

// ---- Brush-cursor overlay Params UBOs (std140 at binding 63) ----
// The cursor ring, the footprint shadow disc and the centre crosshair each ride a
// std140 Params UBO on the gpu:: seam, exactly like matcap. Byte-identical to the
// `Params` block declared in the matching shaders below; pads make the std140
// offsets explicit. The overlay geometry itself is camera-independent (unit ring /
// unit disc / fixed crosshair), so it is built ONCE at init and only these UBOs
// change per frame.
struct CursorParamsGPU {        // ring
    float center[2];            // 0
    float screenSize[2];        // 8   (fills 0..15)
    float normal[3];   float radius;      // 16 (vec3 + scalar packed)
    float camRight[3]; float baseThick;   // 32
    float camUp[3];    float frontBoost;  // 48
    float camFwd[3];   float _pad0;       // 64
    float color[4];                       // 80
};
static_assert(sizeof(CursorParamsGPU) == 96, "cursor Params UBO must be 96 bytes (std140)");

struct ShadowParamsGPU {        // footprint disc
    float center[2];            // 0
    float radius;               // 8
    float _pad0;                // 12
    float screenSize[2];        // 16
    float _pad1[2];             // 24
    float color[4];             // 32
};
static_assert(sizeof(ShadowParamsGPU) == 48, "shadow Params UBO must be 48 bytes (std140)");

struct CrosshairParamsGPU {     // centre X
    float center[2];            // 0
    float screenSize[2];        // 8   (fills 0..15)
    float color[4];             // 16
};
static_assert(sizeof(CrosshairParamsGPU) == 32, "crosshair Params UBO must be 32 bytes (std140)");

// Debug-mesh edge overlay (wireframe) Params UBO — two mat4 (std140 at binding 63).
struct DebugParamsGPU {
    float view[16];
    float proj[16];
};
static_assert(sizeof(DebugParamsGPU) == 128, "debug Params UBO must be 128 bytes (std140)");

// Entity-id pick pass Params UBO — view/proj (constant across the pass) + the
// per-draw entity id (std140 at binding 63).
struct PickParamsGPU {
    float    view[16];     // 0
    float    proj[16];     // 64
    uint32_t entity_id;    // 128
    uint32_t _pad[3];      // 132 (round to 144)
};
static_assert(sizeof(PickParamsGPU) == 144, "pick Params UBO must be 144 bytes (std140)");

// Screen-buffer MRT pass Params UBO — just view/proj (std140 at binding 63).
struct ScreenParamsGPU {
    float view[16];
    float proj[16];
};
static_assert(sizeof(ScreenParamsGPU) == 128, "screen Params UBO must be 128 bytes (std140)");

// Indexed→flat expand compute Params UBO — triangle count (std140 at binding 63).
struct ScreenExpandParamsGPU {
    uint32_t tri_count;
    uint32_t _pad[3];
};
static_assert(sizeof(ScreenExpandParamsGPU) == 16, "expand Params UBO must be 16 bytes (std140)");

// Unit-ring segment count (shared by ring + footprint disc geometry, built once).
static const int CURSOR_SEGS = 64;

// Create-or-grow an owned seam buffer to exactly `size` bytes and fill it with
// `data` (buffer-ownership migration). Same-size re-uploads keep the handle and
// just write_buffer; a size change releases + recreates (the WebGPU-correct path —
// no in-place resize). Callers that cache the handle must re-fetch after this.
static void ensure_buffer(gpu::Device& dev, gpu::Buffer& b,
                          const void* data, uint64_t size, gpu::Usage usage) {
    if (b.handle && b.size == size) gpu::write_buffer(dev, b, 0, data, size);
    else { gpu::release_buffer(b); b = gpu::create_buffer(dev, data, size, usage); }
}

// ---- Shader sources ----

static const char* bg_vert_src = R"(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.999, 1.0);
}
)";

static const char* bg_frag_src = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
void main() {
    // Dark gray gradient: darker at bottom, slightly lighter at top
    float t = vUV.y;
    vec3 bot = vec3(0.18, 0.18, 0.20);
    vec3 top = vec3(0.28, 0.28, 0.30);
    fragColor = vec4(mix(bot, top, t), 1.0);
}
)";

// Matcap Params UBO (std140 at binding 63 — the same convention the compute
// kernels adopted). Declared identically in both stages so the std140 offsets
// agree; each stage uses only the members it needs. Mirrors MatcapParamsGPU.
static const char* matcap_vert_src = R"(
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in float aMask;
layout(location=4) in vec4 aColor;   // RGBA8 normalized -> [0,1]; white = unpainted

layout(std140, binding=63) uniform Params {
    mat4 uView;
    mat4 uProj;
    float uFacingThreshold;
    float uObjMask;
    float uPaintVisible;
};

out vec3 vNormView;
out float vMask;
out vec4 vColor;

void main() {
    mat3 normalMat = mat3(uView);
    vNormView = normalize(normalMat * aNorm);
    vMask = aMask;
    vColor = aColor;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)";

static const char* matcap_frag_src = R"(
#version 430 core
in vec3 vNormView;
in float vMask;
in vec4 vColor;
out vec4 fragColor;

layout(std140, binding=63) uniform Params {
    mat4 uView;
    mat4 uProj;
    float uFacingThreshold;
    float uObjMask;
    float uPaintVisible;
};

void main() {
    vec3 n = normalize(vNormView);
    float rim = 1.0 - abs(n.z);
    float top = n.y * 0.5 + 0.5;
    float base = 0.35 + 0.45 * top + 0.15 * (1.0 - rim * rim);

    float cavity = 1.0 - rim * rim * 0.3;
    float val = base * cavity;

    // Sculpt mask: unmasked (0) = normal, masked (1) = dark
    val *= mix(1.0, 0.4, vMask);

    // Lit albedo: white (default) == matcap grey exactly; paint tints it.
    // uPaintVisible folds the albedo out to white when paint is hidden.
    vec3 col = vec3(val) * mix(vec3(1.0), vColor.rgb, uPaintVisible);

    // Object selection mask: deselected objects get dark muted tint
    if (uObjMask > 0.0) {
        vec3 tint = vec3(0.35, 0.12, 0.18);
        col = mix(col, col * tint, uObjMask);
    }

    float facing = max(n.z, 0.0);
    float edge = abs(facing - uFacingThreshold);
    float px_width = fwidth(facing);
    if (uFacingThreshold > 0.001 && edge < px_width * 0.5) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        fragColor = vec4(col, 1.0);
    }
}
)";

// Entity-id pick shader: writes linear depth (attachment 0) and a per-draw
// entity id (attachment 2, the R32UI buffer otherwise used for triangle ids).
// Reads only position (location 0), so it runs on both the working VAO and the
// inactive display VAOs (which share the pos/norm/mask/mesh_id attribute layout).
// std140 Params block (binding 63) — byte-identical to PickParamsGPU.
#define PICK_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    mat4 uView;\n" \
    "    mat4 uProj;\n" \
    "    uint uEntityId;\n" \
    "};\n"

static const char* pick_vert_src =
"#version 430 core\n"
"layout(location=0) in vec3 aPos;\n"
PICK_PARAMS_BLOCK
R"(
out float vDepth;
void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    vDepth = -viewPos.z;  // linear depth, positive into screen
    gl_Position = uProj * viewPos;
}
)";

static const char* pick_frag_src =
"#version 430 core\n"
PICK_PARAMS_BLOCK
R"(
in float vDepth;
layout(location=0) out float outDepth;
layout(location=2) out uint  outId;
void main() {
    outDepth = vDepth;
    outId = uEntityId;
}
)";

// std140 Params block (binding 63) — byte-identical to CursorParamsGPU. Declared
// in both stages so the offsets agree; each stage uses only the members it needs.
#define CURSOR_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    vec2 uCenter;\n" \
    "    vec2 uScreenSize;\n" \
    "    vec3 uNormal;   float uRadius;\n" \
    "    vec3 uCamRight; float uBaseThick;\n" \
    "    vec3 uCamUp;    float uFrontBoost;\n" \
    "    vec3 uCamFwd;\n" \
    "    vec4 uColor;\n" \
    "};\n"

static const char* cursor_vert_src =
"#version 430 core\n"
"layout(location=0) in vec2 aLocal;\n"   // unit circle (cos, sin)
"layout(location=1) in float aOuter;\n"  // 0 = inner rim vertex, 1 = outer
CURSOR_PARAMS_BLOCK
R"(
out float vFront;

void main() {
    vec3 N = normalize(uNormal);

    // Single tangent-plane basis (U, V) ⊥ N. Used for BOTH the world-space
    // rim (shading) and the screen-space ellipse (shape), so rim_world and
    // ellipse_rim always describe the same point on the disc.
    vec3 ref = (abs(dot(uCamRight, N)) < 0.95) ? uCamRight : uCamUp;
    vec3 U = normalize(ref - dot(ref, N) * N);
    vec3 V = cross(N, U);

    // Project the tangent basis into screen pixel space. Screen Y grows down,
    // so flip the up component.
    vec2 u_screen = vec2(dot(U, uCamRight), -dot(U, uCamUp));
    vec2 v_screen = vec2(dot(V, uCamRight), -dot(V, uCamUp));

    // Visibility clamp: at grazing angles one of the projected basis vectors
    // collapses toward zero, making the ring vanish. Lift the shorter one to
    // a minimum length, perpendicular to the longer one, so the ring keeps a
    // visible sliver at the silhouette without distorting face-on geometry.
    const float MIN_VIS = 0.18;
    float u_len = length(u_screen);
    float v_len = length(v_screen);
    if (u_len < MIN_VIS) {
        if (u_len > 1e-5) {
            u_screen *= (MIN_VIS / u_len);
        } else {
            vec2 vn = normalize(v_screen);
            u_screen = MIN_VIS * vec2(-vn.y, vn.x);
        }
    }
    if (v_len < MIN_VIS) {
        if (v_len > 1e-5) {
            v_screen *= (MIN_VIS / v_len);
        } else {
            vec2 un = normalize(u_screen);
            v_screen = MIN_VIS * vec2(-un.y, un.x);
        }
    }

    // Rim point on the projected ellipse (unit-radius before uRadius scale).
    vec2 ellipse_rim = u_screen * aLocal.x + v_screen * aLocal.y;

    // Outward direction = perpendicular to the parametric tangent,
    // flipped to point away from the ring center.
    vec2 dpos_dt = -u_screen * aLocal.y + v_screen * aLocal.x;
    vec2 outward = vec2(-dpos_dt.y, dpos_dt.x);
    if (dot(outward, ellipse_rim) < 0.0) outward = -outward;
    float out_len = length(outward);
    outward = (out_len > 1e-6) ? (outward / out_len) : vec2(1.0, 0.0);

    // Front/back shading: positive when the rim point sits closer to the
    // camera than the surface center. Same (U, V) parameterization as the
    // screen position, so shading aligns with the visible rim point.
    vec3 rim_world = U * aLocal.x + V * aLocal.y;
    vFront = -dot(rim_world, uCamFwd);

    // Thickness grows on front half, shrinks on back
    float tFront = clamp(vFront, 0.0, 1.0);
    float halfThick = (uBaseThick + uFrontBoost * tFront) * 0.5;

    vec2 screen = uCenter
                + ellipse_rim * uRadius
                + outward * (aOuter - 0.5) * 2.0 * halfThick;

    vec2 ndc = (screen / uScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* cursor_frag_src =
"#version 430 core\n"
CURSOR_PARAMS_BLOCK
R"(
in float vFront;
out vec4 fragColor;
void main() {
    // vFront roughly in [-1, 1]. Darken back, keep front near full.
    float t = clamp(vFront * 0.5 + 0.5, 0.0, 1.0);
    float bright = mix(0.78, 1.12, t);
    float a = uColor.a * mix(0.82, 1.0, t);
    fragColor = vec4(uColor.rgb * bright, a);
}
)";

// ---- Crosshair (center X marker) ----
// std140 Params block (binding 63) — byte-identical to CrosshairParamsGPU.
#define CROSSHAIR_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    vec2 uCenter;\n" \
    "    vec2 uScreenSize;\n" \
    "    vec4 uColor;\n" \
    "};\n"

static const char* crosshair_vert_src =
"#version 430 core\n"
"layout(location=0) in vec2 aPos;\n"
CROSSHAIR_PARAMS_BLOCK
R"(
void main() {
    vec2 screen = uCenter + aPos;
    vec2 ndc = (screen / uScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* crosshair_frag_src =
"#version 430 core\n"
CROSSHAIR_PARAMS_BLOCK
R"(
out vec4 fragColor;
void main() { fragColor = uColor; }
)";

// ---- Brush shadow (screen-space filled disc = actual sculpt footprint) ----
// std140 Params block (binding 63) — byte-identical to ShadowParamsGPU.
#define SHADOW_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    vec2 uCenter;\n" \
    "    float uRadius;\n" \
    "    vec2 uScreenSize;\n" \
    "    vec4 uColor;\n" \
    "};\n"

static const char* cursor_shadow_vert_src =
"#version 430 core\n"
"layout(location=0) in vec2 aPos;\n"   // unit-disc triangle-list vertex (centre or rim)
SHADOW_PARAMS_BLOCK
R"(
out float vDist;
void main() {
    vDist = length(aPos);  // 0 = center, 1 = rim
    vec2 screen = uCenter + aPos * uRadius;
    vec2 ndc = (screen / uScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* cursor_shadow_frag_src =
"#version 430 core\n"
SHADOW_PARAMS_BLOCK
R"(
in float vDist;
out vec4 fragColor;
void main() {
    // Soft falloff near the rim; slight dip in the center so the disc reads
    // as a halo rather than a solid puck.
    float edge = 1.0 - smoothstep(0.72, 1.0, vDist);
    float core = 0.55 + 0.45 * smoothstep(0.0, 0.55, vDist);
    float a = uColor.a * edge * core;
    fragColor = vec4(uColor.rgb, a);
}
)";

// ---- Screen buffer MRT shaders (for brush pipeline) ----

// std140 Params block (binding 63) — byte-identical to ScreenParamsGPU.
#define SCREEN_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    mat4 uView;\n" \
    "    mat4 uProj;\n" \
    "};\n"

static const char* screen_buf_vert_src =
"#version 430 core\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNorm;\n"
"layout(location=2) in float aTriID;\n"
"layout(location=3) in vec2 aBary;\n"
SCREEN_PARAMS_BLOCK
R"(
out vec3 vNormWorld;
out float vDepth;
flat out uint vTriID;
out vec2 vBary;

void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    // World-space normal written to the normal attachment — brush back-projection
    // (apply_draw in brush.cpp) treats this as a world-space direction to displace
    // vertices along. Do NOT transform by uView here: view-space normals would
    // make every visible surface read as ~(0,0,1), producing direction-locked
    // "cannon" extrusions along world +z regardless of camera angle.
    vNormWorld = normalize(aNorm);
    vDepth = -viewPos.z;  // linear depth (positive into screen)
    vTriID = uint(aTriID);
    vBary = aBary;
    gl_Position = uProj * viewPos;
}
)";

static const char* screen_buf_frag_src = R"(
#version 430 core
in vec3 vNormWorld;
in float vDepth;
flat in uint vTriID;
in vec2 vBary;

layout(location=0) out float outDepth;
layout(location=1) out vec4 outNormal;
layout(location=2) out uint outTriID;
layout(location=3) out vec2 outBary;

void main() {
    outDepth = vDepth;
    outNormal = vec4(normalize(vNormWorld), 1.0);
    outTriID = vTriID;
    outBary = vBary;
}
)";

static const char* screen_expand_src = R"(
#version 430
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer PosIn   { float in_pos[]; };
layout(std430, binding = 1) readonly buffer NormIn  { float in_norm[]; };
layout(std430, binding = 2) readonly buffer IdxIn   { uint  in_idx[]; };
layout(std430, binding = 3) writeonly buffer PosOut  { float out_pos[]; };
layout(std430, binding = 4) writeonly buffer NormOut { float out_norm[]; };

layout(std140, binding = 63) uniform Params { uint tri_count; };

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= tri_count) return;

    uint i0 = in_idx[t*3+0];
    uint i1 = in_idx[t*3+1];
    uint i2 = in_idx[t*3+2];
    uint base = t * 9;

    out_pos[base+0] = in_pos[i0*3+0]; out_pos[base+1] = in_pos[i0*3+1]; out_pos[base+2] = in_pos[i0*3+2];
    out_pos[base+3] = in_pos[i1*3+0]; out_pos[base+4] = in_pos[i1*3+1]; out_pos[base+5] = in_pos[i1*3+2];
    out_pos[base+6] = in_pos[i2*3+0]; out_pos[base+7] = in_pos[i2*3+1]; out_pos[base+8] = in_pos[i2*3+2];

    out_norm[base+0] = in_norm[i0*3+0]; out_norm[base+1] = in_norm[i0*3+1]; out_norm[base+2] = in_norm[i0*3+2];
    out_norm[base+3] = in_norm[i1*3+0]; out_norm[base+4] = in_norm[i1*3+1]; out_norm[base+5] = in_norm[i1*3+2];
    out_norm[base+6] = in_norm[i2*3+0]; out_norm[base+7] = in_norm[i2*3+1]; out_norm[base+8] = in_norm[i2*3+2];
}
)";

// std140 Params block (binding 63) — byte-identical to DebugParamsGPU (two mat4).
#define DEBUG_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    mat4 uView;\n" \
    "    mat4 uProj;\n" \
    "};\n"

static const char* debug_vert_src =
"#version 430 core\n"
"layout(location=0) in vec3 aPos;\n"
DEBUG_PARAMS_BLOCK
R"(
void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)";

static const char* debug_edge_frag_src = R"(
#version 430 core
out vec4 fragColor;
void main() {
    fragColor = vec4(0.15, 0.45, 0.15, 1.0);
}
)";

// ---- Renderer ----

Renderer::Renderer()
    : debug_edge_count(0)
    , screen_tri_count(0)
    , initialized(false)
{}

Renderer::~Renderer() {
    gpu::release_buffer(vbo_pos);
    gpu::release_buffer(vbo_norm);
    gpu::release_buffer(vbo_mask);
    gpu::release_buffer(vbo_color);
    gpu::release_buffer(ebo);
    gpu::release_render_pipeline(bg_pipeline);
    gpu::release_buffer(bg_vbuf);
    gpu::release_render_pipeline(matcap_pipeline);
    gpu::release_buffer(matcap_ubo);
    gpu::release_render_pipeline(cursor_pipeline);
    gpu::release_render_pipeline(shadow_pipeline);
    gpu::release_render_pipeline(crosshair_pipeline);
    gpu::release_buffer(cursor_vbuf);
    gpu::release_buffer(shadow_vbuf);
    gpu::release_buffer(crosshair_vbuf);
    gpu::release_buffer(cursor_ubo);
    gpu::release_buffer(shadow_ubo);
    gpu::release_buffer(crosshair_ubo);
    gpu::release_render_pipeline(debug_edge_pipeline);
    gpu::release_buffer(debug_edge_ubo);
    gpu::release_buffer(debug_edge_vbo);
    gpu::release_render_pipeline(pick_pipeline);
    gpu::release_buffer(pick_ubo);
    gpu::release_offscreen_target(screen_target);
    gpu::release_render_pipeline(screen_pipeline);
    gpu::release_buffer(screen_ubo);
    gpu::release_buffer(screen_vbo_pos);
    gpu::release_buffer(screen_vbo_norm);
    gpu::release_buffer(screen_vbo_triid);
    gpu::release_buffer(screen_vbo_bary);
    gpu::release_compute_pipeline(screen_expand_pipeline);
    gpu::release_buffer(screen_expand_ubo);
}

void Renderer::init() {
    gpu_dev = gpu::gl_device();

    // Background gradient quad — first render program on the gpu:: seam.
    {
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x2, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 2 * sizeof(float) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = bg_vert_src;
        d.shaders.frag_glsl = bg_frag_src;
        d.attrs = attrs; d.attr_count = 1;
        d.slots = slots; d.slot_count = 1;
        d.topology = gpu::Topology::TriangleStrip;
        // depth_test off (bg writes no depth), but depth_write=true so set_pipeline
        // leaves GL's depthMask at its default TRUE for the still-raw draw_mesh that
        // follows — those paths assume the default and never reset the mask.
        d.depth_test = false; d.depth_write = true;
        bg_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!bg_pipeline.handle) std::fprintf(stderr, "[renderer] bg pipeline failed\n");
        else std::printf("[renderer] bg pipeline compiled (gpu:: seam)\n");

        float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
        bg_vbuf = gpu::create_buffer(gpu_dev, quad, sizeof(quad), gpu::Usage::Vertex);
    }

    // Matcap shader — on the gpu:: seam. Four separate tightly-packed vertex
    // buffers (pos/norm/mask/color) at locations 0/1/2/4, a std140 Params UBO at
    // binding 63, depth-tested. Shared by the active mesh (draw_mesh) and the
    // inactive display entities (draw_display).
    {
        gpu::VertexAttr attrs[] = {
            { 0, gpu::VertexFormat::F32x3,     0, 0 },  // aPos    <- slot 0
            { 1, gpu::VertexFormat::F32x3,     1, 0 },  // aNorm   <- slot 1
            { 2, gpu::VertexFormat::F32,       2, 0 },  // aMask   <- slot 2
            { 4, gpu::VertexFormat::U8x4_norm, 3, 0 },  // aColor  <- slot 3
        };
        gpu::VertexSlot slots[] = {
            { 3 * sizeof(float) }, { 3 * sizeof(float) },
            { 1 * sizeof(float) }, { 4 },
        };
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(MatcapParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = matcap_vert_src;
        d.shaders.frag_glsl = matcap_frag_src;
        d.attrs = attrs; d.attr_count = 4;
        d.slots = slots; d.slot_count = 4;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Triangles;
        d.depth_test = true; d.depth_write = true;
        matcap_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!matcap_pipeline.handle) std::fprintf(stderr, "[renderer] matcap pipeline failed\n");
        else std::printf("[renderer] matcap pipeline compiled (gpu:: seam)\n");
        matcap_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(MatcapParamsGPU), gpu::Usage::Uniform);
    }

    // Entity-id pick pass — on the gpu:: seam. One pos attribute (slot 0) + a
    // view/proj/entity-id UBO; writes depth (att 0) + id (att 2) into the screen
    // offscreen target. Depth-tested, no blend.
    {
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x3, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 3 * sizeof(float) }};
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(PickParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = pick_vert_src;
        d.shaders.frag_glsl = pick_frag_src;
        d.attrs = attrs; d.attr_count = 1;
        d.slots = slots; d.slot_count = 1;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Triangles;
        d.depth_test = true; d.depth_write = true; d.blend = false;
        // Offscreen MRT signature: same 4 attachments as the screen target it draws
        // into (it only writes depth@0 + id@2, but WebGPU pipelines must match the
        // pass's full attachment set). GL ignores these.
        static const gpu::TexFormat pick_targets[] = {
            gpu::TexFormat::R32F, gpu::TexFormat::RGB16F,
            gpu::TexFormat::R32UI, gpu::TexFormat::RG16F };
        d.color_targets = pick_targets; d.color_target_count = 4;
        pick_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!pick_pipeline.handle) std::fprintf(stderr, "[renderer] pick pipeline failed\n");
        else std::printf("[renderer] pick pipeline compiled (gpu:: seam)\n");
        pick_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(PickParamsGPU), gpu::Usage::Uniform);
    }

    // Mesh buffers vbo_pos/vbo_norm/ebo (Step 1) and vbo_mask/vbo_color (Step 3a) are
    // all owned gpu::Buffers created lazily on first upload_mesh. The draw-time VAO
    // lives on each render pipeline (seam), so the renderer owns none.

    // Brush-cursor overlays — on the gpu:: seam. Three pipelines, each with static
    // camera-independent geometry (built once here) + a std140 Params UBO updated
    // per draw. All blend, no depth test; depth_write=true keeps GL's depthMask at
    // its default TRUE (same reasoning as bg_pipeline) — no depth is actually
    // written since depth_test is off.
    {
        // Ring — unit-circle triangle strip: SEGS+1 (inner,outer) vertex pairs.
        std::vector<float> ring; ring.reserve((CURSOR_SEGS + 1) * 2 * 3);
        for (int i = 0; i <= CURSOR_SEGS; i++) {
            float a = 2.0f * 3.14159265f * (float)i / (float)CURSOR_SEGS;
            float cs = std::cos(a), sn = std::sin(a);
            ring.push_back(cs); ring.push_back(sn); ring.push_back(0.0f); // inner rim
            ring.push_back(cs); ring.push_back(sn); ring.push_back(1.0f); // outer rim
        }
        cursor_vbuf = gpu::create_buffer(gpu_dev, ring.data(),
                                         ring.size() * sizeof(float), gpu::Usage::Vertex);
        gpu::VertexAttr attrs[] = {
            { 0, gpu::VertexFormat::F32x2, 0, 0 },              // aLocal (cos,sin)
            { 1, gpu::VertexFormat::F32,   0, 2 * sizeof(float) }, // aOuter
        };
        gpu::VertexSlot slots[] = {{ 3 * sizeof(float) }};
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(CursorParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = cursor_vert_src;
        d.shaders.frag_glsl = cursor_frag_src;
        d.attrs = attrs; d.attr_count = 2;
        d.slots = slots; d.slot_count = 1;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::TriangleStrip;
        d.depth_test = false; d.depth_write = true; d.blend = true;
        cursor_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!cursor_pipeline.handle) std::fprintf(stderr, "[renderer] cursor pipeline failed\n");
        else std::printf("[renderer] cursor pipeline compiled (gpu:: seam)\n");
        cursor_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(CursorParamsGPU), gpu::Usage::Uniform);
    }
    {
        // Footprint disc — unit triangle LIST (fan converted: WebGPU has no fan).
        // Per segment: (centre, rim[i], rim[i+1]).
        std::vector<float> disc; disc.reserve(CURSOR_SEGS * 3 * 2);
        for (int i = 0; i < CURSOR_SEGS; i++) {
            float a0 = 2.0f * 3.14159265f * (float)i / (float)CURSOR_SEGS;
            float a1 = 2.0f * 3.14159265f * (float)(i + 1) / (float)CURSOR_SEGS;
            disc.push_back(0.0f);        disc.push_back(0.0f);
            disc.push_back(std::cos(a0)); disc.push_back(std::sin(a0));
            disc.push_back(std::cos(a1)); disc.push_back(std::sin(a1));
        }
        shadow_vbuf = gpu::create_buffer(gpu_dev, disc.data(),
                                         disc.size() * sizeof(float), gpu::Usage::Vertex);
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x2, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 2 * sizeof(float) }};
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(ShadowParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = cursor_shadow_vert_src;
        d.shaders.frag_glsl = cursor_shadow_frag_src;
        d.attrs = attrs; d.attr_count = 1;
        d.slots = slots; d.slot_count = 1;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Triangles;
        d.depth_test = false; d.depth_write = true; d.blend = true;
        shadow_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!shadow_pipeline.handle) std::fprintf(stderr, "[renderer] shadow pipeline failed\n");
        else std::printf("[renderer] shadow pipeline compiled (gpu:: seam)\n");
        shadow_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ShadowParamsGPU), gpu::Usage::Uniform);
    }
    {
        // Crosshair — fixed centre X (8 line verts, screen-space offsets).
        const float gap = 3.0f, arm = 8.0f;
        float xh[] = {
            -arm, -arm,  -gap, -gap,
             gap,  gap,   arm,  arm,
             arm, -arm,   gap, -gap,
            -gap,  gap,  -arm,  arm,
        };
        crosshair_vbuf = gpu::create_buffer(gpu_dev, xh, sizeof(xh), gpu::Usage::Vertex);
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x2, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 2 * sizeof(float) }};
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(CrosshairParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = crosshair_vert_src;
        d.shaders.frag_glsl = crosshair_frag_src;
        d.attrs = attrs; d.attr_count = 1;
        d.slots = slots; d.slot_count = 1;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Lines;
        d.depth_test = false; d.depth_write = true; d.blend = true;
        crosshair_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!crosshair_pipeline.handle) std::fprintf(stderr, "[renderer] crosshair pipeline failed\n");
        else std::printf("[renderer] crosshair pipeline compiled (gpu:: seam)\n");
        crosshair_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(CrosshairParamsGPU), gpu::Usage::Uniform);
    }

    // Screen-buffer MRT pipeline — on the gpu:: seam. Four tightly-packed vertex
    // buffers (pos/norm/triid/bary at slots 0-3) + a view/proj UBO; writes all four
    // MRT attachments. The offscreen target itself is created lazily (window size).
    {
        gpu::VertexAttr attrs[] = {
            { 0, gpu::VertexFormat::F32x3, 0, 0 },  // aPos
            { 1, gpu::VertexFormat::F32x3, 1, 0 },  // aNorm
            { 2, gpu::VertexFormat::F32,   2, 0 },  // aTriID
            { 3, gpu::VertexFormat::F32x2, 3, 0 },  // aBary
        };
        gpu::VertexSlot slots[] = {
            { 3 * sizeof(float) }, { 3 * sizeof(float) },
            { 1 * sizeof(float) }, { 2 * sizeof(float) },
        };
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(ScreenParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = screen_buf_vert_src;
        d.shaders.frag_glsl = screen_buf_frag_src;
        d.attrs = attrs; d.attr_count = 4;
        d.slots = slots; d.slot_count = 4;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Triangles;
        d.depth_test = true; d.depth_write = true; d.blend = false;
        // Offscreen MRT signature: writes all four attachments of the screen target.
        static const gpu::TexFormat screen_targets[] = {
            gpu::TexFormat::R32F, gpu::TexFormat::RGB16F,
            gpu::TexFormat::R32UI, gpu::TexFormat::RG16F };
        d.color_targets = screen_targets; d.color_target_count = 4;
        screen_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!screen_pipeline.handle) std::fprintf(stderr, "[renderer] screen pipeline failed\n");
        else std::printf("[renderer] screen pipeline compiled (gpu:: seam)\n");
        screen_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ScreenParamsGPU), gpu::Usage::Uniform);
    }

    // Flat triangle-soup vertex buffers (seam-owned; pos/norm also SSBO targets for
    // the expand kernel). Allocated/filled lazily in upload_screen_mesh.

    // Indexed→flat expansion — on the gpu:: compute seam. 5 storage bindings
    // (0-4) + a tri_count Params UBO at binding 63.
    {
        gpu::BindEntry binds[] = {
            { 0, gpu::Bind::StorageRead,      0 },  // in_pos
            { 1, gpu::Bind::StorageRead,      0 },  // in_norm
            { 2, gpu::Bind::StorageRead,      0 },  // in_idx
            { 3, gpu::Bind::StorageReadWrite, 0 },  // out_pos
            { 4, gpu::Bind::StorageReadWrite, 0 },  // out_norm
            { 63, gpu::Bind::Uniform, sizeof(ScreenExpandParamsGPU) },
        };
        gpu::ShaderSources src; src.glsl = screen_expand_src;
        screen_expand_pipeline = gpu::create_compute_pipeline(gpu_dev, src, binds, 6);
        if (!screen_expand_pipeline.handle) std::fprintf(stderr, "[renderer] screen_expand pipeline failed\n");
        else std::printf("[compute] screen_expand pipeline compiled (gpu:: seam)\n");
        screen_expand_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ScreenExpandParamsGPU), gpu::Usage::Uniform);
    }

    // Debug wireframe overlay — on the gpu:: seam. Lines pipeline reading mesh
    // positions (slot 0) with a std140 view/proj UBO; the edge index buffer is
    // built lazily in draw_debug_mesh.
    {
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x3, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 3 * sizeof(float) }};
        gpu::BindEntry binds[] = {{ 63, gpu::Bind::Uniform, sizeof(DebugParamsGPU) }};
        gpu::RenderPipelineDesc d;
        d.shaders.vert_glsl = debug_vert_src;
        d.shaders.frag_glsl = debug_edge_frag_src;
        d.attrs = attrs; d.attr_count = 1;
        d.slots = slots; d.slot_count = 1;
        d.binds = binds; d.bind_count = 1;
        d.topology = gpu::Topology::Lines;
        d.depth_test = true; d.depth_write = true; d.blend = true;
        debug_edge_pipeline = gpu::create_render_pipeline(gpu_dev, d);
        if (!debug_edge_pipeline.handle) std::fprintf(stderr, "[renderer] debug edge pipeline failed\n");
        else std::printf("[renderer] debug edge pipeline compiled (gpu:: seam)\n");
        debug_edge_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(DebugParamsGPU), gpu::Usage::Uniform);
    }
    // debug_edge_vbo (edge index buffer) is created lazily in draw_debug_mesh.

    initialized = true;
}

void Renderer::upload_mesh(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();

    // Interleave SoA → AoS for VBO upload
    std::vector<float> pos(vc * 3), norm(vc * 3), mask_buf(vc);
    std::vector<uint32_t> col_buf(vc);
    bool has_color = !mesh.color.empty();
    for (uint32_t i = 0; i < vc; i++) {
        pos[i*3+0] = mesh.pos_x[i]; pos[i*3+1] = mesh.pos_y[i]; pos[i*3+2] = mesh.pos_z[i];
        norm[i*3+0] = mesh.norm_x[i]; norm[i*3+1] = mesh.norm_y[i]; norm[i*3+2] = mesh.norm_z[i];
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
        col_buf[i] = (has_color && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    }

    // pos/norm/ebo + mask/color: owned seam buffers. pos/norm carry Vertex|Storage,
    // ebo Index|Storage, mask/color Vertex|Storage (vertex attribute + brush-written
    // storage). The renderer's own `vao` is unused at draw (the seam render pipeline
    // owns the draw-time VAO), so no vertex-attribute pointers are configured here.
    // A size change releases+recreates (new handle) → Scene::bind_active_ refreshes the
    // compute.mask_ssbo / color_ssbo aliases right after this call.
    ensure_buffer(gpu_dev, vbo_pos, pos.data(), (uint64_t)pos.size()*sizeof(float),
                  gpu::Usage::Vertex | gpu::Usage::Storage);
    ensure_buffer(gpu_dev, vbo_norm, norm.data(), (uint64_t)norm.size()*sizeof(float),
                  gpu::Usage::Vertex | gpu::Usage::Storage);
    ensure_buffer(gpu_dev, ebo, mesh.indices.data(),
                  (uint64_t)mesh.indices.size()*sizeof(uint32_t),
                  gpu::Usage::Index | gpu::Usage::Storage);
    ensure_buffer(gpu_dev, vbo_mask, mask_buf.data(), (uint64_t)mask_buf.size()*sizeof(float),
                  gpu::Usage::Vertex | gpu::Usage::Storage);
    ensure_buffer(gpu_dev, vbo_color, col_buf.data(), (uint64_t)col_buf.size()*sizeof(uint32_t),
                  gpu::Usage::Vertex | gpu::Usage::Storage);
}

void Renderer::update_mask(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    std::vector<float> mask_buf(vc);
    for (uint32_t i = 0; i < vc; i++)
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
    gpu::write_buffer(gpu_dev, vbo_mask, 0, mask_buf.data(), (uint64_t)vc * sizeof(float));
}

void Renderer::update_mask_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts) {
    if (dirty_verts.empty()) return;

    uint32_t min_idx = dirty_verts[0], max_idx = dirty_verts[0];
    for (uint32_t v : dirty_verts) {
        if (v < min_idx) min_idx = v;
        if (v > max_idx) max_idx = v;
    }

    uint32_t range = max_idx - min_idx + 1;
    std::vector<float> buf(range);
    for (uint32_t i = 0; i < range; i++) {
        uint32_t v = min_idx + i;
        buf[i] = (v < mesh.mask.size()) ? mesh.mask[v] : 0.0f;
    }

    gpu::write_buffer(gpu_dev, vbo_mask, (uint64_t)min_idx * sizeof(float),
                      buf.data(), (uint64_t)range * sizeof(float));
}

void Renderer::update_mesh_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts) {
    if (dirty_verts.empty()) return;

    uint32_t min_idx = dirty_verts[0], max_idx = dirty_verts[0];
    for (uint32_t v : dirty_verts) {
        if (v < min_idx) min_idx = v;
        if (v > max_idx) max_idx = v;
    }

    uint32_t range = max_idx - min_idx + 1;
    std::vector<float> pos(range * 3), norm(range * 3);
    for (uint32_t i = 0; i < range; i++) {
        uint32_t v = min_idx + i;
        pos[i*3+0] = mesh.pos_x[v]; pos[i*3+1] = mesh.pos_y[v]; pos[i*3+2] = mesh.pos_z[v];
        norm[i*3+0] = mesh.norm_x[v]; norm[i*3+1] = mesh.norm_y[v]; norm[i*3+2] = mesh.norm_z[v];
    }

    uint32_t offset = min_idx * 3 * sizeof(float);
    uint32_t size = range * 3 * sizeof(float);

    gpu::write_buffer(gpu_dev, vbo_pos,  offset, pos.data(),  size);
    gpu::write_buffer(gpu_dev, vbo_norm, offset, norm.data(), size);
}

void Renderer::update_mesh_verts(const Mesh& mesh, const std::vector<uint32_t>& verts) {
    if (verts.empty()) return;

    static std::vector<uint32_t> sorted;
    sorted = verts;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    static std::vector<float> pos, norm;
    size_t i = 0, n = sorted.size();
    while (i < n) {
        // Coalesce a maximal run of consecutive vertex indices. Verts NOT in the
        // list are never included (runs hold no gaps), so untouched geometry —
        // and its GPU normals — is left exactly as the compute brush left it.
        uint32_t start = sorted[i], end = start;
        size_t j = i + 1;
        while (j < n && sorted[j] == end + 1) { end = sorted[j]; ++j; }
        uint32_t run = end - start + 1;

        pos.resize(run * 3);
        norm.resize(run * 3);
        for (uint32_t k = 0; k < run; k++) {
            uint32_t v = start + k;
            pos[k*3+0] = mesh.pos_x[v]; pos[k*3+1] = mesh.pos_y[v]; pos[k*3+2] = mesh.pos_z[v];
            norm[k*3+0] = mesh.norm_x[v]; norm[k*3+1] = mesh.norm_y[v]; norm[k*3+2] = mesh.norm_z[v];
        }
        uint32_t off = start * 3 * sizeof(float);
        uint32_t sz  = run * 3 * sizeof(float);
        gpu::write_buffer(gpu_dev, vbo_pos,  off, pos.data(),  sz);
        gpu::write_buffer(gpu_dev, vbo_norm, off, norm.data(), sz);
        i = j;
    }
}

void Renderer::update_mask_verts(const Mesh& mesh, const std::vector<uint32_t>& verts) {
    if (verts.empty()) return;

    static std::vector<uint32_t> sorted;
    sorted = verts;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    static std::vector<float> buf;
    size_t i = 0, n = sorted.size();
    while (i < n) {
        uint32_t start = sorted[i], end = start;
        size_t j = i + 1;
        while (j < n && sorted[j] == end + 1) { end = sorted[j]; ++j; }
        uint32_t run = end - start + 1;

        buf.resize(run);
        for (uint32_t k = 0; k < run; k++) {
            uint32_t v = start + k;
            buf[k] = (v < mesh.mask.size()) ? mesh.mask[v] : 0.0f;
        }
        gpu::write_buffer(gpu_dev, vbo_mask, (uint64_t)start * sizeof(float),
                          buf.data(), (uint64_t)run * sizeof(float));
        i = j;
    }
}

// Packed RGBA8 color, mirror of the mask uploaders. Missing entries default to
// white (0xFFFFFFFF) so an unpainted vertex renders byte-identical to today.
void Renderer::update_color(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    std::vector<uint32_t> buf(vc);
    bool has = !mesh.color.empty();
    for (uint32_t i = 0; i < vc; i++)
        buf[i] = (has && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    gpu::write_buffer(gpu_dev, vbo_color, 0, buf.data(), (uint64_t)vc * sizeof(uint32_t));
}

void Renderer::update_color_partial(const Mesh& mesh, const std::vector<uint32_t>& dirty_verts) {
    if (dirty_verts.empty()) return;

    uint32_t min_idx = dirty_verts[0], max_idx = dirty_verts[0];
    for (uint32_t v : dirty_verts) {
        if (v < min_idx) min_idx = v;
        if (v > max_idx) max_idx = v;
    }

    uint32_t range = max_idx - min_idx + 1;
    std::vector<uint32_t> buf(range);
    bool has = !mesh.color.empty();
    for (uint32_t i = 0; i < range; i++) {
        uint32_t v = min_idx + i;
        buf[i] = (has && v < mesh.color.size()) ? mesh.color[v] : 0xFFFFFFFFu;
    }

    gpu::write_buffer(gpu_dev, vbo_color, (uint64_t)min_idx * sizeof(uint32_t),
                      buf.data(), (uint64_t)range * sizeof(uint32_t));
}

void Renderer::update_color_verts(const Mesh& mesh, const std::vector<uint32_t>& verts) {
    if (verts.empty()) return;

    static std::vector<uint32_t> sorted;
    sorted = verts;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    static std::vector<uint32_t> buf;
    bool has = !mesh.color.empty();
    size_t i = 0, n = sorted.size();
    while (i < n) {
        uint32_t start = sorted[i], end = start;
        size_t j = i + 1;
        while (j < n && sorted[j] == end + 1) { end = sorted[j]; ++j; }
        uint32_t run = end - start + 1;

        buf.resize(run);
        for (uint32_t k = 0; k < run; k++) {
            uint32_t v = start + k;
            buf[k] = (has && v < mesh.color.size()) ? mesh.color[v] : 0xFFFFFFFFu;
        }
        gpu::write_buffer(gpu_dev, vbo_color, (uint64_t)start * sizeof(uint32_t),
                          buf.data(), (uint64_t)run * sizeof(uint32_t));
        i = j;
    }
}

void Renderer::draw_background(int w, int h) {
    gpu::RenderTarget target;          // fbo 0 (default framebuffer), no clear
    target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, bg_pipeline);
    gpu::set_vertex_buffer(rp, 0, bg_vbuf);
    gpu::draw(rp, 4);
    gpu::end_render_pass(rp);
}

// Fill + upload the matcap Params UBO from the camera and per-draw flags. Shared
// by draw_mesh (active entity) and draw_display (inactive entities).
void Renderer::upload_matcap_params(const Camera& cam, int w, int h,
                                    float facing_threshold, bool selected) {
    MatcapParamsGPU p;
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    p.facing_threshold = facing_threshold;
    p.obj_mask = selected ? 0.0f : 1.0f;
    p.paint_visible = paint_visible;
    p._pad0 = 0.0f;
    gpu::write_buffer(gpu_dev, matcap_ubo, 0, &p, sizeof p);
}

void Renderer::draw_mesh(const Camera& cam, int w, int h, uint32_t index_count,
                          float facing_threshold, bool selected) {
    upload_matcap_params(cam, w, h, facing_threshold, selected);

    gpu::RenderTarget target; target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, matcap_pipeline);
    gpu::BindBufferEntry be[] = {{ 63, &matcap_ubo, sizeof(MatcapParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, matcap_pipeline, be, 1);
    gpu::set_bind_group(rp, matcap_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, vbo_pos);
    gpu::set_vertex_buffer(rp, 1, vbo_norm);
    gpu::set_vertex_buffer(rp, 2, vbo_mask);
    gpu::set_vertex_buffer(rp, 3, vbo_color);
    gpu::set_index_buffer(rp, ebo);
    gpu::draw_indexed(rp, index_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);
}

// ---- Per-entity static display buffers ----

void Renderer::upload_display(EntityGpu& g, const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();

    std::vector<float> pos(vc * 3), norm(vc * 3), mask_buf(vc);
    std::vector<uint32_t> col_buf(vc);
    bool has_color = !mesh.color.empty();
    for (uint32_t i = 0; i < vc; i++) {
        pos[i*3+0] = mesh.pos_x[i]; pos[i*3+1] = mesh.pos_y[i]; pos[i*3+2] = mesh.pos_z[i];
        norm[i*3+0] = mesh.norm_x[i]; norm[i*3+1] = mesh.norm_y[i]; norm[i*3+2] = mesh.norm_z[i];
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
        col_buf[i] = (has_color && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    }

    // Display-only buffers: vertex attributes for the matcap + pick draws (pos also
    // bound as a pick vertex buffer). The render pipeline owns the vertex layout.
    ensure_buffer(gpu_dev, g.vbo_pos, pos.data(), (uint64_t)pos.size()*sizeof(float), gpu::Usage::Vertex);
    ensure_buffer(gpu_dev, g.vbo_norm, norm.data(), (uint64_t)norm.size()*sizeof(float), gpu::Usage::Vertex);
    ensure_buffer(gpu_dev, g.vbo_mask, mask_buf.data(), (uint64_t)mask_buf.size()*sizeof(float), gpu::Usage::Vertex);
    ensure_buffer(gpu_dev, g.vbo_color, col_buf.data(), (uint64_t)col_buf.size()*sizeof(uint32_t), gpu::Usage::Vertex);
    ensure_buffer(gpu_dev, g.ebo, mesh.indices.data(),
                  (uint64_t)mesh.indices.size()*sizeof(uint32_t), gpu::Usage::Index);

    g.index_count = (uint32_t)mesh.indices.size();
    g.dirty = false;
}

void Renderer::free_display(EntityGpu& g) {
    gpu::release_buffer(g.vbo_pos);
    gpu::release_buffer(g.vbo_norm);
    gpu::release_buffer(g.vbo_mask);
    gpu::release_buffer(g.vbo_color);
    gpu::release_buffer(g.ebo);
    g = EntityGpu();
}

void Renderer::draw_display(const Camera& cam, EntityGpu& g, int w, int h,
                            float facing_threshold, bool selected) {
    if (g.index_count == 0) return;
    upload_matcap_params(cam, w, h, facing_threshold, selected);

    gpu::RenderTarget target; target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, matcap_pipeline);
    gpu::BindBufferEntry be[] = {{ 63, &matcap_ubo, sizeof(MatcapParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, matcap_pipeline, be, 1);
    gpu::set_bind_group(rp, matcap_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, g.vbo_pos);
    gpu::set_vertex_buffer(rp, 1, g.vbo_norm);
    gpu::set_vertex_buffer(rp, 2, g.vbo_mask);
    gpu::set_vertex_buffer(rp, 3, g.vbo_color);
    gpu::set_index_buffer(rp, g.ebo);
    gpu::draw_indexed(rp, g.index_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);
}

// ---- Entity-id pick pass (renders into the shared screen offscreen target) ----

void Renderer::pick_begin(const Camera& cam, int w, int h) {
    create_screen_buffers(w, h);

    // Pick writes only depth (att 0) + id (att 2); attachments 1 (normal) and 3
    // (bary) are disabled so the brush data there is left intact.
    gpu::OffscreenPassDesc d;
    d.color[0].enabled = true;  d.color[0].clear = true;  d.color[0].f[0] = 1000.0f; // linear depth
    d.color[1].enabled = false;
    d.color[2].enabled = true;  d.color[2].clear = true;  d.color[2].is_uint = true;  // id (0 = none)
    d.color[3].enabled = false;
    d.clear_depth = true;
    pick_pass = gpu::begin_offscreen_pass(gpu_dev, screen_target, d);
    gpu::set_pipeline(pick_pass, pick_pipeline);

    // view/proj are constant across the pass; the per-entity id is written per draw.
    PickParamsGPU p{};
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    gpu::write_buffer(gpu_dev, pick_ubo, 0,  p.view, sizeof p.view);
    gpu::write_buffer(gpu_dev, pick_ubo, 64, p.proj, sizeof p.proj);

    gpu::BindBufferEntry be[] = {{ 63, &pick_ubo, sizeof(PickParamsGPU) }};
    pick_bg = gpu::create_bind_group(gpu_dev, pick_pipeline, be, 1);
    gpu::set_bind_group(pick_pass, pick_pipeline, pick_bg);
}

void Renderer::pick_draw(uint32_t entity_id, const gpu::Buffer& pos_vbo, const gpu::Buffer& ebo_, uint32_t index_count) {
    if (!pos_vbo.handle || index_count == 0) return;
    gpu::write_buffer(gpu_dev, pick_ubo, 128, &entity_id, sizeof entity_id);
    gpu::set_vertex_buffer(pick_pass, 0, pos_vbo);
    gpu::set_index_buffer(pick_pass, ebo_);
    gpu::draw_indexed(pick_pass, index_count);
}

void Renderer::pick_end() {
    gpu::end_render_pass(pick_pass);   // rebinds the default framebuffer
    gpu::release_bind_group(pick_bg);
}

void Renderer::read_id_region(int x, int y, int w, int h, uint32_t* out) {
    // Entity ids live in attachment 2 (same R32UI buffer as triangle ids).
    read_triid_region(x, y, w, h, out);
}

void Renderer::draw_cursor(const Camera& cam, float cx, float cy, float radius,
                           float nx, float ny, float nz, float hardness,
                           int w, int h, bool on_model) {
    // Color: pale lime (soft) -> bright orange (mid) -> bluish purple (hard).
    float hh = std::max(0.0f, std::min(1.0f, hardness));
    float cr, cg, cb;
    if (hh < 0.5f) {
        float t = hh * 2.0f;
        cr = 0.72f * (1.0f - t) + 1.00f * t;
        cg = 1.00f * (1.0f - t) + 0.55f * t;
        cb = 0.58f * (1.0f - t) + 0.10f * t;
    } else {
        float t = (hh - 0.5f) * 2.0f;
        cr = 1.00f * (1.0f - t) + 0.55f * t;
        cg = 0.55f * (1.0f - t) + 0.40f * t;
        cb = 0.10f * (1.0f - t) + 0.95f * t;
    }
    float ca = 0.75f + 0.15f * hh;

    // Camera basis in world space (matches brush.cpp apply_move derivation)
    Vec3 cam_pos = cam.get_position();
    Vec3 cam_fwd = (cam.target - cam_pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 cam_right = cam_fwd.cross(world_up).normalized();
    Vec3 cam_up = cam_right.cross(cam_fwd).normalized();

    gpu::RenderTarget target;          // fbo 0 (default framebuffer), no clear
    target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);

    // ---- Shadow disc first (under the ring). Tinted with brush color, muted. ----
    {
        ShadowParamsGPU sp{};
        sp.center[0] = cx; sp.center[1] = cy;
        sp.radius = radius;
        sp.screenSize[0] = (float)w; sp.screenSize[1] = (float)h;
        sp.color[0] = cr * 0.5f; sp.color[1] = cg * 0.5f; sp.color[2] = cb * 0.5f; sp.color[3] = 0.22f;
        gpu::write_buffer(gpu_dev, shadow_ubo, 0, &sp, sizeof sp);
        gpu::set_pipeline(rp, shadow_pipeline);
        gpu::BindBufferEntry be[] = {{ 63, &shadow_ubo, sizeof(ShadowParamsGPU) }};
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, shadow_pipeline, be, 1);
        gpu::set_bind_group(rp, shadow_pipeline, grp);
        gpu::set_vertex_buffer(rp, 0, shadow_vbuf);
        gpu::draw(rp, CURSOR_SEGS * 3);
        gpu::release_bind_group(grp);
    }

    // ---- Ring (outer), plus an optional thinner inner hardness ring ----
    {
        gpu::set_pipeline(rp, cursor_pipeline);
        gpu::BindBufferEntry be[] = {{ 63, &cursor_ubo, sizeof(CursorParamsGPU) }};
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, cursor_pipeline, be, 1);
        gpu::set_bind_group(rp, cursor_pipeline, grp);
        gpu::set_vertex_buffer(rp, 0, cursor_vbuf);

        CursorParamsGPU cp{};
        cp.center[0] = cx; cp.center[1] = cy;
        cp.screenSize[0] = (float)w; cp.screenSize[1] = (float)h;
        cp.normal[0] = nx; cp.normal[1] = ny; cp.normal[2] = nz; cp.radius = radius;
        cp.camRight[0] = cam_right.x; cp.camRight[1] = cam_right.y; cp.camRight[2] = cam_right.z;
        cp.baseThick = 2.5f;
        cp.camUp[0] = cam_up.x; cp.camUp[1] = cam_up.y; cp.camUp[2] = cam_up.z;
        cp.frontBoost = 3.0f;
        cp.camFwd[0] = cam_fwd.x; cp.camFwd[1] = cam_fwd.y; cp.camFwd[2] = cam_fwd.z;
        cp.color[0] = cr; cp.color[1] = cg; cp.color[2] = cb; cp.color[3] = ca;
        gpu::write_buffer(gpu_dev, cursor_ubo, 0, &cp, sizeof cp);
        gpu::draw(rp, (CURSOR_SEGS + 1) * 2);

        if (hh > 0.5f) {
            // Re-upload the shared UBO mid-pass for the second draw. Fine on GL
            // (immediate); the web backend will need a 2nd UBO / dynamic offset here.
            cp.radius = std::max(3.0f, radius - 5.0f);
            cp.baseThick = 1.5f; cp.frontBoost = 1.5f;
            cp.color[3] = (hh - 0.5f) * 2.0f * ca;
            gpu::write_buffer(gpu_dev, cursor_ubo, 0, &cp, sizeof cp);
            gpu::draw(rp, (CURSOR_SEGS + 1) * 2);
        }
        gpu::release_bind_group(grp);
    }

    // ---- Crosshair (X at center, on-model only) ----
    if (on_model) {
        CrosshairParamsGPU xp{};
        xp.center[0] = cx; xp.center[1] = cy;
        xp.screenSize[0] = (float)w; xp.screenSize[1] = (float)h;
        xp.color[0] = cr * 1.4f; xp.color[1] = cg * 1.4f; xp.color[2] = cb * 1.4f; xp.color[3] = 1.0f;
        gpu::write_buffer(gpu_dev, crosshair_ubo, 0, &xp, sizeof xp);
        gpu::set_pipeline(rp, crosshair_pipeline);
        gpu::BindBufferEntry be[] = {{ 63, &crosshair_ubo, sizeof(CrosshairParamsGPU) }};
        gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, crosshair_pipeline, be, 1);
        gpu::set_bind_group(rp, crosshair_pipeline, grp);
        gpu::set_vertex_buffer(rp, 0, crosshair_vbuf);
#if defined(CHISEL_BACKEND_GL)
        glLineWidth(3.0f);                    // GL-only nicety (WebGPU ignores line width)
#endif
        gpu::draw(rp, 8);
#if defined(CHISEL_BACKEND_GL)
        glLineWidth(1.0f);
#endif
        gpu::release_bind_group(grp);
    }

    gpu::end_render_pass(rp);
}

// ---- Screen Buffer FBO ----

void Renderer::create_screen_buffers(int w, int h) {
    if (screen_target.color_count) {   // already created → just resize
        resize_screen_buffers(w, h);
        return;
    }
    // Attachments: 0 linear depth (R32F), 1 world normal (RGB16F), 2 triangle id
    // (R32UI), 3 barycentrics (RG16F) + a depth/stencil RBO for z-test.
    gpu::TexFormat fmts[] = { gpu::TexFormat::R32F, gpu::TexFormat::RGB16F,
                              gpu::TexFormat::R32UI, gpu::TexFormat::RG16F };
    screen_target = gpu::create_offscreen_target(gpu_dev, w, h, fmts, 4);
}

void Renderer::resize_screen_buffers(int w, int h) {
    gpu::resize_offscreen_target(gpu_dev, screen_target, w, h);
}

void Renderer::upload_screen_mesh(const Mesh& mesh) {
    uint32_t tc = mesh.tri_count();
    screen_tri_count = tc;
    uint32_t flat_vc = tc * 3;

    // pos/norm are filled by the expand compute (Vertex|Storage); allocate them empty,
    // reallocating only when the flat vertex count changes (contents aren't preserved).
    uint64_t pn_bytes = (uint64_t)flat_vc * 3 * sizeof(float);
    if (screen_vbo_pos.size != pn_bytes) {
        gpu::release_buffer(screen_vbo_pos);
        screen_vbo_pos = gpu::create_buffer(gpu_dev, nullptr, pn_bytes, gpu::Usage::Vertex | gpu::Usage::Storage);
    }
    if (screen_vbo_norm.size != pn_bytes) {
        gpu::release_buffer(screen_vbo_norm);
        screen_vbo_norm = gpu::create_buffer(gpu_dev, nullptr, pn_bytes, gpu::Usage::Vertex | gpu::Usage::Storage);
    }

    // Triid and bary are static per topology — build on CPU.
    std::vector<float> triid(flat_vc);
    std::vector<float> bary(flat_vc * 2);
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t base = t * 3;
        triid[base+0] = (float)t; triid[base+1] = (float)t; triid[base+2] = (float)t;
        bary[(base+0)*2+0] = 1.0f; bary[(base+0)*2+1] = 0.0f;
        bary[(base+1)*2+0] = 0.0f; bary[(base+1)*2+1] = 1.0f;
        bary[(base+2)*2+0] = 0.0f; bary[(base+2)*2+1] = 0.0f;
    }
    ensure_buffer(gpu_dev, screen_vbo_triid, triid.data(), triid.size()*sizeof(float), gpu::Usage::Vertex);
    ensure_buffer(gpu_dev, screen_vbo_bary, bary.data(), bary.size()*sizeof(float), gpu::Usage::Vertex);

    // Fill positions/normals via GPU expansion.
    update_screen_mesh_gpu();
}

void Renderer::update_screen_positions(const Mesh& mesh) {
    (void)mesh;
    update_screen_mesh_gpu();
}

void Renderer::update_screen_mesh_gpu() {
    if (!screen_expand_pipeline.handle || screen_tri_count == 0) return;

    ScreenExpandParamsGPU p{}; p.tri_count = screen_tri_count;
    gpu::write_buffer(gpu_dev, screen_expand_ubo, 0, &p, sizeof p);

    // Seam-owned buffers bound directly. 0-2 read (mesh pos/norm/idx), 3-4 written
    // (flat soup pos/norm), 63 = tri_count UBO.
    gpu::BindBufferEntry be[] = {
        { 0, &vbo_pos, 0 }, { 1, &vbo_norm, 0 }, { 2, &ebo, 0 },
        { 3, &screen_vbo_pos, 0 }, { 4, &screen_vbo_norm, 0 },
        { 63, &screen_expand_ubo, sizeof(ScreenExpandParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, screen_expand_pipeline, be, 6);

    gpu::ComputeBatch batch = gpu::begin_compute(gpu_dev);
    uint32_t groups = (screen_tri_count + 63) / 64;
    gpu::dispatch(batch, screen_expand_pipeline, grp, groups);
    gpu::submit(batch);   // issues the vertex-attrib + buffer-update barriers
    gpu::release_bind_group(grp);
}

void Renderer::render_screen_buffers(const Camera& cam, int w, int h) {
    create_screen_buffers(w, h);

    // Clear: depth→max, normal→0, triid→0xFFFFFFFF (no triangle), bary→0.
    gpu::OffscreenPassDesc d;
    d.color[0].clear = true; d.color[0].f[0] = 1000.0f;
    d.color[1].clear = true;
    d.color[2].clear = true; d.color[2].is_uint = true; d.color[2].u[0] = 0xFFFFFFFFu;
    d.color[3].clear = true;
    d.clear_depth = true;
    gpu::RenderPass rp = gpu::begin_offscreen_pass(gpu_dev, screen_target, d);
    gpu::set_pipeline(rp, screen_pipeline);

    ScreenParamsGPU p;
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    gpu::write_buffer(gpu_dev, screen_ubo, 0, &p, sizeof p);
    gpu::BindBufferEntry be[] = {{ 63, &screen_ubo, sizeof(ScreenParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, screen_pipeline, be, 1);
    gpu::set_bind_group(rp, screen_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, screen_vbo_pos);
    gpu::set_vertex_buffer(rp, 1, screen_vbo_norm);
    gpu::set_vertex_buffer(rp, 2, screen_vbo_triid);
    gpu::set_vertex_buffer(rp, 3, screen_vbo_bary);
    gpu::draw(rp, screen_tri_count * 3);
    gpu::end_render_pass(rp);   // rebinds the default framebuffer
    gpu::release_bind_group(grp);
}

void Renderer::read_depth_region(int x, int y, int w, int h, float* out) {
    gpu::read_target_region(gpu_dev, screen_target, 0, x, y, w, h, out);
}

void Renderer::read_normal_region(int x, int y, int w, int h, float* out) {
    gpu::read_target_region(gpu_dev, screen_target, 1, x, y, w, h, out);
}

void Renderer::read_triid_region(int x, int y, int w, int h, uint32_t* out) {
    gpu::read_target_region(gpu_dev, screen_target, 2, x, y, w, h, out);
}

void Renderer::read_bary_region(int x, int y, int w, int h, float* out) {
    gpu::read_target_region(gpu_dev, screen_target, 3, x, y, w, h, out);
}

void Renderer::draw_debug_mesh(const Camera& cam, const Mesh& mesh, int w, int h) {
    uint32_t tc = mesh.tri_count();

    // Build the edge index buffer lazily (GL-owned; rebuilt after invalidate).
    if (debug_edge_count == 0) {
        std::vector<uint32_t> edges;
        edges.reserve(tc * 6);
        for (uint32_t i = 0; i < tc; i++) {
            uint32_t i0 = mesh.indices[i*3+0];
            uint32_t i1 = mesh.indices[i*3+1];
            uint32_t i2 = mesh.indices[i*3+2];
            edges.push_back(i0); edges.push_back(i1);
            edges.push_back(i1); edges.push_back(i2);
            edges.push_back(i2); edges.push_back(i0);
        }
        debug_edge_count = (uint32_t)edges.size();
        ensure_buffer(gpu_dev, debug_edge_vbo, edges.data(),
                      edges.size() * sizeof(uint32_t), gpu::Usage::Index);
    }

    DebugParamsGPU p;
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    gpu::write_buffer(gpu_dev, debug_edge_ubo, 0, &p, sizeof p);

    // Depth-compare LEQUAL + polygon offset pull the wires just in front of the
    // surface (anti z-fight). The seam doesn't model depth-func / depth-bias yet,
    // so these stay raw GL around the seam draw (they fold into RenderPipelineDesc
    // when the WebGPU render backend lands; WebGPU ignores line width). set_pipeline
    // enables depth test + blend and leaves depth-func untouched, so restore GL_LESS
    // afterwards.
#if defined(CHISEL_BACKEND_GL)
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1.0f, -1.0f);
    glLineWidth(1.0f);
#endif

    gpu::RenderTarget target;          // fbo 0 (default framebuffer), no clear
    target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, debug_edge_pipeline);
    gpu::BindBufferEntry be[] = {{ 63, &debug_edge_ubo, sizeof(DebugParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, debug_edge_pipeline, be, 1);
    gpu::set_bind_group(rp, debug_edge_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, vbo_pos);
    gpu::set_index_buffer(rp, debug_edge_vbo);
    gpu::draw_indexed(rp, debug_edge_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);

#if defined(CHISEL_BACKEND_GL)
    glDisable(GL_POLYGON_OFFSET_LINE);
    glDepthFunc(GL_LESS);
#endif
}
