#include "renderer.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_set>

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

// Debug-mesh edge overlay (wireframe) Params UBO — view/proj + the screen-space
// expansion inputs (std140 at binding 63). The overlay draws edges as expanded
// quads (see debug_vert_src), so it needs the viewport in pixels, the half-width
// to expand by, and a small NDC depth bias to pull the wires off the surface.
struct DebugParamsGPU {
    float view[16];
    float proj[16];
    float viewport[2];       // 128  framebuffer size in px
    float half_width_px;     // 136  line half-width (zoom-compensated)
    float depth_bias_ndc;    // 140  toward-viewer bias (replaces glPolygonOffset)
};
static_assert(sizeof(DebugParamsGPU) == 144, "debug Params UBO must be 144 bytes (std140)");

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
    // 2D group grid: 1D would exceed the 65535 per-dim dispatch limit past
    // ~4.2M tris (multires L9). Same recovery as the SDF kernels.
    uint t = gl_GlobalInvocationID.x
           + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 64u;
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

// Wireframe overlay, screen-space fat lines. GL wide lines are capped/deprecated
// in core profile and WebGPU has no line width at all, so each edge is drawn as
// a camera-facing quad (2 tris = 6 verts) expanded in the VERTEX shader: no
// vertex buffer — gl_VertexID pulls the edge's endpoints from the mesh position
// SSBO via an edge-pair SSBO. The lateral coordinate (±1 across the ribbon)
// feathers alpha over the outermost ~1px in the fragment stage → antialiased on
// every backend, any width. Depth bias moved into the shader (toward-viewer NDC
// nudge) since polygon offset can't apply to what is now fill geometry.
// std140 Params block (binding 63) — byte-identical to DebugParamsGPU.
#define DEBUG_PARAMS_BLOCK \
    "layout(std140, binding=63) uniform Params {\n" \
    "    mat4 uView;\n" \
    "    mat4 uProj;\n" \
    "    vec2 uViewport;\n" \
    "    float uHalfW;\n" \
    "    float uBias;\n" \
    "};\n"

static const char* debug_vert_src =
"#version 430 core\n"
"layout(std430, binding = 0) readonly buffer PosIn { float vpos[]; };\n"
"layout(std430, binding = 1) readonly buffer EdgeIn { uint eidx[]; };\n"
DEBUG_PARAMS_BLOCK
R"(
out float vLat;
void main() {
    uint vid = uint(gl_VertexID);
    uint e = vid / 6u;
    uint c = vid % 6u;
    uint ia = eidx[e*2u+0u], ib = eidx[e*2u+1u];
    vec4 ca = uProj * uView * vec4(vpos[ia*3u+0u], vpos[ia*3u+1u], vpos[ia*3u+2u], 1.0);
    vec4 cb = uProj * uView * vec4(vpos[ib*3u+0u], vpos[ib*3u+1u], vpos[ib*3u+2u], 1.0);
    vec2 sa = (ca.xy / ca.w * 0.5 + 0.5) * uViewport;
    vec2 sb = (cb.xy / cb.w * 0.5 + 0.5) * uViewport;
    vec2 d  = sb - sa;
    vec2 n  = vec2(-d.y, d.x) / max(length(d), 1e-4);
    // 6 verts -> 2 tris over ribbon corners 0:A+ 1:A- 2:B+ 3:B-
    uint corner = (c == 0u) ? 0u : (c == 1u || c == 4u) ? 1u : (c == 5u) ? 3u : 2u;
    bool  atB  = corner >= 2u;
    float side = (corner == 1u || corner == 3u) ? -1.0 : 1.0;
    vec4 clip = atB ? cb : ca;
    vec2 sp   = (atB ? sb : sa) + n * (side * uHalfW);
    vec2 ndc  = sp / uViewport * 2.0 - 1.0;
    gl_Position = vec4(ndc * clip.w, clip.z - uBias * clip.w, clip.w);
    vLat = side;
}
)";

static const char* debug_edge_frag_src =
"#version 430 core\n"
DEBUG_PARAMS_BLOCK
R"(
in float vLat;
out vec4 fragColor;
void main() {
    float aa = clamp((1.0 - abs(vLat)) * uHalfW, 0.0, 1.0);   // ~1px edge feather
    fragColor = vec4(0.08, 0.33, 0.42, aa);   // Aegean blue, cypress-green tinge
}
)";

// ---- WGSL render shaders (WebGPU backend) ----
// One module per program, vs_main + fs_main (the names the seam hardcodes). The
// std140 Params blocks mirror the *ParamsGPU structs above byte-for-byte — WGSL's
// uniform layout matches std140 for these flat scalar/vec/mat members. No clip-Z
// remap: the camera is orthographic with a symmetric (-100,100) near/far, so clip-z
// = 0.01*depth stays inside both GL's [-1,1] and WebGPU's [0,1] (see camera.cpp).

static const char* bg_wgsl_src = R"WGSL(
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@location(0) aPos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.uv = aPos * 0.5 + 0.5;
    o.pos = vec4<f32>(aPos, 0.999, 1.0);
    return o;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let t = in.uv.y;
    let bot = vec3<f32>(0.18, 0.18, 0.20);
    let top = vec3<f32>(0.28, 0.28, 0.30);
    return vec4<f32>(mix(bot, top, t), 1.0);
}
)WGSL";

static const char* matcap_wgsl_src = R"WGSL(
struct Params {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    facing_threshold: f32,
    obj_mask: f32,
    paint_visible: f32,
    _pad0: f32,
};
@group(0) @binding(63) var<uniform> P: Params;
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) mask: f32,
    @location(2) color: vec4<f32>,
};
@vertex
fn vs_main(@location(0) aPos: vec3<f32>, @location(1) aNorm: vec3<f32>,
           @location(2) aMask: f32, @location(4) aColor: vec4<f32>) -> VSOut {
    var o: VSOut;
    let nm = mat3x3<f32>(P.view[0].xyz, P.view[1].xyz, P.view[2].xyz);
    o.nrm = normalize(nm * aNorm);
    o.mask = aMask;
    o.color = aColor;
    o.pos = P.proj * P.view * vec4<f32>(aPos, 1.0);
    return o;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.nrm);
    let rim = 1.0 - abs(n.z);
    let top = n.y * 0.5 + 0.5;
    let base = 0.35 + 0.45 * top + 0.15 * (1.0 - rim * rim);
    let cavity = 1.0 - rim * rim * 0.3;
    var val = base * cavity;
    val = val * mix(1.0, 0.4, in.mask);
    var col = vec3<f32>(val) * mix(vec3<f32>(1.0), in.color.rgb, P.paint_visible);
    if (P.obj_mask > 0.0) {
        let tint = vec3<f32>(0.35, 0.12, 0.18);
        col = mix(col, col * tint, P.obj_mask);
    }
    let facing = max(n.z, 0.0);
    let edge = abs(facing - P.facing_threshold);
    let px_width = fwidth(facing);
    if (P.facing_threshold > 0.001 && edge < px_width * 0.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    return vec4<f32>(col, 1.0);
}
)WGSL";

// Pick MRT: writes depth@0 + id@2, but the pipeline declares all 4 screen-target
// attachments, so the fragment must output all 4 (WebGPU rejects partial writes).
static const char* pick_wgsl_src = R"WGSL(
struct Params { view: mat4x4<f32>, proj: mat4x4<f32>, entity_id: u32 };
@group(0) @binding(63) var<uniform> P: Params;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) depth: f32 };
@vertex
fn vs_main(@location(0) aPos: vec3<f32>) -> VSOut {
    var o: VSOut;
    let viewPos = P.view * vec4<f32>(aPos, 1.0);
    o.depth = -viewPos.z;
    o.pos = P.proj * viewPos;
    return o;
}
struct FSOut {
    @location(0) depth: f32,
    @location(1) nrm: vec4<f32>,
    @location(2) id: u32,
    @location(3) bary: vec2<f32>,
};
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var o: FSOut;
    o.depth = in.depth;
    o.nrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    o.id = P.entity_id;
    o.bary = vec2<f32>(0.0, 0.0);
    return o;
}
)WGSL";

static const char* cursor_wgsl_src = R"WGSL(
struct Params {
    center: vec2<f32>,
    screenSize: vec2<f32>,
    normal: vec3<f32>,    radius: f32,
    camRight: vec3<f32>,  baseThick: f32,
    camUp: vec3<f32>,     frontBoost: f32,
    camFwd: vec3<f32>,    _pad0: f32,
    color: vec4<f32>,
};
@group(0) @binding(63) var<uniform> P: Params;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) front: f32 };
@vertex
fn vs_main(@location(0) aLocal: vec2<f32>, @location(1) aOuter: f32) -> VSOut {
    var o: VSOut;
    let N = normalize(P.normal);
    let r = select(P.camUp, P.camRight, abs(dot(P.camRight, N)) < 0.95);
    let U = normalize(r - dot(r, N) * N);
    let V = cross(N, U);
    var u_screen = vec2<f32>(dot(U, P.camRight), -dot(U, P.camUp));
    var v_screen = vec2<f32>(dot(V, P.camRight), -dot(V, P.camUp));
    let MIN_VIS = 0.18;
    let u_len = length(u_screen);
    let v_len = length(v_screen);
    if (u_len < MIN_VIS) {
        if (u_len > 1e-5) {
            u_screen = u_screen * (MIN_VIS / u_len);
        } else {
            let vn = normalize(v_screen);
            u_screen = MIN_VIS * vec2<f32>(-vn.y, vn.x);
        }
    }
    if (v_len < MIN_VIS) {
        if (v_len > 1e-5) {
            v_screen = v_screen * (MIN_VIS / v_len);
        } else {
            let un = normalize(u_screen);
            v_screen = MIN_VIS * vec2<f32>(-un.y, un.x);
        }
    }
    let ellipse_rim = u_screen * aLocal.x + v_screen * aLocal.y;
    let dpos_dt = -u_screen * aLocal.y + v_screen * aLocal.x;
    var outward = vec2<f32>(-dpos_dt.y, dpos_dt.x);
    if (dot(outward, ellipse_rim) < 0.0) { outward = -outward; }
    let out_len = length(outward);
    outward = select(vec2<f32>(1.0, 0.0), outward / out_len, out_len > 1e-6);
    let rim_world = U * aLocal.x + V * aLocal.y;
    o.front = -dot(rim_world, P.camFwd);
    let tFront = clamp(o.front, 0.0, 1.0);
    let halfThick = (P.baseThick + P.frontBoost * tFront) * 0.5;
    let screen = P.center + ellipse_rim * P.radius + outward * (aOuter - 0.5) * 2.0 * halfThick;
    var ndc = (screen / P.screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    o.pos = vec4<f32>(ndc, 0.0, 1.0);
    return o;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let t = clamp(in.front * 0.5 + 0.5, 0.0, 1.0);
    let bright = mix(0.78, 1.12, t);
    let a = P.color.a * mix(0.82, 1.0, t);
    return vec4<f32>(P.color.rgb * bright, a);
}
)WGSL";

static const char* crosshair_wgsl_src = R"WGSL(
struct Params { center: vec2<f32>, screenSize: vec2<f32>, color: vec4<f32> };
@group(0) @binding(63) var<uniform> P: Params;
@vertex
fn vs_main(@location(0) aPos: vec2<f32>) -> @builtin(position) vec4<f32> {
    let screen = P.center + aPos;
    var ndc = (screen / P.screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    return vec4<f32>(ndc, 0.0, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4<f32> { return P.color; }
)WGSL";

static const char* cursor_shadow_wgsl_src = R"WGSL(
struct Params {
    center: vec2<f32>,
    radius: f32,           _pad0: f32,
    screenSize: vec2<f32>, _pad1: vec2<f32>,
    color: vec4<f32>,
};
@group(0) @binding(63) var<uniform> P: Params;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) dist: f32 };
@vertex
fn vs_main(@location(0) aPos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.dist = length(aPos);
    let screen = P.center + aPos * P.radius;
    var ndc = (screen / P.screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    o.pos = vec4<f32>(ndc, 0.0, 1.0);
    return o;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(0.72, 1.0, in.dist);
    let core = 0.55 + 0.45 * smoothstep(0.0, 0.55, in.dist);
    let a = P.color.a * edge * core;
    return vec4<f32>(P.color.rgb, a);
}
)WGSL";

static const char* screen_buf_wgsl_src = R"WGSL(
struct Params { view: mat4x4<f32>, proj: mat4x4<f32> };
@group(0) @binding(63) var<uniform> P: Params;
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) nrmWorld: vec3<f32>,
    @location(1) depth: f32,
    @location(2) @interpolate(flat) triID: u32,
    @location(3) bary: vec2<f32>,
};
@vertex
fn vs_main(@location(0) aPos: vec3<f32>, @location(1) aNorm: vec3<f32>,
           @location(2) aTriID: f32, @location(3) aBary: vec2<f32>) -> VSOut {
    var o: VSOut;
    let viewPos = P.view * vec4<f32>(aPos, 1.0);
    o.nrmWorld = normalize(aNorm);
    o.depth = -viewPos.z;
    o.triID = u32(aTriID);
    o.bary = aBary;
    o.pos = P.proj * viewPos;
    return o;
}
struct FSOut {
    @location(0) depth: f32,
    @location(1) nrm: vec4<f32>,
    @location(2) triID: u32,
    @location(3) bary: vec2<f32>,
};
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var o: FSOut;
    o.depth = in.depth;
    o.nrm = vec4<f32>(normalize(in.nrmWorld), 1.0);
    o.triID = in.triID;
    o.bary = in.bary;
    return o;
}
)WGSL";

// Indexed->flat expand (compute). writeonly GLSL maps to read_write storage in WGSL.
static const char* screen_expand_wgsl_src = R"WGSL(
@group(0) @binding(0) var<storage, read>       in_pos:  array<f32>;
@group(0) @binding(1) var<storage, read>       in_norm: array<f32>;
@group(0) @binding(2) var<storage, read>       in_idx:  array<u32>;
@group(0) @binding(3) var<storage, read_write> out_pos:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out_norm: array<f32>;
struct Params { tri_count: u32 };
@group(0) @binding(63) var<uniform> P: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)        nwg: vec3<u32>) {
    // 2D group grid past the 65535 per-dim dispatch limit (multires L9).
    let t = gid.x + gid.y * nwg.x * 64u;
    if (t >= P.tri_count) { return; }
    let i0 = in_idx[t*3u+0u];
    let i1 = in_idx[t*3u+1u];
    let i2 = in_idx[t*3u+2u];
    let base = t * 9u;
    out_pos[base+0u] = in_pos[i0*3u+0u]; out_pos[base+1u] = in_pos[i0*3u+1u]; out_pos[base+2u] = in_pos[i0*3u+2u];
    out_pos[base+3u] = in_pos[i1*3u+0u]; out_pos[base+4u] = in_pos[i1*3u+1u]; out_pos[base+5u] = in_pos[i1*3u+2u];
    out_pos[base+6u] = in_pos[i2*3u+0u]; out_pos[base+7u] = in_pos[i2*3u+1u]; out_pos[base+8u] = in_pos[i2*3u+2u];
    out_norm[base+0u] = in_norm[i0*3u+0u]; out_norm[base+1u] = in_norm[i0*3u+1u]; out_norm[base+2u] = in_norm[i0*3u+2u];
    out_norm[base+3u] = in_norm[i1*3u+0u]; out_norm[base+4u] = in_norm[i1*3u+1u]; out_norm[base+5u] = in_norm[i1*3u+2u];
    out_norm[base+6u] = in_norm[i2*3u+0u]; out_norm[base+7u] = in_norm[i2*3u+1u]; out_norm[base+8u] = in_norm[i2*3u+2u];
}
)WGSL";

static const char* debug_wgsl_src = R"WGSL(
struct Params {
    view: mat4x4<f32>, proj: mat4x4<f32>,
    viewport: vec2<f32>, half_w: f32, bias: f32,
};
@group(0) @binding(63) var<uniform> P: Params;
@group(0) @binding(0) var<storage, read> vpos: array<f32>;
@group(0) @binding(1) var<storage, read> eidx: array<u32>;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) lat: f32 };
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    let e = vid / 6u;
    let c = vid % 6u;
    let ia = eidx[e*2u+0u]; let ib = eidx[e*2u+1u];
    let ca = P.proj * P.view * vec4<f32>(vpos[ia*3u], vpos[ia*3u+1u], vpos[ia*3u+2u], 1.0);
    let cb = P.proj * P.view * vec4<f32>(vpos[ib*3u], vpos[ib*3u+1u], vpos[ib*3u+2u], 1.0);
    let sa = (ca.xy / ca.w * 0.5 + vec2<f32>(0.5)) * P.viewport;
    let sb = (cb.xy / cb.w * 0.5 + vec2<f32>(0.5)) * P.viewport;
    let d  = sb - sa;
    let n  = vec2<f32>(-d.y, d.x) / max(length(d), 1e-4);
    // 6 verts -> 2 tris over ribbon corners 0:A+ 1:A- 2:B+ 3:B-
    var corner = 2u;
    if (c == 0u) { corner = 0u; }
    else if (c == 1u || c == 4u) { corner = 1u; }
    else if (c == 5u) { corner = 3u; }
    let atB  = corner >= 2u;
    var side = 1.0;
    if (corner == 1u || corner == 3u) { side = -1.0; }
    var clip = ca; var sp = sa;
    if (atB) { clip = cb; sp = sb; }
    sp = sp + n * (side * P.half_w);
    let ndc = sp / P.viewport * 2.0 - vec2<f32>(1.0);
    var o: VSOut;
    o.pos = vec4<f32>(ndc * clip.w, clip.z - P.bias * clip.w, clip.w);
    o.lat = side;
    return o;
}
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let aa = clamp((1.0 - abs(in.lat)) * P.half_w, 0.0, 1.0);
    return vec4<f32>(0.08, 0.33, 0.42, aa);
}
)WGSL";

// ---- Renderer ----

Renderer::Renderer()
    : debug_edge_count(0)
    , debug_edge_src_tris(0)
    , debug_mesh_radius(1.0f)
    , screen_tri_count(0)
    , initialized(false)
{}

Renderer::~Renderer() {
    gpu::release_buffer(vbo_pos);
    gpu::release_buffer(vbo_norm);
    gpu::release_buffer(vbo_mask);
    gpu::release_buffer(vbo_color);
    gpu::release_buffer(vbo_density);
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
    for (auto& bg : pick_bgs) gpu::release_bind_group(bg);
    for (auto& ubo : pick_ubos) gpu::release_buffer(ubo);
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
    gpu_dev = gpu::app_device();

    // Background gradient quad — first render program on the gpu:: seam.
    {
        gpu::VertexAttr attrs[] = {{ 0, gpu::VertexFormat::F32x2, 0, 0 }};
        gpu::VertexSlot slots[] = {{ 2 * sizeof(float) }};
        gpu::RenderPipelineDesc d;
        d.shaders.wgsl = bg_wgsl_src;
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
        d.shaders.wgsl = matcap_wgsl_src;
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
        d.shaders.wgsl = pick_wgsl_src;
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
        // Per-draw pick UBOs + bind groups are created lazily in pick_draw.
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
        d.shaders.wgsl = cursor_wgsl_src;
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
        d.shaders.wgsl = cursor_shadow_wgsl_src;
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
        d.shaders.wgsl = crosshair_wgsl_src;
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
        d.shaders.wgsl = screen_buf_wgsl_src;
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
        gpu::ShaderSources src; src.wgsl = screen_expand_wgsl_src; src.glsl = screen_expand_src;
        screen_expand_pipeline = gpu::create_compute_pipeline(gpu_dev, src, binds, 6);
        if (!screen_expand_pipeline.handle) std::fprintf(stderr, "[renderer] screen_expand pipeline failed\n");
        else std::printf("[compute] screen_expand pipeline compiled (gpu:: seam)\n");
        screen_expand_ubo = gpu::create_buffer(gpu_dev, nullptr, sizeof(ScreenExpandParamsGPU), gpu::Usage::Uniform);
    }

    // Debug wireframe overlay — on the gpu:: seam. Screen-space fat lines: a
    // Triangles pipeline with NO vertex buffers — the vertex stage pulls edge
    // endpoints from the mesh position SSBO (binding 0) through the edge-pair
    // SSBO (binding 1, built lazily in draw_debug_mesh) and expands each edge
    // into an antialiased ribbon (see debug_vert_src). depth_write off: the
    // feathered fringes must not occlude neighbouring lines' blends.
    {
        gpu::BindEntry binds[] = {
            { 0,  gpu::Bind::StorageRead, 0 },
            { 1,  gpu::Bind::StorageRead, 0 },
            { 63, gpu::Bind::Uniform, sizeof(DebugParamsGPU) },
        };
        gpu::RenderPipelineDesc d;
        d.shaders.wgsl = debug_wgsl_src;
        d.shaders.vert_glsl = debug_vert_src;
        d.shaders.frag_glsl = debug_edge_frag_src;
        d.binds = binds; d.bind_count = 3;
        d.topology = gpu::Topology::Triangles;
        d.depth_test = true; d.depth_write = false; d.blend = true;
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

    // Any full re-upload may carry new topology (remesh, merge, multires level,
    // valence flips) — invalidate the wireframe overlay's cached edge buffer so
    // it rebuilds against the new indices even when the tri count is unchanged.
    invalidate_debug_mesh();

    // Consume the one-shot GPU geometry source (set only between a GPU cascade
    // replay and the sync that follows it). Sizes re-checked here: a mismatch
    // means the source is not this mesh — fall back to the CPU path.
    GeometrySource src = geom_src;
    geom_src = GeometrySource{};
    const uint64_t pn_bytes  = (uint64_t)vc * 3 * sizeof(float);
    const uint64_t ebo_bytes = (uint64_t)mesh.indices.size() * sizeof(uint32_t);
    const bool gpu_geom = src.valid
        && src.vcount == vc && src.index_count == (uint32_t)mesh.indices.size()
        && src.pos  && src.pos->handle  && src.pos->size  >= pn_bytes
        && src.norm && src.norm->handle && src.norm->size >= pn_bytes
        && src.ebo  && src.ebo->handle  && src.ebo->size  == ebo_bytes;

    // Interleave SoA → AoS for VBO upload (positions/normals skipped when they
    // arrive by GPU copy below).
    std::vector<float> pos, norm, mask_buf(vc);
    std::vector<uint32_t> col_buf(vc);
    bool has_color = !mesh.color.empty();
    if (!gpu_geom) {
        pos.resize((size_t)vc * 3); norm.resize((size_t)vc * 3);
        for (uint32_t i = 0; i < vc; i++) {
            pos[i*3+0] = mesh.pos_x[i]; pos[i*3+1] = mesh.pos_y[i]; pos[i*3+2] = mesh.pos_z[i];
            norm[i*3+0] = mesh.norm_x[i]; norm[i*3+1] = mesh.norm_y[i]; norm[i*3+2] = mesh.norm_z[i];
        }
    }
    for (uint32_t i = 0; i < vc; i++) {
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
        col_buf[i] = (has_color && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    }

    // pos/norm/ebo + mask/color: owned seam buffers. pos/norm carry Vertex|Storage,
    // ebo Index|Storage, mask/color Vertex|Storage (vertex attribute + brush-written
    // storage). The renderer's own `vao` is unused at draw (the seam render pipeline
    // owns the draw-time VAO), so no vertex-attribute pointers are configured here.
    // A size change releases+recreates (new handle) → Scene::bind_active_ refreshes the
    // compute.mask_ssbo / color_ssbo aliases right after this call.
    if (gpu_geom) {
        // Residency dedupe: the working geometry is already in VRAM (cascade
        // replay output) — allocate-on-size-change like ensure_buffer, then
        // device-local copies instead of dragging ~V*24B + T*12B across the bus.
        if (!vbo_pos.handle || vbo_pos.size != pn_bytes) {
            gpu::release_buffer(vbo_pos);
            vbo_pos = gpu::create_buffer(gpu_dev, nullptr, pn_bytes,
                                         gpu::Usage::Vertex | gpu::Usage::Storage);
        }
        if (!vbo_norm.handle || vbo_norm.size != pn_bytes) {
            gpu::release_buffer(vbo_norm);
            vbo_norm = gpu::create_buffer(gpu_dev, nullptr, pn_bytes,
                                          gpu::Usage::Vertex | gpu::Usage::Storage);
        }
        if (!ebo.handle || ebo.size != ebo_bytes) {
            gpu::release_buffer(ebo);
            ebo = gpu::create_buffer(gpu_dev, nullptr, ebo_bytes,
                                     gpu::Usage::Index | gpu::Usage::Storage);
        }
        gpu::copy_buffer(gpu_dev, *src.pos,  0, vbo_pos,  0, pn_bytes);
        gpu::copy_buffer(gpu_dev, *src.norm, 0, vbo_norm, 0, pn_bytes);
        gpu::copy_buffer(gpu_dev, *src.ebo,  0, ebo,      0, ebo_bytes);
        std::printf("[residency] display pos/norm/ebo via gpu copy (%u verts)\n", vc);
    } else {
        ensure_buffer(gpu_dev, vbo_pos, pos.data(), pn_bytes,
                      gpu::Usage::Vertex | gpu::Usage::Storage);
        ensure_buffer(gpu_dev, vbo_norm, norm.data(), pn_bytes,
                      gpu::Usage::Vertex | gpu::Usage::Storage);
        ensure_buffer(gpu_dev, ebo, mesh.indices.data(), ebo_bytes,
                      gpu::Usage::Index | gpu::Usage::Storage);
    }
    ensure_buffer(gpu_dev, vbo_mask, mask_buf.data(), (uint64_t)mask_buf.size()*sizeof(float),
                  gpu::Usage::Vertex | gpu::Usage::Storage);
    ensure_buffer(gpu_dev, vbo_color, col_buf.data(), (uint64_t)col_buf.size()*sizeof(uint32_t),
                  gpu::Usage::Vertex | gpu::Usage::Storage);

    // Density field rides the same lifecycle as mask/color (compute.density_ssbo
    // aliases it); unpainted verts default to the neutral 0.5.
    std::vector<float> dens_buf(vc);
    for (uint32_t i = 0; i < vc; i++)
        dens_buf[i] = (i < mesh.density.size()) ? mesh.density[i] : 0.5f;
    ensure_buffer(gpu_dev, vbo_density, dens_buf.data(), (uint64_t)dens_buf.size()*sizeof(float),
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

    // The working VBO is sized for the current level; ids past vertex_count are
    // stale (another level's) and on WebGPU an OOB writeBuffer can kill the device.
    const uint32_t vc = mesh.vertex_count();
    if (sorted.back() >= vc) {
        size_t keep = std::lower_bound(sorted.begin(), sorted.end(), vc) - sorted.begin();
        std::printf("[paint-audit] mask upload CLAMP: %zu/%zu ids >= vcount %u (max %u)\n",
                    sorted.size() - keep, sorted.size(), vc, sorted.back());
        sorted.resize(keep);
        if (sorted.empty()) return;
    }

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

// Density field uploaders, mirror of the mask pair. Missing entries default to
// the neutral 0.5 (unpainted = uniform remesh).
void Renderer::update_density(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    std::vector<float> buf(vc);
    for (uint32_t i = 0; i < vc; i++)
        buf[i] = (i < mesh.density.size()) ? mesh.density[i] : 0.5f;
    gpu::write_buffer(gpu_dev, vbo_density, 0, buf.data(), (uint64_t)vc * sizeof(float));
}

void Renderer::update_density_verts(const Mesh& mesh, const std::vector<uint32_t>& verts) {
    if (verts.empty()) return;
    std::vector<uint32_t> sorted = verts;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    // Same OOB clamp as the mask/color uploaders: ids past the current level's
    // vertex count are stale, and an OOB writeBuffer can kill a WebGPU device.
    const uint32_t vc = mesh.vertex_count();
    if (sorted.back() >= vc) {
        size_t keep = std::lower_bound(sorted.begin(), sorted.end(), vc) - sorted.begin();
        std::printf("[paint-audit] density upload CLAMP: %zu/%zu ids >= vcount %u (max %u)\n",
                    sorted.size() - keep, sorted.size(), vc, sorted.back());
        sorted.resize(keep);
        if (sorted.empty()) return;
    }

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
            buf[k] = (v < mesh.density.size()) ? mesh.density[v] : 0.5f;
        }
        gpu::write_buffer(gpu_dev, vbo_density, (uint64_t)start * sizeof(float),
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

    // Same OOB guard as update_mask_verts — stale ids must not reach writeBuffer.
    const uint32_t vc = mesh.vertex_count();
    if (sorted.back() >= vc) {
        size_t keep = std::lower_bound(sorted.begin(), sorted.end(), vc) - sorted.begin();
        std::printf("[paint-audit] color upload CLAMP: %zu/%zu ids >= vcount %u (max %u)\n",
                    sorted.size() - keep, sorted.size(), vc, sorted.back());
        sorted.resize(keep);
        if (sorted.empty()) return;
    }

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
    gpu::RenderTarget target;
    target.width = w; target.height = h;
#if defined(CHISEL_BACKEND_WEBGPU)
    // First default-framebuffer pass of the frame: it owns the colour+depth clear
    // via loadOp=Clear (there is no separate glClear on webgpu — see backend
    // begin_render_pass, which keys both colour and depth loadOp off t.clear).
    // The gradient quad covers every pixel so the colour clear value is moot; the
    // point is the depth clear to 1.0. GL keeps its own explicit clear in main.
    target.clear = true;
#endif
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

    // view/proj are constant across the pass; cache them so each pick_draw can write
    // them (+ its own id) into its own per-draw UBO. Reset the pool cursor.
    cam.get_view_matrix(pick_view);
    cam.get_projection_matrix(pick_proj, (float)w / (float)h);
    pick_slot = 0;
}

void Renderer::pick_draw(uint32_t entity_id, const gpu::Buffer& pos_vbo, const gpu::Buffer& ebo_, uint32_t index_count) {
    if (!pos_vbo.handle || index_count == 0) return;

    // Grow the per-draw UBO/bind-group pool on demand. Distinct buffer per draw so the
    // per-entity id survives WebGPU's queue-timeline writeBuffer (see renderer.h note).
    if (pick_slot >= pick_ubos.size()) {
        pick_ubos.push_back(gpu::create_buffer(gpu_dev, nullptr, sizeof(PickParamsGPU), gpu::Usage::Uniform));
        gpu::BindBufferEntry be[] = {{ 63, &pick_ubos.back(), sizeof(PickParamsGPU) }};
        pick_bgs.push_back(gpu::create_bind_group(gpu_dev, pick_pipeline, be, 1));
    }
    gpu::Buffer& ubo = pick_ubos[pick_slot];
    gpu::write_buffer(gpu_dev, ubo, 0,   pick_view, sizeof pick_view);
    gpu::write_buffer(gpu_dev, ubo, 64,  pick_proj, sizeof pick_proj);
    gpu::write_buffer(gpu_dev, ubo, 128, &entity_id, sizeof entity_id);

    gpu::set_bind_group(pick_pass, pick_pipeline, pick_bgs[pick_slot]);
    gpu::set_vertex_buffer(pick_pass, 0, pos_vbo);
    gpu::set_index_buffer(pick_pass, ebo_);
    gpu::draw_indexed(pick_pass, index_count);
    pick_slot++;
}

void Renderer::pick_end() {
    gpu::end_render_pass(pick_pass);   // rebinds the default framebuffer
    // Pool (UBOs + bind groups) is retained for reuse across picks.
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

    // pos/norm are filled by the expand compute (Vertex|Storage); allocate them
    // empty, grow-only (contents aren't preserved — the expand kernel rewrites the
    // live tc*3 verts every call, and draws read exactly screen_tri_count*3 verts,
    // so an oversized buffer is harmless on both backends).
    uint64_t pn_bytes = (uint64_t)flat_vc * 3 * sizeof(float);
    if (screen_vbo_pos.size < pn_bytes) {
        gpu::release_buffer(screen_vbo_pos);
        screen_vbo_pos = gpu::create_buffer(gpu_dev, nullptr, pn_bytes, gpu::Usage::Vertex | gpu::Usage::Storage);
    }
    if (screen_vbo_norm.size < pn_bytes) {
        gpu::release_buffer(screen_vbo_norm);
        screen_vbo_norm = gpu::create_buffer(gpu_dev, nullptr, pn_bytes, gpu::Usage::Vertex | gpu::Usage::Storage);
    }

    // Triid and bary are a pure function of the triangle index (t,t,t / fixed
    // bary pattern) — no mesh data involved. Grow-only with tail fill: revisiting
    // a level the buffers already cover uploads nothing (at multires L8 the pair
    // is ~47 MB, previously re-sent on every level switch), and growth uploads
    // only the [screen_static_tris, tc) range.
    if (tc > screen_static_tris || !screen_vbo_triid.handle || !screen_vbo_bary.handle) {
        const uint32_t old_tc = (screen_vbo_triid.handle && screen_vbo_bary.handle)
                              ? screen_static_tris : 0u;
        const uint64_t triid_bytes = (uint64_t)flat_vc * sizeof(float);
        const uint64_t bary_bytes  = (uint64_t)flat_vc * 2 * sizeof(float);
        // A grow reallocates (no in-place resize on WebGPU) and refills from 0;
        // otherwise only the tail is new.
        const uint32_t fill_from = (screen_vbo_triid.size < triid_bytes ||
                                    screen_vbo_bary.size < bary_bytes) ? 0u : old_tc;
        if (screen_vbo_triid.size < triid_bytes) {
            gpu::release_buffer(screen_vbo_triid);
            screen_vbo_triid = gpu::create_buffer(gpu_dev, nullptr, triid_bytes, gpu::Usage::Vertex);
        }
        if (screen_vbo_bary.size < bary_bytes) {
            gpu::release_buffer(screen_vbo_bary);
            screen_vbo_bary = gpu::create_buffer(gpu_dev, nullptr, bary_bytes, gpu::Usage::Vertex);
        }
        const uint32_t fill_tc = tc - fill_from;
        std::vector<float> triid((size_t)fill_tc * 3);
        std::vector<float> bary((size_t)fill_tc * 6);
        for (uint32_t i = 0; i < fill_tc; i++) {
            const float t = (float)(fill_from + i);
            uint32_t base = i * 3;
            triid[base+0] = t; triid[base+1] = t; triid[base+2] = t;
            bary[(base+0)*2+0] = 1.0f; bary[(base+0)*2+1] = 0.0f;
            bary[(base+1)*2+0] = 0.0f; bary[(base+1)*2+1] = 1.0f;
            bary[(base+2)*2+0] = 0.0f; bary[(base+2)*2+1] = 0.0f;
        }
        gpu::write_buffer(gpu_dev, screen_vbo_triid, (uint64_t)fill_from * 3 * sizeof(float),
                          triid.data(), (uint64_t)triid.size() * sizeof(float));
        gpu::write_buffer(gpu_dev, screen_vbo_bary, (uint64_t)fill_from * 6 * sizeof(float),
                          bary.data(), (uint64_t)bary.size() * sizeof(float));
        screen_static_tris = tc;
    }

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
    // 2D grid past 65535 groups (5.2M tris at multires L9 -> 81920): WebGPU
    // hard-caps each dispatch dimension; the kernel recovers the linear index.
    uint32_t groups = (screen_tri_count + 63) / 64;
    uint32_t gx = groups, gy = 1u;
    const uint32_t MAXG = 65535u;
    if (groups > MAXG) { gx = MAXG; gy = (groups + MAXG - 1u) / MAXG; }
    gpu::dispatch(batch, screen_expand_pipeline, grp, gx, gy);
    gpu::submit(batch);   // issues the vertex-attrib + buffer-update barriers
    gpu::release_bind_group(grp);
}

void Renderer::render_screen_buffers(const Camera& cam, int w, int h) {
    create_screen_buffers(w, h);

    // Re-expand the flat soup from the live working VBO first. GPU-side writes
    // (in-place GPU undo/redo, compute kernels) update vbo_pos/vbo_norm without
    // going through update_screen_positions — rendering the stale soup here left
    // the brush picking against ghost (pre-undo) geometry.
    update_screen_mesh_gpu();

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

#if defined(CHISEL_BACKEND_WEBGPU)
    // Kick the async plane reads right behind the render (queue order guarantees the
    // copies see this pass's output even if a pick pass overwrites the FBO later).
    // poll_plane_reads() lands them; until then sample_* report not-ready.
    for (int i = 0; i < 3; ++i)
        if (plane_tk[i]) { gpu::ticket_drop(gpu_dev, plane_tk[i]); plane_tk[i] = 0; }
    plane_valid = false;
    plane_tk[0] = gpu::read_target_region_async(gpu_dev, screen_target, 0, 0, 0, w, h);
    plane_tk[1] = gpu::read_target_region_async(gpu_dev, screen_target, 1, 0, 0, w, h);
    plane_tk[2] = gpu::read_target_region_async(gpu_dev, screen_target, 2, 0, 0, w, h);
    plane_kick_w = w; plane_kick_h = h;
    plane_pending = true;
#endif
}

void Renderer::poll_plane_reads() {
#if defined(CHISEL_BACKEND_WEBGPU)
    if (!plane_pending) return;
    for (int i = 0; i < 3; ++i)
        if (!gpu::ticket_ready(gpu_dev, plane_tk[i])) return;
    size_t px = (size_t)plane_kick_w * plane_kick_h;
    plane_depth.resize(px);
    plane_norm.resize(px * 3);
    plane_triid.resize(px);
    bool ok = px > 0;
    ok &= gpu::ticket_take(gpu_dev, plane_tk[0], plane_depth.data(), px * sizeof(float));
    ok &= gpu::ticket_take(gpu_dev, plane_tk[1], plane_norm.data(),  px * 3 * sizeof(float));
    ok &= gpu::ticket_take(gpu_dev, plane_tk[2], plane_triid.data(), px * sizeof(uint32_t));
    plane_tk[0] = plane_tk[1] = plane_tk[2] = 0;
    plane_pending = false;
    plane_w = plane_kick_w; plane_h = plane_kick_h;
    plane_valid = ok;
#endif
}

bool Renderer::sample_depth(int x, int y, float* out) {
#if defined(CHISEL_BACKEND_WEBGPU)
    if (!plane_valid || x < 0 || y < 0 || x >= plane_w || y >= plane_h) return false;
    *out = plane_depth[(size_t)y * plane_w + x];
    return true;
#else
    if (x < 0 || y < 0 || x >= screen_target.width || y >= screen_target.height) return false;
    read_depth_region(x, y, 1, 1, out);
    return true;
#endif
}

bool Renderer::sample_normal(int x, int y, float out[3]) {
#if defined(CHISEL_BACKEND_WEBGPU)
    if (!plane_valid || x < 0 || y < 0 || x >= plane_w || y >= plane_h) return false;
    size_t i = ((size_t)y * plane_w + x) * 3;
    out[0] = plane_norm[i + 0]; out[1] = plane_norm[i + 1]; out[2] = plane_norm[i + 2];
    return true;
#else
    if (x < 0 || y < 0 || x >= screen_target.width || y >= screen_target.height) return false;
    read_normal_region(x, y, 1, 1, out);
    return true;
#endif
}

bool Renderer::sample_triid(int x, int y, uint32_t* out) {
#if defined(CHISEL_BACKEND_WEBGPU)
    if (!plane_valid || x < 0 || y < 0 || x >= plane_w || y >= plane_h) return false;
    *out = plane_triid[(size_t)y * plane_w + x];
    return true;
#else
    if (x < 0 || y < 0 || x >= screen_target.width || y >= screen_target.height) return false;
    read_triid_region(x, y, 1, 1, out);
    return true;
#endif
}

void Renderer::read_depth_region(int x, int y, int w, int h, float* out) {
    gpu::read_target_region(gpu_dev, screen_target, 0, x, y, w, h, out);
}

bool Renderer::sample_area_normal(int cx, int cy, int radius_px, float out[3],
                                  float out_avg[3]) {
    // Sampling radius is deliberately smaller than the dab (the caller scales it):
    // averaged over the FULL dab the plane stops following real curvature and Clay
    // starts flattening spheres. Capped so a huge brush doesn't turn the GL region
    // read into a multi-megabyte stall.
    int r = radius_px;
    if (r < 2) r = 2;
    if (r > 48) r = 48;

    int x0 = cx - r, y0 = cy - r;
    int side = 2 * r + 1;

    // Clip to the buffer, then read/index whatever survives.
    int bw, bh;
#if defined(CHISEL_BACKEND_WEBGPU)
    if (!plane_valid) return false;
    bw = plane_w; bh = plane_h;
#else
    bw = screen_target.width; bh = screen_target.height;
#endif
    if (x0 < 0) { side += x0; x0 = 0; }
    if (y0 < 0) { side += y0; y0 = 0; }
    int side_x = side, side_y = side;
    if (x0 + side_x > bw) side_x = bw - x0;
    if (y0 + side_y > bh) side_y = bh - y0;
    if (side_x <= 0 || side_y <= 0) return false;

    const float* src;
    const float* dsrc = nullptr;
#if defined(CHISEL_BACKEND_WEBGPU)
    src = nullptr;   // indexed straight out of plane_norm / plane_depth below
#else
    area_norm_scratch.resize((size_t)side_x * side_y * 3);
    read_normal_region(x0, y0, side_x, side_y, area_norm_scratch.data());
    src = area_norm_scratch.data();
    if (out_avg) {
        area_depth_scratch.resize((size_t)side_x * side_y);
        read_depth_region(x0, y0, side_x, side_y, area_depth_scratch.data());
        dsrc = area_depth_scratch.data();
    }
#endif

    float inv_r = 1.0f / (float)r;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float apx = 0.0f, apy = 0.0f, ad = 0.0f, wsum = 0.0f;
    int hits = 0;
    for (int y = 0; y < side_y; y++) {
        for (int x = 0; x < side_x; x++) {
            float dx = (float)(x0 + x - cx) * inv_r;
            float dy = (float)(y0 + y - cy) * inv_r;
            float d2 = dx * dx + dy * dy;
            if (d2 > 1.0f) continue;                 // disc, not box
            float nx, ny, nz;
#if defined(CHISEL_BACKEND_WEBGPU)
            size_t i = ((size_t)(y0 + y) * plane_w + (x0 + x)) * 3;
            nx = plane_norm[i]; ny = plane_norm[i + 1]; nz = plane_norm[i + 2];
#else
            size_t i = ((size_t)y * side_x + x) * 3;
            nx = src[i]; ny = src[i + 1]; nz = src[i + 2];
#endif
            // Background clears to zero, so length is the on-model test — no second
            // region read of the triid attachment just to reject off-mesh texels.
            if (nx * nx + ny * ny + nz * nz < 0.25f) continue;
            float w = 1.0f - d2;                     // smooth, centre-weighted
            ax += nx * w; ay += ny * w; az += nz * w;
            if (out_avg) {
                float depth;
#if defined(CHISEL_BACKEND_WEBGPU)
                depth = plane_depth[(size_t)(y0 + y) * plane_w + (x0 + x)];
#else
                depth = dsrc[(size_t)y * side_x + x];
#endif
                apx += (float)(x0 + x) * w;
                apy += (float)(y0 + y) * w;
                ad  += depth * w;
                wsum += w;
            }
            hits++;
        }
    }
    if (hits == 0) return false;
    float len = std::sqrt(ax * ax + ay * ay + az * az);
    if (len < 1e-6f) return false;                   // taps cancelled out (fold/edge)
    out[0] = ax / len; out[1] = ay / len; out[2] = az / len;
    if (out_avg) {
        if (wsum < 1e-6f) return false;
        out_avg[0] = apx / wsum; out_avg[1] = apy / wsum; out_avg[2] = ad / wsum;
    }
    return true;
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

    // Stale-cache guard: if the mesh's topology changed while the overlay stayed
    // on (multires level switch, remesh, merge), the cached edge buffer no longer
    // matches — rebuild instead of drawing garbage edges against the new verts.
    if (debug_edge_src_tris != tc) debug_edge_count = 0;

    // Build the edge-pair SSBO lazily (rebuilt after invalidate). Unique
    // undirected edges only — each interior edge is shared by two tris, and a
    // double-drawn ribbon would double-blend its AA fringe.
    if (debug_edge_count == 0) {
        std::unordered_set<uint64_t> seen;
        seen.reserve((size_t)tc * 2);
        std::vector<uint32_t> edges;
        edges.reserve((size_t)tc * 3);          // E ≈ 1.5·T for a closed mesh
        for (uint32_t i = 0; i < tc; i++) {
            for (int k = 0; k < 3; k++) {
                uint32_t a = mesh.indices[i*3+k], b = mesh.indices[i*3+(k+1)%3];
                uint64_t key = (a < b) ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a);
                if (!seen.insert(key).second) continue;
                edges.push_back(a); edges.push_back(b);
            }
        }
        debug_edge_count    = (uint32_t)edges.size();
        debug_edge_src_tris = tc;
        ensure_buffer(gpu_dev, debug_edge_vbo, edges.data(),
                      edges.size() * sizeof(uint32_t), gpu::Usage::Storage);
        // Cache the mesh extent for the zoom-adaptive width below. Positions
        // drift while sculpting but the scale doesn't move much between
        // topology rebuilds, and this is only a line-width heuristic.
        Vec3 c; mesh.compute_bounding_sphere(c, debug_mesh_radius);
    }

    DebugParamsGPU p;
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    p.viewport[0] = (float)w;
    p.viewport[1] = (float)h;
    // Zoom-adaptive width: 3px while the model fills the view, swelling up to
    // 2.5× as its projected diameter shrinks below ~600px so the wireframe
    // stays readable from afar. ppwu from the ortho proj (m00 = 2/world-width).
    float ppwu    = 0.5f * p.proj[0] * (float)w;
    float diam_px = std::max(2.0f * debug_mesh_radius * ppwu, 1.0f);
    float swell   = std::min(std::max(600.0f / diam_px, 1.0f), 2.5f);
    p.half_width_px  = 1.5f * swell;
    // NDC bias in place of glPolygonOffset: clears the surface for tilts up to
    // ~70° at this width; ~1% of a unit-radius model's depth, so hidden lines
    // stay hidden except at extreme grazing angles.
    p.depth_bias_ndc = 2.5e-4f;
    gpu::write_buffer(gpu_dev, debug_edge_ubo, 0, &p, sizeof p);

    // No vertex/index buffers — the vertex stage pulls from the two SSBOs and
    // expands each edge into a 6-vertex screen-space ribbon (AA in the frag).
    gpu::RenderTarget target;          // fbo 0 (default framebuffer), no clear
    target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, debug_edge_pipeline);
    gpu::BindBufferEntry be[] = {
        { 0,  &vbo_pos,        vbo_pos.size },
        { 1,  &debug_edge_vbo, debug_edge_vbo.size },
        { 63, &debug_edge_ubo, sizeof(DebugParamsGPU) },
    };
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, debug_edge_pipeline, be, 3);
    gpu::set_bind_group(rp, debug_edge_pipeline, grp);
    gpu::draw(rp, (debug_edge_count / 2u) * 6u);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);
}
