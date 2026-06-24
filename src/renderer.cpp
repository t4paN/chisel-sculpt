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

// Unit-ring segment count (shared by ring + footprint disc geometry, built once).
static const int CURSOR_SEGS = 64;

// Wrap a still-raw GL buffer handle as a transient gpu::Buffer view for binding
// through the seam (size is unused by the GL vertex/index bind path). Same pattern
// the compute kernels use for their GL-owned SSBOs during the staged port.
static inline gpu::Buffer buf_view(GLuint h) { gpu::Buffer b; b.handle = h; return b; }

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
static const char* pick_vert_src = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView;
uniform mat4 uProj;
out float vDepth;
void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    vDepth = -viewPos.z;  // linear depth, positive into screen
    gl_Position = uProj * viewPos;
}
)";

static const char* pick_frag_src = R"(
#version 330 core
in float vDepth;
layout(location=0) out float outDepth;
layout(location=2) out uint  outId;
uniform uint uEntityId;
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

static const char* screen_buf_vert_src = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in float aTriID;
layout(location=3) in vec2 aBary;

uniform mat4 uView;
uniform mat4 uProj;

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
#version 330 core
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

uniform uint tri_count;

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

static GLuint compile_compute_program(const char* src) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[renderer] compute compile error: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[renderer] compute link error: %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

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

// ---- Helpers ----

GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::fprintf(stderr, "Shader compile error: %s\n", log);
    }
    return s;
}

GLuint link_program(GLuint vert, GLuint frag) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vert);
    glAttachShader(p, frag);
    glLinkProgram(p);
    int ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, 512, nullptr, log);
        std::fprintf(stderr, "Program link error: %s\n", log);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return p;
}

// ---- Renderer ----

Renderer::Renderer()
    : vao(0), vbo_pos(0), vbo_norm(0), vbo_mask(0), vbo_color(0), vbo_tri_id(0), vbo_bary(0), ebo(0)
    , debug_edge_vbo(0), debug_edge_count(0)
    , screen_fbo(0), screen_depth_tex(0), screen_normal_tex(0)
    , screen_triid_tex(0), screen_bary_tex(0), screen_depth_rbo(0)
    , screen_buf_w(0), screen_buf_h(0)
    , screen_buf_program(0)
    , screen_vao(0), screen_vbo_pos(0), screen_vbo_norm(0)
    , screen_vbo_triid(0), screen_vbo_bary(0), screen_tri_count(0)
    , screen_expand_program(0)
    , initialized(false)
{}

Renderer::~Renderer() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo_pos) glDeleteBuffers(1, &vbo_pos);
    if (vbo_norm) glDeleteBuffers(1, &vbo_norm);
    if (vbo_mask) glDeleteBuffers(1, &vbo_mask);
    if (vbo_color) glDeleteBuffers(1, &vbo_color);
    if (ebo) glDeleteBuffers(1, &ebo);
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
    if (debug_edge_vbo) glDeleteBuffers(1, &debug_edge_vbo);
    if (screen_fbo) glDeleteFramebuffers(1, &screen_fbo);
    if (screen_depth_tex) glDeleteTextures(1, &screen_depth_tex);
    if (screen_normal_tex) glDeleteTextures(1, &screen_normal_tex);
    if (screen_triid_tex) glDeleteTextures(1, &screen_triid_tex);
    if (screen_bary_tex) glDeleteTextures(1, &screen_bary_tex);
    if (screen_depth_rbo) glDeleteRenderbuffers(1, &screen_depth_rbo);
    if (screen_buf_program) glDeleteProgram(screen_buf_program);
    if (screen_vao) glDeleteVertexArrays(1, &screen_vao);
    if (screen_vbo_pos) glDeleteBuffers(1, &screen_vbo_pos);
    if (screen_vbo_norm) glDeleteBuffers(1, &screen_vbo_norm);
    if (screen_vbo_triid) glDeleteBuffers(1, &screen_vbo_triid);
    if (screen_vbo_bary) glDeleteBuffers(1, &screen_vbo_bary);
    if (screen_expand_program) glDeleteProgram(screen_expand_program);
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

    // Entity-id pick shader
    pick_program = link_program(
        compile_shader(GL_VERTEX_SHADER, pick_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, pick_frag_src)
    );

    // Mesh VAO
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo_pos);
    glGenBuffers(1, &vbo_norm);
    glGenBuffers(1, &vbo_mask);
    glGenBuffers(1, &vbo_color);
    glGenBuffers(1, &ebo);

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

    // Screen buffer MRT shader
    screen_buf_program = link_program(
        compile_shader(GL_VERTEX_SHADER, screen_buf_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, screen_buf_frag_src)
    );

    // Screen buffer VAO (expanded mesh, no index buffer)
    glGenVertexArrays(1, &screen_vao);
    glGenBuffers(1, &screen_vbo_pos);
    glGenBuffers(1, &screen_vbo_norm);
    glGenBuffers(1, &screen_vbo_triid);
    glGenBuffers(1, &screen_vbo_bary);

    // Screen mesh expansion compute shader (indexed → flat triangle-soup)
    screen_expand_program = compile_compute_program(screen_expand_src);
    if (!screen_expand_program)
        std::fprintf(stderr, "[renderer] screen_expand compute shader failed\n");

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
    glGenBuffers(1, &debug_edge_vbo);  // edge index buffer, populated in draw_debug_mesh

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

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, pos.size()*sizeof(float), pos.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
    glBufferData(GL_ARRAY_BUFFER, norm.size()*sizeof(float), norm.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_mask);
    glBufferData(GL_ARRAY_BUFFER, mask_buf.size()*sizeof(float), mask_buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
    glBufferData(GL_ARRAY_BUFFER, col_buf.size()*sizeof(uint32_t), col_buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 mesh.indices.size()*sizeof(uint32_t),
                 mesh.indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void Renderer::update_mask(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    std::vector<float> mask_buf(vc);
    for (uint32_t i = 0; i < vc; i++)
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mask);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vc * sizeof(float), mask_buf.data());
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

    glBindBuffer(GL_ARRAY_BUFFER, vbo_mask);
    glBufferSubData(GL_ARRAY_BUFFER, min_idx * sizeof(float), range * sizeof(float), buf.data());
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

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferSubData(GL_ARRAY_BUFFER, offset, size, pos.data());

    glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
    glBufferSubData(GL_ARRAY_BUFFER, offset, size, norm.data());
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
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
        glBufferSubData(GL_ARRAY_BUFFER, off, sz, pos.data());
        glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
        glBufferSubData(GL_ARRAY_BUFFER, off, sz, norm.data());
        i = j;
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::update_mask_verts(const Mesh& mesh, const std::vector<uint32_t>& verts) {
    if (verts.empty()) return;

    static std::vector<uint32_t> sorted;
    sorted = verts;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    static std::vector<float> buf;
    size_t i = 0, n = sorted.size();
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mask);
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
        glBufferSubData(GL_ARRAY_BUFFER, start * sizeof(float), run * sizeof(float), buf.data());
        i = j;
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Packed RGBA8 color, mirror of the mask uploaders. Missing entries default to
// white (0xFFFFFFFF) so an unpainted vertex renders byte-identical to today.
void Renderer::update_color(const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    std::vector<uint32_t> buf(vc);
    bool has = !mesh.color.empty();
    for (uint32_t i = 0; i < vc; i++)
        buf[i] = (has && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vc * sizeof(uint32_t), buf.data());
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

    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
    glBufferSubData(GL_ARRAY_BUFFER, min_idx * sizeof(uint32_t), range * sizeof(uint32_t), buf.data());
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
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
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
        glBufferSubData(GL_ARRAY_BUFFER, start * sizeof(uint32_t), run * sizeof(uint32_t), buf.data());
        i = j;
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
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
    gpu::set_vertex_buffer(rp, 0, buf_view(vbo_pos));
    gpu::set_vertex_buffer(rp, 1, buf_view(vbo_norm));
    gpu::set_vertex_buffer(rp, 2, buf_view(vbo_mask));
    gpu::set_vertex_buffer(rp, 3, buf_view(vbo_color));
    gpu::set_index_buffer(rp, buf_view(ebo));
    gpu::draw_indexed(rp, index_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);
}

// ---- Per-entity static display buffers ----

void Renderer::upload_display(EntityGpu& g, const Mesh& mesh) {
    uint32_t vc = mesh.vertex_count();
    if (!g.vao) {
        glGenVertexArrays(1, &g.vao);
        glGenBuffers(1, &g.vbo_pos);
        glGenBuffers(1, &g.vbo_norm);
        glGenBuffers(1, &g.vbo_mask);
        glGenBuffers(1, &g.vbo_color);
        glGenBuffers(1, &g.ebo);
    }

    std::vector<float> pos(vc * 3), norm(vc * 3), mask_buf(vc);
    std::vector<uint32_t> col_buf(vc);
    bool has_color = !mesh.color.empty();
    for (uint32_t i = 0; i < vc; i++) {
        pos[i*3+0] = mesh.pos_x[i]; pos[i*3+1] = mesh.pos_y[i]; pos[i*3+2] = mesh.pos_z[i];
        norm[i*3+0] = mesh.norm_x[i]; norm[i*3+1] = mesh.norm_y[i]; norm[i*3+2] = mesh.norm_z[i];
        mask_buf[i] = (i < mesh.mask.size()) ? mesh.mask[i] : 0.0f;
        col_buf[i] = (has_color && i < mesh.color.size()) ? mesh.color[i] : 0xFFFFFFFFu;
    }

    glBindVertexArray(g.vao);

    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, pos.size()*sizeof(float), pos.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_norm);
    glBufferData(GL_ARRAY_BUFFER, norm.size()*sizeof(float), norm.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_mask);
    glBufferData(GL_ARRAY_BUFFER, mask_buf.size()*sizeof(float), mask_buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_color);
    glBufferData(GL_ARRAY_BUFFER, col_buf.size()*sizeof(uint32_t), col_buf.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size()*sizeof(uint32_t),
                 mesh.indices.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
    g.index_count = (uint32_t)mesh.indices.size();
    g.dirty = false;
}

void Renderer::free_display(EntityGpu& g) {
    if (g.vao) glDeleteVertexArrays(1, &g.vao);
    if (g.vbo_pos)     glDeleteBuffers(1, &g.vbo_pos);
    if (g.vbo_norm)    glDeleteBuffers(1, &g.vbo_norm);
    if (g.vbo_mask)    glDeleteBuffers(1, &g.vbo_mask);
    if (g.vbo_color)   glDeleteBuffers(1, &g.vbo_color);
    if (g.ebo)         glDeleteBuffers(1, &g.ebo);
    g = EntityGpu();
}

void Renderer::draw_display(const Camera& cam, EntityGpu& g, int w, int h,
                            float facing_threshold, bool selected) {
    if (!g.vao || g.index_count == 0) return;
    upload_matcap_params(cam, w, h, facing_threshold, selected);

    gpu::RenderTarget target; target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, matcap_pipeline);
    gpu::BindBufferEntry be[] = {{ 63, &matcap_ubo, sizeof(MatcapParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, matcap_pipeline, be, 1);
    gpu::set_bind_group(rp, matcap_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, buf_view(g.vbo_pos));
    gpu::set_vertex_buffer(rp, 1, buf_view(g.vbo_norm));
    gpu::set_vertex_buffer(rp, 2, buf_view(g.vbo_mask));
    gpu::set_vertex_buffer(rp, 3, buf_view(g.vbo_color));
    gpu::set_index_buffer(rp, buf_view(g.ebo));
    gpu::draw_indexed(rp, g.index_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);
}

// ---- Entity-id pick pass (reuses the screen FBO) ----

void Renderer::pick_begin(const Camera& cam, int w, int h) {
    create_screen_buffers(w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, screen_fbo);
    glViewport(0, 0, w, h);

    // Only attachment 0 (depth) and attachment 2 (id) are written by the pick
    // frag. Map output location 2 → attachment 2, leave 1 and 3 disabled.
    GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_NONE, GL_COLOR_ATTACHMENT2, GL_NONE};
    glDrawBuffers(4, bufs);

    float clear_depth = 1000.0f;
    glClearBufferfv(GL_COLOR, 0, &clear_depth);
    uint32_t clear_id = 0;            // 0 = no entity (ids start at 1)
    glClearBufferuiv(GL_COLOR, 2, &clear_id);
    glClear(GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glUseProgram(pick_program);

    float view[16], proj[16];
    cam.get_view_matrix(view);
    cam.get_projection_matrix(proj, (float)w / (float)h);
    glUniformMatrix4fv(glGetUniformLocation(pick_program, "uView"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(pick_program, "uProj"), 1, GL_FALSE, proj);
}

void Renderer::pick_draw(uint32_t entity_id, GLuint vao, uint32_t index_count) {
    if (!vao || index_count == 0) return;
    glUniform1ui(glGetUniformLocation(pick_program, "uEntityId"), entity_id);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void Renderer::pick_end() {
    // Restore the screen FBO's full MRT draw-buffer layout for the brush pass.
    GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                     GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, bufs);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
        glLineWidth(3.0f);                    // GL-only nicety (WebGPU ignores line width)
        gpu::draw(rp, 8);
        glLineWidth(1.0f);
        gpu::release_bind_group(grp);
    }

    gpu::end_render_pass(rp);
}

// ---- Screen Buffer FBO ----

void Renderer::create_screen_buffers(int w, int h) {
    if (screen_fbo) {
        // Already created, just resize
        resize_screen_buffers(w, h);
        return;
    }

    screen_buf_w = w;
    screen_buf_h = h;

    glGenFramebuffers(1, &screen_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, screen_fbo);

    // Attachment 0: linear depth (R32F)
    glGenTextures(1, &screen_depth_tex);
    glBindTexture(GL_TEXTURE_2D, screen_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, screen_depth_tex, 0);

    // Attachment 1: view-space normals (RGB16F)
    glGenTextures(1, &screen_normal_tex);
    glBindTexture(GL_TEXTURE_2D, screen_normal_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, screen_normal_tex, 0);

    // Attachment 2: triangle ID (R32UI)
    glGenTextures(1, &screen_triid_tex);
    glBindTexture(GL_TEXTURE_2D, screen_triid_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, screen_triid_tex, 0);

    // Attachment 3: barycentrics (RG16F)
    glGenTextures(1, &screen_bary_tex);
    glBindTexture(GL_TEXTURE_2D, screen_bary_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, w, h, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, screen_bary_tex, 0);

    // Depth/stencil renderbuffer for actual z-testing
    glGenRenderbuffers(1, &screen_depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, screen_depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, screen_depth_rbo);

    GLenum draw_bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                          GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, draw_bufs);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "Screen FBO incomplete: 0x%x\n", status);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::resize_screen_buffers(int w, int h) {
    if (w == screen_buf_w && h == screen_buf_h) return;
    screen_buf_w = w;
    screen_buf_h = h;

    glBindTexture(GL_TEXTURE_2D, screen_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, screen_normal_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, screen_triid_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    glBindTexture(GL_TEXTURE_2D, screen_bary_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, w, h, 0, GL_RG, GL_FLOAT, nullptr);

    glBindRenderbuffer(GL_RENDERBUFFER, screen_depth_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
}

void Renderer::upload_screen_mesh(const Mesh& mesh) {
    uint32_t tc = mesh.tri_count();
    screen_tri_count = tc;
    uint32_t flat_vc = tc * 3;

    // Allocate pos/norm buffers (compute shader will fill them)
    glBindVertexArray(screen_vao);

    glBindBuffer(GL_ARRAY_BUFFER, screen_vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, (size_t)flat_vc * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, screen_vbo_norm);
    glBufferData(GL_ARRAY_BUFFER, (size_t)flat_vc * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Triid and bary are static per topology — build on CPU
    std::vector<float> triid(flat_vc);
    std::vector<float> bary(flat_vc * 2);
    for (uint32_t t = 0; t < tc; t++) {
        uint32_t base = t * 3;
        triid[base+0] = (float)t; triid[base+1] = (float)t; triid[base+2] = (float)t;
        bary[(base+0)*2+0] = 1.0f; bary[(base+0)*2+1] = 0.0f;
        bary[(base+1)*2+0] = 0.0f; bary[(base+1)*2+1] = 1.0f;
        bary[(base+2)*2+0] = 0.0f; bary[(base+2)*2+1] = 0.0f;
    }

    glBindBuffer(GL_ARRAY_BUFFER, screen_vbo_triid);
    glBufferData(GL_ARRAY_BUFFER, triid.size()*sizeof(float), triid.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, screen_vbo_bary);
    glBufferData(GL_ARRAY_BUFFER, bary.size()*sizeof(float), bary.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);

    // Fill positions/normals via GPU expansion
    update_screen_mesh_gpu();
}

void Renderer::update_screen_positions(const Mesh& mesh) {
    (void)mesh;
    update_screen_mesh_gpu();
}

void Renderer::update_screen_mesh_gpu() {
    if (!screen_expand_program || screen_tri_count == 0) return;

    glUseProgram(screen_expand_program);
    glUniform1ui(glGetUniformLocation(screen_expand_program, "tri_count"), screen_tri_count);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vbo_pos);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo_norm);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, screen_vbo_pos);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, screen_vbo_norm);

    uint32_t groups = (screen_tri_count + 63) / 64;
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    glUseProgram(0);
}

void Renderer::render_screen_buffers(const Camera& cam, int w, int h) {
    create_screen_buffers(w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, screen_fbo);
    glViewport(0, 0, w, h);

    // Clear: depth to max, triid to 0xFFFFFFFF (no triangle), bary to 0
    float clear_depth = 1000.0f;
    glClearBufferfv(GL_COLOR, 0, &clear_depth);
    float clear_normal[] = {0, 0, 0, 0};
    glClearBufferfv(GL_COLOR, 1, clear_normal);
    uint32_t clear_triid = 0xFFFFFFFF;
    glClearBufferuiv(GL_COLOR, 2, &clear_triid);
    float clear_bary[] = {0, 0};
    glClearBufferfv(GL_COLOR, 3, clear_bary);
    glClear(GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glUseProgram(screen_buf_program);

    float view[16], proj[16];
    cam.get_view_matrix(view);
    cam.get_projection_matrix(proj, (float)w / (float)h);

    glUniformMatrix4fv(glGetUniformLocation(screen_buf_program, "uView"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(screen_buf_program, "uProj"), 1, GL_FALSE, proj);

    glBindVertexArray(screen_vao);
    glDrawArrays(GL_TRIANGLES, 0, screen_tri_count * 3);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
}

void Renderer::read_depth_region(int x, int y, int w, int h, float* out) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, screen_fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(x, screen_buf_h - y - h, w, h, GL_RED, GL_FLOAT, out);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Renderer::read_normal_region(int x, int y, int w, int h, float* out) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, screen_fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(x, screen_buf_h - y - h, w, h, GL_RGB, GL_FLOAT, out);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Renderer::read_triid_region(int x, int y, int w, int h, uint32_t* out) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, screen_fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(x, screen_buf_h - y - h, w, h, GL_RED_INTEGER, GL_UNSIGNED_INT, out);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Renderer::read_bary_region(int x, int y, int w, int h, float* out) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, screen_fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(x, screen_buf_h - y - h, w, h, GL_RG, GL_FLOAT, out);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
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
        glBindBuffer(GL_ARRAY_BUFFER, debug_edge_vbo);
        glBufferData(GL_ARRAY_BUFFER, edges.size() * sizeof(uint32_t), edges.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    DebugParamsGPU p;
    cam.get_view_matrix(p.view);
    cam.get_projection_matrix(p.proj, (float)w / (float)h);
    gpu::write_buffer(gpu_dev, debug_edge_ubo, 0, &p, sizeof p);

    // Depth-compare LEQUAL + polygon offset pull the wires just in front of the
    // surface (anti z-fight). The seam doesn't model depth-func / depth-bias yet,
    // so these stay raw GL around the seam draw (they fold into RenderPipelineDesc
    // when the WebGPU render backend lands). set_pipeline enables depth test + blend
    // and leaves depth-func untouched, so restore GL_LESS afterwards.
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1.0f, -1.0f);
    glLineWidth(1.0f);

    gpu::RenderTarget target;          // fbo 0 (default framebuffer), no clear
    target.width = w; target.height = h;
    gpu::RenderPass rp = gpu::begin_render_pass(gpu_dev, target);
    gpu::set_pipeline(rp, debug_edge_pipeline);
    gpu::BindBufferEntry be[] = {{ 63, &debug_edge_ubo, sizeof(DebugParamsGPU) }};
    gpu::BindGroup grp = gpu::create_bind_group(gpu_dev, debug_edge_pipeline, be, 1);
    gpu::set_bind_group(rp, debug_edge_pipeline, grp);
    gpu::set_vertex_buffer(rp, 0, buf_view(vbo_pos));
    gpu::set_index_buffer(rp, buf_view(debug_edge_vbo));
    gpu::draw_indexed(rp, debug_edge_count);
    gpu::end_render_pass(rp);
    gpu::release_bind_group(grp);

    glDisable(GL_POLYGON_OFFSET_LINE);
    glDepthFunc(GL_LESS);
}
