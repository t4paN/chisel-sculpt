#include "renderer.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

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

static const char* matcap_vert_src = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in float aMask;
layout(location=4) in vec4 aColor;   // RGBA8 normalized -> [0,1]; white = unpainted

uniform mat4 uView;
uniform mat4 uProj;

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
#version 330 core
in vec3 vNormView;
in float vMask;
in vec4 vColor;
out vec4 fragColor;
uniform float uFacingThreshold;
// Per-draw object tint: 0 = selected/active (no tint), 1 = deselected (muted).
uniform float uObjMask;
// Vertex-paint visibility: 1 = show albedo, 0 = ignore (plain matcap).
uniform float uPaintVisible;

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

static const char* cursor_vert_src = R"(
#version 330 core
layout(location=0) in vec2 aLocal;   // unit circle (cos, sin)
layout(location=1) in float aOuter;  // 0 = inner rim vertex, 1 = outer
uniform vec2 uCenter;
uniform float uRadius;
uniform vec2 uScreenSize;
uniform vec3 uNormal;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamFwd;      // points from camera into scene
uniform float uBaseThick;  // pixels; half-thickness back side / half front ≈ 1.8x
uniform float uFrontBoost; // extra thickness on front (pixels)

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

static const char* cursor_frag_src = R"(
#version 330 core
in float vFront;
uniform vec4 uColor;
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
static const char* crosshair_vert_src = R"(
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec2 uCenter;
uniform vec2 uScreenSize;
void main() {
    vec2 screen = uCenter + aPos;
    vec2 ndc = (screen / uScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* crosshair_frag_src = R"(
#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }
)";

// ---- Brush shadow (screen-space filled disc = actual sculpt footprint) ----
static const char* cursor_shadow_vert_src = R"(
#version 330 core
layout(location=0) in vec2 aPos;    // (0,0) for fan center, unit circle for rim
uniform vec2 uCenter;
uniform float uRadius;
uniform vec2 uScreenSize;
out float vDist;
void main() {
    vDist = length(aPos);  // 0 = center, 1 = rim
    vec2 screen = uCenter + aPos * uRadius;
    vec2 ndc = (screen / uScreenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char* cursor_shadow_frag_src = R"(
#version 330 core
in float vDist;
uniform vec4 uColor;
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

static const char* debug_vert_src = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView;
uniform mat4 uProj;
void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)";

static const char* debug_vert_frag_src = R"(
#version 330 core
out vec4 fragColor;
void main() {
    fragColor = vec4(0.2, 1.0, 1.0, 1.0);
}
)";

static const char* debug_edge_frag_src = R"(
#version 330 core
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
    , matcap_program(0), bg_program(0), bg_vao(0)
    , cursor_program(0), cursor_shadow_program(0), crosshair_program(0)
    , debug_vert_program(0), debug_edge_program(0)
    , debug_vert_vao(0), debug_edge_vao(0), debug_edge_vbo(0), debug_edge_count(0)
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
    if (bg_vao) glDeleteVertexArrays(1, &bg_vao);
    if (matcap_program) glDeleteProgram(matcap_program);
    if (bg_program) glDeleteProgram(bg_program);
    if (cursor_program) glDeleteProgram(cursor_program);
    if (cursor_shadow_program) glDeleteProgram(cursor_shadow_program);
    if (crosshair_program) glDeleteProgram(crosshair_program);
    if (debug_vert_program) glDeleteProgram(debug_vert_program);
    if (debug_edge_program) glDeleteProgram(debug_edge_program);
    if (debug_vert_vao) glDeleteVertexArrays(1, &debug_vert_vao);
    if (debug_edge_vao) glDeleteVertexArrays(1, &debug_edge_vao);
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
    // Background quad
    bg_program = link_program(
        compile_shader(GL_VERTEX_SHADER, bg_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, bg_frag_src)
    );

    float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint bg_vbo;
    glGenVertexArrays(1, &bg_vao);
    glGenBuffers(1, &bg_vbo);
    glBindVertexArray(bg_vao);
    glBindBuffer(GL_ARRAY_BUFFER, bg_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // Matcap shader
    matcap_program = link_program(
        compile_shader(GL_VERTEX_SHADER, matcap_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, matcap_frag_src)
    );

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

    // Cursor shader
    cursor_program = link_program(
        compile_shader(GL_VERTEX_SHADER, cursor_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, cursor_frag_src)
    );
    cursor_shadow_program = link_program(
        compile_shader(GL_VERTEX_SHADER, cursor_shadow_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, cursor_shadow_frag_src)
    );
    crosshair_program = link_program(
        compile_shader(GL_VERTEX_SHADER, crosshair_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, crosshair_frag_src)
    );

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

    // Debug visualization shaders
    debug_vert_program = link_program(
        compile_shader(GL_VERTEX_SHADER, debug_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, debug_vert_frag_src)
    );
    debug_edge_program = link_program(
        compile_shader(GL_VERTEX_SHADER, debug_vert_src),
        compile_shader(GL_FRAGMENT_SHADER, debug_edge_frag_src)
    );

    // Debug VAOs (will be populated in draw_debug_mesh)
    glGenVertexArrays(1, &debug_vert_vao);
    glGenVertexArrays(1, &debug_edge_vao);
    glGenBuffers(1, &debug_edge_vbo);

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
    glDisable(GL_DEPTH_TEST);
    glUseProgram(bg_program);
    glBindVertexArray(bg_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Renderer::draw_mesh(const Camera& cam, int w, int h, uint32_t index_count,
                          float facing_threshold, bool selected) {
    glEnable(GL_DEPTH_TEST);
    glUseProgram(matcap_program);

    float view[16], proj[16];
    cam.get_view_matrix(view);
    cam.get_projection_matrix(proj, (float)w / (float)h);

    glUniformMatrix4fv(glGetUniformLocation(matcap_program, "uView"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(matcap_program, "uProj"), 1, GL_FALSE, proj);
    glUniform1f(glGetUniformLocation(matcap_program, "uFacingThreshold"), facing_threshold);
    glUniform1f(glGetUniformLocation(matcap_program, "uObjMask"), selected ? 0.0f : 1.0f);
    glUniform1f(glGetUniformLocation(matcap_program, "uPaintVisible"), paint_visible);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
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
    glEnable(GL_DEPTH_TEST);
    glUseProgram(matcap_program);

    float view[16], proj[16];
    cam.get_view_matrix(view);
    cam.get_projection_matrix(proj, (float)w / (float)h);

    glUniformMatrix4fv(glGetUniformLocation(matcap_program, "uView"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(matcap_program, "uProj"), 1, GL_FALSE, proj);
    glUniform1f(glGetUniformLocation(matcap_program, "uFacingThreshold"), facing_threshold);
    glUniform1f(glGetUniformLocation(matcap_program, "uObjMask"), selected ? 0.0f : 1.0f);
    glUniform1f(glGetUniformLocation(matcap_program, "uPaintVisible"), paint_visible);

    glBindVertexArray(g.vao);
    glDrawElements(GL_TRIANGLES, g.index_count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
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
    // Triangle-strip ring: SEGS+1 pairs of (inner, outer) verts closing the loop.
    const int SEGS = 64;
    float verts[(SEGS + 1) * 2 * 3];
    int k = 0;
    for (int i = 0; i <= SEGS; i++) {
        float a = 2.0f * 3.14159265f * (float)i / (float)SEGS;
        float cs = std::cos(a), sn = std::sin(a);
        // Inner then outer so triangle strip winds consistently.
        verts[k++] = cs; verts[k++] = sn; verts[k++] = 0.0f;
        verts[k++] = cs; verts[k++] = sn; verts[k++] = 1.0f;
    }

    GLuint cvao, cvbo;
    glGenVertexArrays(1, &cvao);
    glGenBuffers(1, &cvbo);
    glBindVertexArray(cvao);
    glBindBuffer(GL_ARRAY_BUFFER, cvbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ---- Shadow disc: actual screen-space brush footprint ----
    // Triangle fan: center + (SEGS+1) rim verts closing the loop.
    float disc_verts[(SEGS + 2) * 2];
    disc_verts[0] = 0.0f; disc_verts[1] = 0.0f;
    for (int i = 0; i <= SEGS; i++) {
        float a = 2.0f * 3.14159265f * (float)i / (float)SEGS;
        disc_verts[2 + i*2 + 0] = std::cos(a);
        disc_verts[2 + i*2 + 1] = std::sin(a);
    }
    GLuint dvao, dvbo;
    glGenVertexArrays(1, &dvao);
    glGenBuffers(1, &dvbo);
    glBindVertexArray(dvao);
    glBindBuffer(GL_ARRAY_BUFFER, dvbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(disc_verts), disc_verts, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

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

    // Draw shadow first (under the ring). Tinted with brush color, muted.
    glUseProgram(cursor_shadow_program);
    glUniform2f(glGetUniformLocation(cursor_shadow_program, "uScreenSize"), (float)w, (float)h);
    glUniform2f(glGetUniformLocation(cursor_shadow_program, "uCenter"), cx, cy);
    glUniform1f(glGetUniformLocation(cursor_shadow_program, "uRadius"), radius);
    glUniform4f(glGetUniformLocation(cursor_shadow_program, "uColor"),
                cr * 0.5f, cg * 0.5f, cb * 0.5f, 0.22f);
    glDrawArrays(GL_TRIANGLE_FAN, 0, SEGS + 2);

    // Camera basis in world space (matches brush.cpp apply_move derivation)
    Vec3 cam_pos = cam.get_position();
    Vec3 cam_fwd = (cam.target - cam_pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 cam_right = cam_fwd.cross(world_up).normalized();
    Vec3 cam_up = cam_right.cross(cam_fwd).normalized();

    // ---- Ring geometry (triangle strip) ----
    glBindVertexArray(cvao);
    glUseProgram(cursor_program);
    glUniform2f(glGetUniformLocation(cursor_program, "uScreenSize"), (float)w, (float)h);
    glUniform3f(glGetUniformLocation(cursor_program, "uNormal"), nx, ny, nz);
    glUniform3f(glGetUniformLocation(cursor_program, "uCamRight"), cam_right.x, cam_right.y, cam_right.z);
    glUniform3f(glGetUniformLocation(cursor_program, "uCamUp"),    cam_up.x,    cam_up.y,    cam_up.z);
    glUniform3f(glGetUniformLocation(cursor_program, "uCamFwd"),   cam_fwd.x,   cam_fwd.y,   cam_fwd.z);
    glUniform2f(glGetUniformLocation(cursor_program, "uCenter"), cx, cy);
    glUniform4f(glGetUniformLocation(cursor_program, "uColor"), cr, cg, cb, ca);
    glUniform1f(glGetUniformLocation(cursor_program, "uRadius"), radius);
    glUniform1f(glGetUniformLocation(cursor_program, "uBaseThick"), 2.5f);
    glUniform1f(glGetUniformLocation(cursor_program, "uFrontBoost"), 3.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (SEGS + 1) * 2);

    // Optional inner hardness ring — same 3D treatment, thinner.
    if (hh > 0.5f) {
        float inner_a = (hh - 0.5f) * 2.0f * ca;
        float inner_r = std::max(3.0f, radius - 5.0f);
        glUniform4f(glGetUniformLocation(cursor_program, "uColor"), cr, cg, cb, inner_a);
        glUniform1f(glGetUniformLocation(cursor_program, "uRadius"), inner_r);
        glUniform1f(glGetUniformLocation(cursor_program, "uBaseThick"), 1.5f);
        glUniform1f(glGetUniformLocation(cursor_program, "uFrontBoost"), 1.5f);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (SEGS + 1) * 2);
    }

    // ---- Crosshair (X at center, on-model only) ----
    if (on_model) {
        const float gap = 3.0f, arm = 8.0f;
        float xh[] = {
            -arm, -arm,  -gap, -gap,
             gap,  gap,   arm,  arm,
             arm, -arm,   gap, -gap,
            -gap,  gap,  -arm,  arm,
        };
        GLuint xvao, xvbo;
        glGenVertexArrays(1, &xvao);
        glGenBuffers(1, &xvbo);
        glBindVertexArray(xvao);
        glBindBuffer(GL_ARRAY_BUFFER, xvbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(xh), xh, GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glUseProgram(crosshair_program);
        glUniform2f(glGetUniformLocation(crosshair_program, "uCenter"), cx, cy);
        glUniform2f(glGetUniformLocation(crosshair_program, "uScreenSize"), (float)w, (float)h);
        glUniform4f(glGetUniformLocation(crosshair_program, "uColor"), cr * 1.4f, cg * 1.4f, cb * 1.4f, 1.0f);
        glLineWidth(3.0f);
        glDrawArrays(GL_LINES, 0, 8);
        glLineWidth(1.0f);
        glDeleteBuffers(1, &xvbo);
        glDeleteVertexArrays(1, &xvao);
    }

    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glDeleteBuffers(1, &cvbo);
    glDeleteVertexArrays(1, &cvao);
    glDeleteBuffers(1, &dvbo);
    glDeleteVertexArrays(1, &dvao);
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

    float view[16], proj[16];
    cam.get_view_matrix(view);
    cam.get_projection_matrix(proj, (float)w / (float)h);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1.0f, -1.0f);

    // Build and draw edges
    if (debug_edge_count == 0) {
        std::vector<uint32_t> edges;
        for (uint32_t i = 0; i < tc; i++) {
            uint32_t i0 = mesh.indices[i*3+0];
            uint32_t i1 = mesh.indices[i*3+1];
            uint32_t i2 = mesh.indices[i*3+2];
            edges.push_back(i0);
            edges.push_back(i1);
            edges.push_back(i1);
            edges.push_back(i2);
            edges.push_back(i2);
            edges.push_back(i0);
        }
        debug_edge_count = (uint32_t)edges.size();

        glBindBuffer(GL_ARRAY_BUFFER, debug_edge_vbo);
        glBufferData(GL_ARRAY_BUFFER, edges.size() * sizeof(uint32_t), edges.data(), GL_STATIC_DRAW);
    }

    glBindVertexArray(debug_edge_vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glLineWidth(1.0f);
    glUseProgram(debug_edge_program);
    glUniformMatrix4fv(glGetUniformLocation(debug_edge_program, "uView"), 1, GL_FALSE, view);
    glUniformMatrix4fv(glGetUniformLocation(debug_edge_program, "uProj"), 1, GL_FALSE, proj);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, debug_edge_vbo);
    glDrawElements(GL_LINES, debug_edge_count, GL_UNSIGNED_INT, nullptr);

    glDisable(GL_POLYGON_OFFSET_LINE);
    glDepthFunc(GL_LESS);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}
