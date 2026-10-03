#include "ui_skin.h"
#include "ui_icons.h"
#include "ui_backdrop.h"
#include "input.h"
#include "brush_alpha.h"
#include "imgui.h"
#include "imgui_internal.h"   // ShadeVertsLinearColorGradientKeepAlpha
#include "ui_fonts_generated.h"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

// Layout, tokens and motion follow the handoff's board F (default skin) and G
// (material dial + springs) — HANDOFF.md sections 5 and 6. Sizes are in window
// pixels, like the rest of the UI (DisplayFramebufferScale is pinned to 1).

namespace {

// ---- Tokens ------------------------------------------------------------------
// Set per skin by set_theme(): the modern values below, or the CGA ones.
ImU32 kText, kTextRail, kMuted, kDim, kStatus, kAccent, kAccentHi, kAccentLo, kTint,
      kTintText, kBrush, kUnsaved, kAlert, kDivider, kDividerLo, kBorder, kTipBg, kBadge;

// CGA 16-colour palette (same values as `namespace cga` in ui_overlay.cpp).
namespace cga {
constexpr ImU32 black = IM_COL32(0x00, 0x00, 0x00, 255), blue = IM_COL32(0x00, 0x00, 0xAA, 255);
constexpr ImU32 light_gray = IM_COL32(0xAA, 0xAA, 0xAA, 255), dark_gray = IM_COL32(0x55, 0x55, 0x55, 255);
constexpr ImU32 light_green = IM_COL32(0x55, 0xFF, 0x55, 255), light_cyan = IM_COL32(0x55, 0xFF, 0xFF, 255);
constexpr ImU32 light_red = IM_COL32(0xFF, 0x55, 0x55, 255), light_magenta = IM_COL32(0xFF, 0x55, 0xFF, 255);
constexpr ImU32 yellow = IM_COL32(0xFF, 0xFF, 0x55, 255), white = IM_COL32(0xFF, 0xFF, 0xFF, 255);
}

bool g_dos = false;   // the DOS skin is active (theme + primitives switch on it)
bool g_y2k = false;   // the Y2K skin is active (own layout; shares popups, toasts, menu)

// Y2K palette (board E).
namespace y2k {
constexpr ImU32 text = IM_COL32(0xE6, 0xE6, 0xE6, 255), muted = IM_COL32(0x9C, 0x9C, 0xA2, 255);
constexpr ImU32 label = IM_COL32(0xBD, 0xBD, 0xC2, 255), edge = IM_COL32(0x14, 0x14, 0x16, 255);
constexpr ImU32 well = IM_COL32(0x1D, 0x1D, 0x20, 255), orange = IM_COL32(0xF2, 0x8C, 0x28, 255);
constexpr ImU32 orange_hi = IM_COL32(0xFF, 0xC2, 0x66, 255), orange_lo = IM_COL32(0xD4, 0x6F, 0x0E, 255);
constexpr ImU32 amber = IM_COL32(0xFF, 0xB3, 0x47, 255), value = IM_COL32(0xFF, 0xD0, 0x8A, 255);
constexpr ImU32 on_text = IM_COL32(0x2A, 0x15, 0x00, 255);
}

void set_theme(bool dos, bool y2k = false) {
    g_dos = dos;
    g_y2k = y2k;
    if (y2k) {
        // Only what shared code (toasts, slider HUD, popups, alpha picker) reads;
        // the shelves paint with namespace y2k directly.
        kText = kTextRail = y2k::text;
        kMuted = kStatus = kDim = y2k::muted;
        kAccent = kAccentHi = kAccentLo = y2k::orange;
        kTint = IM_COL32(0xF2, 0x8C, 0x28, 110);
        kTintText = y2k::amber;
        kBrush = y2k::orange;
        kUnsaved = y2k::amber;
        kAlert = IM_COL32(0xFF, 0x4A, 0x3D, 255);
        kDivider = kDividerLo = IM_COL32(0x55, 0x55, 0x5A, 255);
        kBorder = y2k::edge;
        kTipBg = IM_COL32(0xFF, 0xFF, 0xE1, 255);
        kBadge = IM_COL32(0x80, 0x80, 0x80, 255);
        return;
    }
    if (!dos) {
        kText      = IM_COL32(0xEC, 0xEA, 0xF2, 255);
        kTextRail  = IM_COL32(0xDA, 0xD7, 0xE3, 255);
        kMuted     = IM_COL32(0xA9, 0xA6, 0xB6, 255);
        kDim       = IM_COL32(0x9C, 0x99, 0xAA, 255);
        kStatus    = IM_COL32(0xC9, 0xC6, 0xD4, 255);
        kAccent    = IM_COL32(0x76, 0x50, 0xE0, 255);
        kAccentHi  = IM_COL32(0x8A, 0x68, 0xF0, 255);
        kAccentLo  = IM_COL32(0x6B, 0x46, 0xD8, 255);
        kTint      = IM_COL32(118, 80, 224, 82);     // toggles that are on
        kTintText  = IM_COL32(0xE2, 0xD9, 0xFF, 255);
        kBrush     = IM_COL32(0xF5, 0x9E, 0x3B, 255);
        kUnsaved   = IM_COL32(0xF5, 0xB2, 0x6B, 255);
        kAlert     = IM_COL32(0xFF, 0x8A, 0x80, 255);
        kDivider   = IM_COL32(255, 255, 255, 31);
        kDividerLo = IM_COL32(255, 255, 255, 20);
        kBorder    = IM_COL32(255, 255, 255, 20);
        kTipBg     = IM_COL32(0x0E, 0x0E, 0x11, 245);
        kBadge     = IM_COL32(255, 255, 255, 46);
    } else {
        // Board D: yellow for names and headings, light cyan for the active tool and
        // key letters, light green for numbers, light magenta for mirror, light red for
        // unsaved, light gray for frames, blue + white border for tooltips.
        kText = kTextRail = cga::white;
        kMuted = kStatus = cga::light_gray;
        kDim = cga::dark_gray;
        kAccent = kAccentHi = kAccentLo = cga::light_cyan;
        kTint = cga::light_magenta;
        kTintText = cga::light_magenta;
        kBrush = cga::yellow;
        kUnsaved = kAlert = cga::light_red;
        kDivider = kDividerLo = cga::dark_gray;
        kBorder = cga::light_gray;
        kTipBg = cga::blue;
        kBadge = cga::light_gray;
    }
}

constexpr float kEdge      = 16.0f;   // screen margin
constexpr float kRailW     = 56.0f;
constexpr float kRailBtn   = 46.0f;
constexpr float kBarBtn    = 44.0f;

// Corner radius: the modern value, or square in DOS.
float rad(float r) { return g_dos ? 0.0f : r; }

ImU32 with_alpha(ImU32 c, float a) {
    int base = (int)((c >> IM_COL32_A_SHIFT) & 0xFF);
    int na = (int)(base * std::max(0.0f, std::min(1.0f, a)) + 0.5f);
    return (c & ~IM_COL32_A_MASK) | ((ImU32)na << IM_COL32_A_SHIFT);
}

// ---- Fonts -----------------------------------------------------------------
ImFont* g_classic_font = nullptr;
ImFont* g_bitmap_font = nullptr;     // ChiselBitmap.ttf, Plex Mono merged for the rest
ImFont* g_fonts[4] = {nullptr, nullptr, nullptr, nullptr};
ImFont* g_y2k_font = nullptr;        // Verdana / DejaVu Sans / Plex Sans, see load
ImFont* g_y2k_bold = nullptr;
InputState::UiSkin g_active_skin = InputState::UiSkin::CLASSIC;
bool    g_style_ready = false;
ImGuiStyle g_classic_style;

bool bold(UiFont f) { return f == UiFont::SansSemibold || f == UiFont::MonoMedium; }

ImFont* F(UiFont f) {
    if (g_dos) return g_bitmap_font;
    if (g_y2k) return bold(f) ? g_y2k_bold : g_y2k_font;
    return g_fonts[(int)f];
}

// The bitmap font only stays crisp at whole multiples of its 8 px cell, so in DOS
// every size lands on 8 (badges, corner keys) or 16 (everything else). Verdana runs
// big and wide, so Y2K sets the shared code's sizes a notch smaller (board E: 11 px).
float px_for(float px) {
    if (g_dos) return px < 11.0f ? 8.0f : 16.0f;
    if (g_y2k) return std::max(9.0f, px - 2.0f);
    return px;
}

ImVec2 text_size(UiFont f, float px, const char* s) {
    ImFont* font = F(f);
    if (!font) return ImVec2(0, 0);
    ImVec2 sz = font->CalcTextSizeA(px_for(px), FLT_MAX, 0.0f, s);
    if (g_dos && bold(f)) sz.x += 2.0f;
    return sz;
}

void text_at(ImDrawList* dl, UiFont f, float px, ImVec2 pos, ImU32 col, const char* s) {
    ImVec2 p(std::floor(pos.x + 0.5f), std::floor(pos.y + 0.5f));
    dl->AddText(F(f), px_for(px), p, col, s);
    // No bold weight in the bitmap face: the HUD's trick, drawn twice 2 px apart.
    if (g_dos && bold(f)) dl->AddText(F(f), px_for(px), ImVec2(p.x + 2.0f, p.y), col, s);
}

// Names read as titles in DOS (board D: "DRAW", "CLAY").
const char* title_case(const char* s) {
    if (!g_dos) return s;
    static char buf[64];
    size_t i = 0;
    for (; s[i] && i < sizeof buf - 1; i++) buf[i] = (char)std::toupper((unsigned char)s[i]);
    buf[i] = '\0';
    return buf;
}

ImU32 title_col() { return g_dos ? cga::yellow : kText; }

// ---- Motion ----------------------------------------------------------------
bool os_reduced_motion() {
    static double next_check = -1.0;
    static bool reduced = false;
    double now = ImGui::GetTime();
    if (now < next_check) return reduced;
    next_check = now + 2.0;                // a settings flip shows up within 2 s
#if defined(_WIN32)
    BOOL anim = TRUE;
    if (SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, &anim, 0)) reduced = !anim;
#elif defined(__EMSCRIPTEN__)
    reduced = EM_ASM_INT({
        return (window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches) ? 1 : 0;
    }) != 0;
#endif
    return reduced;
}

bool g_motion = true;   // resolved once per frame in draw_modern_ui

struct Spring { float x = 0.0f, v = 0.0f; bool init = false; };
std::unordered_map<ImGuiID, Spring> g_springs;

// Slightly under-damped spring toward `target`: the press squash and its overshoot
// on release (the board's cubic-bezier(.34,1.56,.64,1)), the menu pop and the
// readout bars. With motion off it snaps.
float spring(ImGuiID id, float target, float stiffness = 520.0f, float damping_ratio = 0.52f) {
    Spring& s = g_springs[id];
    if (!s.init || !g_motion) { s.x = target; s.v = 0.0f; s.init = true; return target; }
    float dt = std::min(ImGui::GetIO().DeltaTime, 1.0f / 30.0f);
    const float c = 2.0f * std::sqrt(stiffness) * damping_ratio;
    const int steps = 4;
    float h = dt / steps;
    for (int i = 0; i < steps; i++) {
        float a = -stiffness * (s.x - target) - c * s.v;
        s.v += a * h;
        s.x += s.v * h;
    }
    return s.x;
}
void spring_reset(ImGuiID id, float x) { Spring& s = g_springs[id]; s.x = x; s.v = 0; s.init = true; }

// ---- Material ----------------------------------------------------------------
struct Material { float m, fill_a, rim, sheen; };
Material g_mat{0.45f, 0.62f, 0.19f, 0.066f};

Material material_from(const InputState& in) {
    float m = std::max(0.0f, std::min(1.0f, in.ui_material / 100.0f));
    return Material{m, 0.32f + 0.66f * m, 0.07f + 0.22f * (1.0f - m), 0.12f * (1.0f - m)};
}

// One panel: soft drop shadow, the blurred backdrop, the tint, a top sheen fading
// out by 45 % of the height, a hairline border and a brighter top rim. Order
// matters: the backdrop is opaque, so it hides the part of the shadow beneath the
// panel exactly as CSS box-shadow would.
void panel_bg(ImDrawList* dl, ImVec2 a, ImVec2 b, float r) {
    const Material& mt = g_mat;
    if (b.x - a.x < 2.0f || b.y - a.y < 2.0f) return;
    if (g_y2k) {
        // A loose shelf: vertical brushed-gray gradient, dark edge, lit top line,
        // short soft shadow.
        r = std::min(r, 5.0f);
        dl->AddRectFilled(ImVec2(a.x + 1, a.y + 3), ImVec2(b.x + 3, b.y + 4), IM_COL32(0, 0, 0, 90), r);
        int v0 = dl->VtxBuffer.Size;
        dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
        ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, a, ImVec2(a.x, b.y),
                                                      IM_COL32(0x4E, 0x4E, 0x53, 255),
                                                      IM_COL32(0x38, 0x38, 0x3C, 255));
        dl->AddRect(a, b, y2k::edge, r);
        dl->AddLine(ImVec2(a.x + r, a.y + 1.5f), ImVec2(b.x - r, a.y + 1.5f), IM_COL32(255, 255, 255, 40));
        return;
    }
    if (g_dos) {
        // DOS frame: black box, 2 px light-gray border, hard 8 px drop shadow.
        a = ImVec2(std::floor(a.x), std::floor(a.y));
        b = ImVec2(std::floor(b.x), std::floor(b.y));
        dl->AddRectFilled(ImVec2(a.x + 8, a.y + 8), ImVec2(b.x + 8, b.y + 8), IM_COL32(0, 0, 0, 140));
        dl->AddRectFilled(a, b, cga::black);
        dl->AddRect(ImVec2(a.x + 1, a.y + 1), ImVec2(b.x - 1, b.y - 1), kBorder, 0.0f, 0, 2.0f);
        return;
    }
    for (int i = 1; i <= 4; i++) {
        float e = (float)i * 3.0f;
        dl->AddRectFilled(ImVec2(a.x - e, a.y - e + 6.0f), ImVec2(b.x + e, b.y + e + 6.0f),
                          IM_COL32(0, 0, 0, 13), r + e);
    }
    if (ui_backdrop::available()) {
        ImVec2 uv0, uv1;
        ui_backdrop::uv_for(a, b, &uv0, &uv1);
        dl->AddImageRounded(ui_backdrop::texture(), a, b, uv0, uv1, IM_COL32_WHITE, r);
    }
    dl->AddRectFilled(a, b, IM_COL32(20, 20, 24, (int)(mt.fill_a * 255.0f + 0.5f)), r);
    if (mt.sheen > 0.004f) {
        int v0 = dl->VtxBuffer.Size;
        dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
        int v1 = dl->VtxBuffer.Size;
        float span = (b.y - a.y) * 0.45f;
        for (int i = v0; i < v1; i++) {
            ImDrawVert& v = dl->VtxBuffer[i];
            float t = std::max(0.0f, std::min(1.0f, (v.pos.y - a.y) / span));
            int base = (int)((v.col >> IM_COL32_A_SHIFT) & 0xFF);      // AA fringe < 255
            int al = (int)(base * mt.sheen * (1.0f - t) + 0.5f);
            v.col = (v.col & ~IM_COL32_A_MASK) | ((ImU32)al << IM_COL32_A_SHIFT);
        }
    }
    dl->AddRect(a, b, kBorder, r, 0, 1.0f);
    // Rim: the border again, brighter, clipped to the top edge.
    dl->PushClipRect(a, ImVec2(b.x, a.y + std::max(2.0f, r * 0.6f)), true);
    dl->AddRect(a, b, IM_COL32(255, 255, 255, (int)(mt.rim * 255.0f)), r, 0, 1.0f);
    dl->PopClipRect();
}

// Paint the panel under the current window (pos/size are last frame's for an
// auto-resizing window — the usual one-frame lag the islands already live with).
void window_panel(float r) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetWindowPos(), s = ImGui::GetWindowSize();
    dl->PushClipRectFullScreen();
    panel_bg(dl, p, ImVec2(p.x + s.x, p.y + s.y), r);
    dl->PopClipRect();
}

const ImGuiWindowFlags kPanelFlags =
    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus;

bool begin_panel(const char* name, ImVec2 pos, ImVec2 pivot, ImVec2 pad, float gap, float r) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, pivot);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, pad);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
    bool open = ImGui::Begin(name, nullptr, kPanelFlags);
    window_panel(r);
    return open;
}
void end_panel() { ImGui::End(); ImGui::PopStyleVar(2); }

// ---- Tooltip -----------------------------------------------------------------
enum class TipSide { Right, Below, Left };

// Name + key badge on the near-black chip from board A. Foreground draw list so it
// never fights the panels for z-order, and immediate (the app runs a 0 hover delay).
void tooltip(ImVec2 anchor_min, ImVec2 anchor_max, TipSide side, const char* name, const char* key) {
    if (!name) return;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float fs = 13.0f, ks = 11.0f, padx = 10.0f, pady = 7.0f;
    ImVec2 ns = text_size(UiFont::Sans, fs, name);
    ImVec2 kz = key ? text_size(UiFont::Mono, ks, key) : ImVec2(0, 0);
    float badge_w = key ? kz.x + 10.0f : 0.0f;
    float w = padx * 2 + ns.x + (key ? 10.0f + badge_w : 0.0f);
    float h = pady * 2 + std::max(ns.y, kz.y + 2.0f);
    ImVec2 p;
    if (side == TipSide::Right)
        p = ImVec2(anchor_max.x + 10.0f, (anchor_min.y + anchor_max.y) * 0.5f - h * 0.5f);
    else if (side == TipSide::Left)
        p = ImVec2(anchor_min.x - 10.0f - w, (anchor_min.y + anchor_max.y) * 0.5f - h * 0.5f);
    else
        p = ImVec2((anchor_min.x + anchor_max.x) * 0.5f - w * 0.5f, anchor_max.y + 10.0f);
    ImVec2 disp = ImGui::GetIO().DisplaySize;
    p.x = std::max(8.0f, std::min(p.x, disp.x - w - 8.0f));
    p.y = std::max(8.0f, std::min(p.y, disp.y - h - 8.0f));
    if (g_y2k) {
        // The period tooltip: pale yellow, 1 px black frame, black text, key in gray.
        p = ImVec2(std::floor(p.x), std::floor(p.y));
        ImVec2 n2 = text_size(UiFont::Sans, 13.0f, name);
        ImVec2 k2 = key ? text_size(UiFont::Sans, 13.0f, key) : ImVec2(0, 0);
        float W = 6.0f * 2 + n2.x + (key ? 8.0f + k2.x : 0.0f), Hh = 4.0f * 2 + n2.y;
        if (side == TipSide::Left) p.x = std::floor(anchor_min.x - 8.0f - W);
        else if (side == TipSide::Right) p.y = std::floor((anchor_min.y + anchor_max.y - Hh) * 0.5f);
        dl->AddRectFilled(ImVec2(p.x + 2, p.y + 2), ImVec2(p.x + W + 2, p.y + Hh + 2), IM_COL32(0, 0, 0, 80));
        dl->AddRectFilled(p, ImVec2(p.x + W, p.y + Hh), kTipBg);
        dl->AddRect(p, ImVec2(p.x + W, p.y + Hh), IM_COL32(0, 0, 0, 255));
        text_at(dl, UiFont::Sans, 13.0f, ImVec2(p.x + 6, p.y + 4), IM_COL32(0, 0, 0, 255), name);
        if (key) text_at(dl, UiFont::Sans, 13.0f, ImVec2(p.x + 6 + n2.x + 8, p.y + 4),
                         IM_COL32(0x60, 0x60, 0x60, 255), key);
        return;
    }
    if (g_dos) {
        // Turbo-style help box: blue, white frame, hard shadow, yellow bold name,
        // light-cyan key.
        p = ImVec2(std::floor(p.x), std::floor(p.y));
        const char* nm = title_case(name);
        ImVec2 n2 = text_size(UiFont::SansSemibold, 16.0f, nm);
        ImVec2 k2 = key ? text_size(UiFont::Mono, 16.0f, key) : ImVec2(0, 0);
        float W = 12.0f * 2 + n2.x + (key ? 16.0f + k2.x : 0.0f), Hh = 6.0f * 2 + 16.0f + 4.0f;
        dl->AddRectFilled(ImVec2(p.x + 8, p.y + 8), ImVec2(p.x + W + 8, p.y + Hh + 8), IM_COL32(0, 0, 0, 140));
        dl->AddRectFilled(p, ImVec2(p.x + W, p.y + Hh), cga::blue);
        dl->AddRect(ImVec2(p.x + 1, p.y + 1), ImVec2(p.x + W - 1, p.y + Hh - 1), cga::white, 0.0f, 0, 2.0f);
        text_at(dl, UiFont::SansSemibold, 16.0f, ImVec2(p.x + 12, p.y + 8), cga::yellow, nm);
        if (key) text_at(dl, UiFont::Mono, 16.0f, ImVec2(p.x + 12 + n2.x + 16, p.y + 8), cga::light_cyan, key);
        return;
    }
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), kTipBg, 8.0f);
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), IM_COL32(255, 255, 255, 31), 8.0f);
    text_at(dl, UiFont::Sans, fs, ImVec2(p.x + padx, p.y + (h - ns.y) * 0.5f), kText, name);
    if (key) {
        ImVec2 b0(p.x + padx + ns.x + 10.0f, p.y + (h - kz.y - 2.0f) * 0.5f);
        ImVec2 b1(b0.x + badge_w, b0.y + kz.y + 2.0f);
        dl->AddRect(b0, b1, kBadge, 4.0f);
        text_at(dl, UiFont::Mono, ks, ImVec2(b0.x + 5.0f, b0.y + 1.0f), kStatus, key);
    }
}

// ---- Buttons -----------------------------------------------------------------
enum class Look { Rail, Bar };

struct BtnOpts {
    Look look = Look::Bar;
    bool on = false;            // Rail: the accent tile. Bar: the tinted toggle.
    bool enabled = true;
    const char* tip = nullptr;
    const char* key = nullptr;  // tooltip badge
    const char* corner = nullptr;  // small key label in the tile's corner (rail)
    TipSide side = TipSide::Below;
    float icon_px = 20.0f;
};

// Square icon button. Press squashes it to 0.92 with an inset shade and it springs
// back with a little overshoot. Returns true on click.
bool icon_button(const char* id, Icon icon, float size, const BtnOpts& o) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::PushID(id);
    if (!o.enabled) ImGui::BeginDisabled();
    bool clicked = ImGui::InvisibleButton("##b", ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    bool held    = ImGui::IsItemActive();
    ImGuiID sid  = ImGui::GetItemID();
    if (!o.enabled) ImGui::EndDisabled();
    ImGui::PopID();

    float s = spring(sid, held ? 0.92f : 1.0f, held ? 1400.0f : 520.0f);
    ImVec2 c(pos.x + size * 0.5f, pos.y + size * 0.5f);
    float hs = size * 0.5f * s;
    ImVec2 a(c.x - hs, c.y - hs), b(c.x + hs, c.y + hs);
    float r = (o.look == Look::Rail ? 11.0f : 10.0f) * s;

    if (g_dos) {
        // Solid CGA blocks: light cyan = active tool, light magenta = toggle on,
        // blue while pressed, dark gray on hover. Icons and corner keys go black on
        // a lit block.
        ImU32 bg = 0, icol = cga::white, kcol = cga::light_gray;
        if (held && o.enabled)  bg = cga::blue;
        else if (o.on)          { bg = o.look == Look::Rail ? cga::light_cyan : cga::light_magenta;
                                  icol = kcol = cga::black; }
        else if (hovered && o.enabled) bg = cga::dark_gray;
        if (bg) dl->AddRectFilled(ImVec2(pos.x, pos.y), ImVec2(pos.x + size, pos.y + size), bg);
        if (!o.enabled) icol = cga::dark_gray;
        draw_pixel_icon(dl, icon, c, 2.0f, icol);
        if (o.corner) {
            ImVec2 ks = text_size(UiFont::Mono, 8.0f, o.corner);
            text_at(dl, UiFont::Mono, 8.0f, ImVec2(pos.x + size - 2.0f - ks.x, pos.y + size - 2.0f - ks.y),
                    kcol, o.corner);
        }
        if (hovered && o.tip) tooltip(ImVec2(pos.x, pos.y), ImVec2(pos.x + size, pos.y + size),
                                      o.side, o.tip, o.key);
        return clicked && o.enabled;
    }

    ImU32 icol;
    if (o.on && o.look == Look::Rail) {
        // Raised accent tile: soft glow, top-lit gradient, highlight line.
        for (int i = 1; i <= 3; i++) {
            float e = (float)i * 2.0f;
            dl->AddRectFilled(ImVec2(a.x - e, a.y - e + 2.0f), ImVec2(b.x + e, b.y + e + 2.0f),
                              with_alpha(kAccent, 0.10f), r + e);
        }
        int v0 = dl->VtxBuffer.Size;
        dl->AddRectFilled(a, b, kAccent, r);
        ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, a,
                                                      ImVec2(a.x, b.y), kAccentHi, kAccentLo);
        dl->AddLine(ImVec2(a.x + r, a.y + 1.0f), ImVec2(b.x - r, a.y + 1.0f),
                    IM_COL32(255, 255, 255, 89), 1.0f);
        icol = IM_COL32_WHITE;
    } else if (o.on) {
        dl->AddRectFilled(a, b, kTint, r);
        icol = kTintText;
    } else {
        if (hovered && o.enabled) dl->AddRectFilled(a, b, IM_COL32(255, 255, 255, 18), r);
        icol = o.look == Look::Rail ? kTextRail : kText;
    }
    if (held) {
        dl->AddRectFilled(a, b, IM_COL32(0, 0, 0, 46), r);
        dl->AddRectFilled(a, ImVec2(b.x, a.y + 3.0f), IM_COL32(0, 0, 0, 40), r,
                          ImDrawFlags_RoundCornersTop);
    }
    if (!o.enabled) icol = with_alpha(icol, 0.35f);
    draw_icon(dl, icon, c, o.icon_px * s, icol);
    if (o.corner) {
        ImVec2 ks = text_size(UiFont::Mono, 9.0f, o.corner);
        text_at(dl, UiFont::Mono, 9.0f, ImVec2(b.x - 4.0f - ks.x, b.y - 3.0f - ks.y),
                with_alpha(icol, 0.7f), o.corner);
    }
    if (hovered && o.tip) tooltip(ImVec2(pos.x, pos.y), ImVec2(pos.x + size, pos.y + size),
                                  o.side, o.tip, o.key);
    return clicked && o.enabled;
}

// Plain-text toggle chip (paint target, skin choice): tinted when on.
bool chip(const char* label, bool on, float h = 30.0f) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 ts = text_size(UiFont::Sans, 13.0f, label);
    ImVec2 size(ts.x + 24.0f, h);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(label, size);
    bool hovered = ImGui::IsItemHovered();
    bool held = ImGui::IsItemActive();
    float s = spring(ImGui::GetItemID(), held ? 0.94f : 1.0f, held ? 1400.0f : 520.0f);
    ImVec2 c(pos.x + size.x * 0.5f, pos.y + size.y * 0.5f);
    ImVec2 a(c.x - size.x * 0.5f * s, c.y - size.y * 0.5f * s);
    ImVec2 b(c.x + size.x * 0.5f * s, c.y + size.y * 0.5f * s);
    if (g_dos) {
        if (on) dl->AddRectFilled(a, b, cga::light_cyan);
        else if (hovered) dl->AddRectFilled(a, b, cga::dark_gray);
        dl->AddRect(ImVec2(a.x + 1, a.y + 1), ImVec2(b.x - 1, b.y - 1), on ? cga::light_cyan : cga::light_gray,
                    0.0f, 0, 2.0f);
        text_at(dl, UiFont::Sans, 16.0f, ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f),
                on ? cga::black : cga::white, label);
        return clicked;
    }
    if (on) dl->AddRectFilled(a, b, kTint, 8.0f);
    else if (hovered) dl->AddRectFilled(a, b, IM_COL32(255, 255, 255, 18), 8.0f);
    dl->AddRect(a, b, on ? IM_COL32(118, 80, 224, 140) : kBorder, 8.0f);
    text_at(dl, UiFont::Sans, 13.0f * s, ImVec2(c.x - ts.x * 0.5f * s, c.y - ts.y * 0.5f * s),
            on ? kTintText : kText, label);
    return clicked;
}

void vdivider(float h = 22.0f, float margin = 4.0f) {
    ImGui::SameLine(0.0f, margin);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float bh = kBarBtn;
    const float t = g_dos ? 2.0f : 1.0f;      // DOS rules are 2 px, like its frames
    if (g_dos) h = 28.0f;
    ImGui::Dummy(ImVec2(t, bh));
    float x = std::floor(p.x) + t * 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x, p.y + (bh - h) * 0.5f),
                                        ImVec2(x, p.y + (bh + h) * 0.5f), kDivider, t);
    ImGui::SameLine(0.0f, margin);
}

void hdivider(float width, float margin, ImU32 col, float h) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, h));
    const float t = g_dos ? 2.0f : 1.0f;
    if (g_dos) margin = std::min(margin, 4.0f);
    float y = std::floor(p.y + h * 0.5f) + t * 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x + margin, y), ImVec2(p.x + width - margin, y), col, t);
}

// ---- Formatting ------------------------------------------------------------
void fmt_k(char* out, size_t n, uint32_t v) {
    if (v < 1000)            std::snprintf(out, n, "%u", v);
    else if (v < 1000000)    std::snprintf(out, n, "%.1fk", v / 1000.0);
    else                     std::snprintf(out, n, "%.2fM", v / 1000000.0);
}
void fmt_thousands(char* out, size_t n, uint32_t v) {
    char raw[16];
    std::snprintf(raw, sizeof raw, "%u", v);
    size_t len = std::strlen(raw), o = 0;
    for (size_t i = 0; i < len && o + 2 < n; i++) {
        out[o++] = raw[i];
        size_t left = len - i - 1;
        if (left > 0 && left % 3 == 0) out[o++] = ',';
    }
    out[o] = '\0';
}

// File name for the top-left panel: basename without .chisel, or "Untitled".
void project_name(char* out, size_t n, const char* path) {
    if (!path || !*path) { std::snprintf(out, n, "Untitled"); return; }
    const char* base = path;
    for (const char* p = path; *p; p++) if (*p == '/' || *p == '\\') base = p + 1;
    size_t len = std::strlen(base);
    if (len > 7 && std::strcmp(base + len - 7, ".chisel") == 0) len -= 7;
    const size_t max_chars = 28;
    if (len > max_chars) std::snprintf(out, n, "%.*s...", (int)(max_chars - 3), base);
    else                 std::snprintf(out, n, "%.*s", (int)len, base);
}

// ---- Alpha swatch ------------------------------------------------------------
void draw_alpha_preview(ImDrawList* dl, ImVec2 a, float size, const float* preview) {
    dl->AddRectFilled(a, ImVec2(a.x + size, a.y + size), g_dos ? cga::black : IM_COL32(0x0E, 0x0E, 0x11, 255),
                      rad(7.0f));
    dl->AddRect(a, ImVec2(a.x + size, a.y + size), g_dos ? cga::light_gray : IM_COL32(255, 255, 255, 46),
                rad(7.0f), 0, g_dos ? 2.0f : 1.0f);
    if (!preview) return;
    float inset = size * 0.18f, cell = (size - inset * 2.0f) / 16.0f;
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            float v = preview[y * 16 + x];
            if (v <= 0.02f) continue;
            int g = (int)(v * 242.0f + 0.5f);
            dl->AddRectFilled(ImVec2(a.x + inset + x * cell, a.y + inset + y * cell),
                              ImVec2(a.x + inset + (x + 1) * cell, a.y + inset + (y + 1) * cell),
                              IM_COL32(g, g, (int)(g * 1.02f > 255 ? 255 : g * 1.02f), 255));
        }
}

// ---- Popups ------------------------------------------------------------------
// Popups get the panel material and pop in on the same spring as the buttons.
// Returns true if open (caller must end_skin_popup). `slide` drifts the window
// into place: only for popups whose position is set with ImGuiCond_Always every
// frame, or the offset would accumulate.
bool begin_skin_popup(const char* id, ImVec2 pad, bool slide = true) {
    ImGuiID sid = ImGui::GetID(id);
    bool open = ImGui::IsPopupOpen(id);
    if (!open) spring_reset(sid, 0.0f);
    float t = open ? spring(sid, 1.0f, 300.0f, 0.55f) : 0.0f;
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, pad);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, std::max(0.0f, std::min(1.0f, t * 1.4f)));
    // Popups open from inside the tight-packed panels; give them the normal rhythm back.
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 7.0f));
    bool vis = ImGui::BeginPopup(id);
    if (vis) {
        // The pop: drift down into place. Shift the whole window, not just its
        // background, so content and panel move together.
        float dy = (1.0f - t) * -8.0f;
        if (slide && std::fabs(dy) > 0.05f) {
            ImGuiWindow* w = ImGui::GetCurrentWindow();
            ImGui::SetWindowPos(ImVec2(w->Pos.x, w->Pos.y + dy), ImGuiCond_Always);
        }
        ui_skin_popup_background();
    } else {
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);
    }
    return vis;
}
void end_skin_popup() {
    ImGui::EndPopup();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
}

// ---- Style -------------------------------------------------------------------
void apply_modern_style() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 12.0f; s.PopupRounding = 12.0f; s.ChildRounding = 8.0f;
    s.FrameRounding = 6.0f; s.GrabRounding = 6.0f; s.TabRounding = 6.0f;
    s.ScrollbarRounding = 6.0f;
    s.FramePadding = ImVec2(8.0f, 5.0f);
    s.ItemSpacing = ImVec2(8.0f, 8.0f);
    s.ItemInnerSpacing = ImVec2(8.0f, 6.0f);
    s.GrabMinSize = 12.0f;
    s.PopupBorderSize = 1.0f;
    s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 0.0f;
    s.TabBorderSize = 0.0f;
    ImVec4* c = s.Colors;
    auto rgba = [](int r, int g, int b, float a) { return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a); };
    c[ImGuiCol_Text]              = rgba(0xEC, 0xEA, 0xF2, 1.0f);
    c[ImGuiCol_TextDisabled]      = rgba(0xA9, 0xA6, 0xB6, 1.0f);
    c[ImGuiCol_WindowBg]          = rgba(18, 18, 22, 0.97f);
    c[ImGuiCol_PopupBg]           = rgba(14, 14, 17, 0.97f);
    c[ImGuiCol_Border]            = rgba(255, 255, 255, 0.10f);
    c[ImGuiCol_FrameBg]           = rgba(255, 255, 255, 0.09f);
    c[ImGuiCol_FrameBgHovered]    = rgba(255, 255, 255, 0.13f);
    c[ImGuiCol_FrameBgActive]     = rgba(255, 255, 255, 0.17f);
    c[ImGuiCol_TitleBg]           = rgba(18, 18, 22, 1.0f);
    c[ImGuiCol_TitleBgActive]     = rgba(30, 28, 38, 1.0f);
    c[ImGuiCol_CheckMark]         = rgba(0xE2, 0xD9, 0xFF, 1.0f);
    c[ImGuiCol_SliderGrab]        = rgba(0x8A, 0x68, 0xF0, 1.0f);
    c[ImGuiCol_SliderGrabActive]  = rgba(0xA6, 0x8B, 0xFF, 1.0f);
    c[ImGuiCol_Button]            = rgba(255, 255, 255, 0.06f);
    c[ImGuiCol_ButtonHovered]     = rgba(255, 255, 255, 0.11f);
    c[ImGuiCol_ButtonActive]      = rgba(118, 80, 224, 0.55f);
    c[ImGuiCol_Header]            = rgba(118, 80, 224, 0.30f);
    c[ImGuiCol_HeaderHovered]     = rgba(255, 255, 255, 0.08f);
    c[ImGuiCol_HeaderActive]      = rgba(118, 80, 224, 0.45f);
    c[ImGuiCol_Separator]         = rgba(255, 255, 255, 0.12f);
    c[ImGuiCol_Tab]               = rgba(255, 255, 255, 0.04f);
    c[ImGuiCol_TabHovered]        = rgba(255, 255, 255, 0.10f);
    c[ImGuiCol_TabSelected]       = rgba(118, 80, 224, 0.40f);
    c[ImGuiCol_TabSelectedOverline] = rgba(0x8A, 0x68, 0xF0, 1.0f);
    c[ImGuiCol_ScrollbarBg]       = rgba(0, 0, 0, 0.0f);
    c[ImGuiCol_ScrollbarGrab]     = rgba(255, 255, 255, 0.14f);
    c[ImGuiCol_ModalWindowDimBg]  = rgba(8, 8, 10, 0.55f);
}

// ImGui widgets in Y2K: brushed-gray windows, inset dark fields, orange marks and
// grabs, small radii.
void apply_y2k_style() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = 4.0f; s.PopupRounding = 4.0f; s.ChildRounding = 3.0f;
    s.FrameRounding = 3.0f; s.GrabRounding = 3.0f; s.TabRounding = 3.0f; s.ScrollbarRounding = 3.0f;
    s.FramePadding = ImVec2(6.0f, 3.0f);
    s.ItemSpacing = ImVec2(6.0f, 5.0f);
    s.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    s.GrabMinSize = 12.0f;
    s.PopupBorderSize = s.WindowBorderSize = 1.0f;
    s.FrameBorderSize = 1.0f;
    s.TabBorderSize = 0.0f;
    ImVec4* c = s.Colors;
    auto hex = [](unsigned v, float a = 1.0f) {
        return ImVec4(((v >> 16) & 0xFF) / 255.0f, ((v >> 8) & 0xFF) / 255.0f, (v & 0xFF) / 255.0f, a);
    };
    c[ImGuiCol_Text]              = hex(0xE6E6E6);
    c[ImGuiCol_TextDisabled]      = hex(0x9C9CA2);
    c[ImGuiCol_WindowBg]          = hex(0x3B3B3F);
    c[ImGuiCol_PopupBg]           = hex(0x3B3B3F, 0.98f);
    c[ImGuiCol_Border]            = hex(0x141416);
    c[ImGuiCol_BorderShadow]      = hex(0xFFFFFF, 0.06f);
    c[ImGuiCol_FrameBg]           = hex(0x1D1D20);
    c[ImGuiCol_FrameBgHovered]    = hex(0x26262A);
    c[ImGuiCol_FrameBgActive]     = hex(0x2C2C31);
    c[ImGuiCol_TitleBg]           = hex(0x2F2F33);
    c[ImGuiCol_TitleBgActive]     = hex(0x3F3F43);
    c[ImGuiCol_CheckMark]         = hex(0xF28C28);
    c[ImGuiCol_SliderGrab]        = hex(0xF28C28);
    c[ImGuiCol_SliderGrabActive]  = hex(0xFFC266);
    c[ImGuiCol_Button]            = hex(0x55555A);
    c[ImGuiCol_ButtonHovered]     = hex(0x6C6C71);
    c[ImGuiCol_ButtonActive]      = hex(0xF28C28);
    c[ImGuiCol_Header]            = hex(0xF28C28, 0.55f);
    c[ImGuiCol_HeaderHovered]     = hex(0x6C6C71);
    c[ImGuiCol_HeaderActive]      = hex(0xF28C28, 0.75f);
    c[ImGuiCol_Separator]         = hex(0x141416);
    c[ImGuiCol_Tab]               = hex(0x2F2F33);
    c[ImGuiCol_TabHovered]        = hex(0x6C6C71);
    c[ImGuiCol_TabSelected]       = hex(0x55555A);
    c[ImGuiCol_TabSelectedOverline] = hex(0xF28C28);
    c[ImGuiCol_ScrollbarBg]       = hex(0x1D1D20);
    c[ImGuiCol_ScrollbarGrab]     = hex(0x6C6C71);
    c[ImGuiCol_TextSelectedBg]    = hex(0xF28C28, 0.45f);
    c[ImGuiCol_ModalWindowDimBg]  = hex(0x000000, 0.45f);
}

// ImGui widgets (menu, dialogs, file browser) in CGA: square everything, black
// fields on blue frames, light cyan for marks and grabs, yellow for headings.
void apply_dos_style() {
    ImGuiStyle& s = ImGui::GetStyle();
    s.WindowRounding = s.PopupRounding = s.ChildRounding = 0.0f;
    s.FrameRounding = s.GrabRounding = s.TabRounding = s.ScrollbarRounding = 0.0f;
    s.FramePadding = ImVec2(8.0f, 4.0f);
    s.ItemSpacing = ImVec2(8.0f, 8.0f);
    s.ItemInnerSpacing = ImVec2(8.0f, 6.0f);
    s.GrabMinSize = 16.0f;
    s.PopupBorderSize = s.WindowBorderSize = 2.0f;
    s.FrameBorderSize = 0.0f;
    s.TabBorderSize = 0.0f;
    s.AntiAliasedLines = false;          // 2 px rules stay hard-edged
    s.AntiAliasedFill = false;
    ImVec4* c = s.Colors;
    auto hex = [](unsigned v, float a = 1.0f) {
        return ImVec4(((v >> 16) & 0xFF) / 255.0f, ((v >> 8) & 0xFF) / 255.0f, (v & 0xFF) / 255.0f, a);
    };
    for (int i = 0; i < ImGuiCol_COUNT; i++) c[i] = hex(0x000000);
    c[ImGuiCol_Text]              = hex(0xFFFFFF);
    c[ImGuiCol_TextDisabled]      = hex(0xAAAAAA);
    c[ImGuiCol_WindowBg]          = hex(0x000000);
    c[ImGuiCol_ChildBg]           = hex(0x000000, 0.0f);
    c[ImGuiCol_PopupBg]           = hex(0x0000AA);
    c[ImGuiCol_Border]            = hex(0xAAAAAA);
    c[ImGuiCol_BorderShadow]      = hex(0x000000, 0.0f);
    c[ImGuiCol_FrameBg]           = hex(0x0000AA);
    c[ImGuiCol_FrameBgHovered]    = hex(0x5555FF);
    c[ImGuiCol_FrameBgActive]     = hex(0x5555FF);
    c[ImGuiCol_TitleBg]           = hex(0x0000AA);
    c[ImGuiCol_TitleBgActive]     = hex(0x0000AA);
    c[ImGuiCol_TitleBgCollapsed]  = hex(0x0000AA);
    c[ImGuiCol_MenuBarBg]         = hex(0xAAAAAA);
    c[ImGuiCol_ScrollbarBg]       = hex(0x000000);
    c[ImGuiCol_ScrollbarGrab]     = hex(0x55FFFF);
    c[ImGuiCol_ScrollbarGrabHovered] = hex(0xFFFFFF);
    c[ImGuiCol_ScrollbarGrabActive]  = hex(0xFFFF55);
    c[ImGuiCol_CheckMark]         = hex(0x55FFFF);
    c[ImGuiCol_SliderGrab]        = hex(0x55FFFF);
    c[ImGuiCol_SliderGrabActive]  = hex(0xFFFF55);
    c[ImGuiCol_Button]            = hex(0x0000AA);
    c[ImGuiCol_ButtonHovered]     = hex(0x5555FF);
    c[ImGuiCol_ButtonActive]      = hex(0x00AAAA);
    c[ImGuiCol_Header]            = hex(0x00AAAA);
    c[ImGuiCol_HeaderHovered]     = hex(0x555555);
    c[ImGuiCol_HeaderActive]      = hex(0x00AAAA);
    c[ImGuiCol_Separator]         = hex(0x555555);
    c[ImGuiCol_SeparatorHovered]  = hex(0xAAAAAA);
    c[ImGuiCol_SeparatorActive]   = hex(0xFFFFFF);
    c[ImGuiCol_ResizeGrip]        = hex(0x555555);
    c[ImGuiCol_ResizeGripHovered] = hex(0xAAAAAA);
    c[ImGuiCol_ResizeGripActive]  = hex(0xFFFFFF);
    c[ImGuiCol_InputTextCursor]   = hex(0xFFFF55);
    c[ImGuiCol_Tab]               = hex(0x000000);
    c[ImGuiCol_TabHovered]        = hex(0x555555);
    c[ImGuiCol_TabSelected]       = hex(0x0000AA);
    c[ImGuiCol_TabSelectedOverline] = hex(0x55FFFF);
    c[ImGuiCol_TabDimmed]         = hex(0x000000);
    c[ImGuiCol_TabDimmedSelected] = hex(0x0000AA);
    c[ImGuiCol_PlotLines]         = hex(0x55FF55);
    c[ImGuiCol_PlotHistogram]     = hex(0x55FF55);
    c[ImGuiCol_TableHeaderBg]     = hex(0x0000AA);
    c[ImGuiCol_TableBorderStrong] = hex(0xAAAAAA);
    c[ImGuiCol_TableBorderLight]  = hex(0x555555);
    c[ImGuiCol_TextSelectedBg]    = hex(0x00AAAA);
    c[ImGuiCol_TextLink]          = hex(0x55FFFF);
    c[ImGuiCol_DragDropTarget]    = hex(0xFFFF55);
    c[ImGuiCol_NavCursor]         = hex(0xFFFF55);
    c[ImGuiCol_NavWindowingHighlight] = hex(0xFFFFFF);
    c[ImGuiCol_NavWindowingDimBg] = hex(0x000000, 0.5f);
    c[ImGuiCol_ModalWindowDimBg]  = hex(0x000000, 0.5f);
}

// ---- Panels ------------------------------------------------------------------
// The alpha (stamp) picker: built-ins + loaded images, then "+" to load one.
// Opens beside whatever button armed "##alphapick" (the modern rail, the Y2K shelf).
void alpha_picker_popup(InputState& input, const AlphaLibrary* alpha_lib, ImVec2 at) {
    ImGui::SetNextWindowPos(at, ImGuiCond_Always);
    if (begin_skin_popup("##alphapick", ImVec2(12, 12))) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 6));
        text_at(ImGui::GetWindowDrawList(), UiFont::SansSemibold, 11.0f,
                ImGui::GetCursorScreenPos(), g_dos ? cga::yellow : kMuted, "BRUSH ALPHA");
        ImGui::Dummy(ImVec2(0, g_dos ? 18.0f : 14.0f));
        int n = alpha_lib->count();
        for (int i = 0; i < n; i++) {
            if (i % 5) ImGui::SameLine();
            const AlphaEntry& e = alpha_lib->get(i);
            ImVec2 p = ImGui::GetCursorScreenPos();
            char id[24]; std::snprintf(id, sizeof id, "##al%d", i);
            if (ImGui::InvisibleButton(id, ImVec2(44, 44))) {
                input.active_alpha = i;
                ImGui::CloseCurrentPopup();
            }
            bool hv = ImGui::IsItemHovered();
            ImDrawList* pdl = ImGui::GetWindowDrawList();
            if (input.active_alpha == i)
                pdl->AddRectFilled(p, ImVec2(p.x + 44, p.y + 44), kAccent, rad(10.0f));
            else if (hv)
                pdl->AddRectFilled(p, ImVec2(p.x + 44, p.y + 44),
                                   g_dos ? cga::dark_gray : IM_COL32(255, 255, 255, 18), rad(10.0f));
            draw_alpha_preview(pdl, ImVec2(p.x + 7, p.y + 7), 30.0f, e.preview);
            if (hv) tooltip(p, ImVec2(p.x + 44, p.y + 44), TipSide::Below, e.name.c_str(), nullptr);
        }
        if (n % 5) ImGui::SameLine();
        ImVec2 p = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##alLoad", ImVec2(44, 44))) {
            input.load_alpha_dialog_active = true;
            ImGui::CloseCurrentPopup();
        }
        bool hv = ImGui::IsItemHovered();
        ImDrawList* pdl = ImGui::GetWindowDrawList();
        if (hv) pdl->AddRectFilled(p, ImVec2(p.x + 44, p.y + 44),
                                   g_dos ? cga::dark_gray : IM_COL32(255, 255, 255, 18), rad(10.0f));
        pdl->AddRect(ImVec2(p.x + 7, p.y + 7), ImVec2(p.x + 37, p.y + 37),
                     g_dos ? cga::light_gray : IM_COL32(255, 255, 255, 60), rad(7.0f), 0, g_dos ? 2.0f : 1.0f);
        pdl->AddLine(ImVec2(p.x + 16, p.y + 22), ImVec2(p.x + 28, p.y + 22), kText, 1.75f);
        pdl->AddLine(ImVec2(p.x + 22, p.y + 16), ImVec2(p.x + 22, p.y + 28), kText, 1.75f);
        if (hv) tooltip(p, ImVec2(p.x + 44, p.y + 44), TipSide::Below,
                        "Load a grayscale image", nullptr);
        ImGui::PopStyleVar();
        end_skin_popup();
    }
}

void rail(InputState& input, int win_h, const AlphaLibrary* alpha_lib) {
    (void)win_h;
    BrushType current = input.current_brush;
    const bool smooth_on = input.is_smooth_active();
    const bool paint_on = current == BrushType::PAINT && !smooth_on;
    const auto mode = input.interaction_mode;
    const bool editing = mode == InputState::InteractionMode::EDIT;

    if (!begin_panel("##mRail", ImVec2(kEdge, kEdge), ImVec2(0, 0), ImVec2(5, 5), 2.0f, 14.0f)) {
        end_panel();
        return;
    }
    BtnOpts o;
    o.look = Look::Rail; o.side = TipSide::Right; o.icon_px = 22.0f;

    // Modes — "what am I doing" is one column with the brushes.
    o.on = editing && !paint_on; o.tip = "Sculpt"; o.key = "1"; o.corner = "1";
    if (icon_button("mSculpt", Icon::ModeSculpt, kRailBtn, o)) {
        input.interaction_mode = InputState::InteractionMode::EDIT;
        if (current == BrushType::PAINT) {
            input.clear_smooth_lock();
            input.switch_brush(BrushType::DRAW);
        }
    }
    o.on = mode == InputState::InteractionMode::INSERT; o.tip = "Insert"; o.key = o.corner = "2";
    if (icon_button("mInsert", Icon::ModeInsert, kRailBtn, o))
        input.interaction_mode = InputState::InteractionMode::INSERT;
    o.on = mode == InputState::InteractionMode::SELECT; o.tip = "Select"; o.key = o.corner = "3";
    if (icon_button("mSelect", Icon::ModeSelect, kRailBtn, o))
        input.interaction_mode = InputState::InteractionMode::SELECT;
    o.on = editing && paint_on; o.tip = "Paint"; o.key = o.corner = "4";
    if (icon_button("mPaint", Icon::ModePaint, kRailBtn, o)) {
        input.interaction_mode = InputState::InteractionMode::EDIT;
        input.clear_smooth_lock();
        input.switch_brush(BrushType::PAINT);
        input.subtract_locked = false;
    }

    hdivider(kRailBtn, 6.0f, kDivider, 9.0f);

    struct B { const char* id; const char* name; const char* key; Icon icon; BrushType type; };
    static const B brushes[] = {
        {"bDraw",    "Draw",    "D", Icon::BrushDraw,    BrushType::DRAW},
        {"bClay",    "Clay",    "T", Icon::BrushClay,    BrushType::CLAY},
        {"bInflate", "Inflate", "I", Icon::BrushInflate, BrushType::INFLATE},
        {"bCrease",  "Crease",  "C", Icon::BrushCrease,  BrushType::CREASE},
        {"bPinch",   "Pinch",   "V", Icon::BrushPinch,   BrushType::PINCH},
        {"bMove",    "Move",    "G", Icon::BrushMove,    BrushType::MOVE},
        {"bLimb",    "Limb",    "H", Icon::BrushLimb,    BrushType::LIMB},
        {"bSmooth",  "Smooth",  nullptr, Icon::BrushSmooth, BrushType::SMOOTH},
        {"bMask",    "Mask",    "M", Icon::BrushMask,    BrushType::MASK},
    };
    // Shift-Shift: the glyph if the face has it, else spelled out.
    ImFont* mono = F(UiFont::Mono);
    const char* shift2 = (mono && mono->IsGlyphInFont(0x21E7)) ? "\xE2\x87\xA7\xE2\x87\xA7" : "SS";
    for (const B& b : brushes) {
        if (b.type == BrushType::SMOOTH) hdivider(kRailBtn, 10.0f, kDividerLo, 7.0f);
        o.tip = b.name;
        if (b.type == BrushType::SMOOTH) {
            o.on = smooth_on && editing;
            o.key = "Shift Shift"; o.corner = shift2;
            if (icon_button(b.id, b.icon, kRailBtn, o)) {
                if (!input.smooth_locked) {
                    // Same two lines as the double-tap-Shift path in input.cpp.
                    input.smooth_locked = true;
                    input.sync_live_settings();
                } else {
                    input.clear_smooth_lock();
                }
            }
            continue;
        }
        o.on = editing && current == b.type && !smooth_on;
        o.key = o.corner = b.key;
        if (icon_button(b.id, b.icon, kRailBtn, o)) {
            input.clear_smooth_lock();
            input.switch_brush(b.type);
            input.subtract_locked = false;
            input.interaction_mode = InputState::InteractionMode::EDIT;
        }
    }

    // Current alpha (stamp). Only where an alpha applies: Draw, Mask, Paint.
    const bool alpha_brush = current == BrushType::DRAW || current == BrushType::MASK ||
                             current == BrushType::PAINT;
    if (editing && !smooth_on && alpha_brush && alpha_lib && alpha_lib->count() > 0) {
        hdivider(kRailBtn, 6.0f, kDivider, 9.0f);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        bool clicked = ImGui::InvisibleButton("##alphaBtn", ImVec2(kRailBtn, kRailBtn));
        bool hovered = ImGui::IsItemHovered();
        bool held = ImGui::IsItemActive();
        float s = spring(ImGui::GetItemID(), held ? 0.92f : 1.0f, held ? 1400.0f : 520.0f);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 c(pos.x + kRailBtn * 0.5f, pos.y + kRailBtn * 0.5f);
        if (hovered) dl->AddRectFilled(ImVec2(c.x - 23 * s, c.y - 23 * s), ImVec2(c.x + 23 * s, c.y + 23 * s),
                                       g_dos ? cga::dark_gray : IM_COL32(255, 255, 255, 18), rad(10.0f));
        int ai = std::max(0, std::min(input.active_alpha, alpha_lib->count() - 1));
        const AlphaEntry& ae = alpha_lib->get(ai);
        float sz = 30.0f * s;
        draw_alpha_preview(dl, ImVec2(c.x - sz * 0.5f, c.y - sz * 0.5f), sz, ae.preview);
        if (hovered) tooltip(pos, ImVec2(pos.x + kRailBtn, pos.y + kRailBtn), TipSide::Right,
                             ae.name.c_str(), nullptr);
        if (clicked) ImGui::OpenPopup("##alphapick");

        alpha_picker_popup(input, alpha_lib, ImVec2(pos.x + kRailBtn + 18.0f, pos.y - 8.0f));
    }
    end_panel();
}

void file_panel(InputState& input, const SkinStats& st) {
    if (!begin_panel("##mFile", ImVec2(kEdge + kRailW + kEdge, kEdge), ImVec2(0, 0),
                     ImVec2(4, 4), 2.0f, 12.0f)) { end_panel(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    char name[64];
    project_name(name, sizeof name, st.project_path);
    const bool unsaved = !st.project_path || !*st.project_path;
    if (g_dos) {
        // Board D: yellow CHISEL brand, the name, a light-red * while unsaved.
        ImVec2 bs = text_size(UiFont::SansSemibold, 16.0f, "CHISEL");
        ImVec2 ns = text_size(UiFont::Sans, 16.0f, name);
        float w = 10.0f + bs.x + 18.0f + ns.x + (unsaved ? 6.0f + 16.0f : 0.0f) + 8.0f;
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, kBarBtn));
        float ty = p.y + (kBarBtn - 16.0f) * 0.5f;
        text_at(dl, UiFont::SansSemibold, 16.0f, ImVec2(p.x + 10.0f, ty), cga::yellow, "CHISEL");
        text_at(dl, UiFont::Sans, 16.0f, ImVec2(p.x + 10.0f + bs.x + 18.0f, ty), cga::white, name);
        if (unsaved)
            text_at(dl, UiFont::Sans, 16.0f, ImVec2(p.x + 10.0f + bs.x + 18.0f + ns.x + 6.0f, ty),
                    cga::light_red, "*");
    } else {
    ImVec2 ns = text_size(UiFont::SansSemibold, 14.0f, name);
    ImVec2 us = text_size(UiFont::Sans, 11.0f, "unsaved");
    float w = 10.0f + ns.x + (unsaved ? 6.0f + 10.0f + us.x : 0.0f) + 8.0f;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, kBarBtn));
    text_at(dl, UiFont::SansSemibold, 14.0f, ImVec2(p.x + 10.0f, p.y + (kBarBtn - ns.y) * 0.5f), kText, name);
    if (unsaved) {
        float x = p.x + 10.0f + ns.x + 6.0f;
        dl->AddCircleFilled(ImVec2(x + 3.0f, p.y + kBarBtn * 0.5f), 3.0f, kUnsaved, 12);
        text_at(dl, UiFont::Sans, 11.0f, ImVec2(x + 10.0f, p.y + (kBarBtn - us.y) * 0.5f), kUnsaved, "unsaved");
    }
    }
    vdivider();

    BtnOpts o;
    o.tip = "Save"; o.key = "Ctrl+S";
    if (icon_button("fSave", Icon::CmdSave, kBarBtn, o)) input.save_requested = true;
    ImGui::SameLine();
    o.tip = "Save numbered copy"; o.key = nullptr;
    if (icon_button("fSaveInc", Icon::CmdSaveCopy, kBarBtn, o)) input.save_incremental_requested = true;
    ImGui::SameLine();
    o.tip = "Open"; o.key = "Ctrl+O";
    if (icon_button("fOpen", Icon::CmdOpen, kBarBtn, o)) input.import_dialog_active = true;
    ImGui::SameLine();
    o.tip = "Export"; o.key = "Ctrl+E";
    if (icon_button("fExport", Icon::CmdExport, kBarBtn, o)) input.export_dialog_active = true;
    end_panel();
}

void command_panel(InputState& input, int win_w, int win_h, MultiresInfo mres, const SkinStats& st) {
    if (!begin_panel("##mCmd", ImVec2((float)win_w - kEdge, kEdge), ImVec2(1, 0),
                     ImVec2(4, 4), 2.0f, 12.0f)) { end_panel(); return; }
    BtnOpts o;
    o.tip = "Undo"; o.key = "Ctrl+Z";
    if (icon_button("cUndo", Icon::CmdUndo, kBarBtn, o)) input.undo_requested = true;
    ImGui::SameLine();
    o.tip = "Redo"; o.key = "Ctrl+Shift+Z";
    if (icon_button("cRedo", Icon::CmdRedo, kBarBtn, o)) input.redo_requested = true;
    vdivider();

    // Subdivision stepper: − / L6 / 9 over the triangle count / +.
    o.enabled = input.mesh_locked;
    o.tip = "Subdivision down"; o.key = "Shift+D";
    if (icon_button("cLvlDn", Icon::CmdLevelDown, kBarBtn, o)) input.level_switch_delta = -1;
    ImGui::SameLine(0.0f, 2.0f);
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = 64.0f;
        char lv[16], mx[16], tris[24];
        if (g_dos) {
            // 16 px cells are wide: "L4/9" in light green, the tri count at 8 px below.
            if (mres.locked) std::snprintf(lv, sizeof lv, "L%d/%d", st.level, mres.lmax);
            else             std::snprintf(lv, sizeof lv, "L%d", st.level);
            fmt_k(tris, sizeof tris, st.tris);
            std::strncat(tris, " TRIS", sizeof tris - std::strlen(tris) - 1);
            for (char* c = tris; *c; c++) *c = (char)std::toupper((unsigned char)*c);
            ImVec2 a = text_size(UiFont::Mono, 16.0f, lv), t = text_size(UiFont::Mono, 8.0f, tris);
            const float wd = std::max(80.0f, std::max(a.x, t.x) + 8.0f);
            ImGui::Dummy(ImVec2(wd, kBarBtn));
            float top = p.y + (kBarBtn - 16.0f - 4.0f - 8.0f) * 0.5f;
            text_at(dl, UiFont::Mono, 16.0f, ImVec2(p.x + (wd - a.x) * 0.5f, top), cga::light_green, lv);
            text_at(dl, UiFont::Mono, 8.0f, ImVec2(p.x + (wd - t.x) * 0.5f, top + 20.0f), cga::light_gray, tris);
        } else {
        ImGui::Dummy(ImVec2(w, kBarBtn));
        std::snprintf(lv, sizeof lv, "L%d", st.level);
        if (mres.locked) std::snprintf(mx, sizeof mx, " / %d", mres.lmax);
        else mx[0] = '\0';
        fmt_k(tris, sizeof tris, st.tris);
        std::strncat(tris, " tris", sizeof tris - std::strlen(tris) - 1);
        ImVec2 a = text_size(UiFont::MonoMedium, 15.0f, lv);
        ImVec2 b = text_size(UiFont::MonoMedium, 15.0f, mx);
        ImVec2 t = text_size(UiFont::Sans, 11.0f, tris);
        float top = p.y + (kBarBtn - (a.y + t.y - 2.0f)) * 0.5f;
        float x0 = p.x + (w - (a.x + b.x)) * 0.5f;
        text_at(dl, UiFont::MonoMedium, 15.0f, ImVec2(x0, top), kText, lv);
        text_at(dl, UiFont::MonoMedium, 15.0f, ImVec2(x0 + a.x, top), kDim, mx);
        text_at(dl, UiFont::Sans, 11.0f, ImVec2(p.x + (w - t.x) * 0.5f, top + a.y - 2.0f), kMuted, tris);
        }
    }
    ImGui::SameLine(0.0f, 2.0f);
    o.tip = "Subdivision up"; o.key = "Ctrl+D";
    if (icon_button("cLvlUp", Icon::CmdLevelUp, kBarBtn, o)) input.level_switch_delta = +1;
    o.enabled = true;
    vdivider();

    o.tip = "SDF merge"; o.key = "J";
    if (icon_button("cMerge", Icon::CmdMerge, kBarBtn, o)) input.voxel_merge_confirm_pending = true;
    ImGui::SameLine();
    o.on = input.mirror_x;
    o.tip = input.mirror_x ? (input.mirror_topological ? "Mirror X: on, topological"
                                                       : "Mirror X: on, world space")
                           : "Mirror X: off";
    o.key = "X";
    if (icon_button("cMirror", Icon::ViewMirrorX, kBarBtn, o)) input.mirror_x = !input.mirror_x;
    ImGui::SameLine();
    o.on = false;
    o.tip = input.paint_visible ? "Paint shown while sculpting" : "Paint hidden while sculpting";
    o.key = nullptr;
    if (icon_button("cPaintVis", input.paint_visible ? Icon::ViewPaintShow : Icon::ViewPaintHide,
                    kBarBtn, o))
        input.paint_visible = !input.paint_visible;
    vdivider();

    o.on = ImGui::IsPopupOpen("##helppopup");
    o.tip = "Shortcuts"; o.key = nullptr;
    if (icon_button("cHelp", Icon::UiShortcuts, kBarBtn, o)) ImGui::OpenPopup("##helppopup");
    if (!input.help_seen) {           // first launch ever: open the card once
        input.help_seen = true;
        ImGui::OpenPopup("##helppopup");
    }
    ImGui::SameLine();
    o.on = ImGui::IsPopupOpen("##burgermenu");
    o.tip = "Menu";
    if (icon_button("cMenu", Icon::UiMenu, kBarBtn, o)) ImGui::OpenPopup("##burgermenu");

    // Menu state sync (see the classic islands for why each of these exists).
    static bool prev_menu_open = false;
    bool menu_open = ImGui::IsPopupOpen("##burgermenu");
    bool sync_tabs = menu_open && !prev_menu_open;
    prev_menu_open = menu_open;
    input.settings_menu_open = menu_open;

    float bar_bottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
    ImGui::SetNextWindowPos(ImVec2((float)win_w - kEdge, bar_bottom + 8.0f), ImGuiCond_Always,
                            ImVec2(1, 0));
    ImGui::SetNextWindowSizeConstraints(ImVec2(300, 0), ImVec2(FLT_MAX, (float)win_h - bar_bottom - 24.0f));
    if (begin_skin_popup("##burgermenu", ImVec2(16, 16))) {
        draw_settings_menu_items(input, mres, sync_tabs);
        end_skin_popup();
    }
    draw_help_popup(input, win_w, win_h);
    end_panel();
}

struct Meter { const char* key; const char* label; char value[16]; float pct; ImU32 col; };

void meter_widget(const char* id, const Meter& m) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    if (g_dos) {
        // Board D: "S Size 70" — cyan key, gray label, green value, no bar.
        const char* lab = m.label;
        if (!std::strcmp(lab, "Strength")) lab = "Str";
        else if (!std::strcmp(lab, "Hardness")) lab = "Hard";
        else if (!std::strcmp(lab, "Spacing")) lab = "Spc";
        char mid[24]; std::snprintf(mid, sizeof mid, " %s ", lab);
        ImVec2 k = text_size(UiFont::Mono, 16.0f, m.key), l = text_size(UiFont::Mono, 16.0f, mid);
        ImVec2 v = text_size(UiFont::Mono, 16.0f, m.value);
        const float h = 30.0f, ty = p.y + (h - 16.0f) * 0.5f;
        ImGui::Dummy(ImVec2(k.x + l.x + std::max(v.x, 48.0f), h));   // fixed slot: 3 cells
        text_at(dl, UiFont::Mono, 16.0f, ImVec2(p.x, ty), cga::light_cyan, m.key);
        text_at(dl, UiFont::Mono, 16.0f, ImVec2(p.x + k.x, ty), cga::light_gray, mid);
        text_at(dl, UiFont::Mono, 16.0f, ImVec2(p.x + k.x + l.x, ty), cga::light_green, m.value);
        return;
    }
    const float w = 112.0f, h = 30.0f;
    ImGui::Dummy(ImVec2(w, h));
    // key badge
    ImVec2 b0(p.x, p.y + 2.0f), b1(p.x + 16.0f, p.y + 18.0f);
    dl->AddRect(b0, b1, IM_COL32(255, 255, 255, 51), 4.0f);
    ImVec2 ks = text_size(UiFont::Mono, 10.0f, m.key);
    text_at(dl, UiFont::Mono, 10.0f, ImVec2(b0.x + (16.0f - ks.x) * 0.5f, b0.y + (16.0f - ks.y) * 0.5f), kText, m.key);
    ImVec2 ls = text_size(UiFont::Sans, 13.0f, m.label);
    text_at(dl, UiFont::Sans, 13.0f, ImVec2(p.x + 22.0f, p.y + 10.0f - ls.y * 0.5f), kMuted, m.label);
    ImVec2 vs = text_size(UiFont::Mono, 13.0f, m.value);
    text_at(dl, UiFont::Mono, 13.0f, ImVec2(p.x + w - vs.x, p.y + 10.0f - vs.y * 0.5f), kText, m.value);
    // bar, springing to its new width
    float pct = spring(ImGui::GetID(id), std::max(0.0f, std::min(1.0f, m.pct)), 260.0f, 0.6f);
    pct = std::max(0.0f, std::min(1.0f, pct));
    float y = p.y + h - 4.0f;
    dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + w, y + 3.0f), IM_COL32(255, 255, 255, 26), 2.0f);
    if (pct > 0.0f) dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + w * pct, y + 3.0f), m.col, 2.0f);
}

// A small text label laid out as an item (so SameLine works around it).
void label_item(UiFont f, float px, ImU32 col, const char* s, float min_w = 0.0f, float h = 30.0f) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImVec2 ts = text_size(f, px, s);
    ImGui::Dummy(ImVec2(std::max(ts.x, min_w), h));
    text_at(ImGui::GetWindowDrawList(), f, px, ImVec2(p.x, p.y + (h - ts.y) * 0.5f), col, s);
}

// Insert-shape button: same squash as the icon buttons, its own little line glyphs.
bool shape_button(const char* id, int kind, bool on, const char* tip) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float size = 36.0f;
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();
    bool held = ImGui::IsItemActive();
    float s = spring(ImGui::GetItemID(), held ? 0.92f : 1.0f, held ? 1400.0f : 520.0f);
    ImVec2 c(pos.x + size * 0.5f, pos.y + size * 0.5f);
    ImVec2 a(c.x - size * 0.5f * s, c.y - size * 0.5f * s), b(c.x + size * 0.5f * s, c.y + size * 0.5f * s);
    if (on) dl->AddRectFilled(a, b, kAccent, rad(9.0f));
    else if (hovered) dl->AddRectFilled(a, b, g_dos ? cga::dark_gray : IM_COL32(255, 255, 255, 18), rad(9.0f));
    ImU32 col = on ? (g_dos ? cga::black : IM_COL32_WHITE) : kText;
    float k = s * 0.75f, th = 1.6f;
    if (kind == 0) {
        dl->AddCircle(c, 9.0f * k * 1.33f, col, 0, th);
        dl->AddEllipse(c, ImVec2(12.0f * k, 4.0f * k), col, 0.0f, 0, th * 0.75f);
    } else if (kind == 1) {
        draw_icon(dl, Icon::ModeInsert, c, 24.0f * k * 1.1f, col);
    } else {
        float rw = 8.0f * k * 1.2f, rh = 10.0f * k * 1.2f, ry = 3.2f * k * 1.2f;
        dl->AddEllipse(ImVec2(c.x, c.y - rh), ImVec2(rw, ry), col, 0.0f, 0, th);
        dl->AddLine(ImVec2(c.x - rw, c.y - rh), ImVec2(c.x - rw, c.y + rh), col, th);
        dl->AddLine(ImVec2(c.x + rw, c.y - rh), ImVec2(c.x + rw, c.y + rh), col, th);
        dl->PathArcTo(ImVec2(c.x, c.y + rh), rw, 0.0f, 3.14159265f);
        // squash the lower arc into the ellipse's proportions
        for (ImVec2& v : dl->_Path) v.y = c.y + rh + (v.y - (c.y + rh)) * (ry / rw);
        dl->PathStroke(col, 0, th);
    }
    if (hovered) tooltip(pos, ImVec2(pos.x + size, pos.y + size), TipSide::Below, tip, nullptr);
    return clicked;
}

float g_brush_right = 0.0f;   // readout's right edge, for status_panel's collision check

float brush_panel(InputState& input, int win_h) {
    const auto mode = input.interaction_mode;
    float panel_h = 0.0f;
    if (!begin_panel("##mBrush", ImVec2(kEdge + kRailW + kEdge, (float)win_h - kEdge), ImVec2(0, 1),
                     ImVec2(16, 10), 18.0f, 12.0f)) { end_panel(); return 0.0f; }
    if (mode == InputState::InteractionMode::INSERT) {
        label_item(UiFont::SansSemibold, 14.0f, title_col(), title_case("Insert"), 64.0f, 36.0f);
        struct S { const char* id; int kind; const char* tip; InputState::InsertShape shape; };
        static const S shapes[] = {
            {"##shSphere", 0, "Sphere",   InputState::InsertShape::SPHERE},
            {"##shBox",    1, "Box",      InputState::InsertShape::BOX},
            {"##shCyl",    2, "Cylinder", InputState::InsertShape::CYLINDER},
        };
        ImGui::SameLine(0.0f, 10.0f);
        for (int i = 0; i < 3; i++) {
            if (i) ImGui::SameLine(0.0f, 4.0f);
            if (shape_button(shapes[i].id, shapes[i].kind, input.insert_shape == shapes[i].shape,
                             shapes[i].tip))
                input.insert_shape = shapes[i].shape;
        }
        ImGui::SameLine(0.0f, 14.0f);
        label_item(UiFont::Sans, 12.0f, kMuted, "Click the model to place", 0.0f, 36.0f);
    } else if (mode == InputState::InteractionMode::SELECT) {
        label_item(UiFont::SansSemibold, 14.0f, title_col(), title_case("Select"), 64.0f);
        ImGui::SameLine();
        label_item(UiFont::Sans, 12.0f, kMuted, "Click to pick  \xC2\xB7  drag to move  \xC2\xB7  Q / E rotate");
    } else {
        label_item(UiFont::SansSemibold, 14.0f, title_col(), title_case(input.brush_name()), 64.0f);
        Meter ms[4] = {
            {"S", "Size",     "", input.brush_size / 500.0f, kBrush},
            {"W", "Strength", "", input.brush_strength, kText},
            {"A", "Hardness", "", input.brush_hardness, kText},
            {"O", "Spacing",  "", input.brush_spacing, kText},
        };
        std::snprintf(ms[0].value, sizeof ms[0].value, "%.0f", input.brush_size);
        std::snprintf(ms[1].value, sizeof ms[1].value, "%.0f%%", input.brush_strength * 100.0f);
        std::snprintf(ms[2].value, sizeof ms[2].value, "%.0f%%", input.brush_hardness * 100.0f);
        std::snprintf(ms[3].value, sizeof ms[3].value, "%.0f%%", input.brush_spacing * 100.0f);
        static const char* ids[4] = {"##mS", "##mW", "##mA", "##mO"};
        for (int i = 0; i < 4; i++) {
            ImGui::SameLine();
            meter_widget(ids[i], ms[i]);
        }
    }
    panel_h = ImGui::GetWindowSize().y;
    g_brush_right = ImGui::GetWindowPos().x + ImGui::GetWindowSize().x;
    end_panel();
    return panel_h;
}

// Tool options: what the current tool needs beyond the four meters. Stacked above
// the brush readout; absent when there is nothing to show.
void options_panel(InputState& input, int win_h, float brush_h) {
    const BrushType current = input.current_brush;
    const bool smooth_on = input.is_smooth_active();
    const bool editing = input.interaction_mode == InputState::InteractionMode::EDIT;
    const bool paint_on = editing && current == BrushType::PAINT && !smooth_on;
    const bool clay_on = editing && current == BrushType::CLAY && !smooth_on;
    if (!paint_on && !clay_on) return;

    float y = (float)win_h - kEdge - brush_h - 8.0f;
    if (!begin_panel("##mOpts", ImVec2(kEdge + kRailW + kEdge, y), ImVec2(0, 1),
                     ImVec2(12, 8), 8.0f, 12.0f)) { end_panel(); return; }
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 7));
    if (clay_on) {
        label_item(UiFont::Sans, 13.0f, kMuted, "Melt", 0.0f, 30.0f);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("##clayMelt", &input.clay_melt, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How strongly clay cuts raised areas down to its plane\n"
                              "(0 = ride over them untouched, 1 = flatten to the plane)");
    }
    if (paint_on) {
        if (chip("Colour", !input.paint_target_density)) input.paint_target_density = false;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Paint albedo");
        ImGui::SameLine(0.0f, 4.0f);
        if (chip("Density", input.paint_target_density)) {
            input.paint_target_density = true;
            input.color_pick_active = false;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paint remesh density (green = coarse, red = dense; Ctrl lowers)");
        ImGui::SameLine(0.0f, 12.0f);
        if (!input.paint_target_density) {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 10));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7.0f);
            ImGui::ColorEdit3("##paintA", input.paint_color,
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine(0.0f, 4.0f);
            ImGui::ColorEdit3("##paintB", input.paint_color_alt,
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::PopStyleVar(2);
            ImGui::SameLine(0.0f, 10.0f);
            label_item(UiFont::Sans, 12.0f, input.color_pick_active ? kTintText : kMuted,
                       input.color_pick_active ? "Picking... click the model"
                                               : "Q / E swap  \xC2\xB7  C pick  \xC2\xB7  right-click: wheel");
        } else {
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat("##densCoarse", &input.density_coarse_mult, 1.0f, 4.0f, "green: edge x%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remesh edge length where painted green (coarse)");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(170.0f);
            ImGui::SliderFloat("##densFine", &input.density_fine_mult, 0.2f, 1.0f, "red: edge x%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remesh edge length where painted red (dense)");
        }
    }
    ImGui::PopStyleVar();
    end_panel();
}

// brush_right / brush_h: the readout's right edge and height this frame. On a window
// too narrow for both on one row (or with DOS's wide 16 px cells), the status line
// steps up and sits above the readout's row instead of sliding under it. The panel
// keeps its x either way, so the check can't flip back and forth.
void status_panel(const InputState& input, int win_w, int win_h, const SkinStats& st,
                  float brush_right, float brush_h) {
    static bool lifted = false;
    float y = (float)win_h - kEdge - (lifted ? brush_h + 12.0f : 0.0f);
    if (!begin_panel("##mStatus", ImVec2((float)win_w - kEdge, y), ImVec2(1, 1),
                     ImVec2(14, 10), 14.0f, 12.0f)) { end_panel(); return; }
    if (ImGui::GetWindowSize().x > 1.0f) lifted = ImGui::GetWindowPos().x < brush_right + 12.0f;
    char verts[32], buf[48];
    fmt_thousands(verts, sizeof verts, st.verts);
    if (g_dos) {
        // Board D's status line: gray labels, green numbers, magenta mirror.
        label_item(UiFont::Mono, 16.0f, cga::light_gray, "Verts", 0.0f, 18.0f);
        ImGui::SameLine(0.0f, 8.0f);
        label_item(UiFont::Mono, 16.0f, cga::light_green, verts, 0.0f, 18.0f);
        ImGui::SameLine(0.0f, 24.0f);
        label_item(UiFont::Mono, 16.0f, input.mirror_x ? cga::light_magenta : cga::dark_gray,
                   !input.mirror_x ? "Mirror:Off"
                   : input.mirror_topological ? "Mirror:Topo" : "Mirror:World", 0.0f, 18.0f);
        if (input.autosmooth) {
            ImGui::SameLine(0.0f, 24.0f);
            label_item(UiFont::Mono, 16.0f, cga::light_cyan, "Autosmooth", 0.0f, 18.0f);
        }
        if (input.show_fps) {
            ImGui::SameLine(0.0f, 24.0f);
            std::snprintf(buf, sizeof buf, "%.0ffps", st.fps);
            label_item(UiFont::Mono, 16.0f, cga::light_gray, buf, 0.0f, 18.0f);
        }
        end_panel();
        return;
    }
    std::snprintf(buf, sizeof buf, "%s verts", verts);
    label_item(UiFont::Mono, 12.0f, kStatus, buf, 0.0f, 18.0f);
    ImGui::SameLine();
    if (input.mirror_x)
        label_item(UiFont::Sans, 12.0f, kTintText,
                   input.mirror_topological ? "Mirror \xC2\xB7 Topological" : "Mirror \xC2\xB7 World",
                   0.0f, 18.0f);
    else
        label_item(UiFont::Sans, 12.0f, kMuted, "Mirror off", 0.0f, 18.0f);
    if (input.autosmooth) {
        ImGui::SameLine();
        label_item(UiFont::Sans, 12.0f, kStatus, "Autosmooth", 0.0f, 18.0f);
    }
    if (input.show_fps) {
        ImGui::SameLine();
        std::snprintf(buf, sizeof buf, "%.0f fps", st.fps);
        label_item(UiFont::Mono, 12.0f, kMuted, buf, 0.0f, 18.0f);
    }
    end_panel();
}

// A toast in the panel material, on the foreground list (never steals the mouse).
void toast(const char* msg, float cx, float y, ImU32 col, float alpha) {
    if (alpha <= 0.0f) return;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImVec2 ts = text_size(UiFont::Sans, 14.0f, msg);
    float maxw = ImGui::GetIO().DisplaySize.x - 2 * kEdge - 2 * 18.0f;
    if (ts.x > maxw) ts.x = maxw;
    ImVec2 a(cx - ts.x * 0.5f - 18.0f, y), b(cx + ts.x * 0.5f + 18.0f, y + ts.y + 20.0f);
    Material keep = g_mat;
    g_mat.fill_a = std::min(1.0f, g_mat.fill_a + 0.1f);
    int v0 = dl->VtxBuffer.Size;
    panel_bg(dl, a, b, 12.0f);
    g_mat = keep;
    dl->PushClipRect(ImVec2(a.x + 12, a.y), ImVec2(b.x - 12, b.y), true);
    text_at(dl, UiFont::Sans, 14.0f, ImVec2(a.x + 18.0f, a.y + 10.0f), col, msg);
    dl->PopClipRect();
    if (alpha < 1.0f)
        for (int i = v0; i < dl->VtxBuffer.Size; i++) {
            ImDrawVert& v = dl->VtxBuffer[i];
            int al = (int)(((v.col >> IM_COL32_A_SHIFT) & 0xFF) * alpha);
            v.col = (v.col & ~IM_COL32_A_MASK) | ((ImU32)al << IM_COL32_A_SHIFT);
        }
}

void notifications(InputState& input, int win_w, int win_h) {
    float dt = ImGui::GetIO().DeltaTime;
    if (input.notification_timer > 0.0f) {
        input.notification_timer = std::max(0.0f, input.notification_timer - dt);
        float a = std::min(1.0f, input.notification_timer / 0.3f);
        toast(input.notification, win_w * 0.5f,
              g_y2k ? (float)win_h - 30.0f - 60.0f : (float)win_h - kEdge - 44.0f - 56.0f,
              g_dos ? cga::light_green : kText, a);
    }
    if (input.mirror_unavailable_timer > 0.0f) {
        input.mirror_unavailable_timer = std::max(0.0f, input.mirror_unavailable_timer - dt);
        float a = std::min(1.0f, input.mirror_unavailable_timer / 0.3f);
        toast("Topological mirror not available", win_w * 0.5f, win_h * (2.0f / 3.0f), kAlert, a);
    }
}

// Hold S / W / A / O + drag: name, value and a bar just above the locked point.
void slider_hud(const InputState& input) {
    if (input.slider_mode == InputState::SliderMode::NONE) return;
    const char* name = "";
    char val[16];
    float pct = 0.0f;
    ImU32 col = kText;
    switch (input.slider_mode) {
        case InputState::SliderMode::SIZE:
            name = "Size"; std::snprintf(val, sizeof val, "%.0f", input.brush_size);
            pct = input.brush_size / 500.0f; col = kBrush; break;
        case InputState::SliderMode::STRENGTH:
            name = "Strength"; std::snprintf(val, sizeof val, "%.0f%%", input.brush_strength * 100.0f);
            pct = input.brush_strength; break;
        case InputState::SliderMode::HARDNESS:
            name = "Hardness"; std::snprintf(val, sizeof val, "%.0f%%", input.brush_hardness * 100.0f);
            pct = input.brush_hardness; break;
        case InputState::SliderMode::SPACING:
            name = "Spacing"; std::snprintf(val, sizeof val, "%.0f%%", input.brush_spacing * 100.0f);
            pct = input.brush_spacing; break;
        default: return;
    }
    pct = std::max(0.0f, std::min(1.0f, pct));
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float w = 150.0f, h = 46.0f;
    ImVec2 a((float)input.slider_start_x - w * 0.5f, (float)input.slider_start_y + 22.0f);
    ImVec2 b(a.x + w, a.y + h);
    panel_bg(dl, a, b, 10.0f);
    text_at(dl, UiFont::Sans, 13.0f, ImVec2(a.x + 12.0f, a.y + 8.0f), kMuted, name);
    ImVec2 vs = text_size(UiFont::MonoMedium, 14.0f, val);
    text_at(dl, UiFont::MonoMedium, 14.0f, ImVec2(b.x - 12.0f - vs.x, a.y + 7.0f), kText, val);
    float y = b.y - 12.0f;
    dl->AddRectFilled(ImVec2(a.x + 12.0f, y), ImVec2(b.x - 12.0f, y + 4.0f),
                      g_dos ? cga::dark_gray : IM_COL32(255, 255, 255, 26), rad(2.0f));
    dl->AddRectFilled(ImVec2(a.x + 12.0f, y), ImVec2(a.x + 12.0f + (w - 24.0f) * pct, y + 4.0f), col, rad(2.0f));
}

// ============================ Y2K skin (board E) =============================
// An homage to the docked-shelf 3D apps of the early 2000s (no copied layout or
// branding): opaque shelves on every edge, glossy two-tone tiles with a hard
// step at half height, orange for "on", and an inset bevel framing the viewport.
// The 3D view still renders full-window behind the shelves; the model sits in
// the middle, so the shelves only cover what the floating panels would.

constexpr float kTopH = 78.0f, kTitleH = 22.0f, kLeftW = 112.0f, kRightW = 94.0f, kBotH = 30.0f;

ImFont* yf(bool b) { return b ? g_y2k_bold : g_y2k_font; }
ImVec2 ysize(bool b, float px, const char* s) { return yf(b)->CalcTextSizeA(px, FLT_MAX, 0.0f, s); }

// Text with the era's 1 px drop shadow under it.
void ytext(ImDrawList* dl, bool b, float px, ImVec2 p, ImU32 col, const char* s, bool shadow = true) {
    p = ImVec2(std::floor(p.x + 0.5f), std::floor(p.y + 0.5f));
    if (shadow) dl->AddText(yf(b), px, ImVec2(p.x, p.y + 1.0f), IM_COL32(0, 0, 0, 255), s);
    dl->AddText(yf(b), px, p, col, s);
}

void vgrad(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 top, ImU32 bottom, float r = 0.0f,
           ImDrawFlags f = 0) {
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, IM_COL32_WHITE, r, f);
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, a, ImVec2(a.x, b.y), top, bottom);
}
void hgrad(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 left, ImU32 right) {
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, IM_COL32_WHITE);
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, a, ImVec2(b.x, a.y), left, right);
}

// Recessed well: the dark field everything "inset" sits in.
void inset(ImDrawList* dl, ImVec2 a, ImVec2 b, float r) {
    dl->AddRectFilled(a, b, y2k::well, r);
    dl->AddLine(ImVec2(a.x + r, a.y + 0.5f), ImVec2(b.x - r, a.y + 0.5f), IM_COL32(0, 0, 0, 255));
    dl->AddLine(ImVec2(a.x + r, b.y + 0.5f), ImVec2(b.x - r, b.y + 0.5f), IM_COL32(255, 255, 255, 31));
}

// The glossy two-tone tile: light-to-mid on top, a hard step at half height,
// dark-to-mid below, lit top edge. Orange when on.
void gloss_tile(ImDrawList* dl, ImVec2 a, ImVec2 b, float r, bool on, bool hovered, bool held) {
    ImU32 t0, t1, b0, b1;
    if (on)           { t0 = IM_COL32(0xFF,0xC2,0x66,255); t1 = IM_COL32(0xF7,0xA2,0x3C,255);
                        b0 = IM_COL32(0xEA,0x84,0x18,255); b1 = IM_COL32(0xD4,0x6F,0x0E,255); }
    else if (held)    { t0 = IM_COL32(0x3C,0x3C,0x40,255); t1 = IM_COL32(0x44,0x44,0x49,255);
                        b0 = IM_COL32(0x48,0x48,0x4D,255); b1 = IM_COL32(0x50,0x50,0x55,255); }
    else if (hovered) { t0 = IM_COL32(0x7C,0x7C,0x82,255); t1 = IM_COL32(0x60,0x60,0x66,255);
                        b0 = IM_COL32(0x52,0x52,0x58,255); b1 = IM_COL32(0x5A,0x5A,0x60,255); }
    else              { t0 = IM_COL32(0x6C,0x6C,0x71,255); t1 = IM_COL32(0x52,0x52,0x57,255);
                        b0 = IM_COL32(0x45,0x45,0x4A,255); b1 = IM_COL32(0x4D,0x4D,0x52,255); }
    dl->AddRectFilled(ImVec2(a.x, a.y + 1), ImVec2(b.x, b.y + 2), IM_COL32(0, 0, 0, 120), r);
    float mid = std::floor((a.y + b.y) * 0.5f);
    vgrad(dl, a, ImVec2(b.x, mid), t0, t1, r, ImDrawFlags_RoundCornersTop);
    vgrad(dl, ImVec2(a.x, mid), b, b0, b1, r, ImDrawFlags_RoundCornersBottom);
    dl->AddRect(a, b, y2k::edge, r);
    dl->AddLine(ImVec2(a.x + r, a.y + 1.5f), ImVec2(b.x - r, a.y + 1.5f),
                IM_COL32(255, 255, 255, on ? 140 : (held ? 20 : 72)));
}

// An invisible button at an absolute spot, reporting hover/held.
bool hit(const char* id, ImVec2 a, ImVec2 size, bool* hovered, bool* held, bool enabled = true) {
    ImGui::SetCursorScreenPos(a);
    if (!enabled) ImGui::BeginDisabled();
    bool c = ImGui::InvisibleButton(id, size);
    *hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    *held = ImGui::IsItemActive();
    if (!enabled) ImGui::EndDisabled();
    return c && enabled;
}

bool begin_shelf(const char* name, ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const ImGuiWindowFlags f = (kPanelFlags & ~ImGuiWindowFlags_AlwaysAutoResize) |
                               ImGuiWindowFlags_NoScrollWithMouse;
    bool open = ImGui::Begin(name, nullptr, f);
    ImGui::GetWindowDrawList()->PushClipRectFullScreen();
    return open;
}
void end_shelf() {
    ImGui::GetWindowDrawList()->PopClipRect();
    ImGui::End();
    ImGui::PopStyleVar(3);
}

void apply_mode(InputState& input, int m) {
    const BrushType current = input.current_brush;
    switch (m) {
        case 0:
            input.interaction_mode = InputState::InteractionMode::EDIT;
            if (current == BrushType::PAINT) { input.clear_smooth_lock(); input.switch_brush(BrushType::DRAW); }
            break;
        case 1: input.interaction_mode = InputState::InteractionMode::INSERT; break;
        case 2: input.interaction_mode = InputState::InteractionMode::SELECT; break;
        case 3:
            input.interaction_mode = InputState::InteractionMode::EDIT;
            input.clear_smooth_lock();
            input.switch_brush(BrushType::PAINT);
            input.subtract_locked = false;
            break;
    }
}

// Brush feel written the way the hold-key drag writes it: the live mirror for
// this frame, and per_brush[] / brush_size_of[] as the truth.
void set_feel(InputState& in, int which, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    const BrushType slot = in.live_brush_slot();
    switch (which) {
        case 0: in.brush_size = std::max(5.0f, t * 500.0f);
                in.brush_size_of[(int)in.size_slot()] = in.brush_size; break;
        case 1: in.brush_strength = std::max(0.01f, t); in.per_brush[(int)slot].strength = in.brush_strength; break;
        case 2: in.brush_hardness = std::max(0.01f, t); in.per_brush[(int)slot].hardness = in.brush_hardness; break;
        case 3: in.brush_spacing = std::max(0.05f, std::min(t, max_spacing_for(slot)));
                in.per_brush[(int)slot].spacing = in.brush_spacing; break;
    }
}

// Inset track with the orange gloss fill. Click or drag anywhere on it to set.
// Returns the new 0..1 value while dragged, or -1.
float y2k_track(const char* id, ImDrawList* dl, ImVec2 a, float w, float h, float pct, const char* tip) {
    bool hv, held;
    hit(id, ImVec2(a.x, a.y - 2), ImVec2(w, h + 4), &hv, &held);
    inset(dl, a, ImVec2(a.x + w, a.y + h), 3.0f);
    float fw = std::floor(w * std::max(0.0f, std::min(1.0f, pct)));
    if (fw >= 2.0f) {
        vgrad(dl, ImVec2(a.x + 1, a.y + 1), ImVec2(a.x + fw - 1, a.y + h - 1),
              y2k::orange_hi, IM_COL32(0xC9, 0x65, 0x0A, 255), 2.0f);
        dl->AddLine(ImVec2(a.x + 2, a.y + 1.5f), ImVec2(a.x + fw - 2, a.y + 1.5f), IM_COL32(255, 255, 255, 128));
    }
    if (hv && tip && !held) tooltip(a, ImVec2(a.x + w, a.y + h), TipSide::Below, tip, nullptr);
    if (held) return (ImGui::GetIO().MousePos.x - a.x) / w;
    return -1.0f;
}

void y2k_top(InputState& in, int W, const SkinStats& st) {
    if (!begin_shelf("##yTop", ImVec2(0, 0), ImVec2((float)W, kTopH))) { end_shelf(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float w = (float)W;
    vgrad(dl, ImVec2(0, 0), ImVec2(w, kTopH), IM_COL32(0x5D, 0x5D, 0x62, 255), IM_COL32(0x3F, 0x3F, 0x43, 255));
    dl->AddLine(ImVec2(0, kTopH - 0.5f), ImVec2(w, kTopH - 0.5f), y2k::edge);
    // Title strip.
    vgrad(dl, ImVec2(0, 0), ImVec2(w, kTitleH), IM_COL32(0x2F, 0x2F, 0x33, 255), IM_COL32(0x24, 0x24, 0x27, 255));
    dl->AddLine(ImVec2(0, kTitleH - 0.5f), ImVec2(w, kTitleH - 0.5f), y2k::edge);
    dl->AddLine(ImVec2(0, kTitleH + 0.5f), ImVec2(w, kTitleH + 0.5f), IM_COL32(255, 255, 255, 46));
    float x = 10.0f, ty = 4.0f;
    for (const char* c = "CHISEL"; *c; c++) {                       // letter-spaced wordmark
        char g[2] = {*c, 0};
        ytext(dl, true, 12.0f, ImVec2(x, ty), y2k::amber, g);
        x += ysize(true, 12.0f, g).x + 1.5f;
    }
    x += 12.0f;
    if (st.version && *st.version) {
        char v[24];
        const char* s = (*st.version == 'v' || *st.version == 'V') ? st.version + 1 : st.version;
        size_t i = 0;
        while (s[i] && s[i] != '-' && i < sizeof v - 1) { v[i] = s[i]; i++; }
        v[i] = '\0';
        ytext(dl, false, 11.0f, ImVec2(x, ty + 1), y2k::muted, v, false);
        x += ysize(false, 11.0f, v).x + 14.0f;
    }
    dl->AddLine(ImVec2(x, 5.0f), ImVec2(x, kTitleH - 5.0f), IM_COL32(0x55, 0x55, 0x55, 255));
    x += 14.0f;
    char name[64];
    project_name(name, sizeof name, st.project_path);
    ytext(dl, false, 11.0f, ImVec2(x, ty + 1), y2k::text, name);
    if (!st.project_path || !*st.project_path)
        ytext(dl, false, 11.0f, ImVec2(x + ysize(false, 11.0f, name).x + 2, ty + 1), y2k::amber, " *");
    const char* hint = "Hold S / W / A / O and drag to tune the brush";
    ytext(dl, false, 11.0f, ImVec2(w - 10.0f - ysize(false, 11.0f, hint).x, ty + 1), y2k::muted, hint, false);

    // Slider shelf row.
    const float cy = kTitleH + (kTopH - kTitleH) * 0.5f;
    // Mode well: Sculpt / Insert / Select / Paint, the live one an orange tile.
    static const char* modes[4] = {"Sculpt", "Insert", "Select", "Paint"};
    static const char* mkeys[4] = {"1", "2", "3", "4"};
    const bool smooth_on = in.is_smooth_active();
    const bool paint_on = in.current_brush == BrushType::PAINT && !smooth_on;
    int mode = in.interaction_mode == InputState::InteractionMode::INSERT ? 1
             : in.interaction_mode == InputState::InteractionMode::SELECT ? 2
             : paint_on ? 3 : 0;
    float bw[4], total = 4.0f;
    for (int i = 0; i < 4; i++) {
        bw[i] = ysize(true, 11.0f, modes[i]).x + ysize(false, 9.0f, mkeys[i]).x + 5.0f + 24.0f;
        total += bw[i] + (i ? 2.0f : 0.0f);
    }
    x = 10.0f;
    inset(dl, ImVec2(x, cy - 15), ImVec2(x + total, cy + 15), 4.0f);
    float bx = x + 2.0f;
    for (int i = 0; i < 4; i++) {
        bool hv, held;
        char id[16]; std::snprintf(id, sizeof id, "##ym%d", i);
        ImVec2 a(bx, cy - 13), b(bx + bw[i], cy + 13);
        if (hit(id, a, ImVec2(bw[i], 26), &hv, &held)) apply_mode(in, i);
        if (i == mode) gloss_tile(dl, a, b, 3.0f, true, false, false);
        else if (hv) dl->AddRectFilled(a, b, IM_COL32(255, 255, 255, held ? 10 : 22), 3.0f);
        ImU32 tc = i == mode ? y2k::on_text : y2k::text;
        ImVec2 ns = ysize(true, 11.0f, modes[i]);
        float tx = a.x + 12.0f, tyy = cy - ns.y * 0.5f;
        ytext(dl, true, 11.0f, ImVec2(tx, tyy), tc, modes[i], i != mode);
        ytext(dl, false, 9.0f, ImVec2(tx + ns.x + 5.0f, tyy + 2.0f), with_alpha(tc, 0.75f), mkeys[i], false);
        bx += bw[i] + 2.0f;
    }
    x += total + 14.0f;

    // The four feel sliders, draggable right here (or hold the key in the viewport).
    struct S { const char* label; const char* key; float pct; char val[12]; };
    S sl[4] = {{"Size", "S", in.brush_size / 500.0f, ""}, {"Strength", "W", in.brush_strength, ""},
               {"Hardness", "A", in.brush_hardness, ""}, {"Spacing", "O", in.brush_spacing, ""}};
    std::snprintf(sl[0].val, sizeof sl[0].val, "%.0f", in.brush_size);
    std::snprintf(sl[1].val, sizeof sl[1].val, "%.0f", in.brush_strength * 100.0f);
    std::snprintf(sl[2].val, sizeof sl[2].val, "%.0f", in.brush_hardness * 100.0f);
    std::snprintf(sl[3].val, sizeof sl[3].val, "%.0f", in.brush_spacing * 100.0f);
    const bool editing = in.interaction_mode == InputState::InteractionMode::EDIT;
    for (int i = 0; i < 4; i++) {
        const float sw = 168.0f;
        if (x + sw > w - 10.0f) break;                             // narrow window: drop what won't fit
        ytext(dl, true, 11.0f, ImVec2(x, cy - 17), y2k::text, sl[i].label);
        ytext(dl, false, 11.0f, ImVec2(x + ysize(true, 11.0f, sl[i].label).x + 5, cy - 17), y2k::muted, sl[i].key, false);
        ytext(dl, true, 11.0f, ImVec2(x + sw - ysize(true, 11.0f, sl[i].val).x, cy - 17), y2k::value, sl[i].val);
        char id[16]; std::snprintf(id, sizeof id, "##ys%d", i);
        float t = y2k_track(id, dl, ImVec2(x, cy + 1), sw, 14.0f, sl[i].pct, nullptr);
        if (t >= 0.0f && editing) set_feel(in, i, t);
        x += sw + 14.0f;
    }
    // Autosmooth checkbox.
    if (x + 120.0f <= w - 10.0f) {
        bool hv, held;
        ImVec2 ls = ysize(true, 11.0f, "Autosmooth");
        if (hit("##yAs", ImVec2(x, cy - 10), ImVec2(14 + 6 + ls.x + 18, 20), &hv, &held))
            in.autosmooth = !in.autosmooth;
        ImVec2 a(x, cy - 7), b(x + 14, cy + 7);
        inset(dl, a, b, 2.0f);
        if (in.autosmooth) {
            dl->AddLine(ImVec2(a.x + 3, a.y + 7), ImVec2(a.x + 6, a.y + 10), y2k::orange, 2.0f);
            dl->AddLine(ImVec2(a.x + 6, a.y + 10), ImVec2(a.x + 11, a.y + 3), y2k::orange, 2.0f);
        }
        ytext(dl, true, 11.0f, ImVec2(x + 20, cy - ls.y * 0.5f), hv ? IM_COL32_WHITE : y2k::text, "Autosmooth");
        ytext(dl, false, 11.0f, ImVec2(x + 20 + ls.x + 6, cy - ls.y * 0.5f), y2k::muted, "B", false);
    }
    end_shelf();
}

void y2k_left(InputState& in, int H, const AlphaLibrary* lib) {
    const float h = (float)H - kTopH - kBotH;
    if (!begin_shelf("##yLeft", ImVec2(0, kTopH), ImVec2(kLeftW, h))) { end_shelf(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    hgrad(dl, ImVec2(0, kTopH), ImVec2(kLeftW, kTopH + h), IM_COL32(0x4A, 0x4A, 0x4E, 255), IM_COL32(0x3B, 0x3B, 0x3F, 255));
    dl->AddLine(ImVec2(kLeftW - 0.5f, kTopH), ImVec2(kLeftW - 0.5f, kTopH + h), y2k::edge);
    dl->AddLine(ImVec2(0.5f, kTopH), ImVec2(0.5f, kTopH + h), IM_COL32(255, 255, 255, 31));

    const BrushType current = in.current_brush;
    const bool smooth_on = in.is_smooth_active();
    const auto mode = in.interaction_mode;
    const bool editing = mode == InputState::InteractionMode::EDIT;
    const float x0 = 8.0f, w = kLeftW - 16.0f;
    float y = kTopH + 10.0f;

    struct B { const char* name; const char* key; Icon icon; BrushType type; };
    static const B brushes[] = {
        {"Draw", "D", Icon::BrushDraw, BrushType::DRAW},       {"Clay", "T", Icon::BrushClay, BrushType::CLAY},
        {"Inflate", "I", Icon::BrushInflate, BrushType::INFLATE}, {"Crease", "C", Icon::BrushCrease, BrushType::CREASE},
        {"Pinch", "V", Icon::BrushPinch, BrushType::PINCH},    {"Move", "G", Icon::BrushMove, BrushType::MOVE},
        {"Limb", "H", Icon::BrushLimb, BrushType::LIMB},       {"Smooth", "SS", Icon::BrushSmooth, BrushType::SMOOTH},
        {"Mask", "M", Icon::BrushMask, BrushType::MASK},
    };

    // The big "current tool" tile.
    {
        Icon icon = Icon::BrushDraw;
        const char* nm = "Draw";
        if (mode == InputState::InteractionMode::INSERT)      { icon = Icon::ModeInsert; nm = "Insert"; }
        else if (mode == InputState::InteractionMode::SELECT) { icon = Icon::ModeSelect; nm = "Select"; }
        else if (current == BrushType::PAINT && !smooth_on)   { icon = Icon::ModePaint; nm = "Paint"; }
        else {
            BrushType t = smooth_on ? BrushType::SMOOTH : current;
            for (const B& b : brushes) if (b.type == t) { icon = b.icon; nm = b.name; }
        }
        ImVec2 a(x0, y), b(x0 + w, y + 92);
        gloss_tile(dl, a, b, 6.0f, true, false, false);
        ImVec2 ns = ysize(true, 12.0f, nm);
        draw_icon(dl, icon, ImVec2((a.x + b.x) * 0.5f, a.y + 36), 30.0f, y2k::on_text, 2.0f);
        ytext(dl, true, 12.0f, ImVec2((a.x + b.x - ns.x) * 0.5f, a.y + 60), y2k::on_text, nm, false);
        y += 92 + 10;
    }

    ytext(dl, true, 11.0f, ImVec2(x0 + 2, y), y2k::label, "Brushes");
    y += 18;
    const float tw = (w - 4.0f) * 0.5f, th = 44.0f;
    for (int i = 0; i < 9; i++) {
        const B& br = brushes[i];
        ImVec2 a(x0 + (i % 2) * (tw + 4.0f), y + (i / 2) * (th + 4.0f)), b(a.x + tw, a.y + th);
        bool hv, held;
        char id[16]; std::snprintf(id, sizeof id, "##yb%d", i);
        bool clicked = hit(id, a, ImVec2(tw, th), &hv, &held);
        bool on = br.type == BrushType::SMOOTH ? (smooth_on && editing)
                                               : (editing && current == br.type && !smooth_on);
        gloss_tile(dl, a, b, 5.0f, on, hv, held);
        ImU32 ic = on ? y2k::on_text : IM_COL32(0xEC, 0xEC, 0xEC, 255);
        draw_icon(dl, br.icon, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f - 1), 22.0f, ic);
        ImVec2 ks = ysize(false, 9.0f, br.key);
        ytext(dl, false, 9.0f, ImVec2(b.x - 3 - ks.x, b.y - 2 - ks.y), with_alpha(ic, 0.8f), br.key, false);
        if (hv) tooltip(a, b, TipSide::Right, br.name,
                        br.type == BrushType::SMOOTH ? "Shift Shift" : br.key);
        if (clicked) {
            if (br.type == BrushType::SMOOTH) {
                if (!in.smooth_locked) { in.smooth_locked = true; in.sync_live_settings(); }
                else in.clear_smooth_lock();
            } else {
                in.clear_smooth_lock();
                in.switch_brush(br.type);
                in.subtract_locked = false;
                in.interaction_mode = InputState::InteractionMode::EDIT;
            }
        }
    }
    y += 5 * (th + 4.0f) + 6.0f;

    // Alpha.
    const bool alpha_brush = current == BrushType::DRAW || current == BrushType::MASK ||
                             current == BrushType::PAINT;
    if (editing && !smooth_on && alpha_brush && lib && lib->count() > 0) {
        ytext(dl, true, 11.0f, ImVec2(x0 + 2, y), y2k::label, "Alpha");
        y += 18;
        ImVec2 a(x0, y), b(x0 + w, y + 72);
        bool hv, held;
        if (hit("##yAlpha", a, ImVec2(w, 72), &hv, &held)) ImGui::OpenPopup("##alphapick");
        inset(dl, a, b, 6.0f);
        dl->AddRect(a, b, y2k::edge, 6.0f);
        int ai = std::max(0, std::min(in.active_alpha, lib->count() - 1));
        const AlphaEntry& ae = lib->get(ai);
        draw_alpha_preview(dl, ImVec2((a.x + b.x) * 0.5f - 26, a.y + 10), 52.0f, ae.preview);
        if (hv) tooltip(a, b, TipSide::Right, ae.name.c_str(), nullptr);
        alpha_picker_popup(in, lib, ImVec2(kLeftW + 6.0f, y - 8.0f));
        y += 72 + 10;
    }

    // Tool options.
    const bool paint_on = editing && current == BrushType::PAINT && !smooth_on;
    const bool clay_on = editing && current == BrushType::CLAY && !smooth_on;
    if (clay_on) {
        ytext(dl, true, 11.0f, ImVec2(x0 + 2, y), y2k::label, "Melt");
        char v[12]; std::snprintf(v, sizeof v, "%.2f", in.clay_melt);
        ytext(dl, true, 11.0f, ImVec2(x0 + w - ysize(true, 11.0f, v).x, y), y2k::value, v);
        y += 17;
        float t = y2k_track("##yMelt", dl, ImVec2(x0, y), w, 14.0f, in.clay_melt,
                            "How strongly clay cuts raised areas down to its plane");
        if (t >= 0.0f) in.clay_melt = std::max(0.0f, std::min(1.0f, t));
        y += 24;
    }
    if (paint_on) {
        ytext(dl, true, 11.0f, ImVec2(x0 + 2, y), y2k::label, "Paint");
        y += 18;
        const float hw = (w - 4.0f) * 0.5f;
        bool hv, held;
        ImVec2 a(x0, y);
        if (hit("##yPc", a, ImVec2(hw, 22), &hv, &held)) in.paint_target_density = false;
        gloss_tile(dl, a, ImVec2(a.x + hw, a.y + 22), 3.0f, !in.paint_target_density, hv, held);
        ImVec2 s1 = ysize(true, 10.0f, "Colour");
        ytext(dl, true, 10.0f, ImVec2(a.x + (hw - s1.x) * 0.5f, a.y + 5), !in.paint_target_density ? y2k::on_text : y2k::text,
              "Colour", in.paint_target_density);
        a = ImVec2(x0 + hw + 4, y);
        if (hit("##yPd", a, ImVec2(hw, 22), &hv, &held)) { in.paint_target_density = true; in.color_pick_active = false; }
        gloss_tile(dl, a, ImVec2(a.x + hw, a.y + 22), 3.0f, in.paint_target_density, hv, held);
        ImVec2 s2 = ysize(true, 10.0f, "Density");
        ytext(dl, true, 10.0f, ImVec2(a.x + (hw - s2.x) * 0.5f, a.y + 5), in.paint_target_density ? y2k::on_text : y2k::text,
              "Density", !in.paint_target_density);
        y += 30;
        ImGui::SetCursorScreenPos(ImVec2(x0, y));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 6));
        if (!in.paint_target_density) {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(13, 13));
            ImGui::ColorEdit3("##ypaintA", in.paint_color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine();
            ImGui::ColorEdit3("##ypaintB", in.paint_color_alt, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::PopStyleVar();
            y += 34;
            ytext(dl, false, 10.0f, ImVec2(x0, y), in.color_pick_active ? y2k::amber : y2k::muted,
                  in.color_pick_active ? "Picking..." : "Q/E swap  C pick", false);
        } else {
            ImGui::PushFont(nullptr, 10.0f);
            ImGui::SetNextItemWidth(w);
            ImGui::SliderFloat("##ydC", &in.density_coarse_mult, 1.0f, 4.0f, "green x%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remesh edge length where painted green (coarse)");
            ImGui::SetNextItemWidth(w);
            ImGui::SliderFloat("##ydF", &in.density_fine_mult, 0.2f, 1.0f, "red x%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remesh edge length where painted red (dense)");
            ImGui::PopFont();
        }
        ImGui::PopStyleVar(2);
    }
    if (mode == InputState::InteractionMode::INSERT) {
        ytext(dl, true, 11.0f, ImVec2(x0 + 2, y), y2k::label, "Shape");
        y += 18;
        static const struct { const char* id; int kind; const char* tip; InputState::InsertShape s; } shapes[] = {
            {"##ySh0", 0, "Sphere", InputState::InsertShape::SPHERE},
            {"##ySh1", 1, "Box", InputState::InsertShape::BOX},
            {"##ySh2", 2, "Cylinder", InputState::InsertShape::CYLINDER},
        };
        const float sw = (w - 8.0f) / 3.0f;
        for (int i = 0; i < 3; i++) {
            ImVec2 a(x0 + i * (sw + 4.0f), y);
            bool hv, held;
            if (hit(shapes[i].id, a, ImVec2(sw, 36), &hv, &held)) in.insert_shape = shapes[i].s;
            bool on = in.insert_shape == shapes[i].s;
            gloss_tile(dl, a, ImVec2(a.x + sw, a.y + 36), 4.0f, on, hv, held);
            ImVec2 c(a.x + sw * 0.5f, a.y + 18);
            ImU32 col = on ? y2k::on_text : y2k::text;
            if (shapes[i].kind == 0) { dl->AddCircle(c, 9.0f, col, 0, 1.6f); dl->AddEllipse(c, ImVec2(9, 3), col, 0, 0, 1.2f); }
            else if (shapes[i].kind == 1) draw_icon(dl, Icon::ModeInsert, c, 22.0f, col);
            else {
                dl->AddEllipse(ImVec2(c.x, c.y - 7), ImVec2(7, 2.6f), col, 0, 0, 1.5f);
                dl->AddLine(ImVec2(c.x - 7, c.y - 7), ImVec2(c.x - 7, c.y + 7), col, 1.5f);
                dl->AddLine(ImVec2(c.x + 7, c.y - 7), ImVec2(c.x + 7, c.y + 7), col, 1.5f);
                dl->AddEllipse(ImVec2(c.x, c.y + 7), ImVec2(7, 2.6f), col, 0, 0, 1.5f);
            }
            if (hv) tooltip(a, ImVec2(a.x + sw, a.y + 36), TipSide::Right, shapes[i].tip, nullptr);
        }
        y += 44;
        ytext(dl, false, 10.0f, ImVec2(x0, y), y2k::muted, "Click the model", false);
    }
    if (mode == InputState::InteractionMode::SELECT) {
        ytext(dl, false, 10.0f, ImVec2(x0, y), y2k::muted, "Click to pick,", false);
        ytext(dl, false, 10.0f, ImVec2(x0, y + 13), y2k::muted, "drag to move.", false);
        ytext(dl, false, 10.0f, ImVec2(x0, y + 26), y2k::muted, "Q / E rotate.", false);
    }
    end_shelf();
}

void y2k_right(InputState& in, int W, int H, MultiresInfo mres) {
    const float h = (float)H - kTopH - kBotH, x = (float)W - kRightW;
    if (!begin_shelf("##yRight", ImVec2(x, kTopH), ImVec2(kRightW, h))) { end_shelf(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    hgrad(dl, ImVec2(x, kTopH), ImVec2(x + kRightW, kTopH + h), IM_COL32(0x3B, 0x3B, 0x3F, 255), IM_COL32(0x4A, 0x4A, 0x4E, 255));
    dl->AddLine(ImVec2(x + 0.5f, kTopH), ImVec2(x + 0.5f, kTopH + h), y2k::edge);
    dl->AddLine(ImVec2(x + kRightW - 0.5f, kTopH), ImVec2(x + kRightW - 0.5f, kTopH + h), IM_COL32(255, 255, 255, 31));

    struct C { const char* id; const char* name; const char* tip; const char* key; Icon icon; bool sep; };
    const C cmds[] = {
        {"##yUndo", "Undo", "Undo", "Ctrl+Z", Icon::CmdUndo, false},
        {"##yRedo", "Redo", "Redo", "Ctrl+Shift+Z", Icon::CmdRedo, false},
        {"##yUp",   "Divide", "Subdivision up", "Ctrl+D", Icon::CmdLevelUp, true},
        {"##yDn",   "Lower", "Subdivision down", "Shift+D", Icon::CmdLevelDown, false},
        {"##yMrg",  "Merge", "SDF merge", "J", Icon::CmdMerge, false},
        {"##yMir",  "Mirror X", "Mirror X", "X", Icon::ViewMirrorX, true},
        {"##yVis",  "Paint", in.paint_visible ? "Paint shown while sculpting" : "Paint hidden while sculpting",
                    nullptr, in.paint_visible ? Icon::ViewPaintShow : Icon::ViewPaintHide, false},
        {"##ySave", "Save", "Save", "Ctrl+S", Icon::CmdSave, true},
        {"##ySavI", "Save +", "Save numbered copy", nullptr, Icon::CmdSaveCopy, false},
        {"##yOpen", "Open", "Open", "Ctrl+O", Icon::CmdOpen, false},
        {"##yExp",  "Export", "Export", "Ctrl+E", Icon::CmdExport, false},
        {"##yKeys", "Keys", "Shortcuts", nullptr, Icon::UiShortcuts, true},
        {"##yMenu", "Menu", "Menu", nullptr, Icon::UiMenu, false},
    };
    const int n = (int)(sizeof cmds / sizeof cmds[0]);
    int nsep = 0;
    for (const C& c : cmds) nsep += c.sep ? 1 : 0;
    const float bh = std::max(30.0f, std::min(48.0f, (h - 20.0f - nsep * 14.0f) / n - 4.0f));
    const float bx = x + 8.0f, bw = kRightW - 16.0f;
    float y = kTopH + 10.0f, menu_y = y;
    for (int i = 0; i < n; i++) {
        const C& c = cmds[i];
        if (c.sep) {
            dl->AddLine(ImVec2(bx + 4, y + 5.5f), ImVec2(bx + bw - 4, y + 5.5f), y2k::well);
            dl->AddLine(ImVec2(bx + 4, y + 6.5f), ImVec2(bx + bw - 4, y + 6.5f), IM_COL32(255, 255, 255, 26));
            y += 14.0f;
        }
        bool enabled = (i == 2 || i == 3) ? in.mesh_locked : true;
        bool on = (i == 5 && in.mirror_x) || (i == 11 && ImGui::IsPopupOpen("##helppopup")) ||
                  (i == 12 && ImGui::IsPopupOpen("##burgermenu"));
        bool hv, held;
        ImVec2 a(bx, y), b(bx + bw, y + bh);
        bool clicked = hit(c.id, a, ImVec2(bw, bh), &hv, &held, enabled);
        gloss_tile(dl, a, b, 5.0f, on, hv && enabled, held);
        ImU32 col = on ? y2k::on_text : (enabled ? IM_COL32(0xEC, 0xEC, 0xEC, 255) : IM_COL32(0x80, 0x80, 0x84, 255));
        const float ip = bh >= 40.0f ? 20.0f : 16.0f;
        ImVec2 ns = ysize(true, 10.0f, c.name);
        float stack = ip + 3.0f + ns.y;
        draw_icon(dl, c.icon, ImVec2((a.x + b.x) * 0.5f, a.y + (bh - stack) * 0.5f + ip * 0.5f), ip, col);
        ytext(dl, true, 10.0f, ImVec2((a.x + b.x - ns.x) * 0.5f, a.y + (bh - stack) * 0.5f + ip + 3.0f), col, c.name, !on);
        if (hv) tooltip(a, b, TipSide::Left, c.tip, c.key);
        if (i == 12) menu_y = y;
        if (clicked) {
            switch (i) {
                case 0: in.undo_requested = true; break;
                case 1: in.redo_requested = true; break;
                case 2: in.level_switch_delta = +1; break;
                case 3: in.level_switch_delta = -1; break;
                case 4: in.voxel_merge_confirm_pending = true; break;
                case 5: in.mirror_x = !in.mirror_x; break;
                case 6: in.paint_visible = !in.paint_visible; break;
                case 7: in.save_requested = true; break;
                case 8: in.save_incremental_requested = true; break;
                case 9: in.import_dialog_active = true; break;
                case 10: in.export_dialog_active = true; break;
                case 11: ImGui::OpenPopup("##helppopup"); break;
                case 12: ImGui::OpenPopup("##burgermenu"); break;
            }
        }
        y += bh + 4.0f;
    }
    (void)mres;
    if (!in.help_seen) { in.help_seen = true; ImGui::OpenPopup("##helppopup"); }

    static bool prev_menu_open = false;
    bool menu_open = ImGui::IsPopupOpen("##burgermenu");
    bool sync_tabs = menu_open && !prev_menu_open;
    prev_menu_open = menu_open;
    in.settings_menu_open = menu_open;
    // The menu opens out of the shelf to the left, bottom-aligned with the button.
    ImGui::SetNextWindowPos(ImVec2(x - 6.0f, menu_y + bh), ImGuiCond_Always, ImVec2(1, 1));
    ImGui::SetNextWindowSizeConstraints(ImVec2(300, 0), ImVec2(FLT_MAX, (float)H - kBotH - 16.0f));
    if (begin_skin_popup("##burgermenu", ImVec2(12, 12))) {
        draw_settings_menu_items(in, mres, sync_tabs);
        end_skin_popup();
    }
    draw_help_popup(in, W, H);
    end_shelf();
}

void y2k_bottom(const InputState& in, int W, int H, MultiresInfo mres, const SkinStats& st) {
    const float y0 = (float)H - kBotH;
    if (!begin_shelf("##yBottom", ImVec2(0, y0), ImVec2((float)W, kBotH))) { end_shelf(); return; }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    vgrad(dl, ImVec2(0, y0), ImVec2((float)W, (float)H), IM_COL32(0x3A, 0x3A, 0x3E, 255), IM_COL32(0x2A, 0x2A, 0x2D, 255));
    dl->AddLine(ImVec2(0, y0 + 0.5f), ImVec2((float)W, y0 + 0.5f), y2k::edge);
    dl->AddLine(ImVec2(0, y0 + 1.5f), ImVec2((float)W, y0 + 1.5f), IM_COL32(255, 255, 255, 26));
    const float cy = y0 + kBotH * 0.5f;
    float x = 10.0f;
    auto seg = [&](const char* label, const char* value, ImU32 vcol, const char* tail, bool sep_after) {
        x += 12.0f;
        ImVec2 ls = ysize(false, 11.0f, label);
        ytext(dl, false, 11.0f, ImVec2(x, cy - ls.y * 0.5f), y2k::text, label);
        x += ls.x + 4.0f;
        ImVec2 vs = ysize(true, 11.0f, value);
        ytext(dl, true, 11.0f, ImVec2(x, cy - vs.y * 0.5f), vcol, value);
        x += vs.x;
        if (tail) {
            ImVec2 ts = ysize(false, 11.0f, tail);
            ytext(dl, false, 11.0f, ImVec2(x + 4.0f, cy - ts.y * 0.5f), y2k::text, tail);
            x += 4.0f + ts.x;
        }
        x += 12.0f;
        if (sep_after) dl->AddLine(ImVec2(x + 0.5f, cy - 7), ImVec2(x + 0.5f, cy + 7), IM_COL32(0x55, 0x55, 0x55, 255));
    };
    char tris[24], verts[24], lvl[12], of[16];
    fmt_thousands(tris, sizeof tris, st.tris);
    fmt_thousands(verts, sizeof verts, st.verts);
    std::snprintf(lvl, sizeof lvl, "%d", st.level);
    if (mres.locked) std::snprintf(of, sizeof of, "of %d", mres.lmax); else of[0] = '\0';
    seg("Tris", tris, y2k::text, nullptr, true);
    seg("Verts", verts, y2k::text, nullptr, true);
    seg("Level", lvl, y2k::text, of[0] ? of : nullptr, true);
    seg("Mirror", !in.mirror_x ? "Off" : in.mirror_topological ? "Topological" : "World",
        in.mirror_x ? y2k::amber : y2k::muted, nullptr, true);
    seg("Normals", in.fast_normals ? "Fast" : "Interpolated", y2k::text, nullptr, false);
    if (in.show_fps) {
        char f[16]; std::snprintf(f, sizeof f, "%.0f fps", st.fps);
        ImVec2 fs = ysize(false, 11.0f, f);
        ytext(dl, false, 11.0f, ImVec2((float)W - 22.0f - fs.x, cy - fs.y * 0.5f), y2k::muted, f);
    }
    end_shelf();
}

// The viewport sits sunk into the shelves: dark rim, inner top shadow, a lit lip below.
void y2k_bevel(int W, int H) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 a(kLeftW, kTopH), b((float)W - kRightW, (float)H - kBotH);
    dl->AddRectFilledMultiColor(a, ImVec2(b.x, a.y + 8), IM_COL32(0, 0, 0, 140), IM_COL32(0, 0, 0, 140),
                                IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, 0));
    dl->AddRectFilledMultiColor(a, ImVec2(a.x + 5, b.y), IM_COL32(0, 0, 0, 90), IM_COL32(0, 0, 0, 0),
                                IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, 90));
    dl->AddRect(ImVec2(a.x - 0.5f, a.y - 0.5f), ImVec2(b.x + 0.5f, b.y + 0.5f), IM_COL32(0x12, 0x12, 0x14, 255));
    dl->AddLine(ImVec2(a.x, b.y - 0.5f), ImVec2(b.x, b.y - 0.5f), IM_COL32(255, 255, 255, 20));
}

void draw_y2k_ui(InputState& input, int W, int H, const AlphaLibrary* lib, MultiresInfo mres,
                 const SkinStats& st) {
    y2k_bevel(W, H);
    y2k_top(input, W, st);
    y2k_left(input, H, lib);
    y2k_right(input, W, H, mres);
    y2k_bottom(input, W, H, mres, st);
}

void viewfinder(int win_w, int win_h) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const float ix = std::floor(win_w * 0.2625f) + 0.5f, iy = std::floor(win_h * 0.12f) + 0.5f;
    const float L = 24.0f, th = 1.5f;
    const ImU32 c = IM_COL32(236, 234, 242, 77);
    const float x0 = ix, x1 = win_w - ix, y0 = iy, y1 = win_h - iy;
    dl->AddLine(ImVec2(x0, y0), ImVec2(x0 + L, y0), c, th); dl->AddLine(ImVec2(x0, y0), ImVec2(x0, y0 + L), c, th);
    dl->AddLine(ImVec2(x1, y0), ImVec2(x1 - L, y0), c, th); dl->AddLine(ImVec2(x1, y0), ImVec2(x1, y0 + L), c, th);
    dl->AddLine(ImVec2(x0, y1), ImVec2(x0 + L, y1), c, th); dl->AddLine(ImVec2(x0, y1), ImVec2(x0, y1 - L), c, th);
    dl->AddLine(ImVec2(x1, y1), ImVec2(x1 - L, y1), c, th); dl->AddLine(ImVec2(x1, y1), ImVec2(x1, y1 - L), c, th);
}

} // namespace

// ============================================================================

namespace {
// ImGui sizes a font by its line box (hhea ascender - descender); CSS, and so the
// handoff's specs, size it by the em. For Plex that box is 1.3 em, so "14 px" in
// ImGui came out like 11 px on the boards. Read the ratio from the font's own
// tables and hand it to ExtraSizeScale, so every size in this file means em px.
float em_correction(const unsigned char* ttf, int size) {
    auto u16 = [&](int o) { return (o + 2 <= size) ? (int)((ttf[o] << 8) | ttf[o + 1]) : 0; };
    auto s16 = [&](int o) { int v = u16(o); return v >= 0x8000 ? v - 0x10000 : v; };
    auto u32 = [&](int o) {
        return (o + 4 <= size) ? (int)(((unsigned)ttf[o] << 24) | (ttf[o + 1] << 16) |
                                       (ttf[o + 2] << 8) | ttf[o + 3]) : 0;
    };
    int head = 0, hhea = 0, n = u16(4);
    for (int i = 0; i < n; i++) {
        int rec = 12 + i * 16;
        if (std::memcmp(ttf + rec, "head", 4) == 0) head = u32(rec + 8);
        if (std::memcmp(ttf + rec, "hhea", 4) == 0) hhea = u32(rec + 8);
    }
    if (!head || !hhea) return 1.0f;
    int upem = u16(head + 18);
    int box = s16(hhea + 4) - s16(hhea + 6);
    return (upem > 0 && box > 0) ? (float)box / (float)upem : 1.0f;
}

} // namespace

void ui_skin_load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    // ProggyClean first, explicitly: it is what the classic skin always rendered
    // with (ImGui picks the bitmap default at the old 13 px size).
    g_classic_font = io.Fonts->AddFontDefaultBitmap();
    ImFontConfig cfg;
    cfg.FontDataOwnedByAtlas = false;      // static arrays from ui_fonts_generated.h
    cfg.OversampleH = 2;
    auto add = [&](const unsigned char* data, int size, const char* name) {
        std::snprintf(cfg.Name, sizeof cfg.Name, "%s", name);
        cfg.ExtraSizeScale = em_correction(data, size);
        return io.Fonts->AddFontFromMemoryTTF((void*)data, size, 14.0f, &cfg);
    };
    g_fonts[(int)UiFont::Sans]         = add(k_font_IBMPlexSans_Regular,  k_font_IBMPlexSans_Regular_size,  "IBM Plex Sans");
    g_fonts[(int)UiFont::SansSemibold] = add(k_font_IBMPlexSans_SemiBold, k_font_IBMPlexSans_SemiBold_size, "IBM Plex Sans SemiBold");
    g_fonts[(int)UiFont::Mono]         = add(k_font_IBMPlexMono_Regular,  k_font_IBMPlexMono_Regular_size,  "IBM Plex Mono");
    g_fonts[(int)UiFont::MonoMedium]   = add(k_font_IBMPlexMono_Medium,   k_font_IBMPlexMono_Medium_size,   "IBM Plex Mono Medium");

    // DOS: the HUD's own 8x8 font (ChiselBitmap.ttf, built from text_overlay.cpp's
    // font_data[]), unhinted pixels kept hard. It only covers ASCII 32-122, so Plex
    // Mono is merged in underneath for anything else (braces, tilde, accents, the
    // degree sign in the FOV slider).
    ImFontConfig bc;
    bc.FontDataOwnedByAtlas = false;
    bc.OversampleH = bc.OversampleV = 1;
    bc.PixelSnapH = true;
    std::snprintf(bc.Name, sizeof bc.Name, "Chisel Bitmap");
    bc.ExtraSizeScale = em_correction(k_font_ChiselBitmap, k_font_ChiselBitmap_size);
    g_bitmap_font = io.Fonts->AddFontFromMemoryTTF((void*)k_font_ChiselBitmap, k_font_ChiselBitmap_size,
                                                   16.0f, &bc);
    ImFontConfig mc = cfg;
    mc.MergeMode = true;
    mc.ExtraSizeScale = em_correction(k_font_IBMPlexMono_Regular, k_font_IBMPlexMono_Regular_size);
    std::snprintf(mc.Name, sizeof mc.Name, "IBM Plex Mono (DOS fallback)");
    io.Fonts->AddFontFromMemoryTTF((void*)k_font_IBMPlexMono_Regular, k_font_IBMPlexMono_Regular_size,
                                   16.0f, &mc);
    // Y2K: Verdana is what board E calls for, but it is Microsoft's and can't ship
    // with Chisel, so it is borrowed from the OS when present. DejaVu Sans (a Vera
    // descendant, close in width and feel) is the usual Linux stand-in; failing both,
    // the skin falls back to Plex. Nothing found on the web build, by design.
    {
        std::string win;
        if (const char* wd = std::getenv("WINDIR")) win = std::string(wd) + "\\Fonts\\";
        const std::string reg[] = {
            win + "verdana.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf",
            "/usr/share/fonts/TTF/verdana.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        };
        const std::string bld[] = {
            win + "verdanab.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold.ttf",
            "/usr/share/fonts/TTF/verdanab.ttf",
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        };
        auto load_first = [&](const std::string* paths, int n, const char* name) -> ImFont* {
            for (int i = 0; i < n; i++) {
                if (paths[i].empty() || paths[i][0] == '\\') continue;
                FILE* f = std::fopen(paths[i].c_str(), "rb");
                if (!f) continue;
                std::fseek(f, 0, SEEK_END);
                long sz = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (sz <= 0 || sz > (32L << 20)) { std::fclose(f); continue; }
                unsigned char* buf = (unsigned char*)IM_ALLOC((size_t)sz);   // atlas frees it
                size_t got = std::fread(buf, 1, (size_t)sz, f);
                std::fclose(f);
                if (got != (size_t)sz) { IM_FREE(buf); continue; }
                ImFontConfig yc;
                yc.OversampleH = 2;
                yc.ExtraSizeScale = em_correction(buf, (int)sz);
                std::snprintf(yc.Name, sizeof yc.Name, "%s", name);
                if (ImFont* font = io.Fonts->AddFontFromMemoryTTF(buf, (int)sz, 12.0f, &yc)) return font;
            }
            return nullptr;
        };
        g_y2k_font = load_first(reg, (int)(sizeof reg / sizeof reg[0]), "Y2K regular");
        g_y2k_bold = load_first(bld, (int)(sizeof bld / sizeof bld[0]), "Y2K bold");
        if (!g_y2k_font) g_y2k_font = g_fonts[(int)UiFont::Sans];
        if (!g_y2k_bold) g_y2k_bold = g_fonts[(int)UiFont::SansSemibold];
    }

}

void ui_skin_begin_frame(const InputState& input) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& s = ImGui::GetStyle();
    if (!g_style_ready) {
        g_classic_style = s; g_style_ready = true;
        g_active_skin = InputState::UiSkin::CLASSIC;    // what the context starts as
        set_theme(false);
    }
    InputState::UiSkin want = g_fonts[0] ? input.ui_skin : InputState::UiSkin::CLASSIC;
    if (want == InputState::UiSkin::DOS && !g_bitmap_font) want = InputState::UiSkin::MODERN;
    if (want != g_active_skin) {
        // Keep the app's own non-colour tweaks (hover delays) across the swap.
        float hn = s.HoverDelayNormal, hs = s.HoverDelayShort;
        s = g_classic_style;
        switch (want) {
            case InputState::UiSkin::MODERN:
                apply_modern_style();
                io.FontDefault = g_fonts[(int)UiFont::Sans];
                s.FontSizeBase = 14.0f;
                break;
            case InputState::UiSkin::DOS:
                apply_dos_style();
                io.FontDefault = g_bitmap_font;
                s.FontSizeBase = 16.0f;
                break;
            case InputState::UiSkin::Y2K:
                apply_y2k_style();
                io.FontDefault = g_y2k_font;
                s.FontSizeBase = 12.0f;
                break;
            case InputState::UiSkin::CLASSIC:
                io.FontDefault = g_classic_font;
                s.FontSizeBase = 13.0f;
                break;
        }
        s.HoverDelayNormal = hn; s.HoverDelayShort = hs;
        set_theme(want == InputState::UiSkin::DOS, want == InputState::UiSkin::Y2K);
        g_active_skin = want;
    }
    g_mat = material_from(input);
}

ImFont* ui_skin_font(UiFont f) {
    return g_active_skin == InputState::UiSkin::CLASSIC ? nullptr : F(f);
}

float ui_skin_menu_item_width(float w) { return g_dos ? w * 2.2f : (g_y2k ? w * 1.2f : w); }

void ui_skin_backdrop_params(const InputState& input, float* blur_px, float* saturate) {
    float m = std::max(0.0f, std::min(1.0f, input.ui_material / 100.0f));
    *blur_px = input.ui_skin != InputState::UiSkin::MODERN ? 0.0f : 22.0f * (1.0f - m);
    if (*blur_px < 0.66f) *blur_px = 0.0f;
    *saturate = 1.0f + 0.5f * (1.0f - m);
}

void ui_skin_popup_background() {
    window_panel(16.0f);
}

void draw_appearance_menu_items(InputState& input) {
    ImFont* semi = ui_skin_font(UiFont::SansSemibold);
    if (semi) ImGui::PushFont(semi, (g_dos || g_y2k) ? 0.0f : 11.0f);
    if (g_dos)      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.333f, 1.0f), "APPEARANCE");
    else if (g_y2k) ImGui::TextColored(ImVec4(1.0f, 0.70f, 0.28f, 1.0f), "Appearance");
    else            ImGui::TextDisabled("APPEARANCE");
    if (semi) ImGui::PopFont();

    int skin = (int)input.ui_skin;
    ImGui::RadioButton("Modern", &skin, (int)InputState::UiSkin::MODERN);
    ImGui::SameLine();
    ImGui::RadioButton("DOS", &skin, (int)InputState::UiSkin::DOS);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Chisel's own 8x8 font, the CGA palette and pixel icons.");
    ImGui::SameLine();
    ImGui::RadioButton("2000s", &skin, (int)InputState::UiSkin::Y2K);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Early-2000s 3D app: docked shelves, glossy tiles, orange.");
    ImGui::SameLine();
    ImGui::RadioButton("Classic", &skin, (int)InputState::UiSkin::CLASSIC);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The original button islands and bitmap-font HUD.");
    input.ui_skin = (InputState::UiSkin)skin;
    if (input.ui_skin == InputState::UiSkin::DOS)
        ImGui::Checkbox("Viewfinder corners", &input.ui_viewfinder);
    if (input.ui_skin != InputState::UiSkin::MODERN) return;

    const float m = input.ui_material;
    const char* mname = m < 20.0f ? "Clear" : (m < 75.0f ? "Tinted" : "Solid");
    const float w = 268.0f;
    ImGui::TextUnformatted("Panel material");
    ImGui::SameLine(w - ImGui::CalcTextSize(mname).x + ImGui::GetStyle().WindowPadding.x * 0.0f);
    ImFont* mono = ui_skin_font(UiFont::Mono);
    if (mono) ImGui::PushFont(mono, 0.0f);
    ImGui::TextColored(ImVec4(0xC9 / 255.0f, 0xC6 / 255.0f, 0xD4 / 255.0f, 1.0f), "%s", mname);
    if (mono) ImGui::PopFont();
    ImGui::SetNextItemWidth(w);
    ImGui::SliderFloat("##uiMaterial", &input.ui_material, 0.0f, 100.0f, "%.0f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Clear glass blurs the model behind the panels (one extra\n"
                          "GPU pass per frame); Solid turns the blur off entirely.");
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, 12.0f));
        const float fs = 11.0f;
        ImVec2 t = text_size(UiFont::Sans, fs, "Tinted"), s = text_size(UiFont::Sans, fs, "Solid");
        text_at(dl, UiFont::Sans, fs, ImVec2(p.x, p.y - 4.0f), kMuted, "Clear glass");
        text_at(dl, UiFont::Sans, fs, ImVec2(p.x + (w - t.x) * 0.5f, p.y - 4.0f), kMuted, "Tinted");
        text_at(dl, UiFont::Sans, fs, ImVec2(p.x + w - s.x, p.y - 4.0f), kMuted, "Solid");
    }
    ImGui::Checkbox("Viewfinder corners", &input.ui_viewfinder);
    ImGui::Checkbox("Springy motion", &input.ui_motion);
    if (input.ui_motion && os_reduced_motion()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(off: system reduced motion)");
    }
}

void draw_modern_ui(InputState& input, int win_w, int win_h, const AlphaLibrary* alpha_lib,
                    MultiresInfo mres, const SkinStats& stats) {
    // DOS and Y2K don't bounce: neither era did.
    g_motion = !g_dos && !g_y2k && input.ui_motion && !os_reduced_motion();

    if (g_y2k) {
        draw_y2k_ui(input, win_w, win_h, alpha_lib, mres, stats);
    } else {
        if (input.ui_viewfinder) viewfinder(win_w, win_h);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        rail(input, win_h, alpha_lib);
        file_panel(input, stats);
        command_panel(input, win_w, win_h, mres, stats);
        float bh = brush_panel(input, win_h);
        options_panel(input, win_h, bh);
        status_panel(input, win_w, win_h, stats, g_brush_right, bh);
        ImGui::PopStyleVar();
    }

    // Paint colour wheel at the cursor on right-click (same trigger as classic).
    {
        const bool smooth_on = input.is_smooth_active();
        static bool prev_rmb = false;
        bool rmb_edge = input.mouse2_down && !prev_rmb;
        prev_rmb = input.mouse2_down;
        if (input.current_brush == BrushType::PAINT && !smooth_on && rmb_edge &&
            !input.zoom_modifier_held()) {
            ImGui::SetNextWindowPos(ImVec2((float)input.mouse_x, (float)input.mouse_y));
            ImGui::OpenPopup("##paintswatch");
        }
        if (begin_skin_popup("##paintswatch", ImVec2(12, 12), false)) {
            ImGui::ColorPicker3("##paintpick", input.paint_color,
                                ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoInputs |
                                ImGuiColorEditFlags_NoLabel);
            end_skin_popup();
        }
    }

    notifications(input, win_w, win_h);
    slider_hud(input);
}
