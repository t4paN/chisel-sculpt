#include "ui_overlay.h"
#include "text_overlay.h"
#include "input.h"
#include "brush_alpha.h"
#include "imgui.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// CGA 16-color palette (IBM PC, 1981). Hex values mapped to 0..1 floats.
namespace cga {
    // 0x00 black, 0xAA = 0.667, 0x55 = 0.333, 0xFF = 1.0
    constexpr float black[3]         = {0.000f, 0.000f, 0.000f};
    constexpr float blue[3]          = {0.000f, 0.000f, 0.667f};
    constexpr float green[3]         = {0.000f, 0.667f, 0.000f};
    constexpr float cyan[3]          = {0.000f, 0.667f, 0.667f};
    constexpr float red[3]           = {0.667f, 0.000f, 0.000f};
    constexpr float magenta[3]       = {0.667f, 0.000f, 0.667f};
    constexpr float brown[3]         = {0.667f, 0.333f, 0.000f};
    constexpr float light_gray[3]    = {0.667f, 0.667f, 0.667f};
    constexpr float dark_gray[3]     = {0.333f, 0.333f, 0.333f};
    constexpr float light_blue[3]    = {0.333f, 0.333f, 1.000f};
    constexpr float light_green[3]   = {0.333f, 1.000f, 0.333f};
    constexpr float light_cyan[3]    = {0.333f, 1.000f, 1.000f};
    constexpr float light_red[3]     = {1.000f, 0.333f, 0.333f};
    constexpr float light_magenta[3] = {1.000f, 0.333f, 1.000f};
    constexpr float yellow[3]        = {1.000f, 1.000f, 0.333f};
    constexpr float white[3]         = {1.000f, 1.000f, 1.000f};
}

#define CGA(c) cga::c[0], cga::c[1], cga::c[2]

void draw_quit_dialog(TextOverlay& text, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 180.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    text.draw_text("Quit? Y / N / ESC", msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(yellow), 1.0f);
}

void draw_remesh_confirm(TextOverlay& text, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 260.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    text.draw_text("Remesh stretched regions? Wipes stack. (Y/N)", msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(yellow), 1.0f);
}

void draw_remesh_progress(TextOverlay& text, int win_w, int win_h) {
    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 180.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    text.draw_text("Remeshing...", msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(light_green), 1.0f);
}

void draw_voxel_merge_confirm(TextOverlay& text, int resolution, int n_selected,
                             int n_unselected, bool surface_nets, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float cy = (float)win_h * 0.5f;

    char line[160];
    std::snprintf(line, sizeof(line),
                  "Voxel-merge %d mesh%s into one watertight mesh?",
                  n_selected, n_selected == 1 ? "" : "es");
    text.draw_text(line, cx - 360.0f, cy - 60.0f, scale, win_w, win_h, CGA(yellow), 1.0f);

    text.draw_text("Wipes their multires.", cx - 360.0f, cy - 30.0f, scale,
                  win_w, win_h, CGA(light_gray), 1.0f);

    std::snprintf(line, sizeof(line), "Resolution: %d   ( [ - / + ] )", resolution);
    text.draw_text(line, cx - 360.0f, cy + 4.0f, scale, win_w, win_h, CGA(light_cyan), 1.0f);

    std::snprintf(line, sizeof(line), "Extractor: %s   ( S toggles )",
                  surface_nets ? "Surface Nets" : "Marching Cubes");
    text.draw_text(line, cx - 360.0f, cy + 34.0f, scale, win_w, win_h, CGA(light_cyan), 1.0f);

    text.draw_text("Y merge   M mirror-merge   N / ESC cancel", cx - 360.0f, cy + 70.0f,
                  scale, win_w, win_h, CGA(yellow), 1.0f);

    // Subtract option: carve the unselected (red) meshes out of the selected union.
    if (n_unselected > 0) {
        std::snprintf(line, sizeof(line), "- subtract %d red mesh%s (carve)",
                      n_unselected, n_unselected == 1 ? "" : "es");
        text.draw_text(line, cx - 360.0f, cy + 100.0f, scale, win_w, win_h,
                      CGA(light_red), 1.0f);
    }
}

void draw_voxel_merge_progress(TextOverlay& text, int win_w, int win_h, float progress) {
    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 200.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "Voxel-merging... %d%%",
                  (int)(progress * 100.0f + 0.5f));
    text.draw_text(buf, msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(light_green), 1.0f);
}

void draw_toolbar(TextOverlay& text, const InputState& input,
                  uint32_t tri_count, uint32_t vert_count, const char* ver,
                  int win_w, int win_h) {
    float panel_w = 220.0f;
    float panel_h = 300.0f;
    float panel_x = (float)win_w - panel_w - 10.0f;
    float panel_y = (float)win_h - panel_h - 10.0f;

    text.draw_panel(panel_x, panel_y, panel_w, panel_h,
                   win_w, win_h, CGA(black), 0.75f);

    float tx = panel_x + 12.0f;
    float ty = panel_y + 16.0f;
    float line_h = 18.0f;
    float scale = 1.5f;

    text.draw_text("CHISEL", tx, ty, scale, win_w, win_h,
                  CGA(yellow), 1.0f);

    // Version next to the logo, smaller and dimmer. Strip the leading 'v' and
    // any git-describe suffix (-N-gHASH) so a local build reads like a tag.
    if (ver && *ver) {
        char vbuf[24];
        const char* s = (*ver == 'v' || *ver == 'V') ? ver + 1 : ver;
        size_t vi = 0;
        while (s[vi] && s[vi] != '-' && vi < sizeof(vbuf) - 1) { vbuf[vi] = s[vi]; vi++; }
        vbuf[vi] = '\0';
        float logo_w = 6.0f * 8.0f * scale;   // "CHISEL" = 6 glyphs
        float vscale = 1.1f;
        text.draw_text(vbuf, tx + logo_w + 8.0f, ty + 3.0f, vscale,
                      win_w, win_h, CGA(light_gray), 0.9f);
    }
    ty += line_h + 6;

    char buf[128];
    std::snprintf(buf, sizeof(buf), "Brush: %s", input.brush_name());
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(white), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Size: %.0f", input.brush_size);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_gray), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Strength: %.0f%%",
                 input.brush_strength * 100.0f);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_gray), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Hardness: %.0f%%",
                 input.brush_hardness * 100.0f);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_gray), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Spacing: %.0f%%",
                 input.brush_spacing * 100.0f);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_gray), 1.0f);
    ty += line_h + 6;

    std::snprintf(buf, sizeof(buf), "Tris: %u", tri_count);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_green), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Verts: %u", vert_count);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(light_green), 1.0f);
    ty += line_h;

    std::snprintf(buf, sizeof(buf), "Subdiv level: %d", input.subdiv_level);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(magenta), 1.0f);
    ty += line_h;

    const char* mirror_label = "Mirror: Off";
    if (input.mirror_x)
        mirror_label = "Mirror: X";
    text.draw_text(mirror_label,
                  tx, ty, scale, win_w, win_h,
                  CGA(light_cyan), 1.0f);
    ty += line_h;

    text.draw_text(input.fast_normals ? "Normals: Fast" : "Normals: Interp",
                  tx, ty, scale, win_w, win_h,
                  CGA(light_cyan), 1.0f);
    ty += line_h;

    text.draw_text(input.autosmooth ? "Autosmooth: ON" : "Autosmooth: OFF",
                  tx, ty, scale, win_w, win_h,
                  CGA(light_cyan), 1.0f);
}

void draw_slider(TextOverlay& text, const InputState& input, int win_w, int win_h) {
    char buf[128];
    const char* sname = "";
    float value = 0;
    switch (input.slider_mode) {
        case InputState::SliderMode::SIZE:
            sname = "Size";
            value = input.brush_size;
            break;
        case InputState::SliderMode::STRENGTH:
            sname = "Str";
            value = input.brush_strength * 100.0f;
            break;
        case InputState::SliderMode::HARDNESS:
            sname = "Hard";
            value = input.brush_hardness * 100.0f;
            break;
        case InputState::SliderMode::SPACING:
            sname = "Spacing";
            value = input.brush_spacing * 100.0f;
            break;
        default: break;
    }

    float bar_w = 120.0f;
    float bar_h = 6.0f;
    float bar_x = (float)input.slider_start_x - bar_w * 0.5f;
    float bar_y = (float)input.slider_start_y + 20.0f;

    text.draw_panel(bar_x, bar_y, bar_w, bar_h,
                   win_w, win_h, CGA(dark_gray), 0.8f);

    float fill_pct = 0;
    if (input.slider_mode == InputState::SliderMode::SIZE) {
        fill_pct = input.brush_size / 500.0f;
    } else {
        fill_pct = value / 100.0f;
    }
    fill_pct = std::max(0.0f, std::min(1.0f, fill_pct));
    text.draw_panel(bar_x, bar_y, bar_w * fill_pct, bar_h,
                   win_w, win_h, CGA(yellow), 0.9f);

    std::snprintf(buf, sizeof(buf), "%s: %.0f%s",
                 sname, value,
                 input.slider_mode == InputState::SliderMode::SIZE ? "" : "%");
    text.draw_text(buf, bar_x, bar_y - 20.0f, 2.0f,
                  win_w, win_h, CGA(yellow), 1.0f);
}

void draw_notification(TextOverlay& text, InputState& input, int win_w, int win_h) {
    if (input.notification_timer > 0.0f) {
        input.notification_timer -= 1.0f / 60.0f;
        if (input.notification_timer < 0.0f) input.notification_timer = 0.0f;

        char nbuf[512];
        std::snprintf(nbuf, sizeof(nbuf), "%s", input.notification);
        float nw = std::strlen(nbuf) * 8.0f * 2.5f;
        float nx = (float)win_w * 0.5f - nw * 0.5f;
        float ny = (float)win_h - 80.0f;
        text.draw_panel(nx - 10.0f, ny - 5.0f, nw + 20.0f, 35.0f,
                      win_w, win_h, CGA(black), 0.85f);
        text.draw_text(nbuf, nx, ny, 2.5f, win_w, win_h,
                      CGA(light_green), 1.0f);
    }
}

void draw_fps(TextOverlay& text, float fps, int win_w, int win_h) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "FPS: %.0f", fps);
    float tw = std::strlen(buf) * 8.0f * 2.0f;
    text.draw_text(buf, (float)win_w - tw - 12.0f, 12.0f, 2.0f,
                  win_w, win_h, CGA(light_gray), 0.8f);
}

void draw_mode_indicator(TextOverlay& text, const char* mode_text, int win_w, int win_h) {
    float scale = 2.5f;
    float char_w = 8.0f * scale;
    float tw = std::strlen(mode_text) * char_w;
    float x = (float)win_w * 0.5f - tw * 0.5f;
    float y = (float)win_h - 30.0f;
    text.draw_panel(x - 10.0f, y - 5.0f, tw + 20.0f, 35.0f,
                    win_w, win_h, CGA(black), 0.7f);
    text.draw_text(mode_text, x, y, scale,
                   win_w, win_h, CGA(light_magenta), 1.0f);
}

// ---- ImGui Button Islands ----

static const ImVec4 col_btn       = ImVec4(0.22f, 0.22f, 0.26f, 1.0f);
static const ImVec4 col_btn_hover = ImVec4(0.32f, 0.32f, 0.38f, 1.0f);
static const ImVec4 col_btn_act   = ImVec4(0.55f, 0.35f, 0.70f, 1.0f);
static const ImVec4 col_text      = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
static const ImVec4 col_text_sel  = ImVec4(1.0f, 1.0f, 0.33f, 1.0f);

static bool squircle_button(const char* label, const char* display, const char* tooltip,
                            ImVec2 size, bool selected, bool enabled = true) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::PushID(label);
    if (!enabled) ImGui::BeginDisabled();

    bool clicked = ImGui::InvisibleButton(label, size);
    bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    bool active  = ImGui::IsItemActive();

    if (hovered && tooltip) {
        ImVec2 mouse = ImGui::GetMousePos();
        ImVec2 tt_size = ImGui::CalcTextSize(tooltip);
        float tt_pad = 8.0f;
        float tt_w = tt_size.x + tt_pad * 2.0f;
        float tt_x = mouse.x - tt_w * 0.5f;
        float tt_y = mouse.y + 20.0f;
        ImGui::SetNextWindowPos(ImVec2(tt_x, tt_y));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(tt_pad, 6));
        ImGui::SetTooltip("%s", tooltip);
        ImGui::PopStyleVar();
    }

    ImU32 bg;
    if (selected)     bg = ImGui::GetColorU32(col_btn_act);
    else if (active)  bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);

    float rounding = size.y * 0.28f;

    ImVec2 sh_min = ImVec2(pos.x + 2.0f, pos.y + 2.0f);
    ImVec2 sh_max = ImVec2(pos.x + size.x + 2.0f, pos.y + size.y + 2.0f);
    dl->AddRectFilled(sh_min, sh_max, IM_COL32(0, 0, 0, 80), rounding);

    ImVec2 btn_max = ImVec2(pos.x + size.x, pos.y + size.y);
    dl->AddRectFilled(pos, btn_max, bg, rounding);

    ImVec2 text_size = ImGui::CalcTextSize(display);
    ImVec2 text_pos = ImVec2(
        pos.x + (size.x - text_size.x) * 0.5f,
        pos.y + (size.y - text_size.y) * 0.5f
    );
    ImU32 text_col = selected ? ImGui::GetColorU32(col_text_sel)
                              : ImGui::GetColorU32(col_text);
    dl->AddText(text_pos, text_col, display);

    if (!enabled) ImGui::EndDisabled();
    ImGui::PopID();
    return clicked;
}

// Small square swatch with a vector shape icon (sphere/box/cylinder), styled like
// the paint colour boxes: distinct from the text mode-buttons so it reads as a
// picker. Returns true on click. `kind`: 0=sphere, 1=box, 2=cylinder.
static bool shape_swatch(const char* label, int kind, const char* tooltip,
                         float size, bool selected) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 sz  = ImVec2(size, size);

    ImGui::PushID(label);
    bool clicked = ImGui::InvisibleButton(label, sz);
    bool hovered = ImGui::IsItemHovered();
    bool active  = ImGui::IsItemActive();
    if (hovered && tooltip) ImGui::SetTooltip("%s", tooltip);

    ImU32 bg;
    if (selected)     bg = ImGui::GetColorU32(col_btn_act);
    else if (active)  bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);
    float rounding = size * 0.28f;
    dl->AddRectFilled(ImVec2(pos.x + 2, pos.y + 2), ImVec2(pos.x + size + 2, pos.y + size + 2),
                      IM_COL32(0, 0, 0, 80), rounding);
    dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), bg, rounding);

    float cx = pos.x + size * 0.5f, cy = pos.y + size * 0.5f;
    const ImU32 light = IM_COL32(224, 224, 230, 255);
    const ImU32 mid   = IM_COL32(176, 176, 186, 255);
    const ImU32 dark  = IM_COL32(120, 120, 132, 255);

    if (kind == 0) {
        // Sphere: filled disc with an offset highlight.
        float r = size * 0.28f;
        dl->AddCircleFilled(ImVec2(cx, cy), r, mid, 24);
        dl->AddCircleFilled(ImVec2(cx - r * 0.32f, cy - r * 0.32f), r * 0.42f, light, 16);
    } else if (kind == 1) {
        // Box: isometric cube — top (light), left (mid), right (dark).
        float a = size * 0.26f, e = a * 0.5f;
        ImVec2 T (cx,       cy - a);
        ImVec2 R (cx + a,   cy - e);
        ImVec2 L (cx - a,   cy - e);
        ImVec2 Bc(cx,       cy);
        ImVec2 B (cx,       cy + a);
        ImVec2 Rb(cx + a,   cy + e);
        ImVec2 Lb(cx - a,   cy + e);
        dl->AddQuadFilled(T, R, Bc, L, light);   // top face
        dl->AddQuadFilled(L, Bc, B, Lb, mid);    // left face
        dl->AddQuadFilled(Bc, R, Rb, B, dark);   // right face
    } else {
        // Cylinder: body + bottom cap (mid) with a lighter top ellipse.
        float rw = size * 0.20f, rh = size * 0.24f, ry = rw * 0.42f;
        dl->AddRectFilled(ImVec2(cx - rw, cy - rh), ImVec2(cx + rw, cy + rh), mid);
        dl->AddEllipseFilled(ImVec2(cx, cy + rh), ImVec2(rw, ry), mid, 0.0f, 20);
        dl->AddEllipseFilled(ImVec2(cx, cy - rh), ImVec2(rw, ry), light, 0.0f, 20);
    }

    ImGui::PopID();
    return clicked;
}

// Alpha-stamp swatch: a square button showing the 16x16 preview of a brush-alpha,
// styled like shape_swatch (same 4-state background). `preview` is 256 grayscale
// values in [0,1] row-major, or nullptr to draw a "＋" (load-custom) glyph.
static bool alpha_swatch(const char* label, const float* preview, const char* tooltip,
                         float size, bool selected) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 sz  = ImVec2(size, size);

    ImGui::PushID(label);
    bool clicked = ImGui::InvisibleButton(label, sz);
    bool hovered = ImGui::IsItemHovered();
    bool active  = ImGui::IsItemActive();
    if (hovered && tooltip) ImGui::SetTooltip("%s", tooltip);

    ImU32 bg;
    if (selected)     bg = ImGui::GetColorU32(col_btn_act);
    else if (active)  bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);
    float rounding = size * 0.28f;
    dl->AddRectFilled(ImVec2(pos.x + 2, pos.y + 2), ImVec2(pos.x + size + 2, pos.y + size + 2),
                      IM_COL32(0, 0, 0, 80), rounding);
    dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), bg, rounding);

    const float inset = size * 0.14f;
    float x0 = pos.x + inset, y0 = pos.y + inset;
    float cell = (size - 2 * inset) / 16.0f;

    if (!preview) {
        // Load-custom "＋".
        float cx = pos.x + size * 0.5f, cy = pos.y + size * 0.5f, r = size * 0.22f;
        ImU32 col = IM_COL32(224, 224, 230, 255);
        dl->AddLine(ImVec2(cx - r, cy), ImVec2(cx + r, cy), col, 2.0f);
        dl->AddLine(ImVec2(cx, cy - r), ImVec2(cx, cy + r), col, 2.0f);
    } else {
        // Dark backing so low alpha reads, then the grayscale grid on top.
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cell * 16, y0 + cell * 16),
                          IM_COL32(30, 30, 34, 255));
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                float a = preview[y * 16 + x];
                if (a <= 0.02f) continue;
                int g = (int)(a * 255.0f + 0.5f);
                dl->AddRectFilled(ImVec2(x0 + x * cell, y0 + y * cell),
                                  ImVec2(x0 + (x + 1) * cell, y0 + (y + 1) * cell),
                                  IM_COL32(g, g, g, 255));
            }
        }
    }

    ImGui::PopID();
    return clicked;
}

void draw_button_islands(InputState& input, int win_w, int win_h,
                         const AlphaLibrary* alpha_lib) {
    const float btn_h    = 38.0f;
    const float btn_gap  = 6.0f;
    const float row_gap  = 10.0f;
    const float margin   = 12.0f;
    const float pad_x    = 16.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(btn_gap, btn_gap));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));

    const ImGuiWindowFlags island_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    auto calc_btn_w = [&](const char* text) -> float {
        return ImGui::CalcTextSize(text).x + pad_x * 2.0f;
    };

    BrushType current = input.current_brush;
    bool smooth_on = input.is_smooth_active();
    bool paint_on  = (current == BrushType::PAINT) && !smooth_on;
    InputState::InteractionMode mode = input.interaction_mode;

    // Each island is its OWN auto-resizing window so its mouse-capture rect hugs its
    // own buttons. A single combined window spans the L-shaped bounding box (wide ops
    // row across the top, narrow brush column down the left) and captures the empty
    // viewport enclosed in the bottom-right — the "dead square" that swallowed the
    // brush cursor over pulled tris. Heights are remembered frame-to-frame to stack
    // the islands vertically; with auto-resize a toggled row only lags one frame.
    static float ops_h  = btn_h;
    static float mode_h = btn_h;
    float y = margin;

    // === Island 1 — Ops row ===
    ImGui::SetNextWindowPos(ImVec2(margin, y), ImGuiCond_Always);
    ImGui::Begin("##IslandOps", nullptr, island_flags);

    struct OpsBtn { const char* id; const char* display; const char* tooltip; };
    OpsBtn ops[] = {
        {"MresDown", "Subdiv -",  "Shortcut: Shift+D"},
        {"MresUp",   "Subdiv +",  "Shortcut: Ctrl+D"},
        {"Undo",     "Undo",      "Shortcut: Ctrl+Z"},
        {"Redo",     "Redo",      "Shortcut: Ctrl+Shift+Z"},
        {"Export",   "Export",    "Shortcut: Ctrl+E"},
        {"Save",     "Save",      "Shortcut: Ctrl+S"},
        {"Load",     "Load",      "Shortcut: Ctrl+O"},
        {"Merge",    "Merge",     "Voxel-merge selection for print (Shortcut: J)"},
    };

    for (int i = 0; i < 8; i++) {
        if (i > 0) ImGui::SameLine();
        float w = calc_btn_w(ops[i].display);
        bool enabled = (i < 2) ? input.mesh_locked : true;
        if (squircle_button(ops[i].id, ops[i].display, ops[i].tooltip,
                            ImVec2(w, btn_h), false, enabled)) {
            switch (i) {
                case 0: input.level_switch_delta = -1; break;
                case 1: input.level_switch_delta = +1; break;
                case 2: input.undo_requested = true; break;
                case 3: input.redo_requested = true; break;
                case 4: input.export_dialog_active = true; break;
                case 5: input.save_requested = true; break;
                case 6: input.import_dialog_active = true; break;
                case 7: input.voxel_merge_confirm_pending = true; break;
            }
        }
    }

    ops_h = ImGui::GetWindowSize().y;
    ImGui::End();
    y += ops_h + row_gap;

    // === Island 2 — Mode row: Edit / Insert / Select / Paint (keys 1-4) ===
    ImGui::SetNextWindowPos(ImVec2(margin, y), ImGuiCond_Always);
    ImGui::Begin("##IslandModes", nullptr, island_flags);
    {
        bool edit_sel = (mode == InputState::InteractionMode::EDIT) && !paint_on;
        if (squircle_button("Edit", "Edit", "Sculpt (Shortcut: 1)",
                            ImVec2(calc_btn_w("Edit"), btn_h), edit_sel)) {
            input.interaction_mode = InputState::InteractionMode::EDIT;
            if (current == BrushType::PAINT) {
                input.clear_smooth_lock();
                input.switch_brush(BrushType::DRAW);
            }
        }
        ImGui::SameLine();
        if (squircle_button("Insert", "Insert", "Spawn a primitive (Shortcut: 2)",
                            ImVec2(calc_btn_w("Insert"), btn_h),
                            mode == InputState::InteractionMode::INSERT)) {
            input.interaction_mode = InputState::InteractionMode::INSERT;
        }
        ImGui::SameLine();
        if (squircle_button("Select", "Select", "Pick / move objects (Shortcut: 3)",
                            ImVec2(calc_btn_w("Select"), btn_h),
                            mode == InputState::InteractionMode::SELECT)) {
            input.interaction_mode = InputState::InteractionMode::SELECT;
        }
        ImGui::SameLine();
        if (squircle_button("PaintMode", "Paint", "Vertex paint (Shortcut: 4)",
                            ImVec2(calc_btn_w("Paint"), btn_h), paint_on)) {
            input.interaction_mode = InputState::InteractionMode::EDIT;
            input.clear_smooth_lock();
            input.switch_brush(BrushType::PAINT);
            input.subtract_locked = false;
        }
        // Paint-visibility toggle beside the Paint mode button. Hides albedo while
        // sculpting (the paint brush itself always forces it visible).
        ImGui::SameLine();
        if (squircle_button("PaintVis", input.paint_visible ? "[o]" : "[ ]",
                            input.paint_visible ? "Paint visible (click to hide)"
                                                : "Paint hidden (click to show)",
                            ImVec2(btn_h, btn_h), input.paint_visible)) {
            input.paint_visible = !input.paint_visible;
        }
    }

    // Insert shape picker: Sphere / Box / Cylinder. Only shown in insert mode; the
    // selection drives which primitive InsertController spawns on the next click.
    if (mode == InputState::InteractionMode::INSERT) {
        struct ShapeBtn { const char* id; int kind; const char* tooltip; InputState::InsertShape shape; };
        ShapeBtn shapes[] = {
            {"ShapeSphere",   0, "Sphere",   InputState::InsertShape::SPHERE},
            {"ShapeBox",      1, "Box",      InputState::InsertShape::BOX},
            {"ShapeCylinder", 2, "Cylinder", InputState::InsertShape::CYLINDER},
        };
        for (int i = 0; i < 3; i++) {
            if (i > 0) ImGui::SameLine();
            bool sel = input.insert_shape == shapes[i].shape;
            if (shape_swatch(shapes[i].id, shapes[i].kind, shapes[i].tooltip, btn_h, sel))
                input.insert_shape = shapes[i].shape;
        }
    }

    // Paint colour boxes: active (left) + alternate (right). Q/E swap them. Only
    // shown in paint mode. Drives input.paint_color / paint_color_alt.
    if (paint_on) {
        ImGui::ColorEdit3("##paintA", input.paint_color,
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine();
        ImGui::ColorEdit3("##paintB", input.paint_color_alt,
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine();
        ImGui::TextUnformatted("Q/E swap");
    }

    // Brush-alpha (stamp) picker: Round + built-ins + custom, then a "＋" to load an
    // image. Alphas apply to Draw, Mask and Paint only — the picker is hidden for
    // every other brush (their dabs force the stamp off; see set_alpha_dab).
    // Selection drives input.active_alpha; the main loop uploads on change.
    bool alpha_brush = current == BrushType::DRAW || current == BrushType::MASK ||
                       current == BrushType::PAINT;
    if (mode == InputState::InteractionMode::EDIT && alpha_brush
        && alpha_lib && alpha_lib->count() > 0) {
        const float sw = btn_h;
        int n = alpha_lib->count();
        for (int i = 0; i < n; i++) {
            if (i > 0) ImGui::SameLine();
            const AlphaEntry& ae = alpha_lib->get(i);
            char id[32];
            std::snprintf(id, sizeof(id), "Alpha%d", i);
            if (alpha_swatch(id, ae.preview, ae.name.c_str(), sw, input.active_alpha == i))
                input.active_alpha = i;
        }
        ImGui::SameLine();
        if (alpha_swatch("AlphaLoad", nullptr, "Load a grayscale image as a brush alpha", sw, false))
            input.load_alpha_dialog_active = true;
    }

    mode_h = ImGui::GetWindowSize().y;
    ImGui::End();
    y += mode_h + row_gap;

    // === Island 3 — Brush column: single vertical column ===
    // Order: Draw, Inflate, Crease, Pinch, Move, Limb, Smooth, Mask. Smooth is a lock
    // toggle (not a switch_brush), handled specially in-place. Paint lives in the
    // mode row above, not here.
    ImGui::SetNextWindowPos(ImVec2(margin, y), ImGuiCond_Always);
    ImGui::Begin("##IslandBrushes", nullptr, island_flags);

    float brush_w = calc_btn_w("Inflate");

    struct BrushBtn { const char* id; const char* display; const char* tooltip; BrushType type; };
    BrushBtn brushes[] = {
        {"Draw",    "Draw",    "Shortcut: D",                    BrushType::DRAW},
        {"Inflate", "Inflate", "Shortcut: I",                    BrushType::INFLATE},
        {"Crease",  "Crease",  "Shortcut: C",                    BrushType::CREASE},
        {"Pinch",   "Pinch",   "Shortcut: V",                    BrushType::PINCH},
        {"Move",    "Move",    "Shortcut: G",                    BrushType::MOVE},
        {"Limb",    "Limb",    "Snakehook: pull + redistribute. Shortcut: H", BrushType::LIMB},
        {"Smooth",  "Smooth",  "Shortcut: double-press Shift",   BrushType::SMOOTH},
        {"Mask",    "Mask",    "Shortcut: M",                    BrushType::MASK},
    };

    for (int i = 0; i < 8; i++) {
        if (brushes[i].type == BrushType::SMOOTH) {
            if (squircle_button(brushes[i].id, brushes[i].display, brushes[i].tooltip,
                                ImVec2(brush_w, btn_h), smooth_on)) {
                if (!input.smooth_locked) {
                    input.per_brush[(int)input.current_brush].strength = input.brush_strength;
                    input.per_brush[(int)input.current_brush].hardness = input.brush_hardness;
                    input.per_brush[(int)input.current_brush].spacing  = input.brush_spacing;
                    input.brush_strength = input.per_brush[(int)BrushType::SMOOTH].strength;
                    input.brush_hardness = input.per_brush[(int)BrushType::SMOOTH].hardness;
                    input.brush_spacing  = input.per_brush[(int)BrushType::SMOOTH].spacing;
                    input.smooth_locked = true;
                } else {
                    input.clear_smooth_lock();
                }
            }
            continue;
        }

        bool sel = (current == brushes[i].type) && !smooth_on;
        if (squircle_button(brushes[i].id, brushes[i].display, brushes[i].tooltip,
                            ImVec2(brush_w, btn_h), sel)) {
            input.clear_smooth_lock();
            input.switch_brush(brushes[i].type);
            input.subtract_locked = false;
            // Picking a brush implies the sculpt workspace, not insert/select.
            input.interaction_mode = InputState::InteractionMode::EDIT;
        }
    }

    // Paint color swatch: right-click anywhere on the canvas while the paint brush
    // is active to pop a colour picker at the cursor. Drives input.paint_color.
    static bool prev_rmb = false;
    bool rmb_edge = input.mouse2_down && !prev_rmb;
    prev_rmb = input.mouse2_down;
    if (current == BrushType::PAINT && !smooth_on && rmb_edge && !input.ctrl_held) {
        ImGui::SetNextWindowPos(ImVec2((float)input.mouse_x, (float)input.mouse_y));
        ImGui::OpenPopup("##paintswatch");
    }
    if (ImGui::BeginPopup("##paintswatch")) {
        ImGui::ColorPicker3("##paintpick", input.paint_color,
                            ImGuiColorEditFlags_NoSidePreview |
                            ImGuiColorEditFlags_NoInputs |
                            ImGuiColorEditFlags_NoLabel);
        ImGui::EndPopup();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}
