#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include "ui_overlay.h"
#include "ui_skin.h"
#include "text_overlay.h"
#include "input.h"
#include "settings.h"
#include "brush_alpha.h"
#include "ui_icons.h"
#include "pose.h"
#include "camera.h"
#include "renderer.h"
#include "retopo.h"
#include "uv_unwrap.h"
#include "imgui.h"
#include <functional>
#include <cstdio>
#include <cstring>
#include <cmath>
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

// The bitmap font has no weight, so "bold" is the usual trick: the same string
// twice, offset a couple of pixels horizontally. At scale 3 that reads as a real
// weight; offsetting vertically as well just smears it.
static void draw_text_bold(TextOverlay& text, const char* str, float x, float y,
                           float scale, int win_w, int win_h,
                           float r, float g, float b, float a) {
    text.draw_text(str, x, y, scale, win_w, win_h, r, g, b, a);
    text.draw_text(str, x + 2.0f, y, scale, win_w, win_h, r, g, b, a);
}

// Shared tail of both remesh prompts: what the rescue snapshot will and will not
// do for this scene. The Y keypress is the user's consent to the bake, so the
// cost has to be on screen before they press it.
static void draw_rescue_line(TextOverlay& text, int levels_to_bake, float x, float y,
                             float scale, int win_w, int win_h) {
    char line[128];
    if (levels_to_bake < 0) {
        std::snprintf(line, sizeof(line), "Too big to stash a revert - SAVE FIRST (%s)",
                      live_key_label(KeyAction::SAVE));
        draw_text_bold(text, line, x, y, scale, win_w, win_h, CGA(light_red), 1.0f);
    } else if (levels_to_bake > 0) {
        std::snprintf(line, sizeof(line),
                      "Bakes down %d subdiv level%s first, then %s can revert",
                      levels_to_bake, levels_to_bake == 1 ? "" : "s",
                      live_key_label(KeyAction::UNDO));
        draw_text_bold(text, line, x, y, scale, win_w, win_h, CGA(yellow), 1.0f);
    } else {
        std::snprintf(line, sizeof(line), "%s reverts this for 3 strokes",
                      live_key_label(KeyAction::UNDO));
        draw_text_bold(text, line, x, y, scale, win_w, win_h, CGA(light_green), 1.0f);
    }
}

void draw_quit_dialog(TextOverlay& text, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float msg_scale = 3.0f;
    const char* msg = "Quit? Y / N / S = save first";
    float msg_x = (float)win_w * 0.5f - std::strlen(msg) * 8.0f * msg_scale * 0.5f;   // 8 px cells
    float msg_y = (float)win_h * 0.5f - 20.0f;
    text.draw_text(msg, msg_x, msg_y, msg_scale, win_w, win_h, CGA(yellow), 1.0f);
}

void draw_drop_confirm(TextOverlay& text, const char* path, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float cy = (float)win_h * 0.5f;

    const char* base = path ? std::strrchr(path, '/') : nullptr;
    base = base ? base + 1 : (path ? path : "");

    char line[160];
    std::snprintf(line, sizeof(line), "Open %.120s?", base);
    text.draw_text(line, cx - 360.0f, cy - 50.0f, scale, win_w, win_h, CGA(yellow), 1.0f);

    text.draw_text("Current sculpt will be replaced.", cx - 360.0f, cy - 20.0f, scale,
                  win_w, win_h, CGA(light_gray), 1.0f);

    text.draw_text("Y open, don't save   N / ESC cancel",
                  cx - 360.0f, cy + 20.0f, scale, win_w, win_h, CGA(yellow), 1.0f);
}

void draw_remesh_confirm(TextOverlay& text, int levels_to_bake, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 260.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    text.draw_text("Remesh stretched regions? Wipes stack. (Y/N)", msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(yellow), 1.0f);
    draw_rescue_line(text, levels_to_bake, msg_x, msg_y + 34.0f, msg_scale, win_w, win_h);
}

void draw_remesh_progress(TextOverlay& text, int win_w, int win_h, float progress) {
    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 180.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    char line[48];
    std::snprintf(line, sizeof(line), "Remeshing... %d%%", (int)(progress * 100.0f + 0.5f));
    text.draw_text(line, msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(light_green), 1.0f);
}

void draw_voxel_merge_confirm(TextOverlay& text, int resolution, int n_selected,
                             int n_unselected, bool surface_nets,
                             bool has_density, bool adaptive, int levels_to_bake,
                             int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    float scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float cy = (float)win_h * 0.5f;

    char line[160];
    std::snprintf(line, sizeof(line),
                  "SDF-remesh %d mesh%s into one watertight mesh?",
                  n_selected, n_selected == 1 ? "" : "es");
    text.draw_text(line, cx - 360.0f, cy - 60.0f, scale, win_w, win_h, CGA(yellow), 1.0f);

    draw_rescue_line(text, levels_to_bake, cx - 360.0f, cy - 30.0f, scale, win_w, win_h);

    std::snprintf(line, sizeof(line), "Resolution: %d   ( [ - / + ] )", resolution);
    text.draw_text(line, cx - 360.0f, cy + 4.0f, scale, win_w, win_h, CGA(light_cyan), 1.0f);

    std::snprintf(line, sizeof(line), "Extractor: %s   ( S toggles )",
                  surface_nets ? "Surface Nets" : "Marching Cubes");
    text.draw_text(line, cx - 360.0f, cy + 34.0f, scale, win_w, win_h, CGA(light_cyan), 1.0f);

    text.draw_text("Y remesh   M mirror-remesh   N / ESC cancel", cx - 360.0f, cy + 70.0f,
                  scale, win_w, win_h, CGA(yellow), 1.0f);

    // Subtract option: carve the unselected meshes out of the selected union.
    float oy = 100.0f;
    if (n_unselected > 0) {
        std::snprintf(line, sizeof(line), "- subtract %d unselected mesh%s (carve)",
                      n_unselected, n_unselected == 1 ? "" : "es");
        text.draw_text(line, cx - 360.0f, cy + oy, scale, win_w, win_h,
                      CGA(light_red), 1.0f);
        oy += 30.0f;
    }

    // Density paint present: offer the chained adaptive remesh so the merge
    // output follows the heatmap without a separate '/' press.
    if (has_density) {
        std::snprintf(line, sizeof(line), "Adaptive remesh after: %s   ( D toggles )",
                      adaptive ? "ON" : "OFF");
        text.draw_text(line, cx - 360.0f, cy + oy, scale, win_w, win_h,
                      CGA(light_cyan), 1.0f);
    }
}

void draw_voxel_merge_progress(TextOverlay& text, int win_w, int win_h, float progress) {
    float msg_scale = 3.0f;
    float msg_x = (float)win_w * 0.5f - 200.0f;
    float msg_y = (float)win_h * 0.5f - 20.0f;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "SDF-remeshing... %d%%",
                  (int)(progress * 100.0f + 0.5f));
    text.draw_text(buf, msg_x, msg_y, msg_scale,
                  win_w, win_h, CGA(light_green), 1.0f);
}

void draw_toolbar(TextOverlay& text, const InputState& input,
                  uint32_t tri_count, uint32_t vert_count, const char* ver,
                  int level, const char* project_path, int win_w, int win_h) {
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

    // The LIVE editing level. This used to print input.subdiv_level, which is the
    // OPENING SPHERE's base-cage subdivision — Ctrl+D never touches it, so the readout
    // sat frozen while the multires stack walked underneath it.
    std::snprintf(buf, sizeof(buf), "Subdiv level: %d", level);
    text.draw_text(buf, tx, ty, scale, win_w, win_h,
                  CGA(magenta), 1.0f);
    ty += line_h;

    const char* mirror_label = "Mirror: Off";
    if (input.mirror_x)
        mirror_label = input.mirror_topological ? "Mirror: Topological"
                                                : "Mirror: World Space";
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
    ty += line_h + 6;

    // Sculpt name at the bottom of the panel: basename without the .chisel
    // suffix, or the never-saved placeholder. Truncated to the panel width
    // (glyphs are 8px * scale wide).
    {
        const float nscale = 1.25f;
        text.draw_text("Filename:", tx, ty, scale, win_w, win_h,
                      CGA(light_gray), 1.0f);
        ty += line_h;
        char nbuf[32];
        const char* name = "Unsaved Project";
        if (project_path && *project_path) {
            const char* base = std::strrchr(project_path, '/');
            base = base ? base + 1 : project_path;
            size_t len = std::strlen(base);
            if (len > 7 && std::strcmp(base + len - 7, ".chisel") == 0)
                len -= 7;
            size_t max_chars =
                (size_t)((panel_w - 24.0f) / (8.0f * nscale));
            if (max_chars > sizeof(nbuf) - 1) max_chars = sizeof(nbuf) - 1;
            if (len > max_chars) {
                len = max_chars - 2;
                std::memcpy(nbuf, base, len);
                nbuf[len] = '.'; nbuf[len + 1] = '.'; nbuf[len + 2] = '\0';
            } else {
                std::memcpy(nbuf, base, len);
                nbuf[len] = '\0';
            }
            name = nbuf;
        }
        text.draw_text(name, tx, ty, nscale, win_w, win_h, CGA(white), 1.0f);
    }
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
        case InputState::SliderMode::RINGS:
            sname = "Rings";
            value = (float)(input.retopo_rings - 1) / 7.0f * 100.0f;
            break;
        case InputState::SliderMode::DENSITY:
            sname = "Density";
            value = (input.retopo_density - 0.1f) / 1.9f * 100.0f;
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

    if (input.slider_mode == InputState::SliderMode::RINGS)
        std::snprintf(buf, sizeof(buf), "%s: %d", sname, input.retopo_rings);
    else if (input.slider_mode == InputState::SliderMode::DENSITY)
        std::snprintf(buf, sizeof(buf), "%s: %.1fx", sname, input.retopo_density);
    else
        std::snprintf(buf, sizeof(buf), "%s: %.0f%s",
                     sname, value,
                     input.slider_mode == InputState::SliderMode::SIZE ? "" : "%");
    text.draw_text(buf, bar_x, bar_y - 20.0f, 2.0f,
                  win_w, win_h, CGA(yellow), 1.0f);
}

void draw_drop_level_confirm(TextOverlay& text, int level, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);

    char buf[96];
    std::snprintf(buf, sizeof(buf), "Delete subdiv level %d? (Y/N)", level);
    float msg_scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float msg_y = (float)win_h * 0.5f - 40.0f;
    text.draw_text(buf, cx - std::strlen(buf) * 8.0f * msg_scale * 0.5f, msg_y,
                  msg_scale, win_w, win_h, CGA(yellow), 1.0f);

    // The two consequences worth naming: the fine detail is gone for good, and
    // the history goes with it (its entries point at a level that won't exist).
    const char* sub = "Detail bakes down one level. Cannot be undone - clears history.";
    text.draw_text(sub, cx - std::strlen(sub) * 8.0f * 1.5f * 0.5f, msg_y + 40.0f,
                  1.5f, win_w, win_h, CGA(light_gray), 1.0f);
}

void draw_reset_confirm(TextOverlay& text, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);
    const char* msg = "Reset all settings? (Y/N)";
    float msg_scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float msg_y = (float)win_h * 0.5f - 40.0f;
    text.draw_text(msg, cx - std::strlen(msg) * 8.0f * msg_scale * 0.5f, msg_y,
                  msg_scale, win_w, win_h, CGA(yellow), 1.0f);
    const char* sub = "Profiles, colours, display and keys go back to defaults. Sculpt untouched.";
    text.draw_text(sub, cx - std::strlen(sub) * 8.0f * 1.5f * 0.5f, msg_y + 40.0f,
                  1.5f, win_w, win_h, CGA(light_gray), 1.0f);
}

void draw_pose_clear_confirm(TextOverlay& text, int win_w, int win_h) {
    text.draw_panel(0, 0, (float)win_w, (float)win_h,
                   win_w, win_h, 0.0f, 0.0f, 0.0f, 0.5f);
    const char* msg = "Delete all pose spheres? (Y/N)";
    float msg_scale = 3.0f;
    float cx = (float)win_w * 0.5f;
    float msg_y = (float)win_h * 0.5f - 40.0f;
    text.draw_text(msg, cx - std::strlen(msg) * 8.0f * msg_scale * 0.5f, msg_y,
                  msg_scale, win_w, win_h, CGA(yellow), 1.0f);
    const char* sub = "Every sphere and bone on this mesh goes. The sculpt stays as it is.";
    text.draw_text(sub, cx - std::strlen(sub) * 8.0f * 1.5f * 0.5f, msg_y + 40.0f,
                  1.5f, win_w, win_h, CGA(light_gray), 1.0f);
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

void draw_mirror_unavailable(TextOverlay& text, InputState& input, int win_w, int win_h) {
    if (input.mirror_unavailable_timer <= 0.0f) return;
    input.mirror_unavailable_timer -= 1.0f / 60.0f;
    if (input.mirror_unavailable_timer < 0.0f) input.mirror_unavailable_timer = 0.0f;

    const char* msg = "Topological mirror not available";
    const float scale = 2.5f;
    float tw = std::strlen(msg) * 8.0f * scale;
    float tx = (float)win_w * 0.5f - tw * 0.5f;
    float ty = (float)win_h * (2.0f / 3.0f);
    text.draw_panel(tx - 10.0f, ty - 5.0f, tw + 20.0f, 35.0f,
                    win_w, win_h, CGA(black), 0.85f);
    text.draw_text(msg, tx, ty, scale, win_w, win_h, CGA(light_red), 1.0f);
}

void draw_fps(TextOverlay& text, float fps, int win_w, int win_h) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "FPS: %.0f", fps);
    float tw = std::strlen(buf) * 8.0f * 2.0f;
    // Top-right corner; the burger menu stacks underneath at margin + this line's
    // height (kept in sync in draw_button_islands).
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

// Small sun glyph: disc + eight rays, drawn inline as a label for the lighting
// slider. Not a button — it reserves its own box and paints into it. Brightens
// with `t` (the dial value) so the icon itself previews what the slider does.
static void sun_glyph(float box, float t) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(box, box));

    float cx = pos.x + box * 0.5f, cy = pos.y + box * 0.5f;
    float r  = box * 0.22f;
    int   g  = (int)(120.0f + 135.0f * t);
    ImU32 col = IM_COL32(g, g, (int)(g * 0.82f), 255);

    dl->AddCircleFilled(ImVec2(cx, cy), r, col, 16);
    for (int i = 0; i < 8; i++) {
        float a = (float)i * 0.7853982f;               // 2pi/8
        float ca = std::cos(a), sa = std::sin(a);
        dl->AddLine(ImVec2(cx + ca * r * 1.55f, cy + sa * r * 1.55f),
                    ImVec2(cx + ca * r * 2.30f, cy + sa * r * 2.30f), col, 1.5f);
    }
}

// Hamburger button: three bars in a squircle, same 4-state background as the
// other swatches so it reads as part of the set. Returns true on click.
static bool hamburger_button(float size, bool open) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::PushID("BurgerMenu");
    bool clicked = ImGui::InvisibleButton("burger", ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();
    bool active  = ImGui::IsItemActive();
    if (hovered) ImGui::SetTooltip("Menu");

    ImU32 bg;
    if (open)         bg = ImGui::GetColorU32(col_btn_act);
    else if (active)  bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);
    float rounding = size * 0.28f;
    dl->AddRectFilled(ImVec2(pos.x + 2, pos.y + 2), ImVec2(pos.x + size + 2, pos.y + size + 2),
                      IM_COL32(0, 0, 0, 80), rounding);
    dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), bg, rounding);

    const float bar_w = size * 0.52f, gap = size * 0.17f;
    float x0 = pos.x + (size - bar_w) * 0.5f;
    float cy = pos.y + size * 0.5f;
    ImU32 col = ImGui::GetColorU32(open ? col_text_sel : col_text);
    for (int i = -1; i <= 1; i++)
        dl->AddLine(ImVec2(x0, cy + i * gap), ImVec2(x0 + bar_w, cy + i * gap), col, 1.8f);

    ImGui::PopID();
    return clicked;
}

// Info button: an (i) in a disc, same 4-state squircle background as the
// hamburger so the pair reads as one set. Returns true on click.
static bool info_button(float size, bool open) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::PushID("HelpButton");
    bool clicked = ImGui::InvisibleButton("info", ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();
    bool active  = ImGui::IsItemActive();
    if (hovered) ImGui::SetTooltip("Shortcuts");

    ImU32 bg;
    if (open)         bg = ImGui::GetColorU32(col_btn_act);
    else if (active)  bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);
    float rounding = size * 0.28f;
    dl->AddRectFilled(ImVec2(pos.x + 2, pos.y + 2), ImVec2(pos.x + size + 2, pos.y + size + 2),
                      IM_COL32(0, 0, 0, 80), rounding);
    dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), bg, rounding);

    // Hand-drawn glyph rather than a font "i": at 28 px the default font's
    // lowercase i is a single grey pixel column and reads as dirt.
    float cx = pos.x + size * 0.5f, cy = pos.y + size * 0.5f;
    float r  = size * 0.31f;
    ImU32 col = ImGui::GetColorU32(open ? col_text_sel : col_text);
    dl->AddCircle(ImVec2(cx, cy), r, col, 20, 1.6f);
    dl->AddCircleFilled(ImVec2(cx, cy - r * 0.46f), 1.3f, col, 8);
    dl->AddLine(ImVec2(cx, cy - r * 0.10f), ImVec2(cx, cy + r * 0.50f), col, 1.8f);

    ImGui::PopID();
    return clicked;
}

#ifdef __EMSCRIPTEN__
// Fullscreen button: four corner brackets, pointing out to enter and in to leave. Web
// only — a phone has no Tab key, and a browser tab has no other way to shed its
// address bar. Same squircle as its neighbours. Returns true on click.
static bool fullscreen_button(float size, bool is_fs) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();

    ImGui::PushID("FullscreenButton");
    bool clicked = ImGui::InvisibleButton("fs", ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();
    bool active  = ImGui::IsItemActive();
    if (hovered) ImGui::SetTooltip(is_fs ? "Exit fullscreen" : "Fullscreen");

    ImU32 bg;
    if (active)       bg = ImGui::GetColorU32(ImVec4(0.45f, 0.30f, 0.60f, 1.0f));
    else if (hovered) bg = ImGui::GetColorU32(col_btn_hover);
    else              bg = ImGui::GetColorU32(col_btn);
    float rounding = size * 0.28f;
    dl->AddRectFilled(ImVec2(pos.x + 2, pos.y + 2), ImVec2(pos.x + size + 2, pos.y + size + 2),
                      IM_COL32(0, 0, 0, 80), rounding);
    dl->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size), bg, rounding);

    // Each corner is an L whose elbow sits at the box corner (enter) or pulled in
    // toward the centre with the arms reaching back out (exit).
    ImU32 col = ImGui::GetColorU32(col_text);
    float cx = pos.x + size * 0.5f, cy = pos.y + size * 0.5f;
    float h = size * 0.27f;      // half-extent of the bracket box
    float arm = size * 0.13f;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2) {
            float ex = cx + sx * h, ey = cy + sy * h;              // box corner
            float ax = is_fs ?  sx * arm : -sx * arm;              // arm directions
            float ay = is_fs ?  sy * arm : -sy * arm;
            if (is_fs) { ex -= sx * arm; ey -= sy * arm; }         // elbow pulled in
            ImVec2 pts[3] = { ImVec2(ex + ax, ey), ImVec2(ex, ey), ImVec2(ex, ey + ay) };
            dl->AddPolyline(pts, 3, col, 0, 1.8f);
        }

    ImGui::PopID();
    return clicked;
}
#endif

// One "KEY   what it does" line of the shortcut card. The key column is set in a
// monospaced face (ProggyClean in the classic skin, Plex Mono in the modern one),
// so a %-9s pad is all the column alignment this needs — no table, no per-cell sizing.
static void help_row(const char* key, const char* desc) {
    const bool modern = ui_skin_font(UiFont::Mono) != nullptr;
    if (modern) ImGui::PushFont(ui_skin_font(UiFont::Mono), 0.0f);
    ImGui::TextColored(modern ? ImVec4(0.89f, 0.85f, 1.00f, 1.0f)
                              : ImVec4(1.00f, 1.00f, 0.40f, 1.0f), "%-9s", key);
    if (modern) ImGui::PopFont();
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(ImVec4(0.84f, 0.84f, 0.88f, 1.0f), "%s", desc);
}

static void help_head(const char* title) {
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
    ImGui::TextColored(ImVec4(0.72f, 0.58f, 0.95f, 1.0f), "%s", title);
}

// The shortcut card itself. Deliberately a single screenful: two columns, no
// scrolling, no prose. Anything that needs a sentence lives in MANUAL.md.
void draw_help_popup(InputState& input, int win_w, int win_h) {
    input.help_popup_open = ImGui::IsPopupOpen("##helppopup");
    const bool modern = !input.ui_classic();
    ImGui::SetNextWindowPos(ImVec2(win_w * 0.5f, win_h * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(28, 22));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, modern ? ImVec4(0, 0, 0, 0)
                                                   : ImVec4(0.10f, 0.10f, 0.13f, 0.97f));
    if (ImGui::BeginPopup("##helppopup")) {
        if (modern) {
            // Same material as every other panel; Plex is legible at its normal
            // size, so no blow-up.
            ui_skin_popup_background();
        } else {
            // Big letters: the default font is 13 px, which is a squint at arm's
            // length. Per-window scale, so it does not disturb the rest of the UI.
            ImGui::SetWindowFontScale(1.40f);
        }

        ImGui::TextColored(modern ? ImVec4(0.93f, 0.92f, 0.95f, 1.0f)
                                  : ImVec4(1.0f, 1.0f, 0.40f, 1.0f),
                           modern ? "Shortcuts" : "CHISEL - SHORTCUTS");
        ImGui::Separator();

        // Keys come from the live bindings, so a rebind shows up here at once.
        auto L = [&](KeyAction a) { return input.key_label(a); };
        char k[96], d[96];
        ImGui::BeginGroup();
        help_head("MOUSE");
        help_row("LMB",      "Sculpt  (off mesh: Orbit)");
        help_row("MMB",      "Pan  ~  Scroll: Zoom");
        help_row("Shift",    "Smooth");
        help_row("Ctrl",     "Invert stroke");
        help_row("RMB",      "Colour palette (Paint)");
        help_head("MODES");
        std::snprintf(k, sizeof(k), "%s %s %s %s", L(KeyAction::MODE_SCULPT),
                      L(KeyAction::MODE_INSERT), L(KeyAction::MODE_SELECT), L(KeyAction::MODE_PAINT));
        help_row(k, "Sculpt Insert Select Paint");
        std::snprintf(k, sizeof(k), "%s %s", L(KeyAction::BRUSH_PREV), L(KeyAction::BRUSH_NEXT));
        help_row(k, "Rotate object (Select mode)");
        help_head("BRUSHES");
        std::snprintf(k, sizeof(k), "%s %s %s", L(KeyAction::BRUSH_DRAW),
                      L(KeyAction::BRUSH_CLAY), L(KeyAction::BRUSH_INFLATE));
        help_row(k, "Draw  Clay  Inflate");
        std::snprintf(k, sizeof(k), "%s %s %s", L(KeyAction::BRUSH_CREASE),
                      L(KeyAction::BRUSH_PINCH), L(KeyAction::BRUSH_MOVE));
        help_row(k, "Crease  Pinch  Move");
        std::snprintf(k, sizeof(k), "%s %s", L(KeyAction::BRUSH_LIMB), L(KeyAction::BRUSH_MASK));
        help_row(k, "Limb  Mask");
        std::snprintf(k, sizeof(k), "%s %s", L(KeyAction::BRUSH_PREV), L(KeyAction::BRUSH_NEXT));
        help_row(k, "Cycle alpha (Draw, Mask)");
        std::snprintf(k, sizeof(k), "%s %s", L(KeyAction::PAINT_BUCKET), L(KeyAction::PAINT_PICKER));
        help_row(k, "Fill bucket  Colour picker (Paint)");
        help_head("SIZE + FEEL");
        std::snprintf(k, sizeof(k), "%s %s", L(KeyAction::SIZE_DOWN), L(KeyAction::SIZE_UP));
        help_row(k, "Brush size");
        std::snprintf(k, sizeof(k), "%s %s %s %s", L(KeyAction::HOLD_SIZE),
                      L(KeyAction::HOLD_STRENGTH), L(KeyAction::HOLD_HARDNESS),
                      L(KeyAction::HOLD_SPACING));
        help_row(k, "Size, Strength, Hardness, Spacing");
        help_row("",         "(Hold + Drag)");
        ImGui::EndGroup();

        ImGui::SameLine(0.0f, 44.0f);

        ImGui::BeginGroup();
        help_head("SHAPE");
        std::snprintf(d, sizeof(d), "Subdiv up  (%s: Down)", L(KeyAction::LEVEL_DOWN));
        help_row(L(KeyAction::LEVEL_UP), d);
        help_row(L(KeyAction::PROJECT),    "Project detail down");
        help_row(L(KeyAction::MIRROR),     "Mirror");
        help_row(L(KeyAction::REMESH),     "Remesh");
        help_row(L(KeyAction::MERGE),      "SDF Remesh");
        help_row(L(KeyAction::AUTOSMOOTH), "Autosmooth");
        help_head("MASK");
        std::snprintf(d, sizeof(d), "Invert  ~  %s: Clear", L(KeyAction::MASK_CLEAR));
        help_row(L(KeyAction::MASK_INVERT), d);
        help_head("VIEW");
        help_row(L(KeyAction::FOCUS), "Focus");
        std::snprintf(k, sizeof(k), "%s %s %s", L(KeyAction::VIEW_FRONT),
                      L(KeyAction::VIEW_SIDE), L(KeyAction::VIEW_TOP));
        help_row(k, "Front  Side  Top");
        help_row(L(KeyAction::FULLSCREEN), "Fullscreen");
        help_head("FILES");
        std::snprintf(d, sizeof(d), "Undo  (%s: Redo)", L(KeyAction::REDO));
        help_row(L(KeyAction::UNDO), d);
        std::snprintf(d, sizeof(d), "Save  (%s: Save as)", L(KeyAction::SAVE_AS));
        help_row(L(KeyAction::SAVE), d);
        std::snprintf(d, sizeof(d), "Open  ~  %s: Export", L(KeyAction::EXPORT));
        help_row(L(KeyAction::OPEN), d);
        ImGui::EndGroup();

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        ImGui::Separator();
        ImGui::TextDisabled("Click anywhere outside to close");

        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void draw_button_islands(InputState& input, int win_w, int win_h,
                         const AlphaLibrary* alpha_lib, MultiresInfo mres) {
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
    bool paint_on  = (current == BrushType::PAINT) && !smooth_on
                  && input.interaction_mode != InputState::InteractionMode::RETOPO;
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

    // Tooltips name the live bindings (see KeyAction), so they follow a rebind.
    static char tips[9][96];
    auto tip = [&](int i, const char* fmt, KeyAction a) {
        std::snprintf(tips[i], sizeof(tips[i]), fmt, input.key_label(a));
        return tips[i];
    };
    struct OpsBtn { const char* id; const char* display; const char* tooltip; };
    OpsBtn ops[] = {
        {"MresDown", "Subdiv -",  tip(0, "Shortcut: %s", KeyAction::LEVEL_DOWN)},
        {"MresUp",   "Subdiv +",  tip(1, "Shortcut: %s", KeyAction::LEVEL_UP)},
        {"Undo",     "Undo",      tip(2, "Shortcut: %s", KeyAction::UNDO)},
        {"Redo",     "Redo",      tip(3, "Shortcut: %s", KeyAction::REDO)},
        {"Export",   "Export",    tip(4, "Shortcut: %s", KeyAction::EXPORT)},
        {"Save",     "Save",      tip(5, "Shortcut: %s", KeyAction::SAVE)},
        {"SaveInc",  "+",         "Save incremental: next numbered copy (name_001, name_002, ...)"},
        {"Load",     "Load",      tip(7, "Shortcut: %s", KeyAction::OPEN)},
        {"Merge",    "SDF Merge", tip(8, "SDF remesh selection (Shortcut: %s)", KeyAction::MERGE)},
    };
    const int ops_count = (int)(sizeof(ops) / sizeof(ops[0]));
    const int ops_save_inc = 6;   // square, so it reads as a modifier hanging off Save

    for (int i = 0; i < ops_count; i++) {
        if (i > 0) ImGui::SameLine();
        float w = (i == ops_save_inc) ? btn_h : calc_btn_w(ops[i].display);
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
                case 6: input.save_incremental_requested = true; break;
                case 7: input.import_dialog_active = true; break;
                case 8: input.voxel_merge_confirm_pending = true; break;
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
        static char mtips[4][64];
        std::snprintf(mtips[0], sizeof(mtips[0]), "Sculpt (Shortcut: %s)", input.key_label(KeyAction::MODE_SCULPT));
        std::snprintf(mtips[1], sizeof(mtips[1]), "Spawn a primitive (Shortcut: %s)", input.key_label(KeyAction::MODE_INSERT));
        std::snprintf(mtips[2], sizeof(mtips[2]), "Pick / move objects (Shortcut: %s)", input.key_label(KeyAction::MODE_SELECT));
        std::snprintf(mtips[3], sizeof(mtips[3]), "Vertex paint (Shortcut: %s)", input.key_label(KeyAction::MODE_PAINT));
        bool edit_sel = (mode == InputState::InteractionMode::EDIT) && !paint_on;
        if (squircle_button("Edit", "Edit", mtips[0],
                            ImVec2(calc_btn_w("Edit"), btn_h), edit_sel)) {
            input.enter_sculpt_mode();
        }
        ImGui::SameLine();
        if (squircle_button("Insert", "Insert", mtips[1],
                            ImVec2(calc_btn_w("Insert"), btn_h),
                            mode == InputState::InteractionMode::INSERT)) {
            input.interaction_mode = InputState::InteractionMode::INSERT;
        }
        ImGui::SameLine();
        if (squircle_button("Select", "Select", mtips[2],
                            ImVec2(calc_btn_w("Select"), btn_h),
                            mode == InputState::InteractionMode::SELECT)) {
            input.interaction_mode = InputState::InteractionMode::SELECT;
        }
        ImGui::SameLine();
        if (squircle_button("PaintMode", "Paint", mtips[3],
                            ImVec2(calc_btn_w("Paint"), btn_h), paint_on)) {
            input.enter_paint_mode();
        }
        ImGui::SameLine();
        static char rtip[64];
        std::snprintf(rtip, sizeof(rtip), "Retopo (Shortcut: %s)", input.key_label(KeyAction::MODE_RETOPO));
        if (squircle_button("RetopoMode", "Retopo", rtip, ImVec2(calc_btn_w("Retopo"), btn_h),
                            input.retopo_tools_on()))
            input.enter_retopo_mode();
        ImGui::SameLine();
        static char utip[64];
        std::snprintf(utip, sizeof(utip), "UV (Shortcut: %s)", input.key_label(KeyAction::MODE_UV));
        if (squircle_button("UvMode", "UV", utip, ImVec2(calc_btn_w("UV"), btn_h), input.uv_on()))
            input.enter_uv_mode();
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
        // Target selector: Colour paints albedo, Density paints the remesh-density
        // field (viewport shows green→red colormap while active; Ctrl lowers).
        bool color_target = !input.paint_target_density;
        if (squircle_button("PaintTgtC", "Colour", "Paint albedo",
                            ImVec2(calc_btn_w("Colour"), btn_h), color_target))
            input.paint_target_density = false;
        ImGui::SameLine();
        if (squircle_button("PaintTgtD", "Density",
                            "Paint remesh density (green = coarse, red = dense; Ctrl lowers)",
                            ImVec2(calc_btn_w("Density"), btn_h), !color_target)) {
            input.paint_target_density = true;
            input.bucket_active = false;   // the bucket fills colour only
            input.color_pick_active = false;    // no colour to pick on the heatmap
        }
        if (!input.paint_target_density) {
            ImGui::ColorEdit3("##paintA", input.paint_color,
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine();
            ImGui::ColorEdit3("##paintB", input.paint_color_alt,
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            ImGui::SameLine();
            ImGui::TextUnformatted(input.color_pick_active ? "Picking..."
                                   : input.bucket_active   ? "Click to fill, Ctrl erases"
                                                          : "Q/E swap - C pick - B fill");
            if (input.bucket_active) {
                const AlphaLibrary& pl = fill_patterns();
                const int pi = std::max(0, std::min(input.fill_pattern, pl.count() - 1));
                ImGui::SetNextItemWidth(150.0f);
                if (ImGui::BeginCombo("##fillPattern", pl.get(pi).name.c_str())) {
                    for (int i = 0; i < pl.count(); i++)
                        if (ImGui::Selectable(pl.get(i).name.c_str(), i == pi)) input.fill_pattern = i;
                    ImGui::EndCombo();
                }
                if (pi != kFillSolid && pi != kFillGradient) {
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(120.0f);
                    ImGui::SliderFloat("##fillAngle", &input.fill_angle, 0.0f, 180.0f, "rotate %.0f");
                }
                ImGui::SameLine();
                ImGui::Checkbox("Stop at paint", &input.bucket_paint_walls);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Paint unlike the colour you click walls the fill in, like the mask");
            }
        } else {
            // What the painted extremes mean to the adaptive remesher, as
            // multipliers on the auto target edge length. Defaults (2.0 / 0.5)
            // multiply to 1, so unpainted 0.5 remeshes exactly as today.
            float w = calc_btn_w("Colour") + calc_btn_w("Density") + 8.0f;
            ImGui::SetNextItemWidth(w);
            ImGui::SliderFloat("##densCoarse", &input.density_coarse_mult,
                               1.0f, 4.0f, "green: edge x%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Remesh edge length where painted green (coarse)");
            ImGui::SetNextItemWidth(w);
            ImGui::SliderFloat("##densFine", &input.density_fine_mult,
                               0.2f, 1.0f, "red: edge x%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Remesh edge length where painted red (dense)");
        }
    }

    // Brush-alpha (stamp) picker: Round + built-ins + custom, then a "＋" to load an
    // image. Alphas apply to Draw, Mask and Paint only — the picker is hidden for
    // every other brush (their dabs force the stamp off; see set_alpha_dab). Clay
    // always stamps the Square builtin and offers no choice, so no picker there.
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
    // Order: Draw, Clay, Inflate, Crease, Pinch, Move, Limb, Smooth, Mask. Smooth is a lock
    // toggle (not a switch_brush), handled specially in-place. Paint lives in the
    // mode row above, not here.
    ImGui::SetNextWindowPos(ImVec2(margin, y), ImGuiCond_Always);
    ImGui::Begin("##IslandBrushes", nullptr, island_flags);

    float brush_w = calc_btn_w("Inflate");

    static char btips[10][48];
    auto btip = [&](int i, KeyAction a) {
        std::snprintf(btips[i], sizeof(btips[i]), "Shortcut: %s", input.key_label(a));
        return btips[i];
    };
    struct BrushBtn { const char* id; const char* display; const char* tooltip; BrushType type; };
    BrushBtn brushes[] = {
        {"Draw",    "Draw",    btip(0, KeyAction::BRUSH_DRAW),    BrushType::DRAW},
        {"Clay",    "Clay",    btip(1, KeyAction::BRUSH_CLAY),    BrushType::CLAY},
        {"Inflate", "Inflate", btip(2, KeyAction::BRUSH_INFLATE), BrushType::INFLATE},
        {"Crease",  "Crease",  btip(3, KeyAction::BRUSH_CREASE),  BrushType::CREASE},
        {"Pinch",   "Pinch",   btip(4, KeyAction::BRUSH_PINCH),   BrushType::PINCH},
        {"Move",    "Move",    btip(5, KeyAction::BRUSH_MOVE),    BrushType::MOVE},
        {"Limb",    "Limb",    btip(6, KeyAction::BRUSH_LIMB),    BrushType::LIMB},
        // Pose (K) hidden 2026-10-22: parked, not ready for users. Code stays wired.
        {"Smooth",  "Smooth",  "Shortcut: double-press Shift",    BrushType::SMOOTH},
        {"Mask",    "Mask",    btip(9, KeyAction::BRUSH_MASK),    BrushType::MASK},
    };

    for (int i = 0; i < (int)IM_ARRAYSIZE(brushes); i++) {
        if (brushes[i].type == BrushType::SMOOTH) {
            if (squircle_button(brushes[i].id, brushes[i].display, brushes[i].tooltip,
                                ImVec2(brush_w, btn_h), smooth_on)) {
                if (!input.smooth_locked) {
                    // Same two lines as the double-tap-Shift path in input.cpp — this
                    // copy used to omit the size stash, so locking smooth from the
                    // toolbar and from the keyboard did different things.
                    input.smooth_locked = true;
                    input.sync_live_settings();
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
    if (current == BrushType::PAINT && !smooth_on && rmb_edge && !input.zoom_modifier_held() &&
        input.interaction_mode == InputState::InteractionMode::EDIT) {
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

    // === Island 4 — Burger menu, top-right under the FPS readout ===
    // Right-pivoted so it hugs the corner at any window width. Sits below the FPS
    // line when that is shown, and slides up into its place when it is hidden —
    // no floating gap. draw_fps owns the matching 12 px top margin.
    {
        const float burger = 28.0f;
        const float fps_h  = input.show_fps ? 16.0f + 8.0f : 0.0f;   // 8 px glyphs at 2x
        ImGui::SetNextWindowPos(ImVec2((float)win_w - margin, margin + fps_h),
                                ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        ImGui::Begin("##IslandMenu", nullptr, island_flags);
        // (i) sits to the LEFT of the burger; the island is right-pivoted, so it
        // grows leftward and both stay glued to the corner at any window width.
        if (info_button(burger, ImGui::IsPopupOpen("##helppopup")))
            ImGui::OpenPopup("##helppopup");
        // First launch ever: open the card unprompted, once. Opened from here rather
        // than at startup because the popup belongs to this window's stack — the same
        // reason draw_help_popup is called from inside this Begin/End pair.
        if (!input.help_seen) {
            input.help_seen = true;   // persists via the settings blob
            ImGui::OpenPopup("##helppopup");
        }
        ImGui::SameLine();
#ifdef __EMSCRIPTEN__
        // Between (i) and the burger: in the island's leftmost slot it couldn't be
        // seen on a portrait phone (likely covered by the top toolbar).
        if (fullscreen_button(burger, input.is_fullscreen))
            input.fullscreen_toggle_requested = true;
        ImGui::SameLine();
#endif
        if (hamburger_button(burger, ImGui::IsPopupOpen("##burgermenu")))
            ImGui::OpenPopup("##burgermenu");

        // Read AFTER OpenPopup so the opening frame is the one that syncs. The profile
        // tab bar below stores its selection by ID and defaults to the first tab, so
        // without a one-shot nudge it would report "Mouse" to a Tablet user the moment
        // the menu opens — and that read is what drives switch_profile.
        static bool prev_menu_open = false;
        bool menu_open = ImGui::IsPopupOpen("##burgermenu");
        bool sync_tabs = menu_open && !prev_menu_open;
        prev_menu_open = menu_open;
        // Freezes the device auto-switch for as long as the menu is up, so reaching for
        // a slider with the mouse doesn't yank you off the profile you came here to edit.
        input.settings_menu_open = menu_open;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
        if (ImGui::BeginPopup("##burgermenu")) {
            draw_settings_menu_items(input, mres, sync_tabs);
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();

        // Opened from this window so it inherits the island's popup stack; drawn
        // last so it lands on top of the menu if both were somehow up.
        draw_help_popup(input, win_w, win_h);
        ImGui::End();
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

void draw_settings_menu_items(InputState& input, MultiresInfo mres, bool sync_tabs) {
    // Width of the menu's sliders/buttons: 150 px fits ProggyClean and Plex, but the
    // DOS skin's 16 px character cells need about twice that to show their labels.
    const float mw = ui_skin_menu_item_width(150.0f);
    {
        {
            // Brush-feel profile. Both stay loaded and the active one follows whichever
            // device you are actually using (main.cpp arbitrates); these tabs are how you
            // *edit* the one that isn't in your hand — while this menu is open the
            // auto-switch is frozen, so picking a tab holds until you close it.
            if (ImGui::BeginTabBar("##profiles")) {
                const char* names[(int)InputProfile::COUNT] = { "Mouse", "Tablet" };
                for (int i = 0; i < (int)InputProfile::COUNT; i++) {
                    // SetSelected only on the opening frame (see sync_tabs above);
                    // forcing it every frame would override the user's own clicks.
                    ImGuiTabItemFlags f = (sync_tabs && input.active_profile == (InputProfile)i)
                                        ? ImGuiTabItemFlags_SetSelected : 0;
                    if (ImGui::BeginTabItem(names[i], nullptr, f)) {
                        if (input.active_profile != (InputProfile)i)
                            input.switch_profile((InputProfile)i);
                        ImGui::EndTabItem();
                    }
                }
                ImGui::EndTabBar();
            }
            ImGui::TextDisabled("(?) follows the device you're using");
            if (ImGui::IsItemHovered()) {
                ui_skin_menu_tip("Both profiles stay loaded. Touch the pen and the tablet\n"
                                 "profile takes over; move the mouse and the mouse profile\n"
                                 "does. Never mid-stroke.\n\n"
                                 "Per profile: strength, hardness, spacing, max effect.\n"
                                 "Shared: brush size (per brush if you turn that on\n"
                                 "below), autosmooth, speed falloff, lighting,\n"
                                 "colours, mirror, density.\n\n"
                                 "Pick a tab here to edit the other profile — switching\n"
                                 "is paused while this menu is open.");
            }

            // Max brush effect: the ceiling of the strength ramp for the profile whose
            // tab is selected. Lives above the Separator because everything below it is
            // shared, and this is the one slider that belongs to the tab you just picked.
            // Edited through the LIVE field (not profiles[]) like every other per-profile
            // control — flush_profile parks it in the right slot on the next switch/save.
            //
            // Shown as a percentage because that is how it reads to the user ("60%"),
            // while the stored value stays 0..1 like the rest of the settings. Floor is
            // 0.10, not 0: this is a ramp ceiling and PRESSURE_STR_FLOOR (0.05) is its
            // floor, so going below that would turn the pressure curve upside down.
            float max_pct = input.max_effect * 100.0f;
            ImGui::SetNextItemWidth(mw);
            if (ImGui::SliderFloat("##maxEffect", &max_pct, 10.0f, 100.0f, "max effect %.0f%%"))
                input.max_effect = max_pct * 0.01f;
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("How hard Draw/Inflate/Crease/Pinch hit at FULL input —\n"
                                 "a mouse dab, or the pen pressed to the stop.\n\n"
                                 "Those brushes pile up without bound, so the cap is what\n"
                                 "stops a stroke overshooting. Move, Smooth, Mask and\n"
                                 "Paint aim at a target instead and ignore it.\n\n"
                                 "Defaults: 60%% mouse, 100%% tablet.");

            // Speed falloff (InputState::speed_gain). Sits with max effect because the
            // two are tuned together, but it is ONE shared value, not per profile: the
            // pen has real pressure and never uses it, so a per-profile copy would be a
            // slider that does nothing on one of its two tabs.
            float speed_pct = input.speed_gain * 100.0f;
            ImGui::SetNextItemWidth(mw);
            if (ImGui::SliderFloat("##speedGain", &speed_pct, 0.0f, 100.0f, "speed falloff %.0f%%"))
                input.speed_gain = speed_pct * 0.01f;
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Mouse and finger strokes: the faster you move,\n"
                                 "the lighter the stroke, and a little softer\n"
                                 "and thinner. Slow strokes keep full strength\n"
                                 "and size.\n\n"
                                 "0%% = off. A pen uses its real pressure instead.");

            ImGui::Separator();

            // Which mirror. A button rather than a checkbox because this is a two-state
            // mode, not an option you switch on — and it is genuinely live: main.cpp
            // re-derives it per DabContext, so the very next dab uses the new mode with
            // no reload, no remesh and no map rebuild. That makes it worth flipping
            // mid-sculpt to compare, which is the whole reason it is a button up here
            // instead of a tickbox at the bottom.
            //
            // World Space is the default, and is what ZBrush and Blender do by default
            // too; both also keep a topology-mirror option alongside it, which is what
            // Topological is here.
            {
                const bool topo = input.mirror_topological;
                const ImVec4 on_col  (0.16f, 0.38f, 0.52f, 1.0f);
                const ImVec4 off_col (0.20f, 0.20f, 0.22f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_Button,        topo ? on_col : off_col);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, topo ? ImVec4(0.22f, 0.48f, 0.64f, 1.0f)
                                                                   : ImVec4(0.28f, 0.28f, 0.31f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,  topo ? ImVec4(0.28f, 0.56f, 0.74f, 1.0f)
                                                                   : ImVec4(0.34f, 0.34f, 0.38f, 1.0f));
                if (ImGui::Button(topo ? "Mirror: Topological" : "Mirror: World Space",
                                  ImVec2(mw, 0.0f))) {
                    input.mirror_topological = !topo;
                    // Say it in the viewport, not just on the button: the point of the
                    // toggle is comparing the two mid-sculpt, and the menu is usually
                    // covering the model when you click it.
                    std::snprintf(input.notification, sizeof(input.notification), "%s",
                                  input.mirror_topological ? "Mirror: Topological"
                                                           : "Mirror: World Space");
                    input.notification_timer = 2.0f;
                }
                ImGui::PopStyleColor(3);
            }
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("How X symmetry works. The X key still turns it on and off;\n"
                                 "this is which kind you get, and it changes on the fly —\n"
                                 "the next dab already uses it.\n\n"
                                 "World Space: the dab is reflected in the centre plane\n"
                                 "and applied again to whatever is there. It asks nothing of\n"
                                 "the mesh, so it works on imported models, on remeshed and\n"
                                 "merged ones, and on anything whose two halves are built\n"
                                 "differently. The default.\n\n"
                                 "Topological: vertices are additionally paired across the\n"
                                 "plane and forced to match, so the two sides can never drift\n"
                                 "apart. Only possible on a mesh that really is built\n"
                                 "symmetrically; on one that isn't, Chisel says so and falls\n"
                                 "back to the World Space mirror for that stroke.");

            ImGui::Separator();

            // Per-brush sizes. Seeded from the live size on the frame the box changes, so
            // flipping it never moves the brush that is currently under the cursor —
            // the divergence starts from where you already were.
            if (ImGui::Checkbox("Individual brush sizes", &input.per_brush_sizes))
                input.seed_brush_sizes();
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Off: one brush size for every brush.\n\n"
                                 "On: each brush remembers its own, the way strength and\n"
                                 "spacing already do — Move and Smooth can sit large\n"
                                 "while Paint and Crease stay small, and switching\n"
                                 "between them keeps both.\n\n"
                                 "Sizes last for the session only; they are not saved.");

            // Lighting dial: sun label + the matcap blend (0 = even, 1 = keyed). Worded
            // as "even light" now that "flat shading" means the facet toggle below.
            sun_glyph(ImGui::GetFrameHeight(), input.matcap_contrast);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(mw);
            ImGui::SliderFloat("##matcapLight", &input.matcap_contrast, 0.0f, 1.0f,
                               "light %.2f");
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Viewport lighting: 0 = even, ambient light, 1 = keyed "
                                 "light with contrast and sheen (display only)");

            // Shading. Stored as flat_shading (defaults true, so a settings blob with
            // no key lands faceted like a fresh install), shown inverted because
            // "Smooth shading, on" is how it reads to the user.
            bool smooth = !input.flat_shading;
            if (ImGui::Checkbox("Smooth shading", &smooth))
                input.flat_shading = !smooth;
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("On: normals are averaged across each vertex, so the\n"
                                 "surface reads as one continuous form.\n\n"
                                 "Off: every triangle is shaded by its own facet, so you\n"
                                 "see the actual polygons — useful for judging topology,\n"
                                 "density and what a remesh really did.\n\n"
                                 "Display only: the mesh, the brush and the saved file\n"
                                 "are identical either way.");

            // Viewport projection. Off = the orthographic camera the app has always
            // used and the one the brush feel was tuned against, so it stays default;
            // the slider only bites once perspective is on, hence the disable.
            ImGui::Checkbox("Perspective", &input.camera_perspective);
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Off: orthographic — parallel lines stay parallel and a\n"
                                 "shape reads the same wherever it sits on screen. The\n"
                                 "default, and what the brushes were tuned against.\n\n"
                                 "On: perspective — nearer parts of the model read larger,\n"
                                 "which is how the eye actually sees volume. Worth a look\n"
                                 "when you are judging proportion; ortho is easier to\n"
                                 "sculpt precise detail in.");

            if (!input.camera_perspective) ImGui::BeginDisabled();
            ImGui::SetNextItemWidth(mw);
            ImGui::SliderFloat("##cameraFov", &input.camera_fov, 15.0f, 80.0f, "fov %.0f\xc2\xb0");
            if (!input.camera_perspective) ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ui_skin_menu_tip("Lens angle. Low = long lens, flat and near-orthographic.\n"
                                 "High = wide lens, strong convergence and more distortion\n"
                                 "toward the edges of the view.\n\n"
                                 "The model keeps its on-screen size as you drag this —\n"
                                 "the camera dollies to compensate — so you are changing\n"
                                 "the perspective, not zooming.");

            ImGui::Checkbox("Show FPS", &input.show_fps);

            ImGui::Checkbox("Remesh keeps detail", &input.remesh_keep_detail);
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("On: the remesher keeps its new points on your sculpt\n"
                                 "and won't flip an edge across a ridge or a carved line.\n"
                                 "The SDF remesh/merge also snaps its result back onto\n"
                                 "your meshes, so detail finer than a voxel survives.\n\n"
                                 "Off: the old remeshers, which round small forms and\n"
                                 "creases away.\n\n"
                                 "Either way the console prints a DRIFT line after each\n"
                                 "remesh: how far the result sits from the original.\n"
                                 "Not saved - it resets to On at every launch.");

            // Logarithmic: x0.5 and x2 sit the same distance either side of x1.
            ImGui::SetNextItemWidth(mw);
            ImGui::SliderFloat("##remeshDetail", &input.remesh_detail, 0.25f, 4.0f,
                               "remesh detail x%.2f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("How many triangles a remesh aims for, compared with\n"
                                 "the mesh as it is now. x1 keeps the current average\n"
                                 "and only fixes stretched areas; x2 doubles the count,\n"
                                 "x0.5 halves it.\n\n"
                                 "Anything other than x1 reworks the WHOLE mesh, so it\n"
                                 "takes longer. Ctrl+click to type an exact value.");

            ImGui::Separator();

            draw_appearance_menu_items(input);

            ImGui::Separator();

            // What "sphere" means, for both consumers at once: the ball the app opens
            // with and the one the INSERT shape picker spawns. Deliberately not two
            // settings — a startup ball that disagrees with the insert swatch is the
            // confusing half of this feature.
            int sk = (int)input.sphere_kind;
            ImGui::SetNextItemWidth(mw);
            ui_skin_push_dropdown_bg();
            if (ImGui::Combo("##sphereKind", &sk, "Sphere: icosphere\0Sphere: UV sphere\0"))
                input.sphere_kind = (InputState::SphereKind)sk;
            ImGui::PopStyleColor();
            // Insert picks it up at once; only the opening ball waits for a relaunch.
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Takes effect on next launch");

            // The escape hatch persistence makes necessary: before settings were saved,
            // any value you dragged somewhere unusable was one restart away from gone.
            // Now it comes back, so there has to be a way out that is not "hunt down the
            // config file". It doesn't touch the scene, but it does wipe every setting
            // and binding the user has made, so it asks Y/N first.
            if (ImGui::Selectable("Brush properties...")) {
                input.props_page = InputState::PropsPage::BRUSH;
                input.brush_props_request = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Settings a brush keeps beyond size, strength and hardness.");

            if (ImGui::Selectable("Unselected colour...")) {
                input.props_page = InputState::PropsPage::UNSELECTED_COLOUR;
                input.brush_props_request = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("The tint on objects that aren't selected.");

            if (ImGui::Selectable("Key bindings...")) {
                input.keybind_menu_request = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Change, swap or clear any keyboard shortcut.");

            if (ImGui::Selectable("Reset settings to defaults...")) {
                input.reset_confirm_pending = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::IsItemHovered())
                ui_skin_menu_tip("Both profiles, colours, display options and key\n"
                                 "bindings back to factory values, and clears the\n"
                                 "saved copy.\n"
                                 "Does not touch your sculpt.");

            // Only surfaces when a sink has actually refused us — otherwise silence.
            // Most likely on itch: the game runs in a cross-origin iframe, and a browser
            // set to block third-party storage rejects localStorage outright.
            if (!settings_storage_available()) {
                ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.25f, 1.0f),
                                   "Settings can't be saved");
                if (ImGui::IsItemHovered())
                    ui_skin_menu_tip("Storage was refused, so settings will reset when\n"
                                     "you close the tab. Usually third-party cookies or\n"
                                     "site data being blocked, or a private window.");
            }

            ImGui::Separator();

            // Destructive: arms the Y/N confirm rather than firing directly.
            // Disabled with no stack, or when the only level left IS the base.
            bool can_drop = mres.locked && mres.lmax > mres.base_level;
            char lbl[64];
            if (can_drop) std::snprintf(lbl, sizeof(lbl), "Delete subdiv level %d", mres.lmax);
            else          std::snprintf(lbl, sizeof(lbl), "Delete subdiv level");
            if (!can_drop) ImGui::BeginDisabled();
            if (ImGui::Selectable(lbl)) {
                input.drop_level_confirm_pending = true;
                ImGui::CloseCurrentPopup();
            }
            if (!can_drop) ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ui_skin_menu_tip("%s", can_drop
                    ? "Drop the finest subdivision level and reclaim its memory.\n"
                      "Detail bakes down one level; cannot be undone."
                    : "No subdivision level above the base cage to delete");
            }
        }
    }
}

void draw_pick_cursor(float x, float y, bool on_model) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    int a = on_model ? 235 : 110;
    ImU32 fill = IM_COL32(255, 255, 255, a);
    ImU32 line = IM_COL32(25, 25, 25, a);
    // Glyph points are placed in barrel coordinates: `along` runs from the tip
    // (the exact hotspot at x,y) up the axis toward the bulb, `side` is lateral.
    // Axis points up-right on screen, the classic eyedropper pose.
    const float ax = 0.7071f, ay = -0.7071f;
    auto at = [&](float along, float side) {
        return ImVec2(x + along * ax - side * ay, y + along * ay + side * ax);
    };
    // Needle tip.
    dl->AddTriangleFilled(ImVec2(x, y), at(7.0f, 2.6f), at(7.0f, -2.6f), fill);
    dl->AddTriangle(ImVec2(x, y), at(7.0f, 2.6f), at(7.0f, -2.6f), line, 1.2f);
    // Barrel (outline drawn as a fatter dark line underneath).
    dl->AddLine(at(6.5f, 0), at(17.0f, 0), line, 7.4f);
    dl->AddLine(at(6.5f, 0), at(17.0f, 0), fill, 4.8f);
    // Collar where the glass meets the bulb.
    dl->AddLine(at(15.5f, 0), at(18.0f, 0), line, 10.4f);
    dl->AddLine(at(15.5f, 0), at(18.0f, 0), fill, 7.6f);
    // Rubber bulb.
    ImVec2 c = at(22.5f, 0);
    dl->AddCircleFilled(c, 4.8f, fill);
    dl->AddCircle(c, 4.8f, line, 0, 1.4f);
}

// The armed fill bucket: a dot at the exact hotspot (the fill seeds from the vertex
// under it) and the bucket glyph up and to the right, out of the way of what you aim at.
void draw_bucket_cursor(float x, float y, bool on_model) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    int a = on_model ? 235 : 110;
    ImU32 fill = IM_COL32(255, 255, 255, a);
    ImU32 line = IM_COL32(25, 25, 25, a);
    dl->AddCircleFilled(ImVec2(x, y), 3.2f, line);
    dl->AddCircleFilled(ImVec2(x, y), 1.8f, fill);
    const ImVec2 c(x + 15.0f, y - 15.0f);
    draw_icon(dl, Icon::PaintBucket, c, 24.0f, line, 3.6f);
    draw_icon(dl, Icon::PaintBucket, c, 24.0f, fill, 1.75f);
}

// A gradient being dragged: a filled dot where it is full (the press), a hollow one
// where it runs out (the cursor), joined by a line.
void draw_gradient_drag(float ax, float ay, float bx, float by) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImU32 fill = IM_COL32(255, 255, 255, 235), line = IM_COL32(25, 25, 25, 235);
    dl->AddLine(ImVec2(ax, ay), ImVec2(bx, by), line, 4.0f);
    dl->AddLine(ImVec2(ax, ay), ImVec2(bx, by), fill, 1.8f);
    dl->AddCircleFilled(ImVec2(ax, ay), 6.0f, line);
    dl->AddCircleFilled(ImVec2(ax, ay), 4.5f, fill);
    dl->AddCircle(ImVec2(bx, by), 6.0f, line, 0, 3.6f);
    dl->AddCircle(ImVec2(bx, by), 6.0f, fill, 0, 1.6f);
}

void draw_pose_rig(const PoseRig& rig, const Camera& cam, int win_w, int win_h,
                   int hover, float soft, bool grow) {
    if (rig.empty()) return;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    static std::vector<uint8_t> lit;
    static std::vector<ImVec2>  at;
    static std::vector<float>   rad;
    static std::vector<uint8_t> shown;
    rig.connected(hover, lit);   // all zero when nothing is hovered
    const size_t n = rig.joints.size();
    at.resize(n); rad.resize(n); shown.resize(n);
    for (size_t j = 0; j < n; j++) {
        const PoseJoint& jt = rig.joints[j];
        float x, y;
        cam.world_to_screen(jt.pos, win_w, win_h, x, y);
        shown[j] = !(cam.perspective && cam.view_depth(jt.pos) <= 1e-4f);
        at[j] = ImVec2(x, y);
        rad[j] = std::max(3.0f, jt.radius * pose_px_per_world(cam, jt.pos, win_h));
    }
    const ImU32 dark = IM_COL32(20, 20, 20, 170);
    for (const PoseBone& b : rig.bones) {
        if (!shown[b.a] || !shown[b.b]) continue;
        const ImU32 col = lit[b.a] ? IM_COL32(255, 190, 90, 235) : IM_COL32(235, 235, 235, 170);
        dl->AddLine(at[b.a], at[b.b], dark, 5.0f);
        dl->AddLine(at[b.a], at[b.b], col, 2.5f);
    }
    for (size_t j = 0; j < n; j++) {
        if (!shown[j]) continue;
        const bool is_hover = (int)j == hover;
        const ImU32 fill = is_hover ? IM_COL32(255, 190, 90, 90)
                         : lit[j]   ? IM_COL32(255, 190, 90, 50)
                                    : IM_COL32(255, 255, 255, 36);
        const ImU32 line = is_hover ? IM_COL32(255, 222, 160, 255)
                         : lit[j]   ? IM_COL32(255, 190, 90, 230)
                                    : IM_COL32(235, 235, 235, 170);
        dl->AddCircleFilled(at[j], rad[j], fill, 48);
        dl->AddCircle(at[j], rad[j], dark, 48, 3.5f);
        dl->AddCircle(at[j], rad[j], line, 48, is_hover ? 2.0f : 1.5f);
        if (!is_hover) continue;
        dl->AddCircle(at[j], rad[j] * (1.0f + soft), IM_COL32(255, 190, 90, 80), 64, 1.0f);
        if (grow) {
            const float s = std::min(8.0f, rad[j] * 0.6f);
            const ImVec2 c = at[j];
            dl->AddLine(ImVec2(c.x - s, c.y), ImVec2(c.x + s, c.y), dark, 4.0f);
            dl->AddLine(ImVec2(c.x, c.y - s), ImVec2(c.x, c.y + s), dark, 4.0f);
            dl->AddLine(ImVec2(c.x - s, c.y), ImVec2(c.x + s, c.y), line, 2.0f);
            dl->AddLine(ImVec2(c.x, c.y - s), ImVec2(c.x, c.y + s), line, 2.0f);
        }
    }
}

// The pose brush has no footprint to show: a dot at the hotspot, bright over the
// model or a sphere (a press does something there), dim elsewhere (it orbits).
void draw_pose_cursor(float x, float y, bool on_target) {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const int a = on_target ? 235 : 110;
    dl->AddCircleFilled(ImVec2(x, y), 4.0f, IM_COL32(25, 25, 25, a));
    dl->AddCircleFilled(ImVec2(x, y), 2.4f, IM_COL32(255, 255, 255, a));
}

// ---- Key bindings window ------------------------------------------------------------
//
// Edits a working copy; input.keybinds is only touched by Save. Exit (and Esc) just
// drops the copy — no "discard changes?" prompt, on purpose. Picking a key another
// action already owns raises a warning; going ahead SWAPS the two, so nothing is ever
// silently left without a key it used to have. X clears a binding (shown as ???).
//
// Keys reach this window through key_callback, not ImGui: while keybind_menu_open is
// set the callback runs no shortcuts and drops the next press into keybind_captured.
void draw_keybind_window(InputState& input, int win_w, int win_h) {
    static KeyBind work[(int)KeyAction::COUNT];
    static int     conflict_with = -1;   // action that already owns the captured key
    static int     pending_for   = -1;   // action the captured key is meant for
    static KeyBind pending       = { GLFW_KEY_UNKNOWN, 0 };
    static char    refusal[128]  = "";

    if (input.keybind_menu_request) {
        input.keybind_menu_request = false;
        for (int i = 0; i < (int)KeyAction::COUNT; i++) work[i] = input.keybinds[i];
        conflict_with = pending_for = -1;
        refusal[0] = '\0';
        input.keybind_capture = -1;
        input.keybind_has_capture = false;
        input.keybind_esc = false;
        ImGui::OpenPopup("##keybinds");
    }
    input.keybind_menu_open = ImGui::IsPopupOpen("##keybinds");
    if (!input.keybind_menu_open) return;

    bool close = false;

    // Esc peels one layer: a listening button, then a pending warning, then the window.
    if (input.keybind_esc) {
        input.keybind_esc = false;
        if (input.keybind_capture >= 0) input.keybind_capture = -1;
        else if (conflict_with >= 0)    conflict_with = -1;
        else                            close = true;
    }

    if (input.keybind_has_capture) {
        input.keybind_has_capture = false;
        const int i = input.keybind_capture;
        const KeyBind kb = input.keybind_captured;
        input.keybind_capture = -1;
        if (i >= 0) {
            char name[32];
            if (const char* why = keybind_reserved(kb.key)) {
                std::snprintf(refusal, sizeof(refusal), "%s can't be used: %s.",
                              format_keybind(kb, name, sizeof(name)), why);
            } else {
                // Keys of different modes may share a letter (KeyScope); only a clash
                // where both are live counts.
                int owner = -1;
                const KeyScope sc = key_action_info((KeyAction)i).scope;
                for (int j = 0; j < (int)KeyAction::COUNT; j++)
                    if (j != i && work[j] == kb &&
                        key_scopes_overlap(sc, key_action_info((KeyAction)j).scope)) { owner = j; break; }
                if (owner < 0) {
                    work[i] = kb;
                } else {
                    conflict_with = owner;
                    pending_for = i;
                    pending = kb;
                }
            }
        }
    }

    const bool modern = !input.ui_classic();
    const ImVec4 accent(0.72f, 0.58f, 0.95f, 1.0f);
    const ImVec4 warn  (0.98f, 0.70f, 0.30f, 1.0f);

    ImGui::SetNextWindowPos(ImVec2(win_w * 0.5f, win_h * 0.5f), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 18));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, modern ? ImVec4(0, 0, 0, 0)
                                                   : ImVec4(0.10f, 0.10f, 0.13f, 0.97f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::BeginPopupModal("##keybinds", nullptr, flags)) {
        if (modern) ui_skin_popup_background();

        ImGui::TextColored(modern ? ImVec4(0.93f, 0.92f, 0.95f, 1.0f)
                                  : ImVec4(1.0f, 1.0f, 0.40f, 1.0f), "Key bindings");
        ImGui::TextDisabled("Click a key, then press the new one. Esc cancels.");
        ImGui::Separator();

        // Keys that can't be rebound: { key, what it does }. Listed at the bottom.
        static const char* fixed[][2] = {
            {"Esc",         "Close, cancel, quit"},
            {"Y / N",       "Answer a prompt"},
            {"Shift",       "Smooth (2x: lock)"},
            {"Ctrl",        "Invert (2x: lock)"},
            {"M S D - [ ]", "SDF merge dialog"},
#ifndef __EMSCRIPTEN__
            {"`",           "Console"},
#endif
            {"F9 F10 F12",  "Debug"},
        };

        // Column widths from the actual text (fixed rows included), so the DOS skin's
        // wide cells fit too.
        float label_w = 0.0f;
        for (int i = 0; i < (int)KeyAction::COUNT; i++)
            label_w = std::max(label_w, ImGui::CalcTextSize(
                key_action_info((KeyAction)i).label).x);
        for (const auto& f : fixed)
            label_w = std::max(label_w, ImGui::CalcTextSize(f[1]).x);
        label_w += 24.0f;
        const float key_w = std::max(ImGui::CalcTextSize("press a key...").x,
                                     ImGui::CalcTextSize("Ctrl+Shift+Del").x)
                          + ImGui::GetStyle().FramePadding.x * 2.0f;
        const float x_w = ImGui::GetFrameHeight();
        const float col_w = label_w + key_w + x_w + ImGui::GetStyle().ItemSpacing.x * 2.0f;
        const float col_gap = 32.0f;

        // Lay the groups out in columns so the whole list fits on screen without
        // scrolling. A group never splits across columns; the "Fixed" list is the last
        // unit. Heights mirror what the loop below draws.
        const float sp_y   = ImGui::GetStyle().ItemSpacing.y;
        const float head_h = 4.0f + sp_y + ImGui::GetTextLineHeightWithSpacing();
        const float row_h  = ImGui::GetFrameHeightWithSpacing();
        const float text_h = ImGui::GetTextLineHeightWithSpacing();
        constexpr int kMaxUnits = (int)KeyAction::COUNT + 1;
        int   unit_first[kMaxUnits + 1];        // unit u = actions [first[u], first[u+1])
        float unit_h[kMaxUnits];
        int   units = 0;
        for (int i = 0; i < (int)KeyAction::COUNT; i++) {
            if (i == (int)KeyAction::BRUSH_POSE) continue;   // hidden while the pose brush is parked
            if (i == 0 || std::strcmp(key_action_info((KeyAction)i).group,
                                      key_action_info((KeyAction)(i - 1)).group) != 0) {
                unit_first[units] = i;
                unit_h[units++] = head_h;
            }
            unit_h[units - 1] += row_h;
        }
        unit_first[units] = (int)KeyAction::COUNT;
        const int fixed_unit = units;
        const int n_fixed = (int)(sizeof(fixed) / sizeof(fixed[0]));
        unit_h[units++] = head_h + text_h * (float)n_fixed;

        // Greedy fill, in order: how many columns does a column height of H need?
        auto columns_for = [&](float H, int* col_of) {
            int c = 0;
            float used = 0.0f;
            for (int u = 0; u < units; u++) {
                if (used > 0.0f && used + unit_h[u] > H) { c++; used = 0.0f; }
                used += unit_h[u];
                if (col_of) col_of[u] = c;
            }
            return c + 1;
        };
        // Room left after the title, hint, separators, buttons and a two-line conflict
        // warning, with a margin to the window edges.
        const float chrome = 36.0f + text_h * 6.0f + row_h * 2.0f;
        const float avail_h = std::max(200.0f, win_h - 40.0f - chrome);
        const int max_cols = std::max(1, (int)((win_w - 88.0f + col_gap) / (col_w + col_gap)));
        float total_h = 0.0f, tallest = 0.0f;
        for (int u = 0; u < units; u++) { total_h += unit_h[u]; tallest = std::max(tallest, unit_h[u]); }
        const int cols = std::min(max_cols, columns_for(std::max(avail_h, tallest), nullptr));
        // Balance: the shortest column height that still fits in `cols` columns.
        float H = std::max(tallest, total_h / (float)cols);
        while (columns_for(H, nullptr) > cols) H += row_h * 0.5f;
        int col_of[kMaxUnits];
        columns_for(H, col_of);
        const bool scrolls = H > avail_h;    // window too small even at max_cols
        const float list_w = cols * col_w + (cols - 1) * col_gap
                           + (scrolls ? ImGui::GetStyle().ScrollbarSize : 0.0f) + 8.0f;
        const float list_h = std::min(H, avail_h) + 4.0f;

        ImGui::BeginChild("##kblist", ImVec2(list_w, list_h), false);
        for (int c = 0; c < cols; c++) {
            if (c > 0) ImGui::SameLine(c * (col_w + col_gap));
            ImGui::BeginGroup();
            for (int u = 0; u < units; u++) {
                if (col_of[u] != c) continue;
                if (u == fixed_unit) {
                    // The keys that can't be rebound, listed so the window really shows every
                    // key. Key centred in the key column, like the buttons above it.
                    ImGui::Dummy(ImVec2(0.0f, 4.0f));
                    ImGui::TextColored(accent, "Fixed");
                    for (const auto& f : fixed) {
                        ImGui::TextDisabled("%s", f[1]);
                        ImGui::SameLine(label_w + std::max(0.0f, (key_w - ImGui::CalcTextSize(f[0]).x) * 0.5f));
                        ImGui::TextDisabled("%s", f[0]);
                    }
                    continue;
                }
                ImGui::Dummy(ImVec2(0.0f, 4.0f));
                ImGui::TextColored(accent, "%s", key_action_info((KeyAction)unit_first[u]).group);
                for (int i = unit_first[u]; i < unit_first[u + 1]; i++) {
                    if (i == (int)KeyAction::BRUSH_POSE) continue;
                    const KeyActionInfo& info = key_action_info((KeyAction)i);
                    ImGui::PushID(i);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted(info.label);
                    if (info.note && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", info.note);

                    ImGui::SameLine(label_w);
                    const bool listening = input.keybind_capture == i;
                    char buf[48];
                    if (listening) std::snprintf(buf, sizeof(buf), "press a key...##k");
                    else {
                        char name[32];
                        std::snprintf(buf, sizeof(buf), "%s##k", format_keybind(work[i], name, sizeof(name)));
                    }
                    int pushed = 0;
                    if (listening) {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.46f, 0.31f, 0.88f, 0.85f));
                        pushed++;
                    }
                    if (!work[i].bound()) { ImGui::PushStyleColor(ImGuiCol_Text, warn); pushed++; }
                    if (ImGui::Button(buf, ImVec2(key_w, 0.0f))) {
                        input.keybind_capture = listening ? -1 : i;
                        conflict_with = -1;
                        refusal[0] = '\0';
                    }
                    ImGui::PopStyleColor(pushed);

                    ImGui::SameLine();
                    ImGui::BeginDisabled(!work[i].bound());
                    if (ImGui::Button("X##clear", ImVec2(x_w, 0.0f))) {
                        work[i] = { GLFW_KEY_UNKNOWN, 0 };
                        if (listening) input.keybind_capture = -1;
                        conflict_with = -1;
                    }
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                        ImGui::SetTooltip("Remove this binding (leaves it as \?\?\?)");
                    ImGui::PopID();
                }
            }
            ImGui::EndGroup();
        }
        ImGui::EndChild();

        ImGui::Separator();
        if (conflict_with >= 0) {
            char want[32], old[32];
            format_keybind(pending, want, sizeof(want));
            format_keybind(work[pending_for], old, sizeof(old));
            ImGui::TextColored(warn, "%s is already used by %s.", want,
                               key_action_info((KeyAction)conflict_with).label);
            ImGui::TextColored(warn, "Continue to swap them: %s gets %s instead.",
                               key_action_info((KeyAction)conflict_with).label, old);
            if (ImGui::Button("Continue")) {
                work[conflict_with] = work[pending_for];
                work[pending_for] = pending;
                conflict_with = -1;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) conflict_with = -1;
            ImGui::Separator();
        } else if (refusal[0]) {
            ImGui::TextColored(warn, "%s", refusal);
            ImGui::Separator();
        }

        const float bw = ImGui::CalcTextSize("Save").x * 3.0f;
        if (ImGui::Button("Save", ImVec2(bw, 0.0f))) {
            for (int i = 0; i < (int)KeyAction::COUNT; i++) input.keybinds[i] = work[i];
            std::snprintf(input.notification, sizeof(input.notification), "Key bindings saved");
            input.notification_timer = 1.5f;
            close = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Exit", ImVec2(bw, 0.0f))) close = true;

        if (close) {
            input.keybind_capture = -1;
            conflict_with = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    if (close) input.keybind_menu_open = false;
}

// Brush properties: the knobs a brush has beyond the four meters, kept out of the docks.
// Same frame, padding and background as the key bindings window. Changes apply as you
// drag (and save on the settings debounce), so there is no Save/Cancel, just Close.
// The same window doubles as the unselected-colour picker (input.props_page).
void draw_brush_props_window(InputState& input, int win_w, int win_h) {
    if (input.brush_props_request) {
        input.brush_props_request = false;
        input.brush_props_esc = false;
        ImGui::OpenPopup("##brushprops");
    }
    input.brush_props_open = ImGui::IsPopupOpen("##brushprops");
    if (!input.brush_props_open) return;

    bool close = false;
    if (input.brush_props_esc) {
        input.brush_props_esc = false;
        close = true;
    }

    const bool modern = !input.ui_classic();
    const ImVec4 accent(0.72f, 0.58f, 0.95f, 1.0f);

    ImGui::SetNextWindowPos(ImVec2(win_w * 0.5f, win_h * 0.5f), ImGuiCond_Always,
                            ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 18));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, modern ? ImVec4(0, 0, 0, 0)
                                                   : ImVec4(0.10f, 0.10f, 0.13f, 0.97f));
    const ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::BeginPopupModal("##brushprops", nullptr, flags)) {
        if (modern) ui_skin_popup_background();

        const bool colour_page = input.props_page == InputState::PropsPage::UNSELECTED_COLOUR;
        ImGui::TextColored(modern ? ImVec4(0.93f, 0.92f, 0.95f, 1.0f)
                                  : ImVec4(1.0f, 1.0f, 0.40f, 1.0f),
                           colour_page ? "Unselected colour" : "Brush properties");
        ImGui::TextDisabled("Changes apply as you drag. Esc closes.");
        ImGui::Separator();

        const float label_w = ImGui::CalcTextSize("Build").x + 16.0f;
        // One labelled 0-100% slider row with a muted note under it.
        auto pct_row = [&](const char* label, const char* id, float* v, const char* zero,
                           const char* note) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            ImGui::SameLine(label_w + ImGui::GetStyle().WindowPadding.x);
            ImGui::SetNextItemWidth(240.0f);
            float pct = *v * 100.0f;
            if (ImGui::SliderFloat(id, &pct, 0.0f, 100.0f, zero && pct <= 0.0f ? zero : "%.0f%%"))
                *v = pct * 0.01f;
            ImGui::TextDisabled("%s", note);
        };

        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        if (colour_page) {
            // The tint multiplies the shaded surface, so a mesh shows a little darker
            // than the swatch; pick on the light side.
            ImGui::SetNextItemWidth(260.0f);
            ImGui::ColorPicker3("##unselTint", input.unselected_tint,
                                ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoLabel |
                                ImGuiColorEditFlags_DisplayHex);
            ImGui::TextDisabled("Multiplies the shading, so meshes read a bit darker.");
        } else {
            ImGui::TextColored(accent, "Crease");
            pct_row("Dot", "##creaseDot", &input.crease_dot, nullptr,
                    "0%: a groove along the stroke. 100%: every dab a round pit.");
            ImGui::Dummy(ImVec2(0.0f, 2.0f));
            pct_row("Build", "##creaseBuild", &input.crease_build, "Off",
                    "Hold the brush still and it keeps sinking there. Click and hold: a pit.");
            // Clay melt: how strongly clay eases proud verts back to its plane, so
            // building across an old stroke evens out instead of stacking.
            ImGui::Dummy(ImVec2(0.0f, 6.0f));
            ImGui::TextColored(accent, "Clay");
            pct_row("Melt", "##clayMelt", &input.clay_melt, "Off",
                    "How hard clay cuts raised areas down to its plane. Off: rides over them.");
        }
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        ImGui::Separator();

        const float bw = ImGui::CalcTextSize("Close").x * 3.0f;
        if (ImGui::Button("Close", ImVec2(bw, 0.0f))) close = true;

        if (close) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    if (close) input.brush_props_open = false;
}

// Trash-can button for one layer row: outline only, no background until hovered (then a
// red tint), so it reads as quieter than the row and isn't hit by accident. Returns true
// on click.
static bool trash_button(const char* id, float size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();
    if (hovered)
        dl->AddRectFilled(p, ImVec2(p.x + size, p.y + size),
                          ImGui::GetColorU32(ImVec4(0.60f, 0.18f, 0.18f, ImGui::IsItemActive() ? 1.0f : 0.75f)),
                          size * 0.22f);
    ImU32 col = ImGui::GetColorU32(hovered ? col_text_sel : col_text);
    const float s = size, th = 1.4f;
    auto P = [&](float fx, float fy) { return ImVec2(p.x + s * fx, p.y + s * fy); };
    dl->AddLine(P(0.22f, 0.28f), P(0.78f, 0.28f), col, th);                  // lid
    dl->AddLine(P(0.40f, 0.28f), P(0.42f, 0.18f), col, th);                  // handle
    dl->AddLine(P(0.42f, 0.18f), P(0.58f, 0.18f), col, th);
    dl->AddLine(P(0.58f, 0.18f), P(0.60f, 0.28f), col, th);
    dl->AddLine(P(0.28f, 0.34f), P(0.33f, 0.82f), col, th);                  // can
    dl->AddLine(P(0.33f, 0.82f), P(0.67f, 0.82f), col, th);
    dl->AddLine(P(0.67f, 0.82f), P(0.72f, 0.34f), col, th);
    dl->AddLine(P(0.44f, 0.42f), P(0.45f, 0.72f), col, 1.0f);                // ribs
    dl->AddLine(P(0.56f, 0.42f), P(0.55f, 0.72f), col, 1.0f);
    return clicked;
}

// ---- Paint layers panel ----
// A minimal layer docker: top layer first; the eye toggles visibility, a click
// makes a layer active; the active layer's mode, opacity and clip sit underneath, then
// add / up / down / merge visible; each row ends in its own trash can. Paint mode
// (colour target) only. Edits that move pixels (add, delete, switch, merge) go through Scene::edit_paint_layers so a
// multires entity edits its full-resolution planes; property edits don't move pixels,
// so they mutate the stack and ask for a GPU re-blend.
void draw_paint_layers_panel(InputState& input, Scene& scene, int win_w, int /*win_h*/) {
    if (input.current_brush != BrushType::PAINT || input.paint_target_density) return;
    if (input.interaction_mode == InputState::InteractionMode::RETOPO) return;   // own page
    if (input.interaction_mode == InputState::InteractionMode::SELECT) return;   // the mesh list's spot
    MeshEntity* ent = scene.find_entity(scene.active_mesh_id());
    if (!ent) return;
    const uint32_t eid = ent->id;
    PaintLayerStack& st = ent->mesh.paint_layers;
    using Op = std::function<void(PaintLayerStack&, std::vector<uint32_t>&, uint32_t)>;
    auto move_pixels = [&](const Op& op) { scene.edit_paint_layers(eid, op); };
    auto props       = [&]() { scene.paint_layers_changed(eid, false); };
    // Layer being renamed, and on which entity, since layer ids are only unique per
    // stack. Its own on/off flag: Base is id 0, so no id can mean "none". Static: it has
    // to outlive the frame.
    static bool rename_on = false;
    static uint32_t rename_id = 0, rename_eid = 0;
    static bool rename_focus = false;
    static char rename_buf[64];

    ImGui::SetNextWindowPos(ImVec2((float)win_w - 16.0f, 140.0f), ImGuiCond_FirstUseEver,
                            ImVec2(1.0f, 0.0f));
    if (!ImGui::Begin("Layers", nullptr,
                      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::End();
        return;
    }

    if (st.empty()) {
        // The pre-layer world: one implicit Base. Nothing to toggle until a layer exists.
        ImGui::TextDisabled("Base");
    }
    for (int i = (int)st.layers.size() - 1; i >= 0; i--) {
        PaintLayer& L = st.layers[i];
        ImGui::PushID((int)L.id);
        bool vis = L.visible;
        if (ImGui::Checkbox("##vis", &vis)) { L.visible = vis; props(); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show / hide");
        ImGui::SameLine();
        if (rename_on && rename_id == L.id && rename_eid == eid) {
            // Inline rename: Enter or clicking away keeps the edit, Esc drops it, and
            // an empty name keeps the old one.
            if (rename_focus) { ImGui::SetKeyboardFocusHere(); rename_focus = false; }
            ImGui::SetNextItemWidth(150.0f);
            ImGui::InputText("##rename", rename_buf, sizeof(rename_buf),
                             ImGuiInputTextFlags_AutoSelectAll);
            if (ImGui::IsItemDeactivated()) {
                if (!ImGui::IsKeyPressed(ImGuiKey_Escape) && rename_buf[0]) L.name = rename_buf;
                rename_on = false;
            }
        } else {
            char label[96];
            std::snprintf(label, sizeof(label), "%s%s", L.clip ? "\xC2\xBB " : "", L.name.c_str());
            if (ImGui::Selectable(label, i == st.active, 0, ImVec2(150.0f, 0.0f)) && i != st.active)
                move_pixels([i](PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vc) {
                    paint_layer_set_active(s, act, i, vc);
                });
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                rename_on = true;
                rename_id = L.id;
                rename_eid = eid;
                rename_focus = true;
                std::snprintf(rename_buf, sizeof(rename_buf), "%s", L.name.c_str());
            }
            // Drag a row onto another to reorder. The payload is the layer id, not its
            // index, so it can't go stale. The dragged layer takes the target's slot,
            // which lands above the target when moving up and below it when moving down;
            // a line on that edge says which.
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload("CHISEL_LAYER", &L.id, sizeof(L.id));
                ImGui::TextUnformatted(L.name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginDragDropTarget()) {
                const ImGuiPayload* pl = ImGui::AcceptDragDropPayload(
                    "CHISEL_LAYER", ImGuiDragDropFlags_AcceptBeforeDelivery
                                    | ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
                if (pl) {
                    uint32_t from_id;
                    std::memcpy(&from_id, pl->Data, sizeof(from_id));
                    const int from = paint_layers_index_of(st, from_id);
                    if (from >= 0 && from != i) {
                        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
                        const float y = from < i ? a.y : b.y;
                        ImGui::GetWindowDrawList()->AddLine(ImVec2(a.x, y), ImVec2(b.x, y),
                                                            ImGui::GetColorU32(col_text_sel), 2.0f);
                        if (pl->IsDelivery()) { paint_layer_move(st, from, i); props(); }
                    }
                }
                ImGui::EndDragDropTarget();
            }
        }
        // Delete sits at the far end of its own row, not next to "+": a stray click on
        // the add button must never cost a layer (layer edits aren't undoable yet).
        ImGui::SameLine();
        if (trash_button("##del", ImGui::GetFrameHeight()))
            move_pixels([i](PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vc) {
                paint_layer_remove(s, act, i, vc);
            });
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(st.layers.size() == 1 ? "Clear this layer (it's the only one)"
                                                    : "Delete this layer");
        ImGui::PopID();
    }

    // Under every layer, a textured mesh's texture (an imported glTF): its eye and
    // opacity, no selecting, since brushes paint the layers above it (texture painting
    // comes later). Read straight off the stack each frame, so nothing to re-blend.
    if (ent->mesh.texture && ent->mesh.has_uv()) {
        ImGui::PushID("texture_row");
        bool tv = st.tex_visible;
        if (ImGui::Checkbox("##tvis", &tv)) { st.tex_visible = tv; scene.mark_edited(); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show / hide the texture");
        ImGui::SameLine();
        char label[48];
        std::snprintf(label, sizeof(label), "Texture %dx%d", ent->mesh.texture->w, ent->mesh.texture->h);
        ImGui::BeginDisabled(!st.tex_visible);
        const float label_x = ImGui::GetCursorPosX();   // where the layer names start
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("The imported texture, under every paint layer.\n"
                              "Paint goes on the layers above it.");
        // The slider sits where the layer rows put their trash can, past the 150 px name.
        ImGui::SameLine(label_x + 150.0f + ImGui::GetStyle().ItemSpacing.x);
        float pct = st.tex_opacity * 100.0f;
        ImGui::SetNextItemWidth(ImGui::GetFrameHeight() * 2.6f);
        if (ImGui::SliderFloat("##topacity", &pct, 0.0f, 100.0f, "%.0f%%")) {
            st.tex_opacity = pct / 100.0f;
            scene.mark_edited();
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Texture opacity: fades it toward the bare clay");
        ImGui::EndDisabled();
        ImGui::PopID();
    }

    ImGui::Separator();
    if (!st.empty() && st.active >= 0) {
        PaintLayer& A = st.layers[st.active];
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::BeginCombo("##mode", blend_mode_name(A.mode))) {
            for (int m = 0; m < (int)BlendMode::Count; m++) {
                const bool sel = (int)A.mode == m;
                if (ImGui::Selectable(blend_mode_name((BlendMode)m), sel) && !sel) {
                    A.mode = (BlendMode)m;
                    props();
                }
            }
            ImGui::EndCombo();
        }
        float pct = A.opacity * 100.0f;
        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::SliderFloat("##opacity", &pct, 0.0f, 100.0f, "opacity %.0f%%")) {
            A.opacity = pct / 100.0f;
            props();
        }
        bool clip = A.clip;
        if (ImGui::Checkbox("Clip to layer below", &clip)) { A.clip = clip; props(); }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Only show this layer where the layer under it is painted\n"
                              "(a clipping mask)");
        ImGui::Separator();
    }

    const bool full = (int)st.layers.size() >= kMaxPaintLayers;
    ImGui::BeginDisabled(full);
    if (ImGui::Button("+")) {
        char name[32];
        std::snprintf(name, sizeof(name), "Layer %u", st.next_id);
        const std::string nm = name;
        move_pixels([nm](PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vc) {
            paint_layer_add(s, act, nm, vc);
        });
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(full ? "16 layers is the limit" : "New layer above the active one");
    ImGui::SameLine();
    ImGui::BeginDisabled(st.empty());
    ImGui::BeginDisabled(st.active >= (int)st.layers.size() - 1);
    if (ImGui::Button("Up")) { paint_layer_move(st, st.active, st.active + 1); props(); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(st.active <= 0);
    if (ImGui::Button("Down")) { paint_layer_move(st, st.active, st.active - 1); props(); }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Merge visible"))
        move_pixels([](PaintLayerStack& s, std::vector<uint32_t>& act, uint32_t vc) {
            paint_layers_merge_visible(s, act, vc);
        });
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Merge every visible layer into one (hidden layers are kept)");
    ImGui::EndDisabled();

    ImGui::End();
}

// ---- Retopo mode --------------------------------------------------------------------

namespace {

// Does a surface point at p with normal n face the viewer? Free, needs no plane cache,
// so it is the one occlusion test that still works mid-orbit.
bool retopo_facing(const Camera& cam, Vec3 p, Vec3 n) {
    const Vec3 to_eye = cam.perspective ? cam.get_position() - p : cam.get_view_direction() * -1.0f;
    return n.dot(to_eye) > 0.0f;
}

void retopo_band(ImDrawList* dl, const RetopoStroke& s, const Camera& cam, int win_w, int win_h,
                 ImU32 fill, ImU32 line, bool flip_x, float wscale) {
    const size_t n = s.pts.size();
    if (n < 2) return;
    static std::vector<ImVec2> at;
    static std::vector<float>  rad;
    static std::vector<uint8_t> ok;
    at.resize(n); rad.resize(n); ok.resize(n);
    for (size_t i = 0; i < n; i++) {
        Vec3 p = s.pts[i], nr = s.nrm[i];
        if (flip_x) { p.x = -p.x; nr.x = -nr.x; }
        float x, y;
        ok[i] = cam.world_to_screen(p, win_w, win_h, x, y) && retopo_facing(cam, p, nr)
             && !(cam.perspective && cam.view_depth(p) <= 1e-4f);
        at[i] = ImVec2(x, y);
        const float wpp = 2.0f * cam.half_height_at(cam.view_depth(p)) / (float)win_h;
        rad[i] = wpp > 0.0f ? s.half_w[i] * wscale / wpp : 0.0f;
    }
    // Each point's "across" direction comes from neighbours half a band-width away
    // on either side, not from its own 3 px segment: at that lever arm the hand's
    // jitter swung every edge of the band and it drew as bristles. Neighbouring
    // quads then share their edges, so the band is one continuous ribbon.
    static std::vector<ImVec2> perp;
    perp.resize(n);
    auto wrap = [&](long k) -> size_t {
        if (s.closed) return (size_t)((k % (long)n + (long)n) % (long)n);
        return (size_t)std::max(0L, std::min((long)n - 1, k));
    };
    for (size_t i = 0; i < n; i++) {
        const float reach = std::max(4.0f, rad[i] * 0.5f);
        long lo = (long)i, hi = (long)i;
        float dlo = 0.0f, dhi = 0.0f;
        for (size_t step = 0; step < n && (dlo < reach || dhi < reach); step++) {
            if (dlo < reach) { size_t a = wrap(lo), b = wrap(lo - 1); dlo += std::hypot(at[a].x - at[b].x, at[a].y - at[b].y); lo--; }
            if (dhi < reach) { size_t a = wrap(hi), b = wrap(hi + 1); dhi += std::hypot(at[a].x - at[b].x, at[a].y - at[b].y); hi++; }
        }
        const ImVec2 A = at[wrap(lo)], B = at[wrap(hi)];
        float dx = B.x - A.x, dy = B.y - A.y;
        const float l = std::sqrt(dx * dx + dy * dy);
        perp[i] = l > 1e-3f ? ImVec2(-dy / l, dx / l) : ImVec2(0.0f, 0.0f);
    }
    // No AA fringe on the fills: two fringes overlapping on a shared edge read as a
    // seam line across the ribbon.
    const ImDrawListFlags was = dl->Flags;
    dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;
    const size_t segs = s.closed ? n : n - 1;
    for (size_t i = 0; i < segs; i++) {
        const size_t j = (i + 1) % n;
        if (!ok[i] || !ok[j]) continue;
        const ImVec2 a0(at[i].x + perp[i].x * rad[i], at[i].y + perp[i].y * rad[i]);
        const ImVec2 a1(at[i].x - perp[i].x * rad[i], at[i].y - perp[i].y * rad[i]);
        const ImVec2 b0(at[j].x + perp[j].x * rad[j], at[j].y + perp[j].y * rad[j]);
        const ImVec2 b1(at[j].x - perp[j].x * rad[j], at[j].y - perp[j].y * rad[j]);
        dl->AddQuadFilled(a0, b0, b1, a1, fill);
    }
    dl->Flags = was;
    for (size_t i = 0; i < segs; i++) {
        const size_t j = (i + 1) % n;
        if (ok[i] && ok[j]) dl->AddLine(at[i], at[j], line, 1.5f);
    }
}

}  // namespace

void draw_retopo_overlay(const RetopoState& st, Renderer& renderer, const Camera& cam,
                         int win_w, int win_h, bool depth_fresh, bool mirror, bool uv) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    if (st.show_strokes) {
        auto band = [&](const RetopoStroke& s, int alpha) {
            const ImU32 fill = s.closed ? IM_COL32(70, 200, 220, alpha) : IM_COL32(255, 175, 70, alpha);
            // Selected: a white centreline, so the selection reads without a hover.
            const ImU32 line = s.selected ? IM_COL32(255, 255, 255, 255)
                             : s.closed ? IM_COL32(150, 235, 245, std::min(255, alpha * 3))
                                        : IM_COL32(255, 210, 150, std::min(255, alpha * 3));
            // The live stroke has no settings yet: pen-up gives it 1x.
            const float w = &s == &st.live ? 1.0f : s.width;
            retopo_band(dl, s, cam, win_w, win_h, fill, line, false, w);
            if (!mirror) return;
            bool pos = false, neg = false;
            for (const Vec3& p : s.pts) { pos |= p.x > 1e-4f; neg |= p.x < -1e-4f; }
            if (pos != neg)   // the twin Build will make (retopo_build's rule)
                retopo_band(dl, s, cam, win_w, win_h, fill & 0x3FFFFFFFu, line & 0x5FFFFFFFu, true, w);
        };
        // Hovering a stroke in the bar lights it and dims the rest, hidden ones
        // included — so you can see what a checkbox or trash can is about to hit.
        const bool any_hover = st.hover_id != 0;
        for (const RetopoStroke& s : st.strokes) {
            const bool lit = s.id == st.hover_id;
            // A frozen stroke is geometry now: no band, only its quads (lit when hovered).
            if (!s.frozen && (s.visible || lit)) band(s, lit ? 130 : s.selected ? 90 : any_hover ? 25 : 55);
        }
        if (st.drawing && !st.bridge_from && st.spine_from == ~0u && st.live.pts.size() >= 2) band(st.live, 70);
    }
    // The cut line under the pen: a thin bright line with a dark edge, no band. A bridge
    // (Draw from an edited island) is drawn the same way in blue: its width comes from
    // the edges it joins, not the band.
    auto thin_line = [&](const std::vector<Vec3>& pts, const std::vector<Vec3>& nrm, ImU32 col) {
        if (pts.size() < 2) return;
        static std::vector<ImVec2> at;
        static std::vector<uint8_t> ok;
        const size_t n = pts.size();
        at.resize(n); ok.resize(n);
        for (size_t i = 0; i < n; i++) {
            float x, y;
            ok[i] = cam.world_to_screen(pts[i], win_w, win_h, x, y) && retopo_facing(cam, pts[i], nrm[i]);
            at[i] = ImVec2(x, y);
        }
        for (int pass = 0; pass < 2; pass++)
            for (size_t i = 0; i + 1 < n; i++)
                if (ok[i] && ok[i + 1])
                    dl->AddLine(at[i], at[i + 1], pass ? col : IM_COL32(10, 20, 15, 170), pass ? 1.5f : 3.5f);
    };
    if (st.cutting) thin_line(st.cut_pts, st.cut_nrm, IM_COL32(235, 255, 240, 255));
    if (st.drawing && (st.bridge_from || st.spine_from != ~0u))
        thin_line(st.live.pts, st.live.nrm, IM_COL32(120, 210, 255, 255));
    if (st.drawing && st.extruding) thin_line(st.live.pts, st.live.nrm, IM_COL32(0, 255, 65, 255));
    // The wand's lasso: drawn as dragged, its closing side dimmer (it closes on release).
    if (st.lasso.size() >= 4) {
        static std::vector<ImVec2> lp;
        lp.clear();
        for (size_t i = 0; i + 1 < st.lasso.size(); i += 2) lp.push_back(ImVec2(st.lasso[i], st.lasso[i + 1]));
        dl->AddPolyline(lp.data(), (int)lp.size(), IM_COL32(10, 20, 15, 170), 0, 3.5f);
        dl->AddPolyline(lp.data(), (int)lp.size(), IM_COL32(255, 200, 90, 255), 0, 1.5f);
        dl->AddLine(lp.back(), lp.front(), IM_COL32(255, 200, 90, 110), 1.0f);
    }
    // A paste target (a lasso let go on bare sculpt with something copied): closed, in
    // the clipboard's cyan, while the view is still the one it was drawn in.
    if (st.paste_ready && st.paste_lasso.size() >= 6
        && (cam.get_position() - st.paste_eye).length() <= 1e-4f * std::max(1e-3f, cam.distance)) {
        static std::vector<ImVec2> pp;
        pp.clear();
        for (size_t i = 0; i + 1 < st.paste_lasso.size(); i += 2) pp.push_back(ImVec2(st.paste_lasso[i], st.paste_lasso[i + 1]));
        dl->AddPolyline(pp.data(), (int)pp.size(), IM_COL32(10, 20, 15, 170), ImDrawFlags_Closed, 3.5f);
        dl->AddPolyline(pp.data(), (int)pp.size(), IM_COL32(140, 227, 238, 255), ImDrawFlags_Closed, 1.5f);
    }
    if (!st.built || !st.show_result) return;

    // Wireframe: project every vertex once, then draw an edge only when both ends
    // can be seen. A vertex facing away is hidden; one that faces us but (cache fresh)
    // sits deeper than the sculpt's own depth at that pixel by more than a few pixels'
    // worth of world (the mirror lobe marker's tolerance) is behind a part of the
    // sculpt: its edges still show, darker and thinner, never filled. So does one facing
    // away but only a few quads behind the surface in front of it (an ear's inner fold,
    // under a lip); deeper than that it's the far side of the model and stays hidden.
    // vis: 0 hidden, 1 behind the sculpt, 2 in view.
    const RetopoMesh& rm = st.result;
    static std::vector<ImVec2> at;
    static std::vector<uint8_t> vis;
    at.resize(rm.v.size()); vis.resize(rm.v.size());
    float reach = 0.0f;   // "a few quads": 4x the mean edge
    for (size_t e = 0; e + 1 < rm.edges.size(); e += 2)
        reach += (rm.v[rm.edges[e]] - rm.v[rm.edges[e + 1]]).length();
    reach *= 4.0f / (float)std::max<size_t>(1, rm.edges.size() / 2);
    const float edge_mean = reach / 4.0f;
    for (size_t i = 0; i < rm.v.size(); i++) {
        float x, y;
        // Past the window's edge still counts: its edges run off the screen and get
        // clipped there. Treating it as hidden dropped every edge and face with a corner
        // off screen, so zooming in even a little ate the wire round the borders.
        const bool on_screen = cam.world_to_screen(rm.v[i], win_w, win_h, x, y)
                            || !cam.perspective || cam.view_depth(rm.v[i]) > 1e-4f;
        // How square-on the surface is here (1 = facing us, 0 = edge-on). A vertex
        // just past edge-on still counts as facing: on a curve seen side-on half the
        // silhouette's vertices tip a hair away and their edges used to vanish.
        Vec3 to_eye = cam.perspective ? cam.get_position() - rm.v[i] : cam.get_view_direction() * -1.0f;
        const float te = to_eye.length(), nl = rm.n[i].length();
        const float cosv = te > 1e-6f && nl > 1e-6f ? rm.n[i].dot(to_eye) / (te * nl) : 1.0f;
        const bool facing = cosv > -0.15f;
        uint8_t v = on_screen && facing ? 2 : 0;
        if (on_screen && depth_fresh && x >= 0 && y >= 0 && x < win_w && y < win_h) {
            float front = 0.0f;
            if (renderer.sample_depth((int)x, (int)y, &front) && front < 500.0f) {
                const float depth = cam.view_depth(rm.v[i]);
                // Seen edge-on, one pixel of screen spans many pixels' worth of depth
                // (about 1 / cos), so a flat 6 px slack called most vertices on an
                // oblique surface "behind the sculpt" and drew them dim. Scale it.
                // And never under one mean edge of world: in pixels alone the slack
                // shrinks as you zoom in, until vertices a hair under the sculpt (after
                // a level change, say) read as hidden. Anything that really hides a
                // vertex sits further in front than one of its quads.
                const float tol = std::max(edge_mean, 2.0f * cam.half_height_at(depth) / (float)win_h
                                                      * 6.0f / std::max(std::fabs(cosv), 0.15f));
                if (facing ? depth > front + tol : depth <= front + reach) v = 1;
            }
        }
        at[i] = ImVec2(x, y);
        vis[i] = v;
    }
    const ImU32 dark = IM_COL32(15, 25, 35, 150), lite = IM_COL32(170, 240, 255, 235);
    const ImU32 hot = IM_COL32(255, 235, 120, 255), dim = IM_COL32(170, 240, 255, 110);
    const ImU32 sel_col = IM_COL32(255, 150, 60, 255);
    // Selected strokes' ids: a handful at most, a linear look-up is fine.
    static std::vector<uint32_t> sel;
    sel.clear();
    for (const RetopoStroke& s : st.strokes) if (s.selected) sel.push_back(s.id);
    auto is_sel = [&](uint32_t id) { return std::find(sel.begin(), sel.end(), id) != sel.end(); };
    // Faces: a faint fill under the wire, so what's covered reads at a glance. Plain
    // translucent triangles, no anti-aliased fringe: half the vertices, and neighbours'
    // fringes would otherwise double up into a seam the wire already draws. Quads and
    // triangles take the convex path; only Erase's merged n-gons (maybe concave) the
    // concave one. Hidden if any corner is.
    // UV mode with a fresh layout: each face in its island's colour (the UV panel's),
    // the island hovered or picked there bright and the rest faint.
    const bool tint = uv && st.uv.fits(rm) && st.uv.key == uv_topology_key(rm);
    const bool island_lit = tint && (st.uv_hover < st.uv.charts || st.uv_sel < st.uv.charts);
    // The bake preview stands in for the sculpt: no fill, wire, seams or loops over it,
    // only the brush's ring, so what's judged is the baked maps. Seams come back while a
    // seam stroke is under the pen: painting them blind would be guesswork.
    const bool baked = uv && st.bake.showing();
    if (!baked) {
        const ImDrawListFlags keep_flags = dl->Flags;
        dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;
        static std::vector<ImVec2> poly;
        for (uint32_t f = 0; f < rm.face_count(); f++) {
            const uint32_t ow = rm.face_owner[f];
            if (st.hidden(ow) && ow != st.hover_id) continue;
            const uint32_t* q = rm.face(f);
            const uint32_t k = rm.face_size(f);
            poly.clear();
            bool ok = true;
            for (uint32_t i = 0; i < k && ok; i++) {
                ok = vis[q[i]] == 2;
                if (i == 0 || q[i] != q[i - 1]) poly.push_back(at[q[i]]);   // welds leave repeats
            }
            if (!ok || poly.size() < 3) continue;
            const uint32_t isl = tint ? st.uv.face_chart[f] : ~0u;
            const ImU32 col = isl != ~0u ? uv_island_color(isl, isl == st.uv_hover || isl == st.uv_sel ? 0.45f
                                                                : island_lit ? 0.08f : 0.24f)
                            : ow == st.hover_id ? IM_COL32(255, 240, 160, 48)
                            : is_sel(ow)        ? IM_COL32(255, 190, 120, 40)
                            : st.hover_id       ? IM_COL32(205, 245, 255, 16)
                                                : IM_COL32(205, 245, 255, 36);
            if (poly.size() <= 4) dl->AddConvexPolyFilled(poly.data(), (int)poly.size(), col);
            else                  dl->AddConcavePolyFilled(poly.data(), (int)poly.size(), col);
            // Masked: darkened by its corners' mean mask, as a masked sculpt is.
            float mk = 0.0f;
            for (uint32_t i = 0; i < k; i++) mk += rm.mask_at(q[i]);
            mk /= (float)k;
            if (mk > 0.01f) {
                const ImU32 shade = IM_COL32(8, 12, 22, (int)(170.0f * std::min(1.0f, mk)));
                if (poly.size() <= 4) dl->AddConvexPolyFilled(poly.data(), (int)poly.size(), shade);
                else                  dl->AddConcavePolyFilled(poly.data(), (int)poly.size(), shade);
            }
        }
        dl->Flags = keep_flags;
    }
    // Holes (outlines that aren't a piece's outer edge): a wash over the gap, green when
    // its count of sides is even (Fill makes it quads), amber when odd, rim drawn the same.
    // Only when the whole rim is in view: a half-seen rim would fill the wrong shape.
    if (!baked) {
        const ImDrawListFlags keep_flags = dl->Flags;
        dl->Flags &= ~ImDrawListFlags_AntiAliasedFill;
        static std::vector<ImVec2> poly;
        for (uint32_t r = 0; r < rm.rim_count(); r++) {
            if (!rm.rim_hole[r]) continue;
            poly.clear();
            bool ok = true;
            for (uint32_t k = rm.rim_off[r]; k < rm.rim_off[r + 1] && ok; k++) {
                const uint32_t v = rm.rim[k];
                ok = vis[v] == 2 && !(st.hidden(rm.v_owner[v]) && rm.v_owner[v] != st.hover_id);
                poly.push_back(at[v]);
            }
            if (!ok || poly.size() < 3) continue;
            const bool even = poly.size() % 2 == 0;
            dl->AddConcavePolyFilled(poly.data(), (int)poly.size(),
                                     even ? IM_COL32(60, 230, 120, 60) : IM_COL32(255, 170, 40, 70));
            dl->AddPolyline(poly.data(), (int)poly.size(),
                            even ? IM_COL32(90, 240, 140, 200) : IM_COL32(255, 180, 60, 220), ImDrawFlags_Closed, 1.5f);
        }
        dl->Flags = keep_flags;
    }
    // Behind the sculpt (either end): one dim pass under the rest, no dark outline.
    for (size_t e = 0; e + 1 < rm.edges.size() && !baked; e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (!vis[a] || !vis[b] || (vis[a] == 2 && vis[b] == 2)) continue;
        const uint32_t ow = rm.edge_owner[e / 2];
        if (st.hidden(ow) && ow != st.hover_id) continue;
        const bool lit = ow == st.hover_id, on = is_sel(ow);
        dl->AddLine(at[a], at[b], lit ? IM_COL32(150, 135, 70, 190) : on ? IM_COL32(150, 90, 40, 170)
                                  : IM_COL32(70, 110, 130, st.hover_id ? 70 : 150), 1.0f);
    }
    for (size_t e = 0; e + 1 < rm.edges.size() && !baked; e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (vis[a] != 2 || vis[b] != 2) continue;
        const uint32_t ow = rm.edge_owner[e / 2];
        if (st.hidden(ow) && ow != st.hover_id) continue;   // its eye is shut (hover still shows it)
        dl->AddLine(at[a], at[b], dark, ow == st.hover_id || is_sel(ow) ? 4.0f : 3.0f);
    }
    for (size_t e = 0; e + 1 < rm.edges.size() && !baked; e += 2) {
        const uint32_t a = rm.edges[e], b = rm.edges[e + 1];
        if (vis[a] != 2 || vis[b] != 2) continue;
        const uint32_t ow = rm.edge_owner[e / 2];
        if (st.hidden(ow) && ow != st.hover_id) continue;
        const bool lit = ow == st.hover_id, on = is_sel(ow);
        dl->AddLine(at[a], at[b], lit ? hot : on ? sel_col : st.hover_id ? dim : lite,
                    lit || on ? 2.0f : 1.5f);
    }
    // UV seams: red over the wire, thin where the sculpt is in front.
    if (!baked || st.seam_painting) {
        static std::vector<uint32_t> seams;
        retopo_seam_edges(rm, seams);
        for (size_t e = 0; e + 1 < seams.size(); e += 2) {
            const uint32_t a = seams[e], b = seams[e + 1];
            if (!vis[a] || !vis[b] || st.hidden(rm.v_owner[a])) continue;
            if (vis[a] == 2 && vis[b] == 2) {
                dl->AddLine(at[a], at[b], IM_COL32(30, 0, 0, 220), 5.0f);
                dl->AddLine(at[a], at[b], IM_COL32(255, 50, 40, 255), 2.5f);
            } else {
                dl->AddLine(at[a], at[b], IM_COL32(220, 40, 30, 150), 1.0f);
            }
        }
    }
    // Double-clicked edge loops: green over the wire, thin where the sculpt is in front.
    // A picked edge a weld or the eraser took away isn't an edge any more: skipped.
    for (size_t e = 0; e + 1 < st.loop_edges.size() && !baked; e += 2) {
        const uint32_t a = st.loop_edges[e], b = st.loop_edges[e + 1];
        if (a >= rm.v.size() || b >= rm.v.size() || !vis[a] || !vis[b]) continue;
        if (st.hidden(rm.v_owner[a]) || rm.nb_off.size() != rm.v.size() + 1) continue;
        bool edge = false;
        for (uint32_t k = rm.nb_off[a]; k < rm.nb_off[a + 1] && !edge; k++) edge = rm.nb[k] == b;
        if (!edge) continue;
        if (vis[a] == 2 && vis[b] == 2) {
            dl->AddLine(at[a], at[b], IM_COL32(0, 20, 8, 220), 5.0f);
            dl->AddLine(at[a], at[b], IM_COL32(0, 255, 65, 255), 2.5f);
        } else {
            dl->AddLine(at[a], at[b], IM_COL32(0, 200, 55, 150), 1.0f);
        }
    }
    // Cut's vertex snap: the vertex under the pointer gets a ring, the one a cut started
    // on a filled dot, so a vertex cut is never a surprise.
    auto mark = [&](uint32_t i, bool filled) {
        if (i >= rm.v.size()) return;
        float x, y;
        if (!cam.world_to_screen(rm.v[i], win_w, win_h, x, y)) return;
        const ImU32 green = IM_COL32(0, 255, 65, 255), dark = IM_COL32(0, 20, 8, 200);
        dl->AddCircle(ImVec2(x, y), 6.5f, dark, 16, 4.0f);
        dl->AddCircle(ImVec2(x, y), 6.5f, green, 16, 2.0f);
        if (filled) dl->AddCircleFilled(ImVec2(x, y), 3.5f, green, 12);
    };
    // Ctrl + Erase: the face about to go, outlined red.
    if (st.face_hover < rm.face_count()) {
        const uint32_t* q = rm.face(st.face_hover);
        const uint32_t k = rm.face_size(st.face_hover);
        static std::vector<ImVec2> poly;
        poly.clear();
        for (uint32_t i = 0; i < k; i++) {
            float x, y;
            if (!cam.world_to_screen(rm.v[q[i]], win_w, win_h, x, y)) { poly.clear(); break; }
            poly.push_back(ImVec2(x, y));
        }
        if (poly.size() >= 3) {
            dl->AddPolyline(poly.data(), (int)poly.size(), IM_COL32(20, 0, 0, 200), ImDrawFlags_Closed, 5.0f);
            dl->AddPolyline(poly.data(), (int)poly.size(), IM_COL32(255, 60, 50, 255), ImDrawFlags_Closed, 2.5f);
        }
    }
    // The seam brush's vertex: a red ring, white when Ctrl would erase.
    if (st.seam_v < rm.v.size() && vis[st.seam_v] == 2) {
        const ImVec2 p = at[st.seam_v];
        dl->AddCircle(p, 7.0f, IM_COL32(20, 0, 0, 200), 16, 4.0f);
        dl->AddCircle(p, 7.0f, st.seam_erase ? IM_COL32(255, 255, 255, 255) : IM_COL32(255, 50, 40, 255), 16, 2.0f);
    }
    if (st.cutting) mark(st.cut_from_v, true);
    if (st.snap_v != st.cut_from_v || !st.cutting) mark(st.snap_v, false);
    // Draw's snap: an orange dot on the vertex a line starts or ends on, else the edge
    // in bold blue, so a line meant for a vertex never quietly lands on an edge.
    if (st.draw_v < rm.v.size() && vis[st.draw_v] == 2) {
        float x, y;
        if (cam.world_to_screen(rm.v[st.draw_v], win_w, win_h, x, y)) {
            dl->AddCircleFilled(ImVec2(x, y), 6.5f, IM_COL32(30, 12, 0, 220), 16);
            dl->AddCircleFilled(ImVec2(x, y), 4.5f, IM_COL32(255, 150, 30, 255), 16);
        }
    } else if (st.draw_ea < rm.v.size() && st.draw_eb < rm.v.size()
               && vis[st.draw_ea] == 2 && vis[st.draw_eb] == 2) {
        dl->AddLine(at[st.draw_ea], at[st.draw_eb], IM_COL32(5, 15, 30, 220), 6.0f);
        dl->AddLine(at[st.draw_ea], at[st.draw_eb], IM_COL32(120, 210, 255, 255), 3.0f);
    }
}
