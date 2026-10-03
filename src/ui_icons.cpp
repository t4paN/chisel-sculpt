#include "ui_icons.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cctype>

// Icon paths, verbatim from the handoff's icons-vector/svg/*.svg (24-unit grid,
// stroke 1.75, round caps). Each icon is a few parts; a part is an SVG path, a
// circle or a (rounded) rect. Brush Inflate is the one redraw: the handoff flagged
// its sun-with-rays as reading like "brightness" and colliding with the lighting
// slider's sun glyph, so it is a dome swelling up off the surface line Draw and
// Clay sit on, with an arrow out of it.

namespace {

enum class Kind { Path, FillPath, DashPath, Circle, Rect };

struct Part {
    Kind kind;
    const char* d;                       // Path / FillPath / DashPath
    float x, y, w, h, r;                 // Circle: x,y = centre, r. Rect: x,y,w,h, r = rx
};

constexpr Part P(const char* d)  { return {Kind::Path, d, 0, 0, 0, 0, 0}; }
constexpr Part F(const char* d)  { return {Kind::FillPath, d, 0, 0, 0, 0, 0}; }
constexpr Part D(const char* d)  { return {Kind::DashPath, d, 0, 0, 0, 0, 0}; }
constexpr Part C(float cx, float cy, float r) { return {Kind::Circle, nullptr, cx, cy, 0, 0, r}; }
constexpr Part R(float x, float y, float w, float h, float rx) {
    return {Kind::Rect, nullptr, x, y, w, h, rx};
}

struct IconDef { Part parts[4]; int n; };

const IconDef kIcons[(int)Icon::Count] = {
    /* BrushDraw    */ {{P("M3 17h4c2 0 2.5-7 5-7s3 7 5 7h4")}, 1},
    /* BrushClay    */ {{P("M3 18h3l2-4h8l2 4h3"), P("M9.5 14l1.5-3h2l1.5 3")}, 2},
    /* BrushInflate */ {{P("M3 18h3a6 6 0 0 1 12 0h3"), P("M12 3v6M9.5 5.5L12 3l2.5 2.5")}, 2},
    /* BrushCrease  */ {{P("M3 8h5l4 9 4-9h5")}, 1},
    /* BrushPinch   */ {{P("M12 4v16"), P("M3 12h5M6 9l3 3-3 3"), P("M21 12h-5M18 9l-3 3 3 3")}, 3},
    /* BrushMove    */ {{P("M12 3v18M3 12h18"),
                         P("M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3")}, 2},
    /* BrushLimb    */ {{P("M3 20h5c0-7 3-11 9-13"), P("M13.5 4.5L17 7l-2.5 3.5")}, 2},
    /* BrushSmooth  */ {{P("M3 12c1.5-3 3 3 4.5 0s3-3 4.5 0c1 1.6 2 0 3 0h6")}, 1},
    /* BrushMask    */ {{C(12, 12, 8), F("M12 4a8 8 0 0 0 0 16z")}, 2},
    /* ModeSculpt   */ {{P("M14 4l6 6-9 9H5v-6z"), P("M11 7l6 6")}, 2},
    /* ModeInsert   */ {{P("M12 3l8 4.5v9L12 21l-8-4.5v-9z"),
                         P("M12 12v9M12 12l8-4.5M12 12L4 7.5")}, 2},
    /* ModeSelect   */ {{P("M5 3l14 7-6 2-2 6z")}, 1},
    /* ModePaint    */ {{P("M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z")}, 1},
    /* ViewMirrorX  */ {{D("M12 3v18"), P("M9 7L4 17h5z"), P("M15 7l5 10h-5z")}, 3},
    /* ViewPaintShow*/ {{P("M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"), C(12, 12, 3)}, 2},
    /* ViewPaintHide*/ {{P("M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"), C(12, 12, 3),
                         P("M4 4l16 16")}, 3},
    /* UiShortcuts  */ {{R(2.5f, 6, 19, 12, 2), P("M6 10h1M9.5 10h1M13 10h1M16.5 10h1M7 14h10")}, 2},
    /* UiMenu       */ {{P("M4 6h16M4 12h16M4 18h16")}, 1},
    /* CmdUndo      */ {{P("M9 14L4 9l5-5"), P("M4 9h10.5a5.5 5.5 0 0 1 0 11H11")}, 2},
    /* CmdRedo      */ {{P("M15 14l5-5-5-5"), P("M20 9H9.5a5.5 5.5 0 0 0 0 11H13")}, 2},
    /* CmdLevelDown */ {{R(4, 4, 16, 16, 2), P("M12 4v16M4 12h16")}, 2},
    /* CmdLevelUp   */ {{R(4, 4, 16, 16, 2), P("M9.3 4v16M14.7 4v16M4 9.3h16M4 14.7h16")}, 2},
    /* CmdSave      */ {{P("M5 4h11l3 3v13H5z"), P("M8 4v5h7V4"), R(8, 13, 8, 7, 0)}, 3},
    /* CmdSaveCopy  */ {{P("M13 20H5V4h11l3 3v5"), P("M8 4v5h7V4"), P("M18 15v6M15 18h6")}, 3},
    /* CmdOpen      */ {{P("M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2.5h8.5A1.5 1.5 0 0 1 21 9v9.5"
                           "a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z")}, 1},
    /* CmdExport    */ {{P("M12 15V3M7 8l5-5 5 5"), P("M5 13v7h14v-7")}, 2},
    /* CmdMerge     */ {{P("M8.5 7a5 5 0 1 0 0 10c1.5 0 2.5-.8 3.5-1.6 1 .8 2 1.6 3.5 1.6"
                           "a5 5 0 1 0 0-10c-1.5 0-2.5.8-3.5 1.6C11 7.8 10 7 8.5 7z")}, 1},
};

// ---- Minimal SVG path interpreter -------------------------------------------
// Covers what the set uses: M L H V C S A Z, absolute and relative, implicit
// command repeats, and packed numbers ("2.5-7", ".8"). Arcs are circular (rx==ry,
// no rotation), which is all the set contains.

struct PathCursor {
    const char* s;
    bool more_numbers() {
        while (*s == ' ' || *s == ',') s++;
        return *s == '-' || *s == '+' || *s == '.' || std::isdigit((unsigned char)*s);
    }
    float num() {
        while (*s == ' ' || *s == ',') s++;
        char* end = nullptr;
        float v = std::strtof(s, &end);
        s = end;
        return v;
    }
};

struct Emitter {
    ImDrawList* dl;
    ImVec2 origin;
    float  k;          // px per grid unit
    ImU32  col;
    float  thick;
    Kind   kind;
    bool   open = false;

    ImVec2 map(float x, float y) const { return ImVec2(origin.x + x * k, origin.y + y * k); }

    void flush(bool closed) {
        if (!open) return;
        if (kind == Kind::FillPath) dl->PathFillConvex(col);
        else dl->PathStroke(col, closed ? ImDrawFlags_Closed : ImDrawFlags_None, thick);
        open = false;
    }
};

// SVG endpoint arc -> centre parameterisation (circular, unrotated).
void arc_to(Emitter& e, float x1, float y1, float r, bool large, bool sweep, float x2, float y2) {
    if (r <= 0.0f || (x1 == x2 && y1 == y2)) { e.dl->PathLineTo(e.map(x2, y2)); return; }
    float dx = (x1 - x2) * 0.5f, dy = (y1 - y2) * 0.5f;
    float d2 = dx * dx + dy * dy;
    if (d2 > r * r) r = std::sqrt(d2);                      // radius too small: scale up
    float num = r * r * r * r - r * r * d2;
    float coef = (num > 0.0f && d2 > 0.0f) ? std::sqrt(num / (r * r * d2)) : 0.0f;
    if (large == sweep) coef = -coef;
    float cxp = coef * dy, cyp = -coef * dx;
    float cx = cxp + (x1 + x2) * 0.5f, cy = cyp + (y1 + y2) * 0.5f;
    float a0 = std::atan2((dy - cyp) / r, (dx - cxp) / r);
    float a1 = std::atan2((-dy - cyp) / r, (-dx - cxp) / r);
    float da = a1 - a0;
    const float kTau = 6.28318530718f;
    if (!sweep && da > 0.0f) da -= kTau;
    else if (sweep && da < 0.0f) da += kTau;
    e.dl->PathArcTo(e.map(cx, cy), r * e.k, a0, a0 + da);
}

void run_path(Emitter& e, const char* d) {
    PathCursor pc{d};
    float x = 0, y = 0, sx = 0, sy = 0;        // current point, subpath start
    float lcx = 0, lcy = 0;                    // last cubic's 2nd control (for S)
    char cmd = 0, prev = 0;
    for (;;) {
        while (*pc.s == ' ' || *pc.s == ',') pc.s++;
        if (!*pc.s) break;
        if (std::isalpha((unsigned char)*pc.s)) cmd = *pc.s++;
        else if (!cmd) break;                  // malformed: numbers before any command
        bool rel = std::islower((unsigned char)cmd) != 0;
        float ox = rel ? x : 0.0f, oy = rel ? y : 0.0f;
        switch (std::toupper((unsigned char)cmd)) {
            case 'M': {
                e.flush(false);
                x = ox + pc.num(); y = oy + pc.num();
                sx = x; sy = y;
                e.dl->PathLineTo(e.map(x, y));
                e.open = true;
                cmd = rel ? 'l' : 'L';         // implicit repeats after M are linetos
                prev = 'M';
                continue;
            }
            case 'L': x = ox + pc.num(); y = oy + pc.num(); e.dl->PathLineTo(e.map(x, y)); break;
            case 'H': x = ox + pc.num();                     e.dl->PathLineTo(e.map(x, y)); break;
            case 'V': y = oy + pc.num();                     e.dl->PathLineTo(e.map(x, y)); break;
            case 'C': {
                float c1x = ox + pc.num(), c1y = oy + pc.num();
                float c2x = ox + pc.num(), c2y = oy + pc.num();
                float ex = ox + pc.num(), ey = oy + pc.num();
                e.dl->PathBezierCubicCurveTo(e.map(c1x, c1y), e.map(c2x, c2y), e.map(ex, ey));
                lcx = c2x; lcy = c2y; x = ex; y = ey;
                prev = 'C';
                if (!pc.more_numbers()) cmd = 0;
                continue;
            }
            case 'S': {
                float c1x = x, c1y = y;
                if (prev == 'C') { c1x = 2 * x - lcx; c1y = 2 * y - lcy; }
                float c2x = ox + pc.num(), c2y = oy + pc.num();
                float ex = ox + pc.num(), ey = oy + pc.num();
                e.dl->PathBezierCubicCurveTo(e.map(c1x, c1y), e.map(c2x, c2y), e.map(ex, ey));
                lcx = c2x; lcy = c2y; x = ex; y = ey;
                prev = 'C';                    // an S chain reflects off the S too
                if (!pc.more_numbers()) cmd = 0;
                continue;
            }
            case 'A': {
                float r = pc.num();
                pc.num();                      // ry (== rx throughout the set)
                pc.num();                      // x-axis rotation (always 0)
                bool large = pc.num() != 0.0f;
                bool sweep = pc.num() != 0.0f;
                float ex = ox + pc.num(), ey = oy + pc.num();
                arc_to(e, x, y, r, large, sweep, ex, ey);
                x = ex; y = ey;
                break;
            }
            case 'Z':
                x = sx; y = sy;
                e.flush(true);
                prev = 'Z';
                cmd = 0;
                continue;
            default:
                return;                        // unsupported command: stop, don't spin
        }
        prev = (char)std::toupper((unsigned char)cmd);
        if (!pc.more_numbers()) cmd = 0;
    }
    e.flush(false);
}

// Dashed vertical/horizontal segment (only the mirror icon's centre line uses it):
// "M x y v len", dash 2 / gap 2.6, as in the SVG's stroke-dasharray.
void run_dashed(Emitter& e, const char* d) {
    PathCursor pc{d + 1};                      // skip 'M'
    float x = pc.num(), y = pc.num();
    while (*pc.s == ' ') pc.s++;
    char c = *pc.s++;
    float len = pc.num();
    bool vert = (c == 'v' || c == 'V');
    for (float t = 0.0f; t < len; t += 4.6f) {
        float t1 = std::min(t + 2.0f, len);
        ImVec2 a = vert ? e.map(x, y + t)  : e.map(x + t, y);
        ImVec2 b = vert ? e.map(x, y + t1) : e.map(x + t1, y);
        e.dl->AddLine(a, b, e.col, e.thick);
    }
}

} // namespace

void draw_icon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 col, float stroke) {
    if ((int)icon < 0 || icon >= Icon::Count) return;
    const float k = size / 24.0f;
    Emitter e{dl, ImVec2(center.x - size * 0.5f, center.y - size * 0.5f), k, col,
              std::max(1.0f, stroke * k), Kind::Path};
    const IconDef& def = kIcons[(int)icon];
    for (int i = 0; i < def.n; i++) {
        const Part& p = def.parts[i];
        e.kind = p.kind;
        switch (p.kind) {
            case Kind::Path:
            case Kind::FillPath: run_path(e, p.d); break;
            case Kind::DashPath: run_dashed(e, p.d); break;
            case Kind::Circle:
                dl->AddCircle(e.map(p.x, p.y), p.r * k, col, 0, e.thick);
                break;
            case Kind::Rect:
                dl->AddRect(e.map(p.x, p.y), e.map(p.x + p.w, p.y + p.h), col, p.r * k,
                            ImDrawFlags_None, e.thick);
                break;
        }
    }
}
