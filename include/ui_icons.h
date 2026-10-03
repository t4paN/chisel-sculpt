#pragma once
#include "imgui.h"

// The modern skin's icon set (the 24 px stroke icons from the UI handoff's
// icons-vector/), drawn as vector paths straight into an ImDrawList — no atlas
// texture, so it is identical on the GL, native WebGPU and web builds and stays
// crisp at any size. Path data is copied verbatim from the SVGs; see ui_icons.cpp.
enum class Icon {
    BrushDraw, BrushClay, BrushInflate, BrushCrease, BrushPinch, BrushMove,
    BrushLimb, BrushSmooth, BrushMask,
    ModeSculpt, ModeInsert, ModeSelect, ModePaint,
    ViewMirrorX, ViewPaintShow, ViewPaintHide,
    UiShortcuts, UiMenu,
    CmdUndo, CmdRedo, CmdLevelDown, CmdLevelUp, CmdSave, CmdSaveCopy, CmdOpen,
    CmdExport, CmdMerge,
    Count
};

// Draw `icon` centred on `center`, `size` px square (the 24-unit grid scaled to it).
// `stroke` is in grid units (the set is drawn at 1.75).
void draw_icon(ImDrawList* dl, Icon icon, ImVec2 center, float size, ImU32 col,
               float stroke = 1.75f);

// The DOS skin's 16x16 one-bit version of the same icon, `scale` whole screen
// pixels per icon pixel (2 = 32 px), snapped to the pixel grid.
void draw_pixel_icon(ImDrawList* dl, Icon icon, ImVec2 center, float scale, ImU32 col);
