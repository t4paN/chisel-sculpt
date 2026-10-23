#pragma once
#include "mesh.h"
#include "camera.h"
#include <cstdint>
#include <string>
#include <vector>

// Retopo mode (key 0): the user sketches fat strokes on the sculpt, Build turns each
// into a ribbon of quads lying on the surface. A closed stroke is a donut (concentric
// loops round an eye or a mouth), an open one a strip. Both are the same construction:
// spokes across the stroke, loops along it. The result is a separate low-poly mesh,
// CPU-side, that never enters the sculpt engine.
//
// Strokes are vector data in world space, not paint: they survive level switches and
// remeshes of the sculpt, and Build re-snaps them to whatever surface is there now.

struct RetopoStroke {
    std::vector<Vec3>  pts;      // centreline, on the surface (world)
    std::vector<Vec3>  nrm;      // surface normal at each point
    std::vector<float> half_w;   // half the band's width at each point (world)
    bool closed  = false;        // donut (true) or strip (false)
    // Its eye. Hidden: no band, no wireframe, and picking, Grab, Smooth and welds pass
    // it by; Build and Export still take it (it's out of sight, not out of the mesh).
    bool visible = true;
    // Its quads were grabbed, smoothed or welded: they're geometry now, not a sketch.
    // Build keeps them as they are (visible or not) and the band stops drawing.
    bool frozen  = false;
    bool selected = false;       // the bar's sliders and the right-click menu act on these
    // Its own build settings, taken from the bar's at pen-up; the sliders change them
    // only while it's selected.
    int   rings   = 4;           // quads across the band (1..8)
    float density = 1.0f;        // quads along it: 1 = about square, down to 0.1
    float width   = 1.0f;        // band width as drawn x this (0.1..2): rescues a stroke
                                 // whose path is right but whose brush was too fat
    uint32_t id  = 0;
    std::string name;
};

// The low-poly result. Quads are 4 indices each, wound so their normal faces out of
// the sculpt. Edges are kept unique for the wireframe overlay.
struct RetopoMesh {
    std::vector<Vec3>     v;
    std::vector<Vec3>     n;       // surface normal per vertex (overlay back-face test)
    std::vector<uint32_t> quads;
    std::vector<uint32_t> edges;   // 2 per edge
    std::vector<uint32_t> edge_owner;  // per edge: the RetopoStroke::id it came from (twins share it)
    std::vector<uint32_t> v_owner;     // per vertex, the same (a welded vertex keeps the survivor's)
    std::vector<uint32_t> quad_owner;  // per quad, the same
    // Per vertex, its mirror twin (built as its exact reflection), itself on the x = 0
    // seam, ~0u when it has none. Mirrored edits move pairs by this, never by position.
    std::vector<uint32_t> mirror;
    // For Smooth: edge neighbours (CSR), and for a border vertex its two border
    // neighbours (pinned = a corner, or not a simple border: it never moves).
    std::vector<uint32_t> nb_off, nb;
    std::vector<uint32_t> border_nb;   // 2 per vertex; ~0u for interior vertices
    std::vector<uint8_t>  pinned;
    void clear() {
        v.clear(); n.clear(); quads.clear(); edges.clear(); edge_owner.clear();
        v_owner.clear(); quad_owner.clear(); mirror.clear();
        nb_off.clear(); nb.clear(); border_nb.clear(); pinned.clear();
    }
    uint32_t quad_count() const { return (uint32_t)(quads.size() / 4); }
};

struct RetopoState {
    std::vector<RetopoStroke> strokes;
    std::vector<RetopoStroke> redo;   // strokes taken back by Ctrl+Z, for Ctrl+Shift+Z
    RetopoStroke live;                // the stroke under the pen
    bool drawing = false;
    // Cut: a bare line on the surface (no band), under the pen while `cutting`.
    std::vector<Vec3> cut_pts, cut_nrm;
    bool cutting = false;
    float last_sx = 0, last_sy = 0;   // screen position of the last captured point
    uint32_t next_id = 1;

    RetopoMesh result;
    bool built = false;
    bool show_result = true;
    bool show_strokes = true;
    uint32_t hover_id = 0;            // stroke under the cursor in the bar (0 = none), lit in the overlay

    // Pen-up: keep the stroke if it has enough to build on, decide donut vs strip.
    // Returns false (and drops it) for a tap.
    bool finish_live(int rings, float density);
    void undo();
    // Drop the quads of strokes that no longer exist (trash, undo), right away rather
    // than at the next Build. Welded neighbours keep the vertices they share.
    void prune();
    RetopoStroke* find(uint32_t id) {
        for (RetopoStroke& s : strokes) if (s.id == id) return &s;
        return nullptr;
    }
    // Owner id -> hidden by its eye, for the per-vertex checks (a table, not a search
    // per vertex). The dock refreshes it every frame; an id past its end is visible.
    std::vector<uint8_t> hidden_ids;
    bool hidden(uint32_t owner) const { return owner < hidden_ids.size() && hidden_ids[owner]; }
    void refresh_hidden() {
        hidden_ids.assign(next_id, 0);
        for (const RetopoStroke& s : strokes) if (!s.visible && s.id < next_id) hidden_ids[s.id] = 1;
    }
    void redo_one();
    void clear_all();
    // Selection. A click picks one island (0 = clear); Ctrl adds or takes it out.
    // Frozen and editable islands never share a selection: returns a message saying so
    // and leaves the selection alone, else nullptr.
    const char* click_select(uint32_t id, bool add);
    RetopoStroke* first_selected() {
        for (RetopoStroke& s : strokes) if (s.selected) return &s;
        return nullptr;
    }
};

// The island under screen point (x, y): a built quad (nearest the eye), else an unbuilt
// stroke's band, mirror twins included. Returns its stroke id, 0 for none.
uint32_t retopo_pick(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                     float x, float y, bool mirror);

// Rebuild state.result from every visible stroke against the sculpt `m` (positions
// must be current on the CPU: call Scene::materialize_active_cpu first). With
// `mirror`, a stroke lying wholly on one side of x=0 gets a twin: its exact reflection,
// paired vertex by vertex. Donuts build first, then strips in drawing order; a strip end
// on (or just short of) an island already built joins its outer edge.
// User-paced and CPU-only; returns the build time in ms.
double retopo_build(RetopoState& st, const Mesh& m, bool mirror);

// Editing the built quads. retopo_edit_begin at pen-down for Smooth and Grab alike:
// it re-reads the sculpt for snapping (must be current on the CPU).
// Both freeze every stroke whose quads they move (and every island joined to it: a
// joined group is one piece of geometry), and both weld: an outer-edge vertex
// that comes within ~0.4 of an edge of another outer-edge vertex (another island, or a
// part of its own island that isn't its immediate neighbourhood) snaps onto it and the
// two become one. Inner vertices never weld.
// Smooth (Shift / double Shift in mode 0): dab every frame
// with the cursor: vertices under the ring slide toward their neighbours' average
// along the surface (interior) or along the border (edges — so a strip keeps its
// width and a donut its hole), then snap back onto the sculpt. Corners stay put. With
// mirror, each moved vertex's twin is set to its reflection (welds mirror too).
// Returns how many vertices moved. A rebuild throws the smoothing away.
void retopo_edit_begin(RetopoState& st, const Mesh& m);
int  retopo_smooth_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float cx, float cy, float radius_px, float strength, bool mirror);

// Grab (G in mode 0): at the press, take every vertex under the ring with a soft
// falloff (mirror: their twins follow as reflections); while dragging, move each by the
// pointer's screen offset since the press, carried into the world at that vertex's
// depth, then snap onto the sculpt. Returns how many vertices it holds.
int  retopo_grab_begin(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float cx, float cy, float radius_px, bool mirror);
void retopo_grab_drag(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float dx_px, float dy_px);

// Write state.result as a quad OBJ. Returns false if there's nothing to write or the
// file can't be opened.
bool retopo_export_obj(const RetopoMesh& rm, const char* path);
