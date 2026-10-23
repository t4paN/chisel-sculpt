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

// The low-poly result. Faces have any number of corners: Build makes quads, Cut keeps
// to quads where it can, the eraser merges faces into bigger ones. Face f's corners are
// faces[face_off[f] .. face_off[f + 1]), wound so its normal faces out of the sculpt.
// Edges are kept unique for the wireframe overlay.
struct RetopoMesh {
    std::vector<Vec3>     v;
    std::vector<Vec3>     n;       // surface normal per vertex (overlay back-face test)
    std::vector<uint32_t> faces;
    std::vector<uint32_t> face_off{0};  // face count + 1
    std::vector<uint32_t> edges;   // 2 per edge
    std::vector<uint32_t> edge_owner;  // per edge: the RetopoStroke::id it came from (twins share it)
    std::vector<uint32_t> v_owner;     // per vertex, the same (a welded vertex keeps the survivor's)
    std::vector<uint32_t> face_owner;  // per face, the same
    // Per vertex, its mirror twin (built as its exact reflection), itself on the x = 0
    // seam, ~0u when it has none. Mirrored edits move pairs by this, never by position.
    std::vector<uint32_t> mirror;
    // For Smooth: edge neighbours (CSR), and for a border vertex its two border
    // neighbours (pinned = a corner, or not a simple border: it never moves).
    std::vector<uint32_t> nb_off, nb;
    std::vector<uint32_t> border_nb;   // 2 per vertex; ~0u for interior vertices
    std::vector<uint8_t>  pinned;
    // Every closed outline, walked round (rim r = rim[rim_off[r] .. rim_off[r + 1]), as
    // many sides as vertices), and whether it's a hole: any outline of a piece but its
    // longest, which is the piece's outer edge. Found with the rest of the topology.
    std::vector<uint32_t> rim_off{0}, rim;
    std::vector<uint8_t>  rim_hole;
    uint32_t rim_count() const { return (uint32_t)rim_hole.size(); }
    void clear() {
        v.clear(); n.clear(); faces.clear(); face_off.assign(1, 0); edges.clear(); edge_owner.clear();
        v_owner.clear(); face_owner.clear(); mirror.clear();
        nb_off.clear(); nb.clear(); border_nb.clear(); pinned.clear();
        rim_off.assign(1, 0); rim.clear(); rim_hole.clear();
    }
    void add_face(const uint32_t* p, int k, uint32_t owner) {
        faces.insert(faces.end(), p, p + k);
        face_off.push_back((uint32_t)faces.size());
        face_owner.push_back(owner);
    }
    uint32_t face_count() const { return (uint32_t)face_owner.size(); }
    uint32_t face_size(size_t f) const { return face_off[f + 1] - face_off[f]; }
    const uint32_t* face(size_t f) const { return faces.data() + face_off[f]; }
};

// One step of retopo history: everything an edit can change. Retopo is small (a few
// thousand vertices), so a whole copy per step costs ~100 KB and can't drift the way a
// diff can. Eyes and selection aren't history: undo leaves them as they are.
struct RetopoSnapshot {
    std::vector<RetopoStroke> strokes;
    RetopoMesh result;
    bool built = false;
    uint32_t next_id = 1;
    size_t bytes() const;
};

struct RetopoState {
    std::vector<RetopoStroke> strokes;
    RetopoStroke live;                // the stroke under the pen
    bool drawing = false;
    // Draw started on an edited island (its id): the live stroke is a bridge, not a band.
    uint32_t bridge_from = 0;
    // Cut: a bare line on the surface (no band), under the pen while `cutting`.
    std::vector<Vec3> cut_pts, cut_nrm;
    bool cutting = false;
    // Cut's vertex snap: the built vertex under the pointer (lit while hovering), and the
    // ones the line started and ended on (~0u = it started / ended on no vertex).
    uint32_t snap_v = ~0u, cut_from_v = ~0u, cut_to_v = ~0u;
    // Ctrl + Erase: the face that would go, outlined red while Ctrl is held (~0u = none).
    uint32_t face_hover = ~0u;
    // Edge loops picked by double-click (Draw): 2 vertex ids per edge, mirror twins
    // included. They index `result`, so whatever renumbers it (Build, undo, prune, any
    // checkpointed change) drops them; an edge that stops existing (a weld, Erase) just
    // stops drawing.
    std::vector<uint32_t> loop_edges;
    float last_sx = 0, last_sy = 0;   // screen position of the last captured point
    uint32_t next_id = 1;

    RetopoMesh result;
    bool built = false;
    bool show_result = true;
    bool show_strokes = true;
    uint32_t hover_id = 0;            // stroke under the cursor in the bar (0 = none), lit in the overlay
    // Where the last press of any retopo tool landed on the sculpt: F frames the piece
    // nearest it (the island being worked on, not its mirror twin).
    Vec3 work_pos;
    bool work_valid = false;

    // Pen-up: keep the stroke if it has enough to build on, decide donut vs strip.
    // Returns false (and drops it) for a tap.
    bool finish_live(int rings, float density);
    // History (Ctrl+Z / Ctrl+Shift+Z in mode 0, never the sculpt's), one stack for every
    // island: welds, bridges and cuts share vertices between islands, so undoing one
    // alone could tear its neighbour. checkpoint() before any change; `tag` != 0 folds
    // repeats of the same change into one step (a slider drag) until the tag is reset.
    std::vector<RetopoSnapshot> undo_stack, redo_stack;
    int last_tag = 0;
    void checkpoint(int tag = 0);
    // Grab and Smooth: the press takes the step, the release keeps it only if the
    // quads moved (a press on empty space isn't an undo step).
    RetopoSnapshot pending;
    bool has_pending = false;
    void begin_edit();
    void end_edit();
    bool undo();
    bool redo();
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
    void clear_all();
    // Selection. A click picks one island (0 = clear); Ctrl adds or takes it out.
    // Frozen and editable islands never share a selection: returns a message saying so
    // and leaves the selection alone, else nullptr.
    const char* click_select(uint32_t id, bool add);
    // Merge: the selected edited islands become one, the oldest (first drawn), which
    // keeps its name, row and selection; the others' rows go and their faces, edges and
    // vertices are its. Shared or not, nothing moves. Needs two or more; returns how
    // many went into it (0 = nothing done).
    int merge_selected();
    int merge_candidates() const {
        int n = 0;
        for (const RetopoStroke& s : strokes) n += s.selected && s.frozen;
        return n;
    }
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

// Cut (C in mode 0): the knife, along st.cut_pts (on the sculpt, as drawn). Every built
// edge the line crosses, on the side facing the eye, gets a new vertex, pulled most of
// the way to the edge's middle (the line is drawn by hand, the loop it makes is wanted
// straight), and every face around a cut edge splits into quads: across = two, a corner
// clipped = three round a new turning vertex, crossed 4 times = four round a middle. A
// line that stops inside the island runs on through edge middles to its border (or round
// to meet itself): a dead end inside quads always costs a triangle, a support edge run
// through doesn't. No vertex is left hanging on an edge. With mirror, each cut edge's twin is cut too,
// as its exact reflection. A line that starts (st.cut_from_v) or ends (st.cut_to_v) on a
// vertex cuts from that vertex itself: the face it leaves splits along a chord from the
// corner, triangles and all, and the supports leave that face alone. Start and end on
// two corners of one face and that face splits along the chord between them.
// Freezes what it cuts. Returns how many faces it split.
int retopo_cut(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
               bool mirror);

// The built vertex within a few pixels of screen point (x, y), facing the eye and not
// hidden, nearest first; ~0u for none. Close on purpose: a vertex cut is deliberate.
uint32_t retopo_snap_vertex(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                            float x, float y);

// Double-click in Draw: the edge loop through the built edge nearest screen point
// (x, y) into st.loop_edges. An inner edge runs straight on through every vertex with
// four quads round it and stops at anything else (a pole, a triangle, the outline); an
// outline edge follows the outline all the way round (a hole's rim, a strip's edge).
// `add` (Ctrl) keeps the loops already picked, or takes this one out if it's one of
// them. With mirror, the twin loop comes along. Returns its edges, 0 = no edge there.
int retopo_select_loop(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool add, bool mirror);

// Erase (E in mode 0): a round brush. Each dab takes the inner edge whose middle is
// nearest the ring's centre (inside the ring; the island's outline stays): its two
// faces become one, of any size. An end left as a point on a straight line dissolves,
// so a whole support line erased gives its quads back. With mirror, the twin edge goes
// too. Freezes what it touches. Returns edges removed.
int retopo_erase_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                     float cx, float cy, float radius_px, bool mirror);

// Ctrl + Erase: whole faces. The face under the ring's centre (nearest the eye), else
// the one whose middle is nearest it inside the ring; ~0u for none. The dab deletes it
// (and, with mirror, its twin), outline faces too; vertices only it used go with it.
uint32_t retopo_face_at(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                        float cx, float cy, float radius_px);
int retopo_erase_face_dab(RetopoState& st, const Camera& cam, int win_w, int win_h,
                          float cx, float cy, float radius_px, bool mirror);

// Bridge: a Draw stroke that started on an edited island (st.bridge_from, the line in
// st.live). One strip, one quad wide, from the edge the line first crosses out of that
// island to the edge it last crosses into `to_id` (another edited island, or the same
// one), its ends the two edges' own vertices, its width eased from one edge's length to
// the other's, its steps the size of the quads it joins. to_id 0 = a loose strip grown
// out of the edge to the line's end. The islands and the bridge become one island (the
// oldest keeps its name). Mirror: a twin bridge, or, from an island to its own twin, a
// bridge that is its own twin. Returns nullptr when built, else why not (nothing done).
const char* retopo_bridge(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                          uint32_t to_id, bool mirror);

// Fill (H in mode 0, the dock's Fill): quads over holes. With edge loops picked
// (double-click), every picked loop that is a whole outline, hole or outer edge; else
// every hole. A rim of N = 2(m + n) sides becomes an m x n grid: four corners where the
// rim turns most (sides as near square as they allow), the inside spread from the four
// sides and snapped onto the sculpt `m` (current on the CPU). An odd rim can't be all
// quads: left alone, counted in `odd`. Mirror: a hole's twin gets the reflected fill, a
// hole across x = 0 a symmetric one. The islands round a hole become one island and
// freeze. One undo step. Returns the faces made; `rims` = how many rims they filled.
int retopo_fill(RetopoState& st, const Mesh& m, bool mirror, int& rims, int& odd);

// Write state.result as an OBJ, every face as it is (quads, triangles, bigger ones).
// Returns false if there's nothing to write or the file can't be opened.
bool retopo_export_obj(const RetopoMesh& rm, const char* path);
