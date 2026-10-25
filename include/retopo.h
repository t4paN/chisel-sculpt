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
    int   rings   = 2;           // quads across the band (1..8)
    float density = 1.0f;        // quads along it: 1 = about square, 0.1 to 2
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
    // Draw started on a picked outline loop: the live stroke extrudes it (a bare line).
    bool extruding = false;
    // Draw started on a hole's rim vertex (its id, ~0u = not): if the line runs through
    // the hole to another vertex of that rim, it's a spine (retopo_spine), not a bridge.
    uint32_t spine_from = ~0u;
    // Cut: a bare line on the surface (no band), under the pen while `cutting`.
    std::vector<Vec3> cut_pts, cut_nrm;
    bool cutting = false;
    // Cut's vertex snap: the built vertex under the pointer (lit while hovering), and the
    // ones the line started and ended on (~0u = it started / ended on no vertex).
    uint32_t snap_v = ~0u, cut_from_v = ~0u, cut_to_v = ~0u;
    // Draw's snap (retopo_draw_snap): the vertex a line would start or end on (orange
    // dot), else the edge (bold blue, draw_t = how far along it from draw_ea).
    uint32_t draw_v = ~0u, draw_ea = ~0u, draw_eb = ~0u;
    float draw_t = 0.0f;
    // The vertex the live line started on (its press was on an orange dot), ~0u = none.
    uint32_t line_from_v = ~0u;
    // Ctrl + Erase: the face that would go, outlined red while Ctrl is held (~0u = none).
    uint32_t face_hover = ~0u;
    // Edge loops picked by double-click (Draw): 2 vertex ids per edge, mirror twins
    // included. They index `result`, so whatever renumbers it (Build, undo, prune, any
    // checkpointed change) drops them; an edge that stops existing (a weld, Erase) just
    // stops drawing.
    std::vector<uint32_t> loop_edges;
    // The edge a Shift+Ctrl+click path starts from: the last one clicked to pick. Only
    // trusted while it's still in loop_edges (whatever renumbers clears those).
    uint32_t path_a = ~0u, path_b = ~0u;
    float last_sx = 0, last_sy = 0;   // screen position of the last captured point
    uint32_t next_id = 1;
    // Bumped by every change (each undo step taken, undo, redo, a load): main compares it
    // with its value at the last save to know there's unsaved retopo work.
    uint64_t rev = 0;

    RetopoMesh result;
    bool built = false;
    // Wand: a hole clicked, its fills worked out whole at the click (see retopo_wand_open).
    // While open, `result` shows wand_base or one option swapped in; nothing is history
    // until one is kept.
    struct WandOption {
        char label[32];
        char counts[48];
        RetopoSnapshot after;
    };
    std::vector<WandOption> wand_opts;
    RetopoSnapshot wand_base;
    bool wand_open = false;
    int  wand_shown = -1;           // the option swapped in, -1 = the hole as it was
    bool wand_keep_shown = false;   // opened by a spine: closing it keeps what's shown
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
// `px` > 0 widens it (Draw's lines use kRetopoDrawSnapPx).
uint32_t retopo_snap_vertex(const RetopoState& st, const Camera& cam, int win_w, int win_h,
                            float x, float y, float px = 0.0f);

// Draw's snap reach. Wider than Cut's: lines across gaps start and end on vertices all
// the time, and hitting one by hand at 8 px took the user several tries.
constexpr float kRetopoDrawSnapPx = 14.0f;

// What a Draw line starting or ending at (x, y) lands on, into st.draw_v (a vertex,
// preferred) or st.draw_ea/eb/t (an edge). `any` = everything in reach (the end of a
// line already drawing); else only what a press there would draw a line from: an edited
// island's vertices and edges, any hole's rim vertices. retopo_draw_snap_point: where
// that is on the surface; false = nothing lit.
void retopo_draw_snap(RetopoState& st, const Camera& cam, int win_w, int win_h,
                      float x, float y, bool any);
bool retopo_draw_snap_point(const RetopoState& st, Vec3& p, Vec3& n);

// Double-click in Draw: the edge loop through the built edge nearest screen point
// (x, y) into st.loop_edges. An inner edge runs straight on through every vertex with
// four quads round it and stops at anything else (a pole, a triangle, the outline); an
// outline edge follows the outline all the way round (a hole's rim, a strip's edge).
// `add` (Ctrl) keeps the loops already picked, or takes this one out if it's one of
// them. With mirror, the twin loop comes along. Returns its edges, 0 = no edge there.
int retopo_select_loop(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool add, bool mirror);

// Shift+Ctrl+click in Draw: the shortest run of edges from the last edge picked to the
// edge under (x, y) joins st.loop_edges, and that edge is the next run's start. Both on
// one outline: the run stays on it, the short way round, so Extrude can take it. With
// nothing picked yet it picks just that edge. Returns the edges added (0 = none), -1 =
// nothing joins the two (separate islands).
int retopo_select_path(RetopoState& st, const Camera& cam, int win_w, int win_h,
                       float x, float y, bool mirror);

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

// Del with edges picked: each picked edge goes and the faces either side become one;
// then the picked edges' vertices left with only two edges go too, joining those two
// (a dissolved row: the edges across it run straight through). Mirror takes the twins.
// Outline edges are refused. One undo step; freezes what it touches. Returns nullptr
// when done (`dissolved` = edges taken), else why not.
const char* retopo_dissolve(RetopoState& st, bool mirror, int& dissolved);
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
// Pen-up on a stroke that started on an edited island: true when most of the line ran
// along an island's outer edge (within a band's width, near parallel) rather than out
// of it. The live line is then slid off the island, its band's near edge onto the
// island's, to be kept as a plain strip that Build zips on (join_side). False = a bridge.
bool retopo_alongside(RetopoState& st);

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

// Extrude: a Draw stroke that started on a picked loop (st.extruding, the line in
// st.live). The picked loop nearest the line's start, an outline (open or closed), grows
// new loops outward across the surface, away from the faces it borders, as far as the line
// went out: one loop for a short line, more for a long one so the quads stay about square,
// every loop with the same count of vertices. The outermost new loop becomes the pick, so
// the next stroke carries on from it. Mirror: a loop across x = 0 grows symmetric, a loop
// with a twin grows a twin. Freezes its island. Returns nullptr when done, else why not.
// The press at screen point (x, y) lands on a picked loop (within half an edge's screen
// length of one of its edges, a few pixels at least, facing the eye): Draw extrudes it
// rather than drawing.
bool retopo_on_loop(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y);
const char* retopo_extrude(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                           bool mirror);

// Wand (Q in mode 0): the hole under screen point (x, y) (the nearest one whose rim,
// as seen, contains the point), and every way it can be filled, each applied to a copy
// and kept whole in st.wand_opts, so hovering one is a swap, not a rebuild. Even rims
// get Fill's grid; with mirror, a hole across x = 0 also gets ladders, which add no
// vertices, only edges between the rim's own:
//   Columns: the rim's two seam points joined, then pairs outward along both halves
//            (top i with bottom i) until they meet at the sides: a triangle there when
//            a half's count is even, one in the middle when the rim's is odd.
//   Shifted: the same with one half started a step on, which moves the triangles from
//            the sides to the middle or away entirely.
//   Rows:    each rim vertex to its twin, straight across the seam.
// `m` must be current on the CPU (the grid snaps onto it). Returns nullptr when the menu
// has options, else why not (nothing changes).
const char* retopo_wand_open(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                             float x, float y, bool mirror);
// Show option k (-1 = the hole as it was) while the menu is open.
void retopo_wand_show(RetopoState& st, int k);
// Close the menu: keep option k as one undo step, or k = -1 to leave the hole open.
void retopo_wand_close(RetopoState& st, int keep);

// Spine: a Draw line from a hole's rim vertex (st.spine_from) across the hole to another
// vertex of the same rim. The line becomes a run of edges (its vertices on the line, as
// many as the shorter side of the rim has edges, less two), and each of its vertices gets
// a rung out to the rim on both sides, so the quads grow from the line toward the ring.
// The faces at its two ends each take two rim edges, so they're quads too; a side with
// more edges than the other folds the extra into a triangle against the rim. Mirror: a
// spine from seam vertex to seam vertex of a hole across x = 0 lies on the seam and is
// its own twin; a hole with a twin elsewhere gets the reflection. One undo step.
// retopo_hole_vertex: the hole rim vertex within a few pixels of (x, y), ~0u for none.
// retopo_spine_gesture: whether the live line (released at ex, ey) ran through that
// hole and ended on its rim; false = it's an ordinary stroke (nothing judged yet).
// retopo_wand_open_spine: the wand's menu over that hole with the spine's fills first
// ("Your line", and the same with triangles at its ends), the first already shown and
// kept unless another is picked (closing the menu keeps it, Esc takes it back). `m` as
// for retopo_wand_open. nullptr when open, else why not (nothing changed).
uint32_t retopo_hole_vertex(const RetopoState& st, const Camera& cam, int win_w, int win_h, float x, float y);
bool retopo_spine_gesture(const RetopoState& st, const Camera& cam, int win_w, int win_h, float ex, float ey);
const char* retopo_wand_open_spine(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                                   float ex, float ey, bool mirror);

// Notch: a Draw line from one outline vertex (st.line_from_v, the press's orange dot) to
// another of the same outline (st.draw_v at pen-up), across open space. The notch is the
// short way round the outline between them, plus its legs: the outline carried on past
// both ends, in step, while it keeps heading out of the notch (a Pi's walls below the
// line). A flat mouth joins the legs' feet with new vertices, as many as close it in
// quads, and the line runs through it as a spine. No legs (the line drawn across the
// mouth itself): the line is the mouth. Mirror: across the seam its halves are twins;
// off it, the twin notch gets the same faces. retopo_notch_gesture: whether the live line
// makes one (logs why not); retopo_wand_open_notch: the wand's menu over it, the first
// fill shown and kept, as for a spine. `m` as for retopo_wand_open.
bool retopo_notch_gesture(const RetopoState& st, const Camera& cam, int win_w, int win_h, bool mirror);
const char* retopo_wand_open_notch(RetopoState& st, const Mesh& m, const Camera& cam, int win_w, int win_h,
                                   bool mirror);

// A saved project's retopo (strokes + result as saved) becomes the session's: history,
// picks and tools start fresh, edges / outlines / holes are worked out again from the
// faces. The view toggles (Strokes, Wire) are kept. An empty snapshot clears retopo.
void retopo_load(RetopoState& st, RetopoSnapshot&& s);

// Write state.result as an OBJ, every face as it is (quads, triangles, bigger ones).
// Returns false if there's nothing to write or the file can't be opened.
bool retopo_export_obj(const RetopoMesh& rm, const char* path);
