# Chisel Manual

The complete guide to Chisel: every mode, brush, key, and file format, plus how the
web and native builds differ. If you just want the pitch and the quickstart, read the
[README](README.md) instead.

Chisel is deliberately spartan. There are no nested menus, no settings screens, no
modal panels between you and the mesh. Everything is a keystroke or a hold-and-drag,
and everything runs on the GPU — brushes, normals, selection, masking, remeshing, and
merging execute as compute shaders, so the app stays fast at high polygon counts. The
design goal is simple: don't make the artist wait, and don't make them dig.

---

## Getting started

**In the browser** — play the live build on [itch.io](https://tpn.itch.io/chisel-sculpt).
No install. You need a WebGPU-capable browser: recent Chrome or Edge (113+) works out
of the box; Firefox and Safari currently need their WebGPU flag enabled.

**Native (Linux, Windows)** — grab the AppImage / zip from the releases page, or build
from source (see the README). The native build sculpts identically and reads and
writes the same files.

You start with a sphere, in Edit mode, with the Draw brush active. Left-drag on the
mesh to sculpt; left-drag off the mesh to orbit. That's 90% of the app right there.

## The interface

- **Top toolbar** — Subdiv −/+, Undo, Redo, Export, Save, Load, Merge.
- **Mode row** — Edit, Insert, Select, Paint, plus the `[ ]` paint-visibility toggle.
- **Brush column** (Edit mode) — Draw, Inflate, Crease, Pinch, Move, Limb, Smooth, Mask.
- **HUD panel** (bottom left) — active brush and its size / strength / hardness /
  spacing, triangle and vertex counts, subdivision level, mirror axis, normals mode,
  autosmooth state.
- **Notifications** appear briefly at the top of the canvas (selection counts, load
  results, merge status, warnings).

## Modes

| Key | Mode | What it's for |
|---|---|---|
| 1 | **Edit** | Sculpting the active mesh with brushes |
| 2 | **Insert** | Dropping new sphere primitives into the scene |
| 3 | **Select** | Picking, moving, scaling, deleting whole objects |
| 4 | **Paint** | Vertex-colour painting |

## Sculpting (Edit mode)

Left-drag on the mesh strokes with the active brush. Left-drag off the mesh orbits
the camera, so you never have to switch tools just to look around.

### The brushes

| Key | Brush | Behaviour |
|---|---|---|
| D | **Draw** | The bread-and-butter brush: pushes the surface out along its normal. |
| I | **Inflate** | Swells vertices along their own local normals — great for fattening limbs and rounding forms. |
| C | **Crease** | Pinches while cutting in: sharp valleys and fold lines. |
| V | **Pinch** | Pulls vertices toward the stroke centre — tightens edges and detail. |
| G | **Move** | Grab-and-drag: picks up the surface under the cursor and moves it with you. |
| H | **Limb** | Snakehook: pulls a tube out of the surface, redistributing and densifying geometry at the tip. Grow arms, tails, horns from nothing. |
| Shift | **Smooth** | Hold Shift with any brush to smooth instead. In Paint mode it blends colours instead of geometry. |
| M | **Mask** | Paints a protection mask (see Masking below). |

### Stroke modifiers

| Key | Effect |
|---|---|
| Ctrl + LMB | Subtract — the brush pushes in instead of out |
| Shift + LMB | Smooth (any brush) |
| Double-tap Ctrl | Lock subtract mode (tap again to unlock) |
| Double-tap Shift | Lock smooth mode (tap again to unlock) |
| Q / E | Cycle brush backward / forward |
| B | Toggle autosmooth (a gentle smoothing pass folded into every stroke) |

### Brush tuning — hold a key and drag left/right

No sliders to find, no panels to open: hold the key, drag, watch the cursor ring.

| Hold + drag | Adjusts |
|---|---|
| S | Brush size |
| W | Brush strength |
| A | Brush hardness (falloff) |
| O | Brush spacing (dab distance along the stroke) |

On the web build these drags use pointer lock — your browser will show its standard
"cursor hidden" notice; that's expected.

### Pen pressure

Plug in a tablet and pressure drives both strength and size automatically. Toggle it
with **K**. Sources: XInput2 on Linux (same as Krita), WinTab on Windows, the Pointer
Events pressure channel in the browser. Chisel is designed to work fine with a mouse
— a tablet is nice, never necessary.

## Symmetry

**X** toggles X-axis mirror. Strokes, masks, and paint apply symmetrically in real
time. The HUD shows the active mirror axis. Meshes inserted while mirroring is on can
be committed as a symmetric pair (see Insert mode). Voxel merge can produce a
mirror-symmetric result that stays cleanly editable under mirror afterwards.

## Subdivision levels (multires)

Chisel keeps a displacement stack: a base cage plus per-level detail layers.

| Key | Action |
|---|---|
| Ctrl + D | Subdivision level up (adds detail resolution) |
| Shift + D | Subdivision level down (edit broad forms; detail is preserved) |
| P | Project displacement (bake current-level changes down the stack) |

Step down to move big shapes, step back up and your fine detail rides along on the
adjusted forms. Each object in the scene has its own independent level and stack.

## Remeshing

**/** starts an isotropic remesh of the active mesh (confirm with **Y**). It
rebuilds triangles toward uniform edge length, respects the mask (masked areas keep
their topology) and mirror symmetry, and carries vertex paint across. Use it when
sculpting has stretched triangles thin — the Limb brush and heavy Move strokes are
the usual suspects.

## Multiple objects

### Insert mode (2)

Click to place a sphere, drag to scale it, release to commit. If the placement
straddles the mirror plane it snap-commits as one centred mesh; otherwise you're
prompted: **Y** commits a symmetric pair, **N** keeps it single. Each object gets its
own undo history and subdivision stack.

### Select mode (3)

| Input | Action |
|---|---|
| LMB | Select an object (any object, not just the active one) |
| Ctrl + LMB | Add / remove from the selection — including the active mesh; peeling out the last one deselects the whole scene |
| LMB + drag | Move the selected object in the view plane |
| RMB + drag | Scale the selected object (Ctrl+RMB stays camera zoom) |
| Delete | Delete the selected object |

The selection also drives the voxel merge: selected meshes are the merge set,
deselected (red-tinted) meshes are bystanders — or carving tools (see Subtract).

## Voxel merge (join for print)

**J** opens the merge prompt for the current selection. It voxelizes the selected
meshes into an SDF grid and extracts one watertight, manifold result — ready for 3D
printing.

| Key (in the prompt) | Action |
|---|---|
| Y | Faithful union of the selection |
| M | Mirror-symmetric union — the result stays editable under X-mirror |
| − | **Subtract**: union the selection, then carve away every *unselected* (red) mesh — booleans with your bystander objects as cutters |
| S | Toggle the extractor: Marching Cubes ↔ Surface Nets (Surface Nets gives a smoother, more even result) |
| [ / ] | Lower / raise voxel resolution |
| N / Esc | Cancel |

The merge runs across frames — the app stays responsive during big grids. Output is
tangentially relaxed onto the iso-surface so it comes out even rather than stair-stepped.

## Masking

The Mask brush (M) paints protection: masked vertices ignore sculpting *and* paint.

| Key | Action |
|---|---|
| Ctrl + I | Invert mask (undoable) |
| Ctrl + A | Clear mask |

Masks also steer remeshing (masked topology is preserved) and survive save/load.

## Painting (Paint mode, 4)

Vertex-colour painting with the same brush engine — size, strength, hardness,
spacing, and pressure all apply.

| Input | Action |
|---|---|
| LMB | Paint with the active colour |
| Shift + LMB | Blend/smooth colours |
| RMB | Pick the colour under the cursor |
| Q / E | Swap active / alternate colour |
| `[ ]` toggle (toolbar) | Hide vertex colour while sculpting (Paint mode always shows it) |

Paint lives on the base cage of the multires stack, so it survives level switches,
subdivision, remeshing, and voxel merges. PLY export carries it out of the app.

## Camera

The camera is orthographic — zoom is magnification, so you can zoom as deep as you
like and never clip inside the model.

| Input | Action |
|---|---|
| LMB off mesh / Alt + LMB | Orbit |
| MMB | Pan |
| Scroll | Zoom |
| F | Reframe to the active mesh |
| F1 / F2 / F3 | Snap to front / side / top |
| Space | Toggle fullscreen |

## Undo / redo

**Ctrl+Z / Ctrl+Shift+Z**. Undo is delta-based and per-stroke,
each object has its own history, and on capable GPUs the ring lives GPU-resident —
undoing a stroke never stalls the app. Mask inversion, level switches, and merges are
all undoable.

## Files

### Projects — `.chisel`

**Ctrl+S** saves, **Ctrl+Shift+S** saves under a new name, **Ctrl+O** / the Load
button opens. A project stores every object with its full multires stack, masks,
paint, selection, camera, and mirror settings.

**Cross-platform:** current files (format v4) open identically on native and web —
sculpt on the itch build at lunch, open the same file at home on native. Files saved
by pre-v4 builds migrate automatically when opened on the platform that saved them
(just re-save to upgrade them to v4). Opened on the *other* platform, a pre-v4 file
loads with correct geometry but its level stack flattened — you'll get a notification;
re-save it on its origin platform first if you want the levels.

### Import / export

| Direction | Formats | Notes |
|---|---|---|
| Import (Ctrl+O) | OBJ, PLY, .chisel | One picker for all three |
| Export (Ctrl+E / toolbar) | OBJ, STL, PLY | PLY carries vertex paint; STL is print-ready — pair it with voxel merge |

### On the web

Saves and exports arrive as browser downloads (with a name prompt); where the browser
allows it outside sandboxed iframes, a real save dialog is used. Imports go through
the standard file picker. Tip: enable your browser's "ask where to save each file" if
you want to choose folders on itch.

## Web build notes

- Browser shortcuts that collide with Chisel's (Ctrl+S, Ctrl+D, Ctrl+O, F1, …) are
  captured while the app has focus; devtools shortcuts are left alone.
- `~` toggles a debug console overlay; add `?debug=1` to the URL for on-page logs.
- Performance: hundreds of thousands of triangles flow well on modest hardware; the
  browser tab's memory is usually the ceiling, not the GPU.

## Other keys

| Key | Action |
|---|---|
| N | Toggle fast normals |
| Y | Toggle debug mesh view (outside prompts) |
| Esc | Quit (native only) / cancel prompts |

## Known limitations

- No texture painting — paint is per-vertex.
- Brush feel is still being tuned (falloff curves, stroke interpolation).
- Voxel-merging heavily damaged, non-watertight input can need a resolution bump.
- After a *full* undo of many overlapping carve strokes, normals can look slightly
  off where they overlapped — cosmetic; one smooth pass fixes it.
- Browser pen-pressure fidelity depends on the browser's Pointer Events support.
