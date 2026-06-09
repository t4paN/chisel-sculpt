# Chisel

Shader-based 3D sculpting. C++ / OpenGL 4.3.

Chisel runs the entire sculpt pipeline on the GPU — brush operations, mesh updates, selection, and remeshing all execute as compute shaders. The CPU orchestrates; the GPU does the work. The result is a lightweight, responsive sculpting tool that stays out of your way.

## Features

- **Brushes** — draw, smooth, crease, pinch, move, mask
- **Multiple objects** — insert, select, move, and delete independent meshes in one scene, each with its own undo history and subdivision level
- **Insert primitives** — drop a sphere into the scene, drag to scale, optionally mirror it symmetric on insert
- **Subdivision levels** — step up for detail, step down to edit forms, carry displacement between levels
- **Isotropic remesh** — respects masks and mirror symmetry
- **X-axis mirror** — real-time symmetric sculpting
- **Voxel merge** — weld a selection of meshes into one watertight manifold, ready for 3D printing; optionally mirror-symmetric so the result stays editable under X-mirror
- **Undo / redo** — delta-based, per-stroke, per-object
- **Import / export** — OBJ in and out; STL and PLY export (PLY carries vertex paint)
- **Save / load** — native project files (.chisel)
- **Matcap shading** — one draw call per object
- **Runs on a potato** — as long as the potato supports OpenGL 4.3 (compute shaders are non-negotiable)

## Known limitations / TODO

- Pen pressure is X11/XInput2 only (Wacom et al.); no Wayland tablet support yet
- Voxel merge output is now evened out (tangential relaxation onto the iso-surface) and can be made mirror-symmetric; heavily sculpted, non-watertight input can still need a resolution bump
- Brush feel could use more polish (falloff curves, stroke interpolation)
- No texture painting
- Linux-only dev, Windows builds via CI (not battle-tested)

## No tablet required

Chisel is designed to work well with a mouse. Hold a key and drag to adjust brush size, strength, hardness, or spacing in real time — no menus, no panels, no digging through settings. A tablet is nice to have, but not necessary.

If you do plug in a graphics tablet on X11, pen pressure is picked up automatically (via XInput2 — the same source Krita uses, including any pressure curve you've set there) and drives both brush strength and size. Toggle it with **K**.

## Controls

### Modes

| Key | Mode |
|---|---|
| 1 | Edit (sculpt) |
| 2 | Insert (spawn a sphere) |
| 3 | Select (pick / move / delete objects) |
| 4 | Paint (vertex colour) |

### Sculpting

| Key | Action |
|---|---|
| LMB on mesh | Active brush stroke |
| Ctrl + LMB | Subtract mode |
| Shift + LMB | Smooth (blends colours instead of geometry in paint mode) |
| Double-tap Shift | Lock smooth mode (tap again to unlock) |
| Double-tap Ctrl | Lock subtract mode |
| D | Draw brush |
| I | Inflate brush (swell along local normals) |
| C | Crease brush |
| V | Pinch brush |
| G | Move brush |
| H | Limb brush (snakehook: pull a tube that redistributes + densifies its tip) |
| M | Mask brush |
| 4 | Paint brush (vertex colour) |
| RMB (Paint) | Colour picker at cursor |
| Q / E | Cycle brush backward / forward (Paint: swap active / alternate colour) |
| B | Toggle autosmooth |

Mask shields verts from paint as well as sculpt. The `[ ]` toggle by the Paint icon hides vertex colour while sculpting (the paint brush always shows it).

### Brush sliders (hold + drag left/right)

| Key | Slider |
|---|---|
| S + drag | Brush size |
| W + drag | Brush strength |
| A + drag | Brush hardness |
| O + drag | Brush spacing |

### Objects (Insert / Select modes)

| Key | Action |
|---|---|
| LMB (Insert) | Place a sphere, drag to scale, release to commit |
| Y / N (Insert) | Symmetrize the placed mesh, or keep it single |
| LMB (Select) | Select an object |
| Ctrl + LMB (Select) | Add / remove from selection |
| LMB + drag (Select) | Move the selected object in the view plane |
| Delete | Delete the selected object |

### Mesh operations

| Key | Action |
|---|---|
| Ctrl + D | Subdivision level up |
| Shift + D | Subdivision level down |
| / | Remesh (confirm with Y) |
| J | Voxel merge selection — Y faithful merge, M mirror-symmetric merge, N / Esc cancel; [ / ] adjust resolution |
| P | Project displacement |
| Ctrl + I | Invert mask |
| Ctrl + A | Clear mask |
| X | Toggle X-axis mirror |

### Camera

| Key | Action |
|---|---|
| Alt + LMB / LMB off mesh | Orbit |
| MMB | Pan |
| Scroll | Zoom |
| F | Reframe to mesh |
| F1 / F2 / F3 | Snap to front / side / top |

### File

| Key | Action |
|---|---|
| Ctrl + S | Save |
| Ctrl + Shift + S | Save as |
| Ctrl + O | Import OBJ |
| Ctrl + E | Export OBJ |

### Other

| Key | Action |
|---|---|
| Ctrl + Z | Undo |
| Ctrl + Shift + Z | Redo |
| Space | Toggle borderless fullscreen |
| N | Toggle fast normals |
| K | Toggle pen pressure (tablet) |
| Y | Toggle debug mesh view |
| Esc | Quit (confirm with Y) |

## How it works

Chisel keeps the mesh GPU-resident. Brush strokes, normal recomputation, selection growth, mask painting, and remesh smoothing all run as compute shaders — the CPU never reads vertex data back during a stroke. An FBO snapshot at pen-down gives the CPU a screen-space depth + normal + triangle ID buffer to pick vertices against, so there's zero GPU sync until the stroke ends.

The mesh uses a flat SOA layout with CSR adjacency — no half-edge structures, no linked lists. Normals are recomputed partially (dirty vertices only). GPU upload is partial (dirty range only). The brush hot path allocates zero heap memory.

The goal is simple: stay fast at high polygon counts and don't make the artist wait.

## Install

### Windows

Download `chisel-windows-x64.zip` from the [latest release](https://github.com/t4paN/chisel-sculpt/releases), extract, and run `chisel.exe`. No installer, no dependencies.

### Linux (AppImage)

Download the AppImage from the [latest release](https://github.com/t4paN/chisel-sculpt/releases), make it executable, and run:

```bash
chmod +x Chisel-x86_64.AppImage
./Chisel-x86_64.AppImage
```

### Build from source (Linux)

Requires: `libglfw3-dev`, `libgl-dev`, `libglx-dev`, `libegl-dev`

```bash
cd chisel-sculpt
cmake -B build && cmake --build build -j$(nproc)
./build/chisel
```

## Acknowledgments

Chisel was vibecoded with Anthropic's Claude — the models did the lifting:

| Model | Commits | What it built |
|---|---:|---|
| **Claude Opus 4.6** | 91 | The core architecture: the multimesh scene (selection, insert mode, mirror logic), cross-entity twin/mirror brushes, the ImGui interface, and the CI / AppImage release pipeline. |
| **Claude Opus 4.7** | 35 | The remesher's heavy polish (GPU selection, tangential smoothing, seam discipline, convergence), GPU brush mirror-seam fixes, per-entity undo, and OBJ import. |
| **Claude Haiku 4.5** | 30 | "lil' haiku" — the Big GPU Refactor that moved every brush onto compute shaders (draw, crease, pinch, move, mirrored sculpting), and brought the iso remesher home — symmetry, seam pinning, and tuning — at a point when Opus 4.6 was hamstrung by compute constraints. |
| **Claude Sonnet 4.6** | 28 | The multires displacement stack, the icosphere / Loop-subdivision base, and the per-entity refactor (MeshEntity, per-object undo, per-entity compute dispatch). |
| **Claude Opus 4.8** | 15 | The SDF mirror-symmetric voxel-merge, plus a final polish pass across brushes, rendering, and the merge/remesh pipeline. |

Early testing by **Ariadne**.

## License

MIT License. See [LICENSE](LICENSE) for details.
