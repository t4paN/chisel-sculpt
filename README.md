# Chisel (WebGPU)

Shader-based 3D sculpting, in your browser. C++ / WebGPU (Dawn).

Chisel runs the entire sculpt pipeline on the GPU — brush operations, mesh updates, selection, and remeshing all execute as compute shaders. The CPU orchestrates; the GPU does the work. This is the **WebGPU edition**: the same engine, retargeted from OpenGL compute to WebGPU so it runs natively *and* in the browser, with nothing to install.

> **Status: work in progress.** This is the WebGPU fork of [Chisel](https://github.com/t4paN/chisel-sculpt). The native OpenGL 4.3 build is the source of truth; this tree is porting the GPU backend from GL compute to WebGPU (WGSL) via [Dawn](https://dawn.googlesource.com/dawn), with a browser build as the shipping target on itch.io. Sections below describe the goal — not everything is wired up yet.

## Features

- **Brushes** — draw, smooth, crease, pinch, move, mask
- **Multiple objects** — insert, select, move, and delete independent meshes in one scene, each with its own undo history and subdivision level
- **Insert primitives** — drop a sphere into the scene, drag to scale, optionally mirror it symmetric on insert
- **Subdivision levels** — step up for detail, step down to edit forms, carry displacement between levels
- **Isotropic remesh** — respects masks and mirror symmetry
- **X-axis mirror** — real-time symmetric sculpting
- **Voxel merge** — weld a selection of meshes into one watertight manifold, ready for 3D printing; optionally mirror-symmetric so the result stays editable under X-mirror
- **Undo / redo** — delta-based, per-stroke, per-object
- **Import / export** — OBJ and PLY in; OBJ, STL, and PLY out (PLY carries vertex paint)
- **Save / load** — native project files (.chisel)
- **Matcap shading** — one draw call per object
- **Runs in the browser** — no install, no download; just a WebGPU-capable browser

## Known limitations / TODO

- **WebGPU port is in flight** — the GL 4.3 compute kernels are being translated to WGSL; some brushes/operations may lag the native build during the port
- File and project I/O in the browser goes through the browser's file picker / download (no direct filesystem access)
- Pen pressure in the browser rides the Pointer Events API rather than XInput2; pressure-curve support depends on the browser
- Voxel merge output is evened out (tangential relaxation onto the iso-surface) and can be made mirror-symmetric; heavily sculpted, non-watertight input can still need a resolution bump
- Brush feel could use more polish (falloff curves, stroke interpolation)
- After a *full* undo of inward/carve ("negative") strokes, normals where those strokes overlapped can look slightly off (cosmetic; a quick pass of the smooth brush fixes it)
- No texture painting

## No tablet required

Chisel is designed to work well with a mouse. Hold a key and drag to adjust brush size, strength, hardness, or spacing in real time — no menus, no panels, no digging through settings. A tablet is nice to have, but not necessary.

If you do plug in a graphics tablet, pen pressure is picked up automatically and drives both brush strength and size. Toggle it with **K**. (In the browser this uses the Pointer Events pressure channel; on native Linux it uses XInput2 — the same source Krita uses.)

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
| Ctrl + S | Save project (downloads in the browser) |
| Ctrl + Shift + S | Save as |
| Ctrl + O | Import OBJ (browser file picker) |
| Ctrl + E | Export OBJ (downloads in the browser) |

### Other

| Key | Action |
|---|---|
| Ctrl + Z | Undo |
| Ctrl + Shift + Z | Redo |
| Space | Toggle fullscreen |
| N | Toggle fast normals |
| K | Toggle pen pressure (tablet) |
| Y | Toggle debug mesh view |
| Esc | Quit / leave (native only) |

## How it works

Chisel keeps the mesh GPU-resident. Brush strokes, normal recomputation, selection growth, mask painting, and remesh smoothing all run as compute shaders — the CPU never reads vertex data back during a stroke. A render-target snapshot at pen-down gives the CPU a screen-space depth + normal + triangle ID buffer to pick vertices against, so there's zero GPU sync until the stroke ends.

The mesh uses a flat SOA layout with CSR adjacency — no half-edge structures, no linked lists. Normals are recomputed partially (dirty vertices only). GPU upload is partial (dirty range only). The brush hot path allocates zero heap memory.

For the WebGPU edition, the GPU layer goes through [Dawn](https://dawn.googlesource.com/dawn), so the same C++ engine drives WebGPU compute (WGSL) on the desktop and compiles to a browser build with the same shaders. The GL 4.3 compute kernels are being translated to WGSL one operation at a time.

The goal is simple: stay fast at high polygon counts and don't make the artist wait — in the browser too.

## Play

Chisel (WebGPU) runs in the browser on **itch.io** — *(page link coming once the first build is up)*. No install, no download. You'll need a browser with WebGPU enabled: recent Chrome/Edge (113+), or Firefox/Safari with the WebGPU flag turned on.

## Build from source

> The Dawn/WebGPU build is being set up; these steps will firm up as the toolchain lands.

Chisel (WebGPU) builds from a single C++17 codebase via CMake, with Dawn supplying the WebGPU backend for both native and web (Emscripten) targets.

```bash
cmake -B build && cmake --build build -j$(nproc)
./build/chisel        # native WebGPU (Dawn) build
```

The native OpenGL 4.3 build still lives upstream at [t4paN/chisel-sculpt](https://github.com/t4paN/chisel-sculpt) and remains the reference implementation during the port.

## Acknowledgments

Chisel was vibecoded with Anthropic's Claude — the models did the lifting:

| Model | Commits | What it built |
|---|---:|---|
| **Claude Opus 4.6** | 91 | The core architecture: the multimesh scene (selection, insert mode, mirror logic), cross-entity twin/mirror brushes, the ImGui interface, and the CI / AppImage release pipeline. |
| **Claude Opus 4.7** | 35 | The remesher's heavy polish (GPU selection, tangential smoothing, seam discipline, convergence), GPU brush mirror-seam fixes, per-entity undo, and OBJ import. |
| **Claude Haiku 4.5** | 30 | "lil' haiku" — the Big GPU Refactor that moved every brush onto compute shaders (draw, crease, pinch, move, mirrored sculpting), and brought the iso remesher home — symmetry, seam pinning, and tuning — at a point when Opus 4.6 was hamstrung by compute constraints. |
| **Claude Sonnet 4.6** | 28 | The multires displacement stack, the icosphere / Loop-subdivision base, and the per-entity refactor (MeshEntity, per-object undo, per-entity compute dispatch). |
| **Claude Opus 4.8** | 15 | The SDF mirror-symmetric voxel-merge, a final polish pass across brushes, rendering, and the merge/remesh pipeline — and the WebGPU port. |

Early testing by **Ariadne**.

## License

MIT License. See [LICENSE](LICENSE) for details.
