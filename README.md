# Chisel

GPU sculpting, anywhere. One C++17 codebase; sculpts natively on WebGPU or OpenGL,
and in your browser with nothing to install.

**▶ Sculpt now: [tpn.itch.io/chisel-sculpt](https://tpn.itch.io/chisel-sculpt)** —
needs a WebGPU browser (Chrome/Edge 113+, or Firefox/Safari with the flag on).

Chisel runs the entire sculpt pipeline on the GPU: brushes, normals, masking,
selection, remeshing, and voxel merging all execute as compute shaders. The CPU
orchestrates; the GPU does the work. It is spartan by design — no menus to dig
through, every control a keystroke or a hold-and-drag — because the two things that
matter are raw speed at high polygon counts and never interrupting the artist.

Project files are **cross-platform**: a `.chisel` saved in the browser opens
identically in the native build, and vice versa.

## Features

- **Eight brushes** — draw, inflate, crease, pinch, move, limb (snakehook), smooth, mask
- **Vertex paint mode** with colour picking; paint survives subdivision, remeshing, and merges
- **Multiple objects** — insert, select, move, scale, delete; per-object undo and subdivision
- **Multires displacement stack** — step down to edit forms, back up for detail
- **X-axis mirror** — real-time symmetric sculpting, masking, and painting
- **Voxel merge** — weld a selection into one watertight manifold, ready for 3D
  printing; mirror-symmetric or boolean-subtract variants, Marching Cubes or Surface Nets
- **Isotropic remesh** — respects masks and mirror symmetry
- **Delta-based undo/redo** — per-stroke, per-object, GPU-resident
- **OBJ / PLY in; OBJ / STL / PLY out** (PLY carries vertex paint)
- **Pen pressure** — XInput2 / WinTab / browser Pointer Events; mouse-first design, tablet optional
- **Matcap shading** — one draw call per object

**Full controls and workflow: [MANUAL.md](MANUAL.md).** Quickstart: left-drag on the
mesh sculpts, off the mesh orbits; `1–4` switch modes; hold `S`/`W`/`A` + drag to tune
the brush; `X` mirrors; `Ctrl+Z` undoes.

## How it works

The mesh stays GPU-resident. A render-target snapshot at pen-down gives the CPU
screen-space depth/normal/triangle-ID buffers to pick against, so a stroke runs with
zero GPU sync until pen-up. Mesh data is flat SOA arrays with CSR adjacency — no
half-edges, no linked lists. Normals recompute partially (dirty vertices only), GPU
uploads are partial, and the brush hot path allocates nothing.

The GPU layer is a thin seam with two backends: **WebGPU** (WGSL kernels — wgpu-native
on desktop, the browser's own WebGPU via Emscripten on the web) and **OpenGL 4.3
compute** (GLSL kernels) as the reference/fallback path. Every kernel exists in
lockstep WGSL + GLSL; the same application code drives both.

## Build from source

Dependencies (native): `libglfw3-dev`, `libgl-dev`, `libglx-dev`, `libegl-dev`, CMake.

```bash
# Native, WebGPU backend (wgpu-native, fetched automatically)
cmake -B build-wgpu -DCHISEL_GPU_BACKEND=webgpu && cmake --build build-wgpu -j$(nproc)
./build-wgpu/chisel

# Native, OpenGL 4.3 backend
cmake -B build-gl -DCHISEL_GPU_BACKEND=gl && cmake --build build-gl -j$(nproc)
./build-gl/chisel

# Browser (Emscripten ≥ 4.0.10 with the emdawnwebgpu port)
source ~/emsdk/emsdk_env.sh
emcmake cmake -B build-web -DCHISEL_GPU_BACKEND=webgpu && cmake --build build-web -j$(nproc)
# serve build-web/ over HTTP and open chisel.html
```

CI builds a Linux AppImage and a Windows zip on every tagged release.

## Command line (native builds)

```
chisel [file] [flags]
```

- `file` — a `.chisel` project or `.obj` model to open on launch.
- `--max-level=N` — raise the subdivision cap (default 9, max 12). Levels
  past 9 get heavy on the CPU (level switches, merge, remesh roughly 4x per
  level); the GPU-limit guard still refuses what your card can't hold.
- `--mirror=spatial` — use the spatial-hash mirror pairing instead of the
  topology map.
- `--toaster` — shrink the undo history budgets for low-memory machines.
- `--ring-mb=N` — debug: cap the GPU undo ring at N MB.

## Acknowledgments

Chisel was vibecoded with Anthropic's Claude — the models did the lifting:

| Model | Commits | What it built |
|---|---:|---|
| **Claude Opus 4.8** | 114 | The SDF voxel-merge (mirror-symmetric, Fast Winding Number, Surface Nets), and the entire WebGPU port: the gpu:: seam, every WGSL kernel, the render path, and the first browser builds. |
| **Claude Opus 4.6** | 91 | The core architecture: the multimesh scene (selection, insert mode, mirror logic), cross-entity twin/mirror brushes, the ImGui interface, and the CI / AppImage release pipeline. |
| **Claude Opus 4.7** | 35 | The remesher's heavy polish (GPU selection, tangential smoothing, seam discipline, convergence), GPU brush mirror-seam fixes, per-entity undo, and OBJ import. |
| **Claude Haiku 4.5** | 30 | "lil' haiku" — the Big GPU Refactor that moved every brush onto compute shaders, and brought the iso remesher home at a point when bigger models were hamstrung by compute constraints. |
| **Claude Sonnet 4.6** | 28 | The multires displacement stack, the icosphere / Loop-subdivision base, and the per-entity refactor (MeshEntity, per-object undo, per-entity compute dispatch). |
| **Claude Fable 5** | 10 | The itch.io release sprint (browser save/open, export fix, input bugs), the cross-platform .chisel format (v4 + legacy migration), and the WebGPU camera-clip fix. |
| **Claude Sonnet 5** | 3 | Emscripten SSBO-limit blocker, SELECT-mode picking fixes, a WGSL reserved-word crash. |

Early testing by **Ariadne**.

## License

MIT License. See [LICENSE](LICENSE) for details.
