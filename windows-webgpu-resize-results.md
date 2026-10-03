# Native WebGPU: un-maximize crash, black triangles (2026-10-03)

For the Linux side. Commit `e25146b` on `chisel-windows`.

## Crash on resize (fixed; likely affects Linux too)

Un-maximizing the native WebGPU build aborted every time (`0xc0000409` in
`wgpu_native.dll`). The panic text, once it could be seen:

```
Error in wgpuQueueSubmit: Validation Error
  In a set_scissor_rect command
    Scissor Rect { x: 0, y: 0, w: 1920, h: 1061 } is not contained in the render target (1920, 1017, 1)
```

The frame reads the window size and configures the swapchain **before**
`glfwPollEvents`. `ImGui_ImplGlfw_NewFrame` reads it again **after**. On the frame a
resize lands, ImGui's WebGPU renderer scissors to the new size on the old swapchain.
wgpu-native treats submit-time validation errors as fatal, even with an
uncaptured-error callback. GL tolerates an oversized scissor, so only WebGPU dies. Any
resize can hit it, Linux included; un-maximize just makes it reliable.

Fix: on native WebGPU, clamp `io.DisplaySize` to the configured surface size after
`ImGui_ImplGlfw_NewFrame` (the web build already overrides DisplaySize). Verified: the
user flipped maximize/restore 7 times while sculpting, and a scripted run did 6 more
flips. No crash, stderr empty.

## Why every crash looked silent (affects Linux too)

`debug_console` dup2s stdout **and stderr** into a pipe that is drained once per frame.
Whatever a crashing frame prints, including Rust panic messages, is lost. Set
**`WAYLAND_DEBUG=1`** (the check isn't Linux-only) to turn the capture off when
chasing a crash. Worth considering: drain the pipe from an abort/terminate handler, or
tee to the original fd synchronously.

## Black triangles while not maximized (gone, cause not proven)

On Windows the window was created at the video-mode size **without** maximizing (the
comment said "Start windowed but maximized"). It hung under the taskbar, and in that
state the user saw black triangles while orbiting and sculpting, gone after a maximize.
It runs on **Vulkan** (`[win] adapter: ... via Vulkan`, new log line). Intel's Vulkan
driver reports the swapchain "suboptimal" on nearly every frame after any resize,
including maximized, so that status is not the cause. After this commit the user saw no
triangles in the restored state either. Which change fixed it isn't proven. If it comes
back, compare with `CHISEL_WGPU_BACKEND=dx12`.

## Also in the commit

- `GLFW_MAXIMIZED` at creation (Windows only).
- Outdated/Lost surface: reconfigure and re-acquire, logged.
- Uncaptured-error callback + `wgpuSetLogCallback` (warn level) → `[wgpu]` log lines.
- `CHISEL_WGPU_BACKEND=dx12|vulkan|gl` to pin the API.
