// The frame's one-shot operations and shared helpers, moved out of main.cpp as is.
#include "scene.h"          // first: pulls in glad, which must precede any GL header
#include <GLFW/glfw3.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include "app.h"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cctype>
#include <algorithm>
#include <string>

#include "mesh.h"
#include "camera.h"
#include "renderer.h"
#include "input.h"
#include "settings.h"
#include "text_overlay.h"
#include "brush.h"
#include "tablet.h"
#include "undo.h"
#include "multires_stack.h"
#include "remesh.h"
#include "compute.h"
#include "chisel_debug.h"
#include "scene.h"
#include "sdf.h"
#include "insert_controller.h"
#include "ui_overlay.h"
#include "ui_skin.h"
#include "web_bridge.h"
#include "gpu/wgpu_surface.h"
#include "ui_backdrop.h"
#include "debug_console.h"
#include "bench.h"
#include "imgui.h"
#include "imgui_internal.h"   // InputEventsQueue: the idle frame skip reads it
#include "imgui_impl_glfw.h"
#ifdef CHISEL_BACKEND_WEBGPU
#include "imgui_impl_wgpu.h"
#else
#include "imgui_impl_opengl3.h"
#endif
#include "ImGuiFileDialog.h"
#include "brush_alpha.h"
#include "project_file.h"
#include "scene_snapshot.h"
#include "object_xform.h"
#include <string>
#include <filesystem>

// Phase 1 GPU residency: full re-upload of the active entity's mirrored
// level after any wholesale CPU mutation (lock, level switch, projection,
// cascade). Cheap no-op until the stack is locked / compute supported.
void app_refresh_active_gpu_residency(App& a) {
    auto& scene = a.scene;
    auto& compute = a.compute;
    MeshEntity& ent = scene.active_entity();
    ent.multires_gpu.supported = compute.supported;
    ent.multires_gpu.dev       = &compute.gpu_dev;
    // Residency dedupe: right after a GPU cascade replay, the target
    // level's disp/frames still sit in the replay's layer scratch —
    // upload_level copies them device-local instead of re-sending from
    // CPU. The handoff is consumed here (one switch only) so no later
    // refresh can copy stale scratch.
    const gpu::Buffer* dsrc = nullptr;
    const gpu::Buffer* fsrc = nullptr;
    if (compute.cascade_out_valid &&
        compute.cascade_out_passes ==
            ent.multires.current_level - ent.multires.base_level) {
        dsrc = &compute.cascade_disp_ssbo;
        fsrc = &compute.cascade_frames_ssbo;
        std::printf("[residency] disp/frames via gpu copy\n");
    }
    ent.multires_gpu.upload_level(ent.multires, ent.multires.current_level, dsrc, fsrc);
    compute.cascade_out_valid = false;
}
