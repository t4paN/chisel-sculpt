cmake_minimum_required(VERSION 3.16)
project(chisel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# An unset build type means -O0 — a silent several-times slowdown of all
# CPU-side mesh/brush code. Default to Release unless explicitly overridden.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()

# ---------------------------------------------------------------------------
# GPU backend selection (WebGPU port — see ../webgpu-port-plan.md).
#   gl     : OpenGL 4.3 compute — the reference backend, builds today.
#   webgpu : Dawn (native) / Emscripten (web) WebGPU backend — under construction.
# Default stays `gl` so the native reference build is never broken during the port.
# ---------------------------------------------------------------------------
set(CHISEL_GPU_BACKEND "gl" CACHE STRING "GPU backend: gl | webgpu")
set_property(CACHE CHISEL_GPU_BACKEND PROPERTY STRINGS gl webgpu)
message(STATUS "Chisel GPU backend: ${CHISEL_GPU_BACKEND}")

if(NOT CHISEL_GPU_BACKEND STREQUAL "gl" AND NOT CHISEL_GPU_BACKEND STREQUAL "webgpu")
    message(FATAL_ERROR "CHISEL_GPU_BACKEND must be 'gl' or 'webgpu', got '${CHISEL_GPU_BACKEND}'")
endif()

# ---------------------------------------------------------------------------
# Embedded compute shaders. A build-time generator (cmake/embed_shaders.cmake)
# turns shaders/{glsl,wgsl}/* into one header exposing gpu::embedded_shader(name),
# so the gpu:: seam compiles the canonical kernels straight in (no runtime shader
# dir, no inline copy to drift). Regenerates when any shader changes.
# ---------------------------------------------------------------------------
set(CHISEL_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(CHISEL_SHADERS_HEADER "${CHISEL_GENERATED_DIR}/gpu_shaders_generated.h")
file(GLOB _CHISEL_GLSL CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl/*.comp")
file(GLOB _CHISEL_WGSL CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/wgsl/*.wgsl")
add_custom_command(
    OUTPUT "${CHISEL_SHADERS_HEADER}"
    COMMAND ${CMAKE_COMMAND}
        -DGLSL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl
        -DWGSL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders/wgsl
        -DOUT=${CHISEL_SHADERS_HEADER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shaders.cmake
    DEPENDS ${_CHISEL_GLSL} ${_CHISEL_WGSL} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shaders.cmake
    COMMENT "Embedding compute shaders -> gpu_shaders_generated.h"
    VERBATIM)
add_custom_target(chisel_shaders DEPENDS "${CHISEL_SHADERS_HEADER}")

# ---------------------------------------------------------------------------
# Emscripten web target (Stage 9 bring-up). Cross-compiling with emcmake sets
# EMSCRIPTEN; this must come before the wgpu-native branch below. The web build
# has NO wgpu-native fetch and NO find_package(glfw3/OpenGL/X11) — the browser
# supplies WebGPU via --use-port=emdawnwebgpu (NOT the deprecated -sUSE_WEBGPU)
# and GLFW is emulated by -sUSE_GLFW=3. This block sets up the web-flavored deps
# (imgui) + the shared web flags, then FALLS THROUGH (no return) to the shared
# `chisel` target below, which is guarded backend/EMSCRIPTEN-conditional. The
# earlier probe target (chisel-wgpu-probe) proved the toolchain + async adapter
# pattern; the full app is the next milestone. See ../webgpu-emscripten-handoff.md.
# ---------------------------------------------------------------------------
if(EMSCRIPTEN AND CHISEL_GPU_BACKEND STREQUAL "webgpu")
    # Web flags. The emdawnwebgpu port is needed at compile (headers) AND link;
    # -sUSE_GLFW=3 is a link-only setting (warns if passed at compile), so it's
    # applied to link_options alone below.
    set(CHISEL_WEB_FLAGS --use-port=emdawnwebgpu)

    # Dear ImGui for the web build: core + GLFW platform backend + WebGPU renderer.
    # Same source list as native webgpu, but do NOT define
    # IMGUI_IMPL_WEBGPU_BACKEND_WGPU — imgui_impl_wgpu.h auto-selects the DAWN /
    # emdawnwebgpu flavor under __EMSCRIPTEN__ (the wgpu-native flavor pulls in
    # <webgpu/wgpu.h>, which the web port doesn't ship).
    add_library(imgui_wgpu STATIC
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/imgui_demo.cpp
        external/imgui/imgui_impl_glfw.cpp
        external/imgui/imgui_impl_wgpu.cpp
        external/imgui/ImGuiFileDialog.cpp)  # used by main.cpp; web save/load stubbed later
    target_include_directories(imgui_wgpu PUBLIC external/imgui)
    target_compile_options(imgui_wgpu PUBLIC ${CHISEL_WEB_FLAGS})

    message(STATUS "WebGPU backend (Emscripten): building full app 'chisel' for the browser")
    # no return() — fall through to the shared chisel target below.
endif()

# ---------------------------------------------------------------------------
# WebGPU backend (Stage 2 bring-up). Fetches the wgpu-native prebuilt and builds
# the bring-up probe targets. The full chisel app is NOT portable to webgpu until
# the gpu:: seam lands, so this backend currently builds only the probes; the GL
# build below is skipped entirely (return). Native dev impl = wgpu-native; the web
# build (Emscripten -sUSE_WEBGPU) is wired in a later stage. See ../webgpu-port-plan.md.
# ---------------------------------------------------------------------------
if(NOT EMSCRIPTEN AND CHISEL_GPU_BACKEND STREQUAL "webgpu")
    include(FetchContent)
    set(WGPU_NATIVE_VERSION "v29.0.0.0" CACHE STRING "wgpu-native release tag")
    set(_wgpu_base "https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}")
    if(WIN32)
        set(_wgpu_url "${_wgpu_base}/wgpu-windows-x86_64-msvc-release.zip")
    elseif(APPLE)
        set(_wgpu_url "${_wgpu_base}/wgpu-macos-x86_64-release.zip")
    else()
        set(_wgpu_url "${_wgpu_base}/wgpu-linux-x86_64-release.zip")
    endif()
    message(STATUS "Fetching wgpu-native prebuilt: ${_wgpu_url}")
    FetchContent_Declare(wgpu_native URL "${_wgpu_url}")
    FetchContent_MakeAvailable(wgpu_native)

    add_library(wgpu SHARED IMPORTED GLOBAL)
    if(WIN32)
        # win/apple paths are best-effort (linux-only dev box here) — verified on Linux.
        set_target_properties(wgpu PROPERTIES
            IMPORTED_LOCATION "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll"
            IMPORTED_IMPLIB   "${wgpu_native_SOURCE_DIR}/lib/wgpu_native.dll.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${wgpu_native_SOURCE_DIR}/include")
    else()
        # The prebuilt .so ships without a SONAME; without this CMake records the
        # (relative) link-line path as DT_NEEDED and the loader can't find it from
        # another cwd. IMPORTED_NO_SONAME makes NEEDED the bare filename, resolved
        # via RUNPATH (BUILD_RPATH below).
        set_target_properties(wgpu PROPERTIES
            IMPORTED_LOCATION "${wgpu_native_SOURCE_DIR}/lib/libwgpu_native.so"
            IMPORTED_NO_SONAME TRUE
            INTERFACE_INCLUDE_DIRECTORIES "${wgpu_native_SOURCE_DIR}/include")
    endif()

    add_executable(chisel-wgpu-probe src/gpu/wgpu_probe.cpp)
    target_link_libraries(chisel-wgpu-probe PRIVATE wgpu)
    set_target_properties(chisel-wgpu-probe PROPERTIES BUILD_RPATH "${wgpu_native_SOURCE_DIR}/lib")

    # Stage 2 Step 2: on-screen cleared frame. Needs a windowing system (GLFW with
    # GLFW_NO_API) and, for the X11 surface source, the native X11 handles.
    find_package(glfw3 3.3 REQUIRED)

    # Dear ImGui for the webgpu build: core + GLFW platform backend + WebGPU
    # renderer backend. Do NOT glob external/imgui/*.cpp — that pulls in
    # imgui_impl_opengl3.cpp (needs a GL loader). List sources explicitly. The
    # WGPU renderer needs IMGUI_IMPL_WEBGPU_BACKEND_WGPU (the wgpu-native flavor;
    # it includes <webgpu/wgpu.h>, which the prebuilt ships alongside webgpu.h).
    add_library(imgui_wgpu STATIC
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/imgui_demo.cpp
        external/imgui/imgui_impl_glfw.cpp
        external/imgui/imgui_impl_wgpu.cpp
        external/imgui/ImGuiFileDialog.cpp)  # used by the full app (main.cpp), not the probe
    target_include_directories(imgui_wgpu PUBLIC external/imgui)
    target_compile_definitions(imgui_wgpu PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_WGPU)
    target_link_libraries(imgui_wgpu PUBLIC wgpu glfw)  # wgpu propagates the webgpu headers

    # Stage 3 mesh+camera: pull in the GL-free CPU sources from the reference tree
    # (icosphere generation, mesh normals/adjacency, camera math). None of these
    # touch OpenGL, so they compile clean into the webgpu probe.
    add_executable(chisel-wgpu-window
        src/gpu/wgpu_window.cpp
        src/gpu/webgpu_backend.cpp   # gpu:: seam (WebGPU backend, compute path)
        src/sphere.cpp
        src/mesh.cpp
        src/camera.cpp)
    target_include_directories(chisel-wgpu-window PRIVATE include)
    # Canonical WGSL kernels are loaded from disk at runtime (no drift vs the .wgsl
    # reference files). Build-time embedding lands with the gpu:: seam; until then
    # the probe reads them from the source tree via this absolute path.
    # CHISEL_BACKEND_WEBGPU selects the gpu:: seam's WebGPU backend (gpu/gpu.h).
    target_compile_definitions(chisel-wgpu-window PRIVATE
        CHISEL_WGSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/wgsl"
        CHISEL_BACKEND_WEBGPU)
    target_link_libraries(chisel-wgpu-window PRIVATE wgpu glfw imgui_wgpu)
    set_target_properties(chisel-wgpu-window PROPERTIES BUILD_RPATH "${wgpu_native_SOURCE_DIR}/lib")
    if(UNIX AND NOT APPLE)
        find_package(X11 REQUIRED)
        target_link_libraries(chisel-wgpu-window PRIVATE ${X11_LIBRARIES})
    endif()

    message(STATUS "WebGPU backend: building full app 'chisel' + bring-up probes")
    # NOTE: no return() — the full chisel app is built below, backend-conditional.
elseif(NOT EMSCRIPTEN)
    # ---- GL-only deps (the WebGPU branch set up wgpu / imgui_wgpu / glfw above;
    #      the Emscripten branch set up its own web imgui / flags above) ----
    find_package(OpenGL REQUIRED)
    find_package(glfw3 3.3 REQUIRED)

    # Glad (OpenGL loader) - we'll generate a minimal one
    add_library(glad STATIC src/glad.c)
    target_include_directories(glad PUBLIC include)

    # Dear ImGui (vendored). Exclude imgui_impl_wgpu.cpp — that's the WebGPU renderer
    # backend (vendored for the webgpu build); it needs webgpu headers and must not be
    # compiled into the GL app, which uses imgui_impl_opengl3.cpp.
    file(GLOB IMGUI_SOURCES external/imgui/*.cpp)
    list(FILTER IMGUI_SOURCES EXCLUDE REGEX "imgui_impl_wgpu\\.cpp$")
    add_library(imgui STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui PUBLIC external/imgui)
    target_link_libraries(imgui PRIVATE glfw)
endif()

# The gpu:: seam backend source is the only backend-specific app source: GL uses
# gl_backend.cpp + the glad loader lib; WebGPU uses webgpu_backend.cpp + wgpu.
if(CHISEL_GPU_BACKEND STREQUAL "webgpu")
    set(CHISEL_BACKEND_SRC src/gpu/webgpu_backend.cpp)
else()
    set(CHISEL_BACKEND_SRC src/gpu/gl_backend.cpp)
endif()

# Main executable
if(NOT WIN32 AND NOT EMSCRIPTEN)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
endif()
add_executable(chisel
    src/main.cpp
    src/mesh.cpp
    src/scene.cpp
    src/insert_controller.cpp
    src/renderer.cpp
    src/camera.cpp
    src/input.cpp
    src/sphere.cpp
    src/text_overlay.cpp
    src/brush.cpp
    src/undo.cpp
    src/multires_stack.cpp
    src/multires_gpu.cpp
    src/remesh.cpp
    ${CHISEL_BACKEND_SRC}
    src/compute_common.cpp
    src/compute_draw.cpp
    src/compute_smooth.cpp
    src/compute_crease_pinch.cpp
    src/compute_move.cpp
    src/compute_limb.cpp
    src/compute_mask.cpp
    src/compute_density.cpp
    src/compute_color.cpp
    src/compute_alpha.cpp
    src/compute_remesh.cpp
    src/compute_multires.cpp
    src/brush_alpha.cpp
    src/sdf.cpp
    src/ui_overlay.cpp
    src/project_file.cpp
    src/project_file_v1.cpp
    src/window_icon.cpp
    src/tablet.cpp
)

target_include_directories(chisel PRIVATE include external ${CHISEL_GENERATED_DIR})
# Compute kernels run through the gpu:: seam. The seam selects backend handle types
# at compile time via this macro (see include/gpu/gpu.h). WebGPU also needs the WGSL
# kernel dir (loaded from disk at runtime, like the probe).
if(CHISEL_GPU_BACKEND STREQUAL "webgpu")
    # GLFW must not pull in GL/gl.h: glad.h is still included (via compute.h, for the
    # GL typedefs like GLuint) and errors if a real GL header preceded it. We never
    # call or link GL on this backend, so the typedefs are all we want from glad.
    target_compile_definitions(chisel PRIVATE CHISEL_BACKEND_WEBGPU
        GLFW_INCLUDE_NONE
        CHISEL_WGSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/wgsl")
else()
    target_compile_definitions(chisel PRIVATE CHISEL_BACKEND_GL)
endif()
add_dependencies(chisel chisel_shaders)

# Emscripten web build: emdawnwebgpu port + emulated GLFW (compile+link), ASYNCIFY
# for the event-loop-yielding async readbacks, memory growth, and an .html shell.
# NOTE: JSPI (-sJSPI) was tried 2026-07-02 — near-zero overhead, but it can only
# suspend on a WebAssembly.promising stack. Startup polls (inside main()) work, but
# the RAF main-loop callback is NOT promising, so any in-frame readback throws
# "SuspendError: trying to suspend without WebAssembly.promising". JSPI is viable
# only once NO readback happens inside the frame (see wgpu-finishline.md item 2).
if(EMSCRIPTEN)
    target_compile_options(chisel PRIVATE ${CHISEL_WEB_FLAGS})
    target_link_options(chisel PRIVATE
        ${CHISEL_WEB_FLAGS}
        -sUSE_GLFW=3
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        # ccall backs the file-picker JS→C callback (chisel_web_import_done);
        # without the export it gets DCE'd out of the runtime.
        -sEXPORTED_RUNTIME_METHODS=ccall
        # Custom full-window shell: canvas is 100vw x 100vh (no logo/spinner/textarea,
        # no stock "Resize canvas" button). The app reads clientWidth/clientHeight and
        # sizes the surface/depth to match, so attachments never diverge (see main.cpp).
        --shell-file ${CMAKE_SOURCE_DIR}/packaging/web/shell.html)
    set_target_properties(chisel PROPERTIES SUFFIX ".html"
        LINK_DEPENDS ${CMAKE_SOURCE_DIR}/packaging/web/shell.html)
endif()

# Version: an explicit -DCHISEL_VERSION (CI injects the tag) wins; otherwise
# derive from git; "dev" only as a last resort for a tagless local tree.
if(NOT CHISEL_VERSION)
    execute_process(
        COMMAND git describe --tags --always
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE CHISEL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT CHISEL_VERSION)
    set(CHISEL_VERSION "dev")
endif()
message(STATUS "Chisel version: ${CHISEL_VERSION}")
target_compile_definitions(chisel PRIVATE CHISEL_VERSION="${CHISEL_VERSION}")

if(EMSCRIPTEN)
    # WebGPU + GLFW come from the emdawnwebgpu port / -sUSE_GLFW=3 (no link targets).
    target_link_libraries(chisel PRIVATE imgui_wgpu)
elseif(CHISEL_GPU_BACKEND STREQUAL "webgpu")
    target_link_libraries(chisel PRIVATE wgpu glfw imgui_wgpu)
    # The prebuilt .so ships without a SONAME (see wgpu IMPORTED_NO_SONAME above);
    # RUNPATH lets the loader find it from any cwd.
    set_target_properties(chisel PROPERTIES BUILD_RPATH "${wgpu_native_SOURCE_DIR}/lib")
else()
    target_link_libraries(chisel PRIVATE glad OpenGL::GL glfw imgui)
endif()
if(MSVC)
    target_compile_definitions(chisel PRIVATE _USE_MATH_DEFINES)
endif()
if(NOT WIN32 AND NOT EMSCRIPTEN)
    target_link_libraries(chisel PRIVATE m dl)
endif()

# Pen tablet (src/tablet.cpp): direct Xlib calls need libX11. XInput2 (libXi) is
# dlopen'd at runtime and its headers are vendored under external/xinput2, so
# libxi-dev is NOT a build dependency — only the ubiquitous libX11 dev headers.
# No X11 on the web build (the browser supplies input via GLFW emulation).
if(UNIX AND NOT APPLE AND NOT EMSCRIPTEN)
    find_package(X11 REQUIRED)
    target_link_libraries(chisel PRIVATE ${X11_LIBRARIES})
    target_include_directories(chisel PRIVATE external/xinput2)
endif()

# Windows pen tablet (src/tablet.cpp): WinTab headers are vendored under
# external/wintab and Wintab32.dll is LoadLibrary'd at runtime, so there is no
# build/link dependency (user32/gdi32 for the message window are default).
if(WIN32)
    target_include_directories(chisel PRIVATE external/wintab)
endif()

# GPU seam — GL backend test (Seam Step 2a). Headless GL 4.3 compute check that the
# gpu:: seam's GL backend reproduces the mask kernel (CPU-referenced). GL-only (links
# glad/OpenGL); skipped under the webgpu config.
if(CHISEL_GPU_BACKEND STREQUAL "gl")
    add_executable(chisel-gl-compute-test
        src/gpu/gl_compute_test.cpp
        src/gpu/gl_backend.cpp
        src/sphere.cpp
        src/mesh.cpp)
    target_include_directories(chisel-gl-compute-test PRIVATE include)
    target_compile_definitions(chisel-gl-compute-test PRIVATE
        CHISEL_BACKEND_GL
        CHISEL_GLSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl")
    target_link_libraries(chisel-gl-compute-test PRIVATE glad OpenGL::GL glfw)
    if(UNIX AND NOT WIN32)
        target_link_libraries(chisel-gl-compute-test PRIVATE m dl)
    endif()
endif()

option(CHISEL_DEBUG_MULTIRES "Run post-projection cascade-to-L_max truth check" OFF)
if(CHISEL_DEBUG_MULTIRES)
    target_compile_definitions(chisel PRIVATE CHISEL_DEBUG_MULTIRES)
endif()

option(CHISEL_DEBUG "Enable Chisel debug asserts + synchronous GL debug output" OFF)
if(CHISEL_DEBUG)
    target_compile_definitions(chisel PRIVATE CHISEL_DEBUG)
endif()

# Copy assets next to the executable (works for both single- and multi-config generators)
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
    add_custom_command(TARGET chisel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:chisel>/assets
    )
endif()
