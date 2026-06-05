cmake_minimum_required(VERSION 3.16)
project(chisel LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# Glad (OpenGL loader) - we'll generate a minimal one
add_library(glad STATIC src/glad.c)
target_include_directories(glad PUBLIC include)

# Dear ImGui (vendored)
file(GLOB IMGUI_SOURCES external/imgui/*.cpp)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC external/imgui)
target_link_libraries(imgui PRIVATE glfw)

# Main executable
if(NOT WIN32)
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
    src/remesh.cpp
    src/compute_common.cpp
    src/compute_draw.cpp
    src/compute_smooth.cpp
    src/compute_crease_pinch.cpp
    src/compute_move.cpp
    src/compute_mask.cpp
    src/compute_remesh.cpp
    src/sdf.cpp
    src/ui_overlay.cpp
    src/project_file.cpp
    src/project_file_v1.cpp
    src/tablet.cpp
)

target_include_directories(chisel PRIVATE include)

execute_process(
    COMMAND git describe --tags --always
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CHISEL_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT CHISEL_VERSION)
    set(CHISEL_VERSION "dev")
endif()
target_compile_definitions(chisel PRIVATE CHISEL_VERSION="${CHISEL_VERSION}")

target_link_libraries(chisel PRIVATE glad OpenGL::GL glfw imgui)
if(MSVC)
    target_compile_definitions(chisel PRIVATE _USE_MATH_DEFINES)
endif()
if(NOT WIN32)
    target_link_libraries(chisel PRIVATE m dl)
endif()

# Pen tablet (src/tablet.cpp): direct Xlib calls need libX11. XInput2 (libXi) is
# dlopen'd at runtime and its headers are vendored under external/xinput2, so
# libxi-dev is NOT a build dependency — only the ubiquitous libX11 dev headers.
if(UNIX AND NOT APPLE)
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

option(CHISEL_DEBUG_MULTIRES "Run post-projection cascade-to-L_max truth check" OFF)
if(CHISEL_DEBUG_MULTIRES)
    target_compile_definitions(chisel PRIVATE CHISEL_DEBUG_MULTIRES)
endif()

# Copy assets next to the executable (works for both single- and multi-config generators)
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
    add_custom_command(TARGET chisel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:chisel>/assets
    )
endif()
