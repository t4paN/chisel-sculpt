# glbench — standalone GL micro-benchmark for the per-dab upload/barrier cost
# (see glbench.cpp). Not part of the chisel build; configure it on its own:
#
#   cmake -S tools/glbench -B build-glbench -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-glbench --config Release
#
# Windows (static GLFW from deps/): add
#   -DCMAKE_PREFIX_PATH=<path>/deps/glfw -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake_minimum_required(VERSION 3.16)
project(glbench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# Reuse chisel's generated glad loader rather than vendoring a second copy.
set(CHISEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_executable(glbench glbench.cpp "${CHISEL_ROOT}/src/glad.c")
target_include_directories(glbench PRIVATE "${CHISEL_ROOT}/include")
target_link_libraries(glbench PRIVATE OpenGL::GL glfw)
if(UNIX AND NOT APPLE)
    target_link_libraries(glbench PRIVATE dl)
endif()
