#pragma once
#include "project_file.h"

// --- Legacy (.chisel VERSION 1) loader — REMOVABLE MODULE ---
//
// VERSION 1 stored only a single mesh (the active entity), one multires stack,
// camera and options — there was no multimesh scene. This reader rehydrates
// such a file into a one-entity ProjectData so old projects keep loading after
// the v2 (multimesh) format landed.
//
// It is intentionally self-contained: its own chunk parser, no shared helpers
// with project_file.cpp. To drop v1 support, delete project_file_v1.{h,cpp},
// the `#include` + dispatch branch in project_file.cpp, and the source entry in
// CMakeLists.txt — nothing else depends on it.
//
// `path` is the same file path handed to load_project; the function re-opens it
// and re-validates the magic/version (== 1) itself.
LoadResult load_project_v1(const char* path, ProjectData& data);
