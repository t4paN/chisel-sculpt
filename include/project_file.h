#pragma once
#include "mesh.h"
#include "camera.h"
#include "multires_stack.h"

struct ProjectData {
    Mesh mesh;
    Camera camera;
    MultiresStack multires;
    bool mirror_x = true;
    int  subdiv_level = 4;
};

enum class SaveResult { OK, ERR_OPEN, ERR_WRITE };
enum class LoadResult { OK, ERR_OPEN, ERR_READ, ERR_MAGIC, ERR_VERSION, ERR_CORRUPT };

SaveResult save_project(const char* path, const ProjectData& data);
LoadResult load_project(const char* path, ProjectData& data);

const char* result_string(SaveResult r);
const char* result_string(LoadResult r);
