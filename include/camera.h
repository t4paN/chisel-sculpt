#pragma once
#include "mesh.h"

struct Camera {
    Vec3 target;          // orbit target (last sculpt point or mesh center)
    float distance;       // distance from target
    float yaw;            // horizontal angle in radians
    float pitch;          // vertical angle in radians
    float fov;            // field of view in degrees
    float near_plane;
    float far_plane;

    Camera();

    void orbit(float dx, float dy);
    void pan(float dx, float dy, int screen_w, int screen_h);
    void zoom(float delta);
    void set_target(Vec3 t);

    Vec3 get_position() const;
    void get_view_matrix(float* out) const;
    void get_projection_matrix(float* out, float aspect) const;
    Vec3 get_view_direction() const;
    bool world_to_screen(Vec3 pos, int screen_w, int screen_h, float& sx, float& sy) const;
};
