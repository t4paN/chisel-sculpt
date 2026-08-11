#pragma once
#include "mesh.h"

struct Camera {
    Vec3 target;          // orbit target (last sculpt point or mesh center)
    float distance;       // distance from target
    float yaw;            // horizontal angle in radians
    float pitch;          // vertical angle in radians
    float fov;            // vertical field of view in degrees
    // Orthographic (false) is the historical projection and still the default: every
    // brush was tuned against it, and `fov` there only scales the framing. Perspective
    // is opt-in from the burger menu — see half_height_at() for what changes.
    bool perspective;
    float near_plane;     // ortho only; the perspective pair is derived per frame
    float far_plane;

    Camera();

    void orbit(float dx, float dy);
    void pan(float dx, float dy, int screen_w, int screen_h);
    void zoom(float delta);
    void set_target(Vec3 t);

    // THE foreshortening term. World half-height of the view volume at view-space
    // depth `d`: constant under ortho (fixed by the orbit distance, which is why so
    // much of the brush code could get away with reading `distance` directly), but
    // proportional to `d` under perspective. Any screen<->world conversion must go
    // through this with the depth of the point it is converting, not with `distance`.
    float half_height_at(float d) const;
    // Framing at the orbit target — the "zoom level", identical in both projections.
    float half_height() const { return half_height_at(distance); }
    // Depth of a world point along the view axis, positive into the screen. This is
    // the `d` that half_height_at() wants.
    float view_depth(Vec3 p) const;

    Vec3 get_position() const;
    void get_view_matrix(float* out) const;
    void get_projection_matrix(float* out, float aspect) const;
    Vec3 get_view_direction() const;
    bool world_to_screen(Vec3 pos, int screen_w, int screen_h, float& sx, float& sy) const;
};
