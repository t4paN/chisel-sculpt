#include "camera.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Camera::Camera()
    : target{0, 0, 0}
    , distance(3.0f)
    , yaw(0.0f)
    , pitch(0.0f)
    , fov(45.0f)
    , near_plane(-100.0f)
    , far_plane(100.0f)
{}

void Camera::orbit(float dx, float dy) {
    yaw -= dx * 0.005f;    // negated: drag left = model rotates left
    pitch += dy * 0.005f;
    pitch = std::max(-1.5f, std::min(1.5f, pitch));
}

void Camera::pan(float dx, float dy, int screen_w, int screen_h) {
    float scale = distance * 0.002f;
    Vec3 pos = get_position();
    Vec3 fwd = (target - pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 right = fwd.cross(world_up).normalized();
    Vec3 up = right.cross(fwd).normalized();
    target += right * (-dx * scale) + up * (dy * scale);
}

void Camera::zoom(float delta) {
    distance *= (1.0f - delta * 0.1f);
    distance = std::max(0.05f, std::min(200.0f, distance));
}

void Camera::set_target(Vec3 t) {
    target = t;
}

Vec3 Camera::get_position() const {
    float cp = std::cos(pitch);
    float sp = std::sin(pitch);
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    return {
        target.x + distance * cp * sy,
        target.y + distance * sp,
        target.z + distance * cp * cy
    };
}

Vec3 Camera::get_view_direction() const {
    Vec3 pos = get_position();
    return (target - pos).normalized();
}

void Camera::get_view_matrix(float* m) const {
    Vec3 pos = get_position();
    Vec3 fwd = (target - pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 right = fwd.cross(world_up).normalized();
    Vec3 up = right.cross(fwd).normalized();

    std::memset(m, 0, 16 * sizeof(float));
    m[0] = right.x;    m[4] = right.y;    m[8]  = right.z;
    m[1] = up.x;       m[5] = up.y;       m[9]  = up.z;
    m[2] = -fwd.x;     m[6] = -fwd.y;     m[10] = -fwd.z;
    m[12] = -(right.x*pos.x + right.y*pos.y + right.z*pos.z);
    m[13] = -(up.x*pos.x + up.y*pos.y + up.z*pos.z);
    m[14] = (fwd.x*pos.x + fwd.y*pos.y + fwd.z*pos.z);
    m[15] = 1.0f;
}

bool Camera::world_to_screen(Vec3 pos, int screen_w, int screen_h, float& sx, float& sy) const {
    Vec3 cam_pos = get_position();
    Vec3 fwd = (target - cam_pos).normalized();
    Vec3 world_up = {0, 1, 0};
    Vec3 right = fwd.cross(world_up).normalized();
    Vec3 up = right.cross(fwd).normalized();

    float aspect = (float)screen_w / (float)screen_h;
    float half_h = distance * std::tan(fov * M_PI / 360.0f);
    float half_w = half_h * aspect;

    Vec3 d = {pos.x - cam_pos.x, pos.y - cam_pos.y, pos.z - cam_pos.z};
    float vx = right.x*d.x + right.y*d.y + right.z*d.z;
    float vy = up.x*d.x + up.y*d.y + up.z*d.z;

    float ndcx = vx / half_w;
    float ndcy = vy / half_h;

    sx = (ndcx + 1.0f) * 0.5f * (float)screen_w;
    sy = (1.0f - ndcy) * 0.5f * (float)screen_h;

    return ndcx > -1.0f && ndcx < 1.0f && ndcy > -1.0f && ndcy < 1.0f;
}

void Camera::get_projection_matrix(float* m, float aspect) const {
    float half_h = distance * std::tan(fov * M_PI / 360.0f);
    float half_w = half_h * aspect;

    std::memset(m, 0, 16 * sizeof(float));
    m[0]  = 1.0f / half_w;
    m[5]  = 1.0f / half_h;
    m[10] = -2.0f / (far_plane - near_plane);
    m[14] = -(far_plane + near_plane) / (far_plane - near_plane);
    m[15] = 1.0f;
}
