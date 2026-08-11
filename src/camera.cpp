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
    , perspective(false)
    , near_plane(-100.0f)
    , far_plane(100.0f)
{}

float Camera::half_height_at(float d) const {
    float t = std::tan(fov * M_PI / 360.0f);
    // Ortho ignores `d` outright — that is the whole difference between the two
    // projections, and keeping it in one place is what lets every caller be written
    // the same way regardless of which is active.
    return (perspective ? d : distance) * t;
}

float Camera::view_depth(Vec3 p) const {
    Vec3 cam_pos = get_position();
    Vec3 fwd = (target - cam_pos).normalized();
    return fwd.x * (p.x - cam_pos.x) + fwd.y * (p.y - cam_pos.y) + fwd.z * (p.z - cam_pos.z);
}

void Camera::orbit(float dx, float dy) {
    yaw -= dx * 0.005f;    // negated: drag left = model rotates left
    pitch += dy * 0.005f;
    pitch = std::max(-1.5f, std::min(1.5f, pitch));
}

void Camera::pan(float dx, float dy, int screen_w, int screen_h) {
    // Scaled off the framing, not the raw orbit distance: the FOV slider dollies
    // `distance` to hold the framing fixed, so keying on distance would silently make
    // pan slower at wide angles. The constant is the old 0.002 divided by tan(45°/2),
    // which makes this exactly `distance * 0.002` at the default 45° framing.
    float scale = half_height() * 0.0048284f;
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

    Vec3 d = {pos.x - cam_pos.x, pos.y - cam_pos.y, pos.z - cam_pos.z};
    float vx = right.x*d.x + right.y*d.y + right.z*d.z;
    float vy = up.x*d.x + up.y*d.y + up.z*d.z;
    float vz = fwd.x*d.x + fwd.y*d.y + fwd.z*d.z;

    // Under perspective a point at or behind the eye has no screen position at all
    // (the divide flips its sign and it lands, mirrored, on the far side). Ortho has
    // no such notion — it sees everything — so only the perspective path rejects.
    if (perspective && vz <= 1e-4f) return false;

    float half_h = half_height_at(vz);
    float half_w = half_h * aspect;

    float ndcx = vx / half_w;
    float ndcy = vy / half_h;

    sx = (ndcx + 1.0f) * 0.5f * (float)screen_w;
    sy = (1.0f - ndcy) * 0.5f * (float)screen_h;

    return ndcx > -1.0f && ndcx < 1.0f && ndcy > -1.0f && ndcy < 1.0f;
}

void Camera::get_projection_matrix(float* m, float aspect) const {
    std::memset(m, 0, 16 * sizeof(float));

    if (perspective) {
        // Near/far are derived from the orbit distance rather than stored: the model
        // always sits around the target, so scaling with it tracks the zoom. Ortho's
        // stored (-100, 100) cannot be reused — a perspective near plane must be > 0.
        //
        // near is deliberately 100x closer than the target rather than something
        // tighter. That spends depth precision we can afford (Depth24Plus still leaves
        // ~200k levels across the model) to buy margin against the failure that has
        // actually bitten this code: geometry crossing the near plane and the model
        // opening up as you zoom in — see the WebGPU note below.
        float n = std::max(1e-3f, distance * 0.01f);
        float f = distance + 100.0f;
        float t = std::tan(fov * M_PI / 360.0f);
        m[0]  = 1.0f / (aspect * t);
        m[5]  = 1.0f / t;
        m[10] = -(f + n) / (f - n);
        m[11] = -1.0f;                      // the w = -view_z that does the divide
        m[14] = -2.0f * f * n / (f - n);
        m[15] = 0.0f;
    } else {
        float half_h = half_height();
        float half_w = half_h * aspect;
        m[0]  = 1.0f / half_w;
        m[5]  = 1.0f / half_h;
        m[10] = -2.0f / (far_plane - near_plane);
        m[14] = -(far_plane + near_plane) / (far_plane - near_plane);
        m[15] = 1.0f;
    }

#ifdef CHISEL_BACKEND_WEBGPU
    // GL clips NDC z in [-1,1]; WebGPU clips in [0,1]. Remap the z row
    // (z' = 0.5*z + 0.5*w) or the near_plane=-100 "nothing behind the ortho
    // camera plane ever clips" trick silently dies on WebGPU: the camera plane
    // itself becomes the near plane, and zooming close slices the model open
    // (camera sees inside the mesh — GL builds never did this).
    // Written as a row combine rather than a scale+bias on purpose, so it stays
    // correct for the perspective branch too (where the w row is not constant).
    m[2]  = 0.5f * m[2]  + 0.5f * m[3];
    m[6]  = 0.5f * m[6]  + 0.5f * m[7];
    m[10] = 0.5f * m[10] + 0.5f * m[11];
    m[14] = 0.5f * m[14] + 0.5f * m[15];
#endif
}
