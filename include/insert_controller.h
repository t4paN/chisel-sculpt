#pragma once
#include <cstdint>
#include <vector>
#include "mesh.h"

class Scene;
class Renderer;
struct InputState;
struct Camera;
struct MultiresStack;
struct BrushStroke;
class UndoStack;
struct GLFWwindow;

// InsertController owns the INSERT-mode state machine: click-to-spawn,
// drag-to-scale (PLACING, auto-commits on release), and the
// mirror-symmetrize prompt (MIRROR_PROMPT).
//
// tick() is called every frame regardless of interaction_mode — when the
// user leaves INSERT mode with an active preview, the controller cancels
// it. When mode == INSERT and the caller is in its IDLE app state, the
// state machine runs.
class InsertController {
public:
    InsertController(Scene& scene, Renderer& renderer);

    void tick(InputState& input, Camera& camera,
              MultiresStack& multires,
              BrushStroke& brush_stroke,
              Vec3& mesh_center, float& mesh_radius,
              float dx, bool app_idle,
              int win_w, int win_h, GLFWwindow* window,
              bool& screen_buffers_dirty);

private:
    enum class Phase { IDLE, PLACING, MIRROR_PROMPT };

    struct State {
        Phase phase = Phase::IDLE;

        Vec3 spawn_point = {0, 0, 0};
        float current_radius = 0.0f;
        float base_radius = 0.0f;
        float drag_sensitivity = 0.003f;
        float drag_start_x = 0;
        float drag_start_y = 0;
        float drag_accum = 0.0f;

        int subdiv_level = 4;

        uint32_t preview_mesh_id = 0;
        // Canonical (unit bounding-radius, origin-centred) primitive positions, kept
        // so a drag rescale is just pos = spawn + base_pos * radius. The preview's
        // baked normals need no rescale — uniform scale + translate preserves them.
        std::vector<float> base_pos_x, base_pos_y, base_pos_z;

        void reset() {
            phase = Phase::IDLE;
            spawn_point = {0, 0, 0};
            current_radius = 0.0f;
            base_radius = 0.0f;
            drag_sensitivity = 0.003f;
            drag_accum = 0.0f;
            preview_mesh_id = 0;
            subdiv_level = 4;
            base_pos_x.clear();
            base_pos_y.clear();
            base_pos_z.clear();
        }
    };

    Scene&    scene_;
    Renderer& renderer_;
    State     state_;
};
