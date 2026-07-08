#include "insert_controller.h"
#include "scene.h"
#include "mesh_entity.h"
#include "renderer.h"
#include "input.h"
#include "camera.h"
#include "multires_stack.h"
#include "brush.h"
#include "undo.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// Defined in main.cpp (intentionally non-static so the controller can call it).
bool wrap_cursor(GLFWwindow* window, InputState& input, int win_w, int win_h);

InsertController::InsertController(Scene& scene, Renderer& renderer)
    : scene_(scene), renderer_(renderer) {}

void InsertController::tick(InputState& input, Camera& camera,
                            MultiresStack& /*multires_unused*/,
                            BrushStroke& brush_stroke,
                            Vec3& mesh_center, float& mesh_radius,
                            float dx, bool app_idle,
                            int win_w, int win_h, GLFWwindow* window,
                            bool& screen_buffers_dirty) {
    // Cancel insert preview if user switched away from INSERT mode.
    if (input.interaction_mode != InputState::InteractionMode::INSERT) {
        if (state_.preview_mesh_id != 0) {
            scene_.remove_preview(state_.preview_mesh_id);
            screen_buffers_dirty = true;
            state_.reset();
        }
        return;
    }

    if (!app_idle) return;

    // ESC: cancel active preview, or exit insert mode entirely.
    if (input.quit_requested) {
        input.quit_requested = false;
        if (state_.phase != Phase::IDLE) {
            if (state_.preview_mesh_id != 0) {
                scene_.remove_preview(state_.preview_mesh_id);
                screen_buffers_dirty = true;
            }
            state_.reset();
            std::snprintf(input.notification, sizeof(input.notification), "Insert cancelled");
            input.notification_timer = 1.0f;
        } else {
            state_.reset();
            input.interaction_mode = InputState::InteractionMode::EDIT;
            std::snprintf(input.notification, sizeof(input.notification), "Edit");
            input.notification_timer = 1.0f;
        }
    }

    // Helper: commit the active entity after preview is finalised.
    // Called from PLACING release and MIRROR_PROMPT answer.
    auto finish_commit = [&](uint32_t preview_id, uint32_t vert_count, bool symmetric) {
        scene_.commit_preview(preview_id);

        // Init multires on the newly-active entity.
        multires_stack_init_from_lock(scene_.active_multires(), scene_.active_mesh(), 0);
        scene_.set_mirror_topology(false);
        scene_.refresh_mirror_map();

        // Do NOT clear undo history here. Inserting a new entity does not alter
        // any existing entity's local geometry or multires stack — entries are
        // per-entity and resolve their target (and render offset) at apply time.
        // Clearing would wipe still-valid history for previously sculpted meshes.
        brush_stroke.vertex_count = 0;
        brush_stroke.phase = StrokePhase::NONE;

        scene_.active_mesh().compute_bounding_sphere(mesh_center, mesh_radius);
        scene_.sync();

        if (symmetric) {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Symmetrized mesh %u (%u verts)", preview_id, vert_count);
        } else {
            std::snprintf(input.notification, sizeof(input.notification),
                          "Inserted mesh %u (%u verts)", preview_id, vert_count);
        }
        input.notification_timer = 2.0f;
        state_.reset();
        input.interaction_mode = InputState::InteractionMode::EDIT;
        screen_buffers_dirty = true;
    };

    // MIRROR_PROMPT: Y to symmetrize, N to commit as single.
    if (state_.phase == Phase::MIRROR_PROMPT) {
        bool answer_yes = input.key_y_pressed;
        bool answer_no  = input.key_n_pressed;
        input.key_y_pressed = false;
        input.key_n_pressed = false;

        if (answer_yes || answer_no) {
            if (answer_yes) {
                // Build ONE symmetric mesh: append the preview's X-mirror lobe
                // (X-negated positions, swapped winding) into the same entity, so
                // the committed mesh is sphere ∪ X-mirror with a spatial pair map.
                MeshEntity* pe = scene_.find_entity(state_.preview_mesh_id);
                if (pe) {
                    Mesh& m = pe->mesh;
                    uint32_t vcP = m.vertex_count();
                    uint32_t triN = (uint32_t)m.indices.size();
                    for (uint32_t i = 0; i < vcP; i++) {
                        m.pos_x.push_back(-m.pos_x[i]);
                        m.pos_y.push_back( m.pos_y[i]);
                        m.pos_z.push_back( m.pos_z[i]);
                        m.norm_x.push_back(-m.norm_x[i]);
                        m.norm_y.push_back( m.norm_y[i]);
                        m.norm_z.push_back( m.norm_z[i]);
                    }
                    // Mirrored triangles: offset by vcP, swap 2nd/3rd index so the
                    // lobe faces outward after X-negation.
                    for (uint32_t i = 0; i < triN; i += 3) {
                        m.indices.push_back(m.indices[i + 0] + vcP);
                        m.indices.push_back(m.indices[i + 2] + vcP);
                        m.indices.push_back(m.indices[i + 1] + vcP);
                    }
                    if (!m.mask.empty()) m.mask.resize(m.vertex_count(), 0.0f);
                    m.recompute_normals();
                    m.build_adjacency();
                    m.build_mirror_x_map();   // also stamps mirror_topo_version
                }
            }

            MeshEntity* pe = scene_.find_entity(state_.preview_mesh_id);
            uint32_t vc = pe ? pe->mesh.vertex_count() : 0;
            finish_commit(state_.preview_mesh_id, vc, answer_yes);
        }
    }

    // PLACING: drag to scale, release → auto-commit.
    if (state_.phase == Phase::PLACING) {
        if (input.mouse1_down) {
            bool wrapped = wrap_cursor(window, input, win_w, win_h);
            if (!wrapped)
                state_.drag_accum += dx;
            float new_radius = std::max(0.01f, state_.base_radius + state_.drag_accum * state_.drag_sensitivity);
            if (new_radius != state_.current_radius) {
                state_.current_radius = new_radius;
                MeshEntity* pe = scene_.find_entity(state_.preview_mesh_id);
                if (pe) {
                    uint32_t vc = pe->mesh.vertex_count();
                    for (uint32_t i = 0; i < vc; i++) {
                        pe->mesh.pos_x[i] = state_.spawn_point.x + state_.base_pos_x[i] * state_.current_radius;
                        pe->mesh.pos_y[i] = state_.spawn_point.y + state_.base_pos_y[i] * state_.current_radius;
                        pe->mesh.pos_z[i] = state_.spawn_point.z + state_.base_pos_z[i] * state_.current_radius;
                    }
                    // local indices: 0..vc-1
                    std::vector<uint32_t> local_dirty(vc);
                    for (uint32_t i = 0; i < vc; i++) local_dirty[i] = i;
                    scene_.sync_partial_entity(state_.preview_mesh_id, local_dirty);
                    screen_buffers_dirty = true;
                }
            }
        } else {
            // Mouse released — commit immediately; snap if straddles X=0.
            MeshEntity* pe = scene_.find_entity(state_.preview_mesh_id);
            float snap_tol = state_.current_radius * 0.1f;
            bool touches_plane = false;
            if (pe) {
                uint32_t vc = pe->mesh.vertex_count();
                bool has_pos = false, has_neg = false, has_zero = false;
                for (uint32_t i = 0; i < vc; i++) {
                    float x = pe->mesh.pos_x[i];   // local index, no offset
                    if (std::abs(x) <= snap_tol) has_zero = true;
                    else if (x > 0) has_pos = true;
                    else has_neg = true;
                }
                touches_plane = has_zero || (has_pos && has_neg);
            }

            if (touches_plane && pe) {
                uint32_t vc = pe->mesh.vertex_count();
                state_.spawn_point.x = 0.0f;
                for (uint32_t i = 0; i < vc; i++) {
                    pe->mesh.pos_x[i] = state_.base_pos_x[i] * state_.current_radius;
                    pe->mesh.pos_y[i] = state_.spawn_point.y + state_.base_pos_y[i] * state_.current_radius;
                    pe->mesh.pos_z[i] = state_.spawn_point.z + state_.base_pos_z[i] * state_.current_radius;
                }
                pe->mesh.recompute_normals();
                std::snprintf(input.notification, sizeof(input.notification),
                              "Snapped to center — mesh %u (%u verts)",
                              state_.preview_mesh_id, vc);
                finish_commit(state_.preview_mesh_id, vc, false);
            } else {
                state_.phase = Phase::MIRROR_PROMPT;
                const char* side = (state_.spawn_point.x > 0) ? "X-" : "X+";
                std::snprintf(input.notification, sizeof(input.notification),
                              "Symmetrize to %s? (Y/N)", side);
                input.notification_timer = 60.0f;
            }
        }
    }

    // IDLE: click to place. On-model click hits a tri; empty-canvas click
    // unprojects onto the camera-target plane.
    bool insert_click = state_.phase == Phase::IDLE
        && (input.drag_mode == InputState::DragMode::SCULPT
            || (input.drag_mode == InputState::DragMode::ORBIT && !input.alt_held
                && input.mouse1_just_pressed));
    if (insert_click) {
        input.drag_mode = InputState::DragMode::NONE;

        {
            Vec3 spawn = {0, 0, 0};
            bool valid = false;
            int cx = (int)input.mouse_x;
            int cy = (int)input.mouse_y;

            // Entity-id pick pass over ALL meshes. Read the id under the cursor to
            // test for a hit, and the linear depth to find WHERE it hit. id 0 =
            // miss → fall through to the empty-canvas plane unproject.
            uint32_t hit_id = 0;
            float    hit_depth = 0.0f;
            if (cx >= 0 && cx < win_w && cy >= 0 && cy < win_h) {
                scene_.render_pick(camera, win_w, win_h);
                screen_buffers_dirty = true;  // pick overwrote the shared screen FBO
                renderer_.read_id_region(cx, cy, 1, 1, &hit_id);
                if (hit_id != 0)
                    renderer_.read_depth_region(cx, cy, 1, 1, &hit_depth);
            }

            float ndc_x = 2.0f * (float)cx / (float)win_w - 1.0f;
            float ndc_y = 1.0f - 2.0f * (float)cy / (float)win_h;
            float aspect = (float)win_w / (float)win_h;
            Vec3 fwd = camera.get_view_direction();
            Vec3 world_up = {0, 1, 0};
            Vec3 right = fwd.cross(world_up).normalized();
            Vec3 up = right.cross(fwd).normalized();

            if (hit_id != 0) {
                // Unproject pixel + linear depth → exact world hit point. Same view
                // basis as the off-model branch, distance = the sampled depth.
                float half_h = hit_depth * std::tan(camera.fov * M_PI / 360.0f);
                float half_w = half_h * aspect;
                Vec3 cam_pos = camera.get_position();
                spawn = cam_pos + fwd * hit_depth
                      + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                valid = true;
            } else {
                float half_h = camera.distance * std::tan(camera.fov * M_PI / 360.0f);
                float half_w = half_h * aspect;
                spawn = camera.target + right * (ndc_x * half_w) + up * (ndc_y * half_h);
                valid = true;
            }

            if (valid) {
                state_.spawn_point = spawn;
                float scene_scale = std::max(0.01f, mesh_radius);
                state_.base_radius = scene_scale * 0.15f;
                state_.current_radius = state_.base_radius;
                state_.drag_sensitivity = scene_scale * 0.003f;
                state_.drag_start_x = (float)input.mouse_x;
                state_.drag_start_y = (float)input.mouse_y;

                // Build the chosen primitive in canonical (unit bounding-radius,
                // origin-centred) space with normals already baked. Densities are
                // tuned so each shape ships as a workable base cage.
                Mesh prim;
                switch (input.insert_shape) {
                    case InputState::InsertShape::BOX:
                        prim = box_primitive(12);
                        break;
                    case InputState::InsertShape::CYLINDER:
                        prim = cylinder_primitive(32, 8);
                        break;
                    case InputState::InsertShape::SPHERE:
                    default:
                        prim = icosphere(state_.subdiv_level);
                        prim.recompute_normals();
                        break;
                }
                uint32_t svc = prim.vertex_count();
                state_.base_pos_x.resize(svc);
                state_.base_pos_y.resize(svc);
                state_.base_pos_z.resize(svc);
                for (uint32_t i = 0; i < svc; i++) {
                    Vec3 p = prim.get_pos(i);
                    state_.base_pos_x[i] = p.x;
                    state_.base_pos_y[i] = p.y;
                    state_.base_pos_z[i] = p.z;
                    prim.set_pos(i, spawn + p * state_.current_radius);
                }

                state_.preview_mesh_id = scene_.add_preview(prim, state_.subdiv_level);
                screen_buffers_dirty = true;
                state_.phase = Phase::PLACING;
                std::snprintf(input.notification, sizeof(input.notification),
                              "Insert: drag to scale, release to set size");
                input.notification_timer = 3.0f;
            }
        }
    }
}
