#include "gpu/gpu.h"
#include "settings.h"
#include "input.h"
#include "brush_alpha.h"   // fill_patterns()

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#elif defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

constexpr int   kFormatVersion = 1;
constexpr float kDebounceSec   = 1.0f;   // quiet time after the last edit before a write

// Stable on-disk keys for the brush slots. Deliberately NOT InputState::brush_name():
// that returns display text ("Draw (-)", and it renames Smooth when the lock is on), so
// keying off it would silently orphan every saved brush the day a label changes.
const char* kBrushKey[(int)BrushType::COUNT] = {
    "draw", "clay", "inflate", "crease", "pinch",
    "move", "limb", "pose", "smooth", "mask", "paint", "bucket", "stipple", "paint_stipple", "paint_blend"
};
static_assert((int)BrushType::COUNT == 15, "kBrushKey is out of sync with BrushType");

// Section names of the per-device brush-feel profiles written until 2026-11-12. Read
// once to migrate a stored blob into the single [brush] section, never written.
const char* kLegacyProfileKey[2] = { "mouse", "tablet" };

struct BrushFeel {
    float max_effect;
    BrushSettings per_brush[(int)BrushType::COUNT];
};

// Index 0..4 are the procedural builtins (see AlphaLibrary::init_builtins). A custom
// image lives only in this session's MEMFS/pool, so its index means nothing next run —
// persisting it would restore a stamp the library no longer has at that slot.
constexpr int kBuiltinAlphaCount = 5;

bool g_storage_ok  = true;   // flipped false by the first sink refusal
bool g_dirty       = false;
float g_since_edit = 0.0f;
float g_since_poll = 0.0f;

// Last blob we wrote (or loaded). The change check is "serialize and compare" rather
// than a parallel POD mirror of every field: a mirror means every new setting has to be
// added in two places, and the day someone adds it in one is the day saves go quiet.
// The cost is one small string build a few times a second, never during a stroke.
std::string g_last_blob;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- sink: web (localStorage) ----------------------------------------------------
#ifdef __EMSCRIPTEN__

// All three wrap the access in try/catch: localStorage throws, not returns null, when
// the browser refuses it (Safari private mode, a partitioned third-party iframe with
// storage blocked). An unhandled throw here would take the frame down with it.
std::string storage_read() {
    char* s = (char*)EM_ASM_PTR({
        try {
            var v = window.localStorage.getItem('chisel.settings');
            if (v === null) return 0;
            return stringToNewUTF8(v);
        } catch (e) { return 0; }
    });
    if (!s) return std::string();
    std::string out(s);
    free(s);
    return out;
}

bool storage_write(const std::string& blob) {
    return EM_ASM_INT({
        try {
            window.localStorage.setItem('chisel.settings', UTF8ToString($0));
            return 1;
        } catch (e) { return 0; }   // refused, or quota exhausted
    }, blob.c_str()) != 0;
}

void storage_clear() {
    EM_ASM({ try { window.localStorage.removeItem('chisel.settings'); } catch (e) {} });
}

#else
// ---- sink: native (config file) --------------------------------------------------

// $XDG_CONFIG_HOME/chisel/settings.cfg, falling back to ~/.config; %APPDATA% on Windows.
// Empty when we cannot work out a home dir at all, which disables persistence rather
// than scattering a dotfile into whatever the cwd happens to be.
std::string config_path(bool make_dir) {
    std::string dir;
#ifdef _WIN32
    const char* appdata = std::getenv("APPDATA");
    if (!appdata || !*appdata) return std::string();
    dir = std::string(appdata) + "\\chisel";
    if (make_dir) _mkdir(dir.c_str());
    return dir + "\\settings.cfg";
#else
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) {
        dir = std::string(xdg) + "/chisel";
    } else {
        const char* home = std::getenv("HOME");
        if (!home || !*home) return std::string();
        dir = std::string(home) + "/.config/chisel";
    }
    if (make_dir) {
        // Parent first: ~/.config may not exist on a bare account. EEXIST is fine.
        std::string parent = dir.substr(0, dir.rfind('/'));
        mkdir(parent.c_str(), 0755);
        mkdir(dir.c_str(), 0755);
    }
    return dir + "/settings.cfg";
#endif
}

std::string storage_read() {
    std::string path = config_path(false);
    if (path.empty()) return std::string();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

bool storage_write(const std::string& blob) {
    std::string path = config_path(true);
    if (path.empty()) return false;
    // Write-and-rename so a crash mid-write cannot leave a half-file that parses into
    // a nonsense half-configured app.
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    size_t wrote = std::fwrite(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    if (wrote != blob.size()) { std::remove(tmp.c_str()); return false; }
    std::remove(path.c_str());               // Windows rename won't clobber
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

void storage_clear() {
    std::string path = config_path(false);
    if (!path.empty()) std::remove(path.c_str());
}
#endif

// ---- serialize --------------------------------------------------------------------

void append_kv(std::string& s, const char* key, float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s=%.6g\n", key, (double)v);
    s += buf;
}
void append_kv(std::string& s, const char* key, int v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s=%d\n", key, v);
    s += buf;
}
void append_kv(std::string& s, const char* key, bool v) {
    append_kv(s, key, v ? 1 : 0);
}

void append_brush_feel(std::string& s, const InputState& in) {
    s += "\n[brush]\n";
    append_kv(s, "max_effect", in.max_effect);
    for (int i = 0; i < (int)BrushType::COUNT; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "brush.%s=%.6g %.6g %.6g\n", kBrushKey[i],
                      (double)in.per_brush[i].strength,
                      (double)in.per_brush[i].hardness,
                      (double)in.per_brush[i].spacing);
        s += buf;
    }
}

std::string serialize(const InputState& in) {
    std::string s;
    s.reserve(2048);
    s += "# Chisel settings. Regenerated on change; unknown keys are ignored.\n";
    append_kv(s, "version", kFormatVersion);

    s += "\n[global]\n";
    // brush_size is deliberately absent — see InputState::brush_size. Because the
    // change check is "serialize and compare", leaving it out also means dragging the
    // size slider no longer marks settings dirty at all.
    append_kv(s, "per_brush_sizes",     in.per_brush_sizes);
    append_kv(s, "autosmooth",          in.autosmooth);
    append_kv(s, "speed_gain",          in.speed_gain);
    append_kv(s, "matcap_contrast",     in.matcap_contrast);
    append_kv(s, "flat_shading",        in.flat_shading);
    append_kv(s, "show_fps",            in.show_fps);
    append_kv(s, "ui_skin",             (int)in.ui_skin);
    append_kv(s, "ui_material",         in.ui_material);
    append_kv(s, "ui_motion",           in.ui_motion);
    append_kv(s, "ui_viewfinder",       in.ui_viewfinder);
    append_kv(s, "ui_viewfinder_px",    in.ui_viewfinder_px);
    append_kv(s, "ui_scale_touch",      in.ui_scale_touch);
    append_kv(s, "help_seen",           in.help_seen);
    if (!in.browse_default_dir.empty()) s += "browse_default=" + in.browse_default_dir + "\n";
    append_kv(s, "camera_perspective",  in.camera_perspective);
    append_kv(s, "camera_fov",          in.camera_fov);
    append_kv(s, "remesh_detail",       in.remesh_detail);
    append_kv(s, "uv_padding",          in.uv_padding);
    append_kv(s, "sphere_kind",         (int)in.sphere_kind);
    append_kv(s, "gpu_memory_gb",       in.gpu_memory_gb);
    append_kv(s, "relative_pen",        in.relative_pen);
    append_kv(s, "pen_sensitivity",     in.pen_sensitivity);
    append_kv(s, "pen_lift_height",     in.pen_lift_height);
    // mirror_x is deliberately absent — see InputState::mirror_x. Symmetry is session
    // state that starts ON every run, so it must not survive into the next one. As with
    // brush_size, leaving it out of the "serialize and compare" blob also means pressing
    // X no longer marks settings dirty.
    // mirror_topological IS persisted, unlike mirror_x above: which mirror you want
    // is a preference about how the app works, not per-session state.
    append_kv(s, "mirror_topological", in.mirror_topological);
    append_kv(s, "fast_normals",        in.fast_normals);
    append_kv(s, "paint_visible",       in.paint_visible);
    append_kv(s, "clay_melt",           in.clay_melt);
    append_kv(s, "crease_dot",          in.crease_dot);
    append_kv(s, "crease_build",        in.crease_build);
    append_kv(s, "stipple_rot_jitter",  in.stipple.rot_jitter);
    append_kv(s, "stipple_size_jitter", in.stipple.size_jitter);
    append_kv(s, "stipple_depth_jitter", in.stipple.depth_jitter);
    append_kv(s, "stipple_scatter",     in.stipple.scatter);
    append_kv(s, "stipple_copies",      in.stipple.copies);
    append_kv(s, "stipple_copy_size",   in.stipple.copy_size);
    append_kv(s, "paint_stipple_rot_jitter",   in.paint_stipple.rot_jitter);
    append_kv(s, "paint_stipple_size_jitter",  in.paint_stipple.size_jitter);
    append_kv(s, "paint_stipple_depth_jitter", in.paint_stipple.depth_jitter);
    append_kv(s, "paint_stipple_scatter",      in.paint_stipple.scatter);
    append_kv(s, "paint_stipple_copies",       in.paint_stipple.copies);
    append_kv(s, "paint_stipple_copy_size",    in.paint_stipple.copy_size);
    append_kv(s, "paint_stipple_two_colours",  in.paint_stipple.two_colours);
    append_kv(s, "density_coarse_mult", in.density_coarse_mult);
    append_kv(s, "density_fine_mult",   in.density_fine_mult);
    // Shortcuts: only the ones that differ from the factory binding, as key.<id>=<glfw
    // key>:<mods>. Untouched keys stay unwritten so a later change to a default reaches
    // them; an unbound key is written as -1 so it stays unbound.
    for (int i = 0; i < (int)KeyAction::COUNT; i++) {
        const KeyActionInfo& info = key_action_info((KeyAction)i);
        const KeyBind& b = in.keybinds[i];
        if (b == info.def) continue;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "key.%s=%d:%d\n", info.id, b.bound() ? b.key : -1,
                      b.bound() ? b.mods : 0);
        s += buf;
    }
    if (in.active_alpha >= 0 && in.active_alpha < kBuiltinAlphaCount)
        append_kv(s, "active_alpha", in.active_alpha);
    // Fill patterns are all built in, so an index is stable (new ones are appended).
    append_kv(s, "fill_pattern", in.fill_pattern);
    append_kv(s, "fill_angle",   in.fill_angle);
    append_kv(s, "bucket_paint_walls", in.bucket_paint_walls);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "paint_color=%.6g %.6g %.6g\n",
                  (double)in.paint_color[0], (double)in.paint_color[1],
                  (double)in.paint_color[2]);
    s += buf;
    std::snprintf(buf, sizeof(buf), "paint_color_alt=%.6g %.6g %.6g\n",
                  (double)in.paint_color_alt[0], (double)in.paint_color_alt[1],
                  (double)in.paint_color_alt[2]);
    s += buf;
    std::snprintf(buf, sizeof(buf), "unselected_tint=%.6g %.6g %.6g\n",
                  (double)in.unselected_tint[0], (double)in.unselected_tint[1],
                  (double)in.unselected_tint[2]);
    s += buf;

    append_brush_feel(s, in);
    return s;
}

// ---- parse ------------------------------------------------------------------------

bool parse_bool(const std::string& v) { return v == "1" || v == "true" || v == "yes"; }

void parse_rgb(const std::string& v, float* out) {
    float c[3];
    if (std::sscanf(v.c_str(), "%f %f %f", &c[0], &c[1], &c[2]) != 3) return;
    for (int i = 0; i < 3; i++) out[i] = clampf(c[i], 0.0f, 1.0f);
}

// Every value is clamped to its slider's range on the way in. The blob is user-visible
// (a file on native, devtools-editable on web) and a persisted out-of-range value would
// come back every launch — the one bug class persistence newly makes permanent.
void apply_brush(const std::string& v, BrushSettings& b, BrushType which) {
    float st, hd, sp;
    if (std::sscanf(v.c_str(), "%f %f %f", &st, &hd, &sp) != 3) return;
    b.strength = clampf(st, 0.01f, 1.0f);
    b.hardness = clampf(hd, 0.01f, 1.0f);
    // Per-brush spacing ceiling, not a flat 1.0 — clay breaks up into a stamp lattice
    // above 0.15 (see max_spacing_for). A blob written before this clamp existed can
    // carry clay a spacing from another brush, which is exactly how the lattice
    // shipped; clamping on the way in retires that blob instead of reloading it.
    b.spacing  = clampf(sp, 0.05f, max_spacing_for(which));
}

void apply_feel_key(BrushFeel& p, const std::string& key, const std::string& val) {
    // Low end is the slider's 0.10, NOT 0: max_effect is the ceiling of a ramp whose
    // floor is PRESSURE_STR_FLOOR (0.05), and a ceiling below the floor would invert it.
    if      (key == "max_effect") p.max_effect = clampf(std::strtof(val.c_str(), nullptr), 0.10f, 1.0f);
    else if (key.compare(0, 6, "brush.") == 0) {
        std::string name = key.substr(6);
        for (int i = 0; i < (int)BrushType::COUNT; i++)
            if (name == kBrushKey[i]) { apply_brush(val, p.per_brush[i], (BrushType)i); return; }
    }
}

void apply_global_key(InputState& in, const std::string& key, const std::string& val) {
    // No "brush_size" case: a stored blob from before 2026-08-17 still carries the key,
    // and it falls through as an unknown key — which is exactly the desired migration
    // (the value is dropped, and the next write stops emitting it). Same for "mirror_x"
    // as of 2026-08-24 — a stored `mirror_x=0` must not be able to start a session with
    // symmetry off.
    if      (key == "matcap_contrast")     in.matcap_contrast     = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "clay_melt")           in.clay_melt           = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "crease_dot")          in.crease_dot          = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "crease_build")        in.crease_build        = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "stipple_rot_jitter")  in.stipple.rot_jitter  = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "stipple_size_jitter") in.stipple.size_jitter = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "stipple_depth_jitter") in.stipple.depth_jitter = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "stipple_scatter")     in.stipple.scatter     = parse_bool(val);
    else if (key == "stipple_copies")      in.stipple.copies      = (int)clampf(std::strtof(val.c_str(), nullptr), 1.0f, 64.0f);
    else if (key == "stipple_copy_size")   in.stipple.copy_size   = clampf(std::strtof(val.c_str(), nullptr), 0.05f, 1.0f);
    else if (key == "paint_stipple_rot_jitter")   in.paint_stipple.rot_jitter   = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "paint_stipple_size_jitter")  in.paint_stipple.size_jitter  = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "paint_stipple_depth_jitter") in.paint_stipple.depth_jitter = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "paint_stipple_scatter")      in.paint_stipple.scatter      = parse_bool(val);
    else if (key == "paint_stipple_copies")       in.paint_stipple.copies       = (int)clampf(std::strtof(val.c_str(), nullptr), 1.0f, 64.0f);
    else if (key == "paint_stipple_copy_size")    in.paint_stipple.copy_size    = clampf(std::strtof(val.c_str(), nullptr), 0.05f, 1.0f);
    else if (key == "paint_stipple_two_colours")  in.paint_stipple.two_colours  = parse_bool(val);
    else if (key == "density_coarse_mult") in.density_coarse_mult = clampf(std::strtof(val.c_str(), nullptr), 1.0f, 4.0f);
    else if (key == "density_fine_mult")   in.density_fine_mult   = clampf(std::strtof(val.c_str(), nullptr), 0.2f, 1.0f);
    else if (key == "per_brush_sizes")     in.per_brush_sizes     = parse_bool(val);
    else if (key == "autosmooth")          in.autosmooth          = parse_bool(val);
    else if (key == "speed_gain")          in.speed_gain          = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "camera_fov")          in.camera_fov          = clampf(std::strtof(val.c_str(), nullptr), 15.0f, 80.0f);
    else if (key == "relative_pen")        in.relative_pen        = parse_bool(val);
    else if (key == "pen_sensitivity")     in.pen_sensitivity     = clampf(std::strtof(val.c_str(), nullptr), 0.25f, 4.0f);
    else if (key == "pen_lift_height")     in.pen_lift_height     = clampf(std::strtof(val.c_str(), nullptr), 0.05f, 1.0f);
    else if (key == "remesh_detail")       in.remesh_detail       = clampf(std::strtof(val.c_str(), nullptr), 0.25f, 4.0f);
    else if (key == "flat_shading")        in.flat_shading        = parse_bool(val);
    else if (key == "show_fps")            in.show_fps            = parse_bool(val);
    else if (key.compare(0, 4, "key.") == 0) {
        // Unknown ids (an action a newer build added, then this build reads the blob)
        // are skipped, and a key outside GLFW's range or one that is reserved is
        // ignored rather than trusted — a hand-edited file must not be able to bind Esc.
        const std::string id = key.substr(4);
        int k = GLFW_KEY_UNKNOWN, m = 0;
        if (std::sscanf(val.c_str(), "%d:%d", &k, &m) < 1) return;
        if (k != GLFW_KEY_UNKNOWN && (k < GLFW_KEY_SPACE || k > GLFW_KEY_LAST || keybind_reserved(k)))
            return;
        for (int i = 0; i < (int)KeyAction::COUNT; i++) {
            if (id == key_action_info((KeyAction)i).id) {
                in.keybinds[i] = { k, k == GLFW_KEY_UNKNOWN ? 0 : (m & kKeyModMask) };
                break;
            }
        }
    }
    else if (key == "ui_skin") {
        int k = (int)std::strtol(val.c_str(), nullptr, 10);
        if (k >= 0 && k <= (int)InputState::UiSkin::Y2K) in.ui_skin = (InputState::UiSkin)k;
    }
    // Pre-DOS builds stored the skin as a bool (2026-10-03 only); the next write
    // replaces it with ui_skin.
    else if (key == "ui_classic" && parse_bool(val)) in.ui_skin = InputState::UiSkin::CLASSIC;
    else if (key == "ui_material")         in.ui_material         = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 100.0f);
    else if (key == "ui_motion")           in.ui_motion           = parse_bool(val);
    else if (key == "ui_viewfinder")       in.ui_viewfinder       = parse_bool(val);
    else if (key == "ui_viewfinder_px")    in.ui_viewfinder_px    = clampf(std::strtof(val.c_str(), nullptr), 0.5f, 6.0f);
    else if (key == "ui_scale_touch") {
        int v = (int)std::strtol(val.c_str(), nullptr, 10);
        in.ui_scale_touch = v < 40 ? 40 : (v > 150 ? 150 : v);
    }
    else if (key == "help_seen")           in.help_seen           = parse_bool(val);
    else if (key == "browse_default")      in.browse_default_dir  = val;
    else if (key == "uv_padding") {   // 2..16 in steps of 2, as the slider sets it
        const int v = (int)std::strtol(val.c_str(), nullptr, 10);
        in.uv_padding = std::max(2, std::min(16, (v + 1) / 2 * 2));
    }
    else if (key == "camera_perspective")  in.camera_perspective  = parse_bool(val);
    else if (key == "mirror_topological")  in.mirror_topological  = parse_bool(val);
    else if (key == "fast_normals")        in.fast_normals        = parse_bool(val);
    else if (key == "paint_visible")       in.paint_visible       = parse_bool(val);
    else if (key == "paint_color")         parse_rgb(val, in.paint_color);
    else if (key == "paint_color_alt")     parse_rgb(val, in.paint_color_alt);
    else if (key == "unselected_tint")     parse_rgb(val, in.unselected_tint);
    else if (key == "gpu_memory_gb") {
        // Only a size the table has a row for; anything else keeps the default.
        const int gb = std::atoi(val.c_str());
        for (const gpu::MemoryRow& r : gpu::kMemoryRows) if (r.gb == gb) in.gpu_memory_gb = gb;
    }
    else if (key == "sphere_kind") {
        // Range-checked like every other value on the way in: the blob is
        // user-editable, and an out-of-range enum would come back every launch.
        int k = (int)std::strtol(val.c_str(), nullptr, 10);
        if (k >= 0 && k <= (int)InputState::SphereKind::UV)
            in.sphere_kind = (InputState::SphereKind)k;
    }
    else if (key == "active_alpha") {
        int a = (int)std::strtol(val.c_str(), nullptr, 10);
        if (a >= 0 && a < kBuiltinAlphaCount) in.active_alpha = a;
    }
    else if (key == "fill_pattern") {
        int p = (int)std::strtol(val.c_str(), nullptr, 10);
        if (p >= 0 && p < fill_patterns().count()) in.fill_pattern = p;
    }
    else if (key == "fill_angle") in.fill_angle = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 180.0f);
    else if (key == "bucket_paint_walls") in.bucket_paint_walls = parse_bool(val);
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Applies over whatever `in` already holds, so anything the blob omits keeps its
// constructor default. Section-scoped: -1 = header area, -2 = [global], -3 = [brush],
// 0/1 = a legacy [mouse]/[tablet] profile.
void deserialize(const std::string& blob, InputState& in) {
    int section = -1;
    size_t pos = 0;
    std::string pending_profile;   // legacy: active_profile arrives before the sections do

    BrushFeel feel;
    feel.max_effect = in.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) feel.per_brush[i] = in.per_brush[i];
    bool have_feel = false;
    // Legacy profiles start from the same defaults with their old max-effect defaults
    // (0.6 mouse, 1.0 tablet), so a section that never wrote the key migrates right.
    BrushFeel legacy[2] = { feel, feel };
    legacy[0].max_effect = MOUSE_EFFECT_HOLDBACK;
    legacy[1].max_effect = 1.0f;
    bool have_legacy = false;

    while (pos <= blob.size()) {
        size_t eol = blob.find('\n', pos);
        if (eol == std::string::npos) eol = blob.size();
        std::string line = trim(blob.substr(pos, eol - pos));
        pos = eol + 1;
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '[' && line.back() == ']') {
            std::string name = line.substr(1, line.size() - 2);
            section = -1;
            if (name == "global") section = -2;
            else if (name == "brush") { section = -3; have_feel = true; }
            else for (int i = 0; i < 2; i++)
                if (name == kLegacyProfileKey[i]) { section = i; have_legacy = true; break; }
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (section == -2)      apply_global_key(in, key, val);
        else if (section == -3) apply_feel_key(feel, key, val);
        else if (section >= 0)  apply_feel_key(legacy[section], key, val);
        else if (key == "active_profile") pending_profile = val;
        // `version` is read but unused at v1 — it exists so a future format break has a
        // discriminator to branch on instead of guessing from which keys are present.
    }

    // A blob from before 2026-11-12 has per-device profiles and no [brush]. Keep the
    // one that was active (the one the user was tuning; mouse if unsaid). Its max effect
    // maps onto the shared slider: the mouse now lands at MOUSE_EFFECT_HOLDBACK of the
    // ceiling, so the old mouse value is divided by it (0.6 -> 100%), and the pen's
    // carries over as is. The next save writes [brush] and drops the old sections.
    if (!have_feel && have_legacy) {
        const int a = pending_profile == kLegacyProfileKey[1] ? 1 : 0;
        feel = legacy[a];
        if (a == 0)
            feel.max_effect = clampf(feel.max_effect / MOUSE_EFFECT_HOLDBACK, 0.10f, 1.0f);
    }
    in.max_effect = feel.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) in.per_brush[i] = feel.per_brush[i];
    in.sync_live_settings();
}

} // namespace

// ---- public -----------------------------------------------------------------------

void settings_load(InputState& input) {
    std::string blob = storage_read();
    if (blob.empty()) {
        // Nothing stored yet is the normal first-run path, not a failure. Seed the
        // comparison baseline so the first tick does not write an identical blob back.
        g_last_blob = serialize(input);
        return;
    }
    deserialize(blob, input);
    g_last_blob = serialize(input);
}

void settings_save(InputState& input) {
    std::string blob = serialize(input);
    if (!storage_write(blob)) { g_storage_ok = false; return; }
    g_last_blob = blob;
    g_dirty = false;
}

void settings_tick(InputState& input, float dt) {
    // Never touch storage mid-stroke: this is the one place per frame that could
    // allocate, and the stroke loop is the hot path the architecture rules protect.
    if (input.sculpting) return;

    if (g_dirty) {
        g_since_edit += dt;
        if (g_since_edit >= kDebounceSec) settings_save(input);
        return;
    }

    // Poll a few times a second rather than every frame — a slider drag settles long
    // before the debounce elapses, so finer granularity buys nothing.
    g_since_poll += dt;
    if (g_since_poll < 0.25f) return;
    g_since_poll = 0.0f;

    if (serialize(input) != g_last_blob) {
        g_dirty = true;
        g_since_edit = 0.0f;
    }
}

void settings_reset(InputState& input) {
    InputState fresh;   // the single source of defaults — no second copy to drift
    // brush_size is session state now, so Reset leaves it where the user put it —
    // same as it leaves the current brush and the smooth lock alone.
    input.per_brush_sizes    = fresh.per_brush_sizes;
    input.seed_brush_sizes();
    input.autosmooth         = fresh.autosmooth;
    input.speed_gain         = fresh.speed_gain;
    input.matcap_contrast    = fresh.matcap_contrast;
    input.flat_shading       = fresh.flat_shading;
    input.show_fps           = fresh.show_fps;
    input.browse_default_dir = fresh.browse_default_dir;
    input.ui_skin            = fresh.ui_skin;
    input.ui_material        = fresh.ui_material;
    input.ui_motion          = fresh.ui_motion;
    input.ui_viewfinder      = fresh.ui_viewfinder;
    input.ui_viewfinder_px   = fresh.ui_viewfinder_px;
    input.ui_scale_touch     = fresh.ui_scale_touch;
    input.camera_perspective = fresh.camera_perspective;
    input.camera_fov         = fresh.camera_fov;
    input.remesh_detail      = fresh.remesh_detail;
    input.sphere_kind        = fresh.sphere_kind;
    input.gpu_memory_gb      = fresh.gpu_memory_gb;
    input.relative_pen       = fresh.relative_pen;
    input.pen_sensitivity    = fresh.pen_sensitivity;
    input.pen_lift_height    = fresh.pen_lift_height;
    input.mirror_x           = fresh.mirror_x;
    input.mirror_topological = fresh.mirror_topological;
    input.fast_normals       = fresh.fast_normals;
    input.paint_visible      = fresh.paint_visible;
    input.clay_melt          = fresh.clay_melt;
    input.crease_dot         = fresh.crease_dot;
    input.crease_build       = fresh.crease_build;
    input.stipple            = fresh.stipple;
    input.paint_stipple      = fresh.paint_stipple;
    input.density_coarse_mult = fresh.density_coarse_mult;
    input.density_fine_mult   = fresh.density_fine_mult;
    input.active_alpha       = fresh.active_alpha;
    input.fill_pattern       = fresh.fill_pattern;
    input.fill_angle         = fresh.fill_angle;
    input.bucket_paint_walls = fresh.bucket_paint_walls;
    for (int i = 0; i < (int)KeyAction::COUNT; i++) input.keybinds[i] = fresh.keybinds[i];
    for (int i = 0; i < 3; i++) {
        input.paint_color[i]     = fresh.paint_color[i];
        input.paint_color_alt[i] = fresh.paint_color_alt[i];
        input.unselected_tint[i] = fresh.unselected_tint[i];
    }
    // Brush feel back to defaults. sync_live_settings,
    // not current_brush: reset can fire while a smooth lock is held (or Shift is down).
    const InputState& p = fresh;
    input.max_effect = p.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) input.per_brush[i] = p.per_brush[i];
    input.sync_live_settings();

    storage_clear();
    g_last_blob = serialize(input);
    g_dirty = false;
}

bool settings_storage_available() { return g_storage_ok; }

// The folder settings.cfg lives in (made if missing), for the other files that sit
// beside it: the recent-files list and the autosave folder. Empty on web, or when
// there is no home dir to put it in.
std::string settings_dir() {
#ifdef __EMSCRIPTEN__
    return std::string();
#else
    std::string p = config_path(true);
    if (p.empty()) return p;
    return p.substr(0, p.find_last_of("/\\"));
#endif
}
