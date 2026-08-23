#include "settings.h"
#include "input.h"

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
    "move", "limb", "smooth", "mask", "paint"
};
static_assert((int)BrushType::COUNT == 10, "kBrushKey is out of sync with BrushType");

const char* kProfileKey[(int)InputProfile::COUNT] = { "mouse", "tablet" };

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

void append_profile(std::string& s, const ProfileSettings& p, const char* name) {
    s += "\n[";
    s += name;
    s += "]\n";
    append_kv(s, "max_effect", p.max_effect);
    for (int i = 0; i < (int)BrushType::COUNT; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "brush.%s=%.6g %.6g %.6g\n", kBrushKey[i],
                      (double)p.per_brush[i].strength,
                      (double)p.per_brush[i].hardness,
                      (double)p.per_brush[i].spacing);
        s += buf;
    }
}

// Caller must have flushed the active profile first.
std::string serialize(const InputState& in) {
    std::string s;
    s.reserve(2048);
    s += "# Chisel settings. Regenerated on change; unknown keys are ignored.\n";
    append_kv(s, "version", kFormatVersion);
    s += "active_profile=";
    s += kProfileKey[(int)in.active_profile];
    s += "\n";

    s += "\n[global]\n";
    // brush_size is deliberately absent — see InputState::brush_size. Because the
    // change check is "serialize and compare", leaving it out also means dragging the
    // size slider no longer marks settings dirty at all.
    append_kv(s, "per_brush_sizes",     in.per_brush_sizes);
    append_kv(s, "autosmooth",          in.autosmooth);
    append_kv(s, "matcap_contrast",     in.matcap_contrast);
    append_kv(s, "flat_shading",        in.flat_shading);
    append_kv(s, "show_fps",            in.show_fps);
    append_kv(s, "camera_perspective",  in.camera_perspective);
    append_kv(s, "camera_fov",          in.camera_fov);
    append_kv(s, "mirror_x",            in.mirror_x);
    append_kv(s, "fast_normals",        in.fast_normals);
    append_kv(s, "paint_visible",       in.paint_visible);
    append_kv(s, "clay_melt",           in.clay_melt);
    append_kv(s, "density_coarse_mult", in.density_coarse_mult);
    append_kv(s, "density_fine_mult",   in.density_fine_mult);
    if (in.active_alpha >= 0 && in.active_alpha < kBuiltinAlphaCount)
        append_kv(s, "active_alpha", in.active_alpha);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "paint_color=%.6g %.6g %.6g\n",
                  (double)in.paint_color[0], (double)in.paint_color[1],
                  (double)in.paint_color[2]);
    s += buf;
    std::snprintf(buf, sizeof(buf), "paint_color_alt=%.6g %.6g %.6g\n",
                  (double)in.paint_color_alt[0], (double)in.paint_color_alt[1],
                  (double)in.paint_color_alt[2]);
    s += buf;

    for (int i = 0; i < (int)InputProfile::COUNT; i++)
        append_profile(s, in.profiles[i], kProfileKey[i]);
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

void apply_profile_key(ProfileSettings& p, const std::string& key, const std::string& val) {
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
    // (the value is dropped, and the next write stops emitting it).
    if      (key == "matcap_contrast")     in.matcap_contrast     = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "clay_melt")           in.clay_melt           = clampf(std::strtof(val.c_str(), nullptr), 0.0f, 1.0f);
    else if (key == "density_coarse_mult") in.density_coarse_mult = clampf(std::strtof(val.c_str(), nullptr), 1.0f, 4.0f);
    else if (key == "density_fine_mult")   in.density_fine_mult   = clampf(std::strtof(val.c_str(), nullptr), 0.2f, 1.0f);
    else if (key == "per_brush_sizes")     in.per_brush_sizes     = parse_bool(val);
    else if (key == "autosmooth")          in.autosmooth          = parse_bool(val);
    else if (key == "camera_fov")          in.camera_fov          = clampf(std::strtof(val.c_str(), nullptr), 15.0f, 80.0f);
    else if (key == "flat_shading")        in.flat_shading        = parse_bool(val);
    else if (key == "show_fps")            in.show_fps            = parse_bool(val);
    else if (key == "camera_perspective")  in.camera_perspective  = parse_bool(val);
    else if (key == "mirror_x")            in.mirror_x            = parse_bool(val);
    else if (key == "fast_normals")        in.fast_normals        = parse_bool(val);
    else if (key == "paint_visible")       in.paint_visible       = parse_bool(val);
    else if (key == "paint_color")         parse_rgb(val, in.paint_color);
    else if (key == "paint_color_alt")     parse_rgb(val, in.paint_color_alt);
    else if (key == "active_alpha") {
        int a = (int)std::strtol(val.c_str(), nullptr, 10);
        if (a >= 0 && a < kBuiltinAlphaCount) in.active_alpha = a;
    }
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Applies over whatever `in` already holds, so anything the blob omits keeps its
// constructor default. Section-scoped: -1 = header area, -2 = [global], >=0 = a profile.
void deserialize(const std::string& blob, InputState& in) {
    int section = -1;
    size_t pos = 0;
    std::string pending_profile;   // active_profile arrives before the sections do

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
            else for (int i = 0; i < (int)InputProfile::COUNT; i++)
                if (name == kProfileKey[i]) { section = i; break; }
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (section == -2)      apply_global_key(in, key, val);
        else if (section >= 0)  apply_profile_key(in.profiles[section], key, val);
        else if (key == "active_profile") pending_profile = val;
        // `version` is read but unused at v1 — it exists so a future format break has a
        // discriminator to branch on instead of guessing from which keys are present.
    }

    // Load the requested profile into the live fields LAST, after every [section] has
    // populated profiles[]. Assigning active_profile directly (not via switch_profile)
    // because the live fields are still constructor defaults, not the other profile's —
    // a switch here would flush those defaults over a profile we just parsed.
    in.active_profile = InputProfile::MOUSE;
    for (int i = 0; i < (int)InputProfile::COUNT; i++)
        if (pending_profile == kProfileKey[i]) in.active_profile = (InputProfile)i;

    const ProfileSettings& p = in.profiles[(int)in.active_profile];
    in.max_effect = p.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) in.per_brush[i] = p.per_brush[i];
    in.sync_live_settings();
}

} // namespace

// ---- public -----------------------------------------------------------------------

void settings_load(InputState& input) {
    std::string blob = storage_read();
    if (blob.empty()) {
        // Nothing stored yet is the normal first-run path, not a failure. Seed the
        // comparison baseline so the first tick does not write an identical blob back.
        input.flush_profile();
        g_last_blob = serialize(input);
        return;
    }
    deserialize(blob, input);
    input.flush_profile();
    g_last_blob = serialize(input);
}

void settings_save(InputState& input) {
    input.flush_profile();
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

    input.flush_profile();
    if (serialize(input) != g_last_blob) {
        g_dirty = true;
        g_since_edit = 0.0f;
    }
}

void settings_reset(InputState& input) {
    InputState fresh;   // the single source of defaults — no second copy to drift
    for (int i = 0; i < (int)InputProfile::COUNT; i++) input.profiles[i] = fresh.profiles[i];
    input.active_profile     = fresh.active_profile;
    // brush_size is session state now, so Reset leaves it where the user put it —
    // same as it leaves the current brush and the smooth lock alone.
    input.per_brush_sizes    = fresh.per_brush_sizes;
    input.seed_brush_sizes();
    input.autosmooth         = fresh.autosmooth;
    input.matcap_contrast    = fresh.matcap_contrast;
    input.flat_shading       = fresh.flat_shading;
    input.show_fps           = fresh.show_fps;
    input.camera_perspective = fresh.camera_perspective;
    input.camera_fov         = fresh.camera_fov;
    input.mirror_x           = fresh.mirror_x;
    input.fast_normals       = fresh.fast_normals;
    input.paint_visible      = fresh.paint_visible;
    input.clay_melt          = fresh.clay_melt;
    input.density_coarse_mult = fresh.density_coarse_mult;
    input.density_fine_mult   = fresh.density_fine_mult;
    input.active_alpha       = fresh.active_alpha;
    for (int i = 0; i < 3; i++) {
        input.paint_color[i]     = fresh.paint_color[i];
        input.paint_color_alt[i] = fresh.paint_color_alt[i];
    }
    // Live brush fields come back from the restored active profile. sync_live_settings,
    // not current_brush: reset can fire while a smooth lock is held (or Shift is down).
    const ProfileSettings& p = input.profiles[(int)input.active_profile];
    input.max_effect = p.max_effect;
    for (int i = 0; i < (int)BrushType::COUNT; i++) input.per_brush[i] = p.per_brush[i];
    input.sync_live_settings();

    storage_clear();
    g_last_blob = serialize(input);
    g_dirty = false;
}

bool settings_storage_available() { return g_storage_ok; }
