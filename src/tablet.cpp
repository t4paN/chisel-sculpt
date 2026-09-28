#include "tablet.h"

#if defined(__EMSCRIPTEN__)

// ---- Web: pen pressure via PointerEvent ----
//
// Must come before the __linux__ branch: the Emscripten target has no X11 to dlopen.
//
// There is no device to open and nothing to drain — the browser delivers pressure on the
// same pointermove/pointerdown events the app already listens to for cursor position, so
// the DOM pushes into chisel_set_pen and poll() is a no-op.
//
// pointerType is the discriminator, NOT pressure: a mouse reports a constant 0.5 while
// held (0 while up), which is indistinguishable from a pen at half force. Availability
// therefore latches on the first 'pen' event and stays true — a stylus that has touched
// the tablet once is still present while the user's hand is back on the mouse.
//
// init() returns false because at startup nothing has touched the tablet yet. That is
// not a failure: the main loop already polls available() every frame to catch native
// hotplug, so the first pen contact lights up the same "tablet connected" path for free.

#include <emscripten.h>

namespace {
bool          g_web_pen_seen  = false;
float         g_web_pressure  = 1.0f;
unsigned long g_web_pen_count = 0;
}

extern "C" EMSCRIPTEN_KEEPALIVE void chisel_set_pen(int is_pen, double pressure) {
    if (!is_pen) return;               // mouse/touch events carry no usable force
    g_web_pen_seen = true;
    g_web_pen_count++;
    float p = (float)pressure;
    g_web_pressure = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
}

struct Tablet::Impl { int unused; };

Tablet::Tablet() {}
Tablet::~Tablet() {}
bool  Tablet::init(GLFWwindow*)  { return false; }
void  Tablet::poll(bool)        {}
float Tablet::pressure() const  { return g_web_pen_seen ? g_web_pressure : 1.0f; }
bool  Tablet::available() const { return g_web_pen_seen; }
void  Tablet::shutdown()        {}
unsigned long Tablet::sample_count() const { return g_web_pen_count; }

#elif defined(__linux__) && !defined(CHISEL_NO_TABLET)

#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <dlfcn.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(CHISEL_WAYLAND_TABLET)
#include <GLFW/glfw3.h>
#if GLFW_VERSION_MAJOR > 3 || GLFW_VERSION_MINOR >= 4     // glfwGetPlatform
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#include <wayland-cursor.h>
#include <linux/input-event-codes.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cstdlib>
#include "tablet-unstable-v2-client-protocol.h"
#define CHISEL_WL_TABLET 1
#endif
#endif

// Independent pressure sources, any of which may be absent (the third, the native-
// Wayland tablet protocol, is described below; it also supplies position):
//
//  - XInput2 (X11, and XWayland): the stylus "Abs Pressure" valuator, as below.
//  - hidraw (any Linux session, native Wayland included): the tablet's own USB reports,
//    read directly. This exists for OpenTabletDriver users. OTD's relative ("mouse") mode
//    hands the desktop a plain virtual MOUSE, so no window system ever sees pressure —
//    but Linux lets several readers share a hidraw node, so we read the same reports
//    OTD is reading. It also covers native-Wayland GLFW, which has no tablet support.
//
// The hidraw source only switches on when a node is actually readable by the user. With
// the stock kernel wacom driver the nodes are root-only, so on a normal X11 or Wayland
// desktop it opens nothing and the XInput2 path behaves exactly as before. OTD's udev
// rules are what grant access, so in practice "readable" means "OTD owns this tablet".
//
// Report format = Wacom's "Intuos/Bamboo" protocol (OTD's IntuosReportParser, the
// kernel's wacom_bpt_pen), verified byte-for-byte on a One by Wacom CTL-672:
//   [0] 0x02 report id   [1] 0x80 valid | 0x20 in range | 0x01 tip
//   [2..5] x, y          [6..7] pressure, little-endian   [8] hover distance
// Out of range the tablet sends 02 80 00...; anything else (other report ids, other
// lengths, other vendors' formats) is ignored, so an unknown device is a silent no-op.
namespace {
const int kHidMaxFds      = 4;     // a Wacom exposes 2 hidraw interfaces; slack for two
const int kHidReportLen   = 10;    // id + 9 payload bytes, per the report descriptor
const int kHidDefaultMax  = 2047;  // One by Wacom / Intuos S-M; grows if exceeded
const double kHidRescanSec = 2.0;  // hotplug: look again this often while none is open

// The handful of libXi entry points we need, dlsym'd at runtime so the build
// links nothing from libxi-dev. Signatures match <X11/extensions/XInput2.h>.
typedef Status        (*PFN_XIQueryVersion)(Display*, int*, int*);
typedef XIDeviceInfo* (*PFN_XIQueryDevice)(Display*, int, int*);
typedef void          (*PFN_XIFreeDeviceInfo)(XIDeviceInfo*);
typedef int           (*PFN_XISelectEvents)(Display*, Window, XIEventMask*, int);

double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}

// ---- Native Wayland: the compositor's tablet protocol ----
//
// GLFW binds no tablet protocol, and KWin 6.7 no longer converts pen input into pointer
// input for clients that don't (tablet emulation is gated behind the deprecated
// KWIN_WAYLAND_EMULATE_TABLET env var). So under a real pen device — OTD Artist mode, or
// the "Relative Pen Mode" plugin — a native-Wayland Chisel got no motion and no clicks at
// all. This source binds zwp_tablet_manager_v2 on GLFW's own wl_display and replays the
// pen into the window's GLFW callbacks, so the app and ImGui can't tell it from a mouse.
//
// Everything lives on a PRIVATE event queue. GLFW's glfwPollEvents still reads the socket
// (that distributes our events into our queue), and Tablet::poll — called right after
// it — dispatches them. So the listeners never run inside GLFW's own dispatch, and pen
// events land at the same point in the frame as mouse events.
//
// Compiled only against GLFW 3.4+ with wayland-client found at configure time. The Linux
// CI image ships GLFW 3.3 (X11 only), where the AppImage runs through XWayland — which
// already turns the pen into core pointer events itself.
#if defined(CHISEL_WL_TABLET)
namespace {
struct WlPen {
    GLFWwindow*            win     = nullptr;
    wl_surface*            surface = nullptr;   // the window's; pen events elsewhere are ignored
    wl_display*            dpy     = nullptr;   // GLFW's; not ours to close
    wl_event_queue*        queue   = nullptr;
    wl_registry*           reg     = nullptr;
    wl_compositor*         comp    = nullptr;
    wl_shm*                shm     = nullptr;
    wl_seat*               seat    = nullptr;
    zwp_tablet_manager_v2* mgr     = nullptr;
    zwp_tablet_seat_v2*    tseat   = nullptr;
    std::vector<zwp_tablet_tool_v2*> tools;
    std::vector<zwp_tablet_v2*>      tablets;

    // A tablet tool has its own cursor, separate from the pointer's, and it is blank
    // until the client sets one. Hidden while GLFW's cursor is (Chisel draws its ring),
    // the theme arrow over UI. Unlike the pointer there is no GLFW cursor to reuse.
    wl_cursor_theme* theme = nullptr;
    wl_surface*      arrow = nullptr;
    int arrow_hx = 0, arrow_hy = 0;
    int cursor_applied = -1;             // 1 arrow, 0 hidden, -1 unknown

    // The tool currently in proximity over our surface. Only one pen at a time.
    zwp_tablet_tool_v2* active = nullptr;
    uint32_t prox_serial = 0;
    double   x = 0.0, y = 0.0;
    bool     held_right = false, held_middle = false;

    // Tool events arrive as a batch closed by `frame`; they're collected here and
    // replayed in a fixed order (enter, move, press, release, leave) at the frame.
    bool     f_in = false, f_out = false, f_motion = false, f_down = false, f_up = false;
    zwp_tablet_tool_v2* f_tool = nullptr;
    struct Btn { int button; bool pressed; };
    Btn      f_btn[4];
    int      f_nbtn = 0;

    // Slider drags capture the POINTER (GLFW_CURSOR_DISABLED), which a pen isn't bound
    // by. The drag code measures deltas from GLFW's virtual position, so while it runs
    // the pen is replayed as that position plus the pen's travel since the drag began —
    // otherwise its first event would be a jump of the gap between the two.
    bool     drag = false;
    double   drag_vx = 0, drag_vy = 0;       // GLFW's virtual position at drag start
    double   drag_lx = 0, drag_ly = 0;       // last pen position folded into the sum
    double   drag_sx = 0, drag_sy = 0;       // pen travel since drag start, all sources
    double   ex = 0, ey = 0;                 // last position handed to the app

    // Pen cursor lock, from the OTD "Relative Pen Mode" plugin — the pen's answer to the
    // mouse's pointer lock, which the tablet protocol lacks: while held, the plugin
    // freezes the pen's cursor and sends its motion here instead, so a slider drag
    // neither moves the cursor nor stops at the screen edge. Contract in
    // ~/Projects/CHISEL/otd-relative-pen-plugin.md ("Contract for apps"). No plugin =
    // no socket = the drag works as before, the cursor just travels.
    enum class Lock { NONE, REQUESTED, HELD, RELEASING };
    int      lock_fd = -1;
    Lock     lock    = Lock::NONE;

    bool          seen     = false;      // a pen has been over our window
    float         pressure = 0.0f;
    unsigned long samples  = 0;
};

// GLFW has no getter for a callback. Setting one returns the previous, so swap and
// restore. Whatever is installed is called: ImGui chains in front of the app's own, so
// both get the pen.
void wl_emit_pos(GLFWwindow* w, double x, double y) {
    GLFWcursorposfun cb = glfwSetCursorPosCallback(w, nullptr);
    glfwSetCursorPosCallback(w, cb);
    if (cb) cb(w, x, y);
}
void wl_emit_enter(GLFWwindow* w, int entered) {
    GLFWcursorenterfun cb = glfwSetCursorEnterCallback(w, nullptr);
    glfwSetCursorEnterCallback(w, cb);
    if (cb) cb(w, entered);
}
void wl_emit_button(GLFWwindow* w, int button, int action) {
    // The pen carries no modifier state; the keyboard's is what a mouse click would get.
    int mods = 0;
    auto down = [w](int k) { return glfwGetKey(w, k) == GLFW_PRESS; };
    if (down(GLFW_KEY_LEFT_SHIFT)   || down(GLFW_KEY_RIGHT_SHIFT))   mods |= GLFW_MOD_SHIFT;
    if (down(GLFW_KEY_LEFT_CONTROL) || down(GLFW_KEY_RIGHT_CONTROL)) mods |= GLFW_MOD_CONTROL;
    if (down(GLFW_KEY_LEFT_ALT)     || down(GLFW_KEY_RIGHT_ALT))     mods |= GLFW_MOD_ALT;
    if (down(GLFW_KEY_LEFT_SUPER)   || down(GLFW_KEY_RIGHT_SUPER))   mods |= GLFW_MOD_SUPER;
    GLFWmousebuttonfun cb = glfwSetMouseButtonCallback(w, nullptr);
    glfwSetMouseButtonCallback(w, cb);
    if (cb) cb(w, button, action, mods);
}

// ---- pen cursor lock (plugin socket) ----
void lock_close(WlPen* p) {
    if (p->lock_fd >= 0) close(p->lock_fd);
    p->lock_fd = -1;
    p->lock = WlPen::Lock::NONE;
}

bool lock_send(WlPen* p, const char* msg) {
    if (p->lock_fd < 0) return false;
    // MSG_NOSIGNAL: a daemon that restarted must not SIGPIPE the app.
    if (send(p->lock_fd, msg, std::strlen(msg), MSG_NOSIGNAL) < 0) { lock_close(p); return false; }
    return true;
}

// Connected lazily at the first pen drag and kept open, as the contract asks. A failed
// connect (plugin not loaded) is retried at the next drag — one syscall per drag.
bool lock_connect(WlPen* p) {
    if (p->lock_fd >= 0) return true;
    const char* dir = getenv("XDG_RUNTIME_DIR");
    if (!dir) return false;
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s/otd-relative-pen.sock", dir)
        >= (int)sizeof(addr.sun_path)) return false;
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return false;
    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return false; }
    p->lock_fd = fd;
    return true;
}

void drag_emit(WlPen* p) {
    wl_emit_pos(p->win, p->drag_vx + p->drag_sx, p->drag_vy + p->drag_sy);
}

// Drain the socket. Called every frame from Tablet::poll; replies arrive a few ms after
// a request, so a drag waits a frame or two for `locked` and moves by Wayland motion
// in between, which the contract allows.
void lock_poll(WlPen* p) {
    char buf[128];
    while (p->lock_fd >= 0) {
        ssize_t n = recv(p->lock_fd, buf, sizeof(buf) - 1, MSG_DONTWAIT);
        if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
            // Daemon gone. Closing is also how a lock is released, so nothing is stuck.
            lock_close(p);
            return;
        }
        if (n < 0) return;
        buf[n] = 0;
        while (n > 0 && (buf[n - 1] == ' ' || buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;

        if (!std::strncmp(buf, "motion ", 7)) {
            // Only the held lock's motion counts: after `unlock` the contract says to
            // ignore motion until `unlocked`, and it can't belong to a drag that's over.
            if (p->lock != WlPen::Lock::HELD || !p->drag) continue;
            char* end = nullptr;
            double dx = std::strtod(buf + 7, &end);     // C locale: Chisel never sets one
            double dy = std::strtod(end, nullptr);
            p->drag_sx += dx;
            p->drag_sy += dy;
            drag_emit(p);
        } else if (!std::strcmp(buf, "locked")) {
            if (p->lock == WlPen::Lock::REQUESTED) p->lock = WlPen::Lock::HELD;
            std::printf("[tablet] pen cursor lock: locked\n");
            std::fflush(stdout);
        } else if (!std::strncmp(buf, "denied", 6) || !std::strncmp(buf, "unlocked", 8)) {
            // Denied, or ended by the plugin (pen lifted, timeout, turned off): the drag, if
            // still running, carries on from Wayland motion. The cursor thaws where it froze,
            // which is where drag_lx/ly were last left, so no jump.
            p->lock = WlPen::Lock::NONE;
            std::printf("[tablet] pen cursor lock: %s\n", buf);
            std::fflush(stdout);
        }
        // `hello 1` and anything unknown: nothing to do.
    }
}

// A slider drag began (GLFW's cursor went DISABLED) with the pen over the window. Mouse
// drags never get here: the pen has to be in proximity, and nobody holds both.
void wl_begin_drag(WlPen* p) {
    p->drag = true;
    glfwGetCursorPos(p->win, &p->drag_vx, &p->drag_vy);
    // Anchor on the last position the app saw, not this event's: that's where the drag
    // began, and this event's own motion then counts instead of being eaten.
    p->drag_lx = p->ex;
    p->drag_ly = p->ey;
    p->drag_sx = p->drag_sy = 0.0;
    if (p->lock == WlPen::Lock::NONE && lock_connect(p) && lock_send(p, "lock"))
        p->lock = WlPen::Lock::REQUESTED;
}

void wl_end_drag(WlPen* p) {
    p->drag = false;
    if (p->lock == WlPen::Lock::REQUESTED || p->lock == WlPen::Lock::HELD) {
        if (lock_send(p, "unlock")) p->lock = WlPen::Lock::RELEASING;
    }
}

void wl_move(WlPen* p) {
    if (glfwGetInputMode(p->win, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
        if (!p->drag) wl_begin_drag(p);
        // Wayland motion still counts until the lock lands (and after it lapses); while
        // it's held the cursor is frozen, so this adds nothing.
        p->drag_sx += p->x - p->drag_lx;
        p->drag_sy += p->y - p->drag_ly;
        p->drag_lx = p->x;
        p->drag_ly = p->y;
        drag_emit(p);
        return;
    }
    if (p->drag) wl_end_drag(p);
    p->ex = p->x;
    p->ey = p->y;
    wl_emit_pos(p->win, p->x, p->y);
}

void wl_apply_cursor(WlPen* p) {
    if (!p->active) return;
    int want = glfwGetInputMode(p->win, GLFW_CURSOR) == GLFW_CURSOR_NORMAL && p->arrow ? 1 : 0;
    if (want == p->cursor_applied) return;
    p->cursor_applied = want;
    zwp_tablet_tool_v2_set_cursor(p->active, p->prox_serial, want ? p->arrow : nullptr,
                                  p->arrow_hx, p->arrow_hy);
    wl_display_flush(p->dpy);
}

// ---- tool ----
void tool_type(void*, zwp_tablet_tool_v2*, uint32_t) {}
void tool_hw_serial(void*, zwp_tablet_tool_v2*, uint32_t, uint32_t) {}
void tool_hw_id(void*, zwp_tablet_tool_v2*, uint32_t, uint32_t) {}
void tool_capability(void*, zwp_tablet_tool_v2*, uint32_t) {}
void tool_done(void*, zwp_tablet_tool_v2*) {}
void tool_removed(void* d, zwp_tablet_tool_v2* t) {
    WlPen* p = (WlPen*)d;
    if (p->active == t) p->active = nullptr;
    if (p->f_tool == t) p->f_tool = nullptr;
    for (size_t i = 0; i < p->tools.size(); i++)
        if (p->tools[i] == t) { p->tools.erase(p->tools.begin() + i); break; }
    zwp_tablet_tool_v2_destroy(t);
}
void tool_prox_in(void* d, zwp_tablet_tool_v2* t, uint32_t serial, zwp_tablet_v2*, wl_surface* s) {
    WlPen* p = (WlPen*)d;
    if (s != p->surface) return;         // a popup or someone else's surface
    p->f_in = true;
    p->f_tool = t;
    p->prox_serial = serial;
}
void tool_prox_out(void* d, zwp_tablet_tool_v2* t) {
    WlPen* p = (WlPen*)d;
    p->f_out = true;
    p->f_tool = t;
}
void tool_down(void* d, zwp_tablet_tool_v2*, uint32_t) { ((WlPen*)d)->f_down = true; }
void tool_up(void* d, zwp_tablet_tool_v2*)             { ((WlPen*)d)->f_up = true; }
void tool_motion(void* d, zwp_tablet_tool_v2*, wl_fixed_t x, wl_fixed_t y) {
    WlPen* p = (WlPen*)d;
    p->x = wl_fixed_to_double(x);
    p->y = wl_fixed_to_double(y);
    p->f_motion = true;
}
void tool_pressure(void* d, zwp_tablet_tool_v2*, uint32_t v) {
    ((WlPen*)d)->pressure = (float)v / 65535.0f;   // protocol range is 0..65535
}
void tool_distance(void*, zwp_tablet_tool_v2*, uint32_t) {}
void tool_tilt(void*, zwp_tablet_tool_v2*, wl_fixed_t, wl_fixed_t) {}
void tool_rotation(void*, zwp_tablet_tool_v2*, wl_fixed_t) {}
void tool_slider(void*, zwp_tablet_tool_v2*, int32_t) {}
void tool_wheel(void*, zwp_tablet_tool_v2*, wl_fixed_t, int32_t) {}
void tool_button(void* d, zwp_tablet_tool_v2*, uint32_t, uint32_t button, uint32_t state) {
    WlPen* p = (WlPen*)d;
    // Same mapping as OTD's Artist mode and the Relative Pen plugin: barrel button 1
    // (BTN_STYLUS2) is a right click, button 2 (BTN_STYLUS) a middle click.
    int b = button == BTN_STYLUS2 ? GLFW_MOUSE_BUTTON_RIGHT
          : button == BTN_STYLUS  ? GLFW_MOUSE_BUTTON_MIDDLE : -1;
    if (b < 0 || p->f_nbtn >= 4) return;
    p->f_btn[p->f_nbtn++] = { b, state == ZWP_TABLET_TOOL_V2_BUTTON_STATE_PRESSED };
}
void tool_frame(void* d, zwp_tablet_tool_v2* t, uint32_t) {
    WlPen* p = (WlPen*)d;
    if (p->f_in) {
        p->active = t;
        p->seen = true;
        p->cursor_applied = -1;
        wl_emit_enter(p->win, 1);
        wl_apply_cursor(p);
    }
    if (p->active == t) {
        if (p->f_motion) { wl_move(p); p->samples++; }
        if (p->f_down) wl_emit_button(p->win, GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS);
        for (int i = 0; i < p->f_nbtn; i++) {
            const WlPen::Btn& bt = p->f_btn[i];
            bool& held = bt.button == GLFW_MOUSE_BUTTON_RIGHT ? p->held_right : p->held_middle;
            if (held == bt.pressed) continue;
            held = bt.pressed;
            wl_emit_button(p->win, bt.button, bt.pressed ? GLFW_PRESS : GLFW_RELEASE);
        }
        if (p->f_up) wl_emit_button(p->win, GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE);
        if (p->f_out) {
            // Leaving proximity with a button held (pen whisked off mid-drag) never sends
            // the matching up, so release whatever is still down.
            if (p->held_right)  { p->held_right = false;  wl_emit_button(p->win, GLFW_MOUSE_BUTTON_RIGHT,  GLFW_RELEASE); }
            if (p->held_middle) { p->held_middle = false; wl_emit_button(p->win, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_RELEASE); }
            wl_emit_enter(p->win, 0);
            p->active = nullptr;
            p->drag = false;
        }
    }
    p->f_in = p->f_out = p->f_motion = p->f_down = p->f_up = false;
    p->f_tool = nullptr;
    p->f_nbtn = 0;
}
const zwp_tablet_tool_v2_listener kToolListener = {
    tool_type, tool_hw_serial, tool_hw_id, tool_capability, tool_done, tool_removed,
    tool_prox_in, tool_prox_out, tool_down, tool_up, tool_motion, tool_pressure,
    tool_distance, tool_tilt, tool_rotation, tool_slider, tool_wheel, tool_button, tool_frame,
};

// ---- tablet (only tracked so it can be destroyed when unplugged) ----
void tab_name(void*, zwp_tablet_v2*, const char*) {}
void tab_id(void*, zwp_tablet_v2*, uint32_t, uint32_t) {}
void tab_path(void*, zwp_tablet_v2*, const char*) {}
void tab_done(void*, zwp_tablet_v2*) {}
void tab_removed(void* d, zwp_tablet_v2* t) {
    WlPen* p = (WlPen*)d;
    for (size_t i = 0; i < p->tablets.size(); i++)
        if (p->tablets[i] == t) { p->tablets.erase(p->tablets.begin() + i); break; }
    zwp_tablet_v2_destroy(t);
}
const zwp_tablet_v2_listener kTabletListener = {
    tab_name, tab_id, tab_path, tab_done, tab_removed,
};

// ---- seat ----
void seat_tablet_added(void* d, zwp_tablet_seat_v2*, zwp_tablet_v2* t) {
    WlPen* p = (WlPen*)d;
    p->tablets.push_back(t);
    zwp_tablet_v2_add_listener(t, &kTabletListener, p);
}
void seat_tool_added(void* d, zwp_tablet_seat_v2*, zwp_tablet_tool_v2* t) {
    WlPen* p = (WlPen*)d;
    p->tools.push_back(t);
    zwp_tablet_tool_v2_add_listener(t, &kToolListener, p);
}
void seat_pad_added(void*, zwp_tablet_seat_v2*, zwp_tablet_pad_v2* pad) {
    zwp_tablet_pad_v2_destroy(pad);      // express keys / rings: not used
}
const zwp_tablet_seat_v2_listener kSeatListener = {
    seat_tablet_added, seat_tool_added, seat_pad_added,
};

// ---- registry ----
void reg_global(void* d, wl_registry* r, uint32_t name, const char* iface, uint32_t ver) {
    WlPen* p = (WlPen*)d;
    if (!std::strcmp(iface, zwp_tablet_manager_v2_interface.name))
        p->mgr = (zwp_tablet_manager_v2*)wl_registry_bind(r, name, &zwp_tablet_manager_v2_interface, 1);
    else if (!std::strcmp(iface, wl_seat_interface.name) && !p->seat)
        p->seat = (wl_seat*)wl_registry_bind(r, name, &wl_seat_interface, 1);
    else if (!std::strcmp(iface, wl_compositor_interface.name))
        p->comp = (wl_compositor*)wl_registry_bind(r, name, &wl_compositor_interface, 1);
    else if (!std::strcmp(iface, wl_shm_interface.name))
        p->shm = (wl_shm*)wl_registry_bind(r, name, &wl_shm_interface, 1);
    (void)ver;
}
void reg_global_remove(void*, wl_registry*, uint32_t) {}
const wl_registry_listener kRegListener = { reg_global, reg_global_remove };

void wl_destroy(WlPen* p) {
    lock_close(p);                       // closing releases a held lock at once
    for (zwp_tablet_tool_v2* t : p->tools)   zwp_tablet_tool_v2_destroy(t);
    for (zwp_tablet_v2* t : p->tablets)      zwp_tablet_v2_destroy(t);
    if (p->tseat) zwp_tablet_seat_v2_destroy(p->tseat);
    if (p->mgr)   zwp_tablet_manager_v2_destroy(p->mgr);
    if (p->arrow) wl_surface_destroy(p->arrow);
    if (p->theme) wl_cursor_theme_destroy(p->theme);
    if (p->seat)  wl_seat_destroy(p->seat);
    if (p->shm)   wl_shm_destroy(p->shm);
    if (p->comp)  wl_compositor_destroy(p->comp);
    if (p->reg)   wl_registry_destroy(p->reg);
    if (p->queue) wl_event_queue_destroy(p->queue);
    if (p->dpy)   wl_display_flush(p->dpy);
    delete p;
}

WlPen* wl_create(GLFWwindow* win) {
    if (!win || glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) return nullptr;
    WlPen* p = new WlPen();
    p->win     = win;
    p->dpy     = glfwGetWaylandDisplay();
    p->surface = glfwGetWaylandWindow(win);
    if (!p->dpy || !p->surface) { delete p; return nullptr; }

    p->queue = wl_display_create_queue(p->dpy);
    // Proxies inherit their factory's queue, so a wrapper registry puts every object
    // bound through it (and everything those create) on our queue.
    wl_display* wrapped = (wl_display*)wl_proxy_create_wrapper(p->dpy);
    wl_proxy_set_queue((wl_proxy*)wrapped, p->queue);
    p->reg = wl_display_get_registry(wrapped);
    wl_proxy_wrapper_destroy(wrapped);
    wl_registry_add_listener(p->reg, &kRegListener, p);
    wl_display_roundtrip_queue(p->dpy, p->queue);

    if (!p->mgr || !p->seat) {
        std::printf("[tablet] Wayland: compositor has no tablet protocol — pen via pointer only\n");
        std::fflush(stdout);
        wl_destroy(p);
        return nullptr;
    }
    p->tseat = zwp_tablet_manager_v2_get_tablet_seat(p->mgr, p->seat);
    zwp_tablet_seat_v2_add_listener(p->tseat, &kSeatListener, p);

    if (p->comp && p->shm) {
        const char* size_env = getenv("XCURSOR_SIZE");
        int size = size_env ? atoi(size_env) : 0;
        if (size <= 0) size = 24;
        p->theme = wl_cursor_theme_load(getenv("XCURSOR_THEME"), size, p->shm);
        wl_cursor* c = p->theme ? wl_cursor_theme_get_cursor(p->theme, "default") : nullptr;
        if (!c && p->theme) c = wl_cursor_theme_get_cursor(p->theme, "left_ptr");
        wl_buffer* buf = (c && c->image_count > 0) ? wl_cursor_image_get_buffer(c->images[0]) : nullptr;
        if (buf) {
            p->arrow    = wl_compositor_create_surface(p->comp);
            p->arrow_hx = (int)c->images[0]->hotspot_x;
            p->arrow_hy = (int)c->images[0]->hotspot_y;
            wl_surface_attach(p->arrow, buf, 0, 0);
            wl_surface_damage(p->arrow, 0, 0, (int)c->images[0]->width, (int)c->images[0]->height);
            wl_surface_commit(p->arrow);
        }
    }
    // Second roundtrip: the tablet seat announces the tablets and tools already present.
    wl_display_roundtrip_queue(p->dpy, p->queue);
    std::printf("[tablet] Wayland tablet protocol bound (%zu tablet(s), %zu tool(s))\n",
                p->tablets.size(), p->tools.size());
    std::fflush(stdout);
    return p;
}
}  // namespace
#endif

struct Tablet::Impl {
    void*    xi_lib    = nullptr;
    Display* dpy       = nullptr;   // our own connection (isolated from GLFW's); may be null
    int      xi_opcode = -1;

    PFN_XIQueryVersion   QueryVersion   = nullptr;
    PFN_XIQueryDevice    QueryDevice    = nullptr;
    PFN_XIFreeDeviceInfo FreeDeviceInfo = nullptr;
    PFN_XISelectEvents   SelectEvents   = nullptr;

    // One entry per pressure-capable tool (Wacom reports stylus + eraser).
    struct Pen { int sourceid; int axis; double min, max; };
    std::vector<Pen> pens;
    bool  has_device    = false;    // XInput2 found a pressure valuator
    float last_pressure = 0.0f;
    int   last_sourceid = -1;       // tool the raw stream last attributed motion to
    unsigned long samples = 0;      // bumped on any stylus motion, pressure axis or not

    // hidraw source. Latches "seen" on the first valid pen report rather than on open:
    // an open node only proves a Wacom is plugged in, not that it speaks this format.
    int    hid_fds[kHidMaxFds];
    int    hid_nfds       = 0;
    bool   hid_seen       = false;
    int    hid_max        = kHidDefaultMax;
    double hid_next_scan  = 0.0;

#if defined(CHISEL_WL_TABLET)
    WlPen* wl = nullptr;            // native-Wayland tablet protocol; null elsewhere
#endif
    // The Wayland source, once a pen has been over the window, is the authority: its
    // pressure is the one the compositor delivers after OTD's curve and filters, where
    // hidraw is the raw sensor. Both would otherwise write last_pressure.
    bool wl_seen() const {
#if defined(CHISEL_WL_TABLET)
        return wl && wl->seen;
#else
        return false;
#endif
    }

    bool init_xinput();
    bool scan_devices();
    void query_pressure();          // read live valuator state (grab-independent)
    void hid_scan();
    void hid_poll();
    void hid_close_all();
};

// Read the current pressure straight off the device's valuator state via
// XIQueryDevice. Unlike raw events, this is unaffected by the button-down pointer
// grab, so it keeps pressure live for the whole stroke. We only query the tool the
// raw stream last saw move, so a stylus+eraser pair doesn't read the idle one.
void Tablet::Impl::query_pressure() {
    if (!has_device) return;
    int target = last_sourceid;
    if (target < 0 && !pens.empty()) target = pens[0].sourceid;
    for (const auto& p : pens) {
        if (p.sourceid != target) continue;
        int ndev = 0;
        XIDeviceInfo* d = QueryDevice(dpy, p.sourceid, &ndev);
        if (!d) return;
        if (ndev >= 1) {
            for (int c = 0; c < d[0].num_classes; c++) {
                if (d[0].classes[c]->type != XIValuatorClass) continue;
                XIValuatorClassInfo* v = (XIValuatorClassInfo*)d[0].classes[c];
                if (v->number != p.axis) continue;
                double norm = (v->value - p.min) / (p.max - p.min);
                norm = norm < 0.0 ? 0.0 : (norm > 1.0 ? 1.0 : norm);
                last_pressure = (float)norm;
                break;
            }
        }
        FreeDeviceInfo(d);
        break;
    }
}

// (Re)discover which slave pointers expose an "Abs Pressure" valuator, and at
// what axis number / range. Called at init and on every XI_HierarchyChanged, so
// plugging the tablet in after launch just works.
bool Tablet::Impl::scan_devices() {
    pens.clear();
    int ndev = 0;
    XIDeviceInfo* devs = QueryDevice(dpy, XIAllDevices, &ndev);
    if (!devs) { has_device = false; return false; }

    Atom press_atom = XInternAtom(dpy, "Abs Pressure", True);
    for (int i = 0; i < ndev; i++) {
        XIDeviceInfo& d = devs[i];
        if (d.use != XISlavePointer && d.use != XIFloatingSlave) continue;
        for (int c = 0; c < d.num_classes; c++) {
            if (d.classes[c]->type != XIValuatorClass) continue;
            XIValuatorClassInfo* v = (XIValuatorClassInfo*)d.classes[c];

            bool is_pressure = (press_atom != None && v->label == press_atom);
            if (!is_pressure && v->label != None) {
                char* nm = XGetAtomName(dpy, v->label);
                if (nm) { is_pressure = (std::strcmp(nm, "Abs Pressure") == 0); XFree(nm); }
            }
            if (is_pressure) {
                double mn = v->min, mx = v->max;
                if (mx <= mn) mx = mn + 1.0;   // guard degenerate range
                pens.push_back({ d.deviceid, v->number, mn, mx });
                break;                          // one pressure axis per device
            }
        }
    }
    FreeDeviceInfo(devs);
    has_device = !pens.empty();
    return has_device;
}

// Open every readable hidraw node whose device is a Wacom (USB vendor 0x056A). A node
// we can't open (EACCES — the normal case without OTD) is skipped without a word.
void Tablet::Impl::hid_scan() {
    DIR* dir = opendir("/sys/class/hidraw");
    if (!dir) return;
    while (dirent* e = readdir(dir)) {
        if (hid_nfds >= kHidMaxFds) break;
        if (std::strncmp(e->d_name, "hidraw", 6) != 0) continue;
        char path[160];
        std::snprintf(path, sizeof(path), "/sys/class/hidraw/%s/device/uevent", e->d_name);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        bool wacom = false;
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            // HID_ID=<bus>:<vendor>:<product>, e.g. 0003:0000056A:0000037B
            if (std::strncmp(line, "HID_ID=", 7) == 0) {
                unsigned bus = 0, vendor = 0, product = 0;
                if (std::sscanf(line + 7, "%x:%x:%x", &bus, &vendor, &product) == 3)
                    wacom = (vendor == 0x056A);
                break;
            }
        }
        std::fclose(f);
        if (!wacom) continue;
        std::snprintf(path, sizeof(path), "/dev/%s", e->d_name);
        int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd >= 0) hid_fds[hid_nfds++] = fd;
    }
    closedir(dir);
}

void Tablet::Impl::hid_close_all() {
    for (int i = 0; i < hid_nfds; i++) close(hid_fds[i]);
    hid_nfds = 0;
    hid_seen = false;
}

void Tablet::Impl::hid_poll() {
    if (hid_nfds == 0) {
        double t = now_sec();
        if (t < hid_next_scan) return;
        hid_next_scan = t + kHidRescanSec;
        hid_scan();
        if (hid_nfds == 0) return;
    }
    unsigned char buf[64];
    for (int i = 0; i < hid_nfds; ) {
        ssize_t n;
        while ((n = read(hid_fds[i], buf, sizeof(buf))) > 0) {
            if (n != kHidReportLen || buf[0] != 0x02) continue;
            if (!(buf[1] & 0x80) || !(buf[1] & 0x20)) continue;  // pen out of range
            int raw = buf[6] | (buf[7] << 8);
            // Pressure resolution differs by model and is not in the descriptor (the pen
            // report sits on a vendor usage page). Start at 2047 and widen to the next
            // all-ones value if the pen ever exceeds it, so a 4095-level pen is only too
            // sensitive until its first hard press, never clipped.
            while (raw > hid_max && hid_max < 0xFFFF) hid_max = (hid_max << 1) | 1;
            hid_seen = true;
            if (wl_seen()) continue;    // still drained, but the Wayland pen speaks
            last_pressure = (float)raw / (float)hid_max;
            samples++;
        }
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
            // Unplugged (ENODEV) or otherwise dead: drop this node, keep the rest.
            close(hid_fds[i]);
            hid_fds[i] = hid_fds[--hid_nfds];
            if (hid_nfds == 0) hid_seen = false;
            continue;
        }
        i++;
    }
}

Tablet::Tablet() {}
Tablet::~Tablet() { shutdown(); }

// The two sources are set up independently: no X display (pure Wayland), no libXi, or
// no XInput2 just means the hidraw source runs alone, and vice versa.

bool Tablet::init(GLFWwindow* window) {
    shutdown();
    Impl* m = new Impl();
    if (!m->init_xinput()) {
        if (m->dpy)    { XCloseDisplay(m->dpy); m->dpy = nullptr; }
        if (m->xi_lib) { dlclose(m->xi_lib);   m->xi_lib = nullptr; }
    }
    m->hid_scan();
    m->hid_next_scan = now_sec() + kHidRescanSec;
#if defined(CHISEL_WL_TABLET)
    m->wl = wl_create(window);
#else
    (void)window;
#endif
    impl = m;                 // kept even with no device yet (hotplug on both sources)
    return available();
}

bool Tablet::Impl::init_xinput() {
    Impl* m = this;
    m->xi_lib = dlopen("libXi.so.6", RTLD_NOW | RTLD_LOCAL);
    if (!m->xi_lib) m->xi_lib = dlopen("libXi.so", RTLD_NOW | RTLD_LOCAL);
    if (!m->xi_lib) return false;

    m->QueryVersion   = (PFN_XIQueryVersion)  dlsym(m->xi_lib, "XIQueryVersion");
    m->QueryDevice    = (PFN_XIQueryDevice)   dlsym(m->xi_lib, "XIQueryDevice");
    m->FreeDeviceInfo = (PFN_XIFreeDeviceInfo)dlsym(m->xi_lib, "XIFreeDeviceInfo");
    m->SelectEvents   = (PFN_XISelectEvents)  dlsym(m->xi_lib, "XISelectEvents");
    if (!m->QueryVersion || !m->QueryDevice || !m->FreeDeviceInfo || !m->SelectEvents)
        return false;

    m->dpy = XOpenDisplay(nullptr);
    if (!m->dpy) return false;

    int ev = 0, err = 0;
    if (!XQueryExtension(m->dpy, "XInputExtension", &m->xi_opcode, &ev, &err))
        return false;
    int major = 2, minor = 0;
    if (m->QueryVersion(m->dpy, &major, &minor) != Success) return false;

    m->scan_devices();

    // Two selections on the root window, each with the deviceid the protocol
    // requires (mixing them in one mask is a BadValue):
    //   - raw motion must use XIAllMasterDevices; sourceid identifies the tool.
    //     Focus-independent, so we see pen motion while sculpting regardless of grab.
    //   - hierarchy changes must use XIAllDevices, so plugging a tablet in later works.
    Window root = DefaultRootWindow(m->dpy);
    unsigned char raw_mask[(XI_LASTEVENT + 7) / 8]  = {0};
    unsigned char hier_mask[(XI_LASTEVENT + 7) / 8] = {0};
    XISetMask(raw_mask,  XI_RawMotion);
    XISetMask(hier_mask, XI_HierarchyChanged);
    XIEventMask ems[2];
    ems[0].deviceid = XIAllMasterDevices;
    ems[0].mask_len = sizeof(raw_mask);
    ems[0].mask     = raw_mask;
    ems[1].deviceid = XIAllDevices;
    ems[1].mask_len = sizeof(hier_mask);
    ems[1].mask     = hier_mask;
    m->SelectEvents(m->dpy, root, ems, 2);
    XFlush(m->dpy);
    return true;
}

void Tablet::poll(bool stroke_active) {
    if (!impl) return;
    Impl* m = impl;
    // hidraw first: once it has delivered a pen report it is the authority on pressure,
    // and the XInput2 writes below stand down. Otherwise, with OTD in its absolute
    // "Artist" mode, XWayland also exposes the virtual pen, and the grab fallback's
    // valuator query (stale while our window is a native Wayland one) would overwrite a
    // live hidraw value every stroke frame.
#if defined(CHISEL_WL_TABLET)
    // Runs right after glfwPollEvents, which already read the socket into our queue.
    if (m->wl) {
        wl_display_dispatch_queue_pending(m->wl->dpy, m->wl->queue);
        // Drags start and end on a key, so with the pen held still there's no motion
        // event to notice either edge — check the cursor mode every frame too.
        bool disabled = glfwGetInputMode(m->wl->win, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
        if (disabled && !m->wl->drag && m->wl->active) wl_begin_drag(m->wl);
        if (!disabled && m->wl->drag) wl_end_drag(m->wl);
        lock_poll(m->wl);
        wl_apply_cursor(m->wl);         // follow GLFW's cursor mode (ring vs. UI arrow)
    }
#endif
    m->hid_poll();
    const bool xi_writes_pressure = !m->hid_seen && !m->wl_seen();

    while (m->dpy && XPending(m->dpy)) {
        XEvent ev;
        XNextEvent(m->dpy, &ev);
        if (ev.xcookie.type != GenericEvent || ev.xcookie.extension != m->xi_opcode)
            continue;
        if (!XGetEventData(m->dpy, &ev.xcookie)) continue;

        int evtype = ev.xcookie.evtype;
        if (evtype == XI_HierarchyChanged) {
            m->scan_devices();
        } else if (evtype == XI_RawMotion) {
            XIRawEvent* re = (XIRawEvent*)ev.xcookie.data;
            for (const auto& p : m->pens) {
                if (p.sourceid != re->sourceid) continue;
                m->last_sourceid = re->sourceid;
                // Before the pressure-axis checks below: a hover with no pressure
                // change is still the pen driving, and that is what this counts.
                m->samples++;
                if (!xi_writes_pressure) break;
                const unsigned char* vmask = re->valuators.mask;
                int mask_len = re->valuators.mask_len;
                // Pressure only present in events where that axis actually moved.
                if (p.axis < 0 || (p.axis >> 3) >= mask_len) break;
                if (!XIMaskIsSet(vmask, p.axis)) break;
                // raw_values is packed over the set mask bits, in axis order.
                int idx = 0;
                for (int b = 0; b < p.axis; b++)
                    if (XIMaskIsSet(vmask, b)) idx++;
                double norm = (re->raw_values[idx] - p.min) / (p.max - p.min);
                norm = norm < 0.0 ? 0.0 : (norm > 1.0 ? 1.0 : norm);
                m->last_pressure = (float)norm;
                break;
            }
        }
        XFreeEventData(m->dpy, &ev.xcookie);
    }

    // While the tip is down the pointer grab silences our raw selection, so fall
    // back to polling the live valuator state for the duration of the stroke.
    // hidraw has no grab to dodge; its reports keep coming mid-stroke.
    if (stroke_active && xi_writes_pressure) m->query_pressure();
}

float Tablet::pressure() const {
    if (!available()) return 1.0f;
#if defined(CHISEL_WL_TABLET)
    if (impl->wl_seen()) return impl->wl->pressure;
#endif
    return impl->last_pressure;
}

bool Tablet::available() const {
    return impl && (impl->has_device || impl->hid_seen || impl->wl_seen());
}

unsigned long Tablet::sample_count() const {
    if (!impl) return 0;
#if defined(CHISEL_WL_TABLET)
    if (impl->wl) return impl->samples + impl->wl->samples;
#endif
    return impl->samples;
}

void Tablet::shutdown() {
    if (!impl) return;
#if defined(CHISEL_WL_TABLET)
    if (impl->wl) wl_destroy(impl->wl);
#endif
    impl->hid_close_all();
    if (impl->dpy)    XCloseDisplay(impl->dpy);
    if (impl->xi_lib) dlclose(impl->xi_lib);
    delete impl;
    impl = nullptr;
}

#elif defined(_WIN32) && !defined(CHISEL_NO_TABLET)

// ---- Windows: pen pressure via WinTab (Wintab32.dll) ----
//
// Mirrors the Linux design: the entry points are GetProcAddress'd from the
// runtime DLL (no link against wintab32.lib, no build dependency — the DLL ships
// with every tablet driver), and a self-contained HWND_MESSAGE window owns the
// context, the Windows analogue of the Linux branch's private Display connection.
// Missing DLL / driver / device ⇒ clean no-op, identical to the stub below.
//
// Unlike X11, WinTab packets keep flowing while the mouse button is down, so the
// Linux stroke-grab workaround has no equivalent: poll() ignores stroke_active.

#include <windows.h>
#include "wintab.h"

// PACKET = pen cursor id + normalized pressure, in ascending PK-bit order.
#define PACKETDATA (PK_CURSOR | PK_NORMAL_PRESSURE)
#define PACKETMODE 0
#include "pktdef.h"

namespace {
typedef UINT (*PFN_WTInfoA)(UINT, UINT, LPVOID);
typedef HCTX (*PFN_WTOpenA)(HWND, LPLOGCONTEXTA, BOOL);
typedef BOOL (*PFN_WTClose)(HCTX);
typedef int  (*PFN_WTPacketsGet)(HCTX, int, LPVOID);
typedef BOOL (*PFN_WTEnable)(HCTX, BOOL);

const wchar_t* kMsgWinClass = L"ChiselTabletMsgWin";
}

struct Tablet::Impl {
    HMODULE wintab  = nullptr;
    HWND    msg_win = nullptr;   // own HWND_MESSAGE window (isolated from GLFW's)
    HCTX    ctx     = nullptr;

    PFN_WTInfoA       WTInfo       = nullptr;
    PFN_WTOpenA       WTOpen       = nullptr;
    PFN_WTClose       WTCloseFn    = nullptr;
    PFN_WTPacketsGet  WTPacketsGet = nullptr;
    PFN_WTEnable      WTEnableFn   = nullptr;

    double press_min = 0.0, press_max = 1.0;   // pressure axis range, from the driver
    bool   has_device    = false;
    float  last_pressure = 0.0f;
    unsigned long samples = 0;   // WinTab only queues packets for the pen, so any
                                 // packet at all means the stylus is what is moving
};

Tablet::Tablet() {}
Tablet::~Tablet() { shutdown(); }

bool Tablet::init(GLFWwindow* /*window*/) {
    shutdown();
    Impl* m = new Impl();

    m->wintab = LoadLibraryA("Wintab32.dll");
    if (!m->wintab) { delete m; return false; }

    m->WTInfo       = (PFN_WTInfoA)      GetProcAddress(m->wintab, "WTInfoA");
    m->WTOpen       = (PFN_WTOpenA)      GetProcAddress(m->wintab, "WTOpenA");
    m->WTCloseFn    = (PFN_WTClose)      GetProcAddress(m->wintab, "WTClose");
    m->WTPacketsGet = (PFN_WTPacketsGet) GetProcAddress(m->wintab, "WTPacketsGet");
    m->WTEnableFn   = (PFN_WTEnable)     GetProcAddress(m->wintab, "WTEnable");
    if (!m->WTInfo || !m->WTOpen || !m->WTCloseFn || !m->WTPacketsGet) {
        FreeLibrary(m->wintab); delete m; return false;
    }

    // Availability probe + pressure axis range (guard a degenerate range, as the
    // Linux branch guards a degenerate valuator range).
    if (m->WTInfo(0, 0, nullptr) == 0) { FreeLibrary(m->wintab); delete m; return false; }
    AXIS press = {};
    if (m->WTInfo(WTI_DEVICES, DVC_NPRESSURE, &press) == 0) {
        FreeLibrary(m->wintab); delete m; return false;   // no pressure-capable device
    }
    m->press_min = (double)press.axMin;
    m->press_max = (double)press.axMax;
    if (m->press_max <= m->press_min) m->press_max = m->press_min + 1.0;

    // Message-only window to own the context (no UI, isolated from GLFW's window).
    WNDCLASSEXW wc = {};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = kMsgWinClass;
    RegisterClassExW(&wc);   // benign if already registered
    m->msg_win = CreateWindowExW(0, kMsgWinClass, L"", 0, 0, 0, 0, 0,
                                 HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
    if (!m->msg_win) { FreeLibrary(m->wintab); delete m; return false; }

    // Open a context that delivers cursor id + normal pressure.
    LOGCONTEXTA lc = {};
    if (m->WTInfo(WTI_DEFSYSCTX, 0, &lc) == 0) {
        DestroyWindow(m->msg_win); FreeLibrary(m->wintab); delete m; return false;
    }
    lc.lcOptions |= CXO_MESSAGES;
    lc.lcPktData  = PACKETDATA;
    lc.lcPktMode  = PACKETMODE;
    lc.lcMoveMask = PACKETDATA;
    lc.lcBtnUpMask = lc.lcBtnDnMask;
    m->ctx = m->WTOpen(m->msg_win, &lc, TRUE);

    m->has_device = (m->ctx != nullptr);
    impl = m;                 // keep the lib loaded even with no context (cheap)
    return m->has_device;
}

void Tablet::poll(bool /*stroke_active*/) {
    if (!impl) return;
    Impl* m = impl;
    if (!m->ctx) return;

    // Service our hidden window so the driver fills the context queue.
    MSG msg;
    while (PeekMessageW(&msg, m->msg_win, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Drain queued packets; keep the most recent pressure sample.
    PACKET pkts[32];
    int n = m->WTPacketsGet(m->ctx, 32, pkts);
    if (n > 0) {
        m->samples += (unsigned long)n;
        double norm = ((double)pkts[n - 1].pkNormalPressure - m->press_min)
                    / (m->press_max - m->press_min);
        norm = norm < 0.0 ? 0.0 : (norm > 1.0 ? 1.0 : norm);
        m->last_pressure = (float)norm;
    }
}

float Tablet::pressure() const {
    return (impl && impl->has_device) ? impl->last_pressure : 1.0f;
}

bool Tablet::available() const {
    return impl && impl->has_device;
}

unsigned long Tablet::sample_count() const {
    return impl ? impl->samples : 0;
}

void Tablet::shutdown() {
    if (!impl) return;
    if (impl->ctx)     impl->WTCloseFn(impl->ctx);
    if (impl->msg_win) DestroyWindow(impl->msg_win);
    if (impl->wintab)  FreeLibrary(impl->wintab);
    delete impl;
    impl = nullptr;
}

#else  // ---- other platforms / tablet disabled: no-op stub ----

Tablet::Tablet() {}
Tablet::~Tablet() {}
bool  Tablet::init(GLFWwindow*)  { return false; }
void  Tablet::poll(bool)        {}
float Tablet::pressure() const  { return 1.0f; }
bool  Tablet::available() const { return false; }
void  Tablet::shutdown()        {}
unsigned long Tablet::sample_count() const { return 0; }

#endif
