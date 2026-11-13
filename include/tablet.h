#pragma once

struct GLFWwindow;

// Pen/tablet pressure input.
//
// Linux/X11: reads the stylus "Abs Pressure" valuator straight off XInput2 — the
// same layer other drawing and 3D apps read (they go through Qt/GTK; we talk to libXi
// directly because GLFW has no tablet support). libXi is dlopen'd at runtime, so
// there is NO build- or run-time hard dependency on libxi-dev; libXi.so.6 ships on
// every X11 desktop. xf86-input-wacom's pressure curve is applied driver-side, so
// the values we get are already curved to the user's xsetwacom tuning.
//
// Linux, any session (native Wayland included): also reads a Wacom's own USB reports
// through hidraw when the node is readable — which is what OpenTabletDriver's udev
// rules arrange. Covers OTD's relative mode, where the desktop only ever sees a mouse.
// See tablet.cpp for the format and why it is silent on a stock setup.
//
// Linux, native Wayland: the compositor's tablet protocol (zwp_tablet_v2), the way
// other drawing apps and browsers get the pen. Needed for POSITION, not just pressure: KWin 6.7
// stopped turning pen input into pointer input for apps that don't bind the protocol,
// and GLFW doesn't, so without this a pen never moves or clicks in the window at all.
// Pen motion/tip/buttons are fed through the window's own GLFW callbacks, so the app
// and ImGui see them exactly like mouse events.
//
// Web: PointerEvent pressure from the browser (tablet.cpp's __EMSCRIPTEN__ branch).
//
// Other platforms / no tablet / lib missing: degrades to a no-op — pressure()
// returns 1.0 and available() returns false, so the brush behaves exactly as a
// mouse-only build.
struct Tablet {
    Tablet();
    ~Tablet();

    // true if a pressure-capable device was found. The window is only used by the
    // native-Wayland source (to address its surface and callbacks); may be null.
    bool  init(GLFWwindow* window = nullptr);
    // Drain pending raw events. During a stroke the button-down pointer grab
    // starves the root raw selection, so pass stroke_active=true to also query
    // the device's live valuator state directly (grab-independent).
    void  poll(bool stroke_active = false);
    float pressure() const;    // latest sample in [0,1], or 1.0 when unavailable
    bool  available() const;   // a pressure device is present
    // Monotonic count of reports from the stylus itself. available() only says a tablet
    // is plugged in; this says the pen is what is moving *now* — the caller watches it
    // for change between frames. Needed because on X11 the pen also drives the core
    // pointer, so cursor motion alone cannot tell pen from mouse.
    unsigned long sample_count() const;
    // Hardware proximity of the pen, where the platform reports it (Windows WinTab's
    // WT_PROXIMITY): 1 hovering or touching, 0 away, -1 not known (never reported, or no
    // source). proximity_changes() counts edges, for callers that poll once a frame.
    int   proximity() const;
    unsigned long proximity_changes() const;
    void  shutdown();

    Tablet(const Tablet&) = delete;
    Tablet& operator=(const Tablet&) = delete;

    struct Impl;               // public only so the Win32 message proc can reach it
private:
    Impl* impl = nullptr;      // nullptr on unsupported platforms / failed init
};

// Windows (WinTab) only. With the driver in Mouse mode the pen's side buttons reach the
// window as emulated mouse clicks, and the driver drops some presses (never releases),
// most often right after the pen comes back into range (2026-11-12: 6 of 42 in log19).
// WinTab's own button state never misses one. So once a side button's click is known
// (learned from the first emulated click that lines up with a WinTab edge), the press
// and release come from WinTab and the driver's copies of that click are dropped.
// Called from the mouse-button callback for every middle/right click; `extra` is the
// message's GetMessageExtraInfo (the driver tags its clicks, a real mouse sends 0).
// Returns true when the event is the driver's copy of a click WinTab now delivers.
bool tablet_drop_driver_button(int glfw_button, int action, long long extra);

// Chisel's own relative pen, native. Supported where the pen's absolute position reaches
// tablet.cpp: native Wayland (the tablet protocol). Elsewhere natively, supported() is
// false and set_relative does nothing; the browser has its own (web_bridge.cpp).
// set_relative runs once a frame BEFORE Tablet::poll: gain 0 = off; mouse_drove = the
// arbiter's device is the mouse (a returning pen then carries on from the mouse's spot);
// cx/cy = the app's cursor (input.mouse_x/y). lift = the hover height (0 touching .. 1
// edge of range) above which the pen counts as gone, absolute or relative; 1 = off.
// lift_supported(): the pen reports its height (distance capability).
bool tablet_relative_supported();
bool tablet_lift_supported();
float tablet_pen_height();   // the hovering pen's height now (0..1), -1 out of range / unknown
void tablet_set_relative(float gain, float lift, bool mouse_drove, double cx, double cy);

// Windows (WinTab) only: Tablet::proximity() / proximity_changes(), current even inside
// glfwPollEvents. -1 / 0 where there is no proximity source.
int tablet_live_proximity();
unsigned long tablet_live_proximity_changes();

// Windows (WinTab) only: the position of every pen report drained by the last
// Tablet::poll(), oldest first, in the driver's own units (not pixels; main.cpp's PenPath
// learns the scale). The arrays stay valid until the next poll. Returns the count.
int tablet_packets(const long** xs, const long** ys);

// Native Wayland only: moving the pointer for edge-wrapped drags, through the
// compositor's wp_pointer_warp_v1 (GLFW's glfwSetCursorPos does nothing there). init once
// after the window exists; available() is false elsewhere and on compositors without the
// protocol. warp takes window coords and returns false if it couldn't ask. dispatch runs
// once a frame after glfwPollEvents. See tablet.cpp.
void wl_pointer_warp_init(GLFWwindow* window);
bool wl_pointer_warp_available();
bool wl_pointer_warp(double x, double y);
void wl_pointer_warp_dispatch();
