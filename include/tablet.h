#pragma once

// Pen/tablet pressure input.
//
// Linux/X11: reads the stylus "Abs Pressure" valuator straight off XInput2 — the
// same layer Krita/Blender/GIMP read (they go through Qt/GTK; we talk to libXi
// directly because GLFW has no tablet support). libXi is dlopen'd at runtime, so
// there is NO build- or run-time hard dependency on libxi-dev; libXi.so.6 ships on
// every X11 desktop. xf86-input-wacom's pressure curve is applied driver-side, so
// the values we get are already curved to the user's Krita/xsetwacom tuning.
//
// Other platforms / no tablet / lib missing: degrades to a no-op — pressure()
// returns 1.0 and available() returns false, so the brush behaves exactly as a
// mouse-only build.
struct Tablet {
    Tablet();
    ~Tablet();

    bool  init();              // true if a pressure-capable device was found
    // Drain pending raw events. During a stroke the button-down pointer grab
    // starves the root raw selection, so pass stroke_active=true to also query
    // the device's live valuator state directly (grab-independent).
    void  poll(bool stroke_active = false);
    float pressure() const;    // latest sample in [0,1], or 1.0 when unavailable
    bool  available() const;   // a pressure device is present
    void  shutdown();

    Tablet(const Tablet&) = delete;
    Tablet& operator=(const Tablet&) = delete;

private:
    struct Impl;
    Impl* impl = nullptr;      // nullptr on unsupported platforms / failed init
};
