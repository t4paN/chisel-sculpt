#include "tablet.h"

#if defined(__linux__) && !defined(CHISEL_NO_TABLET)

#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <dlfcn.h>
#include <cstring>
#include <vector>

namespace {
// The handful of libXi entry points we need, dlsym'd at runtime so the build
// links nothing from libxi-dev. Signatures match <X11/extensions/XInput2.h>.
typedef Status        (*PFN_XIQueryVersion)(Display*, int*, int*);
typedef XIDeviceInfo* (*PFN_XIQueryDevice)(Display*, int, int*);
typedef void          (*PFN_XIFreeDeviceInfo)(XIDeviceInfo*);
typedef int           (*PFN_XISelectEvents)(Display*, Window, XIEventMask*, int);
}

struct Tablet::Impl {
    void*    xi_lib    = nullptr;
    Display* dpy       = nullptr;   // our own connection (isolated from GLFW's)
    int      xi_opcode = -1;

    PFN_XIQueryVersion   QueryVersion   = nullptr;
    PFN_XIQueryDevice    QueryDevice    = nullptr;
    PFN_XIFreeDeviceInfo FreeDeviceInfo = nullptr;
    PFN_XISelectEvents   SelectEvents   = nullptr;

    // One entry per pressure-capable tool (Wacom reports stylus + eraser).
    struct Pen { int sourceid; int axis; double min, max; };
    std::vector<Pen> pens;
    bool  has_device    = false;
    float last_pressure = 0.0f;
    int   last_sourceid = -1;       // tool the raw stream last attributed motion to

    bool scan_devices();
    void query_pressure();          // read live valuator state (grab-independent)
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

Tablet::Tablet() {}
Tablet::~Tablet() { shutdown(); }

bool Tablet::init() {
    shutdown();
    Impl* m = new Impl();

    m->xi_lib = dlopen("libXi.so.6", RTLD_NOW | RTLD_LOCAL);
    if (!m->xi_lib) m->xi_lib = dlopen("libXi.so", RTLD_NOW | RTLD_LOCAL);
    if (!m->xi_lib) { delete m; return false; }

    m->QueryVersion   = (PFN_XIQueryVersion)  dlsym(m->xi_lib, "XIQueryVersion");
    m->QueryDevice    = (PFN_XIQueryDevice)   dlsym(m->xi_lib, "XIQueryDevice");
    m->FreeDeviceInfo = (PFN_XIFreeDeviceInfo)dlsym(m->xi_lib, "XIFreeDeviceInfo");
    m->SelectEvents   = (PFN_XISelectEvents)  dlsym(m->xi_lib, "XISelectEvents");
    if (!m->QueryVersion || !m->QueryDevice || !m->FreeDeviceInfo || !m->SelectEvents) {
        dlclose(m->xi_lib); delete m; return false;
    }

    m->dpy = XOpenDisplay(nullptr);
    if (!m->dpy) { dlclose(m->xi_lib); delete m; return false; }

    int ev = 0, err = 0;
    if (!XQueryExtension(m->dpy, "XInputExtension", &m->xi_opcode, &ev, &err)) {
        XCloseDisplay(m->dpy); dlclose(m->xi_lib); delete m; return false;
    }
    int major = 2, minor = 0;
    if (m->QueryVersion(m->dpy, &major, &minor) != Success) {
        XCloseDisplay(m->dpy); dlclose(m->xi_lib); delete m; return false;
    }

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

    impl = m;                 // keep the connection even with no device yet (hotplug)
    return m->has_device;
}

void Tablet::poll(bool stroke_active) {
    if (!impl) return;
    Impl* m = impl;
    while (XPending(m->dpy)) {
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
    if (stroke_active) m->query_pressure();
}

float Tablet::pressure() const {
    return (impl && impl->has_device) ? impl->last_pressure : 1.0f;
}

bool Tablet::available() const {
    return impl && impl->has_device;
}

void Tablet::shutdown() {
    if (!impl) return;
    if (impl->dpy)    XCloseDisplay(impl->dpy);
    if (impl->xi_lib) dlclose(impl->xi_lib);
    delete impl;
    impl = nullptr;
}

#else  // ---- non-Linux / tablet disabled: no-op stub ----

Tablet::Tablet() {}
Tablet::~Tablet() {}
bool  Tablet::init()            { return false; }
void  Tablet::poll(bool)        {}
float Tablet::pressure() const  { return 1.0f; }
bool  Tablet::available() const { return false; }
void  Tablet::shutdown()        {}

#endif
