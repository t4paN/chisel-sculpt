// Stage 2 Step 1 probe: prove wgpu-native links and can reach a GPU adapter.
#include <webgpu/webgpu.h>
#include <cstdio>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

struct AdapterResult { WGPUAdapter adapter = nullptr; bool done = false; bool ok = false; };

static void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                      WGPUStringView message, void* ud1, void* /*ud2*/) {
    AdapterResult* r = static_cast<AdapterResult*>(ud1);
    r->done = true;
    if (status == WGPURequestAdapterStatus_Success) {
        r->adapter = adapter;
        r->ok = true;
    } else {
        std::printf("[probe] adapter request failed: status=%d msg=%.*s\n",
                    (int)status, (int)message.length, message.data ? message.data : "");
    }
}

static void printSV(const char* label, WGPUStringView sv) {
    std::printf("  %-12s: %.*s\n", label, (int)sv.length, sv.data ? sv.data : "");
}

int main() {
    WGPUInstance instance = wgpuCreateInstance(nullptr);
    if (!instance) { std::printf("[probe] wgpuCreateInstance failed\n"); return 1; }
    std::printf("[probe] instance created\n");

    AdapterResult res;
    WGPURequestAdapterCallbackInfo cb = {};
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = onAdapter;
    cb.userdata1 = &res;

    wgpuInstanceRequestAdapter(instance, nullptr, cb);
    for (int i = 0; i < 100 && !res.done; ++i) {
        wgpuInstanceProcessEvents(instance);
#ifdef __EMSCRIPTEN__
        // On the web, requestAdapter is a JS promise that only resolves once
        // control returns to the browser event loop. ASYNCIFY (-sASYNCIFY) makes
        // emscripten_sleep unwind to the loop, let the promise + ProcessEvents
        // deliver the callback, then rewind — without this the busy-wait never
        // yields, res.done stays false, and we release the instance while the
        // request is still pending ("external Instance reference no longer exists").
        emscripten_sleep(10);
#endif
    }

    if (!res.ok) { std::printf("[probe] no adapter available\n"); wgpuInstanceRelease(instance); return 2; }

    WGPUAdapterInfo info = {};
    if (wgpuAdapterGetInfo(res.adapter, &info) == WGPUStatus_Success) {
        std::printf("[probe] adapter:\n");
        printSV("device", info.device);
        printSV("vendor", info.vendor);
        printSV("architecture", info.architecture);
        printSV("description", info.description);
        std::printf("  backendType : %d  adapterType : %d\n", (int)info.backendType, (int)info.adapterType);
        wgpuAdapterInfoFreeMembers(info);
    } else {
        std::printf("[probe] wgpuAdapterGetInfo failed\n");
    }

    wgpuAdapterRelease(res.adapter);
    wgpuInstanceRelease(instance);
    std::printf("[probe] OK\n");
    return 0;
}
