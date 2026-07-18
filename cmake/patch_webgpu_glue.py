#!/usr/bin/env python3
# Post-link hardening for the emdawnwebgpu JS glue (seen through emscripten 6.0.x).
#
# Upstream race: aborting a pending wgpuBufferMapAsync (unmap before the promise
# settles) fires a LATE rejection whose handler unconditionally deletes
# WebGPU.Internals.bufferOnUnmaps[bufferPtr]. If that pointer was freed and reused
# by a new buffer in the meantime, the new buffer's tracking entry is wiped:
# every later getMappedRange throws
#   TypeError: WebGPU.Internals.bufferOnUnmaps[bufferPtr] is undefined
# and _emwgpuBufferUnmap early-returns without calling buffer.unmap(), so the
# buffer stays mapped for the rest of the session (frame-killer until reload).
#
# The C++ side avoids the abort entirely (webgpu_backend.cpp orphan/settle sweep);
# this patch makes the glue self-heal if anything else ever hits the race: the
# mapped-range getters recreate the entry instead of throwing.
#
# Usage: patch_webgpu_glue.py <path-to-chisel.js>   (idempotent; fails loudly if
# the glue changed shape so the patch can be re-fitted instead of silently lost.)
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

GUARDED = "(WebGPU.Internals.bufferOnUnmaps[bufferPtr]??=[]).push"
if GUARDED in src:
    print(f"[glue-patch] {path}: already hardened")
    sys.exit(0)

pat = re.compile(r"WebGPU\s*\.\s*Internals\s*\.\s*bufferOnUnmaps\s*\[\s*bufferPtr\s*\]\s*\.\s*push")
out, n = pat.subn(GUARDED, src)
if n == 0:
    sys.exit(f"[glue-patch] {path}: no bufferOnUnmaps[bufferPtr].push site found — "
             "emscripten glue changed, re-fit this patch (cmake/patch_webgpu_glue.py)")

with open(path, "w", encoding="utf-8") as f:
    f.write(out)
print(f"[glue-patch] {path}: hardened {n} mapped-range push site(s)")
