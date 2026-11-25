#!/usr/bin/env bash
# Package the WebGPU web build for an itch.io HTML upload.
#
# itch's HTML player serves the zip and opens index.html at the zip root, so we
# rename chisel.html -> index.html and zip the runtime files flat (no
# subdirectory). The build is single-threaded ASYNCIFY, so there are no
# .data/.worker/.mem sidecars — just html + js glue + wasm.
#
# Two builds ride in the zip: wasm32 (build-web, 4 GB heap, every WebGPU browser) and
# wasm64 (build-web64, heap up to 16 GB, Chrome/Edge/Firefox; no Safari/WebKit). The page
# asks the browser which it can run and loads that one, so Apple users keep the wasm32 app
# and everyone else gets the bigger heap. The emitted <script src=chisel.js> is swapped
# for that picker here, not in shell.html, so a plain build-web still serves on its own.
#
# Usage:  packaging/web/pack-itch.sh [version]      (default version: v0.1.6)
#         BUILD32=dir BUILD64=dir override the build trees (e.g. from a worktree).
# Output: dist/chisel-itch-<version>.zip
set -euo pipefail

VERSION="${1:-v0.1.6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD32="${BUILD32:-$REPO/build-web}"
BUILD64="${BUILD64:-$REPO/build-web64}"
OUT="$REPO/dist"
ZIP="$OUT/chisel-itch-$VERSION.zip"

for f in "$BUILD32/chisel.html" "$BUILD32/chisel.js" "$BUILD32/chisel.wasm" \
         "$BUILD64/chisel.js" "$BUILD64/chisel.wasm"; do
  [ -f "$f" ] || { echo "missing $f — build both web targets first (build-web, build-web64)" >&2; exit 1; }
done
# Same source in both, or one browser family runs an older app than the other.
v32="$(grep -ao 'v[0-9]\+\.[0-9]\+\.[0-9]\+[-0-9a-z]*' "$BUILD32/chisel.wasm" | sort -u | tr '\n' ' ' || true)"
v64="$(grep -ao 'v[0-9]\+\.[0-9]\+\.[0-9]\+[-0-9a-z]*' "$BUILD64/chisel.wasm" | sort -u | tr '\n' ' ' || true)"
if [ "$v32" != "$v64" ]; then
  echo "build-web ($v32) and build-web64 ($v64) differ — rebuild both" >&2; exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$BUILD32/chisel.js"   "$STAGE/chisel.js"
cp "$BUILD32/chisel.wasm" "$STAGE/chisel.wasm"
cp "$BUILD64/chisel.js"   "$STAGE/chisel64.js"
cp "$BUILD64/chisel.wasm" "$STAGE/chisel64.wasm"

python3 - "$BUILD32/chisel.html" "$STAGE/index.html" <<'EOF'
import sys
src, dst = sys.argv[1], sys.argv[2]
html = open(src, encoding='utf-8').read()
tag = '<script async src=chisel.js></script>'
if html.count(tag) != 1:
    sys.exit('pack-itch: the emitted chisel.js script tag changed shape; update the picker swap')
# 13 bytes: a module whose only content is a 64-bit memory. Only a browser with
# Memory64 validates it. The wasm64 glue still asks for "chisel.wasm" by name, so
# locateFile points it at its own file.
picker = '''<script>
(function () {
  var m64 = false;
  try { m64 = WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,5,3,1,4,1])); } catch (e) {}
  if (/[?&]wasm32(&|$)/.test(location.search)) m64 = false;   // force the Safari path (testing)
  if (m64) Module.locateFile = function (p) { return p === 'chisel.wasm' ? 'chisel64.wasm' : p; };
  console.log('[boot] ' + (m64 ? 'wasm64, heap up to 16 GB' : 'wasm32, 4 GB heap (no Memory64 here)'));
  var s = document.createElement('script');
  s.async = true;
  s.src = m64 ? 'chisel64.js' : 'chisel.js';
  document.body.appendChild(s);
})();
</script>'''
open(dst, 'w', encoding='utf-8').write(html.replace(tag, picker))
EOF

FILES="index.html chisel.js chisel.wasm chisel64.js chisel64.wasm"
mkdir -p "$OUT"
rm -f "$ZIP"
# Python's zipfile where zip isn't installed (the desktop has neither zip nor unzip).
if command -v zip >/dev/null; then
  ( cd "$STAGE" && zip -q "$ZIP" $FILES )
else
  ( cd "$STAGE" && python3 -m zipfile -c "$ZIP" $FILES )
fi

echo "packed: $ZIP"
python3 -m zipfile -l "$ZIP"
