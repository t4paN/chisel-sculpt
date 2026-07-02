#!/usr/bin/env bash
# Package the WebGPU web build for an itch.io HTML upload.
#
# itch's HTML player serves the zip and opens index.html at the zip root, so we
# rename chisel.html -> index.html and zip the three runtime files flat (no
# subdirectory). The build is single-threaded ASYNCIFY, so there are no
# .data/.worker/.mem sidecars — just html + js glue + wasm.
#
# Usage:  packaging/web/pack-itch.sh [version]      (default version: v0.1.6)
# Output: dist/chisel-itch-<version>.zip
set -euo pipefail

VERSION="${1:-v0.1.6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD="$REPO/build-web"
OUT="$REPO/dist"
ZIP="$OUT/chisel-itch-$VERSION.zip"

for f in chisel.html chisel.js chisel.wasm; do
  [ -f "$BUILD/$f" ] || { echo "missing $BUILD/$f — build the web target first (cmake --build build-web)" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$BUILD/chisel.html" "$STAGE/index.html"
cp "$BUILD/chisel.js"   "$STAGE/chisel.js"
cp "$BUILD/chisel.wasm" "$STAGE/chisel.wasm"

mkdir -p "$OUT"
rm -f "$ZIP"
( cd "$STAGE" && zip -q "$ZIP" index.html chisel.js chisel.wasm )

echo "packed: $ZIP"
unzip -l "$ZIP"
