#!/usr/bin/env bash
# Package the WebGPU web build as an installable, offline-capable PWA.
#
# Produces a flat folder you can serve from any static HTTPS host (GitHub Pages,
# your own server) or from localhost for testing. Opening it once in Chrome on
# Android offers "Install app"; after that it launches fullscreen and runs with
# no network. It will NOT work opened as a local file (file://) — service
# workers need HTTPS or localhost.
#
# The service worker's cache name is stamped with a hash of the shipped files,
# so every new build replaces the old cache on the next launch.
#
# Usage:  packaging/web/pack-pwa.sh [version]      (default: git describe)
# Output: dist/chisel-pwa-<version>/   and   dist/chisel-pwa-<version>.zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${1:-$(git -C "$REPO" describe --tags --always 2>/dev/null || echo dev)}"
BUILD="$REPO/build-web"
OUT="$REPO/dist"
DIR="$OUT/chisel-pwa-$VERSION"
ZIP="$OUT/chisel-pwa-$VERSION.zip"

for f in chisel.html chisel.js chisel.wasm; do
  [ -f "$BUILD/$f" ] || { echo "missing $BUILD/$f — build the web target first (cmake --build build-web)" >&2; exit 1; }
done

rm -rf "$DIR" "$ZIP"
mkdir -p "$DIR/icons"
cp "$BUILD/chisel.html" "$DIR/index.html"
cp "$BUILD/chisel.js"   "$BUILD/chisel.wasm" "$DIR/"
cp "$SCRIPT_DIR/manifest.webmanifest" "$DIR/"
cp "$SCRIPT_DIR"/icons/*.png "$DIR/icons/"

# Hash everything the worker caches, then stamp it in.
HASH="$(cd "$DIR" && find . -type f ! -name sw.js -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
sed "s/__BUILD__/$HASH/" "$SCRIPT_DIR/sw.js" > "$DIR/sw.js"

( cd "$DIR" && zip -qr "$ZIP" . )

echo "packed: $DIR  (build $HASH)"
echo "zip:    $ZIP"
echo "test:   cd \"$DIR\" && python3 -m http.server 8080   then open http://localhost:8080"
