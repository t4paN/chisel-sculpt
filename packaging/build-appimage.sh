#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
PKG_DIR="$SCRIPT_DIR/out"
APPDIR="$PKG_DIR/AppDir"
TOOLS_DIR="$SCRIPT_DIR/tools"

echo "=== Building Chisel ==="
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
make -C "$BUILD_DIR" -j"$(nproc)"

echo "=== Preparing AppDir ==="
rm -rf "$PKG_DIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp "$BUILD_DIR/chisel" "$APPDIR/usr/bin/chisel"
cp -r "$BUILD_DIR/assets" "$APPDIR/usr/bin/assets"
cp "$SCRIPT_DIR/chisel.desktop" "$APPDIR/usr/share/applications/chisel.desktop"
cp "$SCRIPT_DIR/chisel.desktop" "$APPDIR/chisel.desktop"

ICON_SRC="$PROJECT_DIR/assets/chisel-icon.png"
cp "$ICON_SRC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/chisel.png"
cp "$ICON_SRC" "$APPDIR/chisel.png"

cp "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"

echo "=== Fetching linuxdeploy (if needed) ==="
mkdir -p "$TOOLS_DIR"
LINUXDEPLOY="$TOOLS_DIR/linuxdeploy-x86_64.AppImage"
if [ ! -x "$LINUXDEPLOY" ]; then
    echo "Downloading linuxdeploy..."
    curl -fSL -o "$LINUXDEPLOY" \
        "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
    chmod +x "$LINUXDEPLOY"
fi

echo "=== Building AppImage ==="
export OUTPUT="$PKG_DIR/Chisel-x86_64.AppImage"
"$LINUXDEPLOY" --appdir "$APPDIR" --output appimage

echo ""
echo "=== Done ==="
echo "AppImage: $OUTPUT"
