#pragma once

// Quake-style `~` console for the NATIVE builds.
//
// The web build already has one — an overlay in packaging/web/shell.html that wraps
// console.log, because on web that is where printf output goes. Native had nothing,
// so reading a sculpting session meant tailing chisel-debug.log in another window
// and matching timestamps by eye. This puts the same text in front of the sculptor.
//
// It captures rather than hooks: stdout and stderr are redirected into a pipe at
// startup and drained once per frame, so EVERY existing printf in the codebase shows
// up without a single call site changing. Drained bytes are written straight back out
// to the real stdout, so chisel-debug.log still receives everything exactly as before.
//
// The cost of that choice: output now reaches the log when the frame drains it rather
// than the instant it is printed, so a hard crash can lose up to one frame of text.
// stdout is forced line-buffered here to keep that window as small as it can be.

struct TextOverlay;

namespace debug_console {

// Redirect stdout/stderr into the capture pipe. Safe to call once, at startup,
// before anything worth reading is printed. No-op on web.
void init();

// Drain the pipe into the ring buffer and echo to the real stdout. Call once per
// frame, whether or not the console is visible — the history is the point.
void pump();

void toggle();
bool visible();

// Positive scrolls back into history, negative returns toward the newest line.
// Clamped; scrolling back then printing leaves the view where the reader put it.
void scroll_lines(int delta);

void draw(TextOverlay& text, int win_w, int win_h);

// Final drain, then put stdout and stderr back. Without the restore, anything
// printed during teardown would vanish into a pipe nobody is reading.
void shutdown();

}  // namespace debug_console
