#include "debug_console.h"
#include "text_overlay.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__EMSCRIPTEN__)
    // Web keeps the shell.html overlay; capture here would fight it for the same text.
#elif defined(_WIN32)
    #include <io.h>
    #include <fcntl.h>
    #include <windows.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
#endif

namespace debug_console {
namespace {

// Matches the web overlay's cap. Long enough to hold a big-brush stroke's whole
// [stage]/[dirty] block plus the strokes either side of it; short enough that the
// ring never becomes a memory question.
constexpr size_t kMaxLines = 400;

// A pipe large enough that a frame-long stall cannot fill it. The alternative — a
// non-blocking write end — silently drops output, and a debug console that lies by
// omission is worse than no console.
constexpr int kPipeBytes = 1 << 20;

std::vector<std::string> g_lines;
std::string g_partial;          // bytes arrived since the last newline
bool   g_visible   = false;
int    g_scroll    = 0;         // lines back from the newest; 0 = pinned to the tail
bool   g_capturing = false;

int g_orig_out = -1;
int g_orig_err = -1;
int g_pipe_r   = -1;
int g_pipe_w   = -1;

void push_line(std::string s) {
    // CR would render as a glyph in a bitmap font that has no notion of one.
    if (!s.empty() && s.back() == '\r') s.pop_back();
    g_lines.push_back(std::move(s));
    if (g_lines.size() > kMaxLines)
        g_lines.erase(g_lines.begin(), g_lines.begin() + (g_lines.size() - kMaxLines));
    // Keep a scrolled-back reader anchored to the same text as new lines arrive.
    if (g_scroll > 0 && g_scroll < (int)g_lines.size()) g_scroll++;
}

void absorb(const char* buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (buf[i] == '\n') { push_line(g_partial); g_partial.clear(); }
        else                { g_partial.push_back(buf[i]); }
    }
}

// Colour by prefix, so an alarm is findable without reading. The [arena]/[cull]
// tags are exactly what chisel-debug.sh greps for when it decides whether a run
// was clean, so they get the loud colour.
void line_color(const std::string& s, float& r, float& g, float& b) {
    if (s.find("[arena]") != std::string::npos || s.find("[cull]") != std::string::npos
        || s.find("CRASH-GUARD") != std::string::npos
        || s.find("FAILED") != std::string::npos) {
        r = 1.000f; g = 0.333f; b = 0.333f;          // CGA light red
    } else if (s.compare(0, 7, "[stage]") == 0 || s.compare(0, 7, "[dirty]") == 0) {
        r = 0.333f; g = 1.000f; b = 1.000f;          // CGA light cyan
    } else if (!s.empty() && s[0] == '[') {
        r = 1.000f; g = 1.000f; b = 0.333f;          // CGA yellow
    } else {
        r = 0.800f; g = 0.800f; b = 0.800f;
    }
}

}  // namespace

void init() {
#if defined(__EMSCRIPTEN__)
    return;
#else
    if (g_capturing) return;

    int fds[2];
  #if defined(_WIN32)
    if (_pipe(fds, kPipeBytes, _O_BINARY) != 0) return;
  #else
    if (pipe(fds) != 0) return;
    fcntl(fds[0], F_SETFL, O_NONBLOCK);
    #if defined(F_SETPIPE_SZ)
    fcntl(fds[1], F_SETPIPE_SZ, kPipeBytes);   // Linux only; the default 64K is thin
    #endif
  #endif
    g_pipe_r = fds[0];
    g_pipe_w = fds[1];

  #if defined(_WIN32)
    g_orig_out = _dup(_fileno(stdout));
    g_orig_err = _dup(_fileno(stderr));
    _dup2(g_pipe_w, _fileno(stdout));
    _dup2(g_pipe_w, _fileno(stderr));
  #else
    g_orig_out = dup(STDOUT_FILENO);
    g_orig_err = dup(STDERR_FILENO);
    dup2(g_pipe_w, STDOUT_FILENO);
    dup2(g_pipe_w, STDERR_FILENO);
  #endif

    // stdout to a pipe is block-buffered by default, which would hold whole strokes
    // of output back until 4K accumulated. Line buffering is what makes the console
    // live, and it shrinks the crash-loss window to the current line.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    g_capturing = true;
    g_lines.reserve(kMaxLines);
    std::printf("[console] ~ toggles this overlay; PgUp/PgDn scroll\n");
#endif
}

void pump() {
#if defined(__EMSCRIPTEN__)
    return;
#else
    if (!g_capturing) return;
    char buf[8192];
    for (;;) {
  #if defined(_WIN32)
        // Windows anonymous pipes have no non-blocking mode, so ask what is there
        // before reading — a blocking _read on an empty pipe would freeze the frame.
        DWORD avail = 0;
        HANDLE h = (HANDLE)_get_osfhandle(g_pipe_r);
        if (h == INVALID_HANDLE_VALUE) return;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) return;
        int want = (int)(avail < sizeof(buf) ? avail : sizeof(buf));
        int n = _read(g_pipe_r, buf, want);
  #else
        ssize_t n = read(g_pipe_r, buf, sizeof(buf));
  #endif
        if (n <= 0) return;
        // Straight back out to the real stdout, so chisel-debug.log is unaffected by
        // the console existing. Errors here are ignored on purpose: a closed or full
        // log must not take the console down with it.
  #if defined(_WIN32)
        if (g_orig_out >= 0) { int w = _write(g_orig_out, buf, n); (void)w; }
  #else
        if (g_orig_out >= 0) { ssize_t w = write(g_orig_out, buf, (size_t)n); (void)w; }
  #endif
        absorb(buf, (size_t)n);
        if ((size_t)n < sizeof(buf)) return;   // drained it; anything more waits a frame
    }
#endif
}

void toggle() {
    g_visible = !g_visible;
    if (g_visible) g_scroll = 0;   // opening always shows the newest text
}

bool visible() { return g_visible; }

void scroll_lines(int delta) {
    if (!g_visible) return;
    g_scroll += delta;
    if (g_scroll < 0) g_scroll = 0;
    int max_back = (int)g_lines.size() - 1;
    if (max_back < 0) max_back = 0;
    if (g_scroll > max_back) g_scroll = max_back;
}

void draw(TextOverlay& text, int win_w, int win_h) {
    if (!g_visible) return;

    const float scale   = 2.0f;
    const float glyph_w = 8.0f * scale;
    const float line_h  = 8.0f * scale + 2.0f;
    const float pad     = 8.0f;

    float panel_h = (float)win_h * 0.45f;
    if (panel_h < line_h * 4.0f) panel_h = line_h * 4.0f;

    text.draw_panel(0.0f, 0.0f, (float)win_w, panel_h, win_w, win_h,
                    0.0f, 0.0f, 0.0f, 0.82f);

    int rows = (int)((panel_h - pad * 2.0f) / line_h);
    if (rows < 1) rows = 1;
    int max_chars = (int)(((float)win_w - pad * 2.0f) / glyph_w);
    if (max_chars < 1) max_chars = 1;

    // Newest line sits at the bottom, so the eye lands on it without hunting.
    int last  = (int)g_lines.size() - 1 - g_scroll;
    int first = last - rows + 1;
    if (first < 0) first = 0;

    float y = pad;
    for (int i = first; i <= last; i++) {
        const std::string& src = g_lines[(size_t)i];
        float r, g, b;
        line_color(src, r, g, b);
        if ((int)src.size() <= max_chars) {
            text.draw_text(src.c_str(), pad, y, scale, win_w, win_h, r, g, b, 1.0f);
        } else {
            std::string cut = src.substr(0, (size_t)max_chars);
            text.draw_text(cut.c_str(), pad, y, scale, win_w, win_h, r, g, b, 1.0f);
        }
        y += line_h;
    }

    // Only worth a status line when the view is not where the reader expects it.
    if (g_scroll > 0) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "-- %d line(s) back --", g_scroll);
        text.draw_text(tag, pad, panel_h - line_h, scale, win_w, win_h,
                       1.0f, 1.0f, 0.333f, 1.0f);
    }
}

void shutdown() {
#if defined(__EMSCRIPTEN__)
    return;
#else
    if (!g_capturing) return;
    pump();
    if (!g_partial.empty()) { push_line(g_partial); g_partial.clear(); }
  #if defined(_WIN32)
    if (g_orig_out >= 0) _dup2(g_orig_out, _fileno(stdout));
    if (g_orig_err >= 0) _dup2(g_orig_err, _fileno(stderr));
    if (g_pipe_r >= 0) _close(g_pipe_r);
    if (g_pipe_w >= 0) _close(g_pipe_w);
  #else
    if (g_orig_out >= 0) dup2(g_orig_out, STDOUT_FILENO);
    if (g_orig_err >= 0) dup2(g_orig_err, STDERR_FILENO);
    if (g_pipe_r >= 0) close(g_pipe_r);
    if (g_pipe_w >= 0) close(g_pipe_w);
  #endif
    g_capturing = false;
    g_pipe_r = g_pipe_w = -1;
#endif
}

}  // namespace debug_console
