// The browser side of the web build: C entry points the page calls, the MEMFS
// download / file pickers, and the DOM listeners main() installs before handing the
// loop to the browser. Empty on native.
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <GLFW/glfw3.h>
#include "web_bridge.h"
#include "input.h"
#include "settings.h"
#include <string>

// Browser save/open bridge. MEMFS is RAM that dies with the tab, and host files
// are invisible to it, so path-based dialogs are meaningless on web: save/export
// write through the normal writers into MEMFS and the bytes leave as a browser
// download; open/import goes through an <input type=file> picker whose bytes
// land back in MEMFS for the normal loaders. No IDBFS persistence by design.

// Settings autosave backstop. The frame loop's debounced write is not enough on its own
// here: a browser can discard a tab without ever running an exit path, and native's
// shutdown flush has no web equivalent (emscripten_set_main_loop_arg never returns).
// pagehide/visibilitychange are the last events that reliably fire — beforeunload does
// not on mobile. Points at main()'s InputState, which outlives every frame (the
// simulate_infinite_loop throw keeps main's stack frame alive).
InputState* g_settings_input = nullptr;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_flush_settings() {
    if (g_settings_input) settings_save(*g_settings_input);
}

// Asked by the page's keydown hook whether the browser should be kept off this key:
// yes for anything while the Key bindings window is open (it is listening, and Ctrl+D
// must reach it rather than bookmark the page), else yes only when a binding wants
// exactly this key and these modifiers. Exact, not the subset rule key_callback uses,
// so a plain F binding does not also swallow the browser's own Ctrl+F.
extern "C" EMSCRIPTEN_KEEPALIVE int chisel_key_wanted(int glfw_key, int ctrl, int shift, int alt) {
    if (!g_settings_input) return 0;
    const InputState& in = *g_settings_input;
    if (in.keybind_menu_open) return 1;
    const int mods = (ctrl ? GLFW_MOD_CONTROL : 0) | (shift ? GLFW_MOD_SHIFT : 0)
                   | (alt ? GLFW_MOD_ALT : 0);
    for (int i = 0; i < (int)KeyAction::COUNT; i++)
        if (in.keybinds[i].bound() && in.keybinds[i].key == glfw_key && in.keybinds[i].mods == mods)
            return 1;
    return 0;
}

// Set async by the picker's JS callback; the frame loop consumes it. Safe to
// assign from JS mid-ASYNCIFY-suspend: plain reentry, nothing here suspends.
std::string g_web_import_path;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_web_import_done(const char* path) {
    g_web_import_path = path ? path : "";
}

// Brush-alpha custom image: same MEMFS bridge as import, separate path so the two
// pickers can't clobber each other. The frame loop consumes it into the alpha pool.
std::string g_web_alpha_path;
extern "C" EMSCRIPTEN_KEEPALIVE void chisel_web_alpha_done(const char* path) {
    g_web_alpha_path = path ? path : "";
}

// Read fs_path out of MEMFS and hand it to the browser, deleting the MEMFS copy.
// Preferred route is showSaveFilePicker (the browser's real save dialog: folder
// browsing + editable name, prefilled with download_name) — but Chrome blocks it
// in cross-origin iframes (itch embeds the game that way), so any refusal falls
// back to a plain anchor download. A user cancel (AbortError) saves nothing.
void web_download_file(const char* fs_path, const char* download_name) {
    EM_ASM({
        var path = UTF8ToString($0);
        var name = UTF8ToString($1);
        var data = FS.readFile(path);
        FS.unlink(path);
        var fallback = function() {
            var a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([data]));
            a.download = name;
            a.click();
            setTimeout(function() { URL.revokeObjectURL(a.href); }, 10000);
        };
        if (window.showSaveFilePicker) {
            window.showSaveFilePicker({ suggestedName: name }).then(function(h) {
                return h.createWritable().then(function(w) {
                    return w.write(data).then(function() { return w.close(); });
                });
            }).catch(function(e) {
                if (e && e.name === 'AbortError') return;   // user cancelled
                fallback();   // SecurityError (cross-origin iframe) etc.
            });
        } else {
            fallback();       // Firefox/Safari: no File System Access API
        }
    }, fs_path, download_name);
}

// Fire the browser's file picker. Needs transient user activation, so it must
// run within a few seconds of the click/keypress that requested it (it does:
// the request flag is consumed the next frame). Result arrives whenever the
// user picks — chisel_web_import_done gets the MEMFS path.
void web_open_file_picker() {
    EM_ASM({
        var inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = '.chisel,.obj,.ply,.glb';
        inp.onchange = function() {
            var f = inp.files && inp.files[0];
            if (!f) return;
            f.arrayBuffer().then(function(buf) {
                var path = '/' + f.name;
                FS.writeFile(path, new Uint8Array(buf));
                Module.ccall('chisel_web_import_done', null, ['string'], [path]);
            });
        };
        inp.click();
    });
}

// Brush-alpha image picker — twin of web_open_file_picker, image types, lands in
// MEMFS under /alpha_ (namespaced so it never collides with an import) and reports
// via chisel_web_alpha_done.
void web_open_alpha_picker() {
    EM_ASM({
        var inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'image/png,image/jpeg,.png,.jpg,.jpeg,.tga,.bmp';
        inp.onchange = function() {
            var f = inp.files && inp.files[0];
            if (!f) return;
            f.arrayBuffer().then(function(buf) {
                var path = '/alpha_' + f.name;
                FS.writeFile(path, new Uint8Array(buf));
                Module.ccall('chisel_web_alpha_done', null, ['string'], [path]);
            });
        };
        inp.click();
    });
}

void web_install_dom_listeners() {
    // Feed pointer position straight from the DOM in CSS-pixel space (canvas-relative),
    // bypassing GLFW's backing-store coords which desync from render/pick space across a
    // resize (the cursor-offset bug). A fresh getBoundingClientRect per event keeps it
    // exact through any resize/DPR change. chisel_set_pointer lives in input.cpp.
    EM_ASM({
        var c = Module.canvas;
        // Under pointer lock (slider drags only) clientX freezes, so the position is
        // rebuilt from movementX/Y, starting from the last free one, and stays
        // continuous when the lock lets go.
        var lx = 0;   // (separate statements: a bare comma would split EM_ASM's
        var ly = 0;   //  macro arguments)
        // Slider drags are fed their own delta, not the raw movementX: browsers on Linux
        // report spikes of hundreds of pixels in it (on lock and at random), which pinned
        // a slider to 0 or 100 with nothing between. Free pointer: the clientX step,
        // which can't spike. Locked: movementX, minus the event that crosses the lock
        // change and anything too big for one frame of a hand.
        // The ring stays where a slider drag began. The position is frozen at the grab
        // point for the whole drag. A held lock (Chromium) returns the pointer there, apart
        // from one ghost event at the unlock (see below). Firefox on Linux holds nothing,
        // so there the next real position gets an offset that continues from the grab
        // point instead of jumping to wherever the pointer went. The offset clears where
        // the hidden pointer must be honest: at the canvas edge (it can't go further),
        // over the UI (the OS arrow is showing there) and for pens and fingers (absolute).
        var sliding = false;
        var anchorX = 0;
        var anchorY = 0;
        var reanchor = false;
        var offX = 0;
        var offY = 0;
        var lastX = 0;
        var lastY = 0;
        // Where the real pointer was at the grab, whether a lock actually held during the
        // drag, and when the drag ended (see the ghost below).
        var grabX = 0;
        var grabY = 0;
        var slideLocked = false;
        var endT = 0;
        Module.chiselSliderBegin = function(x, y) {
            sliding = true; anchorX = x; anchorY = y;
            grabX = lx; grabY = ly; slideLocked = false;
        };
        Module.chiselSliderEnd = function(keep) {
            sliding = false;
            // A held lock puts the real pointer back on the grab point, so the ring already
            // continues from there; only a lockless drag (Firefox on Linux) needs the offset.
            reanchor = keep !== 0 && !slideLocked;
            endT = performance.now();
            lastX = anchorX;
            lastY = anchorY;
        };
        var sx0 = -1;
        var sLocked = false;
        var SPIKE = 400;
        document.addEventListener('pointerlockchange', function() {
            console.log('[slider] pointer lock ' + (document.pointerLockElement === c ? 'on' : 'off'));
        });
        document.addEventListener('pointerlockerror', function() {
            console.log('[slider] pointer lock refused (pointerlockerror)');
        });

        // Two-finger touch: pinch zooms, midpoint pans (applied in the main loop).
        // Only the first finger reaches the app as a click (Emscripten's GLFW), so
        // every finger is tracked here from pointer events. Registered BEFORE feed so
        // the gesture flag is already up when feed sees the second finger. Per-axis
        // maps, not {x, y} objects: same EM_ASM comma rule as above.
        var tpx = {};
        var tpy = {};
        var tn = 0;
        var tFirst = 0;       // when the first finger of this touch landed
        var tGesture = false;
        // The finger GLFW clicks with: the first one on the CANVAS. Not e.isPrimary —
        // that is the first finger on the whole page, which is the thumb whenever it
        // is already holding a modifier button (shell.html), and then the sculpting
        // finger would never move the cursor.
        var tPrimary = -1;
        var tDist = 0;
        var tMx = 0;
        var tMy = 0;
        // A second finger this soon after the first means the first one's stroke or
        // orbit was a stub of the pinch, and the app takes it back. Later than this,
        // whatever the first finger did is kept.
        var TAKE_BACK_MS = 300;
        var tPair = function() {
            var k = Object.keys(tpx);
            var dx = tpx[k[0]] - tpx[k[1]];
            var dy = tpy[k[0]] - tpy[k[1]];
            tDist = Math.sqrt(dx * dx + dy * dy);
            tMx = (tpx[k[0]] + tpx[k[1]]) * 0.5;
            tMy = (tpy[k[0]] + tpy[k[1]]) * 0.5;
        };
        c.addEventListener('pointerdown', function(e) {
            if (e.pointerType !== 'touch') return;
            if (tn === 0) { tFirst = performance.now(); tPrimary = e.pointerId; }
            tpx[e.pointerId] = e.clientX;
            tpy[e.pointerId] = e.clientY;
            tn++;
            var revert = 0;
            if (tn >= 2 && !tGesture) {
                tGesture = true;
                revert = (performance.now() - tFirst < TAKE_BACK_MS) ? 1 : 0;
            }
            if (tn >= 2) tPair();   // re-baseline: a new finger is not a pinch step
            Module._chisel_touch_count(tn, revert);
        });
        c.addEventListener('pointermove', function(e) {
            if (e.pointerType !== 'touch' || !(e.pointerId in tpx)) return;
            tpx[e.pointerId] = e.clientX;
            tpy[e.pointerId] = e.clientY;
            if (!tGesture || tn < 2) return;
            var d0 = tDist;
            var mx0 = tMx;
            var my0 = tMy;
            tPair();
            if (d0 > 1 && tDist > 1)
            {
                // Pan in the app's pixels: client px times the stage's CSS scale-down
                // (1 unless the touch UI scale is on, see shell.html #stage).
                var pk = c.clientWidth / (c.getBoundingClientRect().width || c.clientWidth || 1);
                Module._chisel_touch_pinch(tDist / d0, (tMx - mx0) * pk, (tMy - my0) * pk);
            }
        });
        var tUp = function(e) {
            if (e.pointerType !== 'touch' || !(e.pointerId in tpx)) return;
            delete tpx[e.pointerId];
            delete tpy[e.pointerId];
            tn--;
            if (tn >= 2) tPair();
            if (tn === 0) { tGesture = false; tPrimary = -1; }
            Module._chisel_touch_count(tn, 0);
        };
        c.addEventListener('pointerup', tUp);
        c.addEventListener('pointercancel', tUp);

        var feed = function(e) {
            // During a gesture no finger drives the cursor, and outside one only the
            // first canvas finger does (it is the one GLFW clicks with).
            if (e.pointerType === 'touch' && (tGesture || e.pointerId !== tPrimary)) return;
            var r = c.getBoundingClientRect();
            // Client px -> the app's pixels. 1 normally; under the touch UI scale the
            // canvas is laid out bigger than it shows (shell.html #stage), and the app
            // works in its layout size.
            var k = r.width > 0 ? c.clientWidth / r.width : 1;
            var locked = document.pointerLockElement === c;
            // Browsers deliver one pointermove per frame; the positions merged into it
            // are the stroke's real path (see InputState::path_x). The last coalesced
            // event is `e` itself, so it goes through the normal call below.
            if (e.type === 'pointermove' && e.getCoalescedEvents) {
                var list = e.getCoalescedEvents();
                var px = lx;
                var py = ly;
                for (var i = 0; i + 1 < list.length; i++) {
                    if (locked) {
                        if (!sliding) { px += (list[i].movementX || 0) * k; py += (list[i].movementY || 0) * k; }
                    } else {
                        px = (list[i].clientX - r.left) * k; py = (list[i].clientY - r.top) * k;
                    }
                    Module._chisel_path_sample(px + offX, py + offY);
                }
                if (locked) { lx = px; ly = py; }
            }
            if (locked) {
                if (!sliding) { lx += (e.movementX || 0) * k; ly += (e.movementY || 0) * k; }
                else slideLocked = true;
            } else {
                lx = (e.clientX - r.left) * k; ly = (e.clientY - r.top) * k;
                // Chromium's unlock first reports a ghost: the position the hand would have
                // taken the pointer to, with zero movement. The real pointer arrives back on
                // the grab point right after. Taking the ghost as real threw the ring a whole
                // drag's distance (traced 2026-10-21), so it is held on the grab point.
                if (slideLocked && !sliding && performance.now() - endT < 300
                    && !e.movementX && !e.movementY
                    && Math.abs(lx - grabX) + Math.abs(ly - grabY) > 2) {
                    lx = grabX; ly = grabY;
                }
            }
            var outX = anchorX;
            var outY = anchorY;
            if (!sliding) {
                // The first free position after a drag: continue from where the ring is.
                // (Still locked = Chromium hasn't let go yet; lx is still on the grab
                // point there, so the offset waits for the real position.)
                if (reanchor && !locked) {
                    reanchor = false;
                    offX = lastX - lx;
                    offY = lastY - ly;
                }
                var edge = lx <= 0.5 || ly <= 0.5
                        || lx >= c.clientWidth - 1.5 || ly >= c.clientHeight - 1.5;
                if (e.pointerType !== 'mouse' || c.style.cursor !== 'none' || edge) {
                    offX = 0;
                    offY = 0;
                }
                outX = lx + offX;
                outY = ly + offY;
            }
            lastX = outX;
            lastY = outY;
            var sdx = 0;
            if (locked !== sLocked) {
                sLocked = locked;
            } else if (locked) {
                sdx = (e.movementX || 0) * k;
                if (Math.abs(sdx) > SPIKE) {
                    console.log('[slider] dropped a movementX spike: ' + e.movementX);
                    sdx = 0;
                }
            } else if (sx0 >= 0) {
                sdx = (e.clientX - sx0) * k;
            }
            sx0 = e.clientX;
            Module._chisel_set_pointer(outX, outY, sdx);
        };
        c.addEventListener('pointermove', feed);
        c.addEventListener('pointerdown', feed);
        // Keep a mouse drag's moves coming once it leaves the canvas (or itch's
        // iframe), so an orbit or a stroke carries on past it. Released by the browser
        // on pointerup. Pens are left alone.
        c.addEventListener('pointerdown', function(e) {
            if (e.pointerType === 'mouse' && c.setPointerCapture) {
                try { c.setPointerCapture(e.pointerId); } catch (err) {}
            }
        });

        // Stylus force, which GLFW never surfaces — this is the web build's entire
        // tablet path (see the __EMSCRIPTEN__ branch of tablet.cpp). Guarded on
        // pointerType because a mouse reports a fixed 0.5 while held, which is
        // indistinguishable from a pen at half force.
        //
        // Bound to window in the CAPTURE phase, not to the canvas: capture runs
        // window -> target, so this sees every pointer event that exists, including
        // ones an upstream handler swallows with stopPropagation before they reach
        // the canvas (itch serves us inside its own iframe wrapper). It must stay a
        // SEPARATE handler from feed() above — feed passes e.movementX to
        // chisel_set_pointer, which *accumulates* into the slider drag, so running
        // the same event through both listeners would double every slider's speed.
        var feedPen = function(e) {
            if (e.pointerType === 'pen')
                Module._chisel_set_pen(1, e.pressure);
        };
        window.addEventListener('pointermove', feedPen, true);
        window.addEventListener('pointerdown', feedPen, true);

        // Last-chance settings write. visibilitychange->hidden is the reliable one on
        // mobile (a backgrounded tab may never get pagehide before it is discarded);
        // pagehide covers desktop navigation away. Both are cheap and idempotent.
        window.addEventListener('pagehide', function() { Module._chisel_flush_settings(); });
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'hidden') Module._chisel_flush_settings();
        });

        // Keep app-owned shortcuts away from the browser: Emscripten's GLFW only
        // preventDefault()s Backspace/Tab, so F1 opened help, Ctrl+D bookmarked,
        // Ctrl+S saved the page, … The app still receives these through GLFW's own
        // window-level listener (preventDefault stops the browser default, not
        // propagation). Ctrl+Shift combos stay untouched except Shift+Z (redo) so
        // devtools (Ctrl+Shift+I) keep working. Matched on e.code (physical key),
        // not e.key: under a non-Latin layout (e.g. Greek) Ctrl+D arrives as
        // e.key='δ' and a key-based match lets the browser bookmark anyway.
        //
        // Rebound shortcuts: Ctrl/Alt combos and F-keys are checked against the live
        // bindings (chisel_key_wanted), mapping e.code to the GLFW code — letters and
        // digits share their ASCII value, F1 is 290. Plain keys are left alone: keeping
        // the browser off them would also eat the keypress the Save dialog types with.
        // While the Key bindings window is open it wants every key.
        window.addEventListener('keydown', function(e) {
            var code = e.code || "";
            var ctrl = e.ctrlKey || e.metaKey;
            if (e.key === 'F1'
                || (ctrl && !e.altKey && (!e.shiftKey || code === 'KeyZ')
                    && ['KeyD','KeyS','KeyO','KeyI','KeyZ','KeyY','KeyP'].indexOf(code) !== -1)) {
                e.preventDefault();
                return;
            }
            var gk = -1, fkey = false;
            if (/^Key[A-Z]$/.test(code)) gk = code.charCodeAt(3);
            else if (/^Digit[0-9]$/.test(code)) gk = code.charCodeAt(5);
            else if (/^F([1-9]|1[0-2])$/.test(code)) { gk = 289 + parseInt(code.substring(1), 10); fkey = true; }
            // -2 is no key at all, so this is true only while the window is listening.
            var listening = Module._chisel_key_wanted(-2, 0, 0, 0);
            if (listening || ((ctrl || e.altKey || fkey) && gk !== -1
                    && Module._chisel_key_wanted(gk, ctrl ? 1 : 0, e.shiftKey ? 1 : 0, e.altKey ? 1 : 0)))
                e.preventDefault();
        }, true);
    });
}
#endif // __EMSCRIPTEN__
