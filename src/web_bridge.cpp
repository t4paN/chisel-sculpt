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
void web_open_file_picker(bool meshes_only) {
    EM_ASM({
        var inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = $0 ? '.obj,.ply,.stl,.glb' : '.chisel,.obj,.ply,.stl,.glb';
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
    }, meshes_only ? 1 : 0);
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

int web_tablet_state() {
    return EM_ASM_INT({ return Module.chiselTabletState ? Module.chiselTabletState() : 0; });
}

void web_tablet_connect() {
    EM_ASM({ if (Module.chiselTabletConnect) Module.chiselTabletConnect(); });
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

        // Relative pen (Settings > Relative pen). The driver stays in Pen (absolute) mode,
        // so the browser gets pressure; here each pen move becomes a step of Chisel's own
        // cursor (relVX/Y), like a mouse. Lift the pen and put it down elsewhere and the
        // cursor carries on from where it was: the first report after the pen comes back
        // (leave/enter, or 100 ms of silence) only re-anchors. The OS pointer stays under
        // the tip, hidden (app_render.cpp), so a click lands only while the tip is over
        // the canvas.
        var relVX = 0;
        var relVY = 0;
        var relPX = 0;
        var relPY = 0;
        var relT = -1e9;
        var relFresh = true;
        var relSeen = false;
        var relLive = false;
        var RELGAP = 100;
        // After a re-anchor, every report for this long only re-anchors too: Chromium
        // follows the pen's return with a whole stale pointermove (the spot it left)
        // before the real one ~14 ms later. 40 ms of hover lost is unnoticeable.
        var RELSETTLE = 40;
        var relSettle = -1e9;
        // Chromium on Windows (Chrome, Edge), traced 2026-11-13 with a pointer-event probe.
        // Everything behind winChromium is off everywhere else, which stays as it was.
        //  - After every pen-up it sends one mouse-type pointermove at the tip, in the same
        //    millisecond as the pen's own events. Taken as "the mouse took over", it snapped
        //    the cursor to the tip after each stroke. A mouse event within RELECHO ms of a
        //    pen event is that echo and is dropped; the real mouse gets through once the
        //    pen has been quiet that long.
        //  - A pen held still sends nothing, so RELGAP of silence is no lift there: it
        //    re-anchored (and settled for 40 ms) at the start of every slow move, eating
        //    it. Lifts are always announced (leave/enter, new pointerId), so after a pause
        //    only a hop of more than RELHOP px re-anchors, with no settle window.
        var relUA = navigator.userAgent;
        var winChromium = relUA.indexOf('Windows') >= 0 && relUA.indexOf('Chrome/') >= 0;
        // Windows, any browser: the silence rule above is off in Firefox too. Windows holds
        // back the pen's moves for a moment after every pen-down (mean 30 ms, up to 130 ms
        // in the 2026-11-13 probe logs, Chrome and Firefox alike); in Firefox a hold of over
        // RELGAP read as a lift, so the stroke's first moves were dropped and 40 ms more
        // went to settling: the cursor stuck at the start of a stroke. Firefox announces
        // lifts there as Chrome does (leave/enter).
        var winQuiet = relUA.indexOf('Windows') >= 0;
        // A pen that left and came back (or changed id) since its last report: the next
        // step is the hop to where it landed, which a slider drag must not take.
        var penHop = false;
        var penId = -1;
        var RELECHO = 100;
        var RELHOP = 40;
        var relPenT = -1e9;
        var relLastType = '';
        // TEMPORARY: the last few pen events, printed at every pen-down.
        var relRing = [];
        var relNote = function(what) {
            relRing.push(what);
            if (relRing.length > 10) relRing.shift();
        };
        var relStep = function(cx, cy, t, r, k, gain) {
            var x = (cx - r.left) * k;
            var y = (cy - r.top) * k;
            var gap = t - relT > RELGAP;
            var fresh = relFresh || (gap && !winQuiet);
            var hop = winQuiet && !fresh && gap
                   && Math.abs(x - relPX) + Math.abs(y - relPY) > RELHOP;
            relT = t;
            if (fresh) relSettle = t + RELSETTLE;
            if (fresh || hop || t < relSettle) {
                console.log('[relpen] re-anchor' + (fresh ? '' : ' (settling)') + ': tip '
                            + x.toFixed(0) + ',' + y.toFixed(0)
                            + ' cursor stays ' + lastX.toFixed(0) + ',' + lastY.toFixed(0));
                relFresh = false;
                relPX = x;
                relPY = y;
                // Carry on from wherever the cursor was (the mouse's spot, if it was last);
                // the very first pen report starts under the tip.
                if (relSeen) { relVX = lastX; relVY = lastY; }
                else { relVX = x; relVY = y; relSeen = true; }
                return 0;
            }
            var dx = (x - relPX) * gain;
            var dy = (y - relPY) * gain;
            relPX = x;
            relPY = y;
            if (!sliding) {
                relVX = Math.max(0, Math.min(c.clientWidth - 1, relVX + dx));
                relVY = Math.max(0, Math.min(c.clientHeight - 1, relVY + dy));
            }
            return dx;
        };
        var relId = -1;
        var relFeed = function(e, r, k, gain) {
            var sdx = 0;
            // Chromium gives the pen a new pointerId each time it comes back into range:
            // a lift signal of its own, for browsers that skip pointerleave.
            if (e.pointerId !== relId) { relId = e.pointerId; relFresh = true; }
            // Chromium also re-enters with a STALE sample: the pointerover and the first
            // move's coalesced list carry the spot where the pen left, then the real one.
            // Anchoring on that stale sample turned the whole hop into one cursor step,
            // so the cursor stayed glued to the tip (traced 2026-11-13 with a pointer-event probe). A
            // fresh pen therefore anchors on the event's own, final position only.
            var fresh = relFresh || (!winQuiet && e.timeStamp - relT > RELGAP)
                     || e.timeStamp < relSettle;
            if (!fresh && e.type === 'pointermove' && e.getCoalescedEvents) {
                var list = e.getCoalescedEvents();
                for (var i = 0; i + 1 < list.length; i++) {
                    sdx += relStep(list[i].clientX, list[i].clientY, list[i].timeStamp, r, k, gain);
                    if (!sliding) Module._chisel_path_sample(relVX, relVY);
                }
            }
            sdx += relStep(e.clientX, e.clientY, e.timeStamp, r, k, gain);
            // The mouse's lockless-slider offset belongs to the mouse.
            offX = 0;
            offY = 0;
            reanchor = false;
            // A slider holds the cursor on its grab point; relVX/Y never moved during
            // it, so the pen carries on from there afterwards.
            var outX = sliding ? anchorX : relVX;
            var outY = sliding ? anchorY : relVY;
            lastX = outX;
            lastY = outY;
            sx0 = e.clientX;
            relNote(e.type + '@' + (e.timeStamp | 0) + ' tip ' + (((e.clientX - r.left) * k) | 0) + ','
                    + (((e.clientY - r.top) * k) | 0) + ' out ' + (outX | 0) + ',' + (outY | 0)
                    + ' n=' + (e.getCoalescedEvents ? e.getCoalescedEvents().length : 1));
            if (e.type === 'pointerdown') {
                console.log('[relpen] down; last events: ' + relRing.join(' | '));
                relRing = [];
            }
            Module._chisel_set_pointer(outX, outY, sdx);
        };
        var relGone = function(e) {
            if (e.pointerType !== 'pen') return;
            relFresh = true;
            penHop = true;
            relNote(e.type + '@' + (e.timeStamp | 0));
        };
        c.addEventListener('pointerleave', relGone);
        c.addEventListener('pointerenter', relGone);
        c.addEventListener('pointercancel', relGone);

        // Windows, any browser: a pen side button set to "Erase" in the tablet driver pans,
        // as the middle button. The driver's own "Middle Click" never reaches the page in
        // Chromium (no press, no release; Firefox gets both), and a hovering eraser carries
        // no button bit in either. What does arrive, the same in Chrome and Firefox:
        // pressing or releasing Erase swaps the pen for the eraser tool, seen as the pen
        // leaving and coming back 5-14 ms later with a new pointerId (a real lift took
        // 136 ms or more; probe logs of 2026-11-13). So a
        // return within ERASESWAP ms flips pen <-> eraser and any slower return is the
        // plain pen again. Contact settles it for sure: bit 32 is the eraser, bit 1 the
        // tip. It is a guess from timing: a pen that leaves range with the button held is
        // read as the plain pen when it comes back. Any pen mode, relative or not.
        // The pen's pan button, held as the middle button. Two sources can ask for it: the
        // eraser swap below (bit 1) and the tablet's own report further down (bit 2). One
        // press while either asks, one release when neither does.
        var penPanBits = 0;
        var penPan = function(bit, on, x, y) {
            var was = penPanBits !== 0;
            penPanBits = on ? (penPanBits | bit) : (penPanBits & ~bit);
            var now = penPanBits !== 0;
            if (now === was) return;
            // An ordinary middle press for Emscripten's GLFW; the position is already
            // Chisel's own (chisel_set_pointer).
            c.dispatchEvent(new MouseEvent(now ? 'mousedown' : 'mouseup', { bubbles: true,
                cancelable: true, view: window, button: 1, buttons: now ? 4 : 0,
                clientX: x, clientY: y }));
        };

        if (relUA.indexOf('Windows') >= 0) {
            var ERASESWAP = 40;
            var eraser = false;
            var eraserHeld = false;
            var eraserLeftT = -1e9;
            var eraserIn = false;
            var eraserPan = function(on, e) {
                if (on === eraserHeld) return;
                eraserHeld = on;
                penPan(1, on, e.clientX, e.clientY);
            };
            c.addEventListener('pointerleave', function(e) {
                if (e.pointerType !== 'pen') return;
                eraserLeftT = e.timeStamp;
                eraserIn = false;
                // Gone for longer than a swap: let the button go.
                setTimeout(function() {
                    if (!eraserIn && eraserHeld) { eraser = false; eraserPan(false, e); }
                }, 300);
            });
            c.addEventListener('pointerenter', function(e) {
                if (e.pointerType !== 'pen') return;
                eraserIn = true;
                eraser = e.timeStamp - eraserLeftT < ERASESWAP ? !eraser : false;
                eraserPan(eraser, e);
            });
            var eraserSync = function(e) {
                if (e.pointerType !== 'pen') return;
                if (e.buttons & 32) eraser = true;
                else if (e.buttons & 1) eraser = false;
                eraserPan(eraser, e);
            };
            c.addEventListener('pointermove', eraserSync);
            c.addEventListener('pointerdown', eraserSync);
        }

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
            // For the slider's pointer lock (input.cpp web_pointer_lock): a pen gets none.
            Module.chiselPenDrives = e.pointerType === 'pen';
            if (e.pointerType === 'pen' && e.pointerId !== penId) { penId = e.pointerId; penHop = true; }
            // Chromium on Windows: the mouse-type echo of a relative pen (see winChromium).
            if (e.pointerType === 'pen') relPenT = e.timeStamp;
            else if (winChromium && e.pointerType === 'mouse' && relLive
                     && e.timeStamp - relPenT < RELECHO) return;
            var gain = e.pointerType === 'pen' ? Module._chisel_relpen_gain() : 0;
            var live = gain > 0;
            // TEMPORARY relative-pen diagnostics: what the browser calls this pointer.
            if (e.pointerType !== relLastType) {
                relLastType = e.pointerType;
                console.log('[relpen] pointerType=' + e.pointerType + ' pressure=' + e.pressure
                            + ' gain=' + gain + ' setting=' + Module._chisel_relpen_gain());
            }
            if (live !== relLive) {
                relLive = live;
                if (live) relFresh = true;
                Module._chisel_relpen_live(live ? 1 : 0);
            }
            if (live && !locked) { relFeed(e, r, k, gain); return; }
            if (relSeen) relNote(e.type + '/' + e.pointerType + '@' + (e.timeStamp | 0));
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
            // An absolute pen's slider drag sends the tip's position: the ring follows it
            // (input.cpp g_slider_pen_abs). A relative pen took its own path above.
            var penSlide = sliding && e.pointerType === 'pen';
            var outX = penSlide ? lx : anchorX;
            var outY = penSlide ? ly : anchorY;
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
                // A pen lifted and put down elsewhere mid-drag: not a drag step.
                if (e.pointerType === 'pen' && penHop) sdx = 0;
            }
            if (e.pointerType === 'pen') penHop = false;
            sx0 = e.clientX;
            Module._chisel_set_pointer(outX, outY, sdx);
        };
        c.addEventListener('pointermove', feed);
        c.addEventListener('pointerdown', feed);
        // Keep a mouse drag's moves coming once it leaves the canvas (or itch's
        // iframe), so an orbit or a stroke carries on past it. Released by the browser
        // on pointerup. Pens only in relative mode (below).
        c.addEventListener('pointerdown', function(e) {
            // A relative pen too: its stroke shouldn't stop where the tip leaves the canvas.
            var cap = e.pointerType === 'mouse'
                   || (e.pointerType === 'pen' && Module._chisel_relpen_gain() > 0);
            if (cap && c.setPointerCapture) {
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

        // Pen pressure without Windows Ink (Chrome and Edge on Windows; WebHID). With Ink
        // off, or the driver in Mouse mode, the browser is handed a plain mouse and no DOM
        // event carries pressure. The tablet's own USB reports still do, and its vendor
        // collection opens next to the Wacom driver, so the page reads them itself.
        // Position and buttons stay the mouse's; only the pressure comes from here, into
        // the same chisel_set_pen as above, on every report while the pen is in range
        // (the device arbiter in main.cpp then calls it a tablet, as with a real pen).
        // The user grants the tablet once, from the menu (ui_overlay.cpp); the browser
        // remembers it, so later visits open it with no prompt.
        //
        // Report layout: Wacom's "Bamboo pen" report, measured on a One by Wacom M
        // (056A:037B, 2026-11-15) and the same as Linux wacom_wac.c wacom_bpt_pen:
        //   [0] bit 0 tip, bits 1-2 side buttons, bit 5 ready, bit 6 prox, bit 7 range
        //   [1..2] x   [3..4] y   [5..6] pressure, little-endian, 0..2047   [7] distance
        // The tablet sends it as report id 2. Next to the Wacom driver on Windows it
        // arrives as report id 220 with that 2 as the first byte. Other tablets need
        // their own layout: their reports match neither shape and are ignored.
        //
        // The lower side button pans, read from the same report (bit 1 of byte 0), so it
        // does not matter what the driver maps it to. With Ink off the driver's "Erase"
        // sends the page nothing at all (measured 2026-11-15, Chrome: no button, no wheel,
        // no pointer event), and its "Pan/Scroll" is the driver scrolling by itself. While
        // the button is held the wheel is kept from the app (the driver's own scrolling
        // would zoom), and so is the tip's left press (touching down mid-pan is no stroke).
        var hidOpen = 0;
        var hidInk = false;
        var hidSide = false;
        var hidX = 0;
        var hidY = 0;
        // A real pen event (Ink on) brings its own pressure: the tablet's is not used.
        var hidNote = function(e) {
            hidInk = e.pointerType === 'pen';
            hidX = e.clientX;
            hidY = e.clientY;
        };
        window.addEventListener('pointermove', hidNote, true);
        window.addEventListener('pointerdown', hidNote, true);
        var hidPan = function(on) {
            if (on === hidSide) return;
            hidSide = on;
            penPan(2, on, hidX, hidY);
        };
        var hidReport = function(e) {
            var d = e.data;
            var o = -1;
            if (e.reportId === 2 && d.byteLength >= 7) o = 0;
            else if (e.reportId === 220 && d.byteLength >= 8 && d.getUint8(0) === 2) o = 1;
            if (o < 0) return;
            var b = d.getUint8(o);
            // Bit 5: the pen is in range and reporting. Out of range lets the button go.
            hidPan((b & 0x22) === 0x22);
            if (hidInk || !(b & 0x20)) return;
            Module._chisel_set_pen(1, Math.min(1, d.getUint16(o + 5, true) / 2047));
        };
        window.addEventListener('wheel', function(e) {
            if (!hidSide) return;
            e.preventDefault();
            e.stopImmediatePropagation();
        }, { capture: true, passive: false });
        var hidTipEaten = false;
        var hidTip = function(e) {
            // The app's own made-up middle press is not trusted and passes.
            if (!e.isTrusted || e.button !== 0) return;
            if (e.type === 'mousedown' ? !hidSide : !hidTipEaten) return;
            hidTipEaten = e.type === 'mousedown';
            e.preventDefault();
            e.stopImmediatePropagation();
        };
        window.addEventListener('mousedown', hidTip, true);
        window.addEventListener('mouseup', hidTip, true);
        var hidAttach = function(dev) {
            if (dev.vendorId !== 0x056a || dev.chiselOn) return;
            (dev.opened ? Promise.resolve() : dev.open()).then(function() {
                if (dev.chiselOn) return;
                dev.chiselOn = true;
                dev.addEventListener('inputreport', hidReport);
                hidOpen++;
                console.log('[tablet] reading pressure from ' + dev.productName);
            }).catch(function(err) {
                console.warn('[tablet] could not open ' + dev.productName + ': ' + err.message);
            });
        };
        var hidHere = !!navigator.hid && navigator.userAgent.indexOf('Windows') >= 0;
        // 0 = not offered here, 1 = offered and no tablet open, 2 = a tablet is open.
        Module.chiselTabletState = function() { return !hidHere ? 0 : (hidOpen > 0 ? 2 : 1); };
        // Needs a user gesture: called from a menu click.
        Module.chiselTabletConnect = function() {
            if (!hidHere) return;
            navigator.hid.requestDevice({ filters: [{ vendorId: 0x056a }] }).then(function(list) {
                list.forEach(hidAttach);
            }).catch(function(err) {
                console.warn('[tablet] request failed: ' + err.message);
            });
        };
        if (hidHere) {
            navigator.hid.getDevices().then(function(list) { list.forEach(hidAttach); });
            navigator.hid.addEventListener('connect', function(e) { hidAttach(e.device); });
            navigator.hid.addEventListener('disconnect', function(e) {
                if (!e.device.chiselOn) return;
                e.device.chiselOn = false;
                e.device.removeEventListener('inputreport', hidReport);
                hidOpen = Math.max(0, hidOpen - 1);
                hidPan(false);
            });
        }

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
