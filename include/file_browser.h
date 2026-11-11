#pragma once
// The native file dialogs' browser: one ImGuiFileDialog subclass that every Open /
// Save / Export / Brush Alpha / folder pick shares. It adds a Places pane down the left
// (home folders, the default folder if you picked one, bookmarks, drives, recent
// folders; Places and Drives rows can be removed and reset), back / forward
// / up, and an address bar of clickable folders that turns into a text field on click.
//
// It also remembers where you were: the folder of the last file a dialog returned
// (opened or saved), plus bookmarks and recent folders, in places.txt beside
// settings.cfg. Native only; on web every call is a no-op (the browser's own pickers
// are used there).
#include <string>

namespace IGFD { class FileDialog; }

// The one dialog. Use it in place of IGFD::FileDialog::Instance().
IGFD::FileDialog* file_browser();

// Where a dialog should open: the last folder used, if it still exists; else the
// user's default folder (`user_default`, from the burger menu), if set and present;
// else `builtin` (~/Desktop/chisel-sculpts). Also tells the Places pane which folder
// is the default, so call it as each dialog opens.
std::string file_browser_start_dir(const std::string& user_default, const std::string& builtin);

// A dialog returned this file (or folder): its folder becomes the last one used and
// goes to the top of Recent (unless `recent` is false: a default-folder pick is not a
// place you worked in). Written to disk at once.
void file_browser_note(const std::string& path, bool recent = true);

// The next free numbered copy of `path`: sculpt.chisel -> sculpt_001.chisel -> _002.
// `ext` is the extension to keep (stripped from `path` if it is there). A trailing _NNN
// is the counter, not part of the name, and existing files are skipped, so it never
// overwrites a previous version. Shared by the incremental save and the dialogs' "+".
// Works on web too (it just counts up there: nothing persists in MEMFS).
std::string next_incremental_path(const std::string& path, const std::string& ext);
