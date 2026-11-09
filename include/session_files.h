#pragma once
// Files the app keeps about your work between sessions, in the settings folder beside
// settings.cfg: the recent-projects list, and autosaves of unsaved work.
//
// Native only. On web every call is a no-op or comes back empty: the browser's files
// die with the tab, and browser storage was ruled out for project data.
//
// Autosave writes a normal .chisel (no undo history) on a background thread, so the
// only pause the user can feel is the copy of the scene the main thread hands over.
// One file per session, autosave-<date>-<time>-<pid>.chisel, overwritten each time
// via <name>.tmp + rename, so a crash mid-write never eats the previous copy. A session
// that ends unsaved leaves its file behind; the next launch offers it once (the file
// is renamed seen-autosave-... then), File > Recover lists them all, and only the
// newest 3 are kept.
#include <string>
#include <vector>

struct ProjectData;

// ---- Recent projects ----
void recents_add(const std::string& path);   // to the top, at most 8, written at once
std::vector<std::string> recents_list();     // newest first, only files that still exist

// ---- Autosave ----
// Native, a settings folder to write in, and CHISEL_NO_AUTOSAVE unset (scripted runs).
bool autosave_enabled();

struct AutosaveFile {
    std::string path;
    long long   mtime = 0;      // seconds since the epoch, for sorting
    char        label[32] = ""; // "Nov 09, 14:32", local time of the last write
    bool        seen = false;   // already offered at a startup
};
// Autosaves left by sessions that have ended, newest first. Live sessions' files
// (this one's, or another open window's) are never listed.
std::vector<AutosaveFile> autosave_leftovers();
// Marks every unseen leftover as offered and returns the newest, so the startup prompt
// asks about each file once. False when there was nothing new.
bool autosave_take_unseen(AutosaveFile& newest, int& others);

// Background writer. start() takes ownership of `proj` and deletes it once written;
// one write at a time (a call while busy just drops the snapshot).
bool autosave_busy();
void autosave_start(ProjectData* proj);
// True once per finished write, with how it went.
bool autosave_poll(bool& ok, double& secs, unsigned long long& bytes, std::string& path);
// This session's work was saved for real (or the app is closing clean): delete its
// autosave, and a recovered leftover with it. Waits for a write in flight first.
void autosave_drop_session();
// A leftover was opened: delete it once this session has its own copy (first own
// autosave, or a real save), not before, so a crash in between loses nothing.
void autosave_mark_recovered(const std::string& path);
// Waits for a write in flight. Call before the app exits.
void autosave_shutdown();
