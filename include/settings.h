#pragma once

// Persisted user settings — the look/feel preferences that should survive a restart.
//
// Two sinks behind one serializer: on web the blob goes to localStorage (the tab's
// MEMFS dies with the page, and IDBFS would need an async FS.syncfs at both ends for
// under a kilobyte of scalars); on native it goes to a config file under XDG_CONFIG_HOME
// / %APPDATA%. Only the sink is #ifdef'd, the format is identical on both.
//
// What is saved: brush feel (per profile — see ProfileSettings), plus the single-valued
// look/scene preferences (matcap, paint colours, mirror, density mults). What is NOT
// saved: anything scene- or frame-scoped — subdiv_level, interaction_mode, mesh_locked,
// the dialog and *_requested flags. Persisting those would restore the app into a mode
// the scene does not support.
//
// Format is versioned flat text, `key=value` under `[section]` headers. The parser
// ignores unknown keys and leaves missing ones at their InputState() default, so adding
// a setting never invalidates an existing user's stored blob and never needs a migration.

struct InputState;

// Read the stored blob and apply it over `input`. Absent/unreadable/corrupt storage is
// not an error — it just leaves the constructor defaults in place. Call once, right
// after InputState is constructed and before anything reads its fields.
void settings_load(InputState& input);

// Serialize `input` and write it to the sink. Flushes the active profile first, so the
// caller does not have to. Failures are silent by design: a browser that refuses
// localStorage (see below) must not break sculpting.
void settings_save(InputState& input);

// Per-frame hook. Snapshots the persisted subset, and when it has changed, schedules a
// write ~1s later — so dragging a slider writes once when it settles instead of once per
// frame. Never writes mid-stroke. `dt` is the frame delta in seconds.
void settings_tick(InputState& input, float dt);

// Drop the stored blob and reset every persisted field to its default. This is the
// escape hatch for a bad persisted value: before persistence every bad state was one
// restart away from gone, and it no longer is.
void settings_reset(InputState& input);

// True when the sink rejected us — private-mode/partitioned localStorage, or an
// unwritable config dir. Settings still work for the session, they just will not
// survive it. The burger menu shows this so the user is not left wondering.
//
// On itch the game runs in a cross-origin iframe served from html-classic.itch.zone.
// Chrome and Firefox partition third-party storage (it works, keyed to the itch.io
// top-level site); Safari may refuse it outright. Hence the check rather than an assume.
bool settings_storage_available();
