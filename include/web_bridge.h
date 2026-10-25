#pragma once
// The browser side of the web build (web_bridge.cpp). Nothing here exists on native.
#ifdef __EMSCRIPTEN__
#include <string>

struct InputState;

// main()'s InputState, for the pagehide settings flush and the page's key filter.
extern InputState* g_settings_input;
// MEMFS paths the pickers' JS callbacks leave for the frame loop to consume.
extern std::string g_web_import_path;
extern std::string g_web_alpha_path;

void web_download_file(const char* fs_path, const char* download_name);
void web_open_file_picker();
void web_open_alpha_picker();
// Pointer, pen pressure, pagehide and key listeners. Once, before the main loop.
void web_install_dom_listeners();
#endif
