#include "file_browser.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

std::string next_incremental_path(const std::string& path, const std::string& ext) {
    std::string stem = path;
    if (!ext.empty() && stem.size() >= ext.size() &&
        stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
        stem.resize(stem.size() - ext.size());
    // A browser renames a colliding download to "bust_001 (1).chisel", and opening that
    // file makes it the current project. Strip the marker before reading the counter, or
    // "bust_001 (1)" parses as an un-numbered stem and the next save compounds into
    // "bust_001 (1)_001".
    if (!stem.empty() && stem.back() == ')') {
        size_t op = stem.find_last_of('(');
        if (op != std::string::npos && op + 1 < stem.size() - 1) {
            bool all_digits = true;
            for (size_t i = op + 1; i + 1 < stem.size(); i++)
                if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
            if (all_digits) {
                while (op > 0 && stem[op - 1] == ' ') op--;   // Chrome " (1)", Firefox "(1)"
                stem.resize(op);
            }
        }
    }
    int n = 0;
    const size_t us = stem.find_last_of('_');
    if (us != std::string::npos && us + 1 < stem.size()) {
        bool all_digits = true;
        for (size_t i = us + 1; i < stem.size(); i++)
            if (!std::isdigit((unsigned char)stem[i])) { all_digits = false; break; }
        if (all_digits) {
            n = std::atoi(stem.c_str() + us + 1);
            stem.resize(us);
        }
    }
    char suffix[16];
    for (int i = n + 1; i <= 9999; i++) {
        std::snprintf(suffix, sizeof(suffix), "_%03d", i);
        const std::string cand = stem + suffix + ext;
        std::error_code ec;
        if (!std::filesystem::exists(cand, ec)) return cand;
    }
    return stem + "_9999" + ext;   // series exhausted: overwrite the last one
}

#ifdef __EMSCRIPTEN__
IGFD::FileDialog* file_browser() { return nullptr; }
std::string file_browser_start_dir(const std::string&, const std::string& builtin) { return builtin; }
void file_browser_note(const std::string&, bool) {}
#else

#define IMGUI_DEFINE_MATH_OPERATORS
#include "settings.h"
#include "ui_icons.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuiFileDialog.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---- What is remembered: places.txt beside settings.cfg ---------------------------
// Not in settings.cfg: Reset settings should not throw away your bookmarks, and these
// change on every save, which would churn the settings blob's change check.
struct Store {
    bool loaded = false;
    std::string last;                    // folder of the last file a dialog returned
    std::vector<std::string> bookmarks;  // in the order added
    std::vector<std::string> recent;     // newest first
    std::vector<std::string> hidden;     // Places / Drives rows the user removed
};
Store g;
std::string g_default;                   // the default folder, set as each dialog opens
bool g_user_default = false;             // ...and the user picked it (burger menu), so it gets a Places row
constexpr size_t kMaxRecent = 6;

std::string store_path() {
    const std::string d = settings_dir();
    return d.empty() ? d : d + "/places.txt";
}

void store_load() {
    if (g.loaded) return;
    g.loaded = true;
    const std::string p = store_path();
    if (p.empty()) return;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t eq = line.find('=');
        if (eq == std::string::npos || eq + 1 >= line.size()) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "last") g.last = v;
        else if (k == "bookmark") g.bookmarks.push_back(v);
        else if (k == "recent" && g.recent.size() < kMaxRecent) g.recent.push_back(v);
        else if (k == "hidden") g.hidden.push_back(v);
    }
}

void store_save() {
    const std::string p = store_path();
    if (p.empty()) return;
    const std::string tmp = p + ".tmp";
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!g.last.empty()) f << "last=" << g.last << "\n";
        for (const auto& b : g.bookmarks) f << "bookmark=" << b << "\n";
        for (const auto& r : g.recent) f << "recent=" << r << "\n";
        for (const auto& h : g.hidden) f << "hidden=" << h << "\n";
        f.flush();
        if (!f.good()) { f.close(); fs::remove(tmp, ec); return; }
    }
    fs::rename(tmp, p, ec);
}

bool is_dir(const std::string& p) {
    std::error_code ec;
    return !p.empty() && fs::is_directory(p, ec);
}

// One spelling per folder, so "is this the folder we're in" is a string compare:
// absolute, no "." / "..", no trailing separator (except the root itself).
std::string norm(const std::string& p) {
    if (p.empty()) return p;
    std::error_code ec;
    fs::path a = fs::absolute(fs::path(p), ec);
    if (ec) a = fs::path(p);
    std::string s = a.lexically_normal().string();
    while (s.size() > 1 && (s.back() == '/' || s.back() == '\\') &&
           !(s.size() == 3 && s[1] == ':'))   // keep "C:\"
        s.pop_back();
    return s;
}

std::string leaf_name(const std::string& p) {
    std::string n = fs::path(p).filename().string();
    return n.empty() ? p : n;
}

std::string home_dir() {
#ifdef _WIN32
    const char* h = std::getenv("USERPROFILE");
#else
    const char* h = std::getenv("HOME");
#endif
    return (h && *h) ? std::string(h) : std::string();
}

struct Place { Icon icon; std::string label, path; };

// The home folders, by their real names: ~/.config/user-dirs.dirs on Linux (a
// localised desktop may call Desktop something else), the plain names otherwise.
std::vector<Place> home_places() {
    std::vector<Place> out;
    const std::string home = home_dir();
    if (home.empty()) return out;
    std::map<std::string, std::string> xdg;
#ifndef _WIN32
    {
        const char* cfg = std::getenv("XDG_CONFIG_HOME");
        std::ifstream f(((cfg && *cfg) ? std::string(cfg) : home + "/.config") + "/user-dirs.dirs");
        std::string line;
        while (std::getline(f, line)) {
            const size_t eq = line.find('=');
            if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            std::string v = line.substr(eq + 1);
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            if (v.compare(0, 5, "$HOME") == 0) v = home + v.substr(5);
            xdg[line.substr(0, eq)] = v;
        }
    }
#endif
    auto add = [&](Icon icon, const char* label, const char* key, const char* fallback) {
        auto it = xdg.find(key);
        std::string p = it != xdg.end() ? it->second : (fs::path(home) / fallback).string();
        // user-dirs.dirs points a disabled folder at $HOME itself: no second Home row.
        if (norm(p) != norm(home) && is_dir(p)) out.push_back({icon, label, norm(p)});
    };
    out.push_back({Icon::PlaceHome, "Home", norm(home)});
    add(Icon::PlaceDesktop,   "Desktop",   "XDG_DESKTOP_DIR",   "Desktop");
    add(Icon::PlaceDocuments, "Documents", "XDG_DOCUMENTS_DIR", "Documents");
    add(Icon::PlaceDownloads, "Downloads", "XDG_DOWNLOAD_DIR",  "Downloads");
    add(Icon::PlacePictures,  "Pictures",  "XDG_PICTURES_DIR",  "Pictures");
    return out;
}

// Drives. Linux: the root, and whatever is mounted where desktops mount removable and
// extra disks. Windows: the drive letters.
std::vector<Place> drive_places(IGFD::FileManager& fm) {
    std::vector<Place> out;
#ifdef _WIN32
    if (auto* fsys = fm.GetFileSystemInstance())
        for (const auto& d : fsys->GetDevicesList()) {
            std::string label = d.first;
            if (!d.second.empty()) label += "  " + d.second;
            out.push_back({Icon::PlaceDrive, label, d.first + "\\"});
        }
#else
    (void)fm;
    out.push_back({Icon::PlaceDrive, "Root", "/"});
    const char* user = std::getenv("USER");
    std::vector<std::string> roots = {"/mnt"};
    if (user && *user) {
        roots.push_back(std::string("/run/media/") + user);
        roots.push_back(std::string("/media/") + user);
    }
    for (const auto& r : roots) {
        std::error_code ec;
        if (!fs::is_directory(r, ec)) continue;
        for (const auto& e : fs::directory_iterator(r, fs::directory_options::skip_permission_denied, ec)) {
            std::error_code ec2;
            if (e.is_directory(ec2))
                out.push_back({Icon::PlaceDrive, e.path().filename().string(), e.path().string()});
        }
    }
#endif
    return out;
}

// Length of a file name without its extension: "bust_003.chisel" -> 8. A leading dot
// (".hidden") or no dot at all means the whole name.
size_t stem_length(const std::string& name) {
    const size_t dot = name.find_last_of('.');
    return (dot == std::string::npos || dot == 0) ? name.size() : dot;
}

// ---- The dialog ---------------------------------------------------------------------
class ChiselFileDialog : public IGFD::FileDialog {
public:
    void OpenDialog(const std::string& key, const std::string& title, const char* filters,
                    const IGFD::FileDialogConfig& cfg = {}) override {
        hist_.clear();
        hpos_ = -1;
        editing_ = false;
        focus_name_ = true;   // the name field has the keyboard from the first frame
        match_row_ = -1;
        m_FileDialogInternal.searchManager.Clear();
        store_load();
        // A recent folder since deleted or unmounted is dropped from the list shown.
        g.recent.erase(std::remove_if(g.recent.begin(), g.recent.end(),
                                      [](const std::string& r) { return !is_dir(r); }),
                       g.recent.end());
        // Folder lists are read once per opening, never per frame.
        homes_  = home_places();
        drives_ = drive_places(m_FileDialogInternal.fileManager);
        IGFD::FileDialog::OpenDialog(key, title, filters, cfg);
    }

protected:
    void m_DrawHeader() override;
    void m_DrawContent() override;
    bool m_DrawFooter() override;
    bool m_DrawOkButton() override;

    // Each row's name starts with a blank (see ImGuiFileDialogConfig.h); an icon for
    // what the entry is goes in it.
    bool m_Selectable(int row, const char* label, bool selected, ImGuiSelectableFlags flags,
                      const ImVec2& size) override {
        // While the name field points at a file, that row is the one lit, so the list
        // shows what Enter / Tab will take.
        if (match_row_ >= 0) selected = row == match_row_;
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const bool hit = IGFD::FileDialog::m_Selectable(row, label, selected, flags, size);
        if (row == match_row_ && scroll_match_) {
            ImGui::SetScrollHereY(0.5f);
            scroll_match_ = false;
        }
        auto info = fm().GetFilteredFileAt((size_t)row);
        if (info) {
            Icon icon = Icon::PlaceDocuments;
            if (info->fileNameExt == "..") icon = Icon::NavUp;
            else if (info->fileType.isDir()) icon = Icon::CmdOpen;
            else {
                std::string ext = fs::path(info->fileNameExt).extension().string();
                for (char& c : ext) c = (char)std::tolower((unsigned char)c);
                if (ext == ".chisel") icon = Icon::ModeSculpt;
                else if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".bmp")
                    icon = Icon::PlacePictures;
                else if (ext == ".obj" || ext == ".ply" || ext == ".stl" || ext == ".glb" || ext == ".gltf")
                    icon = Icon::ModeInsert;
            }
            const float h = ImGui::GetTextLineHeight();
            draw_icon(ImGui::GetWindowDrawList(), icon, ImVec2(pos.x + h * 0.55f, pos.y + h * 0.5f), h * 0.95f,
                      ImGui::GetColorU32(ImGuiCol_Text, info->fileType.isDir() ? 1.0f : 0.7f));
        }
        return hit;
    }

private:
    std::vector<std::string> hist_;   // folders visited since this opening
    int hpos_ = -1;                   // where in hist_ we are (back/forward move it)
    bool editing_ = false, edit_focus_ = false;
    int edit_grace_ = 0;              // frames before "not active" means clicked away
    double bad_until_ = 0.0;          // address bar flashes red until then
    char edit_[1024] = "";
    char newdir_[256] = "";
    std::vector<Place> homes_, drives_;

    // Type-ahead in the name field: the first file (in list order) whose name starts with
    // what is typed, case-blind. Found once per frame from the field's text, no allocation.
    bool focus_name_ = false;
    int match_row_ = -1;              // filtered-list row, -1 = none
    bool match_exact_ = false;        // typed text is the whole name
    bool scroll_match_ = false;       // the match moved: bring its row into view
    bool caret_at_end_ = false;       // caret after the last letter, nothing selected

    IGFD::FileManager& fm() { return m_FileDialogInternal.fileManager; }
    // Save and export dialogs are the ones that ask before overwriting.
    bool is_save() {
        return (m_FileDialogInternal.getDialogConfig().flags & ImGuiFileDialogFlags_ConfirmOverwrite) != 0;
    }
    void update_match();
    std::shared_ptr<IGFD::FileInfos> match() {
        return match_row_ >= 0 ? fm().GetFilteredFileAt((size_t)match_row_) : nullptr;
    }
    static int name_callback(ImGuiInputTextCallbackData* d);
    std::string numbered_name();
    std::string cur() { return norm(fm().GetCurrentPath()); }

    bool go(const std::string& path) {
        if (!is_dir(path)) return false;
        fm().SetCurrentPath(norm(path));
        fm().OpenCurrentPath(m_FileDialogInternal);
        return true;
    }
    void start_edit() {
        editing_ = true;
        edit_focus_ = true;
        edit_grace_ = 2;
        IGFD::Utils::SetBuffer(edit_, sizeof(edit_), cur());
    }
    // Text typed into the address bar: a folder goes there; a file in an existing folder
    // goes to the folder and puts the name in the file field (handy for Save).
    bool accept_typed(std::string t) {
        while (!t.empty() && (t.back() == ' ' || t.back() == '\t')) t.pop_back();
        if (!t.empty() && t[0] == '~') t = home_dir() + t.substr(1);
        if (go(t)) return true;
        const fs::path p(t);
        if (p.has_parent_path() && is_dir(p.parent_path().string()) && go(p.parent_path().string())) {
            IGFD::Utils::SetBuffer(fm().fileNameBuffer, MAX_FILE_DIALOG_NAME_BUFFER, p.filename().string());
            return true;
        }
        return false;
    }

    bool icon_button(const char* id, Icon icon, const char* tip, bool enabled = true);
    void draw_address_bar(float width);
    void draw_places(ImVec2 size);
    bool place_row(Icon icon, const std::string& label, const std::string& path, const std::string& here);
};

bool ChiselFileDialog::icon_button(const char* id, Icon icon, const char* tip, bool enabled) {
    const float h = ImGui::GetFrameHeight();
    if (!enabled) ImGui::BeginDisabled();
    const bool hit = ImGui::Button(id, ImVec2(h, h));
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    draw_icon(ImGui::GetWindowDrawList(), icon, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f),
              h * 0.7f, ImGui::GetColorU32(enabled ? ImGuiCol_Text : ImGuiCol_TextDisabled));
    if (!enabled) ImGui::EndDisabled();
    if (tip && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", tip);
    return hit && enabled;
}

void ChiselFileDialog::m_DrawHeader() {
    const std::string here = cur();
    // History: any change of folder we did not make with back/forward is a new step
    // (a double-click into a folder in the list, a place, the address bar).
    if (hpos_ < 0 || hist_[(size_t)hpos_] != here) {
        hist_.resize((size_t)(hpos_ + 1));
        hist_.push_back(here);
        hpos_ = (int)hist_.size() - 1;
    }

    // Keys: Alt+Left / Alt+Right / Alt+Up, Ctrl+L to type a path. Only while no text
    // field has the keyboard, so they never eat a keystroke meant for a name.
    const bool typing = ImGui::GetIO().WantTextInput;
    const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    const bool key_back = focused && !typing && ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_LeftArrow);
    const bool key_fwd  = focused && !typing && ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_RightArrow);
    const bool key_up   = focused && !typing && ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_UpArrow);
    if (focused && !editing_ && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_L)) start_edit();

    const fs::path parent = fs::path(here).parent_path();
    const bool has_up = !parent.empty() && norm(parent.string()) != here;

    if (icon_button("##back", Icon::NavBack, "Back (Alt+Left)", hpos_ > 0) || (key_back && hpos_ > 0)) {
        // A folder deleted meanwhile is skipped, not a dead end.
        while (hpos_ > 0 && !go(hist_[(size_t)--hpos_])) {}
    }
    ImGui::SameLine(0, 2);
    const bool can_fwd = hpos_ + 1 < (int)hist_.size();
    if (icon_button("##fwd", Icon::NavForward, "Forward (Alt+Right)", can_fwd) || (key_fwd && can_fwd)) {
        while (hpos_ + 1 < (int)hist_.size() && !go(hist_[(size_t)++hpos_])) {}
    }
    ImGui::SameLine(0, 2);
    if (icon_button("##up", Icon::NavUp, "Up one folder (Alt+Up)", has_up) || (key_up && has_up))
        go(parent.string());
    ImGui::SameLine();

    const ImGuiStyle& st = ImGui::GetStyle();
    const float search_w = ImGui::GetFontSize() * 10.0f;
    const bool dir_create = !(m_FileDialogInternal.getDialogConfig().flags &
                              ImGuiFileDialogFlags_DisableCreateDirectoryButton);
    float right = search_w + st.ItemSpacing.x;
    if (dir_create) right += ImGui::GetFrameHeight() + st.ItemSpacing.x;
    draw_address_bar(std::max(80.0f, ImGui::GetContentRegionAvail().x - right));

    if (dir_create) {
        ImGui::SameLine();
        if (icon_button("##newdir", Icon::NewFolder, "New folder here")) {
            newdir_[0] = 0;
            ImGui::OpenPopup("##newdirpop");
        }
        if (ImGui::BeginPopup("##newdirpop")) {
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 14.0f);
            bool make = ImGui::InputTextWithHint("##newdirname", "Folder name", newdir_, sizeof(newdir_),
                                                 ImGuiInputTextFlags_EnterReturnsTrue);
            if (ImGui::IsItemActive()) m_FileDialogInternal.searchManager.searchInputIsActive = true;
            ImGui::SameLine();
            make |= ImGui::Button("Create");
            if (make && newdir_[0]) {
                std::error_code ec;
                const fs::path p = fs::path(here) / newdir_;
                if (fs::create_directory(p, ec) || is_dir(p.string())) go(p.string());
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    ImGui::SameLine();
    auto& sm = m_FileDialogInternal.searchManager;
    ImGui::SetNextItemWidth(search_w);
    if (ImGui::InputTextWithHint("##search", "Search", sm.searchBuffer, MAX_FILE_DIALOG_NAME_BUFFER)) {
        sm.searchTag = sm.searchBuffer;
        fm().ApplyFilteringOnFileList(m_FileDialogInternal);
    }
    // Tells the dialog Esc belongs to this field (clears it), not to closing.
    if (ImGui::IsItemActive()) sm.searchInputIsActive = true;
}

// The folders of the current path as buttons with chevrons between; a click on one goes
// there, a click on the bar's empty end (or Ctrl+L) turns it into a text field.
void ChiselFileDialog::draw_address_bar(float width) {
    const ImGuiStyle& st = ImGui::GetStyle();
    const float h = ImGui::GetFrameHeight();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const bool bad = ImGui::GetTime() < bad_until_;

    if (editing_) {
        if (edit_focus_) { ImGui::SetKeyboardFocusHere(); }
        ImGui::SetNextItemWidth(width);
        if (bad) ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.55f, 0.12f, 0.12f, 1.0f));
        const bool enter = ImGui::InputText("##address", edit_, sizeof(edit_),
                                            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        if (bad) ImGui::PopStyleColor();
        const bool active = ImGui::IsItemActive();
        if (active) m_FileDialogInternal.searchManager.searchInputIsActive = true;
        if (enter) {
            if (accept_typed(edit_)) editing_ = false;
            else { bad_until_ = ImGui::GetTime() + 0.6; edit_focus_ = true; edit_grace_ = 2; return; }
        } else if (edit_grace_ <= 0 && !active) {
            editing_ = false;   // clicked away or Esc
        }
        edit_focus_ = false;
        if (edit_grace_ > 0) edit_grace_--;
        return;
    }

    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + width, p0.y + h);
    dl->AddRectFilled(p0, p1, ImGui::GetColorU32(ImGuiCol_FrameBg), st.FrameRounding);

    // Crumbs: (label, folder). The root is its own crumb; on Windows "C:" + "\" is one.
    struct Crumb { std::string label, path; float w; };
    std::vector<Crumb> crumbs;
    fs::path acc;
    for (const auto& part : fs::path(cur())) {
        acc /= part;
        const std::string s = part.string();
        if (s.empty()) continue;
        if (s == "/" || s == "\\") {
            if (crumbs.empty()) crumbs.push_back({"/", acc.string(), 0});
            else crumbs.back().path = acc.string();
            continue;
        }
        crumbs.push_back({s, acc.string(), 0});
    }
    const float pad = st.FramePadding.x * 0.75f;
    const float chev = ImGui::GetFontSize() * 0.75f;
    float total = 0;
    for (auto& c : crumbs) { c.w = ImGui::CalcTextSize(c.label.c_str()).x + pad * 2; total += c.w + chev; }
    // Too long: drop folders from the front, a "..." crumb stands in for them.
    const float dots_w = ImGui::CalcTextSize("...").x + pad * 2 + chev;
    const float room = width - st.FramePadding.x * 2 - h;   // keep a click strip at the end
    size_t first = 0;
    while (first + 1 < crumbs.size() && total > room) {
        total -= crumbs[first].w + chev;
        if (first == 0) total += dots_w;
        first++;
    }

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(pad, st.FramePadding.y));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::SetCursorScreenPos(ImVec2(p0.x + st.FramePadding.x * 0.5f, p0.y));
    const ImU32 dim = ImGui::GetColorU32(ImGuiCol_Text, 0.6f);
    auto chevron = [&]() {
        const ImVec2 c = ImGui::GetCursorScreenPos();
        const float cx = c.x + chev * 0.5f, cy = c.y + h * 0.5f, r = chev * 0.18f;
        const ImVec2 pts[3] = {ImVec2(cx - r * 0.6f, cy - r * 1.4f), ImVec2(cx + r * 0.8f, cy),
                               ImVec2(cx - r * 0.6f, cy + r * 1.4f)};
        dl->AddPolyline(pts, 3, dim, ImDrawFlags_None, 1.5f);
        ImGui::Dummy(ImVec2(chev, h));
        ImGui::SameLine();
    };
    if (first > 0) {
        if (ImGui::Button("...##crumbmore", ImVec2(0, h))) start_edit();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Type a path (Ctrl+L)");
        ImGui::SameLine();
        chevron();
    }
    std::string go_to;
    for (size_t i = first; i < crumbs.size(); i++) {
        const bool last = i + 1 == crumbs.size();
        ImGui::PushID((int)i);
        if (!last) {
            ImVec4 t = ImGui::GetStyleColorVec4(ImGuiCol_Text);
            t.w *= 0.75f;
            ImGui::PushStyleColor(ImGuiCol_Text, t);
        }
        if (ImGui::Button(crumbs[i].label.c_str(), ImVec2(0, h))) go_to = crumbs[i].path;
        if (!last) ImGui::PopStyleColor();
        ImGui::PopID();
        ImGui::SameLine();
        if (!last) chevron();
    }
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    // The rest of the bar: click to type.
    const float rest = p1.x - ImGui::GetCursorScreenPos().x;
    if (rest > 1.0f) {
        if (ImGui::InvisibleButton("##addrrest", ImVec2(rest, h))) start_edit();
        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
    }
    ImGui::SetCursorScreenPos(ImVec2(p1.x, p0.y));
    ImGui::Dummy(ImVec2(0, h));
    if (!go_to.empty()) go(go_to);
}

bool ChiselFileDialog::place_row(Icon icon, const std::string& label, const std::string& path,
                                 const std::string& here) {
    const float h = ImGui::GetFrameHeight();
    ImGui::PushID(path.c_str());
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::Selectable("##place", here == path, ImGuiSelectableFlags_None, ImVec2(0, h));
    const float right = ImGui::GetItemRectMax().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImU32 col = ImGui::GetColorU32(ImGuiCol_Text);
    draw_icon(dl, icon, ImVec2(pos.x + h * 0.5f, pos.y + h * 0.5f), h * 0.68f, col);
    const ImVec2 tp(pos.x + h + 2.0f, pos.y + (h - ImGui::GetTextLineHeight()) * 0.5f);
    dl->PushClipRect(ImVec2(tp.x, pos.y), ImVec2(right - 2.0f, pos.y + h), true);
    dl->AddText(tp, col, label.c_str());
    dl->PopClipRect();
    ImGui::PopID();
    return clicked;
}

void ChiselFileDialog::draw_places(ImVec2 size) {
    ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_FrameBg);
    bg.w *= 0.6f;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg);
    ImGui::BeginChild("##chiselplaces", size, ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleColor();
    const std::string here = cur();
    std::string go_to;
    auto header = [](const char* t) {
        ImGui::Dummy(ImVec2(0, 2));
        ImGui::TextDisabled("%s", t);
    };

    // Places and Drives rows can be removed (right-click); "Reset" brings them all back.
    auto is_hidden = [](const std::string& path) {
        return std::find(g.hidden.begin(), g.hidden.end(), path) != g.hidden.end();
    };
    std::string hide;
    auto removable_row = [&](Icon icon, const std::string& label, const std::string& path) {
        if (is_hidden(path)) return;
        // The menu goes under the row's own ID: one shared popup ID opens every row's
        // menu at once on a single right-click.
        ImGui::PushID(path.c_str());
        if (place_row(icon, label, path, here)) go_to = path;
        if (ImGui::BeginPopupContextItem("##plctx")) {
            if (ImGui::Selectable("Remove from Places")) hide = path;
            ImGui::EndPopup();
        }
        ImGui::PopID();
    };

    header("Places");
    if (!g.hidden.empty()) {
        const float bw = ImGui::CalcTextSize("Reset").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SameLine(ImGui::GetWindowWidth() - bw - ImGui::GetStyle().WindowPadding.x);
        if (ImGui::SmallButton("Reset##plreset")) { g.hidden.clear(); store_save(); }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Show the %d removed place%s again", (int)g.hidden.size(), g.hidden.size() == 1 ? "" : "s");
    }
    ImGui::PushID("homes");
    for (const auto& p : homes_) removable_row(p.icon, p.label, p.path);
    // The default folder gets a row only when the user chose one; the built-in fallback
    // is just where dialogs start, not a place of its own.
    bool default_listed = false;
    for (const auto& p : homes_) default_listed |= p.path == g_default;
    if (g_user_default && !g_default.empty() && !default_listed)
        removable_row(Icon::ModeSculpt, leaf_name(g_default), g_default);
    ImGui::PopID();

    // Bookmarks: + adds the folder you are in; right-click one to remove it.
    header("Bookmarks");
    {
        const bool have = std::find(g.bookmarks.begin(), g.bookmarks.end(), here) != g.bookmarks.end();
        const float bw = ImGui::CalcTextSize("+").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SameLine(ImGui::GetWindowWidth() - bw - ImGui::GetStyle().WindowPadding.x);
        if (have) ImGui::BeginDisabled();
        if (ImGui::SmallButton("+##addbm")) { g.bookmarks.push_back(here); store_save(); }
        if (have) ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(have ? "This folder is bookmarked" : "Bookmark this folder");
    }
    ImGui::PushID("bookmarks");
    int drop = -1;
    for (size_t i = 0; i < g.bookmarks.size(); i++) {
        const std::string& b = g.bookmarks[i];
        ImGui::PushID((int)i);
        if (place_row(Icon::PlaceBookmark, leaf_name(b), b, here)) go_to = b;
        if (ImGui::BeginPopupContextItem("##bmctx")) {
            if (ImGui::Selectable("Remove bookmark")) drop = (int)i;
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    if (drop >= 0) { g.bookmarks.erase(g.bookmarks.begin() + drop); store_save(); }
    if (g.bookmarks.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("+ adds the folder you're in.");
        ImGui::PopStyleColor();
    }

    header("Drives");
    ImGui::PushID("drives");
    for (const auto& p : drives_) removable_row(p.icon, p.label, p.path);
    ImGui::PopID();
    if (!hide.empty()) { g.hidden.push_back(hide); store_save(); }

    // Recent: folders a file was recently opened from or saved to.
    if (!g.recent.empty()) {
        header("Recent");
        ImGui::PushID("recent");
        drop = -1;
        for (size_t i = 0; i < g.recent.size(); i++) {
            const std::string& r = g.recent[i];
            ImGui::PushID((int)i);
            if (place_row(Icon::PlaceRecent, leaf_name(r), r, here)) go_to = r;
            if (ImGui::BeginPopupContextItem("##rcctx")) {
                if (ImGui::Selectable("Remove from Recent")) drop = (int)i;
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        ImGui::PopID();
        if (drop >= 0) { g.recent.erase(g.recent.begin() + drop); store_save(); }
    }

    ImGui::EndChild();
    if (!go_to.empty()) go(go_to);
}

void ChiselFileDialog::update_match() {
    const int prev = match_row_;
    match_row_ = -1;
    match_exact_ = false;
    auto& f = fm();
    const char* typed = f.fileNameBuffer;
    const size_t n = strlen(typed);
    if (n > 0 && !f.dLGDirectoryMode) {
        char low[MAX_FILE_DIALOG_NAME_BUFFER];
        for (size_t i = 0; i <= n; i++) low[i] = (char)std::tolower((unsigned char)typed[i]);
        const size_t count = f.GetFilteredListSize();
        for (size_t i = 0; i < count; i++) {
            auto info = f.GetFilteredFileAt(i);
            if (!info || info->fileType.isDir()) continue;
            const std::string& o = info->fileNameExt_optimized;   // already lower-case
            if (o.size() < n || o.compare(0, n, low) != 0) continue;
            if (o.size() == n) { match_row_ = (int)i; match_exact_ = true; break; }
            if (match_row_ < 0) match_row_ = (int)i;
        }
    }
    if (match_row_ != prev && match_row_ >= 0) scroll_match_ = true;
}

// Tab takes the suggestion; every frame also notes where the caret is, since the grey
// suggestion only makes sense with the caret at the end of the text.
int ChiselFileDialog::name_callback(ImGuiInputTextCallbackData* d) {
    auto* self = static_cast<ChiselFileDialog*>(d->UserData);
    if (d->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
        self->caret_at_end_ = d->CursorPos == d->BufTextLen && !d->HasSelection();
    } else if (d->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
        auto m = self->match();
        if (!m) return 0;
        // The match was found from last frame's text: check it still fits what is typed.
        const std::string& o = m->fileNameExt_optimized;
        const size_t stem = stem_length(m->fileNameExt);
        if ((size_t)d->BufTextLen >= stem) return 0;
        for (int i = 0; i < d->BufTextLen; i++)
            if ((char)std::tolower((unsigned char)d->Buf[i]) != o[(size_t)i]) return 0;
        // The name only, never the dot and extension: the dialog adds its own on save, and
        // the caret lands right after the last digit, so "_003" -> "_004" is one keystroke.
        d->DeleteChars(0, d->BufTextLen);
        d->InsertChars(0, m->fileNameExt.c_str(), m->fileNameExt.c_str() + stem);
    }
    return 0;
}

// The library's footer, with the name field taking the keyboard as the dialog opens,
// the type-ahead, and the grey rest-of-the-name suggestion.
bool ChiselFileDialog::m_DrawFooter() {
    auto& f = fm();
    const float posY = ImGui::GetCursorPos().y;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(f.dLGDirectoryMode ? "Directory Path:" : "File Name:");
    ImGui::SameLine();

    const ImGuiStyle& st = ImGui::GetStyle();
    float width = ImGui::GetContentRegionAvail().x;
    if (!f.dLGDirectoryMode)
        width -= m_FileDialogInternal.filterManager.GetFilterComboBoxWidth() + st.ItemSpacing.x;

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll |
                                ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackAlways;
    if (m_FileDialogInternal.getDialogConfig().flags & ImGuiFileDialogFlags_ReadOnlyFileNameField)
        flags |= ImGuiInputTextFlags_ReadOnly;
    if (focus_name_) {
        ImGui::SetKeyboardFocusHere();
        focus_name_ = false;
    }
    ImGui::SetNextItemWidth(width);
    if (ImGui::InputText("##FileName", f.fileNameBuffer, MAX_FILE_DIALOG_NAME_BUFFER, flags,
                         &ChiselFileDialog::name_callback, this)) {
        // Open: Enter takes the file the letters point at. Save: what you typed is the
        // name (Tab takes the suggestion), so a few letters never overwrite a file.
        if (!is_save() && !match_exact_)
            if (auto m = match()) IGFD::Utils::SetBuffer(f.fileNameBuffer, MAX_FILE_DIALOG_NAME_BUFFER, m->fileNameExt);
        m_FileDialogInternal.isOk = true;
    }
    // Not flagged as fileInputIsActive: that only makes Esc leave the field instead of
    // closing the dialog, and this field has the keyboard from the start.
    const bool active = ImGui::IsItemActive();

    if (active && caret_at_end_ && !match_exact_) {
        if (auto m = match()) {
            const size_t n = strlen(f.fileNameBuffer);
            const size_t stem = stem_length(m->fileNameExt);   // grey = exactly what Tab adds
            if (stem > n) {
                const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
                const float x = a.x + st.FramePadding.x + ImGui::CalcTextSize(f.fileNameBuffer).x;
                const float right = b.x - st.FramePadding.x;
                if (x < right) {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->PushClipRect(a, ImVec2(right, b.y), true);
                    dl->AddText(ImVec2(x, a.y + st.FramePadding.y), ImGui::GetColorU32(ImGuiCol_TextDisabled),
                                m->fileNameExt.c_str() + n, m->fileNameExt.c_str() + stem);
                    dl->PopClipRect();
                }
            }
        }
    }

    m_FileDialogInternal.filterManager.DrawFilterComboBox(m_FileDialogInternal);
    const bool res = m_DrawValidationButtons();
    m_FileDialogInternal.footerHeight = ImGui::GetCursorPosY() - posY;
    return res;
}

// The name the "+" would save to: the next free _NNN of what is in the name field,
// keeping its extension, or the dialog's own when none was typed.
std::string ChiselFileDialog::numbered_name() {
    std::string name = fm().fileNameBuffer;
    std::string ext = fs::path(name).extension().string();
    if (ext.empty()) {
        ext = m_FileDialogInternal.filterManager.GetSelectedFilter().getFirstFilter();
        if (ext.empty() || ext[0] != '.' || ext.find_first_of("*?") != std::string::npos) ext.clear();
    }
    const std::string next = next_incremental_path((fs::path(cur()) / name).string(), ext);
    return fs::path(next).filename().string();
}

// OK, then (save / export only) "+": save as the next numbered copy, never overwriting.
bool ChiselFileDialog::m_DrawOkButton() {
    if (IGFD::FileDialog::m_DrawOkButton()) return true;
    if (!is_save() || !fm().fileNameBuffer[0]) return false;
    const float h = ImGui::GetFrameHeight();
    if (ImGui::Button("+##savenumbered", ImVec2(h, h))) {
        IGFD::Utils::SetBuffer(fm().fileNameBuffer, MAX_FILE_DIALOG_NAME_BUFFER, numbered_name());
        m_FileDialogInternal.isOk = true;
        ImGui::SameLine();
        return true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save a numbered copy: %s", numbered_name().c_str());
    ImGui::SameLine();
    return false;
}

void ChiselFileDialog::m_DrawContent() {
    update_match();
    // The library's own layout, minus its Places pane (ours replaces it).
    ImVec2 size = ImGui::GetContentRegionAvail() - ImVec2(0.0f, m_FileDialogInternal.footerHeight);
    const float places_w = std::min(ImGui::GetFontSize() * 11.0f, size.x * 0.35f);
    draw_places(ImVec2(places_w, size.y));
    ImGui::SameLine();

    auto& cfg = m_FileDialogInternal.getDialogConfigRef();
    size.x = ImGui::GetContentRegionAvail().x - cfg.sidePaneWidth;
    if (cfg.sidePane) {
        ImGui::PushID("splittersidepane");
        IGFD::Utils::ImSplitter(true, 4.0f, &size.x, &cfg.sidePaneWidth, 10.0f, 10.0f, size.y);
        ImGui::PopID();
    }
    m_DrawFileListView(size);
    if (cfg.sidePane) m_DrawSidePane(size.y);
}

}  // namespace

IGFD::FileDialog* file_browser() {
    static ChiselFileDialog dlg;
    return &dlg;
}

std::string file_browser_start_dir(const std::string& user_default, const std::string& builtin) {
    store_load();
    g_user_default = is_dir(user_default);
    g_default = norm(g_user_default ? user_default : builtin);
    if (is_dir(g.last)) return g.last;
    return g_default;
}

void file_browser_note(const std::string& path, bool recent) {
    if (path.empty()) return;
    store_load();
    const std::string dir = is_dir(path) ? norm(path) : norm(fs::path(path).parent_path().string());
    if (!is_dir(dir)) return;
    g.last = dir;
    if (!recent) { store_save(); return; }
    g.recent.erase(std::remove(g.recent.begin(), g.recent.end(), dir), g.recent.end());
    g.recent.insert(g.recent.begin(), dir);
    if (g.recent.size() > kMaxRecent) g.recent.resize(kMaxRecent);
    store_save();
}

#endif  // !__EMSCRIPTEN__
