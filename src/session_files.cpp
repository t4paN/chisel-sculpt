#include "session_files.h"
#include "project_file.h"
#include "settings.h"

#ifdef __EMSCRIPTEN__

void recents_add(const std::string&) {}
std::vector<std::string> recents_list() { return {}; }
bool autosave_enabled() { return false; }
std::vector<AutosaveFile> autosave_leftovers() { return {}; }
bool autosave_busy() { return false; }
void autosave_start(ProjectData* proj) { delete proj; }
bool autosave_poll(bool&, double&, unsigned long long&, std::string&) { return false; }
void autosave_drop_session() {}
void autosave_mark_recovered(const std::string&) {}
void autosave_shutdown() {}

#else

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

// Last, after every project header: windows.h defines macros (near, far, ...) that
// would bite code included after it.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <signal.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

unsigned long my_pid() {
#ifdef _WIN32
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

// Whether the session that wrote a file is still running. A reused pid reads as alive,
// which only hides that file until the other process ends: the safe direction.
bool pid_alive(unsigned long pid) {
    if (pid == 0) return false;
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    DWORD code = 0;
    const bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    return kill((pid_t)pid, 0) == 0 || errno == EPERM;
#endif
}

std::string autosave_dir() {
    std::string d = settings_dir();
    if (d.empty()) return d;
    d += "/autosave";
    std::error_code ec;
    fs::create_directories(d, ec);
    return ec ? std::string() : d;
}

// This session's file name, fixed at the first write.
const std::string& session_path() {
    static std::string p;
    if (p.empty()) {
        const std::string d = autosave_dir();
        if (d.empty()) return p;
        std::time_t t = std::time(nullptr);
        char stamp[32];
        std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&t));
        p = d + "/autosave-" + stamp + "-" + std::to_string(my_pid()) + ".chisel";
    }
    return p;
}

// autosave-...-<pid>.chisel → pid (0 = not ours). seen-autosave-... is what the old
// startup offer renamed files to; still read, so those stay listed and rotated.
unsigned long pid_of(const std::string& name) {
    const std::string ext = ".chisel";
    if (name.size() <= ext.size() || name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
        return 0;
    if (name.rfind("autosave-", 0) != 0 && name.rfind("seen-autosave-", 0) != 0) return 0;
    const std::string stem = name.substr(0, name.size() - ext.size());
    const size_t dash = stem.find_last_of('-');
    if (dash == std::string::npos || dash + 1 >= stem.size()) return 0;
    for (size_t i = dash + 1; i < stem.size(); i++)
        if (stem[i] < '0' || stem[i] > '9') return 0;
    return std::strtoul(stem.c_str() + dash + 1, nullptr, 10);
}

long long mtime_of(const fs::path& p) {
    std::error_code ec;
    const auto ft = fs::last_write_time(p, ec);
    if (ec) return 0;
    // C++17 has no file_clock -> system_clock cast; shift by "now" on both clocks.
    const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return (long long)std::chrono::system_clock::to_time_t(sys);
}

// Every autosave in the folder, live sessions' included, newest first.
std::vector<AutosaveFile> list_all(bool include_live) {
    std::vector<AutosaveFile> out;
    const std::string d = autosave_dir();
    if (d.empty()) return out;
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(d, ec)) {
        const std::string name = e.path().filename().string();
        const unsigned long pid = pid_of(name);
        if (!pid) continue;
        if (!include_live && (pid == my_pid() || pid_alive(pid))) continue;
        AutosaveFile f;
        f.path = e.path().string();
        f.mtime = mtime_of(e.path());
        std::time_t t = (std::time_t)f.mtime;
        std::strftime(f.label, sizeof f.label, "%b %d, %H:%M", std::localtime(&t));
        out.push_back(f);
    }
    std::sort(out.begin(), out.end(),
              [](const AutosaveFile& a, const AutosaveFile& b) { return a.mtime > b.mtime; });
    return out;
}

// Keep the newest 3 files of ended sessions; live sessions' files are never touched.
// Also clears .tmp halves that a crash mid-write left behind.
void rotate() {
    std::vector<AutosaveFile> ended = list_all(false);
    for (size_t i = 3; i < ended.size(); i++) {
        std::error_code ec;
        fs::remove(ended[i].path, ec);
    }
    const std::string d = autosave_dir();
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(d, ec)) {
        std::string name = e.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".tmp") != 0) continue;
        const unsigned long pid = pid_of(name.substr(0, name.size() - 4));
        if (pid && pid != my_pid() && !pid_alive(pid)) {
            std::error_code ec2;
            fs::remove(e.path(), ec2);
        }
    }
}

struct Job {
    ProjectData*       proj = nullptr;
    std::string        path;
    std::atomic<bool>  done{false};
    bool               ok = false;
    double             secs = 0.0;
    unsigned long long bytes = 0;
    std::thread        th;
};
Job*        g_job = nullptr;
bool        g_session_written = false;   // this session's file exists on disk
std::string g_recovered;                 // a leftover opened this session, not yet deleted

void join_job() {
    if (g_job && g_job->th.joinable()) g_job->th.join();
}

const char* recents_file_name = "/recent.txt";

std::vector<std::string> recents_read() {
    std::vector<std::string> out;
    const std::string d = settings_dir();
    if (d.empty()) return out;
    std::ifstream f(d + recents_file_name);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

} // namespace

void recents_add(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    std::string abs = fs::absolute(path, ec).string();
    if (ec) abs = path;
    // Never list an autosave: those live under File > Recover.
    if (pid_of(fs::path(abs).filename().string())) return;
    std::vector<std::string> list = recents_read();
    list.erase(std::remove(list.begin(), list.end(), abs), list.end());
    list.insert(list.begin(), abs);
    if (list.size() > 8) list.resize(8);
    const std::string d = settings_dir();
    if (d.empty()) return;
    const std::string target = d + recents_file_name, tmp = target + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        for (const std::string& s : list) f << s << "\n";
        f.flush();
        if (!f.good()) { f.close(); fs::remove(tmp, ec); return; }
    }
    fs::rename(tmp, target, ec);
}

std::vector<std::string> recents_list() {
    std::vector<std::string> out;
    for (const std::string& s : recents_read()) {
        std::error_code ec;
        if (fs::exists(s, ec)) out.push_back(s);
    }
    return out;
}

bool autosave_enabled() {
    static int on = -1;
    if (on < 0) {
        const char* off = std::getenv("CHISEL_NO_AUTOSAVE");
        on = (off && off[0] == '1') ? 0 : !autosave_dir().empty();
    }
    return on == 1;
}

std::vector<AutosaveFile> autosave_leftovers() {
    if (!autosave_enabled()) return {};
    return list_all(false);
}

bool autosave_busy() { return g_job != nullptr; }

void autosave_start(ProjectData* proj) {
    const std::string& path = session_path();
    if (g_job || path.empty()) { delete proj; return; }
    Job* j = new Job;
    j->proj = proj;
    j->path = path;
    g_job = j;
    j->th = std::thread([j]() {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string tmp = j->path + ".tmp";
        const SaveResult r = save_project(tmp.c_str(), *j->proj);
        delete j->proj;          // the copy is the big part; free it before the rename
        j->proj = nullptr;
        std::error_code ec;
        bool ok = r == SaveResult::OK;
        if (ok) {
            j->bytes = (unsigned long long)fs::file_size(tmp, ec);
            fs::rename(tmp, j->path, ec);   // replaces the previous copy in one step
            ok = !ec;
        }
        if (!ok) fs::remove(tmp, ec);
        j->secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        j->ok = ok;
        j->done.store(true, std::memory_order_release);
    });
}

bool autosave_poll(bool& ok, double& secs, unsigned long long& bytes, std::string& path) {
    if (!g_job || !g_job->done.load(std::memory_order_acquire)) return false;
    join_job();
    ok = g_job->ok; secs = g_job->secs; bytes = g_job->bytes; path = g_job->path;
    delete g_job;
    g_job = nullptr;
    if (ok) {
        g_session_written = true;
        if (!g_recovered.empty()) {   // this session holds the work now
            std::error_code ec;
            fs::remove(g_recovered, ec);
            g_recovered.clear();
        }
        rotate();
    }
    return true;
}

void autosave_drop_session() {
    if (!autosave_enabled()) return;
    if (g_job) {   // a write in flight would land after the delete below
        join_job();
        if (g_job->ok) g_session_written = true;
        delete g_job->proj;
        delete g_job;
        g_job = nullptr;
    }
    std::error_code ec;
    if (g_session_written) {
        fs::remove(session_path(), ec);
        g_session_written = false;
    }
    if (!g_recovered.empty()) {
        fs::remove(g_recovered, ec);
        g_recovered.clear();
    }
}

void autosave_mark_recovered(const std::string& path) { g_recovered = path; }

void autosave_shutdown() {
    if (!g_job) return;
    join_job();
    bool ok; double secs; unsigned long long bytes; std::string path;
    autosave_poll(ok, secs, bytes, path);
}

#endif
