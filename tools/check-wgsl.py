#!/usr/bin/env python3
"""check-wgsl.py — compile every WGSL shader with the browser's own compiler (Tint).

Why this exists: native wgpu-native validates WGSL with naga, but the browser build
uses Tint, which is stricter. Twice a shader that naga accepted shipped as a dead
brush on the web (a reserved keyword; a comparison inside select() that Tint reads as
a template list). And on WebGPU an invalid module still returns a non-null pipeline,
so the app's "compiled" printf proves nothing. Only Tint's own verdict counts.

What it does: serves a tiny page to headless Chrome/Chromium, which asks WebGPU to
compile each shader and, for compute kernels, build the pipeline with the device's
default limits (the same limits the app requests, minus buffer sizes). Every
compilation message comes back here.

Checked: every shaders/wgsl/*.wgsl, plus the WGSL raw-string literals inside src/
(render shaders: anything with @vertex / @fragment / @compute in it).

Usage:  python3 tools/check-wgsl.py [--browser PATH]
Exit:   0 all clean · 1 a shader has errors · 2 couldn't run (no browser/WebGPU/timeout)
No dependencies beyond Python 3 and a Chrome or Chromium.
"""
import glob, http.server, json, os, re, shutil, subprocess, sys, tempfile, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT_S = 120

PAGE = r"""<!doctype html><meta charset="utf-8"><title>wgsl check</title><script>
async function post(r) { await fetch('/report', { method: 'POST', body: JSON.stringify(r) }); }
(async () => {
  const r = { adapter: null, results: [], fatal: null };
  try {
    if (!navigator.gpu) throw new Error('navigator.gpu missing (WebGPU disabled)');
    let adapter = await navigator.gpu.requestAdapter();
    if (!adapter) adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    if (!adapter) throw new Error('no WebGPU adapter');
    const i = adapter.info || {};
    r.adapter = [i.vendor, i.architecture, i.description].filter(Boolean).join(' / ') || 'unknown';
    const device = await adapter.requestDevice();   // default limits, like the app
    const shaders = await (await fetch('/shaders.json')).json();
    for (const s of shaders) {
      const out = { name: s.name, messages: [] };
      device.pushErrorScope('validation');
      const mod = device.createShaderModule({ code: s.code });
      const info = await mod.getCompilationInfo();
      for (const m of info.messages)
        out.messages.push({ type: m.type, line: m.lineNum, col: m.linePos, text: m.message });
      // A module that failed to compile fails every pipeline too; that's just noise.
      const compiled = !out.messages.some(m => m.type === 'error');
      for (const ep of compiled ? s.compute : []) {
        try {
          await device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: ep } });
        } catch (e) {
          out.messages.push({ type: 'error', line: 0, col: 0, text: `pipeline '${ep}': ${e.message}` });
        }
      }
      const err = await device.popErrorScope();
      if (err && !out.messages.some(m => m.type === 'error'))
        out.messages.push({ type: 'error', line: 0, col: 0, text: err.message });
      r.results.push(out);
    }
  } catch (e) { r.fatal = String(e && e.message || e); }
  await post(r);
})();
</script>"""


def collect():
    shaders = []
    for path in sorted(glob.glob(os.path.join(ROOT, "shaders", "wgsl", "*.wgsl"))):
        shaders.append((os.path.relpath(path, ROOT), open(path, encoding="utf-8").read()))
    # Inline render shaders: R"delim( ... )delim" literals that contain WGSL stage attributes.
    raw = re.compile(r'R"([^(\s]{0,16})\((.*?)\)\1"', re.S)
    stage = re.compile(r"@(vertex|fragment|compute)\b")
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.cpp"), recursive=True)):
        text = open(path, encoding="utf-8").read()
        for m in raw.finditer(text):
            if stage.search(m.group(2)):
                line = text.count("\n", 0, m.start()) + 1
                shaders.append((f"{os.path.relpath(path, ROOT)}:{line}", m.group(2)))
    entry = re.compile(r"@compute\b[^{;]*?\bfn\s+(\w+)", re.S)
    return [{"name": n, "code": c, "compute": entry.findall(c)} for n, c in shaders]


def find_browser(arg):
    for c in [arg, os.environ.get("CHROME"), "chromium", "google-chrome", "google-chrome-stable",
              "chromium-browser"]:
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def main():
    arg = sys.argv[sys.argv.index("--browser") + 1] if "--browser" in sys.argv else None
    browser = find_browser(arg)
    if not browser:
        print("check-wgsl: no Chrome/Chromium found (pass --browser PATH or set CHROME)")
        return 2

    shaders = collect()
    payload = json.dumps([{"name": s["name"], "code": s["code"], "compute": s["compute"]} for s in shaders]).encode()
    report, done = {}, threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == "/shaders.json": self._send(payload, "application/json")
            else: self._send(PAGE.encode(), "text/html; charset=utf-8")
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            report.update(json.loads(body))
            self._send(b"ok", "text/plain")
            done.set()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    profile = tempfile.mkdtemp(prefix="check-wgsl-")
    cmd = [browser, "--headless=new", "--no-first-run", "--no-default-browser-check",
           f"--user-data-dir={profile}", "--enable-unsafe-webgpu", "--enable-unsafe-swiftshader",
           "--enable-features=Vulkan", url]
    if os.environ.get("CI"):
        # CI runners block the user namespaces Chrome's sandbox needs. The page only
        # compiles our own shaders from localhost, so there is nothing to contain.
        cmd.insert(1, "--no-sandbox")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        finished = done.wait(TIMEOUT_S)
    finally:
        proc.kill(); proc.wait()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)

    if not finished:
        print(f"check-wgsl: no answer from the browser within {TIMEOUT_S}s")
        return 2
    if report.get("fatal"):
        print(f"check-wgsl: {report['fatal']}")
        return 2

    bad = 0
    for r in report["results"]:
        errors = [m for m in r["messages"] if m["type"] == "error"]
        warns = [m for m in r["messages"] if m["type"] == "warning"]
        if not errors and not warns:
            continue
        bad += bool(errors)
        print(f"{'FAIL' if errors else 'warn'}  {r['name']}")
        for m in errors + warns:
            where = f"{m['line']}:{m['col']}  " if m["line"] else ""
            print(f"      {m['type']}: {where}{m['text']}")
    n = len(report["results"])
    print(f"check-wgsl: {n - bad}/{n} shaders clean under Tint ({report['adapter']}, {os.path.basename(browser)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
