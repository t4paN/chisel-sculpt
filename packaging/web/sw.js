// Service worker for the installable (PWA) web build.
//
// Precaches the whole app on install, then serves it cache-first so it runs
// with no network. pack-pwa.sh stamps BUILD with a hash of the shipped files:
// a new build changes this file's bytes, the browser installs the new worker,
// and activate drops the old cache. The page already running keeps its loaded
// wasm; the next launch picks up the new build.
//
// Only same-origin GETs are handled. Everything else (the shell's POST /log,
// cross-origin requests) goes straight to the network and fails quietly offline.

const BUILD = '__BUILD__';
const CACHE = 'chisel-' + BUILD;
const FILES = [
  './',
  'index.html',
  'chisel.js',
  'chisel.wasm',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/maskable-192.png',
  'icons/maskable-512.png',
];

self.addEventListener('install', (e) => {
  // cache: 'reload' bypasses the HTTP cache so a fresh build never precaches stale bytes.
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(FILES.map((f) => new Request(f, { cache: 'reload' }))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k.startsWith('chisel-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(
    // ignoreSearch: "./?source=pwa" style launch URLs still hit the cached page.
    caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req))
  );
});
