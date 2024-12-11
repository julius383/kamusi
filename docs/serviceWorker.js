const staticSwaEng = "swa-eng-dict-site-v1"
const assets = [
  "/",
  "/manifest.json",
  "/index.html",
  "/js/app.js",
  "/js/elasticlunr.js",
  "/css/pico.colors.min.css",
  "/css/pico.min.css",
  "https://julius383.github.io/kamusi/static/book.png",
  "https://julius383.github.io/kamusi/static/search_index.json",
]

self.addEventListener("install", installEvent => {
  installEvent.waitUntil(
    caches.open(staticSwaEng).then(cache => {
      cache.addAll(assets)
    })
  )
})

self.addEventListener("fetch", fetchEvent => {
  fetchEvent.respondWith(
    caches.match(fetchEvent.request).then(res => {
      return res || fetch(fetchEvent.request)
    })
  )
})
