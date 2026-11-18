// Serves the refresh's public artifacts (../data/publish/public by default)
// at /data/ in `vite` and `vite preview`, so the site can be run locally
// without copying gitignored data into web/. Production builds contain no
// data: the site fetches from VITE_DATA_BASE_URL (DECISIONS D69).
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import type { Connect, Plugin } from 'vite'

export const DATA_ROUTE = '/data/'

/** The file under `root` that a request path names, or null if it names
 * anything else (not JSON, outside root, malformed). */
export function resolveDataFile(root: string, urlPath: string): string | null {
  if (!urlPath.startsWith(DATA_ROUTE)) return null
  let rel: string
  try {
    rel = decodeURIComponent(urlPath.slice(DATA_ROUTE.length).split('?')[0])
  } catch {
    return null
  }
  if (!rel.endsWith('.json') || rel.includes('\0')) return null
  const base = path.resolve(root)
  const file = path.resolve(base, rel)
  if (!file.startsWith(base + path.sep)) return null
  return file
}

function middleware(root: string): Connect.NextHandleFunction {
  return (req, res, next) => {
    const url = req.url ?? ''
    if (!url.startsWith(DATA_ROUTE)) return next()
    const file = resolveDataFile(root, url)
    const send = (status: number, body: string) => {
      res.statusCode = status
      res.setHeader('Content-Type', 'application/json; charset=utf-8')
      res.setHeader('Cache-Control', 'no-store')
      res.end(body)
    }
    if (file === null) return send(404, '{"error":"not found"}')
    stat(file)
      .then((s) => (s.isFile() ? readFile(file, 'utf8') : Promise.reject(new Error('not a file'))))
      .then((body) => send(200, body))
      .catch(() => send(404, '{"error":"not found"}'))
  }
}

export function publishData(root: string): Plugin {
  return {
    name: 'ufc-publish-data',
    configureServer(server) {
      server.middlewares.use(middleware(root))
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware(root))
    },
  }
}
