/** The one place the site fetches data. Every artifact is a static JSON file
 * written by `uv run python -m ufc.predict refresh` (data/publish/public/),
 * read from DATA_BASE_URL. Types are generated from the pydantic models;
 * at run time only the kind and schema version are checked, which catches a
 * wrong URL or a site older than the data. */
import { DATA_BASE_URL } from '../config'
import type { BoutArtifact, Performance, Upcoming } from '../types/artifacts'

export const SUPPORTED_SCHEMA_VERSION = 1

export class DataError extends Error {
  readonly status: number | null
  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'DataError'
    this.status = status
  }
}

type Kind = 'upcoming' | 'bout' | 'performance'

const cache = new Map<string, Promise<unknown>>()

/** Forget fetched files (tests; a manual reload re-reads everything anyway). */
export function clearDataCache(): void {
  cache.clear()
}

export function dataUrl(path: string): string {
  return `${DATA_BASE_URL}/${path.replace(/^\/+/, '')}`
}

async function load<T>(path: string, kind: Kind): Promise<T> {
  const url = dataUrl(path)
  let response: Response
  try {
    response = await fetch(url, { headers: { Accept: 'application/json' } })
  } catch {
    throw new DataError('The data could not be reached. Check your connection and try again.')
  }
  if (!response.ok) {
    throw new DataError(
      response.status === 404
        ? 'This file has not been published.'
        : `The data server answered ${response.status}.`,
      response.status,
    )
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new DataError('The published file is not valid JSON.')
  }
  const head = body as { kind?: unknown; schema_version?: unknown } | null
  if (!head || head.kind !== kind) {
    throw new DataError(`Expected a "${kind}" file at ${path}.`)
  }
  if (head.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    throw new DataError(
      `The data uses schema version ${String(head.schema_version)}; this site reads version ${SUPPORTED_SCHEMA_VERSION}.`,
    )
  }
  return body as T
}

function cached<T>(path: string, kind: Kind): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined
  if (!p) {
    p = load<T>(path, kind)
    cache.set(path, p)
    // a failure is not cached, so "try again" refetches
    p.catch(() => cache.delete(path))
  }
  return p
}

const BOUT_ID = /^wb[0-9a-f]{16}$/

export function isBoutId(id: string | undefined): id is string {
  return id !== undefined && BOUT_ID.test(id)
}

export const fetchUpcoming = () => cached<Upcoming>('upcoming.json', 'upcoming')
export const fetchPerformance = () => cached<Performance>('performance.json', 'performance')
export function fetchBout(boutId: string): Promise<BoutArtifact> {
  if (!isBoutId(boutId)) return Promise.reject(new DataError('Not a bout ID.', 404))
  return cached<BoutArtifact>(`bouts/${boutId}.json`, 'bout')
}
