import { render } from '@testing-library/react'
import { RouterProvider, createMemoryRouter } from 'react-router'
import { vi } from 'vitest'
import { routes } from '../routes'

/** Render the whole site at `path`, as the browser router would. */
export function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  return render(<RouterProvider router={router} />)
}

/** Answer fetches for /data/<file> from a map of file name to JSON body;
 * anything else is a 404. */
export function stubData(files: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    const name = url.replace(/^.*?\/data\//, '')
    if (name in files) {
      return new Response(JSON.stringify(files[name]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    return new Response('{"error":"not found"}', { status: 404 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
