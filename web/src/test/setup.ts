import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'
import { clearDataCache } from '../api/client'

afterEach(() => {
  cleanup()
  clearDataCache()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

// Recharts measures its container; jsdom has no layout.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver

// ScrollRestoration scrolls on navigation; jsdom does not implement it.
if (typeof window !== "undefined") window.scrollTo = () => {}
