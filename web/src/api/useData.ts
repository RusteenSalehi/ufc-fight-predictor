import { useCallback, useEffect, useState } from 'react'

export type DataState<T> =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'ready'; data: T }

/** Runs `load` when `key` changes; `retry` runs it again. */
export function useData<T>(
  key: string,
  load: () => Promise<T>,
): DataState<T> & { retry: () => void } {
  const [state, setState] = useState<{ key: string; attempt: number; value: DataState<T> }>({
    key,
    attempt: 0,
    value: { status: 'loading' },
  })
  const [attempt, setAttempt] = useState(0)

  // A new key or a retry shows the loading state at once.
  if (state.key !== key || state.attempt !== attempt) {
    setState({ key, attempt, value: { status: 'loading' } })
  }

  useEffect(() => {
    let live = true
    load().then(
      (data) => live && setState({ key, attempt, value: { status: 'ready', data } }),
      (error: unknown) =>
        live &&
        setState({
          key,
          attempt,
          value: { status: 'error', error: error instanceof Error ? error : new Error(String(error)) },
        }),
    )
    return () => {
      live = false
    }
    // `load` is a new function on each render; `key` names what it loads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt])

  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  return { ...state.value, retry }
}
