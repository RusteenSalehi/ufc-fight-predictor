import type { ReactNode } from 'react'
import { Link, isRouteErrorResponse, useRouteError } from 'react-router'

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3">
      <span className="sr-only">{label}…</span>
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-24 animate-pulse rounded-xl border border-line bg-surface" />
      ))}
    </div>
  )
}

export function ErrorState({
  title = 'Could not load this page',
  error,
  onRetry,
}: {
  title?: string
  error: Error
  onRetry?: () => void
}) {
  return (
    <div role="alert" className="rounded-xl border border-critical/40 bg-surface p-5">
      <h2 className="font-display text-xl font-semibold tracking-wide uppercase">{title}</h2>
      <p className="mt-1 text-sm text-ink-2">{error.message}</p>
      <div className="mt-4 flex gap-3 text-sm">
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="rounded-md bg-surface-2 px-3 py-1.5 font-medium hover:bg-line-strong"
          >
            Try again
          </button>
        )}
        <Link to="/" className="rounded-md px-3 py-1.5 text-ink-2 hover:text-ink">
          Upcoming events
        </Link>
      </div>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line-strong p-6 text-center">
      <p className="font-display text-lg font-semibold tracking-wide uppercase">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-prose text-sm text-ink-2">{children}</div>}
    </div>
  )
}

/** Shown if rendering throws; keeps the page readable instead of blank. */
export function RouteError() {
  const error = useRouteError()
  const message = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : 'Unknown error'
  return (
    <div className="mx-auto max-w-xl p-6">
      <ErrorState title="Something went wrong" error={new Error(message)} />
    </div>
  )
}

export function Section({
  title,
  id,
  aside,
  children,
}: {
  title: string
  id?: string
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section aria-labelledby={id} className="mt-10 first:mt-0">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={id} className="font-display text-2xl font-semibold tracking-wide uppercase">
          {title}
        </h2>
        {aside && <div className="text-xs text-muted">{aside}</div>}
      </div>
      {children}
    </section>
  )
}

export function Callout({
  tone = 'note',
  title,
  children,
}: {
  tone?: 'note' | 'caution'
  title: string
  children: ReactNode
}) {
  const border = tone === 'caution' ? 'border-l-warning' : 'border-l-accent'
  return (
    <aside className={`rounded-lg border border-line border-l-4 ${border} bg-surface p-4`}>
      <p className="text-sm font-semibold">
        {tone === 'caution' && <span aria-hidden="true">⚠ </span>}
        {title}
      </p>
      <div className="mt-1 text-sm leading-relaxed text-ink-2">{children}</div>
    </aside>
  )
}
