/** "Limited data" and "Stale stats" labels with their reasons (SPEC 1, D9,
 * D52). Each label has an icon and words, never colour alone. */
import { groupReasons } from '../labels'

function sentence(reason: string): string {
  return reason.charAt(0).toUpperCase() + reason.slice(1)
}

function Badge({ kind }: { kind: 'limited' | 'stale' }) {
  const isStale = kind === 'stale'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-semibold tracking-wide uppercase ${
        isStale ? 'border-serious/50 text-serious' : 'border-warning/50 text-warning'
      }`}
    >
      {isStale ? (
        <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8">
          <circle cx="8" cy="8" r="6.2" />
          <path d="M8 4.5V8l2.3 1.6" strokeLinecap="round" />
        </svg>
      ) : (
        <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M8 1.8 14.6 13.6H1.4Z" strokeLinejoin="round" />
          <path d="M8 6.2v3.4" strokeLinecap="round" />
          <circle cx="8" cy="11.6" r="0.4" fill="currentColor" />
        </svg>
      )}
      {isStale ? 'Stale stats' : 'Limited data'}
    </span>
  )
}

export function DataLabels({
  reasons,
  compact = false,
}: {
  reasons: string[]
  compact?: boolean
}) {
  const { limited, stale } = groupReasons(reasons)
  if (limited.length === 0 && stale.length === 0) return null
  const groups = [
    { kind: 'limited' as const, items: limited },
    { kind: 'stale' as const, items: stale },
  ].filter((g) => g.items.length > 0)
  return (
    <div className={compact ? 'space-y-1.5' : 'space-y-3'} data-testid="data-labels">
      {groups.map((g) => (
        <div key={g.kind}>
          <Badge kind={g.kind} />
          <ul className="mt-1 space-y-0.5 text-xs leading-snug text-ink-2">
            {g.items.map((r) => (
              <li key={r}>{sentence(r)}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
