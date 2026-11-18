import type { Reason, TopFactors } from '../types/artifacts'

interface Row {
  reason: Reason
  side: 'fighter_a' | 'fighter_b'
}

/** Diverging bars: each reason's summed contribution to the model's
 * log-odds, toward fighter A to the left and fighter B to the right, on one
 * shared scale. Rows are real text so the long titles wrap on a phone. */
export function FactorChart({
  factors,
  nameA,
  nameB,
}: {
  factors: TopFactors
  nameA: string
  nameB: string
}) {
  const rows: Row[] = [
    ...factors.fighter_a.map((reason) => ({ reason, side: 'fighter_a' as const })),
    ...factors.fighter_b.map((reason) => ({ reason, side: 'fighter_b' as const })),
  ]
  const max = Math.max(...rows.map((r) => r.reason.log_odds), 0)
  if (rows.length === 0 || max <= 0) {
    return <p className="text-sm text-ink-2">No factor carries weight toward either fighter.</p>
  }
  return (
    <figure>
      <div className="grid grid-cols-2 gap-4 text-xs font-semibold tracking-wide uppercase">
        <p className="flex items-center gap-1.5 text-ink-2">
          <span aria-hidden="true" className="h-2.5 w-2.5 rounded-sm bg-fighter-a" />← toward {nameA}
        </p>
        <p className="flex items-center justify-end gap-1.5 text-right text-ink-2">
          toward {nameB} →<span aria-hidden="true" className="h-2.5 w-2.5 rounded-sm bg-fighter-b" />
        </p>
      </div>
      <ul className="mt-3 space-y-4">
        {rows.map(({ reason, side }) => {
          const share = (reason.log_odds / max) * 100
          const isA = side === 'fighter_a'
          return (
            <li key={`${side}-${reason.reason}`} data-testid={`factor-${side}`}>
              <div className={`flex items-baseline gap-2 ${isA ? '' : 'flex-row-reverse text-right'}`}>
                <p className="text-sm font-semibold first-letter:uppercase">{reason.title}</p>
                <p className="shrink-0 text-xs text-muted tabular">{reason.log_odds.toFixed(2)}</p>
              </div>
              <div className="relative mt-1 grid h-2.5 grid-cols-2" aria-hidden="true">
                <div className="flex justify-end pr-px">
                  {isA && <div className="h-full rounded-l bg-fighter-a" style={{ width: `${share}%` }} />}
                </div>
                <div className="flex justify-start pl-px">
                  {!isA && <div className="h-full rounded-r bg-fighter-b" style={{ width: `${share}%` }} />}
                </div>
                <div className="absolute inset-y-[-3px] left-1/2 w-px -translate-x-1/2 bg-axis" />
              </div>
              <p className={`mt-1 text-xs leading-snug text-ink-2 ${isA ? '' : 'text-right'}`}>
                {reason.detail}
              </p>
            </li>
          )
        })}
      </ul>
      <figcaption className="mt-4 text-xs leading-relaxed text-muted">
        Bar length is the weight a group of inputs carries in the model’s log-odds, before calibration, on
        one scale for both fighters. These are what the model weighs, not causes of a result: they explain the
        model, not the sport.
      </figcaption>
    </figure>
  )
}
