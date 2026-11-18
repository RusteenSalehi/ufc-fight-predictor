import { formatEventDate, formatPercent } from '../format'
import type { PastBout, RecentBouts as RecentBoutsData } from '../types/artifacts'

const RESULT: Record<PastBout['result'], { short: string; label: string; cls: string }> = {
  win: { short: 'W', label: 'Win', cls: 'bg-ink text-page' },
  loss: { short: 'L', label: 'Loss', cls: 'border border-line-strong text-ink-2' },
  draw: { short: 'D', label: 'Draw', cls: 'border border-line-strong text-ink-2' },
  nc: { short: 'NC', label: 'No contest', cls: 'border border-line-strong text-ink-2' },
}

function method(b: PastBout): string {
  const parts = [b.method ?? 'Method unknown']
  if (b.end_round !== null) parts.push(`R${b.end_round}${b.end_time ? ` ${b.end_time}` : ''}`)
  return parts.join(', ')
}

function Deserved({ p, color }: { p: number | null; color: string }) {
  if (p === null) return <span className="text-muted">no stats</span>
  return (
    <span className="inline-flex items-center gap-2">
      <span aria-hidden="true" className="relative h-1.5 w-14 overflow-hidden rounded-full bg-surface-2">
        <span className={`absolute inset-y-0 left-0 rounded-full ${color}`} style={{ width: `${p * 100}%` }} />
      </span>
      <span className="tabular">{formatPercent(p)}</span>
    </span>
  )
}

function FighterBouts({ name, bouts, color }: { name: string; bouts: PastBout[]; color: string }) {
  return (
    <div className="min-w-0">
      <h3 className="flex items-center gap-2 font-display text-lg font-semibold tracking-wide uppercase">
        <span aria-hidden="true" className={`h-2.5 w-2.5 rounded-sm ${color}`} />
        {name}
      </h3>
      {bouts.length === 0 ? (
        <p className="mt-2 text-sm text-ink-2">No earlier UFC bouts in the dataset.</p>
      ) : (
        <ol className="mt-2 divide-y divide-line rounded-xl border border-line bg-surface">
          {bouts.map((b) => {
            const r = RESULT[b.result]
            return (
              <li key={`${b.date}-${b.opponent_id}`} className="flex gap-3 p-3">
                <span
                  className={`mt-0.5 flex h-7 w-8 shrink-0 items-center justify-center rounded text-xs font-bold ${r.cls}`}
                  title={r.label}
                >
                  <span aria-hidden="true">{r.short}</span>
                  <span className="sr-only">{r.label}</span>
                </span>
                <div className="min-w-0 flex-1 text-sm">
                  <p className="truncate">
                    <span className="text-muted">vs </span>
                    <span className="font-medium">{b.opponent}</span>
                    {b.title_bout && <span className="ml-1.5 text-xs text-accent">title</span>}
                  </p>
                  <p className="text-xs text-ink-2">
                    {method(b)} · {formatEventDate(b.date, false)}
                  </p>
                  <p className="mt-1 flex items-center justify-between gap-2 text-xs text-ink-2">
                    <span>Deserved-win probability</span>
                    <Deserved p={b.deserved_win_probability} color={color} />
                  </p>
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}

export function RecentBouts({ data, nameA, nameB }: { data: RecentBoutsData; nameA: string; nameB: string }) {
  return (
    <div>
      <div className="grid gap-6 md:grid-cols-2">
        <FighterBouts name={nameA} bouts={data.fighter_a} color="bg-fighter-a" />
        <FighterBouts name={nameB} bouts={data.fighter_b} color="bg-fighter-b" />
      </div>
      <p className="mt-3 text-xs leading-relaxed text-muted">
        <span className="font-semibold text-ink-2">Describes past fights only. </span>
        {data.note}
      </p>
    </div>
  )
}
