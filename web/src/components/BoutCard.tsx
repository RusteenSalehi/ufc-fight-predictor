import { Link } from 'react-router'
import { fairMoneyline, formatMoneyline } from '../probability'
import type { UnpredictedBout, UpcomingBout } from '../types/artifacts'
import { DataLabels } from './DataLabels'
import { ProbabilityBar } from './ProbabilityBar'

function Meta({ weightClass, segment, title }: { weightClass: string; segment: string | null; title: boolean }) {
  return (
    <p className="flex flex-wrap items-center gap-x-2 text-xs font-medium tracking-wide text-muted uppercase">
      <span>{weightClass}</span>
      {title && (
        <>
          <span aria-hidden="true">·</span>
          <span className="text-accent">Title bout</span>
        </>
      )}
      {segment && (
        <>
          <span aria-hidden="true">·</span>
          <span>{segment}</span>
        </>
      )}
    </p>
  )
}

function Name({ name, picked, align, provisional }: { name: string; picked: boolean; align: 'left' | 'right'; provisional: boolean }) {
  return (
    <div className={`min-w-0 ${align === 'right' ? 'text-right' : ''}`}>
      <p className={`truncate font-display text-xl font-semibold tracking-wide uppercase sm:text-2xl ${picked ? 'text-ink' : 'text-ink-2'}`} title={name}>
        {name}
      </p>
      {provisional && <p className="text-xs text-muted">first UFC bout</p>}
    </div>
  )
}

/** One predicted bout on the home page. The fair moneyline is converted here
 * from the published probability with the publisher's formula (D69); the
 * bout page shows the published value. */
export function BoutCard({ bout }: { bout: UpcomingBout }) {
  const a = bout.fighter_a
  const b = bout.fighter_b
  const pick = bout.predicted_winner
  const pickName = pick === 'fighter_a' ? a.name : pick === 'fighter_b' ? b.name : null
  return (
    <article className="rounded-xl border border-line bg-surface p-4 transition-colors hover:border-line-strong sm:p-5">
      <Meta weightClass={bout.weight_class} segment={bout.segment} title={bout.title_bout} />
      <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-end gap-3">
        <Name name={a.name} picked={pick === 'fighter_a'} align="left" provisional={a.provisional} />
        <span className="pb-1 text-xs text-muted">vs</span>
        <Name name={b.name} picked={pick === 'fighter_b'} align="right" provisional={b.provisional} />
      </div>
      <div className="mt-3">
        <ProbabilityBar pA={bout.p_fighter_a} pB={bout.p_fighter_b} nameA={a.name} nameB={b.name} />
      </div>
      <dl className="mt-2 flex justify-between text-xs text-muted tabular">
        <div>
          <dt className="sr-only">Fair moneyline, {a.name}</dt>
          <dd>Fair line {formatMoneyline(fairMoneyline(bout.p_fighter_a))}</dd>
        </div>
        <div className="text-right">
          <dt className="sr-only">Fair moneyline, {b.name}</dt>
          <dd>Fair line {formatMoneyline(fairMoneyline(bout.p_fighter_b))}</dd>
        </div>
      </dl>
      <div className="mt-3 flex flex-wrap items-end justify-between gap-3 border-t border-line pt-3">
        <div className="min-w-0 flex-1">
          <p className="text-sm">
            <span className="text-muted">Model pick: </span>
            <span className="font-semibold">{pickName ?? 'Even'}</span>
          </p>
          {bout.limited_data && (
            <div className="mt-2">
              <DataLabels reasons={bout.limited_data_reasons} compact />
            </div>
          )}
        </div>
        <Link
          to={`/bout/${bout.bout_id}`}
          className="shrink-0 rounded-md bg-surface-2 px-3 py-1.5 text-sm font-medium hover:bg-line-strong"
          aria-label={`Why: ${a.name} vs ${b.name}`}
        >
          Why →
        </Link>
      </div>
    </article>
  )
}

export function UnpredictedCard({ bout }: { bout: UnpredictedBout }) {
  return (
    <article className="rounded-xl border border-dashed border-line-strong p-4">
      <Meta weightClass={bout.weight_class} segment={bout.segment} title={bout.title_bout} />
      <p className="mt-1.5 font-display text-lg font-semibold tracking-wide text-ink-2 uppercase">
        {bout.fighter_a.name} <span className="text-sm text-muted normal-case">vs</span> {bout.fighter_b.name}
      </p>
      <p className="mt-1 text-xs text-muted">
        Not predicted: {bout.reason}
      </p>
    </article>
  )
}
