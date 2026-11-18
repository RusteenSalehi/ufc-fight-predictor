import { Link, useParams } from 'react-router'
import { DataError, fetchBout, isBoutId } from '../api/client'
import { useData } from '../api/useData'
import { DataLabels } from '../components/DataLabels'
import { FactorChart } from '../components/FactorChart'
import { usePageTitle } from '../usePageTitle'
import { ProbabilityBar } from '../components/ProbabilityBar'
import { RecentBouts } from '../components/RecentBouts'
import { StatComparison } from '../components/StatComparison'
import { EmptyState, ErrorState, Loading, Section } from '../components/StatusViews'
import { formatEventDate, formatTimestamp } from '../format'
import { formatMoneyline } from '../probability'
import type { BoutArtifact } from '../types/artifacts'

function Header({ bout }: { bout: BoutArtifact }) {
  const a = bout.fighter_a
  const b = bout.fighter_b
  const p = bout.prediction
  const pickName = p.predicted_winner === 'fighter_a' ? a.name : p.predicted_winner === 'fighter_b' ? b.name : null
  return (
    <header>
      <p className="text-sm text-ink-2">
        <Link to="/" className="link">
          {bout.event.name}
        </Link>{' '}
        · {formatEventDate(bout.event.date)}
      </p>
      <p className="mt-3 flex flex-wrap gap-x-2 text-xs font-medium tracking-wide text-muted uppercase">
        <span>{bout.weight_class}</span>
        {bout.title_bout && <span className="text-accent">· Title bout</span>}
        {bout.segment && <span>· {bout.segment}</span>}
      </p>
      <h1 className="mt-1 grid grid-cols-[1fr_auto_1fr] items-end gap-3 font-display text-3xl leading-none font-bold tracking-wide uppercase sm:text-5xl">
        <span className="min-w-0 break-words">{a.name}</span>
        <span className="pb-1 text-base font-semibold text-muted normal-case">vs</span>
        <span className="min-w-0 break-words text-right">{b.name}</span>
      </h1>
      <div className="mt-5">
        <ProbabilityBar pA={p.p_fighter_a} pB={p.p_fighter_b} nameA={a.name} nameB={b.name} size="lg" />
      </div>
      <dl className="mt-2 flex justify-between text-sm text-ink-2 tabular">
        <div>
          <dt className="sr-only">Fair moneyline, {a.name}</dt>
          <dd>Fair line {formatMoneyline(p.fair_moneyline.fighter_a)}</dd>
        </div>
        <div className="text-right">
          <dt className="sr-only">Fair moneyline, {b.name}</dt>
          <dd>Fair line {formatMoneyline(p.fair_moneyline.fighter_b)}</dd>
        </div>
      </dl>
      <div className="mt-5 rounded-xl border border-line bg-surface p-4 text-sm">
        <p>
          <span className="text-muted">Model pick: </span>
          <span className="font-semibold">{pickName ?? 'Even'}</span>
        </p>
        <p className="mt-1 text-xs text-muted">
          Model <code className="text-ink-2">{p.model_version}</code> · predicted {formatTimestamp(p.predicted_at)}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-muted">{p.fair_moneyline_note}</p>
        {bout.limited_data.limited && (
          <div className="mt-3 border-t border-line pt-3">
            <DataLabels reasons={bout.limited_data.reasons} />
          </div>
        )}
      </div>
    </header>
  )
}

function MatchupContent({ bout }: { bout: BoutArtifact }) {
  const nameA = bout.fighter_a.name
  const nameB = bout.fighter_b.name
  usePageTitle(`${nameA} vs ${nameB}`)
  return (
    <article>
      <Header bout={bout} />

      <Section title="Why" id="why">
        {bout.why && bout.top_factors ? (
          <div className="space-y-6">
            <p className="max-w-prose leading-relaxed text-ink-2">{bout.why.summary}</p>
            <div className="rounded-xl border border-line bg-surface p-4 sm:p-5">
              <FactorChart factors={bout.top_factors} nameA={nameA} nameB={nameB} />
            </div>
          </div>
        ) : (
          <EmptyState title="No explanation stored">
            This prediction was made before explanations were stored. It gets one when its inputs next change.
          </EmptyState>
        )}
      </Section>

      <Section title="Side by side" id="stats" aside="As of the event date">
        <StatComparison stats={bout.stats} nameA={nameA} nameB={nameB} />
      </Section>

      <Section title="Last 5 UFC bouts" id="recent">
        <RecentBouts data={bout.recent_bouts} nameA={nameA} nameB={nameB} />
      </Section>

      <p className="mt-12 text-xs leading-relaxed text-muted">{bout.attribution}</p>
    </article>
  )
}

export function Matchup() {
  const { boutId } = useParams()
  const valid = isBoutId(boutId)
  const state = useData(`bout:${boutId}`, () => fetchBout(boutId ?? ''))
  if (!valid || (state.status === 'error' && state.error instanceof DataError && state.error.status === 404)) {
    return (
      <EmptyState title="Bout not found">
        This bout is not on a published card. It may have been cancelled or the card changed.{' '}
        <Link to="/" className="link">
          See upcoming events
        </Link>
        .
      </EmptyState>
    )
  }
  if (state.status === 'loading') return <Loading label="Loading the matchup" />
  if (state.status === 'error')
    return <ErrorState title="Could not load this matchup" error={state.error} onRetry={state.retry} />
  return <MatchupContent bout={state.data} />
}
