import { fetchPerformance } from '../api/client'
import { useData } from '../api/useData'
import { CalibrationChart } from '../components/CalibrationChart'
import { usePageTitle } from '../usePageTitle'
import { LiveRecord } from '../components/LiveRecord'
import { MetricsTable } from '../components/MetricsTable'
import { Callout, EmptyState, ErrorState, Loading, Section } from '../components/StatusViews'
import { WalkForwardChart } from '../components/WalkForwardChart'
import { formatEventDate, formatTimestamp } from '../format'
import type { Backtest, Performance } from '../types/artifacts'

function BacktestSection({ bt }: { bt: Backtest }) {
  const tp = bt.test_period
  const wf = bt.walk_forward
  const pooled = wf.pooled_before_test_period
  return (
    <div className="space-y-6">
      <p className="text-sm text-ink-2">{bt.label}.</p>

      {tp.already_seen && (
        <Callout tone="caution" title="Already seen: not a held-out result">
          {tp.already_seen_note
            ? `${tp.already_seen_note.charAt(0).toUpperCase()}${tp.already_seen_note.slice(1)}.`
            : 'The test period was evaluated before, so these numbers are not a held-out result.'}
        </Callout>
      )}

      <div>
        <h3 className="font-display text-xl font-semibold tracking-wide uppercase">Test period</h3>
        <p className="mt-1 mb-3 text-sm text-ink-2">
          {formatEventDate(tp.first_date, false)} to {formatEventDate(tp.last_date, false)}:{' '}
          {tp.n_scored.toLocaleString('en-US')} bouts with a winner, {tp.n_no_result} draws or no contests not scored.
          Candidate <code className="text-ink">{tp.candidate_version}</code>.
        </p>
        <MetricsTable metrics={tp.metrics} caption="Test-period metrics" />
      </div>

      <div className="rounded-xl border border-line bg-surface p-4 sm:p-5">
        <h3 className="font-display text-xl font-semibold tracking-wide uppercase">Calibration</h3>
        <p className="mt-1 mb-3 text-sm text-ink-2">Test period, by {tp.calibration_buckets}.</p>
        <CalibrationChart rows={tp.calibration} />
      </div>

      <div className="rounded-xl border border-line bg-surface p-4 sm:p-5">
        <h3 className="font-display text-xl font-semibold tracking-wide uppercase">Results over time</h3>
        <p className="mt-1 mb-3 text-sm text-ink-2">
          Walk-forward backtest, the same recipe refitted each year ({wf.schedule}). Pooled {pooled.years[0]} to{' '}
          {pooled.years[1]}: log loss {pooled.metrics.model?.log_loss.toFixed(4) ?? '—'} for the model over{' '}
          {pooled.metrics.model?.n.toLocaleString('en-US') ?? '—'} bouts. Years before the test period were scored
          with a recipe chosen on 2023 to 2024, so they are a backtest, not a held-out result.
        </p>
        <WalkForwardChart folds={wf.folds} />
      </div>

      <div className="text-xs leading-relaxed text-muted">
        <p>{bt.production_note}</p>
        <p className="mt-2">
          Reproduce:{' '}
          {bt.commands.map((c) => (
            <code key={c} className="mr-2 rounded bg-surface-2 px-1.5 py-0.5 text-ink-2">
              {c}
            </code>
          ))}
        </p>
      </div>
    </div>
  )
}

function PerformanceContent({ perf }: { perf: Performance }) {
  usePageTitle('Performance')
  return (
    <div>
      <header>
        <p className="text-xs font-semibold tracking-[0.18em] text-accent uppercase">Track record</p>
        <h1 className="mt-1 font-display text-4xl leading-none font-bold tracking-wide uppercase sm:text-5xl">
          Model performance
        </h1>
        <p className="mt-3 max-w-prose text-ink-2">{perf.separation}</p>
        <p className="mt-2 text-xs text-muted">
          Live model <code className="text-ink-2">{perf.model.version}</code> ({perf.model.status}) · updated{' '}
          {formatTimestamp(perf.generated_at)}
        </p>
      </header>

      <Section title="Backtest" id="backtest">
        {perf.era_shift_caveat && (
          <div className="mb-5">
            <Callout tone="caution" title="Era shift: the test period looks better than any earlier year">
              {perf.era_shift_caveat.text}
            </Callout>
          </div>
        )}
        {perf.backtest ? (
          <BacktestSection bt={perf.backtest} />
        ) : (
          <EmptyState title="No backtest published">{perf.backtest_unavailable}</EmptyState>
        )}
      </Section>

      <Section title="Live record" id="live">
        <LiveRecord live={perf.live} />
      </Section>
    </div>
  )
}

export function PerformancePage() {
  const state = useData('performance', fetchPerformance)
  if (state.status === 'loading') return <Loading label="Loading performance" />
  if (state.status === 'error')
    return <ErrorState title="Could not load performance" error={state.error} onRetry={state.retry} />
  return <PerformanceContent perf={state.data} />
}
