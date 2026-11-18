import { formatEventDate, formatPercent } from '../format'
import type { LiveBout, LiveRecord as LiveRecordData, Metrics } from '../types/artifacts'
import { CalibrationChart } from './CalibrationChart'
import { EmptyState } from './StatusViews'

const STATUS_LABELS: Record<LiveBout['status'], string> = {
  scored: 'Scored',
  no_result: 'No result (draw or no contest)',
  pending: 'Waiting for results',
  cancelled: 'Off the card before the cut-off',
  no_live_prediction: 'No prediction before the cut-off',
  not_found: 'Result not found in the dataset',
  unlinked: 'Debutant not yet linked to the dataset',
}

const STATUS_ORDER: LiveBout['status'][] = [
  'scored',
  'no_result',
  'pending',
  'cancelled',
  'no_live_prediction',
  'not_found',
  'unlinked',
]

function MetricTiles({ metrics, title }: { metrics: Metrics; title: string }) {
  const tiles = [
    { label: 'Log loss', value: metrics.log_loss.toFixed(4) },
    { label: 'Brier score', value: metrics.brier.toFixed(4) },
    { label: 'Accuracy', value: formatPercent(metrics.accuracy, 1) },
    { label: 'Bouts', value: metrics.n.toLocaleString('en-US') },
  ]
  return (
    <div>
      <h3 className="text-sm font-semibold text-ink-2">{title}</h3>
      <dl className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {tiles.map((t) => (
          <div key={t.label} className="rounded-lg border border-line bg-surface p-3">
            <dt className="text-xs text-muted">{t.label}</dt>
            <dd className="mt-0.5 font-display text-2xl font-bold tabular">{t.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function outcomeText(b: LiveBout): string {
  if (b.outcome === 'fighter_a') return `${b.fighter_a} won`
  if (b.outcome === 'fighter_b') return `${b.fighter_b} won`
  if (b.outcome === 'draw') return 'Draw'
  if (b.outcome === 'nc') return 'No contest'
  return STATUS_LABELS[b.status]
}

function BoutsTable({ bouts }: { bouts: LiveBout[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-[36rem] text-sm tabular">
        <caption className="sr-only">Live predictions and their results</caption>
        <thead className="bg-surface-2 text-xs tracking-wide text-ink-2 uppercase">
          <tr>
            <th scope="col" className="px-3 py-2 text-left font-semibold">Event</th>
            <th scope="col" className="px-3 py-2 text-left font-semibold">Bout</th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">Live prediction</th>
            <th scope="col" className="px-3 py-2 text-left font-semibold">Result</th>
          </tr>
        </thead>
        <tbody>
          {bouts.map((b) => (
            <tr key={b.bout_id} className="border-t border-line bg-surface">
              <td className="px-3 py-2 text-ink-2">
                {b.event_name}
                <span className="block text-xs text-muted">{formatEventDate(b.event_date, false)}</span>
              </td>
              <td className="px-3 py-2">
                {b.fighter_a} <span className="text-muted">vs</span> {b.fighter_b}
              </td>
              <td className="px-3 py-2 text-right">
                {b.p_fighter_a !== null ? `${formatPercent(b.p_fighter_a)} / ${formatPercent(1 - b.p_fighter_a)}` : '—'}
              </td>
              <td className="px-3 py-2">
                {outcomeText(b)}
                {b.status === 'scored' && b.correct !== null && (
                  <span className="ml-1.5 text-xs text-muted">
                    {b.correct === 1 ? '(pick right)' : b.correct === 0 ? '(pick wrong)' : '(even)'}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function LiveRecord({ live }: { live: LiveRecordData }) {
  const counts = STATUS_ORDER.map((s) => ({ status: s, n: live.counts[s] ?? 0 })).filter((c) => c.n > 0)
  const pending = live.counts.pending ?? 0
  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-2">{live.label}.</p>

      {live.metrics === null ? (
        <EmptyState title="No live results yet">
          <p data-testid="live-empty">
            Nothing has been scored yet. Predictions published before each event are scored here once the event’s
            results reach the dataset; until then there is no live track record to report.
            {pending > 0 && ` ${pending} bout${pending === 1 ? ' is' : 's are'} waiting for results.`}
          </p>
        </EmptyState>
      ) : (
        <div className="space-y-5">
          <MetricTiles metrics={live.metrics} title="All scored live predictions" />
          {live.metrics_without_limited_data && (
            <MetricTiles metrics={live.metrics_without_limited_data} title="Without limited-data bouts" />
          )}
          {Object.keys(live.by_model_version).length > 1 && (
            <ul className="text-sm text-ink-2">
              {Object.entries(live.by_model_version).map(([version, m]) => (
                <li key={version}>
                  <code>{version}</code>: log loss {m.log_loss.toFixed(4)} over {m.n} bouts
                </li>
              ))}
            </ul>
          )}
          <div className="rounded-xl border border-line bg-surface p-4">
            <h3 className="text-sm font-semibold text-ink-2">Calibration of live predictions</h3>
            <CalibrationChart rows={live.calibration} />
          </div>
        </div>
      )}

      {counts.length > 0 && (
        <dl className="flex flex-wrap gap-2 text-xs">
          {counts.map((c) => (
            <div key={c.status} className="rounded-full border border-line px-3 py-1">
              <dt className="inline text-muted">{STATUS_LABELS[c.status]}: </dt>
              <dd className="inline font-semibold tabular">{c.n}</dd>
            </div>
          ))}
        </dl>
      )}

      {live.bouts.length > 0 && <BoutsTable bouts={live.bouts} />}

      <details className="rounded-xl border border-line bg-surface p-4 text-sm">
        <summary className="cursor-pointer font-semibold">How the live record is kept</summary>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-ink-2">
          {live.rules.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}
