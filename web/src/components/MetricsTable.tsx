import { MODEL_LABELS } from '../labels'
import type { Metrics } from '../types/artifacts'

const ORDER = ['model', 'logistic', 'elo']

function ordered(metrics: Record<string, Metrics>): [string, Metrics][] {
  return Object.entries(metrics).sort(
    ([a], [b]) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99),
  )
}

export function MetricsTable({ metrics, caption }: { metrics: Record<string, Metrics>; caption: string }) {
  const rows = ordered(metrics)
  const withInterval = rows.some(([, m]) => m.log_loss_low != null && m.log_loss_high != null)
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-[34rem] text-sm tabular">
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-surface-2 text-xs tracking-wide text-ink-2 uppercase">
          <tr>
            <th scope="col" className="px-3 py-2 text-left font-semibold">
              Predictor
            </th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">
              Log loss
            </th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">
              Brier
            </th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">
              Accuracy
            </th>
            <th scope="col" className="px-3 py-2 text-right font-semibold">
              Calib. error
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, m]) => (
            <tr key={key} className={`border-t border-line ${key === 'model' ? 'bg-surface-2/60' : 'bg-surface'}`}>
              <th scope="row" className={`px-3 py-2 text-left ${key === 'model' ? 'font-semibold' : 'font-normal text-ink-2'}`}>
                {MODEL_LABELS[key] ?? key}
              </th>
              <td className="px-3 py-2 text-right">
                {m.log_loss.toFixed(4)}
                {withInterval && m.log_loss_low != null && m.log_loss_high != null && (
                  <span className="block text-xs text-muted">
                    {m.log_loss_low.toFixed(4)}–{m.log_loss_high.toFixed(4)}
                  </span>
                )}
              </td>
              <td className="px-3 py-2 text-right">{m.brier.toFixed(4)}</td>
              <td className="px-3 py-2 text-right">{(m.accuracy * 100).toFixed(1)}%</td>
              <td className="px-3 py-2 text-right">{m.ece != null ? m.ece.toFixed(3) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-line bg-surface px-3 py-2 text-xs text-muted">
        Lower is better for log loss, Brier score and calibration error (favourite-side, ten bins).
        {withInterval && ' Under the log loss: 95% bootstrap interval.'} n = {rows[0]?.[1].n.toLocaleString('en-US')} bouts
        with a winner.
      </p>
    </div>
  )
}
