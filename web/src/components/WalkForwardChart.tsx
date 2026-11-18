import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Fold } from '../types/artifacts'
import { CHART, tickStyle } from './chartTheme'
import { MODEL_LABELS } from '../labels'

const SERIES = [
  { key: 'model', color: CHART.model, width: 2.5, dash: undefined },
  { key: 'logistic', color: CHART.logistic, width: 2, dash: undefined },
  { key: 'elo', color: CHART.elo, width: 2, dash: '5 4' },
] as const

interface Row {
  year: number
  n: number
  test: boolean
  model?: number
  logistic?: number
  elo?: number
}

function rowsOf(folds: Fold[]): Row[] {
  return folds.map((f) => ({
    year: f.year,
    n: f.n_scored,
    test: f.in_test_period,
    model: f.metrics.model?.log_loss,
    logistic: f.metrics.logistic?.log_loss,
    elo: f.metrics.elo?.log_loss,
  }))
}

function TooltipBody({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  const r = payload?.[0]?.payload
  if (!active || !r) return null
  return (
    <div className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-lg tabular">
      <p className="font-semibold text-ink">
        {r.year} · {r.n} bouts{r.test ? ' · test period' : ''}
      </p>
      {SERIES.map((s) => (
        <p key={s.key} className="flex items-center justify-between gap-4 text-ink-2">
          <span className="flex items-center gap-1.5">
            <span aria-hidden="true" className="h-0.5 w-3" style={{ background: s.color }} />
            {MODEL_LABELS[s.key]}
          </span>
          <span>{r[s.key]?.toFixed(4) ?? '—'}</span>
        </p>
      ))}
    </div>
  )
}

/** Log loss by year in the walk-forward backtest, the model against both
 * baselines; test-period years are shaded. */
export function WalkForwardChart({ folds }: { folds: Fold[] }) {
  const rows = rowsOf(folds)
  const test = rows.filter((r) => r.test)
  return (
    <figure>
      <div className="h-72 w-full sm:h-80" role="img" aria-label="Log loss by year; the table below lists the same values">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            {test.length > 0 && (
              <ReferenceArea
                x1={test[0]!.year}
                x2={test.at(-1)!.year}
                fill={CHART.testShade}
                stroke="none"
                label={{ value: 'test period', position: 'insideTop', fill: CHART.tick, fontSize: 11 }}
              />
            )}
            <XAxis dataKey="year" tick={tickStyle} stroke={CHART.axis} interval="preserveStartEnd" />
            <YAxis
              domain={['auto', 'auto']}
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={tickStyle}
              stroke={CHART.axis}
              width={44}
            />
            <Tooltip content={<TooltipBody />} cursor={{ stroke: CHART.axis }} />
            <Legend
              verticalAlign="top"
              height={32}
              iconType="plainline"
              itemSorter={(item) => SERIES.findIndex((s) => s.key === item.dataKey)}
              formatter={(value: string) => <span style={{ color: CHART.ink2, fontSize: 12 }}>{MODEL_LABELS[value] ?? value}</span>}
            />
            {SERIES.map((s) => (
              <Line
                key={s.key}
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={s.width}
                strokeDasharray={s.dash}
                dot={{ r: 3, fill: s.color, stroke: CHART.surface, strokeWidth: 2 }}
                activeDot={{ r: 5, stroke: CHART.surface, strokeWidth: 2 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-1 text-xs text-muted">Log loss by year (lower is better).</figcaption>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-ink-2">Show as a table</summary>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full min-w-[26rem] text-xs tabular">
            <thead className="text-muted">
              <tr>
                <th className="py-1 text-left font-medium">Year</th>
                <th className="py-1 text-right font-medium">Bouts</th>
                {SERIES.map((s) => (
                  <th key={s.key} className="py-1 text-right font-medium">
                    {MODEL_LABELS[s.key]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.year} className="border-t border-line">
                  <td className="py-1">
                    {r.year}
                    {r.test && <span className="ml-1 text-muted">(test)</span>}
                  </td>
                  <td className="py-1 text-right">{r.n}</td>
                  {SERIES.map((s) => (
                    <td key={s.key} className="py-1 text-right">
                      {r[s.key]?.toFixed(4) ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  )
}
