import {
  CartesianGrid,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { formatPercent } from '../format'
import type { CalibrationRow } from '../types/artifacts'
import { CHART, tickStyle } from './chartTheme'

interface Point {
  x: number
  y: number
  err: [number, number]
  n: number
  range: string
  low: number | null
  high: number | null
}

function calibrationPoints(rows: CalibrationRow[]): Point[] {
  return rows
    .filter((r) => r.n > 0 && r.mean_confidence !== null && r.favourite_win_rate !== null)
    .map((r) => {
      const y = r.favourite_win_rate as number
      return {
        x: r.mean_confidence as number,
        y,
        err: [r.win_rate_low !== null ? y - r.win_rate_low : 0, r.win_rate_high !== null ? r.win_rate_high - y : 0],
        n: r.n,
        range: `${formatPercent(r.low)}–${formatPercent(r.high)}`,
        low: r.win_rate_low,
        high: r.win_rate_high,
      }
    })
}

function TooltipBody({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  const p = payload?.[0]?.payload
  if (!active || !p) return null
  return (
    <div className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-xs shadow-lg tabular">
      <p className="font-semibold text-ink">Favourite given {p.range}</p>
      <p className="text-ink-2">Average probability: {formatPercent(p.x, 1)}</p>
      <p className="text-ink-2">
        Favourite won: {formatPercent(p.y, 1)}
        {p.low !== null && p.high !== null && ` (${formatPercent(p.low)}–${formatPercent(p.high)})`}
      </p>
      <p className="text-muted">{p.n} bouts</p>
    </div>
  )
}

/** Reliability diagram: in each bucket of the favourite's probability, the
 * average probability (x) against how often the favourite won (y), with the
 * interval as a whisker. On the diagonal is perfectly calibrated. */
export function CalibrationChart({ rows }: { rows: CalibrationRow[] }) {
  const points = calibrationPoints(rows)
  const yMin = Math.min(0.5, ...points.map((p) => p.low ?? p.y))
  const yFloor = Math.floor(yMin * 10) / 10
  return (
    <figure>
      <div className="h-72 w-full sm:h-80" role="img" aria-label="Calibration plot; the table below lists the same values">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
            <CartesianGrid stroke={CHART.grid} strokeDasharray="0" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0.5, 1]}
              ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              tickFormatter={(v: number) => formatPercent(v)}
              tick={tickStyle}
              stroke={CHART.axis}
              label={{ value: 'Model probability for the favourite', position: 'insideBottom', offset: -14, fill: CHART.tick, fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[yFloor, 1]}
              ticks={Array.from({ length: Math.round((1 - yFloor) * 10) + 1 }, (_, i) => Math.round((yFloor + i / 10) * 10) / 10)}
              tickFormatter={(v: number) => formatPercent(v)}
              tick={tickStyle}
              stroke={CHART.axis}
              width={60}
              label={{ value: 'Favourite won', angle: -90, position: 'insideLeft', offset: 8, fill: CHART.tick, fontSize: 12, style: { textAnchor: 'middle' } }}
            />
            <ReferenceLine
              segment={[
                { x: 0.5, y: 0.5 },
                { x: 1, y: 1 },
              ]}
              stroke={CHART.ink2}
              strokeDasharray="4 4"
              ifOverflow="hidden"
            />
            <Tooltip content={<TooltipBody />} cursor={false} />
            <Scatter data={points} fill={CHART.model} stroke={CHART.surface} strokeWidth={2} isAnimationActive={false}>
              <ErrorBar dataKey="err" direction="y" width={6} stroke={CHART.model} strokeWidth={1.5} />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-1 text-xs text-muted">
        Dashed line: perfect calibration. Dots: how often the favourite won in each probability bucket, with a 95%
        interval.
      </figcaption>
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer text-ink-2">Show as a table</summary>
        <table className="mt-2 w-full text-xs tabular">
          <thead className="text-muted">
            <tr>
              <th className="py-1 text-left font-medium">Bucket</th>
              <th className="py-1 text-right font-medium">Bouts</th>
              <th className="py-1 text-right font-medium">Avg. probability</th>
              <th className="py-1 text-right font-medium">Favourite won</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.low} className="border-t border-line">
                <td className="py-1">
                  {formatPercent(r.low)}–{formatPercent(r.high)}
                </td>
                <td className="py-1 text-right">{r.n}</td>
                <td className="py-1 text-right">{r.mean_confidence !== null ? formatPercent(r.mean_confidence, 1) : '—'}</td>
                <td className="py-1 text-right">
                  {r.favourite_win_rate !== null ? formatPercent(r.favourite_win_rate, 1) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}
