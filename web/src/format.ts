/** Formatting for dates, times and stat values. Pure functions. */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function parseIsoDate(iso: string): { y: number; m: number; d: number } {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!match) throw new Error(`not a date: ${iso}`)
  return { y: Number(match[1]), m: Number(match[2]), d: Number(match[3]) }
}

/** An event date ("2026-11-21") as "Sat 21 Nov 2026". Event dates have no
 * time zone, so this never shifts the day. */
export function formatEventDate(iso: string, withWeekday = true): string {
  const { y, m, d } = parseIsoDate(iso)
  const weekday = WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()]
  const text = `${d} ${MONTHS[m - 1]} ${y}`
  return withWeekday ? `${weekday} ${text}` : text
}

/** A UTC timestamp ("2026-11-18T05:18:27Z") as "18 Nov 2026, 05:18 UTC". */
export function formatTimestamp(iso: string): string {
  const t = new Date(iso)
  if (Number.isNaN(t.getTime())) return iso
  const hh = String(t.getUTCHours()).padStart(2, '0')
  const mm = String(t.getUTCMinutes()).padStart(2, '0')
  return `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]} ${t.getUTCFullYear()}, ${hh}:${mm} UTC`
}

/** Today's date in UTC as YYYY-MM-DD. */
export function todayUtc(now: Date = new Date()): string {
  return now.toISOString().slice(0, 10)
}

export function formatPercent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`
}

function feetInches(cm: number): string {
  const totalInches = Math.round(cm / 2.54)
  return `${Math.floor(totalInches / 12)}′${totalInches % 12}″`
}

/** A stat row's value by its `kind` (StatRow.kind in the bout artifact). */
export function formatStat(value: number | null, kind: string): string {
  if (value === null || !Number.isFinite(value)) return '—'
  switch (kind) {
    case 'years':
      return value.toFixed(1)
    case 'cm':
      return `${Math.round(value)} cm (${feetInches(value)})`
    case 'count':
    case 'days':
      return Math.round(value).toLocaleString('en-US')
    case 'elo':
      return Math.round(value).toLocaleString('en-US')
    case 'pct':
      return formatPercent(value)
    case 'rate':
      return value.toFixed(2)
    default:
      return value.toFixed(2)
  }
}

export function formatNumber(value: number, digits: number): string {
  return value.toFixed(digits)
}
