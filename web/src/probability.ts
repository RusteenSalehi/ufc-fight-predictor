/** Display arithmetic for a bout's two probabilities. Nothing here changes a
 * probability; it decides how the published pair is drawn and labelled. */

export type Side = 'fighter_a' | 'fighter_b'

export interface ProbabilitySplit {
  /** Bar widths in percent; exactly the published probabilities. */
  widthA: number
  widthB: number
  /** Whole-percent labels that add up to 100 and swap when the fighters
   * swap: the favourite's share is rounded, the other is the remainder. */
  labelA: string
  labelB: string
  favourite: Side | null
}

function clamp01(p: number): number {
  if (!Number.isFinite(p)) throw new Error(`not a probability: ${p}`)
  return Math.min(1, Math.max(0, p))
}

export function favouriteOf(pA: number, pB: number): Side | null {
  if (pA > pB) return 'fighter_a'
  if (pB > pA) return 'fighter_b'
  return null
}

export function splitProbability(pA: number, pB: number): ProbabilitySplit {
  const a = clamp01(pA)
  const b = clamp01(pB)
  const favourite = favouriteOf(a, b)
  const top = Math.round(Math.max(a, b) * 100)
  let favLabel: string
  let otherLabel: string
  if (favourite === null) {
    favLabel = otherLabel = '50%'
  } else if (top >= 100) {
    favLabel = '>99%'
    otherLabel = '<1%'
  } else {
    favLabel = `${top}%`
    otherLabel = `${100 - top}%`
  }
  return {
    widthA: a * 100,
    widthB: b * 100,
    labelA: favourite === 'fighter_b' ? otherLabel : favLabel,
    labelB: favourite === 'fighter_b' ? favLabel : otherLabel,
    favourite,
  }
}

/** American odds with no bookmaker margin for a win probability, the same
 * formula as `ufc.publish.schemas.fair_moneyline`: negative for a favourite,
 * positive otherwise, +100 at exactly 50%. Rounded to cents. */
export function fairMoneyline(p: number): number {
  if (!(p > 0 && p < 1)) throw new Error(`no finite moneyline for probability ${p}`)
  const value = p > 0.5 ? (-100 * p) / (1 - p) : (100 * (1 - p)) / p
  return Math.round(value * 100) / 100
}

/** "-295" or "+295": American odds are quoted in whole units. */
export function formatMoneyline(value: number): string {
  const whole = Math.round(value)
  return whole > 0 ? `+${whole}` : `${whole}`
}
