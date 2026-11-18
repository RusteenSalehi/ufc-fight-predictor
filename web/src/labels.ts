/** Words shared by several components. */

export interface LabelGroups {
  limited: string[]
  stale: string[]
}

/** Splits the published reasons: those starting "stale stats" belong to the
 * stale-stats label (the schema's own rule), the rest to limited data. */
export function groupReasons(reasons: string[]): LabelGroups {
  const stale = reasons.filter((r) => r.startsWith('stale stats'))
  const limited = reasons.filter((r) => !r.startsWith('stale stats'))
  return { limited, stale }
}

export const MODEL_LABELS: Record<string, string> = {
  model: 'This model',
  logistic: 'Logistic regression (baseline)',
  elo: 'Elo rating (baseline)',
}
