/* eslint-disable */
/**
 * GENERATED FILE, do not edit.
 * Source: web/schema/artifacts.schema.json, exported from ufc.publish.schemas.
 * Regenerate with `npm run types:gen` (from web/).
 */

/**
 * The JSON files the site reads, generated from ufc.publish.schemas (schema_version 1). Do not edit; run `uv run python -m ufc.publish schema`.
 */
export interface PublicArtifacts {
  bout: BoutArtifact
  performance: Performance
  upcoming: Upcoming
}
export interface BoutArtifact {
  attribution: string
  bout_id: string
  event: EventInfo
  fighter_a: Fighter
  fighter_b: Fighter
  generated_at: string
  kind: 'bout'
  limited_data: LimitedData
  market: Market
  order: number
  prediction: Prediction
  recent_bouts: RecentBouts
  schema_version: 1
  segment: string | null
  stats: Stats
  title_bout: boolean
  top_factors: TopFactors | null
  weight_class: string
  why: Why | null
}
export interface EventInfo {
  date: string
  event_key: string
  name: string
  wikipedia_page: string
  wikipedia_revision: number | null
}
export interface Fighter {
  name: string
  provisional: boolean
  ref: string | null
  resolution: string
  wikipedia_page: string | null
}
export interface LimitedData {
  limited: boolean
  reasons: string[]
  stale_stats: boolean
}
/**
 * Market odds for the bout. Null until odds are matched to bouts (M12);
 * the fields exist now so that filling them is not a schema change.
 */
export interface Market {
  bookmaker: string | null
  captured_at: string | null
  fighter_a_american: number | null
  fighter_b_american: number | null
  p_fighter_a_no_vig: number | null
  p_fighter_b_no_vig: number | null
  source: string | null
}
export interface Prediction {
  fair_moneyline: Moneyline
  fair_moneyline_note: string
  model_version: string
  p_fighter_a: number
  p_fighter_b: number
  predicted_at: string
  predicted_winner: ('fighter_a' | 'fighter_b') | null
  prediction_id: number
}
export interface Moneyline {
  fighter_a: number
  fighter_b: number
}
export interface RecentBouts {
  /**
   * @maxItems 5
   */
  fighter_a: PastBout[]
  /**
   * @maxItems 5
   */
  fighter_b: PastBout[]
  note: string
  performance_model: {
    [k: string]: string | number | null
  }
}
export interface PastBout {
  date: string
  deserved_win_probability: number | null
  end_round: number | null
  end_time: string | null
  event: string
  method: string | null
  opponent: string
  opponent_id: string
  result: 'win' | 'loss' | 'draw' | 'nc'
  title_bout: boolean
  weight_class: string | null
}
export interface Stats {
  attribute_sources: {
    fighter_a?: {
      [k: string]: string | null
    }
    fighter_b?: {
      [k: string]: string | null
    }
  }
  note: string
  rows: StatRow[]
  shrink_minutes: number | null
  stance: {
    fighter_a?: string | null
    fighter_b?: string | null
  }
}
export interface StatRow {
  fighter_a: number | null
  fighter_b: number | null
  group: string
  key: string
  kind: string
  label: string
  shrunk: boolean
}
export interface TopFactors {
  fighter_a: Reason[]
  fighter_b: Reason[]
  method: string
}
export interface Reason {
  component: string | null
  detail: string
  inputs: string[]
  log_odds: number
  reason: string
  title: string
  toward: 'fighter_a' | 'fighter_b'
}
export interface Why {
  against: Reason | null
  favoured: ('fighter_a' | 'fighter_b') | null
  favoured_name: string | null
  /**
   * @maxItems 3
   */
  for: Reason[]
  method: string
  summary: string
}
export interface Performance {
  backtest: Backtest | null
  backtest_unavailable: string | null
  era_shift_caveat: EraShift | null
  generated_at: string
  kind: 'performance'
  live: LiveRecord
  model: ModelInfo
  schema_version: 1
  separation: string
}
export interface Backtest {
  candidate_version: string
  commands: string[]
  label: string
  production_note: string
  production_version: string
  reports: string
  test_period: HeldOutPeriod
  walk_forward: WalkForward
}
export interface HeldOutPeriod {
  already_seen: boolean
  already_seen_note: string | null
  calibration: CalibrationRow[]
  calibration_buckets: string
  candidate_version: string
  first_date: string
  last_date: string
  metrics: {
    [k: string]: Metrics
  }
  n_no_result: number
  n_scored: number
}
export interface CalibrationRow {
  favourite_win_rate: number | null
  high: number
  log_loss: number | null
  low: number
  mean_confidence: number | null
  n: number
  win_rate_high: number | null
  win_rate_low: number | null
}
export interface Metrics {
  accuracy: number
  brier: number
  ece: number | null
  log_loss: number
  log_loss_high: number | null
  log_loss_low: number | null
  n: number
}
export interface WalkForward {
  folds: Fold[]
  pooled_before_test_period: Pooled
  pooled_test_period: Pooled | null
  schedule: string
}
export interface Fold {
  in_test_period: boolean
  metrics: {
    [k: string]: Metrics
  }
  n_scored: number
  year: number
}
export interface Pooled {
  metrics: {
    [k: string]: Metrics
  }
  /**
   * @minItems 2
   * @maxItems 2
   */
  years: [number, number]
}
export interface EraShift {
  backtest_log_loss: number
  /**
   * @minItems 2
   * @maxItems 2
   */
  backtest_year_range: [number, number]
  /**
   * @minItems 2
   * @maxItems 2
   */
  backtest_years: [number, number]
  test_period_log_loss: number
  text: string
}
export interface LiveRecord {
  bouts: LiveBout[]
  by_model_version: {
    [k: string]: Metrics
  }
  calibration: CalibrationRow[]
  calibration_buckets: string
  counts: {
    scored?: number
    no_result?: number
    pending?: number
    cancelled?: number
    no_live_prediction?: number
    not_found?: number
    unlinked?: number
  }
  label: string
  metrics: Metrics | null
  metrics_without_limited_data: Metrics | null
  rules: string[]
}
export interface LiveBout {
  bout_id: string
  brier: number | null
  correct: number | null
  detail: string | null
  event_date: string
  event_key: string
  event_name: string
  fighter_a: string
  fighter_b: string
  limited_data: boolean | null
  log_loss: number | null
  model_version: string | null
  outcome: ('fighter_a' | 'fighter_b' | 'draw' | 'nc') | null
  p_fighter_a: number | null
  predicted_at: string | null
  prediction_id: number | null
  result_bout_id: string | null
  status:
    | 'scored'
    | 'no_result'
    | 'pending'
    | 'cancelled'
    | 'no_live_prediction'
    | 'not_found'
    | 'unlinked'
}
export interface ModelInfo {
  parent: string | null
  status: string
  version: string
}
export interface Upcoming {
  attribution: string
  dataset: Dataset
  /**
   * @minItems 1
   */
  events: [UpcomingEvent, ...UpcomingEvent[]]
  generated_at: string
  kind: 'upcoming'
  model: ModelInfo
  schema_version: 1
  staleness: Staleness
}
export interface Dataset {
  last_event: string | null
  source_commit: string | null
}
export interface UpcomingEvent {
  bouts: UpcomingBout[]
  card_checked_at: string | null
  date: string
  event_key: string
  name: string
  unresolved_bouts: UnpredictedBout[]
  wikipedia_page: string
  wikipedia_revision: number | null
}
export interface UpcomingBout {
  bout_id: string
  detail: string
  fighter_a: FighterBrief
  fighter_b: FighterBrief
  limited_data: boolean
  limited_data_reasons: string[]
  model_version: string
  order: number
  p_fighter_a: number
  p_fighter_b: number
  predicted_at: string
  predicted_winner: ('fighter_a' | 'fighter_b') | null
  segment: string | null
  stale_stats: boolean
  status: 'predicted'
  title_bout: boolean
  weight_class: string
}
export interface FighterBrief {
  name: string
  provisional: boolean
  ref: string | null
  wikipedia_page: string | null
}
export interface UnpredictedBout {
  bout_id: string
  fighter_a: FighterBrief
  fighter_b: FighterBrief
  order: number
  reason: string
  segment: string | null
  status: 'unresolved' | 'not_predicted'
  title_bout: boolean
  weight_class: string
}
export interface Staleness {
  dataset_last_event: string | null
  missing_events: MissingEvent[]
  status: 'current' | 'behind' | 'unknown'
}
export interface MissingEvent {
  bouts: number | null
  date: string
  name: string
  page: string
  read: boolean
}
