import type { UpcomingEvent } from './types/artifacts'

export interface EventSplit {
  next: UpcomingEvent | null
  later: UpcomingEvent[]
  /** True when every published event is dated before today: the refresh
   * has not run since, so the page says so instead of hiding the card. */
  outdated: boolean
}

/** The next event is the first published event dated today or later (UTC);
 * the publisher writes events in date order. */
export function splitEvents(events: UpcomingEvent[], today: string): EventSplit {
  const ahead = events.filter((e) => e.date >= today)
  if (ahead.length === 0) {
    const last = events.at(-1) ?? null
    return { next: last, later: [], outdated: last !== null }
  }
  return { next: ahead[0] ?? null, later: ahead.slice(1), outdated: false }
}
