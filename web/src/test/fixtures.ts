/** The synthetic fixtures (scripts/make_web_fixtures.py), typed. JSON
 * imports widen tuples to arrays, so each is cast once here; pytest checks
 * that every fixture is a valid artifact. */
import boutJson from './fixtures/bout.json'
import perfEmptyJson from './fixtures/performance-empty.json'
import perfLiveJson from './fixtures/performance-live.json'
import upcomingJson from './fixtures/upcoming.json'
import type { BoutArtifact, Performance, Upcoming } from '../types/artifacts'

export const upcoming = upcomingJson as unknown as Upcoming
export const bout = boutJson as unknown as BoutArtifact
export const performanceEmpty = perfEmptyJson as unknown as Performance
export const performanceLive = perfLiveJson as unknown as Performance
