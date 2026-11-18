import { describe, expect, test } from 'vitest'
import { splitEvents } from './events'
import { formatEventDate, formatStat, formatTimestamp } from './format'
import type { UpcomingEvent } from './types/artifacts'

describe('formatting', () => {
  test('event dates never shift with the time zone', () => {
    expect(formatEventDate('2026-11-21')).toBe('Sat 21 Nov 2026')
    expect(formatEventDate('2027-01-01', false)).toBe('1 Jan 2027')
  })

  test('timestamps are shown in UTC', () => {
    expect(formatTimestamp('2026-11-18T05:18:27Z')).toBe('18 Nov 2026, 05:18 UTC')
  })

  test('stat values by kind', () => {
    expect(formatStat(34.10266, 'years')).toBe('34.1')
    expect(formatStat(190.5, 'cm')).toBe('191 cm (6′3″)')
    expect(formatStat(0.555, 'pct')).toBe('56%')
    expect(formatStat(1731.25, 'elo')).toBe('1,731')
    expect(formatStat(5.30508, 'rate')).toBe('5.31')
    expect(formatStat(null, 'rate')).toBe('—')
  })
})

describe('splitEvents', () => {
  const ev = (date: string) => ({ date, event_key: `wp:${date}` }) as UpcomingEvent

  test('the next event is the first dated today or later', () => {
    const s = splitEvents([ev('2026-11-14'), ev('2026-11-21'), ev('2026-12-05')], '2026-11-18')
    expect(s.next?.date).toBe('2026-11-21')
    expect(s.later.map((e) => e.date)).toEqual(['2026-12-05'])
    expect(s.outdated).toBe(false)
  })

  test('an event is still next on its own date', () => {
    expect(splitEvents([ev('2026-11-21')], '2026-11-21').next?.date).toBe('2026-11-21')
  })

  test('only past events: shows the last, marked outdated', () => {
    const s = splitEvents([ev('2026-11-01'), ev('2026-11-08')], '2026-11-18')
    expect(s).toMatchObject({ outdated: true, later: [] })
    expect(s.next?.date).toBe('2026-11-08')
  })
})
