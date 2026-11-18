import { formatStat } from '../format'
import type { Stats } from '../types/artifacts'

const GROUP_TITLES: Record<string, string> = {
  physical: 'Physical',
  record: 'UFC record',
  rating: 'Rating',
  activity: 'Activity',
  striking: 'Striking',
  grappling: 'Grappling',
}

// Which attribute a stat row's value comes from (attribute_sources keys).
const ATTRIBUTE_OF: Record<string, string> = { age: 'dob', height_cm: 'height_cm', reach_cm: 'reach_cm' }

function fromWikipedia(stats: Stats, side: 'fighter_a' | 'fighter_b', key: string): boolean {
  const attr = ATTRIBUTE_OF[key] ?? key
  return stats.attribute_sources[side]?.[attr] === 'wikipedia'
}

function Value({ text, wiki }: { text: string; wiki: boolean }) {
  // "193 cm (6′4″)": the imperial conversion goes on its own line so a
  // phone-width table keeps both fighters' columns on screen.
  const split = /^(.*) \((.*)\)$/.exec(text)
  return (
    <>
      {split ? (
        <>
          {split[1]}
          <span className="block text-xs text-muted">{split[2]}</span>
        </>
      ) : (
        text
      )}
      {wiki && (
        <sup className="ml-0.5 text-muted" title="From Wikipedia">
          W
        </sup>
      )}
    </>
  )
}

export function StatComparison({ stats, nameA, nameB }: { stats: Stats; nameA: string; nameB: string }) {
  const groups: { group: string; rows: Stats['rows'] }[] = []
  for (const row of stats.rows) {
    const last = groups.at(-1)
    if (last && last.group === row.group) last.rows.push(row)
    else groups.push({ group: row.group, rows: [row] })
  }
  const anyShrunk = stats.rows.some((r) => r.shrunk)
  const anyWiki = (['fighter_a', 'fighter_b'] as const).some((side) =>
    Object.values(stats.attribute_sources[side] ?? {}).includes('wikipedia'),
  )
  return (
    <div>
      <div className="overflow-hidden rounded-xl border border-line">
        <table className="w-full table-fixed text-xs tabular sm:text-sm">
          <caption className="sr-only">As-of statistics for {nameA} and {nameB}</caption>
          <thead className="bg-surface-2 text-xs tracking-wide text-ink-2 uppercase">
            <tr>
              <th scope="col" className="w-[44%] px-2 py-2 text-left font-semibold sm:px-3">
                Stat
              </th>
              <th scope="col" className="px-2 py-2 text-right font-semibold sm:px-3">
                <span className="flex min-w-0 items-center justify-end gap-1.5">
                  <span aria-hidden="true" className="h-2 w-2 rounded-sm bg-fighter-a" />
                  <span className="truncate">{nameA}</span>
                </span>
              </th>
              <th scope="col" className="px-2 py-2 text-right font-semibold sm:px-3">
                <span className="flex min-w-0 items-center justify-end gap-1.5">
                  <span aria-hidden="true" className="h-2 w-2 rounded-sm bg-fighter-b" />
                  <span className="truncate">{nameB}</span>
                </span>
              </th>
            </tr>
          </thead>
          {groups.map(({ group, rows }) => (
            <tbody key={group} className="border-t border-line">
              <tr>
                <th colSpan={3} scope="colgroup" className="bg-surface px-2 pt-3 pb-1 text-left sm:px-3 text-xs font-semibold tracking-wide text-muted uppercase">
                  {GROUP_TITLES[group] ?? group}
                </th>
              </tr>
              {rows.map((row) => (
                <tr key={row.key} className="border-t border-line/60 bg-surface">
                  <th scope="row" className="px-2 py-1.5 text-left font-normal text-ink-2 sm:px-3">
                    <span className="first-letter:uppercase">{row.label}</span>
                    {row.shrunk && <span className="text-muted" title="Shrunk toward the weight-class average">†</span>}
                  </th>
                  <td className="px-2 py-1.5 text-right sm:px-3">
                    <Value text={formatStat(row.fighter_a, row.kind)} wiki={fromWikipedia(stats, 'fighter_a', row.key)} />
                  </td>
                  <td className="px-2 py-1.5 text-right sm:px-3">
                    <Value text={formatStat(row.fighter_b, row.kind)} wiki={fromWikipedia(stats, 'fighter_b', row.key)} />
                  </td>
                </tr>
              ))}
              {group === 'physical' && (
                <tr className="border-t border-line/60 bg-surface">
                  <th scope="row" className="px-2 py-1.5 text-left font-normal text-ink-2 sm:px-3">
                    Stance
                  </th>
                  <td className="px-2 py-1.5 text-right sm:px-3">
                    <Value text={stats.stance.fighter_a ?? '—'} wiki={fromWikipedia(stats, 'fighter_a', 'stance')} />
                  </td>
                  <td className="px-2 py-1.5 text-right sm:px-3">
                    <Value text={stats.stance.fighter_b ?? '—'} wiki={fromWikipedia(stats, 'fighter_b', 'stance')} />
                  </td>
                </tr>
              )}
            </tbody>
          ))}
        </table>
      </div>
      <div className="mt-2 space-y-1 text-xs leading-relaxed text-muted">
        <p>{stats.note}</p>
        {anyShrunk && (
          <p>
            † Shrunk toward the weight-class average
            {stats.shrink_minutes !== null ? ` by ${stats.shrink_minutes} cage minutes` : ''}, as the model reads it.
          </p>
        )}
        {anyWiki && <p>W: taken from the fighter’s Wikipedia infobox because the dataset has no value.</p>}
        <p>— means no value: no recorded UFC bouts with stats, or the attribute is unknown.</p>
      </div>
    </div>
  )
}
