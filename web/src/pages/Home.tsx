import { fetchUpcoming } from '../api/client'
import { useData } from '../api/useData'
import { BoutCard, UnpredictedCard } from '../components/BoutCard'
import { usePageTitle } from '../usePageTitle'
import { Callout, ErrorState, Loading } from '../components/StatusViews'
import { splitEvents } from '../events'
import { formatEventDate, formatTimestamp, todayUtc } from '../format'
import type { Staleness, Upcoming, UpcomingEvent } from '../types/artifacts'

function StalenessNote({ staleness }: { staleness: Staleness }) {
  if (staleness.status === 'current') return null
  if (staleness.status === 'unknown') {
    return (
      <Callout tone="caution" title="Stats freshness unknown">
        The list of completed events could not be checked against the dataset at the last refresh.
      </Callout>
    )
  }
  return (
    <Callout tone="caution" title="The stats are behind">
      The dataset ends on{' '}
      {staleness.dataset_last_event ? formatEventDate(staleness.dataset_last_event, false) : 'an unknown date'}{' '}
      and does not yet include:{' '}
      {staleness.missing_events.map((e) => `${e.name} (${formatEventDate(e.date, false)})`).join('; ')}. Bouts
      with a fighter who fought there are labelled “stale stats”.
    </Callout>
  )
}

function Card({ event }: { event: UpcomingEvent }) {
  return (
    <div className="space-y-3">
      {event.bouts.map((b) => (
        <BoutCard key={b.bout_id} bout={b} />
      ))}
      {event.unresolved_bouts.map((b) => (
        <UnpredictedCard key={b.bout_id} bout={b} />
      ))}
    </div>
  )
}

function boutCount(event: UpcomingEvent): string {
  const n = event.bouts.length
  const u = event.unresolved_bouts.length
  return `${n} predicted bout${n === 1 ? '' : 's'}${u ? `, ${u} not predicted` : ''}`
}

function HomeContent({ upcoming }: { upcoming: Upcoming }) {
  const { next, later, outdated } = splitEvents(upcoming.events, todayUtc())
  usePageTitle(next ? next.name : null)
  if (!next) return null
  return (
    <div>
      <header>
        <p className="text-xs font-semibold tracking-[0.18em] text-accent uppercase">
          {outdated ? 'Most recent published card' : 'Next event'}
        </p>
        <h1 className="mt-1 font-display text-4xl leading-none font-bold tracking-wide uppercase sm:text-5xl">
          {next.name}
        </h1>
        <p className="mt-2 text-ink-2">{formatEventDate(next.date)}</p>
        <p className="mt-1 text-xs text-muted">
          Last updated {formatTimestamp(upcoming.generated_at)} · model{' '}
          <code className="text-ink-2">{upcoming.model.version}</code>
        </p>
      </header>

      <div className="mt-5 space-y-3">
        {outdated && (
          <Callout tone="caution" title="No upcoming card has been published since this event">
            The predictions below are from the last refresh and the event date has passed.
          </Callout>
        )}
        <StalenessNote staleness={upcoming.staleness} />
      </div>

      <div className="mt-6">
        <Card event={next} />
      </div>

      {later.length > 0 && (
        <section aria-labelledby="later" className="mt-14">
          <h2 id="later" className="font-display text-2xl font-semibold tracking-wide uppercase">
            Later events
          </h2>
          <div className="mt-3 space-y-2">
            {later.map((event) => (
              <details key={event.event_key} className="group rounded-xl border border-line bg-surface">
                <summary className="flex cursor-pointer list-none flex-wrap items-baseline justify-between gap-x-4 gap-y-1 p-4 [&::-webkit-details-marker]:hidden">
                  <span className="font-display text-lg font-semibold tracking-wide uppercase">
                    <span aria-hidden="true" className="mr-2 inline-block text-muted transition-transform group-open:rotate-90">
                      ›
                    </span>
                    {event.name}
                  </span>
                  <span className="text-sm text-muted">
                    {formatEventDate(event.date)} · {boutCount(event)}
                  </span>
                </summary>
                <div className="border-t border-line p-3 sm:p-4">
                  <Card event={event} />
                </div>
              </details>
            ))}
          </div>
        </section>
      )}

      <p className="mt-12 text-xs leading-relaxed text-muted">{upcoming.attribution}</p>
    </div>
  )
}

export function Home() {
  const state = useData('upcoming', fetchUpcoming)
  if (state.status === 'loading') return <Loading label="Loading upcoming events" />
  if (state.status === 'error')
    return <ErrorState title="Could not load upcoming events" error={state.error} onRetry={state.retry} />
  return <HomeContent upcoming={state.data} />
}
