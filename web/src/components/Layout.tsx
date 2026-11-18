import { Link, NavLink, Outlet, ScrollRestoration } from 'react-router'
import { DISCLAIMER, REPO_URL, SITE_NAME } from '../config'

const NAV = [
  { to: '/', label: 'Upcoming', end: true },
  { to: '/performance', label: 'Performance', end: false },
  { to: '/methodology', label: 'Methodology', end: false },
]

function Wordmark() {
  return (
    <Link to="/" className="group flex items-center gap-2.5" aria-label={`${SITE_NAME} home`}>
      <span aria-hidden="true" className="flex h-3 w-9 overflow-hidden rounded-sm">
        <span className="w-[58%] bg-fighter-a" />
        <span className="w-[2px] bg-page" />
        <span className="flex-1 bg-fighter-b" />
      </span>
      <span className="font-display text-xl font-bold tracking-wide uppercase">{SITE_NAME}</span>
    </Link>
  )
}

export function Layout() {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:rounded focus:bg-surface-2 focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-20 border-b border-line bg-page/85 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-3">
          <Wordmark />
          <nav aria-label="Main">
            <ul className="flex gap-1 text-sm">
              {NAV.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) =>
                      `rounded-md px-2.5 py-1.5 font-medium transition-colors ${
                        isActive ? 'bg-surface-2 text-ink' : 'text-ink-2 hover:text-ink'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-5xl flex-1 px-4 pt-6 pb-16 sm:pt-10">
        <Outlet />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-5xl space-y-2 px-4 py-6 text-xs text-muted">
          <p className="text-sm text-ink-2">{DISCLAIMER}</p>
          <p>
            Fight cards and some fighter attributes from English Wikipedia (CC BY-SA 4.0). Fight
            statistics from the Greco1899/scrape_ufc_stats dataset, derived from ufcstats.com.{' '}
            <a className="link" href={REPO_URL} rel="noreferrer">
              Source code
            </a>
            .
          </p>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  )
}
