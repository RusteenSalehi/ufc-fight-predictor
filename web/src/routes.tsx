import type { RouteObject } from 'react-router'
import { Layout } from './components/Layout'
import { RouteError } from './components/StatusViews'
import { Home } from './pages/Home'
import { Matchup } from './pages/Matchup'
import { Methodology } from './pages/Methodology'
import { NotFound } from './pages/NotFound'

/** Every path the site serves. A static host must answer each of them with
 * index.html (ARCHITECTURE.md, "Frontend hosting"). */
export const routes: RouteObject[] = [
  {
    path: '/',
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Home /> },
      { path: 'bout/:boutId', element: <Matchup /> },
      {
        // Recharts is only needed here, so it loads with this page.
        path: 'performance',
        lazy: async () => ({ Component: (await import('./pages/Performance')).PerformancePage }),
      },
      { path: 'methodology', element: <Methodology /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]
