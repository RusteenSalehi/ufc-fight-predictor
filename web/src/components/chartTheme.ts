/** Chart ink, from the same tokens as index.css (Recharts needs literal
 * values for SVG attributes). The model is the accent; baselines are greys,
 * so the one series the page is about stands out. */
export const CHART = {
  model: '#d4f25a',
  logistic: '#c3c2b7',
  elo: '#8f8d86',
  grid: '#2c2c2a',
  axis: '#4a4a46',
  tick: '#8f8d86',
  surface: '#16181c',
  ink: '#f0efec',
  ink2: '#c3c2b7',
  testShade: 'rgba(255,255,255,0.05)',
} as const

export const tickStyle = { fill: CHART.tick, fontSize: 12 }
