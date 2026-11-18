import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, test } from 'vitest'
import { bout, performanceEmpty as emptyPerf, upcoming } from '../test/fixtures'
import { renderAt, stubData } from '../test/render'

describe('Home', () => {
  test('shows the next event with every bout, and later events below', async () => {
    stubData({ 'upcoming.json': upcoming })
    renderAt('/')
    expect(await screen.findByRole('heading', { level: 1, name: 'Test Fight Night: Stone vs. Vale' })).toBeInTheDocument()
    const cards = screen.getAllByRole('article')
    // three predicted bouts and one unresolved on the next card, then the
    // later event's one bout (inside a closed <details>)
    expect(cards).toHaveLength(5)
    const first = cards[0]!
    expect(within(first).getByText('Bruno Vale', { selector: 'span.font-semibold' })).toBeInTheDocument()
    expect(within(first).getByText('Fair line +233')).toBeInTheDocument()
    expect(within(first).getByText('Fair line -233')).toBeInTheDocument()
    expect(within(cards[1]!).getByText('Limited data')).toBeInTheDocument()
    expect(within(cards[2]!).getByText('Stale stats')).toBeInTheDocument()
    expect(within(cards[2]!).getByText('Even')).toBeInTheDocument()
    expect(within(cards[3]!).getByText(/Not predicted: ambiguous name/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Later events' })).toBeInTheDocument()
    expect(screen.getByText('Test 900: Okafor vs. Lind')).toBeInTheDocument()
  })

  test('a failed fetch shows an error with a retry', async () => {
    stubData({})
    renderAt('/')
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not load upcoming events')
    stubData({ 'upcoming.json': upcoming })
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('heading', { level: 1 })).toHaveTextContent('Stone vs. Vale')
  })

  test('refuses data in a schema version it does not read', async () => {
    stubData({ 'upcoming.json': { ...upcoming, schema_version: 2 } })
    renderAt('/')
    expect(await screen.findByRole('alert')).toHaveTextContent('schema version 2')
  })
})

describe('Matchup', () => {
  test('probability, model version, why, factors, stats and recent bouts', async () => {
    stubData({ 'bouts/wb0000000000000001.json': bout })
    renderAt('/bout/wb0000000000000001')
    expect(await screen.findByRole('heading', { level: 1 })).toHaveTextContent('Avery Stone')
    expect(screen.getByText(/xgboost-0000aaaa-1111bbbb/)).toBeInTheDocument()
    expect(screen.getByText(/predicted 2 Jan 2099, 11:30 UTC/)).toBeInTheDocument()
    expect(screen.getByText('The model favours Bruno Vale (70%). Synthetic summary.')).toBeInTheDocument()
    expect(screen.getAllByTestId('factor-fighter_a')).toHaveLength(2)
    expect(screen.getAllByTestId('factor-fighter_b')).toHaveLength(3)
    expect(screen.getByText(/not causes of a result/)).toBeInTheDocument()
    const reach = screen.getByRole('row', { name: /^reach/ })
    expect(within(reach).getAllByRole('cell').map((c) => c.textContent)).toEqual(['193 cm6′4″', '—'])
    expect(screen.getByText('Southpaw')).toBeInTheDocument()
    expect(screen.getByText('Describes past fights only.')).toBeInTheDocument()
    expect(screen.getByText('81%')).toBeInTheDocument()
    expect(screen.getByText('no stats')).toBeInTheDocument()
  })

  test('an unknown bout is a friendly not-found, not an error', async () => {
    stubData({})
    renderAt('/bout/wb00000000000000ff')
    expect(await screen.findByText('Bout not found')).toBeInTheDocument()
  })

  test('a malformed bout ID is not fetched', async () => {
    const fetchMock = stubData({})
    renderAt('/bout/../../upcoming')
    expect(await screen.findByText(/not found/i)).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('Performance', () => {
  test('backtest with its caveats, and an empty live record', async () => {
    stubData({ 'performance.json': emptyPerf })
    renderAt('/performance')
    expect(await screen.findByRole('heading', { level: 1, name: 'Model performance' })).toBeInTheDocument()
    expect(screen.getByText('Already seen: not a held-out result')).toBeInTheDocument()
    expect(screen.getByText('Synthetic era-shift caveat.')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Test-period metrics' })).toBeInTheDocument()
    expect(screen.getByText('No live results yet')).toBeInTheDocument()
  })
})

describe('Static pages', () => {
  test('methodology links to the repository', () => {
    renderAt('/methodology')
    expect(screen.getByRole('heading', { level: 1, name: 'Methodology' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'GitHub repository' })).toHaveAttribute(
      'href',
      'https://github.com/RusteenSalehi/ufc-fight-predictor',
    )
  })

  test('the footer disclaimer is on every page', () => {
    renderAt('/nowhere')
    expect(screen.getByText('Page not found')).toBeInTheDocument()
    expect(
      screen.getByText('Not affiliated with the UFC. Predictions are statistical estimates, not betting advice.'),
    ).toBeInTheDocument()
  })
})
