import { render, screen, within } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { performanceEmpty, performanceLive } from '../test/fixtures'
import { groupReasons } from '../labels'
import { DataLabels } from './DataLabels'
import { LiveRecord } from './LiveRecord'
import { ProbabilityBar } from './ProbabilityBar'

describe('ProbabilityBar', () => {
  test('draws each side at its probability and labels both', () => {
    render(<ProbabilityBar pA={0.3} pB={0.7} nameA="Avery Stone" nameB="Bruno Vale" />)
    expect(screen.getByTestId('bar-a')).toHaveStyle({ width: '30%' })
    expect(screen.getByTestId('bar-b')).toHaveStyle({ width: '70%' })
    expect(screen.getByTestId('label-a')).toHaveTextContent('30%')
    expect(screen.getByTestId('label-b')).toHaveTextContent('70%')
    expect(screen.getByRole('img')).toHaveAccessibleName('Win probability: Avery Stone 30%, Bruno Vale 70%')
  })

  test('labels add up to 100 where rounding each side would not', () => {
    render(<ProbabilityBar pA={0.495} pB={0.505} nameA="A" nameB="B" />)
    expect(screen.getByTestId('label-a')).toHaveTextContent('49%')
    expect(screen.getByTestId('label-b')).toHaveTextContent('51%')
  })
})

describe('DataLabels', () => {
  const reasons = [
    'debut: Tomas Newman has no earlier UFC bout in the dataset',
    'attribute gap: no height, reach for Tomas Newman',
    'stale stats: June Holt fought at an event the dataset lacks',
  ]

  test('stale-stats reasons get their own label', () => {
    expect(groupReasons(reasons)).toEqual({ limited: reasons.slice(0, 2), stale: reasons.slice(2) })
  })

  test('renders each label with its reasons as text', () => {
    render(<DataLabels reasons={reasons} />)
    expect(screen.getByText('Limited data')).toBeInTheDocument()
    expect(screen.getByText('Stale stats')).toBeInTheDocument()
    expect(screen.getByText('Debut: Tomas Newman has no earlier UFC bout in the dataset')).toBeInTheDocument()
    expect(screen.getByText('Attribute gap: no height, reach for Tomas Newman')).toBeInTheDocument()
    expect(screen.getByText('Stale stats: June Holt fought at an event the dataset lacks')).toBeInTheDocument()
  })

  test('only the labels that apply', () => {
    render(<DataLabels reasons={['stale stats: someone fought recently']} />)
    expect(screen.queryByText('Limited data')).not.toBeInTheDocument()
    expect(screen.getByText('Stale stats')).toBeInTheDocument()
  })

  test('nothing for a bout with full data', () => {
    const { container } = render(<DataLabels reasons={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})

describe('LiveRecord', () => {
  test('an honest empty state before any result exists', () => {
    render(<LiveRecord live={performanceEmpty.live} />)
    expect(screen.getByText('No live results yet')).toBeInTheDocument()
    expect(screen.getByTestId('live-empty')).toHaveTextContent('Nothing has been scored yet')
    // no metric is shown when nothing is scored
    expect(screen.queryByText('Log loss')).not.toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  test('scored bouts show metrics and the bout list', () => {
    render(<LiveRecord live={performanceLive.live} />)
    expect(screen.queryByText('No live results yet')).not.toBeInTheDocument()
    expect(screen.getAllByText('Log loss').length).toBeGreaterThan(0)
    const table = screen.getByRole('table', { name: 'Live predictions and their results' })
    expect(within(table).getByText('Lena Moss won')).toBeInTheDocument()
    expect(within(table).getByText('Waiting for results')).toBeInTheDocument()
  })
})
