import { describe, expect, test } from 'vitest'
import { fairMoneyline, formatMoneyline, splitProbability } from './probability'

describe('splitProbability', () => {
  test('bar widths are exactly the probabilities', () => {
    const s = splitProbability(0.252907683019031, 0.747092316980969)
    expect(s.widthA).toBeCloseTo(25.2907683019031, 10)
    expect(s.widthB).toBeCloseTo(74.7092316980969, 10)
    expect(s.widthA + s.widthB).toBeCloseTo(100, 12)
    expect(s.favourite).toBe('fighter_b')
  })

  test('labels are whole percents that add up to 100', () => {
    for (const p of [0.01, 0.125, 0.3, 0.495, 0.505, 0.555, 0.6249, 0.875, 0.99]) {
      const s = splitProbability(p, 1 - p)
      expect(parseInt(s.labelA) + parseInt(s.labelB)).toBe(100)
    }
    // naive rounding would show 51% + 50% here
    expect(splitProbability(0.505, 0.495)).toMatchObject({ labelA: '51%', labelB: '49%' })
  })

  test('swapping the fighters swaps the labels exactly', () => {
    for (const p of [0.125, 0.505, 0.62, 0.995]) {
      const ab = splitProbability(p, 1 - p)
      const ba = splitProbability(1 - p, p)
      expect([ba.labelA, ba.labelB]).toEqual([ab.labelB, ab.labelA])
      expect(ba.widthA).toBe(ab.widthB)
    }
  })

  test('an even bout and the extremes', () => {
    expect(splitProbability(0.5, 0.5)).toMatchObject({ labelA: '50%', labelB: '50%', favourite: null })
    expect(splitProbability(0.997, 0.003)).toMatchObject({ labelA: '>99%', labelB: '<1%' })
    expect(splitProbability(0.003, 0.997)).toMatchObject({ labelA: '<1%', labelB: '>99%' })
  })

  test('refuses a non-number', () => {
    expect(() => splitProbability(Number.NaN, 0.5)).toThrow()
  })
})

describe('fairMoneyline', () => {
  // Expected values from ufc.publish.schemas.fair_moneyline.
  test.each([
    [0.747092316980969, -295.4],
    [0.252907683019031, 295.4],
    [0.5, 100],
    [0.6, -150],
    [0.4, 150],
    [0.9, -900],
    [0.51, -104.08],
  ])('p = %s gives %s, as the publisher computes it', (p, expected) => {
    expect(fairMoneyline(p)).toBeCloseTo(expected, 2)
  })

  test('has no finite value at 0 or 1', () => {
    expect(() => fairMoneyline(0)).toThrow()
    expect(() => fairMoneyline(1)).toThrow()
  })

  test('is quoted in whole units with a sign', () => {
    expect(formatMoneyline(-295.4)).toBe('-295')
    expect(formatMoneyline(295.4)).toBe('+295')
    expect(formatMoneyline(100)).toBe('+100')
  })
})
