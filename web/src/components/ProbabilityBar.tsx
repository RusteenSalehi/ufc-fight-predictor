import { splitProbability } from '../probability'

/** Two-sided bar: fighter A's share from the left in blue, fighter B's from
 * the right in orange, with a 2px gap at the split. Widths are exactly the
 * published probabilities; labels are whole percents that add up to 100. */
export function ProbabilityBar({
  pA,
  pB,
  nameA,
  nameB,
  size = 'md',
}: {
  pA: number
  pB: number
  nameA: string
  nameB: string
  size?: 'md' | 'lg'
}) {
  const split = splitProbability(pA, pB)
  const height = size === 'lg' ? 'h-4' : 'h-2.5'
  const text = size === 'lg' ? 'text-3xl sm:text-4xl' : 'text-xl'
  const fav = split.favourite
  return (
    <div>
      <div className="flex items-end justify-between gap-3">
        <span
          data-testid="label-a"
          className={`font-display font-bold tabular ${text} ${fav === 'fighter_b' ? 'text-ink-2' : 'text-ink'}`}
        >
          {split.labelA}
        </span>
        <span
          data-testid="label-b"
          className={`font-display font-bold tabular ${text} ${fav === 'fighter_a' ? 'text-ink-2' : 'text-ink'}`}
        >
          {split.labelB}
        </span>
      </div>
      <div
        role="img"
        aria-label={`Win probability: ${nameA} ${split.labelA}, ${nameB} ${split.labelB}`}
        className={`mt-1.5 flex ${height} w-full gap-[2px] overflow-hidden rounded`}
      >
        <div
          data-testid="bar-a"
          className="rounded-l bg-fighter-a"
          style={{ width: `${split.widthA}%` }}
        />
        <div
          data-testid="bar-b"
          className="rounded-r bg-fighter-b"
          style={{ width: `${split.widthB}%` }}
        />
      </div>
    </div>
  )
}
