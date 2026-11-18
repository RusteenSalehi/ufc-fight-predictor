import type { ReactNode } from 'react'
import { Link } from 'react-router'
import { usePageTitle } from '../usePageTitle'
import { REPO_URL, SITE_NAME } from '../config'

/* Written from docs/ (ARCHITECTURE, ML_REQUIREMENTS, DECISIONS, MODEL_REPORT).
 * Metrics live on the Performance page, read from performance.json. The two
 * figures quoted here come from repo commands named next to them:
 * `uv run python -m ufc.ingest load --dry-run` (dropped fights, snapshot
 * 96adace) and the generated block of docs/MODEL_REPORT.md written by
 * `uv run python -m ufc.models report` (the attribute-gap leak). */

const doc = (name: string) => `${REPO_URL}/blob/main/docs/${name}`

const SECTIONS = [
  { id: 'data', title: 'Data sources' },
  { id: 'leakage', title: 'Leakage prevention' },
  { id: 'features', title: 'Features' },
  { id: 'models', title: 'Models' },
  { id: 'evaluation', title: 'Evaluation' },
  { id: 'limitations', title: 'Limitations' },
  { id: 'code', title: 'Source code' },
]

function Block({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="scroll-mt-20 border-t border-line pt-8">
      <h2 id={`${id}-h`} className="font-display text-2xl font-semibold tracking-wide uppercase">
        {title}
      </h2>
      <div className="prose-site mt-3 max-w-prose">{children}</div>
    </section>
  )
}

export function Methodology() {
  usePageTitle('Methodology')
  return (
    <div>
      <header>
        <p className="text-xs font-semibold tracking-[0.18em] text-accent uppercase">How it works</p>
        <h1 className="mt-1 font-display text-4xl leading-none font-bold tracking-wide uppercase sm:text-5xl">
          Methodology
        </h1>
        <p className="mt-3 max-w-prose text-ink-2">
          {SITE_NAME} publishes a win probability for every bout on upcoming UFC cards, explains what the model weighed,
          and scores itself in public. Every number on this site is produced by code in the{' '}
          <a className="link" href={REPO_URL} rel="noreferrer">
            public repository
          </a>{' '}
          and can be reproduced with a documented command.
        </p>
        <nav aria-label="On this page" className="mt-5 flex flex-wrap gap-2 text-sm">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} className="rounded-full border border-line px-3 py-1 text-ink-2 hover:border-line-strong hover:text-ink">
              {s.title}
            </a>
          ))}
        </nav>
      </header>

      <div className="mt-8 space-y-10">
        <Block id="data" title="Data sources">
          <p>
            <strong className="text-ink">Completed fights</strong> come from the{' '}
            <a href="https://github.com/Greco1899/scrape_ufc_stats" rel="noreferrer">
              Greco1899/scrape_ufc_stats
            </a>{' '}
            dataset: CSV files with one row per fighter per round of every UFC bout, collected from ufcstats.com and
            pinned here to a specific upstream commit. This project never requests ufcstats.com itself. Its pages sit
            behind a bot challenge, and the project does not work around it.
          </p>
          <p>
            The dataset is itself a scrape of ufcstats.com, and whether that site’s owner permits this use is not
            settled. The project is non-commercial, commits none of the data, and can swap the source behind one
            module if asked to (<a href={doc('DECISIONS.md')}>DECISIONS</a> D13 to D15).
          </p>
          <p>
            <strong className="text-ink">Upcoming cards</strong> come from English Wikipedia through the MediaWiki
            API (CC BY-SA 4.0). Wikipedia is edited by volunteers, so a card is treated as untrusted input. A card that
            cannot be read, or that suddenly loses most of its bouts, changes nothing. Fighter names are matched to
            the dataset by exact rules and a reviewed override file, never by fuzzy matching. A name with no match is
            treated as a debutant, and an ambiguous name leaves the bout unpredicted.
          </p>
          <p>
            Every load passes a validation gate. If a load fails, the last good data stays in place. Fights that
            cannot be placed with certainty are dropped and reported, never guessed: 3 of 9,003 in snapshot{' '}
            <code>96adace</code> (<code>uv run python -m ufc.ingest load --dry-run</code>).
          </p>
        </Block>

        <Block id="leakage" title="Leakage prevention">
          <p>
            A model that sees the future looks brilliant in testing and fails in use. These rules are
            non-negotiable (<a href={doc('ML_REQUIREMENTS.md')}>ML_REQUIREMENTS</a>):
          </p>
          <ul>
            <li>
              Every feature for a bout on date D uses only bouts from events strictly before D. Other bouts on the same
              card never count. Training and live predictions use the same code, so the two cannot drift apart.
            </li>
            <li>
              Career stats shown on fighter profile pages are never used. Every stat is recomputed from per-fight
              data as of the cut-off.
            </li>
            <li>
              Everything fitted (Elo parameters, league baselines, weight-class averages) is fitted on the training
              period only. Calibration is fitted on validation.
            </li>
            <li>
              Tests prove it. Changing or deleting every bout after a date must leave earlier features exactly
              equal. Other bouts on a card must not change a bout’s features. A known fighter’s stats must match a
              hand calculation.
            </li>
            <li>
              Corner is never a feature. Features are differences between the two fighters, the model is trained on
              both orderings, and P(A) + P(B) = 1 exactly, so swapping the fighters swaps the probabilities exactly.
            </li>
          </ul>
          <p>
            One leak was found and removed before the test period was opened. Fighter profiles are as they stand
            today, and an incomplete profile mostly belongs to someone whose UFC career turned out short. In training
            bouts where only one fighter’s profile lacked an attribute, the fighter with the complete profile won
            91.5% of 895. No model is fitted on such bouts now, and they are labelled “limited data” on the site
            (D44; <code>uv run python -m ufc.models report</code>).
          </p>
        </Block>

        <Block id="features" title="Features">
          <p>
            Each bout is described by A-minus-B differences of the two fighters’ values, computed from their earlier
            bouts only, grouped in families:
          </p>
          <ul>
            <li>Physical: age, height, reach and stance matchup. Age shape counts years past 32 and past the weight class’s median age.</li>
            <li>Record and experience, activity, and moves between weight classes.</li>
            <li>Striking and grappling rates over the career, the last 3 and 5 bouts, and time-decayed. Per-minute rates are shrunk toward the weight-class average, so a fighter with little cage time is not read as extreme.</li>
            <li>Durability, and pace in later rounds compared with the first.</li>
            <li>An Elo rating, the ratings of past opponents, and stats adjusted for what those opponents usually allowed.</li>
            <li>Style matchups: one fighter’s offence times the other’s weakness in the same area.</li>
          </ul>
          <p>
            A debutant’s stats are missing, not zero. Two further families were tested and not adopted because
            they failed rules written down before they were tested: strength of schedule, and a “deserved win”
            measure of past performances. The deserved-win probability still appears on bout pages to describe past
            fights, and no prediction reads it (D54 to D64).
          </p>
        </Block>

        <Block id="models" title="Models">
          <p>
            The headline model is a gradient-boosted tree model (XGBoost) on the difference features. It is trained
            on both orderings of every bout, with recent years weighted more, and calibrated with a one-parameter
            Platt scaling fitted on bouts the model did not train on. Two baselines are always reported next to it:
            an Elo rating and a logistic regression on the same features. The model uses no betting odds.
          </p>
          <p>
            Every setting was chosen on the 2023 to 2024 validation period by log loss. The live model refits that
            recipe on every completed bout. A person approves each model by hand, so a bad retrain cannot ship
            on its own, and rolling back is one edit (D4, D47).
          </p>
          <p>
            The “why” on each bout page comes from TreeSHAP contributions averaged over both orderings of the bout
            and grouped into readable reasons by fixed templates. It describes what the model weighs, never what
            causes a fight’s result (D59, D65).
          </p>
        </Block>

        <Block id="evaluation" title="Evaluation">
          <ul>
            <li>Chronological splits: training through 2022, validation 2023 to 2024, test from 2025. The test period sits behind a lock in code, and every time it is opened is logged.</li>
            <li>Log loss is the primary metric because it rewards honest probabilities. Brier score, accuracy and calibration are reported too, with draws and no contests counted as no result.</li>
            <li>A walk-forward backtest refits the recipe for each year from 2013 and scores the year after. Years before the test period are a backtest, not a held-out result, and are labelled that way.</li>
            <li>
              The live record scores only predictions published before an event’s first bout could have started, and is
              never pooled with backtest numbers.
            </li>
          </ul>
          <p>
            On the test period the model clearly beats Elo, but it is not shown to beat the logistic regression on the
            same features. The figures, their intervals and the caveats are on the{' '}
            <Link to="/performance">Performance</Link> page and in the{' '}
            <a href={doc('MODEL_REPORT.md')}>model report</a>.
          </p>
        </Block>

        <Block id="limitations" title="Limitations">
          <ul>
            <li>
              The test period has been looked at more than once, and it scored better than every earlier year for
              reasons the diagnostics did not establish. Expect live results to be weaker. The live record is the
              real measurement.
            </li>
            <li>
              Debutants and fighters with incomplete profiles are predicted but labelled “limited data”. The model
              knows little about them, and it has done poorly on bouts with an incomplete profile.
            </li>
            <li>
              Only UFC bouts in the dataset count. Records from other promotions, injuries, short-notice replacements,
              weight cuts and betting markets are not inputs.
            </li>
            <li>
              Height, reach, stance and date of birth are treated as fixed, and they come from profiles that were
              sometimes edited after the fact (D46).
            </li>
            <li>
              Cards can change late, and Wikipedia can be wrong or behind. When the dataset has not caught up with a
              recent event, affected bouts are labelled “stale stats”.
            </li>
            <li>The data depends on one volunteer-run feed. If it stops, the site freezes rather than breaks.</li>
            <li>These are statistical estimates, not betting advice. The fair moneyline is a conversion of the model’s probability, not a price.</li>
          </ul>
        </Block>

        <Block id="code" title="Source code">
          <p>
            Everything is in the{' '}
            <a href={REPO_URL} rel="noreferrer">
              GitHub repository
            </a>
            : the ingestion and feature pipeline, the models and their tests, the publishing gate, and this site. The
            reasoning behind each design choice is in <a href={doc('DECISIONS.md')}>DECISIONS.md</a>, and the full
            evaluation is in <a href={doc('MODEL_REPORT.md')}>MODEL_REPORT.md</a>.
          </p>
        </Block>
      </div>
    </div>
  )
}
