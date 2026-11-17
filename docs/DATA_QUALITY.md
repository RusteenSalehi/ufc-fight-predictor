# Data quality report (M2)

Source: `Greco1899/scrape_ufc_stats` at upstream commit `96adace863c0a793579cc06d27d468797545607f`
(committed 2026-11-15T18:03:06Z), loaded into `data/ufc.db` by `uv run python -m ufc.ingest load`.

The block between the GENERATED markers is the output of one command and is replaced on every run:

    uv run python -m ufc.ingest fetch
    uv run python -m ufc.ingest load --verbose
    uv run python scripts/data_quality.py --out docs/DATA_QUALITY.md

Only numbers from that block appear in the prose, which is hand-written and says so. The report was run on
2026-11-16; `--today` reproduces it on a later day, and the freshness numbers are the only ones that depend
on the run date. The active and debut cut-offs (2024-01-01 and 2025-11-22) are `--active-since` and
`--debut-since`.

## Findings

Written by hand from the generated numbers below.

**Freshness.** The most recent event is 2026-11-14, two days before the report date and one day before the
snapshot commit. No event is dated after the commit. Since 2024-11-16 the median gap between event dates is
7 days, so a two-day lag is normal; a stale snapshot would show as a growing number here.

**Attributes are the weak spot, and only for new fighters.** Among the 917 fighters with a bout since
2024-01-01, 9.8% have no height, 9.9% no reach and 8.5% no stance, but only 0.9% (8) have no date of birth.
Splitting by debut, the 769 who debuted before 2025-11-22 are almost complete (4 missing height and reach, 0
missing date of birth), while 58.1% of the 148 who debuted on or after that date have no height, 58.8% no
reach and 50.7% no stance; 5.4% have no date of birth. The by-year and by-quarter tables show this is a
drift, not a cliff at that date: every debut year from 2015 to 2024 is 0.0% missing height, 2025 is 14.7%,
2026 is 56.7%, and the per-date table shows a card where all 6 debutants are missing height (2026-07-25)
next to one where none of 5 is (2026-06-20). The likely cause, a hypothesis not tested here, is that
upstream captures a profile once (DECISIONS D14), so fighters who debuted recently were captured before
their profile pages were filled in.

Consequences for later milestones: height and reach differences will be missing for a large share of
bouts involving recent debutants, while age (from date of birth) is nearly always available. The training
period (through 2022-12-31) comes from debut years with no missing height, so a model will rarely have seen
a missing height or reach difference in training, yet will meet many in the 2025 to 2026 test period and in
live predictions. Imputation and missing-value indicators in M4 are fitted on training data only
(ML_REQUIREMENTS), so that mismatch should be reported: metrics with and without bouts that involve a
fighter missing height or reach.

**Statistics.** 23 of 9,000 loaded bouts have no stats, all in 1994 to 1998. Control time is missing for
every round row in 1994 to 1998 and for 41.8% of 1999, and for none from 2000 on. Two of the 23 are the
ambiguous same-pairing case below.

**Duplicates.** No repeated bout IDs, no repeated event names, no profile pairs with the same name and date
of birth. Nine names are shared by two profiles; one of those names has two profiles with loaded bouts. 67
fighters have two bouts on one date, none since 2000 (the early tournament nights). One pair met twice at
one event (UFC - Ultimate Japan).

**Event gaps.** Event counts per year are 39 to 46 from 2014 on (2026 is a partial year, 40 so far). Since
2010-01-01, eight gaps exceed 28 days and none exceeds 56; the longest is 2020-03-14 to 2020-05-09. The
script cannot tell a missing event from a scheduling gap, so this is a check for gross omissions only.

## Dropped bouts and name resolutions (review)

Three bouts are dropped: the three errors in the report below (the load gate allows 1% of fights to drop).
An entry in `src/ufc/ingest/identity_overrides.csv` maps a fighter name to a fighter ID, so it can only fix
the first kind of drop. **No override was added**: the one that could apply is not settled by the dataset.

1. **`64829a32c78ab656`, Louie Sutherland vs. Jose Montanha, 2026-08-08, heavyweight (unresolved fighter).**
   Two profiles answer to "Jose Montanha": `1f8cd7634e8286ff` and `4e30f2761b6bd3b5`, and the dataset's two
   fighter files contradict each other about which is which (`grep -n "Montanha\|Da Silva Lopes"
   data/external/greco/*.csv` shows the four rows). The details file lists `1f8cd7634e8286ff` as "Jose
   Montanha" and `4e30f2761b6bd3b5` as "Henrique Da Silva Lopes" with the nickname "Montanha". The
   tale-of-the-tape file records the opposite names under the same two IDs: "Henrique Da Silva Lopes" (240 lbs,
   born 1985) at `1f8cd7634e8286ff` and "Jose Montanha" (265 lbs, born 1996) at `4e30f2761b6bd3b5`. Both
   weights fit heavyweight, so the weight class cannot decide. Neither profile has another loaded bout (the
   "loaded bouts" column in the dropped-bouts section), so there is no history or opponent to cross-check.
   Of the three name signals, two (the tale-of-the-tape name and the nickname) point to `4e30f2761b6bd3b5`
   and one (the details-file name) points to `1f8cd7634e8286ff`. That is a lean, not evidence, because the
   files disagree with each other and D22 already blanks both profiles' attributes for exactly that reason.
   Left dropped; the cost is one bout missing from Sutherland's history and one from the other fighter's. If
   you want it loaded anyway, the better-supported line is
   `Jose Montanha,64829a32c78ab656,4e30f2761b6bd3b5,<note>`; I did not add it.
2. **`abf5f4d6f777cf7b` (Shi Ming vs. Bruna Brasil) and `16523b1bec3cd7b0` (Nyamjargal Tumendemberel vs.
   Terrance Saeteurn), event "UFC - Road to UFC 4.6" (orphan event).** The events file has no event whose
   name starts with "UFC - Road to UFC" (0 in the dropped-bouts section), so these fights have no date or event
   ID anywhere in the dataset. This is not a name problem and an override cannot fix it: loading them would
   need a date and an ID from outside the dataset, which this project does not guess. Left dropped. (The
   results file lists them as a women's strawweight bout and a flyweight bout.)

Name resolutions: 56 bouts, 50 by name plus weight class and 6 by override, across nine names (the table
under "Fighter name resolutions by rule"). The six overrides are the three "Rafael Cerquiera" bouts, the
"Nick Galanti" bout and the two shared-name bouts the weight class could not settle; each note in the
override file cites its evidence, and every override is used (the report has no `identity_override`
warning). Two checks that do not depend on the resolution rules say the weight-class resolutions did not
send a bout to the wrong profile: the report has no `date_sanity` or `mixed_division` row (no fighter is
under 17 or over 55 at any bout and none fights in both a women's and a men's class), and no fighter has two
bouts on one date since 2000. Neither check can catch a swap between two same-name men of similar age and
weight, so the weight-class rule remains the one to distrust first if a result looks wrong.

<!-- BEGIN GENERATED: scripts/data_quality.py -->

## Snapshot

- Upstream: `Greco1899/scrape_ufc_stats` at commit `96adace863c0a793579cc06d27d468797545607f` (committed 2026-11-15T18:03:06Z)
- Report run on: 2026-11-16

| table | rows |
|---|---:|
| events | 797 |
| fighters | 4,629 |
| bouts | 9,000 |
| bout_round_stats | 42,256 |
| bout_fighter_stats | 17,954 |
| ingest_issues | 75 |

## Freshness

- Most recent event in the database: 2026-11-14
- Today: 2026-11-16
- Days from most recent event to today: 2
- Snapshot commit date: 2026-11-15T18:03:06Z
- Events dated after the snapshot commit date: 0
- Median gap between consecutive event dates, events since 2024-11-16: 7 days (n=85; 792 distinct event dates in all)

## Fighter attributes

Missing means NULL in `fighters`. That includes the attributes left empty on purpose when a profile's two names disagree (DECISIONS D22). A fighter is a debutant at their first loaded bout; a bout dropped during ingestion is not counted.

| group | fighters | no height | share | no reach | share | no stance | share | no date of birth | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all profiles | 4,629 | 519 | 11.2% | 2,140 | 46.2% | 1,024 | 22.1% | 783 | 16.9% |
| profiles with no loaded bout | 1,850 | 408 | 22.1% | 1,394 | 75.4% | 869 | 47.0% | 619 | 33.5% |
| profiles with at least one loaded bout | 2,779 | 111 | 4.0% | 746 | 26.8% | 155 | 5.6% | 164 | 5.9% |
| active: a bout since 2024-11-16 (730 days before the report date) | 812 | 90 | 11.1% | 91 | 11.2% | 78 | 9.6% | 8 | 1.0% |
| active: a bout since 2024-01-01 | 917 | 90 | 9.8% | 91 | 9.9% | 78 | 8.5% | 8 | 0.9% |
| active since 2024-01-01, debut before 2025-11-22 | 769 | 4 | 0.5% | 4 | 0.5% | 3 | 0.4% | 0 | 0.0% |
| active since 2024-01-01, debut on or after 2025-11-22 | 148 | 86 | 58.1% | 87 | 58.8% | 75 | 50.7% | 8 | 5.4% |

### Share missing by debut year (first loaded bout), 2015 on

| debut year | fighters | no height | no reach | no stance | no date of birth |
|---|---:|---:|---:|---:|---:|
| 2015 | 107 | 0.0% | 26.2% | 8.4% | 0.0% |
| 2016 | 118 | 0.0% | 18.6% | 0.8% | 0.0% |
| 2017 | 121 | 0.0% | 6.6% | 0.8% | 0.0% |
| 2018 | 121 | 0.0% | 2.5% | 0.0% | 0.0% |
| 2019 | 138 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2020 | 123 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2021 | 91 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2022 | 99 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2023 | 127 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2024 | 102 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2025 | 95 | 14.7% | 14.7% | 13.7% | 2.1% |
| 2026 | 134 | 56.7% | 57.5% | 48.5% | 4.5% |

### Share missing by debut quarter, 2024 on

| debut quarter | fighters | no height | no reach | no date of birth | first debut | last debut |
|---|---:|---:|---:|---:|---:|---:|
| 2024-Q1 | 36 | 0.0% | 0.0% | 0.0% | 2024-01-13 | 2024-03-30 |
| 2024-Q2 | 25 | 0.0% | 0.0% | 0.0% | 2024-04-06 | 2024-06-29 |
| 2024-Q3 | 16 | 0.0% | 0.0% | 0.0% | 2024-07-13 | 2024-09-28 |
| 2024-Q4 | 25 | 0.0% | 0.0% | 0.0% | 2024-10-12 | 2024-12-14 |
| 2025-Q1 | 30 | 0.0% | 0.0% | 0.0% | 2025-01-11 | 2025-03-29 |
| 2025-Q2 | 18 | 0.0% | 0.0% | 0.0% | 2025-04-05 | 2025-06-28 |
| 2025-Q3 | 22 | 0.0% | 0.0% | 0.0% | 2025-07-19 | 2025-09-27 |
| 2025-Q4 | 25 | 56.0% | 56.0% | 8.0% | 2025-10-04 | 2025-12-13 |
| 2026-Q1 | 29 | 20.7% | 20.7% | 6.9% | 2026-01-24 | 2026-03-28 |
| 2026-Q2 | 35 | 51.4% | 51.4% | 2.9% | 2026-04-04 | 2026-06-27 |
| 2026-Q3 | 52 | 76.9% | 78.8% | 3.8% | 2026-07-11 | 2026-09-26 |
| 2026-Q4 | 18 | 66.7% | 66.7% | 5.6% | 2026-10-03 | 2026-11-14 |

### Debuts per event date, from 120 days before 2025-11-22

Counts of fighters missing each attribute, by the date of their first loaded bout.

| debut date | debutants | no height | no reach | no date of birth |
|---|---:|---:|---:|---:|
| 2025-07-26 | 1 | 0 | 0 | 0 |
| 2025-08-02 | 1 | 0 | 0 | 0 |
| 2025-08-16 | 4 | 0 | 0 | 0 |
| 2025-08-23 | 1 | 0 | 0 | 0 |
| 2025-09-06 | 4 | 0 | 0 | 0 |
| 2025-09-13 | 5 | 0 | 0 | 0 |
| 2025-09-27 | 4 | 0 | 0 | 0 |
| 2025-10-04 | 2 | 0 | 0 | 0 |
| 2025-10-11 | 1 | 1 | 1 | 0 |
| 2025-10-18 | 2 | 1 | 1 | 0 |
| 2025-10-25 | 1 | 0 | 0 | 0 |
| 2025-11-01 | 2 | 0 | 0 | 0 |
| 2025-11-08 | 2 | 1 | 1 | 0 |
| 2025-11-15 | 1 | 1 | 1 | 0 |
| 2025-11-22 | 6 | 5 | 5 | 1 |
| 2025-12-06 | 1 | 0 | 0 | 0 |
| 2025-12-13 | 7 | 5 | 5 | 1 |
| 2026-01-24 | 1 | 0 | 0 | 0 |
| 2026-01-31 | 6 | 0 | 0 | 0 |
| 2026-02-07 | 1 | 1 | 1 | 0 |
| 2026-02-21 | 2 | 0 | 0 | 0 |
| 2026-02-28 | 6 | 2 | 2 | 0 |
| 2026-03-07 | 3 | 1 | 1 | 0 |
| 2026-03-14 | 3 | 1 | 1 | 1 |
| 2026-03-21 | 5 | 0 | 0 | 0 |
| 2026-03-28 | 2 | 1 | 1 | 1 |
| 2026-04-04 | 3 | 1 | 1 | 0 |
| 2026-04-18 | 6 | 4 | 4 | 0 |
| 2026-04-25 | 3 | 3 | 3 | 0 |
| 2026-05-02 | 2 | 2 | 2 | 0 |
| 2026-05-16 | 4 | 3 | 3 | 1 |
| 2026-05-30 | 7 | 4 | 4 | 0 |
| 2026-06-06 | 1 | 0 | 0 | 0 |
| 2026-06-20 | 5 | 0 | 0 | 0 |
| 2026-06-27 | 4 | 1 | 1 | 0 |
| 2026-07-11 | 2 | 1 | 1 | 0 |
| 2026-07-18 | 5 | 1 | 1 | 0 |
| 2026-07-25 | 6 | 6 | 6 | 0 |
| 2026-08-01 | 9 | 8 | 9 | 0 |
| 2026-08-08 | 3 | 2 | 2 | 0 |
| 2026-08-15 | 2 | 0 | 0 | 0 |
| 2026-08-22 | 4 | 4 | 4 | 1 |
| 2026-08-29 | 7 | 6 | 6 | 0 |
| 2026-09-05 | 7 | 5 | 5 | 0 |
| 2026-09-12 | 2 | 2 | 2 | 0 |
| 2026-09-26 | 5 | 5 | 5 | 1 |
| 2026-10-03 | 4 | 3 | 3 | 1 |
| 2026-10-17 | 7 | 4 | 4 | 0 |
| 2026-10-24 | 1 | 1 | 1 | 0 |
| 2026-10-31 | 2 | 2 | 2 | 0 |
| 2026-11-07 | 3 | 2 | 2 | 0 |
| 2026-11-14 | 1 | 0 | 0 | 0 |

## Events and bouts per year

| year | events | bouts | wins | draws | no contests |
|---|---:|---:|---:|---:|---:|
| 1994 | 3 | 31 | 31 | 0 | 0 |
| 1995 | 4 | 40 | 38 | 2 | 0 |
| 1996 | 5 | 43 | 43 | 0 | 0 |
| 1997 | 5 | 41 | 40 | 0 | 1 |
| 1998 | 3 | 25 | 25 | 0 | 0 |
| 1999 | 6 | 44 | 42 | 2 | 0 |
| 2000 | 6 | 43 | 42 | 1 | 0 |
| 2001 | 5 | 40 | 39 | 0 | 1 |
| 2002 | 7 | 53 | 52 | 0 | 1 |
| 2003 | 5 | 41 | 39 | 2 | 0 |
| 2004 | 5 | 39 | 39 | 0 | 0 |
| 2005 | 10 | 80 | 79 | 0 | 1 |
| 2006 | 18 | 158 | 158 | 0 | 0 |
| 2007 | 19 | 171 | 169 | 1 | 1 |
| 2008 | 20 | 201 | 201 | 0 | 0 |
| 2009 | 20 | 215 | 213 | 1 | 1 |
| 2010 | 24 | 253 | 250 | 3 | 0 |
| 2011 | 27 | 300 | 295 | 2 | 3 |
| 2012 | 31 | 341 | 333 | 2 | 6 |
| 2013 | 33 | 386 | 376 | 1 | 9 |
| 2014 | 46 | 503 | 494 | 1 | 8 |
| 2015 | 41 | 473 | 464 | 2 | 7 |
| 2016 | 41 | 493 | 483 | 7 | 3 |
| 2017 | 39 | 457 | 446 | 5 | 6 |
| 2018 | 39 | 474 | 469 | 2 | 3 |
| 2019 | 42 | 516 | 506 | 5 | 5 |
| 2020 | 41 | 456 | 444 | 5 | 7 |
| 2021 | 43 | 509 | 497 | 5 | 7 |
| 2022 | 42 | 511 | 506 | 3 | 2 |
| 2023 | 43 | 520 | 504 | 6 | 10 |
| 2024 | 42 | 517 | 513 | 0 | 4 |
| 2025 | 42 | 520 | 513 | 4 | 3 |
| 2026 | 40 | 506 | 500 | 5 | 1 |

## Results and methods

| result | bouts | share |
|---|---:|---:|
| win | 8,843 | 98.3% |
| nc | 90 | 1.0% |
| draw | 67 | 0.7% |

| method | bouts | share |
|---|---:|---:|
| Decision - Unanimous | 3,226 | 35.8% |
| KO/TKO | 2,876 | 32.0% |
| Submission | 1,738 | 19.3% |
| Decision - Split | 837 | 9.3% |
| Decision - Majority | 106 | 1.2% |
| TKO - Doctor's Stoppage | 100 | 1.1% |
| Overturned | 58 | 0.6% |
| Could Not Continue | 33 | 0.4% |
| DQ | 24 | 0.3% |
| Other | 2 | 0.0% |

## Statistics coverage by year

A round row is one fighter in one round; control time is missing when the source has no value for it.

| year | bouts | bouts without stats | share | round rows | round rows without control time | share |
|---|---:|---:|---:|---:|---:|---:|
| 1994 | 31 | 2 | 6.5% | 58 | 58 | 100.0% |
| 1995 | 40 | 6 | 15.0% | 74 | 74 | 100.0% |
| 1996 | 43 | 7 | 16.3% | 78 | 78 | 100.0% |
| 1997 | 41 | 4 | 9.8% | 92 | 92 | 100.0% |
| 1998 | 25 | 4 | 16.0% | 60 | 60 | 100.0% |
| 1999 | 44 | 0 | 0.0% | 158 | 66 | 41.8% |
| 2000 | 43 | 0 | 0.0% | 182 | 0 | 0.0% |
| 2001 | 40 | 0 | 0.0% | 184 | 0 | 0.0% |
| 2002 | 53 | 0 | 0.0% | 232 | 0 | 0.0% |
| 2003 | 41 | 0 | 0.0% | 176 | 0 | 0.0% |
| 2004 | 39 | 0 | 0.0% | 146 | 0 | 0.0% |
| 2005 | 80 | 0 | 0.0% | 308 | 0 | 0.0% |
| 2006 | 158 | 0 | 0.0% | 640 | 0 | 0.0% |
| 2007 | 171 | 0 | 0.0% | 724 | 0 | 0.0% |
| 2008 | 201 | 0 | 0.0% | 834 | 0 | 0.0% |
| 2009 | 215 | 0 | 0.0% | 960 | 0 | 0.0% |
| 2010 | 253 | 0 | 0.0% | 1,220 | 0 | 0.0% |
| 2011 | 300 | 0 | 0.0% | 1,396 | 0 | 0.0% |
| 2012 | 341 | 0 | 0.0% | 1,576 | 0 | 0.0% |
| 2013 | 386 | 0 | 0.0% | 1,800 | 0 | 0.0% |
| 2014 | 503 | 0 | 0.0% | 2,402 | 0 | 0.0% |
| 2015 | 473 | 0 | 0.0% | 2,256 | 0 | 0.0% |
| 2016 | 493 | 0 | 0.0% | 2,418 | 0 | 0.0% |
| 2017 | 457 | 0 | 0.0% | 2,220 | 0 | 0.0% |
| 2018 | 474 | 0 | 0.0% | 2,300 | 0 | 0.0% |
| 2019 | 516 | 0 | 0.0% | 2,546 | 0 | 0.0% |
| 2020 | 456 | 0 | 0.0% | 2,228 | 0 | 0.0% |
| 2021 | 509 | 0 | 0.0% | 2,540 | 0 | 0.0% |
| 2022 | 511 | 0 | 0.0% | 2,476 | 0 | 0.0% |
| 2023 | 520 | 0 | 0.0% | 2,496 | 0 | 0.0% |
| 2024 | 517 | 0 | 0.0% | 2,652 | 0 | 0.0% |
| 2025 | 520 | 0 | 0.0% | 2,486 | 0 | 0.0% |
| 2026 | 506 | 0 | 0.0% | 2,338 | 0 | 0.0% |
| all | 9,000 | 23 | 0.3% | 42,256 | 428 | 1.0% |

## Duplicate checks

| check | count |
|---|---:|
| bout IDs appearing more than once | 0 |
| event names shared by more than one event | 0 |
| fighter names shared by more than one profile (names / profiles) | 9 / 18 |
|   of those, profiles with a loaded bout (names / profiles) | 1 / 2 |
| profile pairs with the same name and the same date of birth | 0 |
| fighters with two bouts at one event | 67 |
| fighters with two bouts on one date | 67 |
|   of those, on dates since 2000 | 0 |
| fighter pairs meeting twice at one event | 1 |
| dates with more than one event | 5 |

Dates with more than one event:

- 2014-05-31: UFC Fight Night: Miocic vs Maldonado | UFC Fight Night: Munoz vs Mousasi
- 2014-06-28: UFC Fight Night: Swanson vs Stephens | UFC Fight Night: Te Huna vs Marquardt
- 2014-08-23: UFC Fight Night: Henderson vs Dos Anjos | UFC Fight Night: Bisping vs Le
- 2014-10-04: UFC Fight Night: MacDonald vs Saffiedine | UFC Fight Night: Nelson vs Story
- 2016-11-19: UFC Fight Night: Bader vs. Nogueira | UFC Fight Night: Mousasi vs. Hall 2

## ingest_issues, grouped by check

| check | severity | rows |
|---|---:|---:|
| orphan_event | error | 2 |
| unresolved_fighter | error | 1 |
| ambiguous_stats | warning | 2 |
| profile_name_conflict | warning | 4 |
| date_range | info | 1 |
| duplicate_fight_rows | info | 1 |
| fighter_resolution | info | 56 |
| fighters_without_bouts | info | 1 |
| no_stats | info | 1 |
| no_weight_class | info | 1 |
| results_breakdown | info | 2 |
| row_counts | info | 1 |
| unused_stats_rows | info | 1 |
| womens_bout_inferred | info | 1 |

### Errors and warnings in full

- **error** `orphan_event` [bout abf5f4d6f777cf7b]: 'Shi Ming vs. Bruna Brasil' names event ['UFC - Road to UFC 4.6'] which is not in the events file; bout dropped
- **error** `orphan_event` [bout 16523b1bec3cd7b0]: 'Nyamjargal Tumendemberel vs. Terrance Saeteurn' names event ['UFC - Road to UFC 4.6'] which is not in the events file; bout dropped
- **error** `unresolved_fighter` [bout 64829a32c78ab656]: 'Jose Montanha' in 'Louie Sutherland vs. Jose Montanha' (UFC Fight Night: Gamrot vs. Salkilld, 2026-08-08) shared by fighters ['1f8cd7634e8286ff', '4e30f2761b6bd3b5'] and the weight class does not single one out; bout dropped
- **warning** `ambiguous_stats` [bout ec1bda9a4c2aab42]: 'Kazushi Sakuraba vs. Marcus Silveira' was fought more than once at UFC - Ultimate Japan; stats rows cannot be told apart, so none were loaded
- **warning** `ambiguous_stats` [bout 2750ac5854e8b28b]: 'Kazushi Sakuraba vs. Marcus Silveira' was fought more than once at UFC - Ultimate Japan; stats rows cannot be told apart, so none were loaded
- **warning** `profile_name_conflict` [fighter 1f8cd7634e8286ff]: listed as 'Jose Montanha' but attributes were recorded under 'Henrique Da Silva Lopes'; height, reach, stance and date of birth left empty
- **warning** `profile_name_conflict` [fighter 4e30f2761b6bd3b5]: listed as 'Henrique Da Silva Lopes' but attributes were recorded under 'Jose Montanha'; height, reach, stance and date of birth left empty
- **warning** `profile_name_conflict` [fighter a1a91dbf41a029c4]: listed as 'Tina Black' but attributes were recorded under 'Valesca Machado'; height, reach, stance and date of birth left empty
- **warning** `profile_name_conflict` [fighter bc7f4b2d4c2724a2]: listed as 'Tyrell Fortune' but attributes were recorded under 'Regina Malpica'; height, reach, stance and date of birth left empty

### Fighter name resolutions by rule

| rule | bouts |
|---|---:|
| name + weight class | 50 |
| override | 6 |

Names resolved through a rule other than a unique name, with bouts affected:

| name | bouts |
|---|---:|
| Bruno Silva | 23 |
| Michael McDonald | 9 |
| Jean Silva | 8 |
| Mike Davis | 8 |
| Rafael Cerquiera | 3 |
| Joey Gomez | 2 |
| Anthony Romero | 1 |
| Nick Galanti | 1 |
| Victor Valenzuela | 1 |

## Dropped bouts: what the database says about them

Bout `64829a32c78ab656`: 'Jose Montanha' in 'Louie Sutherland vs. Jose Montanha' (UFC Fight Night: Gamrot vs. Salkilld, 2026-08-08) shared by fighters ['1f8cd7634e8286ff', '4e30f2761b6bd3b5'] and the weight class does not single one out; bout dropped

| fighter_id | listed name | nickname | height_cm | reach_cm | stance | dob | loaded bouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1f8cd7634e8286ff` | Jose Montanha | - | - | - | - | - | 0 |
| `4e30f2761b6bd3b5` | Henrique Da Silva Lopes | Montanha | - | - | - | - | 0 |

Event `UFC - Road to UFC 4.6` is named by 2 dropped bouts (`abf5f4d6f777cf7b`, `16523b1bec3cd7b0`); events in the events file whose name starts with `UFC - Road to UFC`: 0.

## Gaps between consecutive event dates

### Largest 10 gaps overall

| days | previous event | next event | next event name |
|---|---:|---:|---:|
| 182 | 1994-03-11 | 1994-09-09 | UFC 3: The American Dream |
| 154 | 1998-05-15 | 1998-10-16 | UFC - Ultimate Brazil |
| 112 | 1994-12-16 | 1995-04-07 | UFC 5: The Return of the Beast |
| 112 | 1997-02-07 | 1997-05-30 | UFC 13: The Ultimate Force |
| 112 | 1999-11-19 | 2000-03-10 | UFC 24: First Defense |
| 112 | 2003-06-06 | 2003-09-26 | UFC 44: Undisputed |
| 106 | 2004-10-22 | 2005-02-05 | UFC 51: Super Saturday |
| 105 | 2000-06-09 | 2000-09-22 | UFC 27: Ultimate Bad Boyz |
| 99 | 1995-09-08 | 1995-12-16 | UFC - Ultimate Ultimate '95 |
| 98 | 1994-09-09 | 1994-12-16 | UFC 4: Revenge of the Warriors |

### Gaps over 28 days since 2010-01-01

| days | previous event | next event | next event name |
|---|---:|---:|---:|
| 29 | 2010-02-20 | 2010-03-21 | UFC Live: Vera vs Jones |
| 29 | 2010-07-03 | 2010-08-01 | UFC Live: Jones vs Matyushenko |
| 35 | 2011-03-26 | 2011-04-30 | UFC 129: St-Pierre vs Shields |
| 35 | 2011-07-02 | 2011-08-06 | UFC 133: Evans vs Ortiz 2 |
| 43 | 2012-03-02 | 2012-04-14 | UFC on FUEL TV: Gustafsson vs Silva |
| 42 | 2012-08-11 | 2012-09-22 | UFC 152: Jones vs Belfort |
| 56 | 2020-03-14 | 2020-05-09 | UFC 249: Ferguson vs. Gaethje |
| 42 | 2025-12-13 | 2026-01-24 | UFC 324: Gaethje vs. Pimblett |

### Gap per year, 2000 on (gap ending in that year)

| year | gaps | median days | longest days |
|---|---:|---:|---:|
| 2000 | 6 | 56 | 112 |
| 2001 | 5 | 69 | 91 |
| 2002 | 7 | 56 | 76 |
| 2003 | 5 | 56 | 112 |
| 2004 | 5 | 63 | 78 |
| 2005 | 10 | 36.5 | 106 |
| 2006 | 18 | 22 | 58 |
| 2007 | 19 | 17 | 49 |
| 2008 | 20 | 15.5 | 35 |
| 2009 | 20 | 17.5 | 35 |
| 2010 | 24 | 14 | 29 |
| 2011 | 27 | 13 | 35 |
| 2012 | 31 | 8 | 43 |
| 2013 | 33 | 8 | 21 |
| 2014 | 42 | 7 | 21 |
| 2015 | 41 | 7 | 21 |
| 2016 | 40 | 7 | 28 |
| 2017 | 39 | 7 | 28 |
| 2018 | 39 | 7 | 21 |
| 2019 | 42 | 7 | 21 |
| 2020 | 41 | 7 | 56 |
| 2021 | 43 | 7 | 28 |
| 2022 | 42 | 7 | 28 |
| 2023 | 43 | 7 | 28 |
| 2024 | 42 | 7 | 28 |
| 2025 | 42 | 7 | 28 |
| 2026 | 40 | 7 | 42 |

<!-- END GENERATED -->
