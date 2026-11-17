# Synthetic dataset fixtures

These six CSVs follow the column layout of the files published at
https://github.com/Greco1899/scrape_ufc_stats, but **no row in them comes from
that dataset or from ufcstats.com**. Fighters, events, referees, judges' scores
and statistics are invented (statistics from a seeded random generator, except a
few rounds written by hand so tests can state exact values). The IDs are
readable on purpose: `e0..` events, `f1..` fighters, `b0..` fights.

`SOURCE.json` carries a made-up commit hash. `identity_overrides.csv` is this
fixture's own override file; tests pass it explicitly, because the file shipped
in `src/ufc/ingest/` is keyed to the real snapshot.

Regenerate (the output is deterministic):

    uv run python scripts/make_synthetic_fixtures.py

The fights mirror the edge cases ingestion has to handle: draws, no contests,
a disqualification, a doctor's stoppage, tournament finals, a superfight with
overtime, catch-weight and women's bouts (one women's bout not labelled as
such), early events without stats or control time, fights listed twice under a
renamed event, a fight whose event is missing, the same pairing twice on one
card, fighters sharing a name (settled by weight class, by override, or not at
all), misspelled and shortened names, profiles missing height, reach, stance or
date of birth, a profile whose stored attributes belong to a different name,
and names that differ between the two fighter files. Each fight's key in
`scripts/make_synthetic_fixtures.py` names its case.

The one test that needs real rows, the hand-calculated spot check
(`tests/test_features_spot_check.py`), reads them from the downloaded snapshot
in `data/external/greco/` and is skipped when that is absent.
