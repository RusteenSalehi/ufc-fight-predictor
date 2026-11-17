# Spot-check fixture

Khabib Nurmagomedov's first four UFC bouts, cut from the CSV files published
at https://github.com/Greco1899/scrape_ufc_stats at commit
`96adace863c0a793579cc06d27d468797545607f` (2026-11-15). Rows are copied
unchanged; only the selection is ours.

`tests/test_features_spot_check.py` computes his stats before the fourth bout
by hand from the first three and compares them with the feature pipeline's
output. The four fight IDs are listed in `scripts/make_ingest_fixtures.py`.

Regenerate with:

    uv run python -m ufc.ingest fetch --ref 96adace863c0a793579cc06d27d468797545607f
    uv run python scripts/make_ingest_fixtures.py

## Licence and origin

The same caveat as `tests/fixtures/greco/README.md` applies: the upstream
repository is licensed GPL-3.0, its data was collected from ufcstats.com, and
the operator has not granted permission for reuse. These rows are here only
so the test can run against real data.
