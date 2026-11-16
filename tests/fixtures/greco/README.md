# Dataset fixtures

A 29-fight subset of the CSV files published at
https://github.com/Greco1899/scrape_ufc_stats, cut from commit
`96adace863c0a793579cc06d27d468797545607f` (2026-11-15). Rows are copied
unchanged; only the selection is ours.

Regenerate with:

    uv run python -m ufc.ingest fetch --ref 96adace863c0a793579cc06d27d468797545607f
    uv run python scripts/make_ingest_fixtures.py

The fights were chosen to cover the edge cases ingestion has to handle. The
list, with the reason for each, is in `scripts/make_ingest_fixtures.py`.

## Licence and origin

The upstream repository is licensed GPL-3.0. Its data was collected from
ufcstats.com, whose operator has not granted permission for reuse. These
excerpts are kept here only so the tests can run against real rows. The rest
of this repository does not contain the dataset; it is downloaded at run time
into the git-ignored `data/` directory.
