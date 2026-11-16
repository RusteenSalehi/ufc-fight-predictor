"""Cut a small, real subset of the published dataset for use as test fixtures.

Usage:
    uv run python scripts/make_ingest_fixtures.py [SOURCE_DIR] [DEST_DIR]

Defaults: data/external/greco -> tests/fixtures/greco. Rows are copied
unchanged. The fights are chosen to cover the edge cases the ingestion has to
handle; the reason for each is given next to its ID below.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from ufc.ingest import parsing as p
from ufc.ingest import source

FIGHT_IDS = {
    "30dbf1eceb2e1f8d": "men's title bout, unanimous decision, five rounds",
    "32112fd0aaf7fba2": "women's title bout, submission",
    "79c49d244eab9005": "majority draw",
    "2a4b33c38e7b16b0": "catch weight, KO/TKO",
    "b395c89e19a3fec4": "split draw, women's title; listed twice under a renamed event",
    "5d3e046b6bcd49f5": "no contest (overturned); listed twice",
    "697efaf0d162e877": "split decision won by the second-named fighter; listed twice",
    "d822ebc7dbc6c259": "no contest (could not continue); fighter listed under a newer name",
    "de1a3734be60e6a1": "shared name settled by weight class; listed twice",
    "3dd2fde7bff79c51": "tournament final, sudden-victory format; listed twice",
    "54de03e04f571735": "shared name the weight class cannot settle; doctor's stoppage",
    "3a70ed81e5611d24": "disqualification; both fighters listed under newer names",
    "5727d5be8c373346": "interim title bout",
    "6ecfa439ef900bcc": "shared name settled by weight class (middleweight)",
    "20d74ed23d3e9b3a": "shared name settled by weight class (flyweight)",
    "57ff0eb2351979c4": "shared name the weight class cannot settle (bantamweight)",
    "64829a32c78ab656": "shared name with two heavyweight profiles: stays unresolved",
    "5ffb251c6ea48b65": "fighter whose stored attributes belong to someone else",
    "781a2ea50f5ae68f": "fighter name misspelled in the bout row",
    "b6ab5f1be3d3eb9e": "fighter name shortened in the bout row",
    "abf5f4d6f777cf7b": "event missing from the events file",
    "ec1bda9a4c2aab42": "same pairing twice on one card (win)",
    "2750ac5854e8b28b": "same pairing twice on one card (no contest)",
    "a1db4c917777aa79": "early bout with no stats, no time limit",
    "4acab67848e78327": "early bout with stats but no control time",
    "00835554f95fa911": "early tournament final with no weight class",
    "3932f8e9a74f3d11": "superfight draw that ended in overtime",
    "64da2e60f1b472fa": "women's flyweight bout",
    "44d58943ac8e47e1": "women's catch-weight bout, not labelled as women's",
}

EXTRA_FIGHTER_IDS = {
    "09a4447655fd4b7b": "target of the misspelled-name override",
    "8988705468d04ef7": "target of the shortened-name override",
    "5f440fdb94f626c5": "present in the tale-of-the-tape file only",
    "2b33e57c4d0a4502": "name has trailing whitespace; no reach or date of birth",
    "93fe7332d16c6ad9": "no height or reach; appears in no bout",
}


def fighter_id(row: dict[str, str]) -> str:
    return p.id_from_url(row["URL"], "fighter")


def main(source_dir: Path, dest_dir: Path) -> None:
    raw = source.load_raw_tables(source_dir)
    info = source.read_source_info(source_dir)

    results = [
        r for r in raw.fight_results if p.id_from_url(r["URL"], "fight") in FIGHT_IDS
    ]
    found = {p.id_from_url(r["URL"], "fight") for r in results}
    if found != set(FIGHT_IDS):
        raise SystemExit(
            f"fights not found in source: {sorted(set(FIGHT_IDS) - found)}"
        )

    details = [
        r for r in raw.fight_details if p.id_from_url(r["URL"], "fight") in FIGHT_IDS
    ]
    stat_keys = {(p.clean(r["EVENT"]), p.clean(r["BOUT"])) for r in results}
    stats = [
        r
        for r in raw.fight_stats
        if (p.clean(r["EVENT"]), p.clean(r["BOUT"])) in stat_keys
    ]
    event_names = {p.clean(r["EVENT"]) for r in results}
    events = [r for r in raw.events if p.clean(r["EVENT"]) in event_names]

    name_keys = {p.name_key(name) for r in results for name in p.split_bout(r["BOUT"])}
    wanted = set(EXTRA_FIGHTER_IDS)
    for row in raw.fighter_details:
        if p.name_key(f"{row['FIRST']} {row['LAST']}") in name_keys:
            wanted.add(fighter_id(row))
    for row in raw.fighter_tott:
        if p.name_key(row["FIGHTER"]) in name_keys:
            wanted.add(fighter_id(row))

    tables = {
        source.EVENTS_FILE: events,
        source.FIGHT_DETAILS_FILE: details,
        source.FIGHT_RESULTS_FILE: results,
        source.FIGHT_STATS_FILE: stats,
        source.FIGHTER_DETAILS_FILE: [
            r for r in raw.fighter_details if fighter_id(r) in wanted
        ],
        source.FIGHTER_TOTT_FILE: [
            r for r in raw.fighter_tott if fighter_id(r) in wanted
        ],
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_name, rows in tables.items():
        with (dest_dir / file_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=source.EXPECTED_COLUMNS[file_name],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"{file_name}: {len(rows)} rows")

    if info is None:
        raise SystemExit(
            f"{source_dir} has no {source.SOURCE_INFO_FILE}; run the fetch command"
        )
    description = {
        "repo": info.repo,
        "commit": info.commit,
        "commit_date": info.commit_date,
        "subset": "test fixture cut by scripts/make_ingest_fixtures.py",
    }
    (dest_dir / source.SOURCE_INFO_FILE).write_text(
        json.dumps(description, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    arguments = [Path(a) for a in sys.argv[1:]]
    main(
        arguments[0] if arguments else Path("data/external/greco"),
        arguments[1] if len(arguments) > 1 else Path("tests/fixtures/greco"),
    )
