"""Did upstream change fighter profiles after the fact? (era-shift diagnostic)

    uv run python scripts/profile_history.py

Profiles in the pinned snapshot are as they stand today (leakage rule 3). The
upstream repository keeps history, so this downloads its profile file as it
was at the last commit before each of a few dates and compares the parsed
height, reach, stance and date of birth with the pinned snapshot, for every
profile and for the fighters who have a test-period bout.

It then groups the test-period results of ``python -m ufc.models.diagnostics``
(run that first) by whether the fighters' profiles changed, and scores the
saved candidate, without refitting it, on the bouts fought before upstream's
refresh using the profiles upstream held then. That reads test-period rows,
so the run is recorded in ``data/models/test_diagnostics.jsonl`` (DECISIONS
D45). Analysis only.

Requests go to GitHub only (the commits API and raw files), through the
ingest's session: timeouts, retries with backoff, a descriptive User-Agent.
Old files are kept under ``data/external/greco_history/`` (not committed) and
are not downloaded twice. Output: ``data/reports/diagnostics/profile_history.json``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.db.connection import connect
from ufc.features.build import build_features
from ufc.features.config import FeatureConfig
from ufc.features.data import FighterAttrs, load_feature_data
from ufc.ingest import source
from ufc.ingest.fetch import (
    PAUSE_BETWEEN_REQUESTS_S,
    RAW_URL,
    FetchError,
    get_with_retries,
    make_session,
)
from ufc.ingest.parsing import (
    id_from_url,
    parse_dob,
    parse_height_cm,
    parse_reach_cm,
    parse_stance,
)
from ufc.models import registry
from ufc.models.diagnostics import LEDGER_FILE, group_summaries
from ufc.models.metrics import bootstrap_mean_interval, bout_log_loss, summary
from ufc.models.splits import Splits, attribute_gap, labelled

logger = logging.getLogger(__name__)

COMMITS_URL = (
    "https://api.github.com/repos/{repo}/commits?path={path}&until={until}&per_page=1"
)
DEFAULT_BEFORE = (
    "2025-01-01",
    "2025-04-01",
    "2025-07-01",
    "2025-10-01",
    "2025-11-22",
    "2025-11-24",
)
"""The first is before the test period; test-period results are grouped by
what changed since that capture. The rest show when changes happened, up to
either side of the 22 Nov 2025 cut-off in docs/DATA_QUALITY.md."""

ATTRIBUTES: dict[str, tuple[str, Callable[[str], object]]] = {
    "height_cm": ("HEIGHT", parse_height_cm),
    "reach_cm": ("REACH", parse_reach_cm),
    "stance": ("STANCE", parse_stance),
    "dob": ("DOB", parse_dob),
}

Profiles = dict[str, dict[str, object]]


def read_profiles(path: Path) -> Profiles:
    """Parsed attributes per fighter ID. A later row for the same fighter
    replaces an earlier one."""
    profiles: Profiles = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            fighter_id = id_from_url(row["URL"], "fighter")
            profiles[fighter_id] = {
                name: parse(row[column]) for name, (column, parse) in ATTRIBUTES.items()
            }
    return profiles


def compare(
    old: Mapping[str, Mapping[str, object]],
    new: Mapping[str, Mapping[str, object]],
    only: set[str] | None = None,
) -> dict:
    """Counts of what happened to each attribute between two captures, over
    the profiles present in both (and in ``only``, if given)."""
    shared = [f for f in old if f in new and (only is None or f in only)]
    result: dict = {
        "profiles_in_both": len(shared),
        "profiles_only_in_newer": sum(
            1 for f in new if f not in old and (only is None or f in only)
        ),
        "any_attribute_differs": 0,
        "attributes": {},
    }
    differs: set[str] = set()
    for name in ATTRIBUTES:
        counts = {
            "same": 0,
            "empty_in_both": 0,
            "filled_later": 0,
            "emptied": 0,
            "value_changed": 0,
        }
        for fighter in shared:
            before, after = old[fighter][name], new[fighter][name]
            if before is None and after is None:
                counts["empty_in_both"] += 1
            elif before is None:
                counts["filled_later"] += 1
            elif after is None:
                counts["emptied"] += 1
            elif before == after:
                counts["same"] += 1
            else:
                counts["value_changed"] += 1
            if before != after:
                differs.add(fighter)
        result["attributes"][name] = counts
    result["any_attribute_differs"] = len(differs)
    return result


def change_state(
    fighter: str, old: Mapping[str, Mapping[str, object]], new: Mapping, name=None
) -> str:
    """ "absent" (no profile in the older capture), "same" or "changed". With
    ``name``, for that attribute only, and "filled" if it was empty before."""
    if fighter not in old:
        return "absent"
    if name is None:
        return "same" if dict(old[fighter]) == dict(new[fighter]) else "changed"
    before, after = old[fighter][name], new[fighter][name]
    if before == after:
        return "same"
    return "filled" if before is None else "changed"


def results_by_change(
    bouts: pd.DataFrame, old: Profiles, new: Profiles, test_start: str, until: str
) -> dict:
    """Test-period results of the diagnostic's models, by whether the two
    fighters' profiles changed after the older capture. ``until`` is the date
    of the last capture that differs from the pinned snapshot: a bout before
    it could have had its fighters' profiles edited after it was fought."""
    test = bouts[bouts["date"] >= pd.Timestamp(test_start)]
    a = test["fighter_a_id"].map(lambda f: change_state(f, old, new)).to_numpy()
    b = test["fighter_b_id"].map(lambda f: change_state(f, old, new)).to_numpy()
    dob_a = test["fighter_a_id"].map(lambda f: change_state(f, old, new, "dob"))
    dob_b = test["fighter_b_id"].map(lambda f: change_state(f, old, new, "dob"))
    listed = (a != "absent") & (b != "absent")
    same = (a == "same") & (b == "same")
    early = (test["date"] < pd.Timestamp(until)).to_numpy()
    groups = {
        "all test-period bouts": np.ones(len(test), dtype=bool),
        "both profiles unchanged": same,
        "both listed, either profile changed": listed & ~same,
        "either fighter not listed yet": ~listed,
        "bout before the last change; both profiles unchanged": early & same,
        "bout before the last change; both listed, either changed": early
        & listed
        & ~same,
        "either date of birth filled in later": (
            (dob_a == "filled") | (dob_b == "filled")
        ).to_numpy(),
        "either date of birth changed": (
            (dob_a == "changed") | (dob_b == "changed")
        ).to_numpy(),
    }
    return group_summaries(
        test, groups, ("full", "no_profile", "no_dob", "logistic", "elo")
    )


def dob_changes(old: Profiles, new: Profiles) -> dict:
    """How far dates of birth moved, where both captures have one."""
    years = [
        abs((new[f]["dob"] - old[f]["dob"]).days) / 365.25
        for f in old
        if f in new
        and old[f]["dob"]
        and new[f]["dob"]
        and old[f]["dob"] != new[f]["dob"]
    ]
    if not years:
        return {"n": 0}
    return {
        "n": len(years),
        "median_years": float(np.median(years)),
        "largest_years": float(np.max(years)),
        "under_one_year": int(np.sum(np.asarray(years) < 1.0)),
    }


def as_available(
    capture: Profiles,
    captured: str,
    args: argparse.Namespace,
) -> dict:
    """Score the saved candidate on test-period bouts fought before
    ``captured``, twice: with today's profiles and with the profiles upstream
    held in that capture. Nothing is refitted; only the inputs differ. A
    fighter with no profile in the capture gets no attributes."""
    version = json.loads((args.eval / "selection.json").read_text())["version"]
    candidate, _ = registry.load_candidate(args.models, version)
    raw = registry.read_json(args.models, version, "feature_pipeline.json")["config"]
    config = FeatureConfig(**{**raw, "train_end": date.fromisoformat(raw["train_end"])})
    conn = connect(args.db)
    try:
        data = load_feature_data(conn)
    finally:
        conn.close()
    then = {
        fighter: FighterAttrs(fighter, **capture[fighter])
        if fighter in capture
        else FighterAttrs(fighter)
        for fighter in data.fighters
    }
    reason = f"era-shift diagnostic of {version} (analysis only): profiles as available"
    with (args.models / LEDGER_FILE).open("a", encoding="utf-8") as handle:
        entry = {
            "what": reason,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    start, end = date.fromisoformat(args.test_start), date.fromisoformat(captured[:10])
    out: dict = {
        "capture": captured,
        "from": start.isoformat(),
        "before": end.isoformat(),
    }
    losses: dict[str, pd.Series] = {}
    for label, fighters in (("profiles_today", data.fighters), ("profiles_then", then)):
        splits = Splits(build_features(replace(data, fighters=fighters), config).frame)
        splits.unlock_test(reason)
        rows = labelled(splits.between(start, end - timedelta(days=1), "as available"))
        y = rows["label"].to_numpy(dtype=np.float64)
        gap = attribute_gap(rows).to_numpy()
        p = candidate.predict(rows)
        losses[label] = pd.Series(bout_log_loss(p, y), index=rows["bout_id"].to_numpy())
        out[label] = {
            "model": summary(p, y),
            "n_attribute_gap": int(gap.sum()),
            "n_age_missing": int(rows["d_age"].isna().sum()),
            "n_height_or_reach_missing": int(
                (rows["d_height_cm"].isna() | rows["d_reach_cm"].isna()).sum()
            ),
            "model_no_attribute_gap": summary(p[~gap], y[~gap]),
            "model_attribute_gap": summary(p[gap], y[gap]) if gap.any() else None,
        }
    today = losses["profiles_today"]
    out["then_minus_today"] = bootstrap_mean_interval(
        (losses["profiles_then"].loc[today.index] - today).to_numpy()
    )
    return out


def commit_before(session, repo: str, when: str) -> tuple[str, str]:
    """The last commit that touched the profile file before ``when``."""
    url = COMMITS_URL.format(
        repo=repo, path=source.FIGHTER_TOTT_FILE, until=f"{when}T00:00:00Z"
    )
    try:
        commit = get_with_retries(session, url).json()[0]
        return commit["sha"], commit["commit"]["committer"]["date"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise FetchError(f"no commit of the profile file before {when}") from exc


def download(session, repo: str, commit: str, root: Path) -> Path:
    path = root / commit / source.FIGHTER_TOTT_FILE
    if path.exists():
        return path
    time.sleep(PAUSE_BETWEEN_REQUESTS_S)
    url = RAW_URL.format(repo=repo, commit=commit, file_name=source.FIGHTER_TOTT_FILE)
    content = get_with_retries(session, url).content
    header = next(csv.reader(content.decode("utf-8").splitlines()[:1]))
    source.check_header(source.FIGHTER_TOTT_FILE, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_period_fighters(db: Path, since: str) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT b.fighter_a_id, b.fighter_b_id FROM bouts b "
            "JOIN events e ON e.event_id = b.event_id "
            "WHERE e.date >= ? AND b.status = 'completed'",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    return {fighter for row in rows for fighter in row}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshot", type=Path, default=Path("data/external/greco"))
    parser.add_argument(
        "--history", type=Path, default=Path("data/external/greco_history")
    )
    parser.add_argument("--db", type=Path, default=Path("data/ufc.db"))
    parser.add_argument("--before", nargs="+", default=list(DEFAULT_BEFORE))
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--models", type=Path, default=registry.DEFAULT_ROOT)
    parser.add_argument("--eval", type=Path, default=Path("data/reports/eval"))
    parser.add_argument(
        "--bouts",
        type=Path,
        default=Path("data/reports/diagnostics/era_shift_bouts.parquet"),
        help="per-bout probabilities written by python -m ufc.models.diagnostics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/diagnostics/profile_history.json"),
    )
    args = parser.parse_args(argv)

    pinned = json.loads((args.snapshot / source.SOURCE_INFO_FILE).read_text())
    current = read_profiles(args.snapshot / source.FIGHTER_TOTT_FILE)
    fighters = test_period_fighters(args.db, args.test_start)
    session = make_session()
    captures: list[tuple[str, str, Profiles]] = []
    try:
        for when in args.before:
            commit, committed = commit_before(session, pinned["repo"], when)
            path = download(session, pinned["repo"], commit, args.history)
            captures.append((commit, committed, read_profiles(path)))
    except FetchError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    captures.append((pinned["commit"], pinned["commit_date"], current))

    record: dict = {
        "pinned_commit": pinned["commit"],
        "test_period_fighters": len(fighters),
        "against_pinned": [],
        "consecutive": [],
    }
    for commit, committed, profiles in captures[:-1]:
        record["against_pinned"].append(
            {
                "commit": commit,
                "committed": committed,
                "n_profiles": len(profiles),
                "all_profiles": compare(profiles, current),
                "test_period_fighters": compare(profiles, current, fighters),
            }
        )
    for (c1, d1, p1), (c2, d2, p2) in itertools.pairwise(captures):
        record["consecutive"].append(
            {
                "from": {"commit": c1, "committed": d1},
                "to": {"commit": c2, "committed": d2},
                "all_profiles": compare(p1, p2),
            }
        )
    # The last capture that still differs from the pinned snapshot.
    differing = [
        e["committed"]
        for e in record["against_pinned"]
        if e["all_profiles"]["any_attribute_differs"]
    ]
    first = captures[0]
    record["dob_changes_since_first"] = dob_changes(first[2], current)
    if differing and args.bouts.exists():
        record["last_capture_that_differs"] = max(differing)
        record["results_by_change_since_first"] = results_by_change(
            pd.read_parquet(args.bouts),
            first[2],
            current,
            args.test_start,
            max(differing)[:10],
        )
        # Profiles as upstream held them through most of 2025: the last
        # capture before the refresh.
        last = next(c for c in captures if c[1] == max(differing))
        record["as_available"] = as_available(last[2], last[1], args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    for entry in record["against_pinned"]:
        print(f"{entry['committed']} against the pinned snapshot:")
        for scope in ("all_profiles", "test_period_fighters"):
            row = entry[scope]
            print(
                f"  {scope}: {row['profiles_in_both']} in both, "
                f"{row['any_attribute_differs']} differ"
            )
            for name, counts in row["attributes"].items():
                print(f"    {name:10} {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
