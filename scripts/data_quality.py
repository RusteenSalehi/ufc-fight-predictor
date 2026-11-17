"""Data quality report on the loaded dataset (milestone M2).

Usage:
    uv run python scripts/data_quality.py [--db data/ufc.db] [--out docs/DATA_QUALITY.md]
        [--today YYYY-MM-DD] [--active-since 2024-01-01] [--debut-since 2025-11-22]

Reads data/ufc.db only and prints a Markdown report. With ``--out`` the report
replaces the block between the GENERATED markers in that file (the file is
created with the markers if it does not exist), so prose written around the
block survives a rerun. Every number in the block comes from this script; the
only inputs besides the database are the dates given on the command line.
``--today`` exists so a report can be reproduced on a later day.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sqlite3
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED: scripts/data_quality.py -->"
END = "<!-- END GENERATED -->"

# Gaps between consecutive events above this are listed (days), from this date on.
GAP_LIST_DAYS = 28
GAP_LIST_FROM = "2010-01-01"
GAP_TOP = 10

ATTRIBUTES = ("height_cm", "reach_cm", "stance", "dob")
ATTRIBUTE_LABELS = ("height", "reach", "stance", "date of birth")


# --- formatting --------------------------------------------------------------


def pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{100 * part / whole:.1f}%"


def table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return f"{value:,}" if isinstance(value, int) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# --- queries -------------------------------------------------------------------


def scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> object:
    return conn.execute(sql, params).fetchone()[0]


def header_section(conn: sqlite3.Connection, today: date) -> str:
    files = conn.execute(
        "SELECT DISTINCT upstream_repo, upstream_commit, snapshot_at FROM source_files"
    ).fetchall()
    lines = ["## Snapshot", ""]
    for repo, commit, snapshot_at in files:
        lines.append(
            f"- Upstream: `{repo}` at commit `{commit}` (committed {snapshot_at})"
        )
    lines.append(f"- Report run on: {today}")
    counts = {
        table_name: scalar(conn, f"SELECT count(*) FROM {table_name}")
        for table_name in (
            "events",
            "fighters",
            "bouts",
            "bout_round_stats",
            "bout_fighter_stats",
            "ingest_issues",
        )
    }
    lines += ["", table(["table", "rows"], [[k, v] for k, v in counts.items()])]
    return "\n".join(lines)


def freshness_section(conn: sqlite3.Connection, today: date) -> str:
    dates = [
        date.fromisoformat(r[0])
        for r in conn.execute("SELECT DISTINCT date FROM events ORDER BY 1")
    ]
    latest = dates[-1]
    recent_cut = today - timedelta(days=730)
    recent_gaps = [
        (b - a).days for a, b in itertools.pairwise(dates) if a >= recent_cut
    ]
    commit_date = scalar(conn, "SELECT min(snapshot_at) FROM source_files")
    after_snapshot = scalar(
        conn,
        "SELECT count(*) FROM events WHERE date > substr(?, 1, 10)",
        (commit_date,),
    )
    lines = [
        "## Freshness",
        "",
        f"- Most recent event in the database: {latest}",
        f"- Today: {today}",
        f"- Days from most recent event to today: {(today - latest).days}",
        f"- Snapshot commit date: {commit_date}",
        f"- Events dated after the snapshot commit date: {after_snapshot}",
        (
            f"- Median gap between consecutive event dates, events since {recent_cut}: "
            f"{statistics.median(recent_gaps):g} days (n={len(recent_gaps)}; "
            f"{len(dates)} distinct event dates in all)"
        ),
    ]
    return "\n".join(lines)


def per_year_section(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        """
        SELECT substr(e.date, 1, 4) AS year,
               count(DISTINCT e.event_id) AS events,
               count(b.bout_id) AS bouts,
               coalesce(sum(b.result = 'win'), 0),
               coalesce(sum(b.result = 'draw'), 0),
               coalesce(sum(b.result = 'nc'), 0)
        FROM events e LEFT JOIN bouts b ON b.event_id = e.event_id
        GROUP BY year ORDER BY year
        """
    ).fetchall()
    return "\n".join(
        [
            "## Events and bouts per year",
            "",
            table(
                ["year", "events", "bouts", "wins", "draws", "no contests"],
                [list(r) for r in rows],
            ),
        ]
    )


def results_section(conn: sqlite3.Connection) -> str:
    total = scalar(conn, "SELECT count(*) FROM bouts")
    results = conn.execute(
        "SELECT result, count(*) FROM bouts GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    methods = conn.execute(
        "SELECT coalesce(method, '(none)'), count(*) FROM bouts GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    return "\n".join(
        [
            "## Results and methods",
            "",
            table(
                ["result", "bouts", "share"],
                [[r, n, pct(n, total)] for r, n in results],
            ),
            "",
            table(
                ["method", "bouts", "share"],
                [[m, n, pct(n, total)] for m, n in methods],
            ),
        ]
    )


def stats_coverage_section(conn: sqlite3.Connection) -> str:
    bouts = conn.execute(
        """
        SELECT substr(e.date, 1, 4), count(*), sum(b.has_stats = 0)
        FROM bouts b JOIN events e USING (event_id) GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    rounds = {
        year: (n, missing)
        for year, n, missing in conn.execute(
            """
            SELECT substr(e.date, 1, 4), count(*), sum(r.ctrl_s IS NULL)
            FROM bout_round_stats r JOIN bouts b USING (bout_id) JOIN events e USING (event_id)
            GROUP BY 1
            """
        )
    }
    rows = []
    for year, n_bouts, no_stats in bouts:
        n_rounds, no_ctrl = rounds.get(year, (0, 0))
        rows.append(
            [
                year,
                n_bouts,
                no_stats,
                pct(no_stats, n_bouts),
                n_rounds,
                no_ctrl,
                pct(no_ctrl, n_rounds),
            ]
        )
    total_bouts = sum(r[1] for r in rows)
    total_no_stats = sum(r[2] for r in rows)
    total_rounds = sum(r[4] for r in rows)
    total_no_ctrl = sum(r[5] for r in rows)
    rows.append(
        [
            "all",
            total_bouts,
            total_no_stats,
            pct(total_no_stats, total_bouts),
            total_rounds,
            total_no_ctrl,
            pct(total_no_ctrl, total_rounds),
        ]
    )
    return "\n".join(
        [
            "## Statistics coverage by year",
            "",
            "A round row is one fighter in one round; control time is missing when the source has no value for it.",
            "",
            table(
                [
                    "year",
                    "bouts",
                    "bouts without stats",
                    "share",
                    "round rows",
                    "round rows without control time",
                    "share",
                ],
                rows,
            ),
        ]
    )


FIGHTER_CTE = """
WITH appearances AS (
    SELECT b.fighter_a_id AS fid, e.date FROM bouts b JOIN events e USING (event_id)
    UNION ALL
    SELECT b.fighter_b_id, e.date FROM bouts b JOIN events e USING (event_id)
), career AS (
    SELECT fid, min(date) AS first_date, max(date) AS last_date, count(*) AS n_bouts
    FROM appearances GROUP BY fid
)
"""


def missing_row(
    label: str, where: str, conn: sqlite3.Connection, params: tuple = ()
) -> list[object]:
    sql = (
        FIGHTER_CTE
        + "SELECT count(*), "
        + ", ".join(f"coalesce(sum(f.{a} IS NULL), 0)" for a in ATTRIBUTES)
        + f" FROM fighters f LEFT JOIN career c ON c.fid = f.fighter_id WHERE {where}"
    )
    n, *missing = conn.execute(sql, params).fetchone()
    row: list[object] = [label, n]
    for count in missing:
        row += [count, pct(count, n)]
    return row


def attributes_section(
    conn: sqlite3.Connection, today: date, active_since: str, debut_since: str
) -> str:
    two_years = (today - timedelta(days=730)).isoformat()
    headers = ["group", "fighters"]
    for label in ATTRIBUTE_LABELS:
        headers += [f"no {label}", "share"]
    rows = [
        missing_row("all profiles", "1 = 1", conn),
        missing_row("profiles with no loaded bout", "c.fid IS NULL", conn),
        missing_row(
            "profiles with at least one loaded bout", "c.fid IS NOT NULL", conn
        ),
        missing_row(
            f"active: a bout since {two_years} (730 days before the report date)",
            "c.last_date >= ?",
            conn,
            (two_years,),
        ),
        missing_row(
            f"active: a bout since {active_since}",
            "c.last_date >= ?",
            conn,
            (active_since,),
        ),
        missing_row(
            f"active since {active_since}, debut before {debut_since}",
            "c.last_date >= ? AND c.first_date < ?",
            conn,
            (active_since, debut_since),
        ),
        missing_row(
            f"active since {active_since}, debut on or after {debut_since}",
            "c.last_date >= ? AND c.first_date >= ?",
            conn,
            (active_since, debut_since),
        ),
    ]

    by_year = conn.execute(
        FIGHTER_CTE
        + """
        SELECT substr(c.first_date, 1, 4), count(*), sum(f.height_cm IS NULL), sum(f.reach_cm IS NULL),
               sum(f.stance IS NULL), sum(f.dob IS NULL)
        FROM fighters f JOIN career c ON c.fid = f.fighter_id
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    year_rows = [
        [y, n, pct(h, n), pct(r, n), pct(s, n), pct(d, n)]
        for y, n, h, r, s, d in by_year
        if y >= "2015"
    ]

    quarters = conn.execute(
        FIGHTER_CTE
        + """
        SELECT substr(c.first_date, 1, 4) || '-Q' || ((cast(substr(c.first_date, 6, 2) AS integer) + 2) / 3) AS q,
               count(*), sum(f.height_cm IS NULL), sum(f.reach_cm IS NULL), sum(f.dob IS NULL),
               min(c.first_date), max(c.first_date)
        FROM fighters f JOIN career c ON c.fid = f.fighter_id
        WHERE c.first_date >= ?
        GROUP BY 1 ORDER BY 1
        """,
        ("2024-01-01",),
    ).fetchall()
    quarter_rows = [
        [q, n, pct(h, n), pct(r, n), pct(d, n), lo, hi]
        for q, n, h, r, d, lo, hi in quarters
    ]

    last_debuts = conn.execute(
        FIGHTER_CTE
        + """
        SELECT c.first_date, count(*), sum(f.height_cm IS NULL), sum(f.reach_cm IS NULL), sum(f.dob IS NULL)
        FROM fighters f JOIN career c ON c.fid = f.fighter_id
        WHERE c.first_date >= date(?, '-120 days')
        GROUP BY 1 ORDER BY 1
        """,
        (debut_since,),
    ).fetchall()
    debut_rows = [[d, n, h, r, b] for d, n, h, r, b in last_debuts]

    return "\n".join(
        [
            "## Fighter attributes",
            "",
            (
                "Missing means NULL in `fighters`. That includes the attributes left empty "
                "on purpose when a profile's two names disagree (DECISIONS D22). A fighter "
                "is a debutant at their first loaded bout; a bout dropped during ingestion "
                "is not counted."
            ),
            "",
            table(headers, rows),
            "",
            "### Share missing by debut year (first loaded bout), 2015 on",
            "",
            table(
                [
                    "debut year",
                    "fighters",
                    "no height",
                    "no reach",
                    "no stance",
                    "no date of birth",
                ],
                year_rows,
            ),
            "",
            "### Share missing by debut quarter, 2024 on",
            "",
            table(
                [
                    "debut quarter",
                    "fighters",
                    "no height",
                    "no reach",
                    "no date of birth",
                    "first debut",
                    "last debut",
                ],
                quarter_rows,
            ),
            "",
            f"### Debuts per event date, from 120 days before {debut_since}",
            "",
            "Counts of fighters missing each attribute, by the date of their first loaded bout.",
            "",
            table(
                [
                    "debut date",
                    "debutants",
                    "no height",
                    "no reach",
                    "no date of birth",
                ],
                debut_rows,
            ),
        ]
    )


def duplicates_section(conn: sqlite3.Connection) -> str:
    shared_names = conn.execute(
        "SELECT count(*), coalesce(sum(n), 0) FROM (SELECT count(*) AS n FROM fighters GROUP BY lower(name) HAVING n > 1)"
    ).fetchone()
    shared_names_in_bouts = conn.execute(
        FIGHTER_CTE
        + """
        SELECT count(*), coalesce(sum(n), 0) FROM (
            SELECT count(*) AS n FROM fighters f JOIN career c ON c.fid = f.fighter_id
            GROUP BY lower(f.name) HAVING n > 1)
        """
    ).fetchone()
    same_name_dob = scalar(
        conn,
        "SELECT count(*) FROM (SELECT 1 FROM fighters WHERE dob IS NOT NULL GROUP BY lower(name), dob HAVING count(*) > 1)",
    )
    twice_one_event = scalar(
        conn,
        """
        SELECT count(*) FROM (
            SELECT b.event_id, p.fid FROM bouts b
            JOIN (SELECT bout_id, fighter_a_id AS fid FROM bouts UNION ALL SELECT bout_id, fighter_b_id FROM bouts) p USING (bout_id)
            GROUP BY b.event_id, p.fid HAVING count(*) > 1)
        """,
    )
    same_pair_event = scalar(
        conn,
        """
        SELECT count(*) FROM (
            SELECT 1 FROM bouts
            GROUP BY event_id, min(fighter_a_id, fighter_b_id), max(fighter_a_id, fighter_b_id) HAVING count(*) > 1)
        """,
    )
    same_name_event = scalar(
        conn,
        "SELECT count(*) FROM (SELECT 1 FROM events GROUP BY lower(name) HAVING count(*) > 1)",
    )
    same_date_events = conn.execute(
        "SELECT date, group_concat(name, ' | ') FROM events GROUP BY date HAVING count(*) > 1 ORDER BY date"
    ).fetchall()
    same_event_date_fighter = scalar(
        conn,
        """
        SELECT count(*) FROM (
            SELECT p.fid, e.date FROM bouts b JOIN events e USING (event_id)
            JOIN (SELECT bout_id, fighter_a_id AS fid FROM bouts UNION ALL SELECT bout_id, fighter_b_id FROM bouts) p USING (bout_id)
            GROUP BY p.fid, e.date HAVING count(*) > 1)
        """,
    )
    twice_since_2000 = scalar(
        conn,
        """
        SELECT count(*) FROM (
            SELECT p.fid, e.date FROM bouts b JOIN events e USING (event_id)
            JOIN (SELECT bout_id, fighter_a_id AS fid FROM bouts UNION ALL SELECT bout_id, fighter_b_id FROM bouts) p USING (bout_id)
            WHERE e.date >= '2000-01-01'
            GROUP BY p.fid, e.date HAVING count(*) > 1)
        """,
    )
    bout_dupes = scalar(conn, "SELECT count(*) - count(DISTINCT bout_id) FROM bouts")
    lines = [
        "## Duplicate checks",
        "",
        table(
            ["check", "count"],
            [
                ["bout IDs appearing more than once", bout_dupes],
                ["event names shared by more than one event", same_name_event],
                [
                    "fighter names shared by more than one profile (names / profiles)",
                    f"{shared_names[0]} / {shared_names[1]}",
                ],
                [
                    "  of those, profiles with a loaded bout (names / profiles)",
                    f"{shared_names_in_bouts[0]} / {shared_names_in_bouts[1]}",
                ],
                [
                    "profile pairs with the same name and the same date of birth",
                    same_name_dob,
                ],
                ["fighters with two bouts at one event", twice_one_event],
                ["fighters with two bouts on one date", same_event_date_fighter],
                ["  of those, on dates since 2000", twice_since_2000],
                ["fighter pairs meeting twice at one event", same_pair_event],
                ["dates with more than one event", len(same_date_events)],
            ],
        ),
    ]
    if same_date_events:
        lines += ["", "Dates with more than one event:", ""]
        lines += [f"- {d}: {names}" for d, names in same_date_events]
    return "\n".join(lines)


def issues_section(conn: sqlite3.Connection) -> str:
    grouped = conn.execute(
        """
        SELECT check_name, severity, count(*) FROM ingest_issues
        GROUP BY 1, 2 ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, 1
        """
    ).fetchall()
    lines = [
        "## ingest_issues, grouped by check",
        "",
        table(["check", "severity", "rows"], [list(r) for r in grouped]),
        "",
        "### Errors and warnings in full",
        "",
    ]
    for severity, check, entity_type, entity_id, message in conn.execute(
        """
        SELECT severity, check_name, coalesce(entity_type, ''), coalesce(entity_id, ''), message
        FROM ingest_issues WHERE severity IN ('error', 'warning') ORDER BY severity, check_name, seq
        """
    ):
        where = f" [{entity_type} {entity_id}]" if entity_id else ""
        lines.append(f"- **{severity}** `{check}`{where}: {message}")
    resolutions = conn.execute(
        """
        SELECT CASE WHEN message LIKE '%by override' THEN 'override'
                    WHEN message LIKE '%by name+weight' THEN 'name + weight class' ELSE 'other' END,
               count(*) FROM ingest_issues WHERE check_name = 'fighter_resolution' GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    lines += [
        "",
        "### Fighter name resolutions by rule",
        "",
        table(["rule", "bouts"], [list(r) for r in resolutions]),
    ]
    shared = conn.execute(
        """
        SELECT substr(message, instr(message, '''') + 1,
                      instr(substr(message, instr(message, '''') + 1), '''') - 1) AS name, count(*)
        FROM ingest_issues WHERE check_name = 'fighter_resolution' GROUP BY 1 ORDER BY 2 DESC, 1
        """
    ).fetchall()
    lines += [
        "",
        "Names resolved through a rule other than a unique name, with bouts affected:",
        "",
    ]
    lines += [table(["name", "bouts"], [list(r) for r in shared])]
    return "\n".join(lines)


def dropped_section(conn: sqlite3.Connection) -> str:
    """Evidence for reviewing each bout the load dropped (errors in ingest_issues)."""
    lines = ["## Dropped bouts: what the database says about them", ""]
    unresolved = conn.execute(
        "SELECT entity_id, message FROM ingest_issues WHERE check_name = 'unresolved_fighter'"
    ).fetchall()
    for bout_id, message in unresolved:
        lines += [f"Bout `{bout_id}`: {message}", ""]
        ids = sorted(set(re.findall(r"\b[0-9a-f]{16}\b", message)) - {bout_id})
        rows = []
        for fighter_id in ids:
            name, nickname, height, reach, stance, dob = conn.execute(
                "SELECT name, coalesce(nickname, ''), height_cm, reach_cm, stance, dob FROM fighters WHERE fighter_id = ?",
                (fighter_id,),
            ).fetchone()
            loaded = scalar(
                conn,
                "SELECT count(*) FROM bouts WHERE fighter_a_id = ? OR fighter_b_id = ?",
                (fighter_id, fighter_id),
            )
            rows.append(
                [
                    f"`{fighter_id}`",
                    name,
                    nickname or "-",
                    "-" if height is None else height,
                    "-" if reach is None else reach,
                    stance or "-",
                    dob or "-",
                    loaded,
                ]
            )
        lines += [
            table(
                [
                    "fighter_id",
                    "listed name",
                    "nickname",
                    "height_cm",
                    "reach_cm",
                    "stance",
                    "dob",
                    "loaded bouts",
                ],
                rows,
            ),
            "",
        ]
    orphans = conn.execute(
        "SELECT entity_id, message FROM ingest_issues WHERE check_name = 'orphan_event'"
    ).fetchall()
    by_event: dict[str, list[str]] = {}
    for bout_id, message in orphans:
        match = re.search(r"names event \['(.*)'\]", message)
        by_event.setdefault(match.group(1) if match else "(unparsed)", []).append(
            bout_id
        )
    for event_name, bout_ids in by_event.items():
        prefix = event_name.rsplit(" ", 1)[0]
        similar = scalar(
            conn, "SELECT count(*) FROM events WHERE name LIKE ?", (prefix + "%",)
        )
        lines.append(
            f"Event `{event_name}` is named by {len(bout_ids)} dropped bouts "
            f"({', '.join(f'`{b}`' for b in bout_ids)}); events in the events file whose name starts "
            f"with `{prefix}`: {similar}."
        )
    return "\n".join(lines).rstrip("\n")


def gaps_section(conn: sqlite3.Connection) -> str:
    dates = [
        date.fromisoformat(r[0])
        for r in conn.execute("SELECT DISTINCT date FROM events ORDER BY 1")
    ]
    names = dict(
        conn.execute("SELECT date, group_concat(name, ' | ') FROM events GROUP BY date")
    )
    gaps = [(b, (b - a).days, a) for a, b in itertools.pairwise(dates)]
    long_recent = [
        g for g in gaps if g[1] > GAP_LIST_DAYS and g[0].isoformat() >= GAP_LIST_FROM
    ]
    top = sorted(gaps, key=lambda g: -g[1])[:GAP_TOP]

    def row(gap: tuple[date, int, date]) -> list[object]:
        later, days, earlier = gap
        return [days, str(earlier), str(later), names[later.isoformat()]]

    per_year: dict[str, list[int]] = {}
    for later, days, _ in gaps:
        per_year.setdefault(str(later.year), []).append(days)
    year_rows = [
        [y, len(v), f"{statistics.median(v):g}", max(v)]
        for y, v in sorted(per_year.items())
        if y >= "2000"
    ]
    return "\n".join(
        [
            "## Gaps between consecutive event dates",
            "",
            f"### Largest {GAP_TOP} gaps overall",
            "",
            table(
                ["days", "previous event", "next event", "next event name"],
                [row(g) for g in top],
            ),
            "",
            f"### Gaps over {GAP_LIST_DAYS} days since {GAP_LIST_FROM}",
            "",
            table(
                ["days", "previous event", "next event", "next event name"],
                [row(g) for g in long_recent],
            )
            if long_recent
            else "None.",
            "",
            "### Gap per year, 2000 on (gap ending in that year)",
            "",
            table(["year", "gaps", "median days", "longest days"], year_rows),
        ]
    )


def build_report(
    conn: sqlite3.Connection, today: date, active_since: str, debut_since: str
) -> str:
    sections = [
        header_section(conn, today),
        freshness_section(conn, today),
        attributes_section(conn, today, active_since, debut_since),
        per_year_section(conn),
        results_section(conn),
        stats_coverage_section(conn),
        duplicates_section(conn),
        issues_section(conn),
        dropped_section(conn),
        gaps_section(conn),
    ]
    return "\n\n".join(sections) + "\n"


def write_block(path: Path, report: str) -> None:
    block = f"{BEGIN}\n\n{report}\n{END}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            path.write_text(head + block.rstrip("\n") + tail, encoding="utf-8")
            return
        path.write_text(text.rstrip("\n") + "\n\n" + block, encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/ufc.db"))
    parser.add_argument(
        "--out", type=Path, help="Markdown file whose generated block is replaced"
    )
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--active-since", default="2024-01-01")
    parser.add_argument("--debut-since", default="2025-11-22")
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(
            f"{args.db} does not exist; run `uv run python -m ufc.ingest load`",
            file=sys.stderr,
        )
        return 1
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        report = build_report(conn, args.today, args.active_since, args.debut_since)
    finally:
        conn.close()
    if args.out:
        write_block(args.out, report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
