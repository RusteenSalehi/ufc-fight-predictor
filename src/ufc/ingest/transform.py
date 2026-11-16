"""Turn the raw CSV rows of one snapshot into typed tables.

Pure: rows in, records and a list of issues out. No file or database access.

The published files join on text, not IDs: a fight row names its event and
its two fighters, and a stats row names event, bout and fighter. Every such
join is done here, and every row that cannot be joined with certainty is
dropped and reported rather than guessed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ufc.ingest import parsing as p
from ufc.ingest.identity import (
    METHOD_AMBIGUOUS,
    METHOD_NAME,
    FighterResolver,
    Override,
    Profile,
)
from ufc.ingest.models import (
    Bout,
    Dataset,
    Event,
    Fighter,
    FightStats,
    Issue,
    RoundStats,
    StatLine,
    sum_stat_lines,
)
from ufc.ingest.source import RawTables, Row

KNOWN_METHODS = frozenset(
    {
        "KO/TKO",
        "Submission",
        "Decision - Unanimous",
        "Decision - Split",
        "Decision - Majority",
        "TKO - Doctor's Stoppage",
        "DQ",
        "Overturned",
        "Could Not Continue",
        "Other",
    }
)
_NO_CLASS_LIMIT = frozenset({None, "Catch Weight", "Open Weight"})

StatsKey = tuple[str, str]


def _error(
    check: str,
    message: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
):
    return Issue("error", check, message, entity_type, entity_id)


def _warning(
    check: str,
    message: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
):
    return Issue("warning", check, message, entity_type, entity_id)


def _info(
    check: str,
    message: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
):
    return Issue("info", check, message, entity_type, entity_id)


@dataclass(frozen=True, slots=True)
class _PendingBout:
    """A fight whose result and fighters are settled but whose stats are not yet attached."""

    bout: Bout
    name_a: str
    name_b: str
    stats_key: StatsKey


def build_dataset(raw: RawTables, overrides: Sequence[Override] = ()) -> Dataset:
    """Build events, fighters, bouts and stats from one snapshot's raw rows."""
    issues: list[Issue] = []
    events, event_by_name = _build_events(raw.events, issues)
    fighters, profiles = _build_fighters(raw.fighter_details, raw.fighter_tott, issues)

    resolver = FighterResolver(profiles, overrides)
    for override in resolver.invalid_overrides:
        issues.append(
            _warning(
                "identity_override",
                f"override for {override.name!r} points to unknown fighter {override.fighter_id}; "
                "ignored",
                "fighter",
                override.fighter_id,
            )
        )

    pending, source_fights = _build_bouts(
        raw.fight_results, event_by_name, resolver, issues
    )
    _check_fight_details(raw.fight_details, raw.fight_results, issues)
    for override in resolver.unused_overrides():
        issues.append(
            _info(
                "identity_override",
                f"override for {override.name!r} was not needed by any loaded bout",
                "fighter",
                override.fighter_id,
            )
        )

    bouts, round_stats, fight_stats = _attach_stats(pending, raw.fight_stats, issues)
    bouts = _infer_womens_bouts(bouts, issues)
    return Dataset(
        events=events,
        fighters=fighters,
        bouts=bouts,
        round_stats=round_stats,
        fight_stats=fight_stats,
        issues=issues,
        source_fights=source_fights,
    )


def _build_events(
    rows: list[Row], issues: list[Issue]
) -> tuple[list[Event], dict[str, Event]]:
    """Parse events and index them by name, the only key fight rows carry."""
    by_id: dict[str, Event] = {}
    for row in rows:
        try:
            event = Event(
                event_id=p.id_from_url(row["URL"], "event"),
                name=p.clean(row["EVENT"]),
                date=p.parse_event_date(row["DATE"]),
                location=p.clean(row["LOCATION"]) or None,
            )
        except p.ParseError as exc:
            issues.append(_error("unparseable_event", f"{row['EVENT']!r}: {exc}"))
            continue
        if not event.name:
            issues.append(
                _error(
                    "unparseable_event", "event has no name", "event", event.event_id
                )
            )
        elif event.event_id in by_id:
            issues.append(
                _error(
                    "duplicate_id",
                    "event ID appears more than once",
                    "event",
                    event.event_id,
                )
            )
        else:
            by_id[event.event_id] = event

    name_counts = Counter(event.name for event in by_id.values())
    by_name: dict[str, Event] = {}
    for event in by_id.values():
        if name_counts[event.name] > 1:
            issues.append(
                _warning(
                    "duplicate_event_name",
                    f"event name {event.name!r} is shared by several events; fights that name it "
                    "cannot be placed",
                    "event",
                    event.event_id,
                )
            )
        else:
            by_name[event.name] = event
    return list(by_id.values()), by_name


def _index_by_id(
    rows: list[Row], issues: list[Issue], file_label: str
) -> dict[str, Row]:
    by_id: dict[str, Row] = {}
    for row in rows:
        try:
            fighter_id = p.id_from_url(row["URL"], "fighter")
        except p.ParseError as exc:
            issues.append(_error("unparseable_fighter", f"{file_label}: {exc}"))
            continue
        if fighter_id in by_id:
            issues.append(
                _error(
                    "duplicate_id",
                    f"fighter ID appears more than once in {file_label}",
                    "fighter",
                    fighter_id,
                )
            )
        else:
            by_id[fighter_id] = row
    return by_id


def names_conflict(first: str, second: str) -> bool:
    """Whether two names for one profile look like two different people.

    ``"Zach Reese"`` / ``"Zachary Reese"`` share a word and
    ``"ColleyBradford"`` / ``"Colley Bradford"`` share their letters, so
    neither conflicts. ``"Tyrell Fortune"`` / ``"Regina Malpica"`` does.
    """
    key_a, key_b = p.name_key(first), p.name_key(second)
    if set(key_a.split()) & set(key_b.split()):
        return False
    return sorted(key_a.replace(" ", "")) != sorted(key_b.replace(" ", ""))


def _build_fighters(
    listing_rows: list[Row], tape_rows: list[Row], issues: list[Issue]
) -> tuple[list[Fighter], list[Profile]]:
    """Merge the two fighter files into one profile per fighter ID.

    The listing file is rewritten upstream from the live fighter list, so it
    holds current names. The tale-of-the-tape file is appended to once per
    fighter and never refreshed, so its names and attributes date from when
    the fighter was first seen. Both names are kept for matching. If the two
    names look like different people the profile's attributes are not
    trusted and are left empty.
    """
    listing = _index_by_id(listing_rows, issues, "fighter listing")
    tape = _index_by_id(tape_rows, issues, "tale of the tape")
    fighters: list[Fighter] = []
    profiles: list[Profile] = []

    for fighter_id in sorted(listing.keys() | tape.keys()):
        listing_row, tape_row = listing.get(fighter_id), tape.get(fighter_id)
        listing_name = (
            p.clean(f"{listing_row['FIRST']} {listing_row['LAST']}")
            if listing_row
            else ""
        )
        tape_name = p.clean(tape_row["FIGHTER"]) if tape_row else ""
        name = listing_name or tape_name
        if not name:
            issues.append(
                _error(
                    "unparseable_fighter", "fighter has no name", "fighter", fighter_id
                )
            )
            continue

        trusted = tape_row is not None
        if listing_name and tape_name and names_conflict(listing_name, tape_name):
            trusted = False
            issues.append(
                _warning(
                    "profile_name_conflict",
                    f"listed as {listing_name!r} but attributes were recorded under "
                    f"{tape_name!r}; height, reach, stance and date of birth left empty",
                    "fighter",
                    fighter_id,
                )
            )

        attributes: dict[str, object] = {}
        weight_lbs = None
        if trusted and tape_row is not None:
            weight_lbs = _parse_attribute(
                p.parse_weight_lbs, tape_row["WEIGHT"], fighter_id, issues
            )
            attributes = {
                "height_cm": _parse_attribute(
                    p.parse_height_cm, tape_row["HEIGHT"], fighter_id, issues
                ),
                "reach_cm": _parse_attribute(
                    p.parse_reach_cm, tape_row["REACH"], fighter_id, issues
                ),
                "stance": p.parse_stance(tape_row["STANCE"]),
                "dob": _parse_attribute(
                    p.parse_dob, tape_row["DOB"], fighter_id, issues
                ),
            }

        nickname = p.clean(listing_row["NICKNAME"]) if listing_row else ""
        fighters.append(
            Fighter(
                fighter_id=fighter_id,
                name=name,
                nickname=nickname or None,
                height_cm=attributes.get("height_cm"),  # type: ignore[arg-type]
                reach_cm=attributes.get("reach_cm"),  # type: ignore[arg-type]
                stance=attributes.get("stance"),  # type: ignore[arg-type]
                dob=attributes.get("dob"),  # type: ignore[arg-type]
            )
        )
        names = tuple(dict.fromkeys(n for n in (listing_name, tape_name) if n))
        profiles.append(
            Profile(fighter_id=fighter_id, names=names, weight_lbs=weight_lbs)
        )
    return fighters, profiles


def _parse_attribute(parser, text: str, fighter_id: str, issues: list[Issue]):
    try:
        return parser(text)
    except p.ParseError as exc:
        issues.append(
            _warning(
                "bad_fighter_attribute", f"{exc}; left empty", "fighter", fighter_id
            )
        )
        return None


def _build_bouts(
    rows: list[Row],
    event_by_name: dict[str, Event],
    resolver: FighterResolver,
    issues: list[Issue],
) -> tuple[list[_PendingBout], int]:
    """Parse fight results, place each fight in its event and resolve its fighters.

    Returns the bouts that survive and the number of distinct fights in the file.
    """
    rows_by_fight: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        try:
            rows_by_fight[p.id_from_url(row["URL"], "fight")].append(row)
        except p.ParseError as exc:
            issues.append(_error("unparseable_fight", f"{row['BOUT']!r}: {exc}"))

    pending: list[_PendingBout] = []
    relisted = 0
    for fight_id, fight_rows in rows_by_fight.items():
        placed = [r for r in fight_rows if p.clean(r["EVENT"]) in event_by_name]
        event_ids = {event_by_name[p.clean(r["EVENT"])].event_id for r in placed}
        if not placed:
            names = sorted({p.clean(r["EVENT"]) for r in fight_rows})
            issues.append(
                _error(
                    "orphan_event",
                    f"{p.clean(fight_rows[0]['BOUT'])!r} names event {names} which is not in the "
                    "events file; bout dropped",
                    "bout",
                    fight_id,
                )
            )
            continue
        if len(event_ids) > 1:
            issues.append(
                _error(
                    "conflicting_event",
                    "fight is listed under several known events; bout dropped",
                    "bout",
                    fight_id,
                )
            )
            continue
        relisted += len(fight_rows) > 1
        bout = _parse_bout(fight_id, placed[0], event_by_name, resolver, issues)
        if bout is not None:
            pending.append(bout)

    if relisted:
        issues.append(
            _info(
                "duplicate_fight_rows",
                f"{relisted} fights are listed more than once (an event was renamed upstream); "
                "kept the row whose event is in the events file",
            )
        )
    return pending, len(rows_by_fight)


def _parse_bout(
    fight_id: str,
    row: Row,
    event_by_name: dict[str, Event],
    resolver: FighterResolver,
    issues: list[Issue],
) -> _PendingBout | None:
    event = event_by_name[p.clean(row["EVENT"])]
    try:
        name_a, name_b = p.split_bout(row["BOUT"])
        result, winner_slot = p.parse_outcome(row["OUTCOME"])
        end_round = p.parse_round_number(row["ROUND"])
        end_time_s = p.parse_clock(row["TIME"])
    except p.ParseError as exc:
        issues.append(
            _error("unparseable_fight", f"{exc}; bout dropped", "bout", fight_id)
        )
        return None

    bout_type = p.parse_bout_type(row["WEIGHTCLASS"])
    fighter_ids: list[str] = []
    for name in (name_a, name_b):
        resolution = resolver.resolve(name, fight_id, bout_type.weight_class)
        if resolution.fighter_id is None:
            reason = (
                f"shared by fighters {list(resolution.candidates)} and the weight class does not "
                "single one out"
                if resolution.method == METHOD_AMBIGUOUS
                else "matches no fighter profile"
            )
            issues.append(
                _error(
                    "unresolved_fighter",
                    f"{name!r} in {p.clean(row['BOUT'])!r} ({event.name}, {event.date}) {reason}; "
                    "bout dropped",
                    "bout",
                    fight_id,
                )
            )
            return None
        if resolution.method != METHOD_NAME:
            issues.append(
                _info(
                    "fighter_resolution",
                    f"{name!r} resolved to {resolution.fighter_id} by {resolution.method}",
                    "bout",
                    fight_id,
                )
            )
        fighter_ids.append(resolution.fighter_id)

    fighter_a_id, fighter_b_id = fighter_ids
    if fighter_a_id == fighter_b_id:
        issues.append(
            _error(
                "unresolved_fighter",
                f"both names in {p.clean(row['BOUT'])!r} resolve to one fighter; bout dropped",
                "bout",
                fight_id,
            )
        )
        return None

    method = p.clean(row["METHOD"]) or None
    if method not in KNOWN_METHODS:
        issues.append(
            _warning(
                "unknown_method", f"unrecognised method {method!r}", "bout", fight_id
            )
        )

    time_format_raw = p.clean(row["TIME FORMAT"]) or None
    scheduled_rounds = total_time = None
    try:
        time_format = p.parse_time_format(row["TIME FORMAT"])
    except p.ParseError as exc:
        issues.append(_warning("bad_time_format", str(exc), "bout", fight_id))
    else:
        scheduled_rounds = time_format.scheduled_rounds
        total_time = p.fight_time_s(time_format, end_round, end_time_s)
        if total_time is None:
            issues.append(
                _warning(
                    "bad_time_format",
                    f"fight time unknown: ended in round {end_round} of {time_format_raw!r}",
                    "bout",
                    fight_id,
                )
            )

    winner_id = {"a": fighter_a_id, "b": fighter_b_id, None: None}[winner_slot]
    bout = Bout(
        bout_id=fight_id,
        event_id=event.event_id,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        weight_class=bout_type.weight_class,
        bout_type_raw=p.clean(row["WEIGHTCLASS"]),
        is_womens=bout_type.is_womens,
        is_title_bout=bout_type.is_title_bout,
        is_interim_title=bout_type.is_interim_title,
        is_tournament_final=bout_type.is_tournament_final,
        scheduled_rounds=scheduled_rounds,
        time_format=time_format_raw,
        status="completed",
        result=result,
        winner_id=winner_id,
        method=method,
        method_details=p.clean(row["DETAILS"]) or None,
        referee=p.clean(row["REFEREE"]) or None,
        end_round=end_round,
        end_time_s=end_time_s,
        fight_time_s=total_time,
        has_stats=False,
    )
    return _PendingBout(bout, name_a, name_b, (event.name, p.clean(row["BOUT"])))


def _check_fight_details(
    detail_rows: list[Row], result_rows: list[Row], issues: list[Issue]
):
    """The fight list and the fight results should describe the same fights."""

    def ids(rows: list[Row]) -> set[str]:
        found = set()
        for row in rows:
            try:
                found.add(p.id_from_url(row["URL"], "fight"))
            except p.ParseError:
                continue
        return found

    listed, with_results = ids(detail_rows), ids(result_rows)
    if listed != with_results:
        issues.append(
            _warning(
                "fight_files_disagree",
                f"{len(listed - with_results)} fights are listed without a result and "
                f"{len(with_results - listed)} have a result without being listed",
            )
        )


def parse_stat_line(row: Row) -> StatLine:
    """Parse one stats row (one fighter, one round) into counts."""
    sig = p.parse_landed_attempted(row["SIG.STR."])
    total = p.parse_landed_attempted(row["TOTAL STR."])
    takedowns = p.parse_landed_attempted(row["TD"])
    head = p.parse_landed_attempted(row["HEAD"])
    body = p.parse_landed_attempted(row["BODY"])
    leg = p.parse_landed_attempted(row["LEG"])
    distance = p.parse_landed_attempted(row["DISTANCE"])
    clinch = p.parse_landed_attempted(row["CLINCH"])
    ground = p.parse_landed_attempted(row["GROUND"])
    return StatLine(
        knockdowns=p.parse_count(row["KD"]),
        sig_str_landed=sig[0],
        sig_str_attempted=sig[1],
        total_str_landed=total[0],
        total_str_attempted=total[1],
        td_landed=takedowns[0],
        td_attempted=takedowns[1],
        sub_attempts=p.parse_count(row["SUB.ATT"]),
        reversals=p.parse_count(row["REV."]),
        ctrl_s=p.parse_control_time(row["CTRL"]),
        head_landed=head[0],
        head_attempted=head[1],
        body_landed=body[0],
        body_attempted=body[1],
        leg_landed=leg[0],
        leg_attempted=leg[1],
        distance_landed=distance[0],
        distance_attempted=distance[1],
        clinch_landed=clinch[0],
        clinch_attempted=clinch[1],
        ground_landed=ground[0],
        ground_attempted=ground[1],
    )


def _parse_bout_rounds(
    pending: _PendingBout, rows: list[Row]
) -> dict[int, dict[str, StatLine]] | str:
    """Return ``{round: {fighter_id: line}}`` or a description of what is wrong."""
    slot_ids = {
        pending.name_a: pending.bout.fighter_a_id,
        pending.name_b: pending.bout.fighter_b_id,
    }
    if len(slot_ids) != 2:
        return "both fighters have the same name, so stats rows cannot be told apart"
    rounds: dict[int, dict[str, StatLine]] = defaultdict(dict)
    for row in rows:
        try:
            round_number = p.parse_round_label(row["ROUND"])
            line = parse_stat_line(row)
        except p.ParseError as exc:
            return str(exc)
        fighter_id = slot_ids.get(p.clean(row["FIGHTER"]))
        if fighter_id is None:
            return (
                f"stats row names {p.clean(row['FIGHTER'])!r}, who is not in the bout"
            )
        if fighter_id in rounds[round_number]:
            return f"round {round_number} has two rows for one fighter"
        rounds[round_number][fighter_id] = line
    if sorted(rounds) != list(range(1, len(rounds) + 1)):
        return f"rounds are not consecutive from 1: {sorted(rounds)}"
    if any(len(lines) != 2 for lines in rounds.values()):
        return "a round is missing one fighter's row"
    return dict(rounds)


def _attach_stats(
    pending_bouts: list[_PendingBout], stat_rows: list[Row], issues: list[Issue]
) -> tuple[list[Bout], list[RoundStats], list[FightStats]]:
    """Join per-round stats to bouts on (event name, bout text) and total them."""
    rows_by_key: dict[StatsKey, list[Row]] = defaultdict(list)
    for row in stat_rows:
        rows_by_key[(p.clean(row["EVENT"]), p.clean(row["BOUT"]))].append(row)
    bouts_per_key = Counter(pending.stats_key for pending in pending_bouts)

    bouts: list[Bout] = []
    round_stats: list[RoundStats] = []
    fight_stats: list[FightStats] = []
    without_stats = 0
    used_keys: set[StatsKey] = set()

    for pending in pending_bouts:
        bout = pending.bout
        used_keys.add(pending.stats_key)
        if bouts_per_key[pending.stats_key] > 1:
            issues.append(
                _warning(
                    "ambiguous_stats",
                    f"{pending.stats_key[1]!r} was fought more than once at {pending.stats_key[0]}; "
                    "stats rows cannot be told apart, so none were loaded",
                    "bout",
                    bout.bout_id,
                )
            )
            bouts.append(bout)
            continue

        # The dataset writes an empty placeholder row per fighter when a fight has no stats.
        rows = [
            r for r in rows_by_key.get(pending.stats_key, []) if p.clean(r["FIGHTER"])
        ]
        if not rows:
            without_stats += 1
            bouts.append(bout)
            continue

        parsed = _parse_bout_rounds(pending, rows)
        if isinstance(parsed, str):
            issues.append(
                _warning(
                    "malformed_stats",
                    f"{parsed}; no stats loaded",
                    "bout",
                    bout.bout_id,
                )
            )
            bouts.append(bout)
            continue

        if len(parsed) != bout.end_round:
            issues.append(
                _warning(
                    "stats_round_mismatch",
                    f"stats cover {len(parsed)} rounds but the bout ended in round "
                    f"{bout.end_round}",
                    "bout",
                    bout.bout_id,
                )
            )
        bouts.append(replace(bout, has_stats=True))
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id):
            lines = [parsed[n][fighter_id] for n in sorted(parsed)]
            round_stats.extend(
                RoundStats(bout.bout_id, fighter_id, n, line)
                for n, line in enumerate(lines, 1)
            )
            fight_stats.append(
                FightStats(bout.bout_id, fighter_id, sum_stat_lines(lines))
            )

    if without_stats:
        issues.append(
            _info(
                "no_stats",
                f"{without_stats} bouts have no per-fight stats in the source",
            )
        )
    unused_rows = sum(
        len(rows) for key, rows in rows_by_key.items() if key not in used_keys
    )
    if unused_rows:
        issues.append(
            _info(
                "unused_stats_rows",
                f"{unused_rows} stats rows belong to no loaded bout (duplicate listings under a "
                "renamed event, or dropped bouts)",
            )
        )
    return bouts, round_stats, fight_stats


def _infer_womens_bouts(bouts: list[Bout], issues: list[Issue]) -> list[Bout]:
    """Mark catch-weight bouts between women as women's bouts.

    The source says "Women's" only as part of a weight class name, so a
    women's catch-weight bout is not labelled. A fighter who appears in any
    labelled women's bout is taken to be a woman. The same reasoning flags a
    fighter who appears in both a women's and a men's weight class, which
    would mean a name was resolved to the wrong person.
    """
    women = {
        fighter_id
        for bout in bouts
        if bout.is_womens
        for fighter_id in (bout.fighter_a_id, bout.fighter_b_id)
    }
    result: list[Bout] = []
    relabelled = 0
    for bout in bouts:
        if not bout.is_womens and {bout.fighter_a_id, bout.fighter_b_id} & women:
            if bout.weight_class in _NO_CLASS_LIMIT:
                bout = replace(bout, is_womens=True)
                relabelled += 1
            else:
                issues.append(
                    _warning(
                        "mixed_division",
                        f"{bout.weight_class} bout includes a fighter who also appears in a "
                        "women's bout; a name may be resolved to the wrong fighter",
                        "bout",
                        bout.bout_id,
                    )
                )
        result.append(bout)
    if relabelled:
        issues.append(
            _info(
                "womens_bout_inferred",
                f"{relabelled} bouts with no gendered weight class were marked as women's bouts "
                "because a fighter in them also fought in a women's weight class",
            )
        )
    return result
