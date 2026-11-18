"""The read-only commands: show and matchup.

They run on the invented league and cards of the refresh tests. What matters
most is that neither changes the database or data/publish/.
"""

# ruff: noqa: F811, F401
import hashlib
from datetime import date

import pytest

from tests.test_predict_refresh import (
    TODAY,
    league,
    model_root,
    names,
    run,
    world,
)
from ufc.predict import __main__ as cli
from ufc.predict import show
from ufc.predict.show import ShowError


def tree_hash(root):
    """Every file under ``root`` (bytes and names); empty if it is absent."""
    digest = hashlib.sha256()
    if root.exists():
        for path in sorted(root.rglob("*")):
            digest.update(str(path.relative_to(root)).encode())
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


def state(world):
    return (
        hashlib.sha256(world["db"].read_bytes()).hexdigest(),
        tree_hash(world["publish"]),
    )


@pytest.fixture
def refreshed(world):
    result, _ = run(world)
    assert result.ok
    return world


def test_show_prints_the_next_card_and_changes_nothing(refreshed):
    before = state(refreshed)
    conn = show.connect_readonly(refreshed["db"])
    try:
        lines = show.show(conn, TODAY, None, False, 60)
    finally:
        conn.close()
    text = "\n".join(lines)
    assert "UFC Test 1, 2026-11-21" in text
    assert "UFC Test 2" not in text
    assert "Why: The model favours" in text
    assert "UNPREDICTED" in text and "placeholder" in text
    assert "limited data: debut: Newcomer Person" in text
    assert [line for line in lines if len(line) > 60] == []
    assert state(refreshed) == before


def test_show_selects_an_event_or_all(refreshed):
    conn = show.connect_readonly(refreshed["db"])
    try:
        second = "\n".join(show.show(conn, TODAY, "test 2", False, 80))
        everything = "\n".join(show.show(conn, TODAY, None, True, 80))
        with pytest.raises(ShowError, match="No stored event"):
            show.show(conn, TODAY, "UFC Nowhere", False, 80)
        with pytest.raises(ShowError, match="more than one"):
            show.show(conn, TODAY, "UFC Test", False, 80)
    finally:
        conn.close()
    assert "UFC Test 2" in second and "UFC Test 1" not in second
    assert "UFC Test 1" in everything and "UFC Test 2" in everything


def test_show_cli_is_read_only(refreshed, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_today", lambda: TODAY)
    before = state(refreshed)
    assert cli.main(["show", "--db", str(refreshed["db"]), "--all"]) == 0
    assert "UFC Test 1" in capsys.readouterr().out
    assert state(refreshed) == before


def test_show_without_a_database_fails_clearly(tmp_path, capsys):
    assert cli.main(["show", "--db", str(tmp_path / "none.db")]) == 1
    assert "No database" in capsys.readouterr().err
    assert not (tmp_path / "none.db").exists()


def shown_refs(conn, a, b):
    return tuple(
        conn.execute("SELECT fighter_id FROM fighters WHERE name = ?", (n,)).fetchone()[
            0
        ]
        for n in (a, b)
    )


def predict(world, a, b, **kwargs):
    conn = show.connect_readonly(world["db"])
    try:
        return show.predict_matchup(
            conn,
            world["resolver"],
            world["models"],
            a,
            b,
            kwargs.pop("on", TODAY),
            kwargs.pop("weight_class", None),
        )
    finally:
        conn.close()


def test_matchup_predicts_labels_and_changes_nothing(refreshed, names):
    f = names["f"]
    before = state(refreshed)
    h = predict(refreshed, f[1], f[2])
    assert h.p_a + h.p_b == pytest.approx(1.0)
    conn = show.connect_readonly(refreshed["db"])
    try:
        shared = show.shared_division(
            show.load_feature_data(conn), *shown_refs(conn, f[1], f[2]), TODAY
        )
    finally:
        conn.close()
    assert h.weight_class == shared is not None
    text = "\n".join(show.format_hypothetical(h, 70))
    assert "HYPOTHETICAL, NOT STORED" in text and "Why:" in text
    assert state(refreshed) == before
    rows = show.connect_readonly(refreshed["db"])
    try:
        count = rows.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        rows.close()
    assert count == 6


def test_matchup_equals_the_stored_prediction_for_a_scheduled_bout(refreshed, names):
    """Same inputs, same model: a hypothetical of a stored bout reproduces it."""
    f = names["f"]
    conn = show.connect_readonly(refreshed["db"])
    try:
        stored = conn.execute(
            "SELECT p_fighter_a FROM predictions WHERE superseded_at IS NULL "
            "AND event_date = '2026-12-05' AND fighter_a_ref = ("
            "SELECT fighter_id FROM fighters WHERE name = ?)",
            (f[9],),
        ).fetchone()
    finally:
        conn.close()
    h = predict(
        refreshed, f[9], f[10], on=date(2026, 12, 5), weight_class="Lightweight"
    )
    assert h.p_a == pytest.approx(stored[0])


def test_matchup_is_symmetric_in_its_names(refreshed, names):
    f = names["f"]
    ab = predict(refreshed, f[3], f[4])
    ba = predict(refreshed, f[4], f[3])
    assert ab.p_a == pytest.approx(ba.p_b)


def test_matchup_errors_are_clear(refreshed, names):
    f = names["f"]
    with pytest.raises(ShowError, match="was not found"):
        predict(refreshed, f[1], "Nobody Atall")
    with pytest.raises(ShowError, match="same fighter"):
        predict(refreshed, f[1], f[1])
    with pytest.raises(ShowError, match="Unknown weight class"):
        predict(refreshed, f[1], f[2], weight_class="Strawweightish")


def test_matchup_reports_ambiguous_names():
    from ufc.ingest.identity import FighterResolver, Profile

    # Two profiles sharing a name, both at 155 lb: the class cannot settle it.
    twin = FighterResolver(
        [
            Profile("aaaaaaaaaaaaaaa1", ("Twin Name",), 155),
            Profile("aaaaaaaaaaaaaaa2", ("Twin Name",), 155),
        ]
    )
    with pytest.raises(ShowError, match="ambiguous"):
        show.resolve_name(twin, set(), "Twin Name", "Lightweight")


def test_matchup_without_an_approved_model_fails(refreshed, names, tmp_path):
    f = names["f"]
    conn = show.connect_readonly(refreshed["db"])
    try:
        with pytest.raises(ShowError, match="No approved model"):
            show.predict_matchup(
                conn, refreshed["resolver"], tmp_path, f[1], f[2], TODAY, None
            )
    finally:
        conn.close()


def test_matchup_cli_is_read_only(refreshed, names, monkeypatch, tmp_path, capsys):
    f = names["f"]
    snapshot = tmp_path / "greco"
    snapshot.mkdir()
    (snapshot / cli.source.FIGHTER_TOTT_FILE).write_text("")
    monkeypatch.setattr(cli, "_today", lambda: TODAY)
    monkeypatch.setattr(cli, "resolver_from_snapshot", lambda *_: refreshed["resolver"])
    before = state(refreshed)
    argv = [
        "matchup",
        f[1],
        f[2],
        "--db",
        str(refreshed["db"]),
        "--snapshot",
        str(snapshot),
        "--models",
        str(refreshed["models"]),
    ]
    assert cli.main(argv) == 0
    assert "HYPOTHETICAL, NOT STORED" in capsys.readouterr().out
    assert cli.main([argv[0], argv[1], "Nobody Atall", *argv[3:]]) == 1
    assert "was not found" in capsys.readouterr().err
    assert cli.main([*argv, "--date", "17/11/2026"]) == 1
    assert state(refreshed) == before
