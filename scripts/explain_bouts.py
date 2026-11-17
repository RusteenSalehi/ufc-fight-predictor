"""Explain stored predictions for upcoming bouts, one bout at a time.

    uv run python scripts/explain_bouts.py wb9bc38e83195eeeff [more bout IDs]
    uv run python scripts/explain_bouts.py --event-date 2026-11-21   # every bout

Analysis only: it reads data/ufc.db and the model registry and writes
nothing. For each bout it prints the fighters' resolution and UFC history,
the as-of inputs the model sees, the ten largest SHAP contributions and the
probabilities of the approved model and of the candidate it was refitted
from.

Inputs are rebuilt exactly as ``python -m ufc.predict refresh`` builds them:
the model's own feature pipeline, upstream attributes with Wikipedia
backfill in the gaps, ``build_matchup`` at the event date. Contributions are
TreeSHAP from XGBoost (``pred_contribs``), symmetrised over both
orientations, ``(phi(A, B) - phi(B, A)) / 2``, so that they add up to the
symmetric log-odds the model uses before calibration and the bias cancels.
They describe what moved the model, not causes. This is the M6 SHAP method
in draft; the published factors will come from ``src/ufc/predict/``.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb

from ufc.features.data import load_feature_data
from ufc.features.matchup import build_matchup
from ufc.models import registry
from ufc.models.design import matrix, swapped
from ufc.predict.backfill import merged_attributes
from ufc.predict.predict import feature_store, model_inputs

ROOT = Path("data/models")
DB = Path("data/ufc.db")

WINDOW = {
    "career": "career",
    "l3": "last 3",
    "l5": "last 5",
    "decay": "recency-weighted",
}
BASE = {
    "win_rate": "win rate",
    "slpm": "sig. strikes landed/min",
    "sapm": "sig. strikes absorbed/min",
    "sig_acc": "sig. strike accuracy",
    "sig_def": "sig. strike defence",
    "sig_att_pm": "sig. strikes attempted/min",
    "kd_per15": "knockdowns scored/15min",
    "td_per15": "takedowns/15min",
    "td_acc": "takedown accuracy",
    "td_def": "takedown defence",
    "sub_att_per15": "sub attempts/15min",
    "ctrl_share": "control-time share",
    "ctrl_against_share": "share of time controlled by opponent",
    "kd_abs_per15": "knockdowns absorbed/15min",
    "head_abs_pm": "head strikes absorbed/min",
    "opp_elo": "mean opponent Elo",
    "finish_win_share": "share of wins by finish",
    "head_share": "share of strikes to head",
    "body_share": "share of strikes to body",
    "leg_share": "share of strikes to legs",
    "distance_share": "share of strikes at distance",
    "clinch_share": "share of strikes in clinch",
    "ground_share": "share of strikes on ground",
    "n_fights": "UFC fights",
    "wins": "UFC wins",
    "losses": "UFC losses",
    "ko_wins": "KO/TKO wins",
    "sub_wins": "submission wins",
    "title_fights": "title fights",
    "title_wins": "title fight wins",
    "cage_min": "total cage minutes",
    "ko_losses": "KO/TKO losses",
    "sub_losses": "submission losses",
    "age": "age",
    "height_cm": "height",
    "reach_cm": "reach",
    "stance_southpaw": "southpaw stance",
    "stance_switch": "switch stance",
    "win_streak": "win streak",
    "loss_streak": "losing streak",
    "ko_losses_l3": "KO/TKO losses in last 3",
    "ko_losses_l5": "KO/TKO losses in last 5",
    "kd_abs": "knockdowns absorbed (count)",
    "days_since_last": "days since last fight",
    "fights_last_2y": "fights in last 2 years",
    "moved_up": "moving up a weight class",
    "moved_down": "moving down a weight class",
    "fights_at_weight": "fights at this weight class",
    "elo": "Elo rating",
    "debut": "UFC debut",
    "cardio_fights": "fights with cardio data",
    "cardio_sig_acc_delta": "late-round accuracy change",
    "cardio_sig_def_delta": "late-round defence change",
    "stance_southpaw_vs_orthodox": "southpaw-vs-orthodox edge",
    "matchup_td": "takedowns vs opponent's TD defence",
    "matchup_strike": "strike volume vs opponent's defence",
    "matchup_kd": "knockdown power vs opponent's KD absorption",
    "matchup_ctrl": "control vs opponent's tendency to be controlled",
    "matchup_sub": "sub attempts vs opponent's sub-loss rate",
}


def label(name: str) -> str:
    if name.startswith("lvl_"):
        return f"bout-level average {BASE.get(name[4:], name[4:])}"
    n = name[2:]
    adj = n.startswith("adj_")
    if adj:
        n = n[4:]
    if n.startswith("cardio_") and n.endswith(("_career", "_decay")):
        stem, _, win = n.rpartition("_")
        stat = stem[len("cardio_") :].removesuffix("_ratio")
        return f"late/early-round ratio of {BASE.get(stat, stat)} ({WINDOW[win]})"
    stem, _, win = n.rpartition("_")
    if win in WINDOW and stem in BASE:
        s = f"{BASE[stem]} ({WINDOW[win]})"
    else:
        s = BASE.get(n, n)
    return ("opp-adjusted " + s) if adj else s


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def history(conn, fid):
    return conn.execute(
        """
        SELECT e.date, e.name AS event, b.weight_class, b.result, b.winner_id,
               b.method, b.end_round, b.is_title_bout,
               CASE WHEN b.fighter_a_id = ? THEN fb.name ELSE fa.name END AS opp
        FROM bouts b JOIN events e USING (event_id)
        JOIN fighters fa ON fa.fighter_id = b.fighter_a_id
        JOIN fighters fb ON fb.fighter_id = b.fighter_b_id
        WHERE (b.fighter_a_id = ? OR b.fighter_b_id = ?) AND b.status = 'completed'
        ORDER BY e.date
        """,
        (fid, fid, fid),
    ).fetchall()


def outcome(r, fid):
    if r["result"] == "win":
        return "W" if r["winner_id"] == fid else "L"
    return {"draw": "D", "nc": "NC"}.get(r["result"], r["result"] or "?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bouts", nargs="*", help="upcoming bout IDs (wb...)")
    parser.add_argument(
        "--event-date", type=date.fromisoformat, help="explain every bout on this date"
    )
    parser.add_argument("--model", help="registry version (default: approved.json)")
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--models", type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.bouts and args.event_date is None:
        parser.error("give bout IDs or --event-date")
    return args


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    data = load_feature_data(conn)
    merged, sources = merged_attributes(conn, data.fighters)

    primary = args.model or registry.approved_version(args.models)
    lineage = registry.read_manifest(args.models, primary).get("lineage") or {}
    parent = lineage.get("parent")
    versions = [primary] + ([parent] if parent else [])
    models = {}
    for v in versions:
        cand, _ = registry.load_candidate(args.models, v)
        pipe = registry.read_json(args.models, v, "feature_pipeline.json")
        models[v] = (cand, feature_store(data, pipe, merged))

    query = """
        SELECT b.*, e.date AS event_date
        FROM upcoming_bouts b JOIN upcoming_events e USING (event_key)
        WHERE {} ORDER BY e.date, b.card_order
    """
    if args.bouts:
        marks = ",".join("?" * len(args.bouts))
        rows = conn.execute(query.format(f"b.bout_id IN ({marks})"), args.bouts)
    else:
        rows = conn.execute(
            query.format("e.date = ? AND b.status = 'scheduled'"),
            (args.event_date.isoformat(),),
        )
    bouts = rows.fetchall()
    missing = set(args.bouts) - {b["bout_id"] for b in bouts}
    if missing:
        raise SystemExit(f"unknown bout IDs: {sorted(missing)}")

    for bout in bouts:
        bid = bout["bout_id"]
        on = date.fromisoformat(bout["event_date"])
        a_ref, b_ref, wc = (
            bout["fighter_a_ref"],
            bout["fighter_b_ref"],
            bout["weight_class"],
        )
        na, nb = bout["name_a"], bout["name_b"]
        if a_ref is None or b_ref is None:
            print(f"{na} vs {nb}: unresolved ({bout['unresolved_reason']}); skipped")
            continue
        print("=" * 100)
        print(f"{na}  vs  {nb}   ({wc}, {on})   bout {bid}")
        print("=" * 100)

        # 1. Resolution and history
        print("\n## 1. Resolution and UFC history (dataset)")
        for name, ref in ((na, a_ref), (nb, b_ref)):
            f = conn.execute(
                "SELECT * FROM fighters WHERE fighter_id = ?", (ref,)
            ).fetchone()
            prov = (
                conn.execute(
                    "SELECT * FROM provisional_fighters WHERE provisional_id = ?",
                    (ref,),
                ).fetchone()
                if f is None
                else None
            )
            ds_name = f["name"] if f else "(not in dataset)"
            print(
                f"\n{name}: id {ref} -> dataset name '{ds_name}'"
                + (
                    f", dob {f['dob']}, {fmt(f['height_cm'], 0)} cm, reach {fmt(f['reach_cm'], 0)} cm, {f['stance']}"
                    if f
                    else ""
                )
            )
            if prov is not None:
                print(f"  provisional: {dict(prov)}")
            if ref in sources:
                print(f"  backfilled from Wikipedia: {sources[ref]}")
            rows = history(conn, ref)
            print(f"  {len(rows)} UFC bouts")
            for r in rows:
                rnd = f" R{r['end_round']}" if r["end_round"] else ""
                title = " [title]" if r["is_title_bout"] else ""
                print(
                    f"  {r['date']}  {outcome(r, ref):2}  vs {r['opp']:<26} {r['method'] or '':<26}{rnd:<4} {r['weight_class'] or '?'}{title}"
                )

        # 2. As-of inputs (approved pipeline)
        cand, store = models[primary]
        m = build_matchup(store, a_ref, b_ref, on, wc)
        print(
            "\n## 2. As-of inputs (approved model's feature pipeline, bouts before the event date)"
        )
        rows = [
            ("age (years)", "age", 1),
            ("Elo", "elo", 0),
            ("mean opponent Elo (career)", "opp_elo_career", 0),
            ("mean opponent Elo (last 3)", "opp_elo_l3", 0),
            ("UFC fights", "n_fights", 0),
            ("wins", "wins", 0),
            ("losses", "losses", 0),
            ("win streak", "win_streak", 0),
            ("losing streak", "loss_streak", 0),
            ("KO/TKO losses", "ko_losses", 0),
            ("fights at this weight", "fights_at_weight", 0),
            ("moved up / down", None, 0),
            ("days since last fight", "days_since_last", 0),
            ("fights in last 2 years", "fights_last_2y", 0),
            ("height cm", "height_cm", 0),
            ("reach cm", "reach_cm", 0),
        ]
        print(f"  {'':32}{na:>22}{nb:>22}")
        for lab, key, nd in rows:
            if key is None:
                va = f"{fmt(m.a['moved_up'], 0)}/{fmt(m.a['moved_down'], 0)}"
                vb = f"{fmt(m.b['moved_up'], 0)}/{fmt(m.b['moved_down'], 0)}"
            else:
                va, vb = fmt(m.a[key], nd), fmt(m.b[key], nd)
            print(f"  {lab:32}{va:>22}{vb:>22}")
        for name, ref in ((na, a_ref), (nb, b_ref)):
            wcs = store.history(ref).weight_classes
            print(
                f"  weight classes, {name}: "
                + ", ".join(f"{k} x{v}" for k, v in Counter(wcs).items())
            )
        x = model_inputs(m, cand.spec.names)
        miss = [n for n in cand.spec.names if np.isnan(x[n].iloc[0])]
        print(f"  flags: {m.flags}")
        print(
            f"  model inputs missing (NaN): {len(miss)} of {len(cand.spec.names)}"
            + (": " + ", ".join(miss) if miss else "")
        )
        for side, vec, name in (("a", m.a, na), ("b", m.b, nb)):
            nan = [k for k, v in vec.items() if isinstance(v, float) and math.isnan(v)]
            if nan:
                print(f"  {name} per-fighter NaN values: {', '.join(nan)}")

        # 3. SHAP, symmetrized
        booster: xgb.Booster = cand.model._booster
        xa = matrix(x, cand.spec)
        xb = swapped(xa, cand.spec)
        ca = booster.predict(
            xgb.DMatrix(xa, feature_names=list(cand.spec.names)), pred_contribs=True
        )[0]
        cb = booster.predict(
            xgb.DMatrix(xb, feature_names=list(cand.spec.names)), pred_contribs=True
        )[0]
        sym = (ca - cb) / 2.0  # log-odds toward A; bias cancels
        u, v = cand.model.one_sided(xa)[0], cand.model.one_sided(xb)[0]
        assert abs(ca.sum() - logit(u)) < 1e-4 and abs(cb.sum() - logit(v)) < 1e-4
        names = list(cand.spec.names)
        order = np.argsort(-np.abs(sym[:-1]))
        print(
            f"\n## 3. Top 10 SHAP contributions (explained booster, log-odds, + favours {na}, − favours {nb})"
        )
        print(
            f"  one-sided outputs: f(A,B)={u:.4f}  f(B,A)={v:.4f};  symmetrized log-odds sum={sym[:-1].sum():+.3f}"
            f"  (bias {ca[-1]:+.4f} / {cb[-1]:+.4f} cancels)"
        )
        print(
            f"  {'#':>2} {'contrib':>8}  {'favours':<20} {'input value (A−B)':>18}  feature"
        )
        for i, j in enumerate(order[:10], 1):
            n = names[j]
            who = na if sym[j] > 0 else nb
            print(
                f"  {i:>2} {sym[j]:+8.3f}  {who:<20} {fmt(float(xa[0, j]), 3):>18}  {label(n)}  [{n}]"
            )
        rest = sym[:-1].sum() - sym[order[:10]].sum()
        print(f"     {rest:+8.3f}  (other {len(names) - 10} inputs combined)")

        # 4. Probabilities
        print("\n## 4. Probabilities (P(" + na + ") / P(" + nb + "))")
        for v_ in versions:
            c_, s_ = models[v_]
            mm = build_matchup(s_, a_ref, b_ref, on, wc)
            xx = model_inputs(mm, c_.spec.names)
            raw = c_.predict_pair(xx, calibrated=False)
            cal = c_.predict_pair(xx, calibrated=True)
            tag = "explained" if v_ == primary else "parent"
            print(
                f"  {tag:15} {v_}: raw {raw[0][0]:.3f} / {raw[1][0]:.3f}  ->  "
                f"calibrated (Platt slope {c_.calibrator.slope:.3f}) {cal[0][0]:.3f} / {cal[1][0]:.3f}"
            )
        stored = conn.execute(
            "SELECT p_fighter_a FROM predictions WHERE bout_id=? AND superseded_at IS NULL",
            (bid,),
        ).fetchone()
        print(
            "  stored current prediction: " + (f"{stored[0]:.3f}" if stored else "none")
        )
        print()


if __name__ == "__main__":
    main()
