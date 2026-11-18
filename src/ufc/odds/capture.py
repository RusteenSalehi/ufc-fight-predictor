"""One snapshot of MMA moneylines from The Odds API, stored untouched.

The key travels in the ``x-api-key`` header (documented by the API), never in a URL, and every
piece of error text passes through :func:`redact` before it can be logged or printed.
Nothing here writes the database or matches bouts to ours.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "mma_mixed_martial_arts"
KEY_ENV = "ODDS_API_KEY"
MIN_REMAINING = 50
TIMEOUT = (10, 30)
USER_AGENT = "ufc-fight-predictor-odds-capture (portfolio project)"
DEFAULT_DIR = Path("data/odds/raw")
CREDIT_HEADERS = ("x-requests-used", "x-requests-remaining", "x-requests-last")

# A transport takes (url, params, headers) and returns a requests.Response-like object.
Transport = Callable[[str, Mapping[str, str], Mapping[str, str]], requests.Response]


class OddsError(RuntimeError):
    """A capture failure whose message is already free of the API key."""


def load_key(env_file: Path = Path(".env")) -> str:
    """Return the key from the environment, else from ``.env``. Never logs it."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key and env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == KEY_ENV:
                key = value.strip().strip("'\"")
    if not key:
        raise OddsError(f"{KEY_ENV} is not set (environment or .env)")
    return key


def redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def http_transport(url: str, params: Mapping[str, str], headers: Mapping[str, str]):
    return requests.get(url, params=params, headers=headers, timeout=TIMEOUT)


def _credits(headers: Mapping[str, str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for name in CREDIT_HEADERS:
        raw = headers.get(name)
        try:
            out[name] = int(raw) if raw is not None else None
        except ValueError:
            out[name] = None
    return out


def _get(transport: Transport, key: str, path: str, params: Mapping[str, str]):
    try:
        return transport(
            f"{BASE_URL}{path}", params, {"x-api-key": key, "User-Agent": USER_AGENT}
        )
    except requests.RequestException as exc:
        raise OddsError(
            f"request to {path} failed: {redact(type(exc).__name__ + ': ' + str(exc), key)}"
        ) from None


def remaining_credits(transport: Transport, key: str) -> int:
    """Read the quota from the free ``/sports`` endpoint (it does not consume credits)."""
    resp = _get(transport, key, "/sports", {})
    if resp.status_code != 200:
        raise OddsError(
            f"/sports answered HTTP {resp.status_code}; not calling the odds endpoint"
        )
    remaining = _credits(resp.headers)["x-requests-remaining"]
    if remaining is None:
        raise OddsError(
            "no x-requests-remaining header on /sports; refusing to spend credits blind"
        )
    return remaining


@dataclass(frozen=True)
class Snapshot:
    raw_path: Path
    meta_path: Path
    meta: dict
    body: bytes


def take_snapshot(
    key: str,
    out_dir: Path = DEFAULT_DIR,
    regions: str = "us,eu",
    transport: Transport | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    min_remaining: int = MIN_REMAINING,
) -> Snapshot:
    transport = transport or http_transport
    before = remaining_credits(transport, key)
    if before < min_remaining:
        raise OddsError(
            f"only {before} credits remain (minimum {min_remaining}); not calling"
        )
    params = {
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    requested_at = now()
    resp = _get(transport, key, f"/sports/{SPORT_KEY}/odds", params)
    received_at = now()
    if resp.status_code != 200:
        snippet = redact(resp.text[:200], key)
        raise OddsError(f"odds endpoint answered HTTP {resp.status_code}: {snippet}")
    body = resp.content
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = requested_at.strftime("%Y%m%dT%H%M%SZ")
    raw_path = out_dir / f"{stamp}.json"
    meta_path = out_dir / f"{stamp}.meta.json"
    if raw_path.exists():
        raise OddsError(f"{raw_path} already exists; not overwriting")
    meta = {
        "requested_at": requested_at.isoformat(),
        "received_at": received_at.isoformat(),
        "endpoint": f"/v4/sports/{SPORT_KEY}/odds",
        "params": params,
        "http_status": resp.status_code,
        "credits_remaining_before": before,
        "credit_headers": _credits(resp.headers),
        "raw_sha256": hashlib.sha256(body).hexdigest(),
        "raw_bytes": len(body),
    }
    raw_path.write_bytes(body)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return Snapshot(raw_path, meta_path, meta, body)


def summarize(body: bytes) -> dict:
    """Counts for a human to read. Parses the stored bytes; the file itself is never rewritten."""
    events = json.loads(body)
    books: Counter[str] = Counter()
    times: Counter[str] = Counter()
    with_ml = 0
    for ev in events:
        has = False
        for bm in ev.get("bookmakers", []):
            if any(
                m.get("key") == "h2h" and m.get("outcomes")
                for m in bm.get("markets", [])
            ):
                books[bm["key"]] += 1
                has = True
        with_ml += has
        times[ev.get("commence_time", "?")] += 1
    return {
        "events": len(events),
        "with_moneyline": with_ml,
        "bookmakers": dict(books.most_common()),
        "commence_times": dict(sorted(times.items())),
    }
