"""Odds capture: raw file stored unchanged, credit guard, key never leaks. No network."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
import requests

from ufc.odds import __main__ as cli
from ufc.odds import capture

FIXTURE = Path(__file__).parent / "fixtures" / "odds" / "mma_h2h.json"
KEY = "fake-test-key-value"


def _response(status: int, body: bytes, headers: dict[str, str]) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = body
    r.headers.update(headers)
    return r


class FakeTransport:
    def __init__(
        self,
        remaining: str | None = "500",
        status: int = 200,
        body: bytes | None = None,
    ):
        self.remaining = remaining
        self.status = status
        self.body = FIXTURE.read_bytes() if body is None else body
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params), dict(headers)))
        credit = {"x-requests-used": "2", "x-requests-last": "2"}
        if self.remaining is not None:
            credit["x-requests-remaining"] = self.remaining
        if url.endswith("/sports"):
            return _response(200, b"[]", credit)
        return _response(self.status, self.body, credit)


def test_raw_response_stored_byte_for_byte(tmp_path):
    t = FakeTransport(
        body=FIXTURE.read_bytes() + b"\n  \n"
    )  # odd trailing bytes survive
    snap = capture.take_snapshot(KEY, tmp_path, transport=t)
    assert snap.raw_path.read_bytes() == t.body
    assert snap.raw_path.name.endswith("Z.json") and snap.raw_path.parent == tmp_path
    meta = json.loads(snap.meta_path.read_text())
    assert meta["raw_sha256"] == hashlib.sha256(t.body).hexdigest()
    assert meta["credit_headers"] == {
        "x-requests-used": 2,
        "x-requests-remaining": 500,
        "x-requests-last": 2,
    }
    assert meta["requested_at"] and meta["http_status"] == 200


def test_exactly_one_odds_call_and_free_quota_check(tmp_path):
    t = FakeTransport()
    capture.take_snapshot(KEY, tmp_path, transport=t)
    urls = [c[0] for c in t.calls]
    assert [u.rsplit("/v4", 1)[1] for u in urls] == [
        "/sports",
        "/sports/mma_mixed_martial_arts/odds",
    ]
    assert t.calls[1][1]["markets"] == "h2h"


def test_refuses_below_minimum_credits(tmp_path):
    t = FakeTransport(remaining="49")
    with pytest.raises(capture.OddsError, match="49"):
        capture.take_snapshot(KEY, tmp_path, transport=t)
    assert len(t.calls) == 1 and not list(tmp_path.iterdir())


def test_allows_exactly_the_minimum(tmp_path):
    capture.take_snapshot(KEY, tmp_path, transport=FakeTransport(remaining="50"))


def test_refuses_when_quota_header_missing(tmp_path):
    t = FakeTransport(remaining=None)
    with pytest.raises(capture.OddsError, match="header"):
        capture.take_snapshot(KEY, tmp_path, transport=t)
    assert len(t.calls) == 1


def test_error_status_stores_nothing_and_redacts(tmp_path):
    t = FakeTransport(status=401, body=f"bad key {KEY}".encode())
    with pytest.raises(capture.OddsError) as err:
        capture.take_snapshot(KEY, tmp_path, transport=t)
    assert KEY not in str(err.value) and "401" in str(err.value)
    assert not list(tmp_path.iterdir())


def test_key_only_in_header_never_in_url_or_params(tmp_path):
    t = FakeTransport()
    snap = capture.take_snapshot(KEY, tmp_path, transport=t)
    for url, params, headers in t.calls:
        assert KEY not in url and KEY not in json.dumps(params)
        assert headers["x-api-key"] == KEY
    assert (
        KEY not in snap.meta_path.read_text()
        and KEY.encode() not in snap.raw_path.read_bytes()
    )


def test_transport_exception_is_redacted():
    def boom(url, params, headers):
        raise requests.ConnectionError(f"failed for {url}?apiKey={KEY}")

    with pytest.raises(capture.OddsError) as err:
        capture.take_snapshot(KEY, Path("unused"), transport=boom)
    assert KEY not in str(err.value) and err.value.__cause__ is None


def test_cli_output_and_logs_never_contain_key(tmp_path, monkeypatch, capsys, caplog):
    monkeypatch.setenv(capture.KEY_ENV, KEY)
    monkeypatch.setattr(capture, "http_transport", FakeTransport())
    with caplog.at_level(logging.DEBUG):
        assert cli.main(["snapshot", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert KEY not in out.out + out.err + caplog.text
    assert "Events: 3; with a moneyline: 2" in out.out and "book_one 2" in out.out

    monkeypatch.setattr(capture, "http_transport", FakeTransport(remaining="3"))
    assert cli.main(["snapshot", "--out", str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "Not captured" in out.err


def test_load_key_env_beats_file_and_missing_is_error(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"OTHER=1\n{capture.KEY_ENV}='from-file'\n")
    monkeypatch.delenv(capture.KEY_ENV, raising=False)
    assert capture.load_key(env) == "from-file"
    monkeypatch.setenv(capture.KEY_ENV, "from-env")
    assert capture.load_key(env) == "from-env"
    monkeypatch.delenv(capture.KEY_ENV)
    with pytest.raises(capture.OddsError, match=capture.KEY_ENV):
        capture.load_key(tmp_path / "missing")


def test_summarize_counts_fixture():
    s = capture.summarize(FIXTURE.read_bytes())
    assert s["events"] == 3 and s["with_moneyline"] == 2
    assert s["bookmakers"] == {"book_one": 2, "book_two": 1}
    assert s["commence_times"]["2030-01-05T19:00:00Z"] == 1
