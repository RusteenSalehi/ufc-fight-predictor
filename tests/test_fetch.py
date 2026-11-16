"""Snapshot download, with the network replaced by a scripted session."""

import json

import pytest
import requests

from ufc.ingest import fetch, source

COMMIT = "0123456789abcdef0123456789abcdef01234567"
COMMIT_DATE = "2026-11-15T18:03:06Z"


class FakeResponse:
    def __init__(self, status_code=200, content=b"", payload=None):
        self.status_code = status_code
        self.content = content
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Serves the fixture snapshot; ``script`` can override responses per URL, in order."""

    def __init__(self, fixture_dir, script=None):
        self.fixture_dir = fixture_dir
        self.script = script or {}
        self.calls: list[tuple[str, object]] = []

    def get(self, url, timeout=None):
        self.calls.append((url, timeout))
        for fragment, responses in self.script.items():
            if fragment in url and responses:
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
        if url.startswith("https://api.github.com/"):
            payload = {"sha": COMMIT, "commit": {"committer": {"date": COMMIT_DATE}}}
            return FakeResponse(payload=payload)
        file_name = url.rsplit("/", 1)[1]
        return FakeResponse(content=(self.fixture_dir / file_name).read_bytes())


def run(session, dest, sleeps=None):
    recorded = sleeps if sleeps is not None else []
    return fetch.fetch_snapshot(dest, session=session, sleep=recorded.append)


def test_fetch_writes_the_six_files_and_their_provenance(fixture_dir, tmp_path):
    dest = tmp_path / "external" / "greco"
    info = run(FakeSession(fixture_dir), dest)

    assert (info.repo, info.commit, info.commit_date) == (
        source.UPSTREAM_REPO,
        COMMIT,
        COMMIT_DATE,
    )
    for file_name in source.EXPECTED_COLUMNS:
        assert (dest / file_name).read_bytes() == (fixture_dir / file_name).read_bytes()
    described = json.loads((dest / source.SOURCE_INFO_FILE).read_text("utf-8"))
    assert described["commit"] == COMMIT
    assert described["files"][source.EVENTS_FILE]["sha256"] == source.sha256_of(
        dest / source.EVENTS_FILE
    )
    assert source.read_source_info(dest) == info
    assert not (dest.parent / ".greco.partial").exists()


def test_fetch_pins_every_file_to_the_resolved_commit_and_sets_timeouts(
    fixture_dir, tmp_path
):
    session = FakeSession(fixture_dir)
    run(session, tmp_path / "greco")
    urls = [url for url, _ in session.calls]
    assert (
        urls[0] == f"https://api.github.com/repos/{source.UPSTREAM_REPO}/commits/main"
    )
    assert len(urls) == 7
    assert all(f"/{COMMIT}/" in url for url in urls[1:])
    assert all(timeout == fetch.TIMEOUT_S for _, timeout in session.calls)


def test_fetched_snapshot_loads(fixture_dir, tmp_path):
    from ufc.ingest import pipeline

    dest = tmp_path / "greco"
    run(FakeSession(fixture_dir), dest)
    prepared = pipeline.prepare(dest)
    assert prepared.commit == COMMIT
    assert len(prepared.dataset.bouts) == 27


def test_transient_failures_are_retried_with_exponential_backoff(fixture_dir, tmp_path):
    script = {
        source.EVENTS_FILE: [
            FakeResponse(status_code=503),
            requests.ConnectionError("reset"),
            FakeResponse(status_code=429),
        ]
    }
    sleeps: list[float] = []
    run(FakeSession(fixture_dir, script), tmp_path / "greco", sleeps)
    backoff = [s for s in sleeps if s != fetch.PAUSE_BETWEEN_REQUESTS_S]
    assert backoff == [2.0, 4.0, 8.0]
    assert (tmp_path / "greco" / source.EVENTS_FILE).exists()


def test_gives_up_after_the_maximum_number_of_attempts(fixture_dir, tmp_path):
    script = {source.EVENTS_FILE: [FakeResponse(status_code=503)] * fetch.MAX_ATTEMPTS}
    session = FakeSession(fixture_dir, script)
    with pytest.raises(fetch.FetchError, match="HTTP 503"):
        run(session, tmp_path / "greco")
    assert (
        sum(source.EVENTS_FILE in url for url, _ in session.calls) == fetch.MAX_ATTEMPTS
    )


def test_a_missing_file_is_not_retried(fixture_dir, tmp_path):
    session = FakeSession(
        fixture_dir, {source.EVENTS_FILE: [FakeResponse(status_code=404)]}
    )
    with pytest.raises(fetch.FetchError, match="HTTP 404"):
        run(session, tmp_path / "greco")
    assert sum(source.EVENTS_FILE in url for url, _ in session.calls) == 1


@pytest.mark.parametrize(
    "bad_content",
    [
        b"NAME,URL,DATE,LOCATION\nx,y,z,w\n",
        b"EVENT,URL,DATE,LOCATION\n",
        b"",
        b"\xff\xfe\x00",
    ],
)
def test_a_malformed_download_leaves_the_previous_snapshot_in_place(
    fixture_dir, tmp_path, bad_content
):
    dest = tmp_path / "greco"
    run(FakeSession(fixture_dir), dest)
    before = {path.name: path.read_bytes() for path in dest.iterdir()}

    script = {source.FIGHT_STATS_FILE: [FakeResponse(content=bad_content)]}
    with pytest.raises(fetch.FetchError):
        run(FakeSession(fixture_dir, script), dest)
    assert {path.name: path.read_bytes() for path in dest.iterdir()} == before
    assert not (tmp_path / ".greco.partial").exists()


def test_unexpected_commit_description_is_an_error(fixture_dir, tmp_path):
    session = FakeSession(
        fixture_dir, {"api.github.com": [FakeResponse(payload={"message": "nope"})]}
    )
    with pytest.raises(fetch.FetchError, match="unexpected commit description"):
        run(session, tmp_path / "greco")


def test_user_agent_is_descriptive_and_overridable(monkeypatch):
    monkeypatch.delenv(fetch.USER_AGENT_ENV, raising=False)
    assert fetch.make_session().headers["User-Agent"] == fetch.DEFAULT_USER_AGENT
    monkeypatch.setenv(fetch.USER_AGENT_ENV, "custom-agent/1.0 (contact)")
    assert fetch.make_session().headers["User-Agent"] == "custom-agent/1.0 (contact)"


def test_cli_fetch_reports_failure(monkeypatch, tmp_path, capsys):
    from ufc.ingest import __main__ as cli

    def fail(dest, ref):
        raise fetch.FetchError("GET https://example.invalid failed: HTTP 500")

    monkeypatch.setattr(cli, "fetch_snapshot", fail)
    assert cli.main(["fetch", "--dest", str(tmp_path / "greco")]) == 1
    assert "Fetch failed" in capsys.readouterr().err
