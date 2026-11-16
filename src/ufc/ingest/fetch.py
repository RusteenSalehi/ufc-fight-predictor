"""Download a snapshot of the published dataset from GitHub, pinned to a commit.

Seven requests per run: one to resolve the ref to a commit, six for the CSVs.
Files are downloaded to a staging directory and checked before the previous
snapshot is replaced, so a failed or malformed download leaves the last good
snapshot in place.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import requests

from ufc.ingest import source
from ufc.ingest.source import SourceInfo

logger = logging.getLogger(__name__)

COMMIT_URL = "https://api.github.com/repos/{repo}/commits/{ref}"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{file_name}"
DEFAULT_USER_AGENT = "ufcstats-dataset-ingest/0.1 (python-requests)"
USER_AGENT_ENV = "UFC_INGEST_USER_AGENT"
TIMEOUT_S = (10, 120)
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
PAUSE_BETWEEN_REQUESTS_S = 0.5
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class FetchError(Exception):
    """The snapshot could not be downloaded or failed its format check."""


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = os.environ.get(USER_AGENT_ENV, DEFAULT_USER_AGENT)
    return session


def get_with_retries(
    session: requests.Session, url: str, sleep: Callable[[float], None] = time.sleep
) -> requests.Response:
    """GET with a timeout, retrying transient failures with exponential backoff."""
    last_error = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = repr(exc)
        else:
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in RETRYABLE_STATUS:
                break
        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
            logger.warning(
                "GET %s failed (%s); retrying in %.0fs", url, last_error, delay
            )
            sleep(delay)
    raise FetchError(f"GET {url} failed: {last_error}")


def fetch_snapshot(
    dest: Path,
    ref: str = "main",
    *,
    repo: str = source.UPSTREAM_REPO,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SourceInfo:
    """Download the six CSVs at ``ref`` (a branch or commit SHA) into ``dest``.

    Writes ``SOURCE.json`` beside them recording the exact commit, its date,
    and each file's SHA-256, so any load can be traced to and repeated from
    one upstream commit.
    """
    session = session or make_session()
    try:
        commit_data = get_with_retries(
            session, COMMIT_URL.format(repo=repo, ref=ref), sleep
        ).json()
        commit = commit_data["sha"]
        commit_date = commit_data["commit"]["committer"]["date"]
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"unexpected commit description for {repo}@{ref}") from exc
    logger.info("Fetching %s at commit %s (%s)", repo, commit, commit_date)

    staging = dest.parent / f".{dest.name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        files: dict[str, dict[str, object]] = {}
        for file_name in source.EXPECTED_COLUMNS:
            sleep(PAUSE_BETWEEN_REQUESTS_S)
            url = RAW_URL.format(repo=repo, commit=commit, file_name=file_name)
            content = get_with_retries(session, url, sleep).content
            _check_csv(file_name, content)
            (staging / file_name).write_bytes(content)
            files[file_name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            logger.info("Downloaded %s (%d bytes)", file_name, len(content))

        info = SourceInfo(
            repo=repo,
            commit=commit,
            commit_date=commit_date,
            fetched_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        description = {
            "repo": info.repo,
            "commit": info.commit,
            "commit_date": info.commit_date,
            "fetched_at": info.fetched_at,
            "files": files,
        }
        (staging / source.SOURCE_INFO_FILE).write_text(
            json.dumps(description, indent=2) + "\n", encoding="utf-8"
        )

        dest.mkdir(parents=True, exist_ok=True)
        for file_name in [*source.EXPECTED_COLUMNS, source.SOURCE_INFO_FILE]:
            os.replace(staging / file_name, dest / file_name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return info


def _check_csv(file_name: str, content: bytes) -> None:
    """Reject a download that is empty or does not have the documented columns."""
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FetchError(f"{file_name}: not UTF-8 text") from exc
    if len(lines) < 2:
        raise FetchError(f"{file_name}: no data rows")
    try:
        source.check_header(file_name, next(csv.reader(lines[:1])))
    except source.SourceError as exc:
        raise FetchError(str(exc)) from exc
