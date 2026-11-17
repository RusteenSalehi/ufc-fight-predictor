"""A small, polite client for the English Wikipedia MediaWiki API.

Wikimedia's API etiquette and User-Agent policy
(https://meta.wikimedia.org/wiki/User-Agent_policy, https://www.mediawiki.org/wiki/API:Etiquette):

- a descriptive User-Agent with contact information;
- requests made one at a time, never in parallel, here at least
  ``MIN_INTERVAL_S`` apart;
- ``maxlag`` set, and a ``maxlag`` error or an HTTP 429/5xx answered by
  waiting and retrying with backoff;
- many pages in one request where the API allows it (up to 50 titles);
- responses cached, so running the refresh twice in a row makes no new
  requests.

The only network call goes through ``Transport``, which tests replace with
saved responses. No test touches the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
CLIENT = "ufc-fight-predictor/0.1"
DEFAULT_CONTACT = "https://github.com/RusteenSalehi/ufc-fight-predictor"
CONTACT_ENV = "UFC_WIKI_CONTACT"
TIMEOUT_S = (10, 60)
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0
MIN_INTERVAL_S = 1.0
MAXLAG_S = 5
MAX_TITLES = 50
"""The API's limit on titles per query for ordinary clients."""

DEFAULT_CACHE = Path("data/cache/wikipedia")
CARD_MAX_AGE_S = 3600
"""Event pages and the events list: cards change during fight week, so an
hour. A refresh run twice within an hour reads the cache."""
PROFILE_MAX_AGE_S = 7 * 24 * 3600
"""Fighter pages, read only for height, reach and date of birth."""


class WikipediaError(RuntimeError):
    """A request failed after retries, or the API answered with an error."""


def user_agent() -> str:
    contact = os.environ.get(CONTACT_ENV, DEFAULT_CONTACT).strip()
    if not contact:
        raise WikipediaError(
            f"{CONTACT_ENV} is set but empty; it must hold contact info"
        )
    return f"{CLIENT} ({contact}) python-requests/{requests.__version__}"


Transport = Callable[[Mapping[str, str]], dict]
"""Sends one API request (query parameters) and returns the decoded JSON."""


def http_transport(
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Transport:
    """The real transport: GET with timeouts, spacing, retries and backoff."""
    session = session or requests.Session()
    session.headers["User-Agent"] = user_agent()
    last = [-MIN_INTERVAL_S]

    def send(params: Mapping[str, str]) -> dict:
        error = "no attempt made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = MIN_INTERVAL_S - (clock() - last[0])
            if wait > 0:
                sleep(wait)
            last[0] = clock()
            retry_after = None
            try:
                response = session.get(API_URL, params=dict(params), timeout=TIMEOUT_S)
            except requests.RequestException as exc:
                error = repr(exc)
            else:
                retry_after = response.headers.get("Retry-After")
                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError:
                        error = "response was not JSON"
                    else:
                        code = (body.get("error") or {}).get("code")
                        if code is None:
                            return body
                        error = f"API error {code}: {body['error'].get('info', '')}"
                        if code != "maxlag":
                            raise WikipediaError(error)
                else:
                    error = f"HTTP {response.status_code}"
                    if response.status_code not in (429, 500, 502, 503, 504):
                        raise WikipediaError(error)
            if attempt < MAX_ATTEMPTS:
                delay = BACKOFF_BASE_S * 2 ** (attempt - 1)
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                logger.warning(
                    "Wikipedia request failed (%s); retrying in %.0fs", error, delay
                )
                sleep(delay)
        raise WikipediaError(
            f"Wikipedia request failed after {MAX_ATTEMPTS} attempts: {error}"
        )

    return send


@dataclass(frozen=True, slots=True)
class Page:
    """One page's current wikitext. ``title`` is the page's own title after
    redirects; ``requested`` is the title that was asked for."""

    requested: str
    title: str
    pageid: int | None
    revid: int | None
    wikitext: str | None  # None when the page does not exist

    @property
    def exists(self) -> bool:
        return self.wikitext is not None


class Wikipedia:
    """Pages by title, through a transport and an on-disk cache."""

    def __init__(
        self,
        transport: Transport,
        cache_dir: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._transport = transport
        self._cache = cache_dir
        self._clock = clock
        self.requests_made = 0

    def _request(self, params: Mapping[str, str], max_age_s: float) -> dict:
        params = {
            "format": "json",
            "formatversion": "2",
            "maxlag": str(MAXLAG_S),
            **params,
        }
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = self._cache / f"{key}.json" if self._cache else None
        if path is not None and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if self._clock() - stored["fetched_at"] <= max_age_s:
                return stored["response"]
        response = self._transport(params)
        self.requests_made += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "fetched_at": self._clock(),
                        "params": params,
                        "response": response,
                    }
                ),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        return response

    def pages(self, titles: Iterable[str], max_age_s: float) -> dict[str, Page]:
        """Current wikitext of each title, following redirects. Keys are the
        requested titles. Titles are sorted and batched, so the same set of
        titles always makes the same requests."""
        wanted = sorted(dict.fromkeys(t for t in titles if t))
        result: dict[str, Page] = {}
        for start in range(0, len(wanted), MAX_TITLES):
            batch = wanted[start : start + MAX_TITLES]
            body = self._request(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content|ids",
                    "rvslots": "main",
                    "redirects": "1",
                    "titles": "|".join(batch),
                },
                max_age_s,
            )
            result.update(pages_from_query(body, batch))
        return result


def pages_from_query(body: Mapping, requested: Sequence[str]) -> dict[str, Page]:
    """Map a ``query``+``revisions`` response back to the titles asked for."""
    query = body.get("query")
    if not isinstance(query, Mapping):
        raise WikipediaError("query response without a 'query' object")
    renamed: dict[str, str] = {}
    for step in ("normalized", "redirects"):
        for entry in query.get(step, []):
            renamed[entry["from"]] = entry["to"]
    by_title = {page["title"]: page for page in query.get("pages", [])}
    out: dict[str, Page] = {}
    for title in requested:
        final, seen = title, set()
        while final in renamed and final not in seen:
            seen.add(final)
            final = renamed[final]
        page = by_title.get(final)
        if page is None or page.get("missing") or page.get("invalid"):
            out[title] = Page(title, final, None, None, None)
            continue
        revision = (page.get("revisions") or [{}])[0]
        content = (revision.get("slots") or {}).get("main", {}).get("content")
        out[title] = Page(
            title, final, page.get("pageid"), revision.get("revid"), content
        )
    return out
