"""Saved Wikipedia API responses (tests/fixtures/wikipedia/) as a transport.

No test touches the network: ``FixtureTransport`` answers a request with the
saved response whose parameters asked for the same titles, and fails the
test on any other request.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia"


def saved(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def wikitext_of(name: str, title: str) -> str:
    for page in saved(name)["response"]["query"]["pages"]:
        if page["title"] == title:
            return page["revisions"][0]["slots"]["main"]["content"]
    raise KeyError(title)


def legacy_wikitext(name: str) -> str:
    """The two responses saved in M1 used action=parse."""
    return saved(name)["parse"]["wikitext"]


def query_response(pages: dict[str, str | None], revid: int = 1) -> dict:
    """A query+revisions response for invented pages (None = missing)."""
    out = []
    for index, (title, text) in enumerate(pages.items()):
        if text is None:
            out.append({"title": title, "missing": True})
        else:
            out.append(
                {
                    "title": title,
                    "pageid": 1000 + index,
                    "revisions": [
                        {"revid": revid, "slots": {"main": {"content": text}}}
                    ],
                }
            )
    return {"batchcomplete": True, "query": {"pages": out}}


class FixtureTransport:
    """Answers requests from a {frozenset(titles): response} table."""

    def __init__(self, responses: dict[frozenset[str], dict]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def __call__(self, params):
        self.requests.append(dict(params))
        titles = frozenset(params["titles"].split("|"))
        if titles not in self.responses:
            raise AssertionError(f"unexpected request for {sorted(titles)}")
        response = self.responses[titles]
        if isinstance(response, Exception):
            raise response
        return response
