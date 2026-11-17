"""Save the MediaWiki API responses the M5 tests read (tests/fixtures/wikipedia/).

    uv run python scripts/save_wikipedia_fixtures.py

Requests go through the project's Wikipedia client (User-Agent with contact
information, spacing, retries). One response is trimmed: the "Past events"
table of "List of UFC events" holds every UFC event ever held, so only its
first ``PAST_ROWS`` rows are kept; the rest of the page is unchanged. The
README in the fixture directory lists each file, its revision and the trim.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ufc.predict.cards import LIST_PAGE
from ufc.predict.wikipedia import CARD_MAX_AGE_S, Wikipedia, http_transport

OUT = Path("tests/fixtures/wikipedia")
PAST_ROWS = 12
EVENT_PAGES = ("UFC Fight Night: Procházka vs. Stirling", "UFC 334")
FIGHTER_PAGES = (
    "Navajo Stirling",
    "Santiago Luna (fighter)",
    "Jiri Prochazka",  # a redirect
    "Tahir Abdullayev (fighter)",  # expected not to exist
)


def capture(titles: tuple[str, ...]) -> dict:
    """One query, as the client sends it, returning the raw response."""
    responses: list[dict] = []
    real = http_transport()

    def keep(params):
        body = real(params)
        responses.append({"params": dict(params), "response": body})
        return body

    Wikipedia(keep).pages(titles, CARD_MAX_AGE_S)
    if len(responses) != 1:
        raise SystemExit(f"expected one request for {titles}, made {len(responses)}")
    return responses[0]


def trim_past_events(wikitext: str) -> str:
    start = wikitext.index("==Past events==")
    end = wikitext.index("\n|}", start)
    rows = re.split(r"(?m)^(?=\|-)", wikitext[start:end])
    return wikitext[:start] + "".join(rows[: PAST_ROWS + 1]) + wikitext[end:]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    listing = capture((LIST_PAGE,))
    page = listing["response"]["query"]["pages"][0]
    content = page["revisions"][0]["slots"]["main"]
    content["content"] = trim_past_events(content["content"])
    files = {
        "list_of_ufc_events_query.json": listing,
        "event_pages_query.json": capture(EVENT_PAGES),
        "fighter_pages_query.json": capture(FIGHTER_PAGES),
    }
    for name, record in files.items():
        (OUT / name).write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        revisions = {
            p["title"]: p.get("revisions", [{}])[0].get("revid")
            for p in record["response"]["query"]["pages"]
        }
        print(name, revisions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
