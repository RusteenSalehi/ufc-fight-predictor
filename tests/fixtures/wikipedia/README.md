# Wikipedia fixtures

Raw MediaWiki API responses that the tests read instead of the network.

| File | API request | Revisions |
|---|---|---|
| `scheduled_events_section.json` | `action=parse&page=List_of_UFC_events&prop=wikitext\|revid&section=3` (saved 2026-11-16, M1) | 1385888810 |
| `upcoming_event_parse.json` | `action=parse&page=UFC Fight Night: Procházka vs. Stirling&prop=wikitext\|revid\|sections&redirects=1` (saved 2026-11-16, M1) | 1386038886 |
| `list_of_ufc_events_query.json` | `action=query&prop=revisions` for "List of UFC events" (saved 2026-11-17) | 1385888810, **trimmed**: the "Past events" table keeps its first 12 rows; the rest of the page is as published |
| `event_pages_query.json` | `action=query&prop=revisions` for "UFC 334" and "UFC Fight Night: Procházka vs. Stirling" (saved 2026-11-17) | 1385977806, 1386038886 |
| `fighter_pages_query.json` | `action=query&prop=revisions` for "Navajo Stirling", "Santiago Luna (fighter)", "Jiri Prochazka" (a redirect) and "Tahir Abdullayev (fighter)" (no such page) (saved 2026-11-17) | 1382306685, 1385724486, 1383898741 |

The M1 files were requested from `https://en.wikipedia.org/w/api.php` with
`format=json&formatversion=2`. The M5 files were saved by
`uv run python scripts/save_wikipedia_fixtures.py`, through the project's own
client; each records the exact parameters it sent next to the response.

Text is from English Wikipedia and is licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Authors are listed in the
page histories of the articles named above.
