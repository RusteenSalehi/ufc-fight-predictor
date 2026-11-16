# Wikipedia fixtures

Raw MediaWiki API responses saved on 2026-11-16 to confirm that upcoming UFC
cards can be read in a parseable form. No parser is built on them yet (that is
milestone M5).

| File | API request | Revision |
|---|---|---|
| `scheduled_events_section.json` | `action=parse&page=List_of_UFC_events&prop=wikitext\|revid&section=3` | 1385888810 |
| `upcoming_event_parse.json` | `action=parse&page=UFC Fight Night: Procházka vs. Stirling&prop=wikitext\|revid\|sections&redirects=1` | 1386038886 |

Both were requested from `https://en.wikipedia.org/w/api.php` with
`format=json&formatversion=2`.

Text is from English Wikipedia and is licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Authors are listed in the
page histories of the two articles named above.
