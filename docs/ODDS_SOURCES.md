# Odds sources for M12

Research only. No pipeline code was written and no odds were downloaded. Pages were read on 2026-11-17 through a read-only page-fetch tool (plain GETs, no login, no challenge handling). Vendor terms change; every term quoted here must be re-read on the day an account is opened. This is a plain record of what the pages say, not legal advice.

Reliability labels: **verified** means the text was read on the page. **Unverified** means it is a vendor marketing claim, came from a summary of a page rather than its text, or needs a live call to confirm. Nothing in this file is a model result.

## What M12 needs

- Source of the market baseline in `ML_REQUIREMENTS.md` (normalised implied probability, vig removed) and of a blended model and a betting simulation (D7).
- **Historical:** pre-fight moneylines for past bouts, covering at least the evaluation years: validation 2023 to 2024, test 2025 onward, and the yearly walk-forward folds. A source that starts in mid-2020 covers every fold after 2020.
- **Timing matters.** A betting simulation is only honest if the price was available before the bout. "Closing" must mean a price at or just before the event start, and the file must say when it was taken.
- **Bout identity:** odds must join to `fights`/`upcoming_bouts` by event date and fighter names. Names will not match exactly; D17 bans fuzzy matching, so odds sources need the same override-with-evidence mechanism as Wikipedia (`src/ufc/predict/wikipedia_overrides.csv`).
- **Rule 5 applies unchanged:** nothing that needs defeating bot protection, no headless browser pointed at a challenge, no spoofed User-Agent.

## 1. Historical odds

### 1a. BestFightOdds (bestfightodds.com)

- **Coverage:** the archive page says "All odds posted on the site are stored in the archive. The archive contains thousands of matchups and fighter profiles dating back to 2007 when the site was launched." (verified as page text; the 2007 start and per-year completeness were not checked).
- **What a page shows:** per bout, a moneyline row and many prop rows, with prices per sportsbook and movement arrows. The books named in the header include DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Unibet, BetWay, Polymarket and Kalshi (the last two are prediction markets). Which column belongs to which book was not readable in the fetch. There are "Last change: N min ago" lines. Opening and closing values and a "since opening" movement filter appear only in a model summary of the event page (**unverified**).
- **Terms (verified):** `/terms` is two sentences, a warranty disclaimer ("provided on an 'as is' and 'as available' basis"). It says nothing about scraping, automated access, redistribution, commercial use or an API. `/privacy` (effective 2026-04-03) says nothing about them either. There is no API or data-download link on the pages read.
- **robots.txt (verified, verbatim):** `User-agent: *`, `Allow: /`, `Disallow: /cnadm/`, and two sitemaps. Automated crawling of event and fighter pages is not disallowed.
- **Bot protection:** no challenge was served to the fetch tool. The site sits behind Cloudflare and the `ufcscraper` code (below) contains a loop waiting out a "Verify you are human" page, so a challenge can appear. If one does, we stop (rule 5).
- **Bout identification:** by fighter name and event page; no stable cross-source ID.
- **Licence:** none stated. Silence in the terms is not permission to republish. Use would be limited to private analysis; raw BFO odds must not appear in the public site's data files.

### 1b. The `ufcscraper` package (balaustrada/ufcscraper)

Read: docs, README, CHANGELOG, `pyproject.toml`, `bfo_scraper.py`, `base.py`, `utils.py`. PyPI itself returned a JavaScript "Client Challenge" page, so its metadata was not read.

- **Output (verified):** `BestFightOdds_odds.csv` with `fight_id, fighter_id, opening, closing_range_min, closing_range_max`. Opening is one value; closing is a range, not a single line. No bookmaker column, no timestamps. The values are the first three moneyline cells on a BFO fighter page.
- **Matching:** keyed to UFCStats `fight_id`. Fighters are matched to BFO profiles by fuzzy name score (> 90) and bouts by date within 1.5 days. This is the fuzzy matching D17 rejects.
- **How it fetches BFO (verified from source):** Selenium `webdriver.Chrome()`, no robots.txt check, no User-Agent set, no sleep between normal page loads (the `delay` attribute is never used by the BFO code), and a loop that waits (`sleep(5)`, no timeout) while a human-verification element is on screen.
- **UFCStats:** the odds script instantiates the UFCStats scraper and reads UFCStats events and fights, with the `requests` default User-Agent and no custom delay. That is exactly what rule 5 and D13 forbid.
- **Licence:** none found (no licence field, no LICENSE file visible); README says "educational purposes only". Latest version and date unknown (`pyproject.toml` says 1.1.0, CHANGELOG ends at 0.2.1, 2025-03-04).
- **Answer to the question asked:** its approach does not respect robots.txt in the sense of checking it (it does not check; BFO's robots.txt happens to allow crawling), sets no rate limit, drives a real browser at a page that can show a bot challenge, and requires UFCStats. **Do not install or run it.** It is useful only as a description of BFO's page structure and of what the community extracts (an opening value and a closing range).

### 1c. Published datasets

| Dataset | What was verified | Verdict |
|---|---|---|
| Kaggle `mdabbert/ufc-fights-2010-2020-with-betting-odds` and `mdabbert/ultimate-ufc-dataset` | Only the page titles loaded (JavaScript shell). The title says 2010 to 2020. Columns, odds type, bookmaker, licence, collection method: **unverified**. | Ends 2020, so it cannot cover validation or test years. Not useful for M12 even if verified. |
| GitHub `jansen88/ufc-data` | README: odds from Nov 2014, decimal odds, favourite and underdog, from betmma.tips (not BFO); no opening/closing distinction; scrape timestamps only; its own README warns odds names do not always match so some rows are NaN; no licence shown; its scraper also targets ufcstats.com. | Unlicensed, undated odds, scraped. Not recommended. |
| Hugging Face `oliviersportsdata/UFC-Multi-Market-Opening-Closing-Odds` | Tagged "single-commercial-user" licence and under 1K rows (viewer shows a preview). Columns suggest opening and per-book prices. Date range, collection method, update date: **unverified**. | Licence is restrictive and non-standard; size looks too small; origin unknown. Not recommended without reading its full licence. |
| GitHub `michaelcassetti1/ufc-betting` | A modelling project with a Data folder. Odds source and licence unverified. | Not a documented dataset. |

No dataset collected from BestFightOdds was found on GitHub or Hugging Face. Kaggle search pages could not be read, so Kaggle may hold others.

### 1d. Commercial and API sources with history

| Source | Historical coverage | Snapshots / timestamps | Cost | Terms on storage, display, redistribution |
|---|---|---|---|---|
| **The Odds API** historical endpoint | MMA from mid-2020 (vendor page) | 10-minute snapshots from 2020-06-06, 5-minute from September 2022 (verified, v4 guide text). Each snapshot has an exact time. | Paid plans only. $30 per month buys 20,000 credits; historical calls cost 10 credits per market per region and the endpoint is paid-only (verified, v4 guide text). | Storing "indefinitely", displaying, deriving values and ML training are allowed. Standalone resale or redistribution is not. (verified text, terms dated 2026-08-31) |
| **UFCalendar** (ufcalendar.com) | Closing lines back to 2012 for 9,400+ completed fights across UFC and other promotions (vendor claim). 2,700+ have a movement series (Pro and above). | One consensus closing line per bout, defined by the vendor as the last point at or before the start of the bout's card section. Not a sportsbook's published close. | Free: 100 requests a month, 10 a minute (accounts made before 2026-11-11 keep 1,000). Hobby $19 a month, 30,000 requests. No bulk file: "There is no bulk export or dump." Backfill is through the paginated API. | "There is no limit on how much you store or for how long." Commercial use and display allowed on all plans. Not allowed: redistributing the odds "as a standalone odds feed". Circumventing rate limits is prohibited. (verified text, terms updated 2026-11-10) |
| Apify `sports-odds-lab/ufc-fights-odds` | Odds since about 2019 for "about 99%" of fights (developer's own check) | Opening and closing "where ESPN has them" | About $5 per 1,000 fights | Source is ESPN's public feeds. No terms on the page for the output; ESPN's terms were not read. 3 total users, one changelog entry. Not recommended. |
| SportsDataIO | Betting archive from 2020 (search result only) | Not read | Commercial | Not read. Not pursued. |
| OddsPapi | Historical odds only from January 2026 | Endpoint free, 5 s cooldown | Quota unpublished | Useless for a backtest. |
| SportsGameOdds | Pro tier ($299 a month) only; depth not stated | | | Too expensive. |
| odds-api.io | "Historical odds data available" with no depth stated | | No free tier for new users; cheapest paid GBP 49 a month, 2 bookmakers | Not read further. |

## 2. Upcoming odds

| Source | Free tier | UFC moneyline on free tier | Calls for one snapshot of a card | Terms |
|---|---|---|---|---|
| **The Odds API** | 500 credits a month. Cost is markets times regions, "1 per region per market". `/sports` and empty responses are free. | Yes: sport key `mma_mixed_martial_arts`, market `h2h` ("fight winner"). Regions us, us2, uk, eu, au. | One call returns every listed MMA event, so one snapshot of one region is **1 credit**, not one per event. | Clear and permissive for our use (see 1d). Attribution "not required". Key revoked on suspected violation. |
| **UFCalendar** | 100 requests a month, 10 a minute. | Yes, on every plan: consensus moneyline, mean across about 18 books (unnamed; prediction markets and exchanges excluded). | `/v1/events/{slug}/odds` returns a whole card in one call (doc text; call counting not confirmed). | Clear and permissive (see 1d). |
| SportsGameOdds | 2,500 objects a month, 10 requests a minute | UFC page says yes; the pricing page's free league list has no MMA. **Conflict unresolved.** | | Requires deletion of data within 30 days of termination; public display only inside apps with "material value independent of the Service". Worse than the two above. |
| OddsPapi | Free key, quota unpublished | Not explicit | | No storage or display clause found. |
| odds-api.io | None for new users | | | Not usable. |

Four open points were settled on 2026-11-18 (UTC) by one live call and the API guide; see section 2a.

Budget estimate (my arithmetic from the verified cost formula): one region, `h2h` only, is 1 credit a call; the capture command's default of two regions (`us,eu`) is 2 credits a call, confirmed by the live call. Because one call returns every card in the feed, snapshots are per call, not per card: six calls spread over a fight week is 12 credits at two regions, so a month of five cards at six calls each is about 60 credits, 12% of the free quota. Hourly snapshots for a whole fight week would be about 168 credits per card for one region, so hourly capture is not affordable for several cards a month and is not needed.

### 2a. Live call results (one call, 2026-11-18 01:58 UTC, `regions=us,eu`, `markets=h2h`, `oddsFormat=american`)

Run with `uv run python -m ufc.odds snapshot`. The raw response and its sidecar are in `data/odds/raw/` (gitignored; no vendor rows are committed, rule 3). "Verified (call)" is something this response showed; "verified (docs)" is verbatim guide text read the same day.

1. **Which bookmakers carry MMA moneylines: verified (call).** 53 events, all with at least one h2h price. Events carried per bookmaker: betonlineag 52, fanduel 19, onexbet 19, nordicbet 18, betsson 18, draftkings 18, marathonbet 17, betus 15, betanysports 15, betrivers 14, betclic_fr 13, winamax_fr 12, coolbet 11, betfair_ex_eu 11, **pinnacle 11**, matchbook 11, bovada 10, sport888 9, unibet_fr 9, pmu_fr 3. Pinnacle returned a price on all 11 bouts of Saturday's UFC card in the feed (14 to 19 books each); the small regional shows mostly had a single book. Pinnacle and the exchanges (betfair_ex_eu, matchbook) are in the `eu` region, so a US-only call would miss Pinnacle. The `uk` and `au` regions were not called. Each bookmaker and market carries its own `last_update`, which is the best available "price as of" time.
2. **`commence_time`: per bout, not per card (verified, call).** The feed's event is a bout (home and away team are the two fighters; the 32-character `id` is per bout). Saturday 2026-11-21's UFC bouts have staggered start times: 15:00Z (three bouts), 15:15Z, 15:45Z, 17:15Z, 18:00Z, 18:15Z, 18:45Z, 19:00Z, 19:15Z. These are scheduled bout times, probably estimates for later bouts, not an official card start (the verbatim docs only say the value is the start time and that a past value means in-play). The feed covers every MMA promotion (Polish and other regional shows appear on the same date), so UFC bouts must be picked out by matching names and dates to ours in M12. This is why "card start" in `ML_REQUIREMENTS.md` is defined as the earliest `commence_time` among the card's bouts.
3. **Rate limits: partly verified (docs).** The guide says "The current rate limit is 30 requests per second on paid usage plans." It says nothing about free plans, and one call cannot reveal that. 429 means throttling; the spec's 401 description is "The API key might be missing or invalid (unauthenticated), or it might at its usage limit (unauthorized)", so running out of credits looks like a 401, not a 429. The guide documents the `x-api-key` header as the alternative to the `apiKey` query parameter "for extra privacy", and the capture command uses the header so the key is never in a URL. At one call per capture, rate limits are irrelevant.
4. **Credits: verified (call and docs).** The call cost 2 credits (2 regions, 1 market), used 2, remaining 498 of 500, read from `x-requests-used`, `x-requests-remaining` and `x-requests-last`. The `/sports` endpoint "will not affect the quota usage" and returns the same headers, so the capture command reads the quota from `/sports` first and refuses below 50 credits.
5. **Historical credit cost: verified (docs), not called.** "The usage quota cost for historical odds is 10 per region per market" (cost = 10 x markets x regions) and "This endpoint is only available on paid usage plans." The free key cannot show it live, and no call was made. Snapshots: "from June 6th 2020, with snapshots taken at 10 minute intervals", and "From September 2022 ... 5 minute intervals". So one h2h closing snapshot in one region costs 10 credits, two regions 20. For about 290 events and one snapshot each: 2,900 credits in one region, 5,800 in two. Both fit in the 20,000-credit $30 plan.

Still unverified: whether the free plan has a rate limit lower than 30 a second; whether main-card bouts' times move before the event; whether a historical snapshot returns the same books and `last_update` fields as the live feed.

## 3. What the odds are not

- **No source gives true opening and true closing lines with timestamps for free.** BFO's pages mention both but terms and extraction are unsettled; UFCalendar's "closing" is a vendor definition across books; The Odds API's snapshots are real timestamps but only from mid-2020 and cost credits.
- All sources differ in which book they report. A betting simulation must name the book or consensus it used, state the snapshot time, and say whether vig was removed.
- Public pages must show derived values (for example the model's edge) and not raw vendor odds. Every source above prohibits standalone redistribution, and `data/` is gitignored anyway (hard rule 3).

## 4. Recommendation

### (a) Historical backtest source: The Odds API historical endpoint, one paid month

- Verified permissive terms for storage, display of derived values and ML use; a real timestamp per price; named bookmakers; stable bout objects; and no scraping.
- Covers mid-2020 onward: all of validation (2023 to 2024), test (2025 onward) and the walk-forward folds after 2020. It does not cover 2014 to 2020, which the backtest does not need.
- Cost (verified plan table and historical cost formula, section 2a): $30 for one month of 20,000 credits. If a historical call is 10 credits per region per market, one closing-time snapshot of one region for about 290 events (my rough count: about 45 cards a year over 6.5 years) is about 2,900 credits. That fits with room for three snapshots per event.
- **Second source for cross-checking and for 2012 to 2020: UFCalendar free tier**, which keeps a vendor-defined consensus close back to 2012. Its 100 requests a month would not backfill 9,400 bouts unless one request returns many (page size not read). Hobby ($19) would. Use only if the Odds API coverage proves thin; cost-benefit for older years is low.
- **Not recommended:** BFO scraping (no stated permission, challenge risk, no timestamps, no bulk route), the `ufcscraper` package (UFCStats dependency, no robots handling, no delays, Selenium against a challenge-capable site), and the Kaggle, GitHub and Hugging Face files found (end in 2020, unlicensed or restrictive, or unverifiable).
- Spending money is the user's decision; nothing was bought or registered.

### (b) Forward capture source: The Odds API free plan, starting before Saturday

- One credit per snapshot, a month's free quota is about 16 times what a six-snapshot-per-card schedule needs, and the terms allow storing indefinitely.
- Snapshots captured now are the only way to get true pre-fight timestamps for 2026 cards, and they can double as a check on the historical source (same API, same book names).
- **Built (2026-11-18):** `uv run python -m ufc.odds snapshot` makes one h2h call (default regions `us,eu`, 2 credits), refuses below 50 remaining credits (checked on the free `/sports` endpoint first), stores the raw bytes in `data/odds/raw/<UTC timestamp>.json` with a `.meta.json` sidecar (request time, credit headers, SHA-256), and prints a summary. The key is read from `.env` and sent in a header. No database writes and no matching; that is M12. Its design points:
  1. Write each response, raw and unmodified, with the request time and the quota headers, into `data/odds/` (gitignored), so nothing is lost if matching rules change.
  2. Match to `upcoming_bouts` afterwards, not at capture time.
  3. The first live call has been made (section 2a); a snapshot before Saturday's card starts (first bout 15:00Z on 2026-11-21) and one near the start give the first closing-line candidates. Schedule the later calls by hand or with cron; scheduling is not built.
- **Optional second capture:** UFCalendar's free key, one call per card, as a consensus line (a very different kind of number, kept in a separate column).

## 5. Open items before any M12 build

1. User decides on the one paid month for historical data. (The free key is registered.)
2. Done: one live call settled the four open points (section 2a). The test fixture is synthetic in the same shape (no vendor rows in the repo, rule 3).
3. Name matching for odds sources: an override file with evidence, no fuzzy matching (D17).
4. Done: "closing" and the three strategies are pre-registered in `ML_REQUIREMENTS.md` (D62). Still open there: nothing about the book; Pinnacle is the stated reference.
5. Re-read each vendor's terms on the day of sign-up.

## Sources read

- bestfightodds.com: `/robots.txt`, `/`, `/archive`, `/terms`, `/privacy`, `/events/ufc-334-4370`
- ufc-scraper.readthedocs.io (BestFightOdds tables, introduction); github.com/balaustrada/ufcscraper (README, `bfo_scraper.py`, `base.py`, `utils.py`, `pyproject.toml`, `CHANGELOG.md`); pypi.org/project/ufcscraper returned a client challenge and was not read
- the-odds-api.com: home and pricing, `/liveapi/guides/v4/`, `/sports-odds-data/mma-odds.html`, `/terms-and-conditions.html`, `/manage/faqs.html`
- ufcalendar.com: `/developers/ufc-odds-api`, `/developers/guides/ufc-closing-odds-line-movement`, `/developers/terms`
- odds-api.io (`/sports/mma`, `/terms`, `/pricing`); oddspapi.io (requests and quota, terms, MMA, historical odds); sportsgameodds.com (UFC page, pricing, terms); apify.com/sports-odds-lab/ufc-fights-odds
- kaggle.com mdabbert pages (titles only); huggingface.co/datasets/oliviersportsdata/UFC-Multi-Market-Opening-Closing-Odds; github.com/jansen88/ufc-data; github.com/michaelcassetti1/ufc-betting
