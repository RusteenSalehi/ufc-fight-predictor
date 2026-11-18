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
| **The Odds API** historical endpoint | MMA from mid-2020 (vendor page) | 5-minute snapshots from Sept 2022, 10-minute before (summary of the v4 guide; **unverified**, check the API reference). Each snapshot has an exact time. | Paid plans only. $30 per month buys 20,000 credits; historical calls cost 10 credits per market per region (summary; **unverified**). | Storing "indefinitely", displaying, deriving values and ML training are allowed. Standalone resale or redistribution is not. (verified text, terms dated 2026-08-31) |
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

Open points for The Odds API that need one live call with a key:

- Which books return MMA `h2h` (the docs example shows DraftKings only; Pinnacle for MMA is not confirmed).
- Whether `commence_time` for MMA is the bout's time or the card's start. Its event IDs are per bout (fighter names as `home_team`/`away_team`), but main-card and prelim times may differ.
- The rate limit per second; the guide mentions HTTP 429 but gave no number.
- The sport key returns all MMA promotions, so events must be filtered to ones that match our `upcoming_bouts` by name and date.

Budget estimate (my arithmetic from the verified cost formula): one region, `h2h` only, is 1 credit a call. Six snapshots per card (for example 72 h, 48 h, 24 h, 6 h, 1 h before the card start, plus one after lines settle) for a month with five cards is about 30 credits, 6% of the free quota. Two regions (us and eu) doubles it. Hourly snapshots for a whole fight week would be about 168 credits per card for one region, so hourly capture is not affordable for several cards a month and is not needed.

## 3. What the odds are not

- **No source gives true opening and true closing lines with timestamps for free.** BFO's pages mention both but terms and extraction are unsettled; UFCalendar's "closing" is a vendor definition across books; The Odds API's snapshots are real timestamps but only from mid-2020 and cost credits.
- All sources differ in which book they report. A betting simulation must name the book or consensus it used, state the snapshot time, and say whether vig was removed.
- Public pages must show derived values (for example the model's edge) and not raw vendor odds. Every source above prohibits standalone redistribution, and `data/` is gitignored anyway (hard rule 3).

## 4. Recommendation

### (a) Historical backtest source: The Odds API historical endpoint, one paid month

- Verified permissive terms for storage, display of derived values and ML use; a real timestamp per price; named bookmakers; stable bout objects; and no scraping.
- Covers mid-2020 onward: all of validation (2023 to 2024), test (2025 onward) and the walk-forward folds after 2020. It does not cover 2014 to 2020, which the backtest does not need.
- Cost (from the verified plan table and an unverified historical cost summary): $30 for one month of 20,000 credits. If a historical call is 10 credits per region per market, one closing-time snapshot of one region for about 290 events (my rough count: about 45 cards a year over 6.5 years) is about 2,900 credits. That fits with room for three snapshots per event.
- **Second source for cross-checking and for 2012 to 2020: UFCalendar free tier**, which keeps a vendor-defined consensus close back to 2012. Its 100 requests a month would not backfill 9,400 bouts unless one request returns many (page size not read). Hobby ($19) would. Use only if the Odds API coverage proves thin; cost-benefit for older years is low.
- **Not recommended:** BFO scraping (no stated permission, challenge risk, no timestamps, no bulk route), the `ufcscraper` package (UFCStats dependency, no robots handling, no delays, Selenium against a challenge-capable site), and the Kaggle, GitHub and Hugging Face files found (end in 2020, unlicensed or restrictive, or unverifiable).
- Spending money is the user's decision; nothing was bought or registered.

### (b) Forward capture source: The Odds API free plan, starting before Saturday

- One credit per snapshot, a month's free quota is about 16 times what a six-snapshot-per-card schedule needs, and the terms allow storing indefinitely.
- Snapshots captured now are the only way to get true pre-fight timestamps for 2026 cards, and they can double as a check on the historical source (same API, same book names).
- **To start before Saturday (2026-11-21)** the user must create a free key (an account step this session cannot do), and a small snapshot job must be written. That job is code and was out of scope here. It should:
  1. Write each response, raw and unmodified, with the request time and the quota headers, into `data/odds/` (gitignored), so nothing is lost if matching rules change.
  2. Match to `upcoming_bouts` afterwards, not at capture time.
  3. Take a one-off dry run first to settle the open points in section 2.
- **Optional second capture:** UFCalendar's free key, one call per card, as a consensus line (a very different kind of number, kept in a separate column).

## 5. Open items before any M12 build

1. User decides on the one paid month for historical data and registers the keys.
2. One live Odds API call to answer the four open points in section 2, with the response saved as a fixture of synthetic shape (no vendor rows in the repo, rule 3).
3. Name matching for odds sources: an override file with evidence, no fuzzy matching (D17).
4. Define "closing" for the backtest in `ML_REQUIREMENTS.md` before looking at results: the last snapshot at or before the card start, and the book or consensus used.
5. Re-read each vendor's terms on the day of sign-up.

## Sources read

- bestfightodds.com: `/robots.txt`, `/`, `/archive`, `/terms`, `/privacy`, `/events/ufc-334-4370`
- ufc-scraper.readthedocs.io (BestFightOdds tables, introduction); github.com/balaustrada/ufcscraper (README, `bfo_scraper.py`, `base.py`, `utils.py`, `pyproject.toml`, `CHANGELOG.md`); pypi.org/project/ufcscraper returned a client challenge and was not read
- the-odds-api.com: home and pricing, `/liveapi/guides/v4/`, `/sports-odds-data/mma-odds.html`, `/terms-and-conditions.html`, `/manage/faqs.html`
- ufcalendar.com: `/developers/ufc-odds-api`, `/developers/guides/ufc-closing-odds-line-movement`, `/developers/terms`
- odds-api.io (`/sports/mma`, `/terms`, `/pricing`); oddspapi.io (requests and quota, terms, MMA, historical odds); sportsgameodds.com (UFC page, pricing, terms); apify.com/sports-odds-lab/ufc-fights-odds
- kaggle.com mdabbert pages (titles only); huggingface.co/datasets/oliviersportsdata/UFC-Multi-Market-Opening-Closing-Odds; github.com/jansen88/ufc-data; github.com/michaelcassetti1/ufc-betting
