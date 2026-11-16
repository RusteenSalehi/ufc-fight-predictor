# Design Decisions

Format for each entry: decision, alternatives considered, reasoning. Newest at the bottom.

---

### D1. Write a custom UFCStats scraper instead of using an existing Kaggle dataset
- **Alternatives:** Kaggle fighter-statistics datasets and their scrapers.
- **Why:** Those datasets are one row per fighter with career aggregates as of the scrape date. Training on them leaks the future: a 2019 fight would see 2026 stats. The model needs per-fight data so features can be rebuilt as of each fight date.

### D2. SQLite for storage, S3 for published artifacts
- **Alternatives:** PostgreSQL (managed), DynamoDB, plain files.
- **Why:** A few thousand bouts written by one batch job at a time. SQLite has no server and no monthly cost. Managed Postgres would cost money every month to store a few megabytes. Published predictions are static JSON in S3 because page views only read.

### D3. Precompute predictions in batch
- **Alternatives:** an always-on inference API.
- **Why:** Predictions only change when cards or the model change. Batch publishing makes page views nearly free and removes a running server.

### D4. Manual model promotion
- **Alternatives:** automatic retraining and deployment after every event.
- **Why:** Automatic promotion can ship regressions or bad data silently. A registry plus a manually edited `approved.json` pointer keeps promotion deliberate and rollback trivial.

### D5. Corner is never a feature; train on both orientations
- **Alternatives:** keep red/blue as given.
- **Why:** The red corner is usually the favorite or champion, so corner leaks information and makes predictions depend on arbitrary labeling. Difference features, both-orientation training, and averaged inference make outputs symmetric by construction.

### D6. Chronological splits, log loss as the selection metric
- **Alternatives:** random splits, accuracy.
- **Why:** Random splits let the model train on fights after the ones it is tested on. Log loss rewards calibrated probabilities, which matter more for a probability product than raw accuracy.

### D7. Stats-only headline model; odds evaluated separately
- **Alternatives:** odds as a primary input.
- **Why:** Odds would likely dominate a model and make it copy the market. Keeping the headline model stats-only preserves the question of whether it knows something the market does not. A blended model is evaluated separately once odds data exists.

### D8. All history for features, recency-weighted training
- **Alternatives:** cut data at a recent year.
- **Why:** Features need full history (a fighter's 2015 features depend on 2008 fights). The sport has changed, so recent fights get more training weight; the decay rate is chosen on validation.

### D9. Debuting fighters are predicted and labeled
- **Alternatives:** skip their bouts; source pre-UFC records.
- **Why:** Most cards include a debut, and skipping them leaves cards half empty. Stats are marked missing (not zero), the UI labels these bouts "limited data," and their accuracy is reported separately. Pre-UFC records are a later extension.

### D10. Style archetypes use soft clustering and percentiles
- **Alternatives:** hand-assigned tags; k-means with distance-to-centroid as a skill score.
- **Why:** Hand tags are subjective and cannot be reconstructed historically. Distance to a centroid measures how typical a fighter is, not how good: the most extreme power puncher sits far from the center. Gaussian mixtures capture hybrid styles as probabilities; percentiles within weight class measure strength.

### D11. Anthropic API with prepaid credits for the analyst
- **Alternatives:** Amazon Bedrock.
- **Why:** AWS bills after the fact, so caps must be built in the app. Prepaid credits with auto-reload off are a true hard ceiling. A provider interface keeps a later switch to Bedrock simple.

### D12. Development inside WSL2 on ARM64
- **Alternatives:** native Windows.
- **Why:** The development laptop's Smart App Control blocks unsigned compiled extensions in Windows virtual environments, which previously caused silent test failures. Linux aarch64 is better supported by the ML stack.

### D13. Do not solve or work around the UFCStats bot challenge
- **Context:** On 2026-11-16 every UFCStats page, including the events list, returned a "Checking your browser" page that requires running a JavaScript proof-of-work and posting the answer before content is served. `robots.txt` returns 404, so there are no robots rules either way.
- **Alternatives:** solve the proof-of-work in the HTTP client (technically easy); drive a headless browser; present a browser User-Agent.
- **Why:** The challenge is the operator's signal that automated clients are not wanted. Defeating it would be circumventing an access control, and a public portfolio repository that does so loses the credibility it is meant to build. This is a standing rule for the project, not a temporary choice: no code here requests ufcstats.com at all. UFCStats URLs in the data are parsed for their IDs only.

### D14. Ingest a published per-round dataset instead of scraping (replaces the method of D1)
- **Alternatives:** a licensed feed such as Sportradar (clean on terms, unlikely to fit a $5/month budget); asking the operator for access (slow, no guarantee); waiting to see if the challenge is temporary; Kaggle fighter-level datasets (rejected in D1 and still rejected).
- **Why:** `Greco1899/scrape_ufc_stats` publishes six CSVs collected from UFCStats and refreshes them with a daily scheduled job. D1's requirement still holds and is met: the data is one row per fighter per round of each fight, with no career aggregates, so features can be rebuilt as of any fight date. Checked before building on it (commit `96adace`, 2026-11-15): most recent event 2026-11-14; events, fights and fighter profiles carry UFCStats URLs; licence GPL-3.0.
- **Costs accepted:** fights name their fighters as text, not IDs (D17). The dataset starts at UFC 2; UFC 1 is absent. Fighter profiles are captured once and not refreshed, so height and reach are often empty for recent debutants (M2 will quantify this). The project now depends on one volunteer-run feed that could stop; snapshots are pinned to a commit and the last good database is never overwritten by a failed load, so a stoppage freezes the data rather than breaking the site.

### D15. Terms-of-use caveat, stated plainly
- **What is known:** The dataset is itself a scrape of ufcstats.com. UFC.com's terms of use prohibit automated access, scraping, and compiling site content into a database, and say they cover Zuffa's associated sites; ufcstats.com is not named in them, so whether they apply is unclear. UFCStats's own terms could not be read because its pages are behind the challenge. Upstream's published code uses plain HTTP requests with no challenge handling; how it still collects data is not known to us.
- **What this means:** Downloading the CSVs from GitHub means this project makes no requests to UFCStats, but it does not settle whether the data may be used. Upstream's GPL-3.0 licence covers its repository; it cannot grant rights the data's owner has not given. This project uses the data for a non-commercial portfolio and commits none of it, except a 29-fight excerpt in `tests/fixtures/greco/` whose origin and licence are stated beside it.
- **Consequences:** Commercial use stays a non-goal until licensing is resolved (see SPEC). If the operator or upstream objects, the source is swapped behind `src/ufc/ingest/`; nothing downstream reads the CSVs directly. The methodology page must say where the data comes from and repeat this caveat.

### D16. Wikipedia for upcoming cards
- **Alternatives:** UFCStats upcoming-event pages (blocked, D13); ufc.com event pages (its terms prohibit automated access outright); a paid schedule API.
- **Why:** The dataset has completed events only. Wikipedia's MediaWiki API is meant for programmatic use, its content is CC BY-SA, and event pages list bouts in a regular template (`{{MMAevent bout|weight class|fighter|vs.|fighter}}`). Confirmed on a live upcoming event; responses saved in `tests/fixtures/wikipedia/`.
- **Costs accepted:** Wikipedia is edited by volunteers and can be wrong, late, or vandalised, so M5's validation gate has to treat a card as untrusted input. It has no UFCStats IDs: upcoming bouts need their own IDs, and fighter names (with diacritics and disambiguation suffixes) go through the same name resolution as D17. The site must attribute Wikipedia.

### D17. Resolve fighter names to IDs with exact rules and explicit overrides; never fuzzy matching
- **Context:** Fighter profiles have IDs, but a fight row only names its two fighters. Nine names in the snapshot belong to two profiles each (for example two fighters called Bruno Silva), and bout rows sometimes spell a name differently from the profile.
- **Rule, in order:** an entry in `identity_overrides.csv`; a name (ignoring case, accents and punctuation) that belongs to exactly one profile, counting both the current listing name and the name recorded when the profile was first captured; for a shared name, the one profile whose listed weight fits the bout's weight class; otherwise unresolved.
- **Alternatives:** fuzzy string matching; picking the nearest weight; picking the most active fighter.
- **Why:** A wrong match silently gives one fighter another's history and corrupts every feature built on it. A reported gap is visible and fixable. Nearest-weight was rejected on evidence: it assigns one real lightweight bout to the wrong one of two same-named fighters. Each override records the evidence from the dataset that justifies it, and every non-trivial resolution is written to `ingest_issues`. Two cross-checks pass on the full snapshot: no fighter is under 17 or over 55 at any bout, and no fighter appears in both a women's and a men's weight class.

### D18. Drop and report what cannot be joined with certainty; gate the load
- **Alternatives:** best-effort joins; loading partial rows with NULL fighters.
- **Why:** The files join on event name and bout text, which fails in known ways: a renamed event leaves fights listed twice (resolved: keep the row whose event exists), an event missing from the events file orphans its fights (dropped), and the same pairing fought twice on one card makes its stats rows indistinguishable (both bouts load, without stats). At commit `96adace`, 3 of 9,003 fights are dropped and all are listed by `uv run python -m ufc.ingest load --dry-run`. A dropped bout is missing from both fighters' histories, which is a small known error; a guessed one is an unknown error.
- **Gate:** if more than 1% of fights are dropped or the tables would be internally inconsistent, nothing is written. A broken upstream file therefore cannot replace a good database.

### D19. Store per-round stats and derive per-fight totals at load time
- **Alternatives:** store only totals; compute totals in a SQL view.
- **Why:** The source is per round, and rounds allow later features (pace, fading) without re-ingesting. Totals are summed in one tested Python function and stored, so readers get a plain table. Control time, which early events do not record, is NULL in the total if it is NULL in any round: a partial sum is never presented as a total. Bouts with no stats have `has_stats = 0` and no stats rows, never zeros.

### D20. Schema deviations from the original plan
- **raw_pages replaced by source_files:** there are no fetched pages any more. `source_files` records the upstream commit and a SHA-256 per CSV, which is what the model registry needs to name its training data.
- **ingest_issues added:** the validation report of the last load is queryable, so the data quality report and the methodology page can cite it.
- **bout_round_stats added:** see D19.
- **bouts gained** `bout_type_raw`, `is_interim_title`, `is_tournament_final`, `time_format`, `fight_time_s`, `method_details`, `referee`, `has_stats`. `is_title_bout` means a UFC championship bout, including interim titles and the 1990s superfight title. Tournament finals (early UFC tournaments, The Ultimate Fighter, Road to UFC) are flagged separately, because the source labels them "Title Bout" and counting them would inflate title-fight experience. `fight_time_s` is computed from the bout's format so per-minute rates do not have to re-derive round lengths, including the one-round-plus-overtime formats of the 1990s.
- **is_womens is partly inferred:** the source says "Women's" only inside a weight class name, so women's catch-weight bouts are unlabelled. A bout with no gendered class is marked women's if either fighter also appears in a women's weight class.
- **predictions deferred to M5:** upcoming bouts will come from Wikipedia without UFCStats IDs (D16), which changes what a prediction's bout key is. Defining the table now would mean redefining it.

### D21. Timestamps come from the snapshot, not the clock
- **Alternatives:** stamp rows with the time of the load.
- **Why:** `first_seen_at` and `last_seen_at` are set from the upstream commit date of the snapshot being loaded. Loading the same snapshot twice therefore produces byte-identical tables, which is what the idempotency test asserts, and the timestamps mean "when the data said this" rather than "when a job happened to run". They are merged with min and max, so loading snapshots out of order cannot move them the wrong way.

### D22. Fighter attributes are dropped when a profile's two names disagree
- **Context:** Upstream rewrites the fighter listing from the live site but captures height, reach, stance and date of birth once and never refreshes them. In four profiles the current name and the name the attributes were captured under share no word (for example a heavyweight listed with a 125 lbs weight and a 2004 birth date recorded under a woman's name).
- **Alternatives:** trust the attributes; drop the profile.
- **Why:** The attributes may belong to a different person, and age and reach are model features. The profile and its bouts are kept; the four attributes are left empty and a warning is recorded. More generally, D-ML rule 3 treats these attributes as static, and this source makes them static as of first capture, which is the same assumption with an earlier date.
