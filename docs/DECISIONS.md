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
- **What this means:** Downloading the CSVs from GitHub means this project makes no requests to UFCStats, but it does not settle whether the data may be used. Upstream's GPL-3.0 licence covers its repository; it cannot grant rights the data's owner has not given. This project uses the data for a non-commercial portfolio and commits none of it, except two small excerpts whose origin and licence are stated beside them: 29 fights in `tests/fixtures/greco/` and 4 fights in `tests/fixtures/greco_spot_check/`.
- **Consequences:** Commercial use stays a non-goal until licensing is resolved (see SPEC). If the operator or upstream objects, the source is swapped behind `src/ufc/ingest/`; nothing downstream reads the CSVs directly. The methodology page must say where the data comes from and repeat this caveat.

### D16. Wikipedia for upcoming cards
- **Alternatives:** UFCStats upcoming-event pages (blocked, D13); ufc.com event pages (its terms prohibit automated access outright); a paid schedule API.
- **Why:** The dataset has completed events only. Wikipedia's MediaWiki API is meant for programmatic use, its content is CC BY-SA, and event pages list bouts in a regular template (`{{MMAevent bout|weight class|fighter|vs.|fighter}}`). Confirmed on a live upcoming event; responses saved in `tests/fixtures/wikipedia/`.
- **Costs accepted:** Wikipedia is edited by volunteers and can be wrong, late, or vandalised, so M5's validation gate has to treat a card as untrusted input. It has no UFCStats IDs: upcoming bouts need their own IDs, and fighter names (with diacritics and disambiguation suffixes) go through the same name resolution as D17. The site must attribute Wikipedia.

### D17. Resolve fighter names to IDs with exact rules and explicit overrides; never fuzzy matching
- **Context:** Fighter profiles have IDs, but a fight row only names its two fighters. Some names in the snapshot belong to two profiles each (for example two fighters called Bruno Silva), and bout rows sometimes spell a name differently from the profile. At commit `96adace` the load reports 56 fighter names resolved by weight class or override.
- **Rule, in order:** an entry in `identity_overrides.csv`; a name (ignoring case, accents and punctuation) that belongs to exactly one profile, counting both the current listing name and the name recorded when the profile was first captured; for a shared name, the one profile whose listed weight fits the bout's weight class; otherwise unresolved.
- **Alternatives:** fuzzy string matching; picking the nearest weight; picking the most active fighter.
- **Why:** A wrong match silently gives one fighter another's history and corrupts every feature built on it. A reported gap is visible and fixable. Nearest-weight was rejected on evidence: for one lightweight bout it picks a profile born in 1977 with no other bouts over the active featherweight of the same name. Each override records the evidence from the dataset that justifies it, and every non-trivial resolution is written to `ingest_issues`. Two cross-checks pass on the full snapshot: no fighter is under 17 or over 55 at any bout, and no fighter appears in both a women's and a men's weight class.

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
- **Why:** The attributes may belong to a different person, and age and reach are model features. The profile and its bouts are kept; the four attributes are left empty and a warning is recorded. More generally, leakage rule 3 in ML_REQUIREMENTS treats these attributes as static, and this source makes them static as of first capture, which is the same assumption with an earlier date.

### D23. Features are built one date at a time from append-only fighter histories
- **Decision:** Bouts are processed in date order. For each date, features for all of its bouts are computed first, and only then are those bouts added to the store and Elo advanced. A fighter's state is a list of his bouts with running sums of one vector of additive counts per bout (strikes landed and attempted, seconds fought, wins, and so on, for him and for his opponent). Every read goes through one lookup that returns the bouts strictly before a cutoff date. `build_matchup` uses that lookup to price any pair on any date, and the batch build calls the same function.
- **Alternatives:** a SQL query per bout with `WHERE date < ?`; recomputing each history from raw rows for every bout; grouping by event instead of date; separate code for training features and upcoming-bout features.
- **Why:** The date is the unit because the rule is "strictly before D" and five dates in the data hold two events. There are two independent guards against the same mistake: in the batch build later bouts are not in the store yet, and the lookup would exclude them even if they were. One function for training and prediction means the two cannot drift apart, and a test checks that asking the finished store about any past bout reproduces the batch row exactly. Running sums make a full rebuild take seconds, so the pipeline is simply rerun from scratch; there is no incremental state to corrupt.

### D24. Rates are pooled sums; missing values are NaN with two flags per fighter
- **Decision:** A rate is total numerator over total denominator across the bouts in a window (career, last 3 bouts, last 5 bouts, or time-decayed with a 730-day half-life fixed in advance). A bout without recorded stats adds nothing to either side, so a window with no recorded stats has a zero denominator and the rate is NaN. Two flags per fighter say why: `debut` (no earlier UFC bout) and `no_stats` (no earlier bout with recorded stats). Record counts such as bouts and wins are true zeros for a debutant; stat counts and rates are NaN.
- **Alternatives:** averaging per-fight rates; filling missing stats with zero or a league mean; shrinking every rate toward a prior.
- **Why:** Pooling is how the sport reports these stats, it weights a 25-minute fight more than a 20-second one, and it can be checked by hand (the spot-check test does). Zero-filling would tell the model a debutant has no offense and perfect durability. Imputation is a fitted step and belongs in M4, on training data only.
- **Costs accepted:** A fighter with very little cage time has extreme per-minute values, and a ratio with no attempts (never faced a takedown) is NaN without a flag of its own. The differences in bouts and cage minutes tell a model how much evidence stands behind a rate; clipping, if wanted, is a fitted M4 step.
- **Conventions:** doctor's stoppages count as KO/TKO; a no contest is left out of win rates and does not break a streak; windows count a fighter's most recent bouts of any kind. Height, reach, stance and date of birth are treated as static (leakage rule 3, D22); age is computed at the bout date.

### D25. Opponent-adjusted stats: actual minus expected, with opponents' baselines taken at the same cutoff
- **Decision:** For fighter F at cutoff D, each earlier bout of F with stats is compared with its opponent's baseline: the opponent's own totals over his bouts strictly before D, leaving out his bouts against F, plus two league-average bouts as a prior. The baseline gives a rate (for example, the share of takedown attempts opponents usually land on him), which times F's actual exposure in the bout gives an expected count. The adjusted stat is (actual - expected) / exposure over F's history, in career and time-decayed versions. Ten stats are adjusted, including the takedown-defense example in ML_REQUIREMENTS.
- **How the cutoff is respected:** F's bouts and every opponent's bouts are read through the same strictly-before-D lookup, so nothing on or after D is used on either side. The league-average prior is fitted on the training period only.
- **Alternatives:** freezing each opponent's baseline at the date he fought F; including the bouts against F in his baseline; a ratio instead of a difference; solving for all fighters' offense and defense ratings at once.
- **Why:** A baseline frozen at fight time throws away what the opponent did afterwards, which is legitimate evidence about how good he was and is still before D; it also leaves most early-career opponents with no baseline. Bouts against F are excluded because they would make F's own performance part of the yardstick it is measured against. Differences stay finite when an expected value is near zero. A simultaneous solve would have to be rerun for every date to stay as-of and is hard to explain; this method is two lookups and can be checked by hand.
- **Costs accepted:** An opponent's baseline is his whole record before D, so one who has since declined looks weaker than he was when they fought. The prior strength (two bouts) is a judgment, not tuned.

### D26. Elo: one update per date, parameters fitted on the training period
- **Decision:** All bouts on a date are scored against ratings from before that date, and the changes are applied together afterwards. Draws score one half; no contests are not rated. Three parameters are chosen by grid search to minimise the log loss of pre-fight predictions on bouts through 2022-12-31: the step K, a multiplier for finishes, and a multiplier for a fighter's first three rated bouts. The build prints the chosen values and the training log loss; on snapshot `96adace` they are K = 56, finish 1.15, early-career 1.5 (`uv run python -m ufc.features build`).
- **Alternatives:** a conventional fixed K of 32; separate ratings per weight class; Glicko; using judges' margins.
- **Why:** Updating per date keeps the one-night tournaments of the 1990s, where a fighter fought several times in an evening, from leaking one bout into the next. Each prediction in the fitting loss uses earlier dates only, so it is an honest one-step-ahead score. One rating across weight classes keeps fighters who change class connected. The fighter's own rating difference is included as a feature and grouped with opponent quality (family 8), since both come from this system.
- **Note for M4:** ML_REQUIREMENTS also says the Elo baseline's K is tuned on validation. The ratings used as features never see validation data; a validation-tuned baseline would be a separate run of the same code.
- **Detail:** each fighter's expected score is computed from his own side. Computing one as one minus the other differs in the last bit and made ratings depend on which fighter the source lists first; the symmetry tests caught it.

### D27. Fitted parameters are inputs to the build, and the leakage tests treat them that way
- **Context:** Two rules are in play. The as-of rule says a feature for date D uses only earlier bouts. The fit-on-training rule lets Elo parameters and league baselines use the whole training period. So a 2015 feature legitimately depends on parameters chosen with bouts through 2022, exactly as it would on a scaler fitted on the training set.
- **Decision:** The parameters are an explicit input (`FittedParams`), saved in `features_meta.json`. The perturbation test has two forms. With parameters held fixed, changing or deleting every bout after any date D leaves all features on or before D exactly equal. With fitting included, changing every bout after the training period changes neither the parameters nor any training-period feature. The same-event test changes or deletes the other bouts on a card and requires the bout's row to be exactly equal.
- **Alternatives:** refitting parameters at every date so that nothing ever depends on a later bout.
- **Why:** Refitting per date would make the same stat mean different things in different years and is not what rule 4 asks for. Comparisons are exact rather than approximate, so a leak of any size fails.

### D28. Every model feature is antisymmetric; style matchups are products
- **Decision:** Each feature is `f(A) - f(B)` for a per-fighter value, or `m(A, B) - m(B, A)` for a pair value, so swapping the fighters negates it exactly (tested with exact equality, and by rebuilding with every bout's listing order reversed). Style matchups multiply one fighter's offense by the other's weakness: takedowns per 15 minutes times the share of takedowns the opponent fails to stop, strike attempts per minute times the share of strikes he fails to avoid, knockdowns scored times knockdowns absorbed, control share times control conceded, submission attempts times his rate of submission losses. Stance is the two fighters' southpaw and switch indicators differenced, plus a signed southpaw-versus-orthodox term.
- **Alternatives:** sums such as "A's accuracy minus B's defense"; keeping both fighters' raw values as separate columns; a categorical stance pair.
- **Why:** The source lists the winner first far more often than not (the mean of `label` in the feature file shows it), so any trace of listing order would leak the result. A sum of one fighter's offense and the other's defense is only a combination of difference features that already exist; a product is new information. A categorical "orthodox vs southpaw" changes value when the fighters swap, which breaks exact symmetry; the signed term carries the same fact.
- **Costs accepted:** Differences drop the level: two 38-year-olds look the same as two 25-year-olds. Only weight class, the bout format and the per-fighter flags are kept undifferenced. If M4 shows level matters, symmetric features (sums) can be added without touching the antisymmetric ones.

### D29. Family 12: cardio as pooled rounds-3+ rates over round-1 rates
- **Decision:** From per-round stats, over a fighter's earlier bouts that reached round 3: significant strikes landed, attempted and absorbed per second in rounds 3 and later, each divided by the same rate in round 1 (career and time-decayed), plus the change in accuracy and in defense, and the number of such bouts. Rounds before the last ran their full scheduled length; the last ran to the recorded end time. A fighter with no bout past round 2 has NaN for all of these except the count.
- **Alternatives:** averaging per-bout ratios; log ratios; comparing round 2 as well; counting bouts that ended in round 2.
- **Why:** Sums are pooled across bouts before dividing, so one bout with nothing landed in round 1 does not produce a division by zero or a wild ratio by itself. A plain ratio handles "landed nothing late" as zero, where a log ratio is undefined. Round 2 is left out to keep the contrast between fresh and tired.
- **Costs accepted:** Missing for every fighter without a long bout. With a single qualifying bout the ratio is noisy and can be large; the count is provided so a model can discount it. The sample is selected: only bouts a fighter did not finish (or lose) early are measured.
