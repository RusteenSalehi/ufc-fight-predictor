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
