-- SQLite schema. IDs are the 16-character hex IDs from UFCStats URLs.
-- Dates and timestamps are ISO-8601 text. Booleans are 0/1 integers.
-- Every statement is idempotent so the schema can be applied on each run.

CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY CHECK (length(event_id) = 16),
    name          TEXT NOT NULL,
    date          TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    location      TEXT,
    status        TEXT NOT NULL CHECK (status IN ('completed', 'upcoming')),
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);

-- Physical attributes only. Career aggregates are never stored: they reflect
-- today's values and would leak the future into historical features.
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id TEXT PRIMARY KEY CHECK (length(fighter_id) = 16),
    name       TEXT NOT NULL,
    nickname   TEXT,
    height_cm  REAL,
    reach_cm   REAL,
    stance     TEXT,
    dob        TEXT
);

-- fighter_a / fighter_b keep the order the source lists them in. That order
-- is for storage only and is never a model feature.
CREATE TABLE IF NOT EXISTS bouts (
    bout_id             TEXT PRIMARY KEY CHECK (length(bout_id) = 16),
    event_id            TEXT NOT NULL REFERENCES events (event_id),
    fighter_a_id        TEXT NOT NULL REFERENCES fighters (fighter_id),
    fighter_b_id        TEXT NOT NULL REFERENCES fighters (fighter_id),
    weight_class        TEXT,
    bout_type_raw       TEXT NOT NULL,
    is_womens           INTEGER NOT NULL CHECK (is_womens IN (0, 1)),
    is_title_bout       INTEGER NOT NULL CHECK (is_title_bout IN (0, 1)),
    is_interim_title    INTEGER NOT NULL CHECK (is_interim_title IN (0, 1)),
    is_tournament_final INTEGER NOT NULL CHECK (is_tournament_final IN (0, 1)),
    scheduled_rounds    INTEGER,
    time_format         TEXT,
    status              TEXT NOT NULL CHECK (status IN ('scheduled', 'completed', 'cancelled')),
    result              TEXT CHECK (result IN ('win', 'draw', 'nc')),
    winner_id           TEXT REFERENCES fighters (fighter_id),
    method              TEXT,
    method_details      TEXT,
    referee             TEXT,
    end_round           INTEGER,
    end_time_s          INTEGER,
    fight_time_s        INTEGER,
    has_stats           INTEGER NOT NULL CHECK (has_stats IN (0, 1)),
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    CHECK (fighter_a_id <> fighter_b_id),
    CHECK ((result = 'win') = (winner_id IS NOT NULL)),
    CHECK (winner_id IS NULL OR winner_id IN (fighter_a_id, fighter_b_id)),
    CHECK (status <> 'completed' OR result IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_bouts_event ON bouts (event_id);
CREATE INDEX IF NOT EXISTS idx_bouts_fighter_a ON bouts (fighter_a_id);
CREATE INDEX IF NOT EXISTS idx_bouts_fighter_b ON bouts (fighter_b_id);

-- One row per fighter per round, as published. ctrl_s is NULL where control
-- time was not recorded. A bout with has_stats = 0 has no rows here.
CREATE TABLE IF NOT EXISTS bout_round_stats (
    bout_id             TEXT NOT NULL REFERENCES bouts (bout_id),
    fighter_id          TEXT NOT NULL REFERENCES fighters (fighter_id),
    round               INTEGER NOT NULL CHECK (round >= 1),
    knockdowns          INTEGER NOT NULL,
    sig_str_landed      INTEGER NOT NULL,
    sig_str_attempted   INTEGER NOT NULL,
    total_str_landed    INTEGER NOT NULL,
    total_str_attempted INTEGER NOT NULL,
    td_landed           INTEGER NOT NULL,
    td_attempted        INTEGER NOT NULL,
    sub_attempts        INTEGER NOT NULL,
    reversals           INTEGER NOT NULL,
    ctrl_s              INTEGER,
    head_landed         INTEGER NOT NULL,
    head_attempted      INTEGER NOT NULL,
    body_landed         INTEGER NOT NULL,
    body_attempted      INTEGER NOT NULL,
    leg_landed          INTEGER NOT NULL,
    leg_attempted       INTEGER NOT NULL,
    distance_landed     INTEGER NOT NULL,
    distance_attempted  INTEGER NOT NULL,
    clinch_landed       INTEGER NOT NULL,
    clinch_attempted    INTEGER NOT NULL,
    ground_landed       INTEGER NOT NULL,
    ground_attempted    INTEGER NOT NULL,
    PRIMARY KEY (bout_id, fighter_id, round)
);

-- Per-fight totals, derived at load time by summing bout_round_stats.
-- ctrl_s is NULL if it is NULL for any round of the fight.
CREATE TABLE IF NOT EXISTS bout_fighter_stats (
    bout_id             TEXT NOT NULL REFERENCES bouts (bout_id),
    fighter_id          TEXT NOT NULL REFERENCES fighters (fighter_id),
    knockdowns          INTEGER NOT NULL,
    sig_str_landed      INTEGER NOT NULL,
    sig_str_attempted   INTEGER NOT NULL,
    total_str_landed    INTEGER NOT NULL,
    total_str_attempted INTEGER NOT NULL,
    td_landed           INTEGER NOT NULL,
    td_attempted        INTEGER NOT NULL,
    sub_attempts        INTEGER NOT NULL,
    reversals           INTEGER NOT NULL,
    ctrl_s              INTEGER,
    head_landed         INTEGER NOT NULL,
    head_attempted      INTEGER NOT NULL,
    body_landed         INTEGER NOT NULL,
    body_attempted      INTEGER NOT NULL,
    leg_landed          INTEGER NOT NULL,
    leg_attempted       INTEGER NOT NULL,
    distance_landed     INTEGER NOT NULL,
    distance_attempted  INTEGER NOT NULL,
    clinch_landed       INTEGER NOT NULL,
    clinch_attempted    INTEGER NOT NULL,
    ground_landed       INTEGER NOT NULL,
    ground_attempted    INTEGER NOT NULL,
    PRIMARY KEY (bout_id, fighter_id)
);

-- Which snapshot of the published dataset the tables were last loaded from.
-- Replaced on every load.
CREATE TABLE IF NOT EXISTS source_files (
    file_name       TEXT PRIMARY KEY,
    sha256          TEXT NOT NULL,
    row_count       INTEGER NOT NULL,
    upstream_repo   TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    snapshot_at     TEXT NOT NULL
);

-- The validation report of the last load. Replaced on every load.
-- severity: error = data dropped, warning = loaded but altered or suspect,
-- info = summary or audit note.
CREATE TABLE IF NOT EXISTS ingest_issues (
    seq         INTEGER PRIMARY KEY,
    severity    TEXT NOT NULL CHECK (severity IN ('error', 'warning', 'info')),
    check_name  TEXT NOT NULL,
    entity_type TEXT,
    entity_id   TEXT,
    message     TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Upcoming cards and predictions (M5). Read from English Wikipedia, which has
-- no UFCStats IDs, so these tables have keys of their own and never write to
-- the tables above. DECISIONS D48 to D53.

-- One row per event page. The key is the article's page ID, which survives
-- the page being renamed when a headliner changes.
CREATE TABLE IF NOT EXISTS upcoming_events (
    event_key       TEXT PRIMARY KEY CHECK (event_key GLOB 'wp:[0-9]*'),
    pageid          INTEGER NOT NULL UNIQUE,
    page_title      TEXT NOT NULL,
    name            TEXT NOT NULL,
    date            TEXT NOT NULL CHECK (date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    revid           INTEGER,
    status          TEXT NOT NULL CHECK (status IN ('scheduled', 'past', 'unlisted')),
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    card_applied_at TEXT  -- last refresh whose parsed card was applied
);

-- One row per pairing on an event. bout_id is derived from the event's page
-- ID and the two fighters' normalised names, sorted, so it is the same on
-- every refresh and a new opponent makes a new bout. fighter_*_ref is a
-- UFCStats fighter ID or a provisional_id; both are NULL when the bout is
-- unresolved, and unresolved_reason says why.
CREATE TABLE IF NOT EXISTS upcoming_bouts (
    bout_id           TEXT PRIMARY KEY CHECK (bout_id GLOB 'wb[0-9a-f]*' AND length(bout_id) = 18),
    event_key         TEXT NOT NULL REFERENCES upcoming_events (event_key),
    card_order        INTEGER NOT NULL,
    segment           TEXT,
    weight_class_raw  TEXT NOT NULL,
    weight_class      TEXT,
    is_title_bout     INTEGER NOT NULL CHECK (is_title_bout IN (0, 1)),
    name_a            TEXT NOT NULL,
    page_a            TEXT,
    name_b            TEXT NOT NULL,
    page_b            TEXT,
    fighter_a_ref     TEXT,
    fighter_b_ref     TEXT,
    method_a          TEXT NOT NULL,
    method_b          TEXT NOT NULL,
    unresolved_reason TEXT,
    status            TEXT NOT NULL CHECK (status IN ('scheduled', 'cancelled')),
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    cancelled_at      TEXT,
    CHECK ((unresolved_reason IS NULL) = (fighter_a_ref IS NOT NULL AND fighter_b_ref IS NOT NULL)),
    CHECK (status = 'cancelled' OR cancelled_at IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_upcoming_bouts_event ON upcoming_bouts (event_key);

-- A fighter on a card whose name matches no profile in the dataset: almost
-- always a debutant. Keyed by Wikipedia page title ('wp:<title>'), or by the
-- normalised name for a fighter without an article ('wpname:<name>').
-- fighter_id is set once the name resolves to a UFCStats profile.
CREATE TABLE IF NOT EXISTS provisional_fighters (
    provisional_id TEXT PRIMARY KEY CHECK (provisional_id GLOB 'wp:*' OR provisional_id GLOB 'wpname:*'),
    page_title     TEXT,
    name           TEXT NOT NULL,
    fighter_id     TEXT,
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    linked_at      TEXT
);

-- Height, reach and date of birth taken from a fighter's Wikipedia infobox
-- where the dataset has none. One row per fighter and field, with its source.
-- The fighters table is never written: an upstream value always wins.
CREATE TABLE IF NOT EXISTS attribute_backfill (
    fighter_ref TEXT NOT NULL,
    field       TEXT NOT NULL CHECK (field IN ('height_cm', 'reach_cm', 'dob')),
    value       TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('wikipedia')),
    page_title  TEXT NOT NULL,
    revid       INTEGER,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (fighter_ref, field)
);

-- Predictions for upcoming bouts. A row is never changed except to set
-- superseded_at once: when its bout is cancelled or becomes unresolved, or a
-- newer prediction replaces it (different model or different inputs). At
-- most one prediction per bout is current. is_live = 1 when it was made
-- before the event date.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         INTEGER PRIMARY KEY,
    bout_id               TEXT NOT NULL REFERENCES upcoming_bouts (bout_id),
    model_version         TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    event_date            TEXT NOT NULL,
    fighter_a_ref         TEXT NOT NULL,
    fighter_b_ref         TEXT NOT NULL,
    p_fighter_a           REAL NOT NULL CHECK (p_fighter_a BETWEEN 0 AND 1),
    p_fighter_b           REAL NOT NULL CHECK (p_fighter_b BETWEEN 0 AND 1),
    inputs_sha256         TEXT NOT NULL,
    features_json         TEXT NOT NULL,
    shap_json             TEXT,
    limited_data          INTEGER NOT NULL CHECK (limited_data IN (0, 1)),
    limited_reasons_json  TEXT NOT NULL,
    is_live               INTEGER NOT NULL CHECK (is_live IN (0, 1)),
    superseded_at         TEXT,
    superseded_reason     TEXT,
    CHECK ((superseded_at IS NULL) = (superseded_reason IS NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_current
    ON predictions (bout_id) WHERE superseded_at IS NULL;

-- What each refresh did and what went wrong. Append-only.
CREATE TABLE IF NOT EXISTS refresh_log (
    seq       INTEGER PRIMARY KEY,
    run_at    TEXT NOT NULL,
    level     TEXT NOT NULL CHECK (level IN ('error', 'warning', 'info')),
    event_key TEXT,
    message   TEXT NOT NULL
);

