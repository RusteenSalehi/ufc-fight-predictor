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
