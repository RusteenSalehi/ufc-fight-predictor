"""SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database with foreign keys enforced and the schema applied.

    ``path`` may be ``":memory:"``. Parent directories are created.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply ``schema.sql``. Safe to call on an existing database."""
    schema = (
        resources.files("ufc.db").joinpath("schema.sql").read_text(encoding="utf-8")
    )
    conn.executescript(schema)
    conn.commit()
