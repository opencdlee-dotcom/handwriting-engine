"""
SQLite database for benchmark ground truth, provider outputs, and evaluation metrics.

Schema is auto-created on first connection. Migrations via schema_version table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from handwriting_engine.benchmark.models import (
    EvalMetric,
    GroundTruth,
    ProviderOutput,
    RunSummary,
    Sample,
)

DEFAULT_DB_PATH = Path.home() / ".handwriting-engine" / "benchmark.db"
CURRENT_SCHEMA_VERSION = 5

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path  TEXT NOT NULL,
    image_hash  TEXT NOT NULL UNIQUE,
    student     TEXT DEFAULT '',
    category    TEXT DEFAULT '',
    source_dir  TEXT DEFAULT '',
    page_number INTEGER DEFAULT 0,
    notes       TEXT DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS ground_truths (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   INTEGER NOT NULL REFERENCES samples(id),
    text        TEXT NOT NULL,
    source      TEXT DEFAULT 'manual',
    author      TEXT DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_gt_sample ON ground_truths(sample_id);

CREATE TABLE IF NOT EXISTS quality_assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id       INTEGER NOT NULL UNIQUE REFERENCES samples(id),
    blur_score      REAL,
    contrast_score  REAL,
    brightness_mean REAL,
    brightness_std  REAL,
    quality         TEXT,
    issues          TEXT DEFAULT '[]',
    faint_ink       INTEGER DEFAULT 0,
    assessed_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    label            TEXT DEFAULT '',
    providers        TEXT NOT NULL DEFAULT '[]',
    strategies       TEXT NOT NULL DEFAULT '[]',
    domain           TEXT DEFAULT 'biology',
    started_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at      TEXT,
    sample_count     INTEGER DEFAULT 0,
    notes            TEXT DEFAULT '',
    model_version    TEXT DEFAULT NULL,
    iam_partition    TEXT DEFAULT NULL,
    norm_flags       TEXT DEFAULT NULL,
    vocab_hints_off  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS provider_outputs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES runs(id),
    sample_id            INTEGER NOT NULL REFERENCES samples(id),
    provider             TEXT NOT NULL,
    strategy             TEXT NOT NULL DEFAULT 'single',
    output_text          TEXT NOT NULL,
    confidence           REAL DEFAULT 0.0,
    latency_ms           INTEGER DEFAULT 0,
    input_tokens         INTEGER DEFAULT 0,
    output_tokens        INTEGER DEFAULT 0,
    error                TEXT,
    created_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    question_marker_rate REAL DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_po_run ON provider_outputs(run_id);
CREATE INDEX IF NOT EXISTS idx_po_sample ON provider_outputs(sample_id);

CREATE TABLE IF NOT EXISTS eval_metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_output_id INTEGER NOT NULL REFERENCES provider_outputs(id),
    ground_truth_id    INTEGER NOT NULL REFERENCES ground_truths(id),
    cer                REAL NOT NULL,
    wer                REAL NOT NULL,
    char_edits         INTEGER DEFAULT 0,
    word_edits         INTEGER DEFAULT 0,
    ref_chars          INTEGER DEFAULT 0,
    ref_words          INTEGER DEFAULT 0,
    computed_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_em_output ON eval_metrics(provider_output_id);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id       INTEGER NOT NULL REFERENCES samples(id),
    ground_truth_id INTEGER NOT NULL REFERENCES ground_truths(id),
    original_text   TEXT NOT NULL,
    confidence      REAL,
    source          TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_corrections_sample ON corrections(sample_id);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


# Schema migrations — keyed by version number, SQL to upgrade FROM previous version
_MIGRATIONS: dict[int, str] = {
    # v1 is the base schema (already created by _SCHEMA_SQL)
    # v2: Add enhancement_strategy and cache_hit columns to provider_outputs
    2: """
        ALTER TABLE provider_outputs ADD COLUMN enhancement_strategy TEXT DEFAULT NULL;
        ALTER TABLE provider_outputs ADD COLUMN cache_hit INTEGER DEFAULT 0;
        UPDATE schema_version SET version = 2;
    """,
    # v3: Add cost_usd column to runs
    3: """
        ALTER TABLE runs ADD COLUMN total_cost_usd REAL DEFAULT 0.0;
        UPDATE schema_version SET version = 3;
    """,
    # v4: Add provenance columns to runs + marker rate to provider_outputs
    4: """
        ALTER TABLE runs ADD COLUMN model_version TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN iam_partition TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN norm_flags TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN vocab_hints_off INTEGER DEFAULT 0;
        ALTER TABLE provider_outputs ADD COLUMN question_marker_rate REAL DEFAULT NULL;
        UPDATE schema_version SET version = 4;
    """,
    # v5: Add corrections table for instructor-corrected transcriptions (S4)
    5: """
        CREATE TABLE IF NOT EXISTS corrections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id       INTEGER NOT NULL REFERENCES samples(id),
            ground_truth_id INTEGER NOT NULL REFERENCES ground_truths(id),
            original_text   TEXT NOT NULL,
            confidence      REAL,
            source          TEXT NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_corrections_sample ON corrections(sample_id);
        UPDATE schema_version SET version = 5;
    """,
}


def _apply_migrations(conn):
    """Apply any pending schema migrations."""
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row[0] if row else 1
    except Exception:
        current = 1

    for version in sorted(_MIGRATIONS.keys()):
        if version > current:
            logger.info(f"Applying schema migration v{current} -> v{version}")
            try:
                conn.executescript(_MIGRATIONS[version])
                logger.info(f"Migration to v{version} complete")
            except Exception as e:
                logger.warning(f"Migration to v{version} failed (may already be applied): {e}")
                # Try to update version anyway if columns already exist
                try:
                    conn.execute("UPDATE schema_version SET version = ?", (version,))
                    conn.commit()
                except Exception:
                    pass


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the benchmark database, creating schema if needed.

    Args:
        db_path: Override path. Use ':memory:' for in-memory testing.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript(_SCHEMA_SQL)
    # Seed schema version if empty
    row = conn.execute(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (CURRENT_SCHEMA_VERSION,),
        )
        conn.commit()
    _apply_migrations(conn)


# --- Samples ---


def insert_sample(
    conn: sqlite3.Connection,
    image_path: str,
    image_hash: str,
    student: str = "",
    category: str = "",
    source_dir: str = "",
    page_number: int = 0,
    notes: str = "",
    autocommit: bool = True,
) -> int:
    """Insert a sample. Returns its ID. Raises IntegrityError on duplicate hash."""
    cur = conn.execute(
        """INSERT INTO samples (image_path, image_hash, student, category, source_dir, page_number, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (image_path, image_hash, student, category, source_dir, page_number, notes),
    )
    if autocommit:
        conn.commit()
    return cur.lastrowid


def get_sample_by_hash(conn: sqlite3.Connection, image_hash: str) -> Sample | None:
    """Look up a sample by content hash."""
    row = conn.execute(
        "SELECT * FROM samples WHERE image_hash = ?", (image_hash,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_sample(conn, row)


def get_sample_by_id(conn: sqlite3.Connection, sample_id: int) -> Sample | None:
    """Look up a sample by ID."""
    row = conn.execute(
        "SELECT * FROM samples WHERE id = ?", (sample_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_sample(conn, row)


def list_samples(conn: sqlite3.Connection) -> list[Sample]:
    """Return all samples with ground-truth status (single query, no N+1)."""
    rows = conn.execute(
        """SELECT s.*, (EXISTS(SELECT 1 FROM ground_truths gt WHERE gt.sample_id = s.id)) AS has_gt
           FROM samples s ORDER BY s.id"""
    ).fetchall()
    return [_row_to_sample(conn, r, has_gt=bool(r["has_gt"])) for r in rows]


def samples_with_ground_truth(conn: sqlite3.Connection) -> list[Sample]:
    """Return samples that have at least one ground truth transcription."""
    rows = conn.execute(
        """SELECT DISTINCT s.* FROM samples s
           JOIN ground_truths gt ON gt.sample_id = s.id
           ORDER BY s.id"""
    ).fetchall()
    return [_row_to_sample(conn, r, has_gt=True) for r in rows]


def _row_to_sample(
    conn: sqlite3.Connection, row: sqlite3.Row, has_gt: bool | None = None
) -> Sample:
    if has_gt is None:
        gt_row = conn.execute(
            "SELECT 1 FROM ground_truths WHERE sample_id = ? LIMIT 1", (row["id"],)
        ).fetchone()
        has_gt = gt_row is not None
    return Sample(
        id=row["id"],
        image_path=row["image_path"],
        image_hash=row["image_hash"],
        student=row["student"],
        category=row["category"],
        page_number=row["page_number"],
        notes=row["notes"],
        has_ground_truth=has_gt,
    )


# --- Ground Truths ---


def insert_ground_truth(
    conn: sqlite3.Connection,
    sample_id: int,
    text: str,
    source: str = "manual",
    author: str = "",
) -> int:
    """Add a ground truth transcription. Returns its ID."""
    cur = conn.execute(
        """INSERT INTO ground_truths (sample_id, text, source, author)
           VALUES (?, ?, ?, ?)""",
        (sample_id, text, source, author),
    )
    conn.commit()
    return cur.lastrowid


def get_latest_ground_truth(
    conn: sqlite3.Connection, sample_id: int
) -> GroundTruth | None:
    """Get the most recent ground truth for a sample."""
    row = conn.execute(
        """SELECT * FROM ground_truths WHERE sample_id = ?
           ORDER BY id DESC LIMIT 1""",
        (sample_id,),
    ).fetchone()
    if row is None:
        return None
    return GroundTruth(
        id=row["id"],
        sample_id=row["sample_id"],
        text=row["text"],
        source=row["source"],
        created_at=row["created_at"],
    )


# --- Quality Assessments ---


def insert_quality_assessment(
    conn: sqlite3.Connection,
    sample_id: int,
    assessment: dict,
) -> None:
    """Store a quality assessment for a sample (upsert)."""
    conn.execute(
        """INSERT INTO quality_assessments
               (sample_id, blur_score, contrast_score, brightness_mean, brightness_std,
                quality, issues, faint_ink)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(sample_id) DO UPDATE SET
               blur_score=excluded.blur_score,
               contrast_score=excluded.contrast_score,
               brightness_mean=excluded.brightness_mean,
               brightness_std=excluded.brightness_std,
               quality=excluded.quality,
               issues=excluded.issues,
               faint_ink=excluded.faint_ink,
               assessed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
        (
            sample_id,
            assessment.get("blur_score", 0),
            assessment.get("contrast_score", 0),
            assessment.get("brightness_mean", 0),
            assessment.get("brightness_std", 0),
            assessment.get("quality", ""),
            json.dumps(assessment.get("issues", [])),
            int(assessment.get("faint_ink", False)),
        ),
    )
    conn.commit()


# --- Runs ---


def insert_run(
    conn: sqlite3.Connection,
    label: str = "",
    providers: list[str] | None = None,
    strategies: list[str] | None = None,
    domain: str = "biology",
    model_version: str | None = None,
    iam_partition: str | None = None,
    norm_flags: str | None = None,
    vocab_hints_off: int = 0,
) -> int:
    """Create a new benchmark run. Returns its ID."""
    cur = conn.execute(
        """INSERT INTO runs (label, providers, strategies, domain,
                             model_version, iam_partition, norm_flags, vocab_hints_off)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (label, json.dumps(providers or []), json.dumps(strategies or []), domain,
         model_version, iam_partition, norm_flags, vocab_hints_off),
    )
    conn.commit()
    return cur.lastrowid


def finish_run(
    conn: sqlite3.Connection, run_id: int, sample_count: int
) -> None:
    """Mark a run as finished."""
    conn.execute(
        """UPDATE runs SET finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
               sample_count = ? WHERE id = ?""",
        (sample_count, run_id),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection) -> list[RunSummary]:
    """Return all runs (without detailed results)."""
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    results = []
    for r in rows:
        keys = r.keys()
        results.append(RunSummary(
            run_id=r["id"],
            label=r["label"],
            started_at=r["started_at"],
            finished_at=r["finished_at"] or "",
            sample_count=r["sample_count"],
            model_version=r["model_version"] if "model_version" in keys else None,
            iam_partition=r["iam_partition"] if "iam_partition" in keys else None,
            norm_flags=r["norm_flags"] if "norm_flags" in keys else None,
            vocab_hints_off=r["vocab_hints_off"] if "vocab_hints_off" in keys else 0,
        ))
    return results


# --- Provider Outputs ---


def insert_provider_output(
    conn: sqlite3.Connection,
    run_id: int,
    sample_id: int,
    provider: str,
    strategy: str,
    output_text: str,
    confidence: float = 0.0,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    error: str | None = None,
    question_marker_rate: float | None = None,
    autocommit: bool = True,
) -> int:
    """Store a provider output. Returns its ID."""
    cur = conn.execute(
        """INSERT INTO provider_outputs
               (run_id, sample_id, provider, strategy, output_text, confidence,
                latency_ms, input_tokens, output_tokens, error, question_marker_rate)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, sample_id, provider, strategy, output_text, confidence,
            latency_ms, input_tokens, output_tokens, error, question_marker_rate,
        ),
    )
    if autocommit:
        conn.commit()
    return cur.lastrowid


# --- Eval Metrics ---


def insert_eval_metric(
    conn: sqlite3.Connection,
    provider_output_id: int,
    ground_truth_id: int,
    cer: float,
    wer: float,
    char_edits: int = 0,
    word_edits: int = 0,
    ref_chars: int = 0,
    ref_words: int = 0,
    autocommit: bool = True,
) -> int:
    """Store computed CER/WER metrics. Returns its ID."""
    cur = conn.execute(
        """INSERT INTO eval_metrics
               (provider_output_id, ground_truth_id, cer, wer,
                char_edits, word_edits, ref_chars, ref_words)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (provider_output_id, ground_truth_id, cer, wer, char_edits, word_edits, ref_chars, ref_words),
    )
    if autocommit:
        conn.commit()
    return cur.lastrowid


def get_run_results(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    """Fetch all outputs + metrics for a run, joined."""
    rows = conn.execute(
        """SELECT po.*, em.cer, em.wer, em.char_edits, em.word_edits,
                  em.ref_chars, em.ref_words, em.ground_truth_id
           FROM provider_outputs po
           LEFT JOIN eval_metrics em ON em.provider_output_id = po.id
           WHERE po.run_id = ?
           ORDER BY po.sample_id, po.provider, po.strategy""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_run_id(conn: sqlite3.Connection) -> int | None:
    """Get the most recent run ID."""
    row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


# --- Corrections (S4: instructor feedback loop) ---


def record_correction(
    conn: sqlite3.Connection,
    *,
    image_path: str,
    writer_id: str,
    corrected_text: str,
    original_vlm_text: str,
    confidence: float,
    source: str = "labgrader",
) -> int:
    """Idempotently record an instructor-corrected transcription.

    Behavior with identical args called twice:
    - One sample row (deduped by image content hash).
    - One ground_truth row (deduped by sample_id + text).
    - Two corrections rows (history grows; each call records a confirmation).

    Returns the corrections row id from this call.
    """
    from handwriting_engine.benchmark.ingest import hash_file

    image_hash = hash_file(image_path)

    sample = get_sample_by_hash(conn, image_hash)
    if sample is None:
        sample_id = insert_sample(
            conn,
            image_path=image_path,
            image_hash=image_hash,
            student=writer_id,
        )
    else:
        sample_id = sample.id

    gt_row = conn.execute(
        "SELECT id FROM ground_truths WHERE sample_id = ? AND text = ? ORDER BY id DESC LIMIT 1",
        (sample_id, corrected_text),
    ).fetchone()
    if gt_row is None:
        gt_id = insert_ground_truth(conn, sample_id, corrected_text, source=source)
    else:
        gt_id = gt_row["id"]

    cur = conn.execute(
        """INSERT INTO corrections
               (sample_id, ground_truth_id, original_text, confidence, source)
           VALUES (?, ?, ?, ?, ?)""",
        (sample_id, gt_id, original_vlm_text, confidence, source),
    )
    conn.commit()
    return cur.lastrowid
