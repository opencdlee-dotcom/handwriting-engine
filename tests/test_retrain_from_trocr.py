"""Tests for the Stage 3 retrain pipeline (`trained-correction retrain-from-trocr`).

The pipeline gates on a minimum (TrOCR_output, ground_truth) pair count, then
delegates to `train.main`. Tests mock the trainer so no actual fine-tuning
happens — we only verify the gate logic and the argv shape passed to train.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def empty_db(tmp_path: Path) -> str:
    """Spin up an empty benchmark DB and return its path."""
    from handwriting_engine.benchmark.db import get_connection

    db_path = tmp_path / "benchmark.db"
    conn = get_connection(db_path)
    conn.close()
    return str(db_path)


def _seed_pair(conn: sqlite3.Connection, sample_idx: int, corrupted: str, clean: str) -> None:
    """Insert one (sample, ground_truth, run, provider_output) tuple matching
    the benchmark schema.

    Each pair gets a unique image_hash so dedup doesn't merge them.
    """
    conn.execute(
        "INSERT INTO samples (image_path, image_hash) VALUES (?, ?)",
        (f"/tmp/fake_{sample_idx}.png", f"hash_{sample_idx}"),
    )
    sample_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO ground_truths (sample_id, text) VALUES (?, ?)",
        (sample_id, clean),
    )

    # Reuse a single run row across pairs — fine for these tests.
    run_row = conn.execute("SELECT id FROM runs LIMIT 1").fetchone()
    if run_row is None:
        conn.execute(
            "INSERT INTO runs (label, providers, strategies) "
            "VALUES (?, ?, ?)",
            ("test", '["trocr"]', '["single"]'),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    else:
        run_id = run_row[0]

    conn.execute(
        "INSERT INTO provider_outputs (run_id, sample_id, provider, strategy, output_text, error) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        (run_id, sample_id, "trocr", "single", corrupted),
    )


@pytest.fixture
def db_with_n_pairs(tmp_path: Path):
    """Factory: returns a db_path containing N distinct TrOCR (corrupted, clean) pairs."""
    from handwriting_engine.benchmark.db import get_connection

    def _make(n: int) -> str:
        db_path = tmp_path / f"bench_{n}.db"
        conn = get_connection(db_path)
        try:
            for i in range(n):
                # 12+ chars so they survive the min_text_chars=10 filter.
                _seed_pair(
                    conn,
                    sample_idx=i,
                    corrupted=f"corrputed text number {i:03d}",
                    clean=f"corrupted text number {i:03d}",
                )
            conn.commit()
        finally:
            conn.close()
        return str(db_path)

    return _make


# =====================================================================
# Gate: empty DB
# =====================================================================

def test_empty_db_exits_1_with_remediation(empty_db):
    from handwriting_engine.trained_correction.retrain_from_trocr import retrain_from_trocr

    err = io.StringIO()
    train_main = MagicMock()
    rc = retrain_from_trocr(
        db_path=empty_db,
        train_main=train_main,
        out_stream=io.StringIO(),
        err_stream=err,
    )
    assert rc == 1
    msg = err.getvalue()
    assert "Only 0 TrOCR" in msg
    # Remediation must name at least one of the two acquisition entry points.
    assert "bench-my-pages" in msg or "benchmark sweep" in msg
    train_main.assert_not_called()


# =====================================================================
# Gate: under threshold
# =====================================================================

def test_db_with_too_few_pairs_exits_1_with_count_in_message(db_with_n_pairs):
    from handwriting_engine.trained_correction.retrain_from_trocr import retrain_from_trocr

    db_path = db_with_n_pairs(3)  # well under default threshold of 50
    err = io.StringIO()
    train_main = MagicMock()
    rc = retrain_from_trocr(
        db_path=db_path,
        train_main=train_main,
        out_stream=io.StringIO(),
        err_stream=err,
    )
    assert rc == 1
    assert "Only 3 TrOCR" in err.getvalue()
    train_main.assert_not_called()


def test_pairs_min_kwarg_lowers_threshold(db_with_n_pairs, tmp_path):
    """Tests can drop pairs_min so the gate doesn't block on tiny DBs."""
    from handwriting_engine.trained_correction.retrain_from_trocr import retrain_from_trocr

    db_path = db_with_n_pairs(3)
    train_main = MagicMock(return_value=0)

    rc = retrain_from_trocr(
        db_path=db_path,
        v1_path=tmp_path / "v1",
        v3_path=tmp_path / "v3",
        pairs_min=2,
        run_eval=False,  # no eval — checkpoints don't actually exist
        train_main=train_main,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    )
    assert rc == 0
    train_main.assert_called_once()


# =====================================================================
# Happy path: enough pairs → invokes train with correct argv
# =====================================================================

def test_sufficient_pairs_invokes_train_with_continue_from_and_output(db_with_n_pairs, tmp_path):
    from handwriting_engine.trained_correction.retrain_from_trocr import retrain_from_trocr

    db_path = db_with_n_pairs(5)
    v1 = tmp_path / "checkpoints" / "v1"
    v3 = tmp_path / "checkpoints" / "v3"
    train_main = MagicMock(return_value=0)

    rc = retrain_from_trocr(
        db_path=db_path,
        v1_path=v1,
        v3_path=v3,
        num_epochs=3,
        pairs_min=2,
        run_eval=False,
        train_main=train_main,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    )
    assert rc == 0
    train_main.assert_called_once()

    argv = train_main.call_args[0][0]
    # Sanity: all the load-bearing flags made it through
    assert "--from-benchmark-db" in argv
    assert "--continue-from" in argv
    assert str(v1) in argv
    assert "--output-dir" in argv
    assert str(v3) in argv
    assert "--num-epochs" in argv
    assert "3" in argv
    # Provider filter must pin trocr — otherwise we'd train on cloud-VLM pairs too
    assert "--benchmark-providers" in argv
    bp_idx = argv.index("--benchmark-providers")
    assert argv[bp_idx + 1] == "trocr"
    # DB path passthrough
    assert "--benchmark-db-path" in argv
    db_idx = argv.index("--benchmark-db-path")
    assert argv[db_idx + 1] == db_path


def test_train_failure_returns_2(db_with_n_pairs, tmp_path):
    from handwriting_engine.trained_correction.retrain_from_trocr import retrain_from_trocr

    db_path = db_with_n_pairs(3)
    train_main = MagicMock(return_value=42)  # non-zero = failure

    rc = retrain_from_trocr(
        db_path=db_path,
        v1_path=tmp_path / "v1",
        v3_path=tmp_path / "v3",
        pairs_min=2,
        run_eval=False,
        train_main=train_main,
        out_stream=io.StringIO(),
        err_stream=io.StringIO(),
    )
    assert rc == 2


# =====================================================================
# CLI smoke: --help works
# =====================================================================

def test_cli_help_works():
    """The Click subcommand registers and renders help without error."""
    from click.testing import CliRunner
    from handwriting_engine.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["trained-correction", "retrain-from-trocr", "--help"])
    assert result.exit_code == 0
    assert "retrain" in result.output.lower()
    assert "--pairs-min" in result.output
