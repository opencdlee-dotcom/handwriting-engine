"""Phase 9 / RPT-01 — pinned-baseline regression detection.

Covers:
- Schema v6: is_baseline column on runs (durable across sessions).
- set_baseline / get_baseline_run_id atomicity (at-most-one).
- detect_regressions retargets to the pinned baseline rather than runs[1].
- CLI: `benchmark set-baseline RUN_ID`.
"""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from click.testing import CliRunner

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_sample,
    insert_ground_truth,
    insert_run,
    finish_run,
    insert_provider_output,
    insert_eval_metric,
    set_baseline,
    get_baseline_run_id,
    list_runs,
)
from handwriting_engine.benchmark.report import detect_regressions
from handwriting_engine.cli import cli


def _seed_db(db_path: Path, n_samples: int):
    """Build a DB with n_samples + matching ground-truth rows.

    Returns (sample_ids, gt_ids).
    """
    conn = get_connection(db_path)
    img_dir = tempfile.mkdtemp()
    sids, gids = [], []
    for i in range(n_samples):
        p = os.path.join(img_dir, f"i{i}.png")
        Image.new("RGB", (32, 32), (128, 128, 128)).save(p)
        sid = insert_sample(conn, p, f"hash{i}", student=f"w{i % 3}")
        gid = insert_ground_truth(conn, sid, "ground truth")
        sids.append(sid)
        gids.append(gid)
    conn.close()
    return sids, gids


def _seed_run(db_path, label, sids, gids, cers):
    conn = get_connection(db_path)
    rid = insert_run(conn, label=label, providers=["gemini"], strategies=["vote"])
    for sid, gid, cer in zip(sids, gids, cers):
        po = insert_provider_output(
            conn, run_id=rid, sample_id=sid,
            provider="gemini", strategy="vote",
            output_text="x", confidence=0.9,
        )
        insert_eval_metric(
            conn, provider_output_id=po, ground_truth_id=gid,
            cer=cer, wer=cer,
        )
    finish_run(conn, rid, len(cers))
    conn.close()
    return rid


# --- schema migration ---


class TestSchemaV6:
    def test_is_baseline_column_present(self, tmp_path):
        db = tmp_path / "v6.db"
        conn = get_connection(db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        conn.close()
        assert "is_baseline" in cols

    def test_default_is_zero(self, tmp_path):
        db = tmp_path / "v6_default.db"
        conn = get_connection(db)
        rid = insert_run(conn, label="default", providers=["gemini"], strategies=[])
        row = conn.execute("SELECT is_baseline FROM runs WHERE id = ?", (rid,)).fetchone()
        conn.close()
        assert row["is_baseline"] == 0


# --- set_baseline / get_baseline_run_id ---


class TestBaselineFunctions:
    def test_set_and_get(self, tmp_path):
        db = tmp_path / "set.db"
        sids, gids = _seed_db(db, 3)
        r1 = _seed_run(db, "v1", sids, gids, [0.1, 0.1, 0.1])
        r2 = _seed_run(db, "v2", sids, gids, [0.1, 0.1, 0.1])

        conn = get_connection(db)
        assert get_baseline_run_id(conn) is None
        set_baseline(conn, r2)
        assert get_baseline_run_id(conn) == r2
        conn.close()

    def test_at_most_one_baseline(self, tmp_path):
        db = tmp_path / "atmostone.db"
        sids, gids = _seed_db(db, 3)
        r1 = _seed_run(db, "v1", sids, gids, [0.1] * 3)
        r2 = _seed_run(db, "v2", sids, gids, [0.1] * 3)
        r3 = _seed_run(db, "v3", sids, gids, [0.1] * 3)

        conn = get_connection(db)
        set_baseline(conn, r1)
        set_baseline(conn, r2)
        set_baseline(conn, r3)
        # Only r3 should still be flagged.
        flagged = [r["id"] for r in conn.execute(
            "SELECT id FROM runs WHERE is_baseline = 1"
        ).fetchall()]
        conn.close()
        assert flagged == [r3]

    def test_unknown_run_raises(self, tmp_path):
        db = tmp_path / "missing.db"
        conn = get_connection(db)
        with pytest.raises(ValueError, match="not found"):
            set_baseline(conn, 999)
        conn.close()

    def test_durable_across_connections(self, tmp_path):
        # The pinned flag survives closing and reopening the DB —
        # i.e. it lives on disk, not just in process state.
        db = tmp_path / "durable.db"
        sids, gids = _seed_db(db, 3)
        r = _seed_run(db, "v", sids, gids, [0.1] * 3)

        conn = get_connection(db)
        set_baseline(conn, r)
        conn.close()

        conn2 = get_connection(db)
        assert get_baseline_run_id(conn2) == r
        conn2.close()

    def test_list_runs_includes_is_baseline(self, tmp_path):
        db = tmp_path / "list.db"
        sids, gids = _seed_db(db, 3)
        r1 = _seed_run(db, "v1", sids, gids, [0.1] * 3)
        r2 = _seed_run(db, "v2", sids, gids, [0.1] * 3)

        conn = get_connection(db)
        set_baseline(conn, r1)
        runs = list_runs(conn)
        conn.close()

        flagged = {r.run_id: r.is_baseline for r in runs}
        assert flagged[r1] == 1
        assert flagged[r2] == 0


# --- detect_regressions retargets to baseline ---


class TestDetectRegressionsBaseline:
    def test_no_baseline_falls_back_to_previous(self, tmp_path):
        db = tmp_path / "fallback.db"
        sids, gids = _seed_db(db, 5)
        r1 = _seed_run(db, "v1", sids, gids, [0.05] * 5)
        r2 = _seed_run(db, "v2", sids, gids, [0.10] * 5)  # regression vs r1

        regs = detect_regressions(db_path=db)
        # No pinned baseline → falls back to runs[1] = r1 → regression.
        assert any(r["delta"] > 0.04 for r in regs)

    def test_pinned_baseline_used_over_previous(self, tmp_path):
        db = tmp_path / "pinned.db"
        sids, gids = _seed_db(db, 5)
        # r1 = 0.05, r2 = 0.06 (no regression vs r1), r3 = 0.06 (no regression vs r2 but…)
        r1 = _seed_run(db, "v1", sids, gids, [0.05] * 5)
        r2 = _seed_run(db, "v2", sids, gids, [0.06] * 5)
        r3 = _seed_run(db, "v3", sids, gids, [0.10] * 5)

        # Without baseline: detect_regressions(r3) compares r3 vs runs[1]=r2,
        # delta = 0.04 → REGRESSION (above 3% threshold).
        # With baseline pinned at r1: compare r3 vs r1, delta = 0.05 → still
        # REGRESSION but vs the *anchored* run.
        # Sharper test: pin r1, run r2 (which is +1pp vs r1, below 3%
        # threshold) — fallback would compare r2 vs r1 anyway in this case,
        # so we need a scenario where the choice of anchor changes the
        # answer.
        # r3 vs r2 = +4pp REGRESSION, r3 vs r1 = +5pp REGRESSION. Both detect.
        # Need: anchor pick changes the *count* of regressions or the *delta*.
        # Use r2 to demonstrate: r2 vs r1 (anchor) = +1pp, no regression.
        #                        r2 vs runs[1] = r1 anyway → also no regression.
        # Need another run to break the tie.
        # Insert r4 = 0.07. Without baseline: r4 vs r3 = -0.03, no regression.
        # With baseline r1 pinned: r4 vs r1 = +0.02, no regression either.
        # OK simpler: pin r1, query r3. delta_pinned = +0.05, delta_fallback (r3 vs r2) = +0.04.
        # Both are regressions but the reported `previous_cer` differs.
        conn = get_connection(db)
        set_baseline(conn, r1)
        conn.close()

        regs = detect_regressions(run_id=r3, db_path=db)
        assert len(regs) == 1
        # Reported previous_cer comes from the pinned baseline r1 (=0.05),
        # not from the immediately-preceding run r2 (=0.06).
        assert regs[0]["previous_cer"] == pytest.approx(0.05, abs=1e-9)
        assert regs[0]["current_cer"] == pytest.approx(0.10, abs=1e-9)

    def test_self_compare_falls_back_when_current_is_baseline(self, tmp_path):
        db = tmp_path / "self.db"
        sids, gids = _seed_db(db, 5)
        r1 = _seed_run(db, "v1", sids, gids, [0.05] * 5)
        r2 = _seed_run(db, "v2", sids, gids, [0.10] * 5)

        conn = get_connection(db)
        set_baseline(conn, r2)  # the latest run is the baseline
        conn.close()

        # detect_regressions for r2 (which IS the baseline) should fall back
        # to comparing against the prior run, not against itself.
        regs = detect_regressions(run_id=r2, db_path=db)
        # r2 vs r1 = +5pp regression — still detected.
        assert len(regs) == 1
        assert regs[0]["previous_cer"] == pytest.approx(0.05, abs=1e-9)


# --- CLI ---


class TestSetBaselineCli:
    def test_set_baseline_command(self, tmp_path):
        db = tmp_path / "cli.db"
        sids, gids = _seed_db(db, 3)
        r1 = _seed_run(db, "v1", sids, gids, [0.1] * 3)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "set-baseline", str(r1),
            "--db-path", str(db),
        ])
        assert result.exit_code == 0, result.output
        assert f"Baseline pinned: run #{r1}" in result.output

        conn = get_connection(db)
        assert get_baseline_run_id(conn) == r1
        conn.close()

    def test_set_baseline_unknown_run(self, tmp_path):
        db = tmp_path / "cli_missing.db"
        # Make sure the DB exists but has no runs.
        conn = get_connection(db)
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "set-baseline", "999",
            "--db-path", str(db),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output
