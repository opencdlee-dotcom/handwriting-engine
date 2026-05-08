"""Tests for threshold_tuner — pure-Python, no torch / GPU deps."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_ground_truth,
    insert_provider_output,
    insert_run,
    insert_sample,
)
from handwriting_engine.threshold_tuner import (
    DEFAULT_THRESHOLDS,
    ThresholdPoint,
    pareto_frontier,
    pick_knee,
    sweep_thresholds,
)


# ---------- Test helpers ----------


def _seed_pair(
    conn,
    *,
    run_id: int,
    image_hash: str,
    gt_text: str,
    trocr_text: str,
    trocr_conf: float,
    cloud_text: str,
    cloud_provider: str = "gemini",
    student: str = "writer-a",
) -> int:
    """Insert a (sample, gt, trocr_output, cloud_output) quadruple.

    Returns the sample_id.
    """
    sid = insert_sample(
        conn,
        image_path=f"/fake/{image_hash}.png",
        image_hash=image_hash,
        student=student,
    )
    insert_ground_truth(conn, sid, gt_text)
    insert_provider_output(
        conn,
        run_id=run_id,
        sample_id=sid,
        provider="trocr",
        strategy="single",
        output_text=trocr_text,
        confidence=trocr_conf,
    )
    insert_provider_output(
        conn,
        run_id=run_id,
        sample_id=sid,
        provider=cloud_provider,
        strategy="single",
        output_text=cloud_text,
        confidence=0.7,
    )
    return sid


@pytest.fixture
def seeded_db(tmp_path):
    """A DB with 5 paired samples spanning the trocr-confidence range.

    Each cloud output exactly matches GT (so escalation drives CER → 0).
    Each trocr output has 1 character substitution (CER > 0). The trocr
    confidence varies, so threshold choice cleanly trades CER vs cloud-rate.
    """
    db_path = tmp_path / "tune.db"
    conn = get_connection(db_path)
    run_id = insert_run(conn, label="seed", providers=["trocr", "gemini"])

    # Five samples; trocr confidences spaced 0.30..0.90.
    confidences = [0.30, 0.50, 0.65, 0.75, 0.90]
    gt = "the quick brown fox jumps over the lazy dog"
    # 1-char-off trocr text (substitute a single letter):
    trocr_text = "the quack brown fox jumps over the lazy dog"
    for i, conf in enumerate(confidences):
        _seed_pair(
            conn,
            run_id=run_id,
            image_hash=f"hash{i}",
            gt_text=gt,
            trocr_text=trocr_text,
            trocr_conf=conf,
            cloud_text=gt,  # cloud = perfect
        )
    conn.close()
    return db_path


# ---------- sweep_thresholds ----------


class TestSweepThresholds:
    def test_empty_db_returns_empty_list(self, tmp_path):
        db_path = tmp_path / "empty.db"
        # ensure schema exists but no rows
        get_connection(db_path).close()
        out = sweep_thresholds(db_path=db_path)
        assert out == []

    def test_no_crash_when_no_paired_rows(self, tmp_path):
        """Samples + GT but no trocr/cloud outputs → empty result."""
        db_path = tmp_path / "lonely.db"
        conn = get_connection(db_path)
        sid = insert_sample(conn, "/x.png", "h1")
        insert_ground_truth(conn, sid, "hello world")
        conn.close()
        assert sweep_thresholds(db_path=db_path) == []

    def test_returns_one_point_per_threshold(self, seeded_db):
        thresholds = [0.4, 0.6, 0.8]
        out = sweep_thresholds(db_path=seeded_db, thresholds=thresholds)
        assert len(out) == 3
        assert [p.threshold for p in out] == thresholds
        assert all(p.n_samples == 5 for p in out)

    def test_threshold_above_one_forces_full_cloud(self, seeded_db):
        """Threshold > 1.0 → no trocr output ever passes → 100% cloud rate."""
        out = sweep_thresholds(db_path=seeded_db, thresholds=[1.5])
        assert len(out) == 1
        assert out[0].cloud_call_rate == pytest.approx(1.0)
        # Cloud is perfect in the fixture, so CER should be 0.
        assert out[0].cer == pytest.approx(0.0)

    def test_threshold_below_zero_forces_no_cloud(self, seeded_db):
        """Threshold <= 0 → every trocr accept → 0% cloud rate, CER > 0."""
        out = sweep_thresholds(db_path=seeded_db, thresholds=[-0.01])
        assert len(out) == 1
        assert out[0].cloud_call_rate == pytest.approx(0.0)
        # trocr text differs from GT by one character → CER must be > 0.
        assert out[0].cer > 0.0

    def test_lower_threshold_is_lower_cloud_rate_higher_cer(self, seeded_db):
        """Monotonicity: as threshold rises, cloud rate ↑, CER ↓ (in fixture)."""
        thresholds = [0.0, 0.4, 0.7, 1.5]
        out = sweep_thresholds(db_path=seeded_db, thresholds=thresholds)
        rates = [p.cloud_call_rate for p in out]
        cers = [p.cer for p in out]

        # Cloud-call rate must be non-decreasing in threshold.
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)), rates
        # CER must be non-increasing in threshold (cloud is perfect here).
        assert all(cers[i] >= cers[i + 1] for i in range(len(cers) - 1)), cers
        # And the extremes must be strictly different (sanity).
        assert rates[0] < rates[-1]
        assert cers[0] > cers[-1]

    def test_writer_id_filters_samples(self, tmp_path):
        db_path = tmp_path / "writers.db"
        conn = get_connection(db_path)
        run_id = insert_run(conn, label="w", providers=["trocr", "gemini"])
        gt = "alpha beta gamma"
        # writer-a: trocr matches GT (low CER if accepted)
        _seed_pair(
            conn, run_id=run_id, image_hash="ha", gt_text=gt,
            trocr_text=gt, trocr_conf=0.9, cloud_text=gt, student="writer-a",
        )
        # writer-b: trocr is wildly wrong
        _seed_pair(
            conn, run_id=run_id, image_hash="hb", gt_text=gt,
            trocr_text="totally garbled output", trocr_conf=0.9,
            cloud_text=gt, student="writer-b",
        )
        conn.close()

        out_a = sweep_thresholds(
            db_path=db_path, thresholds=[0.5], writer_id="writer-a",
        )
        out_b = sweep_thresholds(
            db_path=db_path, thresholds=[0.5], writer_id="writer-b",
        )
        assert out_a[0].n_samples == 1
        assert out_b[0].n_samples == 1
        # Writer A: trocr accepted and matches GT → CER ~0.
        assert out_a[0].cer == pytest.approx(0.0)
        # Writer B: trocr accepted but garbage → CER high.
        assert out_b[0].cer > 0.5

    def test_uses_latest_provider_output_per_sample(self, tmp_path):
        """When multiple trocr rows exist for one sample, latest wins."""
        db_path = tmp_path / "latest.db"
        conn = get_connection(db_path)
        run_id = insert_run(conn, label="a", providers=["trocr", "gemini"])
        sid = insert_sample(conn, "/p.png", "hh", student="w")
        gt = "hello world"
        insert_ground_truth(conn, sid, gt)

        # First trocr: low confidence (would escalate at thresh=0.5)
        insert_provider_output(
            conn, run_id=run_id, sample_id=sid, provider="trocr",
            strategy="single", output_text="hello world", confidence=0.1,
        )
        # Second trocr: high confidence — this is the "latest" we should pick
        insert_provider_output(
            conn, run_id=run_id, sample_id=sid, provider="trocr",
            strategy="single", output_text="hello world", confidence=0.95,
        )
        insert_provider_output(
            conn, run_id=run_id, sample_id=sid, provider="gemini",
            strategy="single", output_text=gt, confidence=0.7,
        )
        conn.close()

        out = sweep_thresholds(db_path=db_path, thresholds=[0.5])
        assert out[0].n_samples == 1
        # Latest trocr conf is 0.95 ≥ 0.5 → no escalation.
        assert out[0].cloud_call_rate == pytest.approx(0.0)

    def test_default_thresholds_used_when_none(self, seeded_db):
        out = sweep_thresholds(db_path=seeded_db)
        assert len(out) == len(DEFAULT_THRESHOLDS)
        assert [p.threshold for p in out] == list(DEFAULT_THRESHOLDS)

    def test_empty_trocr_text_forces_escalation(self, tmp_path):
        """Empty trocr text → escalate even when confidence clears threshold."""
        db_path = tmp_path / "empty_local.db"
        conn = get_connection(db_path)
        run_id = insert_run(conn, label="e", providers=["trocr", "gemini"])
        sid = insert_sample(conn, "/p.png", "hh", student="w")
        gt = "hello world"
        insert_ground_truth(conn, sid, gt)
        insert_provider_output(
            conn, run_id=run_id, sample_id=sid, provider="trocr",
            strategy="single", output_text="   ", confidence=0.99,
        )
        insert_provider_output(
            conn, run_id=run_id, sample_id=sid, provider="gemini",
            strategy="single", output_text=gt, confidence=0.7,
        )
        conn.close()
        out = sweep_thresholds(db_path=db_path, thresholds=[0.5])
        assert out[0].cloud_call_rate == pytest.approx(1.0)


# ---------- pareto_frontier ----------


class TestParetoFrontier:
    def test_handbuilt_dominance(self):
        # A dominates C (lower CER and lower cloud rate). B is independent.
        a = ThresholdPoint(threshold=0.5, cer=0.05, cloud_call_rate=0.30, n_samples=10)
        b = ThresholdPoint(threshold=0.7, cer=0.02, cloud_call_rate=0.80, n_samples=10)
        c = ThresholdPoint(threshold=0.6, cer=0.10, cloud_call_rate=0.60, n_samples=10)
        front = pareto_frontier([a, b, c])
        assert a in front
        assert b in front
        assert c not in front

    def test_all_pareto(self):
        # Three points, none dominates any other.
        pts = [
            ThresholdPoint(threshold=0.4, cer=0.10, cloud_call_rate=0.10, n_samples=5),
            ThresholdPoint(threshold=0.6, cer=0.05, cloud_call_rate=0.50, n_samples=5),
            ThresholdPoint(threshold=0.8, cer=0.01, cloud_call_rate=0.90, n_samples=5),
        ]
        front = pareto_frontier(pts)
        assert len(front) == 3

    def test_ties_do_not_dominate(self):
        a = ThresholdPoint(threshold=0.5, cer=0.05, cloud_call_rate=0.30, n_samples=10)
        # Same CER, lower cloud rate → does NOT strictly dominate.
        b = ThresholdPoint(threshold=0.6, cer=0.05, cloud_call_rate=0.20, n_samples=10)
        front = pareto_frontier([a, b])
        assert a in front
        assert b in front

    def test_skips_zero_sample_points(self):
        good = ThresholdPoint(threshold=0.5, cer=0.10, cloud_call_rate=0.50, n_samples=5)
        empty = ThresholdPoint(threshold=0.6, cer=0.0, cloud_call_rate=0.0, n_samples=0)
        front = pareto_frontier([good, empty])
        assert front == [good]

    def test_empty_input(self):
        assert pareto_frontier([]) == []


# ---------- pick_knee ----------


class TestPickKnee:
    def test_picks_min_cer_plus_half_rate(self):
        pts = [
            ThresholdPoint(threshold=0.4, cer=0.20, cloud_call_rate=0.10, n_samples=5),  # 0.25
            ThresholdPoint(threshold=0.6, cer=0.05, cloud_call_rate=0.40, n_samples=5),  # 0.25
            ThresholdPoint(threshold=0.8, cer=0.01, cloud_call_rate=0.90, n_samples=5),  # 0.46
        ]
        # Tie between 0.4 and 0.6 at score=0.25; min returns the first encountered.
        knee = pick_knee(pts)
        assert knee is not None
        assert knee.threshold in (0.4, 0.6)

    def test_empty_returns_none(self):
        assert pick_knee([]) is None


# ---------- CLI wiring ----------


class TestCliTuneThreshold:
    def test_help_advertises_subcommand(self):
        from handwriting_engine.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "tune-threshold", "--help"])
        assert result.exit_code == 0, result.output
        assert "--cloud-provider" in result.output
        assert "--writer" in result.output
        assert "--format" in result.output

    def test_insufficient_data_message(self, tmp_path):
        from handwriting_engine.cli import cli
        # Empty DB → fewer than 10 paired samples.
        db_path = tmp_path / "empty.db"
        get_connection(db_path).close()
        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "tune-threshold", "--db-path", str(db_path),
        ])
        assert result.exit_code == 0, result.output
        assert "Not enough paired" in result.output

    def test_table_output_with_seeded_data(self, tmp_path):
        from handwriting_engine.cli import cli
        # Seed 10 paired samples to clear the >=10 gate.
        db_path = tmp_path / "ten.db"
        conn = get_connection(db_path)
        run_id = insert_run(conn, label="cli", providers=["trocr", "gemini"])
        gt = "the quick brown fox"
        for i in range(10):
            _seed_pair(
                conn, run_id=run_id, image_hash=f"h{i}", gt_text=gt,
                trocr_text="the quack brown fox",
                trocr_conf=0.30 + 0.06 * i,
                cloud_text=gt,
            )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "tune-threshold", "--db-path", str(db_path),
        ])
        assert result.exit_code == 0, result.output
        assert "Recommended:" in result.output
        assert "Pareto" in result.output

    def test_json_format(self, tmp_path):
        from handwriting_engine.cli import cli
        db_path = tmp_path / "json.db"
        conn = get_connection(db_path)
        run_id = insert_run(conn, label="j", providers=["trocr", "gemini"])
        gt = "the quick brown fox"
        for i in range(10):
            _seed_pair(
                conn, run_id=run_id, image_hash=f"h{i}", gt_text=gt,
                trocr_text="the quack brown fox",
                trocr_conf=0.30 + 0.06 * i,
                cloud_text=gt,
            )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "tune-threshold",
            "--db-path", str(db_path), "--format", "json",
        ])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["cloud_provider"] == "gemini"
        assert "points" in payload
        assert payload["recommended_threshold"] is not None
        assert any(p["pareto"] for p in payload["points"])
