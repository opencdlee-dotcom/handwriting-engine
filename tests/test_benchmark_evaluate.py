"""Tests for benchmark evaluation — uses mocks to avoid real API calls."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from handwriting_engine.benchmark.db import (
    get_connection,
    get_run_results,
    insert_ground_truth,
    insert_sample,
)
from handwriting_engine.benchmark.evaluate import estimate_cost, run_benchmark
from handwriting_engine.benchmark.report import (
    generate_report,
    compare_runs,
    detect_regressions,
)

# Sweep + report imports — RED until Wave 2
try:
    from handwriting_engine.benchmark.evaluate import run_sweep
except ImportError:
    run_sweep = None  # type: ignore

try:
    from handwriting_engine.benchmark.report import generate_per_writer_report
except ImportError:
    generate_per_writer_report = None  # type: ignore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_eval.db"


@pytest.fixture
def seeded_db(db_path, tmp_path):
    """DB with samples and ground truth, plus a real image file."""
    from PIL import Image

    conn = get_connection(db_path)

    img_path = tmp_path / "test_img.png"
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img.save(img_path)

    sid = insert_sample(conn, str(img_path), "fakehash1", student="test")
    insert_ground_truth(conn, sid, "the mitochondria is the powerhouse of the cell")
    conn.close()
    return db_path


class TestEstimateCost:
    def test_gemini_cheapest(self):
        cost = estimate_cost(1_000_000, 1_000_000, "gemini")
        assert cost < estimate_cost(1_000_000, 1_000_000, "claude")

    def test_zero_tokens(self):
        assert estimate_cost(0, 0, "gemini") == 0.0

    def test_unknown_provider(self):
        assert estimate_cost(1000, 1000, "unknown") == 0.0


class TestRunBenchmark:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_basic_run(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7,
            "latency_ms": 500,
            "input_tokens": 100,
            "output_tokens": 50,
            "error": None,
        }

        run_id = run_benchmark(
            label="test", providers=["gemini"], strategies=[], db_path=seeded_db,
        )
        assert run_id > 0

        conn = get_connection(seeded_db)
        results = get_run_results(conn, run_id)
        conn.close()
        assert len(results) == 1
        assert results[0]["cer"] == pytest.approx(0.0)

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_with_errors(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the sell",
            "confidence": 0.5,
            "latency_ms": 600,
            "input_tokens": 100,
            "output_tokens": 50,
            "error": None,
        }

        run_id = run_benchmark(
            label="errors", providers=["gemini"], strategies=[], db_path=seeded_db,
        )

        conn = get_connection(seeded_db)
        results = get_run_results(conn, run_id)
        conn.close()
        assert results[0]["cer"] > 0  # "sell" != "cell"

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    def test_no_providers_raises(self, mock_providers, seeded_db):
        mock_providers.return_value = []
        with pytest.raises(RuntimeError, match="No providers"):
            run_benchmark(db_path=seeded_db)

    def test_no_ground_truth_raises(self, db_path):
        conn = get_connection(db_path)
        insert_sample(conn, "/img.png", "hash1")
        conn.close()

        with patch("handwriting_engine.benchmark.evaluate._available_providers", return_value=["gemini"]):
            with pytest.raises(RuntimeError, match="No samples with ground truth"):
                run_benchmark(db_path=db_path)


class TestReport:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_table_report(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_id = run_benchmark(label="report test", providers=["gemini"], strategies=[], db_path=seeded_db)

        report = generate_report(run_id, db_path=seeded_db)
        assert "gemini" in report
        assert "0.00%" in report

    def test_no_runs(self, db_path):
        get_connection(db_path).close()  # ensure DB exists
        report = generate_report(db_path=db_path)
        assert "No benchmark runs" in report


class TestSmokeMode:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_smoke_limits_samples(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        run_id = run_benchmark(
            label="smoke", providers=["gemini"], strategies=[], db_path=seeded_db, mode="smoke",
        )
        assert run_id > 0


class TestProgressCallback:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_progress_called(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        progress_calls = []
        def on_progress(current, total, msg):
            progress_calls.append((current, total, msg))

        run_benchmark(
            providers=["gemini"], strategies=[], db_path=seeded_db, on_progress=on_progress,
        )
        assert len(progress_calls) > 0
        assert progress_calls[0][0] == 1  # first call is sample 1


class TestDrillDown:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_drill_down_report(self, mock_read, mock_providers, seeded_db):
        from handwriting_engine.benchmark.report import sample_drill_down
        from handwriting_engine.benchmark.db import get_connection, list_samples

        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_benchmark(providers=["gemini"], strategies=[], db_path=seeded_db)

        conn = get_connection(seeded_db)
        samples = list_samples(conn)
        conn.close()

        report = sample_drill_down(samples[0].id, db_path=seeded_db)
        assert "gemini" in report
        assert "CER" in report


class TestMarkerRate:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_marker_rate_from_raw_text(self, mock_read, mock_providers, seeded_db):
        """Marker rate counts [?] tokens from raw text, not normalized text."""
        mock_providers.return_value = ["gemini"]
        # 2 markers in 5 words = 0.4 rate
        mock_read.return_value = {
            "text": "[?] mitochondria [?] powerhouse cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_id = run_benchmark(label="marker_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        conn = get_connection(seeded_db)
        rows = conn.execute(
            "SELECT question_marker_rate FROM provider_outputs WHERE run_id = ?", (run_id,)
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["question_marker_rate"] == pytest.approx(0.4, abs=0.01)

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_marker_rate_clean_output(self, mock_read, mock_providers, seeded_db):
        """Clean output (no [?]) should have marker_rate == 0.0."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.9, "latency_ms": 300,
            "input_tokens": 80, "output_tokens": 40, "error": None,
        }
        run_id = run_benchmark(label="clean_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        conn = get_connection(seeded_db)
        rows = conn.execute(
            "SELECT question_marker_rate FROM provider_outputs WHERE run_id = ?", (run_id,)
        ).fetchall()
        conn.close()
        assert rows[0]["question_marker_rate"] == pytest.approx(0.0)

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_marker_rate_computed_before_normalization(self, mock_read, mock_providers, seeded_db):
        """If marker_rate is computed after normalization, [?] is stripped and rate=0. Must be > 0."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "[?] unknown term here",
            "confidence": 0.5, "latency_ms": 400,
            "input_tokens": 90, "output_tokens": 45, "error": None,
        }
        run_id = run_benchmark(label="norm_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        conn = get_connection(seeded_db)
        row = conn.execute(
            "SELECT question_marker_rate FROM provider_outputs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()
        # If this is 0.0, marker rate was computed AFTER normalization (bug)
        assert row["question_marker_rate"] > 0.0, (
            "marker_rate is 0 — likely computed after normalize_text() stripped [?] markers"
        )

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_marker_rate_in_report(self, mock_read, mock_providers, seeded_db):
        """generate_report output table must include a marker_rate column."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "[?] mitochondria [?] powerhouse cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_id = run_benchmark(label="marker_report_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        report = generate_report(run_id=run_id, db_path=seeded_db)
        assert "marker_rate" in report.lower(), "Report missing marker_rate column"


class TestCalibrateCommand:
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_calibrate_output_format(self, mock_read, seeded_db):
        """Output must match: 'CER variance: ±X%  |  Min detectable delta: Y% (2σ)'"""
        import re
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.8, "latency_ms": 400,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "calibrate", "--samples", "1", "--provider", "gemini",
            "--db-path", str(seeded_db)
        ])
        assert result.exit_code == 0, f"calibrate failed: {result.output}"
        assert re.search(r"CER variance: ±[\d.]+%\s+\|\s+Min detectable delta: [\d.]+% \(2σ\)", result.output), \
            f"Output format mismatch: {result.output}"

    def test_calibrate_undersample_warning(self, seeded_db):
        """Requesting more samples than available should warn but not abort."""
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "calibrate", "--samples", "9999", "--provider", "gemini",
            "--db-path", str(seeded_db)
        ])
        # Should warn and proceed (or exit 0 if < 2 samples available)
        assert "Warning" in result.output or result.exit_code == 0

    def test_calibrate_no_samples_error(self, tmp_path):
        """Empty DB (no ground truth) should exit non-zero with error message."""
        empty_db = tmp_path / "empty.db"
        conn = get_connection(empty_db)
        conn.close()
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "calibrate", "--samples", "5",
            "--db-path", str(empty_db)
        ])
        assert result.exit_code != 0 or "No samples" in result.output


class TestCostProjection:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_cost_always_shown(self, mock_read, mock_providers, seeded_db):
        """Cost projection must appear before benchmark execution, no threshold."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "test", "confidence": 0.8, "latency_ms": 200,
            "input_tokens": 50, "output_tokens": 20, "error": None,
        }
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "run", "--providers", "gemini", "--yes",
            "--db-path", str(seeded_db)
        ])
        assert "Estimated cost:" in result.output, f"Cost not shown: {result.output}"

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_yes_bypasses_prompt(self, mock_read, mock_providers, seeded_db):
        """--yes flag must skip the 'Proceed?' confirmation prompt."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "test", "confidence": 0.8, "latency_ms": 200,
            "input_tokens": 50, "output_tokens": 20, "error": None,
        }
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "run", "--providers", "gemini", "--yes",
            "--db-path", str(seeded_db)
        ])
        assert "Proceed?" not in result.output

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_decline_exits_cleanly(self, mock_read, mock_providers, seeded_db):
        """Entering 'n' at cost prompt must exit 0 (graceful), not crash."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "test", "confidence": 0.8, "latency_ms": 200,
            "input_tokens": 50, "output_tokens": 20, "error": None,
        }
        runner = CliRunner()
        from handwriting_engine.cli import cli
        result = runner.invoke(cli, [
            "benchmark", "run", "--providers", "gemini",
            "--db-path", str(seeded_db)
        ], input="n\n")
        assert result.exit_code == 0, f"Decline raised non-zero exit: {result.output}"
        assert "Estimated cost:" in result.output


class TestProvenanceCapture:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_provenance_columns_in_db(self, mock_read, mock_providers, seeded_db):
        """After run_benchmark, runs table must have model_version populated."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.9, "latency_ms": 300,
            "input_tokens": 80, "output_tokens": 40, "error": None,
        }
        run_id = run_benchmark(label="prov_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        conn = get_connection(seeded_db)
        row = conn.execute("SELECT model_version, norm_flags FROM runs WHERE id = ?", (run_id,)).fetchone()
        conn.close()
        assert row["model_version"] is not None, "model_version not captured"
        assert row["norm_flags"] is not None, "norm_flags not captured"

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_report_contains_provenance_header(self, mock_read, mock_providers, seeded_db):
        """generate_report output must include 'Provenance:' section header."""
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.9, "latency_ms": 300,
            "input_tokens": 80, "output_tokens": 40, "error": None,
        }
        run_id = run_benchmark(label="prov_report_test", providers=["gemini"], strategies=[], db_path=seeded_db)
        report = generate_report(run_id=run_id, db_path=seeded_db)
        assert "Provenance:" in report, f"Provenance header missing from report: {report[:200]}"


class TestCompareRuns:
    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_compare(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]

        # Run 1: perfect
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        r1 = run_benchmark(label="v1", providers=["gemini"], strategies=[], db_path=seeded_db)

        # Run 2: worse
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the sell",
            "confidence": 0.5, "latency_ms": 600,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        r2 = run_benchmark(label="v2", providers=["gemini"], strategies=[], db_path=seeded_db)

        comparison = compare_runs(r1, r2, db_path=seeded_db)
        assert "REGRESSION" in comparison or "unchanged" in comparison

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_detect_regressions(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]

        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_benchmark(providers=["gemini"], strategies=[], db_path=seeded_db)

        mock_read.return_value = {
            "text": "completely wrong garbage text output",
            "confidence": 0.3, "latency_ms": 800,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_benchmark(providers=["gemini"], strategies=[], db_path=seeded_db)

        regs = detect_regressions(db_path=seeded_db)
        assert len(regs) > 0
        assert regs[0]["delta"] > 0


class TestSweep:
    """Sweep infrastructure (IAM-02) — turned GREEN in Phase 07-03."""

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_benchmark_accepts_line_level(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        run_id = run_benchmark(
            providers=["gemini"], strategies=[], db_path=seeded_db,
            line_level=True,
        )
        assert run_id > 0
        # _read_single must receive line_level=True
        kwargs = mock_read.call_args.kwargs
        assert kwargs.get("line_level") is True

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_benchmark_accepts_auto_retry(self, mock_read, mock_providers, seeded_db):
        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        run_id = run_benchmark(
            providers=["gemini"], strategies=[], db_path=seeded_db,
            auto_retry=True,
        )
        assert run_id > 0
        kwargs = mock_read.call_args.kwargs
        assert kwargs.get("auto_retry") is True

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_sweep_returns_five_run_ids(self, mock_read, mock_providers, seeded_db):
        # run_sweep filters samples to category='iam' only — ensure the seeded
        # sample has that category so each strategy actually evaluates something.
        conn = get_connection(seeded_db)
        try:
            conn.execute("UPDATE samples SET category='iam' WHERE id=1")
            conn.commit()
        finally:
            conn.close()

        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        run_ids = run_sweep(provider="gemini", db_path=seeded_db, yes=True)
        assert isinstance(run_ids, dict)
        assert set(run_ids.keys()) == {
            "baseline", "self_correct", "line_level",
            "prompt_adapted", "zoomed_verify",
        }
        assert all(isinstance(rid, int) and rid > 0 for rid in run_ids.values())

    def test_sweep_cli_shows_cost(self, tmp_path):
        from handwriting_engine.cli import cli
        # Use an empty DB — cost projection must still run before any API call
        db_path = tmp_path / "empty.db"
        get_connection(db_path).close()

        runner = CliRunner()
        # Decline at the prompt — we only care that the cost line appears
        result = runner.invoke(
            cli,
            ["benchmark", "sweep", "--db-path", str(db_path)],
            input="n\n",
        )
        assert "Estimated cost" in result.output
        assert "Sweep projection" in result.output

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_sweep_cli_yes_executes(self, mock_read, mock_providers, tmp_path):
        from handwriting_engine.cli import cli
        from PIL import Image

        db_path = tmp_path / "sweep.db"
        conn = get_connection(db_path)
        img_path = tmp_path / "iam.png"
        Image.new("RGB", (200, 200), color=(128, 128, 128)).save(img_path)
        sid = insert_sample(conn, str(img_path), "iamhash1",
                            student="iam-writer-001", category="iam")
        insert_ground_truth(conn, sid, "the mitochondria is the powerhouse of the cell")
        conn.close()

        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["benchmark", "sweep", "--yes", "--db-path", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        assert "Sweep complete" in result.output
        # All 5 strategies must appear in the report
        for name in ("baseline", "self_correct", "line_level",
                     "prompt_adapted", "zoomed_verify"):
            assert name in result.output


class TestSweepCategoryFilter:
    """Sweep --category support — fixes IAM-hardcoded skip of lab data."""

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_sweep_lab_selects_only_lab_samples(
        self, mock_read, mock_providers, db_path, tmp_path,
    ):
        """category='lab' must restrict the sweep to lab-tagged samples."""
        from PIL import Image

        conn = get_connection(db_path)
        img_iam = tmp_path / "iam.png"
        img_lab = tmp_path / "lab.png"
        Image.new("RGB", (200, 200), color=(128, 128, 128)).save(img_iam)
        Image.new("RGB", (200, 200), color=(64, 64, 64)).save(img_lab)
        sid_iam = insert_sample(conn, str(img_iam), "hashIam",
                                student="iam-writer", category="iam")
        sid_lab = insert_sample(conn, str(img_lab), "hashLab",
                                student="lab-student", category="lab")
        insert_ground_truth(conn, sid_iam, "the mitochondria is the powerhouse of the cell")
        insert_ground_truth(conn, sid_lab, "the cell underwent mitosis today")
        conn.close()

        mock_providers.return_value = ["gemini"]
        # Echo back the right text per image so CER stays 0 either way.
        def fake_read(path, *args, **kwargs):
            if "lab.png" in path:
                text = "the cell underwent mitosis today"
            else:
                text = "the mitochondria is the powerhouse of the cell"
            return {
                "text": text, "confidence": 0.7, "latency_ms": 500,
                "input_tokens": 100, "output_tokens": 50, "error": None,
            }
        mock_read.side_effect = fake_read

        run_ids = run_sweep(
            provider="gemini", db_path=db_path, yes=True, category="lab",
        )
        assert run_ids, "expected at least one strategy run"

        # Every output across every strategy must be from the lab sample only.
        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT DISTINCT sample_id FROM provider_outputs WHERE run_id IN (%s)"
            % ",".join("?" * len(run_ids)),
            tuple(run_ids.values()),
        ).fetchall()
        conn.close()
        sample_ids_seen = {r["sample_id"] for r in rows}
        assert sample_ids_seen == {sid_lab}, (
            f"sweep included non-lab samples: saw {sample_ids_seen}, expected {{{sid_lab}}}"
        )

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_run_sweep_no_category_arg_includes_iam(
        self, mock_read, mock_providers, seeded_db,
    ):
        """Backward compat: no-arg run_sweep must still process IAM samples
        (existing IAM workflows must not regress when category defaults to None=all)."""
        # Tag the seeded sample as IAM, matching the prior hardcoded filter.
        conn = get_connection(seeded_db)
        try:
            conn.execute("UPDATE samples SET category='iam' WHERE id=1")
            conn.commit()
        finally:
            conn.close()

        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }

        # No category argument — should still pick up the IAM sample.
        run_ids = run_sweep(provider="gemini", db_path=seeded_db, yes=True)
        assert run_ids, "no-arg sweep produced zero runs (IAM regression)"

        # And each cloud strategy must have evaluated the IAM sample.
        conn = get_connection(seeded_db)
        rows = conn.execute(
            "SELECT DISTINCT sample_id FROM provider_outputs WHERE run_id IN (%s)"
            % ",".join("?" * len(run_ids)),
            tuple(run_ids.values()),
        ).fetchall()
        conn.close()
        assert {r["sample_id"] for r in rows} == {1}


class TestIngestLabGtSuffix:
    """`benchmark ingest-lab --gt-suffix .txt` runs without prompting."""

    def test_cli_gt_suffix_skips_editor(self, tmp_path):
        """Paired *.png + *.txt directory ingests with no $EDITOR prompt."""
        from PIL import Image
        from handwriting_engine.cli import cli

        img_dir = tmp_path / "lab"
        img_dir.mkdir()
        # Two paired files.
        for i, gt in enumerate(["page zero text", "page one text"]):
            img = img_dir / f"page_{i}.png"
            Image.new("RGB", (64, 64), (10 + i, 20 + i, 30 + i)).save(img)
            (img_dir / f"page_{i}.txt").write_text(gt, encoding="utf-8")

        db = tmp_path / "lab.db"
        runner = CliRunner()
        # Patch click.edit so a regression that re-introduces the editor
        # path will fail loudly (returning None marks every page as skipped).
        with patch("click.edit", return_value=None) as mock_edit:
            result = runner.invoke(cli, [
                "benchmark", "ingest-lab", str(img_dir),
                "--gt-suffix", ".txt",
                "--db-path", str(db),
            ])
        assert result.exit_code == 0, result.output
        assert mock_edit.call_count == 0, (
            "ingest-lab --gt-suffix should not call click.edit"
        )
        assert "2 annotated" in result.output, result.output

        # Ground truth ended up in the DB, sourced from the .txt files.
        conn = get_connection(db)
        rows = conn.execute(
            "SELECT text FROM ground_truths ORDER BY text"
        ).fetchall()
        conn.close()
        assert [r["text"] for r in rows] == ["page one text", "page zero text"]

    def test_cli_gt_suffix_missing_sibling_skips(self, tmp_path):
        """Image without a matching .txt is treated as user-skip."""
        from PIL import Image
        from handwriting_engine.cli import cli

        img_dir = tmp_path / "lab"
        img_dir.mkdir()
        img_a = img_dir / "page_a.png"
        img_b = img_dir / "page_b.png"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(img_a)
        Image.new("RGB", (64, 64), (40, 50, 60)).save(img_b)
        (img_dir / "page_a.txt").write_text("only A has GT", encoding="utf-8")
        # page_b.txt intentionally missing.

        db = tmp_path / "lab.db"
        runner = CliRunner()
        with patch("click.edit", return_value=None) as mock_edit:
            result = runner.invoke(cli, [
                "benchmark", "ingest-lab", str(img_dir),
                "--gt-suffix", ".txt",
                "--db-path", str(db),
            ])
        assert result.exit_code == 0, result.output
        assert mock_edit.call_count == 0
        assert "1 annotated" in result.output, result.output
        assert "1 skipped" in result.output, result.output


class TestPerWriterReport:
    """Per-writer report (IAM-03) — turned GREEN in Phase 07-04."""

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_per_writer_report_groups_by_student(self, mock_read, mock_providers, db_path, tmp_path):
        from PIL import Image

        # Seed two IAM-tagged samples for two different writers
        conn = get_connection(db_path)
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        Image.new("RGB", (200, 200), color=(128, 128, 128)).save(img_a)
        Image.new("RGB", (200, 200), color=(128, 128, 128)).save(img_b)
        sid_a = insert_sample(conn, str(img_a), "hashA",
                              student="iam-writer-a01", category="iam")
        sid_b = insert_sample(conn, str(img_b), "hashB",
                              student="iam-writer-b02", category="iam")
        insert_ground_truth(conn, sid_a, "the mitochondria is the powerhouse of the cell")
        insert_ground_truth(conn, sid_b, "the mitochondria is the powerhouse of the cell")
        conn.close()

        mock_providers.return_value = ["gemini"]
        # Writer a01 perfect, writer b02 has one substitution
        def fake_read(path, *args, **kwargs):
            if "a.png" in path:
                text = "the mitochondria is the powerhouse of the cell"
            else:
                text = "the mitochondria is the powerhouse of the sell"
            return {
                "text": text,
                "confidence": 0.7, "latency_ms": 500,
                "input_tokens": 100, "output_tokens": 50, "error": None,
            }
        mock_read.side_effect = fake_read

        run_id = run_benchmark(
            providers=["gemini"], strategies=[], db_path=db_path,
        )

        report = generate_per_writer_report(run_id=run_id, db_path=db_path)
        assert "iam-writer-a01" in report
        assert "iam-writer-b02" in report
        assert "Writer" in report  # header
        assert "Mean CER" in report

    @patch("handwriting_engine.benchmark.evaluate._available_providers")
    @patch("handwriting_engine.benchmark.evaluate._read_single")
    def test_per_writer_report_no_writers(self, mock_read, mock_providers, seeded_db):
        # seeded_db has student="test" — but the function only includes writers
        # with non-empty student. The seeded sample has student="test", so let's
        # blank it out to trigger the empty path.
        conn = get_connection(seeded_db)
        try:
            conn.execute("UPDATE samples SET student=''")
            conn.commit()
        finally:
            conn.close()

        mock_providers.return_value = ["gemini"]
        mock_read.return_value = {
            "text": "the mitochondria is the powerhouse of the cell",
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
        run_id = run_benchmark(
            providers=["gemini"], strategies=[], db_path=seeded_db,
        )
        report = generate_per_writer_report(run_id=run_id, db_path=seeded_db)
        assert "No writer data" in report

    def test_report_cli_per_writer_flag(self, tmp_path):
        from handwriting_engine.cli import cli

        db_path = tmp_path / "empty.db"
        get_connection(db_path).close()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["benchmark", "report", "--per-writer", "--db-path", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        # Empty DB: function returns "No runs found" or similar
        assert (
            "No runs found" in result.output
            or "No writer data" in result.output
            or "Per-Writer CER" in result.output
        )

        # Help advertises the flag
        help_result = runner.invoke(cli, ["benchmark", "report", "--help"])
        assert "--per-writer" in help_result.output
