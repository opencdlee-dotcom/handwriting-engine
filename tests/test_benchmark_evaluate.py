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
    """RED stubs for sweep infrastructure (IAM-02). All must FAIL until Wave 2."""

    # These are TODO stubs, not tests: each body is a bare pytest.fail() naming work that
    # was never done. Left as hard failures they made the suite permanently red, and a
    # suite that is always red cannot gate a dependency upgrade -- which is how 25
    # security advisories sat unactioned. Declared as expected failures they still print
    # their message on every run (counted as "xfailed"), so the debt stays visible while
    # a NEW failure is once again the only reason the suite goes red. Deleting the marker
    # is part of implementing sweep infrastructure (IAM-02).
    pytestmark = pytest.mark.xfail(
        reason="RED stub: sweep infrastructure (IAM-02) is unimplemented", strict=False
    )

    def test_run_benchmark_accepts_line_level(self, seeded_db):
        pytest.fail(
            "not implemented — run_benchmark must accept line_level=True "
            "and thread it through to _read_single"
        )

    def test_run_benchmark_accepts_auto_retry(self, seeded_db):
        pytest.fail(
            "not implemented — run_benchmark must accept auto_retry=True "
            "and thread it through to _read_single"
        )

    def test_run_sweep_returns_five_run_ids(self, seeded_db):
        pytest.fail(
            "not implemented — run_sweep must return a dict with exactly 5 keys: "
            "baseline, self_correct, line_level, prompt_adapted, zoomed_verify"
        )

    def test_sweep_cli_shows_cost(self, tmp_path):
        pytest.fail(
            "not implemented — `benchmark sweep` CLI must print projected cost "
            "before any API call (even with no real samples)"
        )

    def test_sweep_cli_yes_executes(self, tmp_path):
        pytest.fail(
            "not implemented — `benchmark sweep --yes` must bypass cost confirmation "
            "and attempt to execute all 5 strategies"
        )


class TestPerWriterReport:
    """RED stubs for per-writer report (IAM-03). All must FAIL until Wave 2."""

    # These are TODO stubs, not tests: each body is a bare pytest.fail() naming work that
    # was never done. Left as hard failures they made the suite permanently red, and a
    # suite that is always red cannot gate a dependency upgrade -- which is how 25
    # security advisories sat unactioned. Declared as expected failures they still print
    # their message on every run (counted as "xfailed"), so the debt stays visible while
    # a NEW failure is once again the only reason the suite goes red. Deleting the marker
    # is part of implementing the per-writer report (IAM-03).
    pytestmark = pytest.mark.xfail(
        reason="RED stub: the per-writer report (IAM-03) is unimplemented", strict=False
    )

    def test_per_writer_report_groups_by_student(self, seeded_db):
        pytest.fail(
            "not implemented — generate_per_writer_report must group CER by "
            "samples.student and return a formatted table string"
        )

    def test_per_writer_report_no_writers(self, seeded_db):
        pytest.fail(
            "not implemented — generate_per_writer_report on run with no student "
            "data must return a message indicating no writer data available"
        )

    def test_report_cli_per_writer_flag(self, tmp_path):
        pytest.fail(
            "not implemented — `benchmark report --per-writer` CLI flag must exist "
            "and invoke generate_per_writer_report"
        )
