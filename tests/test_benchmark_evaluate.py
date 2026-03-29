"""Tests for benchmark evaluation — uses mocks to avoid real API calls."""

import pytest
from unittest.mock import patch, MagicMock

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
