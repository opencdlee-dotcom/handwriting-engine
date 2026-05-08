"""Tests for the trocr_local sweep strategy and per-strategy provider override.

Covers Stage 1 of the local-first OCR rollout: the sweep can now compare
zero-token TrOCR runs against cloud strategies on the same IAM samples.

These tests mock both `_read_single` and `_available_providers` so they run
without torch/transformers installed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_ground_truth,
    insert_sample,
)
from handwriting_engine.benchmark.evaluate import (
    SWEEP_STRATEGIES,
    _strategy_providers,
    run_sweep,
)


_GT = "the mitochondria is the powerhouse of the cell"
_N_SAMPLES = 4


@pytest.fixture
def iam_seeded_db(tmp_path):
    db_path = tmp_path / "trocr_sweep.db"
    conn = get_connection(db_path)
    try:
        for i in range(_N_SAMPLES):
            img_path = tmp_path / f"iam_{i:02d}.png"
            Image.new("RGB", (200, 60), color=(240, 240, 240)).save(img_path)
            sid = insert_sample(
                conn,
                str(img_path),
                f"hash_iam_{i:02d}",
                student=f"iam-writer-{i:03d}",
                category="iam",
            )
            insert_ground_truth(conn, sid, _GT)
    finally:
        conn.close()
    return db_path


def test_trocr_local_present_in_sweep_strategies():
    names = [s["name"] for s in SWEEP_STRATEGIES]
    assert "trocr_local" in names, (
        "trocr_local must be in SWEEP_STRATEGIES so `benchmark sweep` "
        "produces a head-to-head local-vs-cloud row."
    )

    trocr = next(s for s in SWEEP_STRATEGIES if s["name"] == "trocr_local")
    assert trocr.get("providers") == ["trocr"], (
        "trocr_local must pin provider=trocr so it ignores the sweep's "
        "--provider flag (the whole point is 'no cloud at all')."
    )


def test_strategy_providers_override_beats_default():
    """Per-strategy `providers` field wins over the sweep-level default."""
    cfg_with_override = {"providers": ["trocr"]}
    cfg_without = {}
    assert _strategy_providers(cfg_with_override, "gemini") == ["trocr"]
    assert _strategy_providers(cfg_without, "gemini") == ["gemini"]


def test_cloud_strategies_have_no_provider_override():
    """The cost-projection logic in cli.benchmark_sweep used to separate cloud
    vs local by checking `s.get("providers")`. With Stage 2's mixed-cost
    local_first strategies, that heuristic became ambiguous (local_first does
    pin providers but DOES cost cloud tokens). Updated rule: any strategy
    that pins providers must declare its cost_class explicitly so the CLI
    projects honest dollar amounts."""
    for s in SWEEP_STRATEGIES:
        if s.get("providers"):
            assert "cost_class" in s, (
                f"Strategy {s['name']!r} pins providers but lacks cost_class. "
                f"Without cost_class the CLI cost projection silently mis-counts."
            )
            assert s["cost_class"] in {"local", "mixed", "cloud"}, (
                f"Strategy {s['name']!r} cost_class={s['cost_class']!r} is "
                f"not one of local/mixed/cloud — CLI doesn't know how to price it."
            )


@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_includes_trocr_when_available(mock_read, mock_providers, iam_seeded_db):
    """When TrOCR is installed, the sweep produces a trocr_local run_id."""
    mock_providers.return_value = ["gemini", "trocr"]
    mock_read.return_value = {
        "text": _GT,
        "confidence": 0.9, "latency_ms": 100,
        "input_tokens": 0, "output_tokens": 0, "error": None,
    }

    run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)
    assert "trocr_local" in run_ids, (
        f"trocr_local missing from sweep results: {list(run_ids)}"
    )
    # Cloud strategies still run.
    assert "baseline" in run_ids
    assert "prompt_adapted" in run_ids


@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_skips_trocr_when_unavailable(mock_read, mock_providers, iam_seeded_db, caplog):
    """When TrOCR is NOT installed, the sweep skips trocr_local with a
    warning rather than crashing the whole run."""
    mock_providers.return_value = ["gemini"]  # no trocr
    mock_read.return_value = {
        "text": _GT,
        "confidence": 0.7, "latency_ms": 500,
        "input_tokens": 100, "output_tokens": 50, "error": None,
    }

    import logging
    with caplog.at_level(logging.WARNING):
        run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)

    assert "trocr_local" not in run_ids, (
        "trocr_local should be skipped when its provider isn't available."
    )
    # All 5 cloud strategies still produce runs.
    assert {"baseline", "self_correct", "line_level", "prompt_adapted", "zoomed_verify"} <= set(run_ids)
    # And the user gets told why.
    assert any("trocr_local" in r.message and "unavailable" in r.message for r in caplog.records), (
        f"Expected a warning naming trocr_local + unavailable, got: "
        f"{[r.message for r in caplog.records]}"
    )
