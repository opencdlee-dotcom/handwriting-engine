"""End-to-end integration: run_sweep -> compare_runs -> recommend_strategy.

Closes the gap that `benchmark sweep` is the *only* path that populates the
multi-run shape STAT-01 (Wilcoxon), STAT-02 (bootstrap CI + Cohen's r), and
RPT-02 (composite-score recommend) consume. Unit tests for those three stamp
synthetic eval_metrics rows directly; this test instead drives the real
sweep -> evaluator -> aggregator pipeline against a mocked provider so the
DB schema, n>=10 gating, and pairing logic are all exercised together.

No API calls — `_read_single` is mocked to emit deterministic per-strategy
text variants of the ground truth, producing distinct CER distributions per
run_id while keeping the same (provider, strategy) key (single-provider
sweep collapses to one key, which is the product-as-designed behavior).
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
from handwriting_engine.benchmark.evaluate import run_benchmark, run_sweep
from handwriting_engine.benchmark.report import compare_runs, recommend_strategy


_GT = "the mitochondria is the powerhouse of the cell"
_N_SAMPLES = 12  # >= _STATS_MIN_PAIRED_N (10) so the stats block must appear


@pytest.fixture
def iam_seeded_db(tmp_path):
    """12 IAM-categorized samples, each with a real on-disk image and ground truth."""
    db_path = tmp_path / "sweep_integration.db"
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


# Sweep strategy order matches SWEEP_STRATEGIES in evaluate.py. The kwargs
# that distinguish baseline from prompt_adapted (vocab_hints_off, auto_enhance,
# strategies list) live at the run_benchmark layer and aren't visible inside
# _read_single, so the mock disambiguates by call-count / N_SAMPLES instead.
_STRATEGY_ORDER = ["baseline", "self_correct", "line_level", "prompt_adapted", "zoomed_verify"]


def _strategy_text(strategy_name: str, sample_idx: int) -> str:
    """Return a per-strategy variant of the ground truth.

    Variants are chosen so prompt_adapted lands at perfect CER and baseline
    lands at a uniformly worse CER. That gives Wilcoxon clean directional
    signal (every paired diff has the same sign) for a low p-value with n=12.
    """
    if strategy_name == "baseline":
        # Drop the last word every sample → ~22% CER on every sample, all paired
        # diffs vs. perfect prompt_adapted are positive.
        return _GT.replace(" of the cell", "")
    if strategy_name == "line_level":
        if sample_idx % 3 == 0:
            return _GT.replace("powerhouse", "powerhose")
        return _GT
    if strategy_name == "zoomed_verify":
        if sample_idx % 2 == 0:
            return _GT.replace("powerhouse", "powerhose")
        return _GT
    # self_correct + prompt_adapted: perfect (with 1 provider, self_correct
    # falls back to single-provider read).
    return _GT


def _make_read_single_mock(n_samples: int):
    """Return a side_effect callable that varies output by strategy + sample.

    Uses a call counter divided by `n_samples` to infer which sweep run is in
    flight (SWEEP_STRATEGIES are executed sequentially, n_samples calls each).
    """
    state = {"call_count": 0}

    def _side_effect(image_path, *args, **kwargs):
        idx_in_run = state["call_count"] % n_samples
        run_idx = (state["call_count"] // n_samples) % len(_STRATEGY_ORDER)
        state["call_count"] += 1

        strategy_name = _STRATEGY_ORDER[run_idx]
        text = _strategy_text(strategy_name, idx_in_run)
        return {
            "text": text,
            "confidence": 0.7,
            "latency_ms": 500,
            "input_tokens": 100,
            "output_tokens": 50,
            "error": None,
        }

    return _side_effect, state


@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_then_compare_emits_stats_block(mock_read, mock_providers, iam_seeded_db):
    """run_sweep populates 5 runs; compare_runs(baseline, prompt_adapted) must
    print the n>=10 stats block (Wilcoxon p, Cohen's r, bootstrap CI95)."""
    mock_providers.return_value = ["gemini"]
    side_effect, _state = _make_read_single_mock(_N_SAMPLES)
    mock_read.side_effect = side_effect

    run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)
    assert set(run_ids.keys()) == {
        "baseline", "self_correct", "line_level", "prompt_adapted", "zoomed_verify",
    }

    # Pair the run that has consistently worse CER (baseline) against the one
    # with consistently better CER (prompt_adapted). Wilcoxon should clearly
    # resolve: every paired diff is non-zero and same-signed.
    out = compare_runs(run_ids["baseline"], run_ids["prompt_adapted"], db_path=iam_seeded_db)

    # STAT-01 + STAT-02 acceptance shape (matches existing test_stats_block_*):
    assert "stats:" in out, f"Wilcoxon stats block missing:\n{out}"
    assert "CI95:" in out, f"bootstrap CI block missing:\n{out}"
    assert f"n={_N_SAMPLES}" in out, f"expected n={_N_SAMPLES} in:\n{out}"

    # baseline (worse CER) - prompt_adapted (perfect) should yield p < 0.05.
    import re
    m = re.search(r"p=([\d.]+)", out)
    assert m, f"p-value not found:\n{out}"
    assert float(m.group(1)) < 0.05, f"expected significant p, got: {out}"

    # Cohen's r should be present and non-trivial (effect is large by construction).
    m_r = re.search(r"r=([\d.]+)", out)
    assert m_r, f"Cohen's r not found:\n{out}"
    assert float(m_r.group(1)) > 0.3, f"expected medium+ effect size, got: {out}"


@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_sweep_then_compare_no_diff_yields_high_p(mock_read, mock_providers, iam_seeded_db):
    """Sanity check: when baseline and prompt_adapted produce identical text,
    Wilcoxon should NOT spuriously reject. Catches a regression where the
    pairing logic accidentally pairs unrelated rows."""
    mock_providers.return_value = ["gemini"]
    mock_read.return_value = {
        "text": _GT,
        "confidence": 0.7, "latency_ms": 500,
        "input_tokens": 100, "output_tokens": 50, "error": None,
    }

    run_ids = run_sweep(provider="gemini", db_path=iam_seeded_db, yes=True)
    out = compare_runs(run_ids["baseline"], run_ids["prompt_adapted"], db_path=iam_seeded_db)

    # All paired diffs are zero; Wilcoxon convention drops them, n -> 0 and
    # the stats block either elides (n < 10) or reports p=1.0. Either is
    # acceptable; what we're guarding against is a false positive.
    if "stats:" in out:
        import re
        m = re.search(r"p=([\d.]+)", out)
        assert m, f"p-value not found:\n{out}"
        # When all diffs are zero, p collapses to 1.0 by the implementation.
        assert float(m.group(1)) > 0.5, f"unexpected significant p with zero deltas:\n{out}"


@patch("handwriting_engine.benchmark.evaluate._available_providers")
@patch("handwriting_engine.benchmark.evaluate._read_single")
def test_recommend_picks_lowest_cer_winner(mock_read, mock_providers, iam_seeded_db):
    """RPT-02: recommend_strategy must rank a known-better config above a
    known-worse one when both are populated by the real run_benchmark path.

    Single-provider sweep collapses to one (provider, strategy) key, so we
    drive recommend with two run_benchmark invocations using *different*
    provider labels — that's the multi-provider shape recommend was designed
    for. The mock keys output text on provider name."""
    mock_providers.return_value = ["fast_provider", "slow_provider"]

    def _by_provider(image_path, provider, *args, **kwargs):
        # fast_provider lands at perfect CER; slow_provider drops the trailing
        # phrase on every sample (~22% CER). Composite score should crown
        # fast_provider.
        text = _GT if provider == "fast_provider" else _GT.replace(" of the cell", "")
        return {
            "text": text,
            "confidence": 0.7, "latency_ms": 500,
            "input_tokens": 100, "output_tokens": 50, "error": None,
        }
    mock_read.side_effect = _by_provider

    run_benchmark(label="fast", providers=["fast_provider"], strategies=[],
                  db_path=iam_seeded_db)
    run_benchmark(label="slow", providers=["slow_provider"], strategies=[],
                  db_path=iam_seeded_db)

    out = recommend_strategy(db_path=iam_seeded_db)

    assert "Winner:" in out, f"recommend output missing winner header:\n{out}"
    assert "fast_provider" in out, f"fast_provider missing from output:\n{out}"
    assert "slow_provider" in out, f"slow_provider missing from output:\n{out}"

    # Winner line is the first "Winner:" — must name fast_provider.
    winner_line = next(line for line in out.splitlines() if "Winner:" in line)
    assert "fast_provider" in winner_line, (
        f"expected fast_provider as winner, got: {winner_line!r}"
    )

    # Composite score: fast_provider should rank #1, slow_provider #2.
    rank_lines = [l for l in out.splitlines() if l.lstrip().startswith(("1 ", "2 "))]
    assert len(rank_lines) >= 2, f"expected ranked rows, got:\n{out}"
    assert "fast_provider" in rank_lines[0]
    assert "slow_provider" in rank_lines[1]
