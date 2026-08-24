"""Tests for Phase 8 statistics layer (STAT-01, STAT-02).

Covers paired Wilcoxon signed-rank, percentile-method bootstrap CI, and
Cohen's r against scipy-validated reference values where applicable.
"""

import pytest

from handwriting_engine.benchmark.stats import (
    WilcoxonResult,
    wilcoxon_signed_rank,
    bootstrap_ci,
    cohens_r,
    _rank_with_ties,
    _normal_cdf,
)


# --- _rank_with_ties ---

class TestRankWithTies:
    def test_no_ties(self):
        ranks, tie_corr = _rank_with_ties([3.0, 1.0, 2.0])
        assert ranks == [3.0, 1.0, 2.0]
        assert tie_corr == 0.0

    def test_two_way_tie(self):
        # Values [1, 1, 2] — the two 1s share ranks 1 and 2 → average 1.5
        ranks, tie_corr = _rank_with_ties([1.0, 1.0, 2.0])
        assert ranks == [1.5, 1.5, 3.0]
        # tie correction = 2^3 - 2 = 6
        assert tie_corr == 6.0

    def test_three_way_tie(self):
        ranks, tie_corr = _rank_with_ties([5.0, 5.0, 5.0, 9.0])
        # Three tied at ranks 1,2,3 → average 2.0
        assert ranks == [2.0, 2.0, 2.0, 4.0]
        # tie correction = 3^3 - 3 = 24
        assert tie_corr == 24.0


# --- _normal_cdf ---

class TestNormalCdf:
    def test_centered(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=1e-9)

    def test_one_sigma(self):
        # P(Z <= 1) ≈ 0.8413
        assert _normal_cdf(1.0) == pytest.approx(0.8413, abs=1e-3)

    def test_two_sigma(self):
        # P(Z <= 2) ≈ 0.9772
        assert _normal_cdf(2.0) == pytest.approx(0.9772, abs=1e-3)

    def test_negative(self):
        assert _normal_cdf(-1.0) == pytest.approx(1.0 - 0.8413, abs=1e-3)


# --- wilcoxon_signed_rank ---

class TestWilcoxonSignedRank:
    def test_returns_wilcoxon_result_dataclass(self):
        # n >= 10 paired samples with a = b - 0.01 (a is consistently lower)
        a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        b = [0.11, 0.13, 0.12, 0.14, 0.10, 0.15, 0.12, 0.11, 0.13, 0.14]
        result = wilcoxon_signed_rank(a, b)
        assert isinstance(result, WilcoxonResult)

    def test_paired_difference_significant(self):
        # All differences negative and consistent: strong signal, p should be small.
        # n = 10, all a_i < b_i  →  all rank-sums on the negative side.
        a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        b = [0.11, 0.13, 0.12, 0.14, 0.10, 0.15, 0.12, 0.11, 0.13, 0.14]
        result = wilcoxon_signed_rank(a, b)
        assert result.p_value < 0.05
        assert result.n == 10
        # b is consistently larger → w_plus is small (most ranks went to negative diffs)
        assert result.statistic < (result.n * (result.n + 1) / 4.0)

    def test_no_difference_yields_high_p(self):
        # Identical samples → all diffs zero → p=1
        a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        result = wilcoxon_signed_rank(a, a)
        assert result.p_value == 1.0
        assert result.n == 0

    def test_random_noise_yields_high_p(self):
        # Symmetric noise around zero diff → no signal → p should not reject
        a = [0.10, 0.13, 0.11, 0.12, 0.10, 0.13, 0.11, 0.12, 0.10, 0.13]
        b = [0.11, 0.12, 0.12, 0.11, 0.11, 0.12, 0.12, 0.11, 0.11, 0.12]
        result = wilcoxon_signed_rank(a, b)
        assert result.p_value > 0.05

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank([0.1, 0.2], [0.1, 0.2, 0.3])

    def test_two_sided_symmetry(self):
        # Swapping a and b must flip sign of z but leave |z| and p_value identical
        a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        b = [0.11, 0.13, 0.12, 0.14, 0.10, 0.15, 0.12, 0.11, 0.13, 0.14]
        r1 = wilcoxon_signed_rank(a, b)
        r2 = wilcoxon_signed_rank(b, a)
        assert r1.p_value == pytest.approx(r2.p_value, abs=1e-9)
        assert abs(r1.z) == pytest.approx(abs(r2.z), abs=1e-9)
        # z should flip sign
        assert (r1.z > 0) != (r2.z > 0)

    def test_small_n_still_runs(self):
        # n = 3 — function should still return a result; callers gate at >=10
        a = [0.1, 0.2, 0.3]
        b = [0.2, 0.3, 0.4]
        result = wilcoxon_signed_rank(a, b)
        assert 0.0 <= result.p_value <= 1.0
        assert result.n == 3

    def test_zero_diffs_dropped(self):
        # 8 paired with diff zero, 2 paired with positive diff. n_effective=2.
        a = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
        b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        result = wilcoxon_signed_rank(a, b)
        assert result.n == 2

    def test_p_value_in_unit_interval(self):
        # Property: p_value must always be in [0, 1].
        cases = [
            ([0.1] * 10, [0.2] * 10),
            ([0.5] * 10, [0.5] * 10),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        ]
        for a, b in cases:
            result = wilcoxon_signed_rank(a, b)
            assert 0.0 <= result.p_value <= 1.0


# --- bootstrap_ci ---

class TestBootstrapCi:
    def test_zero_samples_returns_zero_band(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_single_sample_returns_degenerate_band(self):
        assert bootstrap_ci([0.42]) == (0.42, 0.42)

    def test_constant_sample_returns_constant_band(self):
        # All values 0.1 → resample of any size → mean 0.1
        lo, hi = bootstrap_ci([0.1] * 20, seed=42)
        assert lo == pytest.approx(0.1, abs=1e-9)
        assert hi == pytest.approx(0.1, abs=1e-9)

    def test_variable_sample_brackets_mean(self):
        # 30 samples around 0.15 with realistic CER spread; the CI must
        # contain the empirical mean and have non-zero width.
        values = [0.10, 0.12, 0.14, 0.13, 0.15, 0.11, 0.18, 0.20, 0.13, 0.14,
                  0.15, 0.16, 0.17, 0.12, 0.13, 0.14, 0.11, 0.10, 0.18, 0.19,
                  0.13, 0.14, 0.15, 0.16, 0.12, 0.13, 0.11, 0.10, 0.17, 0.18]
        lo, hi = bootstrap_ci(values, seed=42)
        mean = sum(values) / len(values)
        assert lo < mean < hi
        assert (hi - lo) > 0.0

    def test_higher_confidence_yields_wider_band(self):
        values = [0.1, 0.2, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.11]
        lo_90, hi_90 = bootstrap_ci(values, confidence=0.90, seed=42)
        lo_99, hi_99 = bootstrap_ci(values, confidence=0.99, seed=42)
        assert (hi_99 - lo_99) >= (hi_90 - lo_90)

    def test_seed_makes_output_deterministic(self):
        values = [0.1, 0.2, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.11]
        a = bootstrap_ci(values, seed=123)
        b = bootstrap_ci(values, seed=123)
        assert a == b

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([0.1, 0.2], confidence=1.5)
        with pytest.raises(ValueError):
            bootstrap_ci([0.1, 0.2], confidence=0.0)

    def test_invalid_n_iterations_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([0.1, 0.2], n_iterations=0)


# --- cohens_r ---

class TestCohensR:
    def test_zero_z_yields_zero(self):
        assert cohens_r(0.0, 10) == 0.0

    def test_zero_n_returns_zero(self):
        # Guard against div-by-zero
        assert cohens_r(2.0, 0) == 0.0

    def test_classic_thresholds(self):
        # r = z / sqrt(n)
        # n=25, z=2.5 → r=0.5 (large)
        assert cohens_r(2.5, 25) == pytest.approx(0.5, abs=1e-9)

    def test_negative_z_returns_positive_r(self):
        # Sign of effect lives in z; r is magnitude.
        assert cohens_r(-2.5, 25) == pytest.approx(0.5, abs=1e-9)


# --- end-to-end paired example ---

class TestPairedRunComparison:
    """Mirrors the production wiring path: two runs, n=10 paired by sample."""

    def test_strategy_b_consistently_better(self):
        # Strategy A: ~10% CER. Strategy B: ~6% CER on same samples.
        cer_a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        cer_b = [0.06, 0.08, 0.07, 0.09, 0.05, 0.10, 0.07, 0.06, 0.08, 0.09]

        result = wilcoxon_signed_rank(cer_a, cer_b)
        # All 10 pairs favour B → significant.
        assert result.p_value < 0.01
        # Effect size should be large.
        assert cohens_r(result.z, result.n) > 0.5

        # Bootstrap CIs separate A and B if effect is strong:
        lo_a, hi_a = bootstrap_ci(cer_a, seed=0)
        lo_b, hi_b = bootstrap_ci(cer_b, seed=0)
        # Either A's CI is wholly above B's, or they barely overlap.
        # With this strong separation, no overlap.
        assert lo_a > hi_b or hi_a < lo_b


# --- compare_runs() integration ---

class TestCompareRunsStatsBlock:
    """Integration: compare_runs() must append a stats block when n>=10
    paired samples exist for the same (provider, strategy)."""

    def _seed_two_runs(self, db_path, n_samples, cer_run1, cer_run2):
        """Create two runs sharing n samples, with explicit CERs for each."""
        from PIL import Image
        from handwriting_engine.benchmark.db import (
            get_connection, insert_sample, insert_ground_truth,
            insert_run, finish_run, insert_provider_output, insert_eval_metric,
        )

        conn = get_connection(db_path)
        sample_ids = []
        gt_ids = []
        # Lazy: make a single image; insert_sample requires a real path.
        import tempfile, os
        img_dir = tempfile.mkdtemp()
        for i in range(n_samples):
            img_path = os.path.join(img_dir, f"img_{i}.png")
            Image.new("RGB", (32, 32), (128, 128, 128)).save(img_path)
            sid = insert_sample(conn, img_path, f"hash_{i}", student=f"writer_{i % 3}")
            gt_id = insert_ground_truth(conn, sid, "ground truth text")
            sample_ids.append(sid)
            gt_ids.append(gt_id)

        def _seed_run(label, cers):
            run_id = insert_run(conn, label=label, providers=["gemini"], strategies=["vote"])
            for sid, gt_id, cer in zip(sample_ids, gt_ids, cers):
                po_id = insert_provider_output(
                    conn, run_id=run_id, sample_id=sid,
                    provider="gemini", strategy="vote",
                    output_text="x", confidence=0.9,
                )
                insert_eval_metric(
                    conn, provider_output_id=po_id, ground_truth_id=gt_id,
                    cer=cer, wer=cer,
                )
            finish_run(conn, run_id, len(cers))
            return run_id

        r1 = _seed_run("run_1", cer_run1)
        r2 = _seed_run("run_2", cer_run2)
        conn.close()
        return r1, r2

    def test_stats_block_appears_for_n_geq_10(self, tmp_path):
        from handwriting_engine.benchmark.report import compare_runs

        db = tmp_path / "stats.db"
        # 12 paired samples; run_2 consistently better.
        cer_a = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13, 0.15, 0.16]
        cer_b = [0.06, 0.08, 0.07, 0.09, 0.05, 0.10, 0.07, 0.06, 0.08, 0.09, 0.11, 0.12]
        r1, r2 = self._seed_two_runs(db, 12, cer_a, cer_b)

        out = compare_runs(r1, r2, db_path=db)
        assert "stats:" in out
        assert "CI95:" in out
        assert "n=12" in out
        # Strong separation → low p-value.
        # We can't pin the exact value but it should clearly be < 0.05.
        import re
        m = re.search(r"p=([\d.]+)", out)
        assert m, f"p-value not in output:\n{out}"
        assert float(m.group(1)) < 0.05

    def test_stats_block_omitted_when_n_lt_10(self, tmp_path):
        from handwriting_engine.benchmark.report import compare_runs

        db = tmp_path / "stats_small.db"
        cer_a = [0.10, 0.12, 0.11, 0.13, 0.09]
        cer_b = [0.06, 0.08, 0.07, 0.09, 0.05]
        r1, r2 = self._seed_two_runs(db, 5, cer_a, cer_b)

        out = compare_runs(r1, r2, db_path=db)
        assert "stats:" not in out
        assert "CI95:" not in out

    def test_stats_block_present_for_no_difference(self, tmp_path):
        # Identical runs → p should be high; the block must still render.
        from handwriting_engine.benchmark.report import compare_runs

        db = tmp_path / "stats_identical.db"
        cer = [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.11, 0.10, 0.12, 0.13]
        r1, r2 = self._seed_two_runs(db, 10, cer, list(cer))

        out = compare_runs(r1, r2, db_path=db)
        # All diffs are zero — n_effective is 0, but the block still shows
        # n=0 and p=1.0 (the function gates on len(paired_a) >= 10, not on
        # n_effective). That's the right behavior: tells the user "we tried
        # the test and there was no signal," vs. silently dropping it.
        assert "stats:" in out
        assert "p=1.0000" in out
