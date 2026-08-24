---
phase: 08-statistics-layer
plan: "01"
subsystem: benchmark
tags: [stats, wilcoxon, bootstrap, cohen, no-scipy, tdd]

requires:
  - phase: 07
    provides: get_run_results returning per-sample CER rows that paired analysis can consume

provides:
  - handwriting_engine/benchmark/stats.py (new module)
  - wilcoxon_signed_rank(a, b) -> WilcoxonResult{statistic, p_value, z, n}
  - bootstrap_ci(values, confidence, n_iterations, seed) -> (lo, hi)
  - cohens_r(z, n) -> float
  - compare_runs() emits a "stats:" + "CI95:" block per (provider, strategy) when n_paired >= 10

affects:
  - benchmark compare RUN_A RUN_B — output now includes paired Wilcoxon, Cohen's r, and bootstrap CIs
  - Future Phase 9 recommendation logic (RPT-02) — composite-score "stability" component can lean on these stats

tech-stack:
  added: []
  patterns:
    - Hand-rolled Wilcoxon with normal-approximation + continuity correction (no scipy dep)
    - Average-rank tie handling with explicit tie-correction term (n^3 - n) / 48 in variance
    - Percentile-method bootstrap with seedable RNG for deterministic test output
    - Pairing by sample_id intersection across the two runs (only shared samples count toward n)

key-files:
  created:
    - handwriting_engine/benchmark/stats.py
    - tests/test_benchmark_stats.py
  modified:
    - handwriting_engine/benchmark/report.py

key-decisions:
  - "Hand-roll Wilcoxon and bootstrap rather than add scipy. Trade ~150 LOC of well-tested math for not pulling a 30 MB scientific stack into a project whose existing pattern (Levenshtein in postprocess.py) is hand-rolled small math."
  - "Gate stats block at n_paired >= 10. Below this, normal-approximation Wilcoxon is rough and bootstrap CIs are dominated by sampling noise — better to print nothing than mislead. Aligns with the success-criterion threshold."
  - "Pair by sample_id intersection. Only samples that exist in BOTH runs count. Avoids comparing apples to oranges when run sample sets diverge."
  - "Bootstrap CI seeded by run_id. Deterministic output for the same run pair without exposing a CLI flag."
  - "Two-sided p-value with continuity correction. Matches scipy's wilcoxon(zero_method='wilcox') default; the continuity correction prevents over-rejecting the null on small n."

patterns-established:
  - "Phase 8+ statistics live in a single benchmark/stats.py module; report.py imports from it. Keeps math out of presentation logic and makes the math reusable for Phase 9 / RPT-02."

verification:
  unit_coverage:
    - 29 tests in tests/test_benchmark_stats.py for the three primitives plus a paired end-to-end scenario
    - 3 integration tests exercising compare_runs() against a synthetic two-run DB
  criterion_status:
    - "STAT-01 (#1): Wilcoxon p-value + Cohen's r appear in compare_runs output when n>=10 — IMPLEMENTED, verified on synthetic data; end-to-end IAM verification gated on user IAM download (see .planning/NEXT-STEPS.md)"
    - "STAT-02 (#2): 95% bootstrap CIs for both runs appear in compare_runs output — IMPLEMENTED, verified on synthetic data; same IAM gate"
  pre_existing_failures:
    - tests/test_enhance.py (cv2 missing in this venv)
    - tests/test_trained_correction.py::TestConfidenceGate (transformers/torch missing in this venv)

out_of_scope:
  - "scipy.stats.wilcoxon equivalence audit. We hand-rolled to avoid the dep; if a future user needs scipy parity to ~6 decimals, swap the body of wilcoxon_signed_rank for scipy.stats.wilcoxon — the public WilcoxonResult shape stays."
  - "Two-sample (unpaired) tests. The contract is paired comparison of CERs on the same samples; cross-sample-set comparison is a separate concern."
  - "BCa bootstrap. Percentile method is sufficient for displayed CIs at this sample size; BCa correction for skew is a v2 concern."
  - "STAT-03 (McNemar's), STAT-04 (per-character confusion). v4.0 deferred per REQUIREMENTS.md."
