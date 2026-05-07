---
phase: 09-final-sweep-recommendation-baseline-lock
plan: "02"
subsystem: benchmark
tags: [recommend, composite-score, normalization, cli]

requires:
  - phase: 09-01
    provides: schema v6 / list_runs surface (is_baseline indirectly informs future scoring versions)
  - phase: 07
    provides: get_run_results for per-(provider, strategy) aggregation

provides:
  - benchmark/report.py:recommend_strategy(db_path) -> str
  - CLI: benchmark recommend
  - _RECOMMEND_W_CER / _W_COST / _W_STAB constants — single source of truth for the 70/15/15 weights

affects:
  - Strategy selection workflow — replaces ad-hoc CER staring with a normalized composite score
  - Future RPT-related decisions can lean on the same composite score

tech-stack:
  added: []
  patterns:
    - Min-max normalization within candidate set (per-component)
    - Stability = 1 - normalized(across-run-mean stdev). Single-run candidates get the median stability score (neutral)
    - Pre-Phase-9 schema bumped: CURRENT_SCHEMA_VERSION 5 -> 6 to silence the spurious "migration failed" warning that fresh DBs were emitting because the base _SCHEMA_SQL already includes is_baseline

key-files:
  created:
    - tests/test_benchmark_recommend.py
  modified:
    - handwriting_engine/benchmark/report.py
    - handwriting_engine/benchmark/db.py (CURRENT_SCHEMA_VERSION bump)
    - handwriting_engine/cli.py

key-decisions:
  - "Hand the median to single-run candidates. They can't measure across-run variance; penalizing or rewarding them based on absent data would be noise. Median stability is the neutral choice."
  - "Min-max normalize per-component. Different metrics live on different scales; normalizing to [0, 1] before weighting is the standard composite-score recipe and means the weights mean what they look like."
  - "Bump CURRENT_SCHEMA_VERSION to 6 even though Plan 09-01 already added the migration. Without this bump, fresh DBs (whose base _SCHEMA_SQL already has is_baseline) log a spurious 'migration failed' warning when v6 ALTER tries to add a column that already exists. The warning was harmless but noisy."

verification:
  unit_coverage:
    - 9 tests in tests/test_benchmark_recommend.py
  criterion_status:
    - "RPT-02: IMPLEMENTED on synthetic data. End-to-end verification against multi-strategy IAM sweep gated on user IAM download."
