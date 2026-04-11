# Phase 6: Measurement Foundation - Context

**Gathered:** 2026-04-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Add reproducibility infrastructure to the benchmark CLI: provenance tracking on run records, `[?]_marker_rate` as a separate stored metric alongside CER, a new `benchmark calibrate` command for noise floor measurement, and a cost projection guardrail on `benchmark run`. No new benchmark strategies, no IAM ingestion (Phase 7), no statistics layer (Phase 8).

</domain>

<decisions>
## Implementation Decisions

### Provenance record (FOUND-01)

- **Schema:** v4 migration adds four columns to the `runs` table: `model_version TEXT`, `iam_partition TEXT`, `norm_flags TEXT`, `vocab_hints_off INTEGER` (boolean 0/1)
- **Capture:** Auto-captured at runtime — model version (provider + exact model string, e.g. `gemini-2.0-flash-001`) and active normalization flags written automatically; user adds `--iam-partition <label>` when running IAM (free-text, no enum enforcement)
- **vocab_hints_off:** Boolean only — was the hints mechanism disabled? Not the full hints list
- **norm_flags:** Short string of active normalization flags (e.g. `lowercase,strip_punct`) from `normalize_text()` in metrics.py
- **Display — `benchmark list --runs`:** Provenance columns shown inline always (not behind a flag): model_version and iam_partition as table columns
- **Display — `benchmark report`:** Full provenance block as a header section on every report output

### [?] marker rate (FOUND-02)

- **Storage:** Both per-sample and aggregated — `question_marker_rate REAL` column added to `provider_outputs` table (v4 migration), plus aggregated in run reporting
- **Denominator:** markers per word — `rate = count('[?]') / word_count`. Interpretable as fraction of words that were uncertain
- **Computation:** From raw provider output text **before** normalization (normalization in metrics.py already strips `[?]` markers, so rate must be captured upstream)
- **Report display:** `marker_rate` column shown alongside `mean_cer` in the `benchmark report` results table

### Noise calibration (FOUND-03)

- **Command:** New `benchmark calibrate` subcommand — explicit and separate from `benchmark run`
- **Flags:** `--samples N` (default 20), `--provider` — standard provider selection
- **Samples source:** Random N samples from existing benchmark.db that have ground truth. No extra data needed; for IAM calibration, user ingests IAM first (Phase 7)
- **Storage:** Print-only — no DB write. Calibration is a diagnostic, not a benchmark run
- **Output format:** Compact single line: `CER variance: ±0.42%  |  Min detectable delta: 0.84% (2σ)` — actionable threshold, easy to scan

### Cost guardrail (FOUND-04)

- **Scope:** Wraps existing `benchmark run` command. Phase 7's `benchmark sweep` will inherit this behavior when built
- **Trigger:** Always shown before any run — no sample-count threshold, no dollar threshold
- **Bypass:** `--yes` flag skips confirmation prompt (standard convention, CI-friendly)
- **Output format:**
  ```
  Estimated cost: $0.042
    3 providers × 2 strategies × 20 samples

  Proceed? [y/N]
  ```
  Compact one-liner with multiplier breakdown, then confirm prompt

### Claude's Discretion

- Exact SQL column ordering in the v4 migration
- Whether `norm_flags` is computed dynamically at run time or read from a config
- Aggregation method for `question_marker_rate` at the run level (mean, median, or both)
- How `benchmark calibrate` handles the case where fewer than N samples with ground truth exist in the DB

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

- `estimate_cost(input_tokens, output_tokens, provider)` in `benchmark/evaluate.py` — cost math already done; cost projection can call this per (provider × strategy × samples) and sum
- `_count_uncertainty_markers(text)` in `consensus.py` — counts `[?]` markers; use directly for `question_marker_rate` numerator
- `_MIGRATIONS` dict in `benchmark/db.py` — established pattern for schema migrations; v4 is the next slot
- `normalize_text()` in `benchmark/metrics.py` — strips `[?]` at line 25; marker rate must be captured from raw text before this is called

### Established Patterns

- Schema migrations: `ALTER TABLE ... ADD COLUMN` keyed by version int in `_MIGRATIONS` dict; `CURRENT_SCHEMA_VERSION` bumped to match
- CLI: Click groups/commands via `@cli.group()` / `@benchmark.command()`; new `calibrate` subcommand follows same pattern
- Cost already tracked: `total_cost_usd` in `runs` table (v3), `estimated_cost_usd` in `StrategyResult` dataclass — provenance columns extend this same table

### Integration Points

- `benchmark/db.py`: v4 migration adds columns to `runs` and `provider_outputs`
- `benchmark/evaluate.py`: compute and store `question_marker_rate` per `ProviderOutput` before normalization is applied
- `benchmark/report.py`: add `marker_rate` column to results table; add provenance header block
- `benchmark/models.py`: `ProviderOutput` and `StrategyResult` dataclasses need new fields
- `cli.py`: add `--iam-partition`, `--vocab-hints-off` flags to `benchmark run`; add `benchmark calibrate` subcommand; add pre-flight cost projection + `--yes` bypass

</code_context>

<specifics>
## Specific Ideas

- The cost projection output should feel like `apt-get` confirmation — familiar, not alarming, easy to bypass with `--yes` in scripts
- `benchmark calibrate` output: "if your measured delta is less than X%, it's noise" — actionable framing, not just a raw number
- Provenance shown in report header so every result is self-documenting: model, partition, and normalization context visible without digging into DB

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-measurement-foundation*
*Context gathered: 2026-04-11*
