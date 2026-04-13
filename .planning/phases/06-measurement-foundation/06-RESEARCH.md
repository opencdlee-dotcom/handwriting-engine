# Phase 6: Measurement Foundation - Research

**Researched:** 2026-04-11
**Domain:** Python CLI extension, SQLite schema migration, statistics (stdlib), Click subcommands
**Confidence:** HIGH — all findings are from direct codebase inspection; no external library research required

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Provenance record (FOUND-01)**
- Schema: v4 migration adds four columns to the `runs` table: `model_version TEXT`, `iam_partition TEXT`, `norm_flags TEXT`, `vocab_hints_off INTEGER` (boolean 0/1)
- Capture: Auto-captured at runtime — model version (provider + exact model string, e.g. `gemini-2.0-flash-001`) and active normalization flags written automatically; user adds `--iam-partition <label>` when running IAM (free-text, no enum enforcement)
- `vocab_hints_off`: Boolean only — was the hints mechanism disabled? Not the full hints list
- `norm_flags`: Short string of active normalization flags (e.g. `lowercase,strip_punct`) from `normalize_text()` in metrics.py
- Display — `benchmark list --runs`: Provenance columns shown inline always: model_version and iam_partition as table columns
- Display — `benchmark report`: Full provenance block as a header section on every report output

**[?] marker rate (FOUND-02)**
- Storage: Both per-sample and aggregated — `question_marker_rate REAL` column added to `provider_outputs` table (v4 migration), plus aggregated in run reporting
- Denominator: markers per word — `rate = count('[?]') / word_count`. Interpretable as fraction of words that were uncertain
- Computation: From raw provider output text BEFORE normalization (normalization in metrics.py already strips `[?]` markers)
- Report display: `marker_rate` column shown alongside `mean_cer` in the `benchmark report` results table

**Noise calibration (FOUND-03)**
- Command: New `benchmark calibrate` subcommand — explicit and separate from `benchmark run`
- Flags: `--samples N` (default 20), `--provider` — standard provider selection
- Samples source: Random N samples from existing benchmark.db that have ground truth
- Storage: Print-only — no DB write
- Output format: Compact single line: `CER variance: ±0.42%  |  Min detectable delta: 0.84% (2σ)`

**Cost guardrail (FOUND-04)**
- Scope: Wraps existing `benchmark run` command
- Trigger: Always shown before any run — no threshold
- Bypass: `--yes` flag skips confirmation prompt
- Output format:
  ```
  Estimated cost: $0.042
    3 providers x 2 strategies x 20 samples

  Proceed? [y/N]
  ```

### Claude's Discretion
- Exact SQL column ordering in the v4 migration
- Whether `norm_flags` is computed dynamically at run time or read from a config
- Aggregation method for `question_marker_rate` at the run level (mean, median, or both)
- How `benchmark calibrate` handles the case where fewer than N samples with ground truth exist in the DB

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-01 | Developer can reproduce the 1.67% CER baseline with a documented provenance record (model version, IAM partition ID, normalization flags, vocabulary hints off) so all future comparisons have a valid anchor. | v4 migration pattern is established; `insert_run` and `finish_run` are the write points; `list_runs` and `generate_report` are the read points that need updating. |
| FOUND-02 | Benchmark runs store `[?]_marker_rate` as a separate column alongside CER, so strategies that resolve ambiguity are not conflated with those that improve character accuracy. | `_count_uncertainty_markers()` in consensus.py counts `[?]` markers; raw `output_text` is available in `_run_benchmark_inner` before `character_error_rate()` is called; `provider_outputs` table is where the column lives. |
| FOUND-03 | Developer can run a 20-sample noise floor calibration that measures CER variance at temperature 0.5 and reports the minimum detectable CER difference for this test set. | `statistics.stdev()` already used in `_aggregate_results`; `samples_with_ground_truth()` is the sample source; Click subcommand pattern is established in cli.py. |
| FOUND-04 | CLI warns with an API cost projection (strategies x providers x samples x passes) before executing any sweep run, preventing unintended cost explosions. | `estimate_cost()` in evaluate.py is the cost math function; token estimates can be approximated from historical run data or constants; `click.confirm()` is the confirmation primitive. |
</phase_requirements>

---

## Summary

Phase 6 is a pure extension phase — it adds four capabilities to an already working benchmark system. There are no new external dependencies, no architectural changes, and no ambiguity about which files to touch. Every piece of infrastructure this phase needs already exists: the migration dict pattern (`_MIGRATIONS`), the marker counting function (`_count_uncertainty_markers`), the cost math (`estimate_cost`), the Click CLI group, and the test harness with in-memory SQLite.

The highest implementation risk is the ordering constraint in FOUND-02: `question_marker_rate` must be computed from `output_text` BEFORE `character_error_rate()` is called, because `normalize_text()` in metrics.py strips `[?]` on line 25. In `_run_benchmark_inner`, the `result["text"]` value is the raw provider output — this is the correct capture point.

For FOUND-03, the statistics module is already imported in `report.py` (`import statistics`). The minimum detectable difference at 2-sigma is `2 * stdev`. For FOUND-04, `estimate_cost()` needs token-count projections per strategy-provider pair; since real token counts aren't known before a run, a per-sample average from the most recent run (or a hardcoded conservative estimate of ~2,000 input + 500 output tokens per read) is the correct approach.

**Primary recommendation:** Work top-to-bottom: v4 migration first (unblocks everything), then model data capture in evaluate.py, then marker rate capture, then report display, then the two new CLI subcommands.

---

## Standard Stack

### Core (all already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlite3 | stdlib | Schema migration, column storage | Already the DB engine; ALTER TABLE is idiomatic for incremental schema changes |
| statistics | stdlib | stdev, mean for calibration output | Already used in report.py; no numpy needed |
| click | >=8.1.0 | `benchmark calibrate` subcommand, `--yes` flag, `click.confirm()` | Already the CLI framework; all existing commands use it |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| random | stdlib | Random sample selection for `calibrate` | Needed in `benchmark calibrate` to pick N random samples from DB |
| math | stdlib | sqrt for MDD calculation if not using statistics | Only if 2*stdev formula needs intermediate steps |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `statistics.stdev` | numpy | numpy would be overkill; statistics.stdev is already present in report.py |
| `click.confirm()` | manual input() | click.confirm() handles --yes bypass cleanly via `default=False` |
| Per-sample token average from DB | hardcoded constant | DB average is more accurate; fallback to constant if no prior runs exist |

**Installation:** No new packages required.

---

## Architecture Patterns

### Recommended Execution Order

```
1. db.py       — Add v4 migration (runs columns + provider_outputs column)
2. models.py   — Add fields to ProviderOutput and StrategyResult dataclasses
3. evaluate.py — Capture model_version, norm_flags, question_marker_rate at run time
4. db.py       — Update insert_run() and insert_provider_output() signatures
5. report.py   — Add provenance header block + marker_rate column to table output
6. cli.py      — Add --iam-partition, --vocab-hints-off flags; add calibrate subcommand; add cost projection to benchmark run
```

### Pattern 1: v4 Schema Migration

**What:** SQLite `ALTER TABLE` statements in `_MIGRATIONS[4]`, `CURRENT_SCHEMA_VERSION` bumped to 4.

**When to use:** Any new column on an existing table that must survive across DB sessions.

**Example (from existing v3 pattern):**
```python
# In db.py
CURRENT_SCHEMA_VERSION = 4

_MIGRATIONS: dict[int, str] = {
    2: "...",
    3: "...",
    4: """
        ALTER TABLE runs ADD COLUMN model_version TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN iam_partition TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN norm_flags TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN vocab_hints_off INTEGER DEFAULT 0;
        ALTER TABLE provider_outputs ADD COLUMN question_marker_rate REAL DEFAULT NULL;
        UPDATE schema_version SET version = 4;
    """,
}
```

**Critical:** `executescript()` commits automatically between statements in SQLite. The existing `_apply_migrations` already wraps each migration in a try/except that tolerates "column already exists" errors — this is intentional and safe.

### Pattern 2: Marker Rate Capture (BEFORE normalization)

**What:** Compute `question_marker_rate` from raw `output_text` before calling `character_error_rate()`.

**When to use:** In `_run_benchmark_inner`, immediately after `result = _read_single(...)` returns.

**The constraint:**
- `normalize_text()` (metrics.py line 25) strips `[?]` via `re.sub(r"\[\?\]", "", text)`
- `character_error_rate()` calls `normalize_text()` internally
- Therefore, marker rate MUST be computed from `result["text"]` before `character_error_rate()` is called

**Example:**
```python
# In _run_benchmark_inner, after result = _read_single(...)
from handwriting_engine.consensus import _count_uncertainty_markers

raw_text = result["text"]
word_count = max(1, len(raw_text.split()))
marker_count = _count_uncertainty_markers(raw_text)
marker_rate = marker_count / word_count  # fraction of words that were uncertain

po_id = insert_provider_output(
    conn, ..., question_marker_rate=marker_rate, ...
)
```

**Note:** `_count_uncertainty_markers` is a private function in consensus.py. It is acceptable to import it directly (same package), or to copy the counting logic locally. The regex it uses is `_UNCERTAINTY_RE` which matches `[?]`, `???`, `[illegible...]`, `[unclear]`, and `unable to read`. For FOUND-02's denominator definition (`count('[?]') / word_count`), only literal `[?]` tokens should be counted — not the broader uncertainty regex — to match the user's stated formula. A simple `text.count("[?]")` achieves this.

### Pattern 3: Provenance Capture at Run Time

**What:** Auto-capture model_version and norm_flags when `insert_run` is called.

**The norm_flags derivation:** `normalize_text()` in metrics.py always applies: NFC, lowercase, marker stripping, whitespace collapse. These are always-on and not configurable. Therefore `norm_flags` should be a fixed string like `"nfc,lowercase,strip_markers,collapse_ws"` — computed as a constant or from inspecting the function's behavior, not from a config flag.

**The model_version derivation:** Requires knowing which provider is active. If multiple providers are used, store them comma-separated or as a JSON array. The model string (e.g. `gemini-2.0-flash-001`) should come from the provider's own attribute or from `_constants.py` (which pins model versions to prevent silent regression — confirmed in CLAUDE.md: "Model versions pinned: GPT-4.1-2025-04-14 prevents silent regression").

**Where to call:** In `cli.py`'s `benchmark_run_cmd`, before calling `run_benchmark()`, resolve the provider list and compute `model_version`. Pass these to `run_benchmark()` as new parameters that flow through to `insert_run()`.

### Pattern 4: benchmark calibrate Subcommand

**What:** New `@benchmark.command("calibrate")` in cli.py.

**Logic:**
```python
@benchmark.command("calibrate")
@click.option("--samples", "-n", default=20, type=int)
@click.option("--provider", "-p", default="gemini")
def benchmark_calibrate_cmd(samples, provider):
    """Measure CER variance and minimum detectable delta on N random samples."""
    import random
    import statistics
    from handwriting_engine.benchmark.db import get_connection, samples_with_ground_truth
    
    conn = get_connection()
    all_samples = samples_with_ground_truth(conn)
    conn.close()
    
    if len(all_samples) == 0:
        click.echo("No samples with ground truth in DB.", err=True)
        sys.exit(1)
    
    n = min(samples, len(all_samples))
    if n < samples:
        click.echo(f"Warning: only {n} samples available (requested {samples})")
    
    selected = random.sample(all_samples, n)
    # ... run provider on each selected sample, collect CER values ...
    # ... compute stdev and MDD ...
    sd = statistics.stdev(cers)
    mdd = 2 * sd
    click.echo(f"CER variance: ±{sd*100:.2f}%  |  Min detectable delta: {mdd*100:.2f}% (2σ)")
```

**Fewer-than-N samples handling (Claude's Discretion):** The recommended approach is: cap silently if the shortfall is small (< 5 samples), warn with a one-line message and proceed if larger. Never abort — a calibration with fewer samples is still useful.

### Pattern 5: Cost Projection Guard

**What:** Pre-flight block at the top of `benchmark_run_cmd`, before `run_benchmark()` is called.

**Token estimation:** `estimate_cost()` takes input and output token counts. Before a run, these are unknown. Use the average from the most recent run (query `AVG(input_tokens)`, `AVG(output_tokens)` from `provider_outputs` where `run_id = latest_run_id`). If no prior runs exist, use conservative defaults: 2000 input + 500 output tokens per sample per provider.

**Pass count:** For `benchmark run`, passes = 1 (single strategy per provider). For future `benchmark sweep`, this will be strategies × passes.

**Example:**
```python
# In benchmark_run_cmd, before run_benchmark() call
n_samples = len(samples_with_ground_truth(conn))
n_providers = len(prov_list or available_providers())
n_strategies = max(1, len(strat_list or []))
total_reads = n_providers * n_strategies * n_samples
avg_in, avg_out = _estimate_tokens_per_read(conn)
total_cost = estimate_cost(avg_in * total_reads, avg_out * total_reads, provider)

click.echo(f"Estimated cost: ${total_cost:.3f}")
click.echo(f"  {n_providers} providers x {n_strategies} strategies x {n_samples} samples")
click.echo("")
if not yes_flag:
    if not click.confirm("Proceed?", default=False):
        sys.exit(0)
```

### Anti-Patterns to Avoid

- **Computing marker_rate after normalize_text:** normalize_text strips `[?]` — the rate would always be 0. Must use raw text.
- **Storing norm_flags as a dynamic dict or JSON object:** Keep it as a short human-readable string. It's for display in reports, not for programmatic reconstruction.
- **Aborting calibrate when n < samples:** Warn and proceed. A short calibration is still informative.
- **Showing cost projection only for large runs:** The decision is always-show. The `--yes` flag is the escape valve for CI/automation.
- **Calling executescript() with mixed DDL and DML in separate transactions:** The existing migration pattern uses `executescript()` which auto-commits. Keep the UPDATE schema_version inside the same script string as the ALTER TABLE statements.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Standard deviation | Custom loop | `statistics.stdev()` | Already imported in report.py; handles n < 2 edge case |
| Minimum detectable difference | Custom formula | `2 * statistics.stdev(cers)` | This is the exact 2-sigma formula; no library needed |
| CLI confirmation prompt | `input()` with manual parsing | `click.confirm("Proceed?", default=False)` | Handles --yes bypass, KeyboardInterrupt, TTY detection |
| Marker counting | New regex | `text.count("[?]")` for the rate denominator | Simple and matches the exact user-facing formula |
| Schema migration safety | Rollback logic | The existing `_apply_migrations` try/except | Already handles "column already exists" idempotently |
| Cost-per-token rates | New dict | `COST_PER_1M_TOKENS` from `_constants.py` | Already maintained; `estimate_cost()` already reads it |

**Key insight:** This phase is almost entirely plumbing — connecting existing pieces. The only genuinely new computation is `2 * stdev` for the MDD.

---

## Common Pitfalls

### Pitfall 1: Marker Rate Computed After Normalization
**What goes wrong:** `character_error_rate()` normalizes internally. If marker_rate is derived from the normalized text (e.g., by calling normalize_text() first), `[?]` markers are already stripped and the rate is always 0.
**Why it happens:** It's natural to normalize before any measurement. But here, the marker IS the measurement.
**How to avoid:** In `_run_benchmark_inner`, compute `text.count("[?]")` from `result["text"]` on the line BEFORE calling `character_error_rate(result["text"], gt.text)`.
**Warning signs:** All `question_marker_rate` values are 0.0 in the DB.

### Pitfall 2: insert_run Signature Change Breaks Existing Callers
**What goes wrong:** Adding `model_version`, `iam_partition`, `norm_flags`, `vocab_hints_off` parameters to `insert_run()` without default values breaks `compare_strategies()` in evaluate.py, which calls `insert_run()` directly.
**Why it happens:** `insert_run` is called in multiple places (`_run_benchmark_inner` and `compare_strategies`).
**How to avoid:** Give all four new parameters `None` defaults so existing callers don't need changes.
**Warning signs:** `TypeError: insert_run() missing required argument` at import time or in tests.

### Pitfall 3: CURRENT_SCHEMA_VERSION Not Bumped
**What goes wrong:** `_apply_migrations` only applies migrations for versions > current. If `CURRENT_SCHEMA_VERSION` stays at 3, new DBs will seed version 3 and migration 4 will never run on them.
**Why it happens:** It's easy to add the migration dict entry but forget the constant.
**How to avoid:** Change `CURRENT_SCHEMA_VERSION = 3` to `CURRENT_SCHEMA_VERSION = 4` in the same commit as adding `_MIGRATIONS[4]`.
**Warning signs:** `OperationalError: table runs has no column named model_version` on a fresh DB.

### Pitfall 4: Cost Projection Blocks When No Prior Run Exists
**What goes wrong:** If `_estimate_tokens_per_read()` queries the latest run and there is no run in the DB, it returns None or crashes, and the cost projection aborts before the actual benchmark.
**Why it happens:** First-time use of the benchmark system.
**How to avoid:** Fall back to conservative defaults (2000 input + 500 output tokens) when no prior runs exist. The displayed cost will be an overestimate but that's safe.
**Warning signs:** `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` in the cost projection block.

### Pitfall 5: benchmark calibrate Makes Real API Calls
**What goes wrong:** During testing, `benchmark calibrate` actually calls provider APIs, costing money and requiring valid API keys.
**Why it happens:** The calibrate command runs `_read_single()` which calls real providers.
**How to avoid:** Test the command with mocked providers (same pattern as `test_benchmark_evaluate.py` which patches `_read_single` and `_available_providers`). The command's output formatting can be tested separately with hardcoded CER lists.
**Warning signs:** Tests fail with `GOOGLE_API_KEY not set` or incur unexpected API costs.

### Pitfall 6: word_count Denominator is Zero
**What goes wrong:** `marker_rate = marker_count / word_count` crashes with ZeroDivisionError if the provider returns an empty string (error case, or truly blank image).
**Why it happens:** Error outputs and blank responses produce `output_text = ""`.
**How to avoid:** Use `max(1, len(raw_text.split()))` as the denominator, exactly as `_single_text_confidence()` in consensus.py does.
**Warning signs:** ZeroDivisionError in test runs with mock providers that return empty strings.

---

## Code Examples

Verified patterns from codebase inspection:

### v4 Migration (following v3 pattern exactly)
```python
# Source: handwriting_engine/benchmark/db.py — _MIGRATIONS pattern
_MIGRATIONS: dict[int, str] = {
    2: "...",
    3: "ALTER TABLE runs ADD COLUMN total_cost_usd REAL DEFAULT 0.0;\nUPDATE schema_version SET version = 3;",
    4: """
        ALTER TABLE runs ADD COLUMN model_version TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN iam_partition TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN norm_flags TEXT DEFAULT NULL;
        ALTER TABLE runs ADD COLUMN vocab_hints_off INTEGER DEFAULT 0;
        ALTER TABLE provider_outputs ADD COLUMN question_marker_rate REAL DEFAULT NULL;
        UPDATE schema_version SET version = 4;
    """,
}
CURRENT_SCHEMA_VERSION = 4
```

### Marker Rate Computation (using raw text before CER)
```python
# Source: handwriting_engine/benchmark/evaluate.py — _run_benchmark_inner pattern
# After: result = _read_single(sample.image_path, provider, domain, ...)
raw_text = result["text"]
word_count = max(1, len(raw_text.split()))
marker_count = raw_text.count("[?]")
marker_rate = marker_count / word_count

po_id = insert_provider_output(
    conn, run_id=run_id, sample_id=sample.id, provider=provider,
    strategy="single", output_text=raw_text,
    question_marker_rate=marker_rate,
    ...
)
# Then CER (which normalizes internally, stripping [?]):
if not result["error"]:
    cer, char_edits, ref_chars = character_error_rate(raw_text, gt.text)
```

### Calibration Statistics (stdlib only)
```python
# Source: handwriting_engine/benchmark/report.py — statistics already imported
import statistics
cers = [0.012, 0.018, 0.021, 0.009, ...]  # from N calibration reads
if len(cers) >= 2:
    sd = statistics.stdev(cers)
    mdd = 2 * sd
    click.echo(f"CER variance: ±{sd*100:.2f}%  |  Min detectable delta: {mdd*100:.2f}% (2σ)")
else:
    click.echo("Not enough samples to compute variance (need at least 2).")
```

### Cost Projection with click.confirm
```python
# Source: cli.py benchmark_run_cmd pattern + click docs
# --yes flag:
@click.option("--yes", "-y", is_flag=True, help="Skip cost confirmation (CI-friendly)")
def benchmark_run_cmd(..., yes):
    # Pre-flight cost estimate
    conn = get_connection()
    n_samples = len(samples_with_ground_truth(conn))
    conn.close()
    n_prov = len(prov_list) if prov_list else len(_available_providers())
    n_strat = max(1, len(strat_list) if strat_list else 1)
    
    # Token estimate: use historical average or fallback
    avg_in, avg_out = _get_avg_tokens_per_read() or (2000, 500)
    total_reads = n_prov * n_strat * n_samples
    cost = estimate_cost(int(avg_in * total_reads), int(avg_out * total_reads), 
                        (prov_list[0] if prov_list else "gemini"))
    
    click.echo(f"Estimated cost: ${cost:.3f}")
    click.echo(f"  {n_prov} providers x {n_strat} strategies x {n_samples} samples")
    click.echo("")
    if not yes and not click.confirm("Proceed?", default=False):
        sys.exit(0)
    
    # ... rest of benchmark run ...
```

### Provenance Header in Report
```python
# Source: report.py — _format_table pattern, extended with provenance block
def _format_table(run_id: int, results: list[StrategyResult], run_meta: dict | None = None) -> str:
    lines = [f"Benchmark Run #{run_id}", ""]
    if run_meta:
        lines.append("Provenance:")
        lines.append(f"  Model:      {run_meta.get('model_version', 'unknown')}")
        lines.append(f"  Partition:  {run_meta.get('iam_partition', 'n/a')}")
        lines.append(f"  Norm flags: {run_meta.get('norm_flags', 'unknown')}")
        lines.append(f"  Vocab hints off: {'yes' if run_meta.get('vocab_hints_off') else 'no'}")
        lines.append("")
    # ... existing header and rows ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `detect_regressions()` compares to penultimate run | Will compare to pinned `is_baseline` run (Phase 9) | Phase 9 | Not a Phase 6 concern; don't add `is_baseline` now |
| No provenance on runs | v4 migration adds 4 provenance columns | Phase 6 | Enables reproducibility claims |
| CER conflated with marker resolution | Separate `question_marker_rate` column | Phase 6 | Disambiguates strategy improvements |

**Deprecated/outdated:**
- `detect_regressions()` with penultimate-run comparison: still the current approach; Phase 9 replaces it. Do not change this in Phase 6.

---

## Open Questions

1. **norm_flags: dynamic vs. fixed string**
   - What we know: `normalize_text()` in metrics.py always applies the same transformations (NFC, lowercase, strip markers, collapse whitespace) with no configuration switches
   - What's unclear: Whether future phases might make normalization configurable (e.g., a no-lowercase mode)
   - Recommendation (Claude's Discretion): Use a fixed constant string `"nfc,lowercase,strip_markers,collapse_ws"` for now. If normalization becomes configurable, this can be a computed value. The column name is `norm_flags` which implies it's a flags bitmask, but a human-readable string is clearer for reports.

2. **marker_rate aggregation for run-level reporting**
   - What we know: Per-sample `question_marker_rate` is stored in `provider_outputs`; `StrategyResult` aggregates CER with mean/median/stdev
   - What's unclear: Whether to use mean, median, or both for the run-level marker_rate display
   - Recommendation (Claude's Discretion): Use mean for the run-level summary (consistent with `mean_cer`). Add `mean_marker_rate` to `StrategyResult` dataclass. The per-sample rates are already stored for drill-down.

3. **Cost estimate accuracy for multi-provider consensus**
   - What we know: `estimate_cost()` averages costs across providers in a `+`-joined provider string; consensus reads call multiple providers
   - What's unclear: Whether the cost projection for consensus strategies should show the per-provider cost sum or the averaged cost
   - Recommendation: Show total cost (sum across all providers per sample), not the averaged cost, because the user will actually pay for all providers. This means calling `estimate_cost()` once per provider in the strategy, then summing.

---

## Validation Architecture

> `workflow.nyquist_validation` key is absent from `.planning/config.json` — treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0.0 |
| Config file | none — discovered via `tests/` directory |
| Quick run command | `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | v4 migration adds 4 columns to `runs` table | unit | `pytest tests/test_benchmark_db.py::TestSchemaCreation -x -q` | Partial (file exists, new test class needed) |
| FOUND-01 | Provenance columns written on `insert_run()` call | unit | `pytest tests/test_benchmark_db.py -k "provenance" -x -q` | No — Wave 0 |
| FOUND-01 | `benchmark list --runs` shows model_version and iam_partition columns | unit | `pytest tests/test_benchmark_db.py -k "list_runs_provenance" -x -q` | No — Wave 0 |
| FOUND-01 | `benchmark report` output contains provenance header block | unit | `pytest tests/test_benchmark_evaluate.py -k "report_provenance" -x -q` | No — Wave 0 |
| FOUND-02 | v4 migration adds `question_marker_rate` to `provider_outputs` | unit | `pytest tests/test_benchmark_db.py::TestSchemaCreation -x -q` | Partial |
| FOUND-02 | Marker rate stored correctly for output containing `[?]` | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate" -x -q` | No — Wave 0 |
| FOUND-02 | Marker rate is 0 for clean output (no `[?]`) | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate_clean" -x -q` | No — Wave 0 |
| FOUND-02 | Marker rate computed from RAW text (not normalized) | unit | `pytest tests/test_benchmark_evaluate.py -k "marker_rate_before_norm" -x -q` | No — Wave 0 |
| FOUND-02 | `benchmark report` table includes marker_rate column | unit | `pytest tests/test_benchmark_evaluate.py -k "report_marker_rate" -x -q` | No — Wave 0 |
| FOUND-03 | `benchmark calibrate` command exists and runs | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate" -x -q` | No — Wave 0 |
| FOUND-03 | Calibrate output format matches spec (`CER variance: ±X%  \|  Min detectable delta: Y% (2σ)`) | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate_format" -x -q` | No — Wave 0 |
| FOUND-03 | Calibrate gracefully handles fewer samples than requested | unit | `pytest tests/test_benchmark_evaluate.py -k "calibrate_undersample" -x -q` | No — Wave 0 |
| FOUND-04 | Cost projection printed before benchmark run | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_projection" -x -q` | No — Wave 0 |
| FOUND-04 | `--yes` flag bypasses confirmation prompt | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_yes_bypass" -x -q` | No — Wave 0 |
| FOUND-04 | Declining prompt exits without running benchmark | unit | `pytest tests/test_benchmark_evaluate.py -k "cost_decline" -x -q` | No — Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_benchmark_db.py tests/test_benchmark_evaluate.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_benchmark_db.py` — extend `TestSchemaCreation` with `test_v4_migration_columns` covering both `runs` and `provider_outputs` new columns
- [ ] `tests/test_benchmark_evaluate.py` — add `TestMarkerRate` class covering: rate from raw text, rate=0 for clean output, rate stored in DB, rate in report output
- [ ] `tests/test_benchmark_evaluate.py` — add `TestCalibrateCommand` class (use Click test runner via `CliRunner`) covering: format, undersample warning, no-samples error
- [ ] `tests/test_benchmark_evaluate.py` — add `TestCostProjection` class covering: always-shown, `--yes` bypass, decline exits cleanly
- [ ] `tests/test_benchmark_evaluate.py` — add `TestProvenanceCapture` class covering: provenance columns in DB after run, report header contains provenance block

*(All use `CliRunner` from Click for CLI tests and `:memory:` SQLite for DB tests — both patterns already established in the codebase.)*

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `handwriting_engine/benchmark/db.py` — schema, migration pattern, `CURRENT_SCHEMA_VERSION`, `insert_run`, `insert_provider_output`, `_apply_migrations`
- `handwriting_engine/benchmark/evaluate.py` — `_run_benchmark_inner`, `estimate_cost`, `_read_single` return shape
- `handwriting_engine/benchmark/metrics.py` — `normalize_text()` strips `[?]` on line 25, `character_error_rate()` calls normalize_text internally
- `handwriting_engine/benchmark/report.py` — `_format_table`, `_aggregate_results`, `StrategyResult`, statistics import
- `handwriting_engine/benchmark/models.py` — `ProviderOutput`, `StrategyResult`, `RunSummary` dataclass shapes
- `handwriting_engine/cli.py` — Click command group pattern, `benchmark_run_cmd` signature, existing flags
- `handwriting_engine/consensus.py` — `_count_uncertainty_markers`, `_UNCERTAINTY_RE` pattern
- `tests/test_benchmark_db.py` — in-memory fixture pattern, `TestSchemaCreation`
- `tests/test_benchmark_evaluate.py` — mock pattern for `_read_single` and `_available_providers`
- `tests/conftest.py` — `MockProvider` class, `tmp_image` fixture

### Secondary (MEDIUM confidence)
- `pyproject.toml` — confirms pytest >= 8.0.0 is the test framework, no pytest.ini config file exists
- `CLAUDE.md` — confirms model versions are pinned in `_constants.py`, DB is SQLite at `~/.handwriting-engine/benchmark.db`

### Tertiary (LOW confidence)
- None required — all findings from direct source inspection.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all tools already present
- Architecture: HIGH — migration pattern, CLI pattern, and test pattern all verified from existing code
- Pitfalls: HIGH — all pitfalls identified from reading the actual code paths (normalize_text stripping on line 25, insert_run callers, etc.)

**Research date:** 2026-04-11
**Valid until:** 2026-06-11 (stable codebase; no fast-moving external dependencies)
