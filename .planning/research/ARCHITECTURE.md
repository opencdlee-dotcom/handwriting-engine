# Architecture Patterns: HTR Benchmarking Milestone (v3.0)

**Domain:** Benchmark execution and reporting extension for an existing HTR engine
**Researched:** 2026-04-11
**Confidence:** HIGH — based on direct codebase inspection of all benchmark/ modules

---

## Existing Architecture (What We're Extending)

The benchmark/ subpackage is a self-contained SQLite-backed evaluation system
with a clear internal layering:

```
benchmark/
  models.py      — pure dataclasses, no I/O
  db.py          — SQLite DDL + CRUD, schema_version migrations
  metrics.py     — pure functions (CER, WER, domain_term_accuracy, classify_errors)
  ingest.py      — image import, degrade, bootstrap-gt
  evaluate.py    — run_benchmark(), compare_strategies(), _read_single/_read_consensus
  report.py      — generate_report(), compare_runs(), detect_regressions(), drill-down
  __init__.py    — flat public API surface
cli.py (benchmark group)  — thin Click wrappers only, no business logic
```

Schema (v3, current): `samples`, `ground_truths`, `quality_assessments`, `runs`,
`provider_outputs`, `eval_metrics`, `schema_version`.

`runs.strategies` is a JSON TEXT column listing strategies used per run.
`provider_outputs.provider` holds compound labels like `"gemini+claude"` for consensus runs.
`provider_outputs.strategy` holds the strategy name (`"single"`, `"vote"`, `"self_correct"`, etc.).

---

## Design Decisions — The Five Questions

### 1. IAM Dataset Loading — Same DB, New Category Tag

**Decision: Ingest IAM images into the existing `samples` table using `category="iam"`.**

Do NOT create a separate database or parallel schema.

Rationale:
- The existing `ingest_directory()` already handles deduplication by SHA-256, page-number
  extraction, quality assessment, and `source_dir` metadata — all needed for IAM.
- IAM images paired with their ground-truth `.txt` files are structurally identical to any
  other `(image, ground_truth_text)` pair the system already handles.
- The `category` and `student` columns on `samples` serve as natural dataset-origin tags.
  `category="iam"` + `student="iam-writer-XXX"` gives full slice-and-dice capability via
  existing `list_samples()` and all report queries.
- Keeping one DB means every existing `compare_runs()`, `detect_regressions()`, and
  `sample_drill_down()` command works on IAM results without modification.

The only new piece is a CLI command `benchmark ingest-iam` (or a flag on `ingest`) that:
1. Walks the IAM directory structure (`lines/` or `words/` subdirectory).
2. Reads the paired `.txt` ground truth for each image (IAM distributes GT as `ascii/` files).
3. Calls existing `ingest_single()` then `insert_ground_truth()` with `source="iam"`.

This is a new ~80-line function in `ingest.py` + one new CLI command. Zero schema changes.

**Schema migration needed: None.**

**New function:** `ingest_iam_dataset(iam_root: Path, split: str = "test", db_path=None) -> tuple[int, int]`
- Returns `(samples_ingested, gt_records_created)`.
- `split` selects `lines/` vs `words/` subdirectory (IAM standard splits).
- Lives in `ingest.py` alongside `ingest_directory()`.

---

### 2. Strategy Sweep — Extend `run_benchmark()`, Add `--sweep` CLI Flag

**Decision: Implement the strategy sweep by extending `run_benchmark()` with a `sweep_strategies` parameter, not by writing a new function.**

The existing `compare_strategies()` in `evaluate.py` already does individual per-strategy runs
and prints a comparison table. Its gap is that it runs each strategy as an independent run
(one row per strategy in `runs` table), which means each strategy gets its own `run_id`. This
is intentional and correct — it allows `compare_runs(run_a, run_b)` to compare any two
strategies, not just last vs second-last.

The sweep should:
1. Accept a list of strategies: `["single", "vote", "best_of", "debate", "self_correct", "smart"]`.
2. Accept a list of providers (default: all available).
3. For each `(provider_set, strategy)` combination, call `run_benchmark()` once.
4. Return a list of `run_id` values for downstream analysis.

**New function:** `sweep_strategies(strategies, providers, domain, db_path, label_prefix) -> list[int]`
- Lives in `evaluate.py`.
- Returns list of run IDs — caller passes them to `compare_runs()` or new significance tests.

**New CLI flag:** `benchmark run --sweep vote,best_of,self_correct,smart`
- Replaces existing `--compare-strategies` flag (which is identical in spirit but lacks
  the `--providers` override and returns a formatted table rather than run IDs).
- The `--compare-strategies` flag stays for backward compat; `--sweep` adds run-ID return.

Data layout after a sweep: N rows in `runs` (one per strategy), all sharing the same sample
set. Each run's results can be queried via `get_run_results(conn, run_id)`. The existing
`_aggregate_results()` already handles grouped `(provider, strategy)` aggregation per run.

---

### 3. Statistical Testing — New `benchmark/stats.py` Module + `report` Subcommand Extension

**Decision: Statistical testing lives in a new `benchmark/stats.py` module, exposed via an
extension of the existing `benchmark report` command (new `--significance` flag).**

Do NOT add a separate top-level CLI command. The `report` subcommand already has the right
semantic scope (post-run analysis of stored results), and adding a flag keeps the surface
minimal.

**New module: `benchmark/stats.py`**

```python
# Public API surface
def paired_significance_test(
    run_id_a: int,
    run_id_b: int,
    db_path: Path | None = None,
    alpha: float = 0.05,
) -> SignificanceResult: ...

def bootstrap_confidence_interval(
    run_id: int,
    n_bootstrap: int = 10_000,
    db_path: Path | None = None,
    alpha: float = 0.05,
) -> BootstrapResult: ...
```

**`SignificanceResult` dataclass** (add to `models.py`):
```python
@dataclass
class SignificanceResult:
    run_id_a: int
    run_id_b: int
    test: str          # "wilcoxon" | "ttest_rel" | "mcnemar"
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float  # Cohen's d or rank-biserial r
    note: str           # plain-English interpretation
```

**Test selection logic** (all in `stats.py`, no new dependencies beyond `scipy`):
- Paired Wilcoxon signed-rank test — nonparametric, appropriate for CER distributions
  which are right-skewed and non-normal (typical for HTR). Use when `n >= 10`.
- Paired t-test — fallback when `n < 10` (small IAM subsets).
- McNemar's test — for per-sample binary "correct/incorrect" comparisons.
- Effect size: rank-biserial correlation `r = 1 - (2W)/(n*(n+1))` for Wilcoxon.

`scipy.stats` is available via `scipy` which is already an implicit dependency (numpy is
present via Pillow/OpenCV). Make `scipy` an explicit optional import with a clear error
message if missing (consistent with the project's lazy-import pattern).

**CLI integration:** Add `--significance RUN_A RUN_B` option to `benchmark report`:
```
handwriting-engine benchmark report --significance 12 15
```
This prints the existing report plus a significance section at the bottom.

Alternatively (and more discoverable), extend `benchmark compare RUN_A RUN_B` to print
significance automatically when sample_count >= 10.

**Recommendation:** Add to `benchmark compare` automatically (zero new flags) — the user
who runs `compare` already has intent to evaluate differences. Gate on `n >= 10` silently.

---

### 4. Best-Config Recommendation — Automated, in `report.py`, Driven by Stored Results

**Decision: Automated recommendation, computed from stored `eval_metrics` + `runs` data,
lives in `report.py` as `recommend_best_config()`.**

Manual recommendation (human looks at table and decides) is not acceptable as a milestone
deliverable — it produces no durable artifact and can't be referenced by future regression
detection.

**New function:** `recommend_best_config(db_path, domain, top_n=3) -> BestConfigResult`

Logic:
1. Query all `(provider, strategy)` combinations that have been run on IAM samples
   (`category="iam"`).
2. Filter to runs with `sample_count >= 20` (avoids recommending from sparse data).
3. Score each config on a weighted composite:
   - `0.70 * normalized_mean_cer` (primary accuracy driver)
   - `0.15 * normalized_estimated_cost_usd` (cost matters for production use)
   - `0.15 * normalized_stdev_cer` (stability — lower stdev = more reliable)
4. Return top-N configs with their CER, cost, stability, and a plain-English recommendation
   string.

**`BestConfigResult` dataclass** (add to `models.py`):
```python
@dataclass
class BestConfigResult:
    top_configs: list[StrategyResult]
    recommendation: str   # "Use gemini/self_correct: 1.31% CER, $0.0023/image, stable"
    generated_at: str
    sample_count: int
    domain: str
```

**CLI integration:**
```
handwriting-engine benchmark report --recommend
```
or automatically appended at the bottom of `benchmark report` when multiple strategy runs
exist for the same sample set.

**Persistence:** The recommendation itself is ephemeral (computed on-demand from stored
metrics), but the run results that drive it are permanently stored in the DB. This means
re-running `--recommend` after adding more data gives a refreshed recommendation without
any separate "recommendation" table to maintain.

---

### 5. Regression Baseline — Tag a Run as Baseline; Wire into Existing `detect_regressions()`

**Decision: Add a `baseline_run_id` field to the `runs` table (schema v4 migration), and
update `detect_regressions()` to compare against the tagged baseline run rather than always
using runs[-2].**

The current `detect_regressions()` implementation compares "latest run vs second-latest run"
(lines 231-249 in `report.py`). This breaks down when:
- Multiple exploratory runs are made between baselines.
- A sweep creates many runs at once (IAM strategy sweep produces 6+ runs; runs[-2] points
  to the wrong strategy).
- A future developer runs a test and accidentally sets a regression baseline they didn't intend.

**Schema migration (v4):**
```sql
ALTER TABLE runs ADD COLUMN is_baseline INTEGER DEFAULT 0;
UPDATE schema_version SET version = 4;
```

**New CLI command:** `benchmark set-baseline RUN_ID`
```python
@benchmark.command("set-baseline")
@click.argument("run_id", type=int)
def benchmark_set_baseline_cmd(run_id):
    """Tag a run as the official regression baseline."""
```

This sets `runs.is_baseline = 1` for the given run (and optionally clears other baselines).

**Updated `detect_regressions()` logic:**
1. If any run has `is_baseline = 1`, compare current run against the most recent baseline.
2. Fall back to runs[-2] behavior when no baseline is tagged (preserves backward compat).

**Baseline commit workflow for v3.0:**
1. Run IAM strategy sweep.
2. Identify best config via `recommend_best_config()`.
3. Run `benchmark set-baseline <run_id_of_best_config_run>`.
4. The run_id and DB path are checked into version control as documented outputs
   (not the DB itself — the DB lives in `~/.handwriting-engine/benchmark.db`).
5. Future `benchmark run` calls trigger `detect_regressions()` against this baseline.

---

## Component Boundaries After v3.0

| Component | Responsibility | New/Modified | Communicates With |
|-----------|---------------|--------------|-------------------|
| `benchmark/ingest.py` | Image + GT import, degrade, bootstrap-gt, **IAM import** | Modified (add `ingest_iam_dataset`) | `db.py` |
| `benchmark/db.py` | SQLite CRUD + schema migrations | Modified (v4 migration: `is_baseline`) | — |
| `benchmark/models.py` | Pure dataclasses | Modified (add `SignificanceResult`, `BestConfigResult`) | — |
| `benchmark/metrics.py` | CER/WER/domain/classify | Unchanged | — |
| `benchmark/evaluate.py` | Run execution + cost estimation | Modified (add `sweep_strategies`) | `db.py`, `metrics.py`, `vision.py` |
| `benchmark/report.py` | Aggregate reporting + regression | Modified (add `recommend_best_config`, update `detect_regressions`) | `db.py`, `evaluate.py` |
| `benchmark/stats.py` | Statistical significance testing | **New** | `db.py`, `models.py` |
| `cli.py` | Click commands | Modified (add `ingest-iam`, `set-baseline`, extend `report`/`compare`) | All benchmark modules |

---

## Data Flow: IAM Benchmark Sweep

```
IAM dataset on disk
  └─ ingest_iam_dataset("iam_root/", split="test")
       ├─ ingest_single(image) → samples.id
       └─ insert_ground_truth(sample_id, iam_gt_text, source="iam")
             ↓
         benchmark.db: samples(category="iam"), ground_truths(source="iam")

sweep_strategies(["single","vote","best_of","self_correct","smart"], providers=["gemini"])
  └─ for each strategy:
       run_benchmark(label="iam-sweep-{strategy}", sample_ids=iam_sample_ids)
         ├─ evaluate each (provider, strategy) pair
         ├─ compute CER/WER per sample
         └─ INSERT INTO provider_outputs, eval_metrics
             ↓
         benchmark.db: runs[N..N+5], provider_outputs[M..], eval_metrics[K..]

recommend_best_config(domain="biology")
  └─ SELECT aggregated CER/cost/stdev per (provider, strategy) FROM iam runs
  └─ return BestConfigResult

set-baseline <best_run_id>
  └─ UPDATE runs SET is_baseline = 1 WHERE id = best_run_id
             ↓
         Future detect_regressions() compares against is_baseline=1 run
```

---

## Build Order (Dependency-Respecting)

All new components depend on the existing DB layer, so no circular dependencies are
introduced. Build order:

1. **Schema migration v4** (`db.py`) — `is_baseline` column on `runs`.
   - Blocks: `set-baseline` CLI command, updated `detect_regressions()`.
   - Test: existing `test_benchmark_db.py` + new migration test.

2. **`ingest_iam_dataset()`** (`ingest.py`) — IAM-specific walk + GT pairing.
   - Blocks: all v3.0 benchmark runs.
   - Dependency: existing `ingest_single()` + `insert_ground_truth()` unchanged.
   - Test: new `test_benchmark_iam_ingest.py` with a minimal fixture (3 IAM samples).

3. **`sweep_strategies()`** (`evaluate.py`) — loops `run_benchmark()` per strategy.
   - Blocks: `recommend_best_config()`, statistical testing.
   - Dependency: existing `run_benchmark()` unchanged.

4. **`SignificanceResult` + `BestConfigResult`** (`models.py`) — pure dataclasses.
   - Blocks: `stats.py`, `report.py` additions.
   - No dependencies.

5. **`benchmark/stats.py`** — Wilcoxon/t-test/McNemar + bootstrap CI.
   - Blocks: `benchmark compare` auto-significance output.
   - Dependency: `SignificanceResult` from step 4, `get_run_results()` from `db.py`.

6. **`recommend_best_config()`** (`report.py`) — scoring query over stored metrics.
   - Dependency: completed sweep results in DB (step 3), `BestConfigResult` (step 4).

7. **Updated `detect_regressions()`** (`report.py`) — prefer `is_baseline=1` run.
   - Dependency: schema v4 (step 1).

8. **CLI extensions** (`cli.py`) — thin wrappers over steps 2, 3, 5, 6, 7.

---

## Patterns to Follow

### Pattern: Lazy Optional Imports for scipy

Consistent with existing lazy-import pattern used throughout (paddleocr, transformers,
skimage). In `stats.py`:

```python
def _require_scipy():
    try:
        import scipy.stats
        return scipy.stats
    except ImportError:
        raise ImportError(
            "scipy is required for significance testing. "
            "Install with: pip install scipy"
        )
```

This means `scipy` is not added to core install requirements — it's gated behind
`pip install handwriting-engine[benchmark-stats]` or similar extras.

### Pattern: category= Filtering for Dataset Slices

All new reporting queries should filter by `category` to enable IAM-only vs full-corpus
vs real-notebook-only views. The existing query pattern in `report.py` already works
across all samples; add an optional `category` filter:

```python
def generate_report(run_id=None, db_path=None, fmt="table", category=None):
    # category="iam" restricts to IAM samples only
```

This is the cleanest way to separate IAM accuracy numbers from real-notebook numbers
without schema proliferation.

### Pattern: Single-Run-Per-Strategy is Correct

Do NOT collapse the sweep into one run with strategy as a column on `provider_outputs`.
Multiple strategy runs look redundant but enable: (a) accurate per-run cost tracking,
(b) `compare_runs(a, b)` delta view, (c) independent `is_baseline` tagging per strategy,
(d) future re-runs of just one strategy without re-running all.

---

## Anti-Patterns to Avoid

### Anti-Pattern: Separate IAM Database

Keeping a separate `iam_benchmark.db` breaks `compare_runs()`, `detect_regressions()`,
and `quality_correlation()` — all of which assume a single database.

### Anti-Pattern: Significance Test as a Separate Top-Level Command

A top-level `benchmark significance RUN_A RUN_B` command creates an orphaned command
that most users won't discover. Statistical output belongs alongside the existing
`compare` output — either auto-appended or via `--significance` flag.

### Anti-Pattern: Writing Best-Config as a Static File

Generating a `BEST_CONFIG.json` artifact instead of computing on-demand from stored
metrics means the artifact goes stale whenever new runs are added. The DB is the
single source of truth; `recommend_best_config()` is a query, not a file.

### Anti-Pattern: Regression Baseline as runs[-2]

The current `detect_regressions()` behavior of always comparing against the second-most-
recent run will incorrectly flag a sweep as a regression (sweep run #6 compares to sweep
run #5, not to the pre-sweep baseline). The `is_baseline` tag resolves this.

---

## Scalability Considerations

| Concern | At 100 IAM samples | At 1,000 IAM samples | At 10,000 samples |
|---------|-------------------|----------------------|-------------------|
| DB size | ~5 MB per full sweep | ~50 MB per sweep | ~500 MB — still fine for SQLite |
| Query performance | Instant | Instant (indexed) | Add `idx_po_category` on samples join |
| Sweep cost | ~$0.20/sweep (gemini) | ~$2/sweep | ~$20/sweep — use `--smoke` or batch API |
| Significance tests | N/A (n<30 unreliable) | Wilcoxon valid | Bootstrap CI becomes cheap |

IAM test set is ~1,861 lines. A realistic benchmark uses 100-500 samples (speed/cost
tradeoff). The existing SQLite schema handles this without concern.

---

## Open Questions for Phase-Specific Research

1. **IAM GT file format details**: IAM distributes ground truth in `ascii/` as one `.txt`
   per form, with lines starting with `CSR` and line IDs. Parsing this format needs a
   small dedicated parser. Verify format against current IAM download before implementing
   `ingest_iam_dataset()`.

2. **scipy availability on Apple Silicon**: scipy has arm64 wheels as of scipy 1.11+.
   Verify `pip install scipy` works cleanly in the project's virtual environment before
   adding it to extras.

3. **Line-level vs form-level IAM evaluation**: IAM's `lines/` images are line-level;
   `words/` are word-level. The engine's `read_page(line_level=True)` is designed for
   full pages. Decide whether to benchmark on IAM `lines/` images as single-line pages
   (simplest) or on IAM form images with line-level segmentation enabled (most realistic).
   This choice affects both `ingest_iam_dataset()` and `run_benchmark()` strategy.
