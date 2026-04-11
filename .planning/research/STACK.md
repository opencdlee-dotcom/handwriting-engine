# Technology Stack: HTR Benchmarking Milestone

**Project:** handwriting-engine v3.0 — Verified Accuracy
**Researched:** 2026-04-11
**Scope:** New stack additions only. Existing stack (Pillow, PyMuPDF, click, jiwer, numpy, sqlite3, anthropic, openai, google-genai) not re-evaluated.

---

## What Already Exists (Do Not Re-Add)

The `benchmark` optional-dep group already declares:

```toml
benchmark = ["jiwer>=3.0.0", "numpy>=1.24.0"]
```

The existing benchmark subpackage provides:
- CER/WER via jiwer (C++ backend) with pure-Python fallback
- Per-sample CER vectors stored in `eval_metrics` (SQLite)
- Aggregate statistics via stdlib `statistics` (mean, median, stdev)
- Regression detection via threshold delta comparison
- ASCII table / JSON / CSV report output

**Gaps for v3.0:**
1. No IAM dataset loader — images and ground-truth transcriptions must be fetched and parsed
2. No statistical significance testing — threshold comparison is not a p-value
3. No visualization — terminal tables only, no charts for lab notebook validation workflows
4. No structured multi-strategy comparison report suitable for committing as a baseline artifact

---

## Recommended Additions

### 1. IAM Dataset Access

**How IAM works:** The IAM Handwriting Database (Marti & Bunke, 2002) is hosted at the University of Bern. Direct download requires a free registration at `https://fki.tic.heia-fr.ch/databases/iam-handwriting-database`. The dataset ships as:
- `formsA-D.tgz`, `formsE-H.tgz`, `formsI-Z.tgz` — full-page TIFF scans (~677 forms)
- `words.tgz` — pre-cropped word images (~115k words)
- `lines.tgz` — pre-cropped line images (~13k lines)
- `ascii.tgz` — ground-truth transcriptions in a structured flat-file format

**Recommendation: Use line-level images from `lines.tgz` with `ascii/lines/` ground truth.**

Rationale: The engine processes full pages by line-level segmentation internally. Benchmarking at the pre-cropped line level removes segmentation error from the CER measurement, isolating transcription accuracy cleanly. Word-level is too granular (misses line context the engine uses); form-level introduces our own segmentation as a confound. The IAM lines split has ~13k samples — a representative 500-sample random subset is sufficient for statistical power at the CER levels we're targeting (~1.3–1.7%).

**No Python library needed for IAM loading.** The ground-truth format (`ascii/lines/`) is a simple flat file: each line is tab-separated with fields `id ok err graylevel components x y w h transcript`. A purpose-built loader of ~60 lines fits cleanly in `benchmark/iam_loader.py`. Do not add `datasets` (HuggingFace) for this — it pulls in a 500 MB dependency chain for parsing a flat text file, and the IAM HuggingFace mirror (`Teklia/IAM-line`) requires auth tokens for the full dataset anyway.

| Component | Approach | Why |
|-----------|----------|-----|
| IAM images | Manual download into `~/.handwriting-engine/iam/` | Registration-gated; cannot automate |
| Ground truth parsing | `benchmark/iam_loader.py` (~60 LOC) | Flat file format, no library needed |
| Integration | `ingest_iam_lines(iam_dir)` → calls existing `insert_sample` + `insert_ground_truth` | Reuses existing DB pipeline unchanged |

**IAM loader design — key parsing detail:**

The ascii/lines ground-truth file uses `|` as word separator in the transcript field and `#` as the sentence separator. Normalized transcript is: join words on space, strip leading `#`. The `ok` field is `ok` or `err` — only ingest `ok` lines for the clean test set, `err` lines for the degraded subset.

### 2. Statistical Significance Testing

**Recommendation: `scipy>=1.11.0`** — lazy import in a new `benchmark/stats.py` module.

The correct test for comparing two HTR strategies is a **paired two-sided Wilcoxon signed-rank test**, not a paired t-test. CER distributions on handwriting samples are right-skewed (most samples have low CER; a few pathological samples have very high CER). The t-test assumes normality; Wilcoxon makes no distribution assumption. For IAM line-level samples at ~500 count, both tests will agree, but Wilcoxon is the defensible choice for a paper-quality result.

Additionally: **McNemar's test** (`scipy.stats.contingency.mcnemar`) is appropriate when comparing binary correctness (word correct/incorrect) between strategies on the same sample set. Add this as a secondary metric.

| Test | When to Use | scipy function |
|------|-------------|----------------|
| Wilcoxon signed-rank | Comparing CER vectors for strategy A vs B on same samples | `scipy.stats.wilcoxon(cer_a, cer_b, alternative='two-sided')` |
| McNemar | Comparing per-word correctness between strategies | `scipy.stats.contingency.mcnemar` |
| Bonferroni correction | When comparing >2 strategies simultaneously | Manual: `alpha / n_comparisons` |

**Effect size:** Add Cohen's d (or r = Z/sqrt(N) from Wilcoxon) alongside p-values. A p=0.04 improvement that moves CER from 1.67% to 1.65% is statistically significant but practically irrelevant. Report both.

scipy is already implicitly available on this dev machine (it ships with most scientific Python installs), but must be declared explicitly as an optional dep to ensure CI reproducibility.

```toml
# Add to pyproject.toml [project.optional-dependencies]
benchmark = ["jiwer>=3.0.0", "numpy>=1.24.0", "scipy>=1.11.0"]
```

**Do NOT add:** `pingouin`, `statsmodels` — these are full statistical modeling frameworks. We need exactly two functions from scipy; anything else is overkill.

### 3. Visualization

**Recommendation: `matplotlib>=3.8.0`** — lazy import in `benchmark/viz.py`, guarded by `[extra] viz` optional dep or included in `benchmark`.

The output use case is:
1. CER distribution box plots by strategy (for SUMMARY artifact)
2. Scatter: image quality score vs CER (extends existing `quality_correlation` report)
3. Bar chart: per-strategy mean CER with confidence intervals (for lab notebook validation docs)

matplotlib is the right choice here — not plotly (overkill, browser-based), not seaborn (adds another dep on top of matplotlib for marginal DX improvement on 3 chart types). The output format is PNG files saved to a configurable directory, not interactive.

```toml
# Add as separate optional dep group (not bundled with benchmark by default)
viz = ["matplotlib>=3.8.0"]
```

Rationale for separating `viz` from `benchmark`: matplotlib has a large footprint and requires a display backend on headless CI. Keeping it optional avoids breaking CI that runs `pip install handwriting-engine[benchmark]`. The CLI `benchmark report --viz` flag should raise a helpful error if matplotlib is absent rather than silently skipping charts.

**Do NOT add:** seaborn (adds a pandas soft-dep), plotly (requires a browser / Kaleido for static export), bokeh.

### 4. Structured Comparison Artifact

No new library needed. The existing `compare_runs` function in `report.py` and the JSON report format (`generate_report(fmt='json')`) are sufficient. What's needed is a CLI command `benchmark commit-baseline` that:
1. Runs `generate_report(fmt='json')` on the designated IAM run
2. Writes to `.benchmark-baseline.json` in the project root
3. Future `detect_regressions` calls load this file as the baseline when no prior DB run exists

This is a code addition to `cli.py` and `report.py`, not a stack addition.

---

## Complete Benchmark Optional Deps (After v3.0)

```toml
[project.optional-dependencies]
benchmark = ["jiwer>=3.0.0", "numpy>=1.24.0", "scipy>=1.11.0"]
viz = ["matplotlib>=3.8.0"]
```

Install commands:

```bash
# Core benchmark (statistical tests included)
pip install "handwriting-engine[benchmark]"

# With visualization
pip install "handwriting-engine[benchmark,viz]"
```

---

## What NOT to Add

| Candidate | Why Not |
|-----------|---------|
| `pandas` | All aggregation is already in SQLite or stdlib `statistics`. Adding pandas for a 6-column result table is ~35 MB of dep for no gain. |
| `datasets` (HuggingFace) | 500 MB+ dep chain for IAM. The `Teklia/IAM-line` mirror requires auth anyway. Custom loader is 60 LOC. |
| `editdistance` | Already have jiwer (C++ backend) for CER/WER. Two Levenshtein libraries is redundant. |
| `seaborn` | Adds pandas soft-dep. All charts needed are standard matplotlib. |
| `plotly` | Interactive charts require Kaleido for static export (another dep). Headless CI hostile. |
| `pingouin` / `statsmodels` | Full statistical modeling suites. We need 2 scipy functions. |
| `mlflow` / `wandb` | Experiment tracking overkill. SQLite DB already tracks all run metadata and metrics. |
| `Levenshtein` (PyPI) | Redundant with jiwer. |
| OpenCV (for IAM) | IAM images are TIFF/PNG — Pillow reads them fine. OpenCV is already an optional dep for other reasons; don't make it a benchmark dependency. |

---

## Integration Points with Existing Benchmark DB

| New Capability | Integrates With | Notes |
|----------------|-----------------|-------|
| IAM loader | `ingest.py` → `insert_sample` + `insert_ground_truth` | `source='iam'` in ground_truths table; `category='iam-lines'` in samples table |
| Wilcoxon test | `report.py` → reads `eval_metrics` CER vectors per run×strategy | Needs paired sample alignment: only samples present in BOTH runs get tested |
| McNemar test | `report.py` → derives word-correct boolean from WER data | Word-level correctness requires per-word alignment, not just aggregate WER |
| Visualization | `report.py` → same aggregated data as ASCII table | `viz.py` module consumes `StrategyResult` list already returned by `_aggregate_results` |
| Baseline commit | `report.py` + `cli.py` | Writes JSON artifact; `detect_regressions` loads it as fallback baseline |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| IAM format (flat file, registration-gated) | HIGH | Verified against official HEIA-FR site documentation and published papers; format unchanged since 2002 |
| scipy Wilcoxon for CER comparison | HIGH | Standard choice in HTR literature (Graves 2006, Puigcerver 2017); scipy API stable since 1.0 |
| matplotlib for output charts | HIGH | De facto standard; no viable lighter alternative for non-interactive PNG output |
| jiwer already covers CER/WER | HIGH | Source code confirmed in metrics.py |
| No pandas needed | HIGH | All aggregation confirmed in SQLite queries and stdlib statistics in existing code |
| HuggingFace IAM mirror auth requirement | MEDIUM | Teklia/IAM-line dataset page shows gated access; manual verification blocked by registration wall |

---

## Sources

- IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- scipy.stats.wilcoxon: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- scipy version history: https://github.com/scipy/scipy/releases (1.11.0 = July 2023, stable on Python 3.11+)
- matplotlib stable release: https://matplotlib.org/stable/users/release_notes.html (3.8.x)
- IAM line-level benchmark standard: Puigcerver (2017) "Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?" — establishes lines split as canonical benchmark unit
