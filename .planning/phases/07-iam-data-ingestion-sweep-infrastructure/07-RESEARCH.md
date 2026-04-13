# Phase 7: IAM Data Ingestion + Sweep Infrastructure - Research

**Researched:** 2026-04-11
**Domain:** IAM Handwriting Database parsing, benchmark sweep orchestration, per-writer report extension
**Confidence:** HIGH (codebase fully read; IAM format confirmed from multiple independent sources)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| IAM-01 | Developer runs `benchmark ingest-iam <path>` against IAM ascii/ directory; DB populated with line images tagged `category="iam"` and `student="iam-writer-XXX"`; no manual data wrangling | `insert_sample()` already accepts `student` + `category`; ascii/lines.txt parser needed in ingest.py; image path reconstruction from line ID is deterministic |
| IAM-02 | Developer runs `benchmark sweep` executing all 5 strategies (baseline, self_correct, line_level, prompt_adapted, zoomed_verify) against IAM test set, storing one run_id per strategy | `run_benchmark()` already takes a `strategies` list; sweep = loop over 5 named strategies calling `run_benchmark()` per strategy; new `benchmark sweep` CLI command needed |
| IAM-03 | `benchmark report` shows per-writer CER breakdown table, distinguishing systematic gains from writer-specific gains | `samples.student` already stores writer; SQL GROUP BY student on eval_metrics + provider_outputs is sufficient; new `generate_per_writer_report()` function needed in report.py |
</phase_requirements>

---

## Summary

Phase 7 builds on the complete Phase 6 infrastructure. The benchmark DB schema (v4) already has `student` and `category` columns on the `samples` table, so no schema migration is needed for IAM ingestion — the fields are already there. The IAM ascii/lines.txt file uses a well-documented 9-column space-separated format; each line encodes a unique line ID from which the writer ID and image path are deterministically reconstructable. The five strategies named in IAM-02 (`baseline`, `self_correct`, `line_level`, `prompt_adapted`, `zoomed_verify`) are not existing consensus strategy strings — they are sweep-level names that map to specific `run_benchmark()` parameter combinations. The planner must define this mapping. The per-writer report (IAM-03) requires only a new SQL query against the existing schema; no schema changes are needed.

**Primary recommendation:** Implement three new units: (1) `ingest_iam()` function in ingest.py that parses ascii/lines.txt and bulk-inserts samples; (2) `run_sweep()` function in evaluate.py (or a new sweep.py) that loops over the 5 strategy configs calling `run_benchmark()` each time; (3) `generate_per_writer_report()` function in report.py that groups eval_metrics by student and renders a CER breakdown table. Wire all three into CLI commands.

The cost guardrail from Phase 6 (`benchmark run --yes`) must fire before any sweep run. The sweep command should show projected cost for all 5 strategies × N samples before any API call.

---

## Standard Stack

### Core (already in project — no new dependencies needed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlite3 | stdlib | DB queries for per-writer grouping | Already used throughout benchmark |
| pathlib | stdlib | IAM directory traversal and image path reconstruction | Already used in ingest.py |
| hashlib | stdlib | SHA-256 dedup of IAM line images | Already used in `hash_file()` |
| click | >=8.1 | New CLI commands (ingest-iam, sweep) | Already used throughout cli.py |
| Pillow | >=10.0 | Verify IAM PNG files open correctly | Already used throughout |

### No New Dependencies
All Phase 7 functionality can be implemented using libraries already in the project. IAM ascii/lines.txt is plain text — no special parser needed.

---

## IAM Handwriting Database Format

### What the Developer Downloads

The IAM Handwriting Database (registration-gated at HEIA-FR) provides two downloads relevant to this phase:
1. `data/ascii.tgz` — extracts to `ascii/` directory containing `lines.txt` (and `words.txt`, `forms.txt`)
2. `data/lines.tgz` — extracts to `lines/` directory containing PNG images organized in a two-level folder hierarchy

The `benchmark ingest-iam` command receives the **ascii/ directory path** (as stated in IAM-01). The line image PNGs live in a parallel `lines/` directory at the same level as `ascii/`. The command needs both to function.

**Expected developer usage:**
```
benchmark ingest-iam /path/to/ascii   [lines images found at ../lines/ automatically]
```
Or the command could accept a `--lines-dir` override for non-standard layouts.

### lines.txt Format (HIGH confidence — multiple independent sources agree)

File location: `ascii/lines.txt`

Lines beginning with `#` are comments. Each data line has exactly 9 space-separated fields:

```
line_id  status  graylevel  num_components  x  y  w  h  transcription
```

| Field | Position | Example | Description |
|-------|----------|---------|-------------|
| line_id | 0 | `a01-000u-00` | Hierarchical ID: writer-form-line |
| status | 1 | `ok` | `ok` = good segmentation; `err` = bad (filter these out) |
| graylevel | 2 | `154` | Binarization threshold (not needed for our use case) |
| num_components | 3 | `1` | Number of connected components (not needed) |
| x | 4 | `408` | Bounding box x (not needed) |
| y | 5 | `768` | Bounding box y (not needed) |
| w | 6 | `27` | Bounding box width (not needed) |
| h | 7 | `51` | Bounding box height (not needed) |
| transcription | 8+ | `A MOVE IN` | Ground truth text (may contain spaces — all tokens from col 8 onward) |

**Important:** The transcription uses `|` to separate words in some IAM versions. Always join fields from index 8 onward as the transcription. The `|` pipe character represents a word boundary — replace with a single space when storing as ground truth.

### Line ID to Writer ID and Image Path

```python
line_id = "a01-000u-00"
parts = line_id.split("-")   # ["a01", "000u", "00"]
writer_id = parts[0]          # "a01"
form_id = parts[0] + "-" + parts[1]  # "a01-000u"
# Image path: lines/a01/a01-000u/a01-000u-00.png
image_path = lines_dir / writer_id / form_id / (line_id + ".png")
```

Writer ID mapped to student tag: `f"iam-writer-{writer_id}"` (e.g., `"iam-writer-a01"`)

### IAM Test Partition (Aachen split — what to ingest)

The requirements specify ingesting against the "IAM test set" (STATE.md: "Test/train discipline: only IAM test partition"). The Aachen partition (most widely used) defines:
- **Test:** 2,915 lines from 336 forms
- **Validation:** 966 lines from 115 forms
- **Train:** 6,161 lines from 747 forms

The Aachen partition lists are distributed as separate text files (`testset.txt`, `validationset1.txt`, `validationset2.txt`, `trainset.txt`) that contain form IDs. The `ingest-iam` command should accept an optional `--partition` flag accepting a path to the partition list file, defaulting to ingesting only lines whose form IDs appear in the test partition file.

If no partition file is provided, the command should warn and offer to ingest all lines, or require `--all` to confirm intent.

**Key discipline:** Never ingest train split lines into the benchmark DB to avoid contamination.

### Filtering Logic

```python
def parse_iam_lines(lines_txt: Path, partition_forms: set[str] | None = None) -> list[dict]:
    """Parse lines.txt, returning records for ok-status lines in partition."""
    records = []
    for line in lines_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        line_id, status = fields[0], fields[1]
        if status == "err":
            continue   # filter bad segmentations
        parts = line_id.split("-")
        form_id = parts[0] + "-" + parts[1]
        if partition_forms and form_id not in partition_forms:
            continue
        writer_id = parts[0]
        # Join transcription tokens (fields[8:]), replace | with space
        transcription = " ".join(fields[8:]).replace("|", " ").strip()
        records.append({
            "line_id": line_id,
            "writer_id": writer_id,
            "form_id": form_id,
            "transcription": transcription,
        })
    return records
```

---

## Architecture Patterns

### Recommended Project Structure for Phase 7

```
handwriting_engine/benchmark/
├── db.py             # No changes needed (schema v4 already has student, category)
├── ingest.py         # ADD: ingest_iam() function
├── evaluate.py       # ADD: run_sweep() function
├── report.py         # ADD: generate_per_writer_report() function
└── models.py         # Possibly ADD: WriterResult dataclass for per-writer report

handwriting_engine/
└── cli.py            # ADD: benchmark ingest-iam command, benchmark sweep command,
                      #      extend benchmark report with --per-writer flag
```

### Pattern 1: IAM Ingest Function (ingest.py addition)

```python
# Source: codebase analysis of existing ingest_directory() pattern
def ingest_iam(
    ascii_dir: str | Path,
    lines_dir: str | Path | None = None,
    partition_file: str | Path | None = None,
    db_path: Path | str | None = None,
) -> dict:
    """Parse IAM ascii/lines.txt and ingest line images.

    Returns summary dict: {"ingested": N, "skipped_dup": N, "skipped_missing": N}
    """
    ascii_dir = Path(ascii_dir)
    lines_txt = ascii_dir / "lines.txt"

    # Resolve lines/ directory (default: sibling of ascii/)
    if lines_dir is None:
        lines_dir = ascii_dir.parent / "lines"

    # Load partition filter
    partition_forms = None
    if partition_file:
        partition_forms = set(Path(partition_file).read_text().split())

    records = parse_iam_lines(lines_txt, partition_forms)

    conn = get_connection(db_path)
    try:
        ingested = skipped_dup = skipped_missing = 0
        for r in records:
            img_path = (
                Path(lines_dir) / r["writer_id"] /
                r["form_id"] / (r["line_id"] + ".png")
            )
            if not img_path.exists():
                skipped_missing += 1
                continue
            img_hash = hash_file(img_path)
            existing = get_sample_by_hash(conn, img_hash)
            if existing:
                skipped_dup += 1
                continue
            try:
                sample_id = insert_sample(
                    conn,
                    image_path=str(img_path.resolve()),
                    image_hash=img_hash,
                    student=f"iam-writer-{r['writer_id']}",
                    category="iam",
                    source_dir=str(Path(lines_dir).resolve()),
                    page_number=0,
                    notes=f"iam_line_id:{r['line_id']}",
                    autocommit=False,
                )
                insert_ground_truth(
                    conn, sample_id, r["transcription"],
                    source="iam_ascii"
                )
                conn.commit()
                ingested += 1
            except sqlite3.IntegrityError:
                skipped_dup += 1
    finally:
        conn.close()

    return {"ingested": ingested, "skipped_dup": skipped_dup, "skipped_missing": skipped_missing}
```

**Critical:** Call `insert_ground_truth()` immediately after `insert_sample()` within the same transaction. IAM ground truth comes from the same file as the image metadata — no separate annotation step needed.

**No quality assessment:** Skip `assess_quality` for IAM — line images are already clean, pre-segmented PNG files. This keeps ingest fast (potentially thousands of lines).

### Pattern 2: Five Strategy Configs for Sweep (IAM-02)

The five strategy names in IAM-02 are **sweep-level labels**, not existing consensus strategy strings. They map to specific `run_benchmark()` parameter combinations:

| Strategy Name | run_benchmark() parameters | Notes |
|--------------|---------------------------|-------|
| `baseline` | `strategies=[]`, `auto_enhance=False`, `vocab_hints_off=1` | Single-provider, no enhancement, no vocab hints — reproducible anchor |
| `self_correct` | `strategies=["self_correct"]` | Uses consensus.py `self_correct` strategy |
| `line_level` | pass `line_level=True` to underlying read (requires parameter threading) | `vision.read_page(line_level=True)` — not currently a run_benchmark parameter |
| `prompt_adapted` | `strategies=[]`, uses `prompt_adapter.py` (already applied by default) | Document that prompt adaptation is ON by default; this strategy is baseline+explicit adaptation |
| `zoomed_verify` | `auto_retry=True` on underlying read (not currently a run_benchmark parameter) | `vision.read_page(auto_retry=True)` — not currently a run_benchmark parameter |

**Critical gap:** `run_benchmark()` does not currently expose `line_level` or `auto_retry` as parameters. These are `read_page()` parameters. The sweep infrastructure must either:
1. Add `line_level: bool` and `auto_retry: bool` parameters to `run_benchmark()` / `_read_single()` (preferred — consistent with how `auto_enhance` and `enhance_strategy` were added), OR
2. Create specialized runner functions for those two strategies

**Recommended approach:** Thread `line_level` and `auto_retry` through `_read_single()` → `run_benchmark()`, matching the existing pattern for `auto_enhance` / `enhance_strategy`.

### Pattern 3: Sweep Command

```python
# Source: codebase analysis of existing benchmark run command pattern
SWEEP_STRATEGIES = [
    {
        "name": "baseline",
        "label": "sweep:baseline",
        "kwargs": {"strategies": [], "vocab_hints_off": 1, "auto_enhance": False},
    },
    {
        "name": "self_correct",
        "label": "sweep:self_correct",
        "kwargs": {"strategies": ["self_correct"]},
    },
    {
        "name": "line_level",
        "label": "sweep:line_level",
        "kwargs": {"strategies": [], "line_level": True},
    },
    {
        "name": "prompt_adapted",
        "label": "sweep:prompt_adapted",
        "kwargs": {"strategies": []},  # prompt adaptation ON by default
    },
    {
        "name": "zoomed_verify",
        "label": "sweep:zoomed_verify",
        "kwargs": {"strategies": [], "auto_retry": True},
    },
]

def run_sweep(
    provider: str = "gemini",
    category_filter: str = "iam",
    db_path=None,
    yes: bool = False,
    on_progress=None,
) -> dict[str, int]:
    """Execute all sweep strategies, return {strategy_name: run_id}."""
    run_ids = {}
    for config in SWEEP_STRATEGIES:
        run_id = run_benchmark(
            label=config["label"],
            providers=[provider],
            db_path=db_path,
            # category_filter would filter samples by category="iam"
            **config["kwargs"],
        )
        run_ids[config["name"]] = run_id
    return run_ids
```

**Critical gap:** `run_benchmark()` does not have a `category_filter` parameter. It operates on all samples with ground truth. To run only against IAM samples, either:
1. Add `category_filter: str | None` to `run_benchmark()` that filters `samples_with_ground_truth()` by `category`, OR
2. Pre-collect IAM sample IDs and pass via `sample_ids` parameter (already supported)

**Recommended approach:** Use the existing `sample_ids` parameter. The sweep command queries the DB for `category="iam"` sample IDs first, then passes them to `run_benchmark()`.

### Pattern 4: Per-Writer CER Report (IAM-03)

```python
# Source: codebase analysis of existing report.py pattern
def generate_per_writer_report(
    run_id: int | None = None,
    db_path=None,
) -> str:
    """Per-writer CER breakdown for a run. Uses samples.student column."""
    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        rows = conn.execute(
            """SELECT s.student, AVG(em.cer) as mean_cer,
                      MIN(em.cer) as min_cer, MAX(em.cer) as max_cer,
                      COUNT(*) as n_samples
               FROM provider_outputs po
               JOIN eval_metrics em ON em.provider_output_id = po.id
               JOIN samples s ON s.id = po.sample_id
               WHERE po.run_id = ? AND s.student != ''
               GROUP BY s.student
               ORDER BY mean_cer DESC""",
            (run_id,)
        ).fetchall()
    finally:
        conn.close()

    lines = [f"Per-Writer CER (Run #{run_id})", ""]
    header = f"{'Writer':<25} {'Mean CER':>9} {'Min CER':>9} {'Max CER':>9} {'N':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['student']:<25} {r['mean_cer']:>8.2%} "
            f"{r['min_cer']:>8.2%} {r['max_cer']:>8.2%} {r['n_samples']:>4}"
        )
    return "\n".join(lines)
```

**No schema change required.** The `samples.student` column (already at v4 schema) stores `"iam-writer-XXX"` set during ingest. This query joins the three existing tables.

### Anti-Patterns to Avoid

- **Ingesting train split lines:** The `ingest-iam` command must filter to test partition only (or require explicit `--all`). Ingesting train data would contaminate the benchmark with samples the model indirectly learned from.
- **Running sweep without cost guardrail:** The `benchmark sweep` command must show the same cost projection as `benchmark run` before executing the first strategy. Each strategy run should individually pass `--yes` (or accept a top-level `--yes` that bypasses all 5 confirmations).
- **Embedding partition knowledge in the parser:** The partition file path should be a CLI parameter, not hardcoded. The IAM partition scheme has multiple variants (Aachen, official, etc.).
- **Treating `line_level` and `zoomed_verify` as consensus strategies:** These are `read_page()` flags, not `read_with_consensus()` strategy names. They must be threaded through `_read_single()`.
- **Running quality assessment on IAM images:** IAM line images are already segmented, clean PNGs. Quality assessment adds latency without benefit. Skip it in `ingest_iam()`.
- **Committing ground truth in a separate transaction from the sample:** A failed commit between `insert_sample()` and `insert_ground_truth()` would leave orphaned samples with no GT, causing them to be skipped in `samples_with_ground_truth()`. Both inserts must be committed atomically.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image dedup | Custom hash tracking | `hash_file()` + `get_sample_by_hash()` | Already in ingest.py; handles race conditions |
| GT storage | Custom text file | `insert_ground_truth()` with `source="iam_ascii"` | Already handles versioning, multiple GTs per sample |
| CER computation | Custom edit distance | `character_error_rate()` from metrics.py | Already normalized, handles edge cases |
| Strategy execution | Custom provider calls | `run_benchmark()` with appropriate parameters | Handles token tracking, error capture, DB commits per-sample |
| Cost projection | Custom cost estimator | `estimate_cost()` from evaluate.py + existing guardrail pattern | Already accounts for per-provider rates |

**Key insight:** The benchmark infrastructure is complete. Phase 7 is about wiring IAM data into it and orchestrating multi-strategy execution — not building new measurement machinery.

---

## Common Pitfalls

### Pitfall 1: `err`-status Lines
**What goes wrong:** IAM lines.txt contains lines with `status="err"` indicating poor segmentation. Ingesting these introduces corrupted ground truth that inflates CER.
**Why it happens:** The err marker is in field[1] of each line. Easy to miss when parsing.
**How to avoid:** Always filter `if fields[1] == "err": continue` before processing.
**Warning signs:** CER above 20% on many lines — suggests err-status lines were ingested.

### Pitfall 2: Transcription with `|` Separators
**What goes wrong:** Some IAM versions encode word boundaries in transcription as pipe characters (`put|down|a|resolution`). Storing the raw transcription as-is causes CER inflated by `|` characters.
**Why it happens:** IAM uses `|` as word delimiter in its native format.
**How to avoid:** Replace `"|"` with `" "` when storing ground truth. Already done in pattern above.

### Pitfall 3: Missing Lines Image Directory
**What goes wrong:** Developer provides ascii/ path but lines/ images are in a non-standard location. Command fails with FileNotFoundError on every image.
**Why it happens:** IAM is manually downloaded; directory layout varies by how user extracts archives.
**How to avoid:** Default to `ascii_dir.parent / "lines"` but accept `--lines-dir` override. Emit a clear error message if the directory doesn't exist.

### Pitfall 4: `line_level` / `auto_retry` Not Threaded to `run_benchmark()`
**What goes wrong:** Sweep strategies `line_level` and `zoomed_verify` silently fall back to baseline behavior because the flags aren't threaded through to `_read_single()`.
**Why it happens:** These parameters exist in `read_page()` but not in `run_benchmark()` or `_read_single()`.
**How to avoid:** Add `line_level: bool = False` and `auto_retry: bool = False` to both `_read_single()` and `run_benchmark()` before implementing the sweep. Verify by checking that `_read_single()` passes them through to `read_page()`.
**Warning signs:** `line_level` and baseline CER are identical — the flag is being ignored.

### Pitfall 5: Sweep Cost Surprise
**What goes wrong:** `benchmark sweep` runs all 5 strategies × N IAM samples without warning, triggering large API bills.
**Why it happens:** New `sweep` command doesn't inherit the cost guardrail from `benchmark run`.
**How to avoid:** Implement cost projection at the top of the sweep command, showing total projected cost across all 5 strategies before any API call.

### Pitfall 6: Per-Writer Report on Non-IAM Data
**What goes wrong:** `benchmark report --per-writer` on a non-IAM run shows empty results because samples lack the `student` field.
**Why it happens:** Lab notebook samples may have empty `student` column.
**How to avoid:** The query already filters `WHERE s.student != ''`. Add a note in the report output when no writers are found: "No writer data — ingest IAM samples with ingest-iam first."

### Pitfall 7: Train Partition Contamination
**What goes wrong:** Developer runs `ingest-iam` without specifying a partition file, ingesting all 13,000+ IAM lines including training lines.
**Why it happens:** Without filtering, `parse_iam_lines()` returns all ok-status lines.
**How to avoid:** If no partition file is provided, default to a safe behavior: either (a) require `--partition-file` to be set, or (b) warn loudly and require `--all-partitions` flag to override. Document this in the CLI help text.

---

## Code Examples

### IAM lines.txt Parser (Verified against IAM format documentation)
```python
# Source: IAM database official format, cross-verified via Keras docs + Laia README
def parse_iam_lines(
    lines_txt: Path,
    partition_forms: set[str] | None = None,
) -> list[dict]:
    """Parse ascii/lines.txt. Returns records for ok-status lines in partition.

    lines.txt columns (space-separated):
      [0] line_id  [1] status  [2] graylevel  [3] num_components
      [4] x  [5] y  [6] w  [7] h  [8+] transcription tokens
    """
    records = []
    for raw_line in lines_txt.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("#") or not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) < 9:
            continue
        line_id, status = fields[0], fields[1]
        if status == "err":
            continue
        parts = line_id.split("-")
        if len(parts) < 3:
            continue
        form_id = f"{parts[0]}-{parts[1]}"
        if partition_forms is not None and form_id not in partition_forms:
            continue
        writer_id = parts[0]
        transcription = " ".join(fields[8:]).replace("|", " ").strip()
        records.append({
            "line_id": line_id,
            "writer_id": writer_id,
            "form_id": form_id,
            "transcription": transcription,
        })
    return records
```

### Image Path Construction
```python
# Source: IAM directory structure (confirmed via Keras docs and Laia README)
def iam_image_path(lines_dir: Path, line_id: str) -> Path:
    """Reconstruct image path from line ID.
    line_id "a01-000u-00" -> lines/a01/a01-000u/a01-000u-00.png
    """
    parts = line_id.split("-")
    writer_id = parts[0]
    form_id = f"{parts[0]}-{parts[1]}"
    return lines_dir / writer_id / form_id / f"{line_id}.png"
```

### Sweep Sample ID Pre-collection
```python
# Source: codebase analysis — samples_with_ground_truth() pattern + existing sample_ids param
def _get_iam_sample_ids(conn) -> list[int]:
    """Return sample IDs for category='iam' samples with ground truth."""
    rows = conn.execute(
        """SELECT DISTINCT s.id FROM samples s
           JOIN ground_truths gt ON gt.sample_id = s.id
           WHERE s.category = 'iam'
           ORDER BY s.id"""
    ).fetchall()
    return [r["id"] for r in rows]
```

### Existing `run_benchmark()` signature (for reference)
```python
# Source: handwriting_engine/benchmark/evaluate.py (Phase 6 final state)
def run_benchmark(
    label: str = "",
    providers: list[str] | None = None,
    strategies: list[str] | None = None,
    domain: str = "biology",
    sample_ids: list[int] | None = None,     # <- use this for IAM filtering
    db_path: Path | str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    mode: str = "full",
    auto_enhance: bool = False,
    inject_lessons: bool = False,
    enhance_strategy: str | None = None,
    iam_partition: str | None = None,
    vocabulary_hints: list[str] | None = None,
    vocab_hints_off: int = 0,
    # NEW for Phase 7:
    # line_level: bool = False,     <- needs to be added
    # auto_retry: bool = False,     <- needs to be added
) -> int:
    ...
```

### CLI Command Registration Pattern
```python
# Source: handwriting_engine/cli.py existing pattern
@benchmark.command("ingest-iam")
@click.argument("ascii_dir", type=click.Path(exists=True))
@click.option("--lines-dir", default=None, type=click.Path(),
              help="Path to lines/ image directory (default: sibling of ascii/)")
@click.option("--partition-file", default=None, type=click.Path(exists=True),
              help="Text file listing form IDs to ingest (e.g. testset.txt)")
@click.option("--db-path", default=None, hidden=True)
def benchmark_ingest_iam(ascii_dir, lines_dir, partition_file, db_path):
    """Ingest IAM Handwriting Database line images into benchmark DB."""
    ...

@benchmark.command("sweep")
@click.option("--provider", "-p", default="gemini", help="Provider to use")
@click.option("--yes", "-y", is_flag=True, help="Skip cost confirmation")
@click.option("--db-path", default=None, hidden=True)
def benchmark_sweep(provider, yes, db_path):
    """Run all 5 strategies against IAM test set, storing one run_id per strategy."""
    ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual GT annotation (`benchmark transcribe`) | Auto-parsed from ascii/lines.txt | Phase 7 | No manual step needed for IAM |
| Running strategies one-by-one via `benchmark run` | Single `benchmark sweep` command | Phase 7 | 5 strategy runs orchestrated automatically |
| Aggregate CER only in report | Per-writer CER breakdown added | Phase 7 | Distinguishes systematic gains from writer-specific gains |

**No deprecated patterns in this phase.** All additions are extensions to existing patterns.

---

## Open Questions

1. **Partition file location**
   - What we know: IAM distributes partition files (testset.txt, trainset.txt) as separate downloads from the main database
   - What's unclear: Whether the developer will have these files, and where they'll be relative to the ascii/ directory
   - Recommendation: Accept `--partition-file` as a CLI argument. Document in CLI help that testset.txt is available as a separate download from HEIA-FR. Default behavior if not provided: emit warning + require `--all-partitions` flag.

2. **Strategy naming: `prompt_adapted` vs baseline**
   - What we know: `prompt_adapter.py` is already applied by default in `read_page()` via `adapt_system_prompt()` / `adapt_user_prompt()`
   - What's unclear: What `prompt_adapted` is supposed to measure differently from `baseline`
   - Recommendation: Treat `prompt_adapted` as a run with prompt adaptation explicitly documented ON (vs. `baseline` which should record it as ON but with `vocab_hints_off=1`). The key differentiator between baseline and prompt_adapted may be vocabulary hints: baseline has them off, prompt_adapted has them on. The planner should confirm this interpretation with the developer.

3. **IAM lines/ directory layout when only ascii/ is provided**
   - What we know: Default assumption is `lines/` is a sibling of `ascii/`
   - What's unclear: No guarantee the developer will unpack this way
   - Recommendation: Always require `--lines-dir` to be explicit, OR auto-detect with a clear fallback message.

4. **`run_benchmark()` threading for `line_level` and `auto_retry`**
   - What we know: These parameters exist in `read_page()` but not in `run_benchmark()`
   - What's unclear: Whether adding them is within Phase 7 scope vs. a prerequisite
   - Recommendation: Include in Phase 7 Wave 0 as a prerequisite task. The changes are small (add two boolean params to `_read_single()` and `run_benchmark()`).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0.0 |
| Config file | pyproject.toml (`[tool.pytest.ini_options]` not set — uses defaults) |
| Quick run command | `pytest tests/test_benchmark_ingest.py tests/test_benchmark_evaluate.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IAM-01 | `parse_iam_lines()` skips `#` comments | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_parse_skips_comments -x` | ❌ Wave 0 |
| IAM-01 | `parse_iam_lines()` filters `err`-status lines | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_parse_filters_err -x` | ❌ Wave 0 |
| IAM-01 | `parse_iam_lines()` extracts writer_id, form_id, transcription correctly | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_parse_extracts_fields -x` | ❌ Wave 0 |
| IAM-01 | `parse_iam_lines()` replaces pipe separators with spaces | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_parse_replaces_pipes -x` | ❌ Wave 0 |
| IAM-01 | `parse_iam_lines()` filters to partition when partition_forms provided | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_parse_filters_partition -x` | ❌ Wave 0 |
| IAM-01 | `ingest_iam()` inserts samples with `category="iam"` and `student="iam-writer-XXX"` | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_sets_category_and_student -x` | ❌ Wave 0 |
| IAM-01 | `ingest_iam()` inserts ground truth from ascii transcription | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_inserts_ground_truth -x` | ❌ Wave 0 |
| IAM-01 | `ingest_iam()` deduplicates on reimport | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_ingest_iam_dedup -x` | ❌ Wave 0 |
| IAM-01 | `benchmark ingest-iam` CLI command exists and runs | unit | `pytest tests/test_benchmark_ingest.py::TestIAMIngest::test_cli_ingest_iam_command -x` | ❌ Wave 0 |
| IAM-02 | `run_benchmark()` accepts `line_level` param and passes to `_read_single()` | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_benchmark_accepts_line_level -x` | ❌ Wave 0 |
| IAM-02 | `run_benchmark()` accepts `auto_retry` param and passes to `_read_single()` | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_benchmark_accepts_auto_retry -x` | ❌ Wave 0 |
| IAM-02 | `run_sweep()` returns 5 distinct run_ids, one per strategy | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_run_sweep_returns_five_run_ids -x` | ❌ Wave 0 |
| IAM-02 | `benchmark sweep` CLI command exists and shows cost before running | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_sweep_cli_shows_cost -x` | ❌ Wave 0 |
| IAM-02 | `benchmark sweep --yes` bypasses confirmation and executes all 5 strategies | unit | `pytest tests/test_benchmark_evaluate.py::TestSweep::test_sweep_cli_yes_executes -x` | ❌ Wave 0 |
| IAM-03 | `generate_per_writer_report()` groups by student and shows mean CER per writer | unit | `pytest tests/test_benchmark_evaluate.py::TestPerWriterReport::test_per_writer_report_groups_by_student -x` | ❌ Wave 0 |
| IAM-03 | `generate_per_writer_report()` returns "no writer data" message when student is empty | unit | `pytest tests/test_benchmark_evaluate.py::TestPerWriterReport::test_per_writer_report_no_writers -x` | ❌ Wave 0 |
| IAM-03 | `benchmark report --per-writer` invokes per-writer report | unit | `pytest tests/test_benchmark_evaluate.py::TestPerWriterReport::test_report_cli_per_writer_flag -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_benchmark_ingest.py tests/test_benchmark_evaluate.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_benchmark_ingest.py::TestIAMIngest` — class with all IAM-01 test stubs (8 tests)
- [ ] `tests/test_benchmark_evaluate.py::TestSweep` — class with all IAM-02 test stubs (5 tests)
- [ ] `tests/test_benchmark_evaluate.py::TestPerWriterReport` — class with all IAM-03 test stubs (3 tests)
- [ ] No new framework installs needed — pytest already installed

All 16 tests must be written as RED stubs (failing with `pytest.fail("not implemented")` or `assert False`) before any implementation work begins. This follows the Nyquist compliance pattern established in Phase 6.

---

## Sources

### Primary (HIGH confidence)
- Codebase read: `handwriting_engine/benchmark/db.py` — schema v4, `insert_sample()` signature, `student` + `category` columns confirmed
- Codebase read: `handwriting_engine/benchmark/ingest.py` — `ingest_directory()`, `hash_file()`, `insert_ground_truth()` patterns
- Codebase read: `handwriting_engine/benchmark/evaluate.py` — `run_benchmark()` full signature, `_read_single()` parameters
- Codebase read: `handwriting_engine/benchmark/report.py` — `generate_report()`, `_format_table()`, existing SQL patterns
- Codebase read: `handwriting_engine/cli.py` — all existing benchmark commands, cost guardrail implementation
- Codebase read: `handwriting_engine/vision.py` — `read_page()` `line_level` and `auto_retry` parameters
- Codebase read: `tests/test_benchmark_ingest.py` — existing test patterns
- Codebase read: `tests/test_benchmark_evaluate.py` — existing test patterns, mock conventions

### Secondary (MEDIUM confidence)
- [Keras Handwriting Recognition Tutorial](https://keras.io/examples/vision/handwriting_recognition/) — IAM words.txt column format confirmed: `line_id status graylevel num_components x y w h ... transcription`; ID hierarchy `a01-000u-00` → writer=a01, form=a01-000u, confirmed
- [Laia/egs/iam README](https://github.com/jpuigcerver/Laia/blob/master/egs/iam/README.md) — Aachen partition confirmed: test=2915 lines / 336 forms, val=966/115, train=6161/747
- [Teklia/IAM-line HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line) — test split 2920 lines confirmed (consistent with Aachen), 3 splits total

### Tertiary (LOW confidence)
- WebSearch: IAM lines.txt `|` pipe separator for word boundaries — referenced in multiple community parsers; not directly verified against official docs. Treat as likely-correct but add a defensive `replace("|", " ")` regardless.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries already in pyproject.toml
- Schema analysis: HIGH — db.py read directly; `student` + `category` columns exist at v4
- IAM format (lines.txt columns, ID hierarchy): HIGH — confirmed via Keras docs + Laia README independently
- IAM partition counts: HIGH — confirmed via Laia README + HuggingFace dataset independently
- Pipe separator in transcription: MEDIUM — referenced in community parsers, not in official docs
- `line_level`/`auto_retry` threading gap: HIGH — confirmed by reading `run_benchmark()` signature directly
- Strategy name-to-parameter mapping: MEDIUM — derived from reading codebase; `prompt_adapted` interpretation requires developer confirmation

**Research date:** 2026-04-11
**Valid until:** 2026-05-11 (stable codebase; IAM format unchanged for 20+ years)
