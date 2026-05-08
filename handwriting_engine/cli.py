"""
Handwriting Engine CLI — convert, assess, enhance, read, crop handwritten documents.

This module lives inside the package so ``pip install`` can find the entry point.
The root ``main.py`` imports from here for backwards compatibility.
"""

import atexit
import shutil
import click
import json
import sys
from pathlib import Path

# Track temp dirs for cleanup
_temp_dirs: list[str] = []


def _cleanup_temp():
    for d in _temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Handwriting Engine — unified handwriting recognition with multi-model consensus."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None, help="Output directory (default: temp)")
@click.option("--dpi", default=150, type=click.IntRange(72, 600), help="Rendering DPI (72-600)")
@click.option("--enhance", is_flag=True, help="Auto-enhance images after conversion")
def convert(pdf_path, output_dir, dpi, enhance):
    """Convert PDF to JPEG images."""
    import tempfile
    from handwriting_engine.pdf import convert_pdf

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hwe_")
        _temp_dirs.append(output_dir)

    pages = convert_pdf(pdf_path, output_dir, dpi=dpi, auto_enhance=enhance)
    click.echo(f"Converted {len(pages)} pages to {output_dir}")
    for p in pages:
        click.echo(f"  Page {p['page_number']}: {p['path']}")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def assess(path):
    """Assess image quality (single file or directory)."""
    import os
    from handwriting_engine.quality import assess_image, batch_assess

    if os.path.isdir(path):
        results = batch_assess(path)
        for img_path, report in results.items():
            q = report["quality"]
            icon = {"good": "+", "fair": "~", "poor": "!"}[q]
            click.echo(f"  [{icon}] {os.path.basename(img_path)}: {q} (blur={report['blur_score']}, contrast={report['contrast_score']:.2f})")
            if report["issues"]:
                click.echo(f"      Issues: {', '.join(report['issues'])}")
    else:
        report = assess_image(path)
        click.echo(json.dumps(report, indent=2))


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--strategy", "-s", default="proven", type=click.Choice(["proven", "smart", "llm", "light", "medium", "heavy", "full"]))
@click.option("--output", "-o", default=None, help="Output path")
def enhance(path, strategy, output):
    """Enhance image for better handwriting recognition."""
    from handwriting_engine.enhance import enhance_image

    result = enhance_image(path, strategy=strategy, output_path=output)
    click.echo(f"Enhanced: {result}")


_ALT_MARKER_RE = __import__("re").compile(r"\[\?alt:\s*([^\]]+)\]")


def _extract_alt_markers(text: str) -> list[dict]:
    """Pull `[?alt: a/b]` markers out of consensus text into a structured list.

    Skill-side `--strict` mode uses these to prompt the user once per ambiguity.
    """
    markers = []
    for match in _ALT_MARKER_RE.finditer(text):
        alts = [a.strip() for a in match.group(1).split("/") if a.strip()]
        markers.append({"raw": match.group(0), "alternatives": alts})
    return markers


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--provider", "-p", default="claude", type=click.Choice(["claude", "openai", "gemini", "consensus"]))
@click.option("--domain", "-d", default="general",
              help="Domain hint (general, biology, ...). Default 'general' so skill callers pass --domain=bio explicitly.")
@click.option("--prompt", default="", help="Custom reading prompt")
@click.option("--writer", default=None,
              help="Writer ID for WriterProfileStore lookup (per-writer few-shot/calibration)")
@click.option("--format", "fmt", default="txt",
              type=click.Choice(["txt", "md", "json"]),
              help="Output shape. 'json' emits a structured payload with extracted alt-markers — used by the handwriting-reader skill.")
def read(path, provider, domain, prompt, writer, fmt):
    """Read handwritten text from image or PDF."""
    if path.lower().endswith(".pdf"):
        import tempfile
        from handwriting_engine.pdf import convert_pdf
        tmpdir = tempfile.mkdtemp(prefix="hwe_read_")
        _temp_dirs.append(tmpdir)
        pages = convert_pdf(path, tmpdir)
        image_paths = [(p.get("page_number", i + 1), p["path"]) for i, p in enumerate(pages)]
    else:
        image_paths = [(1, path)]

    page_payloads = []
    for page_no, img in image_paths:
        if provider == "consensus":
            from handwriting_engine import vision as _vision
            result = _vision.read_with_consensus(img, prompt=prompt, domain=domain, writer_id=writer)
            page_payloads.append({
                "page_number": page_no,
                "image_path": img,
                "text": result.text,
                "provider": "consensus",
                "confidence": result.confidence,
                "confidence_level": result.confidence_level,
                "strategy_used": result.strategy_used,
                "disagreements": list(result.disagreements),
                "alt_markers": _extract_alt_markers(result.text),
                "provider_results": dict(result.provider_results),
                "tokens_used": dict(result.tokens_used),
            })
        else:
            from handwriting_engine import vision as _vision
            text = _vision.read_page(img, prompt=prompt, domain=domain, provider=provider)
            page_payloads.append({
                "page_number": page_no,
                "image_path": img,
                "text": text,
                "provider": provider,
                "confidence": None,
                "confidence_level": None,
                "strategy_used": None,
                "disagreements": [],
                "alt_markers": _extract_alt_markers(text),
                "provider_results": {provider: text},
                "tokens_used": {},
            })

    if fmt == "json":
        click.echo(json.dumps({
            "path": path,
            "domain": domain,
            "writer": writer,
            "pages": page_payloads,
        }, indent=2))
        return

    if fmt == "md":
        for p in page_payloads:
            header = f"## Page {p['page_number']}"
            if p["confidence"] is not None:
                header += f" — confidence {p['confidence']:.2f} ({p['confidence_level']})"
            click.echo(header)
            click.echo("")
            click.echo(p["text"])
            click.echo("")
            if p["disagreements"]:
                click.echo(f"_Disagreements: {p['disagreements']}_\n")
        return

    # Default: txt — preserve legacy echo behavior
    for p in page_payloads:
        if p["provider"] == "consensus":
            click.echo(f"--- Confidence: {p['confidence']:.2f} ({p['strategy_used']}) ---")
            click.echo(p["text"])
            if p["disagreements"]:
                click.echo(f"\nDisagreements: {p['disagreements']}")
        else:
            click.echo(p["text"])


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--page", default=0, type=click.IntRange(0, 999), help="Page number (0-indexed)")
@click.option("--questions", default=25, type=click.IntRange(1, 200), help="Number of questions (1-200)")
@click.option("--output-dir", "-o", default=None)
def crop(pdf_path, page, questions, output_dir):
    """Crop individual answer regions from answer sheet PDF."""
    from handwriting_engine.crop import crop_answer_sheet

    crops = crop_answer_sheet(pdf_path, page_number=page, num_questions=questions, output_dir=output_dir)
    click.echo(f"Cropped {len(crops)} answer regions")
    for q, path in sorted(crops.items()):
        click.echo(f"  Q{q}: {path}")


@cli.command("batch")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--enhance", "do_enhance", is_flag=True, help="Enhance all images")
@click.option("--strategy", "-s", default="proven")
@click.option("--read", "do_read", is_flag=True, help="Read all images after enhancing")
@click.option("--provider", "-p", default="claude")
def batch_cmd(directory, do_enhance, strategy, do_read, provider):
    """Batch process a directory of images."""
    import os
    from handwriting_engine.quality import batch_assess

    results = batch_assess(directory)
    click.echo(f"Assessed {len(results)} images")

    if do_enhance:
        from handwriting_engine.enhance import enhance_image
        for img_path, report in results.items():
            if report["quality"] != "good":
                enhance_image(img_path, strategy=strategy)
                click.echo(f"  Enhanced: {os.path.basename(img_path)}")

    if do_read:
        from handwriting_engine.vision import read_page
        for img_path in results:
            text = read_page(img_path, provider=provider)
            click.echo(f"\n--- {os.path.basename(img_path)} ---")
            click.echo(text[:200] + "..." if len(text) > 200 else text)


@cli.group()
def benchmark():
    """Benchmark and regression-test handwriting recognition accuracy."""
    pass


@benchmark.command("ingest")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--student", "-s", default="", help="Student name metadata")
@click.option("--category", "-c", default="", help="Category (e.g. biology)")
def benchmark_ingest(directory, student, category):
    """Import images from a directory into the benchmark database.

    Deduplicates by file content hash. Auto-assesses image quality.
    """
    from handwriting_engine.benchmark.ingest import ingest_directory

    samples = ingest_directory(directory, student=student, category=category)
    click.echo(f"Imported {len(samples)} new samples")
    for s in samples:
        click.echo(f"  [{s.id}] {s.image_path} (page {s.page_number})")


@benchmark.command("ingest-iam")
@click.argument("ascii_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--lines-dir", default=None, type=click.Path(file_okay=False),
    help="Path to lines/ image directory (default: sibling of ascii/ directory)"
)
@click.option(
    "--partition-file", default=None, type=click.Path(exists=True, dir_okay=False),
    help="Text file listing form IDs to ingest (e.g. testset.txt from Aachen split)"
)
@click.option(
    "--all-partitions", is_flag=True, default=False,
    help="Ingest ALL lines regardless of partition (includes training data — use with caution)"
)
@click.option("--db-path", default=None, hidden=True)
def benchmark_ingest_iam(ascii_dir, lines_dir, partition_file, all_partitions, db_path):
    """Ingest IAM Handwriting Database line images into benchmark DB.

    ASCII_DIR: path to the extracted ascii/ directory from IAM.
    Line images are expected in a sibling lines/ directory, or use --lines-dir.

    PARTITION SAFETY: Pass --partition-file testset.txt to ingest only the test split.
    If neither --partition-file nor --all-partitions is given, this command will abort
    to prevent accidental ingestion of training data.
    """
    if partition_file is None and not all_partitions:
        click.echo(
            "ERROR: Partition safety guard triggered.\n"
            "  Provide --partition-file <testset.txt> to ingest only the test split, OR\n"
            "  pass --all-partitions to ingest all lines (INCLUDES training data).\n"
            "  Ingesting training data contaminates the benchmark — use with caution.",
            err=True,
        )
        raise SystemExit(1)

    if all_partitions and partition_file is None:
        click.echo(
            "WARNING: Ingesting ALL partitions (including training data). "
            "This may contaminate benchmark results.",
            err=True,
        )

    from handwriting_engine.benchmark.ingest import ingest_iam

    try:
        result = ingest_iam(
            ascii_dir=ascii_dir,
            lines_dir=lines_dir,
            partition_file=partition_file,
            db_path=db_path,
        )
        click.echo(
            f"IAM ingest complete: "
            f"{result['ingested']} ingested, "
            f"{result['skipped_dup']} duplicates skipped, "
            f"{result['skipped_missing']} images missing."
        )
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@benchmark.command("ingest-lab")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--student", "-s", default="lab", help="Student / author tag for these images")
@click.option("--with-suggestion", is_flag=True, default=False,
              help="Pre-fill each prompt with a single VLM read (costs ~$/image).")
@click.option("--vlm-provider", default="gemini", show_default=True,
              help="Provider for VLM suggestion (when --with-suggestion).")
@click.option("--gt-suffix", default=None,
              help="Non-interactive mode: read ground truth from a sibling file "
                   "with this suffix (e.g. '.txt' pairs page1.png with page1.txt). "
                   "Skips $EDITOR entirely — required for scripted workflows.")
@click.option("--db-path", default=None, hidden=True)
def benchmark_ingest_lab(directory, student, with_suggestion, vlm_provider, gt_suffix, db_path):
    """Guided annotation: capture ground-truth transcriptions for lab notebook images.

    For each image in DIRECTORY, opens $EDITOR (or prompts inline) with an
    optional VLM-suggested transcription. Save to record as ground truth;
    leave empty / cancel to skip an image. Re-running on the same directory
    skips images that already have ground truth — safe to resume.

    Pass --gt-suffix .txt to skip the editor and pull ground truth from a
    sibling file (page1.png + page1.txt). Empty or missing sibling files are
    treated as user-skip — same disposition as cancelling the editor.
    """
    from handwriting_engine.benchmark.ingest import ingest_lab

    if gt_suffix is not None:
        # Normalize suffix so callers can pass "txt" or ".txt" interchangeably.
        suffix = gt_suffix if gt_suffix.startswith(".") else f".{gt_suffix}"

        def _prompt(image_path: str, _suggestion: str) -> str | None:
            sibling = Path(image_path).with_suffix(suffix)
            if not sibling.exists():
                click.echo(f"  skip: no sibling {sibling.name} for {Path(image_path).name}")
                return None
            try:
                text = sibling.read_text(encoding="utf-8").strip()
            except Exception as exc:
                click.echo(f"  skip: could not read {sibling.name}: {exc}", err=True)
                return None
            if not text:
                click.echo(f"  skip: {sibling.name} is empty")
                return None
            return text
    else:
        def _prompt(image_path: str, suggestion: str) -> str | None:
            click.echo(f"\n--- {image_path} ---")
            if suggestion:
                click.echo(f"VLM suggestion: {suggestion}")
            # click.edit returns None if EDITOR exits without saving (skip);
            # otherwise it returns the buffer with trailing whitespace.
            edited = click.edit(suggestion or "")
            if edited is None:
                return None
            return edited.strip()

    counts = ingest_lab(
        directory,
        student=student,
        prompt_fn=_prompt,
        db_path=db_path,
        use_vlm_suggestion=with_suggestion,
        vlm_provider=vlm_provider,
    )
    click.echo(
        "\nLab ingest complete: "
        f"{counts['annotated']} annotated, "
        f"{counts['newly_added']} new samples, "
        f"{counts['skipped_existing_gt']} already had ground truth, "
        f"{counts['skipped_user']} skipped, "
        f"{counts['errors']} errors."
    )


@benchmark.command("transcribe")
@click.argument("sample_id", type=int)
@click.option("--text", "-t", default=None, help="Ground truth transcription text")
@click.option("--text-file", "-f", type=click.Path(exists=True), default=None, help="Read ground truth from file")
@click.option("--source", default="manual", help="How this GT was obtained")
def benchmark_transcribe(sample_id, text, text_file, source):
    """Add or update ground-truth transcription for a sample."""
    from handwriting_engine.benchmark.db import get_connection, get_sample_by_id, insert_ground_truth

    if text_file:
        with open(text_file, encoding="utf-8") as f:
            text = f.read().strip()
    if not text:
        click.echo("Provide --text or --text-file", err=True)
        sys.exit(1)

    conn = get_connection()
    try:
        sample = get_sample_by_id(conn, sample_id)
        if not sample:
            click.echo(f"Sample {sample_id} not found.", err=True)
            sys.exit(1)
        gt_id = insert_ground_truth(conn, sample_id, text, source=source)
    finally:
        conn.close()
    click.echo(f"Ground truth #{gt_id} saved for sample {sample_id}")


@benchmark.command("list")
@click.option("--samples", "show_samples", is_flag=True, help="List all samples")
@click.option("--runs", "show_runs", is_flag=True, help="List all runs")
def benchmark_list(show_samples, show_runs):
    """List benchmark samples or runs."""
    from handwriting_engine.benchmark.db import get_connection, list_samples, list_runs
    import os

    conn = get_connection()

    if show_samples or not show_runs:
        samples = list_samples(conn)
        click.echo(f"Samples ({len(samples)}):")
        for s in samples:
            gt = "[GT]" if s.has_ground_truth else "    "
            name = os.path.basename(s.image_path)
            student = f" ({s.student})" if s.student else ""
            click.echo(f"  {gt} [{s.id}] {name}{student} pg{s.page_number}")

    if show_runs:
        runs = list_runs(conn)
        click.echo(f"\nRuns ({len(runs)}):")
        for r in runs:
            label = f' "{r.label}"' if r.label else ""
            click.echo(f"  [{r.run_id}]{label} {r.started_at} — {r.sample_count} samples")

    conn.close()


def _get_avg_tokens_per_read(conn) -> tuple:
    """Estimate average input/output tokens per read from the most recent run.

    Falls back to conservative defaults (2000 input, 500 output) if no prior runs exist.
    """
    try:
        latest_row = conn.execute(
            "SELECT id FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if latest_row is None:
            return (2000.0, 500.0)
        latest_run_id = latest_row["id"]
        avg_row = conn.execute(
            """SELECT AVG(input_tokens) as avg_in, AVG(output_tokens) as avg_out
               FROM provider_outputs WHERE run_id = ?""",
            (latest_run_id,)
        ).fetchone()
        avg_in = avg_row["avg_in"] or 2000.0
        avg_out = avg_row["avg_out"] or 500.0
        return (avg_in, avg_out)
    except Exception:
        return (2000.0, 500.0)


@benchmark.command("run")
@click.option("--label", "-l", default="", help="Label for this run")
@click.option("--providers", "-p", default=None, help="Comma-separated providers (e.g. gemini,claude)")
@click.option("--strategies", "-s", default=None, help="Comma-separated consensus strategies (e.g. vote,best_of)")
@click.option("--domain", "-d", default="biology")
@click.option("--feed-lessons", is_flag=True, help="Feed errors back to lessons system")
@click.option("--smoke", is_flag=True, help="Smoke test: 3 hardest samples, cheapest provider")
@click.option("--enhance", is_flag=True, help="Apply smart enhancement before reading (matches production)")
@click.option("--inject-lessons", is_flag=True, help="Inject lessons into prompts (matches production)")
@click.option("--compare-strategies", default=None, help="Run multiple strategies and print CER comparison table (e.g. vote,best_of,self_correct)")
@click.option("--preprocessing", default=None, help="Apply a named enhance strategy before reading (e.g. sauvola, proven, clahe)")
@click.option("--yes", "-y", is_flag=True, help="Skip cost confirmation prompt (CI-friendly)")
@click.option("--iam-partition", default=None, help="IAM partition label for provenance (e.g. 'test2023')")
@click.option("--vocab-hints-off", is_flag=True, help="Record that vocabulary hints were disabled for this run")
@click.option("--db-path", default=None, hidden=True, help="Override DB path (for testing)")
def benchmark_run_cmd(label, providers, strategies, domain, feed_lessons, smoke, enhance, inject_lessons, compare_strategies, preprocessing, yes, iam_partition, vocab_hints_off, db_path):
    """Run all providers/strategies against samples with ground truth.

    Only samples that have ground-truth transcriptions are evaluated.
    Skips providers whose SDK is not installed.
    """
    from handwriting_engine.benchmark.evaluate import run_benchmark, compare_strategies as run_compare, estimate_cost, _available_providers
    from handwriting_engine.benchmark.db import get_connection as _get_conn, samples_with_ground_truth
    from handwriting_engine.benchmark.report import generate_report
    from handwriting_engine.benchmark.lessons_bridge import feed_errors_to_lessons

    if compare_strategies:
        strat_list = [s.strip() for s in compare_strategies.split(",")]
        click.echo(run_compare(strat_list, domain=domain))
        return

    prov_list = [p.strip() for p in providers.split(",")] if providers else None
    strat_list = [s.strip() for s in strategies.split(",")] if strategies else None
    mode = "smoke" if smoke else "full"

    if preprocessing and not enhance:
        enhance = True

    # --- Cost projection guardrail (FOUND-04) ---
    # Always shown before any benchmark run. Use --yes to bypass in scripts/CI.
    _conn = _get_conn(db_path)
    _all_samples = samples_with_ground_truth(_conn)
    _n_samples = len(_all_samples)
    _avg_in, _avg_out = _get_avg_tokens_per_read(_conn)
    _conn.close()

    _prov_list_for_cost = [p.strip() for p in providers.split(",")] if providers else _available_providers()
    _strat_list_for_cost = [s.strip() for s in strategies.split(",")] if strategies else []
    _n_prov = len(_prov_list_for_cost)
    _n_strat = max(1, len(_strat_list_for_cost))

    # Sum cost across providers (user pays for all, not an average)
    _total_cost = sum(
        estimate_cost(int(_avg_in * _n_strat * _n_samples),
                      int(_avg_out * _n_strat * _n_samples),
                      p)
        for p in _prov_list_for_cost
    )

    click.echo(f"Estimated cost: ${_total_cost:.3f}")
    click.echo(f"  {_n_prov} provider{'s' if _n_prov != 1 else ''} x {_n_strat} strateg{'ies' if _n_strat != 1 else 'y'} x {_n_samples} samples")
    click.echo("")
    if not yes:
        if not click.confirm("Proceed?", default=False):
            sys.exit(0)
    # --- End cost projection ---

    def progress(current, total, msg):
        click.echo(f"  [{current}/{total}] {msg}")

    try:
        run_id = run_benchmark(
            label=label, providers=prov_list, strategies=strat_list, domain=domain,
            on_progress=progress, mode=mode,
            auto_enhance=enhance, inject_lessons=inject_lessons,
            enhance_strategy=preprocessing,
            iam_partition=iam_partition,
            vocab_hints_off=int(vocab_hints_off),
            db_path=db_path,
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(generate_report(run_id, db_path=db_path))

    if feed_lessons:
        count = feed_errors_to_lessons(run_id)
        click.echo(f"\nFed {count} lessons back to the lessons system.")


@benchmark.command("sweep")
@click.option("--provider", "-p", default="gemini",
              help="Provider used for all cloud strategies (default: gemini). "
                   "Local strategies (e.g. trocr_local) ignore this flag.")
@click.option("--category", default="iam", show_default=True,
              help="Sample category to sweep (e.g. 'iam', 'lab'). "
                   "Pass '' (empty string) to sweep every category in the DB.")
@click.option("--yes", "-y", is_flag=True,
              help="Bypass cost confirmation")
@click.option("--db-path", default=None, hidden=True)
def benchmark_sweep(provider, category, yes, db_path):
    """Run every sweep strategy against samples in a category, one run_id per strategy.

    Cloud strategies (cost API tokens, run on --provider):
      baseline, self_correct, line_level, prompt_adapted, zoomed_verify
    Local strategies (zero API tokens, run on the named provider):
      trocr_local (requires `pip install -e '.[trained-correction]'`)

    Defaults to category='iam' for backward compat with the IAM rollout.
    Pass --category lab to sweep ingested lab notebook pages, or --category ''
    to sweep every category in the DB. Requires matching samples in the DB
    with ground truth — run `benchmark ingest-iam` or `benchmark ingest-lab`
    first. Prints projected cost before any API call. Strategies whose
    provider isn't installed are skipped with a warning rather than failing
    the sweep.
    """
    from handwriting_engine.benchmark.evaluate import (
        run_sweep, SWEEP_STRATEGIES, estimate_cost,
    )
    from handwriting_engine.benchmark.db import get_connection as _get_conn

    # Empty string -> sweep all categories (None signals run_sweep to skip the WHERE filter).
    category_filter: str | None = category if category else None

    # Count samples with ground truth (filtered by category when set)
    conn = _get_conn(db_path)
    try:
        if category_filter is None:
            row = conn.execute(
                "SELECT COUNT(DISTINCT s.id) AS n FROM samples s "
                "JOIN ground_truths gt ON gt.sample_id = s.id"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(DISTINCT s.id) AS n FROM samples s "
                "JOIN ground_truths gt ON gt.sample_id = s.id "
                "WHERE s.category = ?",
                (category_filter,),
            ).fetchone()
        n_samples = row["n"] if row else 0
    finally:
        conn.close()

    # Cost projection: ~2000 input + ~200 output tokens per sample per strategy.
    # Three cost classes:
    #   cloud  — full API call per sample, charged at --provider's rate
    #   local  — zero API tokens (e.g. trocr_local)
    #   mixed  — local-first with cloud fallback; assume worst-case 100%
    #            escalation rate so the user never gets a surprise bill.
    #            cost_provider can pin a specific cloud model (claude vs gemini).
    def _cost_class(s):
        # Backward-compat: strategies without an explicit cost_class fall back
        # to the legacy rule "providers field present == local".
        if "cost_class" in s:
            return s["cost_class"]
        return "local" if s.get("providers") else "cloud"

    cloud_strategies = [s for s in SWEEP_STRATEGIES if _cost_class(s) == "cloud"]
    local_strategies = [s for s in SWEEP_STRATEGIES if _cost_class(s) == "local"]
    mixed_strategies = [s for s in SWEEP_STRATEGIES if _cost_class(s) == "mixed"]

    est_per_cloud = estimate_cost(2000 * n_samples, 200 * n_samples, provider)
    est_total = est_per_cloud * len(cloud_strategies)
    for s in mixed_strategies:
        # Worst case: every sample escalates. Charged at the strategy's pinned
        # provider when set (some local_first variants use claude, not gemini).
        mixed_provider = s.get("cost_provider") or provider
        est_total += estimate_cost(2000 * n_samples, 200 * n_samples, mixed_provider)

    samples_label = f"category={category}" if category_filter else "all categories"
    click.echo(
        f"\nSweep projection: {len(SWEEP_STRATEGIES)} strategies "
        f"x {n_samples} samples ({samples_label}, provider={provider})"
    )
    click.echo(
        f"  Estimated cost (cloud): ~${est_per_cloud:.4f}/strategy "
        f"x {len(cloud_strategies)} = ~${est_per_cloud * len(cloud_strategies):.4f}"
    )
    if mixed_strategies:
        mixed_added = est_total - est_per_cloud * len(cloud_strategies)
        click.echo(
            f"  Estimated cost (mixed, local-first + cloud fallback at "
            f"worst-case 100% escalation): ~${mixed_added:.4f} for "
            f"{', '.join(s['name'] for s in mixed_strategies)}"
        )
    if local_strategies:
        click.echo(
            f"  Local (zero token cost): "
            f"{', '.join(s['name'] for s in local_strategies)}"
        )
    click.echo(
        f"  Strategies: {', '.join(s['name'] for s in SWEEP_STRATEGIES)}\n"
    )

    if n_samples == 0:
        if category_filter == "iam":
            hint = "Run `benchmark ingest-iam` first."
        elif category_filter == "lab":
            hint = "Run `benchmark ingest-lab` first."
        elif category_filter:
            hint = f"No samples ingested with category='{category_filter}'."
        else:
            hint = "No samples with ground truth in DB. Run an ingest command first."
        click.echo(
            f"WARNING: No samples with ground truth match {samples_label}. {hint}",
            err=True,
        )

    if not yes:
        click.confirm(
            f"Proceed with sweep (~${est_total:.4f} projected)?",
            abort=True,
        )

    try:
        run_ids = run_sweep(
            provider=provider, db_path=db_path, yes=yes, category=category_filter,
        )
    except Exception as exc:
        click.echo(f"ERROR during sweep: {exc}", err=True)
        sys.exit(1)

    click.echo("\nSweep complete:")
    for name, run_id in run_ids.items():
        click.echo(f"  {name:20s}: run_id={run_id}")


@benchmark.command("tune-threshold")
@click.option("--cloud-provider", default="gemini", show_default=True,
              help="Provider name to use as the local-first escalation target.")
@click.option("--writer", default=None,
              help="Filter to a single writer (matches samples.student).")
@click.option("--db-path", default=None, type=click.Path(),
              help="Override database path (default: ~/.handwriting-engine/benchmark.db).")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json"]), show_default=True)
def benchmark_tune_threshold_cmd(cloud_provider, writer, db_path, fmt):
    """Pareto sweep over local-first acceptance thresholds.

    Replays the local-first decision retrospectively against stored
    `provider='trocr'` and `provider=<cloud_provider>` outputs to plot
    CER vs cloud-call rate. Marks the Pareto frontier and recommends
    a knee point that minimizes ``cer + 0.5 * cloud_call_rate`` —
    a Lagrangian-ish tradeoff weighting one unit of cloud usage at
    half a unit of CER.

    Use the recommended threshold via ``HE_LOCAL_FIRST_THRESHOLD``
    or by passing ``threshold=`` to ``read_local_first``.
    """
    from dataclasses import asdict
    from handwriting_engine.threshold_tuner import (
        pareto_frontier, pick_knee, sweep_thresholds,
    )

    points = sweep_thresholds(
        db_path=db_path, cloud_provider=cloud_provider, writer_id=writer,
    )

    n_paired = points[0].n_samples if points else 0
    if n_paired < 10:
        click.echo(
            f"Not enough paired (trocr, {cloud_provider}) samples in DB "
            f"to tune threshold (found {n_paired}, need >=10). "
            "Run `benchmark sweep` against both providers and ensure "
            "ground truth is present, then retry."
        )
        return

    frontier_set = {id(p) for p in pareto_frontier(points)}
    knee = pick_knee([p for p in points if id(p) in frontier_set])

    if fmt == "json":
        payload = {
            "cloud_provider": cloud_provider,
            "writer": writer,
            "points": [
                {**asdict(p), "pareto": id(p) in frontier_set}
                for p in points
            ],
            "recommended_threshold": knee.threshold if knee else None,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    # Table format
    sorted_points = sorted(points, key=lambda p: p.threshold)
    click.echo(
        f"\nThreshold sweep (cloud_provider={cloud_provider}"
        f"{f', writer={writer}' if writer else ''}, n={n_paired}):\n"
    )
    click.echo(f"  {'thresh':>7}  {'CER %':>7}  {'cloud %':>8}  {'samples':>8}  pareto")
    click.echo(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}")
    for p in sorted_points:
        is_pareto = id(p) in frontier_set
        marker = "*" if is_pareto else " "
        click.echo(
            f"  {p.threshold:>7.2f}  "
            f"{p.cer * 100:>7.2f}  "
            f"{p.cloud_call_rate * 100:>8.2f}  "
            f"{p.n_samples:>8d}  "
            f"{marker}"
        )
    click.echo("\n  (* = Pareto-optimal: no other point beats it on both CER and cloud-call rate)")
    if knee:
        click.echo(
            f"\nRecommended: threshold={knee.threshold:.2f} "
            f"(knee of Pareto frontier; CER={knee.cer * 100:.2f}%, "
            f"cloud-call rate={knee.cloud_call_rate * 100:.2f}%)"
        )


@benchmark.command("report")
@click.option("--run-id", "-r", default=None, type=int, help="Specific run (default: latest)")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("--per-writer", is_flag=True, default=False,
              help="Show per-writer CER breakdown (requires IAM samples with student tags)")
@click.option("--db-path", default=None, hidden=True)
def benchmark_report_cmd(run_id, fmt, per_writer, db_path):
    """Show accuracy comparison table for a benchmark run."""
    if per_writer:
        from handwriting_engine.benchmark.report import generate_per_writer_report
        click.echo(generate_per_writer_report(run_id=run_id, db_path=db_path))
        return

    from handwriting_engine.benchmark.report import generate_report
    click.echo(generate_report(run_id, fmt=fmt))


@benchmark.command("compare")
@click.argument("run_id_1", type=int)
@click.argument("run_id_2", type=int)
def benchmark_compare_cmd(run_id_1, run_id_2):
    """Compare two benchmark runs side by side.

    Shows regressions, improvements, and unchanged metrics.
    """
    from handwriting_engine.benchmark.report import compare_runs

    click.echo(compare_runs(run_id_1, run_id_2))


@benchmark.command("recommend")
@click.option("--db-path", default=None, type=click.Path(),
              help="Override database path (default: ~/.handwriting-engine/benchmark.db).")
def benchmark_recommend_cmd(db_path):
    """Recommend the best (provider, strategy) configuration.

    Composite score: 70% CER, 15% cost, 15% stability across runs.
    Each component is min-max normalized within the candidate set.
    Single-run candidates get the median stability score (neutral).
    """
    from handwriting_engine.benchmark.report import recommend_strategy

    click.echo(recommend_strategy(db_path=db_path))


@benchmark.command("set-baseline")
@click.argument("run_id", type=int)
@click.option("--db-path", default=None, type=click.Path(),
              help="Override database path (default: ~/.handwriting-engine/benchmark.db).")
def benchmark_set_baseline_cmd(run_id, db_path):
    """Pin RUN_ID as the regression-detection anchor.

    `detect_regressions` and `benchmark report` then compare future runs
    against this pinned run rather than the immediately-preceding run.
    Exactly one run is the baseline at any time; this command atomically
    clears the prior pin and sets the new one.
    """
    from handwriting_engine.benchmark.db import get_connection, set_baseline

    conn = get_connection(db_path)
    try:
        set_baseline(conn, run_id)
    except ValueError as e:
        raise click.ClickException(str(e))
    finally:
        conn.close()
    click.echo(f"Baseline pinned: run #{run_id}")


@benchmark.command("drill-down")
@click.argument("sample_id", type=int)
@click.option("--run-id", "-r", default=None, type=int, help="Specific run (default: latest)")
def benchmark_drill_down_cmd(sample_id, run_id):
    """Show per-sample detail: all provider outputs, errors, and metrics."""
    from handwriting_engine.benchmark.report import sample_drill_down

    click.echo(sample_drill_down(sample_id, run_id=run_id))


@benchmark.command("quality")
@click.option("--run-id", "-r", default=None, type=int, help="Specific run (default: latest)")
def benchmark_quality_cmd(run_id):
    """Show correlation between image quality and recognition accuracy."""
    from handwriting_engine.benchmark.report import quality_correlation

    click.echo(quality_correlation(run_id=run_id))


@benchmark.command("calibrate")
@click.option("--samples", "-n", default=20, type=int, show_default=True,
              help="Number of random samples to use for calibration")
@click.option("--provider", "-p", default="gemini", show_default=True,
              help="Provider to use for calibration reads")
@click.option("--db-path", default=None, hidden=True, help="Override DB path (for testing)")
def benchmark_calibrate_cmd(samples, provider, db_path):
    """Measure CER variance and minimum detectable delta on N random samples.

    Runs N benchmark reads against randomly selected samples with ground truth.
    Prints the noise floor so you know whether a measured CER delta is real.

    Output: CER variance: ±0.42%  |  Min detectable delta: 0.84% (2\u03c3)
    """
    import random
    import statistics as _stats

    import handwriting_engine.benchmark.evaluate as _evaluate
    from handwriting_engine.benchmark.db import get_connection, samples_with_ground_truth, get_latest_ground_truth
    from handwriting_engine.benchmark.metrics import character_error_rate

    conn = get_connection(db_path)
    all_samples = samples_with_ground_truth(conn)
    conn.close()

    if not all_samples:
        click.echo("No samples with ground truth in DB. Ingest some samples first.", err=True)
        sys.exit(1)

    n = min(samples, len(all_samples))
    if n < samples:
        click.echo(f"Warning: only {n} samples available (requested {samples})")

    selected = random.sample(all_samples, n)

    cers = []
    for sample in selected:
        result = _evaluate._read_single(sample.image_path, provider, "biology", False, False, None)
        if result.get("error"):
            continue
        conn2 = get_connection(db_path)
        gt = get_latest_ground_truth(conn2, sample.id)
        conn2.close()
        if gt is None:
            continue
        cer, _, _ = character_error_rate(result["text"], gt.text)
        cers.append(cer)

    if not cers:
        click.echo("Not enough successful reads to compute variance (need at least 1).")
        sys.exit(0)

    sd = _stats.pstdev(cers)
    mdd = 2 * sd
    click.echo(f"CER variance: \u00b1{sd * 100:.2f}%  |  Min detectable delta: {mdd * 100:.2f}% (2\u03c3)")


@benchmark.command("calibration")
@click.option("--run-id", "-r", default=None, type=int, help="Specific run (default: latest)")
def benchmark_calibration_cmd(run_id):
    """Show how well confidence scores predict actual accuracy."""
    from handwriting_engine.benchmark.report import confidence_calibration

    click.echo(confidence_calibration(run_id=run_id))


@benchmark.command("degrade")
@click.argument("sample_id", type=int)
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Directory for degraded images")
def benchmark_degrade_cmd(sample_id, output_dir):
    """Generate synthetic degraded variants of a sample for data amplification.

    Creates blurred, low-contrast, rotated, noisy, and cropped variants
    that share the same ground truth as the original.
    """
    from handwriting_engine.benchmark.ingest import generate_degraded_variants

    variants = generate_degraded_variants(sample_id, output_dir)
    click.echo(f"Generated {len(variants)} variants")
    for v in variants:
        click.echo(f"  [{v.id}] {v.image_path}")


@benchmark.command("bootstrap-gt")
@click.option("--agreement", default=0.02, type=float, help="Max CER between providers for auto-GT")
@click.option("--confidence", default=0.85, type=float, help="Min confidence for auto-GT")
def benchmark_bootstrap_gt_cmd(agreement, confidence):
    """Auto-generate ground truth from high-agreement provider outputs.

    For samples where all providers agree within the CER threshold
    and confidence exceeds the minimum, auto-register as ground truth.
    """
    from handwriting_engine.benchmark.ingest import bootstrap_ground_truth

    count = bootstrap_ground_truth(
        agreement_threshold=agreement, confidence_threshold=confidence,
    )
    click.echo(f"Auto-generated {count} ground truths from consensus")


# =====================================================================
# Trained post-correction (optional — requires [trained-correction] extras)
# =====================================================================

@cli.group(name="trained-correction")
def trained_correction_group():
    """Train and evaluate the optional trained post-correction model.

    Requires: pip install handwriting-engine[trained-correction]
    """


@trained_correction_group.command(name="train")
@click.option("--output-dir", "-o", required=True, type=click.Path(),
              help="Directory for the trained checkpoint + manifest")
@click.option("--num-pairs", default=50000, type=int, help="Synthetic training pairs to generate")
@click.option("--num-epochs", default=2, type=int)
@click.option("--batch-size", default=8, type=int)
@click.option("--learning-rate", default=3e-4, type=float)
@click.option("--max-input-length", default=256, type=int)
@click.option("--max-target-length", default=256, type=int)
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--seed", default=42, type=int)
@click.option("--quick", is_flag=True, help="Tiny smoke run")
@click.option("--no-system-wordlist", is_flag=True)
@click.option("--model-name", default="google/flan-t5-small", show_default=True)
@click.option("--from-benchmark-db", is_flag=True,
              help="Mix real (VLM_output, ground_truth) pairs from the benchmark DB")
@click.option("--benchmark-db-path", default=None, type=click.Path(),
              help="Override the default benchmark.db path")
@click.option("--benchmark-providers", default=None,
              help="Comma-separated provider filter (e.g. 'gemini,claude')")
@click.option("--real-data-weight", default=3, type=int,
              help="Replication factor for real pairs (default 3)")
@click.option("--continue-from", default=None, type=click.Path(),
              help="Continue fine-tuning from an existing checkpoint")
def trained_correction_train(**kwargs):
    """Fine-tune the synthetic-data corrector. Long-running."""
    from handwriting_engine.trained_correction.train import main as train_main
    argv: list[str] = []
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif v is not None:
            argv.extend([flag, str(v)])
    sys.exit(train_main(argv))


@trained_correction_group.command(name="retrain-from-trocr")
@click.option("--db-path", default=None, type=click.Path(),
              help="Override benchmark DB path "
                   "(default: ~/.handwriting-engine/benchmark.db)")
@click.option("--v1-path", default=None, type=click.Path(),
              help="Synthetic v1 checkpoint to continue from "
                   "(default: ~/.handwriting-engine/models/trained-corrector-v1)")
@click.option("--v3-path", default=None, type=click.Path(),
              help="Output directory for the retrained v3 checkpoint "
                   "(default: ~/.handwriting-engine/models/trained-corrector-v3)")
@click.option("--num-epochs", default=3, type=int, show_default=True)
@click.option("--pairs-min", default=50, type=int, show_default=True,
              help="Minimum TrOCR pair count required to proceed")
@click.option("--no-eval", is_flag=True,
              help="Skip the post-training v1-vs-v3 CER comparison")
@click.option("--eval-n-pairs", default=200, type=int, show_default=True)
def trained_correction_retrain_from_trocr(
    db_path, v1_path, v3_path, num_epochs, pairs_min, no_eval, eval_n_pairs,
):
    """Retrain v1 -> v3 on real (TrOCR_output, ground_truth) pairs.

    Stage 3 of the local-first OCR rollout. Reads pairs from the benchmark
    DB (provider=trocr + has ground truth), continues fine-tuning from the
    v1 synthetic checkpoint, and writes a v3 checkpoint. Exits 1 with a
    remediation message if the DB doesn't have enough pairs yet — run
    `bench-my-pages.py` or `benchmark sweep` first to populate.
    """
    from handwriting_engine.trained_correction.retrain_from_trocr import (
        retrain_from_trocr,
    )
    rc = retrain_from_trocr(
        db_path=db_path,
        v1_path=v1_path,
        v3_path=v3_path,
        num_epochs=num_epochs,
        pairs_min=pairs_min,
        run_eval=not no_eval,
        eval_n_pairs=eval_n_pairs,
    )
    sys.exit(rc)


@trained_correction_group.command(name="eval")
@click.option("--n-pairs", default=1000, type=int)
@click.option("--seed", default=1234, type=int)
@click.option("--domain", default="biology")
@click.option("--skip-trained", is_flag=True, help="Heuristic-only baseline (no model load)")
@click.option("--output", default=None, type=click.Path(), help="Optional JSON output path")
def trained_correction_eval(n_pairs, seed, domain, skip_trained, output):
    """A/B evaluate post-correction pipelines on synthetic pairs."""
    from handwriting_engine.trained_correction.eval import main as eval_main
    argv = ["--n-pairs", str(n_pairs), "--seed", str(seed), "--domain", domain]
    if skip_trained:
        argv.append("--skip-trained")
    if output:
        argv.extend(["--output", output])
    sys.exit(eval_main(argv))


# =====================================================================
# Stage 4 — per-writer TrOCR fine-tune (local-first rollout)
# =====================================================================


@cli.command("fine-tune-writer")
@click.argument("writer_id")
@click.option("--epochs", default=3, type=click.IntRange(1, 50), show_default=True,
              help="Number of fine-tuning epochs (default 3 per arXiv 2305.02593).")
@click.option("--min-pairs", default=5, type=click.IntRange(1, 1000), show_default=True,
              help="Minimum (image, ground-truth) pairs required. Aborts otherwise.")
@click.option("--lr", default=5e-5, type=float, show_default=True,
              help="AdamW learning rate.")
@click.option("--db-path", default=None, hidden=True,
              help="Override the benchmark DB path (testing only).")
def fine_tune_writer_cmd(writer_id, epochs, min_pairs, lr, db_path):
    """Fine-tune TrOCR on every ground-truthed sample for WRITER_ID.

    Pulls (image, ground_truth) pairs where samples.student = WRITER_ID, then
    calls TrOCRProvider.fine_tune_for_writer. Saves a writer-specific
    checkpoint to ~/.handwriting-engine/writer-models/<WRITER_ID>/, which the
    next read of that writer auto-loads.

    Per arXiv 2305.02593, 5 lines per writer is "very effective" for
    single-writer HTR — that's the default --min-pairs floor.
    """
    from handwriting_engine.writer_profile_store import get_ground_truth_pairs_for_writer

    pairs = get_ground_truth_pairs_for_writer(writer_id, db_path=db_path)
    if len(pairs) < min_pairs:
        click.echo(
            f"ERROR: writer {writer_id!r} has {len(pairs)} ground-truthed pair"
            f"{'s' if len(pairs) != 1 else ''} in the benchmark DB; need at "
            f"least {min_pairs}. Capture more lines via "
            f"`benchmark ingest-lab` or `benchmark transcribe` first.",
            err=True,
        )
        sys.exit(1)

    # Lazy-import: keep base64/torch off the import path until fine-tune runs.
    import base64

    from handwriting_engine.providers import get_provider

    image_b64s: list[str] = []
    transcriptions: list[str] = []
    for image_path, gt_text in pairs:
        try:
            with open(image_path, "rb") as f:
                image_b64s.append(base64.standard_b64encode(f.read()).decode("utf-8"))
            transcriptions.append(gt_text)
        except (OSError, IOError) as exc:
            click.echo(f"  skipping unreadable image {image_path}: {exc}", err=True)

    if len(image_b64s) < min_pairs:
        click.echo(
            f"ERROR: only {len(image_b64s)} pairs had readable images on disk "
            f"(need {min_pairs}). Re-ingest or fix paths and retry.",
            err=True,
        )
        sys.exit(1)

    click.echo(
        f"Fine-tuning TrOCR for writer {writer_id!r} on {len(image_b64s)} pairs "
        f"(epochs={epochs}, lr={lr})..."
    )
    # Always start from the BASE checkpoint, never an existing writer model —
    # iterating on top of yesterday's checkpoint snowballs drift.
    provider = get_provider("trocr")
    save_dir = provider.fine_tune_for_writer(
        writer_id=writer_id,
        line_images_b64=image_b64s,
        transcriptions=transcriptions,
        epochs=epochs,
        lr=lr,
    )
    click.echo(f"Checkpoint saved: {save_dir}")
    click.echo(
        f"Next reads with `--writer {writer_id}` will auto-load this checkpoint."
    )


if __name__ == "__main__":
    cli()
