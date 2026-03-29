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


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--provider", "-p", default="claude", type=click.Choice(["claude", "openai", "gemini", "consensus"]))
@click.option("--domain", "-d", default="biology")
@click.option("--prompt", default="", help="Custom reading prompt")
def read(path, provider, domain, prompt):
    """Read handwritten text from image or PDF."""
    import os

    if path.lower().endswith(".pdf"):
        import tempfile
        from handwriting_engine.pdf import convert_pdf
        tmpdir = tempfile.mkdtemp(prefix="hwe_read_")
        _temp_dirs.append(tmpdir)
        pages = convert_pdf(path, tmpdir)
        image_paths = [p["path"] for p in pages]
    else:
        image_paths = [path]

    for img in image_paths:
        if provider == "consensus":
            from handwriting_engine.vision import read_with_consensus
            result = read_with_consensus(img, prompt=prompt, domain=domain)
            click.echo(f"--- Confidence: {result.confidence:.2f} ({result.strategy_used}) ---")
            click.echo(result.text)
            if result.disagreements:
                click.echo(f"\nDisagreements: {result.disagreements}")
        else:
            from handwriting_engine.vision import read_page
            text = read_page(img, prompt=prompt, domain=domain, provider=provider)
            click.echo(text)


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


@benchmark.command("run")
@click.option("--label", "-l", default="", help="Label for this run")
@click.option("--providers", "-p", default=None, help="Comma-separated providers (e.g. gemini,claude)")
@click.option("--strategies", "-s", default=None, help="Comma-separated consensus strategies (e.g. vote,best_of)")
@click.option("--domain", "-d", default="biology")
@click.option("--feed-lessons", is_flag=True, help="Feed errors back to lessons system")
@click.option("--smoke", is_flag=True, help="Smoke test: 3 hardest samples, cheapest provider")
@click.option("--enhance", is_flag=True, help="Apply smart enhancement before reading (matches production)")
@click.option("--inject-lessons", is_flag=True, help="Inject lessons into prompts (matches production)")
def benchmark_run_cmd(label, providers, strategies, domain, feed_lessons, smoke, enhance, inject_lessons):
    """Run all providers/strategies against samples with ground truth.

    Only samples that have ground-truth transcriptions are evaluated.
    Skips providers whose SDK is not installed.
    """
    from handwriting_engine.benchmark.evaluate import run_benchmark
    from handwriting_engine.benchmark.report import generate_report
    from handwriting_engine.benchmark.lessons_bridge import feed_errors_to_lessons

    prov_list = [p.strip() for p in providers.split(",")] if providers else None
    strat_list = [s.strip() for s in strategies.split(",")] if strategies else None
    mode = "smoke" if smoke else "full"

    def progress(current, total, msg):
        click.echo(f"  [{current}/{total}] {msg}")

    try:
        run_id = run_benchmark(
            label=label, providers=prov_list, strategies=strat_list, domain=domain,
            on_progress=progress, mode=mode,
            auto_enhance=enhance, inject_lessons=inject_lessons,
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(generate_report(run_id))

    if feed_lessons:
        count = feed_errors_to_lessons(run_id)
        click.echo(f"\nFed {count} lessons back to the lessons system.")


@benchmark.command("report")
@click.option("--run-id", "-r", default=None, type=int, help="Specific run (default: latest)")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
def benchmark_report_cmd(run_id, fmt):
    """Show accuracy comparison table for a benchmark run."""
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


if __name__ == "__main__":
    cli()
