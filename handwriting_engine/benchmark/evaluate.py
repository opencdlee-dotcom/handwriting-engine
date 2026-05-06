"""
Benchmark evaluation engine.

Runs providers and consensus strategies against samples with ground truth,
computes CER/WER, and stores everything in the database.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from handwriting_engine.benchmark.db import (
    finish_run,
    get_connection,
    get_latest_ground_truth,
    insert_eval_metric,
    insert_provider_output,
    insert_run,
    samples_with_ground_truth,
)
from handwriting_engine.benchmark.metrics import character_error_rate, word_error_rate

logger = logging.getLogger(__name__)

from handwriting_engine._constants import COST_PER_1M_TOKENS

# Fixed string describing the always-on normalization transformations in metrics.py
_NORM_FLAGS = "nfc,lowercase,strip_markers,collapse_ws"


def _resolve_model_version(providers: list[str]) -> str:
    """Build 'provider/model_string' label for provenance tracking."""
    from handwriting_engine import _constants as C
    _PROVIDER_MODELS = {
        "gemini": getattr(C, "DEFAULT_GEMINI_MODEL", "gemini-unknown"),
        "claude": getattr(C, "DEFAULT_CLAUDE_MODEL", "claude-unknown"),
        "openai": getattr(C, "DEFAULT_OPENAI_MODEL", "openai-unknown"),
    }
    return ",".join(
        f"{p}/{_PROVIDER_MODELS.get(p, p)}" for p in providers
    )


def _compute_marker_rate(text: str) -> float:
    """Compute the fraction of words that are [?] uncertainty markers.

    Computed from raw provider output BEFORE normalization so markers are not stripped.
    Returns 0.0 for empty text. Returns 1.0 if all tokens are markers.
    """
    if not text:
        return 0.0
    import re
    tokens = text.split()
    if not tokens:
        return 0.0
    marker_count = sum(1 for t in tokens if re.fullmatch(r"\[\?\]", t))
    return marker_count / len(tokens)


def estimate_cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """Estimate USD cost from token counts.

    For consensus providers like 'gemini+claude', averages costs across all
    providers in the combination.
    """
    providers = provider.split("+") if "+" in provider else [provider]
    total = 0.0
    for p in providers:
        rates = COST_PER_1M_TOKENS.get(p, {"input": 0, "output": 0})
        total += (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return total / len(providers) if len(providers) > 1 else total


def _available_providers() -> list[str]:
    """Get list of providers with installed SDKs and API keys."""
    try:
        from handwriting_engine.providers import available_providers
        return available_providers()
    except ImportError:
        return []


def _read_single(
    image_path: str, provider: str, domain: str,
    auto_enhance: bool = False, inject_lessons: bool = False,
    enhance_strategy: str | None = None,
    line_level: bool = False,
    auto_retry: bool = False,
) -> dict:
    """Read a single image with one provider. Returns result dict."""
    from handwriting_engine.vision import read_page

    # Optionally enhance before reading (matches production pipeline)
    # Always write to a temp file to avoid corrupting benchmark source images
    actual_path = image_path
    if auto_enhance or enhance_strategy:
        strategy = enhance_strategy or "smart"
        try:
            import tempfile
            from handwriting_engine.enhance import enhance_image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
            actual_path = enhance_image(image_path, strategy=strategy, output_path=tmp_path)
        except Exception:
            actual_path = image_path

    # Snapshot token usage BEFORE call to compute delta (providers are singletons)
    pre_input = pre_output = 0
    try:
        from handwriting_engine.providers import get_provider
        p = get_provider(provider)
        pre_usage = p.usage
        pre_input = pre_usage.get("input_tokens", 0)
        pre_output = pre_usage.get("output_tokens", 0)
    except Exception:
        pass

    start = time.monotonic()
    error = None
    text = ""
    confidence = 0.0

    try:
        text = read_page(
            actual_path, domain=domain, provider=provider,
            inject_lessons=inject_lessons,
            line_level=line_level,
            auto_retry=auto_retry,
        )
        penalty = text.count("[?]") * 0.05 + text.count("[illegible") * 0.1
        confidence = max(0.0, min(0.95, 1.0 - penalty))
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning("Provider %s failed on %s: %s", provider, image_path, error)

    latency_ms = int((time.monotonic() - start) * 1000)

    # Compute token DELTA (not cumulative total)
    input_tokens = output_tokens = 0
    try:
        from handwriting_engine.providers import get_provider
        p = get_provider(provider)
        post_usage = p.usage
        input_tokens = max(0, post_usage.get("input_tokens", 0) - pre_input)
        output_tokens = max(0, post_usage.get("output_tokens", 0) - pre_output)
    except Exception:
        pass

    return {
        "text": text,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }


def _read_consensus(
    image_path: str, providers: list[str], strategy: str, domain: str
) -> dict:
    """Read with a consensus strategy. Returns result dict."""
    from handwriting_engine.vision import read_with_consensus

    start = time.monotonic()
    error = None
    text = ""
    confidence = 0.0
    input_tokens = output_tokens = 0

    try:
        result = read_with_consensus(
            image_path,
            domain=domain,
            providers=providers,
            strategy=strategy,
        )
        text = result.text
        confidence = result.confidence
        # tokens_used may be flat {"input_tokens": N, "output_tokens": N}
        # or nested by provider {"gemini": {"input_tokens": N, ...}}
        tokens = result.tokens_used
        if tokens and isinstance(next(iter(tokens.values()), None), dict):
            # Nested: sum across all providers
            input_tokens = sum(v.get("input_tokens", 0) for v in tokens.values() if isinstance(v, dict))
            output_tokens = sum(v.get("output_tokens", 0) for v in tokens.values() if isinstance(v, dict))
        else:
            input_tokens = tokens.get("input_tokens", 0)
            output_tokens = tokens.get("output_tokens", 0)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning("Consensus %s failed on %s: %s", strategy, image_path, error)

    latency_ms = int((time.monotonic() - start) * 1000)

    return {
        "text": text,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": error,
    }


def run_benchmark(
    label: str = "",
    providers: list[str] | None = None,
    strategies: list[str] | None = None,
    domain: str = "biology",
    sample_ids: list[int] | None = None,
    db_path: Path | str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    mode: str = "full",
    auto_enhance: bool = False,
    inject_lessons: bool = False,
    enhance_strategy: str | None = None,
    iam_partition: str | None = None,
    vocabulary_hints: list[str] | None = None,
    vocab_hints_off: int = 0,
    line_level: bool = False,
    auto_retry: bool = False,
) -> int:
    """Execute a full benchmark run.

    For each sample with ground truth:
      - For each available provider: single-provider read
      - For each consensus strategy (if 2+ providers): consensus read
    Computes CER/WER against ground truth and stores everything.

    Args:
        label: Human-readable label for this run.
        providers: Provider names to test. Default: all available.
        strategies: Consensus strategies to test. Default: ['vote', 'best_of'].
        domain: Domain for reading strategies.
        sample_ids: Limit to specific sample IDs. Default: all with ground truth.
        db_path: Override database path.
        on_progress: Optional callback(current, total, message) for progress.
        mode: 'full' (all samples) or 'smoke' (3 hardest samples only).
        auto_enhance: Apply smart enhancement before reading (matches production).
        inject_lessons: Inject lessons into vision prompts (matches production).
        enhance_strategy: Named enhancement strategy (e.g. 'sauvola', 'proven').
            When set, implies auto_enhance=True with the specified strategy.
        iam_partition: IAM database partition name for provenance tracking (e.g. 'test', 'train').
        vocabulary_hints: Optional domain vocabulary hints list. If provided, enables vocab hints mode.

    Returns:
        The run ID.
    """
    conn = get_connection(db_path)
    try:
        return _run_benchmark_inner(
            conn, label, providers, strategies, domain, sample_ids,
            on_progress, mode, auto_enhance, inject_lessons, enhance_strategy,
            iam_partition=iam_partition,
            vocabulary_hints=vocabulary_hints,
            vocab_hints_off=vocab_hints_off,
            line_level=line_level,
            auto_retry=auto_retry,
        )
    finally:
        conn.close()


def _run_benchmark_inner(
    conn,
    label: str,
    providers: list[str] | None,
    strategies: list[str] | None,
    domain: str,
    sample_ids: list[int] | None,
    on_progress: Callable[[int, int, str], None] | None,
    mode: str,
    auto_enhance: bool = False,
    inject_lessons: bool = False,
    enhance_strategy: str | None = None,
    iam_partition: str | None = None,
    vocabulary_hints: list[str] | None = None,
    vocab_hints_off: int = 0,
    line_level: bool = False,
    auto_retry: bool = False,
) -> int:
    """Inner benchmark logic with connection managed by caller."""
    # Resolve providers
    avail = _available_providers()
    if providers:
        providers = [p for p in providers if p in avail]
    else:
        providers = avail

    if not providers:
        raise RuntimeError("No providers available. Install at least one SDK and set API key.")

    # Default strategies
    if strategies is None:
        strategies = ["vote", "best_of"] if len(providers) >= 2 else []

    # Get samples
    samples = samples_with_ground_truth(conn)
    if sample_ids:
        samples = [s for s in samples if s.id in sample_ids]

    # Smoke test mode: pick 3 hardest samples based on historical CER
    if mode == "smoke":
        samples = _select_smoke_samples(conn, samples, limit=3)

    if not samples:
        raise RuntimeError("No samples with ground truth. Run 'benchmark ingest' and 'benchmark transcribe' first.")

    # Create run with provenance metadata
    all_strategies = ["single"] + strategies
    run_id = insert_run(
        conn,
        label=label,
        providers=providers,
        strategies=all_strategies,
        domain=domain,
        model_version=_resolve_model_version(providers),
        norm_flags=_NORM_FLAGS,
        iam_partition=iam_partition,
        vocab_hints_off=vocab_hints_off,
    )
    logger.info("Benchmark run %d: %d samples, providers=%s, strategies=%s", run_id, len(samples), providers, all_strategies)

    evaluated = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        gt = get_latest_ground_truth(conn, sample.id)
        if not gt:
            continue

        if not Path(sample.image_path).exists():
            logger.warning("Image missing for sample %d: %s", sample.id, sample.image_path)
            continue

        if on_progress:
            on_progress(i + 1, total, f"Sample {sample.id} ({sample.student or 'unknown'})")

        # Single-provider reads
        for provider in providers:
            result = _read_single(
                sample.image_path, provider, domain,
                auto_enhance, inject_lessons, enhance_strategy,
                line_level=line_level,
                auto_retry=auto_retry,
            )
            # Compute marker rate from raw text BEFORE any normalization
            raw_text = result["text"]
            marker_rate = _compute_marker_rate(raw_text) if raw_text else None
            po_id = insert_provider_output(
                conn,
                run_id=run_id,
                sample_id=sample.id,
                provider=provider,
                strategy="single",
                output_text=result["text"],
                confidence=result["confidence"],
                latency_ms=result["latency_ms"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                error=result["error"],
                question_marker_rate=marker_rate,
                autocommit=False,
            )

            if not result["error"]:
                cer, char_edits, ref_chars = character_error_rate(result["text"], gt.text)
                wer, word_edits, ref_words = word_error_rate(result["text"], gt.text)
                insert_eval_metric(
                    conn, po_id, gt.id,
                    cer=cer, wer=wer,
                    char_edits=char_edits, word_edits=word_edits,
                    ref_chars=ref_chars, ref_words=ref_words,
                    autocommit=False,
                )

        # Consensus strategies (need 2+ providers)
        if len(providers) >= 2:
            for strategy in strategies:
                result = _read_consensus(sample.image_path, providers, strategy, domain)
                provider_label = "+".join(providers)
                # Compute marker rate from RAW text BEFORE character_error_rate() normalizes it
                # CRITICAL: normalize_text() in metrics.py strips [?] — must capture here
                raw_text = result["text"]
                consensus_marker_rate = _compute_marker_rate(raw_text) if raw_text else None
                po_id = insert_provider_output(
                    conn,
                    run_id=run_id,
                    sample_id=sample.id,
                    provider=provider_label,
                    strategy=strategy,
                    output_text=result["text"],
                    confidence=result["confidence"],
                    latency_ms=result["latency_ms"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    error=result["error"],
                    question_marker_rate=consensus_marker_rate,
                    autocommit=False,
                )

                if not result["error"]:
                    cer, char_edits, ref_chars = character_error_rate(result["text"], gt.text)
                    wer, word_edits, ref_words = word_error_rate(result["text"], gt.text)
                    insert_eval_metric(
                        conn, po_id, gt.id,
                        cer=cer, wer=wer,
                        char_edits=char_edits, word_edits=word_edits,
                        ref_chars=ref_chars, ref_words=ref_words,
                        autocommit=False,
                    )

        # Commit per sample (batch within sample, not per-row)
        conn.commit()
        evaluated += 1

    finish_run(conn, run_id, evaluated)
    logger.info("Benchmark run %d complete: %d samples evaluated", run_id, evaluated)
    return run_id


def compare_strategies(
    strategies: list[str],
    domain: str = "biology",
    db_path: Path | str | None = None,
    regression_threshold: float = 0.5,
) -> str:
    """Run benchmark for each strategy and print a comparison table.

    Args:
        strategies: List of consensus strategy names, e.g. ['vote', 'best_of', 'self_correct'].
        domain: Domain for reading strategies.
        db_path: Override database path.
        regression_threshold: Warn if CER increases by more than this many percentage points.

    Returns:
        Formatted comparison table as a string.
    """
    from handwriting_engine.benchmark.db import get_run_results
    from handwriting_engine.benchmark.report import detect_regressions

    results: list[dict] = []

    for strategy in strategies:
        label = f"compare-strategies:{strategy}"
        run_id = run_benchmark(
            label=label,
            strategies=[strategy],
            domain=domain,
            db_path=db_path,
        )
        conn = get_connection(db_path)
        try:
            rows = get_run_results(conn, run_id)
        finally:
            conn.close()

        if not rows:
            results.append({"strategy": strategy, "cer": None, "wer": None, "samples": 0})
            continue

        cers = [r["cer"] for r in rows if r.get("cer") is not None]
        wers = [r["wer"] for r in rows if r.get("wer") is not None]
        avg_cer = sum(cers) / len(cers) if cers else None
        avg_wer = sum(wers) / len(wers) if wers else None
        results.append({
            "strategy": strategy,
            "cer": avg_cer,
            "wer": avg_wer,
            "samples": len(cers),
            "run_id": run_id,
        })

        # Regression alerting: compare to previous run for this strategy
        conn = get_connection(db_path)
        try:
            regressions = detect_regressions(conn, threshold=regression_threshold / 100.0)
        finally:
            conn.close()
        for reg in regressions:
            prev_pct = reg.get("prev_cer", 0) * 100
            curr_pct = reg.get("current_cer", 0) * 100
            print(
                f"WARNING: CER regression detected: {prev_pct:.2f}% → {curr_pct:.2f}%"
                f" (>{regression_threshold:.1f}% increase)"
            )

    # Format comparison table
    lines = [
        "Strategy Comparison:",
        f"{'strategy':<20} | {'CER':>7} | {'WER':>7} | {'samples':>7}",
        f"{'-'*20}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}",
    ]
    for r in results:
        cer_str = f"{r['cer']*100:.2f}%" if r["cer"] is not None else "N/A"
        wer_str = f"{r['wer']*100:.2f}%" if r["wer"] is not None else "N/A"
        lines.append(f"{r['strategy']:<20} | {cer_str:>7} | {wer_str:>7} | {r['samples']:>7}")

    return "\n".join(lines)


def _select_smoke_samples(conn, samples: list, limit: int = 3) -> list:
    """Select the hardest samples for smoke testing based on historical CER."""
    from handwriting_engine.benchmark.db import get_run_results, get_latest_run_id

    latest_run = get_latest_run_id(conn)
    if not latest_run:
        return samples[:limit]

    results = get_run_results(conn, latest_run)
    # Average CER per sample
    sample_cer: dict[int, list[float]] = {}
    for r in results:
        if r.get("cer") is not None:
            sample_cer.setdefault(r["sample_id"], []).append(r["cer"])

    sample_ids = {s.id for s in samples}
    ranked = sorted(
        ((sid, sum(cers) / len(cers)) for sid, cers in sample_cer.items() if sid in sample_ids),
        key=lambda x: x[1],
        reverse=True,
    )

    hard_ids = {sid for sid, _ in ranked[:limit]}
    selected = [s for s in samples if s.id in hard_ids]

    # Fill remaining slots if not enough history
    if len(selected) < limit:
        remaining = [s for s in samples if s.id not in hard_ids]
        selected.extend(remaining[: limit - len(selected)])

    return selected


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
        # prompt_adapter is applied by default in read_page(); distinguished from
        # baseline by leaving vocab_hints_off at default (0 = hints ON).
        "kwargs": {"strategies": []},
    },
    {
        "name": "zoomed_verify",
        "label": "sweep:zoomed_verify",
        "kwargs": {"strategies": [], "auto_retry": True},
    },
]


def run_sweep(
    provider: str = "gemini",
    db_path: Path | str | None = None,
    yes: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Execute all 5 sweep strategies against IAM samples (IAM-02).

    Fetches IAM sample IDs (samples.category='iam' with ground truth) from the
    DB and passes them to run_benchmark() once per strategy. Returns a dict
    mapping strategy name to run_id.

    Args:
        provider: Provider used for all strategies.
        db_path: Override database path.
        yes: Reserved for future per-strategy confirmation; CLI guards cost upstream.
        on_progress: Optional progress callback forwarded to run_benchmark.

    Returns:
        Dict {strategy_name: run_id} with exactly 5 keys matching SWEEP_STRATEGIES.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT DISTINCT s.id AS id FROM samples s
               JOIN ground_truths gt ON gt.sample_id = s.id
               WHERE s.category = 'iam'
               ORDER BY s.id"""
        ).fetchall()
        sample_ids = [r["id"] for r in rows]
    finally:
        conn.close()

    run_ids: dict[str, int] = {}
    for config in SWEEP_STRATEGIES:
        run_id = run_benchmark(
            label=config["label"],
            providers=[provider],
            sample_ids=sample_ids if sample_ids else None,
            db_path=db_path,
            on_progress=on_progress,
            **config["kwargs"],
        )
        run_ids[config["name"]] = run_id

    return run_ids
