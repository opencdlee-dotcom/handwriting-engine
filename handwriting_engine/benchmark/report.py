"""
Benchmark reporting — comparison tables, regression detection, per-sample drill-down,
quality correlation, and output formatting.
"""

from __future__ import annotations

import csv
import io
import statistics
from pathlib import Path

from handwriting_engine.benchmark.db import (
    get_connection,
    get_latest_run_id,
    get_run_results,
    list_runs,
)
from handwriting_engine.benchmark.evaluate import estimate_cost
from handwriting_engine.benchmark.models import StrategyResult
from handwriting_engine.benchmark.stats import (
    bootstrap_ci,
    cohens_r,
    wilcoxon_signed_rank,
)


# Minimum paired-sample count for the statistics layer to attach. Below this,
# normal-approximation Wilcoxon is too rough and bootstrap CIs are dominated
# by sampling noise — better to print nothing than to print a misleading
# p-value. Aligns with Phase 8 success criterion #1.
_STATS_MIN_PAIRED_N = 10


def _aggregate_results(rows: list[dict]) -> list[StrategyResult]:
    """Group run results by (provider, strategy) and compute aggregate metrics."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["provider"], row["strategy"])
        groups.setdefault(key, []).append(row)

    results = []
    for (provider, strategy), group_rows in sorted(groups.items()):
        cers = [r["cer"] for r in group_rows if r.get("cer") is not None]
        wers = [r["wer"] for r in group_rows if r.get("wer") is not None]
        failures = sum(1 for r in group_rows if r.get("error"))
        total_in = sum(r.get("input_tokens", 0) or 0 for r in group_rows)
        total_out = sum(r.get("output_tokens", 0) or 0 for r in group_rows)

        cost = estimate_cost(total_in, total_out, provider)
        marker_rates = [r.get("question_marker_rate") for r in group_rows if r.get("question_marker_rate") is not None]
        mean_marker_rate = statistics.mean(marker_rates) if marker_rates else 0.0

        results.append(StrategyResult(
            provider=provider,
            strategy=strategy,
            mean_cer=statistics.mean(cers) if cers else -1,
            mean_wer=statistics.mean(wers) if wers else -1,
            median_cer=statistics.median(cers) if cers else -1,
            median_wer=statistics.median(wers) if wers else -1,
            stdev_cer=statistics.stdev(cers) if len(cers) >= 2 else 0.0,
            stdev_wer=statistics.stdev(wers) if len(wers) >= 2 else 0.0,
            total_tokens=total_in + total_out,
            estimated_cost_usd=cost,
            sample_count=len(group_rows),
            failures=failures,
            mean_marker_rate=mean_marker_rate,
        ))

    return results


def generate_report(
    run_id: int | None = None,
    db_path: Path | str | None = None,
    fmt: str = "table",
) -> str:
    """Generate a human-readable report for a benchmark run.

    Args:
        run_id: Specific run ID. Default: latest run.
        db_path: Override database path.
        fmt: Output format — 'table', 'json', or 'csv'.

    Returns:
        Formatted report string.
    """
    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
            if run_id is None:
                return "No benchmark runs found. Run 'handwriting-engine benchmark run' first."

        rows = get_run_results(conn, run_id)
        run_meta_row = conn.execute(
            "SELECT model_version, iam_partition, norm_flags, vocab_hints_off FROM runs WHERE id = ?",
            (run_id,)
        ).fetchone()
        run_meta = dict(run_meta_row) if run_meta_row else None
    finally:
        conn.close()

    if not rows:
        return f"No results found for run {run_id}."

    results = _aggregate_results(rows)

    if fmt == "json":
        return _format_json(run_id, results)
    elif fmt == "csv":
        return _format_csv(results)
    return _format_table(run_id, results, run_meta=run_meta)


def _format_table(run_id: int, results: list[StrategyResult], run_meta: dict | None = None) -> str:
    """Render results as an ASCII table with optional provenance header."""
    lines = [f"Benchmark Run #{run_id}", ""]

    if run_meta:
        lines.append("Provenance:")
        lines.append(f"  Model:      {run_meta.get('model_version') or 'unknown'}")
        partition = run_meta.get('iam_partition') or 'n/a'
        lines.append(f"  Partition:  {partition}")
        lines.append(f"  Norm flags: {run_meta.get('norm_flags') or 'unknown'}")
        vocab_off = 'yes' if run_meta.get('vocab_hints_off') else 'no'
        lines.append(f"  Vocab hints off: {vocab_off}")
        lines.append("")

    header = (
        f"{'Provider':<20} {'Strategy':<10} {'CER':>8} {'±sd':>6} {'WER':>8} "
        f"{'marker_rate':>11} {'Tokens':>10} {'Cost':>8} {'N':>4} {'Fail':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in sorted(results, key=lambda x: x.mean_cer if x.mean_cer >= 0 else 999):
        cer_str = f"{r.mean_cer:.2%}" if r.mean_cer >= 0 else "N/A"
        sd_str = f"{r.stdev_cer:.2%}" if r.stdev_cer > 0 else "---"
        wer_str = f"{r.mean_wer:.2%}" if r.mean_wer >= 0 else "N/A"
        cost_str = f"${r.estimated_cost_usd:.4f}"
        marker_pct = f"{r.mean_marker_rate * 100:.2f}%"
        lines.append(
            f"{r.provider:<20} {r.strategy:<10} {cer_str:>8} {sd_str:>6} {wer_str:>8} "
            f"{marker_pct:>11} {r.total_tokens:>10,} {cost_str:>8} {r.sample_count:>4} {r.failures:>4}"
        )

    lines.append("")
    return "\n".join(lines)


def _format_json(run_id: int, results: list[StrategyResult]) -> str:
    """Render results as JSON."""
    import json
    data = {
        "run_id": run_id,
        "results": [
            {
                "provider": r.provider,
                "strategy": r.strategy,
                "mean_cer": round(r.mean_cer, 6) if r.mean_cer >= 0 else None,
                "mean_wer": round(r.mean_wer, 6) if r.mean_wer >= 0 else None,
                "median_cer": round(r.median_cer, 6) if r.median_cer >= 0 else None,
                "median_wer": round(r.median_wer, 6) if r.median_wer >= 0 else None,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": round(r.estimated_cost_usd, 6),
                "sample_count": r.sample_count,
                "failures": r.failures,
            }
            for r in results
        ],
    }
    return json.dumps(data, indent=2)


def _format_csv(results: list[StrategyResult]) -> str:
    """Render results as properly escaped CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["provider", "strategy", "mean_cer", "mean_wer", "median_cer",
                      "median_wer", "total_tokens", "cost_usd", "samples", "failures"])
    for r in results:
        writer.writerow([
            r.provider,
            r.strategy,
            f"{r.mean_cer:.6f}" if r.mean_cer >= 0 else "",
            f"{r.mean_wer:.6f}" if r.mean_wer >= 0 else "",
            f"{r.median_cer:.6f}" if r.median_cer >= 0 else "",
            f"{r.median_wer:.6f}" if r.median_wer >= 0 else "",
            r.total_tokens,
            f"{r.estimated_cost_usd:.6f}",
            r.sample_count,
            r.failures,
        ])
    return output.getvalue().strip()


def _paired_cers(
    rows_1: list[dict],
    rows_2: list[dict],
    provider: str,
    strategy: str,
) -> tuple[list[float], list[float]]:
    """Pair per-sample CERs from two runs by sample_id, restricted to the
    given (provider, strategy). Order is sample_id-sorted and identical
    on both sides — that's what makes the test paired."""
    by_sample_1 = {
        r["sample_id"]: r["cer"]
        for r in rows_1
        if r.get("provider") == provider
        and r.get("strategy") == strategy
        and r.get("cer") is not None
    }
    by_sample_2 = {
        r["sample_id"]: r["cer"]
        for r in rows_2
        if r.get("provider") == provider
        and r.get("strategy") == strategy
        and r.get("cer") is not None
    }
    shared = sorted(set(by_sample_1) & set(by_sample_2))
    return ([by_sample_1[sid] for sid in shared],
            [by_sample_2[sid] for sid in shared])


def compare_runs(
    run_id_1: int,
    run_id_2: int,
    db_path: Path | str | None = None,
) -> str:
    """Compare two benchmark runs side by side.

    For each (provider, strategy) pair shared between the runs, when the
    paired sample count is >= 10 the output appends:
    - paired Wilcoxon signed-rank p-value and z-score
    - Cohen's r effect size
    - 95% bootstrap CIs for each run's CER

    Pairing is by sample_id — same image, different run.
    """
    conn = get_connection(db_path)
    try:
        rows_1 = get_run_results(conn, run_id_1)
        rows_2 = get_run_results(conn, run_id_2)
    finally:
        conn.close()

    if not rows_1:
        return f"No results found for run {run_id_1}."
    if not rows_2:
        return f"No results found for run {run_id_2}."

    results_1 = {(r.provider, r.strategy): r for r in _aggregate_results(rows_1)}
    results_2 = {(r.provider, r.strategy): r for r in _aggregate_results(rows_2)}

    all_keys = sorted(set(results_1) | set(results_2))

    lines = [f"Comparison: Run #{run_id_1} vs Run #{run_id_2}", ""]
    header = f"{'Provider':<20} {'Strategy':<10} {'CER #1':>8} {'CER #2':>8} {'Delta':>8} {'Status':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    regressions = 0
    improvements = 0

    for key in all_keys:
        r1 = results_1.get(key)
        r2 = results_2.get(key)
        provider, strategy = key

        if r1 and r2 and r1.mean_cer >= 0 and r2.mean_cer >= 0:
            delta = r2.mean_cer - r1.mean_cer
            if delta > 0.005:
                status = "REGRESSION"
                regressions += 1
            elif delta < -0.005:
                status = "IMPROVED"
                improvements += 1
            else:
                status = "unchanged"
            lines.append(
                f"{provider:<20} {strategy:<10} {r1.mean_cer:>7.2%} {r2.mean_cer:>7.2%} "
                f"{delta:>+7.2%} {status:>12}"
            )

            paired_a, paired_b = _paired_cers(rows_1, rows_2, provider, strategy)
            if len(paired_a) >= _STATS_MIN_PAIRED_N:
                wilcox = wilcoxon_signed_rank(paired_a, paired_b)
                r_effect = cohens_r(wilcox.z, wilcox.n)
                lo_a, hi_a = bootstrap_ci(paired_a, seed=run_id_1)
                lo_b, hi_b = bootstrap_ci(paired_b, seed=run_id_2)
                lines.append(
                    f"{'  stats:':<31} "
                    f"n={wilcox.n} "
                    f"W={wilcox.statistic:.1f} "
                    f"z={wilcox.z:+.2f} "
                    f"p={wilcox.p_value:.4f} "
                    f"r={r_effect:.2f}"
                )
                lines.append(
                    f"{'  CI95:':<31} "
                    f"run#{run_id_1} [{lo_a:.2%}, {hi_a:.2%}]  "
                    f"run#{run_id_2} [{lo_b:.2%}, {hi_b:.2%}]"
                )
        elif r1 and not r2:
            cer_str = f"{r1.mean_cer:>7.2%}" if r1.mean_cer >= 0 else "    N/A"
            lines.append(f"{provider:<20} {strategy:<10} {cer_str} {'---':>8} {'---':>8} {'removed':>12}")
        elif r2 and not r1:
            cer_str = f"{r2.mean_cer:>7.2%}" if r2.mean_cer >= 0 else "    N/A"
            lines.append(f"{provider:<20} {strategy:<10} {'---':>8} {cer_str} {'---':>8} {'new':>12}")

    lines.append("")
    lines.append(f"Summary: {improvements} improved, {regressions} regressed, {len(all_keys) - improvements - regressions} unchanged")
    return "\n".join(lines)


# --- Phase 9 / RPT-02: configuration recommendation ---


# Composite score weights. The contract from REQUIREMENTS.md is 70/15/15;
# changing these is a product decision, not a code change.
_RECOMMEND_W_CER = 0.70
_RECOMMEND_W_COST = 0.15
_RECOMMEND_W_STAB = 0.15


def recommend_strategy(db_path: Path | str | None = None) -> str:
    """Rank (provider, strategy) configurations by a composite score.

    Score = 0.70 * (1 - cer_norm) + 0.15 * (1 - cost_norm) + 0.15 * stab_norm,
    where each component is min-max normalized within the candidate set.
    Lower CER and lower cost score higher; higher stability scores higher.

    Stability is the inverse of CER stdev across runs of the same
    (provider, strategy). When a candidate has only one run, it is given
    the median stability score of the candidate set (neutral) and the
    output flags it as `n=1`.

    Returns a ranked human-readable table with the winner annotated.
    """
    conn = get_connection(db_path)
    try:
        runs = list_runs(conn)
        if not runs:
            return "No runs in database — nothing to recommend."

        # Per-(provider, strategy) accumulator across all runs.
        accum: dict[tuple[str, str], dict] = {}
        for run in runs:
            rows = get_run_results(conn, run.run_id)
            for r in _aggregate_results(rows):
                if r.mean_cer < 0:
                    continue
                key = (r.provider, r.strategy)
                bucket = accum.setdefault(key, {
                    "cers_per_run": [],
                    "cost_per_sample_runs": [],
                    "total_runs": 0,
                })
                bucket["cers_per_run"].append(r.mean_cer)
                if r.sample_count > 0:
                    bucket["cost_per_sample_runs"].append(
                        r.estimated_cost_usd / r.sample_count
                    )
                bucket["total_runs"] += 1
    finally:
        conn.close()

    if not accum:
        return "No (provider, strategy) configurations with measured CER — nothing to recommend."

    # Compute summary stats per candidate.
    summaries: list[dict] = []
    for (provider, strategy), bucket in accum.items():
        cers = bucket["cers_per_run"]
        costs = bucket["cost_per_sample_runs"]
        n_runs = bucket["total_runs"]
        mean_cer = sum(cers) / len(cers)
        stdev_cer = statistics.stdev(cers) if len(cers) >= 2 else None
        mean_cost = sum(costs) / len(costs) if costs else 0.0
        summaries.append({
            "provider": provider,
            "strategy": strategy,
            "n_runs": n_runs,
            "mean_cer": mean_cer,
            "stdev_cer": stdev_cer,
            "mean_cost_per_sample": mean_cost,
        })

    # Normalize each component to [0, 1] within the candidate set.
    cers = [s["mean_cer"] for s in summaries]
    costs = [s["mean_cost_per_sample"] for s in summaries]
    cer_min, cer_max = min(cers), max(cers)
    cost_min, cost_max = min(costs), max(costs)

    measured_stdevs = [s["stdev_cer"] for s in summaries if s["stdev_cer"] is not None]
    median_stdev = statistics.median(measured_stdevs) if measured_stdevs else 0.0
    stdevs_for_norm = [
        s["stdev_cer"] if s["stdev_cer"] is not None else median_stdev
        for s in summaries
    ]
    stdev_min, stdev_max = min(stdevs_for_norm), max(stdevs_for_norm)

    def _norm(x: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-12:
            return 0.5  # all candidates equal on this axis
        return (x - lo) / (hi - lo)

    for s, eff_stdev in zip(summaries, stdevs_for_norm):
        cer_n = _norm(s["mean_cer"], cer_min, cer_max)
        cost_n = _norm(s["mean_cost_per_sample"], cost_min, cost_max)
        stab_n = 1.0 - _norm(eff_stdev, stdev_min, stdev_max)
        s["score"] = (
            _RECOMMEND_W_CER * (1.0 - cer_n)
            + _RECOMMEND_W_COST * (1.0 - cost_n)
            + _RECOMMEND_W_STAB * stab_n
        )

    summaries.sort(key=lambda s: s["score"], reverse=True)
    winner = summaries[0]

    lines = [
        "Strategy + provider recommendation",
        f"  weights: CER {_RECOMMEND_W_CER:.0%}  "
        f"cost {_RECOMMEND_W_COST:.0%}  "
        f"stability {_RECOMMEND_W_STAB:.0%}",
        "",
        f"  Winner: {winner['provider']} + {winner['strategy']}  "
        f"(score {winner['score']:.3f})",
        "",
    ]
    header = (
        f"{'Rank':<4} {'Provider':<12} {'Strategy':<14} "
        f"{'CER':>7} {'$/sample':>10} {'stdev':>9} {'n':>3} {'score':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for rank, s in enumerate(summaries, start=1):
        stdev_str = f"{s['stdev_cer']:.3f}" if s["stdev_cer"] is not None else "  n=1"
        lines.append(
            f"{rank:<4} {s['provider']:<12} {s['strategy']:<14} "
            f"{s['mean_cer']:>6.2%} {s['mean_cost_per_sample']:>9.4f}$ "
            f"{stdev_str:>9} {s['n_runs']:>3} {s['score']:>7.3f}"
        )

    return "\n".join(lines)


def detect_regressions(
    run_id: int | None = None,
    threshold: float = 0.03,
    db_path: Path | str | None = None,
) -> list[dict]:
    """Compare a run against the pinned baseline. Returns list of regressions.

    Phase 9 / RPT-01: anchor is the run pinned via `benchmark set-baseline`.
    Falls back to the previous run when no baseline is pinned, preserving
    pre-Phase-9 behavior on a freshly-initialized DB. Excludes self-compare
    when the current run IS the baseline.

    Default threshold is 3% — with small sample sizes (<30), differences
    below this are within the noise floor and not meaningful.
    """
    from handwriting_engine.benchmark.db import get_baseline_run_id

    conn = get_connection(db_path)
    try:
        runs = list_runs(conn)

        if len(runs) < 2:
            return []

        if run_id is None:
            current = runs[0]
        else:
            current = next((r for r in runs if r.run_id == run_id), None)
        if not current:
            return []

        baseline_run_id = get_baseline_run_id(conn)
        if baseline_run_id is not None and baseline_run_id != current.run_id:
            previous = next((r for r in runs if r.run_id == baseline_run_id), None)
        else:
            # No pinned baseline (or current IS the baseline) — use the run
            # immediately preceding `current`, matching pre-Phase-9 behavior.
            idx = next((i for i, r in enumerate(runs) if r.run_id == current.run_id), None)
            previous = runs[idx + 1] if idx is not None and idx + 1 < len(runs) else None

        if not previous:
            return []

        rows_curr = get_run_results(conn, current.run_id)
        rows_prev = get_run_results(conn, previous.run_id)
    finally:
        conn.close()

    agg_curr = {(r.provider, r.strategy): r for r in _aggregate_results(rows_curr)}
    agg_prev = {(r.provider, r.strategy): r for r in _aggregate_results(rows_prev)}

    regressions = []
    for key, curr in agg_curr.items():
        prev = agg_prev.get(key)
        if prev and curr.mean_cer >= 0 and prev.mean_cer >= 0:
            delta = curr.mean_cer - prev.mean_cer
            if delta > threshold:
                regressions.append({
                    "provider": curr.provider,
                    "strategy": curr.strategy,
                    "previous_cer": prev.mean_cer,
                    "current_cer": curr.mean_cer,
                    "delta": delta,
                })

    return regressions


def sample_drill_down(
    sample_id: int,
    run_id: int | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Generate a per-sample detail report showing all provider outputs and errors.

    Args:
        sample_id: The sample to drill into.
        run_id: Specific run. Default: latest.
        db_path: Override database path.

    Returns:
        Formatted per-sample report.
    """
    from handwriting_engine.benchmark.db import get_sample_by_id, get_latest_ground_truth

    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        if run_id is None:
            return "No benchmark runs found."

        sample = get_sample_by_id(conn, sample_id)
        if not sample:
            return f"Sample {sample_id} not found."

        gt = get_latest_ground_truth(conn, sample_id)
        rows = conn.execute(
            """SELECT po.*, em.cer, em.wer
               FROM provider_outputs po
               LEFT JOIN eval_metrics em ON em.provider_output_id = po.id
               WHERE po.run_id = ? AND po.sample_id = ?
               ORDER BY po.provider, po.strategy""",
            (run_id, sample_id),
        ).fetchall()
    finally:
        conn.close()

    import os
    lines = [
        f"Sample #{sample_id}: {os.path.basename(sample.image_path)}",
        f"Student: {sample.student or 'unknown'} | Page: {sample.page_number}",
    ]

    if gt:
        gt_preview = gt.text[:100] + "..." if len(gt.text) > 100 else gt.text
        lines.append(f"Ground Truth: {gt_preview}")
    else:
        lines.append("Ground Truth: NONE")

    lines.append("")

    if not rows:
        lines.append("No provider outputs for this run.")
        return "\n".join(lines)

    for row in rows:
        row = dict(row)
        cer_str = f"{row['cer']:.2%}" if row.get("cer") is not None else "N/A"
        wer_str = f"{row['wer']:.2%}" if row.get("wer") is not None else "N/A"

        lines.append(f"  {row['provider']}/{row['strategy']} — CER: {cer_str}, WER: {wer_str}, "
                     f"{row['latency_ms']}ms, {row.get('input_tokens', 0)}+{row.get('output_tokens', 0)} tokens")

        if row.get("error"):
            lines.append(f"    ERROR: {row['error']}")
        else:
            preview = row["output_text"][:100] + "..." if len(row["output_text"]) > 100 else row["output_text"]
            lines.append(f"    Output: {preview}")

    lines.append("")
    return "\n".join(lines)


def quality_correlation(
    run_id: int | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Report correlation between image quality metrics and CER.

    Joins quality_assessments with eval_metrics to show whether
    blur/contrast/brightness predicts recognition accuracy.

    Returns:
        Formatted correlation report.
    """
    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        if run_id is None:
            return "No benchmark runs found."

        rows = conn.execute(
            """SELECT qa.blur_score, qa.contrast_score, qa.brightness_mean, qa.quality,
                      AVG(em.cer) as avg_cer, AVG(em.wer) as avg_wer, COUNT(*) as n
               FROM quality_assessments qa
               JOIN provider_outputs po ON po.sample_id = qa.sample_id
               JOIN eval_metrics em ON em.provider_output_id = po.id
               WHERE po.run_id = ?
               GROUP BY qa.sample_id
               ORDER BY avg_cer DESC""",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No quality/accuracy data available. Ensure samples were ingested with quality assessment."

    lines = [f"Quality vs Accuracy (Run #{run_id})", ""]
    header = f"{'Quality':<8} {'Blur':>8} {'Contrast':>8} {'Brightness':>10} {'Avg CER':>8} {'Avg WER':>8} {'N':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(
            f"{r['quality'] or '?':<8} {r['blur_score'] or 0:>8.1f} {r['contrast_score'] or 0:>8.2f} "
            f"{r['brightness_mean'] or 0:>10.1f} {r['avg_cer']:>7.2%} {r['avg_wer']:>7.2%} {r['n']:>4}"
        )

    # Summary by quality tier
    tiers: dict[str, list[float]] = {}
    for r in rows:
        q = r["quality"] or "unknown"
        tiers.setdefault(q, []).append(r["avg_cer"])

    if len(tiers) > 1:
        lines.append("")
        lines.append("By quality tier:")
        for q in ["good", "fair", "poor"]:
            if q in tiers:
                cers = tiers[q]
                lines.append(f"  {q}: avg CER {statistics.mean(cers):.2%} ({len(cers)} samples)")

    lines.append("")
    return "\n".join(lines)


def confidence_calibration(
    run_id: int | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Report how well confidence scores predict actual accuracy.

    Computes correlation between reported confidence and actual CER.
    Flags miscalibrated outputs (high confidence + high error).
    """
    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        if run_id is None:
            return "No benchmark runs found."

        rows = conn.execute(
            """SELECT po.provider, po.strategy, po.confidence, em.cer
               FROM provider_outputs po
               JOIN eval_metrics em ON em.provider_output_id = po.id
               WHERE po.run_id = ? AND po.error IS NULL
               ORDER BY po.confidence DESC""",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    if len(rows) < 3:
        return "Not enough data for calibration (need at least 3 successful outputs)."

    confidences = [r["confidence"] for r in rows]
    cers = [r["cer"] for r in rows]

    # Pearson correlation (confidence should negatively correlate with CER)
    n = len(confidences)
    mean_conf = sum(confidences) / n
    mean_cer = sum(cers) / n
    cov = sum((c - mean_conf) * (e - mean_cer) for c, e in zip(confidences, cers)) / n
    std_conf = (sum((c - mean_conf) ** 2 for c in confidences) / n) ** 0.5
    std_cer = (sum((e - mean_cer) ** 2 for e in cers) / n) ** 0.5

    if std_conf > 0 and std_cer > 0:
        correlation = cov / (std_conf * std_cer)
    else:
        correlation = 0.0

    lines = [f"Confidence Calibration (Run #{run_id})", ""]
    lines.append(f"Correlation (confidence vs CER): {correlation:+.3f}")
    if correlation > -0.3:
        lines.append("  WARNING: Weak correlation — confidence scores are poorly calibrated")
    elif correlation < -0.7:
        lines.append("  Good: Strong negative correlation — confidence is meaningful")
    else:
        lines.append("  Moderate: Confidence has some predictive value")

    lines.append("")

    # Bucket by confidence ranges
    buckets = {"0.0-0.3": [], "0.3-0.5": [], "0.5-0.7": [], "0.7-0.9": [], "0.9-1.0": []}
    for r in rows:
        c = r["confidence"]
        if c < 0.3:
            buckets["0.0-0.3"].append(r["cer"])
        elif c < 0.5:
            buckets["0.3-0.5"].append(r["cer"])
        elif c < 0.7:
            buckets["0.5-0.7"].append(r["cer"])
        elif c < 0.9:
            buckets["0.7-0.9"].append(r["cer"])
        else:
            buckets["0.9-1.0"].append(r["cer"])

    lines.append(f"{'Confidence':>12} {'Avg CER':>8} {'N':>4}")
    lines.append("-" * 28)
    for bucket, bucket_cers in buckets.items():
        if bucket_cers:
            avg = statistics.mean(bucket_cers)
            lines.append(f"{bucket:>12} {avg:>7.2%} {len(bucket_cers):>4}")

    miscalibrated = [r for r in rows if r["confidence"] > 0.7 and r["cer"] > 0.1]
    if miscalibrated:
        lines.append("")
        lines.append(f"Miscalibrated outputs ({len(miscalibrated)} found — high confidence, high error):")
        for r in miscalibrated[:5]:
            lines.append(f"  {r['provider']}/{r['strategy']}: confidence={r['confidence']:.2f}, CER={r['cer']:.2%}")

    lines.append("")
    return "\n".join(lines)


def generate_per_writer_report(
    run_id: int | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Per-writer CER breakdown for a benchmark run (IAM-03).

    Groups eval_metrics by samples.student so the developer can tell whether a
    strategy's CER gain is consistent across writers or driven by a few easy
    ones. Requires samples to have been ingested with student tags (e.g. via
    `benchmark ingest-iam`, which sets student='iam-writer-XXX').
    """
    conn = get_connection(db_path)
    try:
        if run_id is None:
            run_id = get_latest_run_id(conn)
        if run_id is None:
            return "No runs found in database."

        rows = conn.execute(
            """SELECT s.student         AS student,
                      AVG(em.cer)       AS mean_cer,
                      MIN(em.cer)       AS min_cer,
                      MAX(em.cer)       AS max_cer,
                      COUNT(*)          AS n_samples
               FROM provider_outputs po
               JOIN eval_metrics em ON em.provider_output_id = po.id
               JOIN samples s      ON s.id = po.sample_id
               WHERE po.run_id = ?
                 AND s.student IS NOT NULL
                 AND s.student != ''
               GROUP BY s.student
               ORDER BY mean_cer DESC""",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return (
            f"Per-Writer CER (Run #{run_id})\n\n"
            "No writer data found for this run.\n"
            "Tip: ingest IAM samples with `benchmark ingest-iam` first — "
            "only IAM samples carry per-writer tags."
        )

    header = f"{'Writer':<25} {'Mean CER':>9} {'Min CER':>9} {'Max CER':>9} {'N':>4}"
    separator = "-" * len(header)
    lines = [
        f"Per-Writer CER (Run #{run_id})",
        "",
        header,
        separator,
    ]
    for r in rows:
        lines.append(
            f"{r['student']:<25} "
            f"{r['mean_cer']:>8.2%} "
            f"{r['min_cer']:>8.2%} "
            f"{r['max_cer']:>8.2%} "
            f"{r['n_samples']:>4}"
        )
    lines.append(separator)
    lines.append(f"  {len(rows)} writer(s) shown")
    return "\n".join(lines)
