"""
Data-driven Pareto threshold sweep for the local-first OCR strategy.

Replays the local-first decision retrospectively against the benchmark DB:
for each candidate threshold T, simulate "if trocr.confidence >= T and trocr
output non-empty, accept local; else escalate to cloud", then compute
aggregate CER vs ground truth and the cloud-call rate.

Outputs a list of ``ThresholdPoint`` rows. ``pareto_frontier`` filters to
non-dominated points (a point is dominated when another exists with
*both* lower CER and lower cloud-call rate). ``pick_knee`` picks the
recommended operating point from the frontier by minimizing
``cer + 0.5 * cloud_call_rate`` — a Lagrangian-ish tradeoff that prices
each unit of cloud usage at half a unit of CER, picking the elbow where
spending more cloud calls stops buying meaningful accuracy.

No I/O outside the SQLite read; safe to call from tests with an
in-memory or tmp-path DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from handwriting_engine.benchmark.db import get_connection
from handwriting_engine.benchmark.metrics import character_error_rate

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
)


@dataclass(frozen=True)
class ThresholdPoint:
    """One point in the Pareto sweep.

    Attributes:
        threshold: The local-first acceptance threshold being simulated.
        cer: Aggregate CER (edits/ref_chars summed across the chosen-text set).
        cloud_call_rate: Fraction in [0, 1] of samples that escalated.
        n_samples: Number of (sample, gt) pairs evaluated at this threshold.
    """
    threshold: float
    cer: float
    cloud_call_rate: float
    n_samples: int


def _load_paired_rows(
    conn,
    cloud_provider: str,
    writer_id: str | None,
) -> list[dict]:
    """Fetch the latest (trocr, cloud) output pair plus ground truth per sample.

    "Latest" is per-provider per-sample (max provider_outputs.id). A sample is
    only included when *both* providers have at least one output AND the sample
    has a ground truth row.

    Returns rows with keys: sample_id, writer_id, gt_text,
    trocr_text, trocr_conf, cloud_text.
    """
    sql = """
        WITH latest_trocr AS (
            SELECT po.sample_id,
                   po.output_text AS trocr_text,
                   po.confidence  AS trocr_conf
            FROM provider_outputs po
            JOIN (
                SELECT sample_id, MAX(id) AS max_id
                FROM provider_outputs
                WHERE provider = 'trocr'
                GROUP BY sample_id
            ) m ON m.max_id = po.id
        ),
        latest_cloud AS (
            SELECT po.sample_id,
                   po.output_text AS cloud_text
            FROM provider_outputs po
            JOIN (
                SELECT sample_id, MAX(id) AS max_id
                FROM provider_outputs
                WHERE provider = ?
                GROUP BY sample_id
            ) m ON m.max_id = po.id
        ),
        latest_gt AS (
            SELECT gt.sample_id, gt.text AS gt_text
            FROM ground_truths gt
            JOIN (
                SELECT sample_id, MAX(id) AS max_id
                FROM ground_truths
                GROUP BY sample_id
            ) m ON m.max_id = gt.id
        )
        SELECT s.id          AS sample_id,
               s.student     AS writer_id,
               g.gt_text     AS gt_text,
               t.trocr_text  AS trocr_text,
               t.trocr_conf  AS trocr_conf,
               c.cloud_text  AS cloud_text
        FROM samples s
        JOIN latest_gt    g ON g.sample_id = s.id
        JOIN latest_trocr t ON t.sample_id = s.id
        JOIN latest_cloud c ON c.sample_id = s.id
    """
    params: list = [cloud_provider]
    if writer_id is not None:
        sql += " WHERE s.student = ?"
        params.append(writer_id)

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def sweep_thresholds(
    *,
    db_path: Path | str | None = None,
    cloud_provider: str = "gemini",
    thresholds: list[float] | None = None,
    writer_id: str | None = None,
) -> list[ThresholdPoint]:
    """Sweep candidate thresholds and report (CER, cloud_call_rate) at each.

    For each (sample, ground_truth) where both a 'trocr' and a
    ``cloud_provider`` ``provider_outputs`` row exist (latest per sample),
    simulate the local-first decision:

      - If trocr.confidence >= threshold AND trocr.text.strip() != ""
        => accept local (cost = 0)
      - Else escalate to cloud (cost = 1 cloud call)

    Aggregate CER is computed micro-average style (sum of edits / sum of
    ref_chars) across the chosen-text set per threshold.

    Args:
        db_path: Override DB path. None = default benchmark.db.
        cloud_provider: provider name to use as the escalation target.
        thresholds: candidate thresholds. None => DEFAULT_THRESHOLDS.
        writer_id: filter to a single writer (matches samples.student).

    Returns:
        List of ``ThresholdPoint`` in the same order as ``thresholds``.
        Empty list when no paired (trocr, cloud, gt) rows exist.
    """
    candidate = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLDS)

    conn = get_connection(db_path)
    try:
        rows = _load_paired_rows(conn, cloud_provider, writer_id)
    finally:
        conn.close()

    if not rows:
        return []

    points: list[ThresholdPoint] = []
    for t in candidate:
        total_edits = 0
        total_ref = 0
        escalations = 0
        n = 0
        for r in rows:
            trocr_text = r["trocr_text"] or ""
            trocr_conf = r["trocr_conf"] if r["trocr_conf"] is not None else 0.0
            cloud_text = r["cloud_text"] or ""
            gt_text = r["gt_text"] or ""

            if trocr_conf >= t and trocr_text.strip():
                chosen = trocr_text
            else:
                chosen = cloud_text
                escalations += 1

            _, edits, ref_chars = character_error_rate(chosen, gt_text)
            # Skip samples with empty ref to avoid distorting the aggregate
            # (character_error_rate returns 0 or inf in that case).
            if ref_chars == 0:
                continue
            total_edits += edits
            total_ref += ref_chars
            n += 1

        if n == 0:
            # No usable samples (all had empty ground truth). Report zeros.
            points.append(ThresholdPoint(
                threshold=float(t), cer=0.0, cloud_call_rate=0.0, n_samples=0,
            ))
            continue

        cer = total_edits / total_ref if total_ref else 0.0
        cloud_rate = escalations / n
        points.append(ThresholdPoint(
            threshold=float(t),
            cer=cer,
            cloud_call_rate=cloud_rate,
            n_samples=n,
        ))

    return points


def pareto_frontier(points: Iterable[ThresholdPoint]) -> list[ThresholdPoint]:
    """Return only the non-dominated points.

    A point P is dominated iff some other point Q has Q.cer < P.cer AND
    Q.cloud_call_rate < P.cloud_call_rate (strict on both axes). Ties on
    either axis do not dominate. Frontier is returned sorted by threshold.
    """
    pts = [p for p in points if p.n_samples > 0]
    frontier: list[ThresholdPoint] = []
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            if q.cer < p.cer and q.cloud_call_rate < p.cloud_call_rate:
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda x: x.threshold)
    return frontier


def pick_knee(frontier: list[ThresholdPoint]) -> ThresholdPoint | None:
    """Pick the recommended operating point from a Pareto frontier.

    Minimizes ``cer + 0.5 * cloud_call_rate``. The 0.5 weight prices each
    unit of cloud-call rate at half a unit of CER; in practice this picks
    the knee where buying more cloud calls stops yielding meaningful CER
    drops. Returns None on empty frontier.
    """
    if not frontier:
        return None
    return min(frontier, key=lambda p: p.cer + 0.5 * p.cloud_call_rate)
