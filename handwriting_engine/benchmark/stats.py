"""
Statistical defensibility for benchmark comparisons (Phase 8 / STAT-01, STAT-02).

Hand-rolled to avoid pulling scipy as a dependency for three functions. The
project already hand-rolls Levenshtein inline in postprocess.py for the same
reason: trade ~150 LOC of well-tested math for a 30 MB scientific-stack dep.

Public API:
- wilcoxon_signed_rank(a, b)  -> {"statistic", "p_value", "z", "n"}
- bootstrap_ci(values, ...)   -> (lower, upper)
- cohens_r(z, n)              -> float

All three operate on plain lists of floats; no numpy required.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (paired, two-sided)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WilcoxonResult:
    """Outcome of a paired Wilcoxon signed-rank test."""
    statistic: float   # W+ (sum of ranks of positive differences)
    p_value: float     # two-sided
    z: float           # standard normal z-score (sign matches direction of effect)
    n: int             # number of non-zero paired differences


def _rank_with_ties(abs_values: list[float]) -> tuple[list[float], float]:
    """Average-rank values; also return the tie-correction sum used by the
    Wilcoxon variance formula.

    The tie-correction is sum(t^3 - t) over each tie group, which gets
    subtracted (divided by 48) inside the variance.
    """
    indexed = sorted(enumerate(abs_values), key=lambda x: x[1])
    ranks = [0.0] * len(abs_values)
    tie_correction = 0.0

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # Average rank for positions i..j (1-indexed)
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        run_len = j - i + 1
        if run_len > 1:
            tie_correction += run_len ** 3 - run_len
        i = j + 1

    return ranks, tie_correction


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test using normal approximation
    with continuity correction.

    Drops zero differences (Wilcoxon's reduced-sample convention; matches
    scipy's default `zero_method="wilcox"`).

    Use only when n >= 10. For small n the normal approximation is rough;
    callers gate the integration at n >= 10 per the Phase 8 contract.

    Args:
        a, b: equal-length sequences of paired observations.

    Returns:
        WilcoxonResult. p_value is in [0, 1]; statistic is the sum of ranks
        of positive differences (W+).
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples must be same length; got {len(a)} vs {len(b)}")

    diffs = [x - y for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)

    if n == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, z=0.0, n=0)

    abs_diffs = [abs(d) for d in nonzero]
    ranks, tie_correction = _rank_with_ties(abs_diffs)

    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0

    if var_w <= 0:
        # All differences identical — degenerate but well-defined: any non-
        # zero w_plus implies p=1 (no information); zero implies p=1 too.
        return WilcoxonResult(statistic=w_plus, p_value=1.0, z=0.0, n=n)

    # Continuity correction toward the mean.
    diff_from_mean = w_plus - mean_w
    if diff_from_mean > 0:
        z = (diff_from_mean - 0.5) / math.sqrt(var_w)
    elif diff_from_mean < 0:
        z = (diff_from_mean + 0.5) / math.sqrt(var_w)
    else:
        z = 0.0

    # Two-sided p-value via standard-normal survival function.
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    p_value = max(0.0, min(1.0, p_value))

    return WilcoxonResult(statistic=w_plus, p_value=p_value, z=z, n=n)


def _normal_cdf(z: float) -> float:
    """Standard-normal cumulative distribution function via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval (percentile method)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_iterations: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile-method bootstrap CI for the mean.

    Resamples `values` with replacement `n_iterations` times, computes the
    mean of each resample, and returns the (lower, upper) percentile bounds
    for the requested confidence level.

    Returns (mean, mean) when given fewer than 2 values — a degenerate CI is
    more useful than raising for callers that just want to render a band.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive; got {n_iterations}")

    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    means = [0.0] * n_iterations
    for i in range(n_iterations):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means[i] = s / n
    means.sort()

    alpha = 1.0 - confidence
    low_idx = int(round((alpha / 2.0) * n_iterations))
    high_idx = int(round((1.0 - alpha / 2.0) * n_iterations)) - 1
    low_idx = max(0, min(n_iterations - 1, low_idx))
    high_idx = max(0, min(n_iterations - 1, high_idx))

    return (means[low_idx], means[high_idx])


# ---------------------------------------------------------------------------
# Cohen's r effect size for Wilcoxon
# ---------------------------------------------------------------------------

def cohens_r(z: float, n: int) -> float:
    """Cohen's r effect size: r = z / sqrt(n).

    Conventional thresholds (Cohen 1988): 0.1 small, 0.3 medium, 0.5 large.
    Returns the absolute magnitude — sign of effect is already captured in
    the z-score and the CER deltas.
    """
    if n <= 0:
        return 0.0
    return abs(z) / math.sqrt(n)
