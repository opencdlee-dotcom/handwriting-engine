"""A/B evaluation harness — heuristic vs trained vs combined post-correction.

Reports CER (character error rate) on a held-out synthetic test set or on a
list of `(corrupted, clean)` pairs the caller supplies (e.g., from the
benchmark DB once Phase 7 lands).

CER computed via character-level Levenshtein distance / reference length.
Pure-stdlib implementation — `jiwer` is the optional benchmark dep but we
don't want to require it here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from handwriting_engine.postprocess import correct as full_correct, correct_domain_terms

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur[0] = i
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, dele, sub)
        prev, cur = cur, prev
    return prev[len(b)]


def cer(prediction: str, reference: str) -> float:
    """Character error rate. Returns 0.0 for empty reference (treat as N/A)."""
    if not reference:
        return 0.0
    return _levenshtein(prediction, reference) / len(reference)


@dataclass
class EvalResult:
    n: int
    avg_cer_input: float          # CER of corrupted vs clean (baseline)
    avg_cer_heuristic: float      # CER after heuristic correction
    avg_cer_trained: float        # CER after trained correction (raw, no gates)
    avg_cer_combined: float       # CER after heuristic THEN trained (raw, no gates)
    avg_cer_combined_gated: float # CER after heuristic + trained with confidence gate + fidelity check
    fidelity_rejections: int      # how many times the fidelity check fired
    confidence_skips: int         # how many times the confidence gate skipped trained pass

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "avg_cer_input": round(self.avg_cer_input, 4),
            "avg_cer_heuristic": round(self.avg_cer_heuristic, 4),
            "avg_cer_trained_raw": round(self.avg_cer_trained, 4),
            "avg_cer_combined_raw": round(self.avg_cer_combined, 4),
            "avg_cer_combined_gated": round(self.avg_cer_combined_gated, 4),
            "delta_heuristic_vs_input": round(self.avg_cer_heuristic - self.avg_cer_input, 4),
            "delta_combined_raw_vs_heuristic": round(self.avg_cer_combined - self.avg_cer_heuristic, 4),
            "delta_combined_gated_vs_heuristic": round(self.avg_cer_combined_gated - self.avg_cer_heuristic, 4),
            "delta_gated_vs_raw": round(self.avg_cer_combined_gated - self.avg_cer_combined, 4),
            "fidelity_rejections": self.fidelity_rejections,
            "confidence_skips": self.confidence_skips,
        }


def evaluate(
    pairs: Iterable[tuple[str, str]],
    domain: str = "biology",
    skip_trained: bool = False,
    fidelity_threshold: float = 0.35,
) -> EvalResult:
    """Evaluate all four pipelines on each (corrupted, clean) pair.

    Tracks both raw combined (heuristic → trained, no safeguards) and
    gated combined (with confidence gate + fidelity check) so callers can
    see what each safeguard buys.
    """
    pairs = list(pairs)
    if not pairs:
        return EvalResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    from handwriting_engine.postprocess import _change_ratio

    cer_in = []
    cer_heur = []
    cer_trained = []
    cer_combined = []
    cer_combined_gated = []
    fidelity_rejections = 0
    confidence_skips = 0

    # Lazy: trained corrector loads on first call
    trained_corrector = None
    if not skip_trained:
        try:
            from handwriting_engine.trained_correction.corrector import get_default_corrector
            trained_corrector = get_default_corrector()
        except ImportError:
            trained_corrector = None

    for i, (corrupted, clean) in enumerate(pairs):
        cer_in.append(cer(corrupted, clean))

        heur = correct_domain_terms(corrupted, domain)
        cer_heur.append(cer(heur, clean))

        if trained_corrector is not None:
            tr = trained_corrector.correct(corrupted)
            cer_trained.append(cer(tr, clean))
            both_raw = trained_corrector.correct(heur)
            cer_combined.append(cer(both_raw, clean))

            # Gated combined: confidence gate + fidelity check
            heuristic_made_changes = heur != corrupted
            if not heuristic_made_changes:
                # Confidence gate: skip trained pass
                gated = heur
                confidence_skips += 1
            else:
                if _change_ratio(heur, both_raw) > fidelity_threshold:
                    gated = heur
                    fidelity_rejections += 1
                else:
                    gated = both_raw
            cer_combined_gated.append(cer(gated, clean))
        else:
            cer_trained.append(cer_in[-1])
            cer_combined.append(cer_heur[-1])
            cer_combined_gated.append(cer_heur[-1])

        if (i + 1) % 100 == 0:
            logger.info("Evaluated %d / %d", i + 1, len(pairs))

    n = len(pairs)
    return EvalResult(
        n=n,
        avg_cer_input=sum(cer_in) / n,
        avg_cer_heuristic=sum(cer_heur) / n,
        avg_cer_trained=sum(cer_trained) / n,
        avg_cer_combined=sum(cer_combined) / n,
        avg_cer_combined_gated=sum(cer_combined_gated) / n,
        fidelity_rejections=fidelity_rejections,
        confidence_skips=confidence_skips,
    )


def evaluate_synthetic(
    n_pairs: int = 1000,
    seed: int = 1234,
    domain: str = "biology",
    skip_trained: bool = False,
) -> EvalResult:
    """Generate synthetic pairs and evaluate. Uses a different seed than training
    so the eval set is held out from anything the trained model saw."""
    from handwriting_engine.trained_correction.dataset import build_pairs
    pairs_obj = build_pairs(n=n_pairs, seed=seed)
    pairs = [(p.corrupted, p.clean) for p in pairs_obj]
    return evaluate(pairs, domain=domain, skip_trained=skip_trained)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B evaluate post-correction pipelines.")
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--domain", default="biology")
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = evaluate_synthetic(
        n_pairs=args.n_pairs,
        seed=args.seed,
        domain=args.domain,
        skip_trained=args.skip_trained,
    )

    print(json.dumps(result.to_dict(), indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
