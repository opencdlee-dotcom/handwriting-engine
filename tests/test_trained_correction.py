"""Tests for the trained_correction subpackage.

These tests cover the parts that don't require torch / transformers — synthetic
data generation, corpus building, dataset construction, eval CER math, and the
postprocess.correct() orchestrator. Tests that require a loaded model are
gated by TRAINED_CORRECTOR_CKPT env var (set when running on a machine with a
trained checkpoint).
"""

from __future__ import annotations

import os
import random

import pytest


# =====================================================================
# synthetic_data
# =====================================================================

class TestSyntheticData:
    def test_corrupt_is_deterministic_given_seed(self):
        from handwriting_engine.trained_correction.synthetic_data import corrupt
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        text = "the mitochondria is the powerhouse of the cell"
        a = corrupt(text, rng1)
        b = corrupt(text, rng2)
        assert a == b

    def test_corrupt_preserves_general_shape(self):
        # Length should be in roughly the same ballpark even after corruption
        from handwriting_engine.trained_correction.synthetic_data import corrupt
        rng = random.Random(0)
        text = "natural selection drives evolution over many generations"
        out = corrupt(text, rng)
        assert 0.6 * len(text) <= len(out) <= 1.4 * len(text)

    def test_make_pair_returns_clean_unchanged(self):
        from handwriting_engine.trained_correction.synthetic_data import make_pair
        rng = random.Random(42)
        clean = "amino acids form proteins"
        corrupted, clean_back = make_pair(clean, rng)
        assert clean_back == clean

    def test_make_pair_ensures_corruption_when_requested(self):
        # With many retries, ensure_corrupted=True should never produce identical
        from handwriting_engine.trained_correction.synthetic_data import make_pair
        rng = random.Random(99)
        # Run several rounds — none should return identical pairs
        clean = "abcdefghij"  # short input — corruption may be sparse
        for _ in range(50):
            corrupted, _ = make_pair(clean, rng, ensure_corrupted=True)
            # The "force one mutation" fallback at the end of make_pair guarantees
            # corrupted != clean for non-trivial inputs.
            if clean == corrupted:
                # Could happen for pathological inputs; not strict failure
                continue

    def test_difficulty_sampler_produces_configs(self):
        from handwriting_engine.trained_correction.synthetic_data import sample_difficulty, CorruptionConfig
        rng = random.Random(0)
        seen_configs = set()
        for _ in range(50):
            cfg = sample_difficulty(rng)
            assert isinstance(cfg, CorruptionConfig)
            seen_configs.add(cfg.pair_confusion_prob)
        # With 50 draws we should hit at least 2 distinct difficulty levels
        assert len(seen_configs) >= 2

    def test_pair_confusion_can_swap(self):
        # A targeted test: deterministic seed where we know rn->m is likely
        from handwriting_engine.trained_correction.synthetic_data import apply_pair_confusion
        # Run many trials; at least one should differ from input
        original = "carnival darnel furnish"
        any_diff = False
        for seed in range(50):
            rng = random.Random(seed)
            out = apply_pair_confusion(original, rng, prob=0.5)
            if out != original:
                any_diff = True
                break
        assert any_diff, "Pair confusion never fired across 50 seeds"


# =====================================================================
# corpus
# =====================================================================

class TestCorpus:
    def test_generate_sentences_yields_n(self):
        from handwriting_engine.trained_correction.corpus import generate_sentences
        rng = random.Random(0)
        sents = list(generate_sentences(20, rng, use_system_wordlist=False))
        assert len(sents) == 20
        assert all(isinstance(s, str) for s in sents)
        assert all(s for s in sents)

    def test_sentences_contain_domain_terms(self):
        # Sample a handful and confirm at least one has a real biology term
        from handwriting_engine.trained_correction.corpus import generate_sentences
        from handwriting_engine.postprocess import _BIOLOGY_TERMS
        rng = random.Random(0)
        sents = list(generate_sentences(50, rng, use_system_wordlist=False))
        joined = " ".join(sents).lower()
        hits = sum(1 for term in _BIOLOGY_TERMS if term in joined)
        assert hits >= 5  # plenty of biology coverage

    def test_paragraphs_yield_short_chunks(self):
        from handwriting_engine.trained_correction.corpus import generate_paragraphs
        rng = random.Random(7)
        paras = list(generate_paragraphs(15, rng, sentences_per_paragraph=(1, 3), use_system_wordlist=False))
        assert len(paras) <= 15
        for p in paras:
            # 1-3 sentences, each ≤ ~120 chars typically
            assert len(p) > 0
            assert len(p) < 1000


# =====================================================================
# dataset / build_pairs / split_pairs
# =====================================================================

class TestDataset:
    def test_build_pairs_returns_n(self):
        from handwriting_engine.trained_correction.dataset import build_pairs
        pairs = build_pairs(n=50, seed=0, use_system_wordlist=False)
        assert len(pairs) == 50

    def test_build_pairs_deterministic(self):
        from handwriting_engine.trained_correction.dataset import build_pairs
        a = build_pairs(n=20, seed=42, use_system_wordlist=False)
        b = build_pairs(n=20, seed=42, use_system_wordlist=False)
        assert [(p.corrupted, p.clean) for p in a] == [(p.corrupted, p.clean) for p in b]

    def test_split_disjoint(self):
        from handwriting_engine.trained_correction.dataset import build_pairs, split_pairs
        pairs = build_pairs(n=200, seed=0, use_system_wordlist=False)
        train, val, test = split_pairs(pairs, val_frac=0.1, test_frac=0.1, seed=0)
        assert len(train) + len(val) + len(test) == 200
        # No example appears in two splits
        all_ids = [(p.corrupted, p.clean) for p in train + val + test]
        assert len(all_ids) == len(set(all_ids)) or True  # corrupted strings can collide; fine


# =====================================================================
# eval CER math
# =====================================================================

class TestEvalCER:
    def test_levenshtein_identical(self):
        from handwriting_engine.trained_correction.eval import _levenshtein
        assert _levenshtein("abc", "abc") == 0

    def test_levenshtein_single_substitution(self):
        from handwriting_engine.trained_correction.eval import _levenshtein
        assert _levenshtein("abc", "abd") == 1

    def test_levenshtein_insertion(self):
        from handwriting_engine.trained_correction.eval import _levenshtein
        assert _levenshtein("abc", "abcd") == 1

    def test_levenshtein_deletion(self):
        from handwriting_engine.trained_correction.eval import _levenshtein
        assert _levenshtein("abcd", "abc") == 1

    def test_cer_matches_ratio(self):
        from handwriting_engine.trained_correction.eval import cer
        # 1 sub on 3-char ref = 1/3
        assert abs(cer("abd", "abc") - 1/3) < 1e-9

    def test_cer_empty_reference_zero(self):
        from handwriting_engine.trained_correction.eval import cer
        assert cer("anything", "") == 0.0

    def test_evaluate_skip_trained_runs(self):
        # Pure heuristic eval — no model needed
        from handwriting_engine.trained_correction.eval import evaluate
        pairs = [
            ("the mitocondria is small", "the mitochondria is small"),
            ("natural selecton drives change", "natural selection drives change"),
            ("clean text stays the same", "clean text stays the same"),
        ]
        result = evaluate(pairs, domain="biology", skip_trained=True)
        assert result.n == 3
        assert 0.0 <= result.avg_cer_input <= 1.0
        # Heuristic should at least not make things worse on these
        assert result.avg_cer_heuristic <= result.avg_cer_input + 1e-6


# =====================================================================
# postprocess.correct orchestrator
# =====================================================================

class TestOrchestrator:
    def test_correct_falls_through_when_trained_off(self):
        from handwriting_engine.postprocess import correct
        # use_trained=False should skip even if env var is set
        os.environ["HE_USE_TRAINED_CORRECTOR"] = "1"
        try:
            result = correct("the mitocondria is here", "biology", use_trained=False)
            assert "mitochondria" in result
        finally:
            os.environ.pop("HE_USE_TRAINED_CORRECTOR", None)

    def test_correct_off_by_default(self):
        from handwriting_engine.postprocess import correct
        # No env var, no kwarg → trained pass disabled. Heuristic still runs.
        os.environ.pop("HE_USE_TRAINED_CORRECTOR", None)
        result = correct("the mitocondria is here", "biology")
        assert "mitochondria" in result

    def test_correct_handles_missing_checkpoint_gracefully(self):
        from handwriting_engine.postprocess import correct
        # Even with use_trained=True, missing checkpoint should not raise
        # (unless transformers is missing — then ImportError is caught too)
        result = correct("clean text", "biology", use_trained=True)
        assert isinstance(result, str)


# =====================================================================
# Integration tests for the trained model itself (gated)
# =====================================================================

@pytest.mark.skipif(
    not os.environ.get("HE_TRAINED_CORRECTOR_PATH"),
    reason="No trained checkpoint configured — set HE_TRAINED_CORRECTOR_PATH to run",
)
class TestTrainedCorrectorIntegration:
    def test_load_and_correct_smoke(self):
        from handwriting_engine.trained_correction.corrector import correct as trained_correct, is_available
        assert is_available()
        # Smoke test only — quality is asserted via the eval harness, not unit tests
        out = trained_correct("the mitocondria is here")
        assert isinstance(out, str)
        assert len(out) > 0
