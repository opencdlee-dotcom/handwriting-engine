"""Tests for benchmark metrics — CER, WER, Levenshtein distance."""

import math
import pytest
from handwriting_engine.benchmark.metrics import (
    character_error_rate,
    levenshtein_distance,
    normalize_text,
    word_error_rate,
)


class TestLevenshteinDistance:
    def test_identical(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_insert(self):
        assert levenshtein_distance("abc", "abcd") == 1

    def test_single_delete(self):
        assert levenshtein_distance("abcd", "abc") == 1

    def test_single_replace(self):
        assert levenshtein_distance("abc", "axc") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_known_distance(self):
        # "kitten" -> "sitting" = 3 edits
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_works_on_lists(self):
        # Word-level distance
        assert levenshtein_distance(["the", "cat"], ["the", "dog"]) == 1
        assert levenshtein_distance(["a", "b", "c"], ["a", "c"]) == 1


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello   world\n\tfoo") == "hello world foo"

    def test_strip_uncertainty_markers(self):
        assert normalize_text("the [?] answer is [unclear] yes") == "the answer is yes"
        assert normalize_text("read [?alt: B/D] here") == "read here"
        assert normalize_text("text [illegible: ~3 chars] more") == "text more"

    def test_strip_triple_question_marks(self):
        assert normalize_text("the ??? answer") == "the answer"

    def test_strip_unable_to_read(self):
        assert normalize_text("text unable to read more") == "text more"

    def test_strip_diagram_markers(self):
        assert normalize_text("before [diagram: cell structure] after") == "before after"
        assert normalize_text("see [graph: bar chart] here") == "see here"

    def test_strip_margin_and_inserted(self):
        assert normalize_text("text [margin: note here] more") == "text more"
        assert normalize_text("text [inserted: correction] more") == "text more"

    def test_strip_empty_marker(self):
        assert normalize_text("cell [empty] data") == "cell data"

    def test_strip_crossed_out(self):
        assert normalize_text("text [crossed out: old word] new") == "text new"

    def test_unicode_nfc_normalization(self):
        # cafe + combining acute vs precomposed café
        assert normalize_text("caf\u0065\u0301") == normalize_text("caf\u00e9")

    def test_combined(self):
        assert normalize_text("  Hello  [?]  World  ") == "hello world"


class TestCharacterErrorRate:
    def test_perfect(self):
        cer, edits, ref_len = character_error_rate("hello", "hello")
        assert cer == 0.0
        assert edits == 0
        assert ref_len == 5

    def test_single_error(self):
        # "hxllo" vs "hello" = 1 edit / 5 chars = 0.2
        cer, edits, ref_len = character_error_rate("hxllo", "hello")
        assert cer == pytest.approx(0.2)
        assert edits == 1

    def test_empty_reference(self):
        cer, _, _ = character_error_rate("some text", "")
        assert cer == float("inf")

    def test_both_empty(self):
        cer, edits, ref_len = character_error_rate("", "")
        assert cer == 0.0

    def test_hallucination_exceeds_one(self):
        # Hypothesis much longer than reference -> CER > 1.0
        cer, _, _ = character_error_rate("this is a very long hallucinated text", "hi")
        assert cer > 1.0

    def test_normalization_applied(self):
        # Case difference should not count as errors
        cer, _, _ = character_error_rate("Hello World", "hello world")
        assert cer == 0.0

    def test_uncertainty_markers_stripped(self):
        cer, _, _ = character_error_rate("the [?] answer", "the answer")
        assert cer == 0.0

    def test_unicode_nfc(self):
        # Composed vs decomposed should be equal
        cer, _, _ = character_error_rate("caf\u00e9", "caf\u0065\u0301")
        assert cer == 0.0

    def test_engine_markers_stripped(self):
        cer, _, _ = character_error_rate("the ??? answer [diagram: box]", "the answer")
        assert cer == 0.0


class TestWordErrorRate:
    def test_perfect(self):
        wer, edits, ref_words = word_error_rate("the cat sat", "the cat sat")
        assert wer == 0.0
        assert edits == 0
        assert ref_words == 3

    def test_one_word_wrong(self):
        # "the dog sat" vs "the cat sat" = 1 edit / 3 words
        wer, edits, ref_words = word_error_rate("the dog sat", "the cat sat")
        assert wer == pytest.approx(1.0 / 3.0)
        assert edits == 1

    def test_empty_reference(self):
        wer, _, _ = word_error_rate("some text", "")
        assert wer == float("inf")

    def test_both_empty(self):
        wer, edits, ref_words = word_error_rate("", "")
        assert wer == 0.0

    def test_extra_words(self):
        # Insertion: "the big cat sat" vs "the cat sat" = 1 edit / 3 words
        wer, edits, _ = word_error_rate("the big cat sat", "the cat sat")
        assert edits == 1

    def test_normalization_applied(self):
        wer, _, _ = word_error_rate("The Cat Sat", "the cat sat")
        assert wer == 0.0


class TestDomainTermAccuracy:
    def test_all_terms_correct(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        ref = "the mitochondria produces atp through respiration"
        hyp = "the mitochondria produces atp through respiration"
        acc, correct, total, missed = domain_term_accuracy(hyp, ref)
        assert acc == 1.0
        assert correct == total
        assert missed == []

    def test_one_term_wrong(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        ref = "the enzyme catalase breaks down peroxide"
        hyp = "the enzyme catolase breaks down peroxide"
        acc, correct, total, missed = domain_term_accuracy(hyp, ref)
        assert "catalase" in missed
        assert acc < 1.0

    def test_no_domain_terms(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jumps over the lazy dog"
        acc, _, total, _ = domain_term_accuracy(hyp, ref)
        assert acc == -1.0
        assert total == 0

    def test_custom_terms(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        custom = {"velocity", "acceleration", "force"}
        ref = "calculate the velocity and force"
        hyp = "calculate the velocity and farce"
        acc, correct, total, missed = domain_term_accuracy(hyp, ref, domain_terms=custom)
        assert total == 2
        assert "force" in missed

    def test_multi_word_terms(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        ref = "the endoplasmic reticulum is connected to the golgi apparatus"
        hyp = "the endoplasmic reticulum is connected to the golgi apparatus"
        acc, correct, total, missed = domain_term_accuracy(hyp, ref)
        assert acc == 1.0
        # Should find both "endoplasmic reticulum" and "golgi apparatus" as multi-word terms
        assert total >= 2

    def test_multi_word_term_missed(self):
        from handwriting_engine.benchmark.metrics import domain_term_accuracy
        ref = "the endoplasmic reticulum produces proteins"
        hyp = "the endoplasmic reticulem produces proteins"
        acc, correct, total, missed = domain_term_accuracy(hyp, ref)
        assert "endoplasmic reticulum" in missed


class TestClassifyErrors:
    def test_no_errors(self):
        from handwriting_engine.benchmark.metrics import classify_errors
        counts = classify_errors("hello world", "hello world")
        assert sum(counts.values()) == 0

    def test_confusion_pair(self):
        from handwriting_engine.benchmark.metrics import classify_errors
        # 'o' vs 'a' is a known confusion pair
        counts = classify_errors("cat", "cot")
        assert counts["confusion_pair"] >= 1

    def test_insertion(self):
        from handwriting_engine.benchmark.metrics import classify_errors
        counts = classify_errors("hello world extra", "hello world")
        assert counts["insertion"] > 0

    def test_deletion(self):
        from handwriting_engine.benchmark.metrics import classify_errors
        counts = classify_errors("hello", "hello world")
        assert counts["deletion"] > 0

    def test_mixed_errors(self):
        from handwriting_engine.benchmark.metrics import classify_errors
        counts = classify_errors("the d0g sat", "the dog sat on the mat")
        total = sum(counts.values())
        assert total > 0
