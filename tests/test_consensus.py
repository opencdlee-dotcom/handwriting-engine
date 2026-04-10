"""Tests for consensus engine — all strategies, edge cases, and the cascade."""

from unittest.mock import patch, MagicMock
import pytest

from handwriting_engine.consensus import (
    read_with_consensus,
    _single_text_confidence,
    _estimate_confidence,
    _word_diff,
    _word_agreement_ratio,
    _word_level_vote,
    _count_uncertainty_markers,
    PROVIDER_WEIGHTS,
)


# --- _single_text_confidence ---

def test_confidence_empty():
    assert _single_text_confidence("") == 0.0
    assert _single_text_confidence("   ") == 0.0


def test_confidence_normal_text():
    score = _single_text_confidence("The student wrote a detailed answer about mitosis and cell division.")
    assert score > 0.5
    # Single-provider reads are capped at 0.70
    assert score <= 0.70


def test_confidence_uncertain_text():
    score = _single_text_confidence("The [?] wrote [illegible] about [unclear]")
    # Should be penalized heavily for uncertainty markers
    assert score < 0.5


def test_confidence_short_text():
    score = _single_text_confidence("yes")
    # Short but not empty — penalized for brevity
    assert 0.0 < score < 0.6


def test_estimate_confidence_legacy_alias():
    """_estimate_confidence is a backwards-compatible alias."""
    assert _estimate_confidence("hello world test") == _single_text_confidence("hello world test")


# --- _word_diff ---

def test_word_diff_identical():
    assert _word_diff("hello world", "hello world") == []


def test_word_diff_single_word():
    diffs = _word_diff("the cat sat", "the cut sat")
    assert len(diffs) >= 1
    found = any("cat" in d["word_a"] and "cut" in d["word_b"] for d in diffs)
    assert found


def test_word_diff_capped_at_30():
    a = " ".join(f"word{i}" for i in range(50))
    b = " ".join(f"other{i}" for i in range(50))
    diffs = _word_diff(a, b)
    assert len(diffs) <= 30


# --- _word_agreement_ratio ---

def test_agreement_identical():
    ratio = _word_agreement_ratio("the cat sat on the mat", "the cat sat on the mat")
    assert ratio == 1.0


def test_agreement_completely_different():
    ratio = _word_agreement_ratio("hello world", "goodbye moon")
    assert ratio < 0.3


def test_agreement_partial():
    ratio = _word_agreement_ratio("the cat sat", "the dog sat")
    assert 0.3 < ratio < 1.0


def test_agreement_empty():
    assert _word_agreement_ratio("", "") == 1.0
    assert _word_agreement_ratio("hello", "") == 0.0


# --- _word_level_vote ---

def test_vote_unanimous():
    texts = ["the cat sat", "the cat sat"]
    weights = [("a", 1.0), ("b", 1.0)]
    text, disagreements, confidence = _word_level_vote(texts, weights)
    assert "the" in text and "cat" in text and "sat" in text
    assert confidence > 0.9
    assert len(disagreements) == 0


def test_vote_disagreement_marks_alt():
    texts = ["the cat sat", "the dog sat"]
    weights = [("a", 1.5), ("b", 1.0)]
    text, disagreements, confidence = _word_level_vote(texts, weights)
    # Higher-weighted provider wins
    assert "cat" in text
    assert len(disagreements) > 0
    assert confidence < 1.0


def test_vote_three_providers_majority():
    texts = ["the cat sat", "the cat sat", "the dog sat"]
    weights = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
    text, disagreements, confidence = _word_level_vote(texts, weights)
    # Majority says "cat"
    assert "cat" in text


# --- read_with_consensus with mocked providers ---

def _mock_provider(name, response):
    """Create a mock provider that returns given response."""
    mock = MagicMock()
    mock.name = name
    mock.read_image.return_value = response
    mock.usage = {"input_tokens": 100, "output_tokens": 50}
    return mock


def test_best_of_strategy():
    mock_claude = _mock_provider("claude", "test response with enough words for decent confidence")

    with patch("handwriting_engine.consensus.available_providers", return_value=["claude"]), \
         patch("handwriting_engine.consensus.get_provider", return_value=mock_claude):
        result = read_with_consensus("b64data", "image/jpeg", "read this", strategy="best_of")

    assert result.text == "test response with enough words for decent confidence"
    assert result.strategy_used == "best_of"
    # Confidence is now derived from text signals, capped at 0.70
    assert 0.0 < result.confidence <= 0.70


def test_best_of_routing():
    """best_of should route handwriting to gemini (2026 benchmarks)."""
    mock_gemini = _mock_provider("gemini", "handwriting result from gemini")

    with patch("handwriting_engine.consensus.available_providers", return_value=["gemini", "claude"]), \
         patch("handwriting_engine.consensus.get_provider", return_value=mock_gemini):
        result = read_with_consensus(
            "b64data", "image/jpeg", "read this",
            strategy="best_of", content_type="handwriting",
        )

    assert result.text == "handwriting result from gemini"


def test_vote_single_fallback():
    mock = _mock_provider("claude", "solo response with several words here")

    with patch("handwriting_engine.consensus.available_providers", return_value=["claude"]), \
         patch("handwriting_engine.consensus.get_provider", return_value=mock):
        result = read_with_consensus("b64", "image/jpeg", "read", strategy="vote")

    assert result.text == "solo response with several words here"
    assert result.strategy_used == "vote_single_fallback"
    # Single provider capped at 0.70
    assert result.confidence <= 0.70


def test_vote_merges_similar():
    """Two similar responses should vote with high confidence."""
    providers_map = {
        "openai": _mock_provider("openai", "The student answered: mitosis"),
        "claude": _mock_provider("claude", "The student answered: meiosis"),
    }

    with patch("handwriting_engine.consensus.available_providers", return_value=["openai", "claude"]), \
         patch("handwriting_engine.consensus.get_provider", side_effect=lambda n: providers_map[n]):
        result = read_with_consensus(
            "b64", "image/jpeg", "read", strategy="vote",
            content_type="handwriting",
        )

    assert result.strategy_used == "vote"
    assert result.confidence > 0.5
    assert len(result.disagreements) > 0


def test_vote_all_providers_fail():
    mock = MagicMock()
    mock.read_image.side_effect = RuntimeError("fail")
    mock.usage = {}

    with patch("handwriting_engine.consensus.available_providers", return_value=["openai", "claude"]), \
         patch("handwriting_engine.consensus.get_provider", return_value=mock):
        result = read_with_consensus("b64", "image/jpeg", "read", strategy="vote")

    assert result.text == ""
    assert result.confidence == 0.0


def test_debate_agree():
    """When both providers produce identical text, should return high confidence."""
    proposer = _mock_provider("openai", "The cat sat on the mat")
    critic = _mock_provider("claude", "The cat sat on the mat")

    def get_prov(name):
        if name == "openai":
            return proposer
        return critic

    with patch("handwriting_engine.consensus.available_providers", return_value=["openai", "claude"]), \
         patch("handwriting_engine.consensus.get_provider", side_effect=get_prov):
        result = read_with_consensus(
            "b64", "image/jpeg", "read",
            strategy="debate", providers=["openai", "claude"],
        )

    assert result.strategy_used == "debate_agreed"
    assert result.confidence >= 0.9


def test_debate_disagree_resolves():
    """When providers disagree, should go through resolution phase."""
    call_count = [0]

    class CriticProvider:
        name = "claude"
        _usage = {"input_tokens": 100, "output_tokens": 50}

        def read_image(self, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Independent read — different from proposer
                return "The cat sat on the rug"
            # Resolution phase
            return "The cat sat on the mat"

        @property
        def usage(self):
            return dict(self._usage)

    proposer = _mock_provider("openai", "The cat sat on the mat")

    def get_prov(name):
        if name == "openai":
            return proposer
        return CriticProvider()

    with patch("handwriting_engine.consensus.available_providers", return_value=["openai", "claude"]), \
         patch("handwriting_engine.consensus.get_provider", side_effect=get_prov):
        result = read_with_consensus(
            "b64", "image/jpeg", "read",
            strategy="debate", providers=["openai", "claude"],
        )

    assert result.strategy_used == "debate_resolved"
    # Confidence derived from agreement, not hardcoded
    assert 0.5 < result.confidence < 0.95


def test_cascade_stops_at_confident():
    """Cascade should stop at first provider that meets threshold."""
    long_text = (
        "The student wrote a detailed multi-paragraph answer about mitosis and cell division. "
        "The answer covers all phases including prophase, metaphase, anaphase, and telophase. "
        "Each phase is described with correct terminology and appropriate scientific detail."
    )
    gemini = _mock_provider("gemini", long_text)
    openai = _mock_provider("openai", "should not be called")

    def get_prov(name):
        if name == "gemini":
            return gemini
        return openai

    with patch("handwriting_engine.consensus.available_providers", return_value=["gemini", "openai"]), \
         patch("handwriting_engine.consensus.get_provider", side_effect=get_prov):
        result = read_with_consensus(
            "b64", "image/jpeg", "read",
            strategy="cascade", confidence_threshold=0.5,
        )

    assert "cascade_gemini" in result.strategy_used
    openai.read_image.assert_not_called()


def test_cascade_escalates_on_low_confidence():
    """Cascade should escalate when first provider returns low-confidence result."""
    gemini = _mock_provider("gemini", "[?] [illegible] [unclear]")  # Low confidence
    openai = _mock_provider("openai", "The student clearly wrote mitosis as the answer to question five")

    def get_prov(name):
        if name == "gemini":
            return gemini
        return openai

    with patch("handwriting_engine.consensus.available_providers", return_value=["gemini", "openai"]), \
         patch("handwriting_engine.consensus.get_provider", side_effect=get_prov):
        result = read_with_consensus(
            "b64", "image/jpeg", "read",
            strategy="cascade", confidence_threshold=0.7,
        )

    # Should have escalated past gemini
    assert "openai" in result.strategy_used or "best_effort" in result.strategy_used


def test_no_providers_raises():
    with patch("handwriting_engine.consensus.available_providers", return_value=[]):
        with pytest.raises(ValueError, match="No vision providers"):
            read_with_consensus("b64", "image/jpeg", "read", strategy="best_of")

        with pytest.raises(ValueError, match="No vision providers"):
            read_with_consensus("b64", "image/jpeg", "read", strategy="vote")

        with pytest.raises(ValueError, match="No vision providers"):
            read_with_consensus("b64", "image/jpeg", "read", strategy="debate")

        with pytest.raises(ValueError, match="No vision providers"):
            read_with_consensus("b64", "image/jpeg", "read", strategy="cascade")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        read_with_consensus("b64", "image/jpeg", "read", strategy="invalid")


def test_provider_weights_structure():
    """All weight dicts should contain the three provider names."""
    for content_type, weights in PROVIDER_WEIGHTS.items():
        assert "claude" in weights
        assert "openai" in weights
        assert "gemini" in weights


# --- _count_uncertainty_markers ---

def test_count_uncertainty_markers_none():
    assert _count_uncertainty_markers("The student wrote mitosis") == 0


def test_count_uncertainty_markers_question():
    assert _count_uncertainty_markers("The [?] wrote [illegible: 3 chars]") == 2


def test_count_uncertainty_markers_multiple():
    text = "[?] [?] [unclear] ??? [illegible]"
    count = _count_uncertainty_markers(text)
    assert count >= 3


# --- self_correct strategy ---

class TestSelfCorrectStrategy:

    def test_self_correct_calls_provider_twice(self):
        """self_correct must call read_image twice: initial read + correction."""
        call_count = [0]
        responses = [
            "The [?] wrote about mitosis",  # initial read with uncertainty
            "The student wrote about mitosis",  # corrected read
        ]

        class CountingProvider:
            name = "gemini"
            _usage = {"input_tokens": 100, "output_tokens": 50}

            def read_image(self, *args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                return responses[idx] if idx < len(responses) else responses[-1]

            @property
            def usage(self):
                return dict(self._usage)

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=CountingProvider()):
            result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

        assert call_count[0] == 2  # Initial + correction
        assert result.strategy_used == "self_correct_1pass"
        assert result.text == "The student wrote about mitosis"

    def test_self_correct_no_extra_pass_when_clean(self):
        """If initial read has no [?] markers, correction still runs once (max_rounds=1)."""
        call_count = [0]

        class CleanProvider:
            name = "gemini"
            _usage = {"input_tokens": 100, "output_tokens": 50}

            def read_image(self, *args, **kwargs):
                call_count[0] += 1
                return "The student wrote about mitosis"

            @property
            def usage(self):
                return dict(self._usage)

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=CleanProvider()):
            result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

        assert call_count[0] == 2  # Always does at least 1 correction pass
        assert result.strategy_used == "self_correct_1pass"

    def test_self_correct_max_rounds_respected(self):
        """max_self_correct_rounds=2 should do up to 2 correction passes."""
        call_count = [0]
        responses = [
            "The [?] wrote [?] mitosis",    # initial
            "The [?] wrote about mitosis",   # round 1 — still has [?]
            "The student wrote about mitosis",  # round 2 — clean
        ]

        class MultiRoundProvider:
            name = "gemini"
            _usage = {"input_tokens": 100, "output_tokens": 50}

            def read_image(self, *args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                return responses[idx] if idx < len(responses) else responses[-1]

            @property
            def usage(self):
                return dict(self._usage)

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=MultiRoundProvider()):
            result = read_with_consensus(
                "b64", "image/jpeg", "read",
                strategy="self_correct", max_self_correct_rounds=2
            )

        assert call_count[0] == 3  # initial + 2 correction passes
        assert "2pass" in result.strategy_used
        assert result.text == "The student wrote about mitosis"

    def test_self_correct_strategy_used_field(self):
        """strategy_used should reflect self_correct."""
        mock = MagicMock()
        mock.name = "gemini"
        mock.read_image.return_value = "clean text without markers here"
        mock.usage = {"input_tokens": 50, "output_tokens": 25}

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=mock):
            result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

        assert "self_correct" in result.strategy_used

    def test_self_correct_provider_results_has_initial_and_corrected(self):
        """provider_results must contain 'initial' and 'corrected' keys."""
        call_count = [0]

        class TwoPassProvider:
            name = "gemini"
            _usage = {"input_tokens": 100, "output_tokens": 50}

            def read_image(self, *args, **kwargs):
                call_count[0] += 1
                return "initial text" if call_count[0] == 1 else "corrected text"

            @property
            def usage(self):
                return dict(self._usage)

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=TwoPassProvider()):
            result = read_with_consensus("b64", "image/jpeg", "read", strategy="self_correct")

        assert "initial" in result.provider_results
        assert "corrected" in result.provider_results
        assert result.provider_results["initial"] == "initial text"
        assert result.provider_results["corrected"] == "corrected text"


# --- smart strategy escalation ---

class TestSmartEscalation:

    def test_smart_escalates_on_high_uncertainty(self):
        """smart strategy should escalate to self_correct when [?] count > threshold."""
        call_count = [0]
        responses = [
            "[?] wrote [?] about [?] and [?] more",  # initial easy read — 4 [?] markers
            "The student wrote about mitosis and cell division",  # self-correction
        ]

        class EscalatingProvider:
            name = "gemini"
            _usage = {"input_tokens": 100, "output_tokens": 50}

            def read_image(self, *args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                return responses[idx] if idx < len(responses) else responses[-1]

            @property
            def usage(self):
                return dict(self._usage)

        good_quality = {"quality": "good", "blur_score": 200, "contrast_score": 0.7, "faint_ink": False, "issues": []}

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=EscalatingProvider()):
            result = read_with_consensus(
                "b64", "image/jpeg", "read",
                strategy="smart",
                quality_assessment=good_quality,
                uncertainty_threshold=3,
            )

        assert "self_correct" in result.strategy_used

    def test_smart_no_escalation_on_clean_output(self):
        """smart strategy should NOT escalate when output is clean."""
        mock = MagicMock()
        mock.name = "gemini"
        mock.read_image.return_value = "The student answered mitosis correctly"
        mock.usage = {"input_tokens": 100, "output_tokens": 50}

        good_quality = {"quality": "good", "blur_score": 200, "contrast_score": 0.7, "faint_ink": False, "issues": []}

        with patch("handwriting_engine.consensus.available_providers", return_value=["gemini"]), \
             patch("handwriting_engine.consensus.get_provider", return_value=mock):
            result = read_with_consensus(
                "b64", "image/jpeg", "read",
                strategy="smart",
                quality_assessment=good_quality,
                uncertainty_threshold=3,
            )

        # Should be best_of (single read), not self_correct
        assert "self_correct" not in result.strategy_used
        # Only called once (best_of, no escalation)
        assert mock.read_image.call_count == 1
