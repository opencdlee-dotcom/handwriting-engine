"""Tests for S5 char-level consensus resolution.

Falsifies:
- #1 char-level resolves a known confusion case (rn↔m)
- #2 defers cleanly when not a confusion case (returns None)
- #3 writer-specific bias overrides default
+ supporting determinism, ordering, and length-mismatch tests
"""

from handwriting_engine.consensus import (
    resolve_char_level,
    _word_level_vote,
)


class TestResolveCharLevel:
    def test_rn_m_confusion_resolves_to_m_canonical(self):
        # Criterion #1 char-level half: no-majority pair where the only
        # difference is the rn↔m pair → returns the canonical "m" form.
        result = resolve_char_level(["modern", "rnodern"], [1.0, 1.0])
        assert result == "modern"

    def test_unrelated_words_return_none(self):
        # Criterion #2: completely different candidates are NOT a confusion
        # case; defer to the existing [?alt: …] fallback.
        assert resolve_char_level(["apple", "orange"], [1.0, 1.0]) is None

    def test_writer_resolution_overrides_default(self):
        # Criterion #3: a writer who consistently writes 'rn' instead of 'm'
        # gets their preference applied even when global default would pick 'm'.
        result = resolve_char_level(
            ["modern", "rnodern"],
            [1.0, 1.0],
            writer_resolutions={"rn↔m": "rn"},
        )
        assert result == "rnodern"

    def test_writer_resolution_for_m_form(self):
        # Symmetric: writer prefers the 'm' form → still returns 'modern'.
        result = resolve_char_level(
            ["modern", "rnodern"],
            [1.0, 1.0],
            writer_resolutions={"rn↔m": "m"},
        )
        assert result == "modern"

    def test_cl_d_pair(self):
        # cl↔d is a separate well-known confusion pair.
        result = resolve_char_level(["could", "coulcl"], [1.0, 1.0])
        # Default (canonical d): "could" wins.
        assert result == "could"

    def test_identical_candidates_return_canonical(self):
        # When candidates collapse to a single value, that's the answer.
        assert resolve_char_level(["modern", "modern"], [1.0, 1.0]) == "modern"

    def test_three_or_more_candidates_returns_none(self):
        # v0 only handles pairwise. Three distinct candidates defer.
        assert resolve_char_level(
            ["modern", "rnodern", "modarn"], [1.0, 1.0, 1.0]
        ) is None

    def test_provider_order_independent(self):
        # Determinism: swapping the order of candidates must not change the
        # result.
        a = resolve_char_level(["modern", "rnodern"], [1.0, 1.0])
        b = resolve_char_level(["rnodern", "modern"], [1.0, 1.0])
        assert a == b == "modern"

    def test_length_mismatch_handled(self):
        # rn↔m is by definition length-mismatched; the resolver must align
        # via SequenceMatcher rather than positional indexing.
        result = resolve_char_level(["learn", "leamr"], [1.0, 1.0])
        # leamr → learn? "amr" vs "arn" — replace 'm' at idx 2 with 'rn'
        # gives "learn" but the source has 'mr' which doesn't fit cleanly.
        # The candidates aren't a clean rn↔m pair → None.
        assert result is None

    def test_case_sensitive_pair_I_vs_l(self):
        # I↔l is a case-sensitive pair (capital I vs lowercase l).
        result = resolve_char_level(["cell", "ceII"], [1.0, 1.0])
        assert result == "cell"

    def test_empty_candidates_return_none(self):
        assert resolve_char_level([], []) is None

    def test_single_candidate_returns_self(self):
        assert resolve_char_level(["modern"], [1.0]) == "modern"


class TestWordLevelVoteIntegration:
    """End-to-end: when a no-majority disagreement is a confusion pair,
    _word_level_vote should resolve it without emitting [?alt: …]."""

    def test_no_alt_marker_when_char_level_resolves(self):
        # 2 providers, equal weight, "modern" vs "rnodern" — would normally
        # emit [?alt: rnodern]. With char-level resolution it shouldn't.
        text, disagreements, conf = _word_level_vote(
            ["the modern cell", "the rnodern cell"],
            [("gemini", 1.0), ("claude", 1.0)],
        )
        assert "[?alt:" not in text
        assert "modern" in text
        assert "rnodern" not in text

    def test_alt_marker_still_emits_for_unrelated(self):
        # Criterion #2 integration: non-confusion disagreement still gets
        # the [?alt: …] marker.
        text, disagreements, conf = _word_level_vote(
            ["the apple is red", "the orange is red"],
            [("gemini", 1.0), ("claude", 1.0)],
        )
        # One of "apple"/"orange" wins; the other appears in [?alt: …].
        assert "[?alt:" in text

    def test_writer_profile_threaded_through_vote(self):
        # When writer_profile carries a confusion_resolution, it overrides
        # the global default at the word-level voting stage.
        text, _, _ = _word_level_vote(
            ["the modern cell", "the rnodern cell"],
            [("gemini", 1.0), ("claude", 1.0)],
            writer_profile={"confusion_resolutions": {"rn↔m": "rn"}},
        )
        assert "rnodern" in text


class TestWriterProfileEndToEndWireUp:
    """End-to-end: writer_profile passed to consensus.read_with_consensus
    must reach _word_level_vote unchanged (vote and smart strategies).

    Direct text-level assertions are fragile here because provider weights
    bias the vote independent of char-level resolution, so we instead probe
    the _word_level_vote call and assert the kwarg arrived intact.
    """

    def _mock_provider(self, name, response):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.read_image.return_value = response
        mock.usage = {}
        mock.get_mean_confidence = MagicMock(return_value=0.0)
        return mock

    def test_vote_strategy_propagates_writer_profile(self):
        from unittest.mock import patch
        from handwriting_engine.consensus import read_with_consensus

        providers_map = {
            "openai": self._mock_provider("openai", "the modern cell"),
            "claude": self._mock_provider("claude", "the rnodern cell"),
        }
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured["writer_profile"] = kwargs.get("writer_profile")
            return ("the modern cell", [], 0.95)

        with patch("handwriting_engine.consensus.available_providers",
                   return_value=["openai", "claude"]), \
             patch("handwriting_engine.consensus.get_provider",
                   side_effect=lambda n: providers_map[n]), \
             patch("handwriting_engine.consensus._word_level_vote",
                   side_effect=_spy):
            read_with_consensus(
                "b64", "image/jpeg", "read", strategy="vote",
                content_type="handwriting",
                writer_profile={"confusion_resolutions": {"rn↔m": "rn"}},
            )

        assert captured["writer_profile"] == {
            "confusion_resolutions": {"rn↔m": "rn"}
        }

    def test_smart_strategy_no_quality_propagates_writer_profile(self):
        # smart with quality_assessment=None falls back to vote internally.
        from unittest.mock import patch
        from handwriting_engine.consensus import read_with_consensus

        providers_map = {
            "openai": self._mock_provider("openai", "the modern cell"),
            "claude": self._mock_provider("claude", "the rnodern cell"),
        }
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured["writer_profile"] = kwargs.get("writer_profile")
            return ("the modern cell", [], 0.95)

        with patch("handwriting_engine.consensus.available_providers",
                   return_value=["openai", "claude"]), \
             patch("handwriting_engine.consensus.get_provider",
                   side_effect=lambda n: providers_map[n]), \
             patch("handwriting_engine.consensus._word_level_vote",
                   side_effect=_spy):
            read_with_consensus(
                "b64", "image/jpeg", "read", strategy="smart",
                content_type="handwriting",
                writer_profile={"confusion_resolutions": {"rn↔m": "rn"}},
            )

        assert captured["writer_profile"] == {
            "confusion_resolutions": {"rn↔m": "rn"}
        }

    def test_default_no_writer_profile_passes_none(self):
        # Sanity: without writer_profile, the kwarg arrives as None
        # (not as an empty dict or some other sentinel that would mask
        # missing wires).
        from unittest.mock import patch
        from handwriting_engine.consensus import read_with_consensus

        providers_map = {
            "openai": self._mock_provider("openai", "the modern cell"),
            "claude": self._mock_provider("claude", "the modern cell"),
        }
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured["writer_profile"] = kwargs.get("writer_profile", "MISSING")
            return ("the modern cell", [], 0.95)

        with patch("handwriting_engine.consensus.available_providers",
                   return_value=["openai", "claude"]), \
             patch("handwriting_engine.consensus.get_provider",
                   side_effect=lambda n: providers_map[n]), \
             patch("handwriting_engine.consensus._word_level_vote",
                   side_effect=_spy):
            read_with_consensus(
                "b64", "image/jpeg", "read", strategy="vote",
                content_type="handwriting",
            )

        assert captured["writer_profile"] is None
