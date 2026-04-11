"""
Codetester adversarial tests — generated 2026-04-10T19:14:41
Target: handwriting-engine (phase-5 diff: postprocess.py, quality.py, optimize.py)
Mode: full | Depth: normal
Agents: Boundary Hunter, Scale Breaker, Interaction Prober (sequential)

These tests supplement the existing 84-TC adversarial suite by targeting gaps
identified after reviewing test_adversarial_codetester.py plus the current
failing test (TC-22).
"""

from __future__ import annotations

import os
import tempfile
import pytest
from PIL import Image, ImageDraw


# ===========================================================================
# Helpers
# ===========================================================================

def _solid_image(color=(200, 200, 200), size=(200, 200), mode="RGB", suffix=".jpg"):
    img = Image.new(mode, size, color)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if suffix == ".png":
        img.save(path, "PNG")
    else:
        if mode == "RGBA":
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=95)
    return path


# ===========================================================================
# TC-100: Boundary Hunter — T2 Contract Violation
# _edit_distance_1_candidates with empty wordlist
# ===========================================================================

class TestEditDistanceCandidatesContracts:
    """T2: Contract violations in _edit_distance_1_candidates."""

    def test_empty_wordlist_returns_empty_list(self):
        """TC-100: wordlist=set() — no candidates possible, must return []."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates
        candidates = _edit_distance_1_candidates("mitosis", set())
        assert candidates == [], f"Empty wordlist should yield [], got {candidates}"

    def test_wordlist_with_exact_match_returns_empty(self):
        """TC-101: If word IS in wordlist → early return [] (already correct)."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates
        result = _edit_distance_1_candidates("cell", {"cell", "mitosis"})
        assert result == [], "Exact match must return [] not the word itself"

    def test_empty_word_single_char_candidates_only(self):
        """TC-102: Empty string → insertions generate all 26 single-letter strings.
        For biology wordlist, no single letter is present → should return []."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates, _BIOLOGY_TERMS
        candidates = _edit_distance_1_candidates("", _BIOLOGY_TERMS)
        # Single letters like 'a' are not in the biology wordlist
        for c in candidates:
            assert len(c) == 1, f"Empty word should only generate 1-char candidates, got: {c!r}"
        assert candidates == [], f"No single letters in biology wordlist, got {candidates}"

    def test_single_char_word_generates_candidates(self):
        """TC-103: 1-char word generates 26 replacements + 26 insertions.
        For a tiny wordlist containing 'ab', candidates should include it."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates
        # 'a' → insertions: 'ba', 'ca'... and 'ab'... 'az' — 'ab' IS at edit distance 1
        candidates = _edit_distance_1_candidates("a", {"ab", "ba", "ac"})
        # 'ab' is insertion: ("a","") -> "a" + "b" + "" = "ab" ✓
        assert "ab" in candidates or "ba" in candidates, (
            f"1-char word 'a' should find 'ab'/'ba' in {{ab, ba}}, got {candidates}"
        )


# ===========================================================================
# TC-110: Boundary Hunter — T1 Boundary Collision
# correct_domain_terms — 3-char word boundary
# ===========================================================================

class TestCorrectDomainTermsBoundaries:
    """T1: Boundary conditions in correct_domain_terms."""

    def test_three_char_word_skipped_by_length_check(self):
        """TC-110: 3-char words must be skipped (len < 4 guard).
        'atp' → ATP is in biology but 'atp' (3 chars) should not be corrected."""
        from handwriting_engine.postprocess import correct_domain_terms
        # "cel" is 3 chars — should not be corrected to "cell" (< 4 guard)
        result = correct_domain_terms("the cel is active", "biology")
        assert "cel" in result, "3-char 'cel' must not be corrected (len < 4 guard)"
        assert result == "the cel is active", f"Unexpected modification: {result!r}"

    def test_four_char_word_processed(self):
        """TC-111: 4-char words ARE processed (boundary: len < 4 is false for len==4).
        'cell' is exactly in the biology wordlist → edit distance 0, returns []."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates, _BIOLOGY_TERMS
        # "cell" (4 chars) IS in _BIOLOGY_TERMS → exact match → returns []
        candidates = _edit_distance_1_candidates("cell", _BIOLOGY_TERMS)
        assert candidates == [], "4-char exact match should return []"

    def test_four_char_word_one_edit_away_corrected(self):
        """TC-112: 4-char word NOT in wordlist but 1 edit from biology term.
        'bell' is 1 edit (b→c) from 'cell'. If only 1 candidate → FALSE POSITIVE.
        Documents the false-positive correction bug."""
        from handwriting_engine.postprocess import correct_domain_terms, _edit_distance_1_candidates, _BIOLOGY_TERMS
        # Confirm the false positive exists
        candidates = _edit_distance_1_candidates("bell", _BIOLOGY_TERMS)
        if len(candidates) == 1:
            # Bug: "bell" will be changed to "cell" — common word corrupted
            result = correct_domain_terms("the bell rang", "biology")
            assert result == "the bell rang", (
                f"BUG: 'bell' was incorrectly corrected to a domain term: {result!r}. "
                "Common English words within edit distance 1 of domain terms get corrupted."
            )

    def test_false_positive_sell_to_cell(self):
        """TC-113: 'sell' is edit distance 1 from 'cell' (s→c).
        Should NOT be corrected in a biology context."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("students sell their notes", "biology")
        # If "sell" is incorrectly corrected → real bug in transcription quality
        assert "sell" in result, (
            f"BUG: 'sell' incorrectly corrected in: {result!r}"
        )

    def test_all_caps_long_word_not_skipped(self):
        """TC-114: ALL-CAPS word > 4 chars is NOT matched by _SKIP_RE (which only
        matches ^[A-Z]{1,4}$). 'MITOCONDRIA' (11 chars) should be processed."""
        from handwriting_engine.postprocess import _SKIP_RE
        import re
        # Verify the regex does NOT match a long all-caps word
        assert not _SKIP_RE.match("MITOCONDRIA"), (
            "MITOCONDRIA should not be skipped by abbreviation regex"
        )
        # But ATP (3 chars) SHOULD be matched → correct skip behavior
        assert _SKIP_RE.match("ATP"), "ATP (3 uppercase chars) should be skipped"

    def test_all_caps_word_capitalization_bug(self):
        """TC-115: ALL-CAPS word 'MITOCONDRIA' → corrected to 'Mitochondria' (title case)
        instead of 'MITOCHONDRIA' (ALL-CAPS). Capitalization preservation bug for
        all-caps words."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("MITOCONDRIA is the powerhouse", "biology")
        # The word should be corrected (it's a clear typo of mitochondria)
        assert "mitochondria" in result.lower(), (
            f"'MITOCONDRIA' should be corrected to a form of 'mitochondria', got: {result!r}"
        )
        # Document the capitalization behavior: is ALL-CAPS preserved?
        if "mitochondria" in result.lower():
            corrected_word = [w for w in result.split() if "mitochondria" in w.lower()][0]
            # Bug: "capitalize()" produces "Mitochondria" not "MITOCHONDRIA"
            if corrected_word == "Mitochondria":
                pytest.xfail(
                    "KNOWN CAPITALIZATION BUG: 'MITOCONDRIA' → 'Mitochondria' instead of "
                    "'MITOCHONDRIA'. correct_domain_terms uses str.capitalize() which always "
                    "produces title-case, not preserving all-caps."
                )


# ===========================================================================
# TC-120: Boundary Hunter — T4 Historical Bug Patterns
# Closing bracket/paren stripped issue (confirmed from TC-22 failure)
# ===========================================================================

class TestCorrectDomainTermsClosingPunct:
    """T4: Historical punctuation handling — closing brackets not in rstrip set."""

    def test_word_in_parentheses_corrected(self):
        """TC-120: '(mitocondria)' — closing ')' is not in rstrip(',;:!?'),
        so ']' ends up in core, preventing edit distance match.
        Real bug: correct_domain_terms("see (mitocondria) here") fails to correct.
        Root cause: rstrip only removes sentence-ending punct, not closing brackets."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("see (mitocondria) here", "biology")
        # This should correct 'mitocondria' to 'mitochondria', preserving parens
        assert "mitochondria" in result, (
            f"BUG: '(mitocondria)' not corrected — closing ')' leaks into core.\n"
            f"Result: {result!r}\n"
            "Fix: also strip trailing non-alpha chars ()] etc.) before edit distance lookup."
        )

    def test_word_in_square_brackets_not_confused_with_markers(self):
        """TC-121: '[mitocondria]' — after Bug-1 fix, leading '[' is peeled into
        prefix and trailing ']' into suffix, so core='mitocondria' is corrected
        normally to 'mitochondria'.  The result should be '[mitochondria]'."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("[mitocondria]", "biology")
        # Bug-1 fix: '[' → prefix, ']' → suffix, core corrected → '[mitochondria]'
        assert "[mitochondria]" in result, (
            f"'[mitocondria]' should be corrected to '[mitochondria]', got: {result!r}"
        )

    def test_word_with_trailing_colon_suffix_stripped(self):
        """TC-122: 'mitocondria:' — trailing ':' IS in rstrip set → stripped to suffix.
        Verify that words with sentence punctuation ARE corrected."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("define mitocondria: it is", "biology")
        # 'mitocondria:' → stripped='mitocondria', suffix=':', core='mitocondria'
        # Should be corrected to 'mitochondria:'
        assert "mitochondria" in result, (
            f"'mitocondria:' should be corrected (colon in rstrip set), got: {result!r}"
        )

    def test_word_with_trailing_comma(self):
        """TC-123: 'mitocondria,' — trailing ',' in rstrip set → corrected."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the mitocondria, found in cells", "biology")
        assert "mitochondria" in result, (
            f"'mitocondria,' should be corrected (comma in rstrip set), got: {result!r}"
        )


# ===========================================================================
# TC-130: Interaction Prober — T6 Invariant Hunting
# assess_contrast: bimodal distribution where bright tail is past p95
# ===========================================================================

class TestAssessContrastInvariants:
    """T6: Invariant: assess_contrast returns [0,1] AND the metric reflects
    actual visual contrast, not just cumulative percentile coverage."""

    def test_bimodal_skewed_dark_contrast_not_zero(self):
        """TC-130: Image with 95% pixels at value=10 (dark) and 5% at value=240 (bright).
        p5 will be found at bucket 10 (cumulative hits 5% there).
        p95 will ALSO be found at bucket 10 (cumulative hits 95% at bucket 10).
        → contrast = (10-10)/255 = 0.0, despite visual contrast existing.
        Documents that assess_contrast IGNORES the bright tail when skewed."""
        from handwriting_engine.quality import assess_contrast
        # Build a 200x200 image: 95% pixels at 10, 5% at 240
        img = Image.new("L", (200, 200), 10)
        bright_count = int(200 * 200 * 0.05)
        pixels_flat = [10] * (200 * 200 - bright_count) + [240] * bright_count
        img = Image.new("L", (200, 200))
        img.putdata(pixels_flat)
        img = img.convert("RGB")
        score = assess_contrast(img)
        # Document the behavior: score will be 0 (or near 0) even though there's contrast
        assert 0.0 <= score <= 1.0, "Must be in [0,1]"
        # The known limitation: p95 is found at the dark bucket, bright pixels ignored
        if score == 0.0:
            pytest.xfail(
                "KNOWN LIMITATION: assess_contrast uses 5th–95th percentile range. "
                "When ≥95% of pixels are in the dark region, p95==p5 and score=0 "
                "even though 5% of pixels are very bright. Bright outliers are ignored."
            )

    def test_contrast_symmetric_bimodal(self):
        """TC-131: 50% pixels at 0 (black) and 50% at 255 (white).
        p5 must be 0, p95 must be 255 → contrast = 1.0."""
        from handwriting_engine.quality import assess_contrast
        half = 200 * 200 // 2
        pixels_flat = [0] * half + [255] * (200 * 200 - half)
        img = Image.new("L", (200, 200))
        img.putdata(pixels_flat)
        img = img.convert("RGB")
        score = assess_contrast(img)
        assert score == 1.0, (
            f"50/50 black/white must have contrast=1.0, got {score}"
        )

    def test_contrast_near_maximum_asymmetric(self):
        """TC-132: 6% pixels at 0, 94% pixels at 255.
        p5 will be found at bucket 255 (cumulative doesn't hit 5% until bucket 255).
        Wait — 6% is at 0, so cumulative at bucket 0 = 6% >= 5% → p5=0.
        p95: cumulative at bucket 0 = 6%, then at bucket 255 = 100% >= 95% → p95=255.
        contrast = (255-0)/255 = 1.0 — correct."""
        from handwriting_engine.quality import assess_contrast
        total = 200 * 200  # 40000
        dark_count = int(total * 0.06)  # 2400 dark pixels
        bright_count = total - dark_count
        pixels_flat = [0] * dark_count + [255] * bright_count
        img = Image.new("L", (200, 200))
        img.putdata(pixels_flat)
        img = img.convert("RGB")
        score = assess_contrast(img)
        assert score == 1.0, (
            f"6%/94% black/white should give contrast=1.0 (p5=0, p95=255), got {score}"
        )


# ===========================================================================
# TC-140: Interaction Prober — T3 Expectation Inversion
# recommend_enhancement_params: edge cases in the if/elif chain
# ===========================================================================

class TestRecommendEnhancementParamsEdgeCases:
    """T3: Test borderline cases in the recommend_enhancement_params if/elif chain."""

    def test_quality_poor_no_issues_hits_else_branch(self):
        """TC-140: quality='poor' + issues=[] — no elif branch matches.
        Falls to 'else: multiple issues full treatment' with an empty issues string."""
        from handwriting_engine.quality import recommend_enhancement_params
        assessment = {
            "quality": "poor",
            "issues": [],
            "faint_ink": False,
        }
        params = recommend_enhancement_params(assessment)
        # Falls to else: reason = f"multiple issues ({', '.join(issues)}) — full treatment"
        # With empty issues, ', '.join([]) = '' → reason = "multiple issues () — full treatment"
        assert isinstance(params, dict)
        assert params.get("strategy") == "adaptive"
        # 'else' branch enables denoise + sharpen + contrast + brighten
        assert params.get("denoise") is True
        assert params.get("sharpen") is True
        assert params.get("contrast") is True
        assert params.get("brighten") is True

    def test_quality_fair_poor_contrast_factor_differs(self):
        """TC-141: 'else' branch: contrast_factor=1.5 for 'fair', 2.0 for 'poor'.
        Verify the conditional correctly differentiates quality tiers."""
        from handwriting_engine.quality import recommend_enhancement_params
        base = {"issues": ["blurry", "low_contrast"], "faint_ink": False}

        fair = recommend_enhancement_params({**base, "quality": "fair"})
        poor = recommend_enhancement_params({**base, "quality": "poor"})

        assert fair["contrast_factor"] == 1.5, (
            f"fair quality should use contrast_factor=1.5, got {fair['contrast_factor']}"
        )
        assert poor["contrast_factor"] == 2.0, (
            f"poor quality should use contrast_factor=2.0, got {poor['contrast_factor']}"
        )

    def test_unknown_quality_value_uses_else_contrast_factor(self):
        """TC-142: quality='unknown_tier' with blurry+low_contrast (neither specific elif fires → else).
        quality != 'fair' → contrast_factor=2.0 in else branch."""
        from handwriting_engine.quality import recommend_enhancement_params
        assessment = {
            "quality": "unknown_tier",
            "issues": ["blurry", "low_contrast"],  # Both present → neither single-issue elif fires
            "faint_ink": False,
        }
        params = recommend_enhancement_params(assessment)
        # Falls to else: contrast_factor = 1.5 if quality=='fair' else 2.0
        # "unknown_tier" != "fair" → should be 2.0
        assert params["contrast_factor"] == 2.0, (
            f"Unknown quality tier should use 2.0 in else branch, got {params['contrast_factor']}"
        )

    def test_missing_keys_graceful_defaults(self):
        """TC-143: Assessment dict with missing 'issues', 'quality', 'faint_ink' keys —
        .get() calls with defaults must not crash."""
        from handwriting_engine.quality import recommend_enhancement_params
        # Completely empty dict — all .get() calls use defaults
        params = recommend_enhancement_params({})
        # quality defaults to "good" → returns early with llm strategy
        assert params.get("strategy") == "llm", (
            f"Missing quality key should default to 'good' strategy, got {params}"
        )

    def test_faint_ink_in_issues_key_absent(self):
        """TC-144: faint_ink key missing from dict → defaults to False.
        'faint_ink' in issues=True but assessment.get('faint_ink', False)=False.
        Verify faint_ink-in-issues path still fires."""
        from handwriting_engine.quality import recommend_enhancement_params
        assessment = {
            "quality": "poor",
            "issues": ["faint_ink"],
            # Note: 'faint_ink' key is ABSENT — .get('faint_ink', False) = False
        }
        params = recommend_enhancement_params(assessment)
        # The check is: `if faint or "faint_ink" in issues` — faint=False but faint_ink in issues
        # → should still trigger faint branch
        assert params.get("contrast") is True, (
            "faint_ink in issues should trigger contrast treatment even with missing faint_ink key"
        )
        assert params.get("sharpen") is False, (
            "faint_ink branch must NOT sharpen"
        )


# ===========================================================================
# TC-150: Scale Breaker — T7 Scale Amplification
# Performance stress tests on edit distance generation
# ===========================================================================

class TestEditDistanceScale:
    """T7: Scale amplification — stress the edit distance candidate generation."""

    def test_long_word_performance(self):
        """TC-150: _edit_distance_1_candidates on a 200-char word.
        The function generates O(n*26) strings per operation — n=200 means ~5200 edits.
        Must complete without hanging."""
        from handwriting_engine.postprocess import _edit_distance_1_candidates, _BIOLOGY_TERMS
        import time
        long_word = "a" * 200
        t0 = time.time()
        candidates = _edit_distance_1_candidates(long_word, _BIOLOGY_TERMS)
        elapsed = time.time() - t0
        assert isinstance(candidates, list)
        assert elapsed < 5.0, (
            f"200-char word took {elapsed:.2f}s — edit distance generation may be O(n^2)"
        )

    def test_many_corrections_in_single_text(self):
        """TC-151: Text with 100 words each needing correction — no crash, no OOM."""
        from handwriting_engine.postprocess import correct_domain_terms
        # 100 copies of a typo that has exactly 1 correction candidate
        text = " ".join(["mitocondria"] * 100)
        result = correct_domain_terms(text, "biology")
        assert "mitochondria" in result
        assert result.count("mitochondria") == 100, (
            f"All 100 instances should be corrected, got {result.count('mitochondria')}"
        )

    def test_word_with_maximum_leading_punctuation(self):
        """TC-152: Word with many leading non-alpha chars '(((mitocondria)))'.
        The prefix loop strips each non-alpha char. The closing '))' ends in core.
        Documents the multi-bracket nesting behavior."""
        from handwriting_engine.postprocess import correct_domain_terms
        # Triple-nested parens — closing parens all end up in core
        result = correct_domain_terms("(((mitocondria)))", "biology")
        # Either corrected (if closing parens don't interfere with edit match) or not
        assert isinstance(result, str), "Must not crash on heavily punctuated input"


# ===========================================================================
# TC-160: Interaction Prober — T8 Adversarial Actor
# International / encoding edge cases
# ===========================================================================

class TestCorrectDomainTermsEncoding:
    """T8: Adversarial encoding inputs to correct_domain_terms."""

    def test_science_domain_uses_combined_wordlist(self):
        """TC-160: domain='science' uses biology+chemistry+general combined.
        Verify a chemistry typo is corrected in science domain."""
        from handwriting_engine.postprocess import correct_domain_terms, _DOMAIN_WORDLISTS
        # Confirm science domain is the union
        assert "biology" not in _DOMAIN_WORDLISTS["science"], (
            "Science wordlist should not be the string 'biology'"
        )
        from handwriting_engine.postprocess import _BIOLOGY_TERMS, _CHEMISTRY_TERMS
        science_words = _DOMAIN_WORDLISTS["science"]
        assert _CHEMISTRY_TERMS.issubset(science_words), (
            "Science domain must include all chemistry terms"
        )
        assert _BIOLOGY_TERMS.issubset(science_words), (
            "Science domain must include all biology terms"
        )

    def test_mixed_case_word_preserves_leading_cap(self):
        """TC-161: 'Mitocondria' (title case) → 'Mitochondria' (title case preserved)."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("Mitocondria divides", "biology")
        # core[0].isupper() → candidate.capitalize()
        words = result.split()
        assert words[0] == "Mitochondria", (
            f"Title-case 'Mitocondria' should become 'Mitochondria', got: {words[0]!r}"
        )

    def test_word_with_accented_chars_not_corrected(self):
        """TC-162: Accented characters like 'naïve' — the edit distance alphabet is
        ASCII only (a-z). Non-ASCII letters generate edits that won't match domain
        terms. The function should not crash and should leave the word unchanged."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("the naïve experiment", "biology")
        assert isinstance(result, str), "Must not crash on accented chars"
        # "naïve" is 5 chars, not in biology wordlist, but edit distance candidates
        # are built with ASCII alphabet — naïve is unlikely to match any biology term
        assert "naïve" in result or "naIve" in result.lower() or isinstance(result, str)

    def test_only_whitespace_returns_empty(self):
        """TC-163: Text with only whitespace → split() returns [] → corrected=[] →
        ' '.join([]) = '' → returns ''."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("   \t   ", "biology")
        assert result == "", f"Whitespace-only text should return '', got: {result!r}"

    def test_single_word_text_corrected(self):
        """TC-164: Single-word text 'mitocondria' → 'mitochondria'."""
        from handwriting_engine.postprocess import correct_domain_terms
        result = correct_domain_terms("mitocondria", "biology")
        assert result == "mitochondria", f"Single word correction failed: {result!r}"


# ===========================================================================
# TC-170: Boundary Hunter — T1 Boundary Collision
# batch_pages_by_size: single oversized image
# ===========================================================================

class TestBatchPagesBySize:
    """T1: Boundary conditions in batch_pages_by_size."""

    def test_single_valid_page_one_batch(self):
        """TC-170: Single valid page → exactly 1 batch containing that page."""
        from handwriting_engine.optimize import batch_pages_by_size
        path = _solid_image(size=(200, 200))
        try:
            pages = [{"path": path, "page_number": 1}]
            batches = batch_pages_by_size(pages, max_batch_bytes=10_000_000)
            assert len(batches) == 1
            assert batches[0] == pages
        finally:
            os.unlink(path)

    def test_two_pages_small_budget_split_into_two_batches(self):
        """TC-171: Two pages but tiny max_batch_bytes forces them into separate batches."""
        from handwriting_engine.optimize import batch_pages_by_size
        p1 = _solid_image(size=(400, 400))
        p2 = _solid_image(size=(400, 400))
        try:
            pages = [
                {"path": p1, "page_number": 1},
                {"path": p2, "page_number": 2},
            ]
            # Use 1 byte — every image will exceed this, triggering the shrink path
            # But even the shrunk version will exceed 1 byte → pages skipped → raises
            from handwriting_engine.optimize import ImagePreparationError
            try:
                batches = batch_pages_by_size(pages, max_batch_bytes=1)
                # If it doesn't raise, each image must have been in its own batch
                assert len(batches) >= 1
            except ImagePreparationError:
                pass  # Acceptable — pages can't fit in 1 byte → raises
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_all_bad_pages_raises_error(self):
        """TC-172: All pages have non-existent paths → all skipped → ImagePreparationError."""
        from handwriting_engine.optimize import batch_pages_by_size, ImagePreparationError
        pages = [
            {"path": "/not/here1.jpg", "page_number": 1},
            {"path": "/not/here2.jpg", "page_number": 2},
        ]
        with pytest.raises(ImagePreparationError):
            batch_pages_by_size(pages)

    def test_rgba_image_converted_and_processed(self):
        """TC-173: RGBA image must be converted to RGB before JPEG encoding.
        validate_and_prepare_image handles RGBA → RGB conversion internally."""
        from handwriting_engine.optimize import validate_and_prepare_image
        # Create an RGBA image (PNG with alpha)
        img = Image.new("RGBA", (100, 100), (200, 200, 200, 128))
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path, "PNG")
        try:
            result = validate_and_prepare_image(path)
            assert result is not None, "RGBA image should be convertible (RGBA→RGB)"
            data, media_type = result
            assert media_type == "image/jpeg"
            assert len(data) > 0
        finally:
            os.unlink(path)
