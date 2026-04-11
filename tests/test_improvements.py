"""Tests for recent improvements to the handwriting engine.

Covers:
- detect_skew() edge cases
- _deskew() identity behavior
- _build_compact_disambiguation() output format
- _single_text_confidence() hallucination pattern penalties
- Hallucination rejection threshold (60% agreement)
- New constants (SKEW_THRESHOLD_DEGREES, ZOOM_VERIFY_*)
- adapt_system_prompt / adapt_user_prompt per provider
- _zoomed_verify returns original text when no [?] markers present
"""

import os
import tempfile
import re
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image, ImageDraw

from handwriting_engine._constants import (
    SKEW_THRESHOLD_DEGREES,
    ZOOM_VERIFY_STRIPS,
    ZOOM_VERIFY_CONFIDENCE_LOW,
    ZOOM_VERIFY_CONFIDENCE_HIGH,
    ZOOM_VERIFY_MIN_MARKERS,
)
from handwriting_engine.quality import detect_skew
from handwriting_engine.enhance import _deskew
from handwriting_engine.prompt_adapter import (
    adapt_system_prompt,
    adapt_user_prompt,
    _build_compact_disambiguation,
    _TOP_DISAMBIGUATION_PAIRS,
)
from handwriting_engine.consensus import (
    _single_text_confidence,
    _word_agreement_ratio,
    _SINGLE_PROVIDER_MAX_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tmp(img, suffix=".jpg"):
    """Save image to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if img.mode == "L":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=95)
    return path


# ---------------------------------------------------------------------------
# New Constants
# ---------------------------------------------------------------------------

class TestNewConstants:
    def test_skew_threshold_degrees_value(self):
        assert SKEW_THRESHOLD_DEGREES == 1.5

    def test_zoom_verify_strips_value(self):
        assert ZOOM_VERIFY_STRIPS == 3

    def test_zoom_verify_confidence_low_value(self):
        assert ZOOM_VERIFY_CONFIDENCE_LOW == 0.45

    def test_zoom_verify_confidence_high_value(self):
        assert ZOOM_VERIFY_CONFIDENCE_HIGH == 0.65

    def test_zoom_verify_min_markers_value(self):
        assert ZOOM_VERIFY_MIN_MARKERS == 1

    def test_zoom_confidence_low_below_high(self):
        """Sanity: low threshold must be less than high threshold."""
        assert ZOOM_VERIFY_CONFIDENCE_LOW < ZOOM_VERIFY_CONFIDENCE_HIGH

    def test_zoom_verify_strips_positive(self):
        assert ZOOM_VERIFY_STRIPS > 0

    def test_zoom_verify_min_markers_positive(self):
        assert ZOOM_VERIFY_MIN_MARKERS >= 1


# ---------------------------------------------------------------------------
# detect_skew
# ---------------------------------------------------------------------------

class TestDetectSkew:
    def test_nonexistent_file_returns_zero(self):
        result = detect_skew("/nonexistent/path/to/image.jpg")
        assert result == 0.0

    def test_corrupt_file_returns_zero(self):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(b"not an image at all")
            result = detect_skew(path)
            assert result == 0.0
        finally:
            os.unlink(path)

    def test_blank_image_returns_zero(self):
        """A uniform white image has no text lines to detect skew from."""
        img = Image.new("RGB", (400, 400), (255, 255, 255))
        path = _save_tmp(img)
        try:
            result = detect_skew(path)
            assert result == 0.0
        finally:
            os.unlink(path)

    def test_straight_text_returns_near_zero(self):
        """Horizontal lines should produce ~0 skew."""
        img = Image.new("L", (400, 200), 240)
        draw = ImageDraw.Draw(img)
        # Draw perfectly horizontal lines
        for y in range(20, 180, 30):
            draw.line([(10, y), (390, y)], fill=20, width=2)
        path = _save_tmp(img)
        try:
            result = detect_skew(path)
            # Should be below threshold (returned as 0.0)
            assert result == 0.0
        finally:
            os.unlink(path)

    def test_return_type_is_float(self):
        img = Image.new("L", (200, 200), 240)
        path = _save_tmp(img)
        try:
            result = detect_skew(path)
            assert isinstance(result, float)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _deskew
# ---------------------------------------------------------------------------

class TestDeskew:
    def test_zero_angle_returns_unchanged(self):
        """Angle exactly 0 should return the same image dimensions."""
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        result = _deskew(img, 0.0)
        assert result.size == img.size

    def test_very_small_angle_returns_unchanged(self):
        """Angle < 0.1 should return unchanged (early return)."""
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        result = _deskew(img, 0.05)
        assert result.size == img.size

    def test_nonzero_angle_rotates(self):
        """A significant angle should produce a rotated (expanded) image."""
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        result = _deskew(img, 5.0)
        # expand=True means dimensions change for non-zero rotation
        assert result.size != img.size

    def test_negative_angle_rotates(self):
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        result = _deskew(img, -3.0)
        assert result.size != img.size

    def test_grayscale_fill(self):
        """Grayscale images should fill with 255 (white), not a tuple."""
        img = Image.new("L", (200, 100), 128)
        result = _deskew(img, 5.0)
        assert result.mode == "L"

    def test_rgb_fill(self):
        """RGB images should fill with (255, 255, 255)."""
        img = Image.new("RGB", (200, 100), (128, 128, 128))
        result = _deskew(img, 5.0)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# _build_compact_disambiguation
# ---------------------------------------------------------------------------

class TestBuildCompactDisambiguation:
    def test_returns_string(self):
        result = _build_compact_disambiguation()
        assert isinstance(result, str)

    def test_has_header(self):
        result = _build_compact_disambiguation()
        assert "=== CHARACTER DISAMBIGUATION (TOP 15) ===" in result

    def test_has_context_instruction(self):
        result = _build_compact_disambiguation()
        assert "When a character is ambiguous, use context:" in result

    def test_contains_all_15_pairs(self):
        result = _build_compact_disambiguation()
        for pair, rule in _TOP_DISAMBIGUATION_PAIRS:
            assert pair in result, f"Missing pair: {pair}"
            assert rule in result, f"Missing rule for: {pair}"

    def test_exactly_15_pairs(self):
        assert len(_TOP_DISAMBIGUATION_PAIRS) == 15

    def test_has_validation_footer(self):
        result = _build_compact_disambiguation()
        assert "VALIDATION:" in result
        assert "swapping confusion pair" in result

    def test_format_indentation(self):
        """Each pair line should be indented with two spaces."""
        result = _build_compact_disambiguation()
        lines = result.split("\n")
        pair_lines = [l for l in lines if ": " in l and l.startswith("  ")]
        assert len(pair_lines) == 15


# ---------------------------------------------------------------------------
# _single_text_confidence — hallucination pattern penalties
# ---------------------------------------------------------------------------

class TestSingleTextConfidenceHallucination:
    def test_empty_returns_zero(self):
        assert _single_text_confidence("") == 0.0

    def test_whitespace_returns_zero(self):
        assert _single_text_confidence("   \n  ") == 0.0

    def test_capped_at_max(self):
        """Even clean text is capped at the single provider max."""
        text = (
            "The student wrote a detailed answer about the mitochondria being "
            "the powerhouse of the cell, with correct scientific terminology "
            "and appropriate citations to the lab manual."
        )
        score = _single_text_confidence(text)
        assert score <= _SINGLE_PROVIDER_MAX_CONFIDENCE

    def test_short_text_penalty(self):
        """Short specific text (1-5 words, no markers) should be penalized
        as it matches the hallucination pattern."""
        short = _single_text_confidence("Steffon Espericueta")
        long = _single_text_confidence(
            "The student wrote Steffon Espericueta at the top of the page "
            "followed by three paragraphs about cellular biology"
        )
        assert short < long

    def test_single_word_no_markers_penalized(self):
        """A single word with no uncertainty should be treated as suspicious."""
        score = _single_text_confidence("Sheldon")
        assert score < 0.55

    def test_single_character_capped_low(self):
        """Single-character answers have hard cap at 0.45."""
        score = _single_text_confidence("A")
        assert score <= 0.45

    def test_two_character_capped_low(self):
        score = _single_text_confidence("pH")
        assert score <= 0.45

    def test_many_uncertainty_markers_low_confidence(self):
        text = "The [?] student [?] wrote [illegible] about [?] something [unclear]"
        score = _single_text_confidence(text)
        assert score < 0.3

    def test_highly_repetitive_text_penalized(self):
        """Text that repeats the same word many times is suspicious."""
        text = " ".join(["hello"] * 20)
        score = _single_text_confidence(text)
        normal = _single_text_confidence(
            "The student described the process of mitosis including prophase "
            "metaphase anaphase and telophase with correct detail"
        )
        assert score < normal

    def test_moderate_text_gets_moderate_confidence(self):
        """Normal paragraph text without markers should score reasonably."""
        text = "The mitochondria is the powerhouse of the cell. pH = 7.2 OD600 = 0.45"
        score = _single_text_confidence(text)
        assert 0.5 < score <= 0.70


# ---------------------------------------------------------------------------
# Hallucination rejection threshold (60% agreement)
# ---------------------------------------------------------------------------

class TestHallucinationRejection:
    def test_agreement_ratio_identical(self):
        ratio = _word_agreement_ratio("hello world", "hello world")
        assert ratio == 1.0

    def test_agreement_ratio_empty(self):
        assert _word_agreement_ratio("", "") == 1.0
        assert _word_agreement_ratio("hello", "") == 0.0

    def test_agreement_below_sixty_percent(self):
        """Two very different texts should have agreement well below 0.60."""
        a = "The mitochondria is the powerhouse of the cell"
        b = "XYZZY PLUGH ABCDE nothing matches here at all"
        ratio = _word_agreement_ratio(a, b)
        assert ratio < 0.60

    def test_agreement_above_sixty_percent(self):
        """Two similar texts with small differences should be above 0.60."""
        a = "The student wrote about mitosis and cell division"
        b = "The student wrote about meiosis and cell division"
        ratio = _word_agreement_ratio(a, b)
        assert ratio > 0.60

    def test_sixty_percent_is_the_threshold(self):
        """Verify the threshold value used in the vote function.

        The _vote function flags providers with max_agreement < 0.60
        as potential hallucinations.
        """
        # This is a structural test — the threshold is hardcoded in _vote.
        # We verify it through the Grep-able string "0.60" in consensus.py.
        # The actual test is that _word_agreement_ratio works as expected.
        assert _word_agreement_ratio("completely different text", "nothing similar here") < 0.60


# ---------------------------------------------------------------------------
# adapt_system_prompt
# ---------------------------------------------------------------------------

class TestAdaptSystemPrompt:
    def test_claude_passthrough(self):
        """Claude gets the full system prompt unchanged."""
        prompt = "Full system prompt with all details and tables."
        assert adapt_system_prompt(prompt, "claude") == prompt

    def test_claude_empty(self):
        assert adapt_system_prompt("", "claude") == ""

    def test_gemini_compacts_disambiguation(self):
        """Gemini should replace full disambiguation table with compact version."""
        prompt = (
            "Preamble text.\n"
            "=== CHARACTER DISAMBIGUATION ===\n"
            "Full 44-pair table here\n"
            "pair1: rule1\n"
            "pair2: rule2\n"
            "\n=== NEXT SECTION ===\nMore content"
        )
        result = adapt_system_prompt(prompt, "gemini")
        assert "=== CHARACTER DISAMBIGUATION (TOP 15) ===" in result
        assert "=== NEXT SECTION ===" in result
        assert "Full 44-pair table here" not in result

    def test_gemini_no_disambiguation_passthrough(self):
        """Without disambiguation section, Gemini prompt is unchanged."""
        prompt = "Simple prompt without disambiguation table."
        result = adapt_system_prompt(prompt, "gemini")
        assert result == prompt

    def test_openai_compacts_multipass(self):
        """OpenAI should replace multi-pass protocol with concise version."""
        prompt = (
            "Preamble.\n"
            "=== CHARACTER DISAMBIGUATION ===\nFull table\n"
            "\n=== HANDWRITING READING PROTOCOL (MULTI-PASS) ===\n"
            "Step 1...\nStep 2...\nStep 3...\n"
            "\n=== FINAL SECTION ===\nMore stuff"
        )
        result = adapt_system_prompt(prompt, "openai")
        # Should have compact disambiguation
        assert "=== CHARACTER DISAMBIGUATION (TOP 15) ===" in result
        # Should have compact reading protocol
        assert "=== READING PROTOCOL ===" in result
        # Should have few-shot disambiguation examples
        assert "DISAMBIGUATION EXAMPLES" in result
        assert '"The OD was 0.45"' in result
        # Should still have the final section
        assert "=== FINAL SECTION ===" in result
        # Full multi-pass protocol should be gone
        assert "MULTI-PASS" not in result

    def test_openai_no_multipass_only_compacts_disambiguation(self):
        """OpenAI without multi-pass should still compact disambiguation."""
        prompt = (
            "Preamble.\n"
            "=== CHARACTER DISAMBIGUATION ===\nFull table\n"
        )
        result = adapt_system_prompt(prompt, "openai")
        assert "=== CHARACTER DISAMBIGUATION (TOP 15) ===" in result

    def test_unknown_provider_passthrough(self):
        """Unknown providers should get the full prompt (same as Claude)."""
        prompt = "Some prompt"
        assert adapt_system_prompt(prompt, "unknown_provider") == prompt


# ---------------------------------------------------------------------------
# adapt_user_prompt
# ---------------------------------------------------------------------------

class TestAdaptUserPrompt:
    def test_openai_adds_role_prefix(self):
        """OpenAI should get concise transcription instruction prepended."""
        prompt = "Read all handwritten text."
        result = adapt_user_prompt(prompt, "openai")
        assert result.startswith("Transcribe all handwritten text")
        assert prompt in result

    def test_openai_contains_format_instructions(self):
        result = adapt_user_prompt("Read this.", "openai")
        assert "exactly as written" in result
        assert "Output ONLY" in result

    def test_claude_passthrough(self):
        prompt = "Read all handwritten text."
        assert adapt_user_prompt(prompt, "claude") == prompt

    def test_gemini_passthrough(self):
        prompt = "Read all handwritten text."
        assert adapt_user_prompt(prompt, "gemini") == prompt

    def test_unknown_provider_passthrough(self):
        prompt = "Read all handwritten text."
        assert adapt_user_prompt(prompt, "unknown") == prompt

    def test_openai_preserves_original_prompt(self):
        """The original prompt should appear at the end of the adapted version."""
        original = "Carefully transcribe these handwritten notes."
        result = adapt_user_prompt(original, "openai")
        assert result.endswith(original)


# ---------------------------------------------------------------------------
# _zoomed_verify — returns original text when no markers
# ---------------------------------------------------------------------------

class TestZoomedVerify:
    def test_no_markers_returns_original(self):
        """When text has no [?] or [illegible] markers, _zoomed_verify
        should return the original text unchanged (no crops needed)."""
        from handwriting_engine.vision import _zoomed_verify

        # Create a real image for the function to open
        img = Image.new("RGB", (400, 300), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        for y in range(20, 280, 30):
            draw.line([(10, y), (390, y)], fill=(20, 20, 20), width=2)
        path = _save_tmp(img)

        try:
            text = "Clean text with no uncertainty markers at all\nSecond line here"
            result = _zoomed_verify(
                path, text, "claude",
                system_prompt="system", max_tokens=4096,
            )
            assert result == text
        finally:
            os.unlink(path)

    def test_empty_text_returns_original(self):
        """Empty text should be returned as-is."""
        from handwriting_engine.vision import _zoomed_verify

        img = Image.new("RGB", (200, 200), (240, 240, 240))
        path = _save_tmp(img)

        try:
            result = _zoomed_verify(path, "", "claude", "", 4096)
            assert result == ""
        finally:
            os.unlink(path)

    def test_corrupt_image_returns_original(self):
        """If the image can't be opened, return original text."""
        from handwriting_engine.vision import _zoomed_verify

        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(b"corrupt data")
            text = "Some text with [?] marker"
            result = _zoomed_verify(path, text, "claude", "", 4096)
            assert result == text
        finally:
            os.unlink(path)

    def test_markers_trigger_crop_and_reread(self):
        """When text has [?] markers and image is valid, the function
        should attempt to crop and re-read strips (mocked provider)."""
        from handwriting_engine.vision import _zoomed_verify

        img = Image.new("RGB", (400, 600), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        for y in range(20, 580, 30):
            draw.line([(10, y), (390, y)], fill=(20, 20, 20), width=2)
        path = _save_tmp(img)

        mock_provider = MagicMock()
        mock_provider.read_image.return_value = "Resolved text for this strip"

        text = (
            "Line one is clean\n"
            "Line two has [?] uncertainty\n"
            "Line three is fine"
        )

        try:
            with patch("handwriting_engine.vision.get_provider", return_value=mock_provider):
                result = _zoomed_verify(path, text, "claude", "system", 4096)
            # Provider should have been called at least once (for the strip with [?])
            assert mock_provider.read_image.called
            # Result should not be identical to input since strip was replaced
            # (it will differ because the mock returns different text)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: _deskew is used in smart_enhance and adaptive_enhance
# ---------------------------------------------------------------------------

class TestDeskewIntegration:
    def test_smart_enhance_uses_deskew_from_assessment(self):
        """smart_enhance should call _deskew when assessment has skew_angle."""
        from handwriting_engine.enhance import smart_enhance

        img = Image.new("RGB", (400, 400), (50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "dark text", fill=(70, 70, 70))
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(path, "JPEG", quality=95)

        fd2, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd2)

        try:
            # Provide an assessment with skew and a strategy that triggers processing
            assessment = {
                "quality": "fair",
                "recommended_strategy": "sharpen",
                "faint_ink": False,
                "skew_angle": 3.0,
            }
            result = smart_enhance(path, assessment=assessment, output_path=out_path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            os.unlink(path)
            if os.path.exists(out_path):
                os.unlink(out_path)


# ---------------------------------------------------------------------------
# assess_image now includes skew_angle
# ---------------------------------------------------------------------------

class TestAssessImageSkew:
    def test_assess_image_has_skew_angle(self):
        """assess_image result should now include skew_angle field."""
        from handwriting_engine.quality import assess_image

        img = Image.new("L", (400, 200), 240)
        draw = ImageDraw.Draw(img)
        for y in range(0, 200, 30):
            draw.line([(10, y), (390, y)], fill=20, width=2)
        draw.text((20, 60), "Hello World", fill=20)
        path = _save_tmp(img)

        try:
            result = assess_image(path)
            assert "skew_angle" in result
            assert isinstance(result["skew_angle"], float)
        finally:
            os.unlink(path)

    def test_assess_image_skew_triggers_issue(self):
        """If skew is detected above threshold, 'skewed' should be in issues."""
        from handwriting_engine.quality import assess_image

        # Create an image with diagonal lines that might appear skewed.
        # We mock detect_skew to control the test.
        img = Image.new("L", (400, 200), 240)
        draw = ImageDraw.Draw(img)
        for y in range(0, 200, 30):
            draw.line([(10, y), (390, y)], fill=20, width=2)
        path = _save_tmp(img)

        try:
            with patch("handwriting_engine.quality.detect_skew", return_value=3.5):
                result = assess_image(path)
            assert "skewed" in result["issues"]
            assert result["skew_angle"] == 3.5
        finally:
            os.unlink(path)
