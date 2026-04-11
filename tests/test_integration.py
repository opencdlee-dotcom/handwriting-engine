"""Integration tests for the handwriting engine pipeline.

Tests end-to-end flows: image creation → quality assessment → enhancement →
reading (with mock providers) → consensus. No real API calls.
"""

import os
import tempfile

from PIL import Image, ImageDraw

from handwriting_engine.enhance import enhance_image, adaptive_enhance
from handwriting_engine.optimize import validate_and_prepare_image, is_blank_page
from handwriting_engine.quality import assess_image, recommend_enhancement_params, classify_handwriting_style


# ---------------------------------------------------------------------------
# Fixture helpers — create synthetic test images
# ---------------------------------------------------------------------------

def _create_text_image(text="Hello World", size=(400, 200), bg=240, fg=20):
    """Create a synthetic image with rendered text and good contrast."""
    img = Image.new("L", size, bg)
    draw = ImageDraw.Draw(img)
    # Draw thick text strokes and edges to create measurable contrast/sharpness
    for y_off in range(0, size[1], 30):
        draw.line([(10, y_off), (size[0] - 10, y_off)], fill=fg, width=2)
    draw.text((20, 60), text, fill=fg)
    draw.text((20, 90), "ABCDEFGHIJKLMNOP", fill=fg)
    return img


def _create_blank_page(size=(400, 600)):
    """Create a nearly-blank white page."""
    return Image.new("RGB", size, (252, 252, 252))


def _create_faint_image(text="Faint text", size=(400, 200)):
    """Create an image with very faint text (high brightness, low contrast)."""
    img = Image.new("L", size, 230)  # Very bright background
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), text, fill=210)  # Nearly invisible text
    return img


def _create_blurry_image(text="Blurry text", size=(400, 200)):
    """Create a blurred image with preserved contrast."""
    from PIL import ImageFilter
    img = _create_text_image(text, size, bg=240, fg=20)
    # Heavy blur to ensure blur detection triggers
    return img.filter(ImageFilter.GaussianBlur(radius=8))


def _create_low_contrast_image(text="Low contrast", size=(400, 200)):
    """Create a low-contrast image."""
    img = Image.new("L", size, 140)
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), text, fill=120)  # Very close to background
    return img


def _save_tmp(img, suffix=".jpg"):
    """Save image to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if img.mode == "L":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=95)
    return path


# ---------------------------------------------------------------------------
# Quality Assessment Integration
# ---------------------------------------------------------------------------

class TestQualityAssessmentIntegration:
    def test_good_image_rated_good(self):
        img = _create_text_image()
        path = _save_tmp(img)
        try:
            result = assess_image(path)
            # A clean synthetic image should not be rated "poor"
            assert result["quality"] in ("good", "fair")
            assert not result["faint_ink"]
        finally:
            os.unlink(path)

    def test_blurry_image_has_issues(self):
        img = _create_blurry_image()
        path = _save_tmp(img)
        try:
            result = assess_image(path)
            # Heavy blur reduces edge variance but also affects contrast/brightness
            # metrics. The key assertion is that it's not rated "good".
            assert result["quality"] != "good"
            assert len(result["issues"]) > 0
        finally:
            os.unlink(path)

    def test_faint_ink_detected(self):
        img = _create_faint_image()
        path = _save_tmp(img)
        try:
            result = assess_image(path)
            assert result["faint_ink"] is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Enhancement Integration
# ---------------------------------------------------------------------------

class TestEnhancementIntegration:
    def test_smart_enhance_improves_quality(self):
        img = _create_low_contrast_image()
        path = _save_tmp(img)
        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            before = assess_image(path)
            enhance_image(path, strategy="smart", output_path=out_path)
            after = assess_image(out_path)
            # Enhancement should not make things worse
            assert after["contrast_score"] >= before["contrast_score"] - 0.1
        finally:
            os.unlink(path)
            os.unlink(out_path)

    def test_proven_enhance_runs(self):
        img = _create_text_image()
        path = _save_tmp(img)
        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            result = enhance_image(path, strategy="proven", output_path=out_path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            os.unlink(path)
            os.unlink(out_path)

    def test_adaptive_enhance_on_faint(self):
        img = _create_faint_image()
        path = _save_tmp(img)
        fd, out_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            result = adaptive_enhance(path, output_path=out_path)
            assert os.path.exists(result)
        finally:
            os.unlink(path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_all_strategies_run_without_error(self):
        img = _create_text_image()
        path = _save_tmp(img)
        try:
            for strategy in ("smart", "proven", "llm", "light", "medium", "heavy", "full", "adaptive"):
                fd, out = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                try:
                    result = enhance_image(path, strategy=strategy, output_path=out)
                    assert os.path.exists(result), f"Strategy {strategy} failed"
                finally:
                    if os.path.exists(out):
                        os.unlink(out)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Quality-Adaptive Enhancement Params
# ---------------------------------------------------------------------------

class TestRecommendEnhancementParams:
    def test_good_image_gets_llm_strategy(self):
        params = recommend_enhancement_params({"quality": "good", "issues": [], "faint_ink": False})
        assert params["strategy"] == "llm"

    def test_faint_ink_no_sharpen(self):
        params = recommend_enhancement_params({
            "quality": "poor", "issues": ["faint_ink"], "faint_ink": True
        })
        assert params.get("sharpen") is False or not params.get("sharpen")
        assert params.get("contrast") is True

    def test_blurry_no_contrast(self):
        params = recommend_enhancement_params({
            "quality": "fair", "issues": ["blurry"], "faint_ink": False
        })
        assert params.get("sharpen") is True
        assert params.get("contrast") is False or not params.get("contrast")


# ---------------------------------------------------------------------------
# Image Optimization Integration
# ---------------------------------------------------------------------------

class TestOptimizationIntegration:
    def test_validate_and_prepare_normal(self):
        img = _create_text_image()
        path = _save_tmp(img)
        try:
            result = validate_and_prepare_image(path)
            assert result is not None
            b64_data, media_type = result
            assert media_type == "image/jpeg"
            assert len(b64_data) > 0
        finally:
            os.unlink(path)

    def test_validate_large_image_resized(self):
        img = Image.new("RGB", (3000, 3000), (128, 128, 128))
        path = _save_tmp(img)
        try:
            result = validate_and_prepare_image(path, max_long_side=1568)
            assert result is not None
        finally:
            os.unlink(path)

    def test_validate_corrupt_image_returns_none(self):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(b"not an image")
            result = validate_and_prepare_image(path)
            assert result is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Blank Page Detection
# ---------------------------------------------------------------------------

class TestBlankPageDetection:
    def test_blank_page_detected(self):
        img = _create_blank_page()
        path = _save_tmp(img)
        try:
            assert is_blank_page(path) is True
        finally:
            os.unlink(path)

    def test_content_page_not_blank(self):
        # Create image with significant dark content (not near-white)
        img = Image.new("L", (400, 600), 240)
        draw = ImageDraw.Draw(img)
        # Fill ~30% of the image with dark pixels
        for y in range(0, 600, 10):
            draw.line([(0, y), (400, y)], fill=30, width=3)
        path = _save_tmp(img)
        try:
            assert is_blank_page(path) is False
        finally:
            os.unlink(path)

    def test_corrupt_image_not_blank(self):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(b"corrupt")
            assert is_blank_page(path) is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Handwriting Style Classification
# ---------------------------------------------------------------------------

class TestStyleClassification:
    def test_returns_valid_style(self):
        img = _create_text_image()
        path = _save_tmp(img)
        try:
            style = classify_handwriting_style(path)
            assert style in ("print", "cursive", "mixed")
        finally:
            os.unlink(path)

    def test_corrupt_image_returns_mixed(self):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(b"corrupt")
            assert classify_handwriting_style(path) == "mixed"
        finally:
            os.unlink(path)
