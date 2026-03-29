"""
Academic integrity detection utilities.

Provides lightweight checks for text similarity between submissions,
typed-text detection in handwriting images, and anomaly flagging.
Uses simple approaches (Jaccard similarity, pixel variance) to avoid
heavy dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Text similarity (Jaccard on word-level n-grams)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return text.lower().split()


def _ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
    """Generate word-level n-grams from a token list."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def check_text_similarity(
    texts: list[str],
    threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """Compare all pairs of texts, return pairs exceeding similarity threshold.

    Uses Jaccard similarity on word-level trigrams -- simple but effective
    for detecting near-identical submissions.

    Args:
        texts: List of transcribed submission texts.
        threshold: Minimum similarity score to flag (0.0 - 1.0).

    Returns:
        List of (index_a, index_b, similarity_score) for flagged pairs,
        sorted by similarity descending.
    """
    if len(texts) < 2:
        return []

    # Pre-compute n-gram sets
    gram_sets = [_ngrams(_tokenize(t)) for t in texts]

    flagged: list[tuple[int, int, float]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = _jaccard(gram_sets[i], gram_sets[j])
            if score >= threshold:
                flagged.append((i, j, round(score, 4)))

    flagged.sort(key=lambda x: x[2], reverse=True)
    return flagged


# ---------------------------------------------------------------------------
# Typed-text detection via pixel variance analysis
# ---------------------------------------------------------------------------

def detect_typed_text(image_path: str) -> dict:
    """Analyze an image for signs of typed (non-handwritten) text.

    Uses pixel-level variance analysis.  Typed text tends to have very
    uniform stroke widths and consistent baseline spacing, resulting in
    lower local variance in certain metrics compared to handwriting.

    Requires Pillow (PIL).  If Pillow is not installed, returns a result
    with is_typed=False and a note about the missing dependency.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys:
            is_typed (bool): Whether the image likely contains typed text.
            confidence (float): 0.0 - 1.0 confidence in the assessment.
            indicators (list[str]): Human-readable reasons for the assessment.
    """
    path = Path(image_path)
    if not path.exists():
        return {
            "is_typed": False,
            "confidence": 0.0,
            "indicators": [f"File not found: {image_path}"],
        }

    try:
        from PIL import Image, ImageStat
    except ImportError:
        return {
            "is_typed": False,
            "confidence": 0.0,
            "indicators": ["Pillow not installed -- cannot analyze image"],
        }

    indicators: list[str] = []
    typed_score = 0.0

    try:
        img = Image.open(path).convert("L")  # grayscale
    except Exception as e:
        return {
            "is_typed": False,
            "confidence": 0.0,
            "indicators": [f"Could not open image: {e}"],
        }

    width, height = img.size
    if width < 50 or height < 50:
        return {
            "is_typed": False,
            "confidence": 0.0,
            "indicators": ["Image too small for analysis"],
        }

    # --- Metric 1: Global standard deviation ---
    # Typed text on white background has high contrast, low mid-tone variance.
    stat = ImageStat.Stat(img)
    global_std = stat.stddev[0]

    # --- Metric 2: Row-wise variance consistency ---
    # Typed text has very consistent row variance (uniform line heights).
    row_variances: list[float] = []
    sample_rows = min(height, 100)
    step = max(1, height // sample_rows)
    for y in range(0, height, step):
        row_pixels = list(img.crop((0, y, width, y + 1)).getdata())
        if len(row_pixels) < 2:
            continue
        mean = sum(row_pixels) / len(row_pixels)
        var = sum((p - mean) ** 2 for p in row_pixels) / len(row_pixels)
        row_variances.append(var)

    if row_variances:
        rv_mean = sum(row_variances) / len(row_variances)
        rv_std = math.sqrt(
            sum((v - rv_mean) ** 2 for v in row_variances) / len(row_variances)
        )
        # Coefficient of variation of row variances
        rv_cv = rv_std / rv_mean if rv_mean > 0 else 0

        # Typed text: lower CV (more uniform rows)
        if rv_cv < 0.4:
            typed_score += 0.3
            indicators.append(
                f"Row variance is very uniform (CV={rv_cv:.2f}) -- typical of typed text"
            )
        elif rv_cv < 0.7:
            typed_score += 0.1
            indicators.append(
                f"Row variance is moderately uniform (CV={rv_cv:.2f})"
            )
        else:
            indicators.append(
                f"Row variance is irregular (CV={rv_cv:.2f}) -- typical of handwriting"
            )

    # --- Metric 3: Bimodal pixel distribution ---
    # Typed text on white paper is strongly bimodal (black text, white bg).
    histogram = img.histogram()
    total_pixels = width * height
    dark_pixels = sum(histogram[:80])
    light_pixels = sum(histogram[200:])
    mid_pixels = sum(histogram[80:200])

    dark_ratio = dark_pixels / total_pixels
    light_ratio = light_pixels / total_pixels
    mid_ratio = mid_pixels / total_pixels

    if mid_ratio < 0.15 and (dark_ratio > 0.05 and light_ratio > 0.5):
        typed_score += 0.35
        indicators.append(
            f"Strongly bimodal pixel distribution (mid={mid_ratio:.1%}) -- "
            f"typical of typed/printed text"
        )
    elif mid_ratio < 0.25:
        typed_score += 0.15
        indicators.append(
            f"Somewhat bimodal pixel distribution (mid={mid_ratio:.1%})"
        )
    else:
        indicators.append(
            f"Smooth pixel distribution (mid={mid_ratio:.1%}) -- "
            f"typical of handwriting or photos"
        )

    # --- Metric 4: Horizontal projection regularity ---
    # Typed text has very regular line spacing visible in horizontal projection.
    col_sums: list[int] = []
    for x in range(0, width, max(1, width // 100)):
        col = list(img.crop((x, 0, x + 1, height)).getdata())
        col_sums.append(sum(255 - p for p in col))  # ink density

    if col_sums:
        cs_mean = sum(col_sums) / len(col_sums)
        cs_std = math.sqrt(
            sum((c - cs_mean) ** 2 for c in col_sums) / len(col_sums)
        )
        cs_cv = cs_std / cs_mean if cs_mean > 0 else 0

        if cs_cv < 0.3:
            typed_score += 0.2
            indicators.append(
                f"Very even horizontal ink distribution (CV={cs_cv:.2f}) -- "
                f"suggests typed/printed text"
            )
        elif cs_cv > 0.6:
            indicators.append(
                f"Uneven horizontal ink distribution (CV={cs_cv:.2f}) -- "
                f"suggests handwriting"
            )

    # --- Metric 5: High global contrast with low noise ---
    if global_std > 60 and mid_ratio < 0.2:
        typed_score += 0.15
        indicators.append(
            f"High contrast (std={global_std:.1f}) with clean background"
        )

    # Clamp confidence
    confidence = min(typed_score, 1.0)
    is_typed = confidence >= 0.5

    if is_typed:
        indicators.insert(0, "LIKELY TYPED/PRINTED TEXT DETECTED")
    else:
        indicators.insert(0, "Text appears to be handwritten")

    return {
        "is_typed": is_typed,
        "confidence": round(confidence, 3),
        "indicators": indicators,
    }


# ---------------------------------------------------------------------------
# Anomaly flagging across a batch of submissions
# ---------------------------------------------------------------------------

def flag_anomalies(submissions: list[dict]) -> list[dict]:
    """Flag submissions with integrity concerns.

    Runs text similarity checks across all submissions and (if image
    paths are provided) checks for typed text.

    Args:
        submissions: List of dicts, each with keys:
            - 'name' (str): Student/submission identifier
            - 'text' (str): Transcribed submission text
            - 'image_path' (str, optional): Path to the submission image

    Returns:
        List of flagged items.  Each dict contains:
            - 'name': The submission name
            - 'reasons': List of human-readable concern strings
            - 'details': Dict with supporting data
    """
    if not submissions:
        return []

    flagged: list[dict] = []
    flagged_indices: set[int] = set()

    # --- Text similarity ---
    texts = [s.get("text", "") for s in submissions]
    similar_pairs = check_text_similarity(texts, threshold=0.85)

    for idx_a, idx_b, score in similar_pairs:
        flagged_indices.add(idx_a)
        flagged_indices.add(idx_b)

    # --- Typed text detection ---
    typed_indices: set[int] = set()
    for i, sub in enumerate(submissions):
        img_path = sub.get("image_path")
        if img_path:
            result = detect_typed_text(img_path)
            if result["is_typed"]:
                typed_indices.add(i)

    # --- Assemble flagged items ---
    all_flagged = flagged_indices | typed_indices
    for i in sorted(all_flagged):
        sub = submissions[i]
        reasons: list[str] = []
        details: dict = {}

        # Similarity reasons
        related_pairs = [(a, b, s) for a, b, s in similar_pairs if a == i or b == i]
        if related_pairs:
            for a, b, score in related_pairs:
                other = b if a == i else a
                other_name = submissions[other].get("name", f"submission_{other}")
                reasons.append(
                    f"High text similarity ({score:.0%}) with {other_name}"
                )
            details["similarity_pairs"] = [
                {
                    "other_index": b if a == i else a,
                    "other_name": submissions[b if a == i else a].get("name", ""),
                    "score": s,
                }
                for a, b, s in related_pairs
            ]

        # Typed text reasons
        if i in typed_indices:
            img_path = sub.get("image_path", "")
            result = detect_typed_text(img_path)
            reasons.append(
                f"Possible typed/printed text (confidence: {result['confidence']:.0%})"
            )
            details["typed_detection"] = result

        flagged.append({
            "name": sub.get("name", f"submission_{i}"),
            "reasons": reasons,
            "details": details,
        })

    return flagged
