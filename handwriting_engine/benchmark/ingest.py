"""
Image ingestion for the benchmark database.

Imports images from directories, deduplicates by SHA-256, and auto-assesses quality.
Supports synthetic degradation for data amplification.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from pathlib import Path

from handwriting_engine.benchmark.db import (
    get_connection,
    get_sample_by_hash,
    insert_ground_truth,
    insert_quality_assessment,
    insert_sample,
)
from handwriting_engine.benchmark.models import Sample

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def hash_file(path: str | Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_page_number(filename: str) -> int:
    """Try to extract a page number from filenames like 'page_003.png'.

    Prefers 'page_N' patterns. Falls back to last number in filename.
    """
    # Try page-specific pattern first
    match = re.search(r"page[_\-]?(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fall back to last number in filename
    numbers = re.findall(r"(\d+)", filename)
    return int(numbers[-1]) if numbers else 0


def ingest_directory(
    directory: str | Path,
    student: str = "",
    category: str = "",
    db_path: Path | str | None = None,
    assess_quality: bool = True,
) -> list[Sample]:
    """Import all images from a directory into the benchmark database.

    Deduplicates by file content hash — re-importing the same images is safe.

    Args:
        directory: Path to directory containing image files.
        student: Student name metadata for all imported samples.
        category: Category metadata (e.g. 'biology', 'biol107').
        db_path: Override database path. Default: ~/.handwriting-engine/benchmark.db.
        assess_quality: Run quality assessment on each imported image.

    Returns:
        List of Sample objects for newly imported images (skips duplicates).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    conn = get_connection(db_path)
    try:
        imported: list[Sample] = []
        source_dir = str(directory.resolve())

        image_files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        for img_path in image_files:
            img_hash = hash_file(img_path)
            existing = get_sample_by_hash(conn, img_hash)
            if existing:
                logger.info("Skipping duplicate: %s (matches sample %d)", img_path.name, existing.id)
                continue

            page_num = _extract_page_number(img_path.stem)
            try:
                sample_id = insert_sample(
                    conn,
                    image_path=str(img_path.resolve()),
                    image_hash=img_hash,
                    student=student,
                    category=category,
                    source_dir=source_dir,
                    page_number=page_num,
                )
            except sqlite3.IntegrityError:
                logger.warning("Duplicate hash race condition for %s, skipping", img_path.name)
                continue

            sample = Sample(
                id=sample_id,
                image_path=str(img_path.resolve()),
                image_hash=img_hash,
                student=student,
                category=category,
                page_number=page_num,
            )
            imported.append(sample)

            if assess_quality:
                try:
                    from handwriting_engine.quality import assess_image
                    assessment = assess_image(str(img_path.resolve()))
                    insert_quality_assessment(conn, sample_id, assessment)
                except Exception as e:
                    logger.warning("Quality assessment failed for %s: %s", img_path.name, e)

        return imported
    finally:
        conn.close()


def ingest_single(
    image_path: str | Path,
    student: str = "",
    category: str = "",
    db_path: Path | str | None = None,
) -> Sample | None:
    """Import a single image. Returns None if it's a duplicate.

    Args:
        image_path: Path to the image file.
        student: Student name metadata.
        category: Category metadata.
        db_path: Override database path.

    Returns:
        Sample object if newly imported, None if duplicate.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")

    conn = get_connection(db_path)
    try:
        img_hash = hash_file(image_path)
        existing = get_sample_by_hash(conn, img_hash)
        if existing:
            return None

        page_num = _extract_page_number(image_path.stem)
        try:
            sample_id = insert_sample(
                conn,
                image_path=str(image_path.resolve()),
                image_hash=img_hash,
                student=student,
                category=category,
                source_dir=str(image_path.parent.resolve()),
                page_number=page_num,
            )
        except sqlite3.IntegrityError:
            return None

        return Sample(
            id=sample_id,
            image_path=str(image_path.resolve()),
            image_hash=img_hash,
            student=student,
            category=category,
            page_number=page_num,
        )
    finally:
        conn.close()


def generate_degraded_variants(
    sample_id: int,
    output_dir: str | Path,
    db_path: Path | str | None = None,
) -> list[Sample]:
    """Generate synthetic degraded variants of a sample for data amplification.

    Creates: blurred, low-contrast, rotated, noisy, and cropped variants.
    Each variant shares the same ground truth as the original.

    Args:
        sample_id: ID of the original sample to degrade.
        output_dir: Directory to save generated images.
        db_path: Override database path.

    Returns:
        List of newly created Sample objects for the variants.
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import random

    from handwriting_engine.benchmark.db import get_sample_by_id, get_latest_ground_truth

    conn = get_connection(db_path)
    try:
        sample = get_sample_by_id(conn, sample_id)
        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        gt = get_latest_ground_truth(conn, sample_id)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(sample.image_path).convert("RGB")  # Normalize to RGB
        variants: list[Sample] = []
        stem = Path(sample.image_path).stem

        # Seed RNG from image hash for reproducible variants
        rng = random.Random(sample.image_hash)

        degradations = {
            "blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)),
            "lowcontrast": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
            "rotate": lambda im: im.rotate(rng.uniform(2, 5), expand=True, fillcolor=(255, 255, 255)),
            "noise": lambda im: _add_noise(im, seed=hash(sample.image_hash)),
            "crop80": lambda im: _center_crop(im, 0.8),
            "perspective": lambda im: _perspective_warp(im, seed=hash(sample.image_hash)),
            "elastic": lambda im: _elastic_distortion(im, seed=hash(sample.image_hash)),
        }

        for name, transform in degradations.items():
            out_path = output_dir / f"{stem}_{name}.png"
            degraded = transform(img.copy())
            degraded.save(str(out_path))

            img_hash = hash_file(out_path)
            try:
                sid = insert_sample(
                    conn,
                    image_path=str(out_path.resolve()),
                    image_hash=img_hash,
                    student=sample.student,
                    category=sample.category,
                    page_number=sample.page_number,
                    notes=f"degraded:{name} from sample {sample_id}",
                )
            except sqlite3.IntegrityError:
                continue

            # Share ground truth with original
            if gt:
                insert_ground_truth(conn, sid, gt.text, source=f"degraded:{name}")

            variants.append(Sample(
                id=sid,
                image_path=str(out_path.resolve()),
                image_hash=img_hash,
                student=sample.student,
                category=sample.category,
                page_number=sample.page_number,
                has_ground_truth=gt is not None,
            ))

        return variants
    finally:
        conn.close()


def _add_noise(img, seed: int = 0):
    """Add salt-and-pepper noise to an image. Works on any mode."""
    import random as rng_mod

    rng = rng_mod.Random(seed)
    img = img.convert("RGB")
    pixels = img.load()
    w, h = img.size
    num_noise = int(w * h * 0.01)
    for _ in range(num_noise):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        pixels[x, y] = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
    return img


def _center_crop(img, fraction: float):
    """Crop to center fraction of image."""
    w, h = img.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def _perspective_warp(img, seed: int = 0):
    """Simulate phone camera angle with a perspective transform."""
    import random as rng_mod
    from PIL import Image

    rng = rng_mod.Random(seed)
    w, h = img.size
    # Slight perspective shift (5-10% of dimensions)
    dx = int(w * rng.uniform(0.03, 0.07))
    dy = int(h * rng.uniform(0.03, 0.07))

    coeffs = _find_perspective_coeffs(
        [(dx, dy), (w - dx, 0), (w, h - dy), (0, h)],  # destination
        [(0, 0), (w, 0), (w, h), (0, h)],               # source
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))


def _find_perspective_coeffs(dst, src):
    """Compute perspective transform coefficients."""
    import numpy as np

    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=float)
    B = np.array([c for pair in src for c in pair], dtype=float)
    res = np.linalg.solve(A, B)
    return tuple(res.flatten())


def _elastic_distortion(img, seed: int = 0, alpha: float = 15.0, sigma: float = 3.0):
    """Apply elastic distortion — the highest-impact HTR augmentation.

    Creates a smooth random displacement field and applies it to the image.
    """
    import numpy as np
    from PIL import Image as PILImage

    rng = np.random.RandomState(seed % (2**31))
    w, h = img.size
    arr = np.array(img)

    # Random displacement fields
    dx = rng.uniform(-1, 1, (h, w)) * alpha
    dy = rng.uniform(-1, 1, (h, w)) * alpha

    # Smooth with Gaussian-like averaging (simple box filter approximation)
    kernel_size = int(sigma * 3) | 1
    for _ in range(3):
        dx = _box_filter_2d(dx, kernel_size)
        dy = _box_filter_2d(dy, kernel_size)

    # Apply displacement
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.int32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.int32)
    result = arr[map_y, map_x]

    return PILImage.fromarray(result)


def _box_filter_2d(arr, kernel_size: int):
    """Simple box filter for smoothing displacement fields."""
    import numpy as np

    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode="reflect")
    result = np.zeros_like(arr)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            result += padded[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]
    return result / (kernel_size * kernel_size)


def bootstrap_ground_truth(
    db_path: Path | str | None = None,
    agreement_threshold: float = 0.02,
    confidence_threshold: float = 0.85,
) -> int:
    """Auto-generate ground truth from high-agreement consensus reads.

    For samples without ground truth, if all available provider outputs
    for that sample agree within the CER threshold and confidence exceeds
    the threshold, register the consensus as auto ground truth.

    Args:
        db_path: Override database path.
        agreement_threshold: Max CER between any two provider outputs.
        confidence_threshold: Min confidence for auto-GT.

    Returns:
        Number of auto-generated ground truths.
    """
    from handwriting_engine.benchmark.db import list_samples, get_latest_ground_truth
    from handwriting_engine.benchmark.metrics import character_error_rate

    conn = get_connection(db_path)
    try:
        samples = list_samples(conn)
        count = 0

        for sample in samples:
            if sample.has_ground_truth:
                continue

            # Get best output per DISTINCT provider (not per-run duplicates)
            rows = conn.execute(
                """SELECT provider, output_text, confidence FROM provider_outputs
                   WHERE sample_id = ? AND error IS NULL AND strategy = 'single'
                   AND provider NOT LIKE '%+%'
                   ORDER BY created_at DESC""",
                (sample.id,),
            ).fetchall()

            # Deduplicate: keep only the most recent output per provider
            seen_providers: dict[str, dict] = {}
            for r in rows:
                if r["provider"] not in seen_providers:
                    seen_providers[r["provider"]] = dict(r)

            if len(seen_providers) < 2:
                continue

            texts = [v["output_text"] for v in seen_providers.values()]
            confidences = [v["confidence"] for v in seen_providers.values()]

            # Check all pairs agree within threshold
            all_agree = True
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    cer, _, _ = character_error_rate(texts[i], texts[j])
                    if cer > agreement_threshold:
                        all_agree = False
                        break
                if not all_agree:
                    break

            if all_agree and min(confidences) >= confidence_threshold:
                # Use the longest text as ground truth (most complete)
                best_text = max(texts, key=len)
                insert_ground_truth(conn, sample.id, best_text, source="auto_consensus")
                count += 1
                logger.info("Auto-GT for sample %d (%.0f%% min confidence, %d providers agree)",
                           sample.id, min(confidences) * 100, len(texts))

        return count
    finally:
        conn.close()
