"""
Writer identification via Gemini multimodal embeddings.

Uses gemini-embedding-2-preview to embed handwriting images into a vector
space where similar handwriting styles cluster together. This enables:
- Automatic writer identification from handwriting samples
- Writer style clustering without manual labeling
- Auto-loading calibration profiles for recognized writers

Embedding dimensions: 768 (fast), 1536 (balanced), 3072 (max discrimination)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_DIMENSIONS = 768  # Fast and sufficient for style clustering
PROFILES_DIR = Path.home() / ".handwriting-engine" / "writer_profiles"


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length.

    Gemini embeddings below 3072 dimensions are NOT normalized by default.
    Without normalization, cosine similarity and centroid averaging produce
    distorted results.
    """
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def embed_image(
    image_path: str,
    dimensions: int = DEFAULT_DIMENSIONS,
    api_key: str | None = None,
) -> list[float]:
    """Generate an embedding vector for a handwriting image.

    Args:
        image_path: Path to the handwriting image
        dimensions: Embedding size (768, 1536, or 3072)
        api_key: Google API key (defaults to GOOGLE_API_KEY env var)

    Returns:
        List of floats representing the embedding vector
    """
    from google import genai
    from google.genai import types

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Detect media type
    ext = Path(image_path).suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    media_type = media_types.get(ext, "image/jpeg")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=media_type)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[image_part],
        config=types.EmbedContentConfig(output_dimensionality=dimensions),
    )

    vec = list(response.embeddings[0].values)
    # Normalize: Gemini embeddings < 3072 dims are NOT unit-length by default
    if dimensions < 3072:
        vec = _l2_normalize(vec)
    return vec


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def save_writer_profile(
    writer_id: str,
    embeddings: list[list[float]],
    metadata: dict | None = None,
) -> Path:
    """Save a writer's embedding profile to disk.

    A profile is the average of multiple sample embeddings from the same writer.
    More samples = more robust identification.

    Args:
        writer_id: Unique writer identifier
        embeddings: List of embedding vectors from multiple samples
        metadata: Optional metadata (name, notes, etc.)

    Returns:
        Path to the saved profile
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Average the embeddings to get a centroid
    if not embeddings:
        raise ValueError("At least one embedding required")

    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            centroid[i] += v
    centroid = [v / len(embeddings) for v in centroid]
    # Re-normalize: averaging normalized vectors does NOT produce a normalized vector
    centroid = _l2_normalize(centroid)

    profile = {
        "writer_id": writer_id,
        "centroid": centroid,
        "sample_count": len(embeddings),
        "dimensions": dim,
        "metadata": metadata or {},
    }

    path = PROFILES_DIR / f"{writer_id}.json"
    with open(path, "w") as f:
        json.dump(profile, f)

    logger.info("Saved writer profile %s (%d samples, %d dims)", writer_id, len(embeddings), dim)
    return path


def load_writer_profiles() -> dict[str, dict]:
    """Load all saved writer profiles.

    Returns:
        Dict mapping writer_id -> {centroid, sample_count, dimensions, metadata}
    """
    profiles = {}
    if not PROFILES_DIR.exists():
        return profiles

    for path in PROFILES_DIR.glob("*.json"):
        try:
            with open(path) as f:
                profile = json.load(f)
            profiles[profile["writer_id"]] = profile
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping corrupt profile %s: %s", path, e)

    return profiles


def identify_writer(
    image_path: str,
    threshold: float = 0.85,
    dimensions: int = DEFAULT_DIMENSIONS,
    api_key: str | None = None,
) -> tuple[str | None, float]:
    """Identify the writer of a handwriting sample.

    Embeds the image and compares against all saved writer profiles.

    Args:
        image_path: Path to the handwriting image
        threshold: Minimum cosine similarity to consider a match
        dimensions: Embedding dimensions (must match saved profiles)
        api_key: Google API key

    Returns:
        Tuple of (writer_id or None, similarity_score)
    """
    profiles = load_writer_profiles()
    if not profiles:
        return None, 0.0

    embedding = embed_image(image_path, dimensions=dimensions, api_key=api_key)

    best_id = None
    best_score = 0.0

    for writer_id, profile in profiles.items():
        if profile["dimensions"] != dimensions:
            continue
        score = cosine_similarity(embedding, profile["centroid"])
        if score > best_score:
            best_score = score
            best_id = writer_id

    if best_score >= threshold:
        logger.info("Writer identified: %s (similarity %.3f)", best_id, best_score)
        return best_id, best_score

    logger.debug("No writer match above threshold %.2f (best: %.3f)", threshold, best_score)
    return None, best_score


def enroll_writer(
    writer_id: str,
    image_paths: list[str],
    dimensions: int = DEFAULT_DIMENSIONS,
    metadata: dict | None = None,
    api_key: str | None = None,
) -> Path:
    """Enroll a new writer by embedding multiple handwriting samples.

    Provide 3-5 samples for best results. The profile is the centroid
    of all sample embeddings.

    Args:
        writer_id: Unique identifier for this writer
        image_paths: Paths to handwriting samples (3-5 recommended)
        dimensions: Embedding dimensions
        metadata: Optional metadata
        api_key: Google API key

    Returns:
        Path to the saved profile
    """
    if len(image_paths) < 1:
        raise ValueError("At least 1 image required (3-5 recommended)")

    embeddings = []
    for path in image_paths:
        try:
            emb = embed_image(path, dimensions=dimensions, api_key=api_key)
            embeddings.append(emb)
            logger.debug("Embedded %s for writer %s", path, writer_id)
        except Exception as e:
            logger.warning("Failed to embed %s: %s", path, e)

    if not embeddings:
        raise RuntimeError(f"No images could be embedded for writer {writer_id}")

    return save_writer_profile(writer_id, embeddings, metadata=metadata)
