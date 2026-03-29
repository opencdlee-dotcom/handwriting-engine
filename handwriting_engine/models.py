"""
Core data models for the handwriting engine.

Provides structured representations for page metadata, quality assessments,
multi-provider consensus results, and exception hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Canonical ConsensusResult lives in providers.base — re-export for backwards compat
from handwriting_engine.providers.base import ConsensusResult  # noqa: F401


# --- Exception hierarchy ---

class HandwritingEngineError(Exception):
    """Base exception for all handwriting engine errors."""


class ProviderError(HandwritingEngineError):
    """Error from a vision provider (API failure, auth, rate limit exhausted)."""


class ImageError(HandwritingEngineError):
    """Error processing an image (corrupt, unsupported format, too large)."""


class ConfigError(HandwritingEngineError):
    """Configuration error (missing API key, invalid settings)."""


# --- Enums ---

class PageOrientation(Enum):
    """Detected orientation of a scanned page."""

    NORMAL = "normal"
    SIDEWAYS = "sideways"
    UPSIDE_DOWN = "upside_down"


class ContentType(Enum):
    """Classification of the content found on a page."""

    UNKNOWN = "unknown"
    ANSWER_SHEET = "answer_sheet"
    FREE_RESPONSE = "free_response"
    LAB_NOTEBOOK = "lab_notebook"
    DATA_TABLE = "data_table"
    GRAPH = "graph"


class ImageQuality(Enum):
    """Overall quality rating for a page image."""

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# --- Data classes ---

@dataclass
class PageInfo:
    """
    Metadata for a single page image in the grading pipeline.

    Tracks the page's source path, detected orientation, content type,
    quality rating, optional enhanced copy, and any transcription results.
    """

    page_number: int
    image_path: str
    orientation: PageOrientation = PageOrientation.NORMAL
    content_type: ContentType = ContentType.UNKNOWN
    quality: ImageQuality = ImageQuality.GOOD
    enhanced_path: Optional[str] = None
    was_auto_rotated: bool = False
    original_rotation: int = 0
    transcription: str = ""
    confidence_notes: str = ""
    assessment: Optional[dict] = None
