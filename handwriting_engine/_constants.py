"""
Single source of truth for all handwriting engine constants.
Consumers override via their own config.py or by passing parameters explicitly.
"""

# PDF rendering (200 DPI — research shows 150 is at the accuracy cliff for LLM-based OCR on handwriting)
PDF_DPI = 200
JPEG_QUALITY = 85

# Image optimization (Claude Vision API recommendations)
MAX_IMAGE_LONG_SIDE = 1568
MAX_IMAGE_FILE_BYTES = 3_500_000
MAX_REQUEST_BYTES = 19_000_000
RETRY_MAX_LONG_SIDE = 1024
RETRY_JPEG_QUALITY = 70

# Chunking
CHUNK_SIZE = 10
SURVEY_DPI = 72
SURVEY_QUALITY = 60
SURVEY_MAX_LONG_SIDE = 800

# Adaptive DPI tiers: (max_pages, dpi, quality, max_long_side)
DPI_TIERS = [
    (15, 200, 85, 1568),
    (30, 150, 80, 1568),
    (50, 120, 75, 1280),
    (999, 100, 70, 1024),
]

# Answer sheet cropping
CROP_DPI = 300
CROP_PADDING = 20
CROP_MIN_HEIGHT = 80
CROP_JPEG_QUALITY = 95

# Quality assessment thresholds
BLUR_THRESHOLD = 100
CONTRAST_THRESHOLD = 0.4
BRIGHTNESS_LOW = 100
BRIGHTNESS_HIGH = 220
FAINT_INK_BRIGHTNESS = 180
FAINT_INK_CONTRAST = 0.5

# Proven enhancement pipeline defaults (from BIOL 107 session — 15-20% improvement)
PROVEN_SHARPEN = 3.0
PROVEN_CONTRAST = 2.5
PROVEN_BRIGHTNESS = 1.1
PROVEN_AUTOCONTRAST_CUTOFF = 2
PROVEN_UPSCALE = 2

# Image-crisp sharpen presets: (unsharp_radius, unsharp_percent, unsharp_threshold, detail_passes, contrast_factor)
CRISP_PRESETS = {
    "light": (1.5, 80, 2, 1, 1.05),
    "medium": (2.0, 120, 1, 1, 1.10),
    "heavy": (2.5, 180, 0, 2, 1.15),
}

# Default model names
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_QUALITY_MODEL = "gemini-2.5-pro"  # For quality reads (better accuracy, higher cost)

# API timeout (seconds) — prevents hanging calls from blocking the pipeline
API_TIMEOUT_SECONDS = 120

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60
