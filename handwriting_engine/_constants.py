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

# Proven enhancement pipeline defaults (from BIOL 107 session)
# Contrast lowered from 2.5 to 2.0: PreP-OCR (ACL 2025) found heavy contrast
# causes LLMs to produce "contextually plausible but factually incorrect hallucinations"
# Lowered from 3.0: PreP-OCR (ACL 2025) found heavy sharpening creates artifacts
# that LLMs misinterpret. 2.0 matches the contrast reduction rationale.
PROVEN_SHARPEN = 2.0
PROVEN_CONTRAST = 2.0
PROVEN_BRIGHTNESS = 1.1
PROVEN_AUTOCONTRAST_CUTOFF = 2
PROVEN_UPSCALE = 2

# Image-crisp sharpen presets: (unsharp_radius, unsharp_percent, unsharp_threshold, detail_passes, contrast_factor)
CRISP_PRESETS = {
    "light": (1.5, 80, 2, 1, 1.05),
    "medium": (2.0, 120, 1, 1, 1.10),
    "heavy": (2.5, 180, 0, 2, 1.15),
}

# Default model names — pinned versions prevent silent regression
# (GPT-4o-mini study showed WER doubling from 15% to 32% over 6 months with rolling model)
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_QUALITY_MODEL = "gemini-2.5-pro"  # For quality reads (better accuracy, higher cost)

# Gemini temperature: 0 is optimal for OCR determinism on Gemini 2.5 Flash
# (Earlier Southbridge.AI concern about degenerate sampling was specific to 1.x Flash;
# 2.5 Flash handles temp=0 correctly. Research consensus favors temp=0 for OCR.)
DEFAULT_GEMINI_TEMPERATURE = 0.0

# Deskew — auto-correct rotated scans before enhancement
SKEW_THRESHOLD_DEGREES = 1.5  # Only deskew if detected angle exceeds this

# Zoomed crop verification — re-read uncertain regions at higher resolution
ZOOM_VERIFY_STRIPS = 3           # Split page into N horizontal strips
ZOOM_VERIFY_CONFIDENCE_LOW = 0.45   # Below this: full retry (existing behavior)
ZOOM_VERIFY_CONFIDENCE_HIGH = 0.65  # Above this: no retry needed
ZOOM_VERIFY_MIN_MARKERS = 1     # Minimum [?] markers to trigger zoom verify

# Smart routing — adaptive consensus based on image quality
SMART_ROUTE_EASY_BLUR = 200       # High blur score = sharp image -> single provider
SMART_ROUTE_EASY_CONTRAST = 0.6   # High contrast = clear text -> single provider
SMART_ROUTE_HARD_BLUR = 80        # Low blur score = blurry -> full consensus
SMART_ROUTE_HARD_CONTRAST = 0.3   # Low contrast = faded -> full consensus

# Vocabulary priming — cheap pre-scan extracts legible words for context
VOCAB_PRIMING_MAX_TOKENS = 200

# Dual-polarity reading — inverted images for faint ink detection
DUAL_POLARITY_ENABLED = True

# max_tokens defaults — typical handwriting OCR outputs are 500–1,500 tokens.
# Prior defaults (4096 single, 8192 batch) left large headroom that bills against
# Anthropic's reserved allocation and doesn't help accuracy. Callers can still
# override via the `max_tokens` parameter for long-output edge cases.
MAX_TOKENS_SINGLE = 2048
MAX_TOKENS_BATCH = 4096

# API timeout (seconds) — prevents hanging calls from blocking the pipeline
API_TIMEOUT_SECONDS = 120

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60

# Cost per 1M tokens (USD) — single source of truth
COST_PER_1M_TOKENS = {
    "claude": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 0.15, "output": 0.60},
    "openai": {"input": 2.00, "output": 8.00},
}
