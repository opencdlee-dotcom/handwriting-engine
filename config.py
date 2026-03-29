"""
User-facing config. Loads from .env, falls back to engine defaults.
Consumers can override by importing and modifying values after import.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional for library consumers

# Re-export all engine defaults
from handwriting_engine._constants import *  # noqa: F401, F403
from handwriting_engine._constants import DEFAULT_CLAUDE_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_GEMINI_MODEL

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
