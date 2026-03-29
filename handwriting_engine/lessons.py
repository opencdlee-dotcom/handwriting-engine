"""
Learn-from-mistakes persistence engine.

Saves and loads "lessons" -- short notes about past misreadings or grading
errors -- so that subsequent vision calls can be primed with corrections
the user has already made.

Lessons are stored as newline-delimited JSON files, one per category,
inside ~/.handwriting-engine/lessons/ by default.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LESSONS_DIR = Path.home() / ".handwriting-engine" / "lessons"


def _lessons_file(category: str, lessons_dir: Path | None) -> Path:
    """Return the path to the lessons file for a category."""
    base = Path(lessons_dir) if lessons_dir else DEFAULT_LESSONS_DIR
    base.mkdir(parents=True, exist_ok=True)
    # Sanitize category name for filesystem safety
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in category)
    return base / f"{safe}.jsonl"


def save_lesson(
    lesson_text: str,
    category: str = "general",
    lessons_dir: Path | str | None = None,
) -> None:
    """Append a lesson to the lessons file.

    Args:
        lesson_text: Human-readable description of what was learned
            (e.g. "Student writes 'B' with very angular strokes that look
            like 'D' -- always check both loops").
        category: Grouping key (e.g. 'general', 'biology', 'math',
            'student_alice').
        lessons_dir: Override the default storage directory.
    """
    # Deduplicate: skip if identical lesson already exists in recent entries
    existing = load_lessons(category=category, limit=15, lessons_dir=lessons_dir)
    normalized = lesson_text.strip()
    if any(normalized == e.strip() for e in existing):
        return

    path = _lessons_file(category, lessons_dir)
    entry = {
        "text": normalized,
        "category": category,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_lessons(
    category: str = "general",
    limit: int = 15,
    lessons_dir: Path | str | None = None,
) -> list[str]:
    """Load the most recent lessons for a category.

    Args:
        category: Which lesson category to load.
        limit: Maximum number of lessons to return (most recent first).
        lessons_dir: Override the default storage directory.

    Returns:
        List of lesson text strings, most recent first, up to *limit*.
    """
    path = _lessons_file(category, lessons_dir)
    if not path.exists():
        return []

    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Most recent first, capped at limit
    entries = entries[-limit:]
    entries.reverse()
    return [e["text"] for e in entries if "text" in e]


def build_lessons_prompt(
    category: str = "general",
    lessons_dir: Path | str | None = None,
) -> str:
    """Build a prompt section from saved lessons for injection into vision calls.

    Returns an empty string if no lessons exist, so callers can safely
    concatenate without checking.

    Args:
        category: Which lesson category to pull from.
        lessons_dir: Override the default storage directory.

    Returns:
        A formatted prompt block, or '' if no lessons are available.
    """
    lessons = load_lessons(category=category, lessons_dir=lessons_dir)
    if not lessons:
        return ""

    lines = ["=== LESSONS FROM PREVIOUS GRADING SESSIONS ==="]
    lines.append("Apply these corrections learned from past errors:\n")
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"{i}. {lesson}")
    lines.append("")
    return "\n".join(lines)


def auto_extract_lessons(
    model_output: str,
    corrected_text: str,
    category: str = "general",
    lessons_dir: Path | str | None = None,
) -> int:
    """Automatically extract and save lessons by diffing model output vs correction.

    Compares the model's reading against the human-corrected version and saves
    each meaningful word-level difference as a lesson for future reads.

    Args:
        model_output: What the vision model produced.
        corrected_text: The human-corrected version.
        category: Lesson category (e.g. 'biology', 'student_alice').
        lessons_dir: Override the default storage directory.

    Returns:
        Number of lessons extracted and saved.
    """
    from difflib import SequenceMatcher

    words_model = model_output.split()
    words_correct = corrected_text.split()
    matcher = SequenceMatcher(None, words_model, words_correct)
    count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            original = " ".join(words_model[i1:i2])
            fixed = " ".join(words_correct[j1:j2])
            if original.strip() and fixed.strip() and original != fixed:
                lesson = f"Model read '{original}' but correct reading is '{fixed}'"
                save_lesson(lesson, category=category, lessons_dir=lessons_dir)
                count += 1
        elif tag == "delete":
            original = " ".join(words_model[i1:i2])
            if original.strip():
                lesson = f"Model hallucinated '{original}' — this text was not present"
                save_lesson(lesson, category=category, lessons_dir=lessons_dir)
                count += 1
        elif tag == "insert":
            missed = " ".join(words_correct[j1:j2])
            if missed.strip():
                lesson = f"Model missed '{missed}' — this text was present but not read"
                save_lesson(lesson, category=category, lessons_dir=lessons_dir)
                count += 1

    return count


def clear_lessons(
    category: str = "general",
    lessons_dir: Path | str | None = None,
) -> None:
    """Clear all lessons for a category.

    Args:
        category: Which lesson category to clear.
        lessons_dir: Override the default storage directory.
    """
    path = _lessons_file(category, lessons_dir)
    if path.exists():
        path.unlink()
