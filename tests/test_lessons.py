"""Tests for the lessons persistence engine."""

import json
import tempfile
from pathlib import Path

import pytest

from handwriting_engine.lessons import (
    auto_extract_lessons,
    build_lessons_prompt,
    clear_lessons,
    load_lessons,
    save_lesson,
)


@pytest.fixture
def lessons_dir(tmp_path):
    """Provide a fresh temporary lessons directory."""
    return tmp_path / "lessons"


class TestSaveAndLoad:
    def test_save_and_load_basic(self, lessons_dir):
        save_lesson("Model read 'B' but correct is 'D'", category="bio", lessons_dir=lessons_dir)
        loaded = load_lessons(category="bio", lessons_dir=lessons_dir)
        assert len(loaded) == 1
        assert "Model read 'B' but correct is 'D'" in loaded[0]

    def test_save_multiple(self, lessons_dir):
        save_lesson("Lesson 1", category="general", lessons_dir=lessons_dir)
        save_lesson("Lesson 2", category="general", lessons_dir=lessons_dir)
        save_lesson("Lesson 3", category="general", lessons_dir=lessons_dir)
        loaded = load_lessons(category="general", lessons_dir=lessons_dir)
        assert len(loaded) == 3
        # Most recent first
        assert loaded[0] == "Lesson 3"
        assert loaded[2] == "Lesson 1"

    def test_load_respects_limit(self, lessons_dir):
        for i in range(20):
            save_lesson(f"Lesson {i}", category="general", lessons_dir=lessons_dir)
        loaded = load_lessons(category="general", limit=5, lessons_dir=lessons_dir)
        assert len(loaded) == 5

    def test_load_empty_category(self, lessons_dir):
        loaded = load_lessons(category="nonexistent", lessons_dir=lessons_dir)
        assert loaded == []

    def test_categories_are_isolated(self, lessons_dir):
        save_lesson("Bio lesson", category="biology", lessons_dir=lessons_dir)
        save_lesson("Chem lesson", category="chemistry", lessons_dir=lessons_dir)
        bio = load_lessons(category="biology", lessons_dir=lessons_dir)
        chem = load_lessons(category="chemistry", lessons_dir=lessons_dir)
        assert len(bio) == 1
        assert len(chem) == 1
        assert bio[0] == "Bio lesson"
        assert chem[0] == "Chem lesson"

    def test_deduplication(self, lessons_dir):
        save_lesson("Same lesson", category="general", lessons_dir=lessons_dir)
        save_lesson("Same lesson", category="general", lessons_dir=lessons_dir)
        save_lesson("Same lesson", category="general", lessons_dir=lessons_dir)
        loaded = load_lessons(category="general", lessons_dir=lessons_dir)
        assert len(loaded) == 1

    def test_whitespace_normalization_in_dedup(self, lessons_dir):
        save_lesson("  Same lesson  ", category="general", lessons_dir=lessons_dir)
        save_lesson("Same lesson", category="general", lessons_dir=lessons_dir)
        loaded = load_lessons(category="general", lessons_dir=lessons_dir)
        assert len(loaded) == 1

    def test_category_sanitization(self, lessons_dir):
        # Category names with special chars should be sanitized
        save_lesson("Test", category="student/alice", lessons_dir=lessons_dir)
        loaded = load_lessons(category="student/alice", lessons_dir=lessons_dir)
        assert len(loaded) == 1

    def test_jsonl_format(self, lessons_dir):
        save_lesson("Test lesson", category="general", lessons_dir=lessons_dir)
        path = lessons_dir / "general.jsonl"
        assert path.exists()
        with open(path) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["text"] == "Test lesson"
        assert entry["category"] == "general"
        assert "ts" in entry


class TestBuildLessonsPrompt:
    def test_empty_returns_empty_string(self, lessons_dir):
        result = build_lessons_prompt(category="empty", lessons_dir=lessons_dir)
        assert result == ""

    def test_prompt_format(self, lessons_dir):
        save_lesson("Always check B vs D", category="bio", lessons_dir=lessons_dir)
        save_lesson("Watch for rn vs m", category="bio", lessons_dir=lessons_dir)
        prompt = build_lessons_prompt(category="bio", lessons_dir=lessons_dir)
        assert "LESSONS FROM PREVIOUS GRADING SESSIONS" in prompt
        assert "Always check B vs D" in prompt
        assert "Watch for rn vs m" in prompt
        assert "1." in prompt  # Numbered list

    def test_prompt_is_injectable(self, lessons_dir):
        save_lesson("Test", category="general", lessons_dir=lessons_dir)
        prompt = build_lessons_prompt(category="general", lessons_dir=lessons_dir)
        # Should be safe to concatenate into a system prompt
        system = "You are a transcription engine.\n\n" + prompt
        assert "LESSONS" in system


class TestAutoExtractLessons:
    def test_extract_replacements(self, lessons_dir):
        model = "The mitocondria is the powerhouse"
        corrected = "The mitochondria is the powerhouse"
        count = auto_extract_lessons(model, corrected, category="bio", lessons_dir=lessons_dir)
        assert count == 1
        lessons = load_lessons(category="bio", lessons_dir=lessons_dir)
        assert any("mitocondria" in l and "mitochondria" in l for l in lessons)

    def test_extract_insertions(self, lessons_dir):
        model = "The cell has membrane"
        corrected = "The cell has a membrane"
        count = auto_extract_lessons(model, corrected, category="general", lessons_dir=lessons_dir)
        assert count >= 1

    def test_extract_deletions(self, lessons_dir):
        model = "The the cell has a membrane"
        corrected = "The cell has a membrane"
        count = auto_extract_lessons(model, corrected, category="general", lessons_dir=lessons_dir)
        assert count >= 1

    def test_identical_texts_no_lessons(self, lessons_dir):
        text = "The cell has a membrane"
        count = auto_extract_lessons(text, text, category="general", lessons_dir=lessons_dir)
        assert count == 0


class TestClearLessons:
    def test_clear(self, lessons_dir):
        save_lesson("Test", category="general", lessons_dir=lessons_dir)
        assert len(load_lessons(category="general", lessons_dir=lessons_dir)) == 1
        clear_lessons(category="general", lessons_dir=lessons_dir)
        assert len(load_lessons(category="general", lessons_dir=lessons_dir)) == 0

    def test_clear_nonexistent(self, lessons_dir):
        # Should not raise
        clear_lessons(category="nonexistent", lessons_dir=lessons_dir)
