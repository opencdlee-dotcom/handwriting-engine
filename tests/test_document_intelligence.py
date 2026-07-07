"""Tests for the document-intelligence layer (offline — mock providers)."""

import pytest

from handwriting_engine import document_intelligence as di
from handwriting_engine.document_intelligence import (
    DocumentAnalysis, Table, Figure, Region,
    analyze_document, analyze_pages, build_core_schema, _package_schema_for,
)
from handwriting_engine.models import ProviderError, ImageError
from handwriting_engine import providers as providers_pkg


# --- canned model output ----------------------------------------------------

SAMPLE_RAW = {
    "document_type": "lab_notebook",
    "summary": "A titration experiment page: a data table and a titration curve.",
    "full_text": "Trial 1: 12.4 mL\nTrial 2: 12.6 mL",
    "regions": [
        {"type": "heading", "content": "Titration of HCl"},
        {"type": "table", "content": "volume data"},
        {"type": "chart", "content": "titration curve"},
    ],
    "tables": [
        {
            "title": "Titrant volumes",
            "columns": ["Trial", "Volume (mL)"],
            "rows": [{"cells": ["1", "12.4"]}, {"cells": ["2", "12.6"]}],
            "notes": "mean 12.5 mL",
        }
    ],
    "figures": [
        {
            "kind": "chart",
            "title": "Titration curve",
            "description": "pH vs volume, sigmoidal.",
            "interpretation": "Equivalence point near 12.5 mL, consistent with the table.",
            "data_points": [{"label": "equivalence", "value": "12.5 mL"}],
            "relationships": ["volume -> pH (increasing)"],
        }
    ],
    "equations": [{"content": "M1V1 = M2V2", "meaning": "dilution/neutralization relation"}],
    "key_findings": ["Endpoint ~12.5 mL", "Table and curve agree"],
    "reading_confidence": "high",
    "uncertainties": ["Second decimal on trial 2 slightly smudged"],
}


class StructuredMockProvider:
    """Mock provider exposing read_structured for document-intelligence tests."""

    name = "mockstruct"

    def __init__(self, model=None, response=None, **kwargs):
        self._model = model
        self._response = SAMPLE_RAW if response is None else response
        self._usage = {"input_tokens": 100, "output_tokens": 50}
        self.last_schema = None

    def read_image(self, *a, **k):
        return self._response.get("full_text", "")

    def read_batch(self, *a, **k):
        return self._response.get("full_text", "")

    def read_structured(self, image_blocks, prompt, schema, system_prompt="", max_tokens=8192):
        self.last_schema = schema
        return self._response

    @property
    def usage(self):
        return dict(self._usage)

    @classmethod
    def is_available(cls):
        return True


class PlainMockProvider:
    """Mock provider WITHOUT read_structured (like local OCR)."""

    name = "mockplain"

    def __init__(self, **kwargs):
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def read_image(self, *a, **k):
        return "text"

    def read_batch(self, *a, **k):
        return "text"

    @property
    def usage(self):
        return dict(self._usage)

    @classmethod
    def is_available(cls):
        return True


@pytest.fixture(autouse=True)
def _register_mocks():
    providers_pkg.register("mockstruct", StructuredMockProvider)
    providers_pkg.register("mockplain", PlainMockProvider)
    # Drop cached singletons so each test gets a fresh mock instance.
    providers_pkg._INSTANCES.pop("mockstruct", None)
    providers_pkg._INSTANCES.pop("mockplain", None)
    yield
    providers_pkg._INSTANCES.pop("mockstruct", None)
    providers_pkg._INSTANCES.pop("mockplain", None)


# --- from_dict parsing ------------------------------------------------------

def test_from_dict_full():
    a = DocumentAnalysis.from_dict(SAMPLE_RAW, provider="claude", model="claude-fable-5")
    assert a.document_type == "lab_notebook"
    assert a.reading_confidence == "high"
    assert len(a.tables) == 1
    assert a.tables[0].columns == ["Trial", "Volume (mL)"]
    assert a.tables[0].rows == [["1", "12.4"], ["2", "12.6"]]
    assert len(a.figures) == 1
    assert a.figures[0].interpretation.startswith("Equivalence point")
    assert a.figures[0].data_points == [{"label": "equivalence", "value": "12.5 mL"}]
    assert a.key_findings == ["Endpoint ~12.5 mL", "Table and curve agree"]
    assert a.provider == "claude"
    assert a.model == "claude-fable-5"


def test_from_dict_empty_is_safe():
    a = DocumentAnalysis.from_dict({})
    assert a.document_type == ""
    assert a.tables == [] and a.figures == [] and a.regions == []
    assert a.reading_confidence == "medium"


def test_from_dict_tolerates_bare_row_lists():
    """Rows may arrive as bare lists instead of {cells: [...]} objects."""
    raw = {"tables": [{"columns": ["a"], "rows": [["x"], ["y"]]}]}
    a = DocumentAnalysis.from_dict(raw)
    assert a.tables[0].rows == [["x"], ["y"]]


def test_from_dict_skips_malformed_entries():
    raw = {"tables": ["not a dict", {"columns": ["a"], "rows": []}], "figures": [None]}
    a = DocumentAnalysis.from_dict(raw)
    assert len(a.tables) == 1
    assert a.figures == []


# --- rendering --------------------------------------------------------------

def test_to_markdown_contains_key_sections():
    a = DocumentAnalysis.from_dict(SAMPLE_RAW)
    md = a.to_markdown()
    assert "# Document Analysis — lab_notebook" in md
    assert "## Summary" in md
    assert "## Key findings" in md
    assert "| Trial | Volume (mL) |" in md
    assert "| 1 | 12.4 |" in md
    assert "**Interpretation:**" in md
    assert "## Full transcription" in md
    assert "## Uncertainties" in md


def test_to_markdown_ragged_table_padded():
    tbl = Table(columns=["a", "b", "c"], rows=[["1"], ["2", "3", "4"]])
    a = DocumentAnalysis(document_type="t", summary="s", full_text="f", tables=[tbl])
    md = a.to_markdown()
    # Header has 3 columns; short row is padded to 3 cells (4 pipes -> 3 gaps).
    assert "| a | b | c |" in md
    assert "| 1 |  |  |" in md


def test_to_dict_roundtrips_types():
    a = DocumentAnalysis.from_dict(SAMPLE_RAW)
    d = a.to_dict()
    assert isinstance(d["tables"], list)
    assert d["tables"][0]["title"] == "Titrant volumes"
    assert d["figures"][0]["kind"] == "chart"


# --- schema packaging -------------------------------------------------------

def test_core_schema_shape():
    s = build_core_schema()
    assert s["type"] == "object"
    for req in ("document_type", "summary", "full_text"):
        assert req in s["properties"]
        assert req in s["required"]
    # Region type enum is present and reasonable.
    region_type = s["properties"]["regions"]["items"]["properties"]["type"]
    assert "table" in region_type["enum"] and "chart" in region_type["enum"]


def test_package_schema_per_provider():
    core = build_core_schema()
    claude = _package_schema_for("claude", core)
    assert claude["input_schema"] is core and claude["name"] == "document_analysis"

    openai = _package_schema_for("openai", core)
    assert openai["schema"] is core and openai["strict"] is False

    gemini = _package_schema_for("gemini", core)
    assert gemini is core  # raw schema


# --- analyze_document dispatch ---------------------------------------------

def test_analyze_document_happy_path(tmp_image):
    img = tmp_image()
    a = analyze_document(img, provider="mockstruct", domain="chemistry")
    assert a.provider == "mockstruct"
    assert a.document_type == "lab_notebook"
    assert len(a.tables) == 1


def test_analyze_document_passes_raw_schema_to_generic_provider(tmp_image):
    img = tmp_image()
    analyze_document(img, provider="mockstruct")
    inst = providers_pkg.get_provider("mockstruct")
    # mockstruct isn't claude/openai, so it receives the raw (un-wrapped) schema.
    assert inst.last_schema["type"] == "object"


def test_analyze_document_model_override_forwarded(tmp_image):
    img = tmp_image()
    a = analyze_document(img, provider="mockstruct", model="explicit-model")
    assert a.model == "explicit-model"


def test_analyze_document_without_read_structured_raises(tmp_image):
    img = tmp_image()
    with pytest.raises(ProviderError, match="read_structured"):
        analyze_document(img, provider="mockplain")


def test_analyze_document_bad_image_raises():
    with pytest.raises(ImageError):
        analyze_document("/nonexistent/path/to/image.jpg", provider="mockstruct")


def test_analyze_document_empty_output_is_safe(tmp_image, monkeypatch):
    img = tmp_image()
    providers_pkg.register(
        "mockempty",
        type("E", (StructuredMockProvider,), {"read_structured": lambda self, *a, **k: {}}),
    )
    providers_pkg._INSTANCES.pop("mockempty", None)
    a = analyze_document(img, provider="mockempty")
    assert a.document_type == ""
    providers_pkg._INSTANCES.pop("mockempty", None)


# --- analyze_pages ----------------------------------------------------------

def test_analyze_pages_isolates_failures(tmp_image):
    good = tmp_image()
    results = analyze_pages([good, "/nope/missing.jpg"], provider="mockstruct")
    assert len(results) == 2
    assert results[0].document_type == "lab_notebook"
    assert results[1].document_type == "error"
    assert results[1].uncertainties  # records the failure
