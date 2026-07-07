"""
Document intelligence — structured, in-context interpretation of a page.

Where ``vision.read_page`` answers "what does this say?" and returns a flat
transcription, ``analyze_document`` answers "what IS this, and what does it
mean?" — it reads charts, tables, diagrams and handwriting together and
returns a structured :class:`DocumentAnalysis`: layout regions, tables as
data, figures interpreted in context, equations, and cross-content findings.

It is built on the providers' ``read_structured`` (tool-use / JSON-schema)
path, so the model fills a schema in a single pass rather than emitting prose
we then have to parse. The default model is the Fable-5 intelligence tier —
interpreting a figure in context is exactly what a deep-reasoning model is for
(unlike verbatim transcription, where reasoning hurts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

from handwriting_engine._constants import (
    CLAUDE_INTELLIGENCE_MODEL,
    GEMINI_QUALITY_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from handwriting_engine.models import ProviderError, ImageError
from handwriting_engine.providers import get_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

REGION_TYPES = [
    "heading", "paragraph", "handwriting", "table", "figure", "chart",
    "diagram", "equation", "form_field", "margin_note", "caption", "other",
]
FIGURE_KINDS = ["chart", "graph", "diagram", "drawing", "photo", "map", "other"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]


@dataclass
class Table:
    """A table lifted off the page as structured data (not flattened text)."""
    title: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    notes: str = ""


@dataclass
class Figure:
    """A chart/diagram/drawing, interpreted — not just described."""
    kind: str = "other"          # one of FIGURE_KINDS
    title: str = ""
    description: str = ""        # what is literally drawn
    interpretation: str = ""     # what it means in the document's context
    data_points: list[dict] = field(default_factory=list)  # [{label, value}]
    relationships: list[str] = field(default_factory=list)  # e.g. "A -> B (activates)"


@dataclass
class Region:
    """One laid-out region of the page (reading order preserved by list order)."""
    type: str = "other"          # one of REGION_TYPES
    content: str = ""            # transcription / label for this region
    note: str = ""


@dataclass
class DocumentAnalysis:
    """Structured, in-context interpretation of a single page image."""
    document_type: str = ""
    summary: str = ""            # context-integrated reading of the whole page
    full_text: str = ""          # verbatim transcription of all text + handwriting
    regions: list[Region] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    equations: list[dict] = field(default_factory=list)   # [{content, meaning}]
    key_findings: list[str] = field(default_factory=list)
    reading_confidence: str = "medium"
    uncertainties: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict, *, provider: str = "", model: str = "") -> "DocumentAnalysis":
        """Build from the model's raw structured output, tolerating omissions."""
        d = d or {}

        def _tables(raw) -> list[Table]:
            out = []
            for t in raw or []:
                if not isinstance(t, dict):
                    continue
                rows = []
                for r in t.get("rows") or []:
                    # Accept either ["a","b"] or {"cells": ["a","b"]}.
                    cells = r.get("cells") if isinstance(r, dict) else r
                    rows.append([str(c) for c in (cells or [])])
                out.append(Table(
                    title=str(t.get("title", "")),
                    columns=[str(c) for c in (t.get("columns") or [])],
                    rows=rows,
                    notes=str(t.get("notes", "")),
                ))
            return out

        def _figures(raw) -> list[Figure]:
            out = []
            for f in raw or []:
                if not isinstance(f, dict):
                    continue
                out.append(Figure(
                    kind=str(f.get("kind", "other")),
                    title=str(f.get("title", "")),
                    description=str(f.get("description", "")),
                    interpretation=str(f.get("interpretation", "")),
                    data_points=[dp for dp in (f.get("data_points") or []) if isinstance(dp, dict)],
                    relationships=[str(x) for x in (f.get("relationships") or [])],
                ))
            return out

        def _regions(raw) -> list[Region]:
            out = []
            for r in raw or []:
                if not isinstance(r, dict):
                    continue
                out.append(Region(
                    type=str(r.get("type", "other")),
                    content=str(r.get("content", "")),
                    note=str(r.get("note", "")),
                ))
            return out

        return cls(
            document_type=str(d.get("document_type", "")),
            summary=str(d.get("summary", "")),
            full_text=str(d.get("full_text", "")),
            regions=_regions(d.get("regions")),
            tables=_tables(d.get("tables")),
            figures=_figures(d.get("figures")),
            equations=[e for e in (d.get("equations") or []) if isinstance(e, dict)],
            key_findings=[str(x) for x in (d.get("key_findings") or [])],
            reading_confidence=str(d.get("reading_confidence", "medium")),
            uncertainties=[str(x) for x in (d.get("uncertainties") or [])],
            provider=provider,
            model=model,
        )

    # -- rendering ----------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """Human-readable report of the analysis."""
        L: list[str] = []
        title = self.document_type or "Document"
        L.append(f"# Document Analysis — {title}")
        if self.reading_confidence:
            L.append(f"_Reading confidence: {self.reading_confidence}_")

        if self.summary:
            L.append("\n## Summary\n" + self.summary)

        if self.key_findings:
            L.append("\n## Key findings")
            L.extend(f"- {k}" for k in self.key_findings)

        for tbl in self.tables:
            L.append(f"\n## Table{': ' + tbl.title if tbl.title else ''}")
            L.append(_render_table_md(tbl))
            if tbl.notes:
                L.append(f"\n_{tbl.notes}_")

        for fig in self.figures:
            head = f"\n## Figure — {fig.kind}"
            if fig.title:
                head += f": {fig.title}"
            L.append(head)
            if fig.description:
                L.append(fig.description)
            if fig.interpretation:
                L.append(f"\n**Interpretation:** {fig.interpretation}")
            if fig.data_points:
                L.append("\nData points:")
                for dp in fig.data_points:
                    label = dp.get("label", "")
                    value = dp.get("value", "")
                    L.append(f"- {label}: {value}")
            if fig.relationships:
                L.append("\nRelationships:")
                L.extend(f"- {r}" for r in fig.relationships)

        if self.equations:
            L.append("\n## Equations")
            for eq in self.equations:
                content = eq.get("content", "")
                meaning = eq.get("meaning", "")
                L.append(f"- `{content}`" + (f" — {meaning}" if meaning else ""))

        if self.full_text:
            L.append("\n## Full transcription\n" + self.full_text)

        if self.uncertainties:
            L.append("\n## Uncertainties")
            L.extend(f"- {u}" for u in self.uncertainties)

        return "\n".join(L).strip() + "\n"


def _render_table_md(tbl: Table) -> str:
    cols = tbl.columns or (["" for _ in tbl.rows[0]] if tbl.rows else [])
    width = max([len(cols)] + [len(r) for r in tbl.rows], default=0)
    if width == 0:
        return "_(empty table)_"
    cols = (cols + [""] * width)[:width]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in tbl.rows:
        cells = ([str(c) for c in r] + [""] * width)[:width]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema — one core schema, packaged per provider's structured-output API
# ---------------------------------------------------------------------------

_SCHEMA_NAME = "document_analysis"
_SCHEMA_DESC = (
    "Structured, in-context interpretation of a document page: layout regions, "
    "tables as data, figures interpreted (not just described), equations, and "
    "cross-content findings."
)


def build_core_schema() -> dict:
    """The provider-neutral JSON Schema for a DocumentAnalysis.

    Kept to the common subset all three providers accept (object/array/string/
    number/boolean + string enums) — no ``$ref``/``oneOf``/``additionalProperties``.
    """
    s = lambda **kw: {"type": "string", **kw}  # noqa: E731
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "document_type": s(description="e.g. lab_notebook, exam, worksheet, form, article, receipt, mixed"),
            "summary": s(description="What this page is and what it conveys, reasoning across text and visuals."),
            "full_text": s(description="Verbatim transcription of ALL text and handwriting; mark unsure chars with [?]."),
            "regions": {
                "type": "array",
                "description": "Laid-out regions in natural reading order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": s(enum=REGION_TYPES),
                        "content": s(description="Transcription or label for this region."),
                        "note": s(description="Optional note, e.g. why a region was flagged."),
                    },
                    "required": ["type", "content"],
                },
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": s(),
                        "columns": str_array,
                        "rows": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"cells": str_array}, "required": ["cells"]},
                        },
                        "notes": s(),
                    },
                    "required": ["columns", "rows"],
                },
            },
            "figures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": s(enum=FIGURE_KINDS),
                        "title": s(),
                        "description": s(description="What is literally drawn/plotted."),
                        "interpretation": s(description="What it means in this document's context."),
                        "data_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"label": s(), "value": s()},
                                "required": ["label", "value"],
                            },
                        },
                        "relationships": str_array,
                    },
                    "required": ["kind", "description"],
                },
            },
            "equations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"content": s(), "meaning": s()},
                    "required": ["content"],
                },
            },
            "key_findings": str_array,
            "reading_confidence": s(enum=CONFIDENCE_LEVELS),
            "uncertainties": str_array,
        },
        "required": ["document_type", "summary", "full_text"],
    }


def _package_schema_for(provider: str, core: dict) -> dict:
    """Wrap the core schema in the container each provider's read_structured wants."""
    if provider == "claude":
        return {"name": _SCHEMA_NAME, "description": _SCHEMA_DESC, "input_schema": core}
    if provider == "openai":
        # Non-strict: strict mode would demand additionalProperties:false + every
        # field required, which fights the "omit what isn't present" design.
        return {"name": _SCHEMA_NAME, "schema": core, "strict": False}
    # gemini and anything else: the raw schema object.
    return core


_DEFAULT_MODEL_BY_PROVIDER = {
    "claude": CLAUDE_INTELLIGENCE_MODEL,
    "gemini": GEMINI_QUALITY_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
}


def _default_prompt(domain: str, extra_instructions: str) -> str:
    domain_line = f"Domain context: {domain}.\n" if domain and domain != "general" else ""
    extra = f"\n{extra_instructions}" if extra_instructions else ""
    return (
        "Analyze this document page and fill the schema.\n"
        f"{domain_line}"
        "Do NOT merely describe what you see — interpret it in context:\n"
        "- Transcribe every piece of text and handwriting verbatim into full_text; "
        "mark characters you are less than ~80% sure of with [?].\n"
        "- Extract tables as structured rows/columns, preserving every value.\n"
        "- For each chart/graph/diagram/drawing, give both what is drawn (description) "
        "and what it means here (interpretation), plus any readable data points and relationships.\n"
        "- Capture equations with their meaning.\n"
        "- In key_findings, state what the page actually establishes, reasoning across "
        "text and visuals together.\n"
        "- List anything you could not read or were unsure about in uncertainties."
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def analyze_document(
    image_path: str,
    *,
    provider: str = "claude",
    model: str | None = None,
    domain: str = "general",
    system_prompt: str = "",
    prompt: str = "",
    extra_instructions: str = "",
    max_tokens: int = 8192,
) -> DocumentAnalysis:
    """Interpret a single page image as structured document intelligence.

    Args:
        image_path: Path to a page image.
        provider: Vision provider with a ``read_structured`` method
            (``claude`` recommended — most permissive schema support, and the
            Fable-5 home; ``gemini`` and ``openai`` also supported).
        model: Model override. Defaults to the provider's intelligence tier
            (Claude → Fable 5, Gemini → 2.5 Pro).
        domain: Domain hint folded into the prompt (e.g. "biology").
        prompt: Full prompt override. When empty, a document-intelligence
            prompt is used and ``extra_instructions`` is appended to it.
        extra_instructions: Extra guidance appended to the default prompt.

    Returns:
        A :class:`DocumentAnalysis`.

    Raises:
        ProviderError: if the provider has no ``read_structured`` capability
            (e.g. local OCR providers) or the structured call fails.
        ImageError: if the image cannot be prepared.
    """
    from handwriting_engine.optimize import validate_and_prepare_image

    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    p = get_provider(provider, model=resolved_model) if resolved_model else get_provider(provider)

    read_structured = getattr(p, "read_structured", None)
    if not callable(read_structured):
        raise ProviderError(
            f"Provider '{provider}' does not support structured document analysis "
            "(no read_structured). Use claude, gemini, or openai."
        )

    prepared = validate_and_prepare_image(image_path)
    if prepared is None:
        raise ImageError(f"Could not prepare image for analysis: {image_path}")
    b64_data, media_type = prepared
    image_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
    }

    used_prompt = prompt or _default_prompt(domain, extra_instructions)
    packaged = _package_schema_for(provider, build_core_schema())

    try:
        raw = read_structured([image_block], used_prompt, packaged, system_prompt, max_tokens)
    except Exception as e:  # noqa: BLE001 — surface as ProviderError with context
        raise ProviderError(f"Structured document analysis failed ({provider}): {e}") from e

    if not raw:
        logger.warning("Document analysis returned empty structured output (%s)", provider)
    return DocumentAnalysis.from_dict(raw or {}, provider=provider, model=resolved_model or "")


def analyze_pages(
    image_paths: list[str],
    **kwargs,
) -> list[DocumentAnalysis]:
    """Analyze several page images. Failures are surfaced per page, not fatal.

    Returns one :class:`DocumentAnalysis` per input path; a page that fails
    yields an analysis whose ``uncertainties`` records the error, so a bad page
    never drops a good one from the batch.
    """
    results: list[DocumentAnalysis] = []
    for path in image_paths:
        try:
            results.append(analyze_document(path, **kwargs))
        except (ProviderError, ImageError) as e:
            logger.error("Analysis failed for %s: %s", path, e)
            results.append(DocumentAnalysis(
                document_type="error",
                summary=f"Analysis failed: {e}",
                uncertainties=[str(e)],
            ))
    return results
