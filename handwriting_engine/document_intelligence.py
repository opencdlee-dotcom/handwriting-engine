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
(unlike verbatim transcription, where reasoning hurts) — and on Claude the
call runs with extended thinking enabled (INTELLIGENCE_THINKING_BUDGET) so the
model reasons across the page before committing to the schema.

:func:`analyze_document_set` extends this to multi-page documents in one
call: all pages go up together, and the model connects content across pages
(a table on page 1 feeding a chart on page 3) instead of reading each page
in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

from handwriting_engine._constants import (
    CLAUDE_INTELLIGENCE_MODEL,
    GEMINI_QUALITY_MODEL,
    DEFAULT_OPENAI_MODEL,
    INTELLIGENCE_THINKING_BUDGET,
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
    page: int = 0                # 1-based page (multi-page analysis); 0 = unset


@dataclass
class Figure:
    """A chart/diagram/drawing, interpreted — not just described."""
    kind: str = "other"          # one of FIGURE_KINDS
    title: str = ""
    description: str = ""        # what is literally drawn
    interpretation: str = ""     # what it means in the document's context
    axes: dict = field(default_factory=dict)  # {"x": "volume (mL)", "y": "pH"}
    data_points: list[dict] = field(default_factory=list)  # [{label, value}]
    relationships: list[str] = field(default_factory=list)  # e.g. "A -> B (activates)"
    page: int = 0                # 1-based page (multi-page analysis); 0 = unset


@dataclass
class Region:
    """One laid-out region of the page (reading order preserved by list order)."""
    type: str = "other"          # one of REGION_TYPES
    content: str = ""            # transcription / label for this region
    note: str = ""
    page: int = 0                # 1-based page (multi-page analysis); 0 = unset


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

        def _page(v) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

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
                    page=_page(t.get("page")),
                ))
            return out

        def _figures(raw) -> list[Figure]:
            out = []
            for f in raw or []:
                if not isinstance(f, dict):
                    continue
                axes = f.get("axes")
                out.append(Figure(
                    kind=str(f.get("kind", "other")),
                    title=str(f.get("title", "")),
                    description=str(f.get("description", "")),
                    interpretation=str(f.get("interpretation", "")),
                    axes={str(k): str(v) for k, v in axes.items()} if isinstance(axes, dict) else {},
                    data_points=[dp for dp in (f.get("data_points") or []) if isinstance(dp, dict)],
                    relationships=[str(x) for x in (f.get("relationships") or [])],
                    page=_page(f.get("page")),
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
                    page=_page(r.get("page")),
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
            head = f"\n## Table{': ' + tbl.title if tbl.title else ''}"
            if tbl.page:
                head += f" (p. {tbl.page})"
            L.append(head)
            L.append(_render_table_md(tbl))
            if tbl.notes:
                L.append(f"\n_{tbl.notes}_")

        for fig in self.figures:
            head = f"\n## Figure — {fig.kind}"
            if fig.title:
                head += f": {fig.title}"
            if fig.page:
                head += f" (p. {fig.page})"
            L.append(head)
            if fig.description:
                L.append(fig.description)
            if fig.axes:
                L.append("Axes: " + ", ".join(f"{k} = {v}" for k, v in fig.axes.items()))
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
    page = {"type": "integer", "description": "1-based page this item appears on (multi-page analysis only)."}
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
                        "page": page,
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
                        "page": page,
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
                        "axes": {
                            "type": "object",
                            "description": "Axis labels with units for charts/graphs, e.g. {\"x\": \"volume (mL)\", \"y\": \"pH\"}.",
                            "properties": {"x": s(), "y": s()},
                        },
                        "data_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"label": s(), "value": s()},
                                "required": ["label", "value"],
                            },
                        },
                        "relationships": str_array,
                        "page": page,
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


_ANALYSIS_RULES = (
    "Do NOT merely describe what you see — interpret it in context:\n"
    "- Transcribe every piece of text and handwriting verbatim into full_text; "
    "mark characters you are less than ~80% sure of with [?].\n"
    "- Extract tables as structured rows/columns, preserving every value.\n"
    "- For each chart/graph/diagram/drawing, give both what is drawn (description) "
    "and what it means here (interpretation). Read axis labels, units and scales "
    "into axes, lift every legible data point, and name trends and relationships.\n"
    "- Connect visuals to the text that references them ('the table above', 'Fig. 2') — "
    "a figure's interpretation should use what the surrounding text says it is for.\n"
    "- Capture equations with their meaning.\n"
    "- In key_findings, state what the document actually establishes, reasoning across "
    "text and visuals together — including contradictions (a value in the text that "
    "disagrees with the table, an implausible unit, a curve that doesn't match its caption).\n"
    "- List anything you could not read or were unsure about in uncertainties."
)


def _default_prompt(domain: str, extra_instructions: str) -> str:
    domain_line = f"Domain context: {domain}.\n" if domain and domain != "general" else ""
    extra = f"\n{extra_instructions}" if extra_instructions else ""
    return (
        "Analyze this document page and fill the schema.\n"
        f"{domain_line}"
        f"{_ANALYSIS_RULES}"
        f"{extra}"
    )


def _doc_set_prompt(n_pages: int, domain: str, extra_instructions: str) -> str:
    domain_line = f"Domain context: {domain}.\n" if domain and domain != "general" else ""
    extra = f"\n{extra_instructions}" if extra_instructions else ""
    return (
        f"The {n_pages} images above are consecutive pages of ONE document. "
        "Analyze the document as a whole and fill the schema.\n"
        f"{domain_line}"
        f"{_ANALYSIS_RULES}\n"
        "- Reason ACROSS pages: connect a table on one page to the chart or text that "
        "uses it on another, and track anything that continues over a page break.\n"
        "- Tag every region, table and figure with the 1-based page it appears on.\n"
        "- In full_text, transcribe all pages in order, separated by '--- Page N ---' lines."
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

MAX_DOC_SET_PAGES = 20  # one-call multi-page cap: request-size limits + reading fidelity


def _resolve_structured_provider(provider: str, model: str | None):
    """Provider instance + its read_structured + the resolved model name."""
    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    p = get_provider(provider, model=resolved_model) if resolved_model else get_provider(provider)

    read_structured = getattr(p, "read_structured", None)
    if not callable(read_structured):
        raise ProviderError(
            f"Provider '{provider}' does not support structured document analysis "
            "(no read_structured). Use claude, gemini, or openai."
        )
    return read_structured, resolved_model


def _prepare_image_block(image_path: str) -> dict:
    from handwriting_engine.optimize import validate_and_prepare_image

    prepared = validate_and_prepare_image(image_path)
    if prepared is None:
        raise ImageError(f"Could not prepare image for analysis: {image_path}")
    b64_data, media_type = prepared
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
    }


def _call_structured(
    provider: str,
    read_structured,
    blocks: list[dict],
    used_prompt: str,
    system_prompt: str,
    max_tokens: int,
    thinking_budget: int | None,
) -> dict:
    packaged = _package_schema_for(provider, build_core_schema())

    # Only Claude exposes a thinking knob on read_structured; Gemini 2.5 Pro
    # thinks by default and OpenAI's pinned model has no reasoning mode.
    kwargs = {}
    if provider == "claude":
        budget = INTELLIGENCE_THINKING_BUDGET if thinking_budget is None else thinking_budget
        if budget > 0:
            kwargs["thinking_budget"] = budget

    try:
        raw = read_structured(blocks, used_prompt, packaged, system_prompt, max_tokens, **kwargs)
    except Exception as e:  # noqa: BLE001 — surface as ProviderError with context
        raise ProviderError(f"Structured document analysis failed ({provider}): {e}") from e

    if not raw:
        logger.warning("Document analysis returned empty structured output (%s)", provider)
    return raw


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
    thinking_budget: int | None = None,
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
        thinking_budget: Extended-thinking tokens before the schema fill
            (Claude only). ``None`` uses INTELLIGENCE_THINKING_BUDGET; 0
            disables thinking (the old forced-tool behavior).

    Returns:
        A :class:`DocumentAnalysis`.

    Raises:
        ProviderError: if the provider has no ``read_structured`` capability
            (e.g. local OCR providers) or the structured call fails.
        ImageError: if the image cannot be prepared.
    """
    read_structured, resolved_model = _resolve_structured_provider(provider, model)
    image_block = _prepare_image_block(image_path)
    used_prompt = prompt or _default_prompt(domain, extra_instructions)

    raw = _call_structured(
        provider, read_structured, [image_block], used_prompt,
        system_prompt, max_tokens, thinking_budget,
    )
    return DocumentAnalysis.from_dict(raw or {}, provider=provider, model=resolved_model or "")


def analyze_document_set(
    image_paths: list[str],
    *,
    provider: str = "claude",
    model: str | None = None,
    domain: str = "general",
    system_prompt: str = "",
    prompt: str = "",
    extra_instructions: str = "",
    max_tokens: int = 16384,
    thinking_budget: int | None = None,
) -> DocumentAnalysis:
    """Interpret a multi-page document in ONE model call, reasoning across pages.

    Where :func:`analyze_pages` analyzes each page in isolation, this sends
    every page in a single request (interleaved with "Page N" labels) so the
    model can connect a table on one page to the chart that plots it three
    pages later. Regions/tables/figures come back tagged with their 1-based
    ``page``, and ``full_text`` carries ``--- Page N ---`` separators.

    Accepts the same knobs as :func:`analyze_document`. Capped at
    ``MAX_DOC_SET_PAGES`` pages per call — split longer documents, or use
    :func:`analyze_pages` for independent per-page analyses.

    Raises:
        ValueError: on an empty list or more than ``MAX_DOC_SET_PAGES`` pages.
        ProviderError / ImageError: as :func:`analyze_document`.
    """
    if not image_paths:
        raise ValueError("analyze_document_set needs at least one image path")
    if len(image_paths) > MAX_DOC_SET_PAGES:
        raise ValueError(
            f"analyze_document_set is capped at {MAX_DOC_SET_PAGES} pages per call "
            f"(got {len(image_paths)}). Split the document, or use analyze_pages "
            "for independent per-page analyses."
        )

    read_structured, resolved_model = _resolve_structured_provider(provider, model)

    blocks: list[dict] = []
    for i, path in enumerate(image_paths, start=1):
        blocks.append({"type": "text", "text": f"Page {i} of {len(image_paths)}:"})
        blocks.append(_prepare_image_block(path))

    used_prompt = prompt or _doc_set_prompt(len(image_paths), domain, extra_instructions)

    raw = _call_structured(
        provider, read_structured, blocks, used_prompt,
        system_prompt, max_tokens, thinking_budget,
    )
    return DocumentAnalysis.from_dict(raw or {}, provider=provider, model=resolved_model or "")


# ---------------------------------------------------------------------------
# Grounded question-answering — targeted read instead of a full analysis
# ---------------------------------------------------------------------------

@dataclass
class QuestionAnswer:
    """A document-grounded answer to a specific question.

    Where :func:`analyze_document` dumps the whole page as structured data,
    this answers one question and returns only the answer plus the evidence
    on the page that supports it — far fewer output tokens when the caller
    already knows what they want to know.
    """
    question: str = ""
    answer: str = ""
    answerable: bool = True          # False when the page can't support an answer
    supported_by: list[str] = field(default_factory=list)  # quotes / figure / table refs
    confidence: str = "medium"       # one of CONFIDENCE_LEVELS
    uncertainties: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    @classmethod
    def from_dict(cls, d: dict, *, question: str = "", provider: str = "", model: str = "") -> "QuestionAnswer":
        d = d or {}
        return cls(
            question=question,
            answer=str(d.get("answer", "")),
            answerable=bool(d.get("answerable", True)),
            supported_by=[str(x) for x in (d.get("supported_by") or [])],
            confidence=str(d.get("confidence", "medium")),
            uncertainties=[str(x) for x in (d.get("uncertainties") or [])],
            provider=provider,
            model=model,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        L: list[str] = []
        if self.question:
            L.append(f"**Q:** {self.question}")
        if not self.answerable:
            L.append(f"\n**Not answerable from the document.** {self.answer}".rstrip())
        else:
            L.append(f"\n**A:** {self.answer}")
        if self.confidence:
            L.append(f"_Confidence: {self.confidence}_")
        if self.supported_by:
            L.append("\nSupported by:")
            L.extend(f"- {s}" for s in self.supported_by)
        if self.uncertainties:
            L.append("\nUncertainties:")
            L.extend(f"- {u}" for u in self.uncertainties)
        return "\n".join(L).strip() + "\n"


_QA_SCHEMA_NAME = "document_answer"
_QA_SCHEMA_DESC = "A document-grounded answer to a specific question, with the on-page evidence for it."


def build_qa_schema() -> dict:
    """Provider-neutral JSON Schema for a :class:`QuestionAnswer`."""
    s = lambda **kw: {"type": "string", **kw}  # noqa: E731
    return {
        "type": "object",
        "properties": {
            "answer": s(description="The direct answer, grounded in the document. If not answerable, say what's missing."),
            "answerable": {"type": "boolean", "description": "False if the document lacks the information to answer."},
            "supported_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact quotes, figure titles, or table cells from the document that support the answer.",
            },
            "confidence": s(enum=CONFIDENCE_LEVELS),
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "answerable"],
    }


def _package_qa_schema_for(provider: str, core: dict) -> dict:
    if provider == "claude":
        return {"name": _QA_SCHEMA_NAME, "description": _QA_SCHEMA_DESC, "input_schema": core}
    if provider == "openai":
        return {"name": _QA_SCHEMA_NAME, "schema": core, "strict": False}
    return core


def _qa_prompt(question: str, n_pages: int, domain: str, extra_instructions: str) -> str:
    domain_line = f"Domain context: {domain}.\n" if domain and domain != "general" else ""
    scope = (
        f"The {n_pages} images above are consecutive pages of ONE document. "
        if n_pages > 1 else ""
    )
    extra = f"\n{extra_instructions}" if extra_instructions else ""
    return (
        f"{scope}Answer this question about the document using ONLY what is shown:\n"
        f"Q: {question}\n"
        f"{domain_line}"
        "- Interpret charts, tables, diagrams and handwriting in context — do not stop at "
        "the literal text.\n"
        "- Ground your answer in the document: put the exact quotes, figure titles or table "
        "cells that support it in supported_by.\n"
        "- If the document does not contain enough to answer, set answerable=false and say "
        "what is missing in answer. Do not guess from outside knowledge."
        f"{extra}"
    )


def ask_document(
    image_paths: "str | list[str]",
    question: str,
    *,
    provider: str = "claude",
    model: str | None = None,
    domain: str = "general",
    system_prompt: str = "",
    prompt: str = "",
    extra_instructions: str = "",
    max_tokens: int = 4096,
    thinking_budget: int | None = None,
) -> QuestionAnswer:
    """Answer a specific question about a document, grounded in what's on the page(s).

    A targeted alternative to :func:`analyze_document`: instead of returning the
    whole page as structured data, it returns just the answer and the on-page
    evidence for it — the deep-reasoning read (charts/tables/handwriting in
    context) without the full-analysis output-token cost. Pass one path or a
    list of pages (sent together, so the answer can span pages).

    Accepts the same knobs as :func:`analyze_document`. Raises ``ValueError`` on
    an empty question or page list; ``ProviderError``/``ImageError`` as
    :func:`analyze_document`.
    """
    if not question or not question.strip():
        raise ValueError("ask_document needs a non-empty question")
    paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    if not paths:
        raise ValueError("ask_document needs at least one image path")
    if len(paths) > MAX_DOC_SET_PAGES:
        raise ValueError(
            f"ask_document is capped at {MAX_DOC_SET_PAGES} pages per call (got {len(paths)})."
        )

    resolved_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    p = get_provider(provider, model=resolved_model) if resolved_model else get_provider(provider)
    read_structured = getattr(p, "read_structured", None)
    if not callable(read_structured):
        raise ProviderError(
            f"Provider '{provider}' does not support grounded Q&A (no read_structured). "
            "Use claude, gemini, or openai."
        )

    blocks: list[dict] = []
    if len(paths) == 1:
        blocks.append(_prepare_image_block(paths[0]))
    else:
        for i, path in enumerate(paths, start=1):
            blocks.append({"type": "text", "text": f"Page {i} of {len(paths)}:"})
            blocks.append(_prepare_image_block(path))

    used_prompt = prompt or _qa_prompt(question, len(paths), domain, extra_instructions)
    packaged = _package_qa_schema_for(provider, build_qa_schema())

    kwargs = {}
    if provider == "claude":
        budget = INTELLIGENCE_THINKING_BUDGET if thinking_budget is None else thinking_budget
        if budget > 0:
            kwargs["thinking_budget"] = budget

    try:
        raw = read_structured(blocks, used_prompt, packaged, system_prompt, max_tokens, **kwargs)
    except Exception as e:  # noqa: BLE001 — surface as ProviderError with context
        raise ProviderError(f"Grounded Q&A failed ({provider}): {e}") from e

    if not raw:
        logger.warning("Grounded Q&A returned empty structured output (%s)", provider)
    return QuestionAnswer.from_dict(raw or {}, question=question, provider=provider, model=resolved_model or "")


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
