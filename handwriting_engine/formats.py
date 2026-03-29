"""
Output format templates for different content types.

Each template tells the vision model how to structure its transcription
output.  Callers pick a template by content type and (optionally) an
output format variant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "lab_notebook": """
OUTPUT FORMAT: Markdown table

For each page, produce a structured markdown table:

| Field            | Content                        |
|------------------|--------------------------------|
| Page Number      | [page #]                       |
| Date             | [date if visible]              |
| Title/Objective  | [section title or objective]   |
| Procedure Notes  | [step-by-step notes]           |
| Data/Tables      | [reproduce tables in markdown] |
| Calculations     | [show work and results]        |
| Observations     | [qualitative notes]            |
| Diagrams         | [diagram: description]         |
| Conclusions      | [summary statements]           |
| Margin Notes     | [any margin annotations]       |

Preserve all original data values exactly as written.
Use [?] markers for ambiguous readings.
Use [illegible] for unreadable sections.
""".strip(),

    "student_work": """
OUTPUT FORMAT: Problem/Answer

For each problem or question on the page:

**Problem [#]:**
- Question (if visible): [transcribe question text]
- Student Answer: [transcribe answer exactly as written]
- Work Shown: [transcribe any intermediate steps]
- Diagrams: [diagram: description] if present

Number problems sequentially as they appear on the page.
If problem numbers are visible, use those instead.
Preserve original formatting (lists, equations, diagrams).
""".strip(),

    "form": """
OUTPUT FORMAT: JSON (field/value pairs)

Return a JSON object where each key is a form field label
and each value is what the student/person wrote:

```json
{
  "field_label_1": "handwritten value",
  "field_label_2": "handwritten value",
  ...
}
```

Rules:
- Use the printed field label as the key (clean, no trailing colon)
- Use the handwritten response as the value
- For checkboxes: true/false
- For empty fields: null
- For illegible fields: "[illegible]"
- For ambiguous readings: include [?] marker in value
""".strip(),

    "general": """
OUTPUT FORMAT: Freeform Markdown Notes

Transcribe the handwritten content preserving its natural structure:

- Use headers (##) for section titles or large headings
- Use bullet points for lists
- Use > blockquotes for margin notes or asides
- Use **bold** for underlined or emphasized text
- Use `code` for formulas, equations, or technical notation
- Preserve paragraph breaks as they appear
- Note non-text elements: [diagram: description], [graph: description]
- Mark ambiguous readings with [?]
- Mark illegible text with [illegible: ~N characters]
""".strip(),

    "plain": """
OUTPUT FORMAT: Plain Text

Transcribe all handwritten text as plain unformatted text.

Rules:
- No markdown formatting
- No special markers except [?] for ambiguous and [illegible] for unreadable
- Preserve line breaks as they appear on the page
- Preserve original spelling (do not correct)
- Separate distinct sections with a blank line
- Output ONLY the transcribed text, no commentary
""".strip(),

    "exam_answers": """
OUTPUT FORMAT: Exam Answers (Q&A)

Transcribe each answer keyed to its question number:

Q1: [student's answer to question 1]
Q2: [student's answer to question 2]
Q3: [student's answer to question 3]
...

Rules:
- Use the question numbers printed on the exam
- If sub-parts exist: Q1a:, Q1b:, Q1c:, etc.
- For multiple choice: just the letter (e.g., Q1: B)
- For matching: Q1: C, Q2: A, Q3: D, etc.
- For short answer: transcribe the full response
- For show-your-work: include intermediate steps
- Mark ambiguous single-letter answers: Q5: B [?alt: D]
- For blank/unanswered: Q7: [no answer]
""".strip(),
}

# Aliases for convenience
TEMPLATES["notebook"] = TEMPLATES["lab_notebook"]
TEMPLATES["answers"] = TEMPLATES["exam_answers"]
TEMPLATES["text"] = TEMPLATES["plain"]
TEMPLATES["json"] = TEMPLATES["form"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_format_template(content_type: str, output_format: str = "markdown") -> str:
    """Get the appropriate output format template.

    Args:
        content_type: One of the keys in TEMPLATES (e.g. 'lab_notebook',
            'student_work', 'form', 'general', 'plain', 'exam_answers').
            Also accepts aliases: 'notebook', 'answers', 'text', 'json'.
        output_format: Output format hint ('markdown', 'json', 'plain').
            If content_type is not found, this is used as a fallback key.

    Returns:
        The template string, or the 'general' template if nothing matches.
    """
    # Direct match
    if content_type in TEMPLATES:
        return TEMPLATES[content_type]

    # Try output_format as a key
    if output_format in TEMPLATES:
        return TEMPLATES[output_format]

    # Map output_format hints to known templates
    format_map = {
        "markdown": "general",
        "json": "form",
        "plain": "plain",
        "table": "lab_notebook",
        "qa": "exam_answers",
    }
    fallback_key = format_map.get(output_format, "general")
    return TEMPLATES[fallback_key]


def list_content_types() -> list[str]:
    """Return all available content type keys (excluding aliases)."""
    primary = ["lab_notebook", "student_work", "form", "general", "plain", "exam_answers"]
    return primary
