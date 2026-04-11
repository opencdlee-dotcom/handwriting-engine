"""
Provider-specific prompt adaptation for handwriting reading strategies.

Research shows each LLM provider responds differently to the same prompt:
- Gemini: Concise instructions, top confusion pairs only (20-50 domain terms sweet spot)
- OpenAI: Role+task+format pattern, few-shot examples over rule tables
- Claude: Full detail, optimized for prompt caching (stable prefix)
"""

from __future__ import annotations

# Top 15 disambiguation pairs by frequency in handwriting OCR errors
_TOP_DISAMBIGUATION_PAIRS = [
    ("1 / l / I", "In numbers -> 1, in words -> l or I"),
    ("0 / O", "In numbers -> 0, in words -> O"),
    ("rn / m", "Check: does 'rn' or 'm' make a valid word?"),
    ("5 / S", "In numbers -> 5, in words -> S"),
    ("u / v", "Rounded bottom = u, pointed = v"),
    ("a / o", "Has tail = a, fully closed = o"),
    ("cl / d", "Check: does 'cl' or 'd' make a valid word?"),
    ("n / h", "Tall ascender = h, short = n"),
    ("e / c", "Loop closed = e, open = c"),
    ("9 / q", "In numbers -> 9, in words -> q"),
    ("7 / 1", "7 has horizontal top stroke, 1 is vertical"),
    ("B / D", "B has TWO loops, D has ONE smooth curve"),
    ("t / +", "In text -> t, in math -> +"),
    ("3 / 8", "3 is open on left, 8 is fully closed"),
    ("H / K", "H has parallel verticals, K has diagonals"),
]


def adapt_system_prompt(system_prompt: str, provider: str) -> str:
    """Adapt a system prompt for a specific provider.

    Args:
        system_prompt: The full system prompt from get_reading_strategies() + lessons + calibration
        provider: Provider name ('gemini', 'openai', 'claude')

    Returns:
        Adapted system prompt optimized for the target provider
    """
    if provider == "gemini":
        return _adapt_for_gemini(system_prompt)
    elif provider == "openai":
        return _adapt_for_openai(system_prompt)
    # Claude gets full prompt — it handles verbose instructions well
    return system_prompt


def adapt_user_prompt(prompt: str, provider: str) -> str:
    """Adapt the user-facing prompt for a specific provider.

    GPT-4.1 follows instructions more literally and doesn't need verbose role preambles.
    Others use the prompt as-is.
    """
    if provider == "openai":
        return (
            "Transcribe all handwritten text from this image exactly as written. "
            "Preserve original spelling, line breaks, and formatting. "
            "Output ONLY the transcribed text.\n\n"
            + prompt
        )
    return prompt


def _adapt_for_gemini(system_prompt: str) -> str:
    """Gemini: Replace the full 44-pair disambiguation table with top 15 pairs.

    Research: Gemini benefits from concise instructions. The full table
    dilutes attention. Domain term sweet spot is 20-50 terms.
    """
    # Find and replace the CHARACTER_DISAMBIGUATION section with a compact version
    compact_disambiguation = _build_compact_disambiguation()

    # Replace the full table if present
    if "=== CHARACTER DISAMBIGUATION ===" in system_prompt:
        # Find the section boundaries
        start = system_prompt.index("=== CHARACTER DISAMBIGUATION ===")
        # Find the next section (starts with ===) or end
        rest = system_prompt[start + len("=== CHARACTER DISAMBIGUATION ==="):]
        next_section = rest.find("\n===")
        if next_section != -1:
            # Keep everything before disambiguation + compact version + everything after
            before = system_prompt[:start]
            after = rest[next_section:]
            return before + compact_disambiguation + after
        else:
            # Disambiguation is the last section
            before = system_prompt[:start]
            return before + compact_disambiguation

    return system_prompt


def _adapt_for_openai(system_prompt: str) -> str:
    """OpenAI: Restructure rules into concise format with examples.

    Research: GPT-4.1 performs best with token-efficient prompting —
    shorter, direct instructions with few-shot examples rather than exhaustive tables.
    """
    # Replace full disambiguation table with compact version (same as Gemini)
    # but also trim the multi-pass instructions to be more direct
    result = _adapt_for_gemini(system_prompt)  # Start with compact disambiguation

    # Trim the multi-pass reading instructions to essentials
    if "=== HANDWRITING READING PROTOCOL (MULTI-PASS) ===" in result:
        compact_multipass = (
            "=== READING PROTOCOL ===\n"
            "1. Map layout first: tables, headers, diagrams, margin notes\n"
            "2. Read text carefully, digit by digit for numbers\n"
            "3. Flag ambiguous characters with [?], illegible regions with [illegible: ~N chars]\n"
            "4. Cross-validate [?] marks using context and disambiguation rules\n"
            "5. Preserve original spelling — do NOT correct errors\n\n"
            "=== DISAMBIGUATION EXAMPLES ===\n"
            "These show how context resolves ambiguous characters:\n"
            '  "The OD was 0.45" → \'0\' not \'O\' (number follows number pattern)\n'
            '  "cells in meiosis" → \'m\' not \'rn\' (\'rneiosis\' is not a word)\n'
            '  "measured 1.5 mL" → \'1\' not \'l\' (precedes decimal point)\n'
            '  "add 5 uL" → \'u\' not \'v\' (\'vL\' is not a valid unit)\n'
            '  "plate 3B" → \'B\' not \'D\' (plate labels use sequential letters)'
        )
        start = result.index("=== HANDWRITING READING PROTOCOL (MULTI-PASS) ===")
        rest = result[start + len("=== HANDWRITING READING PROTOCOL (MULTI-PASS) ==="):]
        next_section = rest.find("\n===")
        if next_section != -1:
            before = result[:start]
            after = rest[next_section:]
            result = before + compact_multipass + after
        else:
            before = result[:start]
            result = before + compact_multipass

    # GPT-4.1 prioritizes instructions closer to the END of the prompt.
    # Move the critical transcription fidelity rule to the end if present.
    fidelity_marker = "=== CRITICAL TRANSCRIPTION RULE"
    if fidelity_marker in result:
        fidelity_start = result.index(fidelity_marker)
        fidelity_rest = result[fidelity_start:]
        next_section = fidelity_rest.find("\n===", len(fidelity_marker))
        if next_section != -1:
            fidelity_block = fidelity_rest[:next_section]
            after_fidelity = fidelity_rest[next_section:]
            result = result[:fidelity_start] + after_fidelity + "\n\n" + fidelity_block
        else:
            # Fidelity rule is already last — no reordering needed
            pass

    return result


def _build_compact_disambiguation() -> str:
    """Build a compact disambiguation table from top 15 pairs."""
    lines = ["=== CHARACTER DISAMBIGUATION (TOP 15) ==="]
    lines.append("When a character is ambiguous, use context:")
    for pair, rule in _TOP_DISAMBIGUATION_PAIRS:
        lines.append(f"  {pair}: {rule}")
    lines.append("")
    lines.append("VALIDATION: If a word doesn't look right, try swapping confusion pair characters.")
    return "\n".join(lines)
