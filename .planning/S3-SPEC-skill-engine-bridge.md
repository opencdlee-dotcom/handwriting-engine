# S3 — Bridge `handwriting-reader` Skill to Engine Library

**Status:** SPEC (not yet a phase). Implementation unblocked — no data dependency.
**Authored:** 2026-05-06
**Source:** `.planning/NEXT-STEPS.md` § Out-of-band side projects.

---

## Goal

When the user invokes `/handwriting-reader <image>`, the skill calls `handwriting_engine.read_with_consensus()` (or `read_page()` for single-provider) directly instead of executing its own multi-pass workflow described in [SKILL.md](~/.claude/skills/handwriting-reader/SKILL.md). One source of truth: any future improvement to the engine — new strategy, new provider, postprocess upgrade, S2 few-shot, S5 char-consensus — propagates to the skill on the next invocation, no skill edits needed.

## Why this is high-leverage

- The skill currently re-implements Phase 1 (classify) → Phase 2 (multi-pass extract) → confidence markers in its own prompt logic. The engine has all of this and more (`read_with_consensus`, `assess_image`, `proven_enhance`, `correct`, the writer-profile path). Two implementations means two failure surfaces and predictable drift.
- Both repos are local. Importing the engine from the skill is a one-time wiring task.
- Pre-requisite for S4 to pay off: if the grader writes corrections back to `benchmark.db` while the skill bypasses the engine entirely, the skill's per-writer accuracy never compounds — half the feedback loop is missing.

## Non-goals

- Eliminating the skill. The skill is the conversational entry point (`/handwriting-reader …`); only its *internals* change.
- Changing the skill's user-visible interface. `--format=`, `--strict`, `--domain=`, `--output` arguments and HEIC/PDF handling stay identical.
- Engine changes. The engine's public API is sufficient as-is — verified by inspection of [`handwriting_engine/__init__.py`](../handwriting_engine/__init__.py).

---

## Design

### 1. Skill restructure ([~/.claude/skills/handwriting-reader/SKILL.md](~/.claude/skills/handwriting-reader/SKILL.md))

Replace the Phase 1/Phase 2 prose workflow with a thin orchestration layer:

```python
from handwriting_engine import (
    read_with_consensus,
    assess_image,
    proven_enhance,
    convert_pdf,
)
from handwriting_engine.writer_profile_store import WriterProfileStore

# 1. Path validation (unchanged from current skill — this is policy, not engine concern)
# 2. PDF expansion via convert_pdf() if .pdf
# 3. HEIC conversion via sips shell (already documented; stays in skill)
# 4. quality = assess_image(path); if quality.needs_enhancement: proven_enhance()
# 5. profile = WriterProfileStore().load(writer_id) if writer_id else None
# 6. result = read_with_consensus(path, writer_profile=profile, domain=domain)
# 7. Format result per --format= flag
# 8. If --strict: prompt user to resolve each [?alt: …] marker before output
```

The skill's job becomes: input parsing, file I/O, output formatting, and `--strict` interactive UX. Engine handles every transcription decision.

### 2. Domain auto-detection

Currently the skill auto-detects `--domain=bio` by scanning content. That can't happen *before* transcription. Two valid orderings:

- **A. Two-pass:** quick `read_page()` with no domain, scan output for biology terms, then `read_with_consensus()` with the detected domain. Cost: 2× the cheap-provider call.
- **B. Trust user / default to general:** require explicit `--domain=` for non-default; otherwise use `domain="general"`. Cost: 0 extra calls, but loses auto-detection.

**Decision:** B (default to general; `--domain=bio` opt-in). Auto-detection adds latency + cost for marginal accuracy gain on a feature the user is already explicitly invoking. Document the tradeoff in SKILL.md so the user can pass `--domain=bio` when needed. If real-world usage shows users frequently forget the flag, revisit.

### 3. `--strict` mode

Today it's a hand-rolled "confirm every `[?]`" loop in the skill. Engine's consensus output already produces `[?alt: X/Y]` markers in `read_with_consensus`. The skill iterates them post-hoc:

```python
for marker in extract_alt_markers(result.text):
    chosen = ask_user(f"Choose: {marker.alternatives}")
    result.text = result.text.replace(marker.raw, chosen, 1)
```

`extract_alt_markers` is regex over `\[\?alt: ([^\]]+)\]` — keep it skill-side, no engine API needed.

### 4. Writer profile binding

If the user passes `--writer=<name>`, look up the profile via `WriterProfileStore().load()` and forward to `read_with_consensus`. New flag — additive, doesn't break callers.

---

## Falsifiable success criteria

When this phase is complete, all of the following are TRUE:

1. **Single source of truth for transcription.** `grep -r "Pass 1\|Pass 2\|multi-pass" ~/.claude/skills/handwriting-reader/` returns nothing — the multi-pass logic is removed from the skill prose.
2. **End-to-end parity.** Running `/handwriting-reader sample.jpg --format=md` on a fixture image produces output whose CER vs. ground truth is **≤** the pre-S3 skill's CER on the same image. Tested across a 10-image fixture covering: typed text page (control), neat printed handwriting, cursive, lab-notebook table, and a deliberately blurry image.
3. **Engine improvements propagate.** Bumping a postprocess threshold in [`handwriting_engine/postprocess.py`](../handwriting_engine/postprocess.py) and re-running the skill produces a measurably different output **without** touching SKILL.md. Verified by a git-bisect-style test: change → invoke → diff.
4. **Format flags preserved.** `--format=json` returns the schema documented in pre-S3 SKILL.md (or a richer engine-native schema with the previous fields as a strict subset). No breaking changes for existing automations.
5. **Strict mode works.** `--strict` prompts the user once per `[?alt: …]` marker emitted by `read_with_consensus`, no more, no fewer.
6. **PDF + HEIC unchanged.** Same input handling rules (`pages=`, `sips` conversion note) — verified by re-running each pre-S3 SKILL.md example.
7. **Performance baseline.** Skill latency on a typical lab-notebook page is within 1.2× of pre-S3 (consensus is more expensive than single-pass; this is acceptable, not free). Documented in SKILL.md.

## Out of scope (queued for follow-up)

- **Streaming output to chat.** Today the skill renders output as one block. Streaming partial transcriptions during the call is a separate UX project.
- **GUI/TUI for `--strict`.** Stays text-prompt-based.
- **Skill-level caching of recent reads.** Engine doesn't cache; if added later, do it engine-side.

---

## Risks & open questions

| Risk | Mitigation / decision needed |
|------|------------------------------|
| The skill runs in Claude Code's runtime, not a Python process. The skill is markdown + tool calls, not Python imports. | The skill's "implementation" is Claude executing instructions in SKILL.md. The bridge is: SKILL.md instructs Claude to invoke the engine via the Bash tool: `python3 -m handwriting_engine.cli read <path> --format=json --domain=…`. So S3 reduces to **(a)** ensuring the engine has a CLI surface that maps to every SKILL.md feature, and **(b)** rewriting SKILL.md to call that CLI rather than doing prose-driven multi-pass. **This is the dominant design correction vs. the original framing.** |
| The engine CLI today (`handwriting_engine/cli.py`) is benchmark-focused. Does it expose a top-level `read` command? | Audit needed — pre-implementation check. If missing, scope adds an engine-side `cli read` subcommand that wraps `read_with_consensus`. Likely already exists in some form; verify before sizing. |
| Skill output format may not match what the engine CLI emits today. | Acceptable spec change: engine CLI gains `--format=md|json|txt` flags; skill's role becomes a thin wrapper + UX shell. |
| HEIC handling logic (the `sips` shell command) is pure environmental tooling, not engine concern. | Stays in SKILL.md. |
| `--strict` interactive prompting can't happen inside the engine CLI (Claude can't interact mid-CLI-call). | Engine CLI returns a structured payload (JSON) including the alt-markers as separate fields; skill (Claude) iterates and prompts. |

## Touchpoints (preliminary)

- **Edit:** [~/.claude/skills/handwriting-reader/SKILL.md](~/.claude/skills/handwriting-reader/SKILL.md) — replace Phase 1/2 prose with engine CLI invocation + UX layer.
- **Edit (likely):** [`handwriting_engine/cli.py`](../handwriting_engine/cli.py) — add `read` top-level command if missing; ensure `--format=md|json|txt` is supported with stable JSON schema.
- **Edit:** [`~/.claude/skills/handwriting-reader/references/`](~/.claude/skills/handwriting-reader/references/) — likely contains heuristics that move into the engine or get retired.
- **Doc:** SKILL.md changelog note explaining the engine bridge (so future-Claude doesn't re-grow the multi-pass logic in SKILL.md the next time someone "improves" it).

## Estimated scope

- 1 plan, ~150 LOC engine-side (CLI surface), ~0 LOC skill (prose rewrite), ~200 LOC tests.
- No data dependency. Can ship now.

---

## Promotion path

`/gsd-add-phase` (this is naturally next in the v3.0 milestone — it lifts a recurring drift cost). Use this SPEC.md as the seed for `/gsd-discuss-phase` to nail down the engine CLI schema before coding.
