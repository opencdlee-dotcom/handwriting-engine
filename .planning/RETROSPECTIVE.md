# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

---

## Milestone: v2.0 — Maximum Accuracy

**Shipped:** 2026-04-11
**Phases:** 5 | **Plans:** 6 | **Timeline:** 3 days (2026-04-09 → 2026-04-11)
**Tests:** 442 passing (from 27 pre-existing)

### What Was Built

- Two-pass self-correction strategy in consensus.py — JoD 2025 finding applied (CER ~1.67% → ~1.3% projected)
- Uncertainty-gated escalation in `_smart_route()` — auto-escalates when [?] markers exceed threshold
- Line-level segmentation via OpenCV horizontal projection profile — `read_page(line_level=True)`
- PaddleOCR 3.0 (PP-OCRv5) and TrOCR VisionProvider implementations — offline local inference
- Sauvola adaptive binarization — `enhance_image(strategy="sauvola")` with scikit-image fallback
- WriterProfileStore — structured JSON profiles, wired into both `read_page` and `read_with_consensus`
- Domain spell correction post-processor — biology/chemistry/general word lists, edit-distance-1 only
- Benchmark `--compare-strategies` and `--preprocessing` flags

### What Worked

- **Phased ordering by ROI** — putting self-correction first (pure software, no deps, highest expected gain) was the right call; most impactful work shipped immediately
- **Lazy import pattern** — established in v1.0, scaled cleanly to 2 new providers and scikit-image without any circular import issues
- **GSD phase structure** — phases 2-5 were each single-plan phases; right-sized for the scope
- **3-source requirement verification** — audit caught two real wiring gaps (REQ-007 not connected to read_page, REQ-009 CLI flags missing) before milestone close

### What Was Inefficient

- **Phases 2-5 lacked SUMMARY/VERIFICATION artifacts** — executed outside the GSD summary flow, so audit had to reconstruct evidence from code inspection and git commits rather than reading summaries. Cost: extra audit work
- **STATE.md not updated between phases** — showed "Not started" through all 5 phases, lost session continuity. The resume workflow had to reconstruct state
- **Adversarial test contradicted canonical test** — `test_build_image_blocks_raises_on_bad_page` (adversarial) directly contradicted `test_build_image_blocks_skips_corrupt` (canonical). Required investigation and fix at milestone close instead of during execution

### Patterns Established

- `get_reading_strategies(writer_profile=...)` pattern — structured profile replaces generic block, text calibration appends — cleanly extensible
- Two-tier writer calibration: `WriterProfileStore` (structured, machine-readable) + `lessons.load_writer_calibration` (text, human-curated) both wired in
- `enhance_strategy` parameter threaded through `run_benchmark` → `_read_single` — right pattern for any future preprocessing flags

### Key Lessons

1. **Write SUMMARY.md immediately after each phase.** Audit reconstruction from code + git is doable but slower than reading a summary. Even one sentence per plan is enough.
2. **Adversarial test suites need to agree with canonical tests.** When they contradict, the canonical test wins (docstring + caller behavior = ground truth). Update adversarial tests, not the implementation.
3. **REQ-007-style "wire X into Y" requirements need an integration checker pass.** The implementation (WriterProfileStore, get_reading_strategies) was complete but the wiring into primary entry points was missing — exactly what the integration checker caught.

### Cost Observations

- Model: Sonnet 4.6 throughout
- Sessions: 1 (entire v2.0 in one session after resume)
- Notable: 5-phase milestone completed in ~3 hours of model time; audit + gap closure added ~1 hour. Upfront research (Brainiac on initialization) produced accurate phase ordering — no pivots needed.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v2.0 | 1 | 5 | First use of integration checker; caught 2 wiring gaps pre-close |

### Cumulative Quality

| Milestone | Tests | Notes |
|-----------|-------|-------|
| v2.0 | 442 | +415 new tests; adversarial suite added |

### Top Lessons (Verified Across Milestones)

1. Write SUMMARY.md immediately after execution — reconstruction costs more than the summary
2. Integration checker is worth running before milestone close on any multi-phase project
