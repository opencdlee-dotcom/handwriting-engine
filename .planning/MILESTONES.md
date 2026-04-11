# Milestones

## v2.0 Maximum Accuracy (Shipped: 2026-04-11)

**Phases:** 5 phases, 6 plans
**Timeline:** 2026-04-09 → 2026-04-11 (3 days)
**Files changed:** 22 files, +3,222 / -31 lines
**Test suite:** 442 passed, 0 failed
**Git range:** ad6fe35..752a687

**What shipped:**
Pushed the handwriting engine accuracy ceiling by adding the Journal of Documentation 2025 self-correction technique (two-pass LLM read), line-level page segmentation, two new local model providers (PaddleOCR 3.0 + TrOCR with per-writer fine-tuning), Sauvola adaptive binarization, WriterProfileStore for cross-session calibration, domain spell correction, and a full benchmark comparison harness.

**Key accomplishments:**
- Self-correction consensus strategy + uncertainty-gated escalation in `_smart_route()` (REQ-001, REQ-002)
- Line-level segmentation pipeline via OpenCV horizontal projection profile (REQ-003)
- PaddleOCR 3.0 (PP-OCRv5) and TrOCR VisionProvider implementations (REQ-004, REQ-005)
- Sauvola adaptive binarization in `enhance_image(strategy="sauvola")` (REQ-006)
- WriterProfileStore wired into `read_page()` and `read_with_consensus()` (REQ-007)
- Domain spell correction post-processor with biology/chemistry/general word lists (REQ-008)
- Benchmark `--compare-strategies` and `--preprocessing` CLI flags (REQ-009)

**Archive:** `.planning/milestones/v2.0-ROADMAP.md` | `.planning/milestones/v2.0-REQUIREMENTS.md`

---
