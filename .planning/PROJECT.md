# Handwriting Engine — Project

> The world's most accurate LLM-vision handwriting recognition engine, with self-correction, line-level segmentation, local model providers, and writer adaptation.

## What This Is

A Python library (`handwriting_engine`) that uses LLM vision APIs (Gemini, Claude, GPT-4) to transcribe handwritten documents with industry-leading accuracy. Combines multi-model consensus strategies, self-correction loops, line-level segmentation, and optional local models (PaddleOCR, TrOCR) into a single reusable engine used for lab notebook grading and document digitization.

**Core Value:** Highest accuracy handwriting transcription available — better than Azure, GPT-4o, and all dedicated HTR models — now with self-correction and ensemble expansion.

## Current State (post v2.0)

| Metric | Value |
|--------|-------|
| Codebase | ~15,700 LOC Python |
| Test suite | 442 passing |
| Providers | Gemini, Claude, OpenAI, PaddleOCR (optional), TrOCR (optional) |
| Consensus strategies | vote, best_of, debate, self_correct, smart (uncertainty-gated) |
| Baseline CER | 1.67% (Gemini Flash, IAM) |
| Self-correction target | ~1.3% CER (JoD 2025 finding applied) |

## Requirements

### Validated (v2.0)

- ✓ Self-correction consensus strategy — `read_with_consensus(strategy="self_correct")` — v2.0
- ✓ Uncertainty-gated escalation in smart strategy — auto-escalates when [?] markers > threshold — v2.0
- ✓ Line-level segmentation pipeline — `read_page(line_level=True)` — v2.0
- ✓ PaddleOCR 3.0 (PP-OCRv5) provider — lazy import, VisionProvider protocol — v2.0
- ✓ TrOCR provider with `fine_tune_for_writer()` — 5-line writer adaptation — v2.0
- ✓ Sauvola adaptive binarization — `enhance_image(strategy="sauvola")` — v2.0
- ✓ WriterProfileStore — cross-session calibration injected via `get_reading_strategies(writer_profile=...)` — v2.0
- ✓ Domain spell correction — `correct_domain_terms(text, domain)` — v2.0
- ✓ Benchmark `--compare-strategies` and `--preprocessing` CLI flags — v2.0

### Active (next milestone)

- Run IAM benchmark to measure actual CER improvement from self_correct strategy
- Measure actual [?] marker reduction from line-level segmentation on lab notebooks
- Install and benchmark PaddleOCR against IAM test set (not yet in dev env)
- Connect writer_embeddings.py cluster observations → auto-populate WriterProfileStore

### Out of Scope

- CRNN/CTC model training from scratch — LLM ensemble already beats all open-weight HTR
- Multi-language support beyond Gemini/Claude coverage
- Historical manuscript support (Kraken/Transkribus domain)
- Web UI or SaaS product
- Distributed training infrastructure

## Key Decisions

| Decision | Choice | Outcome | Status |
|----------|--------|---------|--------|
| Self-correction strategy | New `self_correct` strategy in consensus.py | Built — awaiting benchmark vs IAM | ✓ |
| Line segmentation | OpenCV projection profile | Built — fallback to whole-page on < 2 lines | ✓ |
| PaddleOCR | PP-OCRv5 via paddlepaddle + paddleocr | Built — lazy import, not yet benchmarked | ✓ |
| TrOCR | microsoft/trocr-base-handwritten via HuggingFace | Built — fine_tune_for_writer() implemented | ✓ |
| Binarization | scikit-image threshold_sauvola | Built — CLAHE fallback on missing dependency | ✓ |
| build_image_blocks error handling | Skip gracefully (return [] + warn) vs raise | Graceful skip — callers handle empty list | ✓ |
| Writer calibration architecture | WriterProfileStore (structured JSON) + lessons store (text) | Both wired in read_page — structured profile replaces generic WRITER_CALIBRATION block | ✓ |
| Gemini temperature | 0.5 (not 0) | Research shows temp 0 causes degenerate sampling on Flash OCR | ✓ |
| Gemini system_instruction | Separate field, not string concat | Proper API separation improves instruction following | ✓ |
| Optional dependencies | Lazy imports throughout | No crash when paddleocr/transformers/skimage not installed | ✓ |

## Constraints

| Constraint | Impact |
|-----------|--------|
| Python 3.10+ | All providers must be 3.10-compatible |
| Optional dependencies pattern | New libraries (paddlepaddle, transformers, scikit-image) must be lazy-imported |
| Existing VisionProvider protocol | New providers implement protocol without changes to base.py |
| No changes to consensus.py public API | New strategies added only, existing unchanged |
| Apple Silicon (M-series) dev machine | PaddleOCR/TrOCR must support arm64/CPU inference |

## Target Users

| User type | Description | Technical level |
|-----------|-------------|-----------------|
| Primary developer | Charlie building/using the engine for lab notebook grading | High (Python, ML) |
| Engine consumers | Skills/tools that call handwriting_engine | API users |

---
*Last updated: 2026-04-11 after v2.0 milestone*
