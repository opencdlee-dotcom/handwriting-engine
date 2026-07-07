# Handwriting Engine — Project Memory

## Quick Context
Project: handwriting-engine | Version: 0.1.0 | Status: Initial build
Consolidates ~3,000 lines from 8 locations into one reusable library.

## Architecture
- **Entry point**: `main.py` (click CLI) + `handwriting_engine/__init__.py` (library API)
- **Core modules**: quality, enhance, pdf, optimize, crop, handwriting, models
- **Providers**: claude, openai, gemini — each implements VisionProvider protocol
- **Transcription vs. intelligence**: `vision.read_page` = flat verbatim text; `document_intelligence.analyze_document` = structured in-context interpretation (layout regions, tables-as-data, figures interpreted, equations, cross-content findings). The latter reuses the providers' previously-unused `read_structured` (tool-use / JSON-schema) path.
- **Consensus**: vote/best_of/debate strategies for multi-model reads
- **Benchmark**: `benchmark/` subpackage — SQLite ground-truth DB, CER/WER metrics, regression detection
- **Prompt adaptation**: `prompt_adapter.py` — provider-specific prompt optimization (Gemini=concise, OpenAI=role+task, Claude=full)
- **Writer identification**: `writer_embeddings.py` — Gemini multimodal embeddings for automatic writer clustering
- **Batch API**: `providers/batch_openai.py` — 50% cost OpenAI batch processing for benchmarks
- **Config**: `_constants.py` (defaults) + `config.py` (loads .env)

## Key Design Decisions
- **Multi-model consensus**: Gemini best overall (1.67% CER), Claude for layouts, GPT for disambiguation
- **Gemini uses system_instruction**: NOT string concatenation — proper API separation improves instruction following
- **Gemini media_resolution=HIGH**: Explicit max resolution for OCR (not left to auto-select)
- **Gemini temperature 0.5**: Research shows temp 0 triggers degenerate sampling on Flash OCR tasks
- **Proven enhancement pipeline**: Grayscale→autocontrast(2)→sharpen(3.0)→contrast(2.0)→brightness(1.1)→2x upscale
- **Contrast lowered to 2.0**: PreP-OCR (ACL 2025) found 2.5 causes LLM hallucinations
- **Tool-use structured output**: Claude's tool_choice="any" for reliable JSON (from LabNoteBookGrader)
- **OpenAI detail="high"**: Best available for GPT-4.1 ("original" is GPT-5.4+ only)
- **Lazy provider imports**: Missing SDK won't crash the engine
- **Model versions pinned**: GPT-4.1-2025-04-14 prevents silent regression (WER doubled in 6mo with rolling models)
- **Gemini context caching**: enable_context_cache() for 90% discount on batch workflows
- **Confidence-weighted voting**: Consensus weights adjusted per-read by text quality signals
- **Vocabulary hints**: 20-50 domain terms injected via vocabulary_hints parameter (research sweet spot)
- **All functions accept parameter overrides**: Defaults from _constants.py, never hardcoded

## Benchmark System
- **DB location**: `~/.handwriting-engine/benchmark.db` (SQLite, auto-created)
- **CLI**: `handwriting-engine benchmark {ingest,transcribe,list,run,report,compare,drill-down,quality,degrade,bootstrap-gt}`
- **Workflow**: ingest images → add ground-truth transcriptions → run benchmarks → view reports
- **Metrics**: CER, WER (Levenshtein), Domain Term Accuracy (biology terms), Error Taxonomy (confusion pairs/substitution/insertion/deletion)
- **Data amplification**: `degrade` generates 5 synthetic variants (blur, low-contrast, rotate, noise, crop) sharing original ground truth
- **Bootstrap GT**: `bootstrap-gt` auto-generates ground truth when all providers agree within 2% CER
- **Smoke mode**: `--smoke` flag on `benchmark run` tests 3 hardest samples only for fast CI checks
- **Lessons bridge**: `--feed-lessons` flag feeds high-error outputs back into lessons system
- **Reports**: table/json/csv aggregate, per-sample drill-down, quality-vs-accuracy correlation, run comparison with regression detection
- **No new dependencies**: sqlite3 is stdlib, all DB functions use try/finally for connection safety

## Gotchas
- Never send images >1568px long side to Claude without resizing
- OpenCV is optional — crop.py falls back to Pillow for CLAHE
- anthropic/openai/google-genai are all optional dependencies
- Memory safety: process large PDFs in 5-page batches with gc.collect()
- RGBA preservation required in all enhancement functions
