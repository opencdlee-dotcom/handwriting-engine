---
phase: 09-final-sweep-recommendation-baseline-lock
plan: "03"
subsystem: benchmark
tags: [ingest, ground-truth, lab, click-edit, cli, tdd]

requires:
  - phase: 07
    provides: hash_file, insert_sample, get_sample_by_hash patterns reused for dedup

provides:
  - benchmark/ingest.py:ingest_lab(directory, *, student, prompt_fn,
                                   db_path, use_vlm_suggestion, vlm_provider) -> dict
  - CLI: benchmark ingest-lab DIRECTORY [--student S] [--with-suggestion] [--vlm-provider P]
  - Ground-truth rows inserted with source='lab-grader' and the student tag stored as `author` for provenance

affects:
  - Production-distribution test set is now collectable without IAM dependency
  - benchmark recommend / compare can run against lab samples once they have GT
  - Future graders (S4 / labgrader bridge) can read these GTs as the trusted reference

tech-stack:
  added: []
  patterns:
    - prompt_fn dependency injection: production CLI uses click.edit, tests inject deterministic callables
    - Resumable: dedup by file hash; samples with existing ground_truth are silently skipped
    - VLM suggestion is opt-in via --with-suggestion to keep cost predictable; failures degrade to empty suggestion rather than aborting the workflow

key-files:
  created:
    - tests/test_benchmark_ingest_lab.py
  modified:
    - handwriting_engine/benchmark/ingest.py
    - handwriting_engine/cli.py

key-decisions:
  - "prompt_fn callable, not a hardcoded EDITOR call. Lets tests run without spawning $EDITOR and lets a future GUI / web frontend reuse the same workflow function."
  - "Whitespace-only return treated as skip. The user clearing the buffer signals 'I can't read this' just as clearly as click.edit returning None."
  - "Sample inserted BEFORE prompt, not after. If the user skips, the sample row still exists so they can revisit it later via `benchmark transcribe`. Re-running ingest-lab won't re-create the row (hash dedup) but also won't re-prompt unless GT is missing."
  - "VLM failures swallowed with warning, not raised. The point of ingest-lab is capturing GT; a transient VLM outage shouldn't block the workflow — the user just types the transcription manually."

verification:
  unit_coverage:
    - 10 tests in tests/test_benchmark_ingest_lab.py
  criterion_status:
    - "RPT-03: IMPLEMENTED. Includes resume semantics, VLM degradation, and CLI."
