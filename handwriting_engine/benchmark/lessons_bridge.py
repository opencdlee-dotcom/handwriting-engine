"""
Bidirectional bridge between the benchmark database and the lessons system.

Direction 1: Extract ground-truth fragments from existing lessons.
Direction 2: Feed high-error benchmark outputs back into lessons for future reads.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from handwriting_engine.benchmark.db import (
    get_connection,
    get_latest_ground_truth,
    get_run_results,
)

logger = logging.getLogger(__name__)


def extract_gt_from_lessons(
    lessons_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Parse lessons JSONL files for correction patterns.

    Looks for patterns like:
        "Model read 'X' but correct reading is 'Y'"
    and logs them. These are partial (word/phrase level), useful for
    understanding common error patterns.

    Returns:
        Number of correction patterns found.
    """
    from handwriting_engine.lessons import DEFAULT_LESSONS_DIR

    base = Path(lessons_dir) if lessons_dir else DEFAULT_LESSONS_DIR
    if not base.exists():
        return 0

    pattern = re.compile(
        r"Model read '([^']+)' but correct reading is '([^']+)'"
    )
    count = 0

    for jsonl_file in base.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = entry.get("text", "")
                if pattern.search(text):
                    count += 1

    return count


def feed_errors_to_lessons(
    run_id: int,
    min_cer: float = 0.1,
    category: str = "benchmark",
    db_path: Path | str | None = None,
    lessons_dir: Path | str | None = None,
) -> int:
    """Feed high-error outputs from a benchmark run back into the lessons system.

    For provider outputs where CER > min_cer, diffs the output against ground
    truth using the existing auto_extract_lessons() function.

    Args:
        run_id: Benchmark run to process.
        min_cer: Minimum CER threshold — only process outputs worse than this.
        category: Lessons category to save corrections under.
        db_path: Override benchmark database path.
        lessons_dir: Override lessons storage directory.

    Returns:
        Number of lessons generated.
    """
    from handwriting_engine.lessons import auto_extract_lessons

    conn = get_connection(db_path)
    try:
        rows = get_run_results(conn, run_id)

        total_lessons = 0
        for row in rows:
            cer = row.get("cer")
            if cer is None or cer <= min_cer:
                continue

            output_text = row.get("output_text", "")
            if not output_text or row.get("error"):
                continue

            # Only use manually verified GT to avoid poisoning the lessons
            # system with auto-consensus GT that may contain systematic errors
            gt = get_latest_ground_truth(conn, row["sample_id"])
            if not gt or gt.source not in ("manual", "corrected_output"):
                continue

            count = auto_extract_lessons(
                model_output=output_text,
                corrected_text=gt.text,
                category=category,
                lessons_dir=lessons_dir,
            )
            total_lessons += count
            if count > 0:
                logger.info(
                    "Extracted %d lessons from %s/%s on sample %d (CER=%.2f%%)",
                    count, row["provider"], row["strategy"], row["sample_id"], cer * 100,
                )

        return total_lessons
    finally:
        conn.close()
