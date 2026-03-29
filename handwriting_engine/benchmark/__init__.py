"""
Benchmark system for handwriting recognition accuracy measurement.

Provides ground-truth storage, multi-provider evaluation, CER/WER metrics,
domain term accuracy, error taxonomy, regression detection, data amplification,
and lessons feedback integration.
"""

from handwriting_engine.benchmark.db import get_connection
from handwriting_engine.benchmark.evaluate import run_benchmark
from handwriting_engine.benchmark.ingest import (
    bootstrap_ground_truth,
    generate_degraded_variants,
    ingest_directory,
    ingest_single,
)
from handwriting_engine.benchmark.lessons_bridge import (
    extract_gt_from_lessons,
    feed_errors_to_lessons,
)
from handwriting_engine.benchmark.metrics import (
    character_error_rate,
    classify_errors,
    domain_term_accuracy,
    levenshtein_distance,
    normalize_text,
    word_error_rate,
)
from handwriting_engine.benchmark.models import (
    EvalMetric,
    GroundTruth,
    ProviderOutput,
    RunSummary,
    Sample,
    StrategyResult,
)
from handwriting_engine.benchmark.report import (
    compare_runs,
    confidence_calibration,
    detect_regressions,
    generate_report,
    quality_correlation,
    sample_drill_down,
)

__all__ = [
    # Database
    "get_connection",
    # Ingestion
    "ingest_directory",
    "ingest_single",
    "generate_degraded_variants",
    "bootstrap_ground_truth",
    # Evaluation
    "run_benchmark",
    # Metrics
    "character_error_rate",
    "word_error_rate",
    "levenshtein_distance",
    "normalize_text",
    "domain_term_accuracy",
    "classify_errors",
    # Reporting
    "generate_report",
    "compare_runs",
    "detect_regressions",
    "sample_drill_down",
    "quality_correlation",
    "confidence_calibration",
    # Lessons bridge
    "extract_gt_from_lessons",
    "feed_errors_to_lessons",
    # Models
    "Sample",
    "GroundTruth",
    "ProviderOutput",
    "EvalMetric",
    "StrategyResult",
    "RunSummary",
]
