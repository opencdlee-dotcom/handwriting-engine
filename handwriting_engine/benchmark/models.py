"""
Data models for the benchmark system.

Pure dataclasses — no DB logic, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sample:
    """A registered image in the benchmark database."""

    id: int
    image_path: str
    image_hash: str
    student: str = ""
    category: str = ""
    page_number: int = 0
    notes: str = ""
    has_ground_truth: bool = False


@dataclass
class GroundTruth:
    """Human-verified transcription for a sample."""

    id: int
    sample_id: int
    text: str
    source: str = "manual"  # 'manual', 'corrected_output', 'lesson_extract'
    created_at: str = ""


@dataclass
class ProviderOutput:
    """Raw output from a single provider read during a benchmark run."""

    id: int
    run_id: int
    sample_id: int
    provider: str
    strategy: str
    output_text: str
    confidence: float = 0.0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    question_marker_rate: float | None = None


@dataclass
class EvalMetric:
    """CER/WER metrics for a provider output against ground truth."""

    provider_output_id: int
    ground_truth_id: int
    cer: float
    wer: float
    char_edits: int = 0
    word_edits: int = 0
    ref_chars: int = 0
    ref_words: int = 0


@dataclass
class StrategyResult:
    """Aggregated metrics for one (provider, strategy) combination in a run."""

    provider: str
    strategy: str
    mean_cer: float
    mean_wer: float
    median_cer: float
    median_wer: float
    stdev_cer: float
    stdev_wer: float
    total_tokens: int
    estimated_cost_usd: float
    sample_count: int
    failures: int = 0
    mean_marker_rate: float = 0.0


@dataclass
class RunSummary:
    """Overview of a completed benchmark run."""

    run_id: int
    label: str
    started_at: str
    finished_at: str = ""
    sample_count: int = 0
    results: list[StrategyResult] = field(default_factory=list)
    model_version: str | None = None
    iam_partition: str | None = None
    norm_flags: str | None = None
    vocab_hints_off: int = 0
