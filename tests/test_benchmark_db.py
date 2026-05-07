"""Tests for benchmark database — schema creation, CRUD, queries."""

import pytest
from handwriting_engine.benchmark.db import (
    finish_run,
    get_connection,
    get_latest_ground_truth,
    get_latest_run_id,
    get_run_results,
    get_sample_by_hash,
    get_sample_by_id,
    insert_eval_metric,
    insert_ground_truth,
    insert_provider_output,
    insert_quality_assessment,
    insert_run,
    insert_sample,
    list_runs,
    list_samples,
    record_correction,
    samples_with_ground_truth,
)


@pytest.fixture
def db():
    """In-memory SQLite database for testing."""
    conn = get_connection(":memory:")
    yield conn
    conn.close()


class TestSchemaCreation:
    def test_tables_created(self, db):
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "samples" in table_names
        assert "ground_truths" in table_names
        assert "quality_assessments" in table_names
        assert "runs" in table_names
        assert "provider_outputs" in table_names
        assert "eval_metrics" in table_names
        assert "schema_version" in table_names

    def test_schema_version(self, db):
        row = db.execute("SELECT version FROM schema_version").fetchone()
        from handwriting_engine.benchmark.db import CURRENT_SCHEMA_VERSION
        assert row["version"] == CURRENT_SCHEMA_VERSION

    def test_idempotent(self, db):
        # Calling ensure_schema again should not fail
        from handwriting_engine.benchmark.db import ensure_schema
        ensure_schema(db)

    def test_v4_migration_columns(self, db):
        """v4 migration must add provenance columns to runs and marker_rate to provider_outputs."""
        runs_cols = {row["name"] for row in db.execute("PRAGMA table_info(runs)").fetchall()}
        assert "model_version" in runs_cols, "runs.model_version missing — v4 migration not applied"
        assert "iam_partition" in runs_cols, "runs.iam_partition missing — v4 migration not applied"
        assert "norm_flags" in runs_cols, "runs.norm_flags missing — v4 migration not applied"
        assert "vocab_hints_off" in runs_cols, "runs.vocab_hints_off missing — v4 migration not applied"

        po_cols = {row["name"] for row in db.execute("PRAGMA table_info(provider_outputs)").fetchall()}
        assert "question_marker_rate" in po_cols, "provider_outputs.question_marker_rate missing — v4 migration not applied"

    def test_v5_corrections_table(self, db):
        """v5 migration must create the corrections table with expected columns."""
        tables = {r["name"] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "corrections" in tables, "corrections table missing — v5 migration not applied"

        cols = {r["name"] for r in db.execute("PRAGMA table_info(corrections)").fetchall()}
        for required in {"sample_id", "ground_truth_id", "original_text", "confidence", "source", "created_at"}:
            assert required in cols, f"corrections.{required} missing"


class TestSamples:
    def test_insert_and_retrieve(self, db):
        sid = insert_sample(db, "/path/img.png", "abc123", student="alice")
        assert sid > 0

        sample = get_sample_by_hash(db, "abc123")
        assert sample is not None
        assert sample.image_path == "/path/img.png"
        assert sample.student == "alice"

    def test_duplicate_hash_raises(self, db):
        insert_sample(db, "/path/a.png", "hash1")
        with pytest.raises(Exception):
            insert_sample(db, "/path/b.png", "hash1")

    def test_get_by_id(self, db):
        sid = insert_sample(db, "/path/img.png", "hash2")
        sample = get_sample_by_id(db, sid)
        assert sample is not None
        assert sample.id == sid

    def test_get_missing(self, db):
        assert get_sample_by_hash(db, "nonexistent") is None
        assert get_sample_by_id(db, 999) is None

    def test_list_samples(self, db):
        insert_sample(db, "/a.png", "h1")
        insert_sample(db, "/b.png", "h2")
        samples = list_samples(db)
        assert len(samples) == 2


class TestGroundTruths:
    def test_insert_and_retrieve(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        gt_id = insert_ground_truth(db, sid, "the answer is 42")
        assert gt_id > 0

        gt = get_latest_ground_truth(db, sid)
        assert gt is not None
        assert gt.text == "the answer is 42"
        assert gt.source == "manual"

    def test_versioning_latest_wins(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        insert_ground_truth(db, sid, "first version")
        insert_ground_truth(db, sid, "corrected version")

        gt = get_latest_ground_truth(db, sid)
        assert gt.text == "corrected version"

    def test_samples_with_ground_truth(self, db):
        s1 = insert_sample(db, "/a.png", "h1")
        s2 = insert_sample(db, "/b.png", "h2")
        insert_ground_truth(db, s1, "text for a")

        with_gt = samples_with_ground_truth(db)
        assert len(with_gt) == 1
        assert with_gt[0].id == s1
        assert with_gt[0].has_ground_truth is True

    def test_no_ground_truth(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        gt = get_latest_ground_truth(db, sid)
        assert gt is None


class TestQualityAssessments:
    def test_insert(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        assessment = {
            "blur_score": 150.0,
            "contrast_score": 0.6,
            "brightness_mean": 130.0,
            "brightness_std": 45.0,
            "quality": "good",
            "issues": [],
            "faint_ink": False,
        }
        insert_quality_assessment(db, sid, assessment)

        row = db.execute("SELECT * FROM quality_assessments WHERE sample_id = ?", (sid,)).fetchone()
        assert row["quality"] == "good"
        assert row["blur_score"] == 150.0

    def test_upsert(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        insert_quality_assessment(db, sid, {"quality": "poor", "blur_score": 50})
        insert_quality_assessment(db, sid, {"quality": "good", "blur_score": 200})

        row = db.execute("SELECT * FROM quality_assessments WHERE sample_id = ?", (sid,)).fetchone()
        assert row["quality"] == "good"


class TestRuns:
    def test_create_and_finish(self, db):
        run_id = insert_run(db, label="test run", providers=["gemini"], strategies=["single"])
        assert run_id > 0

        finish_run(db, run_id, sample_count=5)

        runs = list_runs(db)
        assert len(runs) == 1
        assert runs[0].label == "test run"
        assert runs[0].sample_count == 5
        assert runs[0].finished_at != ""

    def test_latest_run_id(self, db):
        assert get_latest_run_id(db) is None
        r1 = insert_run(db, label="first")
        r2 = insert_run(db, label="second")
        assert get_latest_run_id(db) == r2


class TestProviderOutputsAndMetrics:
    def test_full_pipeline(self, db):
        sid = insert_sample(db, "/img.png", "h1")
        gt_id = insert_ground_truth(db, sid, "ground truth text")
        run_id = insert_run(db, label="test")

        po_id = insert_provider_output(
            db, run_id, sid,
            provider="gemini", strategy="single",
            output_text="ground truth text",
            confidence=0.7, latency_ms=500,
            input_tokens=100, output_tokens=50,
        )
        assert po_id > 0

        em_id = insert_eval_metric(
            db, po_id, gt_id,
            cer=0.0, wer=0.0,
            char_edits=0, word_edits=0,
            ref_chars=17, ref_words=3,
        )
        assert em_id > 0

        results = get_run_results(db, run_id)
        assert len(results) == 1
        assert results[0]["provider"] == "gemini"
        assert results[0]["cer"] == 0.0


class TestCorrections:
    """S4: record_correction() instructor-feedback API."""

    @pytest.fixture
    def fake_image(self, tmp_path):
        path = tmp_path / "page.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes-for-hashing")
        return str(path)

    def test_first_call_creates_sample_gt_and_correction(self, db, fake_image):
        cid = record_correction(
            db,
            image_path=fake_image,
            writer_id="prof-bio101-jdoe",
            corrected_text="OD600 = 0.45",
            original_vlm_text="OD60O = 0.45",
            confidence=0.62,
        )
        assert cid > 0

        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 1

        sample = db.execute("SELECT * FROM samples").fetchone()
        assert sample["student"] == "prof-bio101-jdoe"

    def test_idempotent_identical_args(self, db, fake_image):
        """Identical-args double-call: 1 sample + 1 GT + 2 corrections (history grows)."""
        kwargs = dict(
            image_path=fake_image,
            writer_id="prof-bio101-jdoe",
            corrected_text="OD600 = 0.45",
            original_vlm_text="OD60O = 0.45",
            confidence=0.62,
        )
        record_correction(db, **kwargs)
        record_correction(db, **kwargs)

        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 2

    def test_different_correction_text_adds_new_gt(self, db, fake_image):
        """Same image, two different corrected_text values → 1 sample, 2 GTs, 2 corrections."""
        record_correction(
            db, image_path=fake_image, writer_id="w1",
            corrected_text="first read", original_vlm_text="orig", confidence=0.6,
        )
        record_correction(
            db, image_path=fake_image, writer_id="w1",
            corrected_text="revised read", original_vlm_text="orig", confidence=0.6,
        )
        assert db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM ground_truths").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 2

    def test_per_writer_accumulation(self, db, tmp_path):
        """Per spec criterion #5: same writer across 'sessions' adds to same student value."""
        for i in range(3):
            img = tmp_path / f"p{i}.png"
            img.write_bytes(f"png-bytes-{i}".encode())
            record_correction(
                db, image_path=str(img), writer_id="prof-bio101-jdoe",
                corrected_text=f"text {i}", original_vlm_text=f"orig {i}", confidence=0.5,
            )
        count = db.execute(
            "SELECT COUNT(*) FROM samples WHERE student = ?", ("prof-bio101-jdoe",)
        ).fetchone()[0]
        assert count == 3

    def test_correction_links_to_real_sample_and_gt(self, db, fake_image):
        cid = record_correction(
            db, image_path=fake_image, writer_id="w1",
            corrected_text="real text", original_vlm_text="vlm text", confidence=0.8,
            source="labgrader",
        )
        row = db.execute("SELECT * FROM corrections WHERE id = ?", (cid,)).fetchone()
        assert row["original_text"] == "vlm text"
        assert row["confidence"] == 0.8
        assert row["source"] == "labgrader"

        # FK integrity: linked sample and GT must exist
        assert db.execute("SELECT 1 FROM samples WHERE id = ?", (row["sample_id"],)).fetchone()
        assert db.execute("SELECT 1 FROM ground_truths WHERE id = ?", (row["ground_truth_id"],)).fetchone()
