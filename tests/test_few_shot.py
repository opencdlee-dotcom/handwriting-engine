"""Tests for S2 — per-writer few-shot exemplars.

Maps onto the S2-SPEC falsifiable criteria:
* #1  -- eligibility gate (select_exemplars)
* #2  -- provider calls carry exemplars before the target
* #4  -- cold-writer (<2 GT samples) falls back to single-image read
* #5  -- TrOCR passthrough (no error, no exemplars)
* #6  -- HE_FEW_SHOT_K env honored (cost guardrail surface)

Criterion #3 (CER on real IAM data) requires the populated benchmark DB and
the Phase 8 stats infrastructure; it's covered by a separate eval, not here.
"""

from __future__ import annotations

import os

import pytest

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_ground_truth,
    insert_sample,
)
from handwriting_engine.few_shot import (
    DEFAULT_FEW_SHOT_K,
    EXEMPLAR_LABEL_TEMPLATE,
    EXEMPLAR_PROVIDERS,
    FEW_SHOT_K_ENV,
    build_exemplar_blocks,
    env_few_shot_k,
    provider_supports_exemplars,
    select_and_build_exemplar_blocks,
)
from handwriting_engine.writer_profile_store import (
    Exemplar,
    select_exemplars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _png_bytes(seed: int = 0) -> bytes:
    # Distinct content per seed so image_hash uniqueness holds.
    base = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfa\xcf"
        b"\x00\x00\x00\x03\x00\x01\x16\xfb\x96\xea\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base + (b"\x00" * seed)


@pytest.fixture
def png_factory(tmp_path):
    counter = {"i": 0}

    def make(name: str = None) -> str:
        counter["i"] += 1
        path = tmp_path / (name or f"img-{counter['i']}.png")
        path.write_bytes(_png_bytes(counter["i"]))
        return str(path)

    return make


@pytest.fixture
def db():
    conn = get_connection(":memory:")
    yield conn
    conn.close()


def _seed_writer(db, *, writer_id: str, n: int, png_factory) -> list[int]:
    """Insert n samples + GTs for a writer; returns the sample ids in order."""
    sample_ids = []
    for idx in range(n):
        path = png_factory(name=f"{writer_id}-{idx}.png")
        from handwriting_engine.benchmark.ingest import hash_file

        sid = insert_sample(
            db,
            image_path=path,
            image_hash=hash_file(path),
            student=writer_id,
        )
        insert_ground_truth(db, sid, f"sample-{idx} text for {writer_id}")
        sample_ids.append(sid)
    return sample_ids


# ---------------------------------------------------------------------------
# Criterion #1 -- eligibility gate
# ---------------------------------------------------------------------------


class TestSelectExemplars:
    def test_returns_empty_when_writer_has_no_samples(self, db):
        assert select_exemplars("absent-writer", k=3, conn=db) == []

    def test_returns_empty_when_writer_has_one_sample(self, db, png_factory):
        # SPEC #1: caller decides to skip when len < 2; select_exemplars
        # itself returns the 1 row -- it's not the gate's job.
        # We assert exact count so the consumer's gate is unambiguous.
        _seed_writer(db, writer_id="solo", n=1, png_factory=png_factory)
        assert len(select_exemplars("solo", k=3, conn=db)) == 1

    def test_returns_k_rows_when_available(self, db, png_factory):
        _seed_writer(db, writer_id="prolific", n=5, png_factory=png_factory)
        result = select_exemplars("prolific", k=3, conn=db)
        assert len(result) == 3
        assert all(isinstance(e, Exemplar) for e in result)
        assert all(e.ground_truth.startswith("sample-") for e in result)

    def test_caps_at_available_samples(self, db, png_factory):
        _seed_writer(db, writer_id="meager", n=2, png_factory=png_factory)
        result = select_exemplars("meager", k=5, conn=db)
        assert len(result) == 2

    def test_deterministic_order_by_sample_id(self, db, png_factory):
        ids = _seed_writer(db, writer_id="det", n=4, png_factory=png_factory)
        first = select_exemplars("det", k=3, conn=db)
        second = select_exemplars("det", k=3, conn=db)
        assert [e.sample_id for e in first] == [e.sample_id for e in second]
        # And the order matches the deterministic id-ascending tiebreak.
        assert [e.sample_id for e in first] == sorted(ids)[:3]

    def test_exclude_sample_id_filters_target_image(self, db, png_factory):
        ids = _seed_writer(db, writer_id="exclude", n=3, png_factory=png_factory)
        target_id = ids[1]
        result = select_exemplars("exclude", k=5, conn=db, exclude_sample_id=target_id)
        assert target_id not in [e.sample_id for e in result]
        assert len(result) == 2

    def test_k_zero_returns_empty(self, db, png_factory):
        _seed_writer(db, writer_id="zero", n=3, png_factory=png_factory)
        assert select_exemplars("zero", k=0, conn=db) == []

    def test_k_negative_returns_empty(self, db, png_factory):
        _seed_writer(db, writer_id="neg", n=3, png_factory=png_factory)
        assert select_exemplars("neg", k=-1, conn=db) == []

    def test_blank_writer_id_returns_empty(self, db, png_factory):
        _seed_writer(db, writer_id="something", n=2, png_factory=png_factory)
        assert select_exemplars("", k=3, conn=db) == []

    def test_does_not_leak_across_writers(self, db, png_factory):
        _seed_writer(db, writer_id="a", n=3, png_factory=png_factory)
        _seed_writer(db, writer_id="b", n=2, png_factory=png_factory)
        result_a = select_exemplars("a", k=10, conn=db)
        result_b = select_exemplars("b", k=10, conn=db)
        assert len(result_a) == 3
        assert len(result_b) == 2
        assert {e.sample_id for e in result_a}.isdisjoint(
            {e.sample_id for e in result_b}
        )


# ---------------------------------------------------------------------------
# Criterion #2 -- block layout: exemplars before target, each followed by label
# ---------------------------------------------------------------------------


class TestBuildExemplarBlocks:
    def test_layout_interleaves_image_text_pairs_then_target(self, png_factory):
        e1_path = png_factory(name="ex1.png")
        e2_path = png_factory(name="ex2.png")
        exemplars = [
            Exemplar(sample_id=1, image_path=e1_path, ground_truth="hello"),
            Exemplar(sample_id=2, image_path=e2_path, ground_truth="world"),
        ]
        blocks = build_exemplar_blocks(
            target_image_b64="TGT_B64",
            target_media_type="image/jpeg",
            exemplars=exemplars,
        )
        # Expected sequence: image, text, image, text, image (target).
        types = [b["type"] for b in blocks]
        assert types == ["image", "text", "image", "text", "image"]

    def test_target_is_last_block(self, png_factory):
        e_path = png_factory(name="ex.png")
        blocks = build_exemplar_blocks(
            target_image_b64="TGT_B64",
            target_media_type="image/png",
            exemplars=[
                Exemplar(sample_id=1, image_path=e_path, ground_truth="x"),
                Exemplar(sample_id=2, image_path=e_path, ground_truth="y"),
            ],
        )
        last = blocks[-1]
        assert last["type"] == "image"
        assert last["source"]["data"] == "TGT_B64"
        assert last["source"]["media_type"] == "image/png"

    def test_label_text_includes_ground_truth_quoted(self, png_factory):
        e_path = png_factory(name="ex.png")
        blocks = build_exemplar_blocks(
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            exemplars=[
                Exemplar(sample_id=1, image_path=e_path, ground_truth="quick fox")
            ],
        )
        # blocks: image, text, image -> the text block at index 1 carries GT.
        assert blocks[1]["type"] == "text"
        assert "«quick fox»" in blocks[1]["text"]

    def test_anti_cargo_cult_warning_present_in_label(self, png_factory):
        # SPEC § Risks: label must tell the model the reference text is from
        # the same writer but DIFFERENT TEXT, to avoid copy-prior-output.
        e_path = png_factory(name="ex.png")
        blocks = build_exemplar_blocks(
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            exemplars=[
                Exemplar(sample_id=1, image_path=e_path, ground_truth="abc")
            ],
        )
        label = blocks[1]["text"]
        assert "DIFFERENT TEXT" in label
        assert "do not repeat" in label.lower()

    def test_missing_exemplar_image_is_silently_skipped(self, png_factory):
        good = png_factory(name="ok.png")
        blocks = build_exemplar_blocks(
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            exemplars=[
                Exemplar(sample_id=1, image_path="/nonexistent/missing.png", ground_truth="x"),
                Exemplar(sample_id=2, image_path=good, ground_truth="ok"),
            ],
        )
        # missing skipped -> only 1 exemplar pair + target = 3 blocks
        types = [b["type"] for b in blocks]
        assert types == ["image", "text", "image"]
        assert "«ok»" in blocks[1]["text"]

    def test_all_exemplars_missing_collapses_to_target_only(self):
        blocks = build_exemplar_blocks(
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            exemplars=[
                Exemplar(sample_id=1, image_path="/no/where.png", ground_truth="x"),
                Exemplar(sample_id=2, image_path="/no/here.png", ground_truth="y"),
            ],
        )
        assert len(blocks) == 1
        assert blocks[0]["source"]["data"] == "TGT"


# ---------------------------------------------------------------------------
# Criterion #6 -- HE_FEW_SHOT_K env honored
# ---------------------------------------------------------------------------


class TestEnvFewShotK:
    def test_default_when_unset(self):
        assert env_few_shot_k({}) == DEFAULT_FEW_SHOT_K

    def test_zero_disables(self):
        assert env_few_shot_k({FEW_SHOT_K_ENV: "0"}) == 0

    def test_explicit_value_honored(self):
        assert env_few_shot_k({FEW_SHOT_K_ENV: "5"}) == 5

    def test_garbage_falls_back_to_default(self):
        assert env_few_shot_k({FEW_SHOT_K_ENV: "abc"}) == DEFAULT_FEW_SHOT_K

    def test_negative_falls_back_to_default(self):
        # Negative is "invalid", not "disabled" -- 0 disables.
        assert env_few_shot_k({FEW_SHOT_K_ENV: "-3"}) == DEFAULT_FEW_SHOT_K


# ---------------------------------------------------------------------------
# Criterion #5 -- TrOCR (and other non-allowlisted) passthrough
# ---------------------------------------------------------------------------


class TestProviderAllowlist:
    def test_claude_and_gemini_are_supported(self):
        assert provider_supports_exemplars("claude") is True
        assert provider_supports_exemplars("gemini") is True

    def test_trocr_is_not_supported(self):
        assert provider_supports_exemplars("trocr") is False

    def test_other_ocr_providers_default_to_unsupported(self):
        assert provider_supports_exemplars("paddleocr") is False
        assert provider_supports_exemplars("openai") is False

    def test_allowlist_membership(self):
        assert "claude" in EXEMPLAR_PROVIDERS
        assert "gemini" in EXEMPLAR_PROVIDERS
        assert "trocr" not in EXEMPLAR_PROVIDERS


# ---------------------------------------------------------------------------
# select_and_build_exemplar_blocks: covers the integration gates end-to-end
# ---------------------------------------------------------------------------


class TestSelectAndBuildOrchestrator:
    def test_returns_none_when_no_writer_id(self, db, png_factory):
        result = select_and_build_exemplar_blocks(
            writer_id=None,
            provider="claude",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
        )
        assert result is None

    def test_returns_none_for_trocr(self, db, png_factory):
        # Criterion #5: writer_id set but provider=trocr -> no exemplars,
        # no error.
        _seed_writer(db, writer_id="alice", n=3, png_factory=png_factory)
        result = select_and_build_exemplar_blocks(
            writer_id="alice",
            provider="trocr",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
        )
        assert result is None

    def test_returns_none_when_k_env_zero(self, db, png_factory):
        # Criterion #6: HE_FEW_SHOT_K=0 disables.
        _seed_writer(db, writer_id="alice", n=3, png_factory=png_factory)
        result = select_and_build_exemplar_blocks(
            writer_id="alice",
            provider="claude",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
            env={FEW_SHOT_K_ENV: "0"},
        )
        assert result is None

    def test_returns_none_when_writer_below_threshold(self, db, png_factory):
        # Criterion #4: <2 GT samples -> fall back to single-image read.
        _seed_writer(db, writer_id="cold", n=1, png_factory=png_factory)
        result = select_and_build_exemplar_blocks(
            writer_id="cold",
            provider="claude",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
            env={FEW_SHOT_K_ENV: "3"},
        )
        assert result is None

    def test_returns_blocks_when_eligible(self, db, png_factory):
        _seed_writer(db, writer_id="warm", n=4, png_factory=png_factory)
        result = select_and_build_exemplar_blocks(
            writer_id="warm",
            provider="gemini",
            target_image_b64="TGT_B64",
            target_media_type="image/jpeg",
            conn=db,
            env={FEW_SHOT_K_ENV: "3"},
        )
        assert result is not None
        # 3 exemplars selected -> 3*(image+text) + 1 target image = 7 blocks
        types = [b["type"] for b in result]
        assert types == ["image", "text", "image", "text", "image", "text", "image"]
        assert result[-1]["source"]["data"] == "TGT_B64"

    def test_explicit_k_argument_overrides_env(self, db, png_factory):
        _seed_writer(db, writer_id="cap", n=5, png_factory=png_factory)
        result = select_and_build_exemplar_blocks(
            writer_id="cap",
            provider="claude",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
            env={FEW_SHOT_K_ENV: "0"},  # would disable, but k= overrides
            k=2,
        )
        assert result is not None
        # 2 exemplars + 1 target = 5 blocks
        assert len(result) == 5

    def test_exclude_sample_id_propagates(self, db, png_factory):
        ids = _seed_writer(db, writer_id="excl", n=3, png_factory=png_factory)
        target_id = ids[0]
        result = select_and_build_exemplar_blocks(
            writer_id="excl",
            provider="claude",
            target_image_b64="TGT",
            target_media_type="image/jpeg",
            conn=db,
            env={FEW_SHOT_K_ENV: "5"},
            exclude_sample_id=target_id,
        )
        # 2 surviving exemplars + 1 target = 5 blocks
        assert result is not None
        assert len(result) == 5
