"""Tests for Stage 4: per-writer TrOCR fine-tuning.

Covers the new `fine-tune-writer` CLI subcommand, writer-specific checkpoint
auto-loading in TrOCRProvider, and the `get_provider("trocr", writer_id=...)`
caching contract. All torch/transformers calls are mocked so these run on a
vanilla install.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_ground_truth,
    insert_sample,
)
from handwriting_engine.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_writer(db_path: Path, writer_id: str, n: int, image_dir: Path) -> list[int]:
    """Insert N samples + ground truths under student=writer_id. Returns sample ids."""
    conn = get_connection(db_path)
    sids: list[int] = []
    try:
        for i in range(n):
            img = image_dir / f"{writer_id}_{i:02d}.png"
            Image.new("RGB", (200, 60), color=(240, 240, 240)).save(img)
            sid = insert_sample(
                conn,
                str(img),
                f"hash_{writer_id}_{i:02d}",
                student=writer_id,
                category="lab",
            )
            insert_ground_truth(conn, sid, f"line {i} for {writer_id}")
            sids.append(sid)
    finally:
        conn.close()
    return sids


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def stub_trocr_provider(monkeypatch):
    """Replace TrOCRProvider with a MagicMock so fine-tune is mockable.

    Returns the mock instance so tests can assert on call args. Patches the
    provider registry's cached singleton, NOT the import path — the CLI
    imports through `get_provider` which reads from the registry.
    """
    from handwriting_engine.providers import _INSTANCES, _WRITER_INSTANCES

    fake_provider = MagicMock()
    fake_provider.fine_tune_for_writer.return_value = Path(
        "/tmp/fake-writer-model"
    )
    # Pre-seed the cache so get_provider("trocr") returns our mock without
    # ever importing torch.
    _INSTANCES["trocr"] = fake_provider
    yield fake_provider
    _INSTANCES.pop("trocr", None)
    # Also clear writer-keyed cache so other tests don't see leftovers.
    keys_to_drop = [k for k in _WRITER_INSTANCES if k[0] == "trocr"]
    for k in keys_to_drop:
        _WRITER_INSTANCES.pop(k, None)


# ---------------------------------------------------------------------------
# CLI: insufficient pairs
# ---------------------------------------------------------------------------


class TestFineTuneWriterCli:
    def test_too_few_pairs_exits_nonzero_with_count(self, tmp_path, runner):
        """A writer with <5 ground-truthed pairs aborts and names the count."""
        db = tmp_path / "ft.db"
        _seed_writer(db, "alice", n=3, image_dir=tmp_path)

        result = runner.invoke(
            cli,
            ["fine-tune-writer", "alice", "--db-path", str(db)],
        )
        assert result.exit_code == 1, result.output
        # The error names how many it found AND the floor that wasn't met.
        assert "3" in result.output
        assert "5" in result.output
        assert "alice" in result.output

    def test_zero_pairs_exits_nonzero(self, tmp_path, runner):
        """No samples at all for the writer → still exits cleanly with count 0."""
        db = tmp_path / "ft_empty.db"
        # Create the db with schema but no rows for "ghost".
        get_connection(db).close()

        result = runner.invoke(
            cli,
            ["fine-tune-writer", "ghost", "--db-path", str(db)],
        )
        assert result.exit_code == 1, result.output
        assert "0" in result.output

    def test_custom_min_pairs_below_threshold(self, tmp_path, runner):
        """--min-pairs lower can let a smaller writer through."""
        db = tmp_path / "ft_min.db"
        _seed_writer(db, "bob", n=2, image_dir=tmp_path)

        # min-pairs=3 still too high
        result = runner.invoke(
            cli,
            ["fine-tune-writer", "bob", "--db-path", str(db), "--min-pairs", "3"],
        )
        assert result.exit_code == 1
        assert "2" in result.output
        assert "3" in result.output

    def test_enough_pairs_calls_fine_tune_with_right_args(
        self, tmp_path, runner, stub_trocr_provider,
    ):
        """≥5 pairs → fine_tune_for_writer is called with the matching args."""
        db = tmp_path / "ft_ok.db"
        _seed_writer(db, "carol", n=6, image_dir=tmp_path)

        result = runner.invoke(
            cli,
            [
                "fine-tune-writer", "carol",
                "--db-path", str(db),
                "--epochs", "2",
                "--lr", "1e-4",
            ],
        )
        assert result.exit_code == 0, result.output
        stub_trocr_provider.fine_tune_for_writer.assert_called_once()
        call_kwargs = stub_trocr_provider.fine_tune_for_writer.call_args.kwargs
        assert call_kwargs["writer_id"] == "carol"
        assert call_kwargs["epochs"] == 2
        assert call_kwargs["lr"] == pytest.approx(1e-4)
        # All 6 pairs forwarded as parallel lists.
        assert len(call_kwargs["line_images_b64"]) == 6
        assert len(call_kwargs["transcriptions"]) == 6
        assert all(t.startswith("line ") for t in call_kwargs["transcriptions"])
        # Confirm output names the saved checkpoint dir.
        assert "Checkpoint saved" in result.output

    def test_help_flag_works(self, runner):
        """Smoke: `fine-tune-writer --help` exits 0 and prints the option block."""
        result = runner.invoke(cli, ["fine-tune-writer", "--help"])
        assert result.exit_code == 0
        # Each documented flag should appear.
        assert "--epochs" in result.output
        assert "--min-pairs" in result.output
        assert "--lr" in result.output


# ---------------------------------------------------------------------------
# get_provider("trocr", writer_id=...) caching contract
# ---------------------------------------------------------------------------


class TestGetProviderWriterId:
    def test_writer_id_returns_distinct_instance(self):
        """Passing writer_id MUST NOT return the cached base singleton."""
        from handwriting_engine.providers import (
            _INSTANCES,
            _WRITER_INSTANCES,
            get_provider,
            register,
        )

        # Stub a registry entry we fully control; this isolates us from torch.
        class _Stub:
            name = "trocr_stub"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            @classmethod
            def is_available(cls):
                return True

        register("trocr_stub", _Stub)
        try:
            base = get_provider("trocr_stub")  # caches the bare singleton
            specific = get_provider("trocr_stub", writer_id="abc")
            assert base is not specific, (
                "writer_id must bypass the base singleton or per-writer "
                "checkpoints get silently discarded."
            )
            # writer_id is also stamped on the new instance.
            assert specific.kwargs == {"writer_id": "abc"}
            # Repeated lookups for the same writer_id reuse the cached instance.
            specific2 = get_provider("trocr_stub", writer_id="abc")
            assert specific is specific2
            # Different writer → different instance.
            other = get_provider("trocr_stub", writer_id="xyz")
            assert other is not specific
        finally:
            _INSTANCES.pop("trocr_stub", None)
            keys = [k for k in _WRITER_INSTANCES if k[0] == "trocr_stub"]
            for k in keys:
                _WRITER_INSTANCES.pop(k, None)
            # Pop the registry entry so other tests don't see it.
            from handwriting_engine.providers import _REGISTRY
            _REGISTRY.pop("trocr_stub", None)


# ---------------------------------------------------------------------------
# TrOCRProvider checkpoint auto-load smoke test
# ---------------------------------------------------------------------------


class TestTrOCRWriterCheckpointAutoLoad:
    def test_existing_checkpoint_loads_from_writer_dir(self, tmp_path, monkeypatch):
        """When ~/.handwriting-engine/writer-models/<writer_id>/ exists,
        TrOCRProvider must call from_pretrained on THAT path, not the base
        microsoft/trocr-base-handwritten name."""
        # Re-route WRITER_MODELS_DIR to a tmp location and create the dir.
        from handwriting_engine.providers import trocr_provider as tp_mod

        fake_root = tmp_path / "writer-models"
        ckpt_dir = fake_root / "alice"
        ckpt_dir.mkdir(parents=True)

        monkeypatch.setattr(tp_mod, "WRITER_MODELS_DIR", fake_root)

        # Stub torch + transformers BEFORE TrOCRProvider.__init__ imports them.
        fake_torch = types.ModuleType("torch")
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        fake_torch.no_grad = lambda: _NullCtx()  # noqa: E731
        sys.modules["torch"] = fake_torch

        fake_transformers = types.ModuleType("transformers")
        proc_loaded_with: list[str] = []
        model_loaded_with: list[str] = []

        class _FakeProcessor:
            @classmethod
            def from_pretrained(cls, name):
                proc_loaded_with.append(name)
                return cls()

        class _FakeModel:
            @classmethod
            def from_pretrained(cls, name):
                model_loaded_with.append(name)
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

        fake_transformers.TrOCRProcessor = _FakeProcessor
        fake_transformers.VisionEncoderDecoderModel = _FakeModel
        sys.modules["transformers"] = fake_transformers

        try:
            # Re-resolve: writer_checkpoint_dir uses module-level global.
            provider = tp_mod.TrOCRProvider(writer_id="alice")
            assert provider._writer_id == "alice"
            # The processor + model must have been fetched from the writer dir,
            # NOT the base model name.
            assert proc_loaded_with == [str(ckpt_dir)]
            assert model_loaded_with == [str(ckpt_dir)]
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("transformers", None)

    def test_missing_checkpoint_falls_back_to_base(self, tmp_path, monkeypatch):
        """When the writer dir is absent, TrOCRProvider must load the base
        model name (current default: microsoft/trocr-base-handwritten)."""
        from handwriting_engine.providers import trocr_provider as tp_mod

        # Point WRITER_MODELS_DIR somewhere empty.
        empty_root = tmp_path / "writer-models-empty"
        empty_root.mkdir()
        monkeypatch.setattr(tp_mod, "WRITER_MODELS_DIR", empty_root)

        fake_torch = types.ModuleType("torch")
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        fake_torch.no_grad = lambda: _NullCtx()  # noqa: E731
        sys.modules["torch"] = fake_torch

        fake_transformers = types.ModuleType("transformers")
        loaded_with: list[str] = []

        class _FakeProcessor:
            @classmethod
            def from_pretrained(cls, name):
                loaded_with.append(name)
                return cls()

        class _FakeModel:
            @classmethod
            def from_pretrained(cls, name):
                loaded_with.append(name)
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

        fake_transformers.TrOCRProcessor = _FakeProcessor
        fake_transformers.VisionEncoderDecoderModel = _FakeModel
        sys.modules["transformers"] = fake_transformers

        try:
            provider = tp_mod.TrOCRProvider(writer_id="ghost-writer")
            # Both calls must hit the base model name — NOT the (nonexistent)
            # writer dir.
            assert all(name == tp_mod.DEFAULT_TROCR_MODEL for name in loaded_with), (
                f"Expected base model load, got: {loaded_with}"
            )
            assert provider._writer_id == "ghost-writer"
        finally:
            sys.modules.pop("torch", None)
            sys.modules.pop("transformers", None)


class _NullCtx:
    """Minimal context-manager stub for `with torch.no_grad():` paths."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
