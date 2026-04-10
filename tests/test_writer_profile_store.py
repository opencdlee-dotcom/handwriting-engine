"""Tests for WriterProfileStore."""
import pytest
from handwriting_engine.writer_profile_store import WriterProfileStore


def test_save_and_load_round_trip(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    profile = {"crosses_sevens": True, "open_four": False, "connects_letters": True}
    store.save("test_writer", profile)
    loaded = store.load("test_writer")
    assert loaded["crosses_sevens"] is True
    assert loaded["open_four"] is False
    assert loaded["writer_id"] == "test_writer"


def test_load_nonexistent_returns_none(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    assert store.load("no_such_writer") is None


def test_delete(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("writer_a", {"crosses_sevens": False})
    assert store.delete("writer_a") is True
    assert store.load("writer_a") is None
    assert store.delete("writer_a") is False


def test_list_writers(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("alice", {"crosses_sevens": True})
    store.save("bob", {"crosses_sevens": False})
    writers = store.list_writers()
    assert "alice" in writers
    assert "bob" in writers


def test_build_calibration_block_with_full_profile(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    profile = {
        "crosses_sevens": True,
        "open_four": True,
        "connects_letters": False,
        "confusion_resolutions": {"u/v": "v", "1/l": "1"},
    }
    block = store.build_calibration_block(profile)
    assert "7s" in block
    assert "4s" in block
    assert "printed" in block
    assert "u/v" in block
    assert "v" in block


def test_build_calibration_block_empty(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    assert store.build_calibration_block({}) == ""


def test_writer_id_sanitization(tmp_path):
    store = WriterProfileStore(profiles_dir=tmp_path)
    store.save("writer/with/slashes", {"crosses_sevens": True})
    # Slashes are stripped — should not create subdirectories
    writers = store.list_writers()
    assert any("writer" in w for w in writers)
