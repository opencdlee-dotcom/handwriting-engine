"""Phase 9 / RPT-02 — composite-scored configuration recommendation.

Covers:
- Composite weights 70% CER / 15% cost / 15% stability.
- Min-max normalization within candidate set.
- Single-run candidates flagged n=1, given median stability.
- Empty DB / no measured CER → graceful messages.
- CLI: `benchmark recommend`.
"""

import os
import tempfile

from PIL import Image
from click.testing import CliRunner

from handwriting_engine.benchmark.db import (
    get_connection,
    insert_sample,
    insert_ground_truth,
    insert_run,
    finish_run,
    insert_provider_output,
    insert_eval_metric,
)
from handwriting_engine.benchmark.report import recommend_strategy
from handwriting_engine.cli import cli


def _seed_db_with_runs(db_path, configs):
    """configs: list of dicts with keys provider, strategy, cers (list[float]),
    cost_per_sample (float). Each config produces ONE run with len(cers) samples."""
    conn = get_connection(db_path)
    img_dir = tempfile.mkdtemp()
    sids, gids = [], []
    n = max(len(c["cers"]) for c in configs)
    for i in range(n):
        p = os.path.join(img_dir, f"i{i}.png")
        Image.new("RGB", (32, 32), (128, 128, 128)).save(p)
        sid = insert_sample(conn, p, f"hash{i}", student=f"w{i % 3}")
        gid = insert_ground_truth(conn, sid, "ground truth")
        sids.append(sid)
        gids.append(gid)

    run_ids = []
    for cfg in configs:
        rid = insert_run(
            conn, label=cfg.get("label", ""),
            providers=[cfg["provider"]], strategies=[cfg["strategy"]],
        )
        # Cost per sample is achieved by setting input/output tokens such
        # that estimate_cost yields the desired total. Simpler: set
        # input_tokens = total_input_tokens_for_run, then cost is computed
        # by report._aggregate_results. We'll just dial output_tokens.
        # Easier: we don't pin cost exactly here; tests inspect *relative*
        # ordering, not absolute $ values.
        for sid, gid, cer in zip(sids, gids, cfg["cers"]):
            po = insert_provider_output(
                conn, run_id=rid, sample_id=sid,
                provider=cfg["provider"], strategy=cfg["strategy"],
                output_text="x", confidence=0.9,
                input_tokens=cfg.get("input_tokens", 1000),
                output_tokens=cfg.get("output_tokens", 500),
            )
            insert_eval_metric(
                conn, provider_output_id=po, ground_truth_id=gid,
                cer=cer, wer=cer,
            )
        finish_run(conn, rid, len(cfg["cers"]))
        run_ids.append(rid)
    conn.close()
    return run_ids


# --- recommend_strategy() ---


class TestRecommendCore:
    def test_empty_db_message(self, tmp_path):
        db = tmp_path / "empty.db"
        # Initialize the DB but insert no runs.
        get_connection(db).close()
        out = recommend_strategy(db_path=db)
        assert "No runs" in out

    def test_lower_cer_wins_when_cost_equal(self, tmp_path):
        # Two candidates, equal cost, A has lower CER.
        db = tmp_path / "cer.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote", "cers": [0.05] * 10},
            {"provider": "claude", "strategy": "vote", "cers": [0.10] * 10},
        ])
        out = recommend_strategy(db_path=db)
        assert "Winner: gemini + vote" in out

    def test_lower_cost_wins_when_cer_equal(self, tmp_path):
        # Equal CER → composite reduces to cost + stability. Both candidates
        # have one run so stability is neutral; cost is the only differentiator.
        db = tmp_path / "cost.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote",
             "cers": [0.10] * 10, "input_tokens": 100, "output_tokens": 50},
            {"provider": "claude", "strategy": "vote",
             "cers": [0.10] * 10, "input_tokens": 1000, "output_tokens": 500},
        ])
        out = recommend_strategy(db_path=db)
        # Gemini is cheaper per-token AND uses fewer tokens → lower cost.
        assert "Winner: gemini + vote" in out

    def test_more_stable_wins_when_cer_and_cost_equal(self, tmp_path):
        # Same overall mean CER (0.10), same cost, different across-run
        # variance. Stable's three runs all average to 0.10. Wobbly's
        # three runs average to 0.05, 0.10, 0.15 — same overall mean,
        # but stdev across runs is much higher.
        db = tmp_path / "stab.db"
        stable_runs = [[0.10] * 10, [0.10] * 10, [0.10] * 10]
        wobbly_runs = [[0.05] * 10, [0.10] * 10, [0.15] * 10]
        configs = []
        for i, cers in enumerate(stable_runs):
            configs.append({
                "provider": "stable", "strategy": "vote",
                "cers": cers, "label": f"stable_{i}",
            })
        for i, cers in enumerate(wobbly_runs):
            configs.append({
                "provider": "wobbly", "strategy": "vote",
                "cers": cers, "label": f"wobbly_{i}",
            })
        _seed_db_with_runs(db, configs)
        out = recommend_strategy(db_path=db)
        # Same overall mean, same cost, lower across-run stdev → stable wins.
        assert "Winner: stable + vote" in out

    def test_single_run_flagged_n_1(self, tmp_path):
        db = tmp_path / "single.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote", "cers": [0.05] * 10},
            {"provider": "claude", "strategy": "vote", "cers": [0.10] * 10},
        ])
        out = recommend_strategy(db_path=db)
        # Both candidates have only one run → both should show n=1 marker
        # in the stdev column.
        assert "n=1" in out

    def test_score_ordering_descending(self, tmp_path):
        db = tmp_path / "order.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote", "cers": [0.05] * 10},
            {"provider": "claude", "strategy": "vote", "cers": [0.10] * 10},
            {"provider": "openai", "strategy": "vote", "cers": [0.15] * 10},
        ])
        out = recommend_strategy(db_path=db)
        # Gemini at rank 1, openai (worst) at rank 3.
        gemini_idx = out.find("gemini")
        openai_idx = out.find("openai")
        assert 0 < gemini_idx < openai_idx

    def test_winner_is_top_of_ranked_table(self, tmp_path):
        db = tmp_path / "header.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote", "cers": [0.05] * 10},
            {"provider": "claude", "strategy": "vote", "cers": [0.10] * 10},
        ])
        out = recommend_strategy(db_path=db)
        # The "Winner: …" line and the "1 …" rank line must reference the
        # same configuration.
        winner_line = next(line for line in out.split("\n") if line.startswith("  Winner:"))
        rank_one = next(line for line in out.split("\n") if line.startswith("1 "))
        assert "gemini" in winner_line and "vote" in winner_line
        assert "gemini" in rank_one and "vote" in rank_one


# --- CLI ---


class TestRecommendCli:
    def test_recommend_cmd_runs(self, tmp_path):
        db = tmp_path / "cli.db"
        _seed_db_with_runs(db, [
            {"provider": "gemini", "strategy": "vote", "cers": [0.05] * 10},
            {"provider": "claude", "strategy": "vote", "cers": [0.10] * 10},
        ])
        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "recommend",
            "--db-path", str(db),
        ])
        assert result.exit_code == 0, result.output
        assert "Winner:" in result.output

    def test_recommend_cmd_empty_db(self, tmp_path):
        db = tmp_path / "empty_cli.db"
        get_connection(db).close()
        runner = CliRunner()
        result = runner.invoke(cli, [
            "benchmark", "recommend",
            "--db-path", str(db),
        ])
        # No runs is a graceful empty state, not an error.
        assert result.exit_code == 0
        assert "No runs" in result.output
