from __future__ import annotations

import json

import polars as pl

from embodied_data_infra.cleaning.cleaning import run_clean
from embodied_data_infra.cleaning.validation import validate_clean_contract


def _assert_expected_clean(result):
    assert result.kept_episodes == 2
    assert result.kept_frames == 10
    decisions = {int(row["episode_index"]): row for row in result.metrics.iter_rows(named=True)}
    assert decisions[0]["decision"] == "trim"
    assert decisions[0]["trim_head"] == 4
    assert decisions[0]["n_frames_out"] == 4
    assert decisions[1]["decision"] == "keep"
    assert decisions[2]["decision"] == "drop"
    assert "action_domain" in decisions[2]["reasons"]


def test_clean_stage_materializes_contract(mini_config):
    result = run_clean(mini_config)
    _assert_expected_clean(result)

    root = result.final_dir
    assert (root / "data.parquet").is_file()
    assert (root / "manifest.parquet").is_file()
    assert (root / "selection.parquet").is_file()
    assert (root / "source_schema.json").is_file()
    assert (root / "tasks.parquet").is_file()
    assert (root / "run.json").is_file()
    assert (root / "qc" / "report.md").is_file()
    assert (root / "videos" / "observation.images.front" / "episode_00000.mp4").is_file()
    assert (root / "videos" / "observation.images.wrist" / "episode_00000.mp4").is_file()
    assert not (root / "videos" / "observation.images.front" / "episode_00002.mp4").exists()

    data = pl.read_parquet(root / "data.parquet")
    manifest = pl.read_parquet(root / "manifest.parquet")
    selection = pl.read_parquet(root / "selection.parquet")
    assert data.height == 10
    assert manifest.height == 3
    assert selection.height == 24
    assert int(selection["kept"].sum()) == 10

    run = json.loads((root / "run.json").read_text())
    assert run["stage"] == "clean"
    assert run["is_lerobot_dataset"] is False
    assert run["counts"]["output_episodes"] == 2
    assert run["counts"]["output_frames"] == 10
    assert run["output_fingerprint"]["digest"]

    layer = validate_clean_contract(root)
    assert layer.kept_episodes == [0, 1]


def test_dry_run_writes_no_materialized_layer(mini_config):
    result = run_clean(mini_config, dry_run=True)
    _assert_expected_clean(result)
    assert result.final_dir is None
    assert not mini_config.output.clean_root.exists()
    assert (result.report_dir / "summary.json").is_file()
    assert (result.report_dir / "proposed_manifest.parquet").is_file()
    assert (result.report_dir / "proposed_selection.parquet").is_file()


def test_clean_is_idempotent(mini_config):
    first = run_clean(mini_config)
    second = run_clean(mini_config)
    assert second.reused is True
    assert second.final_dir == first.final_dir


def test_clean_overwrite_rebuilds(mini_config):
    first = run_clean(mini_config)
    second = run_clean(mini_config, overwrite=True)
    assert second.reused is False
    assert second.final_dir == first.final_dir


def test_contract_rejects_tampered_clip(mini_config):
    import pytest

    from embodied_data_infra.cleaning.validation import ValidationError

    result = run_clean(mini_config)
    clip = result.final_dir / "videos" / "observation.images.front" / "episode_00000.mp4"
    with clip.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        validate_clean_contract(result.final_dir)
