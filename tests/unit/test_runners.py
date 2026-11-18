"""Unit tests for the collection and training stage wrappers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from embodied_data_infra.collection.runner import (
    CollectionError,
    build_record_command,
    load_json_config,
    run_record,
    validate_record_config,
)
from embodied_data_infra.training.runner import (
    TrainingError,
    build_train_command,
    resolve_latest_curated,
    run_train,
    validate_train_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_CONFIG = PROJECT_ROOT / "configs" / "collect" / "gym_hil_pick_cube.json"
TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train" / "sac_gym_hil_pick_cube.json"


def _make_curated_run(root: Path, name: str, mtime: float) -> Path:
    dataset = root / name / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")
    os.utime(root / name, (mtime, mtime))
    return dataset


def test_shipped_record_config_is_valid():
    config = load_json_config(RECORD_CONFIG)
    validate_record_config(config)
    assert config["dataset"]["repo_id"]


def test_shipped_train_config_is_valid():
    train_config = load_json_config(TRAIN_CONFIG)
    validate_train_config(train_config)
    assert train_config["policy"]["type"]


def test_record_config_rejects_missing_sections():
    with pytest.raises(CollectionError, match="'env'"):
        validate_record_config({"dataset": {"repo_id": "a/b"}})
    with pytest.raises(CollectionError, match="'dataset'"):
        validate_record_config({"env": {}})


def test_build_record_command_applies_overrides(tmp_path):
    command = build_record_command(
        RECORD_CONFIG,
        dataset_root=tmp_path,
        repo_id="user/new_repo",
        num_episodes=3,
        push_to_hub=False,
        extra_args=["--display_data", "true"],
    )
    joined = " ".join(command)
    assert "lerobot.scripts.lerobot_record" in joined
    assert str(RECORD_CONFIG) in joined
    assert f"--dataset.root {tmp_path}" in joined
    assert "--dataset.repo_id user/new_repo" in joined
    assert "--dataset.num_episodes 3" in joined
    assert "--dataset.push_to_hub false" in joined
    assert command[-2:] == ["--display_data", "true"]


def test_record_dry_run_does_not_spawn(monkeypatch):
    def _fail(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("subprocess.run must not be called during a dry run")

    monkeypatch.setattr("embodied_data_infra.collection.runner.subprocess.run", _fail)
    result = run_record(RECORD_CONFIG, dry_run=True)
    assert result.dry_run is True
    assert result.returncode == 0


def test_train_config_requires_policy():
    with pytest.raises(TrainingError, match="'policy'"):
        validate_train_config({"dataset": {}})


def test_build_train_command_applies_overrides(tmp_path):
    command = build_train_command(
        TRAIN_CONFIG,
        dataset_root=tmp_path,
        output_dir=tmp_path / "run",
        job_name="smoke",
        resume=True,
        extra_args=["--steps", "10"],
    )
    joined = " ".join(command)
    assert "lerobot.scripts.lerobot_train" in joined
    assert f"--dataset.root {tmp_path}" in joined
    assert f"--output_dir {tmp_path / 'run'}" in joined
    assert "--job_name smoke" in joined
    assert "--resume true" in joined
    assert command[-2:] == ["--steps", "10"]


def test_train_dry_run_does_not_spawn(monkeypatch, tmp_path):
    dataset = _make_curated_run(tmp_path, "run_a", mtime=1_700_000_000)

    def _fail(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("subprocess.run must not be called during a dry run")

    monkeypatch.setattr("embodied_data_infra.training.runner.subprocess.run", _fail)
    result = run_train(TRAIN_CONFIG, dataset_root=dataset, dry_run=True)
    assert result.dry_run is True
    assert result.returncode == 0


def test_resolve_latest_curated_picks_newest(tmp_path):
    _make_curated_run(tmp_path, "run_old", mtime=1_700_000_000)
    newest = _make_curated_run(tmp_path, "run_new", mtime=1_800_000_000)
    assert resolve_latest_curated(tmp_path) == newest


def test_resolve_latest_curated_requires_datasets(tmp_path):
    with pytest.raises(TrainingError, match="No curated datasets"):
        resolve_latest_curated(tmp_path)


def test_train_rejects_missing_dataset_root(tmp_path):
    with pytest.raises(TrainingError, match="meta/info.json"):
        run_train(TRAIN_CONFIG, dataset_root=tmp_path, dry_run=True)


def test_train_rejects_ambiguous_dataset_source(tmp_path):
    dataset = _make_curated_run(tmp_path, "run_a", mtime=1_700_000_000)
    with pytest.raises(TrainingError, match="not both"):
        run_train(
            TRAIN_CONFIG,
            dataset_root=dataset,
            latest_curated=True,
            curated_root=tmp_path,
            dry_run=True,
        )
