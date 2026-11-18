from __future__ import annotations

import json

from embodied_data_infra.cleaning.cleaning import run_clean
from embodied_data_infra.cleaning.conversion import run_convert


def test_clean_convert_validate_roundtrip(mini_config):
    clean_result = run_clean(mini_config)
    assert clean_result.kept_episodes == 2
    assert clean_result.kept_frames == 10

    convert_result = run_convert(mini_config)
    assert convert_result.validation["passed"] is True
    assert convert_result.dataset_dir.is_dir()
    assert (convert_result.dataset_dir / "meta" / "info.json").is_file()
    assert (convert_result.dataset_dir / "data" / "chunk-000" / "file-000.parquet").is_file()
    assert (convert_result.dataset_dir / "videos" / "observation.images.front").is_dir()

    run = json.loads((convert_result.final_dir / "run.json").read_text())
    assert run["stage"] == "convert"
    assert run["is_lerobot_dataset"] is True
    assert run["counts"]["episodes"] == 2
    assert run["counts"]["frames"] == 10
    assert run["validation"]["passed"] is True

    conversion = json.loads((convert_result.final_dir / "conversion.json").read_text())
    assert conversion["episode_mapping"] == [
        {"source_episode_index": 0, "curated_episode_index": 0, "frames": 4},
        {"source_episode_index": 1, "curated_episode_index": 1, "frames": 6},
    ]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=mini_config.output.repo_id,
        root=convert_result.dataset_dir,
        video_backend="pyav",
    )
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 10
    assert len(dataset) == 10
    assert [dataset.meta.episodes[i]["length"] for i in range(2)] == [4, 6]
    item = dataset[0]
    assert item["observation.state"].shape == (3,)
    assert item["action"].shape == (3,)
    assert item["observation.images.front"].shape == (3, 64, 64)


def test_convert_is_idempotent(mini_config):
    run_clean(mini_config)
    first = run_convert(mini_config)
    second = run_convert(mini_config)
    assert second.reused is True
    assert second.final_dir == first.final_dir


def test_convert_overwrite_rebuilds(mini_config):
    run_clean(mini_config)
    first = run_convert(mini_config)
    second = run_convert(mini_config, overwrite=True)
    assert second.reused is False
    assert second.final_dir == first.final_dir
