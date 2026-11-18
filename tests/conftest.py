"""Synthetic LeRobot fixtures; tests never need network access."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from embodied_data_infra.cleaning.config import (
    ActionConfig,
    CleaningConfig,
    ConversionConfig,
    IdleConfig,
    OutputConfig,
    PipelineConfig,
    PushConfig,
    SourceConfig,
    SuccessConfig,
)


def _image(
    episode: int, frame: int, width: int = 64, height: int = 64, offset: int = 0
) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    phase = (episode * 17 + frame * 3 + offset) % 32
    base = ((xx + yy + phase * 4) % 256).astype(np.uint8)
    rgb = np.stack([base, np.clip(base + 20, 0, 255).astype(np.uint8), base], axis=-1)
    return rgb


@pytest.fixture(scope="session")
def mini_raw_dataset(tmp_path_factory) -> Path:
    """3 episodes x 8/6/10 frames, two cameras, videos encoded by LeRobot.

    ep0: 4 leading idle frames, then a valid pick (should trim, keep).
    ep1: valid pick (keep).
    ep2: invalid gripper=0 at frame 3 (should drop).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path_factory.mktemp("mini_raw")
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": [3, 64, 64],
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [3, 64, 64],
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": [3], "names": None},
        "action": {"dtype": "float32", "shape": [3], "names": ["dx", "dy", "gripper"]},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "complementary_info.discrete_penalty": {"dtype": "float32", "shape": [1], "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="test/mini_raw",
        fps=10,
        features=features,
        root=root / "raw",
        robot_type="test",
        use_videos=True,
        rgb_encoder=None,
    )

    episode_lengths = {0: 8, 1: 6, 2: 10}
    for episode, length in episode_lengths.items():
        for frame in range(length):
            idle = episode == 0 and frame < 4
            action = np.array([0.0, 0.0, 1.0] if idle else [0.5, 0.0, 1.0], dtype=np.float32)
            if episode == 2 and frame == 3:
                action[2] = 0.0  # invalid gripper value
            if not idle and frame == length - 1:
                action[2] = 2.0  # close gripper at the end
            state = np.array(
                [0.001 * (frame if idle else frame + 1), float(episode), float(frame)],
                dtype=np.float32,
            )
            dataset.add_frame(
                {
                    "observation.images.front": _image(episode, frame, offset=0),
                    "observation.images.wrist": _image(episode, frame, offset=7),
                    "observation.state": state,
                    "action": action,
                    "next.reward": np.array(
                        [1.0 if frame == length - 1 else 0.0], dtype=np.float32
                    ),
                    "next.done": np.array([frame == length - 1], dtype=np.bool_),
                    "complementary_info.discrete_penalty": np.array([-0.02], dtype=np.float32),
                    "task": "pick cube",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root / "raw"


@pytest.fixture
def mini_config(tmp_path, mini_raw_dataset) -> PipelineConfig:
    return PipelineConfig(
        source=SourceConfig(
            repo_id="test/mini_raw",
            revision="test",
            local_root=mini_raw_dataset,
            download_dir=tmp_path / "raw_cache",
            hash_videos=True,
        ),
        cleaning=CleaningConfig(
            fps_tolerance_s=1.0e-3,
            fail_on_nan=True,
            min_frames=2,
            max_frames=50,
            duration_outlier_factor=3.0,
            clip_crf=30,
            success=SuccessConfig(require=True),
            action=ActionConfig(
                bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
                allowed_values={2: (1.0, 2.0)},
                atol=1.0e-5,
            ),
            idle=IdleConfig(
                state_motion_dims=(0, 1, 2),
                action_translation_dims=(0, 1),
                motion_eps=1.0e-3,
                action_eps=1.0e-6,
                trim_leading=True,
                trim_trailing=True,
                min_trim_frames=3,
                max_trim_ratio=0.8,
                interior_idle_ratio_max=0.5,
                interior_idle_run_max=100,
            ),
        ),
        output=OutputConfig(
            clean_root=tmp_path / "cleaned",
            curated_root=tmp_path / "curated",
            report_root=tmp_path / "reports",
            repo_id="test/mini_clean",
            overwrite=False,
        ),
        conversion=ConversionConfig(
            parallel_encoding=False,
            validate_after=True,
            video_corr_min=0.5,
            video_mae_max=30.0,
        ),
        push=PushConfig(enabled=False, private=True),
    )
