from __future__ import annotations

from dataclasses import replace

import polars as pl

from embodied_data_infra.cleaning.config import (
    ActionConfig,
    CleaningConfig,
    IdleConfig,
    SuccessConfig,
)
from embodied_data_infra.cleaning.rules import compute_metrics, decide_episodes


def _cfg(**kwargs) -> CleaningConfig:
    base = CleaningConfig(
        min_frames=2,
        max_frames=150,
        duration_outlier_factor=3.0,
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
            interior_idle_ratio_max=0.2,
            interior_idle_run_max=10,
        ),
    )
    return replace(base, **kwargs)


def _frames(episodes: list[dict], fps: int = 10) -> tuple[pl.DataFrame, pl.DataFrame]:
    frame_rows = []
    meta_rows = []
    global_index = 0
    for episode, spec in enumerate(episodes):
        actions = spec["actions"]
        states = spec["states"]
        n = len(actions)
        timestamps = spec.get("timestamps", [i / fps for i in range(n)])
        frame_indices = spec.get("frame_indices", list(range(n)))
        rewards = spec.get("rewards", [0.0] * (n - 1) + [1.0])
        dones = spec.get("dones", [False] * (n - 1) + [True])
        for i in range(n):
            frame_rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame_indices[i],
                    "timestamp": timestamps[i],
                    "action": [float(v) for v in actions[i]],
                    "observation.state": [float(v) for v in states[i]],
                    "next.reward": rewards[i],
                    "next.done": dones[i],
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
        meta_rows.append(
            {
                "episode_index": episode,
                "length": n,
                "dataset_from_index": global_index - n,
                "dataset_to_index": global_index,
                "tasks": ["pick cube"],
            }
        )
    frame_df = pl.DataFrame(
        {
            "episode_index": [r["episode_index"] for r in frame_rows],
            "frame_index": [r["frame_index"] for r in frame_rows],
            "timestamp": pl.Series([r["timestamp"] for r in frame_rows], dtype=pl.Float32),
            "action": pl.Series([r["action"] for r in frame_rows], dtype=pl.Array(pl.Float32, 3)),
            "observation.state": pl.Series(
                [r["observation.state"] for r in frame_rows], dtype=pl.Array(pl.Float32, 3)
            ),
            "next.reward": [r["next.reward"] for r in frame_rows],
            "next.done": [r["next.done"] for r in frame_rows],
            "index": [r["index"] for r in frame_rows],
            "task_index": [r["task_index"] for r in frame_rows],
        }
    )
    meta_df = pl.DataFrame(meta_rows)
    return frame_df, meta_df


def _run(episodes: list[dict], cfg: CleaningConfig | None = None) -> pl.DataFrame:
    cfg = cfg or _cfg()
    frames, meta = _frames(episodes)
    metrics = compute_metrics(frames, meta, cfg, 10, reward_key="next.reward", done_key="next.done")
    return decide_episodes(metrics, cfg)


def _moving(n: int) -> dict:
    return {
        "actions": [[0.5, 0.0, 1.0] for _ in range(n)],
        "states": [[0.01 * i, 0.0, 0.0] for i in range(n)],
    }


def test_leading_idle_is_trimmed():
    episodes = [
        {
            "actions": [[0.0, 0.0, 1.0]] * 4 + [[0.5, 0.0, 1.0]] * 4,
            "states": [[0.01 * i, 0.0, 0.0] for i in range(8)],
        }
    ]
    result = _run(episodes)
    row = result.row(0, named=True)
    assert row["decision"] == "trim"
    assert row["trim_head"] == 4
    assert row["n_frames_out"] == 4


def test_short_idle_below_threshold_is_kept():
    episodes = [
        {
            "actions": [[0.0, 0.0, 1.0]] * 2 + [[0.5, 0.0, 1.0]] * 4,
            "states": [[0.01 * i, 0.0, 0.0] for i in range(6)],
        }
    ]
    row = _run(episodes).row(0, named=True)
    assert row["decision"] == "keep"
    assert row["trim_head"] == 0


def test_invalid_gripper_drops_episode():
    spec = _moving(6)
    spec["actions"][3] = [0.5, 0.0, 0.0]
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "action_domain" in row["reasons"]


def test_missing_success_drops_episode():
    spec = _moving(6)
    spec["dones"] = [False] * 6
    spec["rewards"] = [0.0] * 6
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "success_missing" in row["reasons"]


def test_nan_state_drops_episode():
    spec = _moving(6)
    spec["states"][2] = [float("nan"), 0.0, 0.0]
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "nan_inf" in row["reasons"]


def test_timestamp_gap_drops_episode():
    spec = _moving(6)
    spec["timestamps"] = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6]
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "timestamp_grid" in row["reasons"]


def test_frame_index_gap_drops_episode():
    spec = _moving(6)
    spec["frame_indices"] = [0, 1, 3, 4, 5, 6]
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "frame_index_gap" in row["reasons"]


def test_duration_outlier_drops_long_episode():
    episodes = [_moving(5), _moving(5), _moving(40)]
    result = _run(episodes)
    long_row = result.filter(pl.col("episode_index") == 2).row(0, named=True)
    assert long_row["decision"] == "drop"
    assert "duration_outlier" in long_row["reasons"]


def test_interior_idle_drops_episode():
    spec = {
        "actions": [[0.5, 0.0, 1.0] for _ in range(12)],
        "states": [[0.0, 0.0, 0.0] for _ in range(12)],
    }
    row = _run([spec]).row(0, named=True)
    assert row["decision"] == "drop"
    assert "interior_idle_ratio" in row["reasons"]
    assert "interior_idle_run" in row["reasons"]
