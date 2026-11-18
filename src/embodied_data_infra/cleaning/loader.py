"""Polars loading of the raw frame table."""

from __future__ import annotations

import polars as pl

from .source import SourceDataset, SourceError

REQUIRED_FRAME_COLUMNS = {
    "episode_index",
    "frame_index",
    "timestamp",
    "action",
    "observation.state",
    "index",
    "task_index",
}


def load_frames(source: SourceDataset, *, reward_key: str, done_key: str) -> pl.DataFrame:
    """Load all data parquet files lazily, then collect one in-memory table.

    The demo dataset is 666 rows. Keeping this as a lazy scan makes the loader
    reusable for larger inputs; the cleaning rules themselves are in-memory.
    """
    frames = pl.concat(
        [pl.scan_parquet(path) for path in source.data_paths],
        how="vertical_relaxed",
    ).collect()

    missing = (REQUIRED_FRAME_COLUMNS | {reward_key, done_key}) - set(frames.columns)
    if missing:
        raise SourceError(f"Raw data is missing required column(s): {sorted(missing)}")

    action_dtype = frames.schema["action"]
    state_dtype = frames.schema["observation.state"]
    if not isinstance(action_dtype, pl.Array) or not isinstance(state_dtype, pl.Array):
        raise SourceError(
            "Expected action and observation.state to be fixed-size Array columns, got "
            f"{action_dtype!r} and {state_dtype!r}"
        )

    return frames.sort(["episode_index", "frame_index"])


def action_width(frames: pl.DataFrame) -> int:
    return frames.schema["action"].size


def state_width(frames: pl.DataFrame) -> int:
    return frames.schema["observation.state"].size
