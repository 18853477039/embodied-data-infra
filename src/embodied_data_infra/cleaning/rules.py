"""Frame-level metrics and episode-level cleaning decisions.

All heavy lifting is done with Polars expressions. Decisions are a small,
explicit policy layer on top of the computed metrics so every drop/trim is
explainable in the QC report.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .config import CleaningConfig, PipelineConfig
from .loader import action_width, state_width
from .source import SourceError


def validate_widths(cfg: PipelineConfig, frames: pl.DataFrame) -> None:
    a_width = action_width(frames)
    s_width = state_width(frames)
    bounds = cfg.cleaning.action.bounds
    if len(bounds) != a_width:
        raise SourceError(
            f"action.bounds has {len(bounds)} dimensions but action is {a_width}-dimensional"
        )
    for dim in cfg.cleaning.action.allowed_values:
        if dim >= a_width:
            raise SourceError(
                f"action.allowed_values references dim {dim}, action width is {a_width}"
            )
    for dim in cfg.cleaning.idle.state_motion_dims:
        if dim >= s_width:
            raise SourceError(
                f"idle.state_motion_dims references dim {dim}, state width is {s_width}"
            )
    for dim in cfg.cleaning.idle.action_translation_dims:
        if dim >= a_width:
            raise SourceError(
                f"idle.action_translation_dims references dim {dim}, action width is {a_width}"
            )


def _nonfinite_mask(column: str, width: int) -> pl.Expr:
    return pl.any_horizontal(
        [
            pl.col(column).arr.get(dim).is_nan() | pl.col(column).arr.get(dim).is_infinite()
            for dim in range(width)
        ]
    )


def add_frame_metrics(
    frames: pl.DataFrame, cfg: CleaningConfig, fps: int, *, reward_key: str, done_key: str
) -> pl.DataFrame:
    """Add derived per-frame columns used by the episode rules."""
    a_width = action_width(frames)
    s_width = state_width(frames)

    state_delta_parts = [
        (
            pl.col("observation.state")
            .arr.get(dim)
            .cast(pl.Float64)
            .diff()
            .abs()
            .fill_null(0.0)
            .over("episode_index")
        )
        for dim in cfg.idle.state_motion_dims
    ]
    state_delta = pl.sum_horizontal(state_delta_parts).alias("state_delta")

    noop = pl.all_horizontal(
        [
            pl.col("action").arr.get(dim).abs() < cfg.idle.action_eps
            for dim in cfg.idle.action_translation_dims
        ]
    ).alias("is_noop")

    domain_parts: list[pl.Expr] = []
    for dim, (low, high) in enumerate(cfg.action.bounds):
        value = pl.col("action").arr.get(dim)
        domain_parts.append((value >= low - cfg.action.atol) & (value <= high + cfg.action.atol))
    for dim, allowed in cfg.action.allowed_values.items():
        value = pl.col("action").arr.get(dim)
        domain_parts.append(
            pl.any_horizontal(
                [(value - allowed_value).abs() <= cfg.action.atol for allowed_value in allowed]
            )
        )
    action_domain_ok = pl.all_horizontal(domain_parts).alias("action_domain_ok")

    step = 1.0 / fps
    timestamp_ok = (
        (
            (pl.col("timestamp").diff().over("episode_index") - step).abs() <= cfg.fps_tolerance_s
        ).fill_null(True)
        & (pl.col("timestamp").first().over("episode_index").abs() <= cfg.fps_tolerance_s)
    ).alias("timestamp_ok")
    frame_index_ok = (
        (pl.col("frame_index").diff().over("episode_index") == 1).fill_null(True)
    ).alias("frame_index_ok")

    nan_inf = (
        _nonfinite_mask("action", a_width) | _nonfinite_mask("observation.state", s_width)
    ).alias("nan_inf")

    frames = frames.with_columns(
        [
            state_delta,
            noop,
            action_domain_ok,
            timestamp_ok,
            frame_index_ok,
            nan_inf,
        ]
    )
    frames = frames.with_columns(
        [(pl.col("state_delta") < cfg.idle.motion_eps).alias("is_motionless")]
    )
    return frames.with_columns(
        [
            (
                ~pl.col("nan_inf")
                & pl.col("action_domain_ok")
                & pl.col("timestamp_ok")
                & pl.col("frame_index_ok")
            ).alias("frame_ok")
        ]
    )


def _aggregate_frames(frames: pl.DataFrame, *, reward_key: str, done_key: str) -> pl.DataFrame:
    return frames.group_by("episode_index").agg(
        [
            pl.len().alias("n_frames"),
            pl.col("frame_index").min().alias("first_frame_index"),
            pl.col("frame_index").max().alias("last_frame_index"),
            pl.col("timestamp").min().alias("first_timestamp"),
            pl.col("timestamp").max().alias("last_timestamp"),
            pl.col(reward_key).last().alias("last_reward"),
            pl.col(done_key).last().alias("last_done"),
            (~pl.col("frame_ok")).sum().alias("invalid_frame_count"),
            pl.col("nan_inf").sum().alias("nan_inf_count"),
            (~pl.col("action_domain_ok")).sum().alias("action_domain_violations"),
            (~pl.col("timestamp_ok")).sum().alias("timestamp_violations"),
            (~pl.col("frame_index_ok")).sum().alias("frame_index_violations"),
            (pl.col("frame_index") == 0).sum().alias("zero_frame_count"),
            pl.when(~pl.col("is_noop"))
            .then(pl.col("frame_index"))
            .otherwise(None)
            .min()
            .alias("first_active_frame"),
            pl.when(~pl.col("is_noop"))
            .then(pl.col("frame_index"))
            .otherwise(None)
            .max()
            .alias("last_active_frame"),
            pl.col("is_motionless").sum().alias("motionless_total"),
        ]
    )


def _add_trim_candidates(metrics: pl.DataFrame, cfg: CleaningConfig) -> pl.DataFrame:
    idle = cfg.idle
    return (
        metrics.with_columns(
            [
                pl.when(
                    pl.lit(idle.trim_leading)
                    & (pl.col("leading_noop") >= idle.min_trim_frames)
                    & (pl.col("leading_noop") < pl.col("n_frames"))
                )
                .then(pl.col("leading_noop"))
                .otherwise(0)
                .alias("_head_candidate"),
                pl.when(
                    pl.lit(idle.trim_trailing)
                    & (pl.col("trailing_noop") >= idle.min_trim_frames)
                    & (pl.col("trailing_noop") < pl.col("n_frames"))
                )
                .then(pl.col("trailing_noop"))
                .otherwise(0)
                .alias("_tail_candidate"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("_head_candidate") + pl.col("_tail_candidate")
                    > pl.col("n_frames") - cfg.min_frames
                ).alias("trim_would_be_too_short"),
                (
                    (pl.col("_head_candidate") + pl.col("_tail_candidate")) / pl.col("n_frames")
                    > idle.max_trim_ratio
                ).alias("trim_skipped_large"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("trim_would_be_too_short") | pl.col("trim_skipped_large"))
                .then(0)
                .otherwise(pl.col("_head_candidate"))
                .alias("trim_head"),
                pl.when(pl.col("trim_would_be_too_short") | pl.col("trim_skipped_large"))
                .then(0)
                .otherwise(pl.col("_tail_candidate"))
                .alias("trim_tail"),
            ]
        )
    )


def _interior_idle(metrics: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    trims = metrics.select(["episode_index", "n_frames", "trim_head", "trim_tail"])
    interior = (
        frames.select(["episode_index", "frame_index", "is_motionless"])
        .join(trims, on="episode_index", how="left")
        .filter(
            (pl.col("frame_index") >= pl.col("trim_head"))
            & (pl.col("frame_index") < pl.col("n_frames") - pl.col("trim_tail"))
        )
    )

    base = interior.group_by("episode_index").agg(
        [
            pl.len().alias("interior_frames"),
            pl.col("is_motionless").sum().alias("interior_motionless"),
        ]
    )
    base = base.with_columns(
        pl.when(pl.col("interior_frames") > 0)
        .then(pl.col("interior_motionless") / pl.col("interior_frames"))
        .otherwise(0.0)
        .alias("interior_idle_ratio")
    )

    runs = (
        interior.with_columns(
            (
                (pl.col("is_motionless") != pl.col("is_motionless").shift(1).fill_null(False))
                .cum_sum()
                .over("episode_index")
            ).alias("_run_id")
        )
        .filter(pl.col("is_motionless"))
        .group_by(["episode_index", "_run_id"])
        .agg(pl.len().alias("_run_len"))
        .group_by("episode_index")
        .agg(pl.col("_run_len").max().alias("max_interior_idle_run"))
    )

    base = base.join(runs, on="episode_index", how="left").with_columns(
        pl.col("max_interior_idle_run").fill_null(0)
    )
    return metrics.join(base, on="episode_index", how="left").with_columns(
        [
            pl.col("interior_frames").fill_null(0),
            pl.col("interior_motionless").fill_null(0),
            pl.col("interior_idle_ratio").fill_null(0.0),
            pl.col("max_interior_idle_run").fill_null(0),
        ]
    )


def compute_metrics(
    frames: pl.DataFrame,
    episodes_meta: pl.DataFrame,
    cfg: CleaningConfig,
    fps: int,
    *,
    reward_key: str,
    done_key: str,
) -> pl.DataFrame:
    """Compute one metrics row per episode, including interior idle after trims."""
    frames = add_frame_metrics(frames, cfg, fps, reward_key=reward_key, done_key=done_key)
    metrics = _aggregate_frames(frames, reward_key=reward_key, done_key=done_key)

    metrics = metrics.with_columns(
        [
            pl.col("first_active_frame").fill_null(pl.col("n_frames")).alias("leading_noop"),
            pl.when(pl.col("last_active_frame").is_null())
            .then(pl.col("n_frames"))
            .otherwise(pl.col("n_frames") - 1 - pl.col("last_active_frame"))
            .alias("trailing_noop"),
        ]
    ).with_columns(
        [
            (pl.col("n_frames") / fps).alias("duration_s"),
            (pl.col("last_done").fill_null(False) & (pl.col("last_reward") > 0)).alias("success"),
            (pl.col("leading_noop") == pl.col("n_frames")).alias("all_idle"),
        ]
    )

    meta = episodes_meta.select(
        [
            col
            for col in [
                "episode_index",
                "length",
                "dataset_from_index",
                "dataset_to_index",
                "tasks",
            ]
            if col in episodes_meta.columns
        ]
    ).rename({"length": "meta_length"})
    metrics = metrics.join(meta, on="episode_index", how="left")
    if "tasks" in metrics.columns:
        metrics = metrics.with_columns(pl.col("tasks").list.first().alias("task"))
    else:
        metrics = metrics.with_columns(pl.lit(None, dtype=pl.String).alias("task"))
    metrics = metrics.with_columns(
        (pl.col("n_frames") == pl.col("meta_length")).fill_null(False).alias("meta_length_match")
    )

    median_n = float(metrics["n_frames"].median() or 0.0)
    metrics = metrics.with_columns(
        pl.lit(median_n).alias("median_n_frames"),
        (pl.col("n_frames") > cfg.duration_outlier_factor * median_n).alias("duration_outlier"),
    )

    metrics = _add_trim_candidates(metrics, cfg)
    metrics = _interior_idle(metrics, frames)
    return metrics.sort("episode_index")


def decide_episodes(metrics: pl.DataFrame, cfg: CleaningConfig) -> pl.DataFrame:
    """Apply the hard-drop / trim / keep policy and attach human-readable reasons."""
    rows: list[dict[str, Any]] = []
    for row in metrics.iter_rows(named=True):
        reasons: list[str] = []
        warnings: list[str] = []

        if row["nan_inf_count"] > 0 and cfg.fail_on_nan:
            reasons.append("nan_inf")
        if row["timestamp_violations"] > 0:
            reasons.append("timestamp_grid")
        if row["frame_index_violations"] > 0 or row["zero_frame_count"] != 1:
            reasons.append("frame_index_gap")
        if not row["meta_length_match"]:
            reasons.append("meta_length_mismatch")
        if cfg.success.require and not row["success"]:
            reasons.append("success_missing")
        if row["action_domain_violations"] > 0:
            reasons.append("action_domain")
        if row["n_frames"] < cfg.min_frames:
            reasons.append("too_short")
        if row["n_frames"] > cfg.max_frames:
            reasons.append("too_long")
        if row["duration_outlier"]:
            reasons.append("duration_outlier")
        if row["all_idle"]:
            reasons.append("all_idle")
        if not row["all_idle"] and row["interior_idle_ratio"] > cfg.idle.interior_idle_ratio_max:
            reasons.append("interior_idle_ratio")
        if not row["all_idle"] and row["max_interior_idle_run"] > cfg.idle.interior_idle_run_max:
            reasons.append("interior_idle_run")

        if row["n_frames"] > 1.5 * row["median_n_frames"] and not row["duration_outlier"]:
            warnings.append("long_but_valid")
        if row["trim_would_be_too_short"]:
            warnings.append("trim_would_be_too_short")
        if row["trim_skipped_large"]:
            warnings.append("trim_skipped_large")

        decision = (
            "drop" if reasons else ("trim" if row["trim_head"] + row["trim_tail"] > 0 else "keep")
        )
        n_frames_out = (
            0 if decision == "drop" else row["n_frames"] - row["trim_head"] - row["trim_tail"]
        )

        rows.append(
            {
                "episode_index": row["episode_index"],
                "decision": decision,
                "reasons": reasons,
                "warnings": warnings,
                "n_frames_out": n_frames_out,
            }
        )

    decisions = pl.DataFrame(
        rows,
        schema={
            "episode_index": pl.Int64,
            "decision": pl.String,
            "reasons": pl.List(pl.String),
            "warnings": pl.List(pl.String),
            "n_frames_out": pl.Int64,
        },
    )
    return metrics.join(decisions, on="episode_index", how="left")
