"""Stage 1: raw LeRobot v3 dataset -> self-contained cleaned layer."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from . import CLEAN_CONTRACT_VERSION, PIPELINE_NAME, __version__
from .config import PipelineConfig
from .layout import (
    artifact_fingerprint,
    assert_replaceable,
    safe_remove_previous,
    staging_dir,
    write_json,
)
from .loader import load_frames
from .media import build_encoder_configs, encode_clip, load_episode_frames
from .report import rule_hit_counts, write_qc
from .rules import compute_metrics, decide_episodes, validate_widths
from .source import SourceError, load_source, sha256_file

logger = logging.getLogger(__name__)


class CleanError(RuntimeError):
    """Raised when the cleaning stage cannot produce a valid cleaned layer."""


@dataclass
class CleanResult:
    run_key: str
    report_dir: Path
    metrics: pl.DataFrame
    source_fp: str
    final_dir: Path | None
    dry_run: bool
    reused: bool = False

    @property
    def kept_episodes(self) -> int:
        return self.metrics.filter(pl.col("decision") != "drop").height

    @property
    def kept_frames(self) -> int:
        return int(self.metrics.filter(pl.col("decision") != "drop")["n_frames_out"].sum())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tool_versions() -> dict[str, str]:
    import av
    import polars
    import pyarrow

    versions = {
        "python": os.sys.version.split()[0],
        "polars": polars.__version__,
        "pyarrow": pyarrow.__version__,
        "av": av.__version__,
    }
    try:
        import lerobot

        versions["lerobot"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        versions["lerobot"] = "not-installed"
    return versions


def _build_selection(frames: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    kept_episodes = (
        metrics.filter(pl.col("decision") != "drop")
        .sort("episode_index")["episode_index"]
        .to_list()
    )
    curated_map = pl.DataFrame(
        {
            "episode_index": kept_episodes,
            "curated_episode_index": list(range(len(kept_episodes))),
        },
        schema={"episode_index": pl.Int64, "curated_episode_index": pl.Int64},
    )

    selected = frames.join(
        metrics.select(
            [
                "episode_index",
                "decision",
                "reasons",
                "warnings",
                "trim_head",
                "trim_tail",
                "n_frames",
            ]
        ),
        on="episode_index",
        how="left",
        maintain_order="left",
    )
    selected = selected.with_columns(
        [
            (pl.col("decision") != "drop").alias("_episode_kept"),
            (pl.col("frame_index") >= pl.col("trim_head")).alias("_after_head"),
            (pl.col("frame_index") < pl.col("n_frames") - pl.col("trim_tail")).alias(
                "_before_tail"
            ),
        ]
    ).with_columns(
        [(pl.col("_episode_kept") & pl.col("_after_head") & pl.col("_before_tail")).alias("kept")]
    )
    selected = selected.with_columns(
        [
            pl.when(pl.col("kept"))
            .then(pl.col("frame_index") - pl.col("trim_head"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("clip_frame_index")
        ]
    )
    selected = selected.join(
        curated_map, on="episode_index", how="left", maintain_order="left"
    ).with_columns(
        [
            pl.col("clip_frame_index").alias("curated_frame_index"),
        ]
    )
    return selected.select(
        [
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "kept",
            "decision",
            "reasons",
            "warnings",
            "trim_head",
            "trim_tail",
            "clip_frame_index",
            "curated_episode_index",
            "curated_frame_index",
        ]
    )


def _build_manifest(
    metrics: pl.DataFrame,
    source_episodes: pl.DataFrame,
    video_keys: list[str],
    clips: dict[tuple[int, str], dict],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    meta_by_episode = {
        int(row["episode_index"]): row for row in source_episodes.iter_rows(named=True)
    }

    clean_cursor = 0
    offsets: dict[int, tuple[int, int]] = {}
    curated_map: dict[int, int] = {}
    for new_index, row in enumerate(
        metrics.filter(pl.col("decision") != "drop").sort("episode_index").iter_rows(named=True)
    ):
        episode = int(row["episode_index"])
        n_out = int(row["n_frames_out"])
        offsets[episode] = (clean_cursor, clean_cursor + n_out)
        curated_map[episode] = new_index
        clean_cursor += n_out

    for row in metrics.sort("episode_index").iter_rows(named=True):
        episode = int(row["episode_index"])
        meta = meta_by_episode.get(episode, {})
        start, stop = offsets.get(episode, (None, None))
        record: dict[str, Any] = {
            "episode_index": episode,
            "curated_episode_index": curated_map.get(episode),
            "task": row.get("task"),
            "decision": row["decision"],
            "reasons": row["reasons"] or [],
            "warnings": row["warnings"] or [],
            "success": row["success"],
            "n_frames_in": int(row["n_frames"]),
            "trim_head": int(row["trim_head"]),
            "trim_tail": int(row["trim_tail"]),
            "n_frames_out": int(row["n_frames_out"]),
            "duration_in_s": float(row["duration_s"]),
            "duration_out_s": float(row["duration_out_s"]),
            "median_n_frames": float(row["median_n_frames"]),
            "duration_outlier": bool(row["duration_outlier"]),
            "action_domain_violations": int(row["action_domain_violations"]),
            "interior_idle_ratio": float(row["interior_idle_ratio"]),
            "max_interior_idle_run": int(row["max_interior_idle_run"]),
            "leading_noop": int(row["leading_noop"]),
            "trailing_noop": int(row["trailing_noop"]),
            "meta_length": int(row["meta_length"]) if row["meta_length"] is not None else None,
            "dataset_from_index": int(meta["dataset_from_index"])
            if "dataset_from_index" in meta
            else None,
            "dataset_to_index": int(meta["dataset_to_index"])
            if "dataset_to_index" in meta
            else None,
            "clean_row_start": start,
            "clean_row_stop": stop,
        }
        for camera in video_keys:
            clip = clips.get((episode, camera))
            record[f"videos/{camera}/source_chunk_index"] = (
                int(meta[f"videos/{camera}/chunk_index"])
                if f"videos/{camera}/chunk_index" in meta
                else None
            )
            record[f"videos/{camera}/source_file_index"] = (
                int(meta[f"videos/{camera}/file_index"])
                if f"videos/{camera}/file_index" in meta
                else None
            )
            record[f"videos/{camera}/source_from_timestamp"] = (
                float(meta[f"videos/{camera}/from_timestamp"])
                if f"videos/{camera}/from_timestamp" in meta
                else None
            )
            record[f"videos/{camera}/source_to_timestamp"] = (
                float(meta[f"videos/{camera}/to_timestamp"])
                if f"videos/{camera}/to_timestamp" in meta
                else None
            )
            record[f"videos/{camera}/clip_path"] = clip["path"] if clip else None
            record[f"videos/{camera}/clip_sha256"] = clip["sha256"] if clip else None
            record[f"videos/{camera}/clip_frames"] = int(clip["frames"]) if clip else 0
        rows.append(record)
    return pl.DataFrame(rows)


def _duration_out_s(metrics: pl.DataFrame, fps: int) -> pl.DataFrame:
    return metrics.with_columns((pl.col("n_frames_out") / fps).alias("duration_out_s"))


def _decision_counts(metrics: pl.DataFrame) -> dict[str, int]:
    counts = metrics.group_by("decision").agg(pl.len().alias("n"))
    result = {"keep": 0, "trim": 0, "drop": 0}
    for row in counts.iter_rows(named=True):
        result[row["decision"]] = int(row["n"])
    return result


def _run_json(
    *,
    cfg: PipelineConfig,
    source,
    run_key: str,
    metrics: pl.DataFrame,
    dry_run: bool,
    started_at: str,
    artifacts: dict | None,
    output_fingerprint: dict | None,
) -> dict:
    kept = metrics.filter(pl.col("decision") != "drop")
    return {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": __version__,
        "clean_contract_version": CLEAN_CONTRACT_VERSION,
        "stage": "clean",
        "run_key": run_key,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": _now(),
        "source": {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "root": str(source.root),
            "fingerprint": source.fingerprint,
        },
        "config": {
            "fingerprint": cfg.fingerprint(),
            "snapshot": cfg.snapshot(),
        },
        "counts": {
            "input_episodes": metrics.height,
            "output_episodes": kept.height,
            "input_frames": int(metrics["n_frames"].sum()),
            "output_frames": int(kept["n_frames_out"].sum()),
            "trimmed_frames": int((metrics["trim_head"] + metrics["trim_tail"]).sum()),
            "clips": len(artifacts["clips"]) if artifacts else 0,
        },
        "decision_counts": _decision_counts(metrics),
        "rule_hits": rule_hit_counts(metrics),
        "artifacts": artifacts,
        "output_fingerprint": output_fingerprint,
        "is_lerobot_dataset": False,
        "notes": [
            "Cleaned parquet preserves source episode/frame/index/timestamp values; indices may have gaps.",
            "This directory is intentionally not a loadable LeRobotDataset; run the convert stage for that.",
            "Stage-1 clips are near-lossless; conversion re-encodes them with the source encoder settings.",
        ],
        "tool_versions": _tool_versions(),
    }


def run_clean(
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> CleanResult:
    started_at = _now()
    t0 = time.monotonic()

    source = load_source(cfg)
    frames = load_frames(
        source,
        reward_key=cfg.cleaning.success.reward_key,
        done_key=cfg.cleaning.success.done_key,
    )
    validate_widths(cfg, frames)

    meta_episodes = set(source.episodes["episode_index"].to_list())
    data_episodes = set(frames["episode_index"].unique().to_list())
    if meta_episodes != data_episodes:
        raise SourceError(
            "Episode metadata and data parquet disagree: "
            f"missing_in_data={sorted(meta_episodes - data_episodes)}, "
            f"missing_in_meta={sorted(data_episodes - meta_episodes)}"
        )

    metrics = compute_metrics(
        frames,
        source.episodes,
        cfg.cleaning,
        source.fps,
        reward_key=cfg.cleaning.success.reward_key,
        done_key=cfg.cleaning.success.done_key,
    )
    metrics = decide_episodes(metrics, cfg.cleaning)
    metrics = _duration_out_s(metrics, source.fps)
    if int((metrics["decision"] != "drop").sum()) == 0:
        raise CleanError("All episodes were dropped; refusing to write an empty cleaned layer")

    run_key = f"{source.source_fp}__{cfg.fingerprint()[:12]}"
    final_dir = cfg.output.clean_root / run_key
    report_dir = cfg.output.report_root / run_key

    if final_dir.exists() and not dry_run:
        if overwrite or cfg.output.overwrite:
            assert_replaceable(final_dir, cfg.output.clean_root, pipeline_name=PIPELINE_NAME)
        else:
            marker = final_dir / "run.json"
            if not marker.is_file():
                raise CleanError(
                    f"Output exists but has no run.json marker: {final_dir}. Use --overwrite to replace it."
                )
            existing = json.loads(marker.read_text(encoding="utf-8"))
            same_source = (
                existing.get("source", {}).get("fingerprint", {}).get("digest")
                == source.fingerprint["digest"]
            )
            same_config = existing.get("config", {}).get("fingerprint") == cfg.fingerprint()
            if same_source and same_config and existing.get("output_fingerprint"):
                logger.info("Reusing existing cleaned layer: %s", final_dir)
                write_qc(
                    report_dir,
                    metrics,
                    cfg.cleaning,
                    run_key=run_key,
                    source_repo=source.repo_id,
                    source_revision=source.revision,
                    source_fp=source.source_fp,
                    dry_run=False,
                )
                (report_dir / "run.json").write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                return CleanResult(
                    run_key=run_key,
                    report_dir=report_dir,
                    metrics=metrics,
                    source_fp=source.source_fp,
                    final_dir=final_dir,
                    dry_run=False,
                    reused=True,
                )
            raise CleanError(
                f"Output exists with a different source/config fingerprint: {final_dir}. "
                "Use --overwrite or change the output root."
            )

    selection = _build_selection(frames, metrics)
    cleaned = frames.filter(selection["kept"]).select(frames.columns)
    if cleaned.schema != frames.schema:
        raise CleanError("Internal error: cleaned parquet schema changed")

    if dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        proposed_manifest = _build_manifest(metrics, source.episodes, source.video_keys, {})
        proposed_manifest.write_parquet(report_dir / "proposed_manifest.parquet")
        selection.write_parquet(report_dir / "proposed_selection.parquet")
        write_qc(
            report_dir,
            metrics,
            cfg.cleaning,
            run_key=run_key,
            source_repo=source.repo_id,
            source_revision=source.revision,
            source_fp=source.source_fp,
            dry_run=True,
        )
        run_json = _run_json(
            cfg=cfg,
            source=source,
            run_key=run_key,
            metrics=metrics,
            dry_run=True,
            started_at=started_at,
            artifacts=None,
            output_fingerprint=None,
        )
        write_json(report_dir / "run.json", run_json)
        logger.info("Dry run complete; no cleaned data written. Report: %s", report_dir)
        return CleanResult(
            run_key=run_key,
            report_dir=report_dir,
            metrics=metrics,
            source_fp=source.source_fp,
            final_dir=None,
            dry_run=True,
        )

    staging = staging_dir(cfg.output.clean_root, run_key)

    try:
        cleaned.write_parquet(staging / "data.parquet")
        selection.write_parquet(staging / "selection.parquet")
        shutil.copy2(source.root / "meta" / "tasks.parquet", staging / "tasks.parquet")
        write_json(staging / "source_schema.json", source.info)
        write_json(staging / "config.snapshot.json", cfg.snapshot())

        encoders = build_encoder_configs(source, cfg.cleaning)
        clips: dict[tuple[int, str], dict] = {}
        kept_episodes = (
            metrics.filter(pl.col("decision") != "drop")
            .sort("episode_index")["episode_index"]
            .to_list()
        )
        metrics_by_episode = {
            int(row["episode_index"]): row for row in metrics.iter_rows(named=True)
        }

        for camera in source.video_keys:
            logger.info("Slicing %s source video into %d clips", camera, len(kept_episodes))
            episode_frames = load_episode_frames(source, camera, kept_episodes)
            for episode in kept_episodes:
                row = metrics_by_episode[episode]
                frames_ep = episode_frames[episode]
                head = int(row["trim_head"])
                tail = int(row["trim_tail"])
                trimmed = frames_ep[head : len(frames_ep) - tail if tail else len(frames_ep)]
                expected = int(row["n_frames_out"])
                if len(trimmed) != expected:
                    raise CleanError(
                        f"Episode {episode} {camera}: trimmed clip has {len(trimmed)} frames, "
                        f"expected {expected}"
                    )
                rel_path = Path("videos") / camera / f"episode_{episode:05d}.mp4"
                out_path = staging / rel_path
                logger.debug("Encoding %s (%d frames)", rel_path, len(trimmed))
                encode_clip(trimmed, source.fps, out_path, encoders[camera])
                clips[(episode, camera)] = {
                    "path": rel_path.as_posix(),
                    "sha256": sha256_file(out_path),
                    "frames": len(trimmed),
                }

        manifest = _build_manifest(metrics, source.episodes, source.video_keys, clips)
        manifest.write_parquet(staging / "manifest.parquet")
        write_json(staging / "manifest.json", manifest.to_dicts())

        write_qc(
            staging / "qc",
            metrics,
            cfg.cleaning,
            run_key=run_key,
            source_repo=source.repo_id,
            source_revision=source.revision,
            source_fp=source.source_fp,
            dry_run=False,
        )

        artifacts = {
            "data": "data.parquet",
            "manifest": "manifest.parquet",
            "selection": "selection.parquet",
            "source_schema": "source_schema.json",
            "tasks": "tasks.parquet",
            "clips": {
                f"{camera}/episode_{episode:05d}.mp4": info
                for (episode, camera), info in sorted(clips.items())
            },
        }
        output_fingerprint = artifact_fingerprint(staging)
        run_json = _run_json(
            cfg=cfg,
            source=source,
            run_key=run_key,
            metrics=metrics,
            dry_run=False,
            started_at=started_at,
            artifacts=artifacts,
            output_fingerprint=output_fingerprint,
        )
        write_json(staging / "run.json", run_json)

        if final_dir.exists():
            safe_remove_previous(final_dir, cfg.output.clean_root, pipeline_name=PIPELINE_NAME)
        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    report_dir.mkdir(parents=True, exist_ok=True)
    for name in ["summary.json", "episodes.parquet", "report.md"]:
        shutil.copy2(final_dir / "qc" / name, report_dir / name)
    shutil.copy2(final_dir / "run.json", report_dir / "run.json")
    shutil.copy2(final_dir / "manifest.json", report_dir / "manifest.json")

    logger.info(
        "Clean stage complete in %.1fs: %d -> %d episodes, %d -> %d frames. Output: %s",
        time.monotonic() - t0,
        metrics.height,
        int((metrics["decision"] != "drop").sum()),
        int(metrics["n_frames"].sum()),
        int(metrics.filter(pl.col("decision") != "drop")["n_frames_out"].sum()),
        final_dir,
    )
    return CleanResult(
        run_key=run_key,
        report_dir=report_dir,
        metrics=metrics,
        source_fp=source.source_fp,
        final_dir=final_dir,
        dry_run=False,
    )
