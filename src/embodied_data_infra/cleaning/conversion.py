"""Stage 2: cleaned layer -> canonical LeRobot v3 dataset."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import PIPELINE_NAME, __version__
from .config import PipelineConfig
from .layout import (
    artifact_fingerprint,
    assert_replaceable,
    safe_remove_previous,
    staging_dir,
    write_json,
)
from .media import build_encoder_configs_from_features, encoder_signature, iter_clip_frames
from .validation import CleanLayer, ValidationError, validate_clean_contract, validate_curated

logger = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Raised when the cleaned layer cannot be converted into a valid LeRobot dataset."""


@dataclass
class ConversionResult:
    run_key: str
    final_dir: Path
    dataset_dir: Path
    validation: dict
    reused: bool = False


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


def resolve_clean_run(cfg: PipelineConfig, clean_run: str | Path | None = None) -> Path:
    if clean_run is not None:
        root = Path(clean_run).expanduser().resolve()
        if not (root / "run.json").is_file():
            raise ConversionError(f"Not a clean-stage run directory: {root}")
        return root

    candidates = sorted(
        path
        for path in cfg.output.clean_root.glob(f"*__{cfg.fingerprint()[:12]}")
        if (path / "run.json").is_file()
    )
    if not candidates:
        raise ConversionError(
            f"No cleaned layer found under {cfg.output.clean_root}. Run `embodied clean` first."
        )
    if len(candidates) > 1:
        raise ConversionError(
            "Multiple cleaned layers match the current config; pass --clean-run explicitly:\n"
            + "\n".join(f"- {path}" for path in candidates)
        )
    return candidates[0].resolve()


def _writer_features(info: dict) -> dict:
    from lerobot.utils.constants import DEFAULT_FEATURES

    features = {
        key: deepcopy(feature)
        for key, feature in info["features"].items()
        if key not in DEFAULT_FEATURES
    }
    if not features:
        raise ConversionError("source_schema.json has no writer features after removing defaults")
    return features


def _frame_value(value: Any, feature: dict) -> Any:
    dtype = feature["dtype"]
    shape = tuple(feature.get("shape") or ())
    if dtype in {"image", "video"}:
        raise ConversionError(f"Video/image feature handled elsewhere: {feature}")
    if dtype == "string":
        return str(value)
    if dtype in {"language", "language_events"}:
        raise ConversionError(f"Unsupported feature dtype for conversion: {dtype}")
    array = np.asarray(value, dtype=np.dtype(dtype))
    if shape == (1,) and array.size == 1:
        array = array.reshape(1)
    if shape and tuple(array.shape) != shape:
        raise ConversionError(f"Feature value has shape {array.shape}, expected {shape}")
    return array


def _iter_clips(
    layer: CleanLayer, episode: int, cameras: list[str]
) -> dict[str, Iterator[np.ndarray]]:
    return {camera: iter(iter_clip_frames(layer.clip_path(episode, camera))) for camera in cameras}


def _check_reuse(
    final_root: Path, layer: CleanLayer, cfg: PipelineConfig
) -> ConversionResult | None:
    marker = final_root / "run.json"
    validation_file = final_root / "validation.json"
    if not marker.is_file() or not validation_file.is_file():
        return None
    run = json.loads(marker.read_text(encoding="utf-8"))
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    clean_digest = (layer.run.get("output_fingerprint") or {}).get("digest")
    same_clean = (run.get("clean_fingerprint") or {}).get("digest") == clean_digest
    same_config = (run.get("config") or {}).get("fingerprint") == cfg.fingerprint()
    if same_clean and same_config and validation.get("passed"):
        logger.info("Reusing existing curated dataset: %s", final_root)
        return ConversionResult(
            run_key=run.get("run_key", final_root.name),
            final_dir=final_root,
            dataset_dir=final_root / "dataset",
            validation=validation,
            reused=True,
        )
    return None


def run_convert(
    cfg: PipelineConfig,
    *,
    clean_run: str | Path | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> ConversionResult:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    t0 = time.monotonic()
    clean_dir = resolve_clean_run(cfg, clean_run)
    layer = validate_clean_contract(clean_dir)
    clean_digest = layer.run["output_fingerprint"]

    run_key = str(layer.run["run_key"])
    final_root = cfg.output.curated_root / run_key

    if final_root.exists():
        if overwrite or cfg.output.overwrite:
            assert_replaceable(final_root, cfg.output.curated_root, pipeline_name=PIPELINE_NAME)
        else:
            reused = _check_reuse(final_root, layer, cfg)
            if reused is not None:
                return reused
            raise ConversionError(
                f"Curated output exists but does not match this clean/config fingerprint: {final_root}. "
                "Use --overwrite."
            )

    features = _writer_features(layer.info)
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    encoders = build_encoder_configs_from_features(features, crf=None)
    if set(encoders) != set(video_keys):
        raise ConversionError("Video feature set and encoder config set disagree")

    for key, feature in features.items():
        if feature.get("dtype") in {"image", "video"}:
            continue
        if key not in layer.data.columns:
            raise ConversionError(
                f"Feature {key!r} is declared in source_schema.json but missing from data.parquet"
            )

    for row in layer.kept_manifest.iter_rows(named=True):
        for key in video_keys:
            if not row.get(f"videos/{key}/clip_path"):
                raise ConversionError(f"Episode {row['episode_index']} is missing clip for {key}")

    staging = staging_dir(cfg.output.curated_root, run_key)
    staging_dataset = staging / "dataset"
    writer = None
    mapping: list[dict] = []
    try:
        writer = LeRobotDataset.create(
            repo_id=cfg.output.repo_id,
            fps=int(layer.info["fps"]),
            features=features,
            root=staging_dataset,
            robot_type=layer.info.get("robot_type"),
            use_videos=bool(video_keys),
            video_backend="pyav",
            rgb_encoder=next(iter(encoders.values())) if encoders else None,
        )

        kept = layer.kept_manifest
        total_frames = int(kept["n_frames_out"].sum())
        logger.info(
            "Writing %d episodes / %d frames into %s",
            kept.height,
            total_frames,
            staging_dataset,
        )

        for new_episode, row in enumerate(kept.iter_rows(named=True)):
            episode = int(row["episode_index"])
            n_frames = int(row["n_frames_out"])
            task = row.get("task")
            if not task:
                raise ConversionError(f"Episode {episode} has no task string")
            episode_rows = layer.data.filter(pl.col("episode_index") == episode).sort("frame_index")
            if episode_rows.height != n_frames:
                raise ConversionError(
                    f"Episode {episode}: data.parquet has {episode_rows.height} rows, expected {n_frames}"
                )

            clip_iters = _iter_clips(layer, episode, video_keys)
            for data_row in episode_rows.iter_rows(named=True):
                frame: dict[str, Any] = {}
                for key, feature in features.items():
                    if feature.get("dtype") == "video":
                        try:
                            frame[key] = next(clip_iters[key])
                        except StopIteration as exc:
                            raise ConversionError(
                                f"Episode {episode} {key}: clip ended before parquet rows"
                            ) from exc
                    else:
                        frame[key] = _frame_value(data_row[key], feature)
                frame["task"] = str(task)
                writer.add_frame(frame)

            for key, iterator in clip_iters.items():
                if next(iterator, None) is not None:
                    raise ConversionError(
                        f"Episode {episode} {key}: clip has more frames than parquet rows"
                    )

            writer.save_episode(parallel_encoding=cfg.conversion.parallel_encoding)
            if writer.meta.total_episodes != new_episode + 1:
                raise ConversionError(
                    f"Writer episode counter {writer.meta.total_episodes} != {new_episode + 1}"
                )
            mapping.append(
                {
                    "source_episode_index": episode,
                    "curated_episode_index": new_episode,
                    "frames": n_frames,
                }
            )

        writer.finalize()
        if not getattr(writer, "_is_finalized", False):
            raise ConversionError("LeRobot writer did not finalize")

        validation = validate_curated(staging_dataset, layer, cfg)
        write_json(staging / "validation.json", validation)
        if not validation.get("passed"):
            raise ValidationError(
                "Curated dataset validation failed:\n"
                + "\n".join(f"- {failure}" for failure in validation.get("failures", []))
            )

        conversion = {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": __version__,
            "stage": "convert",
            "run_key": run_key,
            "created_at": _now(),
            "clean_run": str(clean_dir),
            "clean_fingerprint": clean_digest,
            "repo_id": cfg.output.repo_id,
            "dataset_root": "dataset",
            "episode_mapping": mapping,
            "counts": {
                "episodes": kept.height,
                "frames": total_frames,
                "cameras": len(video_keys),
            },
            "encoders": {key: encoder_signature(value) for key, value in encoders.items()},
            "tool_versions": _tool_versions(),
        }
        write_json(staging / "conversion.json", conversion)

        output_fingerprint = artifact_fingerprint(staging)
        run_json = {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": __version__,
            "stage": "convert",
            "run_key": run_key,
            "created_at": _now(),
            "clean_run": str(clean_dir),
            "clean_fingerprint": clean_digest,
            "config": {"fingerprint": cfg.fingerprint(), "snapshot": cfg.snapshot()},
            "counts": conversion["counts"],
            "validation": {
                "passed": validation.get("passed"),
                "failures": validation.get("failures", []),
            },
            "output_fingerprint": output_fingerprint,
            "dataset_root": "dataset",
            "is_lerobot_dataset": True,
            "tool_versions": _tool_versions(),
        }
        write_json(staging / "run.json", run_json)

        if final_root.exists():
            safe_remove_previous(final_root, cfg.output.curated_root, pipeline_name=PIPELINE_NAME)
        os.replace(staging, final_root)
    except Exception:
        if writer is not None and not getattr(writer, "_is_finalized", False):
            try:
                writer.finalize()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    logger.info(
        "Convert stage complete in %.1fs: %d episodes / %d frames. Output: %s",
        time.monotonic() - t0,
        len(mapping),
        sum(item["frames"] for item in mapping),
        final_root,
    )

    if push_to_hub:
        logger.info("Pushing %s to the Hugging Face Hub", cfg.output.repo_id)
        dataset = LeRobotDataset(
            repo_id=cfg.output.repo_id,
            root=final_root / "dataset",
            video_backend="pyav",
        )
        dataset.push_to_hub(private=cfg.push.private)
        write_json(
            final_root / "push.json",
            {"repo_id": cfg.output.repo_id, "private": cfg.push.private, "pushed_at": _now()},
        )

    return ConversionResult(
        run_key=run_key,
        final_dir=final_root,
        dataset_dir=final_root / "dataset",
        validation=json.loads((final_root / "validation.json").read_text(encoding="utf-8")),
        reused=False,
    )
