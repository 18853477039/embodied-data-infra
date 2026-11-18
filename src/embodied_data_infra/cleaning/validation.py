"""Contract checks for the cleaned layer and read-back validation of the curated dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import CLEAN_CONTRACT_VERSION
from .config import PipelineConfig
from .layout import artifact_fingerprint
from .media import iter_clip_frames, probe_video
from .source import sha256_file


class ValidationError(RuntimeError):
    """Raised when an artifact contract or read-back check fails."""


@dataclass
class CleanLayer:
    root: Path
    run: dict
    info: dict
    tasks: pl.DataFrame
    data: pl.DataFrame
    manifest: pl.DataFrame
    selection: pl.DataFrame

    @property
    def kept_manifest(self) -> pl.DataFrame:
        return self.manifest.filter(pl.col("decision") != "drop").sort("episode_index")

    @property
    def kept_episodes(self) -> list[int]:
        return [int(v) for v in self.kept_manifest["episode_index"].to_list()]

    def clip_path(self, episode: int, camera: str) -> Path:
        row = self.manifest.filter(pl.col("episode_index") == episode).row(0, named=True)
        clip = row.get(f"videos/{camera}/clip_path")
        if not clip:
            raise ValidationError(f"No clip for episode {episode} camera {camera}")
        return self.root / clip


def _require(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Missing required cleaned-layer file: {path}")


def load_clean_layer(clean_dir: str | Path) -> CleanLayer:
    root = Path(clean_dir).expanduser().resolve()
    for name in [
        "run.json",
        "data.parquet",
        "manifest.parquet",
        "selection.parquet",
        "source_schema.json",
        "tasks.parquet",
    ]:
        _require(root / name)

    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run.get("stage") != "clean":
        raise ValidationError(f"{root}/run.json is not a clean-stage artifact")
    if run.get("clean_contract_version") != CLEAN_CONTRACT_VERSION:
        raise ValidationError(
            f"Unsupported clean contract version {run.get('clean_contract_version')!r}; "
            f"expected {CLEAN_CONTRACT_VERSION}"
        )
    if run.get("is_lerobot_dataset"):
        raise ValidationError("Cleaned layer must not claim to be a loadable LeRobot dataset")

    info = json.loads((root / "source_schema.json").read_text(encoding="utf-8"))
    tasks = pl.read_parquet(root / "tasks.parquet")
    data = pl.read_parquet(root / "data.parquet")
    manifest = pl.read_parquet(root / "manifest.parquet")
    selection = pl.read_parquet(root / "selection.parquet")
    return CleanLayer(
        root=root,
        run=run,
        info=info,
        tasks=tasks,
        data=data,
        manifest=manifest,
        selection=selection,
    )


def validate_clean_contract(clean_dir: str | Path) -> CleanLayer:
    layer = load_clean_layer(clean_dir)

    recorded = layer.run.get("output_fingerprint") or {}
    actual = artifact_fingerprint(layer.root)
    if not recorded.get("digest"):
        raise ValidationError("run.json has no output_fingerprint")
    if recorded["digest"] != actual["digest"]:
        raise ValidationError(
            "Cleaned-layer fingerprint mismatch: files changed after the clean stage "
            f"(recorded {recorded['digest'][:12]}, actual {actual['digest'][:12]})"
        )

    kept = layer.kept_manifest
    if kept.is_empty():
        raise ValidationError("Cleaned layer contains no kept episodes")

    kept_rows = int(kept["n_frames_out"].sum())
    if layer.data.height != kept_rows:
        raise ValidationError(
            f"data.parquet has {layer.data.height} rows, manifest expects {kept_rows}"
        )
    if int(layer.selection["kept"].sum()) != layer.data.height:
        raise ValidationError("selection.parquet kept count does not match data.parquet")

    mapping = (
        layer.selection.filter(pl.col("kept"))
        .select(["episode_index", "curated_episode_index"])
        .unique()
        .sort("curated_episode_index")
    )
    if mapping.height != kept.height:
        raise ValidationError("selection.parquet does not map every kept episode exactly once")
    if mapping["curated_episode_index"].null_count() > 0:
        raise ValidationError("selection.parquet has null curated_episode_index for kept frames")
    if mapping["curated_episode_index"].to_list() != list(range(kept.height)):
        raise ValidationError("curated_episode_index is not contiguous from 0")

    cameras = [
        key for key, feature in layer.info["features"].items() if feature.get("dtype") == "video"
    ]
    for row in kept.iter_rows(named=True):
        episode = int(row["episode_index"])
        frames = int(row["n_frames_out"])
        if frames <= 0:
            raise ValidationError(f"Episode {episode} has non-positive output length")
        for camera in cameras:
            clip = row.get(f"videos/{camera}/clip_path")
            checksum = row.get(f"videos/{camera}/clip_sha256")
            if not clip or not checksum:
                raise ValidationError(f"Episode {episode} {camera} has no clip/checksum")
            path = layer.root / clip
            if not path.is_file():
                raise ValidationError(f"Missing clip: {path}")
            if sha256_file(path) != checksum:
                raise ValidationError(f"Clip checksum mismatch: {path}")
            probe = probe_video(path)
            if probe["frames"] != frames:
                raise ValidationError(
                    f"{path} has {probe['frames']} frames, manifest expects {frames}"
                )

    return layer


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


def _video_sample_check(
    dataset,
    layer: CleanLayer,
    camera: str,
    cfg: PipelineConfig,
) -> list[str]:
    failures: list[str] = []
    kept = layer.kept_manifest
    for new_episode, row in enumerate(kept.iter_rows(named=True)):
        old_episode = int(row["episode_index"])
        n = int(row["n_frames_out"])
        clip_frames = list(iter_clip_frames(layer.clip_path(old_episode, camera)))
        if len(clip_frames) != n:
            failures.append(f"clip {old_episode}/{camera}: {len(clip_frames)} frames, expected {n}")
            continue
        meta = dataset.meta.episodes[new_episode]
        start = int(meta["dataset_from_index"])
        for offset in sorted({0, n // 2, n - 1}):
            item = dataset[start + offset]
            image = item[camera]
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            expected = clip_frames[offset]
            mae = float(np.abs(image.astype(np.float32) - expected.astype(np.float32)).mean())
            corr = _pearson(image, expected)
            if image.mean() <= 1:
                failures.append(f"episode {new_episode}/{camera} frame {offset}: black frame")
            if corr < cfg.conversion.video_corr_min or mae > cfg.conversion.video_mae_max:
                failures.append(
                    f"episode {new_episode}/{camera} frame {offset}: corr={corr:.3f}, mae={mae:.2f}"
                )
    return failures


def validate_curated(
    dataset_dir: str | Path,
    layer: CleanLayer,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise ValidationError(
            f"Not a local LeRobot dataset (missing meta/info.json): {dataset_root}"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    failures: list[str] = []
    dataset = LeRobotDataset(
        repo_id=cfg.output.repo_id,
        root=dataset_root,
        video_backend="pyav",
    )

    kept = layer.kept_manifest
    expected_episodes = kept.height
    expected_frames = int(kept["n_frames_out"].sum())
    if dataset.meta.total_episodes != expected_episodes:
        failures.append(
            f"total_episodes={dataset.meta.total_episodes}, expected {expected_episodes}"
        )
    if dataset.meta.total_frames != expected_frames:
        failures.append(f"total_frames={dataset.meta.total_frames}, expected {expected_frames}")
    if dataset.meta.fps != int(layer.info["fps"]):
        failures.append(f"fps={dataset.meta.fps}, expected {layer.info['fps']}")
    if len(dataset) != expected_frames:
        failures.append(f"len(dataset)={len(dataset)}, expected {expected_frames}")

    for key, feature in layer.info["features"].items():
        if key in {"timestamp", "frame_index", "episode_index", "index", "task_index"}:
            continue
        if key not in dataset.meta.features:
            failures.append(f"curated dataset is missing feature {key}")
            continue
        expected_shape = tuple(feature.get("shape") or ())
        actual_shape = tuple(dataset.meta.features[key].get("shape") or ())
        if expected_shape and actual_shape and expected_shape != actual_shape:
            failures.append(f"feature {key} shape {actual_shape}, expected {expected_shape}")

    hf = dataset.hf_dataset
    if hf is None:
        raise ValidationError("Curated dataset has no hf_dataset table")
    hf_len = len(hf)
    if hf_len != expected_frames:
        failures.append(f"hf_dataset length {hf_len}, expected {expected_frames}")
    else:
        expected_rows = layer.data.sort("index").select(
            ["observation.state", "action", "next.reward", "next.done"]
        )
        actual_states = np.asarray(hf["observation.state"], dtype=np.float32)
        actual_actions = np.asarray(hf["action"], dtype=np.float32)
        expected_states = np.stack(
            [np.asarray(v, dtype=np.float32) for v in expected_rows["observation.state"]]
        )
        expected_actions = np.stack(
            [np.asarray(v, dtype=np.float32) for v in expected_rows["action"]]
        )
        if actual_states.shape != expected_states.shape:
            failures.append(f"state shape {actual_states.shape}, expected {expected_states.shape}")
        elif not np.allclose(actual_states, expected_states, atol=1e-6, rtol=0):
            failures.append("state values differ from cleaned data")
        if actual_actions.shape != expected_actions.shape:
            failures.append(
                f"action shape {actual_actions.shape}, expected {expected_actions.shape}"
            )
        elif not np.allclose(actual_actions, expected_actions, atol=1e-6, rtol=0):
            failures.append("action values differ from cleaned data")

        actual_rewards = np.asarray(hf["next.reward"], dtype=np.float32).reshape(-1)
        expected_rewards = np.asarray(expected_rows["next.reward"], dtype=np.float32).reshape(-1)
        if not np.allclose(actual_rewards, expected_rewards, atol=1e-6, rtol=0):
            failures.append("next.reward values differ from cleaned data")
        actual_done = np.asarray(hf["next.done"]).astype(bool).reshape(-1)
        expected_done = np.asarray(expected_rows["next.done"]).astype(bool).reshape(-1)
        if not np.array_equal(actual_done, expected_done):
            failures.append("next.done values differ from cleaned data")

        timestamps = np.asarray(hf["timestamp"], dtype=np.float64).reshape(-1)
        episode_indices = np.asarray(hf["episode_index"]).reshape(-1)
        frame_indices = np.asarray(hf["frame_index"]).reshape(-1)
        cursor = 0
        for new_episode, row in enumerate(kept.iter_rows(named=True)):
            n = int(row["n_frames_out"])
            expected_ts = np.arange(n, dtype=np.float64) / dataset.meta.fps
            if not np.allclose(timestamps[cursor : cursor + n], expected_ts, atol=1e-6):
                failures.append(f"episode {row['episode_index']}: timestamp grid mismatch")
            if not np.array_equal(frame_indices[cursor : cursor + n], np.arange(n)):
                failures.append(f"episode {row['episode_index']}: frame_index grid mismatch")
            if not np.all(episode_indices[cursor : cursor + n] == new_episode):
                failures.append(f"episode {row['episode_index']}: episode_index mismatch")
            cursor += n

    if dataset.meta.stats is None:
        failures.append("curated dataset has no stats")
    else:
        for key in ["observation.state", "action", "next.reward", "next.done"]:
            if key not in dataset.meta.stats:
                failures.append(f"stats missing feature {key}")

    for camera in [
        key for key, feature in layer.info["features"].items() if feature.get("dtype") == "video"
    ]:
        failures.extend(_video_sample_check(dataset, layer, camera, cfg))

    result = {
        "dataset_root": str(dataset_root),
        "expected_episodes": expected_episodes,
        "expected_frames": expected_frames,
        "actual_episodes": int(dataset.meta.total_episodes),
        "actual_frames": int(dataset.meta.total_frames),
        "checks": {
            "meta_counts": not any("total_" in f or "len(dataset)" in f for f in failures),
            "features": not any("feature" in f or "shape" in f for f in failures),
            "tabular": not any(
                token in f
                for f in failures
                for token in ["state values", "action values", "reward", "done"]
            ),
            "timestamps": not any(
                "timestamp" in f or "frame_index" in f or "episode_index" in f for f in failures
            ),
            "stats": not any("stats" in f for f in failures),
            "video": not any("frame" in f or "corr=" in f or "black" in f for f in failures),
        },
        "failures": failures,
        "passed": not failures,
        "video_corr_min": cfg.conversion.video_corr_min,
        "video_mae_max": cfg.conversion.video_mae_max,
    }
    return result
