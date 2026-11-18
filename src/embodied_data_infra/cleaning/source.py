"""Resolve a raw LeRobot v3 dataset (Hub or local) and fingerprint it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import PipelineConfig, SourceConfig


class SourceError(RuntimeError):
    """Raised when the raw dataset cannot be resolved or is not LeRobot v3."""


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_root(cfg: SourceConfig) -> Path:
    if cfg.local_root is not None:
        root = cfg.local_root.expanduser().resolve()
        if not (root / "meta" / "info.json").is_file():
            raise SourceError(f"local_root does not contain meta/info.json: {root}")
        return root

    local_dir = (cfg.download_dir / sanitize_repo_id(cfg.repo_id)).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SourceError("huggingface_hub is required to download the source dataset") from exc

    try:
        snapshot_download(
            repo_id=cfg.repo_id,
            revision=cfg.revision,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["meta/**", "data/**", "videos/**"],
        )
    except Exception as exc:  # noqa: BLE001 - surface a useful message
        raise SourceError(
            f"Could not download {cfg.repo_id}@{cfg.revision}: {exc}. "
            "If the dataset is already local, set source.local_root instead."
        ) from exc
    return local_dir.resolve()


def _fingerprint(root: Path, cfg: SourceConfig) -> dict:
    files: list[dict] = []
    for subdir in ("meta", "data", "videos"):
        base = root / subdir
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            record = {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
            }
            if subdir == "videos" and not cfg.hash_videos:
                record["sha256"] = None
            else:
                record["sha256"] = sha256_file(path)
            files.append(record)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "files": files,
        "digest": digest,
    }


@dataclass
class SourceDataset:
    root: Path
    repo_id: str
    revision: str
    info: dict
    tasks: pl.DataFrame
    episodes: pl.DataFrame
    data_paths: list[Path]
    fingerprint: dict

    @property
    def source_fp(self) -> str:
        return self.fingerprint["digest"][:12]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict:
        return self.info["features"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    def video_path(self, video_key: str, chunk_index: int, file_index: int) -> Path:
        template = self.info["video_path"]
        return self.root / template.format(
            video_key=video_key, chunk_index=chunk_index, file_index=file_index
        )


def load_source(cfg: PipelineConfig) -> SourceDataset:
    root = _resolve_root(cfg.source)
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"Could not read {info_path}: {exc}") from exc

    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3."):
        raise SourceError(
            f"Unsupported LeRobot codebase_version={version!r}; this pipeline targets v3.x datasets."
        )

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise SourceError(f"Missing {tasks_path}")
    tasks = pl.read_parquet(tasks_path)

    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise SourceError(f"No episode metadata parquet files under {root / 'meta' / 'episodes'}")
    episodes = pl.concat([pl.read_parquet(path) for path in episode_files], how="vertical_relaxed")
    if "episode_index" not in episodes.columns:
        raise SourceError("meta/episodes parquet has no episode_index column")
    episodes = episodes.sort("episode_index")
    if episodes["episode_index"].n_unique() != len(episodes):
        raise SourceError("meta/episodes contains duplicate episode_index values")

    data_paths = sorted((root / "data").rglob("*.parquet"))
    if not data_paths:
        raise SourceError(f"No data parquet files under {root / 'data'}")

    if not isinstance(info.get("features"), dict):
        raise SourceError("meta/info.json has no valid 'features' mapping")
    if "fps" not in info:
        raise SourceError("meta/info.json has no 'fps'")

    return SourceDataset(
        root=root,
        repo_id=cfg.source.repo_id,
        revision=cfg.source.revision,
        info=info,
        tasks=tasks,
        episodes=episodes,
        data_paths=data_paths,
        fingerprint=_fingerprint(root, cfg.source),
    )
