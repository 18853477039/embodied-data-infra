"""Launch LeRobot training runs against curated datasets.

Training is delegated to ``lerobot.scripts.lerobot_train``. This module keeps
the operational glue explicit: config validation, selecting the newest curated
dataset, and deterministic CLI overrides for dataset root and output paths.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAIN_MODULE = "lerobot.scripts.lerobot_train"


class TrainingError(RuntimeError):
    """Raised when a training run cannot be started or its config is invalid."""


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a (possibly dry-run) training launch."""

    command: list[str]
    returncode: int
    dry_run: bool


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the top-level shape of a LeRobot train config."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise TrainingError(f"Training config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"Invalid JSON in training config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingError(f"Training config must be a JSON object: {config_path}")
    return payload


def validate_train_config(config: dict[str, Any]) -> None:
    """Check the minimum contract shared by all LeRobot train configs."""
    dataset = config.get("dataset")
    policy = config.get("policy")
    if not isinstance(dataset, dict):
        raise TrainingError("Training config is missing a 'dataset' object")
    if not isinstance(policy, dict):
        raise TrainingError("Training config is missing a 'policy' object")
    if not isinstance(policy.get("type"), str) or not policy["type"]:
        raise TrainingError("Training config policy.type must be a non-empty string")
    output_dir = config.get("output_dir")
    if output_dir is not None and not isinstance(output_dir, str):
        raise TrainingError("Training config output_dir must be a string or null")


def resolve_latest_curated(curated_root: str | Path = Path("data/curated")) -> Path:
    """Return the newest ``<curated_root>/<run_key>/dataset`` directory."""
    root = Path(curated_root).expanduser()
    if not root.is_dir():
        raise TrainingError(f"Curated root does not exist: {root}")

    candidates = [
        (run_dir.stat().st_mtime, run_dir.name, run_dir / "dataset")
        for run_dir in root.iterdir()
        if (run_dir / "dataset" / "meta" / "info.json").is_file()
    ]
    if not candidates:
        raise TrainingError(
            f"No curated datasets found under {root}. Run `embodied convert` first."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def build_train_command(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    output_dir: str | Path | None = None,
    job_name: str | None = None,
    resume: bool | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the argv for ``lerobot-train`` without executing it."""
    command = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--config_path",
        str(Path(config_path).expanduser().resolve()),
    ]
    if dataset_root is not None:
        command += ["--dataset.root", str(Path(dataset_root).expanduser().resolve())]
    if repo_id is not None:
        command += ["--dataset.repo_id", repo_id]
    if output_dir is not None:
        command += ["--output_dir", str(Path(output_dir).expanduser().resolve())]
    if job_name is not None:
        command += ["--job_name", job_name]
    if resume is not None:
        command += ["--resume", "true" if resume else "false"]
    command += list(extra_args)
    return command


def run_train(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    latest_curated: bool = False,
    curated_root: str | Path = Path("data/curated"),
    repo_id: str | None = None,
    output_dir: str | Path | None = None,
    job_name: str | None = None,
    resume: bool | None = None,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> TrainResult:
    """Validate the config and launch a LeRobot training run."""
    config_file = Path(config_path).expanduser().resolve()
    validate_train_config(load_train_config(config_file))

    if latest_curated and dataset_root is not None:
        raise TrainingError("Use either --dataset-root or --latest-curated, not both")
    if latest_curated:
        dataset_root = resolve_latest_curated(curated_root)
    if dataset_root is not None:
        resolved = Path(dataset_root).expanduser().resolve()
        if not (resolved / "meta" / "info.json").is_file():
            raise TrainingError(f"Dataset root has no meta/info.json: {resolved}")
        dataset_root = resolved

    command = build_train_command(
        config_file,
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=output_dir,
        job_name=job_name,
        resume=resume,
        extra_args=extra_args,
    )
    logger.info("Training command: %s", shlex.join(command))

    if dry_run:
        return TrainResult(command=command, returncode=0, dry_run=True)

    completed = subprocess.run(command, check=False)
    return TrainResult(command=command, returncode=completed.returncode, dry_run=False)
