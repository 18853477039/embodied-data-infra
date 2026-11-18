"""Launch LeRobot recording sessions from a validated JSON config.

The collection stage is a thin, auditable wrapper around
``lerobot.scripts.lerobot_record``: it validates the declarative config,
applies explicit CLI overrides, and forwards everything else unchanged.
Recording itself (robot I/O, cameras, gamepad) stays in LeRobot.
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

RECORD_MODULE = "lerobot.scripts.lerobot_record"


class CollectionError(RuntimeError):
    """Raised when recording cannot be started or the config is invalid."""


@dataclass(frozen=True)
class CollectResult:
    """Outcome of a (possibly dry-run) recording launch."""

    command: list[str]
    returncode: int
    dry_run: bool


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config and require a top-level object."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise CollectionError(f"Collection config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CollectionError(f"Invalid JSON in collection config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CollectionError(f"Collection config must be a JSON object: {config_path}")
    return payload


def validate_record_config(config: dict[str, Any]) -> None:
    """Check the minimum contract shared by all LeRobot record configs."""
    env = config.get("env")
    dataset = config.get("dataset")
    if not isinstance(env, dict):
        raise CollectionError("Collection config is missing an 'env' object")
    if not isinstance(dataset, dict):
        raise CollectionError("Collection config is missing a 'dataset' object")

    for key, expected in (("name", str), ("task", str), ("fps", (int, float)), ("features", dict)):
        value = env.get(key)
        if value is None:
            raise CollectionError(f"Collection config is missing env.{key}")
        if not isinstance(value, expected):
            raise CollectionError(
                f"Collection config env.{key} must be {expected}, got {type(value).__name__}"
            )
    if not env["features"]:
        raise CollectionError("Collection config env.features must not be empty")

    repo_id = dataset.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise CollectionError("Collection config dataset.repo_id must be a non-empty string")

    episodes = dataset.get("num_episodes_to_record")
    if episodes is not None and (not isinstance(episodes, int) or episodes <= 0):
        raise CollectionError(
            "Collection config dataset.num_episodes_to_record must be a positive int"
        )

    root = dataset.get("root")
    if root is not None and not isinstance(root, str):
        raise CollectionError("Collection config dataset.root must be a string or null")


def build_record_command(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    num_episodes: int | None = None,
    push_to_hub: bool | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the argv for ``lerobot-record`` without executing it."""
    if num_episodes is not None and num_episodes <= 0:
        raise CollectionError("--episodes must be a positive integer")

    command = [
        sys.executable,
        "-m",
        RECORD_MODULE,
        "--config_path",
        str(Path(config_path).expanduser().resolve()),
    ]
    if dataset_root is not None:
        command += ["--dataset.root", str(Path(dataset_root).expanduser().resolve())]
    if repo_id is not None:
        command += ["--dataset.repo_id", repo_id]
    if num_episodes is not None:
        command += ["--dataset.num_episodes", str(num_episodes)]
    if push_to_hub is not None:
        command += ["--dataset.push_to_hub", "true" if push_to_hub else "false"]
    command += list(extra_args)
    return command


def run_record(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    num_episodes: int | None = None,
    push_to_hub: bool | None = None,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
) -> CollectResult:
    """Validate the config and launch a LeRobot recording session."""
    config_file = Path(config_path).expanduser().resolve()
    validate_record_config(load_json_config(config_file))
    command = build_record_command(
        config_file,
        dataset_root=dataset_root,
        repo_id=repo_id,
        num_episodes=num_episodes,
        push_to_hub=push_to_hub,
        extra_args=extra_args,
    )
    logger.info("Recording command: %s", shlex.join(command))

    if dry_run:
        return CollectResult(command=command, returncode=0, dry_run=True)

    completed = subprocess.run(command, check=False)
    return CollectResult(command=command, returncode=completed.returncode, dry_run=False)
