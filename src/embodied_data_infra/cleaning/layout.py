"""Run-directory helpers: fingerprints, atomic staging, safe overwrite."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .source import sha256_file


class LayoutError(RuntimeError):
    """Raised when an output directory cannot be created or safely replaced."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def artifact_fingerprint(directory: Path, *, exclude: tuple[str, ...] = ("run.json",)) -> dict:
    records = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path.relative_to(directory).as_posix() in exclude:
            continue
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "files": records,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def staging_dir(root: Path, run_key: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f".{run_key}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def assert_replaceable(path: Path, root: Path, *, pipeline_name: str) -> None:
    root = root.resolve()
    path = path.resolve()
    if path.parent != root:
        raise LayoutError(f"Refusing to replace output outside output root: {path}")
    marker = path / "run.json"
    if not marker.is_file():
        raise LayoutError(f"Refusing to overwrite non-pipeline directory (no run.json): {path}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Refusing to overwrite unreadable run marker: {marker}") from exc
    if payload.get("pipeline") != pipeline_name:
        raise LayoutError(f"Refusing to overwrite output from another pipeline: {path}")


def safe_remove_previous(path: Path, root: Path, *, pipeline_name: str) -> None:
    assert_replaceable(path, root, pipeline_name=pipeline_name)
    shutil.rmtree(path)
