"""Typed YAML configuration for the cleaning pipeline.

The config is deliberately explicit: every rule threshold lives in YAML and is
hashed into the run key so that two different policies can never silently share
an output directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML config is missing, malformed, or inconsistent."""


def _reject_unknown(section: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in {path}: {sorted(unknown)}")


def _section(data: dict[str, Any], name: str, *, required: bool = True) -> dict[str, Any]:
    value = data.get(name)
    if value is None:
        if required:
            raise ConfigError(f"Missing required config section: {name}")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping, got {type(value).__name__}")
    return value


def _path(value: Any, name: str) -> Path:
    if value is None:
        raise ConfigError(f"'{name}' must be a path, got null")
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"'{name}' must be a path string, got {type(value).__name__}")
    return Path(value).expanduser()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, got {value!r}")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {value!r}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True)
class SourceConfig:
    repo_id: str
    revision: str = "main"
    local_root: Path | None = None
    download_dir: Path = Path("data/raw")
    hash_videos: bool = True


@dataclass(frozen=True)
class SuccessConfig:
    require: bool = True
    reward_key: str = "next.reward"
    done_key: str = "next.done"


@dataclass(frozen=True)
class ActionConfig:
    bounds: tuple[tuple[float, float], ...] = ()
    allowed_values: dict[int, tuple[float, ...]] = field(default_factory=dict)
    atol: float = 1.0e-5


@dataclass(frozen=True)
class IdleConfig:
    state_motion_dims: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    action_translation_dims: tuple[int, ...] = (0, 1, 2)
    motion_eps: float = 1.0e-4
    action_eps: float = 1.0e-6
    trim_leading: bool = True
    trim_trailing: bool = True
    min_trim_frames: int = 10
    max_trim_ratio: float = 0.5
    interior_idle_ratio_max: float = 0.20
    interior_idle_run_max: int = 10


@dataclass(frozen=True)
class CleaningConfig:
    fps_tolerance_s: float = 1.0e-3
    fail_on_nan: bool = True
    min_frames: int = 5
    max_frames: int = 150
    duration_outlier_factor: float = 3.0
    clip_crf: int = 18
    success: SuccessConfig = field(default_factory=SuccessConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)


@dataclass(frozen=True)
class OutputConfig:
    clean_root: Path = Path("data/cleaned")
    curated_root: Path = Path("data/curated")
    report_root: Path = Path("data/reports")
    repo_id: str = "local/pick_cube_clean"
    overwrite: bool = False


@dataclass(frozen=True)
class ConversionConfig:
    parallel_encoding: bool = False
    validate_after: bool = True
    video_corr_min: float = 0.95
    video_mae_max: float = 10.0


@dataclass(frozen=True)
class PushConfig:
    enabled: bool = False
    private: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    source: SourceConfig
    cleaning: CleaningConfig
    output: OutputConfig
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    push: PushConfig = field(default_factory=PushConfig)

    def snapshot(self) -> dict[str, Any]:
        return _canonicalize(asdict(self))

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.snapshot(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(self, **kwargs: Any) -> PipelineConfig:
        return replace(self, **kwargs)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def _parse_success(raw: dict[str, Any]) -> SuccessConfig:
    _reject_unknown(raw, {"require", "reward_key", "done_key"}, "cleaning.success")
    return SuccessConfig(
        require=_boolean(raw.get("require", True), "cleaning.success.require"),
        reward_key=str(raw.get("reward_key", "next.reward")),
        done_key=str(raw.get("done_key", "next.done")),
    )


def _parse_action(raw: dict[str, Any]) -> ActionConfig:
    _reject_unknown(raw, {"bounds", "allowed_values", "atol"}, "cleaning.action")
    bounds_raw = raw.get("bounds", [])
    if not isinstance(bounds_raw, list) or not bounds_raw:
        raise ConfigError("cleaning.action.bounds must be a non-empty list of [low, high] pairs")
    bounds: list[tuple[float, float]] = []
    for i, pair in enumerate(bounds_raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"cleaning.action.bounds[{i}] must be [low, high]")
        low, high = (
            _number(pair[0], f"cleaning.action.bounds[{i}].low"),
            _number(pair[1], f"cleaning.action.bounds[{i}].high"),
        )
        if low > high:
            raise ConfigError(f"cleaning.action.bounds[{i}] has low > high")
        bounds.append((low, high))

    allowed_raw = raw.get("allowed_values", {}) or {}
    if not isinstance(allowed_raw, dict):
        raise ConfigError("cleaning.action.allowed_values must be a mapping of dimension -> values")
    allowed: dict[int, tuple[float, ...]] = {}
    for key, values in allowed_raw.items():
        try:
            dim = int(key)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"cleaning.action.allowed_values key must be an integer, got {key!r}"
            ) from exc
        if dim < 0:
            raise ConfigError("cleaning.action.allowed_values dimension must be >= 0")
        if not isinstance(values, list) or not values:
            raise ConfigError(f"cleaning.action.allowed_values[{dim}] must be a non-empty list")
        allowed[dim] = tuple(_number(v, f"cleaning.action.allowed_values[{dim}]") for v in values)

    atol = _number(raw.get("atol", 1.0e-5), "cleaning.action.atol")
    if atol < 0:
        raise ConfigError("cleaning.action.atol must be >= 0")
    return ActionConfig(bounds=tuple(bounds), allowed_values=allowed, atol=atol)


def _parse_idle(raw: dict[str, Any]) -> IdleConfig:
    _reject_unknown(
        raw,
        {
            "state_motion_dims",
            "action_translation_dims",
            "motion_eps",
            "action_eps",
            "trim_leading",
            "trim_trailing",
            "min_trim_frames",
            "max_trim_ratio",
            "interior_idle_ratio_max",
            "interior_idle_run_max",
        },
        "cleaning.idle",
    )

    def dims(name: str, default: list[int]) -> tuple[int, ...]:
        values = raw.get(name, default)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"cleaning.idle.{name} must be a non-empty list")
        out = tuple(_integer(v, f"cleaning.idle.{name}") for v in values)
        if any(v < 0 for v in out):
            raise ConfigError(f"cleaning.idle.{name} entries must be >= 0")
        if len(set(out)) != len(out):
            raise ConfigError(f"cleaning.idle.{name} contains duplicates")
        return out

    cfg = IdleConfig(
        state_motion_dims=dims("state_motion_dims", [0, 1, 2, 3, 4, 5, 6]),
        action_translation_dims=dims("action_translation_dims", [0, 1, 2]),
        motion_eps=_number(raw.get("motion_eps", 1.0e-4), "cleaning.idle.motion_eps"),
        action_eps=_number(raw.get("action_eps", 1.0e-6), "cleaning.idle.action_eps"),
        trim_leading=_boolean(raw.get("trim_leading", True), "cleaning.idle.trim_leading"),
        trim_trailing=_boolean(raw.get("trim_trailing", True), "cleaning.idle.trim_trailing"),
        min_trim_frames=_integer(raw.get("min_trim_frames", 10), "cleaning.idle.min_trim_frames"),
        max_trim_ratio=_number(raw.get("max_trim_ratio", 0.5), "cleaning.idle.max_trim_ratio"),
        interior_idle_ratio_max=_number(
            raw.get("interior_idle_ratio_max", 0.20), "cleaning.idle.interior_idle_ratio_max"
        ),
        interior_idle_run_max=_integer(
            raw.get("interior_idle_run_max", 10), "cleaning.idle.interior_idle_run_max"
        ),
    )
    if cfg.motion_eps < 0 or cfg.action_eps < 0:
        raise ConfigError("cleaning.idle.motion_eps / action_eps must be >= 0")
    if cfg.min_trim_frames < 1:
        raise ConfigError("cleaning.idle.min_trim_frames must be >= 1")
    if not 0 < cfg.max_trim_ratio < 1:
        raise ConfigError("cleaning.idle.max_trim_ratio must be in (0, 1)")
    if not 0 <= cfg.interior_idle_ratio_max <= 1:
        raise ConfigError("cleaning.idle.interior_idle_ratio_max must be in [0, 1]")
    if cfg.interior_idle_run_max < 1:
        raise ConfigError("cleaning.idle.interior_idle_run_max must be >= 1")
    return cfg


def _parse_cleaning(raw: dict[str, Any]) -> CleaningConfig:
    _reject_unknown(
        raw,
        {
            "fps_tolerance_s",
            "fail_on_nan",
            "min_frames",
            "max_frames",
            "duration_outlier_factor",
            "clip_crf",
            "success",
            "action",
            "idle",
        },
        "cleaning",
    )
    min_frames = _integer(raw.get("min_frames", 5), "cleaning.min_frames")
    max_frames = _integer(raw.get("max_frames", 150), "cleaning.max_frames")
    if min_frames < 1:
        raise ConfigError("cleaning.min_frames must be >= 1")
    if max_frames <= min_frames:
        raise ConfigError("cleaning.max_frames must be > cleaning.min_frames")
    factor = _number(raw.get("duration_outlier_factor", 3.0), "cleaning.duration_outlier_factor")
    if factor <= 1:
        raise ConfigError("cleaning.duration_outlier_factor must be > 1")
    clip_crf = _integer(raw.get("clip_crf", 18), "cleaning.clip_crf")
    if not 0 <= clip_crf <= 63:
        raise ConfigError("cleaning.clip_crf must be in [0, 63]")
    fps_tol = _number(raw.get("fps_tolerance_s", 1.0e-3), "cleaning.fps_tolerance_s")
    if fps_tol < 0:
        raise ConfigError("cleaning.fps_tolerance_s must be >= 0")

    return CleaningConfig(
        fps_tolerance_s=fps_tol,
        fail_on_nan=_boolean(raw.get("fail_on_nan", True), "cleaning.fail_on_nan"),
        min_frames=min_frames,
        max_frames=max_frames,
        duration_outlier_factor=factor,
        clip_crf=clip_crf,
        success=_parse_success(raw.get("success", {}) or {}),
        action=_parse_action(raw.get("action", {}) or {}),
        idle=_parse_idle(raw.get("idle", {}) or {}),
    )


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    _reject_unknown(raw, {"source", "cleaning", "output", "conversion", "push"}, "config root")

    source_raw = _section(raw, "source")
    _reject_unknown(
        source_raw,
        {"repo_id", "revision", "local_root", "download_dir", "hash_videos"},
        "source",
    )
    repo_id = source_raw.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ConfigError("source.repo_id must be a non-empty string")
    local_root_raw = source_raw.get("local_root")
    source = SourceConfig(
        repo_id=repo_id.strip(),
        revision=str(source_raw.get("revision", "main")),
        local_root=_path(local_root_raw, "source.local_root")
        if local_root_raw is not None
        else None,
        download_dir=_path(source_raw.get("download_dir", "data/raw"), "source.download_dir"),
        hash_videos=_boolean(source_raw.get("hash_videos", True), "source.hash_videos"),
    )

    cleaning = _parse_cleaning(_section(raw, "cleaning"))

    output_raw = _section(raw, "output")
    _reject_unknown(
        output_raw,
        {"clean_root", "curated_root", "report_root", "repo_id", "overwrite"},
        "output",
    )
    output_repo_id = output_raw.get("repo_id", "local/pick_cube_clean")
    if not isinstance(output_repo_id, str) or not output_repo_id.strip():
        raise ConfigError("output.repo_id must be a non-empty string")
    output = OutputConfig(
        clean_root=_path(output_raw.get("clean_root", "data/cleaned"), "output.clean_root"),
        curated_root=_path(output_raw.get("curated_root", "data/curated"), "output.curated_root"),
        report_root=_path(output_raw.get("report_root", "data/reports"), "output.report_root"),
        repo_id=output_repo_id.strip(),
        overwrite=_boolean(output_raw.get("overwrite", False), "output.overwrite"),
    )

    conversion_raw = _section(raw, "conversion", required=False)
    _reject_unknown(
        conversion_raw,
        {"parallel_encoding", "validate_after", "video_corr_min", "video_mae_max"},
        "conversion",
    )
    conversion = ConversionConfig(
        parallel_encoding=_boolean(
            conversion_raw.get("parallel_encoding", False), "conversion.parallel_encoding"
        ),
        validate_after=_boolean(
            conversion_raw.get("validate_after", True), "conversion.validate_after"
        ),
        video_corr_min=_number(
            conversion_raw.get("video_corr_min", 0.95), "conversion.video_corr_min"
        ),
        video_mae_max=_number(
            conversion_raw.get("video_mae_max", 10.0), "conversion.video_mae_max"
        ),
    )
    if not 0 <= conversion.video_corr_min <= 1:
        raise ConfigError("conversion.video_corr_min must be in [0, 1]")
    if conversion.video_mae_max < 0:
        raise ConfigError("conversion.video_mae_max must be >= 0")

    push_raw = _section(raw, "push", required=False)
    _reject_unknown(push_raw, {"enabled", "private"}, "push")
    push = PushConfig(
        enabled=_boolean(push_raw.get("enabled", False), "push.enabled"),
        private=_boolean(push_raw.get("private", True), "push.private"),
    )

    return PipelineConfig(
        source=source,
        cleaning=cleaning,
        output=output,
        conversion=conversion,
        push=push,
    )
