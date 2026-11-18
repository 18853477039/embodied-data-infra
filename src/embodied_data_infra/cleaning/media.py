"""Video decode, episode slicing, clip encoding, and frame-count verification.

Stage 1 slices source episodes by integer frame index, never by float seek
boundaries. That keeps trims exactly aligned with the parquet rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import av
import numpy as np
from lerobot.configs.video import VideoEncoderConfig, encoder_config_from_video_info

from .config import CleaningConfig
from .source import SourceDataset, SourceError


class MediaError(RuntimeError):
    """Raised when a video cannot be decoded, encoded, or verified."""


def build_encoder_configs_from_features(
    features: dict, *, crf: int | None
) -> dict[str, VideoEncoderConfig]:
    """Build resolved encoder configs, one per video camera.

    `video.codec` is stored as a stream codec name ("av1"). Without
    `resolve_vcodec()`, PyAV would pick libaom-av1; the source was encoded with
    libsvtav1 and its preset/CRF options are SVT-specific. Passing `crf=None`
    keeps the source CRF (used for the canonical conversion output).
    """
    configs: dict[str, VideoEncoderConfig] = {}
    for key, feature in features.items():
        if feature.get("dtype") != "video":
            continue
        info = feature.get("info") or {}
        cfg = encoder_config_from_video_info(info)
        cfg.resolve_vcodec()
        if crf is not None:
            cfg.crf = crf
        cfg.validate()
        configs[key] = cfg

    signatures = {(cfg.vcodec, cfg.pix_fmt) for cfg in configs.values()}
    if len(signatures) > 1:
        raise SourceError(
            "Cameras use different video encoders/pixel formats; this demo uses one writer config. "
            f"Found: {sorted(signatures)}"
        )
    return configs


def build_encoder_configs(
    source: SourceDataset, cleaning: CleaningConfig
) -> dict[str, VideoEncoderConfig]:
    return build_encoder_configs_from_features(source.features, crf=cleaning.clip_crf)


def decode_video_all(path: Path) -> tuple[np.ndarray, float]:
    """Decode every frame of a video to uint8 HWC and return (frames, fps)."""
    frames: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Could not decode {path}: {exc}") from exc
    if not frames:
        raise MediaError(f"Video has no frames: {path}")
    return np.stack(frames), fps


def iter_clip_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield decoded uint8 HWC frames from a clip."""
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Could not decode {path}: {exc}") from exc


def probe_video(path: Path) -> dict:
    """Decode a video and report frame count, fps, and size."""
    count = 0
    fps = 0.0
    width = height = 0
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            width, height = int(stream.width), int(stream.height)
            for _ in container.decode(video=0):
                count += 1
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Could not probe {path}: {exc}") from exc
    return {"frames": count, "fps": fps, "width": width, "height": height}


def _episode_video_groups(source: SourceDataset, camera: str) -> dict[tuple[int, int], list[dict]]:
    columns = {
        "episode_index",
        "length",
        f"videos/{camera}/chunk_index",
        f"videos/{camera}/file_index",
        f"videos/{camera}/from_timestamp",
        f"videos/{camera}/to_timestamp",
    }
    missing = columns - set(source.episodes.columns)
    if missing:
        raise SourceError(
            f"Episode metadata is missing video columns for {camera}: {sorted(missing)}"
        )

    groups: dict[tuple[int, int], list[dict]] = {}
    for row in source.episodes.iter_rows(named=True):
        key = (int(row[f"videos/{camera}/chunk_index"]), int(row[f"videos/{camera}/file_index"]))
        groups.setdefault(key, []).append(
            {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "from_timestamp": float(row[f"videos/{camera}/from_timestamp"]),
                "to_timestamp": float(row[f"videos/{camera}/to_timestamp"]),
            }
        )
    for episodes in groups.values():
        episodes.sort(key=lambda item: (item["from_timestamp"], item["episode_index"]))
    return groups


def load_episode_frames(
    source: SourceDataset,
    camera: str,
    wanted_episodes: Iterable[int] | None = None,
) -> dict[int, np.ndarray]:
    """Decode each source video file once and slice out episodes by frame count.

    Demo scale: all source frames for a camera are held in memory (~32 MB for
    128x128x3x666). For larger datasets this should become seek/stream based.
    """
    wanted = None if wanted_episodes is None else set(int(ep) for ep in wanted_episodes)
    result: dict[int, np.ndarray] = {}
    for (chunk_index, file_index), episodes in _episode_video_groups(source, camera).items():
        path = source.video_path(camera, chunk_index, file_index)
        frames, fps = decode_video_all(path)
        if abs(fps - source.fps) > 0.01:
            raise MediaError(f"{path} is {fps} fps, expected {source.fps}")

        for episode in episodes:
            start = round(episode["from_timestamp"] * source.fps)
            end = round(episode["to_timestamp"] * source.fps)
            length = episode["length"]
            if end - start != length:
                raise MediaError(
                    f"Episode {episode['episode_index']} {camera}: metadata says {length} frames, "
                    f"timestamps imply {end - start} ({episode['from_timestamp']}..{episode['to_timestamp']})"
                )
            if start < 0 or end > len(frames):
                raise MediaError(
                    f"Episode {episode['episode_index']} {camera}: frame range {start}:{end} "
                    f"outside video with {len(frames)} frames"
                )
            out = frames[start:end]
            if len(out) != length:
                raise MediaError(
                    f"Episode {episode['episode_index']} {camera}: sliced {len(out)} frames, expected {length}"
                )
            if wanted is None or episode["episode_index"] in wanted:
                result[episode["episode_index"]] = out
    return result


def encode_clip(
    frames: np.ndarray,
    fps: int,
    out_path: Path,
    encoder: VideoEncoderConfig,
) -> None:
    """Encode uint8 HWC frames directly with PyAV and verify the frame count."""
    if len(frames) == 0:
        raise MediaError(f"Refusing to encode an empty clip: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    options = encoder.get_codec_options(encoder_threads=None, as_strings=True)
    try:
        with av.open(str(out_path), "w", options={"movflags": "faststart"}) as container:
            stream = container.add_stream(encoder.vcodec, rate=fps, options=options)
            stream.pix_fmt = encoder.pix_fmt
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            for array in frames:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Could not encode {out_path}: {exc}") from exc

    probe = probe_video(out_path)
    expected = {
        "frames": len(frames),
        "fps": float(fps),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
    }
    if probe["frames"] != expected["frames"]:
        raise MediaError(
            f"{out_path}: decoded {probe['frames']} frames, expected {expected['frames']}"
        )
    if abs(probe["fps"] - expected["fps"]) > 0.01:
        raise MediaError(f"{out_path}: fps {probe['fps']}, expected {expected['fps']}")
    if (probe["width"], probe["height"]) != (expected["width"], expected["height"]):
        raise MediaError(
            f"{out_path}: size {probe['width']}x{probe['height']}, "
            f"expected {expected['width']}x{expected['height']}"
        )


def encoder_signature(encoder: VideoEncoderConfig) -> dict:
    return {
        "vcodec": encoder.vcodec,
        "pix_fmt": encoder.pix_fmt,
        "g": encoder.g,
        "crf": encoder.crf,
        "preset": encoder.preset,
        "fast_decode": encoder.fast_decode,
        "extra_options": dict(encoder.extra_options),
    }
