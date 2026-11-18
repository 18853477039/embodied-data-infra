"""Unified command-line entry point for collection, cleaning, and training."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import traceback
from pathlib import Path

from .cleaning.cleaning import CleanError, run_clean
from .cleaning.config import ConfigError, load_config
from .cleaning.conversion import ConversionError, resolve_clean_run, run_convert
from .cleaning.layout import LayoutError, write_json
from .cleaning.report import decision_table
from .cleaning.source import SourceError
from .cleaning.validation import ValidationError, validate_clean_contract, validate_curated
from .collection.runner import CollectionError, run_record
from .training.runner import TrainingError, run_train

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except Exception:  # noqa: BLE001 - optional logging polish
        logger.debug("PyAV not available; skipping its log-level configuration")


def _extra_args(values: list[str]) -> list[str]:
    """Drop the ``--`` separator that argparse keeps in a REMAINDER list."""
    return values[1:] if values and values[0] == "--" else values


def _print_clean_summary(result) -> None:
    print("\n" + decision_table(result.metrics))
    dropped = int((result.metrics["decision"] == "drop").sum())
    print(
        f"\nEpisodes: {result.metrics.height} in -> {result.kept_episodes} kept / {dropped} dropped"
    )
    print(f"Frames: {int(result.metrics['n_frames'].sum())} in -> {result.kept_frames} out")
    if result.dry_run:
        print(f"Dry run (no cleaned data written). Report: {result.report_dir}")
    else:
        print(f"Cleaned layer: {result.final_dir}")
        print(f"QC report:     {result.report_dir}")
    print(f"Run key:       {result.run_key}")


def _cmd_collect(args: argparse.Namespace) -> int:
    result = run_record(
        args.config,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        num_episodes=args.episodes,
        push_to_hub=args.push_to_hub,
        extra_args=_extra_args(args.extra),
        dry_run=args.dry_run,
    )
    print(f"Command: {shlex.join(result.command)}")
    if result.dry_run:
        print("Dry run: recording not started.")
        return 0
    return 0 if result.returncode == 0 else 1


def _cmd_clean(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = run_clean(cfg, dry_run=args.dry_run, overwrite=args.overwrite)
    _print_clean_summary(result)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = run_convert(
        cfg,
        clean_run=args.clean_run,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )
    print(f"Curated dataset: {result.dataset_dir}")
    print(f"Conversion run:  {result.final_dir}")
    print(f"Validation:      {'PASS' if result.validation.get('passed') else 'FAIL'}")
    if not result.validation.get("passed"):
        for failure in result.validation.get("failures", []):
            print(f"  - {failure}")
        return 1
    print(f"Reused:          {result.reused}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dataset = Path(args.dataset).expanduser().resolve()
    if args.clean_run:
        clean_dir = Path(args.clean_run).expanduser().resolve()
    else:
        conversion_file = dataset.parent / "conversion.json"
        if dataset.name == "dataset" and conversion_file.is_file():
            clean_dir = Path(
                json.loads(conversion_file.read_text(encoding="utf-8"))["clean_run"]
            ).expanduser()
        else:
            clean_dir = resolve_clean_run(cfg)
    layer = validate_clean_contract(clean_dir)
    result = validate_curated(dataset, layer, cfg)
    write_json(dataset.parent / "validation.json", result)
    print(f"Validation: {'PASS' if result.get('passed') else 'FAIL'}")
    print(
        f"Episodes:   {result.get('actual_episodes')} (expected {result.get('expected_episodes')})"
    )
    print(f"Frames:     {result.get('actual_frames')} (expected {result.get('expected_frames')})")
    if not result.get("passed"):
        for failure in result.get("failures", []):
            print(f"  - {failure}")
        return 1
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    result = run_train(
        args.config,
        dataset_root=args.dataset_root,
        latest_curated=args.latest_curated,
        curated_root=args.curated_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        job_name=args.job_name,
        resume=args.resume,
        extra_args=_extra_args(args.extra),
        dry_run=args.dry_run,
    )
    print(f"Command: {shlex.join(result.command)}")
    if result.dry_run:
        print("Dry run: training not started.")
        return 0
    return 0 if result.returncode == 0 else 1


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")


def _add_passthrough(parser: argparse.ArgumentParser, target: str) -> None:
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help=f"Arguments after `--` are forwarded verbatim to {target}",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodied",
        description="Collection, cleaning, conversion, and training for LeRobot datasets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Record raw episodes with lerobot-record")
    collect.add_argument("--config", required=True, type=Path, help="Collection JSON config")
    collect.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root")
    collect.add_argument("--repo-id", default=None, help="Override dataset repo id")
    collect.add_argument("--episodes", type=int, default=None, help="Override episode count")
    collect.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dataset.push_to_hub",
    )
    collect.add_argument("--dry-run", action="store_true", help="Print the command only")
    _add_verbose(collect)
    _add_passthrough(collect, "lerobot-record")

    clean = sub.add_parser("clean", help="raw -> cleaned parquet + clips + manifest")
    clean.add_argument("--config", required=True, type=Path, help="Cleaning YAML config")
    clean.add_argument("--dry-run", action="store_true", help="Compute decisions and write QC only")
    clean.add_argument("--overwrite", action="store_true", help="Replace an existing cleaned layer")
    _add_verbose(clean)

    convert = sub.add_parser("convert", help="cleaned layer -> canonical LeRobot v3 dataset")
    convert.add_argument("--config", required=True, type=Path, help="Cleaning YAML config")
    convert.add_argument("--clean-run", default=None, help="Explicit cleaned-layer directory")
    convert.add_argument(
        "--overwrite", action="store_true", help="Replace an existing curated dataset"
    )
    convert.add_argument("--push-to-hub", action="store_true", help="Push after validation passes")
    _add_verbose(convert)

    validate = sub.add_parser("validate", help="Re-run read-back validation on a curated dataset")
    validate.add_argument("--config", required=True, type=Path, help="Cleaning YAML config")
    validate.add_argument(
        "--dataset", required=True, type=Path, help="Curated dataset directory (the one with meta/)"
    )
    validate.add_argument("--clean-run", default=None, help="Explicit cleaned-layer directory")
    _add_verbose(validate)

    train = sub.add_parser("train", help="Train a policy with lerobot-train")
    train.add_argument("--config", required=True, type=Path, help="Training JSON config")
    dataset_group = train.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset-root", type=Path, default=None, help="Local curated dataset directory"
    )
    dataset_group.add_argument(
        "--latest-curated",
        action="store_true",
        help="Use the newest dataset under --curated-root",
    )
    train.add_argument(
        "--curated-root",
        type=Path,
        default=Path("data/curated"),
        help="Root used by --latest-curated (default: data/curated)",
    )
    train.add_argument("--repo-id", default=None, help="Override dataset repo id")
    train.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    train.add_argument("--job-name", default=None, help="Override job name")
    train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override resume behaviour",
    )
    train.add_argument("--dry-run", action="store_true", help="Print the command only")
    _add_verbose(train)
    _add_passthrough(train, "lerobot-train")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    handlers = {
        "collect": _cmd_collect,
        "clean": _cmd_clean,
        "convert": _cmd_convert,
        "validate": _cmd_validate,
        "train": _cmd_train,
    }
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    except (
        SourceError,
        CleanError,
        ConversionError,
        LayoutError,
        CollectionError,
        TrainingError,
    ) as exc:
        logger.error("%s", exc)
        return 2
    except ValidationError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error: %s", exc)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
