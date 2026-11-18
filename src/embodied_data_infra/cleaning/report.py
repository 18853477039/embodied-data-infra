"""QC reports: machine-readable tables plus a human review document."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from .config import CleaningConfig


def rule_hit_counts(metrics: pl.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for reasons in metrics["reasons"].to_list():
        counter.update(reasons or [])
    return dict(sorted(counter.items()))


def summary_dict(
    metrics: pl.DataFrame,
    cfg: CleaningConfig,
    *,
    run_key: str,
    source_repo: str,
    source_revision: str,
    source_fp: str,
    dry_run: bool,
) -> dict[str, Any]:
    kept = metrics.filter(pl.col("decision") != "drop")
    dropped = metrics.filter(pl.col("decision") == "drop")
    trimmed = metrics.filter(pl.col("decision") == "trim")
    decisions = {
        "keep": int((metrics["decision"] == "keep").sum()),
        "trim": trimmed.height,
        "drop": dropped.height,
    }
    return {
        "run_key": run_key,
        "dry_run": dry_run,
        "source": {
            "repo_id": source_repo,
            "revision": source_revision,
            "fingerprint": source_fp,
        },
        "episodes": {
            "input": metrics.height,
            "kept": kept.height,
            "trimmed": trimmed.height,
            "dropped": dropped.height,
            "dropped_episodes": dropped["episode_index"].to_list(),
            "decisions": decisions,
        },
        "frames": {
            "input": int(metrics["n_frames"].sum()),
            "output": int(kept["n_frames_out"].sum()),
            "trimmed": int((metrics["trim_head"] + metrics["trim_tail"]).sum()),
        },
        "rule_hits": rule_hit_counts(metrics),
        "thresholds": asdict(cfg),
    }


def decision_table(metrics: pl.DataFrame) -> str:
    lines = [
        "| ep | in | decision | trim(head/tail) | out | reasons | warnings |",
        "|---:|---:|---|---:|---:|---|---|",
    ]
    for row in metrics.sort("episode_index").iter_rows(named=True):
        reasons = ", ".join(row["reasons"] or []) or "-"
        warnings = ", ".join(row["warnings"] or []) or "-"
        trim = f"{row['trim_head']}/{row['trim_tail']}"
        lines.append(
            f"| {row['episode_index']} | {row['n_frames']} | {row['decision']} | {trim} | "
            f"{row['n_frames_out']} | {reasons} | {warnings} |"
        )
    return "\n".join(lines)


def format_report_md(
    metrics: pl.DataFrame,
    cfg: CleaningConfig,
    *,
    run_key: str,
    source_repo: str,
    source_revision: str,
    source_fp: str,
    dry_run: bool,
) -> str:
    summary = summary_dict(
        metrics,
        cfg,
        run_key=run_key,
        source_repo=source_repo,
        source_revision=source_revision,
        source_fp=source_fp,
        dry_run=dry_run,
    )
    hits = summary["rule_hits"]
    hit_lines = "\n".join(f"- `{name}`: {count}" for name, count in hits.items()) or "- none"
    idle = cfg.idle
    action = cfg.action
    return f"""# Pick Cube cleaning QC report

- Source: `{source_repo}@{source_revision}` (fingerprint `{source_fp}`)
- Run key: `{run_key}`
- Mode: {"dry-run" if dry_run else "materialized"}
- Episodes: {summary["episodes"]["input"]} in / {summary["episodes"]["kept"]} kept / {summary["episodes"]["trimmed"]} trimmed / {summary["episodes"]["dropped"]} dropped
- Decisions: {summary["episodes"]["decisions"]}
- Frames: {summary["frames"]["input"]} in / {summary["frames"]["output"]} out / {summary["frames"]["trimmed"]} trimmed
- Dropped episodes: {summary["episodes"]["dropped_episodes"] or "none"}

## Rule hits

{hit_lines}

## Decisions

{decision_table(metrics)}

## Thresholds used

- Success required: `{cfg.success.require}` (last `{cfg.success.done_key}` and `{cfg.success.reward_key} > 0`)
- Length bounds: `[{cfg.min_frames}, {cfg.max_frames}]` frames; duration outlier: `> {cfg.duration_outlier_factor} x median`
- Action bounds: `{action.bounds}`; allowed values: `{action.allowed_values}`; atol `{action.atol}`
- Idle: motion dims `{idle.state_motion_dims}`, eps `{idle.motion_eps}`; translation dims `{idle.action_translation_dims}`, eps `{idle.action_eps}`
- Trim: leading `{idle.trim_leading}`, trailing `{idle.trim_trailing}`, min run `{idle.min_trim_frames}` frames, max trim ratio `{idle.max_trim_ratio}`
- Interior idle drop: ratio `> {idle.interior_idle_ratio_max}` or run `> {idle.interior_idle_run_max}` frames
- Stage-1 clip CRF: `{cfg.clip_crf}` (near-lossless; conversion re-encodes canonical output)

## Notes

- `is_motionless` uses the L1 sum of joint-state deltas (`state_motion_dims`), not max-abs.
- Interior idle is evaluated after proposed head/tail trims.
- Startup idle shorter than `min_trim_frames` is intentionally kept: every episode in this
  dataset has 4-7 normal startup frames, and trimming them all would delete valid observations.
"""


def write_qc(
    out_dir: Path,
    metrics: pl.DataFrame,
    cfg: CleaningConfig,
    *,
    run_key: str,
    source_repo: str,
    source_revision: str,
    source_fp: str,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_dict(
        metrics,
        cfg,
        run_key=run_key,
        source_repo=source_repo,
        source_revision=source_revision,
        source_fp=source_fp,
        dry_run=dry_run,
    )
    metrics.write_parquet(out_dir / "episodes.parquet")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        format_report_md(
            metrics,
            cfg,
            run_key=run_key,
            source_repo=source_repo,
            source_revision=source_revision,
            source_fp=source_fp,
            dry_run=dry_run,
        ),
        encoding="utf-8",
    )
    return summary
