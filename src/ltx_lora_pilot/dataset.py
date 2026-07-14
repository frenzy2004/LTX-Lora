from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


ALLOWED_OUTPUT_DIRECTORIES = {"training", "holdout"}


def portrait_video_filter() -> str:
    return (
        "fps=24,"
        "crop='if(gt(iw/ih,9/16),ih*9/16,iw)':"
        "'if(gt(iw/ih,9/16),ih,iw*16/9)':"
        "'(iw-ow)/2':'(ih-oh)/2',"
        "scale=720:1280"
    )


def select_records(
    records: list[dict[str, Any]],
    training_ids: list[str],
    holdout_ids: list[str],
    *,
    clip_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlap = set(training_ids) & set(holdout_ids)
    if overlap:
        raise ValueError(f"training and holdout selections overlap: {sorted(overlap)}")
    eligible = {
        record["source_id"]: record
        for record in records
        if float(record["duration"]) >= clip_seconds and min(int(record["width"]), int(record["height"])) >= 480
    }
    requested = training_ids + holdout_ids
    missing = [source_id for source_id in requested if source_id not in eligible]
    if missing:
        raise ValueError(f"missing or ineligible sources: {missing}")
    return [eligible[source_id] for source_id in training_ids], [eligible[source_id] for source_id in holdout_ids]


def safe_reset_output_directory(output_root: Path, candidate: Path) -> None:
    root = output_root.resolve()
    target = candidate.resolve()
    if target.parent != root or target.name not in ALLOWED_OUTPUT_DIRECTORIES:
        raise ValueError(f"refusing to reset unsafe dataset path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
