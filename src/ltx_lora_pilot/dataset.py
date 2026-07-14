from __future__ import annotations

import shutil
from pathlib import Path


ALLOWED_OUTPUT_DIRECTORIES = {"training", "holdout"}


def safe_reset_output_directory(output_root: Path, candidate: Path) -> None:
    root = output_root.resolve()
    target = candidate.resolve()
    if target.parent != root or target.name not in ALLOWED_OUTPUT_DIRECTORIES:
        raise ValueError(f"refusing to reset unsafe dataset path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
