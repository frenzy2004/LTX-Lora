from __future__ import annotations

import argparse
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

from ltx_lora_pilot.a2v_bundle import (
    build_dataset_manifest,
    build_root_manifest,
    build_training_archive,
    compute_bundle_id,
)
from ltx_lora_pilot.a2v_quality import validate_quality_and_splits
from ltx_lora_pilot.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    strict_load_json,
)
from ltx_lora_pilot.provider_validation import (
    validate_provider_validation_selection,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = frozenset(
    {"training-data.zip", "dataset-manifest.json", "bundle-manifest.json"}
)


class _NeutralArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "A2V_BUNDLE_ARGUMENT_ERROR\n")


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_regular_file(path: Path) -> None:
    if _is_symlink_or_junction(path):
        raise ValueError("private bundle input must not be a link")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError("required private bundle input is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("required private bundle input must be a regular file")


def _require_regular_directory(path: Path) -> None:
    if _is_symlink_or_junction(path) or not path.is_dir():
        raise ValueError("required private bundle directory is unavailable")


def _load_canonical_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    value = strict_load_json(path)
    if type(value) is not dict:
        raise ValueError("private bundle JSON input must be an object")
    if path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("private bundle JSON input is not canonical")
    return value


def _prepare_bundle_directory(run_dir: Path) -> Path:
    _require_regular_directory(run_dir)
    bundle_dir = run_dir / "bundle"
    if bundle_dir.exists() or _is_symlink_or_junction(bundle_dir):
        _require_regular_directory(bundle_dir)
        unexpected = sorted(path.name for path in bundle_dir.iterdir() if path.name not in OUTPUT_NAMES)
        if unexpected:
            raise ValueError("bundle directory contains unexpected entries")
        for path in bundle_dir.iterdir():
            if _is_symlink_or_junction(path) or (path.exists() and not path.is_file()):
                raise ValueError("bundle output destination is not a regular file")
    else:
        bundle_dir.mkdir()
    if bundle_dir.resolve(strict=True).parent != run_dir.resolve(strict=True):
        raise ValueError("bundle directory escapes the private run directory")
    return bundle_dir


def _resolved_groups(
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    train_ids = set(quality_summary["accepted_train_group_ids"])
    holdout_ids = set(quality_summary["accepted_holdout_group_ids"])
    result = []
    for group in structural_report["groups"]:
        group_id = group["group_id"]
        if group_id in train_ids:
            split = "train"
        elif group_id in holdout_ids:
            split = "holdout"
        else:
            continue
        result.append(
            {
                "group_id": group_id,
                "split": split,
                "files": [
                    {**digest, "path": candidate_dir / digest["name"]}
                    for digest in group["files"]
                ],
            }
        )
    return result


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit, re.ASCII) is None:
        raise ValueError("repository commit is unavailable")
    return commit


def _build(run_dir: Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    bundle_dir = _prepare_bundle_directory(run_dir)
    candidate_dir = run_dir / "candidates"
    control_dir = run_dir / "control"
    validation_dir = run_dir / "validation"
    for directory in (candidate_dir, control_dir, validation_dir):
        _require_regular_directory(directory)

    plan_path = run_dir / "plan.md"
    structural_path = control_dir / "structural-report.json"
    attestation_path = control_dir / "quality-attestation.json"
    policy_path = control_dir / "standing-authorization.json"
    price_path = control_dir / "price-evidence.json"
    execution_config_path = control_dir / "execution-config.json"
    selection_path = validation_dir / "provider-validation-selection.json"
    _require_regular_file(plan_path)
    structural_report = _load_canonical_json(structural_path)
    quality_attestation = _load_canonical_json(attestation_path)
    _load_canonical_json(policy_path)
    _load_canonical_json(price_path)
    execution_config = _load_canonical_json(execution_config_path)
    selection = _load_canonical_json(selection_path)

    quality_summary = validate_quality_and_splits(
        quality_attestation,
        structural_report,
    )
    validate_provider_validation_selection(
        selection,
        structural_report,
        quality_summary,
        execution_config,
        candidate_dir,
    )
    groups = _resolved_groups(structural_report, quality_summary, candidate_dir)
    archive_path = bundle_dir / "training-data.zip"
    archive_digest = build_training_archive(groups, archive_path)
    dataset_manifest = build_dataset_manifest(
        structural_report,
        quality_attestation,
        archive_digest,
        candidate_dir=candidate_dir,
    )
    dataset_manifest_path = bundle_dir / "dataset-manifest.json"
    atomic_write_json(dataset_manifest_path, dataset_manifest)

    artifacts = {
        "plan": sha256_file(plan_path),
        "standing_authorization": sha256_file(policy_path),
        "price_evidence": sha256_file(price_path),
        "structural_report": sha256_file(structural_path),
        "quality_attestation": sha256_file(attestation_path),
        "dataset_manifest": sha256_file(dataset_manifest_path),
        "training_archive": archive_digest,
        "execution_config": sha256_file(execution_config_path),
        "provider_validation_selection": sha256_file(selection_path),
    }
    root_manifest = build_root_manifest(
        execution_id=execution_config["execution_id"],
        created_at_utc=execution_config["created_at_utc"],
        expires_at_utc=execution_config["expires_at_utc"],
        repository_commit=_repository_commit(),
        artifacts=artifacts,
        holdout_groups=dataset_manifest["groups"]["holdout"],
    )
    atomic_write_json(bundle_dir / "bundle-manifest.json", root_manifest)
    return {"status": "built", "bundle_id": compute_bundle_id(root_manifest)}


def main() -> None:
    parser = _NeutralArgumentParser(
        description="Build a deterministic private A2V execution bundle"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        output = _build(args.run_dir)
    except Exception:
        parser.exit(2, "A2V_BUNDLE_BUILD_FAILED\n")
    print(canonical_json_bytes(output).decode("utf-8"))


if __name__ == "__main__":
    main()
