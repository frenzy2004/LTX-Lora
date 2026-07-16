from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

from .a2v_dataset import A2VSpec, validate_a2v_directory
from .a2v_quality import validate_quality_and_splits
from .artifacts import canonical_json_bytes
from .authorization import validate_execution_config
from .preflight import verify_static_a2v_bundle
from .private_workspace import require_canonical_run_dir
from .staging import _copy_sealed_file


@dataclass(frozen=True)
class SourceRunSnapshot:
    run_dir: Path
    structural_report: dict[str, Any]
    quality_attestation: dict[str, Any]
    quality_summary: dict[str, Any]
    source_config: dict[str, Any]


def _require_dataset_contract(
    run_dir: Path,
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
) -> set[str]:
    groups = structural_report.get("groups")
    train_ids = quality_summary.get("accepted_train_group_ids")
    holdout_ids = quality_summary.get("accepted_holdout_group_ids")
    if (
        type(groups) is not list
        or type(train_ids) is not list
        or type(holdout_ids) is not list
        or len(groups) != 17
        or len(train_ids) != 12
        or len(holdout_ids) != 5
    ):
        raise ValueError("source dataset must contain exactly 17 accepted groups")
    group_ids = [group.get("group_id") for group in groups if type(group) is dict]
    if len(group_ids) != 17 or len(set(group_ids)) != 17:
        raise ValueError("source structural groups are invalid")
    if set(train_ids) & set(holdout_ids) or set(train_ids) | set(holdout_ids) != set(group_ids):
        raise ValueError("source split must cover the exact structural group set")
    expected_names: set[str] = set()
    for group in groups:
        if type(group) is not dict or type(group.get("files")) is not list:
            raise ValueError("source structural groups are invalid")
        files = group["files"]
        if len(files) != 4:
            raise ValueError("source groups must contain exactly four files")
        for record in files:
            if type(record) is not dict or type(record.get("name")) is not str:
                raise ValueError("source structural files are invalid")
            expected_names.add(record["name"])
    if len(expected_names) != 68:
        raise ValueError("source dataset must contain exactly 68 candidate files")
    candidate_dir = Path(run_dir) / "candidates"
    try:
        actual_names = {path.name for path in candidate_dir.iterdir()}
    except OSError as exc:
        raise ValueError("source candidate directory is unavailable") from exc
    if actual_names != expected_names or len(actual_names) != 68:
        raise ValueError("source candidate directory does not match the sealed groups")
    return expected_names


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_junction = getattr(os.path, "isjunction", None)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or bool(is_junction is not None and is_junction(path))
    )


def _has_ads_syntax(path: Path) -> bool:
    parts = path.parts[1:] if path.anchor else path.parts
    return any(":" in part for part in parts)


def _has_case_alias(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        try:
            with os.scandir(current) as entries:
                matching = {
                    entry.name
                    for entry in entries
                    if entry.name.casefold() == part.casefold()
                }
        except OSError:
            return True
        if matching != {part}:
            return True
        current /= part
    return False


def _has_alias_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _is_reparse_or_link(current):
            return True
    return False


def _require_canonical_source_run(run_dir: Path) -> Path:
    candidate = Path(run_dir)
    raw = str(candidate)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or _has_ads_syntax(candidate)
    ):
        raise ValueError("source run directory is invalid")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise ValueError("source run directory is invalid")
    if _has_alias_component(absolute) or _has_case_alias(absolute):
        raise ValueError("source run directory must be canonical")
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError("source run directory is unavailable") from exc
    if (
        _is_reparse_or_link(absolute)
        or not stat.S_ISDIR(metadata.st_mode)
        or str(resolved) != str(absolute)
    ):
        raise ValueError("source run directory must be canonical")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(parent).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _nearest_existing_ancestor(path: Path) -> Path:
    current = Path(path)
    while not current.exists() and not current.is_symlink():
        parent = current.parent
        if parent == current:
            raise ValueError("candidate staging destination is invalid")
        current = parent
    return current


def _prepare_empty_destination(destination: Path, *, prohibited_root: Path) -> Path:
    candidate = Path(destination)
    raw = str(candidate)
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or _has_ads_syntax(candidate)
    ):
        raise ValueError("candidate staging destination is invalid")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise ValueError("candidate staging destination is invalid")
    if _is_within(absolute, prohibited_root):
        raise ValueError("candidate staging destination must not be within the source run")
    existing_parent = _nearest_existing_ancestor(absolute.parent)
    if _has_alias_component(existing_parent) or _has_case_alias(existing_parent):
        raise ValueError("candidate staging destination is invalid")
    try:
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if absolute.exists() or absolute.is_symlink():
            metadata = absolute.lstat()
            if _is_reparse_or_link(absolute) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("candidate staging destination is invalid")
            if any(absolute.iterdir()):
                raise ValueError("candidate staging destination must be empty")
        else:
            absolute.mkdir(mode=0o700)
            if os.name != "nt":
                os.chmod(absolute, 0o700)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("candidate staging destination is invalid") from exc
    if _has_alias_component(absolute) or _has_case_alias(absolute):
        raise ValueError("candidate staging destination is invalid")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError("candidate staging destination is invalid") from exc
    if str(resolved) != str(absolute):
        raise ValueError("candidate staging destination is invalid")
    if _is_within(resolved, prohibited_root):
        raise ValueError("candidate staging destination must not be within the source run")
    return resolved


def _snapshot_copy_inputs(
    snapshot: SourceRunSnapshot,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str]]:
    if not isinstance(snapshot, SourceRunSnapshot):
        raise ValueError("source snapshot is invalid")
    structural = copy.deepcopy(snapshot.structural_report)
    attestation = copy.deepcopy(snapshot.quality_attestation)
    quality_summary = validate_quality_and_splits(attestation, structural)
    if canonical_json_bytes(quality_summary) != canonical_json_bytes(snapshot.quality_summary):
        raise ValueError("source quality split changed after verification")
    source_config = validate_execution_config(copy.deepcopy(snapshot.source_config))
    expected_names = _require_dataset_contract(
        snapshot.run_dir,
        structural,
        quality_summary,
    )
    return structural, attestation, source_config, expected_names


def verify_source_run_static(
    *,
    private_root: Path,
    pilot_id: str,
    source_execution_id: str,
    expected_source_bundle_id: str,
) -> SourceRunSnapshot:
    """Return a detached, static snapshot of the accepted source candidate set."""

    root = Path(private_root)
    run_dir = require_canonical_run_dir(
        root,
        pilot_id,
        source_execution_id,
        root / "pilots" / pilot_id / "runs" / source_execution_id,
    )
    bundle = verify_static_a2v_bundle(root, run_dir, expected_source_bundle_id)
    _require_dataset_contract(
        bundle.run_dir,
        bundle.structural_report,
        bundle.quality_summary,
    )
    return SourceRunSnapshot(
        run_dir=bundle.run_dir,
        structural_report=copy.deepcopy(bundle.structural_report),
        quality_attestation=copy.deepcopy(bundle.quality_attestation),
        quality_summary=copy.deepcopy(bundle.quality_summary),
        source_config=copy.deepcopy(bundle.execution_config),
    )


def copy_accepted_candidates(
    snapshot: SourceRunSnapshot,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Securely copy the exact accepted candidate set into an empty staging directory."""

    if not isinstance(snapshot, SourceRunSnapshot):
        raise ValueError("source snapshot is invalid")
    source_run = _require_canonical_source_run(snapshot.run_dir)
    structural, attestation, source_config, expected_names = _snapshot_copy_inputs(snapshot)
    target = _prepare_empty_destination(destination, prohibited_root=source_run)
    source_dir = source_run / "candidates"
    if _is_reparse_or_link(source_dir) or not source_dir.is_dir():
        raise ValueError("source candidate directory is invalid")
    try:
        source_names = {path.name for path in source_dir.iterdir()}
    except OSError as exc:
        raise ValueError("source candidate directory is unavailable") from exc
    if source_names != expected_names or len(source_names) != 68:
        raise ValueError("source candidate directory does not match the accepted set")

    for group in structural["groups"]:
        for record in group["files"]:
            name = record["name"]
            _copy_sealed_file(
                source_dir / name,
                target / name,
                record,
                label="accepted source candidate",
            )

    try:
        copied_entries = list(target.iterdir())
    except OSError as exc:
        raise ValueError("copied candidate directory is unavailable") from exc
    if {path.name for path in copied_entries} != expected_names or len(copied_entries) != 68:
        raise ValueError("copied candidate directory does not match the accepted set")
    for path in copied_entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError("copied candidate is unavailable") from exc
        if (
            _is_reparse_or_link(path)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError("copied candidate is not an independent regular file")

    copied_structural = validate_a2v_directory(
        target,
        spec=A2VSpec(min_groups=17),
        trigger_phrase=source_config["trigger_phrase"],
    )
    if canonical_json_bytes(copied_structural) != canonical_json_bytes(structural):
        raise ValueError("copied candidate structural report changed")
    copied_summary = validate_quality_and_splits(attestation, copied_structural)
    if canonical_json_bytes(copied_summary) != canonical_json_bytes(snapshot.quality_summary):
        raise ValueError("copied candidate quality split changed")
    return copy.deepcopy(copied_structural), copy.deepcopy(attestation)
