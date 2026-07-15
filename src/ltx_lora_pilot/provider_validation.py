from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from .authorization import validate_execution_config
from .artifacts import canonical_json_bytes, safe_relative_name, sha256_file


SCHEMA_VERSION = "a2v-provider-validation-selection-v1"
CANONICAL_JSON_VERSION = 1
UUID4_HEX = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
GROUP_ID_PATTERN = re.compile(rf"grp_{UUID4_HEX}", re.ASCII)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
URL_PATTERN = re.compile(r"\b(?:https?|file|data):", re.IGNORECASE)

SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_json_version",
        "structural_report_sha256",
        "execution_config_sha256",
        "items",
    }
)
ITEM_FIELDS = frozenset({"group_id", "prompt", "image", "audio"})
DIGEST_FIELDS = frozenset({"name", "bytes", "sha256"})
STRUCTURAL_REPORT_FIELDS = frozenset(
    {"schema_version", "status", "spec", "groups"}
)
STRUCTURAL_GROUP_FIELDS = frozenset({"group_id", "files"})
STRUCTURAL_SPEC_FIELDS = frozenset(
    {"width", "height", "frames", "fps", "sample_rate"}
)
EXPECTED_STRUCTURAL_SPEC = {
    "width": 544,
    "height": 960,
    "frames": 89,
    "fps": 24,
    "sample_rate": 48_000,
}
QUALITY_SUMMARY_FIELDS = frozenset(
    {
        "status",
        "accepted_train_group_ids",
        "accepted_holdout_group_ids",
        "location_coverage",
        "coverage_counts",
    }
)
LOCATION_COVERAGE_FIELDS = frozenset({"isolated_holdout_group_ids"})
COVERAGE_COUNT_FIELDS = frozenset(
    {
        "accepted_train_groups",
        "accepted_holdout_groups",
        "location_isolated_holdout_groups",
        "training_teeth_or_inner_mouth_visible",
        "holdout_teeth_or_inner_mouth_visible",
    }
)
PROHIBITED_PROMPT_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co"})


def _exact_dict(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} must contain the exact fields")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _group_id(value: Any, *, label: str) -> str:
    if type(value) is not str or GROUP_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a machine-generated opaque group ID")
    return value


def _canonical_prompt(value: Any) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("provider validation prompt must be canonical prompt text")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("provider validation prompt must be NFC-normalized")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "provider validation prompt contains a prohibited Unicode character"
        ) from exc
    if len(encoded) > 1_024:
        raise ValueError("provider validation prompt must be at most 1024 UTF-8 bytes")
    if any(
        unicodedata.category(character) in PROHIBITED_PROMPT_CATEGORIES
        for character in value
    ):
        raise ValueError(
            "provider validation prompt contains a prohibited Unicode character"
        )
    if URL_PATTERN.search(value) is not None:
        raise ValueError("provider validation prompt must not contain a URL")
    return value


def _canonical_filename(
    value: Any,
    *,
    label: str,
    expected: str,
) -> str:
    if type(value) is not str or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a canonical local filename")
    try:
        safe_relative_name(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical local filename") from exc
    if value != expected:
        raise ValueError(f"{label} must be a canonical local filename")
    return value


def _digest_record(
    value: Any,
    *,
    label: str,
    expected_name: str,
) -> dict[str, Any]:
    record = _exact_dict(value, DIGEST_FIELDS, label=label)
    _canonical_filename(record["name"], label=label, expected=expected_name)
    if type(record["bytes"]) is not int or record["bytes"] <= 0:
        raise ValueError(f"{label} bytes must be a positive integer")
    _sha256(record["sha256"], label=f"{label} sha256")
    return record


def _canonical_group_id_list(value: Any, *, label: str) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a canonical group-ID list")
    result = [_group_id(item, label=label) for item in value]
    if result != sorted(result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must be a canonical group-ID list")
    return result


def _validated_quality_summary(value: Any) -> tuple[set[str], set[str]]:
    summary = _exact_dict(value, QUALITY_SUMMARY_FIELDS, label="quality summary")
    if summary["status"] != "valid":
        raise ValueError("quality summary status must be valid")
    train = _canonical_group_id_list(
        summary["accepted_train_group_ids"],
        label="accepted training group IDs",
    )
    holdout = _canonical_group_id_list(
        summary["accepted_holdout_group_ids"],
        label="accepted holdout group IDs",
    )
    if set(train) & set(holdout):
        raise ValueError("quality summary split group IDs must be disjoint")
    location = _exact_dict(
        summary["location_coverage"],
        LOCATION_COVERAGE_FIELDS,
        label="quality summary location coverage",
    )
    isolated = _canonical_group_id_list(
        location["isolated_holdout_group_ids"],
        label="isolated holdout group IDs",
    )
    if not set(isolated).issubset(holdout):
        raise ValueError("isolated holdout group IDs must be accepted holdouts")
    counts = _exact_dict(
        summary["coverage_counts"],
        COVERAGE_COUNT_FIELDS,
        label="quality summary coverage counts",
    )
    if any(type(counts[field]) is not int or counts[field] < 0 for field in counts):
        raise ValueError("quality summary coverage counts must be non-negative integers")
    expected_counts = {
        "accepted_train_groups": len(train),
        "accepted_holdout_groups": len(holdout),
        "location_isolated_holdout_groups": len(isolated),
    }
    if any(counts[field] != expected for field, expected in expected_counts.items()):
        raise ValueError("quality summary coverage counts mismatch")
    return set(train), set(holdout)


def _validated_structural_groups(value: Any) -> dict[str, dict[str, Any]]:
    report = _exact_dict(
        value,
        STRUCTURAL_REPORT_FIELDS,
        label="structural report",
    )
    if report["schema_version"] != "a2v-structural-report-v1":
        raise ValueError("structural report schema mismatch")
    if report["status"] != "valid":
        raise ValueError("structural report status must be valid")
    spec = _exact_dict(
        report["spec"],
        STRUCTURAL_SPEC_FIELDS,
        label="structural report spec",
    )
    if spec != EXPECTED_STRUCTURAL_SPEC or any(
        type(spec[field]) is not int for field in spec
    ):
        raise ValueError("structural report normalized spec mismatch")
    if type(report["groups"]) is not list:
        raise ValueError("structural report groups must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, group_value in enumerate(report["groups"]):
        group = _exact_dict(
            group_value,
            STRUCTURAL_GROUP_FIELDS,
            label=f"structural group {index}",
        )
        group_id = _group_id(group["group_id"], label="structural group_id")
        if group_id in result:
            raise ValueError("structural report contains a duplicate group ID")
        if type(group["files"]) is not list:
            raise ValueError(f"structural group {group_id} files must be a list")
        files: dict[str, dict[str, Any]] = {}
        for record_value in group["files"]:
            if type(record_value) is not dict:
                raise ValueError(f"structural group {group_id} file must be an object")
            name = record_value.get("name")
            if type(name) is not str:
                raise ValueError(f"structural group {group_id} file name is invalid")
            if name in files:
                raise ValueError(f"structural group {group_id} contains a duplicate filename")
            if name == f"{group_id}_start.png":
                kind = "image"
            elif name == f"{group_id}_audio.wav":
                kind = "audio"
            else:
                continue
            files[name] = _digest_record(
                record_value,
                label=f"structural {kind}",
                expected_name=name,
            )
        for expected in (f"{group_id}_start.png", f"{group_id}_audio.wav"):
            if expected not in files:
                raise ValueError(f"structural group {group_id} is missing provider media")
        result[group_id] = files
    return result


def _validate_execution_config_reference(value: Any) -> dict[str, Any]:
    return validate_execution_config(value)


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _candidate_path(candidate_dir: Path, name: str) -> Path:
    if _is_symlink_or_junction(candidate_dir) or not candidate_dir.is_dir():
        raise ValueError("candidate directory must be a regular local directory")
    path = candidate_dir / name
    if _is_symlink_or_junction(path):
        raise ValueError("provider validation candidate must not be a link or alias")
    try:
        metadata = path.stat()
        root = candidate_dir.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("provider validation candidate is unavailable") from exc
    if metadata.st_nlink != 1:
        raise ValueError("provider validation candidate must not be a link or alias")
    if not stat.S_ISREG(metadata.st_mode) or resolved.parent != root:
        raise ValueError("provider validation candidate must be a regular local file")
    return path


def _validate_current_candidate(path: Path, record: Mapping[str, Any]) -> None:
    digest = sha256_file(path)
    if digest.bytes != record["bytes"] or digest.sha256 != record["sha256"]:
        raise ValueError("provider validation current candidate bytes mismatch")


def _paths_alias(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError as exc:
        raise ValueError("provider validation candidate identity is unavailable") from exc


def validate_provider_validation_selection(
    selection: dict[str, Any],
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
    execution_config: dict[str, Any],
    candidate_dir: Path,
) -> dict[str, Any]:
    """Validate an immutable two-item Fal provider-validation selection."""

    value = _exact_dict(selection, SELECTION_FIELDS, label="selection")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("provider validation selection schema mismatch")
    if (
        type(value["canonical_json_version"]) is not int
        or value["canonical_json_version"] != CANONICAL_JSON_VERSION
    ):
        raise ValueError("provider validation selection canonical JSON version mismatch")

    _validate_execution_config_reference(execution_config)
    structural_digest = hashlib.sha256(
        canonical_json_bytes(structural_report)
    ).hexdigest()
    execution_config_digest = hashlib.sha256(
        canonical_json_bytes(execution_config)
    ).hexdigest()
    _sha256(
        value["structural_report_sha256"],
        label="structural_report_sha256",
    )
    _sha256(
        value["execution_config_sha256"],
        label="execution_config_sha256",
    )
    if value["structural_report_sha256"] != structural_digest:
        raise ValueError("provider validation structural report digest mismatch")
    if value["execution_config_sha256"] != execution_config_digest:
        raise ValueError("provider validation execution config digest mismatch")

    _train_ids, accepted_holdout_ids = _validated_quality_summary(quality_summary)
    structural_groups = _validated_structural_groups(structural_report)
    items_value = value["items"]
    if type(items_value) is not list or len(items_value) != 2:
        raise ValueError("provider validation selection requires exactly two items")

    group_ids: list[str] = []
    selected_paths: list[Path] = []
    selected_hashes: list[str] = []
    candidate_root = Path(candidate_dir)
    for index, item_value in enumerate(items_value):
        item = _exact_dict(
            item_value,
            ITEM_FIELDS,
            label=f"selection item {index}",
        )
        group_id = _group_id(item["group_id"], label=f"selection item {index} group_id")
        if group_id not in structural_groups:
            raise ValueError("provider validation selection references a missing structural group")
        if group_id not in accepted_holdout_ids:
            raise ValueError("provider validation selection requires an accepted holdout group")
        _canonical_prompt(item["prompt"])
        structural_files = structural_groups[group_id]
        for role, expected_name in (
            ("image", f"{group_id}_start.png"),
            ("audio", f"{group_id}_audio.wav"),
        ):
            record = _digest_record(
                item[role],
                label=f"selection item {index} {role}",
                expected_name=expected_name,
            )
            if record != structural_files[expected_name]:
                raise ValueError(
                    f"provider validation {role} structural record mismatch"
                )
            path = _candidate_path(candidate_root, expected_name)
            _validate_current_candidate(path, record)
            selected_paths.append(path)
            selected_hashes.append(record["sha256"])
        group_ids.append(group_id)

    if len(set(group_ids)) != 2:
        raise ValueError("provider validation selection requires distinct holdout groups")
    if group_ids != sorted(group_ids):
        raise ValueError("provider validation selection requires canonical group-ID order")
    for index, first in enumerate(selected_paths):
        for second in selected_paths[index + 1 :]:
            if _paths_alias(first, second):
                raise ValueError(
                    "provider validation candidate must not be a link or alias"
                )
    if len(set(selected_hashes)) != len(selected_hashes):
        raise ValueError("provider validation selection contains duplicate selected media")

    canonical_json_bytes(value)
    return copy.deepcopy(value)


def build_provider_validation_selection(
    *,
    structural_report: dict[str, Any],
    quality_summary: dict[str, Any],
    execution_config: dict[str, Any],
    candidate_dir: Path,
    prompts: Mapping[str, str],
) -> dict[str, Any]:
    """Build a canonical selection from exactly two accepted holdout prompts."""

    if type(prompts) is not dict or len(prompts) != 2:
        raise ValueError("provider validation selection requires exactly two prompts")
    _validated_quality_summary(quality_summary)
    structural_groups = _validated_structural_groups(structural_report)
    _validate_execution_config_reference(execution_config)
    items = []
    for group_id in sorted(prompts):
        _group_id(group_id, label="provider validation prompt group_id")
        if group_id not in structural_groups:
            raise ValueError("provider validation selection references a missing structural group")
        prompt = _canonical_prompt(prompts[group_id])
        files = structural_groups[group_id]
        items.append(
            {
                "group_id": group_id,
                "prompt": prompt,
                "image": copy.deepcopy(files[f"{group_id}_start.png"]),
                "audio": copy.deepcopy(files[f"{group_id}_audio.wav"]),
            }
        )
    selection = {
        "schema_version": SCHEMA_VERSION,
        "canonical_json_version": CANONICAL_JSON_VERSION,
        "structural_report_sha256": hashlib.sha256(
            canonical_json_bytes(structural_report)
        ).hexdigest(),
        "execution_config_sha256": hashlib.sha256(
            canonical_json_bytes(execution_config)
        ).hexdigest(),
        "items": items,
    }
    return validate_provider_validation_selection(
        selection,
        structural_report,
        quality_summary,
        execution_config,
        candidate_dir,
    )
