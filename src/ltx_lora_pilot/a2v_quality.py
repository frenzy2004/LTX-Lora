from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .a2v_dataset import GROUP_ID_PATTERN, STRUCTURAL_REPORT_SCHEMA
from .artifacts import safe_relative_name, strict_load_json


QUALITY_ATTESTATION_SCHEMA = "a2v-quality-attestation-v1"
ATTESTATION_KEYS = frozenset(
    {"schema_version", "dataset_id", "rights_and_consent", "groups"}
)
RIGHTS_KEYS = frozenset({"confirmed", "reviewer_id", "reviewed_at_utc"})
GROUP_KEYS = frozenset(
    {
        "group_id",
        "split",
        "accepted",
        "source_asset_id",
        "source_session_id",
        "location_id",
        "source_start_ms",
        "source_end_ms",
        "checks",
        "notes",
    }
)
REQUIRED_TRUE_CHECKS = frozenset(
    {
        "one_visible_speaker",
        "close_or_medium_close_framing",
        "face_mouth_jaw_cheeks_and_eyes_unobstructed",
        "continuous_real_speech_motion",
        "no_internal_cut",
        "no_overlapping_speaker_dubbing_or_music",
        "no_watermark_burned_captions_or_beauty_filter",
        "audio_and_video_are_from_the_same_interval",
        "rights_and_likeness_use_confirmed",
    }
)
TEETH_CHECK = "teeth_or_inner_mouth_visible"
CHECK_KEYS = REQUIRED_TRUE_CHECKS | {TEETH_CHECK}

STRUCTURAL_REPORT_KEYS = frozenset({"schema_version", "status", "spec", "groups"})
STRUCTURAL_SPEC_KEYS = frozenset({"width", "height", "frames", "fps", "sample_rate"})
STRUCTURAL_GROUP_KEYS = frozenset({"group_id", "files"})
DIGEST_KEYS = frozenset({"name", "bytes", "sha256"})
EXPECTED_SPEC = {
    "width": 544,
    "height": 960,
    "frames": 89,
    "fps": 24,
    "sample_rate": 48_000,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
OPAQUE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", re.ASCII)


def load_quality_attestation(path: Path) -> dict:
    value = strict_load_json(path)
    if type(value) is not dict:
        raise ValueError("quality attestation must be a JSON object")
    return value


def _exact_object(value: Any, expected: frozenset[str], *, label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    unknown = sorted(keys - expected)
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {unknown}")
    missing = sorted(expected - keys)
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _opaque_id(value: Any, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    if OPAQUE_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be a canonical opaque ID")
    return text


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _utc_timestamp(value: Any, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be UTC")
    return text


def _validated_attestation_groups(attestation: dict) -> list[dict]:
    _exact_object(attestation, ATTESTATION_KEYS, label="quality attestation")
    if attestation["schema_version"] != QUALITY_ATTESTATION_SCHEMA:
        raise ValueError("unsupported quality attestation schema_version")
    _nonempty_string(attestation["dataset_id"], label="dataset_id")

    rights = _exact_object(
        attestation["rights_and_consent"],
        RIGHTS_KEYS,
        label="rights_and_consent",
    )
    if not _boolean(rights["confirmed"], label="rights_and_consent.confirmed"):
        raise ValueError("rights and consent must be confirmed")
    _nonempty_string(rights["reviewer_id"], label="rights_and_consent.reviewer_id")
    _utc_timestamp(rights["reviewed_at_utc"], label="rights_and_consent.reviewed_at_utc")

    groups = attestation["groups"]
    if type(groups) is not list:
        raise ValueError("quality attestation groups must be a list")

    seen_ids: set[str] = set()
    validated: list[dict] = []
    for index, group_value in enumerate(groups):
        group = _exact_object(
            group_value,
            GROUP_KEYS,
            label=f"quality attestation group {index}",
        )
        group_id = _nonempty_string(group["group_id"], label="group_id")
        if GROUP_ID_PATTERN.fullmatch(group_id) is None:
            raise ValueError("quality attestation contains an unsafe group ID")
        if group_id in seen_ids:
            raise ValueError(f"duplicate group ID in quality attestation: {group_id}")
        seen_ids.add(group_id)

        if type(group["split"]) is not str or group["split"] not in {"train", "holdout"}:
            raise ValueError(f"group {group_id} split must be train or holdout")
        accepted = _boolean(group["accepted"], label=f"group {group_id} accepted")
        for field in ("source_asset_id", "source_session_id", "location_id"):
            _opaque_id(group[field], label=f"group {group_id} {field}")
        start_ms = _nonnegative_integer(
            group["source_start_ms"],
            label=f"group {group_id} source_start_ms",
        )
        end_ms = _nonnegative_integer(
            group["source_end_ms"],
            label=f"group {group_id} source_end_ms",
        )
        if end_ms <= start_ms:
            raise ValueError(f"group {group_id} source interval must have end after start")

        checks = group["checks"]
        if type(checks) is not dict:
            raise ValueError(f"group {group_id} checks must be an object")
        check_keys = set(checks)
        unknown_checks = sorted(check_keys - CHECK_KEYS)
        if unknown_checks:
            raise ValueError(f"group {group_id} checks contains unknown keys: {unknown_checks}")
        missing_checks = sorted(CHECK_KEYS - check_keys)
        if missing_checks:
            raise ValueError(f"group {group_id} is missing required check: {missing_checks}")
        for check_name in CHECK_KEYS:
            _boolean(checks[check_name], label=f"group {group_id} check {check_name}")
        if accepted and any(not checks[check_name] for check_name in REQUIRED_TRUE_CHECKS):
            raise ValueError(f"group {group_id} accepted group has a false required check")

        if type(group["notes"]) is not str:
            raise ValueError(f"group {group_id} notes must be a string")
        if not accepted and not group["notes"].strip():
            raise ValueError(f"group {group_id} rejected group requires a reason in notes")
        validated.append(group)
    return validated


def _validated_structural_groups(structural_report: dict) -> dict[str, dict]:
    _exact_object(
        structural_report,
        STRUCTURAL_REPORT_KEYS,
        label="structural report",
    )
    if structural_report["schema_version"] != STRUCTURAL_REPORT_SCHEMA:
        raise ValueError("unsupported structural report schema_version")
    if structural_report["status"] != "valid":
        raise ValueError("structural report status must be valid")
    spec = _exact_object(
        structural_report["spec"],
        STRUCTURAL_SPEC_KEYS,
        label="structural report spec",
    )
    if (
        any(type(spec[key]) is not int for key in STRUCTURAL_SPEC_KEYS)
        or spec != EXPECTED_SPEC
    ):
        raise ValueError("structural report does not use the exact normalized A2V spec")

    groups_value = structural_report["groups"]
    if type(groups_value) is not list:
        raise ValueError("structural report groups must be a list")
    groups: dict[str, dict] = {}
    for index, group_value in enumerate(groups_value):
        group = _exact_object(
            group_value,
            STRUCTURAL_GROUP_KEYS,
            label=f"structural group {index}",
        )
        group_id = _nonempty_string(group["group_id"], label="structural group_id")
        if GROUP_ID_PATTERN.fullmatch(group_id) is None:
            raise ValueError("structural report contains an unsafe group ID")
        if group_id in groups:
            raise ValueError(f"duplicate group ID in structural report: {group_id}")

        files = group["files"]
        if type(files) is not list or len(files) != 4:
            raise ValueError(f"structural group {group_id} must contain exactly four file digests")
        expected_names = {
            f"{group_id}_start.png",
            f"{group_id}_audio.wav",
            f"{group_id}_end.mp4",
            f"{group_id}.txt",
        }
        seen_names: set[str] = set()
        for file_index, digest_value in enumerate(files):
            digest = _exact_object(
                digest_value,
                DIGEST_KEYS,
                label=f"structural group {group_id} file {file_index}",
            )
            name = _nonempty_string(digest["name"], label=f"structural group {group_id} file name")
            safe_relative_name(name)
            if name in seen_names:
                raise ValueError(f"structural group {group_id} contains a duplicate filename")
            seen_names.add(name)
            byte_count = digest["bytes"]
            if type(byte_count) is not int or byte_count <= 0:
                raise ValueError(f"structural group {group_id} file bytes must be a positive integer")
            sha256 = digest["sha256"]
            if type(sha256) is not str or SHA256_PATTERN.fullmatch(sha256) is None:
                raise ValueError(f"structural group {group_id} file has an invalid SHA-256")
        if seen_names != expected_names:
            raise ValueError(f"structural group {group_id} does not contain the exact required filenames")
        groups[group_id] = group
    return groups


def _reject_cross_split_provenance(train: list[dict], holdout: list[dict]) -> None:
    train_sessions = {group["source_session_id"] for group in train}
    holdout_sessions = {group["source_session_id"] for group in holdout}
    if train_sessions & holdout_sessions:
        raise ValueError("source session crosses the train and holdout splits")

    for train_group in train:
        for holdout_group in holdout:
            if train_group["source_asset_id"] != holdout_group["source_asset_id"]:
                continue
            if (
                train_group["source_start_ms"] < holdout_group["source_end_ms"]
                and holdout_group["source_start_ms"] < train_group["source_end_ms"]
            ):
                raise ValueError("source interval overlaps across the train and holdout splits")


def _reject_duplicate_media_digests(
    accepted_groups: list[dict],
    structural_groups: dict[str, dict],
) -> None:
    seen: set[str] = set()
    media_suffixes = ("_start.png", "_audio.wav", "_end.mp4")
    for group in accepted_groups:
        structural_group = structural_groups[group["group_id"]]
        for digest in structural_group["files"]:
            if not digest["name"].endswith(media_suffixes):
                continue
            sha256 = digest["sha256"]
            if sha256 in seen:
                raise ValueError("duplicate media digest among accepted A2V groups")
            seen.add(sha256)


def validate_quality_and_splits(attestation: dict, structural_report: dict) -> dict:
    groups = _validated_attestation_groups(attestation)
    structural_groups = _validated_structural_groups(structural_report)

    attestation_ids = {group["group_id"] for group in groups}
    structural_ids = set(structural_groups)
    missing_structural = sorted(attestation_ids - structural_ids)
    if missing_structural:
        raise ValueError(f"quality attestation references a missing structural group: {missing_structural}")
    missing_attestation = sorted(structural_ids - attestation_ids)
    if missing_attestation:
        raise ValueError(f"structural report has a missing quality attestation group: {missing_attestation}")

    accepted = [group for group in groups if group["accepted"]]
    train = [group for group in accepted if group["split"] == "train"]
    holdout = [group for group in accepted if group["split"] == "holdout"]
    if len(train) < 10:
        raise ValueError("quality gate requires at least 10 accepted training groups")
    if len(holdout) < 5:
        raise ValueError("quality gate requires at least 5 accepted holdout groups")

    _reject_cross_split_provenance(train, holdout)
    _reject_duplicate_media_digests(accepted, structural_groups)

    training_locations = {group["location_id"] for group in train}
    isolated_holdouts = [
        group for group in holdout if group["location_id"] not in training_locations
    ]
    if len(isolated_holdouts) < 2:
        raise ValueError("quality gate requires at least two location-isolated holdouts")

    training_teeth_count = sum(
        group["checks"][TEETH_CHECK] for group in train
    )
    holdout_teeth_count = sum(
        group["checks"][TEETH_CHECK] for group in holdout
    )
    if holdout_teeth_count < 1:
        raise ValueError("quality gate requires held-out teeth or inner-mouth coverage")

    train_ids = sorted(group["group_id"] for group in train)
    holdout_ids = sorted(group["group_id"] for group in holdout)
    isolated_ids = sorted(group["group_id"] for group in isolated_holdouts)
    return {
        "status": "valid",
        "accepted_train_group_ids": train_ids,
        "accepted_holdout_group_ids": holdout_ids,
        "location_coverage": {
            "isolated_holdout_group_ids": isolated_ids,
        },
        "coverage_counts": {
            "accepted_train_groups": len(train_ids),
            "accepted_holdout_groups": len(holdout_ids),
            "location_isolated_holdout_groups": len(isolated_ids),
            "training_teeth_or_inner_mouth_visible": training_teeth_count,
            "holdout_teeth_or_inner_mouth_visible": holdout_teeth_count,
        },
    }
