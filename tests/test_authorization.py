from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
import urllib.request

import pytest

from ltx_lora_pilot.a2v_bundle import build_root_manifest, compute_bundle_id
from ltx_lora_pilot.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)


FIXED_TIME = "2026-07-15T02:00:00Z"
POLICY_ID = "policy_00000000000040008000000000000001"
APPROVAL_ID = "approval_00000000000040008000000000000002"
ISSUER_PROCESS_ID = "process_00000000000040008000000000000003"
PILOT_ID = "pilot_00000000000040008000000000000004"
LEDGER_ID = "ledger_00000000000040008000000000000005"
EXECUTION_ID = "exec_00000000000040008000000000000006"
ENDPOINT = "fal-ai/ltx23-trainer-v2/a2v"
OFFICIAL_PRICE_URL = "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v"
ROOT = Path(__file__).resolve().parents[1]
RECORD_SCRIPT = ROOT / "scripts" / "record_standing_authorization.py"
PRICE_SCRIPT = ROOT / "scripts" / "capture_fal_price.py"
ISSUE_SCRIPT = ROOT / "scripts" / "issue_a2v_approval.py"
DATASET_MANIFEST = {"schema_version": "synthetic-dataset-v1"}
ARCHIVE_CONTENT = b"synthetic deterministic archive"
SYNTHETIC_REPORT = {"schema_version": "synthetic-report-v1"}
SYNTHETIC_SELECTION = {"schema_version": "synthetic-selection-v1"}


def _api() -> Any:
    try:
        return import_module("ltx_lora_pilot.authorization")
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"authorization API is unavailable: {exc}")


def _load_script(path: Path) -> Any:
    module_name = f"test_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def valid_policy(**overrides: Any) -> dict[str, Any]:
    value = {
        "policy_id": POLICY_ID,
        "source_sha256": "1" * 64,
        "endpoint": ENDPOINT,
        "executions": 1,
        "steps": 1_000,
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
        "expires_at_utc": "2026-07-17T02:00:00Z",
    }
    value.update(overrides)
    return value


def valid_price_evidence(**overrides: Any) -> dict[str, Any]:
    response = b"Training costs $0.006 * steps; 1,000 steps cost $6.00."
    value = {
        "source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": "0.006",
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "retrieved_at_utc": "2026-07-15T01:00:00Z",
        "expires_at_utc": "2026-07-16T01:00:00Z",
    }
    value.update(overrides)
    return value


def _group_id(index: int) -> str:
    return f"grp_000000000000400080000000{index:08x}"


def _holdout_groups() -> list[dict[str, Any]]:
    groups = []
    for index in range(1, 6):
        group_id = _group_id(index)
        files = []
        for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
            name = f"{group_id}{suffix}"
            content = name.encode("ascii")
            files.append(
                {
                    "name": name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        groups.append({"group_id": group_id, "files": files})
    return groups


def _execution_config(
    *,
    policy_value: dict[str, Any] | None = None,
    price_value: dict[str, Any] | None = None,
    dataset_content: bytes | None = None,
    archive_content: bytes = ARCHIVE_CONTENT,
    **overrides: Any,
) -> dict[str, Any]:
    bound_policy = valid_policy() if policy_value is None else policy_value
    bound_price = valid_price_evidence() if price_value is None else price_value
    bound_dataset = (
        canonical_json_bytes(DATASET_MANIFEST)
        if dataset_content is None
        else dataset_content
    )
    value = {
        "schema_version": "a2v-execution-config-v1",
        "canonical_json_version": 1,
        "execution_id": EXECUTION_ID,
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "created_at_utc": "2026-07-15T01:30:00Z",
        "expires_at_utc": "2026-07-15T20:00:00Z",
        "endpoint": ENDPOINT,
        "trigger_phrase": "chrx9_speech",
        "rank": 32,
        "steps": 1_000,
        "learning_rate": "0.0002",
        "training_frames": 89,
        "training_fps": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "audio_normalize": True,
        "audio_preserve_pitch": True,
        "debug_dataset": False,
        "negative_prompt": "synthetic artifacts, distortion",
        "validation": [
            {
                "image_filename": "validation_01_start.png",
                "image_sha256": hashlib.sha256(b"validation-image-1").hexdigest(),
                "audio_filename": "validation_01_audio.wav",
                "audio_sha256": hashlib.sha256(b"validation-audio-1").hexdigest(),
                "prompt": "chrx9_speech speaks in a synthetic test scene",
                "frames": 89,
                "fps": 24,
                "resolution": "high",
                "aspect_ratio": "9:16",
            },
            {
                "image_filename": "validation_02_start.png",
                "image_sha256": hashlib.sha256(b"validation-image-2").hexdigest(),
                "audio_filename": "validation_02_audio.wav",
                "audio_sha256": hashlib.sha256(b"validation-audio-2").hexdigest(),
                "prompt": "chrx9_speech speaks in another synthetic test scene",
                "frames": 89,
                "fps": 24,
                "resolution": "high",
                "aspect_ratio": "9:16",
            },
        ],
        "dataset_manifest_sha256": hashlib.sha256(bound_dataset).hexdigest(),
        "training_archive_sha256": hashlib.sha256(archive_content).hexdigest(),
        "standing_authorization_sha256": hashlib.sha256(
            canonical_json_bytes(bound_policy)
        ).hexdigest(),
        "price_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(bound_price)
        ).hexdigest(),
        "price_source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": "0.006",
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
    }
    value.update(overrides)
    return value


def _write_bundle(
    tmp_path: Path,
    *,
    name: str = "run-a",
    policy: dict[str, Any] | None = None,
    price: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    root_execution_id: str | None = None,
    plan_content: bytes = b"synthetic approved plan",
    root_created_at_utc: str | None = None,
    root_expires_at_utc: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    run_dir = tmp_path / name
    control_dir = run_dir / "control"
    bundle_dir = run_dir / "bundle"
    validation_dir = run_dir / "validation"
    control_dir.mkdir(parents=True)
    bundle_dir.mkdir()
    validation_dir.mkdir()

    policy_value = valid_policy() if policy is None else policy
    price_value = valid_price_evidence() if price is None else price
    plan_path = run_dir / "plan.md"
    policy_path = control_dir / "standing-authorization.json"
    price_path = control_dir / "price-evidence.json"
    config_path = control_dir / "execution-config.json"
    structural_path = control_dir / "structural-report.json"
    quality_path = control_dir / "quality-attestation.json"
    selection_path = validation_dir / "provider-validation-selection.json"
    dataset_path = bundle_dir / "dataset-manifest.json"
    archive_path = bundle_dir / "training-data.zip"

    plan_path.write_bytes(plan_content)
    atomic_write_json(policy_path, policy_value)
    atomic_write_json(price_path, price_value)
    atomic_write_json(structural_path, SYNTHETIC_REPORT)
    atomic_write_json(quality_path, SYNTHETIC_REPORT)
    atomic_write_json(selection_path, SYNTHETIC_SELECTION)
    atomic_write_json(dataset_path, DATASET_MANIFEST)
    archive_path.write_bytes(ARCHIVE_CONTENT)
    if config is None:
        timestamp_overrides = {}
        if root_created_at_utc is not None:
            timestamp_overrides["created_at_utc"] = root_created_at_utc
        if root_expires_at_utc is not None:
            timestamp_overrides["expires_at_utc"] = root_expires_at_utc
        config_value = _execution_config(
            policy_value=policy_value,
            price_value=price_value,
            dataset_content=dataset_path.read_bytes(),
            archive_content=archive_path.read_bytes(),
            **timestamp_overrides,
        )
    else:
        config_value = config
    atomic_write_json(config_path, config_value)

    artifacts = {
        "plan": sha256_file(plan_path),
        "standing_authorization": sha256_file(policy_path),
        "price_evidence": sha256_file(price_path),
        "structural_report": sha256_file(structural_path),
        "quality_attestation": sha256_file(quality_path),
        "provider_validation_selection": sha256_file(selection_path),
        "execution_config": sha256_file(config_path),
        "dataset_manifest": sha256_file(dataset_path),
        "training_archive": sha256_file(archive_path),
    }
    root_manifest = build_root_manifest(
        execution_id=root_execution_id or config_value["execution_id"],
        created_at_utc=(
            root_created_at_utc or config_value["created_at_utc"]
        ),
        expires_at_utc=(
            root_expires_at_utc or config_value["expires_at_utc"]
        ),
        repository_commit="f" * 40,
        artifacts=artifacts,
        holdout_groups=_holdout_groups(),
    )
    atomic_write_json(bundle_dir / "bundle-manifest.json", root_manifest)
    return run_dir, root_manifest, compute_bundle_id(root_manifest)


def test_valid_policy_round_trips_exactly() -> None:
    authorization = _api().StandingAuthorization.from_dict(
        valid_policy(), now=FIXED_TIME
    )

    assert authorization.to_dict() == valid_policy()


def test_policy_rejects_extra_two_dollar_cap() -> None:
    policy = valid_policy(cumulative_cap_usd="14.0000")

    with pytest.raises(ValueError, match="cumulative cap must be 12.0000"):
        _api().StandingAuthorization.from_dict(policy, now=FIXED_TIME)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("endpoint", "fal-ai/other", "endpoint must be"),
        ("executions", 2, "executions must be 1"),
        ("executions", True, "executions must be 1"),
        ("steps", 999, "steps must be 1000"),
        ("steps", True, "steps must be 1000"),
        ("training_max_usd", "6.0001", "training ceiling must be 6.0000"),
        (
            "validation_allocation_usd",
            "1.2501",
            "validation allocation must be at most 1.2500",
        ),
        (
            "validation_allocation_usd",
            "1.25",
            "validation allocation must use four decimal places",
        ),
        ("source_sha256", "A" * 64, "source_sha256 must be a lowercase SHA-256"),
    ],
)
def test_policy_rejects_non_fixed_or_malformed_values(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _api().StandingAuthorization.from_dict(
            valid_policy(**{field: value}), now=FIXED_TIME
        )


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_policy_rejects_non_exact_schema(mutation: str) -> None:
    policy = valid_policy()
    if mutation == "unknown":
        policy["contingency_usd"] = "2.0000"
    else:
        del policy["endpoint"]

    with pytest.raises(ValueError, match="exact fields"):
        _api().StandingAuthorization.from_dict(policy, now=FIXED_TIME)


def test_policy_rejects_expiry_and_noncanonical_time() -> None:
    with pytest.raises(ValueError, match="standing authorization expired"):
        _api().StandingAuthorization.from_dict(
            valid_policy(expires_at_utc="2026-07-15T01:59:59Z"),
            now=FIXED_TIME,
        )
    with pytest.raises(ValueError, match="canonical UTC timestamp"):
        _api().StandingAuthorization.from_dict(
            valid_policy(expires_at_utc="2026-07-17T02:00:00+00:00"),
            now=FIXED_TIME,
        )


def test_receipt_uses_exact_structured_approval_schema(tmp_path: Path) -> None:
    run_dir, root_manifest, bundle_id = _write_bundle(tmp_path)

    receipt = _api().issue_execution_receipt(
        valid_policy(),
        run_dir,
        expected_bundle_id=bundle_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )

    assert set(receipt.to_dict()) == {
        "schema_version",
        "approval_id",
        "status",
        "approval_mode",
        "policy_id",
        "standing_authorization_sha256",
        "issuer_process_id",
        "issued_at_utc",
        "bundle_id",
        "execution_id",
        "expires_at_utc",
        "plan_sha256",
        "dataset_manifest_sha256",
        "training_archive_sha256",
        "execution_config_sha256",
        "pilot_id",
        "ledger_id",
        "training_max_usd",
        "validation_allocation_usd",
        "cumulative_cap_usd",
        "steps",
    }
    assert receipt.schema_version == "a2v-execution-approval-v1"
    assert receipt.status == "approved_for_paid_execution"
    assert receipt.approval_mode == "standing_policy"
    assert receipt.bundle_id == compute_bundle_id(root_manifest)
    assert receipt.execution_id == EXECUTION_ID
    assert receipt.expires_at_utc == "2026-07-15T20:00:00Z"
    assert receipt.training_max_usd == "6.0000"
    assert receipt.validation_allocation_usd == "1.2500"
    assert receipt.cumulative_cap_usd == "12.0000"
    assert receipt.steps == 1_000
    assert _api().verify_execution_receipt(
        receipt,
        valid_policy(),
        run_dir,
        now=FIXED_TIME,
    ) == receipt


def test_receipt_for_bundle_a_cannot_approve_bundle_b(tmp_path: Path) -> None:
    bundle_a, _, bundle_a_id = _write_bundle(tmp_path, name="run-a")
    bundle_b, _, _ = _write_bundle(
        tmp_path,
        name="run-b",
        plan_content=b"different synthetic approved plan",
    )
    receipt = _api().issue_execution_receipt(
        valid_policy(),
        bundle_a,
        expected_bundle_id=bundle_a_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )

    with pytest.raises(ValueError, match="bundle mismatch"):
        _api().verify_execution_receipt(
            receipt, valid_policy(), bundle_b, now=FIXED_TIME
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"endpoint": "fal-ai/other"}, "endpoint mismatch"),
        ({"steps": 1_001}, "step count mismatch"),
        ({"training_max_usd": "6.0001"}, "training ceiling mismatch"),
        (
            {"validation_allocation_usd": "1.2501"},
            "validation allocation mismatch",
        ),
        ({"cumulative_cap_usd": "14.0000"}, "cumulative cap mismatch"),
        (
            {"execution_id": "exec_00000000000040008000000000000007"},
            "execution ID mismatch",
        ),
    ],
)
def test_issuer_rejects_execution_config_mismatch(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        config=_execution_config(**overrides),
        root_execution_id=(
            EXECUTION_ID if "execution_id" in overrides else None
        ),
    )

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_issuer_requires_exact_versioned_execution_config_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _execution_config()
    if mutation == "unknown":
        config["steps_override"] = 2_000
    else:
        del config["negative_prompt"]
    run_dir, _, bundle_id = _write_bundle(tmp_path, config=config)

    with pytest.raises(ValueError, match="execution configuration.*exact fields"):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "created_at_utc",
            "2026-07-15T01:30:00+00:00",
            "created_at_utc must be a canonical UTC timestamp",
        ),
        (
            "created_at_utc",
            True,
            "created_at_utc must be a canonical UTC timestamp",
        ),
        (
            "expires_at_utc",
            "2026-07-15T20:00:00+00:00",
            "expires_at_utc must be a canonical UTC timestamp",
        ),
        (
            "expires_at_utc",
            "2026-07-15T01:30:00Z",
            "expires_at_utc must be after created_at_utc",
        ),
    ],
)
def test_issuer_rejects_noncanonical_or_reversed_config_timestamps(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        config=_execution_config(**{field: value}),
        root_created_at_utc="2026-07-15T01:30:00Z",
        root_expires_at_utc="2026-07-15T20:00:00Z",
    )

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("root_field", "value", "message"),
    [
        (
            "root_created_at_utc",
            "2026-07-15T01:31:00Z",
            "execution configuration creation timestamp mismatch",
        ),
        (
            "root_expires_at_utc",
            "2026-07-15T19:59:59Z",
            "execution configuration expiry timestamp mismatch",
        ),
    ],
)
def test_issuer_requires_config_timestamps_to_equal_root_manifest(
    tmp_path: Path,
    root_field: str,
    value: str,
    message: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        config=_execution_config(),
        **{root_field: value},
    )

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_builder_compatible_config_timestamps_are_root_bound(
    tmp_path: Path,
) -> None:
    run_dir, root_manifest, bundle_id = _write_bundle(tmp_path)
    config = json.loads(
        (run_dir / "control" / "execution-config.json").read_text("utf-8")
    )

    assert len(config) == 32
    assert set(config) == _api().EXECUTION_CONFIG_FIELDS
    assert config["created_at_utc"] == root_manifest["created_at_utc"]
    assert config["expires_at_utc"] == root_manifest["expires_at_utc"]
    receipt = _api().issue_execution_receipt(
        valid_policy(),
        run_dir,
        expected_bundle_id=bundle_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )
    assert receipt.expires_at_utc == config["expires_at_utc"]


def test_issuer_rejects_bundle_expiry_beyond_standing_authorization(
    tmp_path: Path,
) -> None:
    policy = valid_policy(expires_at_utc="2026-07-15T19:59:59Z")
    config = _execution_config(policy_value=policy)
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        policy=policy,
        config=config,
    )

    with pytest.raises(ValueError, match="bundle expiry exceeds policy expiry"):
        _api().issue_execution_receipt(
            policy,
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_issuer_rejects_bundle_expiry_beyond_price_evidence(
    tmp_path: Path,
) -> None:
    price = valid_price_evidence(expires_at_utc="2026-07-15T19:59:59Z")
    config = _execution_config(price_value=price)
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        price=price,
        config=config,
    )

    with pytest.raises(ValueError, match="bundle expiry exceeds price evidence expiry"):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "a2v-execution-config-v2", "schema mismatch"),
        (
            "canonical_json_version",
            2,
            "canonical JSON version mismatch",
        ),
        (
            "canonical_json_version",
            True,
            "canonical JSON version mismatch",
        ),
        ("rank", 16, "rank mismatch"),
        ("rank", True, "rank mismatch"),
        ("learning_rate", "0.00020", "learning rate mismatch"),
        ("learning_rate", 0, "learning rate mismatch"),
        ("training_frames", 88, "training frames mismatch"),
        ("training_fps", 25, "training fps mismatch"),
        ("resolution", "low", "resolution mismatch"),
        ("aspect_ratio", "16:9", "aspect ratio mismatch"),
        ("auto_scale_input", True, "auto-scale mismatch"),
        ("split_input_into_scenes", True, "split-scenes mismatch"),
        ("audio_normalize", False, "audio normalization mismatch"),
        ("audio_preserve_pitch", False, "pitch preservation mismatch"),
        ("debug_dataset", True, "debug_dataset must be false"),
        ("trigger_phrase", "Human Name", "neutral trigger phrase"),
    ],
)
def test_issuer_rejects_non_fixed_a2v_request_values(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        config=_execution_config(**{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "exactly two validation entries"),
        ("unknown", "validation entry.*exact fields"),
        ("missing", "validation entry.*exact fields"),
        ("frames", "validation frames mismatch"),
        ("fps_bool", "validation fps mismatch"),
        ("resolution", "validation resolution mismatch"),
        ("aspect", "validation aspect ratio mismatch"),
        ("filename", "canonical local filename"),
        ("hash", "image_sha256 must be a lowercase SHA-256"),
        ("prompt", "canonical non-empty text"),
        ("order", "canonical validation order"),
    ],
)
def test_issuer_rejects_malformed_nested_validation_entries(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config = _execution_config()
    validation = copy.deepcopy(config["validation"])
    if mutation == "count":
        validation.pop()
    elif mutation == "unknown":
        validation[0]["image_url"] = "https://example.invalid/private"
    elif mutation == "missing":
        del validation[0]["prompt"]
    elif mutation == "frames":
        validation[0]["frames"] = 88
    elif mutation == "fps_bool":
        validation[0]["fps"] = True
    elif mutation == "resolution":
        validation[0]["resolution"] = "low"
    elif mutation == "aspect":
        validation[0]["aspect_ratio"] = "16:9"
    elif mutation == "filename":
        validation[0]["image_filename"] = "../private.png"
    elif mutation == "hash":
        validation[0]["image_sha256"] = "A" * 64
    elif mutation == "prompt":
        validation[0]["prompt"] = " trailing space "
    else:
        validation.reverse()
    config["validation"] = validation
    run_dir, _, bundle_id = _write_bundle(tmp_path, config=config)

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("field", "suffix", "message"),
    [
        (
            "image_filename",
            ".png",
            "validation image filenames must be case-insensitively unique",
        ),
        (
            "audio_filename",
            ".wav",
            "validation audio filenames must be case-insensitively unique",
        ),
    ],
)
def test_issuer_rejects_casefold_colliding_validation_filenames(
    tmp_path: Path,
    field: str,
    suffix: str,
    message: str,
) -> None:
    config = _execution_config()
    validation = copy.deepcopy(config["validation"])
    validation[0][field] = f"Validation{suffix}"
    validation[1][field] = f"validation{suffix}"
    config["validation"] = validation
    run_dir, _, bundle_id = _write_bundle(tmp_path, config=config)

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_manifest_sha256", "0" * 64, "dataset manifest hash mismatch"),
        ("training_archive_sha256", "0" * 64, "training archive hash mismatch"),
        (
            "standing_authorization_sha256",
            "0" * 64,
            "standing authorization config hash mismatch",
        ),
        ("price_evidence_sha256", "0" * 64, "price evidence config hash mismatch"),
        ("price_source_url", "https://example.invalid", "price source URL mismatch"),
        ("rate_usd_per_step", "0.007", "price rate mismatch"),
    ],
)
def test_issuer_rejects_execution_config_binding_changes(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(
        tmp_path,
        config=_execution_config(**{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "plan.md",
        "control/standing-authorization.json",
        "control/price-evidence.json",
        "control/structural-report.json",
        "control/quality-attestation.json",
        "control/execution-config.json",
        "validation/provider-validation-selection.json",
        "bundle/dataset-manifest.json",
        "bundle/training-data.zip",
    ],
)
def test_issuer_requires_every_root_bound_file(
    tmp_path: Path,
    relative_path: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    (run_dir / relative_path).unlink()

    with pytest.raises(ValueError, match="private bundle input is unavailable"):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("control/structural-report.json", "structural report root binding mismatch"),
        ("control/quality-attestation.json", "quality attestation root binding mismatch"),
        (
            "validation/provider-validation-selection.json",
            "provider validation selection root binding mismatch",
        ),
    ],
)
def test_issuer_freshly_hashes_previously_unchecked_root_artifacts(
    tmp_path: Path,
    relative_path: str,
    message: str,
) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    atomic_write_json(run_dir / relative_path, {"tampered": True})

    with pytest.raises(ValueError, match=message):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_issuer_rejects_aliasing_root_bound_files(tmp_path: Path) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    structural_path = run_dir / "control" / "structural-report.json"
    quality_path = run_dir / "control" / "quality-attestation.json"
    quality_path.unlink()
    os.link(structural_path, quality_path)

    with pytest.raises(ValueError, match="root-bound files must not alias"):
        _api().issue_execution_receipt(
            valid_policy(),
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_issuer_rejects_expired_bundle_wrong_policy_hash_and_replay_id(
    tmp_path: Path,
) -> None:
    expired_run, _, expired_id = _write_bundle(
        tmp_path,
        name="expired",
        root_expires_at_utc="2026-07-15T01:59:59Z",
    )
    with pytest.raises(ValueError, match="bundle expired"):
        _api().issue_execution_receipt(
            valid_policy(),
            expired_run,
            expected_bundle_id=expired_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )

    other_policy = valid_policy(source_sha256="2" * 64)
    wrong_policy_run, _, wrong_policy_id = _write_bundle(
        tmp_path,
        name="wrong-policy",
        policy=other_policy,
    )
    with pytest.raises(ValueError, match="standing authorization hash mismatch"):
        _api().issue_execution_receipt(
            valid_policy(),
            wrong_policy_run,
            expected_bundle_id=wrong_policy_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )

    valid_run, _, valid_id = _write_bundle(tmp_path, name="valid")
    with pytest.raises(ValueError, match="replay ID"):
        _api().issue_execution_receipt(
            valid_policy(),
            valid_run,
            expected_bundle_id=valid_id,
            approval_id=EXECUTION_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_receipt_rejects_unknown_fields_and_expiry(tmp_path: Path) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    receipt = _api().issue_execution_receipt(
        valid_policy(),
        run_dir,
        expected_bundle_id=bundle_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )
    serialized = receipt.to_dict()
    serialized["request_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="exact fields"):
        _api().ExecutionReceipt.from_dict(serialized, now=FIXED_TIME)
    with pytest.raises(ValueError, match="execution approval expired"):
        _api().verify_execution_receipt(
            receipt,
            valid_policy(),
            run_dir,
            now="2026-07-15T20:00:01Z",
        )


def test_receipt_hashes_are_the_exact_bound_artifacts(tmp_path: Path) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    receipt = _api().issue_execution_receipt(
        valid_policy(),
        run_dir,
        expected_bundle_id=bundle_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )

    assert receipt.standing_authorization_sha256 == hashlib.sha256(
        canonical_json_bytes(valid_policy())
    ).hexdigest()
    assert receipt.plan_sha256 == sha256_file(run_dir / "plan.md").sha256
    assert receipt.dataset_manifest_sha256 == sha256_file(
        run_dir / "bundle" / "dataset-manifest.json"
    ).sha256
    assert receipt.training_archive_sha256 == sha256_file(
        run_dir / "bundle" / "training-data.zip"
    ).sha256
    assert receipt.execution_config_sha256 == sha256_file(
        run_dir / "control" / "execution-config.json"
    ).sha256


def test_price_capture_binds_official_url_formula_hash_and_24_hour_expiry() -> None:
    response = b"Training costs 0.006 * steps; 1000 steps cost $6.00."
    requested_urls: list[str] = []

    def fetch(url: str) -> bytes:
        requested_urls.append(url)
        return response

    evidence = _api().capture_price_evidence(fetch=fetch, now=FIXED_TIME)

    assert evidence.to_dict() == {
        "source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": "0.006",
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "retrieved_at_utc": "2026-07-15T02:00:00Z",
        "expires_at_utc": "2026-07-16T02:00:00Z",
    }
    assert requested_urls == [OFFICIAL_PRICE_URL]


def test_price_capture_requires_official_formula() -> None:
    fetch = lambda _url: b"The cost is 0.007 * steps."

    with pytest.raises(ValueError, match="unexpected A2V rate"):
        _api().capture_price_evidence(fetch=fetch, now=FIXED_TIME)


def test_price_capture_isolates_a2v_and_accepts_identical_live_duplicates() -> None:
    response = (
        b'<section data-model="fal-ai/other">$0.007 per step; $0.007 * steps; '
        b'1000 steps cost $7.00.</section>'
        b'<section data-model="fal-ai/ltx23-trainer-v2/a2v">0.006 * steps; '
        b'$0.006 per step; 1000 steps cost $6.00.</section>'
        b'<section data-model="fal-ai/other-after">$0.009 per step; $0.009 * steps; '
        b'1000 steps cost $9.00.</section>'
        b'<script>{"endpoint":"fal-ai/ltx23-trainer-v2/a2v",'
        b'"formula":"0.006 * steps","example":"1000 steps costs $6.00"}'
        b"</script>"
    )

    evidence = _api().capture_price_evidence(
        fetch=lambda _url: response,
        now=FIXED_TIME,
    )

    assert evidence.rate_usd_per_step == "0.006"
    assert evidence.response_sha256 == hashlib.sha256(response).hexdigest()


def test_price_capture_rejects_conflicting_a2v_anchored_serializations() -> None:
    response = (
        b'<section data-model="fal-ai/ltx23-trainer-v2/a2v">0.006 * steps; '
        b'1000 steps cost $6.00.</section>'
        b'<section data-model="fal-ai/ltx23-trainer-v2/a2v">0.007 * steps; '
        b'1000 steps cost $7.00.</section>'
    )

    with pytest.raises(ValueError, match="unexpected A2V rate"):
        _api().capture_price_evidence(
            fetch=lambda _url: response,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "alternate_rate",
    [
        "$0.007 per step",
        "0.007 USD per training step",
        "$0.007/step",
        "per step rate is $0.007",
    ],
)
def test_price_capture_rejects_a2v_scoped_alternate_rate_wording(
    alternate_rate: str,
) -> None:
    response = (
        f'<section data-model="{ENDPOINT}">{alternate_rate}; '
        f"{'archived-' * 12} calculator: 0.006 * steps; "
        "1000 steps cost $6.00.</section>"
    ).encode("ascii")

    with pytest.raises(ValueError, match="unexpected A2V rate"):
        _api().capture_price_evidence(
            fetch=lambda _url: response,
            now=FIXED_TIME,
        )


def test_price_capture_rejects_punctuated_reverse_order_a2v_rate() -> None:
    response = (
        f'<section data-model="{ENDPOINT}">per step rate is $0.007. '
        "Archived calculator: 0.006 * steps; 1000 steps cost $6.00.</section>"
    ).encode("ascii")

    with pytest.raises(ValueError, match="unexpected A2V rate"):
        _api().capture_price_evidence(
            fetch=lambda _url: response,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"Training costs $0.006 * steps.", "unexpected 1,000-step cost"),
        (
            b"Training costs $0.006 * steps; 1,000 steps cost $7.00.",
            "unexpected 1,000-step cost",
        ),
    ],
)
def test_price_capture_rejects_ambiguous_or_changed_statements(
    response: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _api().capture_price_evidence(
            fetch=lambda _url: response,
            now=FIXED_TIME,
        )


def test_price_capture_sanitizes_fetch_failure_and_never_reads_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_environment(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("credential access attempted")

    def failed_fetch(_url: str) -> bytes:
        raise OSError("SENSITIVE-FETCH-DETAIL")

    monkeypatch.setattr(os, "getenv", forbidden_environment)
    monkeypatch.setenv("FAL_KEY", "SYNTHETIC-SECRET-MUST-NOT-BE-READ")

    with pytest.raises(ValueError, match="^price fetch failed$") as captured:
        _api().capture_price_evidence(fetch=failed_fetch, now=FIXED_TIME)

    assert "SENSITIVE" not in str(captured.value)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"x" * (1_048_576 + 1), id="oversize"),
        pytest.param("not-bytes", id="wrong-type"),
    ],
)
def test_price_capture_rejects_invalid_or_oversize_responses(response: Any) -> None:
    with pytest.raises(ValueError, match="official price response"):
        _api().capture_price_evidence(
            fetch=lambda _url: response,
            now=FIXED_TIME,
        )


@pytest.mark.parametrize(
    "source_url",
    [
        "http://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
        "https://example.invalid/models/fal-ai/ltx23-trainer-v2/a2v",
        "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v/api",
        "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v?changed=1",
        "https://user@fal.ai/models/fal-ai/ltx23-trainer-v2/a2v",
    ],
)
def test_price_evidence_rejects_non_official_url(source_url: str) -> None:
    with pytest.raises(ValueError, match="official HTTPS URL"):
        _api().PriceEvidence.from_dict(
            valid_price_evidence(source_url=source_url),
            now=FIXED_TIME,
        )


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_price_evidence_rejects_non_exact_schema(mutation: str) -> None:
    evidence = valid_price_evidence()
    if mutation == "unknown":
        evidence["response_body"] = "private"
    else:
        del evidence["response_sha256"]

    with pytest.raises(ValueError, match="exact fields"):
        _api().PriceEvidence.from_dict(evidence, now=FIXED_TIME)


def test_price_evidence_rejects_stale_overlong_and_malformed_hash() -> None:
    with pytest.raises(ValueError, match="price evidence expired"):
        _api().PriceEvidence.from_dict(
            valid_price_evidence(expires_at_utc="2026-07-15T01:59:59Z"),
            now=FIXED_TIME,
        )
    with pytest.raises(ValueError, match="within 24 hours"):
        _api().PriceEvidence.from_dict(
            valid_price_evidence(expires_at_utc="2026-07-16T01:00:01Z"),
            now=FIXED_TIME,
        )
    with pytest.raises(ValueError, match="response_sha256 must be a lowercase SHA-256"):
        _api().PriceEvidence.from_dict(
            valid_price_evidence(response_sha256="not-a-hash"),
            now=FIXED_TIME,
        )


def test_price_evidence_rejects_future_retrieval_time() -> None:
    with pytest.raises(ValueError, match="retrieval is in the future"):
        _api().PriceEvidence.from_dict(
            valid_price_evidence(
                retrieved_at_utc="2026-07-15T02:00:01Z",
                expires_at_utc="2026-07-16T02:00:01Z",
            ),
            now=FIXED_TIME,
        )


def test_issuer_revalidates_direct_policy_dataclass(tmp_path: Path) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    invalid = _api().StandingAuthorization(
        **valid_policy(endpoint="fal-ai/other")
    )

    with pytest.raises(ValueError, match="endpoint must be"):
        _api().issue_execution_receipt(
            invalid,
            run_dir,
            expected_bundle_id=bundle_id,
            approval_id=APPROVAL_ID,
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )


def test_default_price_fetch_is_bounded_unauthenticated_and_redirect_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body = b"Training costs $0.006 * steps; 1,000 steps cost $6.00."
    opened: dict[str, Any] = {"handlers": []}

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def geturl(self) -> str:
            return OFFICIAL_PRICE_URL

        def read(self, byte_count: int) -> bytes:
            opened["read_size"] = byte_count
            return response_body

    class FakeOpener:
        def open(self, request: Any, *, timeout: int) -> FakeResponse:
            opened["request"] = request
            opened["timeout"] = timeout
            return FakeResponse()

    def build_opener(*handlers: Any) -> FakeOpener:
        opened["handlers"].extend(handlers)
        return FakeOpener()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(_api().urllib_request, "build_opener", build_opener)

    assert _api()._fetch_official_price(OFFICIAL_PRICE_URL) == response_body
    request = opened["request"]
    headers = {name.lower(): value for name, value in request.header_items()}
    assert request.full_url == OFFICIAL_PRICE_URL
    assert request.get_method() == "GET"
    assert "authorization" not in headers
    assert "proxy-authorization" not in headers
    assert "cookie" not in headers
    assert opened["timeout"] == 10
    assert opened["read_size"] == 1_048_577
    proxy_handlers = [
        handler
        for handler in opened["handlers"]
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}

    handler = _api()._OfficialPriceRedirectHandler()
    with pytest.raises(ValueError, match="redirect left"):
        handler.redirect_request(
            urllib.request.Request(OFFICIAL_PRICE_URL),
            None,
            302,
            "Found",
            {},
            "https://example.invalid/redirect",
        )


def test_standing_authorization_recorder_hashes_without_copying_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "PRIVATE-SOURCE-NAME.txt"
    output_path = tmp_path / "standing-authorization.json"
    source_content = b"synthetic private instruction marker; never serialize this"
    source_path.write_bytes(source_content)
    recorder = _load_script(RECORD_SCRIPT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RECORD_SCRIPT),
            "--source-file",
            str(source_path),
            "--policy-id",
            POLICY_ID,
            "--expires-at-utc",
            "2099-01-01T00:00:00Z",
            "--output",
            str(output_path),
        ],
    )

    recorder.main()

    captured = capsys.readouterr()
    assert captured.out == "STANDING_AUTHORIZATION_RECORDED\n"
    assert captured.err == ""
    serialized = output_path.read_bytes()
    policy = json.loads(serialized)
    assert serialized == canonical_json_bytes(policy)
    assert policy == valid_policy(
        source_sha256=hashlib.sha256(source_content).hexdigest(),
        expires_at_utc="2099-01-01T00:00:00Z",
    )
    assert source_content not in serialized
    assert source_path.name.encode("utf-8") not in serialized
    assert str(source_path).encode("utf-8") not in serialized
    assert hashlib.sha256(source_content).hexdigest() not in captured.out


def test_standing_authorization_recorder_sanitizes_argument_and_runtime_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _load_script(RECORD_SCRIPT)
    sensitive = "PRIVATE-PATH-MARKER"
    monkeypatch.setattr(sys, "argv", [str(RECORD_SCRIPT), sensitive])
    with pytest.raises(SystemExit) as parse_exit:
        recorder.main()
    parse_output = capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RECORD_SCRIPT),
            "--source-file",
            str(tmp_path / sensitive),
            "--policy-id",
            POLICY_ID,
            "--expires-at-utc",
            "2099-01-01T00:00:00Z",
            "--output",
            str(tmp_path / "authorization.json"),
        ],
    )
    with pytest.raises(SystemExit) as runtime_exit:
        recorder.main()
    runtime_output = capsys.readouterr()

    assert parse_exit.value.code == 2
    assert parse_output.out == ""
    assert parse_output.err == "STANDING_AUTHORIZATION_ARGUMENT_ERROR\n"
    assert runtime_exit.value.code == 2
    assert runtime_output.out == ""
    assert runtime_output.err == "STANDING_AUTHORIZATION_RECORD_FAILED\n"
    assert sensitive not in parse_output.err + runtime_output.err


def test_concurrent_recorders_publish_exactly_one_complete_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "synthetic-source.txt"
    output_path = tmp_path / "standing-authorization.json"
    source_path.write_bytes(b"synthetic standing authorization source")
    recorder = _load_script(RECORD_SCRIPT)
    original_hash = recorder.sha256_file
    publication_barrier = threading.Barrier(2)

    def synchronized_hash(path: Path) -> Any:
        digest = original_hash(path)
        publication_barrier.wait(timeout=10)
        return digest

    monkeypatch.setattr(recorder, "sha256_file", synchronized_hash)
    policy_ids = [
        "policy_0000000000004000800000000000000b",
        "policy_0000000000004000800000000000000c",
    ]

    def attempt(policy_id: str) -> Any:
        try:
            return recorder._record(
                source_path,
                output_path,
                policy_id=policy_id,
                expires_at_utc="2026-07-17T02:00:00Z",
                now=FIXED_TIME,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, policy_ids))

    policies = [
        result for result in results if isinstance(result, _api().StandingAuthorization)
    ]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(policies) == 1
    assert len(failures) == 1
    assert str(failures[0]) == "authorization destination already exists"

    serialized = output_path.read_bytes()
    published = json.loads(serialized)
    assert serialized == canonical_json_bytes(published)
    assert published == policies[0].to_dict()
    assert not list(output_path.parent.glob(".standing-authorization.json.*.tmp"))


def test_price_command_writes_only_canonical_minimal_evidence(tmp_path: Path) -> None:
    price_command = _load_script(PRICE_SCRIPT)
    output_path = tmp_path / "price-evidence.json"
    response = b"Training costs $0.006 * steps; 1,000 steps cost $6.00."

    price_command._capture(
        output_path,
        fetch=lambda _url: response,
        now=FIXED_TIME,
    )

    serialized = output_path.read_bytes()
    value = json.loads(serialized)
    assert serialized == canonical_json_bytes(value)
    assert value == _api().capture_price_evidence(
        fetch=lambda _url: response,
        now=FIXED_TIME,
    ).to_dict()
    assert set(value) == {
        "source_url",
        "rate_usd_per_step",
        "response_sha256",
        "retrieved_at_utc",
        "expires_at_utc",
    }


def test_concurrent_price_captures_publish_exactly_one_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_command = _load_script(PRICE_SCRIPT)
    output_path = tmp_path / "price-evidence.json"
    original_capture = price_command.capture_price_evidence
    publication_barrier = threading.Barrier(2)

    def synchronized_capture(*args: Any, **kwargs: Any) -> Any:
        evidence = original_capture(*args, **kwargs)
        publication_barrier.wait(timeout=10)
        return evidence

    monkeypatch.setattr(
        price_command,
        "capture_price_evidence",
        synchronized_capture,
    )
    responses = [
        b"variant-a: 0.006 * steps; 1000 steps cost $6.00.",
        b"variant-b: 0.006 * steps; 1000 steps cost $6.00.",
    ]

    def attempt(response: bytes) -> Any:
        try:
            return price_command._capture(
                output_path,
                fetch=lambda _url: response,
                now=FIXED_TIME,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, responses))

    evidence = [result for result in results if isinstance(result, _api().PriceEvidence)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(evidence) == 1
    assert len(failures) == 1
    assert str(failures[0]) == "price evidence destination already exists"

    serialized = output_path.read_bytes()
    published = json.loads(serialized)
    assert serialized == canonical_json_bytes(published)
    assert published == evidence[0].to_dict()
    assert not list(output_path.parent.glob(".price-evidence.json.*.tmp"))


def test_issuer_writes_once_and_has_no_paid_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ltx_lora_pilot import budget, fal_api

    run_dir, _, bundle_id = _write_bundle(tmp_path)
    issuer = _load_script(ISSUE_SCRIPT)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("paid or external capability was accessed")

    monkeypatch.setenv("FAL_KEY", "SYNTHETIC-KEY-MUST-NOT-BE-READ")
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(fal_api, "upload", forbidden)
    monkeypatch.setattr(fal_api, "submit", forbidden)
    monkeypatch.setattr(budget.BudgetLedger, "__init__", forbidden)
    monkeypatch.setattr(budget.BudgetLedger, "reserve", forbidden)

    receipt = issuer._issue(
        run_dir,
        bundle_id,
        approval_id=APPROVAL_ID,
        issuer_process_id=ISSUER_PROCESS_ID,
        now=FIXED_TIME,
    )

    approval_path = run_dir / "control" / "execution-approval.json"
    assert approval_path.read_bytes() == canonical_json_bytes(receipt.to_dict())
    assert _api().verify_execution_receipt(
        json.loads(approval_path.read_text("utf-8")),
        valid_policy(),
        run_dir,
        now=FIXED_TIME,
    ) == receipt
    with pytest.raises(ValueError, match="already exists"):
        issuer._issue(
            run_dir,
            bundle_id,
            approval_id="approval_00000000000040008000000000000007",
            issuer_process_id=ISSUER_PROCESS_ID,
            now=FIXED_TIME,
        )

    issuer_source = ISSUE_SCRIPT.read_text(encoding="utf-8")
    for prohibited in (
        "fal_client",
        "ltx_lora_pilot.fal_api",
        "ltx_lora_pilot.budget",
        "BudgetLedger",
        "upload",
        "submit",
    ):
        assert prohibited not in issuer_source


def test_concurrent_issuers_publish_exactly_one_complete_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _, bundle_id = _write_bundle(tmp_path)
    issuer = _load_script(ISSUE_SCRIPT)
    original_issue = issuer.issue_execution_receipt
    publication_barrier = threading.Barrier(2)

    def synchronized_issue(*args: Any, **kwargs: Any) -> Any:
        receipt = original_issue(*args, **kwargs)
        publication_barrier.wait(timeout=10)
        return receipt

    monkeypatch.setattr(issuer, "issue_execution_receipt", synchronized_issue)
    identifiers = [
        (
            "approval_00000000000040008000000000000007",
            "process_00000000000040008000000000000009",
        ),
        (
            "approval_00000000000040008000000000000008",
            "process_0000000000004000800000000000000a",
        ),
    ]

    def attempt(approval_id: str, process_id: str) -> Any:
        try:
            return issuer._issue(
                run_dir,
                bundle_id,
                approval_id=approval_id,
                issuer_process_id=process_id,
                now=FIXED_TIME,
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda pair: attempt(*pair), identifiers))

    receipts = [result for result in results if isinstance(result, _api().ExecutionReceipt)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(receipts) == 1
    assert len(failures) == 1
    assert str(failures[0]) == "execution approval already exists"

    approval_path = run_dir / "control" / "execution-approval.json"
    serialized = approval_path.read_bytes()
    published = json.loads(serialized)
    assert serialized == canonical_json_bytes(published)
    assert published == receipts[0].to_dict()
    assert not list(approval_path.parent.glob(".execution-approval.json.*.tmp"))


def test_issuer_cli_requires_full_bundle_id_and_neutral_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    issuer = _load_script(ISSUE_SCRIPT)
    sensitive = "PRIVATE-BUNDLE-ID-MARKER"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ISSUE_SCRIPT),
            "--run-dir",
            str(tmp_path),
            "--bundle-id",
            sensitive,
            "--approval-id",
            APPROVAL_ID,
            "--issuer-process-id",
            ISSUER_PROCESS_ID,
        ],
    )
    with pytest.raises(SystemExit) as parse_exit:
        issuer.main()
    parse_output = capsys.readouterr()

    monkeypatch.setattr(issuer, "_issue", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ISSUE_SCRIPT),
            "--run-dir",
            str(tmp_path),
            "--bundle-id",
            "f" * 64,
            "--approval-id",
            APPROVAL_ID,
            "--issuer-process-id",
            ISSUER_PROCESS_ID,
        ],
    )
    issuer.main()
    success_output = capsys.readouterr()

    assert parse_exit.value.code == 2
    assert parse_output.out == ""
    assert parse_output.err == "A2V_APPROVAL_ARGUMENT_ERROR\n"
    assert sensitive not in parse_output.err
    assert success_output.out == "A2V_APPROVAL_ISSUED\n"
    assert success_output.err == ""
    assert "f" * 64 not in success_output.out
