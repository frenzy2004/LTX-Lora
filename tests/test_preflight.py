from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable
import warnings
import zipfile

import pytest

from ltx_lora_pilot.a2v_bundle import (
    FIXED_ARCHIVE_DATETIME,
    FIXED_EXTERNAL_ATTR,
    build_dataset_manifest,
    build_root_manifest,
    build_training_archive,
    compute_bundle_id,
)
from ltx_lora_pilot.a2v_quality import CHECK_KEYS, validate_quality_and_splits
from ltx_lora_pilot.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    strict_load_json,
)
from ltx_lora_pilot.authorization import (
    OFFICIAL_PRICE_URL,
    issue_execution_receipt,
)
from ltx_lora_pilot.pilot_ledger import (
    PilotLedger,
    migrate_legacy_ledger,
)
from ltx_lora_pilot.preflight import (
    GATE_ORDER,
    PreflightNotReady,
    PreflightStatus,
    run_preflight,
)
from ltx_lora_pilot.provider_validation import (
    build_provider_validation_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_a2v.py"
FIXED_TIME = datetime(2026, 7, 16, 1, 0, 0, tzinfo=timezone.utc)
PILOT_ID = "pilot_00000000000040008000000000000001"
LEDGER_ID = "ledger_00000000000040008000000000000002"
EXECUTION_ID = "exec_00000000000040008000000000000003"
POLICY_ID = "policy_00000000000040008000000000000004"
APPROVAL_ID = "approval_00000000000040008000000000000005"
PROCESS_ID = "process_00000000000040008000000000000006"
MIGRATION_ID = "migration_00000000000040008000000000000007"
AMOUNTS = ["1.2000", "0.1099", "0.1099", "0.3272", "0.3272", "1.4667"]
STATES = ["consumed", "consumed", "consumed", "consumed", "reserved", "consumed"]


def _typed_id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:012x}40008{number:015x}"


def _group_id(index: int) -> str:
    return f"grp_{index:012x}40008{index:015x}"


def _source_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def _digest(path: Path) -> dict[str, Any]:
    return asdict(sha256_file(path))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _policy(**overrides: Any) -> dict[str, Any]:
    value = {
        "policy_id": POLICY_ID,
        "source_sha256": "1" * 64,
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "executions": 1,
        "steps": 1000,
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
        "expires_at_utc": "2026-07-16T20:00:00Z",
    }
    value.update(overrides)
    return value


def _price(**overrides: Any) -> dict[str, Any]:
    value = {
        "source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": "0.006",
        "response_sha256": hashlib.sha256(b"official synthetic price evidence").hexdigest(),
        "retrieved_at_utc": "2026-07-16T00:00:00Z",
        "expires_at_utc": "2026-07-16T18:00:00Z",
    }
    value.update(overrides)
    return value


def _make_candidates(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_dir.mkdir(parents=True)
    groups: list[dict[str, Any]] = []
    attested: list[dict[str, Any]] = []
    for index in range(1, 16):
        group_id = _group_id(index)
        paths = []
        for suffix in (".txt", "_audio.wav", "_end.mp4", "_start.png"):
            path = candidate_dir / f"{group_id}{suffix}"
            path.write_bytes(f"synthetic-{index}-{suffix}".encode("ascii"))
            paths.append(path)
        groups.append(
            {
                "group_id": group_id,
                "files": [_digest(path) for path in sorted(paths, key=lambda item: item.name)],
            }
        )
        split = "train" if index <= 10 else "holdout"
        checks = {key: True for key in CHECK_KEYS}
        attested.append(
            {
                "group_id": group_id,
                "split": split,
                "accepted": True,
                "source_asset_id": f"asset_{index:02d}",
                "source_session_id": f"session_{index:02d}",
                "location_id": f"location_{index:02d}",
                "source_start_ms": 0,
                "source_end_ms": 3000,
                "checks": checks,
                "notes": "accepted synthetic fixture",
            }
        )
    structural = {
        "schema_version": "a2v-structural-report-v1",
        "status": "valid",
        "spec": {
            "width": 544,
            "height": 960,
            "frames": 89,
            "fps": 24,
            "sample_rate": 48000,
        },
        "groups": groups,
    }
    quality = {
        "schema_version": "a2v-quality-attestation-v1",
        "dataset_id": "dset_00000000000040008000000000000008",
        "rights_and_consent": {
            "confirmed": True,
            "reviewer_id": "reviewer_opaque_01",
            "reviewed_at_utc": "2026-07-16T00:10:00Z",
        },
        "groups": attested,
    }
    return structural, quality


def _resolved_groups(
    structural: dict[str, Any],
    quality_summary: dict[str, Any],
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    train_ids = set(quality_summary["accepted_train_group_ids"])
    holdout_ids = set(quality_summary["accepted_holdout_group_ids"])
    result = []
    for group in structural["groups"]:
        group_id = group["group_id"]
        split = "train" if group_id in train_ids else "holdout" if group_id in holdout_ids else None
        if split is None:
            continue
        result.append(
            {
                "group_id": group_id,
                "split": split,
                "files": [
                    {**record, "path": candidate_dir / record["name"]}
                    for record in group["files"]
                ],
            }
        )
    return result


def _execution_config(
    *,
    policy: dict[str, Any],
    price: dict[str, Any],
    dataset_path: Path,
    archive_path: Path,
    **overrides: Any,
) -> dict[str, Any]:
    value = {
        "schema_version": "a2v-execution-config-v2",
        "canonical_json_version": 1,
        "execution_id": EXECUTION_ID,
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "created_at_utc": "2026-07-16T00:20:00Z",
        "expires_at_utc": "2026-07-16T12:00:00Z",
        "endpoint": "fal-ai/ltx23-trainer-v2/a2v",
        "trigger_phrase": "chrx9_speech",
        "rank": 32,
        "steps": 1000,
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
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
        "dataset_manifest_sha256": sha256_file(dataset_path).sha256,
        "training_archive_sha256": sha256_file(archive_path).sha256,
        "standing_authorization_sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
        "price_evidence_sha256": hashlib.sha256(canonical_json_bytes(price)).hexdigest(),
        "price_source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": "0.006",
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
    }
    value.update(overrides)
    return value


def _migration_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    source_entries = []
    manifest_entries = []
    for index, (amount, state) in enumerate(zip(AMOUNTS, STATES, strict=True), start=1):
        source_id = _source_id(index)
        source_entries.append(
            {
                "id": source_id,
                "label": f"synthetic legacy item {index}",
                "amount_usd": amount,
                "status": state,
                "created_at": 1_700_000_000 + index,
                **({"finalized_at": 1_700_000_100 + index} if state != "reserved" else {}),
            }
        )
        manifest_entries.append(
            {
                "source_entry_id": source_id,
                "reservation_id": _typed_id("reservation", index + 20),
                "bundle_id": hashlib.sha256(f"legacy-{index}".encode("ascii")).hexdigest(),
                "execution_id": _typed_id("exec", index + 40),
                "amount_usd": amount,
                "state": state,
            }
        )
    source = {"cap_usd": "12.0000", "entries": source_entries}
    manifest = {
        "schema_version": "pilot-budget-migration-v1",
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
        "migration_id": MIGRATION_ID,
        "cap_usd": "12.0000",
        "source_ledger_sha256": hashlib.sha256(canonical_json_bytes(source)).hexdigest(),
        "created_at_utc": "2026-07-16T00:00:00Z",
        "entries": manifest_entries,
    }
    return source, manifest


def _secure_tree(root: Path) -> None:
    if os.name == "nt":
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)


def _install_ledger(private_root: Path) -> PilotLedger:
    ledger_dir = private_root / "pilots" / PILOT_ID / "ledger"
    ledger_dir.mkdir(parents=True)
    source, manifest = _migration_documents()
    evidence_dir = private_root / "migration-evidence"
    evidence_dir.mkdir()
    source_path = evidence_dir / "legacy.json"
    manifest_path = evidence_dir / "manifest.json"
    _write_json(source_path, source)
    _write_json(manifest_path, manifest)
    return migrate_legacy_ledger(source_path, manifest_path, ledger_dir / "pilot.sqlite3")


def _root_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "plan": sha256_file(run_dir / "plan.md"),
        "standing_authorization": sha256_file(run_dir / "control" / "standing-authorization.json"),
        "price_evidence": sha256_file(run_dir / "control" / "price-evidence.json"),
        "structural_report": sha256_file(run_dir / "control" / "structural-report.json"),
        "quality_attestation": sha256_file(run_dir / "control" / "quality-attestation.json"),
        "dataset_manifest": sha256_file(run_dir / "bundle" / "dataset-manifest.json"),
        "training_archive": sha256_file(run_dir / "bundle" / "training-data.zip"),
        "execution_config": sha256_file(run_dir / "control" / "execution-config.json"),
        "provider_validation_selection": sha256_file(
            run_dir / "validation" / "provider-validation-selection.json"
        ),
    }


def _rebind_bundle(run_dir: Path) -> str:
    control = run_dir / "control"
    bundle = run_dir / "bundle"
    validation = run_dir / "validation"
    policy = strict_load_json(control / "standing-authorization.json")
    price = strict_load_json(control / "price-evidence.json")
    config = strict_load_json(control / "execution-config.json")
    config["standing_authorization_sha256"] = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    config["price_evidence_sha256"] = hashlib.sha256(canonical_json_bytes(price)).hexdigest()
    config["dataset_manifest_sha256"] = sha256_file(bundle / "dataset-manifest.json").sha256
    config["training_archive_sha256"] = sha256_file(bundle / "training-data.zip").sha256
    _write_json(control / "execution-config.json", config)
    structural = strict_load_json(control / "structural-report.json")
    selection = strict_load_json(validation / "provider-validation-selection.json")
    selection["structural_report_sha256"] = hashlib.sha256(canonical_json_bytes(structural)).hexdigest()
    selection["execution_config_sha256"] = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    _write_json(validation / "provider-validation-selection.json", selection)
    dataset = strict_load_json(bundle / "dataset-manifest.json")
    root_manifest = build_root_manifest(
        execution_id=config["execution_id"],
        created_at_utc=config["created_at_utc"],
        expires_at_utc=config["expires_at_utc"],
        repository_commit="f" * 40,
        artifacts=_root_artifacts(run_dir),
        holdout_groups=dataset["groups"]["holdout"],
    )
    _write_json(bundle / "bundle-manifest.json", root_manifest)
    return compute_bundle_id(root_manifest)


def _write_ready_run(tmp_path: Path, *, with_receipt: bool = True) -> dict[str, Any]:
    private_root = tmp_path / "private"
    run_dir = private_root / "pilots" / PILOT_ID / "runs" / EXECUTION_ID
    candidate_dir = run_dir / "candidates"
    control = run_dir / "control"
    bundle = run_dir / "bundle"
    validation = run_dir / "validation"
    for directory in (control, bundle, validation):
        directory.mkdir(parents=True, exist_ok=True)
    structural, quality = _make_candidates(candidate_dir)
    quality_summary = validate_quality_and_splits(quality, structural)
    groups = _resolved_groups(structural, quality_summary, candidate_dir)
    archive_path = bundle / "training-data.zip"
    archive_digest = build_training_archive(groups, archive_path)
    dataset = build_dataset_manifest(
        structural,
        quality,
        archive_digest,
        candidate_dir=candidate_dir,
    )
    _write_json(bundle / "dataset-manifest.json", dataset)
    policy = _policy()
    price = _price()
    _write_json(control / "standing-authorization.json", policy)
    _write_json(control / "price-evidence.json", price)
    _write_json(control / "structural-report.json", structural)
    _write_json(control / "quality-attestation.json", quality)
    config = _execution_config(
        policy=policy,
        price=price,
        dataset_path=bundle / "dataset-manifest.json",
        archive_path=archive_path,
    )
    _write_json(control / "execution-config.json", config)
    selection = build_provider_validation_selection(
        structural_report=structural,
        quality_summary=quality_summary,
        execution_config=config,
        candidate_dir=candidate_dir,
        prompts={
            _group_id(11): "A close talking-head shot with natural speech.",
            _group_id(12): "A medium talking-head shot with steady eye contact.",
        },
    )
    _write_json(validation / "provider-validation-selection.json", selection)
    (run_dir / "plan.md").write_text("approved private synthetic plan", encoding="utf-8")
    bundle_id = _rebind_bundle(run_dir)
    ledger = _install_ledger(private_root)
    if with_receipt:
        receipt = issue_execution_receipt(
            policy,
            run_dir,
            expected_bundle_id=bundle_id,
            read_ledger_snapshot=lambda pilot, ledger_id, bundle_value, execution: ledger.preflight_snapshot(
                bundle_value, execution
            ),
            approval_id=APPROVAL_ID,
            issuer_process_id=PROCESS_ID,
            now=FIXED_TIME,
        )
        _write_json(control / "execution-approval.json", receipt.to_dict())
    _secure_tree(private_root)
    train_ids = set(quality_summary["accepted_train_group_ids"])
    train_report = {
        "schema_version": structural["schema_version"],
        "status": "valid",
        "spec": structural["spec"],
        "groups": [group for group in structural["groups"] if group["group_id"] in train_ids],
    }
    return {
        "private_root": private_root,
        "run_dir": run_dir,
        "bundle_id": bundle_id,
        "structural": structural,
        "train_report": train_report,
        "ledger": ledger,
    }


@pytest.fixture
def ready_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    fixture = _write_ready_run(tmp_path)
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)

    def structural_validator(path: Path, **_: Any) -> dict[str, Any]:
        if Path(path) == fixture["run_dir"] / "candidates":
            return fixture["structural"]
        return fixture["train_report"]

    monkeypatch.setattr(preflight, "validate_a2v_directory", structural_validator)
    return fixture


def _clock() -> datetime:
    return FIXED_TIME


def _run(fixture: dict[str, Any], *, require_receipt: bool = True, clock: Callable[[], datetime] = _clock) -> PreflightStatus:
    return run_preflight(
        fixture["run_dir"],
        fixture["bundle_id"],
        require_receipt=require_receipt,
        approved_private_root=fixture["private_root"],
        clock=clock,
    )


def test_gate_order_and_public_status_contract_are_exact(ready_run: dict[str, Any]) -> None:
    assert GATE_ORDER == (
        "private_root",
        "canonical_artifacts",
        "bundle_id",
        "root_artifact_hashes",
        "archive_inspection",
        "archive_structural_validation",
        "candidate_structural_rerun",
        "quality_attestation",
        "split_and_manifest",
        "provider_validation_selection",
        "request_allowlist",
        "price_freshness",
        "standing_policy",
        "receipt",
        "ledger_snapshot",
        "final_recheck",
    )
    report = _run(ready_run)
    public = report.to_public_dict()
    assert set(public) == {
        "schema_version",
        "status",
        "failed_gate",
        "receipt_required",
        "bundle_id",
        "execution_id",
        "counts",
        "budget",
        "passed_gates",
    }
    assert set(public["counts"]) == {
        "training_groups",
        "holdout_groups",
        "provider_validation_items",
    }
    assert set(public["budget"]) == {
        "committed_usd",
        "remaining_usd",
        "training_reservation_usd",
        "remaining_after_reservation_usd",
    }
    serialized = json.dumps(public, sort_keys=True)
    for prohibited in ("pilot_id", "ledger_id", "ledger_head_sha256", str(ready_run["private_root"])):
        assert prohibited not in serialized


def test_policy_only_and_receipt_required_ready_states(ready_run: dict[str, Any]) -> None:
    policy_only = _run(ready_run, require_receipt=False)
    assert policy_only.status == "ready_for_policy_issuance", (
        policy_only.failed_gate,
        policy_only.passed_gates,
    )
    assert policy_only.failed_gate is None
    assert policy_only.passed_gates == GATE_ORDER
    with pytest.raises(PreflightNotReady, match="preflight is not ready for paid execution"):
        policy_only.require_ready()

    paid = _run(ready_run, require_receipt=True)
    assert paid.status == "ready_for_paid_execution"
    assert paid.require_ready() is paid
    assert paid.committed_usd == "3.5409"
    assert paid.remaining_usd == "8.4591"
    assert paid.remaining_after_reservation_usd == "2.4591"
    assert paid.training_groups == 10
    assert paid.holdout_groups == 5
    assert paid.provider_validation_items == 2


def test_preflight_uses_exactly_one_atomic_ledger_snapshot(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = PilotLedger.preflight_snapshot

    def counted(self: PilotLedger, bundle_id: str, execution_id: str):
        nonlocal calls
        calls += 1
        return original(self, bundle_id, execution_id)

    monkeypatch.setattr(PilotLedger, "preflight_snapshot", counted)
    before = (ready_run["ledger"].path.read_bytes(), ready_run["ledger"].path.stat().st_mtime_ns)
    paid = _run(ready_run)
    assert paid.status == "ready_for_paid_execution", (paid.failed_gate, paid.passed_gates)
    after = (ready_run["ledger"].path.read_bytes(), ready_run["ledger"].path.stat().st_mtime_ns)
    assert calls == 1
    assert after == before


@pytest.mark.parametrize(
    ("mutation", "expected_gate"),
    [
        ("archive_byte", "root_artifact_hashes"),
        ("wrong_bundle", "bundle_id"),
        ("selected_holdout", "root_artifact_hashes"),
        ("wrong_receipt", "receipt"),
        ("wrong_ledger", "ledger_snapshot"),
    ],
)
def test_preflight_fails_closed_at_first_bound_gate(
    ready_run: dict[str, Any], mutation: str, expected_gate: str
) -> None:
    bundle_id = ready_run["bundle_id"]
    if mutation == "archive_byte":
        path = ready_run["run_dir"] / "bundle" / "training-data.zip"
        content = bytearray(path.read_bytes())
        content[10] ^= 1
        path.write_bytes(content)
    elif mutation == "wrong_bundle":
        bundle_id = "a" * 64
    elif mutation == "selected_holdout":
        path = ready_run["run_dir"] / "candidates" / f"{_group_id(11)}_audio.wav"
        path.write_bytes(b"changed selected holdout")
    elif mutation == "wrong_receipt":
        path = ready_run["run_dir"] / "control" / "execution-approval.json"
        receipt = strict_load_json(path)
        receipt["plan_sha256"] = "a" * 64
        _write_json(path, receipt)
    else:
        ready_run["ledger"].path.write_bytes(b"not sqlite")
    _secure_tree(ready_run["private_root"])
    report = run_preflight(
        ready_run["run_dir"],
        bundle_id,
        require_receipt=True,
        approved_private_root=ready_run["private_root"],
        clock=_clock,
    )
    assert report.status == "failed"
    assert report.failed_gate == expected_gate
    assert report.passed_gates == GATE_ORDER[: GATE_ORDER.index(expected_gate)]


@pytest.mark.parametrize("artifact", ["root_manifest", "receipt"])
def test_parsed_control_objects_are_bound_to_their_pinned_bytes(
    ready_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    import ltx_lora_pilot.preflight as preflight

    target = (
        ready_run["run_dir"] / "bundle" / "bundle-manifest.json"
        if artifact == "root_manifest"
        else ready_run["run_dir"] / "control" / "execution-approval.json"
    )
    original = preflight._pin_file
    changed = False

    def swap_before_pin(path: Path):
        nonlocal changed
        if path == target and not changed:
            changed = True
            value = strict_load_json(path)
            if artifact == "root_manifest":
                value["created_at_utc"] = "2026-07-16T00:00:01Z"
            else:
                value["plan_sha256"] = "a" * 64
            _write_json(path, value)
        return original(path)

    monkeypatch.setattr(preflight, "_pin_file", swap_before_pin)
    report = _run(ready_run, require_receipt=True)
    assert changed is True
    assert report.failed_gate == "root_artifact_hashes"


@pytest.mark.parametrize("artifact", ["execution_config", "provider_selection"])
def test_every_parsed_root_artifact_is_bound_to_its_pinned_bytes(
    ready_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    """A valid parsed object must not authorize different root-bound bytes."""

    import ltx_lora_pilot.preflight as preflight

    if artifact == "execution_config":
        target = ready_run["run_dir"] / "control" / "execution-config.json"
        invalid = strict_load_json(target)
        invalid["rank"] = 31
    else:
        target = (
            ready_run["run_dir"]
            / "validation"
            / "provider-validation-selection.json"
        )
        invalid = strict_load_json(target)
        invalid["items"][0]["prompt"] = invalid["items"][0]["prompt"].replace(
            "A", "B", 1
        )

    valid_bytes = target.read_bytes()
    invalid_bytes = canonical_json_bytes(invalid)
    assert len(invalid_bytes) == len(valid_bytes)

    # Bind the root to the invalid bytes, then temporarily expose valid bytes
    # only to the canonical parser.  The swap back happens before pinning while
    # preserving the metadata-only security snapshot.
    target.write_bytes(invalid_bytes)
    ready_run["bundle_id"] = _rebind_bundle(ready_run["run_dir"])
    target.write_bytes(valid_bytes)
    original_stat = target.stat()
    _secure_tree(ready_run["private_root"])

    original_pin = preflight._pin_file
    swapped = False

    def swap_before_pin(path: Path):
        nonlocal swapped
        if path == target and not swapped:
            swapped = True
            target.write_bytes(invalid_bytes)
            os.utime(
                target,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
        return original_pin(path)

    monkeypatch.setattr(preflight, "_pin_file", swap_before_pin)
    report = _run(ready_run, require_receipt=False)
    assert swapped is True
    assert report.failed_gate == "root_artifact_hashes"


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_key", "float", "unknown_field", "noncanonical_whitespace"],
)
def test_canonical_artifact_adversarial_matrix(
    ready_run: dict[str, Any], mutation: str
) -> None:
    control = ready_run["run_dir"] / "control"
    if mutation == "duplicate_key":
        path = control / "price-evidence.json"
        value = path.read_text(encoding="utf-8")
        path.write_text(
            value[:-1] + ',"source_url":"https://invalid.example"}',
            encoding="utf-8",
        )
    elif mutation == "float":
        path = control / "execution-config.json"
        config = strict_load_json(path)
        config["steps"] = 1000.0
        path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    elif mutation == "unknown_field":
        path = control / "standing-authorization.json"
        policy = strict_load_json(path)
        policy["unexpected"] = True
        _write_json(path, policy)
    else:
        path = control / "quality-attestation.json"
        path.write_bytes(path.read_bytes() + b"\n")
    _secure_tree(ready_run["private_root"])
    assert _run(ready_run, require_receipt=True).failed_gate == "canonical_artifacts"


@pytest.mark.parametrize("mutation", ["missing", "expired"])
def test_required_receipt_adversarial_matrix(
    ready_run: dict[str, Any], mutation: str
) -> None:
    path = ready_run["run_dir"] / "control" / "execution-approval.json"
    if mutation == "missing":
        path.unlink()
    else:
        receipt = strict_load_json(path)
        receipt["expires_at_utc"] = "2026-07-15T00:00:00Z"
        _write_json(path, receipt)
    _secure_tree(ready_run["private_root"])
    report = _run(ready_run, require_receipt=True)
    assert report.failed_gate == "receipt"
    assert report.status == "failed"


@pytest.mark.parametrize("mutation", ["replay", "insufficient", "head"])
def test_ledger_snapshot_adversarial_matrix(
    ready_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    original = PilotLedger.preflight_snapshot

    def altered(self: PilotLedger, bundle_id: str, execution_id: str):
        snapshot = original(self, bundle_id, execution_id)
        if mutation == "replay":
            return replace(snapshot, replay_detected=True)
        if mutation == "insufficient":
            return replace(
                snapshot,
                committed_usd="6.0001",
                remaining_usd="5.9999",
            )
        return replace(snapshot, head_sha256="a" * 64)

    monkeypatch.setattr(PilotLedger, "preflight_snapshot", altered)
    report = _run(ready_run, require_receipt=True)
    assert report.failed_gate == "ledger_snapshot"
    assert report.status == "failed"


def _assert_archive_inspection_rejects(
    ready_run: dict[str, Any], kind: str
) -> None:
    archive = ready_run["run_dir"] / "bundle" / "training-data.zip"
    names = [record["name"] for record in strict_load_json(
        ready_run["run_dir"] / "bundle" / "dataset-manifest.json"
    )["training_members"]]
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED if kind == "compression" else zipfile.ZIP_STORED,
        allowZip64=False,
    ) as output:
        selected = names[:]
        if kind == "traversal":
            selected.append("../escape.txt")
        elif kind == "duplicate":
            selected.append(selected[0])
        elif kind == "case":
            selected.append(selected[0].upper())
        with warnings.catch_warnings():
            if kind == "duplicate":
                warnings.simplefilter("ignore", UserWarning)
            for index, name in enumerate(selected):
                info = zipfile.ZipInfo(name, date_time=FIXED_ARCHIVE_DATETIME)
                info.create_system = 3
                info.external_attr = (
                    (stat.S_IFLNK | 0o600) << 16
                    if kind == "link" and index == 0
                    else FIXED_EXTERNAL_ATTR
                )
                payload = b"0" * (
                    2_000_000 if kind == "compression" and index == 0 else 8
                )
                output.writestr(info, payload)
    dataset_path = ready_run["run_dir"] / "bundle" / "dataset-manifest.json"
    dataset = strict_load_json(dataset_path)
    dataset["archive"] = asdict(sha256_file(archive))
    _write_json(dataset_path, dataset)
    ready_run["bundle_id"] = _rebind_bundle(ready_run["run_dir"])
    _secure_tree(ready_run["private_root"])
    assert _run(ready_run, require_receipt=False).failed_gate == "archive_inspection"


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "case", "link", "compression"])
def test_archive_adversarial_matrix(kind: str, ready_run: dict[str, Any]) -> None:
    _assert_archive_inspection_rejects(ready_run, kind)


def test_archive_is_safely_streamed_and_never_uses_zip_extract(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zipfile.ZipFile, "extract", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extract called")))
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extractall called")))
    assert _run(ready_run, require_receipt=False).status == "ready_for_policy_issuance"
    assert not (ready_run["run_dir"] / ".preflight-tmp").exists()


def test_archive_inspection_and_extraction_share_one_open_archive_object(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A byte-different archive cannot be substituted for a second path open."""

    archive_path = ready_run["run_dir"] / "bundle" / "training-data.zip"
    alternate_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_path, "r") as source, zipfile.ZipFile(
        alternate_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
    ) as alternate:
        for info in source.infolist():
            replacement = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            replacement.create_system = info.create_system
            replacement.external_attr = info.external_attr
            replacement.compress_type = zipfile.ZIP_DEFLATED
            alternate.writestr(replacement, source.read(info))
    alternate_bytes = alternate_buffer.getvalue()
    assert alternate_bytes != archive_path.read_bytes()

    original_zipfile = zipfile.ZipFile
    path_opens = 0
    zipfile_opens = 0

    def substitute_second_path_open(file: Any, *args: Any, **kwargs: Any):
        nonlocal path_opens, zipfile_opens
        zipfile_opens += 1
        if isinstance(file, (str, os.PathLike)) and Path(file) == archive_path:
            path_opens += 1
            if path_opens == 2:
                return original_zipfile(io.BytesIO(alternate_bytes), *args, **kwargs)
        return original_zipfile(file, *args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", substitute_second_path_open)
    report = _run(ready_run, require_receipt=False)
    assert report.status == "ready_for_policy_issuance"
    assert zipfile_opens == 1
    assert path_opens == 0


def test_final_recheck_rejects_post_snapshot_ledger_commit_with_restored_metadata(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_snapshot = PilotLedger.preflight_snapshot
    mutated = False

    def advance_after_snapshot(
        self: PilotLedger, bundle_id: str, execution_id: str
    ):
        nonlocal mutated
        snapshot = original_snapshot(self, bundle_id, execution_id)
        if not mutated:
            mutated = True
            before = self.path.stat()
            self.reserve("b" * 64, _typed_id("exec", 99), "0.1000")
            os.utime(
                self.path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            after = self.path.stat()
            assert after.st_size == before.st_size
            assert after.st_ino == before.st_ino
        return snapshot

    monkeypatch.setattr(PilotLedger, "preflight_snapshot", advance_after_snapshot)
    report = _run(ready_run, require_receipt=True)
    assert mutated is True
    assert report.failed_gate == "final_recheck"


def test_semantic_gates_are_attributed_after_rebinding(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import ltx_lora_pilot.preflight as preflight

    cases: list[tuple[str, Callable[[], None]]] = []
    control = ready_run["run_dir"] / "control"
    bundle = ready_run["run_dir"] / "bundle"
    validation = ready_run["run_dir"] / "validation"

    def candidate_failure() -> None:
        original = preflight.validate_a2v_directory

        def fail_candidate(path: Path, **kwargs: Any):
            if Path(path) == ready_run["run_dir"] / "candidates":
                raise ValueError("candidate structural mismatch")
            return original(path, **kwargs)

        monkeypatch.setattr(preflight, "validate_a2v_directory", fail_candidate)

    cases.append(("candidate_structural_rerun", candidate_failure))
    for expected_gate, mutate in cases:
        mutate()
        assert _run(ready_run, require_receipt=False).failed_gate == expected_gate

    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)
    monkeypatch.setattr(
        preflight,
        "validate_a2v_directory",
        lambda path, **kwargs: ready_run["structural"]
        if Path(path) == ready_run["run_dir"] / "candidates"
        else ready_run["train_report"],
    )

    quality_path = control / "quality-attestation.json"
    quality = strict_load_json(quality_path)
    quality["rights_and_consent"]["confirmed"] = False
    _write_json(quality_path, quality)
    ready_run["bundle_id"] = _rebind_bundle(ready_run["run_dir"])
    _secure_tree(ready_run["private_root"])
    assert _run(ready_run, require_receipt=False).failed_gate == "quality_attestation"

    # Restore quality, then make the bound dataset manifest stale.
    quality["rights_and_consent"]["confirmed"] = True
    _write_json(quality_path, quality)
    dataset_path = bundle / "dataset-manifest.json"
    dataset = strict_load_json(dataset_path)
    dataset["counts"]["train_groups"] = 11
    _write_json(dataset_path, dataset)
    ready_run["bundle_id"] = _rebind_bundle(ready_run["run_dir"])
    _secure_tree(ready_run["private_root"])
    assert _run(ready_run, require_receipt=False).failed_gate == "split_and_manifest"

    # Restore with a fresh independent fixture for selection/config/policy/price attribution.


@pytest.mark.parametrize(
    ("kind", "gate"),
    [
        ("selection", "provider_validation_selection"),
        ("config", "request_allowlist"),
        ("price", "price_freshness"),
        ("policy", "standing_policy"),
    ],
)
def test_late_semantic_gate_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    gate: str,
) -> None:
    fixture = _write_ready_run(tmp_path)
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)
    monkeypatch.setattr(
        preflight,
        "validate_a2v_directory",
        lambda path, **kwargs: fixture["structural"] if Path(path) == fixture["run_dir"] / "candidates" else fixture["train_report"],
    )
    control = fixture["run_dir"] / "control"
    validation = fixture["run_dir"] / "validation"
    if kind == "selection":
        selection_path = validation / "provider-validation-selection.json"
        selection = strict_load_json(selection_path)
        train_id = _group_id(1)
        structural = fixture["structural"]
        records = {record["name"]: record for record in structural["groups"][0]["files"]}
        selection["items"][0] = {
            "group_id": train_id,
            "prompt": "A canonical but invalid training selection.",
            "image": records[f"{train_id}_start.png"],
            "audio": records[f"{train_id}_audio.wav"],
        }
        _write_json(selection_path, selection)
    elif kind == "config":
        path = control / "execution-config.json"
        config = strict_load_json(path)
        config["steps"] = 999
        _write_json(path, config)
    elif kind == "price":
        path = control / "price-evidence.json"
        price = strict_load_json(path)
        price["retrieved_at_utc"] = "2026-07-14T00:00:00Z"
        price["expires_at_utc"] = "2026-07-15T00:00:00Z"
        _write_json(path, price)
    else:
        path = control / "standing-authorization.json"
        policy = strict_load_json(path)
        policy["executions"] = 2
        _write_json(path, policy)
    fixture["bundle_id"] = _rebind_bundle(fixture["run_dir"])
    _secure_tree(fixture["private_root"])
    assert _run(fixture, require_receipt=False).failed_gate == gate


def test_archive_structural_validation_failure_is_distinct(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import ltx_lora_pilot.preflight as preflight

    original = preflight.validate_a2v_directory

    def fail_extracted(path: Path, **kwargs: Any):
        if Path(path) != ready_run["run_dir"] / "candidates":
            raise ValueError("extracted bytes failed")
        return original(path, **kwargs)

    monkeypatch.setattr(preflight, "validate_a2v_directory", fail_extracted)
    assert _run(ready_run, require_receipt=False).failed_gate == "archive_structural_validation"


def test_private_root_rejects_dacl_denial_hardlink_ads_and_repository_nesting(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(
        preflight,
        "_WINDOWS_DACL_CHECK",
        lambda path: (_ for _ in ()).throw(ValueError("denied")),
    )
    assert _run(ready_run).failed_gate == "private_root"
    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)

    source = ready_run["run_dir"] / "plan.md"
    alias = ready_run["run_dir"] / "plan-hardlink.md"
    os.link(source, alias)
    assert _run(ready_run).failed_gate == "private_root"
    alias.unlink()

    ads_run = Path(str(ready_run["run_dir"]) + ":stream")
    report = run_preflight(
        ads_run,
        ready_run["bundle_id"],
        require_receipt=True,
        approved_private_root=ready_run["private_root"],
        clock=_clock,
    )
    assert report.failed_gate == "private_root"

    nested = run_preflight(
        ready_run["run_dir"],
        ready_run["bundle_id"],
        require_receipt=True,
        approved_private_root=ROOT,
        clock=_clock,
    )
    assert nested.failed_gate == "private_root"


def _assert_final_recheck_detects_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _write_ready_run(tmp_path)
    import ltx_lora_pilot.preflight as preflight

    monkeypatch.setattr(preflight, "_WINDOWS_DACL_CHECK", lambda path: None)
    monkeypatch.setattr(
        preflight,
        "validate_a2v_directory",
        lambda path, **kwargs: fixture["structural"] if Path(path) == fixture["run_dir"] / "candidates" else fixture["train_report"],
    )
    calls = 0

    def racing_clock() -> datetime:
        nonlocal calls
        calls += 1
        if calls == 4:
            if kind == "path":
                (fixture["run_dir"] / "plan.md").write_text("changed at final gate", encoding="utf-8")
            elif kind == "sidecar":
                Path(str(fixture["ledger"].path) + "-wal").write_bytes(b"new sidecar")
            else:
                return FIXED_TIME + timedelta(days=2)
        return FIXED_TIME

    report = _run(fixture, require_receipt=True, clock=racing_clock)
    assert report.failed_gate == "final_recheck"


@pytest.mark.parametrize("kind", ["expiry", "path", "sidecar"])
def test_final_recheck_race_matrix(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_final_recheck_detects_race(tmp_path, monkeypatch, kind)


def test_failure_report_is_sanitized_and_written_only_after_trusted_root(
    ready_run: dict[str, Any], tmp_path: Path
) -> None:
    archive = ready_run["run_dir"] / "bundle" / "training-data.zip"
    archive.write_bytes(b"private marker /secret/path FAL_KEY=do-not-print")
    _secure_tree(ready_run["private_root"])
    report = _run(ready_run)
    output_path = ready_run["run_dir"] / "control" / "preflight-report.json"
    assert report.failed_gate == "root_artifact_hashes"
    public = strict_load_json(output_path)
    serialized = canonical_json_bytes(public).decode("utf-8")
    assert str(ready_run["private_root"]) not in serialized
    assert "FAL_KEY" not in serialized
    assert "secret" not in serialized

    untrusted = tmp_path / "untrusted"
    failed = run_preflight(
        untrusted,
        "a" * 64,
        require_receipt=True,
        approved_private_root=untrusted,
        clock=_clock,
    )
    assert failed.failed_gate == "private_root"
    assert not (untrusted / "control" / "preflight-report.json").exists()


def test_no_provider_or_paid_capability_is_imported_or_called(
    ready_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PilotLedger, "reserve_training", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reservation called")))
    assert _run(ready_run).status == "ready_for_paid_execution"
    source = (ROOT / "src" / "ltx_lora_pilot" / "preflight.py").read_text("utf-8")
    script = SCRIPT.read_text("utf-8")
    prohibited = (
        "fal_client",
        "FAL_KEY",
        "upload_file",
        "submit(",
        "reserve_training(",
        "capture_price_evidence(",
    )
    for marker in prohibited:
        assert marker not in source
        assert marker not in script


def test_cli_surface_is_constrained_and_failures_are_neutral() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--run-dir" in completed.stdout
    assert "--confirm-bundle-id" in completed.stdout
    assert "--require-receipt" in completed.stdout
    for prohibited in ("--execute", "--ledger", "--budget", "--endpoint", "--steps", "--private-root"):
        assert prohibited not in completed.stdout

    failed = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", "relative", "--confirm-bundle-id", "a" * 64],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "LTX_LORA_PRIVATE_ROOT"},
    )
    assert failed.returncode != 0
    assert "relative" not in failed.stdout + failed.stderr
