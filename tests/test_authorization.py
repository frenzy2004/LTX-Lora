from __future__ import annotations

import hashlib
import importlib.util
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request

import pytest

from ltx_lora_pilot.a2v_bundle import build_root_manifest, compute_bundle_id
from ltx_lora_pilot.artifacts import (
    FileDigest,
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


def _arbitrary_digest(label: str) -> FileDigest:
    content = label.encode("ascii")
    return FileDigest(
        name=f"{label}.json",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _execution_config(**overrides: Any) -> dict[str, Any]:
    value = {
        "execution_id": EXECUTION_ID,
        "endpoint": ENDPOINT,
        "steps": 1_000,
        "training_max_usd": "6.0000",
        "validation_allocation_usd": "1.2500",
        "cumulative_cap_usd": "12.0000",
        "pilot_id": PILOT_ID,
        "ledger_id": LEDGER_ID,
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
    root_expires_at_utc: str = "2026-07-15T20:00:00Z",
) -> tuple[Path, dict[str, Any], str]:
    run_dir = tmp_path / name
    control_dir = run_dir / "control"
    bundle_dir = run_dir / "bundle"
    control_dir.mkdir(parents=True)
    bundle_dir.mkdir()

    policy_value = valid_policy() if policy is None else policy
    price_value = valid_price_evidence() if price is None else price
    config_value = _execution_config() if config is None else config
    plan_path = run_dir / "plan.md"
    policy_path = control_dir / "standing-authorization.json"
    price_path = control_dir / "price-evidence.json"
    config_path = control_dir / "execution-config.json"
    dataset_path = bundle_dir / "dataset-manifest.json"
    archive_path = bundle_dir / "training-data.zip"

    plan_path.write_bytes(plan_content)
    atomic_write_json(policy_path, policy_value)
    atomic_write_json(price_path, price_value)
    atomic_write_json(config_path, config_value)
    atomic_write_json(dataset_path, {"schema_version": "synthetic-dataset-v1"})
    archive_path.write_bytes(b"synthetic deterministic archive")

    artifacts = {
        role: _arbitrary_digest(role)
        for role in {
            "structural_report",
            "quality_attestation",
            "provider_validation_selection",
        }
    }
    artifacts.update(
        {
            "plan": sha256_file(plan_path),
            "standing_authorization": sha256_file(policy_path),
            "price_evidence": sha256_file(price_path),
            "execution_config": sha256_file(config_path),
            "dataset_manifest": sha256_file(dataset_path),
            "training_archive": sha256_file(archive_path),
        }
    )
    root_manifest = build_root_manifest(
        execution_id=root_execution_id or config_value["execution_id"],
        created_at_utc="2026-07-15T01:30:00Z",
        expires_at_utc=root_expires_at_utc,
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
    response = b"Training costs $0.006 * steps; 1,000 steps cost $6.00."
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


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            b"Training costs 0.006 * steps; 1,000 steps cost $6.00.",
            "unexpected A2V rate",
        ),
        (
            b"$0.006 * steps; $0.006 * steps; 1,000 steps cost $6.00.",
            "unexpected A2V rate",
        ),
        (b"Training costs $0.006 * steps.", "unexpected 1,000-step cost"),
        (
            b"Training costs $0.006 * steps; 1,000 steps cost $7.00.",
            "unexpected 1,000-step cost",
        ),
        (
            b"Training costs $0.006 * steps; 1,000 steps cost $6.00; "
            b"1,000 steps cost $6.00.",
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
    opened: dict[str, Any] = {}

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

    monkeypatch.setattr(
        _api().urllib_request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    assert _api()._fetch_official_price(OFFICIAL_PRICE_URL) == response_body
    request = opened["request"]
    headers = {name.lower(): value for name, value in request.header_items()}
    assert request.full_url == OFFICIAL_PRICE_URL
    assert request.get_method() == "GET"
    assert "authorization" not in headers
    assert "cookie" not in headers
    assert opened["timeout"] == 10
    assert opened["read_size"] == 1_048_577

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
