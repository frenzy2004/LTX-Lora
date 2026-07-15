from __future__ import annotations

import hashlib
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import request as urllib_request

from .a2v_bundle import compute_bundle_id
from .artifacts import canonical_json_bytes, sha256_file, strict_load_json


A2V_ENDPOINT = "fal-ai/ltx23-trainer-v2/a2v"
OFFICIAL_PRICE_URL = "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v"
A2V_EXECUTIONS = 1
A2V_STEPS = 1_000
A2V_RATE_USD_PER_STEP = "0.006"
TRAINING_MAX_USD = "6.0000"
VALIDATION_ALLOCATION_MAX_USD = "1.2500"
CUMULATIVE_CAP_USD = "12.0000"
APPROVAL_SCHEMA_VERSION = "a2v-execution-approval-v1"
APPROVAL_STATUS = "approved_for_paid_execution"
APPROVAL_MODE = "standing_policy"
PRICE_EVIDENCE_MAX_AGE = timedelta(hours=24)
PRICE_FETCH_TIMEOUT_SECONDS = 10
MAX_PRICE_RESPONSE_BYTES = 1_048_576

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
MONEY_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{4}", re.ASCII)
UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
    re.ASCII,
)
UUID4_HEX = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
POLICY_ID_PATTERN = re.compile(rf"policy_{UUID4_HEX}", re.ASCII)
APPROVAL_ID_PATTERN = re.compile(rf"approval_{UUID4_HEX}", re.ASCII)
PROCESS_ID_PATTERN = re.compile(rf"process_{UUID4_HEX}", re.ASCII)
PILOT_ID_PATTERN = re.compile(rf"pilot_{UUID4_HEX}", re.ASCII)
LEDGER_ID_PATTERN = re.compile(rf"ledger_{UUID4_HEX}", re.ASCII)
EXECUTION_ID_PATTERN = re.compile(rf"exec_{UUID4_HEX}", re.ASCII)

STANDING_AUTHORIZATION_FIELDS = frozenset(
    {
        "policy_id",
        "source_sha256",
        "endpoint",
        "executions",
        "steps",
        "training_max_usd",
        "validation_allocation_usd",
        "cumulative_cap_usd",
        "expires_at_utc",
    }
)
PRICE_EVIDENCE_FIELDS = frozenset(
    {
        "source_url",
        "rate_usd_per_step",
        "response_sha256",
        "retrieved_at_utc",
        "expires_at_utc",
    }
)
EXECUTION_RECEIPT_FIELDS = frozenset(
    {
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
)
EXECUTION_CONFIG_REQUIRED_FIELDS = frozenset(
    {
        "execution_id",
        "endpoint",
        "steps",
        "training_max_usd",
        "validation_allocation_usd",
        "cumulative_cap_usd",
        "pilot_id",
        "ledger_id",
    }
)
RATE_FORMULA_PATTERN = re.compile(
    r"(?<![0-9.])(?P<currency>\$?)(?P<rate>[0-9]+\.[0-9]+)"
    r"\s*\*\s*steps\b",
    re.IGNORECASE | re.ASCII,
)
COST_AFTER_STEPS_PATTERN = re.compile(
    r"\b1,000\s+steps\b.{0,80}?\$(?P<cost>[0-9]+\.[0-9]+)",
    re.IGNORECASE | re.ASCII | re.DOTALL,
)
COST_BEFORE_STEPS_PATTERN = re.compile(
    r"\$(?P<cost>[0-9]+\.[0-9]+)(?![0-9.])(?!\s*\*\s*steps)"
    r".{0,80}?\b(?:for\s+)?1,000\s+steps\b",
    re.IGNORECASE | re.ASCII | re.DOTALL,
)


def _exact_dict(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} must contain the exact fields")
    return value


def _parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    if type(value) is not str or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from exc


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _now(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    if type(value) is str:
        return _parse_utc_timestamp(value, label="current time")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("current time must be UTC")
        return value.astimezone(timezone.utc).replace(microsecond=0)
    raise ValueError("current time must be a UTC datetime")


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _typed_id(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a typed opaque UUIDv4")
    return value


def _money(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or MONEY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must use four decimal places")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must use four decimal places") from exc


@dataclass(frozen=True)
class StandingAuthorization:
    policy_id: str
    source_sha256: str
    endpoint: str
    executions: int
    steps: int
    training_max_usd: str
    validation_allocation_usd: str
    cumulative_cap_usd: str
    expires_at_utc: str

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        now: str | datetime | None = None,
    ) -> StandingAuthorization:
        data = _exact_dict(
            value,
            STANDING_AUTHORIZATION_FIELDS,
            label="standing authorization",
        )
        _typed_id(data["policy_id"], POLICY_ID_PATTERN, label="policy_id")
        _sha256(data["source_sha256"], label="source_sha256")
        if type(data["endpoint"]) is not str or data["endpoint"] != A2V_ENDPOINT:
            raise ValueError(f"endpoint must be {A2V_ENDPOINT}")
        if type(data["executions"]) is not int or data["executions"] != A2V_EXECUTIONS:
            raise ValueError("executions must be 1")
        if type(data["steps"]) is not int or data["steps"] != A2V_STEPS:
            raise ValueError("steps must be 1000")
        if data["training_max_usd"] != TRAINING_MAX_USD:
            raise ValueError("training ceiling must be 6.0000")
        validation_allocation = _money(
            data["validation_allocation_usd"],
            label="validation allocation",
        )
        if validation_allocation < 0 or validation_allocation > Decimal(
            VALIDATION_ALLOCATION_MAX_USD
        ):
            raise ValueError("validation allocation must be at most 1.2500")
        if data["cumulative_cap_usd"] != CUMULATIVE_CAP_USD:
            raise ValueError("cumulative cap must be 12.0000")
        expires = _parse_utc_timestamp(
            data["expires_at_utc"], label="expires_at_utc"
        )
        if expires <= _now(now):
            raise ValueError("standing authorization expired")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriceEvidence:
    source_url: str
    rate_usd_per_step: str
    response_sha256: str
    retrieved_at_utc: str
    expires_at_utc: str

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        now: str | datetime | None = None,
    ) -> PriceEvidence:
        data = _exact_dict(value, PRICE_EVIDENCE_FIELDS, label="price evidence")
        if type(data["source_url"]) is not str or data["source_url"] != OFFICIAL_PRICE_URL:
            raise ValueError("price evidence must use the official HTTPS URL")
        if (
            type(data["rate_usd_per_step"]) is not str
            or data["rate_usd_per_step"] != A2V_RATE_USD_PER_STEP
        ):
            raise ValueError("unexpected A2V rate")
        _sha256(data["response_sha256"], label="response_sha256")
        retrieved = _parse_utc_timestamp(
            data["retrieved_at_utc"], label="retrieved_at_utc"
        )
        expires = _parse_utc_timestamp(
            data["expires_at_utc"], label="expires_at_utc"
        )
        if expires <= retrieved or expires - retrieved > PRICE_EVIDENCE_MAX_AGE:
            raise ValueError("price evidence expiry must be within 24 hours")
        current = _now(now)
        if retrieved > current:
            raise ValueError("price evidence retrieval is in the future")
        if expires <= current:
            raise ValueError("price evidence expired")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _OfficialPriceRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib_request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib_request.Request | None:
        if newurl != OFFICIAL_PRICE_URL:
            raise ValueError("price redirect left the official HTTPS URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_official_price(url: str) -> bytes:
    if type(url) is not str or url != OFFICIAL_PRICE_URL:
        raise ValueError("price fetch requires the official HTTPS URL")
    opener = urllib_request.build_opener(_OfficialPriceRedirectHandler())
    request = urllib_request.Request(
        url,
        headers={"Accept": "text/html,application/xhtml+xml"},
        method="GET",
    )
    with opener.open(request, timeout=PRICE_FETCH_TIMEOUT_SECONDS) as response:
        if response.geturl() != OFFICIAL_PRICE_URL:
            raise ValueError("price response left the official HTTPS URL")
        status = getattr(response, "status", 200)
        if type(status) is not int or status != 200:
            raise ValueError("official price response was unsuccessful")
        content = response.read(MAX_PRICE_RESPONSE_BYTES + 1)
    return content


def _verify_price_statement(content: bytes) -> None:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("official price response must be UTF-8") from exc
    formulas = list(RATE_FORMULA_PATTERN.finditer(text))
    if (
        len(formulas) != 1
        or formulas[0].group("currency") != "$"
        or formulas[0].group("rate") != A2V_RATE_USD_PER_STEP
    ):
        raise ValueError("unexpected A2V rate")
    costs = [
        match.group("cost")
        for pattern in (COST_AFTER_STEPS_PATTERN, COST_BEFORE_STEPS_PATTERN)
        for match in pattern.finditer(text)
    ]
    if costs != ["6.00"]:
        raise ValueError("unexpected 1,000-step cost")


def capture_price_evidence(
    *,
    fetch: Callable[[str], bytes] | None = None,
    now: str | datetime | None = None,
) -> PriceEvidence:
    current = _now(now)
    fetcher = fetch or _fetch_official_price
    try:
        content = fetcher(OFFICIAL_PRICE_URL)
    except Exception as exc:
        raise ValueError("price fetch failed") from exc
    if type(content) is not bytes or not content or len(content) > MAX_PRICE_RESPONSE_BYTES:
        raise ValueError("official price response is invalid or too large")
    _verify_price_statement(content)
    evidence = {
        "source_url": OFFICIAL_PRICE_URL,
        "rate_usd_per_step": A2V_RATE_USD_PER_STEP,
        "response_sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": _format_utc(current),
        "expires_at_utc": _format_utc(current + PRICE_EVIDENCE_MAX_AGE),
    }
    return PriceEvidence.from_dict(evidence, now=current)


@dataclass(frozen=True)
class ExecutionReceipt:
    schema_version: str
    approval_id: str
    status: str
    approval_mode: str
    policy_id: str
    standing_authorization_sha256: str
    issuer_process_id: str
    issued_at_utc: str
    bundle_id: str
    execution_id: str
    expires_at_utc: str
    plan_sha256: str
    dataset_manifest_sha256: str
    training_archive_sha256: str
    execution_config_sha256: str
    pilot_id: str
    ledger_id: str
    training_max_usd: str
    validation_allocation_usd: str
    cumulative_cap_usd: str
    steps: int

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        now: str | datetime | None = None,
    ) -> ExecutionReceipt:
        data = _exact_dict(
            value,
            EXECUTION_RECEIPT_FIELDS,
            label="execution approval",
        )
        if data["schema_version"] != APPROVAL_SCHEMA_VERSION:
            raise ValueError("execution approval schema mismatch")
        if data["status"] != APPROVAL_STATUS:
            raise ValueError("execution approval status mismatch")
        if data["approval_mode"] != APPROVAL_MODE:
            raise ValueError("execution approval mode mismatch")
        if (
            type(data["approval_id"]) is str
            and EXECUTION_ID_PATTERN.fullmatch(data["approval_id"]) is not None
        ):
            raise ValueError("approval_id must not use a replay ID")
        _typed_id(data["approval_id"], APPROVAL_ID_PATTERN, label="approval_id")
        _typed_id(data["policy_id"], POLICY_ID_PATTERN, label="policy_id")
        _typed_id(
            data["issuer_process_id"],
            PROCESS_ID_PATTERN,
            label="issuer_process_id",
        )
        _typed_id(data["execution_id"], EXECUTION_ID_PATTERN, label="execution_id")
        _typed_id(data["pilot_id"], PILOT_ID_PATTERN, label="pilot_id")
        _typed_id(data["ledger_id"], LEDGER_ID_PATTERN, label="ledger_id")
        for field in (
            "standing_authorization_sha256",
            "bundle_id",
            "plan_sha256",
            "dataset_manifest_sha256",
            "training_archive_sha256",
            "execution_config_sha256",
        ):
            _sha256(data[field], label=field)
        issued = _parse_utc_timestamp(data["issued_at_utc"], label="issued_at_utc")
        expires = _parse_utc_timestamp(data["expires_at_utc"], label="expires_at_utc")
        if expires <= issued:
            raise ValueError("execution approval expiry must follow issuance")
        current = _now(now)
        if expires <= current:
            raise ValueError("execution approval expired")
        if issued > current:
            raise ValueError("execution approval issuance is in the future")
        if data["training_max_usd"] != TRAINING_MAX_USD:
            raise ValueError("execution approval training ceiling mismatch")
        if data["validation_allocation_usd"] != VALIDATION_ALLOCATION_MAX_USD:
            raise ValueError("execution approval validation allocation mismatch")
        if data["cumulative_cap_usd"] != CUMULATIVE_CAP_USD:
            raise ValueError("execution approval cumulative cap mismatch")
        if type(data["steps"]) is not int or data["steps"] != A2V_STEPS:
            raise ValueError("execution approval step count mismatch")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _BundleFacts:
    bundle_id: str
    execution_id: str
    expires_at_utc: str
    policy: StandingAuthorization
    price_evidence: PriceEvidence
    standing_authorization_sha256: str
    plan_sha256: str
    dataset_manifest_sha256: str
    training_archive_sha256: str
    execution_config_sha256: str
    pilot_id: str
    ledger_id: str


def _is_symlink_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_directory(path: Path) -> None:
    if _is_symlink_or_junction(path) or not path.is_dir():
        raise ValueError("private bundle directory is unavailable")


def _require_regular_file(path: Path) -> None:
    if _is_symlink_or_junction(path):
        raise ValueError("private bundle input is unavailable")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError("private bundle input is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("private bundle input is unavailable")


def _load_canonical_object(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    value = strict_load_json(path)
    if type(value) is not dict or path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("private bundle JSON is not canonical")
    return value


def _expected_digest(path: Path, root_record: Any, *, label: str) -> str:
    digest = sha256_file(path)
    if (
        type(root_record) is not dict
        or set(root_record) != {"bytes", "sha256"}
        or type(root_record["bytes"]) is not int
        or root_record["bytes"] != digest.bytes
        or root_record["sha256"] != digest.sha256
    ):
        raise ValueError(f"{label} root binding mismatch")
    return digest.sha256


def _load_bundle(
    policy_value: Mapping[str, Any] | StandingAuthorization,
    run_dir: str | Path,
    *,
    expected_bundle_id: str | None,
    now: str | datetime | None,
) -> _BundleFacts:
    current = _now(now)
    policy = StandingAuthorization.from_dict(
        (
            policy_value.to_dict()
            if isinstance(policy_value, StandingAuthorization)
            else dict(policy_value)
        ),
        now=current,
    )
    run_path = Path(run_dir)
    _require_directory(run_path)
    control_dir = run_path / "control"
    bundle_dir = run_path / "bundle"
    _require_directory(control_dir)
    _require_directory(bundle_dir)

    plan_path = run_path / "plan.md"
    policy_path = control_dir / "standing-authorization.json"
    price_path = control_dir / "price-evidence.json"
    config_path = control_dir / "execution-config.json"
    root_path = bundle_dir / "bundle-manifest.json"
    dataset_path = bundle_dir / "dataset-manifest.json"
    archive_path = bundle_dir / "training-data.zip"
    for path in (
        plan_path,
        policy_path,
        price_path,
        config_path,
        root_path,
        dataset_path,
        archive_path,
    ):
        _require_regular_file(path)

    on_disk_policy = StandingAuthorization.from_dict(
        _load_canonical_object(policy_path), now=current
    )
    price = PriceEvidence.from_dict(_load_canonical_object(price_path), now=current)
    config = _load_canonical_object(config_path)
    if not EXECUTION_CONFIG_REQUIRED_FIELDS.issubset(config):
        raise ValueError("execution configuration is missing required fields")
    root_manifest = _load_canonical_object(root_path)
    bundle_id = compute_bundle_id(root_manifest)
    if expected_bundle_id is not None:
        _sha256(expected_bundle_id, label="bundle_id")
        if expected_bundle_id != bundle_id:
            raise ValueError("bundle mismatch")
    root_expiry = _parse_utc_timestamp(
        root_manifest["expires_at_utc"], label="bundle expires_at_utc"
    )
    if root_expiry <= current:
        raise ValueError("bundle expired")

    root_artifacts = root_manifest["artifacts"]
    policy_sha256 = _expected_digest(
        policy_path,
        root_artifacts["standing_authorization"],
        label="standing authorization",
    )
    passed_policy_sha256 = hashlib.sha256(
        canonical_json_bytes(policy.to_dict())
    ).hexdigest()
    if passed_policy_sha256 != policy_sha256 or on_disk_policy != policy:
        raise ValueError("standing authorization hash mismatch")
    _expected_digest(
        price_path,
        root_artifacts["price_evidence"],
        label="price evidence",
    )
    plan_sha256 = _expected_digest(
        plan_path, root_artifacts["plan"], label="plan"
    )
    dataset_sha256 = _expected_digest(
        dataset_path,
        root_artifacts["dataset_manifest"],
        label="dataset manifest",
    )
    archive_sha256 = _expected_digest(
        archive_path,
        root_artifacts["training_archive"],
        label="training archive",
    )
    config_sha256 = _expected_digest(
        config_path,
        root_artifacts["execution_config"],
        label="execution configuration",
    )

    if config["endpoint"] != policy.endpoint:
        raise ValueError("endpoint mismatch")
    if type(config["steps"]) is not int or config["steps"] != policy.steps:
        raise ValueError("step count mismatch")
    if config["training_max_usd"] != policy.training_max_usd:
        raise ValueError("training ceiling mismatch")
    if config["validation_allocation_usd"] != policy.validation_allocation_usd:
        raise ValueError("validation allocation mismatch")
    if config["cumulative_cap_usd"] != policy.cumulative_cap_usd:
        raise ValueError("cumulative cap mismatch")
    if config["execution_id"] != root_manifest["execution_id"]:
        raise ValueError("execution ID mismatch")
    execution_id = _typed_id(
        config["execution_id"], EXECUTION_ID_PATTERN, label="execution_id"
    )
    pilot_id = _typed_id(config["pilot_id"], PILOT_ID_PATTERN, label="pilot_id")
    ledger_id = _typed_id(config["ledger_id"], LEDGER_ID_PATTERN, label="ledger_id")

    expiries = {
        policy.expires_at_utc: _parse_utc_timestamp(
            policy.expires_at_utc, label="policy expires_at_utc"
        ),
        price.expires_at_utc: _parse_utc_timestamp(
            price.expires_at_utc, label="price expires_at_utc"
        ),
        root_manifest["expires_at_utc"]: root_expiry,
    }
    expires_at_utc = min(expiries, key=expiries.get)  # type: ignore[arg-type]
    return _BundleFacts(
        bundle_id=bundle_id,
        execution_id=execution_id,
        expires_at_utc=expires_at_utc,
        policy=policy,
        price_evidence=price,
        standing_authorization_sha256=policy_sha256,
        plan_sha256=plan_sha256,
        dataset_manifest_sha256=dataset_sha256,
        training_archive_sha256=archive_sha256,
        execution_config_sha256=config_sha256,
        pilot_id=pilot_id,
        ledger_id=ledger_id,
    )


def _new_typed_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def issue_execution_receipt(
    policy: Mapping[str, Any] | StandingAuthorization,
    bundle: str | Path,
    *,
    expected_bundle_id: str | None = None,
    approval_id: str | None = None,
    issuer_process_id: str | None = None,
    now: str | datetime | None = None,
) -> ExecutionReceipt:
    current = _now(now)
    facts = _load_bundle(
        policy,
        bundle,
        expected_bundle_id=expected_bundle_id,
        now=current,
    )
    selected_approval_id = approval_id or _new_typed_id("approval")
    selected_process_id = issuer_process_id or _new_typed_id("process")
    if EXECUTION_ID_PATTERN.fullmatch(selected_approval_id) is not None:
        raise ValueError("approval_id must not use a replay ID")
    receipt = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_id": selected_approval_id,
        "status": APPROVAL_STATUS,
        "approval_mode": APPROVAL_MODE,
        "policy_id": facts.policy.policy_id,
        "standing_authorization_sha256": facts.standing_authorization_sha256,
        "issuer_process_id": selected_process_id,
        "issued_at_utc": _format_utc(current),
        "bundle_id": facts.bundle_id,
        "execution_id": facts.execution_id,
        "expires_at_utc": facts.expires_at_utc,
        "plan_sha256": facts.plan_sha256,
        "dataset_manifest_sha256": facts.dataset_manifest_sha256,
        "training_archive_sha256": facts.training_archive_sha256,
        "execution_config_sha256": facts.execution_config_sha256,
        "pilot_id": facts.pilot_id,
        "ledger_id": facts.ledger_id,
        "training_max_usd": facts.policy.training_max_usd,
        "validation_allocation_usd": facts.policy.validation_allocation_usd,
        "cumulative_cap_usd": facts.policy.cumulative_cap_usd,
        "steps": facts.policy.steps,
    }
    return ExecutionReceipt.from_dict(receipt, now=current)


def verify_execution_receipt(
    receipt: Mapping[str, Any] | ExecutionReceipt,
    policy: Mapping[str, Any] | StandingAuthorization,
    bundle: str | Path,
    *,
    now: str | datetime | None = None,
) -> ExecutionReceipt:
    current = _now(now)
    approval = (
        receipt
        if isinstance(receipt, ExecutionReceipt)
        else ExecutionReceipt.from_dict(dict(receipt), now=current)
    )
    if isinstance(receipt, ExecutionReceipt):
        approval = ExecutionReceipt.from_dict(receipt.to_dict(), now=current)
    facts = _load_bundle(
        policy,
        bundle,
        expected_bundle_id=approval.bundle_id,
        now=current,
    )
    expected = {
        "policy_id": facts.policy.policy_id,
        "standing_authorization_sha256": facts.standing_authorization_sha256,
        "bundle_id": facts.bundle_id,
        "execution_id": facts.execution_id,
        "expires_at_utc": facts.expires_at_utc,
        "plan_sha256": facts.plan_sha256,
        "dataset_manifest_sha256": facts.dataset_manifest_sha256,
        "training_archive_sha256": facts.training_archive_sha256,
        "execution_config_sha256": facts.execution_config_sha256,
        "pilot_id": facts.pilot_id,
        "ledger_id": facts.ledger_id,
        "training_max_usd": facts.policy.training_max_usd,
        "validation_allocation_usd": facts.policy.validation_allocation_usd,
        "cumulative_cap_usd": facts.policy.cumulative_cap_usd,
        "steps": facts.policy.steps,
    }
    for field, value in expected.items():
        if getattr(approval, field) != value:
            if field == "bundle_id":
                raise ValueError("bundle mismatch")
            raise ValueError(f"execution approval {field} mismatch")
    return approval
