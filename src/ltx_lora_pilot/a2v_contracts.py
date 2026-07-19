"""Pure schemas and validators for the immutable A2V training contract.

This module is deliberately authority-free.  It validates public control
artifacts but never retrieves pricing, resolves credentials, issues receipts,
or opens a budget ledger.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any


A2V_ENDPOINT = "fal-ai/ltx23-trainer-v2/a2v"
OFFICIAL_PRICE_URL = "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v"
A2V_EXECUTIONS = 1
A2V_STEPS = 1_000
A2V_RATE_USD_PER_STEP = "0.006"
TRAINING_MAX_USD = "6.0000"
VALIDATION_ALLOCATION_MAX_USD = "1.2500"
CUMULATIVE_CAP_USD = "12.0000"
APPROVAL_SCHEMA_VERSION = "a2v-execution-approval-v2"
APPROVAL_STATUS = "approved_for_paid_execution"
APPROVAL_MODE = "standing_policy"
EXECUTION_CONFIG_SCHEMA_VERSION = "a2v-execution-config-v2"
PRICE_EVIDENCE_MAX_AGE = timedelta(hours=24)

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
TRIGGER_PHRASE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}", re.ASCII)

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
        "ledger_head_sha256",
        "training_max_usd",
        "validation_allocation_usd",
        "cumulative_cap_usd",
        "steps",
    }
)
EXECUTION_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "canonical_json_version",
        "execution_id",
        "pilot_id",
        "ledger_id",
        "created_at_utc",
        "expires_at_utc",
        "endpoint",
        "trigger_phrase",
        "rank",
        "steps",
        "learning_rate",
        "training_frames",
        "training_fps",
        "resolution",
        "aspect_ratio",
        "auto_scale_input",
        "split_input_into_scenes",
        "audio_normalize",
        "audio_preserve_pitch",
        "debug_dataset",
        "negative_prompt",
        "validation_number_of_frames",
        "validation_frame_rate",
        "validation_resolution",
        "validation_aspect_ratio",
        "dataset_manifest_sha256",
        "training_archive_sha256",
        "standing_authorization_sha256",
        "price_evidence_sha256",
        "price_source_url",
        "rate_usd_per_step",
        "training_max_usd",
        "validation_allocation_usd",
        "cumulative_cap_usd",
    }
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


def _canonical_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 4_096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _validate_execution_config(value: Any) -> dict[str, Any]:
    data = _exact_dict(
        value,
        EXECUTION_CONFIG_FIELDS,
        label="execution configuration",
    )
    if data["schema_version"] != EXECUTION_CONFIG_SCHEMA_VERSION:
        raise ValueError("execution configuration schema mismatch")
    if (
        type(data["canonical_json_version"]) is not int
        or data["canonical_json_version"] != 1
    ):
        raise ValueError("execution configuration canonical JSON version mismatch")
    _typed_id(data["execution_id"], EXECUTION_ID_PATTERN, label="execution_id")
    _typed_id(data["pilot_id"], PILOT_ID_PATTERN, label="pilot_id")
    _typed_id(data["ledger_id"], LEDGER_ID_PATTERN, label="ledger_id")
    created = _parse_utc_timestamp(
        data["created_at_utc"],
        label="execution configuration created_at_utc",
    )
    expires = _parse_utc_timestamp(
        data["expires_at_utc"],
        label="execution configuration expires_at_utc",
    )
    if expires <= created:
        raise ValueError(
            "execution configuration expires_at_utc must be after created_at_utc"
        )
    if type(data["endpoint"]) is not str or data["endpoint"] != A2V_ENDPOINT:
        raise ValueError("endpoint mismatch")
    if (
        type(data["trigger_phrase"]) is not str
        or TRIGGER_PHRASE_PATTERN.fullmatch(data["trigger_phrase"]) is None
    ):
        raise ValueError("execution configuration requires a neutral trigger phrase")
    if type(data["rank"]) is not int or data["rank"] != 32:
        raise ValueError("execution configuration rank mismatch")
    if type(data["steps"]) is not int or data["steps"] != A2V_STEPS:
        raise ValueError("step count mismatch")
    if type(data["learning_rate"]) is not str or data["learning_rate"] != "0.0002":
        raise ValueError("execution configuration learning rate mismatch")
    if type(data["training_frames"]) is not int or data["training_frames"] != 89:
        raise ValueError("execution configuration training frames mismatch")
    if type(data["training_fps"]) is not int or data["training_fps"] != 24:
        raise ValueError("execution configuration training fps mismatch")
    if type(data["resolution"]) is not str or data["resolution"] != "high":
        raise ValueError("execution configuration resolution mismatch")
    if type(data["aspect_ratio"]) is not str or data["aspect_ratio"] != "9:16":
        raise ValueError("execution configuration aspect ratio mismatch")
    if data["auto_scale_input"] is not False:
        raise ValueError("execution configuration auto-scale mismatch")
    if data["split_input_into_scenes"] is not False:
        raise ValueError("execution configuration split-scenes mismatch")
    if data["audio_normalize"] is not True:
        raise ValueError("execution configuration audio normalization mismatch")
    if data["audio_preserve_pitch"] is not True:
        raise ValueError("execution configuration pitch preservation mismatch")
    if data["debug_dataset"] is not False:
        raise ValueError("execution configuration debug_dataset must be false")
    _canonical_text(data["negative_prompt"], label="negative_prompt")
    if (
        type(data["validation_number_of_frames"]) is not int
        or data["validation_number_of_frames"] != 89
    ):
        raise ValueError("execution configuration validation number of frames mismatch")
    if (
        type(data["validation_frame_rate"]) is not int
        or data["validation_frame_rate"] != 24
    ):
        raise ValueError("execution configuration validation frame rate mismatch")
    if (
        type(data["validation_resolution"]) is not str
        or data["validation_resolution"] != "high"
    ):
        raise ValueError("execution configuration validation resolution mismatch")
    if (
        type(data["validation_aspect_ratio"]) is not str
        or data["validation_aspect_ratio"] != "9:16"
    ):
        raise ValueError("execution configuration validation aspect ratio mismatch")
    for field in (
        "dataset_manifest_sha256",
        "training_archive_sha256",
        "standing_authorization_sha256",
        "price_evidence_sha256",
    ):
        _sha256(data[field], label=field)
    if data["price_source_url"] != OFFICIAL_PRICE_URL:
        raise ValueError("execution configuration price source URL mismatch")
    if data["rate_usd_per_step"] != A2V_RATE_USD_PER_STEP:
        raise ValueError("execution configuration price rate mismatch")
    if data["training_max_usd"] != TRAINING_MAX_USD:
        raise ValueError("training ceiling mismatch")
    if data["validation_allocation_usd"] != VALIDATION_ALLOCATION_MAX_USD:
        raise ValueError("validation allocation mismatch")
    if data["cumulative_cap_usd"] != CUMULATIVE_CAP_USD:
        raise ValueError("cumulative cap mismatch")
    return data


def validate_execution_config(value: Any) -> dict[str, Any]:
    """Return a defensive copy of an exact fixed A2V execution config."""

    return dict(_validate_execution_config(value))


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
        expires = _parse_utc_timestamp(data["expires_at_utc"], label="expires_at_utc")
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
        expires = _parse_utc_timestamp(data["expires_at_utc"], label="expires_at_utc")
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
