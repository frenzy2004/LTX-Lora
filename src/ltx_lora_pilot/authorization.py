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
from .artifacts import (
    canonical_json_bytes,
    safe_relative_name,
    sha256_file,
    strict_load_json,
)


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
EXECUTION_CONFIG_SCHEMA_VERSION = "a2v-execution-config-v1"
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
        "validation",
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
VALIDATION_ENTRY_FIELDS = frozenset(
    {
        "image_filename",
        "image_sha256",
        "audio_filename",
        "audio_sha256",
        "prompt",
        "frames",
        "fps",
        "resolution",
        "aspect_ratio",
    }
)
TRIGGER_PHRASE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}", re.ASCII)
RATE_FORMULA_PATTERN = re.compile(
    r"(?<![0-9.])(?P<currency>\$?)(?P<rate>[0-9]+\.[0-9]+)"
    r"\s*\*\s*steps\b",
    re.IGNORECASE | re.ASCII,
)
UNIT_STEP_RATE_MODIFIERS = (
    r"(?:(?:additional|individual|training)[\s-]+){0,3}"
)
UNIT_STEP_RATE_CONTEXT = (
    r"(?:"
    rf"per[\s-]+{UNIT_STEP_RATE_MODIFIERS}steps?"
    rf"|(?:for[\s-]+)?(?:each|every)[\s-]+{UNIT_STEP_RATE_MODIFIERS}steps?"
    rf"|(?:for[\s-]+)?(?:1|a|an|one|single)[\s-]+"
    rf"{UNIT_STEP_RATE_MODIFIERS}step"
    rf"|(?<![0-9][\s-]){UNIT_STEP_RATE_MODIFIERS}"
    r"step[\s-]+(?:rate|price|cost)"
    r")"
)
UNIT_STEP_RATE_CONTEXT_PATTERN = re.compile(
    rf"\b{UNIT_STEP_RATE_CONTEXT}\b",
    re.IGNORECASE | re.ASCII,
)
RATE_UNIT_AMOUNT_CONNECTOR = (
    r"\s*(?:"
    r",?\s*(?:the\s+)?(?:rate|price|cost|charge)\s*"
    r"(?:(?:is|equals)\s*|[:=]\s*)?"
    r"|(?:costs?|is(?:\s+(?:priced|charged)\s+at)?|equals)\s*"
    r"|[:=]\s*"
    r")"
)
RATE_PER_STEP_AFTER_PATTERN = re.compile(
    r"(?<![0-9.])\$?(?P<rate>[0-9]+\.[0-9]+)(?![0-9.])"
    r"(?P<bridge>[^0-9.;?!<>\r\n]{0,48}?)"
    rf"(?:/\s*(?:training\s+)?steps?\b|\b{UNIT_STEP_RATE_CONTEXT}\b)",
    re.IGNORECASE | re.ASCII,
)
DIRECT_RATE_AFTER_BRIDGE_PATTERN = re.compile(
    r"\s*(?:USD\s*)?",
    re.IGNORECASE | re.ASCII,
)
ASSERTIVE_RATE_AFTER_BRIDGE_PATTERN = re.compile(
    r"\s*(?:USD\s*)?(?:"
    r"(?:is\s+)?(?:charged|billed)"
    r"|is\s+(?:the\s+)?(?:(?:overall|total)\s+)?"
    r"(?:charge|cost|fee|price|rate)"
    r"|applies?"
    r")\s*",
    re.IGNORECASE | re.ASCII,
)
RATE_PER_STEP_BEFORE_PATTERN = re.compile(
    rf"\b{UNIT_STEP_RATE_CONTEXT}\b"
    rf"(?P<bridge>{RATE_UNIT_AMOUNT_CONNECTOR})"
    r"(?<![0-9.])\$?(?P<rate>[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])",
    re.IGNORECASE | re.ASCII,
)
RATE_UNIT_SAME_CLAUSE_BEFORE_PATTERN = re.compile(
    rf"\b{UNIT_STEP_RATE_CONTEXT}\b"
    r"(?P<bridge>[^0-9,.;?!<>\r\n]{0,48}?)"
    r"(?<![0-9.])\$?(?P<rate>[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])",
    re.IGNORECASE | re.ASCII,
)
RATE_UNIT_INTRO_BEFORE_PATTERN = re.compile(
    rf"\b{UNIT_STEP_RATE_CONTEXT}\b,"
    r"(?P<bridge>[^0-9,.;?!<>\r\n]{0,48}?)"
    r"(?<![0-9.])\$?(?P<rate>[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])",
    re.IGNORECASE | re.ASCII,
)
STEP_RATE_SUFFIX = (
    rf"(?:\*\s*steps\b|(?:USD\s*)?(?:/\s*(?:training\s+)?steps?\b|"
    rf"\b{UNIT_STEP_RATE_CONTEXT}\b))"
)
STEP_RATE_OPERAND = r"\$?[0-9]+\.[0-9]+\s*" + STEP_RATE_SUFFIX
COST_AFTER_STEPS_PATTERN = re.compile(
    r"\b1,?000\s+steps\b"
    r"(?P<bridge>(?:[^.;?!<>\r\n]|\.(?=[0-9])){0,80}?)"
    r"\b(?:costs?|prices?|priced|totals?)\b"
    rf"(?:(?:{STEP_RATE_OPERAND})|[^$.;?!<>\r\n]){{0,64}}?"
    r"\$(?P<cost>[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])"
    rf"(?!\s*{STEP_RATE_SUFFIX})",
    re.IGNORECASE | re.ASCII,
)
COST_BEFORE_STEPS_PATTERN = re.compile(
    r"\$(?P<cost>[0-9]+\.[0-9]+)(?![0-9.])"
    r"(?P<bridge>[^.;?!<>\r\n]{0,48}?)"
    r"\b(?:for|per)\s+1,?000\s+steps\b",
    re.IGNORECASE | re.ASCII,
)
MODEL_ENDPOINT_PATTERN = re.compile(
    r"fal-ai/[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*",
    re.IGNORECASE | re.ASCII,
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


def _canonical_local_filename(value: Any, *, label: str, suffix: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a canonical local filename")
    try:
        safe_relative_name(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical local filename") from exc
    if "/" in value or not value.endswith(suffix):
        raise ValueError(f"{label} must be a canonical local filename")
    return value


def _validate_validation_entries(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != 2:
        raise ValueError("execution configuration requires exactly two validation entries")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        entry = _exact_dict(
            item,
            VALIDATION_ENTRY_FIELDS,
            label=f"validation entry {index}",
        )
        _canonical_local_filename(
            entry["image_filename"],
            label="validation image_filename",
            suffix=".png",
        )
        _sha256(entry["image_sha256"], label="image_sha256")
        _canonical_local_filename(
            entry["audio_filename"],
            label="validation audio_filename",
            suffix=".wav",
        )
        _sha256(entry["audio_sha256"], label="audio_sha256")
        _canonical_text(entry["prompt"], label="validation prompt")
        if type(entry["frames"]) is not int or entry["frames"] != 89:
            raise ValueError("validation frames mismatch")
        if type(entry["fps"]) is not int or entry["fps"] != 24:
            raise ValueError("validation fps mismatch")
        if type(entry["resolution"]) is not str or entry["resolution"] != "high":
            raise ValueError("validation resolution mismatch")
        if type(entry["aspect_ratio"]) is not str or entry["aspect_ratio"] != "9:16":
            raise ValueError("validation aspect ratio mismatch")
        entries.append(entry)
    canonical_entries = sorted(
        entries,
        key=lambda entry: (entry["image_filename"], entry["audio_filename"]),
    )
    image_names = [entry["image_filename"] for entry in entries]
    audio_names = [entry["audio_filename"] for entry in entries]
    if len(set(image_names)) != 2 or len(set(audio_names)) != 2:
        raise ValueError("execution configuration requires canonical validation order")
    if len({name.casefold() for name in image_names}) != 2:
        raise ValueError(
            "validation image filenames must be case-insensitively unique"
        )
    if len({name.casefold() for name in audio_names}) != 2:
        raise ValueError(
            "validation audio filenames must be case-insensitively unique"
        )
    if entries != canonical_entries:
        raise ValueError("execution configuration requires canonical validation order")
    return entries


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
    _validate_validation_entries(data["validation"])
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
    opener = urllib_request.build_opener(
        urllib_request.ProxyHandler({}),
        _OfficialPriceRedirectHandler(),
    )
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


def _a2v_price_contexts(text: str) -> list[str]:
    markers = list(MODEL_ENDPOINT_PATTERN.finditer(text))
    if not markers:
        return [text]
    contexts: list[str] = []
    for index, marker in enumerate(markers):
        if marker.group(0).lower() != A2V_ENDPOINT:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        contexts.append(text[marker.start() : end])
    return contexts


def _verify_price_statement(content: bytes) -> None:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("official price response must be UTF-8") from exc
    contexts = _a2v_price_contexts(text)
    formula_rates = {
        match.group("rate")
        for context in contexts
        for match in RATE_FORMULA_PATTERN.finditer(context)
    }
    explicit_rates = set(formula_rates)
    costs: set[str] = set()
    for context in contexts:
        formula_spans = {
            match.span("rate") for match in RATE_FORMULA_PATTERN.finditer(context)
        }
        rate_before_matches = list(
            RATE_PER_STEP_BEFORE_PATTERN.finditer(context)
        )
        rate_after_candidates = list(
            RATE_PER_STEP_AFTER_PATTERN.finditer(context)
        )
        direct_rate_after_matches = [
            match
            for match in rate_after_candidates
            if DIRECT_RATE_AFTER_BRIDGE_PATTERN.fullmatch(match.group("bridge"))
            is not None
            or ASSERTIVE_RATE_AFTER_BRIDGE_PATTERN.fullmatch(
                match.group("bridge")
            )
            is not None
        ]
        strong_rate_matches = rate_before_matches + direct_rate_after_matches
        strong_rate_spans = formula_spans | {
            match.span("rate") for match in strong_rate_matches
        }
        cost_after_matches = list(COST_AFTER_STEPS_PATTERN.finditer(context))
        cost_before_matches = [
            match
            for match in COST_BEFORE_STEPS_PATTERN.finditer(context)
            if UNIT_STEP_RATE_CONTEXT_PATTERN.search(match.group("bridge"))
            is None
        ]
        cost_matches = [
            match
            for match in cost_after_matches + cost_before_matches
            if match.span("cost") not in strong_rate_spans
        ]
        cost_spans = {match.span("cost") for match in cost_matches}
        weak_rate_candidates = [
            match
            for pattern in (
                RATE_UNIT_SAME_CLAUSE_BEFORE_PATTERN,
                RATE_UNIT_INTRO_BEFORE_PATTERN,
            )
            for match in pattern.finditer(context)
            if match.span("rate") not in strong_rate_spans
        ] + [
            match
            for match in rate_after_candidates
            if match.span("rate") not in strong_rate_spans
        ]
        weak_rate_matches = [
            match
            for match in weak_rate_candidates
            if match.span("rate") not in cost_spans
        ]
        rate_matches = strong_rate_matches + weak_rate_matches
        explicit_rates.update(match.group("rate") for match in rate_matches)
        costs.update(match.group("cost") for match in cost_matches)
    if (
        formula_rates != {A2V_RATE_USD_PER_STEP}
        or explicit_rates != {A2V_RATE_USD_PER_STEP}
    ):
        raise ValueError("unexpected A2V rate")
    if costs != {"6.00"}:
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


def _ensure_distinct_files(paths: list[Path]) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            try:
                aliases = left.samefile(right)
            except OSError as exc:
                raise ValueError("private bundle input is unavailable") from exc
            if aliases:
                raise ValueError("root-bound files must not alias")


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
    validation_dir = run_path / "validation"
    _require_directory(control_dir)
    _require_directory(bundle_dir)
    _require_directory(validation_dir)

    plan_path = run_path / "plan.md"
    policy_path = control_dir / "standing-authorization.json"
    price_path = control_dir / "price-evidence.json"
    config_path = control_dir / "execution-config.json"
    structural_path = control_dir / "structural-report.json"
    quality_path = control_dir / "quality-attestation.json"
    selection_path = validation_dir / "provider-validation-selection.json"
    root_path = bundle_dir / "bundle-manifest.json"
    dataset_path = bundle_dir / "dataset-manifest.json"
    archive_path = bundle_dir / "training-data.zip"
    root_bound_paths = {
        "plan": plan_path,
        "standing_authorization": policy_path,
        "price_evidence": price_path,
        "structural_report": structural_path,
        "quality_attestation": quality_path,
        "execution_config": config_path,
        "provider_validation_selection": selection_path,
        "dataset_manifest": dataset_path,
        "training_archive": archive_path,
    }
    for path in (root_path, *root_bound_paths.values()):
        _require_regular_file(path)
    _ensure_distinct_files(list(root_bound_paths.values()))

    on_disk_policy = StandingAuthorization.from_dict(
        _load_canonical_object(policy_path), now=current
    )
    price = PriceEvidence.from_dict(_load_canonical_object(price_path), now=current)
    config = _validate_execution_config(_load_canonical_object(config_path))
    _load_canonical_object(structural_path)
    _load_canonical_object(quality_path)
    _load_canonical_object(selection_path)
    _load_canonical_object(dataset_path)
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
    if config["created_at_utc"] != root_manifest["created_at_utc"]:
        raise ValueError("execution configuration creation timestamp mismatch")
    if config["expires_at_utc"] != root_manifest["expires_at_utc"]:
        raise ValueError("execution configuration expiry timestamp mismatch")
    policy_expiry = _parse_utc_timestamp(
        policy.expires_at_utc,
        label="policy expires_at_utc",
    )
    price_expiry = _parse_utc_timestamp(
        price.expires_at_utc,
        label="price expires_at_utc",
    )
    if root_expiry > policy_expiry:
        raise ValueError("bundle expiry exceeds policy expiry")
    if root_expiry > price_expiry:
        raise ValueError("bundle expiry exceeds price evidence expiry")

    root_artifacts = root_manifest["artifacts"]
    labels = {
        "plan": "plan",
        "standing_authorization": "standing authorization",
        "price_evidence": "price evidence",
        "structural_report": "structural report",
        "quality_attestation": "quality attestation",
        "execution_config": "execution configuration",
        "provider_validation_selection": "provider validation selection",
        "dataset_manifest": "dataset manifest",
        "training_archive": "training archive",
    }
    artifact_sha256 = {
        role: _expected_digest(
            path,
            root_artifacts[role],
            label=labels[role],
        )
        for role, path in root_bound_paths.items()
    }
    policy_sha256 = artifact_sha256["standing_authorization"]
    price_sha256 = artifact_sha256["price_evidence"]
    plan_sha256 = artifact_sha256["plan"]
    dataset_sha256 = artifact_sha256["dataset_manifest"]
    archive_sha256 = artifact_sha256["training_archive"]
    config_sha256 = artifact_sha256["execution_config"]
    passed_policy_sha256 = hashlib.sha256(
        canonical_json_bytes(policy.to_dict())
    ).hexdigest()
    if passed_policy_sha256 != policy_sha256 or on_disk_policy != policy:
        raise ValueError("standing authorization hash mismatch")

    if config["standing_authorization_sha256"] != policy_sha256:
        raise ValueError("standing authorization config hash mismatch")
    if config["price_evidence_sha256"] != price_sha256:
        raise ValueError("price evidence config hash mismatch")
    if config["dataset_manifest_sha256"] != dataset_sha256:
        raise ValueError("dataset manifest hash mismatch")
    if config["training_archive_sha256"] != archive_sha256:
        raise ValueError("training archive hash mismatch")
    if config["price_source_url"] != price.source_url:
        raise ValueError("price source URL mismatch")
    if config["rate_usd_per_step"] != price.rate_usd_per_step:
        raise ValueError("price rate mismatch")

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
        policy.expires_at_utc: policy_expiry,
        price.expires_at_utc: price_expiry,
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
