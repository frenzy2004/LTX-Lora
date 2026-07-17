from __future__ import annotations

import hashlib
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html import unescape as html_unescape
from pathlib import Path
from typing import Any, Callable, Mapping

from .a2v_contracts import (
    A2V_ENDPOINT,
    OFFICIAL_PRICE_URL,
    A2V_EXECUTIONS,
    A2V_STEPS,
    A2V_RATE_USD_PER_STEP,
    TRAINING_MAX_USD,
    VALIDATION_ALLOCATION_MAX_USD,
    CUMULATIVE_CAP_USD,
    APPROVAL_SCHEMA_VERSION,
    APPROVAL_STATUS,
    APPROVAL_MODE,
    EXECUTION_CONFIG_SCHEMA_VERSION,
    PRICE_EVIDENCE_MAX_AGE,
    SHA256_PATTERN,
    MONEY_PATTERN,
    UTC_TIMESTAMP_PATTERN,
    UUID4_HEX,
    POLICY_ID_PATTERN,
    APPROVAL_ID_PATTERN,
    PROCESS_ID_PATTERN,
    PILOT_ID_PATTERN,
    LEDGER_ID_PATTERN,
    EXECUTION_ID_PATTERN,
    TRIGGER_PHRASE_PATTERN,
    STANDING_AUTHORIZATION_FIELDS,
    PRICE_EVIDENCE_FIELDS,
    EXECUTION_RECEIPT_FIELDS,
    EXECUTION_CONFIG_FIELDS,
    _exact_dict,
    _parse_utc_timestamp,
    _format_utc,
    _now,
    _sha256,
    _typed_id,
    _money,
    _canonical_text,
    _validate_execution_config,
    validate_execution_config,
    StandingAuthorization,
    PriceEvidence,
)

from .a2v_bundle import compute_bundle_id
from .artifacts import (
    canonical_json_bytes,
    sha256_file,
    strict_load_json,
)
from .pilot_ledger import LedgerPreflightSnapshot


PRICE_FETCH_TIMEOUT_SECONDS = 10
MAX_PRICE_RESPONSE_BYTES = 1_048_576

RATE_FORMULA_OPERATOR = (
    r"(?:\*|x|times|multiplied\s+by|"
    r"\N{MULTIPLICATION SIGN}|\N{MIDDLE DOT})"
)
RATE_BEFORE_STEPS_FORMULA_PATTERN = re.compile(
    r"(?<![0-9.])(?P<currency>\$?)(?P<rate>[0-9]+\.[0-9]+)"
    rf"\s*{RATE_FORMULA_OPERATOR}\s*steps\b",
    re.IGNORECASE | re.ASCII,
)
RATE_AFTER_STEPS_FORMULA_PATTERN = re.compile(
    rf"\bsteps\s*{RATE_FORMULA_OPERATOR}\s*"
    r"(?P<currency>\$?)(?P<rate>[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])",
    re.IGNORECASE | re.ASCII,
)
RATE_FORMULA_PATTERNS = (
    RATE_BEFORE_STEPS_FORMULA_PATTERN,
    RATE_AFTER_STEPS_FORMULA_PATTERN,
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
STRUCTURED_BILLING_OBJECT_PATTERN = re.compile(
    r'(?:\\?")(?:endpointBilling|publicEndpointBilling)(?:\\?")'
    r"\s*:\s*\{(?P<body>[^{}]{0,1024})\}",
    re.ASCII,
)
HTML_TAG_PATTERN = re.compile(r"<[^<>]*>", re.DOTALL)
HTML_ATTRIBUTE_VALUE_PATTERN = re.compile(
    r'''=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)')''',
    re.DOTALL,
)
MONETARY_DECIMAL_PATTERN = re.compile(
    r"(?<![0-9.])(?P<currency>\$)?"
    r"(?P<amount>(?:[0-9]+\.[0-9]+|\.[0-9]+))"
    r"(?![0-9]|\.[0-9])",
    re.ASCII,
)
STEP_WORD_PATTERN = re.compile(r"\bsteps?\b", re.IGNORECASE | re.ASCII)
THOUSAND_STEP_PATTERN = re.compile(
    r"\b1,?000[\s-]+steps?\b",
    re.IGNORECASE | re.ASCII,
)
USD_BEFORE_AMOUNT_PATTERN = re.compile(
    r"\bUSD\s*$",
    re.IGNORECASE | re.ASCII,
)
USD_AFTER_AMOUNT_PATTERN = re.compile(
    r"\s*USD\b",
    re.IGNORECASE | re.ASCII,
)
PRICE_LABEL_PATTERN = re.compile(
    r"\b(?:rate|price|fee|charge|cost)\b",
    re.IGNORECASE | re.ASCII,
)
FORMULA_LABEL_PATTERN = re.compile(r"\bformula\b", re.IGNORECASE | re.ASCII)
UNMARKED_PER_STEP_AFTER_PATTERN = re.compile(
    r"\s*(?:/|per\b)[^0-9.;?!<>\r\n]{0,48}\bstep\b",
    re.IGNORECASE | re.ASCII,
)
FORMULA_STEP_AFTER_AMOUNT_PATTERN = re.compile(
    r"[^0-9.;?!<>\r\n]{0,64}\bsteps?\b",
    re.IGNORECASE | re.ASCII,
)
FORMULA_STEP_BEFORE_AMOUNT_PATTERN = re.compile(
    r"\bsteps?\b[^0-9.;?!<>\r\n]{0,64}$",
    re.IGNORECASE | re.ASCII,
)
PRICE_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[;?!\r\n]|\.(?![0-9])")


def _fetch_official_price(url: str) -> bytes:
    from urllib import request as urllib_request

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
    if not any(
        marker.group(0).lower() == A2V_ENDPOINT for marker in markers
    ):
        return []
    return [text]


def _replace_price_tag(match: re.Match[str]) -> str:
    values = [
        attribute.group("double") or attribute.group("single") or ""
        for attribute in HTML_ATTRIBUTE_VALUE_PATTERN.finditer(match.group(0))
    ]
    return " " + " ".join(values) + " "


def _normalize_price_context(text: str) -> str:
    return " ".join(HTML_TAG_PATTERN.sub(_replace_price_tag, text).split())


def _bounded_price_clause(context: str, start: int, end: int) -> str:
    left = context[max(0, start - 96) : start]
    right = context[end : min(len(context), end + 96)]
    left_clause = PRICE_CLAUSE_BOUNDARY_PATTERN.split(left)[-1]
    right_clause = PRICE_CLAUSE_BOUNDARY_PATTERN.split(right)[0]
    return left_clause + context[start:end] + right_clause


def _step_associated_monetary_matches(
    context: str,
) -> list[tuple[re.Match[str], str]]:
    matches: list[tuple[re.Match[str], str]] = []
    for match in MONETARY_DECIMAL_PATTERN.finditer(context):
        clause = _bounded_price_clause(context, *match.span("amount"))
        if match.group("currency") is None:
            usd_before = context[
                max(0, match.start("amount") - 8) : match.start("amount")
            ]
            usd_after = context[match.end("amount") : match.end("amount") + 8]
            if (
                USD_BEFORE_AMOUNT_PATTERN.search(usd_before) is None
                and USD_AFTER_AMOUNT_PATTERN.match(usd_after) is None
            ):
                rate_suffix = context[
                    match.end("amount") : match.end("amount") + 64
                ]
                rate_prefix = context[
                    max(0, match.start("amount") - 64) : match.start("amount")
                ]
                formula_binding = (
                    FORMULA_LABEL_PATTERN.search(rate_prefix) is not None
                    and (
                        FORMULA_STEP_AFTER_AMOUNT_PATTERN.match(rate_suffix)
                        is not None
                        or FORMULA_STEP_BEFORE_AMOUNT_PATTERN.search(rate_prefix)
                        is not None
                    )
                )
                if (
                    PRICE_LABEL_PATTERN.search(clause) is None
                    and UNMARKED_PER_STEP_AFTER_PATTERN.match(rate_suffix)
                    is None
                    and not formula_binding
                ):
                    continue
        if STEP_WORD_PATTERN.search(clause) is None:
            if PRICE_LABEL_PATTERN.search(clause) is None:
                continue
            neighborhood = context[
                max(0, match.start("amount") - 160) : min(
                    len(context), match.end("amount") + 160
                )
            ]
            if STEP_WORD_PATTERN.search(neighborhood) is None:
                continue
            clause = neighborhood
        matches.append((match, clause))
    return matches


def _rate_formula_matches(context: str) -> list[re.Match[str]]:
    return [
        match
        for pattern in RATE_FORMULA_PATTERNS
        for match in pattern.finditer(context)
    ]


def _jsonish_field_values(body: str, field: str) -> tuple[str, ...]:
    normalized = body.replace(r'\"', '"')
    pattern = re.compile(
        rf'"{re.escape(field)}"\s*:\s*'
        r'(?P<value>"[^"\\]*"|[0-9]+\.[0-9]+)',
        re.ASCII,
    )
    values = []
    for match in pattern.finditer(normalized):
        value = match.group("value")
        if value.startswith('"'):
            value = value[1:-1]
        values.append(value)
    return tuple(values)


def _structured_a2v_billing_records(
    text: str,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    records: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for match in STRUCTURED_BILLING_OBJECT_PATTERN.finditer(text):
        body = match.group("body")
        endpoints = _jsonish_field_values(body, "endpoint")
        if A2V_ENDPOINT not in endpoints:
            continue
        records.append(
            (
                _jsonish_field_values(body, "billing_unit"),
                _jsonish_field_values(body, "price"),
            )
        )
    return records


def _verify_price_statement(content: bytes) -> None:
    try:
        text = html_unescape(content.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise ValueError("official price response must be UTF-8") from exc
    structured_records = _structured_a2v_billing_records(text)
    if structured_records:
        if any(
            units != ("steps",) or rates != (A2V_RATE_USD_PER_STEP,)
            for units, rates in structured_records
        ):
            raise ValueError("unexpected A2V rate")
        return
    contexts = [
        _normalize_price_context(context) for context in _a2v_price_contexts(text)
    ]
    formula_rates = {
        match.group("rate")
        for context in contexts
        for match in _rate_formula_matches(context)
    }
    explicit_rates = set(formula_rates)
    costs: set[str] = set()
    for context in contexts:
        formula_spans = {
            match.span("rate") for match in _rate_formula_matches(context)
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
        classified_spans = strong_rate_spans | cost_spans | {
            match.span("rate") for match in weak_rate_matches
        }
        for match, clause in _step_associated_monetary_matches(context):
            if match.span("amount") in classified_spans:
                continue
            amount = match.group("amount")
            if amount == "6.00" and THOUSAND_STEP_PATTERN.search(clause):
                costs.add(amount)
            else:
                explicit_rates.add(amount)
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
    ledger_head_sha256: str
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
            "ledger_head_sha256",
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


LedgerSnapshotReader = Callable[
    [str, str, str, str],
    LedgerPreflightSnapshot,
]


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
    config = validate_execution_config(_load_canonical_object(config_path))
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


def _ledger_head_for_receipt(
    facts: _BundleFacts,
    read_ledger_snapshot: LedgerSnapshotReader,
) -> str:
    snapshot = read_ledger_snapshot(
        facts.pilot_id,
        facts.ledger_id,
        facts.bundle_id,
        facts.execution_id,
    )
    if not isinstance(snapshot, LedgerPreflightSnapshot):
        raise ValueError("ledger preflight snapshot is invalid")
    expected_identities = {
        "pilot_id": facts.pilot_id,
        "ledger_id": facts.ledger_id,
        "bundle_id": facts.bundle_id,
        "execution_id": facts.execution_id,
    }
    for field, expected in expected_identities.items():
        if getattr(snapshot, field) != expected:
            raise ValueError(f"ledger preflight snapshot {field} mismatch")
    if type(snapshot.replay_detected) is not bool:
        raise ValueError("ledger preflight snapshot replay flag is invalid")
    if snapshot.replay_detected:
        raise ValueError("ledger preflight snapshot detected replay")
    return _sha256(snapshot.head_sha256, label="ledger_head_sha256")


def issue_execution_receipt(
    policy: Mapping[str, Any] | StandingAuthorization,
    bundle: str | Path,
    *,
    read_ledger_snapshot: LedgerSnapshotReader,
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
    ledger_head_sha256 = _ledger_head_for_receipt(
        facts,
        read_ledger_snapshot,
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
        "ledger_head_sha256": ledger_head_sha256,
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
