from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
from typing import Any, Callable

from . import a2v_static_verification as _static_verification
from .a2v_static_verification import (
    _LoadedArtifacts,
    _OpenPinnedArchive,
    _PathContext,
    _PinnedFile,
    StaticA2VBundle,
    verify_static_a2v_bundle,
)
from .a2v_contracts import (
    CUMULATIVE_CAP_USD,
    MONEY_PATTERN,
    SHA256_PATTERN,
    TRAINING_MAX_USD,
)
from .artifacts import atomic_write_json
from .authorization import (
    ExecutionReceipt,
    PriceEvidence,
    StandingAuthorization,
    verify_execution_receipt,
)
from .pilot_ledger import (
    CAP_USD_TEXT,
    SQLITE_SIDECAR_SUFFIXES,
    LedgerPreflightSnapshot,
    PilotLedger,
)
from .private_workspace import resolve_pilot_ledger


GATE_ORDER = (
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

PREFLIGHT_REPORT_SCHEMA = "a2v-preflight-report-v1"
TRAINING_RESERVATION_USD = TRAINING_MAX_USD


class PreflightNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class PreflightStatus:
    schema_version: str
    status: str
    failed_gate: str | None
    receipt_required: bool
    bundle_id: str
    execution_id: str | None
    training_groups: int | None
    holdout_groups: int | None
    provider_validation_items: int | None
    committed_usd: str | None
    remaining_usd: str | None
    training_reservation_usd: str
    remaining_after_reservation_usd: str | None
    passed_gates: tuple[str, ...]
    pilot_id: str | None = field(repr=False)
    ledger_id: str | None = field(repr=False)
    ledger_head_sha256: str | None = field(repr=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "failed_gate": self.failed_gate,
            "receipt_required": self.receipt_required,
            "bundle_id": self.bundle_id,
            "execution_id": self.execution_id,
            "counts": {
                "training_groups": self.training_groups,
                "holdout_groups": self.holdout_groups,
                "provider_validation_items": self.provider_validation_items,
            },
            "budget": {
                "committed_usd": self.committed_usd,
                "remaining_usd": self.remaining_usd,
                "training_reservation_usd": self.training_reservation_usd,
                "remaining_after_reservation_usd": self.remaining_after_reservation_usd,
            },
            "passed_gates": list(self.passed_gates),
        }

    def require_ready(self) -> PreflightStatus:
        if not (
            self.status == "ready_for_paid_execution"
            and self.receipt_required is True
            and self.failed_gate is None
            and self.ledger_head_sha256 is not None
            and self.passed_gates == GATE_ORDER
        ):
            raise PreflightNotReady("preflight is not ready for paid execution")
        return self


def _parse_expiry(value: Any) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("expiry timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("expiry timestamp is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError("expiry timestamp is invalid")
    return parsed


def _clock_read(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo != timezone.utc:
        raise ValueError("preflight clock must return UTC")
    return value.replace(microsecond=0)


def _money(value: Any) -> Decimal:
    if type(value) is not str or MONEY_PATTERN.fullmatch(value) is None:
        raise ValueError("ledger money is invalid")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("ledger money is invalid") from exc


def _ledger_sidecars_absent(path: Path) -> None:
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError("ledger sidecar is prohibited")


def _ledger_snapshot(
    context: _PathContext,
    config: dict[str, Any],
    bundle_id: str,
    receipt: ExecutionReceipt | None,
) -> tuple[LedgerPreflightSnapshot, Path, str, _PinnedFile]:
    ledger_id = config.get("ledger_id")
    if type(ledger_id) is not str:
        raise ValueError("ledger identity is invalid")
    ledger_path = resolve_pilot_ledger(context.private_root, context.pilot_id)
    _ledger_sidecars_absent(ledger_path)
    ledger_pin = _static_verification._pin_file(ledger_path)
    ledger = PilotLedger.open_existing(
        ledger_path,
        context.pilot_id,
        expected_ledger_id=ledger_id,
    )
    snapshot = ledger.preflight_snapshot(bundle_id, context.execution_id)
    expected = {
        "pilot_id": context.pilot_id,
        "ledger_id": ledger_id,
        "bundle_id": bundle_id,
        "execution_id": context.execution_id,
    }
    for name, value in expected.items():
        if getattr(snapshot, name) != value:
            raise ValueError("ledger snapshot identity mismatch")
    if type(snapshot.replay_detected) is not bool or snapshot.replay_detected:
        raise ValueError("ledger snapshot replay detected")
    committed = _money(snapshot.committed_usd)
    remaining = _money(snapshot.remaining_usd)
    if committed + remaining != Decimal(CAP_USD_TEXT):
        raise ValueError("ledger snapshot balance mismatch")
    if remaining < Decimal(TRAINING_RESERVATION_USD):
        raise ValueError("ledger has insufficient remaining budget")
    if SHA256_PATTERN.fullmatch(snapshot.head_sha256) is None:
        raise ValueError("ledger snapshot head is invalid")
    if receipt is not None and receipt.ledger_head_sha256 != snapshot.head_sha256:
        raise ValueError("execution receipt ledger head mismatch")
    _ledger_sidecars_absent(ledger_path)
    return (
        snapshot,
        ledger_path,
        f"{remaining - Decimal(TRAINING_RESERVATION_USD):.4f}",
        ledger_pin,
    )


def _final_recheck(
    *,
    context: _PathContext,
    artifacts: _LoadedArtifacts,
    pins: dict[str, _PinnedFile],
    clock: Callable[[], datetime],
    receipt: ExecutionReceipt | None,
    ledger_path: Path,
) -> None:
    current = _clock_read(clock)
    PriceEvidence.from_dict(artifacts.price_evidence, now=current)
    StandingAuthorization.from_dict(artifacts.standing_policy, now=current)
    if receipt is not None:
        ExecutionReceipt.from_dict(receipt.to_dict(), now=current)
    for value in (
        artifacts.root_manifest.get("expires_at_utc"),
        artifacts.execution_config.get("expires_at_utc"),
    ):
        if _parse_expiry(value) <= current:
            raise ValueError("preflight artifact expired")
    _ledger_sidecars_absent(ledger_path)
    if (
        _static_verification._security_snapshot(context.private_root)
        != context.security_snapshot
    ):
        raise ValueError("private path identity changed during preflight")
    for pin in pins.values():
        if _static_verification._pin_file(pin.path) != pin:
            raise ValueError("private artifact changed during preflight")
    _ledger_sidecars_absent(ledger_path)


def _atomic_report(context: _PathContext, report: PreflightStatus) -> None:
    path = context.run_dir / "control" / "preflight-report.json"
    atomic_write_json(path, report.to_public_dict())
    if os.name != "nt":
        os.chmod(path, 0o600)


def run_preflight(
    run_dir: Path,
    confirmed_bundle_id: str,
    *,
    require_receipt: bool,
    approved_private_root: Path,
    clock: Callable[[], datetime],
) -> PreflightStatus:
    if type(require_receipt) is not bool or not callable(clock):
        raise TypeError("preflight arguments are invalid")
    public_bundle_id = (
        confirmed_bundle_id
        if type(confirmed_bundle_id) is str and SHA256_PATTERN.fullmatch(confirmed_bundle_id)
        else ""
    )
    passed: list[str] = []
    context: _PathContext | None = None
    execution_id: str | None = None
    training_groups: int | None = None
    holdout_groups: int | None = None
    provider_items: int | None = None
    committed_usd: str | None = None
    remaining_usd: str | None = None
    remaining_after: str | None = None
    pilot_id: str | None = None
    ledger_id: str | None = None
    ledger_head: str | None = None
    opened_archive: _OpenPinnedArchive | None = None

    def finish(failed_gate: str | None) -> PreflightStatus:
        if opened_archive is not None:
            opened_archive.close()
        if failed_gate is None:
            status = (
                "ready_for_paid_execution"
                if require_receipt
                else "ready_for_policy_issuance"
            )
        else:
            status = "failed"
        report = PreflightStatus(
            schema_version=PREFLIGHT_REPORT_SCHEMA,
            status=status,
            failed_gate=failed_gate,
            receipt_required=require_receipt,
            bundle_id=public_bundle_id,
            execution_id=execution_id,
            training_groups=training_groups,
            holdout_groups=holdout_groups,
            provider_validation_items=provider_items,
            committed_usd=committed_usd,
            remaining_usd=remaining_usd,
            training_reservation_usd=TRAINING_RESERVATION_USD,
            remaining_after_reservation_usd=remaining_after,
            passed_gates=tuple(passed),
            pilot_id=pilot_id,
            ledger_id=ledger_id,
            ledger_head_sha256=ledger_head,
        )
        if context is not None:
            _atomic_report(context, report)
        return report

    try:
        context = _static_verification._lexical_context(
            Path(approved_private_root), Path(run_dir)
        )
        execution_id = context.execution_id
        pilot_id = context.pilot_id
    except Exception:
        return finish("private_root")
    passed.append("private_root")

    try:
        static = _static_verification._verify_static_gates(
            context,
            public_bundle_id,
            require_receipt=require_receipt,
        )
    except _static_verification._StaticGateFailure as exc:
        passed.extend(exc.passed_gates)
        return finish(exc.gate)
    artifacts = static.artifacts
    pins = static.pins
    config = static.config
    training_groups = static.quality_summary["coverage_counts"]["accepted_train_groups"]
    holdout_groups = static.quality_summary["coverage_counts"]["accepted_holdout_groups"]
    provider_items = static.provider_items
    ledger_id = config["ledger_id"]
    passed.extend(static.passed_gates)

    try:
        price = PriceEvidence.from_dict(artifacts.price_evidence, now=_clock_read(clock))
    except Exception:
        return finish("price_freshness")
    passed.append("price_freshness")

    try:
        policy = StandingAuthorization.from_dict(
            artifacts.standing_policy,
            now=_clock_read(clock),
        )
        if config["endpoint"] != policy.endpoint or config["steps"] != policy.steps:
            raise ValueError("standing policy request mismatch")
        if (
            config["training_max_usd"] != policy.training_max_usd
            or config["validation_allocation_usd"] != policy.validation_allocation_usd
            or config["cumulative_cap_usd"] != policy.cumulative_cap_usd
            or config["cumulative_cap_usd"] != CUMULATIVE_CAP_USD
        ):
            raise ValueError("standing policy cost mismatch")
        root_expiry = _parse_expiry(artifacts.root_manifest["expires_at_utc"])
        if root_expiry > _parse_expiry(policy.expires_at_utc) or root_expiry > _parse_expiry(price.expires_at_utc):
            raise ValueError("bundle expiry exceeds authorization evidence")
    except Exception:
        return finish("standing_policy")
    passed.append("standing_policy")

    try:
        receipt: ExecutionReceipt | None = None
        if require_receipt:
            if artifacts.receipt is None:
                raise ValueError("execution receipt is required")
            receipt = verify_execution_receipt(
                artifacts.receipt,
                policy,
                context.run_dir,
                now=_clock_read(clock),
            )
    except Exception:
        return finish("receipt")
    passed.append("receipt")

    try:
        snapshot, ledger_path, remaining_after, ledger_pin = _ledger_snapshot(
            context,
            config,
            public_bundle_id,
            receipt,
        )
        pins["ledger"] = ledger_pin
        committed_usd = snapshot.committed_usd
        remaining_usd = snapshot.remaining_usd
        ledger_head = snapshot.head_sha256
    except Exception:
        return finish("ledger_snapshot")
    passed.append("ledger_snapshot")

    try:
        _final_recheck(
            context=context,
            artifacts=artifacts,
            pins=pins,
            clock=clock,
            receipt=receipt,
            ledger_path=ledger_path,
        )
    except Exception:
        return finish("final_recheck")
    passed.append("final_recheck")
    return finish(None)
