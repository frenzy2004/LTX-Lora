from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .artifacts import canonical_json_bytes
from .fal_api import A2V_QUEUE_URL, resolve_key as resolve_fal_key
from .fal_api import submit_a2v_once, upload_staged_file
from .pilot_ledger import PilotLedger, Reservation
from .preflight import PreflightStatus, run_preflight
from .private_workspace import resolve_pilot_ledger
from .staging import StagedArtifactChanged, StagedArtifactGuard, stage_bundle


TRAINING_RESERVATION_USD = Decimal("6.0000")
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,256}", re.ASCII)


class AmbiguousProviderSubmission(RuntimeError):
    """A provider might have received the request; no automatic retry is safe."""


@dataclass(frozen=True)
class SubmissionRecord:
    bundle_id: str
    execution_id: str
    reservation_id: str
    request_id: str
    request_body_sha256: str


def system_utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_timestamp(clock: Callable[[], datetime]) -> str:
    current = clock()
    if not isinstance(current, datetime):
        raise TypeError("execution clock returned an invalid value")
    if current.tzinfo is None:
        raise ValueError("execution clock must be timezone-aware")
    return current.astimezone(timezone.utc).replace(microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _secure_private_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    content = canonical_json_bytes(dict(value))
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = None
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if os.name != "nt":
            os.chmod(path, 0o600)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _submission_directory(private_root: Path) -> Path:
    # Keep durable provider state shallow; a deep run path plus a 64-byte bundle
    # identity exceeds the legacy Windows path limit.
    return private_root / ".a2v-provider-state"


def _build_request_body(
    staged: StagedArtifactGuard,
    urls: Mapping[str, str],
) -> bytes:
    config = staged.execution_config
    validation = []
    for pair in staged.validation_pairs:
        image_url = urls.get(f"{pair.group_id}:image")
        audio_url = urls.get(f"{pair.group_id}:audio")
        if not _is_secure_runtime_url(image_url) or not _is_secure_runtime_url(audio_url):
            raise RuntimeError("provider upload did not return secure validation URLs")
        validation.append(
            {
                "prompt": pair.prompt,
                "image_url": image_url,
                "audio_url": audio_url,
            }
        )
    training_data_url = urls.get("training")
    if not _is_secure_runtime_url(training_data_url):
        raise RuntimeError("provider upload did not return a secure training URL")
    payload = {
        "training_data_url": training_data_url,
        "rank": config["rank"],
        "number_of_steps": config["steps"],
        "learning_rate": float(config["learning_rate"]),
        "number_of_frames": config["training_frames"],
        "frame_rate": config["training_fps"],
        "resolution": config["resolution"],
        "aspect_ratio": config["aspect_ratio"],
        "trigger_phrase": config["trigger_phrase"],
        "auto_scale_input": config["auto_scale_input"],
        "split_input_into_scenes": config["split_input_into_scenes"],
        "debug_dataset": config["debug_dataset"],
        "audio_normalize": config["audio_normalize"],
        "audio_preserve_pitch": config["audio_preserve_pitch"],
        "validation": validation,
        "validation_negative_prompt": config["negative_prompt"],
        "validation_number_of_frames": config["validation_number_of_frames"],
        "validation_frame_rate": config["validation_frame_rate"],
        "validation_resolution": config["validation_resolution"],
        "validation_aspect_ratio": config["validation_aspect_ratio"],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _is_secure_runtime_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith("https://") and len(value) <= 16_384


def _upload_staged_assets(
    staged: StagedArtifactGuard,
    upload_fn: Callable[[Path], str],
) -> dict[str, str]:
    urls: dict[str, str] = {}
    staged.require_unchanged()
    urls["training"] = upload_fn(staged.training_zip)
    staged.require_unchanged()
    for pair in staged.validation_pairs:
        urls[f"{pair.group_id}:image"] = upload_fn(pair.image)
        staged.require_unchanged()
        urls[f"{pair.group_id}:audio"] = upload_fn(pair.audio)
        staged.require_unchanged()
    return urls


def _persist_submit_intent(
    private_root: Path,
    report: PreflightStatus,
    reservation: Reservation,
    body_sha256: str,
    *,
    clock: Callable[[], datetime],
) -> None:
    _secure_private_write(
        _submission_directory(private_root) / f"{report.bundle_id}.submit-intent.json",
        {
            "schema_version": "a2v-submit-intent-v1",
            "bundle_id": report.bundle_id,
            "execution_id": report.execution_id,
            "reservation_id": reservation.id,
            "endpoint": A2V_QUEUE_URL,
            "request_body_sha256": body_sha256,
            "created_at_utc": _canonical_timestamp(clock),
        },
    )


def _persist_request_id(
    private_root: Path,
    report: PreflightStatus,
    reservation: Reservation,
    request_id: str,
    body_sha256: str,
    *,
    clock: Callable[[], datetime],
) -> None:
    if REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise RuntimeError("provider acknowledgement is malformed")
    _secure_private_write(
        _submission_directory(private_root) / f"{report.bundle_id}.request.json",
        {
            "schema_version": "a2v-provider-request-v1",
            "bundle_id": report.bundle_id,
            "execution_id": report.execution_id,
            "reservation_id": reservation.id,
            "endpoint": A2V_QUEUE_URL,
            "request_body_sha256": body_sha256,
            "request_id": request_id,
            "recorded_at_utc": _canonical_timestamp(clock),
        },
    )


def _release_pre_submit(ledger: PilotLedger, reservation: Reservation | None, reason: str) -> None:
    if reservation is None:
        return
    try:
        ledger.release_pre_submit(reservation.id, reason)
    except Exception:
        # A failed pre-submit release must not lead to an unsafe automatic resubmit.
        pass


def execute_training_bundle(
    run_dir: Path,
    confirmed_bundle_id: str,
    *,
    approved_private_root: Path,
    resolve_key: Callable[[], str] = resolve_fal_key,
    upload_fn: Callable[[Path], str] | None = None,
    submit_fn: Callable[[str, bytes, str], Mapping[str, Any]] = submit_a2v_once,
    clock: Callable[[], datetime] = system_utc_clock,
) -> SubmissionRecord:
    """Reserve and submit one immutable A2V training request exactly once."""

    reservation: Reservation | None = None
    ledger: PilotLedger | None = None
    submit_started = False
    phase = "preflight"
    try:
        report = run_preflight(
            Path(run_dir),
            confirmed_bundle_id,
            require_receipt=True,
            approved_private_root=Path(approved_private_root),
            clock=clock,
        )
        report.require_ready()
        if (
            report.pilot_id is None
            or report.ledger_id is None
            or report.execution_id is None
            or report.ledger_head_sha256 is None
        ):
            raise RuntimeError("preflight omitted private execution identities")
        phase = "reserve"
        ledger_path = resolve_pilot_ledger(Path(approved_private_root), report.pilot_id)
        ledger = PilotLedger.open_existing(
            ledger_path,
            report.pilot_id,
            expected_ledger_id=report.ledger_id,
        )
        reservation = ledger.reserve_training(
            report.bundle_id,
            report.execution_id,
            TRAINING_RESERVATION_USD,
            expected_head_sha256=report.ledger_head_sha256,
        )
        phase = "stage"
        with stage_bundle(
            Path(run_dir),
            approved_private_root=Path(approved_private_root),
            confirmed_bundle_id=report.bundle_id,
            pilot_id=report.pilot_id,
            execution_id=report.execution_id,
        ) as staged:
            phase = "credential"
            key = resolve_key()
            if not isinstance(key, str) or not key:
                raise RuntimeError("FAL_KEY is not set in the process environment")
            resolved_upload = upload_fn or (lambda path: upload_staged_file(path, key))
            phase = "upload"
            ledger.transition(reservation.id, "uploading")
            urls = _upload_staged_assets(staged, resolved_upload)
            staged.require_unchanged()
            body = _build_request_body(staged, urls)
            body_sha256 = hashlib.sha256(body).hexdigest()
            phase = "intent"
            _persist_submit_intent(
                Path(approved_private_root), report, reservation, body_sha256, clock=clock
            )
            ledger.transition(reservation.id, "submit_started")
            submit_started = True
            phase = "submit"
            acknowledgement = submit_fn(A2V_QUEUE_URL, body, key)
            if type(acknowledgement) is not dict:
                raise RuntimeError("provider acknowledgement is malformed")
            request_id = acknowledgement.get("request_id")
            if not isinstance(request_id, str):
                raise RuntimeError("provider acknowledgement is malformed")
            phase = "acknowledgement"
            _persist_request_id(
                Path(approved_private_root),
                report,
                reservation,
                request_id,
                body_sha256,
                clock=clock,
            )
            ledger.transition(reservation.id, "submitted")
            return SubmissionRecord(
                bundle_id=report.bundle_id,
                execution_id=report.execution_id,
                reservation_id=reservation.id,
                request_id=request_id,
                request_body_sha256=body_sha256,
            )
    except Exception as exc:
        if submit_started:
            raise AmbiguousProviderSubmission(
                "provider submission is ambiguous; reservation remains committed"
            ) from None
        if ledger is not None:
            _release_pre_submit(
                ledger,
                reservation,
                f"{phase}_failed",
            )
        if isinstance(exc, StagedArtifactChanged):
            raise RuntimeError("staged artifact changed before provider submission") from None
        if phase in {"credential", "upload"}:
            raise RuntimeError(f"{phase} failed before provider submission") from None
        raise
