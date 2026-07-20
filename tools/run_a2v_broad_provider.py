from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import httpx

from build_ic_lora_dataset import BudgetExceeded, reserve_budget


A2V_APPLICATION = "fal-ai/ltx23-trainer-v2/a2v"
A2V_QUEUE_URL = "https://queue.fal.run/fal-ai/ltx23-trainer-v2/a2v"
A2V_RATE_USD_PER_STEP = Decimal("0.006")
OFFICIAL_PRICE_URL = "https://fal.ai/models/fal-ai/ltx23-trainer-v2/a2v"
ALLOWED_STEP_COUNTS = frozenset({100, 4_000})
IN_FLIGHT_BUDGET_STATUSES = frozenset(
    {"reserved", "uploaded", "submit_intent", "submitted", "submission_ambiguous"}
)
TRIGGER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}", re.ASCII)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}:[0-9a-fA-F]{32})"
    r"(?![A-Za-z0-9])"
)
LOSS_TEXT_PATTERN = re.compile(
    r"\bloss(?:\s+[a-z_][a-z0-9_]*)?\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


class ProviderExecutionError(RuntimeError):
    pass


def utc_now(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("time must be timezone-aware")
    return (
        current.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProviderExecutionError(f"{label} must be a canonical UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as exc:
        raise ProviderExecutionError(
            f"{label} must be a canonical UTC timestamp"
        ) from exc


def validate_price_evidence(
    value: object,
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    fields = {
        "source_url",
        "rate_usd_per_step",
        "response_sha256",
        "retrieved_at_utc",
        "expires_at_utc",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ProviderExecutionError("price evidence has an unexpected shape")
    if value["source_url"] != OFFICIAL_PRICE_URL:
        raise ProviderExecutionError("price evidence must use the official Fal URL")
    if value["rate_usd_per_step"] != str(A2V_RATE_USD_PER_STEP):
        raise ProviderExecutionError("unexpected A2V price rate")
    if (
        not isinstance(value["response_sha256"], str)
        or SHA256_PATTERN.fullmatch(value["response_sha256"]) is None
    ):
        raise ProviderExecutionError("price evidence response hash is invalid")
    retrieved = _parse_utc(value["retrieved_at_utc"], label="price retrieval time")
    expires = _parse_utc(value["expires_at_utc"], label="price expiry time")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if retrieved > current:
        raise ProviderExecutionError("price evidence retrieval is in the future")
    if expires <= current:
        raise ProviderExecutionError("price evidence has expired")
    if expires <= retrieved or expires - retrieved > timedelta(hours=24):
        raise ProviderExecutionError("price evidence lifetime must not exceed 24 hours")
    return {key: str(value[key]) for key in sorted(fields)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def extract_unique_fal_key(path: Path) -> str:
    if not path.is_file():
        raise ProviderExecutionError("credential attachment is unavailable")
    matches = sorted(
        set(KEY_PATTERN.findall(path.read_text(encoding="utf-8", errors="strict")))
    )
    if len(matches) != 1:
        raise ProviderExecutionError(
            "credential attachment must contain exactly one unique Fal key"
        )
    return matches[0]


def load_run_config(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    fields = {"schema_version", "steps", "trigger_phrase", "validation"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ProviderExecutionError("run configuration has an unexpected shape")
    if payload["schema_version"] != "broad-a2v-provider-run-v1":
        raise ProviderExecutionError("run configuration schema mismatch")
    steps = int(payload["steps"])
    expected_training_cost(steps)
    trigger = str(payload["trigger_phrase"])
    if TRIGGER_PATTERN.fullmatch(trigger) is None:
        raise ProviderExecutionError("run configuration trigger phrase is invalid")
    raw_validation = payload["validation"]
    if not isinstance(raw_validation, list) or len(raw_validation) > 2:
        raise ProviderExecutionError("run configuration permits at most two validations")
    validation: list[dict[str, object]] = []
    for item in raw_validation:
        if not isinstance(item, dict) or set(item) != {"prompt", "image", "audio"}:
            raise ProviderExecutionError(
                "run validation entries require prompt/image/audio"
            )
        prompt = str(item["prompt"]).strip()
        if not prompt:
            raise ProviderExecutionError("run validation prompt must not be empty")
        paths: dict[str, str] = {}
        for field in ("image", "audio"):
            candidate = Path(str(item[field]))
            if not candidate.is_absolute():
                candidate = resolved.parent / candidate
            candidate = candidate.resolve(strict=True)
            if not candidate.is_file():
                raise ProviderExecutionError(
                    f"run validation {field} is unavailable"
                )
            paths[field] = str(candidate)
        validation.append({"prompt": prompt, **paths})
    return {
        "schema_version": payload["schema_version"],
        "steps": steps,
        "trigger_phrase": trigger,
        "validation": validation,
    }


def expected_training_cost(steps: int) -> Decimal:
    if steps not in ALLOWED_STEP_COUNTS:
        raise ValueError("steps must be exactly 100 or 4000")
    return (Decimal(steps) * A2V_RATE_USD_PER_STEP).quantize(Decimal("0.0001"))


def _secure_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith("https://") and len(value) <= 16_384


def build_training_input(
    training_data_url: str,
    *,
    steps: int,
    trigger_phrase: str,
    validation: list[dict[str, str]],
) -> dict[str, Any]:
    expected_training_cost(steps)
    if not _secure_url(training_data_url):
        raise ValueError("a secure provider training-data URL is required")
    if TRIGGER_PATTERN.fullmatch(trigger_phrase) is None:
        raise ValueError("trigger phrase must be a neutral lowercase token")
    canonical_validation: list[dict[str, str]] = []
    for item in validation:
        if type(item) is not dict or set(item) != {"prompt", "image_url", "audio_url"}:
            raise ValueError("validation items must contain prompt/image_url/audio_url")
        if not item["prompt"].strip():
            raise ValueError("validation prompt must not be empty")
        if not _secure_url(item["image_url"]) or not _secure_url(item["audio_url"]):
            raise ValueError("validation inputs require secure provider URLs")
        canonical_validation.append(dict(item))
    return {
        "training_data_url": training_data_url,
        "rank": 32,
        "number_of_steps": steps,
        "learning_rate": 0.0001,
        "number_of_frames": 89,
        "frame_rate": 24,
        "resolution": "high",
        "aspect_ratio": "9:16",
        "trigger_phrase": trigger_phrase,
        "auto_scale_input": False,
        "split_input_into_scenes": False,
        "debug_dataset": True,
        "audio_normalize": True,
        "audio_preserve_pitch": True,
        "validation": canonical_validation,
        "validation_negative_prompt": (
            "distorted face, deformed mouth, extra teeth, identity drift, "
            "flicker, text, watermark"
        ),
        "validation_number_of_frames": 89,
        "validation_frame_rate": 24,
        "validation_resolution": "high",
        "validation_aspect_ratio": "9:16",
    }


def assert_no_paid_request_in_flight(budget: object) -> None:
    if not isinstance(budget, dict) or not isinstance(budget.get("entries"), list):
        raise ProviderExecutionError("budget ledger has an unexpected shape")
    active = [
        entry
        for entry in budget["entries"]
        if isinstance(entry, dict)
        and entry.get("status") in IN_FLIGHT_BUDGET_STATUSES
    ]
    if active:
        raise ProviderExecutionError("another paid request is already in flight")


def reserve_budget_file(path: Path, label: str, steps: int) -> dict[str, Any]:
    if not path.is_file():
        raise ProviderExecutionError("budget ledger is unavailable")
    budget = json.loads(path.read_text(encoding="utf-8"))
    assert_no_paid_request_in_flight(budget)
    if any(
        isinstance(entry, dict) and entry.get("label") == label
        for entry in budget.get("entries", [])
    ):
        raise ProviderExecutionError(f"budget label already exists: {label}")
    amount = expected_training_cost(steps)
    try:
        updated = reserve_budget(budget, label, float(amount))
    except BudgetExceeded as exc:
        raise ProviderExecutionError(str(exc)) from exc
    atomic_write_json(path, updated)
    return updated


def update_budget_entry(
    path: Path,
    label: str,
    status: str,
    **extra: object,
) -> dict[str, Any]:
    budget = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in budget.get("entries", [])
        if isinstance(entry, dict) and entry.get("label") == label
    ]
    if len(matches) != 1:
        raise ProviderExecutionError("reserved budget entry is unavailable or ambiguous")
    matches[0]["status"] = status
    matches[0].update(extra)
    atomic_write_json(path, budget)
    return budget


def upload_file(path: Path, key: str) -> str:
    if not path.is_file():
        raise ProviderExecutionError("reviewed provider input is unavailable")
    import fal_client

    url = fal_client.SyncClient(key=key, default_timeout=1200.0).upload_file(path)
    if not _secure_url(url):
        raise ProviderExecutionError("provider upload did not return a secure URL")
    return str(url)


def start_run(
    *,
    archive: Path,
    budget_path: Path,
    key_source: Path,
    price_evidence_path: Path,
    state_dir: Path,
    label: str,
    steps: int,
    trigger_phrase: str,
    validation: list[dict[str, object]],
    upload_fn: Callable[[Path, str], str] = upload_file,
    submit_fn: Callable[[str, dict[str, Any], str], dict[str, str]] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    amount = expected_training_cost(steps)
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise ProviderExecutionError("reviewed A2V training archive is unavailable")
    if not price_evidence_path.is_file():
        raise ProviderExecutionError("current official price evidence is unavailable")
    price = validate_price_evidence(
        json.loads(price_evidence_path.read_text(encoding="utf-8")),
        now=now,
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "execution.private.json"
    if state_path.exists():
        raise ProviderExecutionError(
            "execution state already exists; refusing duplicate submission"
        )

    reserve_budget_file(budget_path, label, steps)
    state: dict[str, Any] = {
        "application": A2V_APPLICATION,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "budget_label": label,
        "reserved_amount_usd": float(amount),
        "steps": steps,
        "learning_rate": 0.0001,
        "price_evidence_sha256": sha256_file(price_evidence_path),
        "price_rate_usd_per_step": price["rate_usd_per_step"],
        "phase": "reserved",
        "created_at_utc": utc_now(now),
    }
    atomic_write_json(state_path, state)

    key = extract_unique_fal_key(key_source)
    try:
        training_url = upload_fn(archive, key)
        if not _secure_url(training_url):
            raise ProviderExecutionError(
                "provider training upload did not return a secure URL"
            )
        provider_validation: list[dict[str, str]] = []
        for index, item in enumerate(validation, start=1):
            if not isinstance(item, dict) or set(item) != {"prompt", "image", "audio"}:
                raise ProviderExecutionError(
                    "local validation entries require prompt/image/audio"
                )
            image = Path(str(item["image"]))
            audio = Path(str(item["audio"]))
            image_url = upload_fn(image, key)
            audio_url = upload_fn(audio, key)
            provider_validation.append(
                {
                    "prompt": str(item["prompt"]),
                    "image_url": image_url,
                    "audio_url": audio_url,
                }
            )
            state[f"validation_{index}_image_sha256"] = sha256_file(image)
            state[f"validation_{index}_audio_sha256"] = sha256_file(audio)
        arguments = build_training_input(
            training_url,
            steps=steps,
            trigger_phrase=trigger_phrase,
            validation=provider_validation,
        )
    except Exception:
        state["phase"] = "upload_failed_unsubmitted"
        state["last_checked_at_utc"] = utc_now(now)
        atomic_write_json(state_path, state)
        raise

    state.update(
        {
            "phase": "uploaded",
            "training_upload_url_sha256": hashlib.sha256(
                training_url.encode("utf-8")
            ).hexdigest(),
            "request_body_sha256": hashlib.sha256(
                canonical_json_bytes(arguments)
            ).hexdigest(),
            "uploaded_at_utc": utc_now(now),
            "validation_count": len(provider_validation),
        }
    )
    atomic_write_json(state_path, state)
    update_budget_entry(budget_path, label, "uploaded")

    state["phase"] = "submit_intent"
    state["submit_intent_at_utc"] = utc_now(now)
    atomic_write_json(state_path, state)
    update_budget_entry(budget_path, label, "submit_intent")
    resolved_submit = submit_fn or submit_once
    try:
        acknowledgement = resolved_submit(A2V_APPLICATION, arguments, key)
    except Exception:
        state["phase"] = "submission_ambiguous"
        state["last_checked_at_utc"] = utc_now(now)
        atomic_write_json(state_path, state)
        update_budget_entry(budget_path, label, "submission_ambiguous")
        raise
    request_id = acknowledgement.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        state["phase"] = "submission_ambiguous"
        atomic_write_json(state_path, state)
        update_budget_entry(budget_path, label, "submission_ambiguous")
        raise ProviderExecutionError(
            "provider acknowledgement is malformed; do not retry"
        )
    state.update(
        {
            "phase": "submitted",
            "request_id": request_id,
            "submitted_at_utc": utc_now(now),
        }
    )
    atomic_write_json(state_path, state)
    update_budget_entry(
        budget_path,
        label,
        "submitted",
        submitted_at_utc=state["submitted_at_utc"],
    )
    return {
        "phase": "submitted",
        "reserved_amount_usd": float(amount),
        "steps": steps,
    }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def submit_once(
    application: str,
    arguments: dict[str, Any],
    key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    if application != A2V_APPLICATION:
        raise ValueError("A2V queue endpoint is fixed")
    if not isinstance(key, str) or not key:
        raise ProviderExecutionError("Fal key is unavailable")
    selected_transport = transport or httpx.HTTPTransport(
        retries=0,
        verify=True,
        trust_env=False,
        http1=True,
        http2=False,
    )
    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Fal-No-Retry": "1",
        "x-app-fal-disable-fallback": "true",
        "X-Fal-Store-IO": "0",
    }
    try:
        with httpx.Client(
            transport=selected_transport,
            follow_redirects=False,
            trust_env=False,
            http1=True,
            http2=False,
            timeout=120.0,
        ) as client:
            response = client.post(
                A2V_QUEUE_URL,
                content=canonical_json_bytes(arguments),
                headers=headers,
            )
    except Exception as exc:
        raise ProviderExecutionError(
            "provider submission transport outcome is ambiguous; do not retry"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise ProviderExecutionError(
            f"provider submission returned non-success status {response.status_code}"
        )
    try:
        acknowledgement = response.json()
    except Exception as exc:
        raise ProviderExecutionError(
            "provider submission acknowledgement is malformed; do not retry"
        ) from exc
    request_id = (
        acknowledgement.get("request_id")
        if isinstance(acknowledgement, dict)
        else None
    )
    if not isinstance(request_id, str) or not request_id.strip():
        raise ProviderExecutionError(
            "provider submission acknowledgement is malformed; do not retry"
        )
    return {"request_id": request_id}


def extract_loss_observations(value: object) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []

    def walk(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if (
                    isinstance(child, (int, float))
                    and not isinstance(child, bool)
                    and "loss" in str(key).lower()
                ):
                    observations.append(
                        {
                            "source": "field",
                            "path": child_path,
                            "value": float(child),
                        }
                    )
                elif isinstance(child, str):
                    for match in LOSS_TEXT_PATTERN.finditer(child):
                        observations.append(
                            {
                                "source": "text",
                                "path": child_path,
                                "value": float(match.group(1)),
                            }
                        )
                else:
                    walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "")
    return observations


def record_telemetry_snapshot(
    path: Path,
    snapshot: object,
    *,
    observed_at_utc: str,
) -> dict[str, object]:
    snapshot_hash = hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()
    telemetry = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "snapshots": []}
    )
    snapshots = telemetry.get("snapshots")
    if not isinstance(snapshots, list):
        raise ProviderExecutionError("provider telemetry file has an unexpected shape")
    if not any(
        isinstance(item, dict) and item.get("snapshot_sha256") == snapshot_hash
        for item in snapshots
    ):
        snapshots.append(
            {
                "observed_at_utc": observed_at_utc,
                "snapshot_sha256": snapshot_hash,
                "provider_snapshot": snapshot,
            }
        )
    loss_observations: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for item in snapshots:
        if not isinstance(item, dict):
            continue
        for observation in extract_loss_observations(item.get("provider_snapshot")):
            key = (
                observation.get("source"),
                observation.get("path"),
                observation.get("value"),
            )
            if key in seen:
                continue
            seen.add(key)
            loss_observations.append(observation)
    telemetry["snapshot_count"] = len(snapshots)
    telemetry["loss_observations"] = loss_observations
    telemetry["provider_loss_status"] = (
        "exposed" if loss_observations else "not_exposed_yet"
    )
    atomic_write_json(path, telemetry)
    return {
        "snapshot_count": len(snapshots),
        "loss_observations": loss_observations,
        "provider_loss_status": telemetry["provider_loss_status"],
    }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _jsonable(attributes)
    return str(value)


def provider_status(key: str, request_id: str) -> object:
    import fal_client

    return fal_client.SyncClient(key=key, default_timeout=120.0).status(
        A2V_APPLICATION,
        request_id,
        with_logs=True,
    )


def provider_result(key: str, request_id: str) -> dict[str, Any]:
    import fal_client

    result = fal_client.SyncClient(key=key, default_timeout=120.0).result(
        A2V_APPLICATION,
        request_id,
    )
    if not isinstance(result, dict):
        raise ProviderExecutionError("provider result has an unexpected shape")
    return result


def _safe_artifact_name(field: str, value: dict[str, Any]) -> str:
    provider_name = value.get("file_name")
    suffix = Path(provider_name).suffix if isinstance(provider_name, str) else ""
    default_suffix = {
        "lora_file": ".safetensors",
        "config_file": ".json",
        "debug_dataset": ".zip",
        "video": ".mp4",
        "audio": ".wav",
    }.get(field, ".bin")
    if not suffix or len(suffix) > 12:
        suffix = default_suffix
    return re.sub(r"[^A-Za-z0-9_.-]", "_", field) + suffix


def download_result_artifacts(
    result: dict[str, Any],
    destination: Path,
) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for field in ("lora_file", "config_file", "debug_dataset", "video", "audio"):
        value = result.get(field)
        if not isinstance(value, dict):
            continue
        url = value.get("url")
        if not _secure_url(url):
            continue
        name = _safe_artifact_name(field, value)
        target = destination / name
        temporary = target.with_name(target.name + ".part")
        transport = httpx.HTTPTransport(retries=0, verify=True, trust_env=False)
        with httpx.Client(
            transport=transport,
            follow_redirects=True,
            trust_env=False,
            timeout=600.0,
        ) as client:
            with client.stream("GET", str(url)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
        os.replace(temporary, target)
        downloaded.append(name)
    return downloaded


def monitor_run(
    *,
    state_dir: Path,
    budget_path: Path,
    key_source: Path,
    status_fn: Callable[[str, str], object] = provider_status,
    result_fn: Callable[[str, str], dict[str, Any]] = provider_result,
    download_fn: Callable[[dict[str, Any], Path], list[str]] = download_result_artifacts,
    now: datetime | None = None,
) -> dict[str, object]:
    state_path = state_dir / "execution.private.json"
    if not state_path.is_file():
        raise ProviderExecutionError("execution state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("phase") == "completed":
        artifacts = state.get("artifacts", {})
        return {
            "phase": "completed",
            "provider_status": "completed",
            "provider_loss_status": state.get("provider_loss_status"),
            "provider_loss_observation_count": state.get(
                "provider_loss_observation_count", 0
            ),
            "reserved_amount_usd": state.get("reserved_amount_usd"),
            "artifacts": sorted(artifacts) if isinstance(artifacts, dict) else [],
        }
    request_id = state.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProviderExecutionError("execution has no submitted provider request")
    if state.get("application") != A2V_APPLICATION:
        raise ProviderExecutionError("execution endpoint mismatch")
    key = extract_unique_fal_key(key_source)
    raw_status = status_fn(key, request_id)
    status_type = type(raw_status).__name__.lower()
    snapshot = _jsonable(raw_status)
    if isinstance(snapshot, dict) and isinstance(snapshot.get("status"), str):
        status_name = str(snapshot["status"]).lower()
    else:
        status_name = status_type
    observed_at = utc_now(now)
    telemetry = record_telemetry_snapshot(
        state_dir / "telemetry.private.json",
        snapshot,
        observed_at_utc=observed_at,
    )
    state.update(
        {
            "last_provider_status": status_name,
            "last_checked_at_utc": observed_at,
            "telemetry_snapshot_count": telemetry["snapshot_count"],
            "provider_loss_status": telemetry["provider_loss_status"],
            "provider_loss_observation_count": len(telemetry["loss_observations"]),
        }
    )
    atomic_write_json(state_path, state)

    if status_name != "completed":
        return {
            "phase": state.get("phase"),
            "provider_status": status_name,
            "provider_loss_status": telemetry["provider_loss_status"],
            "provider_loss_observation_count": len(telemetry["loss_observations"]),
            "reserved_amount_usd": state.get("reserved_amount_usd"),
        }

    provider_error = snapshot.get("error") if isinstance(snapshot, dict) else None
    if provider_error:
        state["phase"] = "failed_pending_billing_verification"
        state["provider_error_present"] = True
        state["completed_at_utc"] = observed_at
        atomic_write_json(state_path, state)
        update_budget_entry(
            budget_path,
            str(state["budget_label"]),
            "failed_pending_billing_verification",
            completed_at_utc=observed_at,
        )
        return {
            "phase": state["phase"],
            "provider_status": status_name,
            "reserved_amount_usd": state.get("reserved_amount_usd"),
        }

    result = result_fn(key, request_id)
    if not isinstance(result, dict):
        raise ProviderExecutionError("provider result has an unexpected shape")
    atomic_write_json(state_dir / "result.private.json", result)
    artifacts = download_fn(result, state_dir / "artifacts")
    required = {"lora_file", "config_file", "debug_dataset"}
    present = {name.split(".", 1)[0] for name in artifacts}
    missing = sorted(required - present)
    if missing:
        raise ProviderExecutionError(
            "completed provider result is missing required artifacts: "
            + ", ".join(missing)
        )
    final_loss_status = (
        "exposed"
        if telemetry["loss_observations"]
        else "provider_loss_not_exposed"
    )
    state.update(
        {
            "phase": "completed",
            "completed_at_utc": observed_at,
            "provider_loss_status": final_loss_status,
            "artifacts": {
                name: {
                    "size_bytes": (state_dir / "artifacts" / name).stat().st_size,
                    "sha256": sha256_file(state_dir / "artifacts" / name),
                }
                for name in artifacts
            },
        }
    )
    atomic_write_json(state_path, state)
    update_budget_entry(
        budget_path,
        str(state["budget_label"]),
        "charged_expected",
        completed_at_utc=observed_at,
    )
    return {
        "phase": "completed",
        "provider_status": status_name,
        "provider_loss_status": final_loss_status,
        "provider_loss_observation_count": len(telemetry["loss_observations"]),
        "reserved_amount_usd": state.get("reserved_amount_usd"),
        "artifacts": artifacts,
    }


def release_unsubmitted_budget(
    *,
    state_dir: Path,
    budget_path: Path,
    evidence: str,
    now: datetime | None = None,
) -> dict[str, object]:
    if not evidence.strip():
        raise ValueError("release evidence is required")
    state_path = state_dir / "execution.private.json"
    if not state_path.is_file():
        raise ProviderExecutionError("execution state is unavailable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("request_id") or state.get("submitted_at_utc"):
        raise ProviderExecutionError("submitted or ambiguous reservations cannot be released")
    if state.get("phase") not in {
        "reserved",
        "uploaded",
        "upload_failed_unsubmitted",
    }:
        raise ProviderExecutionError("execution state is not provably unsubmitted")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    label = state.get("budget_label")
    matches = [
        entry
        for entry in budget.get("entries", [])
        if isinstance(entry, dict) and entry.get("label") == label
    ]
    if len(matches) != 1 or matches[0].get("status") not in {"reserved", "uploaded"}:
        raise ProviderExecutionError("budget reservation is unavailable or ambiguous")
    amount = Decimal(str(matches[0].get("amount_usd", 0)))
    current = Decimal(str(budget.get("incremental_accounted_or_reserved", 0)))
    updated_total = current - amount
    if updated_total < 0:
        raise ProviderExecutionError("budget release would make accounted spend negative")
    budget["incremental_accounted_or_reserved"] = float(updated_total)
    if "incremental_absolute_stop" in budget:
        budget["incremental_remaining_absolute"] = float(
            Decimal(str(budget["incremental_absolute_stop"])) - updated_total
        )
    if "incremental_normal_cap" in budget:
        budget["incremental_remaining_normal_cap"] = float(
            Decimal(str(budget["incremental_normal_cap"])) - updated_total
        )
    released_at = utc_now(now)
    matches[0].update(
        {
            "status": "released_unsubmitted",
            "released_at_utc": released_at,
            "release_evidence": evidence.strip(),
        }
    )
    atomic_write_json(budget_path, budget)
    state.update(
        {
            "phase": "released_unsubmitted",
            "released_at_utc": released_at,
            "release_evidence": evidence.strip(),
        }
    )
    atomic_write_json(state_path, state)
    return {
        "phase": "released_unsubmitted",
        "released_amount_usd": float(amount),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot Fal LTX-2.3 broad A2V trainer with telemetry capture."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start")
    start.add_argument("--archive", type=Path, required=True)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--key-source", type=Path, required=True)
    start.add_argument("--price-evidence", type=Path, required=True)
    start.add_argument("--state-dir", type=Path, required=True)
    start.add_argument("--label", required=True)
    start.add_argument("--config", type=Path, required=True)

    monitor = commands.add_parser("monitor")
    monitor.add_argument("--budget", type=Path, required=True)
    monitor.add_argument("--key-source", type=Path, required=True)
    monitor.add_argument("--state-dir", type=Path, required=True)

    release = commands.add_parser("release-unsubmitted")
    release.add_argument("--budget", type=Path, required=True)
    release.add_argument("--state-dir", type=Path, required=True)
    release.add_argument("--evidence", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        config = load_run_config(args.config)
        output = start_run(
            archive=args.archive,
            budget_path=args.budget,
            key_source=args.key_source,
            price_evidence_path=args.price_evidence,
            state_dir=args.state_dir,
            label=args.label,
            steps=int(config["steps"]),
            trigger_phrase=str(config["trigger_phrase"]),
            validation=list(config["validation"]),
        )
    elif args.command == "monitor":
        output = monitor_run(
            state_dir=args.state_dir,
            budget_path=args.budget,
            key_source=args.key_source,
        )
    else:
        output = release_unsubmitted_budget(
            state_dir=args.state_dir,
            budget_path=args.budget,
            evidence=args.evidence,
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
